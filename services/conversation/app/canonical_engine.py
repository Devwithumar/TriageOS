"""Application-facing runtime for the canonical conversation aggregate."""

from dataclasses import dataclass, replace
import re
from threading import Lock, RLock
from typing import Any

from libs.ai.config import load_llm_config
from libs.ai.conversation_intelligence import detect_intent
from libs.ai.proposal_adapter import ProposalAdapter, ProposalAdapterError, ProposalCompletion
from libs.ai.proposal_fallback import build_recovery_proposal
from libs.conversation.domain import (
    AppointmentRequestResultData,
    AvailabilityResultData,
    Channel,
    CompleteOperationCommand,
    ConversationSession,
    ConversationState,
    OperationName,
    OperationRequestedEvent,
    PracticeProfileResultData,
    ProviderResultRecord,
    ProviderSearchResultData,
    RequestOperationCommand,
    SlotSource,
    StartOperationCommand,
    TaskName,
    WorkflowState,
    command_to_event,
    reduce_state,
)
from libs.conversation.orchestrator import (
    ConversationOrchestrator,
    OrchestrationResult,
    ProposalSource,
)
from libs.conversation.persistence import InMemorySessionRepository
from libs.conversation.proposals import ConversationProposal
from libs.conversation.proposals import ProposalConfidenceBand, ProposedSlot, ToolSelectionProposal
from libs.conversation.contracts import DialogueAct, IntentName
from libs.conversation.provider_matching import (
    is_provider_details_request,
    is_provider_retry_request,
    resolve_provider_reference,
)
from services.conversation.app.provider_directory import ProviderDirectory
from services.conversation.app.practice_profile import PracticeProfileLookup
from services.conversation.app.scheduling import SchedulingService, build_scheduling_service
from services.conversation.app.response_policy import ResponseDecision, build_response


@dataclass(frozen=True)
class CanonicalTurnResult:
    reply: str
    state: dict[str, Any]
    usage: dict[str, Any]


class CanonicalConversationEngine:
    """Single application boundary for text turns and canonical state."""

    def __init__(
        self,
        adapter: ProposalSource | None = None,
        provider_directory: ProviderDirectory | None = None,
        practice_profile: PracticeProfileLookup | None = None,
        scheduling_service: SchedulingService | None = None,
    ) -> None:
        self.repository = InMemorySessionRepository()
        self.provider_directory = provider_directory or ProviderDirectory()
        self.practice_profile = practice_profile or PracticeProfileLookup()
        self.scheduling_service = scheduling_service or build_scheduling_service()
        adapter = adapter or ProposalAdapter(load_llm_config())
        self.orchestrator = ConversationOrchestrator(
            self.repository,
            _PolicyAwareProposalSource(adapter),
        )
        self._messages: dict[str, list[dict[str, str]]] = {}
        self._session_locks: dict[str, RLock] = {}
        self._session_locks_guard = Lock()

    def handle_turn(self, session_id: str, user_text: str) -> CanonicalTurnResult:
        with self._lock_for_session(session_id):
            return self._handle_turn(session_id, user_text)

    def _handle_turn(self, session_id: str, user_text: str) -> CanonicalTurnResult:
        recent_messages = self._messages.get(session_id, [])[-8:]
        result = self.orchestrator.process_turn(
            session_id,
            user_text,
            channel=Channel.TEXT,
            recent_messages=recent_messages,
        )
        if not result.error:
            result = self._ensure_provider_operation(result)
            result = self._execute_provider_operation(result)
            result = self._ensure_availability_operation(result)
            result = self._execute_availability_operation(result)
            result = self._execute_appointment_operation(result)
            result = self._execute_practice_profile_operation(result)
        decision = build_response(result)
        self._messages.setdefault(session_id, []).extend(
            [
                {"role": "user", "content": user_text.strip()},
                {"role": "assistant", "content": decision.text},
            ]
        )
        return CanonicalTurnResult(
            reply=decision.text,
            state=self._compact_state(result.session, self._messages[session_id]),
            usage=self._usage(result, decision, len(recent_messages)),
        )

    def _lock_for_session(self, session_id: str) -> RLock:
        with self._session_locks_guard:
            lock = self._session_locks.get(session_id)
            if lock is None:
                lock = RLock()
                self._session_locks[session_id] = lock
            return lock

    def _ensure_provider_operation(self, result: OrchestrationResult) -> OrchestrationResult:
        state = result.session.state
        if state.pending_operation or state.workflow_state == WorkflowState.SELECTING_PROVIDER:
            return result
        if state.last_operation_result is not None:
            return result
        if state.active_task == TaskName.PROVIDER_LOOKUP:
            required = ("care_setting", "location")
        elif state.active_task == TaskName.APPOINTMENT_REQUEST:
            required = ("care_setting", "location", "appointment_reason")
        else:
            return result
        if any(slot not in state.slots for slot in required):
            return result
        request_id = f"{state.session_id}:search:{state.state_version}"
        command = RequestOperationCommand(
            session_id=state.session_id,
            expected_state_version=state.state_version,
            correlation_id=result.turn_event.correlation_id,
            request_id=request_id,
            operation=OperationName.SEARCH_PROVIDERS,
            idempotency_key=request_id,
        )
        event = command_to_event(state, command)
        committed = self.repository.commit_batch(
            state.session_id,
            state.state_version,
            [event],
        )
        return replace(result, session=committed.session, events=result.events + (event,))

    def _execute_provider_operation(self, result: OrchestrationResult) -> OrchestrationResult:
        state = result.session.state
        operation = state.pending_operation
        if operation is None or operation.operation != OperationName.SEARCH_PROVIDERS:
            return result
        start_command = StartOperationCommand(
            session_id=state.session_id,
            expected_state_version=state.state_version,
            correlation_id=operation.correlation_id,
            request_id=operation.request_id,
        )
        started_event = command_to_event(state, start_command)
        started_state = reduce_state(state, started_event)
        care_setting = started_state.slots.get("care_setting")
        location = started_state.slots.get("location")
        if care_setting is None or location is None:
            return result
        directory_result = self.provider_directory.search(
            care_setting=care_setting.value,
            location=location.value,
            reason=(started_state.slots.get("appointment_reason").value
                    if started_state.slots.get("appointment_reason")
                    else None),
        )
        output = ProviderSearchResultData(
            location=directory_result.location,
            providers=[
                ProviderResultRecord(
                    provider_id=provider.provider_id,
                    name=provider.name,
                    address=provider.address,
                    category=provider.category,
                    distance_km=provider.distance_km,
                    phone=provider.phone,
                    website=provider.website,
                )
                for provider in directory_result.providers
            ],
            source=directory_result.source,
            error=directory_result.error,
            error_code=directory_result.error_code,
        )
        complete_command = CompleteOperationCommand(
            session_id=started_state.session_id,
            expected_state_version=started_state.state_version,
            correlation_id=operation.correlation_id,
            request_id=operation.request_id,
            requested_state_version=operation.requested_state_version,
            result=output,
        )
        completed_event = command_to_event(started_state, complete_command)
        committed = self.repository.commit_batch(
            state.session_id,
            state.state_version,
            [started_event, completed_event],
        )
        return replace(
            result,
            session=committed.session,
            events=result.events + (started_event, completed_event),
        )

    def _execute_practice_profile_operation(self, result: OrchestrationResult) -> OrchestrationResult:
        state = result.session.state
        operation = state.pending_operation
        if operation is None or operation.operation != OperationName.GET_PRACTICE_PROFILE:
            return result
        start_command = StartOperationCommand(
            session_id=state.session_id,
            expected_state_version=state.state_version,
            correlation_id=operation.correlation_id,
            request_id=operation.request_id,
        )
        started_event = command_to_event(state, start_command)
        started_state = reduce_state(state, started_event)
        profile_result = self.practice_profile.lookup()
        complete_command = CompleteOperationCommand(
            session_id=started_state.session_id,
            expected_state_version=started_state.state_version,
            correlation_id=operation.correlation_id,
            request_id=operation.request_id,
            requested_state_version=operation.requested_state_version,
            result=profile_result,
        )
        completed_event = command_to_event(started_state, complete_command)
        committed = self.repository.commit_batch(
            state.session_id,
            state.state_version,
            [started_event, completed_event],
        )
        return replace(
            result,
            session=committed.session,
            events=result.events + (started_event, completed_event),
        )

    def _ensure_availability_operation(self, result: OrchestrationResult) -> OrchestrationResult:
        state = result.session.state
        if (
            state.active_task != TaskName.APPOINTMENT_REQUEST
            or state.pending_operation
            or "provider_id" not in state.slots
            or not isinstance(state.last_operation_result, ProviderSearchResultData)
        ):
            return result
        request_id = f"{state.session_id}:availability:{state.state_version}"
        command = RequestOperationCommand(
            session_id=state.session_id,
            expected_state_version=state.state_version,
            correlation_id=result.turn_event.correlation_id,
            request_id=request_id,
            operation=OperationName.GET_AVAILABILITY,
            idempotency_key=request_id,
        )
        event = command_to_event(state, command)
        committed = self.repository.commit_batch(
            state.session_id,
            state.state_version,
            [event],
        )
        return replace(result, session=committed.session, events=result.events + (event,))

    def _execute_availability_operation(self, result: OrchestrationResult) -> OrchestrationResult:
        state = result.session.state
        operation = state.pending_operation
        if operation is None or operation.operation != OperationName.GET_AVAILABILITY:
            return result
        provider = state.slots.get("provider_id")
        if provider is None:
            return result
        start_command = StartOperationCommand(
            session_id=state.session_id,
            expected_state_version=state.state_version,
            correlation_id=operation.correlation_id,
            request_id=operation.request_id,
        )
        started_event = command_to_event(state, start_command)
        started_state = reduce_state(state, started_event)
        availability = self.scheduling_service.get_availability(provider.value)
        output = AvailabilityResultData(
            provider_id=availability.provider_id,
            slots=availability.slots,
            source=availability.source,
            error=availability.error,
            error_code=availability.error_code or ("unavailable" if availability.error else None),
        )
        complete_command = CompleteOperationCommand(
            session_id=started_state.session_id,
            expected_state_version=started_state.state_version,
            correlation_id=operation.correlation_id,
            request_id=operation.request_id,
            requested_state_version=operation.requested_state_version,
            result=output,
        )
        completed_event = command_to_event(started_state, complete_command)
        committed = self.repository.commit_batch(
            state.session_id,
            state.state_version,
            [started_event, completed_event],
        )
        return replace(result, session=committed.session, events=result.events + (started_event, completed_event))

    def _execute_appointment_operation(self, result: OrchestrationResult) -> OrchestrationResult:
        state = result.session.state
        operation = state.pending_operation
        if operation is None or operation.operation != OperationName.CREATE_APPOINTMENT_REQUEST:
            return result
        preferred_time = state.slots.get("preferred_time")
        if preferred_time is None:
            return result
        start_command = StartOperationCommand(
            session_id=state.session_id,
            expected_state_version=state.state_version,
            correlation_id=operation.correlation_id,
            request_id=operation.request_id,
        )
        started_event = command_to_event(state, start_command)
        started_state = reduce_state(state, started_event)
        submission = self.scheduling_service.submit_request(
            idempotency_key=operation.idempotency_key,
            preferred_time=preferred_time.value,
        )
        output = AppointmentRequestResultData(
            request_reference=submission.request_reference,
            status=submission.status,
            error=submission.error,
        )
        complete_command = CompleteOperationCommand(
            session_id=started_state.session_id,
            expected_state_version=started_state.state_version,
            correlation_id=operation.correlation_id,
            request_id=operation.request_id,
            requested_state_version=operation.requested_state_version,
            result=output,
        )
        completed_event = command_to_event(started_state, complete_command)
        committed = self.repository.commit_batch(
            state.session_id,
            state.state_version,
            [started_event, completed_event],
        )
        return replace(result, session=committed.session, events=result.events + (started_event, completed_event))

    @staticmethod
    def _compact_state(session: ConversationSession, messages: list[dict[str, str]]) -> dict[str, Any]:
        state = session.state
        status = {
            WorkflowState.COLLECTING_CONTEXT: "collecting_details",
            WorkflowState.SEARCHING_PROVIDERS: "collecting_details",
            WorkflowState.SELECTING_PROVIDER: "collecting_details",
            WorkflowState.COLLECTING_DETAILS: "collecting_details",
            WorkflowState.REVIEWING_REQUEST: "reviewing_request",
            WorkflowState.AWAITING_CONFIRMATION: "reviewing_request",
            WorkflowState.COMPLETED: "completed",
            WorkflowState.CANCELLED: "cancelled",
            WorkflowState.FAILED: "submission_failed",
        }.get(state.workflow_state, "idle")
        ordered_slots = [
            "care_setting",
            "location",
            "appointment_reason",
            "provider_id",
            "preferred_time",
            "caller_name",
            "callback_number",
        ]
        missing = [
            slot
            for slot in ordered_slots
            if state.active_task == TaskName.APPOINTMENT_REQUEST
            and slot not in state.slots
            and not (slot == "provider_id" and state.provider_options)
        ]
        if state.active_task == TaskName.PROVIDER_LOOKUP:
            missing = [slot for slot in ("care_setting", "location") if slot not in state.slots]
        next_action = "continue_conversation"
        if state.workflow_state == WorkflowState.SEARCHING_PROVIDERS:
            next_action = "search_providers"
        elif state.workflow_state == WorkflowState.SELECTING_PROVIDER:
            next_action = "select_provider"
        elif missing:
            next_action = f"collect_{missing[0]}"
        elif state.workflow_state == WorkflowState.REVIEWING_REQUEST:
            next_action = "confirm_request"
        structured = {
            **{name: slot.value for name, slot in state.slots.items()},
            "provider_options": [option.model_dump(mode="json") for option in state.provider_options],
            "appointment_status": status,
            "workflow": "receptionist" if state.active_task != TaskName.NONE else "general",
            "next_action": next_action,
            "missing_fields": missing,
        }
        return {
            "recent_messages": messages[-20:],
            "summary": "Canonical conversation state is active.",
            "structured_state": structured,
        }

    @staticmethod
    def _usage(
        result: OrchestrationResult,
        decision: ResponseDecision,
        context_messages: int,
    ) -> dict[str, Any]:
        proposal = result.proposal
        completion = result.proposal_completion
        return {
            "provider": decision.provider,
            "model": completion.model if completion else "canonical-policy",
            "route_reason": decision.reason,
            "estimated_input_messages": context_messages,
            "prompt_tokens": completion.prompt_tokens if completion else None,
            "completion_tokens": completion.completion_tokens if completion else None,
            "error": result.error,
            "intent": (
                {
                    "name": proposal.intent.value,
                    "dialogue_act": proposal.dialogue_act.value,
                    "confidence_band": proposal.confidence_band.value,
                    "requested_task": proposal.requested_task.value,
                }
                if proposal
                else {}
            ),
            "proposal": proposal.model_dump(mode="json") if proposal else {},
            "tool_call": (
                result.session.state.pending_operation.model_dump(mode="json")
                if result.session.state.pending_operation
                else None
            ),
            "tool_result": (
                result.session.state.last_operation_result.model_dump(mode="json")
                if result.session.state.last_operation_result
                else None
            ),
        }


class _PolicyAwareProposalSource:
    def __init__(self, adapter: ProposalSource) -> None:
        self._adapter = adapter

    def propose(
        self,
        user_text: str,
        state: ConversationState,
        recent_messages: list[dict[str, str]],
        correlation_id: str,
    ) -> ProposalCompletion:
        detected = detect_intent(user_text)
        if detected.name == "urgent_safety":
            proposal = ConversationProposal(
                session_id=state.session_id,
                based_on_state_version=state.state_version,
                correlation_id=correlation_id,
                intent=IntentName.URGENT_SAFETY,
                dialogue_act=DialogueAct.ESCALATE,
                confidence_band="high",
            )
            return ProposalCompletion(proposal=proposal, model="policy", provider="guardrail")
        if detected.name == "practice_information" and not state.provider_options:
            proposal = ConversationProposal(
                session_id=state.session_id,
                based_on_state_version=state.state_version,
                correlation_id=correlation_id,
                intent=IntentName.PRACTICE_INFORMATION,
                dialogue_act=DialogueAct.REQUEST_INFORMATION,
                confidence_band=ProposalConfidenceBand.HIGH,
                requested_task=TaskName.PRACTICE_INFORMATION,
                tool_selection=ToolSelectionProposal(
                    operation=OperationName.GET_PRACTICE_PROFILE,
                    rationale="answer only from the configured practice profile",
                ),
            )
            return ProposalCompletion(proposal=proposal, model="policy", provider="practice_profile_policy")
        if detected.name == "appointment_cancellation":
            proposal = ConversationProposal(
                session_id=state.session_id,
                based_on_state_version=state.state_version,
                correlation_id=correlation_id,
                intent=IntentName.APPOINTMENT_CANCELLATION,
                dialogue_act=DialogueAct.CANCEL,
                confidence_band="high",
                cancel_requested=True,
            )
            return ProposalCompletion(proposal=proposal, model="policy", provider="guardrail")
        if (
            detected.name == "confirmation"
            and state.active_task == TaskName.APPOINTMENT_REQUEST
            and state.workflow_state == WorkflowState.REVIEWING_REQUEST
        ):
            proposal = ConversationProposal(
                session_id=state.session_id,
                based_on_state_version=state.state_version,
                correlation_id=correlation_id,
                intent=IntentName.CONFIRMATION,
                dialogue_act=DialogueAct.CONFIRM,
                confidence_band=ProposalConfidenceBand.HIGH,
                tool_selection=ToolSelectionProposal(
                    operation=OperationName.CREATE_APPOINTMENT_REQUEST,
                    rationale="submit the reviewed appointment request after explicit confirmation",
                ),
            )
            return ProposalCompletion(proposal=proposal, model="policy", provider="confirmation_policy")
        if detected.name == "correction":
            recovery = build_recovery_proposal(user_text, state, correlation_id)
            if recovery.corrections:
                return ProposalCompletion(
                    proposal=recovery,
                    model="policy",
                    provider="correction_policy",
                )
        if (
            is_provider_retry_request(user_text)
            and isinstance(state.last_operation_result, ProviderSearchResultData)
            and state.last_operation_result.error
            and state.active_task in {TaskName.APPOINTMENT_REQUEST, TaskName.PROVIDER_LOOKUP}
            and state.pending_operation is None
        ):
            proposal = ConversationProposal(
                session_id=state.session_id,
                based_on_state_version=state.state_version,
                correlation_id=correlation_id,
                intent=IntentName.PROVIDER_LOOKUP,
                dialogue_act=DialogueAct.INFORM,
                confidence_band=ProposalConfidenceBand.HIGH,
                tool_selection=ToolSelectionProposal(
                    operation=OperationName.SEARCH_PROVIDERS,
                    rationale="retry the previous verified provider search",
                ),
            )
            return ProposalCompletion(proposal=proposal, model="policy", provider="retry_policy")
        if state.provider_options and state.workflow_state == WorkflowState.SELECTING_PROVIDER:
            provider = resolve_provider_reference(user_text, state.provider_options)
            if provider is not None:
                slots = []
                if not is_provider_details_request(user_text):
                    slots = [
                        ProposedSlot(
                            name="provider_id",
                            value=provider.provider_id,
                            source=SlotSource.USER_EXPLICIT,
                            confidence=0.99,
                        ),
                        ProposedSlot(
                            name="provider_name",
                            value=provider.name,
                            source=SlotSource.USER_EXPLICIT,
                            confidence=0.99,
                        ),
                    ]
                proposal = ConversationProposal(
                    session_id=state.session_id,
                    based_on_state_version=state.state_version,
                    correlation_id=correlation_id,
                    intent=IntentName.GENERAL_CONVERSATION,
                    dialogue_act=DialogueAct.INFORM,
                    confidence_band=ProposalConfidenceBand.HIGH,
                    slots=slots,
                )
                return ProposalCompletion(proposal=proposal, model="policy", provider="provider_policy")
        try:
            completion = self._adapter.propose(user_text, state, recent_messages, correlation_id)
            proposal = _remove_ungrounded_explicit_slots(user_text, state, completion.proposal)
            if detected.name == IntentName.PROVIDER_LOOKUP.value and state.active_task in {
                TaskName.NONE,
                TaskName.PROVIDER_LOOKUP,
            }:
                proposal = proposal.model_copy(
                    update={
                        "intent": IntentName.PROVIDER_LOOKUP,
                        "requested_task": TaskName.PROVIDER_LOOKUP,
                    }
                )
            recovery = build_recovery_proposal(user_text, state, correlation_id)
            if recovery.slots:
                captured_names = {slot.name for slot in proposal.slots}
                missing_explicit_slots = [
                    slot for slot in recovery.slots if slot.name not in captured_names
                ]
                if missing_explicit_slots:
                    proposal = proposal.model_copy(
                        update={
                            "slots": [*proposal.slots, *missing_explicit_slots],
                            "dialogue_act": DialogueAct.INFORM,
                        }
                    )
            return replace(completion, proposal=proposal)
        except ProposalAdapterError:
            return ProposalCompletion(
                proposal=build_recovery_proposal(user_text, state, correlation_id),
                model="local-recovery",
                provider="local_recovery",
            )


def _remove_ungrounded_explicit_slots(
    user_text: str,
    state: ConversationState,
    proposal: ConversationProposal,
) -> ConversationProposal:
    text_tokens = set(re.findall(r"[a-z0-9]+", user_text.lower()))
    grounded_slots = []
    changed = False
    for slot in proposal.slots:
        if slot.source.value != "user_explicit":
            grounded_slots.append(slot)
            continue
        if slot.name in {"care_setting", "location", "appointment_reason"}:
            if " ".join(slot.value.lower().split()) == " ".join(user_text.lower().split()):
                changed = True
                continue
        if slot.name in {"provider_id", "provider_name"} and state.workflow_state == WorkflowState.SELECTING_PROVIDER:
            option_values = {
                value
                for option in state.provider_options
                for value in (option.provider_id.lower(), option.name.lower())
            }
            grounded = slot.value.lower() in option_values or any(
                token in text_tokens
                for token in re.findall(r"[a-z0-9]+", slot.value.lower())
                if len(token) > 2
            )
        else:
            value_tokens = [
                token for token in re.findall(r"[a-z0-9]+", slot.value.lower()) if len(token) > 2
            ]
            grounded = bool(value_tokens) and any(token in text_tokens for token in value_tokens)
        if grounded:
            grounded_slots.append(slot)
        else:
            changed = True
    if not changed:
        return proposal
    return proposal.model_copy(update={"slots": grounded_slots})
