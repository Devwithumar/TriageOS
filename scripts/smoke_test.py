"""Phase 1 smoke tests — run with voice (8000) and conversation (8001) services up."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx
import websockets
from pydantic import ValidationError

from libs.ai.conversation_intelligence import detect_intent
from libs.ai.llm import CompletionResult
from libs.ai.config import LLMConfig
from libs.ai.proposal_adapter import ProposalAdapter, ProposalAdapterError, parse_proposal_payload
from libs.ai.proposal_adapter import ProposalCompletion
from libs.ai.proposal_fallback import build_recovery_proposal
from libs.conversation.contracts import (
    ConversationState,
    DialogueAct,
    ExtractedSlot,
    IntentClassification,
    IntentName,
    ReceptionistState,
)
from libs.conversation.domain import (
    CaptureSlotCommand,
    AvailabilityResultData,
    AvailabilitySlotRecord,
    ConversationSession,
    ConversationState as DomainConversationState,
    DomainInvariantError,
    OperationName,
    OperationRequestedEvent,
    OperationStartedEvent,
    OperationSucceededEvent,
    ProviderResultRecord,
    RequestOperationCommand,
    SlotCapturedEvent,
    SlotRecord,
    SlotSource,
    StartOperationCommand,
    StartWorkflowCommand,
    StaleOperationResultError,
    TaskName,
    TaskCancelledEvent,
    WorkflowStartedEvent,
    WorkflowState,
    command_to_event,
    reduce_session,
    reduce_state,
)
from libs.conversation.workflow import apply_receptionist_turn
from libs.conversation.transitions import TRANSITION_TABLE, transition_for
from libs.conversation.orchestrator import ConversationOrchestrator
from libs.conversation.persistence import (
    InMemorySessionRepository,
    PersistenceConflictError,
)
from libs.conversation.proposals import (
    ConversationProposal,
    ProposalConfidenceBand,
    ProposalValidationError,
    ProposedSlot,
    ToolSelectionProposal,
    validate_proposal,
)
from libs.conversation.provider_matching import resolve_provider_reference
from libs.conversation.tool_contracts import (
    ProviderRecord,
    ProviderSearchOutput,
    SearchProvidersInput,
    ToolContractError,
    ToolRequest,
    ToolResult,
    ToolResultStatus,
)
from services.conversation.app.agent import (
    _extract_location_hint,
    _guard_generated_reply,
    generate_reply,
)
from services.conversation.app.canonical_engine import CanonicalConversationEngine
from services.conversation.app.provider_directory import (
    Provider,
    ProviderDirectory,
    ProviderDirectoryError,
    ProviderSearchResult,
)
from services.conversation.app.practice_profile import PracticeProfileLookup
from services.conversation.app.scheduling import (
    AppointmentDetails,
    GoogleCalendarSchedulingService,
    MockSchedulingService,
    UnavailableSchedulingService,
    build_scheduling_service,
)

VOICE_URL = os.getenv("TRIAGEOS_VOICE_URL", "http://localhost:8000")
CONVERSATION_URL = os.getenv("TRIAGEOS_CONVERSATION_URL", "http://localhost:8001")
WS_BASE = os.getenv("TRIAGEOS_WS_BASE", "ws://localhost:8000")


@dataclass
class Results:
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def ok(self, name: str) -> None:
        self.passed.append(name)
        print(f"  PASS  {name}")

    def fail(self, name: str, detail: str) -> None:
        self.failed.append(f"{name}: {detail}")
        print(f"  FAIL  {name} — {detail}")

    def warn(self, name: str, detail: str) -> None:
        self.warnings.append(f"{name}: {detail}")
        print(f"  WARN  {name} — {detail}")


def test_domain_contracts(results: Results) -> None:
    try:
        state = DomainConversationState(session_id="domain_smoke")
        state = reduce_state(
            state,
            WorkflowStartedEvent(
                aggregate_id=state.session_id,
                aggregate_version=1,
                correlation_id="corr-domain",
                task=TaskName.APPOINTMENT_REQUEST,
            ),
        )
        state = reduce_state(
            state,
            SlotCapturedEvent(
                aggregate_id=state.session_id,
                aggregate_version=2,
                correlation_id="corr-domain",
                slot="location",
                value="Lagos",
                source="user_explicit",
                confidence=1,
            ),
        )
        state = reduce_state(
            state,
            OperationRequestedEvent(
                aggregate_id=state.session_id,
                aggregate_version=3,
                correlation_id="corr-domain",
                request_id="req-provider-1",
                operation=OperationName.SEARCH_PROVIDERS,
                requested_state_version=2,
                idempotency_key="idem-provider-1",
            ),
        )
        state = reduce_state(
            state,
            OperationStartedEvent(
                aggregate_id=state.session_id,
                aggregate_version=4,
                correlation_id="corr-domain",
                request_id="req-provider-1",
            ),
        )
        state = reduce_state(
            state,
            OperationSucceededEvent(
                aggregate_id=state.session_id,
                aggregate_version=5,
                correlation_id="corr-domain",
                request_id="req-provider-1",
                requested_state_version=2,
            ),
        )
        if state.workflow_state != WorkflowState.SELECTING_PROVIDER or state.pending_operation:
            results.fail("domain contracts", "successful provider operation did not advance state")
            return
        try:
            reduce_state(
                state,
                OperationSucceededEvent(
                    aggregate_id=state.session_id,
                    aggregate_version=6,
                    correlation_id="corr-domain",
                    request_id="req-provider-1",
                    requested_state_version=2,
                ),
            )
            results.fail("domain contracts", "stale operation result was accepted")
            return
        except StaleOperationResultError:
            pass
        try:
            DomainConversationState(
                session_id="invalid-domain",
                active_task=TaskName.APPOINTMENT_REQUEST,
                workflow_state=WorkflowState.IDLE,
            )
            results.fail("domain contracts", "invalid idle state was accepted")
            return
        except (DomainInvariantError, ValidationError):
            pass
        cancelled = reduce_state(
            state,
            TaskCancelledEvent(
                aggregate_id=state.session_id,
                aggregate_version=6,
                correlation_id="corr-domain",
                reason="user cancelled the request",
            ),
        )
        if cancelled.workflow_state != WorkflowState.CANCELLED or cancelled.active_task != TaskName.NONE:
            results.fail("domain contracts", "cancellation did not clear the active task")
            return
        results.ok("domain contracts")
    except Exception as exc:
        results.fail("domain contracts", str(exc))


def test_transition_table(results: Results) -> None:
    try:
        session = ConversationSession.create("transition_smoke")
        start = StartWorkflowCommand(
            session_id=session.session_id,
            expected_state_version=0,
            correlation_id="corr-transition",
            task=TaskName.APPOINTMENT_REQUEST,
        )
        session = reduce_session(session, command_to_event(session.state, start))
        capture = CaptureSlotCommand(
            session_id=session.session_id,
            expected_state_version=session.state.state_version,
            correlation_id="corr-transition",
            slot="location",
            value="Lagos",
            source="user_explicit",
            confidence=1,
        )
        session = reduce_session(session, command_to_event(session.state, capture))
        request = RequestOperationCommand(
            session_id=session.session_id,
            expected_state_version=session.state.state_version,
            correlation_id="corr-transition",
            request_id="req-transition-1",
            operation=OperationName.SEARCH_PROVIDERS,
            idempotency_key="idem-transition-1",
        )
        session = reduce_session(session, command_to_event(session.state, request))
        start_operation = StartOperationCommand(
            session_id=session.session_id,
            expected_state_version=session.state.state_version,
            correlation_id="corr-transition",
            request_id="req-transition-1",
        )
        session = reduce_session(session, command_to_event(session.state, start_operation))
        decision = transition_for(
            session.state,
            OperationSucceededEvent(
                aggregate_id=session.session_id,
                aggregate_version=session.state.state_version + 1,
                correlation_id="corr-transition",
                request_id="req-transition-1",
                requested_state_version=2,
            ),
        )
        if decision.target_state != WorkflowState.SELECTING_PROVIDER:
            results.fail("transition table", "provider success transition is incorrect")
            return
        if len(TRANSITION_TABLE) < 10:
            results.fail("transition table", "transition table is incomplete")
            return
        results.ok("transition table")
    except Exception as exc:
        results.fail("transition table", str(exc))


def test_tool_and_persistence_contracts(results: Results) -> None:
    try:
        request = ToolRequest(
            request_id="tool-request-1",
            session_id="tool-session",
            operation=OperationName.SEARCH_PROVIDERS,
            requested_state_version=3,
            idempotency_key="tool-idem-1",
            correlation_id="tool-correlation-1",
            input=SearchProvidersInput(care_setting="hospital", location="Lagos"),
        )
        result = ToolResult(
            request_id=request.request_id,
            session_id=request.session_id,
            operation=request.operation,
            requested_state_version=request.requested_state_version,
            correlation_id=request.correlation_id,
            status=ToolResultStatus.SUCCEEDED,
            output=ProviderSearchOutput(
                resolved_location="Lagos, Nigeria",
                providers=[
                    ProviderRecord(
                        provider_id="osm:node:1",
                        name="Verified Hospital",
                        category="hospital",
                        source="test-directory",
                        observed_at=request.created_at,
                    )
                ],
            ),
        )
        if result.output is None or result.output.tool_name != request.operation:
            results.fail("tool and persistence contracts", "typed tool output was not preserved")
            return
        try:
            ToolResult(
                request_id=request.request_id,
                session_id=request.session_id,
                operation=request.operation,
                requested_state_version=request.requested_state_version,
                correlation_id=request.correlation_id,
                status=ToolResultStatus.FAILED,
            )
            results.fail("tool and persistence contracts", "failed result without error was accepted")
            return
        except (ToolContractError, ValidationError):
            pass

        repository = InMemorySessionRepository()
        session = repository.create("persist-session")
        start_command = StartWorkflowCommand(
            session_id=session.session_id,
            expected_state_version=0,
            correlation_id="persist-correlation",
            task=TaskName.APPOINTMENT_REQUEST,
        )
        event = command_to_event(session.state, start_command)
        first_commit = repository.commit(session.session_id, 0, event)
        replay = repository.commit(session.session_id, 0, event)
        if not replay.idempotent_replay or first_commit.session.state.state_version != 1:
            results.fail("tool and persistence contracts", "duplicate event was not idempotent")
            return
        try:
            repository.commit(session.session_id, 0, event.model_copy(update={"event_id": "different-event"}))
            results.fail("tool and persistence contracts", "stale persistence write was accepted")
            return
        except PersistenceConflictError:
            pass
        results.ok("tool and persistence contracts")
    except Exception as exc:
        results.fail("tool and persistence contracts", str(exc))


def test_proposal_validation(results: Results) -> None:
    try:
        state = DomainConversationState(session_id="proposal-session")
        proposal = ConversationProposal(
            session_id=state.session_id,
            based_on_state_version=state.state_version,
            correlation_id="proposal-correlation",
            intent=IntentName.APPOINTMENT_REQUEST,
            dialogue_act="request_information",
            confidence_band=ProposalConfidenceBand.HIGH,
            requested_task=TaskName.APPOINTMENT_REQUEST,
            slots=[
                ProposedSlot(name="care_setting", value="hospital", source="user_explicit", confidence=1),
                ProposedSlot(name="location", value="Lagos", source="user_explicit", confidence=1),
                ProposedSlot(name="appointment_reason", value="general consultation", source="user_explicit", confidence=1),
            ],
            tool_selection=ToolSelectionProposal(operation=OperationName.SEARCH_PROVIDERS),
            response_draft="I’ll look for verified options.",
        )
        validated = validate_proposal(state, proposal)
        if len(validated.commands) != 5 or [command.expected_state_version for command in validated.commands] != [0, 1, 2, 3, 4]:
            results.fail("proposal validation", "valid proposal did not produce sequential commands")
            return
        for invalid in (
            proposal.model_copy(update={"based_on_state_version": 1}),
            proposal.model_copy(update={"confidence_band": ProposalConfidenceBand.LOW}),
            proposal.model_copy(
                update={
                    "slots": [
                        ProposedSlot(
                            name="location",
                            value="Lagos",
                            source="model_inference",
                            confidence=1,
                        )
                    ],
                    "tool_selection": None,
                }
            ),
        ):
            try:
                validate_proposal(state, invalid)
                results.fail("proposal validation", "unsafe proposal was accepted")
                return
            except ProposalValidationError:
                pass
        results.ok("proposal validation")
    except Exception as exc:
        results.fail("proposal validation", str(exc))


def test_structured_proposal_adapter(results: Results) -> None:
    try:
        state = DomainConversationState(session_id="adapter-session")
        adapter = ProposalAdapter(LLMConfig(provider="stub", api_key=None, model="stub"))
        completion = adapter.propose(
            "I want to book an appointment",
            state,
            [],
            "adapter-correlation",
        )
        if (
            completion.proposal.session_id != state.session_id
            or completion.proposal.based_on_state_version != state.state_version
            or completion.proposal.requested_task != TaskName.APPOINTMENT_REQUEST
        ):
            results.fail("structured proposal adapter", "stub proposal metadata or task was incorrect")
            return
        valid_payload = completion.proposal.model_dump(mode="json")
        parse_proposal_payload(
            valid_payload,
            session_id=state.session_id,
            state_version=state.state_version,
            correlation_id="adapter-correlation",
        )
        for invalid_payload in (
            {**valid_payload, "session_id": "other-session"},
            {**valid_payload, "based_on_state_version": 1},
            {**valid_payload, "unexpected": True},
        ):
            try:
                parse_proposal_payload(
                    invalid_payload,
                    session_id=state.session_id,
                    state_version=state.state_version,
                    correlation_id="adapter-correlation",
                )
                results.fail("structured proposal adapter", "invalid structured output was accepted")
                return
            except ProposalAdapterError:
                pass
        try:
            parse_proposal_payload(
                "not-json",
                session_id=state.session_id,
                state_version=state.state_version,
                correlation_id="adapter-correlation",
            )
            results.fail("structured proposal adapter", "non-object payload was accepted")
            return
        except ProposalAdapterError:
            pass
        results.ok("structured proposal adapter")
    except Exception as exc:
        results.fail("structured proposal adapter", str(exc))


def test_orchestration_pipeline(results: Results) -> None:
    try:
        repository = InMemorySessionRepository()
        adapter = ProposalAdapter(LLMConfig(provider="stub", api_key=None, model="stub"))
        orchestrator = ConversationOrchestrator(repository, adapter)
        first = orchestrator.process_turn(
            "orchestration-session",
            "I want to book an appointment",
            correlation_id="orchestration-correlation-1",
        )
        if (
            not first.succeeded
            or first.proposal is None
            or first.validated_proposal is None
            or len(first.events) != 2
            or first.session.state.state_version != 2
            or first.session.state.active_task != TaskName.APPOINTMENT_REQUEST
            or first.events[0].event_type != "conversation.turn.received"
            or first.events[1].event_type != "workflow.started"
        ):
            results.fail("orchestration pipeline", "valid turn was not committed as one event batch")
            return

        second = orchestrator.process_turn(
            "orchestration-session",
            "Hello, how are you?",
            correlation_id="orchestration-correlation-2",
        )
        if not second.succeeded or len(second.events) != 1 or second.session.state.state_version != 3:
            results.fail("orchestration pipeline", "turn without a state mutation was not committed")
            return

        replay = repository.commit_batch("orchestration-session", 0, list(first.events))
        if not replay.idempotent_replay or replay.session.state.state_version != 3:
            results.fail("orchestration pipeline", "complete batch replay was not idempotent")
            return

        repository = InMemorySessionRepository()
        session = repository.create("atomic-session")
        valid_turn = first.events[0].model_copy(
            update={
                "aggregate_id": session.session_id,
                "aggregate_version": 1,
            }
        )
        invalid_follow_up = valid_turn.model_copy(
            update={"event_id": "invalid-follow-up", "aggregate_version": 99}
        )
        try:
            repository.commit_batch(session.session_id, 0, [valid_turn, invalid_follow_up])
            results.fail("orchestration pipeline", "invalid batch was partially committed")
            return
        except PersistenceConflictError:
            pass
        if repository.load(session.session_id).state.state_version != 0 or repository.events(session.session_id):
            results.fail("orchestration pipeline", "failed batch changed the session")
            return
        results.ok("orchestration pipeline")
    except Exception as exc:
        results.fail("orchestration pipeline", str(exc))


def test_canonical_conversation_engine(results: Results) -> None:
    class FakeProposalSource:
        def propose(self, user_text, state, recent_messages, correlation_id):
            normalized = user_text.lower()
            slots = []
            if "veterinary" in normalized:
                slots.append(ProposedSlot(name="care_setting", value="veterinary care", source="user_explicit", confidence=1))
            if "lagos" in normalized:
                slots.append(ProposedSlot(name="location", value="Lagos", source="user_explicit", confidence=1))
            if "cough" in normalized:
                slots.append(ProposedSlot(name="appointment_reason", value="persistent cough", source="user_explicit", confidence=1))
            task = TaskName.APPOINTMENT_REQUEST if "appointment" in normalized or state.active_task == TaskName.APPOINTMENT_REQUEST else TaskName.NONE
            proposal = ConversationProposal(
                session_id=state.session_id,
                based_on_state_version=state.state_version,
                correlation_id=correlation_id,
                intent=IntentName.APPOINTMENT_REQUEST if task != TaskName.NONE else IntentName.GENERAL_CONVERSATION,
                dialogue_act=DialogueAct.INFORM,
                confidence_band=ProposalConfidenceBand.HIGH,
                requested_task=task,
                slots=slots,
            )
            return ProposalCompletion(proposal=proposal, model="fake", provider="fake")

    class FakeProviderDirectory:
        def search(self, care_setting, location, reason=None):
            return ProviderSearchResult(
                location=location,
                source="fake-directory",
                providers=[
                    Provider(
                        provider_id="fake:1",
                        name="Verified Test Clinic",
                        address="1 Test Street, Lagos",
                        category=care_setting,
                        latitude=0,
                        longitude=0,
                        distance_km=1.2,
                    )
                ],
            )

    try:
        engine = CanonicalConversationEngine(FakeProposalSource(), FakeProviderDirectory())
        session_id = "canonical-engine-smoke"
        first = engine.handle_turn(session_id, "I want to book an appointment")
        second = engine.handle_turn(session_id, "veterinary care")
        third = engine.handle_turn(session_id, "Lagos")
        fourth = engine.handle_turn(session_id, "It is for a persistent cough")
        cancelled = engine.handle_turn(session_id, "Never mind the appointment")
        urgent = engine.handle_turn("canonical-safety-smoke", "I have chest pain right now")
        if (
            "kind of care" not in first.reply.lower()
            or "postal code" not in second.reply.lower()
            or "visit" not in third.reply.lower()
            or "verified test clinic" not in fourth.reply.lower()
            or "stopped" not in cancelled.reply.lower()
            or "emergency" not in urgent.reply.lower()
            or cancelled.state["structured_state"].get("appointment_status") != "cancelled"
        ):
            results.fail(
                "canonical conversation engine",
                f"canonical workflow did not progress safely: replies={[first.reply, second.reply, third.reply, fourth.reply, cancelled.reply, urgent.reply]} state={cancelled.state}",
            )
            return
        results.ok("canonical conversation engine")
    except Exception as exc:
        results.fail("canonical conversation engine", str(exc))


def test_practice_profile_slice(results: Results) -> None:
    profile_payload = {
        "profile_id": "practice:test",
        "profile_version": 3,
        "display_name": "Configured Test Practice",
        "address": "1 Test Street",
        "phone": "+234 000 000 0000",
        "website": "https://example.test",
        "hours": [
            {"day": "Monday", "opens_at": "08:00", "closes_at": "17:00"},
            {"day": "Sunday", "closed": True},
        ],
        "services": ["primary care"],
        "accepted_insurance": ["Example Health"],
    }
    try:
        with tempfile.TemporaryDirectory() as directory:
            profile_path = f"{directory}/practice_profile.json"
            with open(profile_path, "w", encoding="utf-8") as profile_file:
                json.dump(profile_payload, profile_file)
            lookup = PracticeProfileLookup(profile_path)
            loaded = lookup.lookup()
            engine = CanonicalConversationEngine(practice_profile=lookup)

            hours = engine.handle_turn("practice-profile-smoke", "What are your opening hours?")
            missing = CanonicalConversationEngine().handle_turn(
                "unconfigured-profile-smoke", "What services does the practice offer?"
            )
            if (
                loaded.profile_id != "practice:test"
                or loaded.profile_version != 3
                or "Monday: 08:00" not in hours.reply
                or "17:00" not in hours.reply
                or "Sunday: closed" not in hours.reply
                or "practice profile" not in missing.reply.lower()
                or "services" not in missing.reply.lower()
            ):
                results.fail(
                    "practice profile slice",
                    f"unexpected loaded={loaded} hours={hours.reply!r} missing={missing.reply!r}",
                )
                return
        results.ok("practice profile slice")
    except Exception as exc:
        results.fail("practice profile slice", str(exc))


def test_mock_scheduling_slice(results: Results) -> None:
    class SchedulingProposalSource:
        def propose(self, user_text, state, recent_messages, correlation_id):
            normalized = user_text.lower()
            slots = []
            if state.turn_count == 1:
                proposal = ConversationProposal(
                    session_id=state.session_id,
                    based_on_state_version=state.state_version,
                    correlation_id=correlation_id,
                    intent=IntentName.APPOINTMENT_REQUEST,
                    dialogue_act=DialogueAct.INFORM,
                    confidence_band=ProposalConfidenceBand.HIGH,
                    requested_task=TaskName.APPOINTMENT_REQUEST,
                    slots=[
                        ProposedSlot(name="care_setting", value="hospital", source="user_explicit", confidence=1),
                        ProposedSlot(name="location", value="Lagos", source="user_explicit", confidence=1),
                        ProposedSlot(name="appointment_reason", value="general consultation", source="user_explicit", confidence=1),
                    ],
                )
            elif "option 2" in normalized and isinstance(state.last_operation_result, AvailabilityResultData):
                slots = [
                    ProposedSlot(
                        name="preferred_time",
                        value=state.last_operation_result.slots[1].start_at,
                        source="user_explicit",
                        confidence=1,
                    )
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
            elif "my name is" in normalized:
                slots = [ProposedSlot(name="caller_name", value="Alex Morgan", source="user_explicit", confidence=1)]
                proposal = ConversationProposal(
                    session_id=state.session_id,
                    based_on_state_version=state.state_version,
                    correlation_id=correlation_id,
                    intent=IntentName.GENERAL_CONVERSATION,
                    dialogue_act=DialogueAct.INFORM,
                    confidence_band=ProposalConfidenceBand.HIGH,
                    slots=slots,
                )
            elif "+44" in normalized:
                slots = [ProposedSlot(name="callback_number", value="+44 20 1234 5678", source="user_explicit", confidence=1)]
                proposal = ConversationProposal(
                    session_id=state.session_id,
                    based_on_state_version=state.state_version,
                    correlation_id=correlation_id,
                    intent=IntentName.GENERAL_CONVERSATION,
                    dialogue_act=DialogueAct.INFORM,
                    confidence_band=ProposalConfidenceBand.HIGH,
                    slots=slots,
                )
            else:
                proposal = ConversationProposal(
                    session_id=state.session_id,
                    based_on_state_version=state.state_version,
                    correlation_id=correlation_id,
                    intent=IntentName.CONFIRMATION,
                    dialogue_act=DialogueAct.CONFIRM,
                    confidence_band=ProposalConfidenceBand.HIGH,
                )
            return ProposalCompletion(proposal=proposal, model="fake", provider="fake")

    class SchedulingProviderDirectory:
        def search(self, care_setting, location, reason=None):
            return ProviderSearchResult(
                location=location,
                providers=[
                    Provider(
                        provider_id="provider:scheduling",
                        name="Verified Scheduling Clinic",
                        address="1 Scheduling Street, Lagos",
                        category=care_setting,
                        latitude=0,
                        longitude=0,
                        distance_km=1,
                    )
                ],
            )

    try:
        slots = [
            AvailabilitySlotRecord(
                slot_id="configured-1",
                start_at="2026-10-01T09:00:00+01:00",
                end_at="2026-10-01T09:30:00+01:00",
                label="Thursday at 9:00 AM",
            ),
            AvailabilitySlotRecord(
                slot_id="configured-2",
                start_at="2026-10-01T14:00:00+01:00",
                end_at="2026-10-01T14:30:00+01:00",
                label="Thursday at 2:00 PM",
            ),
        ]
        scheduler = MockSchedulingService(slots)
        engine = CanonicalConversationEngine(
            SchedulingProposalSource(),
            SchedulingProviderDirectory(),
            scheduling_service=scheduler,
        )
        session_id = "mock-scheduling-smoke"
        started = engine.handle_turn(session_id, "I want to book a hospital appointment in Lagos for a consultation")
        availability = engine.handle_turn(session_id, "option 1")
        preferred = engine.handle_turn(session_id, "option 2")
        named = engine.handle_turn(session_id, "My name is Alex Morgan")
        phone = engine.handle_turn(session_id, "+44 20 1234 5678")
        submitted = engine.handle_turn(session_id, "Yes, submit it")
        duplicate = scheduler.submit_request(
            idempotency_key="mock-scheduling-idempotency",
            preferred_time=slots[0].start_at,
        )
        duplicate_again = scheduler.submit_request(
            idempotency_key="mock-scheduling-idempotency",
            preferred_time=slots[0].start_at,
        )
        if (
            "verified scheduling clinic" not in started.reply.lower()
            or "demonstration slots" not in availability.reply.lower()
            or "name should" not in preferred.reply.lower()
            or "phone number" not in named.reply.lower()
            or "submit this request" not in phone.reply.lower()
            or "mock scheduler" not in submitted.reply.lower()
            or duplicate.request_reference != duplicate_again.request_reference
            or submitted.state["structured_state"].get("appointment_status") != "completed"
        ):
            results.fail(
                "mock scheduling slice",
                f"unexpected replies={[started.reply, availability.reply, preferred.reply, named.reply, phone.reply, submitted.reply]}",
            )
            return
        results.ok("mock scheduling slice")
    except Exception as exc:
        results.fail("mock scheduling slice", str(exc))


def test_scheduling_configuration(results: Results) -> None:
    previous = os.environ.get("SCHEDULING_PROVIDER")
    try:
        os.environ["SCHEDULING_PROVIDER"] = "unconfigured"
        unavailable = build_scheduling_service()
        result = unavailable.get_availability("provider:configuration")
        os.environ["SCHEDULING_PROVIDER"] = "mock"
        mock = build_scheduling_service()
        if (
            not isinstance(unavailable, UnavailableSchedulingService)
            or result.error_code != "not_configured"
            or not isinstance(mock, MockSchedulingService)
        ):
            results.fail("scheduling configuration", "scheduling provider selection did not fail closed")
            return
        results.ok("scheduling configuration")
    except Exception as exc:
        results.fail("scheduling configuration", str(exc))
    finally:
        if previous is None:
            os.environ.pop("SCHEDULING_PROVIDER", None)
        else:
            os.environ["SCHEDULING_PROVIDER"] = previous


def test_google_calendar_adapter(results: Results) -> None:
    env_names = (
        "GOOGLE_CALENDAR_ACCESS_TOKEN",
        "GOOGLE_CALENDAR_ID",
        "SCHEDULING_TIMEZONE",
        "SCHEDULING_WINDOW_DAYS",
        "SCHEDULING_SLOT_MINUTES",
        "SCHEDULING_WORKDAY_START",
        "SCHEDULING_WORKDAY_END",
    )
    previous = {name: os.environ.get(name) for name in env_names}
    requests = []

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def request(self, method, url, params=None, json=None):
            requests.append({"method": method, "url": url, "params": params, "json": json})
            request = httpx.Request(method, url)
            if url.endswith("/freeBusy"):
                return httpx.Response(
                    200,
                    json={
                        "calendars": {
                            "primary": {
                                "busy": [
                                    {
                                        "start": "2026-10-01T09:00:00+00:00",
                                        "end": "2026-10-01T09:30:00+00:00",
                                    }
                                ]
                            }
                        }
                    },
                    request=request,
                )
            return httpx.Response(200, json={"id": "event-123"}, request=request)

    try:
        os.environ.update(
            {
                "GOOGLE_CALENDAR_ACCESS_TOKEN": "test-token",
                "GOOGLE_CALENDAR_ID": "primary",
                "SCHEDULING_TIMEZONE": "UTC",
                "SCHEDULING_WINDOW_DAYS": "1",
                "SCHEDULING_SLOT_MINUTES": "30",
                "SCHEDULING_WORKDAY_START": "09:00",
                "SCHEDULING_WORKDAY_END": "10:00",
            }
        )
        service = GoogleCalendarSchedulingService(
            http_client_factory=lambda **kwargs: FakeClient(),
            now=lambda: datetime(2026, 10, 1, 8, 0, tzinfo=timezone.utc),
        )
        availability = service.get_availability("provider:calendar")
        if (
            availability.error
            or len(availability.slots) != 1
            or availability.slots[0].start_at != "2026-10-01T09:30:00+00:00"
        ):
            results.fail("google calendar adapter", f"busy-time filtering failed: {availability}")
            return
        submission = service.submit_request(
            idempotency_key="calendar-idempotency",
            preferred_time=availability.slots[0].start_at,
            details=AppointmentDetails(
                provider_id="provider:calendar",
                preferred_time=availability.slots[0].start_at,
                caller_name="Alex Morgan",
                callback_number="+44 20 1234 5678",
                appointment_reason="consultation",
            ),
        )
        if (
            submission.status != "submitted"
            or submission.source != "google_calendar"
            or submission.request_reference != "google:event-123"
            or len(requests) != 2
            or requests[1]["json"]["extendedProperties"]["private"]["triageos_idempotency_key"]
            != "calendar-idempotency"
        ):
            results.fail("google calendar adapter", f"event submission failed: {submission}")
            return
        results.ok("google calendar adapter")
    except Exception as exc:
        results.fail("google calendar adapter", str(exc))
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_reset_and_correction_boundaries(results: Results) -> None:
    try:
        captured_at = datetime.now(timezone.utc)
        state = DomainConversationState(
            session_id="reset-correction-boundary",
            active_task=TaskName.APPOINTMENT_REQUEST,
            workflow_state=WorkflowState.COLLECTING_DETAILS,
            slots={
                "care_setting": SlotRecord(
                    value="hospital",
                    source=SlotSource.USER_EXPLICIT,
                    confidence=1,
                    event_id="care",
                    captured_at=captured_at,
                ),
                "location": SlotRecord(
                    value="Garki, Abuja",
                    source=SlotSource.USER_EXPLICIT,
                    confidence=1,
                    event_id="location",
                    captured_at=captured_at,
                ),
                "provider_id": SlotRecord(
                    value="provider:one",
                    source=SlotSource.USER_EXPLICIT,
                    confidence=1,
                    event_id="provider-id",
                    captured_at=captured_at,
                ),
                "provider_name": SlotRecord(
                    value="First Hospital",
                    source=SlotSource.USER_EXPLICIT,
                    confidence=1,
                    event_id="provider-name",
                    captured_at=captured_at,
                ),
            },
            provider_options=[
                ProviderResultRecord(
                    provider_id="provider:one",
                    name="First Hospital",
                    address="1 First Street",
                    category="hospital",
                ),
                ProviderResultRecord(
                    provider_id="provider:two",
                    name="Second Hospital",
                    address="2 Second Street",
                    category="hospital",
                ),
            ],
        )

        provider_correction = build_recovery_proposal(
            "Actually, option number 2 instead.", state, "provider-correction"
        )
        if {correction.name for correction in provider_correction.corrections} != {
            "provider_id",
            "provider_name",
        }:
            results.fail("reset and correction boundaries", "provider correction was not extracted")
            return

        working_state = state
        validated = validate_proposal(working_state, provider_correction)
        for command in validated.commands:
            event = command_to_event(working_state, command)
            working_state = reduce_state(working_state, event)
        location_correction = build_recovery_proposal(
            "Actually, I meant Ibadan instead.", working_state, "location-correction"
        )
        validated = validate_proposal(working_state, location_correction)
        for command in validated.commands:
            event = command_to_event(working_state, command)
            working_state = reduce_state(working_state, event)
        if (
            working_state.slots.get("location").value != "Ibadan"
            or working_state.provider_options
            or "provider_id" in working_state.slots
            or working_state.workflow_state != WorkflowState.COLLECTING_CONTEXT
        ):
            results.fail("reset and correction boundaries", "location correction retained stale provider context")
            return

        cancellation = ConversationProposal(
            session_id=working_state.session_id,
            based_on_state_version=working_state.state_version,
            correlation_id="cancel-boundary",
            intent=IntentName.APPOINTMENT_CANCELLATION,
            dialogue_act=DialogueAct.CANCEL,
            confidence_band=ProposalConfidenceBand.HIGH,
            cancel_requested=True,
        )
        validated = validate_proposal(working_state, cancellation)
        for command in validated.commands:
            event = command_to_event(working_state, command)
            working_state = reduce_state(working_state, event)
        if (
            working_state.active_task != TaskName.NONE
            or working_state.workflow_state != WorkflowState.CANCELLED
            or working_state.slots
            or working_state.provider_options
        ):
            results.fail("reset and correction boundaries", "cancellation did not clear session context")
            return
        results.ok("reset and correction boundaries")
    except Exception as exc:
        results.fail("reset and correction boundaries", str(exc))


def test_provider_lookup_boundary(results: Results) -> None:
    try:
        cases = (
            ("Find a hospital near Abuja, Nigeria.", "hospital", "Abuja, Nigeria"),
            ("I'm in Greater London and I want to find a clinic.", "clinic", "Greater London"),
            ("Could you look for a doctor in Toronto?", "doctor", "Toronto"),
            ("I live around Nairobi and need a hospital.", "hospital", "Nairobi"),
        )
        for index, (text, expected_care, expected_location) in enumerate(cases):
            proposal = build_recovery_proposal(
                text,
                DomainConversationState(session_id=f"provider-lookup-boundary-{index}"),
                f"provider-lookup-correlation-{index}",
            )
            slots = {slot.name: slot.value for slot in proposal.slots}
            if (
                proposal.intent != IntentName.PROVIDER_LOOKUP
                or proposal.requested_task != TaskName.PROVIDER_LOOKUP
                or slots != {"care_setting": expected_care, "location": expected_location}
            ):
                results.fail("provider lookup boundary", f"unexpected proposal={proposal}")
                return
        results.ok("provider lookup boundary")
    except Exception as exc:
        results.fail("provider lookup boundary", str(exc))


def test_provider_search_reliability(results: Results) -> None:
    class RetryProposalSource:
        def propose(self, user_text, state, recent_messages, correlation_id):
            if "try again" in user_text.lower():
                proposal = ConversationProposal(
                    session_id=state.session_id,
                    based_on_state_version=state.state_version,
                    correlation_id=correlation_id,
                    intent=IntentName.PROVIDER_LOOKUP,
                    dialogue_act=DialogueAct.INFORM,
                    confidence_band=ProposalConfidenceBand.HIGH,
                    tool_selection=ToolSelectionProposal(operation=OperationName.SEARCH_PROVIDERS),
                )
            else:
                proposal = ConversationProposal(
                    session_id=state.session_id,
                    based_on_state_version=state.state_version,
                    correlation_id=correlation_id,
                    intent=IntentName.PROVIDER_LOOKUP,
                    dialogue_act=DialogueAct.INFORM,
                    confidence_band=ProposalConfidenceBand.HIGH,
                    requested_task=TaskName.PROVIDER_LOOKUP,
                    slots=[
                        ProposedSlot(name="care_setting", value="hospital", source="user_explicit", confidence=1),
                        ProposedSlot(name="location", value="Lagos", source="user_explicit", confidence=1),
                    ],
                )
            return ProposalCompletion(proposal=proposal, model="fake", provider="fake")

    class FlakyProviderDirectory:
        def __init__(self):
            self.calls = 0

        def search(self, care_setting, location, reason=None):
            self.calls += 1
            if self.calls == 1:
                return ProviderSearchResult(
                    location=location,
                    providers=[],
                    error="temporary outage",
                    error_code="unavailable",
                )
            return ProviderSearchResult(
                location=location,
                providers=[
                    Provider(
                        provider_id="retry:1",
                        name="Verified Retry Hospital",
                        address="1 Retry Street, Lagos",
                        category=care_setting,
                        latitude=0,
                        longitude=0,
                        distance_km=1,
                    )
                ],
            )

    class EmptyProviderDirectory:
        def search(self, care_setting, location, reason=None):
            return ProviderSearchResult(location=location, providers=[])

    try:
        flaky = FlakyProviderDirectory()
        engine = CanonicalConversationEngine(RetryProposalSource(), flaky)
        first = engine.handle_turn("provider-retry", "Find a hospital near Lagos")
        second = engine.handle_turn("provider-retry", "try again")
        if (
            "try again" not in first.reply.lower()
            or "verified retry hospital" not in second.reply.lower()
            or flaky.calls != 2
        ):
            results.fail("provider search reliability", f"retry flow failed: first={first.reply} second={second.reply}")
            return

        empty = CanonicalConversationEngine(RetryProposalSource(), EmptyProviderDirectory())
        no_match = empty.handle_turn("provider-no-match", "Find a hospital near Lagos")
        if "couldn’t find a verified provider" not in no_match.reply.lower():
            results.fail("provider search reliability", f"no-match response was misclassified: {no_match.reply}")
            return

        class ResumeConversationSource:
            def propose(self, user_text, state, recent_messages, correlation_id):
                if "find" in user_text.lower():
                    proposal = ConversationProposal(
                        session_id=state.session_id,
                        based_on_state_version=state.state_version,
                        correlation_id=correlation_id,
                        intent=IntentName.PROVIDER_LOOKUP,
                        dialogue_act=DialogueAct.INFORM,
                        confidence_band=ProposalConfidenceBand.HIGH,
                        requested_task=TaskName.PROVIDER_LOOKUP,
                        slots=[
                            ProposedSlot(name="care_setting", value="hospital", source="user_explicit", confidence=1),
                            ProposedSlot(name="location", value="Lagos", source="user_explicit", confidence=1),
                        ],
                    )
                else:
                    proposal = ConversationProposal(
                        session_id=state.session_id,
                        based_on_state_version=state.state_version,
                        correlation_id=correlation_id,
                        intent=IntentName.GENERAL_CONVERSATION,
                        dialogue_act=DialogueAct.INFORM,
                        confidence_band=ProposalConfidenceBand.HIGH,
                        response_draft="Of course. What would you like to talk about?",
                    )
                return ProposalCompletion(proposal=proposal, model="fake", provider="fake")

        class AlwaysFailProviderDirectory:
            def search(self, care_setting, location, reason=None):
                return ProviderSearchResult(
                    location=location,
                    providers=[],
                    error="persistent outage",
                    error_code="unavailable",
                )

        resumed_engine = CanonicalConversationEngine(ResumeConversationSource(), AlwaysFailProviderDirectory())
        resumed_engine.handle_turn("provider-resume", "Find a hospital near Lagos")
        resumed = resumed_engine.handle_turn("provider-resume", "hello, I have another question")
        if resumed.reply != "Of course. What would you like to talk about?":
            results.fail("provider search reliability", f"ordinary conversation was trapped: {resumed.reply}")
            return

        directory = ProviderDirectory()
        directory._geocode = lambda location, deadline=None: (6.5, 3.4, location)
        directory._search_openstreetmap = lambda *args: (_ for _ in ()).throw(
            ProviderDirectoryError("temporary directory outage")
        )
        fallback_calls = 0

        def fallback_search(**kwargs):
            nonlocal fallback_calls
            fallback_calls += 1
            if fallback_calls == 1:
                raise ProviderDirectoryError("temporary directory outage")
            return [
                Provider(
                    provider_id="recovered:1",
                    name="Recovered Hospital",
                    address="Lagos",
                    category="hospital",
                    latitude=6.51,
                    longitude=3.41,
                    distance_km=1,
                )
            ]

        directory._search_nominatim_providers = fallback_search
        first = directory.search("hospital", "Lagos")
        second = directory.search("hospital", "Lagos")
        if first.error_code != "unavailable" or not second.providers or fallback_calls != 2:
            results.fail("provider search reliability", "transient directory failure was cached")
            return

        class FakeClient:
            def __init__(self, request_fn):
                self.request_fn = request_fn

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def request(self, method, url, params=None, data=None):
                return self.request_fn(method, url, params, data)

        retry_directory = ProviderDirectory()
        retry_directory._max_attempts = 2
        retry_directory._retry_backoff_seconds = 0
        retry_attempts = 0

        def retry_request(method, url, params, data):
            nonlocal retry_attempts
            retry_attempts += 1
            if retry_attempts == 1:
                raise httpx.ConnectError("temporary network failure", request=httpx.Request(method, url))
            return httpx.Response(
                200,
                json={"elements": []},
                request=httpx.Request(method, url),
            )

        retry_directory._http_client_factory = lambda **kwargs: FakeClient(retry_request)
        payload = retry_directory._request_json(
            "POST",
            "https://directory.test/search",
            headers={},
            data={"data": "query"},
            timeout_seconds=1,
            deadline=None,
        )
        if retry_attempts != 2 or payload != {"elements": []}:
            results.fail("provider transport resilience", "transient failure did not recover")
            return

        circuit_directory = ProviderDirectory()
        circuit_directory._max_attempts = 1
        circuit_directory._circuit_failure_threshold = 1
        circuit_calls = 0

        def failing_request(method, url, params, data):
            nonlocal circuit_calls
            circuit_calls += 1
            raise httpx.ConnectError("persistent network failure", request=httpx.Request(method, url))

        circuit_directory._http_client_factory = lambda **kwargs: FakeClient(failing_request)
        for _ in range(2):
            try:
                circuit_directory._request_json(
                    "GET",
                    "https://circuit.test/search",
                    headers={},
                    timeout_seconds=1,
                    deadline=None,
                )
            except ProviderDirectoryError:
                pass
        circuit_health = circuit_directory.health()
        circuit_status = circuit_health["endpoints"]["circuit.test"]["status"]
        if circuit_calls != 1 or circuit_status != "open":
            results.fail("provider transport resilience", "circuit breaker did not stop repeated failures")
            return
        results.ok("provider search reliability")
        results.ok("provider transport resilience")
    except Exception as exc:
        results.fail("provider search reliability", str(exc))


def test_provider_option_matching(results: Results) -> None:
    try:
        options = [
            ProviderResultRecord(
                provider_id="provider:1",
                name="Eye Foundation Hospital",
                address="648 Mobolaji Johnson Street, Abuja",
                category="hospital",
            ),
            ProviderResultRecord(
                provider_id="provider:2",
                name="CedarCrest Abuja Hospital",
                address="2 Ahmad Daku Street, Abuja",
                category="hospital",
            ),
        ]
        references = (
            "cedar crest sounds good",
            "I would like to use option number 2",
            "can you get me more details about cedarcrest abuja",
        )
        if not all(resolve_provider_reference(text, options) == options[1] for text in references):
            results.fail("provider option matching", "informal provider references did not resolve")
            return

        class OptionProposalSource:
            def propose(self, user_text, state, recent_messages, correlation_id):
                proposal = ConversationProposal(
                    session_id=state.session_id,
                    based_on_state_version=state.state_version,
                    correlation_id=correlation_id,
                    intent=IntentName.PROVIDER_LOOKUP if state.turn_count == 1 else IntentName.GENERAL_CONVERSATION,
                    dialogue_act=DialogueAct.INFORM,
                    confidence_band=ProposalConfidenceBand.HIGH,
                    requested_task=TaskName.PROVIDER_LOOKUP if state.turn_count == 1 else TaskName.NONE,
                    slots=(
                        [
                            ProposedSlot(name="care_setting", value="hospital", source="user_explicit", confidence=1),
                            ProposedSlot(name="location", value="Lagos", source="user_explicit", confidence=1),
                        ]
                        if state.turn_count == 1
                        else []
                    ),
                )
                return ProposalCompletion(proposal=proposal, model="fake", provider="fake")

        class OptionDirectory:
            def search(self, care_setting, location, reason=None):
                return ProviderSearchResult(
                    location=location,
                    providers=[
                        Provider(
                            provider_id="provider:1",
                            name="Eye Foundation Hospital",
                            address="648 Mobolaji Johnson Street, Lagos",
                            category=care_setting,
                            latitude=0,
                            longitude=0,
                            distance_km=1,
                        ),
                        Provider(
                            provider_id="provider:2",
                            name="CedarCrest Abuja Hospital",
                            address="2 Ahmad Daku Street, Abuja",
                            category=care_setting,
                            latitude=0,
                            longitude=0,
                            distance_km=2,
                            phone="+234 0809 515 7906",
                        ),
                    ],
                )

        engine = CanonicalConversationEngine(OptionProposalSource(), OptionDirectory())
        engine.handle_turn("provider-semantics", "Find a hospital near Lagos")
        details = engine.handle_turn(
            "provider-semantics",
            "The second one sounds good, but can you tell me a bit about it first?",
        )
        selected = engine.handle_turn("provider-semantics", "The second one sounds good")
        phone = engine.handle_turn("provider-semantics", "Can you give me their phone number?")
        if (
            "verified directory details" not in details.reply.lower()
            or details.state["structured_state"].get("provider_name")
            or "selected cedarcrest abuja hospital" not in selected.reply.lower()
            or "+234 0809 515 7906" not in phone.reply
        ):
            results.fail("provider option matching", "details and selection semantics were conflated")
            return
        results.ok("provider option matching")
    except Exception as exc:
        results.fail("provider option matching", str(exc))


def test_workflow_contracts(results: Results) -> None:
    try:
        state = ConversationState(session_id="workflow_smoke")
        intent = IntentClassification(name=IntentName.APPOINTMENT_REQUEST, confidence=1)
        entities = {
            name: ExtractedSlot(value=value, confidence=1)
            for name, value in {
                "care_setting": "primary care clinic",
                "location": "Lagos",
                "appointment_reason": "general consultation",
            }.items()
        }
        review = apply_receptionist_turn(state, intent, entities)
        review.state.provider_options = [
            {
                "provider_id": "osm:node:123",
                "name": "Example Clinic",
                "address": "1 Main Street",
            }
        ]
        review.state.workflow.next_action = "select_provider"
        selected = apply_receptionist_turn(
            review.state,
            IntentClassification(name=IntentName.GENERAL_CONVERSATION, confidence=1),
            {
                "provider_name": ExtractedSlot(value="Example Clinic", confidence=1),
                "provider_id": ExtractedSlot(value="osm:node:123", confidence=1),
            },
        )
        if (
            review.state.workflow.state == ReceptionistState.COLLECTING_DETAILS
            and review.next_action == "search_providers"
            and review.tool_call
            and review.tool_call.name == "search_providers"
            and "caller_name" not in review.state.slots
            and selected.next_action == "collect_preferred_time"
        ):
            results.ok("workflow contract transitions")
        else:
            results.fail("workflow contract transitions", "unexpected workflow decision")
    except Exception as exc:
        results.fail("workflow contract transitions", str(exc))


def test_factual_routing_boundaries(results: Results) -> None:
    cases = [
        ("What is a good nearby hospital I could go to?", "provider_lookup", "city"),
        ("Someone told me about a clinic called Vetlane. How close is it to me?", "provider_lookup", "verify Vetlane"),
        ("Is there a nearby supermarket?", "unsupported_local_search", "healthcare-related"),
    ]
    intent_cases = [
        ("What services can TriageOS help with today?", "capabilities"),
        ("What can you help me with?", "capabilities"),
        ("What services does the clinic offer?", "practice_information"),
        ("Does the hospital accept insurance?", "practice_information"),
    ]
    try:
        for text, expected_intent in intent_cases:
            intent = detect_intent(text)
            if intent.name != expected_intent:
                results.fail("factual routing boundaries", f"misclassified {text}: {intent}")
                return
        for text, expected_intent, expected_phrase in cases:
            intent = detect_intent(text)
            response = generate_reply(text, [], {})
            if intent.name != expected_intent or expected_phrase.lower() not in response["reply"].lower():
                results.fail("factual routing boundaries", f"unexpected result for {text}: {response}")
                return
        guarded = _guard_generated_reply(
            CompletionResult(
                text="I can give directions to our clinic.",
                model="test",
                provider="test",
                reason="test",
            ),
            {},
        )
        if guarded.provider != "policy" or "verified practice" not in guarded.text.lower():
            results.fail("factual routing boundaries", f"unverified claim was not suppressed: {guarded}")
            return
        if _extract_location_hint("I need a hospital near Garki, Abuja") != "Garki, Abuja":
            results.fail("factual routing boundaries", "comma-separated location was not extracted")
            return
        if _extract_location_hint(
            "I live within central area, Abuja, Nigeria, so just somewhere good."
        ) != "central area, Abuja, Nigeria":
            results.fail("factual routing boundaries", "location sentence was not cleaned")
            return
        if _extract_location_hint("I need a clinic near me") is not None:
            results.fail("factual routing boundaries", "near me was treated as a literal location")
            return
        results.ok("factual routing boundaries")
    except Exception as exc:
        results.fail("factual routing boundaries", str(exc))


async def recv_json(ws: websockets.ClientConnection, timeout: float = 5.0) -> dict:
    raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
    return json.loads(raw)


async def test_health(results: Results) -> None:
    async with httpx.AsyncClient() as client:
        for name, url in [("voice health", f"{VOICE_URL}/health"), ("conversation health", f"{CONVERSATION_URL}/health")]:
            try:
                response = await client.get(url, timeout=5)
                if response.status_code == 200 and response.json().get("status") == "ok":
                    results.ok(name)
                else:
                    results.fail(name, f"unexpected response: {response.status_code} {response.text}")
            except Exception as exc:
                results.fail(name, str(exc))


async def test_web_client(results: Results) -> None:
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{VOICE_URL}/", timeout=5)
            if response.status_code == 200 and "TriageOS" in response.text:
                results.ok("web client index")
            else:
                results.fail("web client index", f"status={response.status_code}")
        except Exception as exc:
            results.fail("web client index", str(exc))


async def test_conversation_turn(results: Results) -> None:
    session_id = f"smoke_{uuid.uuid4()}"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{CONVERSATION_URL}/v1/conversations/{session_id}/turn",
                json={"session_id": session_id, "text": "hello there"},
                timeout=10,
            )
            response.raise_for_status()
            body = response.json()
            if body.get("reply") and body.get("session_id") == session_id:
                results.ok("conversation turn (greeting)")
            else:
                results.fail("conversation turn (greeting)", f"unexpected body: {body}")

            response2 = await client.post(
                f"{CONVERSATION_URL}/v1/conversations/{session_id}/turn",
                json={"session_id": session_id, "text": "tell me about scheduling"},
                timeout=10,
            )
            response2.raise_for_status()
            state = response2.json().get("state", {})
            messages = state.get("recent_messages", [])
            if len(messages) >= 4:
                results.ok("conversation state persistence")
            else:
                results.fail("conversation state persistence", f"expected >=4 messages, got {len(messages)}")
        except Exception as exc:
            results.fail("conversation turn", str(exc))


async def test_urgent_safety_response(results: Results) -> None:
    session_id = f"safety_{uuid.uuid4()}"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{CONVERSATION_URL}/v1/conversations/{session_id}/turn",
                json={"session_id": session_id, "text": "I am having chest pains"},
                timeout=10,
            )
            response.raise_for_status()
            body = response.json()
            reply = body.get("reply", "").lower()
            usage = body.get("usage", {})
            intent = usage.get("intent", {}).get("name")
            if (
                intent == "urgent_safety"
                and usage.get("provider") == "guardrail"
                and "emergency" in reply
                and "do not drive" in reply
            ):
                results.ok("urgent safety escalation")
            else:
                results.fail("urgent safety escalation", f"unexpected response: {body}")
        except Exception as exc:
            results.fail("urgent safety escalation", str(exc))


async def test_receptionist_appointment_flow(results: Results) -> None:
    session_id = f"receptionist_{uuid.uuid4()}"
    turns = [
        "I need to book an appointment",
        "veterinary care",
        "Lagos",
    ]
    async with httpx.AsyncClient() as client:
        try:
            body = None
            for text in turns:
                response = await client.post(
                    f"{CONVERSATION_URL}/v1/conversations/{session_id}/turn",
                    json={"session_id": session_id, "text": text},
                    timeout=10,
                )
                response.raise_for_status()
                body = response.json()

            state = body.get("state", {}).get("structured_state", {})
            usage = body.get("usage", {})
            if (
                usage.get("provider") == "workflow"
                and state.get("appointment_status") == "collecting_details"
                and state.get("care_setting") == "veterinary care"
                and state.get("location") == "Lagos"
                and state.get("next_action") == "collect_appointment_reason"
                and "caller_name" not in state
            ):
                results.ok("receptionist appointment context")
            else:
                results.fail("receptionist appointment context", f"unexpected response: {body}")
        except Exception as exc:
            results.fail("receptionist appointment request", str(exc))


async def test_receptionist_quality_boundaries(results: Results) -> None:
    async with httpx.AsyncClient() as client:
        try:
            appointment_session = f"clarification_{uuid.uuid4()}"
            for text in [
                "I need to book an appointment",
                "veterinary care",
                "What clinic are you talking about?",
            ]:
                clarification_response = await client.post(
                    f"{CONVERSATION_URL}/v1/conversations/{appointment_session}/turn",
                    json={"session_id": appointment_session, "text": text},
                    timeout=10,
                )
                clarification_response.raise_for_status()
            clarification_body = clarification_response.json()
            clarification_reply = clarification_body.get("reply", "").lower()

            if (
                "appointment context" in clarification_reply
                and "caller name" not in clarification_reply
                and "caller_name" not in clarification_body.get("state", {}).get("structured_state", {})
            ):
                results.ok("receptionist provider and clarification boundaries")
            else:
                results.fail(
                    "receptionist provider and clarification boundaries",
                    f"unexpected clarification={clarification_body}",
                )
        except Exception as exc:
            results.fail("receptionist domain and clarification boundaries", str(exc))


async def test_websocket_session_lifecycle(results: Results) -> None:
    session_id = f"smoke_{uuid.uuid4()}"
    uri = f"{WS_BASE}/v1/voice/sessions/{session_id}/stream"
    try:
        async with websockets.connect(uri) as ws:
            started = await recv_json(ws)
            if started.get("event") != "voice.session.started":
                results.fail("websocket session.started", f"got {started}")
                return
            if not started.get("timestamp"):
                results.fail("server event timestamp", "voice.session.started missing timestamp")
                return
            results.ok("websocket session.started")
            results.ok("server event timestamp")

            await ws.send(
                json.dumps(
                    {
                        "event": "voice.user.transcript.final",
                        "session_id": session_id,
                        "payload": {"text": "hello", "stt_ms": 420},
                    }
                )
            )

            response = await recv_json(ws)
            if response.get("event") != "voice.assistant.response.created":
                results.fail("websocket assistant response", f"got {response}")
                return

            payload = response.get("payload", {})
            latency = payload.get("latency", {})
            if (
                payload.get("text")
                and payload.get("turn_id", 0) > 0
                and payload.get("tts", {}).get("provider") == "browser"
                and latency.get("llm_ms") is not None
                and latency.get("stt_ms") == 420
            ):
                results.ok("websocket assistant response")
                results.ok("turn latency payload")
            else:
                results.fail("websocket assistant response", f"unexpected payload: {payload}")

            await ws.send(
                json.dumps(
                    {
                        "event": "voice.session.ended",
                        "session_id": session_id,
                        "payload": {},
                    }
                )
            )
            ended = await recv_json(ws)
            if ended.get("event") == "voice.session.ended":
                results.ok("websocket session.ended")
            else:
                results.fail("websocket session.ended", f"got {ended}")
    except Exception as exc:
        results.fail("websocket session lifecycle", str(exc))


async def test_websocket_partial_transcript(results: Results) -> None:
    session_id = f"smoke_{uuid.uuid4()}"
    uri = f"{WS_BASE}/v1/voice/sessions/{session_id}/stream"
    try:
        async with websockets.connect(uri) as ws:
            await recv_json(ws)  # session.started

            await ws.send(
                json.dumps(
                    {
                        "event": "voice.user.transcript.partial",
                        "session_id": session_id,
                        "payload": {"text": "hel"},
                    }
                )
            )
            partial = await recv_json(ws)
            if partial.get("event") == "voice.user.transcript.partial":
                results.ok("websocket partial transcript echo")
            else:
                results.fail("websocket partial transcript echo", f"got {partial}")
    except Exception as exc:
        results.fail("websocket partial transcript", str(exc))


async def test_websocket_audio_chunk_ack(results: Results) -> None:
    session_id = f"smoke_{uuid.uuid4()}"
    uri = f"{WS_BASE}/v1/voice/sessions/{session_id}/stream"
    try:
        async with websockets.connect(uri) as ws:
            await recv_json(ws)

            await ws.send(
                json.dumps(
                    {
                        "event": "voice.audio.chunk",
                        "session_id": session_id,
                        "payload": {"seq": 1, "bytes": 0},
                    }
                )
            )
            ack = await recv_json(ws)
            if ack.get("event") == "voice.audio.chunk.ack":
                results.ok("websocket audio chunk ack")
            else:
                results.fail("websocket audio chunk ack", f"got {ack}")
    except Exception as exc:
        results.fail("websocket audio chunk ack", str(exc))


async def test_interruption_layer(results: Results) -> None:
    """Interruption is foundation-critical — verify cancel propagates and turns stay consistent."""
    session_id = f"smoke_{uuid.uuid4()}"
    uri = f"{WS_BASE}/v1/voice/sessions/{session_id}/stream"
    try:
        async with websockets.connect(uri) as ws:
            await recv_json(ws)

            await ws.send(
                json.dumps(
                    {
                        "event": "voice.interruption.detected",
                        "session_id": session_id,
                        "payload": {},
                    }
                )
            )
            interruption = await recv_json(ws)
            if interruption.get("event") != "voice.interruption.detected":
                results.fail("interruption event echo", f"got {interruption}")
                return
            if interruption.get("payload", {}).get("turn_seq", 0) <= 0:
                results.fail("interruption turn_seq", f"missing turn_seq in {interruption}")
                return
            results.ok("interruption event echo (server acknowledges)")
            results.ok("interruption turn_seq in payload")

            await ws.send(
                json.dumps(
                    {
                        "event": "voice.user.transcript.final",
                        "session_id": session_id,
                        "payload": {"text": "wait stop, I meant something else", "stt_ms": 300},
                    }
                )
            )
            response = await recv_json(ws)
            if response.get("event") == "voice.assistant.response.created":
                results.ok("turn after interruption (no server block)")
            else:
                results.fail("turn after interruption", f"got {response}")

            results.warn(
                "interruption — automatic barge-in",
                "no always-on mic / VAD; user must press Start Talking to interrupt",
            )
    except Exception as exc:
        results.fail("interruption layer", str(exc))


async def test_stale_turn_superseded(results: Results) -> None:
    """A rapid second final must invalidate the first in-flight turn."""
    session_id = f"smoke_{uuid.uuid4()}"
    uri = f"{WS_BASE}/v1/voice/sessions/{session_id}/stream"
    try:
        async with websockets.connect(uri) as ws:
            await recv_json(ws)

            await ws.send(
                json.dumps(
                    {
                        "event": "voice.user.transcript.final",
                        "session_id": session_id,
                        "payload": {"text": "first utterance"},
                    }
                )
            )
            await ws.send(
                json.dumps(
                    {
                        "event": "voice.user.transcript.final",
                        "session_id": session_id,
                        "payload": {"text": "second utterance"},
                    }
                )
            )

            responses: list[dict] = []
            while len(responses) < 2:
                try:
                    message = await recv_json(ws, timeout=3.0)
                except TimeoutError:
                    break
                if message.get("event") == "voice.assistant.response.created":
                    responses.append(message)

            if len(responses) != 1:
                results.fail("stale turn superseded", f"expected 1 response, got {len(responses)}")
                return

            results.ok("stale turn superseded")
    except Exception as exc:
        results.fail("stale turn superseded", str(exc))


async def test_invalid_event(results: Results) -> None:
    session_id = f"smoke_{uuid.uuid4()}"
    uri = f"{WS_BASE}/v1/voice/sessions/{session_id}/stream"
    try:
        async with websockets.connect(uri) as ws:
            await recv_json(ws)
            await ws.send(json.dumps({"event": "voice.session.started", "session_id": session_id, "payload": {}}))
            # voice.session.started is valid ClientEventType but not handled — should be silently ignored
            # Send malformed payload instead
            await ws.send(json.dumps({"event": "not.a.real.event", "session_id": session_id, "payload": {}}))
            error = await recv_json(ws)
            if error.get("event") == "voice.error":
                results.ok("invalid event returns voice.error")
            else:
                results.fail("invalid event handling", f"got {error}")
    except Exception as exc:
        results.fail("invalid event handling", str(exc))


async def main() -> int:
    results = Results()
    print("\nTriageOS Voice and Receptionist Smoke Tests\n" + "=" * 44)

    test_domain_contracts(results)
    test_transition_table(results)
    test_tool_and_persistence_contracts(results)
    test_proposal_validation(results)
    test_structured_proposal_adapter(results)
    test_orchestration_pipeline(results)
    test_canonical_conversation_engine(results)
    test_practice_profile_slice(results)
    test_mock_scheduling_slice(results)
    test_scheduling_configuration(results)
    test_google_calendar_adapter(results)
    test_reset_and_correction_boundaries(results)
    test_provider_lookup_boundary(results)
    test_provider_search_reliability(results)
    test_provider_option_matching(results)
    test_workflow_contracts(results)
    test_factual_routing_boundaries(results)
    await test_health(results)
    await test_web_client(results)
    await test_conversation_turn(results)
    await test_urgent_safety_response(results)
    await test_receptionist_appointment_flow(results)
    await test_receptionist_quality_boundaries(results)
    await test_websocket_session_lifecycle(results)
    await test_websocket_partial_transcript(results)
    await test_websocket_audio_chunk_ack(results)
    await test_interruption_layer(results)
    await test_stale_turn_superseded(results)
    await test_invalid_event(results)

    print("\n" + "=" * 32)
    print(f"Passed: {len(results.passed)}  Failed: {len(results.failed)}  Warnings: {len(results.warnings)}")

    if results.failed:
        print("\nFailures:")
        for item in results.failed:
            print(f"  - {item}")

    if results.warnings:
        print("\nInterruption / foundation gaps:")
        for item in results.warnings:
            print(f"  - {item}")

    return 1 if results.failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
