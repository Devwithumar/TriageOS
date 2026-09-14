"""Deterministic response policy for canonical conversation state."""

from dataclasses import dataclass

from libs.conversation.domain import (
    ConversationState,
    ProviderSearchResultData,
    TaskName,
    WorkflowState,
)
from libs.conversation.proposals import ConversationProposal
from libs.conversation.orchestrator import OrchestrationResult


@dataclass(frozen=True)
class ResponseDecision:
    text: str
    provider: str
    reason: str


_SLOT_PROMPTS = {
    "care_setting": "What kind of care or provider are you looking for?",
    "location": "What city, neighborhood, or postal code should I use?",
    "appointment_reason": "What would you like help with during the visit?",
    "provider_id": "Which provider would you like to use?",
    "preferred_time": "What day or time would you prefer?",
    "caller_name": "What name should I put on the appointment request?",
    "callback_number": "What phone number should the practice use to reach you?",
}

_TASK_SLOTS = {
    TaskName.PROVIDER_LOOKUP: ("care_setting", "location"),
    TaskName.APPOINTMENT_REQUEST: (
        "care_setting",
        "location",
        "appointment_reason",
        "provider_id",
        "preferred_time",
        "caller_name",
        "callback_number",
    ),
}

_UNSAFE_DRAFT_TERMS = (
    "diagnose",
    "medical advice",
    "treat your",
    "prescribe",
    "health-related questions",
    "health related questions",
    "our clinic",
    "our hospital",
    "located at",
    "minutes away",
    "well-rated",
    "well rated",
)


def build_response(result: OrchestrationResult) -> ResponseDecision:
    if result.error:
        return ResponseDecision(
            text="I couldn’t safely apply that message. Could you say it another way?",
            provider="policy",
            reason="proposal rejected or unavailable",
        )

    state = result.session.state
    proposal = result.proposal
    if proposal and proposal.intent.value == "urgent_safety":
        return ResponseDecision(
            text=(
                "Chest pain or breathing difficulty can be serious. Contact your local emergency "
                "services immediately or have someone take you to the nearest emergency department. "
                "Please do not drive yourself."
            ),
            provider="guardrail",
            reason="urgent safety escalation",
        )
    if state.workflow_state == WorkflowState.CANCELLED:
        return ResponseDecision(
            text="Understood. I’ve stopped that request. What would you like help with instead?",
            provider="policy",
            reason="task cancelled",
        )

    operation_response = _operation_response(state)
    if operation_response:
        return operation_response

    if state.active_task != TaskName.NONE:
        workflow_response = _workflow_response(state)
        if workflow_response:
            return workflow_response

    return _general_response(proposal)


def _operation_response(state: ConversationState) -> ResponseDecision | None:
    result = state.last_operation_result
    if isinstance(result, ProviderSearchResultData):
        if result.providers:
            options = "; ".join(
                _format_provider(index, provider.name, provider.address)
                for index, provider in enumerate(result.providers, start=1)
            )
            return ResponseDecision(
                text=(
                    f"I found these verified provider options near {result.location}: {options}. "
                    "Which one would you like to use?"
                ),
                provider="provider_directory",
                reason="rendered verified provider search result",
            )
        if result.error:
            return ResponseDecision(
                text=(
                    "I couldn’t reach the provider directory right now, so I won’t invent a provider "
                    "or address. Would you like to try another location?"
                ),
                provider="provider_directory",
                reason="provider directory failed",
            )
        return ResponseDecision(
            text=(
                f"I couldn’t find a verified provider near {result.location}. "
                "Would you like to try another area or care type?"
            ),
            provider="provider_directory",
            reason="provider directory returned no matches",
        )
    if state.workflow_state == WorkflowState.SEARCHING_PROVIDERS:
        location = state.slots.get("location")
        location_text = f" near {location.value}" if location else ""
        return ResponseDecision(
            text=f"I’m checking the verified provider directory{location_text}. I won’t invent a result.",
            provider="policy",
            reason="provider search pending",
        )
    return None


def _workflow_response(state: ConversationState) -> ResponseDecision | None:
    if state.workflow_state == WorkflowState.SELECTING_PROVIDER:
        if state.provider_options:
            return ResponseDecision(
                text="Which verified provider would you like to use? You can give me its number or name.",
                provider="workflow",
                reason="provider selection required",
            )
        return ResponseDecision(
            text="I need a verified provider result before collecting appointment details.",
            provider="workflow",
            reason="provider selection unavailable",
        )
    required_slots = _TASK_SLOTS.get(state.active_task, ())
    for slot_name in required_slots:
        if slot_name == "provider_id" and state.provider_options and slot_name not in state.slots:
            continue
        if slot_name not in state.slots:
            return ResponseDecision(
                text=_missing_slot_prompt(state, slot_name),
                provider="workflow",
                reason=f"missing slot: {slot_name}",
            )
    if state.workflow_state == WorkflowState.REVIEWING_REQUEST:
        return ResponseDecision(
            text="I have the appointment details. Would you like me to submit this request?",
            provider="workflow",
            reason="appointment request ready for confirmation",
        )
    return None


def _general_response(proposal: ConversationProposal | None) -> ResponseDecision:
    if proposal and proposal.intent.value == "unsupported_local_search":
        return ResponseDecision(
            text=(
                "I can help search for verified healthcare providers, but I can’t search general local "
                "businesses. Would you like to find a clinic, hospital, pharmacy, or another healthcare provider?"
            ),
            provider="policy",
            reason="unsupported non-healthcare local search",
        )
    if proposal and proposal.intent.value == "capabilities":
        return ResponseDecision(
            text=(
                "I can help with healthcare receptionist tasks such as finding verified providers, "
                "preparing appointment requests, and answering configured practice questions. "
                "What would you like to do?"
            ),
            provider="policy",
            reason="capabilities response",
        )
    draft = proposal.response_draft.strip() if proposal and proposal.response_draft else ""
    if draft and not _contains_unsafe_claim(draft):
        return ResponseDecision(text=draft, provider="llm_proposal", reason="validated conversational draft")
    return ResponseDecision(
        text="I’m here to help with healthcare receptionist needs. What would you like to do?",
        provider="policy",
        reason="safe conversational fallback",
    )


def _contains_unsafe_claim(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    return any(term in normalized for term in _UNSAFE_DRAFT_TERMS)


def _missing_slot_prompt(state: ConversationState, slot_name: str) -> str:
    if state.active_task == TaskName.APPOINTMENT_REQUEST and not state.provider_options:
        if slot_name == "care_setting":
            return "I haven’t selected a clinic yet. What kind of care or provider are you looking for?"
        if slot_name == "location":
            return "I haven’t selected a clinic yet. What city, neighborhood, or postal code should I search?"
        if slot_name == "appointment_reason":
            return "I haven’t selected a clinic yet. What would you like help with during the visit?"
    return _SLOT_PROMPTS[slot_name]


def _format_provider(index: int, name: str, address: str | None) -> str:
    address_text = f" at {address}" if address else ""
    return f"{index}. {name}{address_text}"
