"""Deterministic response policy for canonical conversation state."""

from dataclasses import dataclass

from libs.conversation.contracts import IntentName
from libs.conversation.domain import (
    ConversationState,
    AppointmentRequestResultData,
    AvailabilityResultData,
    OperationName,
    OperationSucceededEvent,
    PracticeProfileResultData,
    ProviderSearchResultData,
    SlotCapturedEvent,
    TaskCancelledEvent,
    TaskName,
    WorkflowState,
)
from libs.conversation.proposals import ConversationProposal
from libs.conversation.orchestrator import OrchestrationResult
from libs.conversation.provider_matching import (
    is_provider_details_request,
    resolve_provider_reference,
)
from services.conversation.app.practice_profile import practice_profile_topic


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

_WORKFLOW_INTENTS = {
    IntentName.PROVIDER_LOOKUP,
    IntentName.APPOINTMENT_REQUEST,
    IntentName.APPOINTMENT_CHANGE,
    IntentName.APPOINTMENT_CANCELLATION,
    IntentName.CONFIRMATION,
    IntentName.CORRECTION,
    IntentName.PRACTICE_INFORMATION,
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
    if any(isinstance(event, TaskCancelledEvent) for event in result.events):
        return ResponseDecision(
            text="Understood. I’ve stopped that request. What would you like help with instead?",
            provider="policy",
            reason="task cancelled",
        )
    if proposal and proposal.intent.value == "appointment_cancellation":
        return ResponseDecision(
            text="There isn’t an active request to cancel. What would you like help with?",
            provider="policy",
            reason="no active task to cancel",
        )

    operation_response = _operation_response(result)
    if operation_response:
        return operation_response

    provider_details = _provider_details_response(result)
    if provider_details:
        return provider_details

    provider_selection = _provider_selection_response(result)
    if provider_selection:
        return provider_selection

    provider_failure = _provider_failure_response(result)
    if provider_failure:
        return provider_failure

    if (
        state.active_task != TaskName.NONE
        and proposal is not None
        and proposal.intent not in _WORKFLOW_INTENTS
        and not proposal.slots
        and not proposal.corrections
    ):
        return _general_response(proposal)

    if state.active_task != TaskName.NONE:
        clarification = _workflow_clarification_response(result)
        if clarification:
            return clarification
        workflow_response = _workflow_response(state)
        if workflow_response:
            return workflow_response

    return _general_response(proposal)


def _operation_response(result: OrchestrationResult) -> ResponseDecision | None:
    state = result.session.state
    if not any(isinstance(event, OperationSucceededEvent) for event in result.events):
        return None
    question = result.turn_event.text
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
            if result.error_code == "location_not_found":
                return ResponseDecision(
                    text=(
                        f"I couldn’t locate {result.location} in the directory. "
                        "Could you check the city, neighborhood, or postal code?"
                    ),
                    provider="provider_directory",
                    reason="provider directory location not found",
                )
            if result.error_code == "timeout":
                return ResponseDecision(
                    text=(
                        "The verified provider directory timed out. You can say ‘try again’ "
                        "or give me a different location."
                    ),
                    provider="provider_directory",
                    reason="provider directory timed out",
                )
            if result.error_code == "misconfigured":
                return ResponseDecision(
                    text="The verified provider directory is not configured yet, so I can’t search it safely.",
                    provider="provider_directory",
                    reason="provider directory misconfigured",
                )
            return ResponseDecision(
                text=(
                    "I couldn’t reach the verified provider directory right now, so I won’t invent a provider "
                    "or address. You can say ‘try again’ or give me a different location."
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
    if isinstance(result, PracticeProfileResultData):
        if result.error:
            return ResponseDecision(
                text=(
                    "I don’t have a verified practice profile configured yet, so I can’t confirm "
                    "the practice’s hours, location, services, contact details, or insurance policies."
                ),
                provider="practice_profile",
                reason=f"practice profile unavailable: {result.error_code or 'unknown'}",
            )
        return ResponseDecision(
            text=_format_practice_profile(result, question),
            provider="practice_profile",
            reason="rendered configured practice profile",
        )
    if isinstance(result, AvailabilityResultData):
        if result.error:
            if result.error_code == "not_configured":
                return ResponseDecision(
                    text=(
                        "Appointment scheduling is not connected yet, so I can’t confirm availability "
                        "or submit an appointment request."
                    ),
                    provider="scheduling",
                    reason="scheduling provider not configured",
                )
            if result.source != "mock_scheduling":
                return ResponseDecision(
                    text="I couldn’t retrieve appointment availability. No appointment request was submitted.",
                    provider="scheduling",
                    reason="scheduling availability lookup failed",
                )
            return ResponseDecision(
                text="I couldn’t retrieve availability from the mock scheduler. No appointment request was submitted.",
                provider="mock_scheduling",
                reason="mock availability lookup failed",
            )
        if not result.slots:
            return ResponseDecision(
                text="The selected provider has no available mock slots. No appointment request was submitted.",
                provider="mock_scheduling",
                reason="mock scheduler returned no slots",
            )
        options = "; ".join(
            f"{index}. {slot.label}"
            for index, slot in enumerate(result.slots, start=1)
        )
        return ResponseDecision(
            text=f"These are the available demonstration slots: {options}. Which one would you prefer?",
            provider="mock_scheduling",
            reason="rendered mock availability",
        )
    if isinstance(result, AppointmentRequestResultData):
        if result.status == "submitted":
            if result.source == "mock_scheduling":
                return ResponseDecision(
                    text=(
                        f"Your appointment request was submitted to the mock scheduler. Reference: {result.request_reference}. "
                        "This is a demonstration only; no real appointment was booked."
                    ),
                    provider="mock_scheduling",
                    reason="mock appointment request submitted",
                )
            return ResponseDecision(
                text=(
                    f"Your appointment request was submitted to the connected scheduling service. "
                    f"Reference: {result.request_reference}."
                ),
                provider=result.source,
                reason="appointment request submitted",
            )
        return ResponseDecision(
            text="The appointment request could not be submitted, so no appointment was booked.",
            provider="mock_scheduling",
            reason="mock appointment request failed",
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


def _format_practice_profile(profile: PracticeProfileResultData, question: str) -> str:
    topic = practice_profile_topic(question)
    if topic == "hours":
        if not profile.hours:
            return "The configured practice profile does not include opening hours."
        hours = "; ".join(
            (
                f"{entry.day}: closed"
                if entry.closed
                else f"{entry.day}: {entry.opens_at or 'hours unavailable'}–{entry.closes_at or 'hours unavailable'}"
            )
            for entry in profile.hours
        )
        return f"The configured hours for {profile.display_name} are: {hours}."
    if topic == "location":
        return (
            f"{profile.display_name} is listed at {profile.address}."
            if profile.address
            else "The configured practice profile does not include an address."
        )
    if topic == "contact":
        contact = ". ".join(
            value
            for value in (
                f"Phone: {profile.phone}" if profile.phone else None,
                f"Website: {profile.website}" if profile.website else None,
            )
            if value
        )
        return contact or "The configured practice profile does not include contact details."
    if topic == "insurance":
        return (
            f"The configured accepted insurance list is: {', '.join(profile.accepted_insurance)}."
            if profile.accepted_insurance
            else "The configured practice profile does not include accepted insurance information."
        )
    if topic == "services":
        return (
            f"The configured services are: {', '.join(profile.services)}."
            if profile.services
            else "The configured practice profile does not include a services list."
        )
    details = [profile.display_name]
    if profile.address:
        details.append(f"Address: {profile.address}")
    if profile.phone:
        details.append(f"Phone: {profile.phone}")
    return ". ".join(details) + "."


def _provider_failure_response(result: OrchestrationResult) -> ResponseDecision | None:
    state = result.session.state
    operation_result = state.last_operation_result
    if not isinstance(operation_result, ProviderSearchResultData) or not operation_result.error:
        return None
    if (
        result.proposal is None
        or result.proposal.tool_selection is None
        or result.proposal.tool_selection.operation != OperationName.SEARCH_PROVIDERS
    ):
        return None
    if any(isinstance(event, OperationSucceededEvent) for event in result.events):
        return None
    if state.provider_options:
        return None
    if operation_result.error_code == "location_not_found":
        text = (
            f"I couldn’t locate {operation_result.location} in the verified directory. "
            "Could you check the city, neighborhood, or postal code?"
        )
        reason = "provider directory location not found"
    elif operation_result.error_code == "timeout":
        text = "The verified provider directory timed out. You can say ‘try again’ or give me a different location."
        reason = "provider directory timed out"
    elif operation_result.error_code == "misconfigured":
        text = "The verified provider directory is not configured yet, so I can’t search it safely."
        reason = "provider directory misconfigured"
    else:
        text = "The verified provider directory is unavailable. You can say ‘try again’ or give me a different location."
        reason = "provider directory failure remains active"
    return ResponseDecision(text=text, provider="provider_directory", reason=reason)


def _provider_details_response(result: OrchestrationResult) -> ResponseDecision | None:
    state = result.session.state
    if not state.provider_options:
        return None
    if not is_provider_details_request(result.turn_event.text):
        return None
    provider = resolve_provider_reference(result.turn_event.text, state.provider_options)
    if provider is None:
        selected_provider_id = state.slots.get("provider_id")
        if selected_provider_id:
            provider = next(
                (
                    option
                    for option in state.provider_options
                    if option.provider_id == selected_provider_id.value
                ),
                None,
            )
    if provider is None:
        return None
    details = [f"{provider.name} is listed at {provider.address}" if provider.address else provider.name]
    if provider.phone:
        details.append(f"Phone: {provider.phone}")
    if provider.website:
        details.append(f"Website: {provider.website}")
    return ResponseDecision(
        text=(
            "Here are the verified directory details I have: "
            + ". ".join(details)
            + ". Would you like to use this provider?"
        ),
        provider="provider_directory",
        reason="rendered verified provider details",
    )


def _provider_selection_response(result: OrchestrationResult) -> ResponseDecision | None:
    if not any(
        isinstance(event, SlotCapturedEvent) and event.slot in {"provider_id", "provider_name"}
        for event in result.events
    ):
        return None
    provider_name = result.session.state.slots.get("provider_name")
    if provider_name is None:
        return None
    if result.session.state.active_task == TaskName.APPOINTMENT_REQUEST:
        text = f"Great, I’ve selected {provider_name.value}. What day or time would you prefer?"
    else:
        text = (
            f"Great, I’ve selected {provider_name.value}. Would you like more details, "
            "or would you like help preparing an appointment request?"
        )
    return ResponseDecision(
        text=text,
        provider="workflow",
        reason="provider selection acknowledged",
    )


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


def _workflow_clarification_response(result: OrchestrationResult) -> ResponseDecision | None:
    state = result.session.state
    normalized = " ".join(result.turn_event.text.lower().split())
    asks_about_provider = any(
        phrase in normalized
        for phrase in (
            "what clinic",
            "which clinic",
            "what provider",
            "which provider",
            "what hospital",
            "which hospital",
            "where are you booking",
        )
    )
    if not asks_about_provider or "location" in state.slots:
        return None
    care_setting = state.slots.get("care_setting")
    care_text = f" for {care_setting.value}" if care_setting else ""
    return ResponseDecision(
        text=(
            f"We’re still gathering the appointment context{care_text}; I haven’t searched for a provider yet. "
            "What city, neighborhood, or postal code should I use?"
        ),
        provider="workflow",
        reason="clarified provider search prerequisites",
    )


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
