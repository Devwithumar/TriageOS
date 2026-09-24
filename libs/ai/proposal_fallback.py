"""Conservative recovery proposals for transient structured-model failures."""

import re

from libs.ai.conversation_intelligence import canonical_care_setting, detect_intent
from libs.conversation.contracts import DialogueAct, IntentName
from libs.conversation.domain import (
    AvailabilityResultData,
    ConversationState,
    SlotSource,
    TaskName,
    WorkflowState,
)
from libs.conversation.provider_matching import resolve_provider_reference
from libs.conversation.proposals import (
    ConversationProposal,
    ProposalConfidenceBand,
    ProposedCorrection,
    ProposedSlot,
)


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


def build_recovery_proposal(
    user_text: str,
    state: ConversationState,
    correlation_id: str,
) -> ConversationProposal:
    """Recover only explicit, low-risk task transitions from the current state."""

    detected = detect_intent(user_text)
    corrections = _extract_corrections(user_text, state)
    if detected.name == IntentName.CORRECTION.value:
        return ConversationProposal(
            session_id=state.session_id,
            based_on_state_version=state.state_version,
            correlation_id=correlation_id,
            intent=IntentName.CORRECTION,
            dialogue_act=DialogueAct.CORRECT if corrections else DialogueAct.REQUEST_INFORMATION,
            confidence_band=ProposalConfidenceBand.HIGH,
            corrections=corrections,
        )
    requested_task = _requested_task(state, detected.name)
    slots = _extract_explicit_slots(user_text, state, requested_task)
    dialogue_act = DialogueAct.INFORM if slots else DialogueAct.REQUEST_INFORMATION
    return ConversationProposal(
        session_id=state.session_id,
        based_on_state_version=state.state_version,
        correlation_id=correlation_id,
        intent=_intent_name(detected.name),
        dialogue_act=dialogue_act,
        confidence_band=ProposalConfidenceBand.HIGH,
        requested_task=requested_task,
        slots=slots,
        response_draft=None,
    )


def _requested_task(state: ConversationState, detected_name: str) -> TaskName:
    if state.active_task != TaskName.NONE:
        return TaskName.NONE
    if detected_name == IntentName.APPOINTMENT_REQUEST.value:
        return TaskName.APPOINTMENT_REQUEST
    if detected_name == IntentName.PROVIDER_LOOKUP.value:
        return TaskName.PROVIDER_LOOKUP
    return TaskName.NONE


def _extract_explicit_slots(
    user_text: str,
    state: ConversationState,
    requested_task: TaskName,
) -> list[ProposedSlot]:
    task = state.active_task if state.active_task != TaskName.NONE else requested_task
    if task == TaskName.NONE:
        return []
    if state.active_task == TaskName.NONE and requested_task == TaskName.PROVIDER_LOOKUP:
        return _extract_provider_lookup_slots(user_text)
    if state.active_task == TaskName.NONE and requested_task != TaskName.NONE:
        return []
    if state.workflow_state == WorkflowState.SELECTING_PROVIDER:
        return _select_provider(user_text, state)

    expected = _next_missing_slot(state, task)
    if expected is None:
        return []
    value = (
        _resolve_availability_slot(user_text, state.last_operation_result)
        if expected == "preferred_time"
        else _extract_value(user_text, expected)
    )
    if not value:
        return []
    return [
        ProposedSlot(
            name=expected,
            value=value,
            source=SlotSource.USER_EXPLICIT,
            confidence=0.9,
        )
    ]


def _resolve_availability_slot(
    user_text: str,
    result: object,
) -> str | None:
    if not isinstance(result, AvailabilityResultData) or not result.slots:
        return _extract_value(user_text, "preferred_time")
    normalized = " ".join(user_text.lower().split())
    ordinal = re.search(
        r"\b(?:option|choice|number|no\.?)\s*(1|2|3|4|5)\b",
        normalized,
    )
    if ordinal:
        index = int(ordinal.group(1)) - 1
        if index < len(result.slots):
            return result.slots[index].start_at
    for slot in result.slots:
        if slot.label.lower() in normalized or slot.start_at.lower() in normalized:
            return slot.start_at
    return _extract_value(user_text, "preferred_time")


def _extract_provider_lookup_slots(user_text: str) -> list[ProposedSlot]:
    normalized = " ".join(user_text.lower().split())
    care_setting = canonical_care_setting(user_text)
    if care_setting is None and "provider" in normalized:
        care_setting = "doctor"
    conjunction = r"\s*(?:,\s*)?(?:and|but|so|plus)\s+(?:i|we|please|want|wanna|need|would|looking|trying|hoping|just)\b"
    location_match = re.search(
        r"\b(?:i\s+(?:am|'m)|we\s+(?:are|'re)|live|located|based|stay)\s+"
        r"(?:in|at|near)\s+(?P<location>[^?.!;]+?)"
        rf"(?={conjunction}|[?.!;]|$)",
        user_text,
        re.IGNORECASE,
    )
    if location_match is None:
        location_match = re.search(
            r"\b(?:near|in|around|at|close to)\s+(?P<location>[^?.!;]+?)"
            rf"(?={conjunction}|[?.!;]|$)",
            user_text,
            re.IGNORECASE,
        )
    location = location_match.group("location").strip(" ,") if location_match else None
    if location and location.lower() in {"me", "here", "there"}:
        location = None
    slots: list[ProposedSlot] = []
    if care_setting:
        slots.append(
            ProposedSlot(
                name="care_setting",
                value=care_setting,
                source=SlotSource.USER_EXPLICIT,
                confidence=0.94,
            )
        )
    if location:
        slots.append(
            ProposedSlot(
                name="location",
                value=location,
                source=SlotSource.USER_EXPLICIT,
                confidence=0.94,
            )
        )
    return slots


def _next_missing_slot(state: ConversationState, task: TaskName) -> str | None:
    for slot_name in _TASK_SLOTS[task]:
        if slot_name == "provider_id" and state.provider_options and slot_name not in state.slots:
            continue
        if slot_name not in state.slots:
            return slot_name
    return None


def _extract_value(text: str, slot_name: str) -> str | None:
    stripped = text.strip()
    if not stripped or stripped.endswith("?"):
        return None
    if slot_name == "caller_name":
        match = re.search(
            r"\b(?:my name is|i am|i'm|the name is|put the name)\s+([A-Za-z][A-Za-z .'-]{1,50})",
            stripped,
            re.IGNORECASE,
        )
        return match.group(1).strip(" .") if match else stripped
    if slot_name == "callback_number":
        match = re.search(r"(?:\+?\d[\d ()-]{7,}\d)", stripped)
        return match.group(0).strip() if match else None
    if slot_name == "email":
        match = re.search(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b", stripped)
        return match.group(0).lower() if match else None
    if slot_name == "preferred_time":
        lowered = stripped.lower()
        if any(
            marker in lowered
            for marker in (
                "morning",
                "afternoon",
                "evening",
                "tomorrow",
                "next week",
                "monday",
                "tuesday",
                "wednesday",
                "thursday",
                "friday",
                "saturday",
                "sunday",
            )
        ):
            return stripped
        return None
    return stripped


def _select_provider(user_text: str, state: ConversationState) -> list[ProposedSlot]:
    selected = resolve_provider_reference(user_text, state.provider_options)
    if selected is None:
        return []
    return [
        ProposedSlot(
            name="provider_id",
            value=selected.provider_id,
            source=SlotSource.USER_EXPLICIT,
            confidence=0.96,
        ),
        ProposedSlot(
            name="provider_name",
            value=selected.name,
            source=SlotSource.USER_EXPLICIT,
            confidence=0.96,
        ),
    ]


def _extract_corrections(
    user_text: str,
    state: ConversationState,
) -> list[ProposedCorrection]:
    normalized = " ".join(user_text.lower().split())
    corrections: list[ProposedCorrection] = []

    if state.provider_options and state.slots.get("provider_id"):
        selected = resolve_provider_reference(user_text, state.provider_options)
        if selected is not None:
            corrections.extend(
                [
                    ProposedCorrection(name="provider_id", value=selected.provider_id),
                    ProposedCorrection(name="provider_name", value=selected.name),
                ]
            )
            return corrections

    labeled_patterns = {
        "care_setting": r"(?:care|provider|specialty|clinic type)\s*(?:is|should be|to|=)?\s*(?P<value>[^?.!]+)",
        "location": r"(?:location|city|area|neighborhood|postal code|zip code|postcode)\s*(?:is|should be|to|=)?\s*(?P<value>[^?.!]+)",
        "appointment_reason": r"(?:appointment reason|reason for (?:the )?visit|visit reason)\s*(?:is|should be|to|=)?\s*(?P<value>[^?.!]+)",
        "preferred_time": r"(?:appointment time|preferred time|date|time)\s*(?:is|should be|to|=)?\s*(?P<value>[^?.!]+)",
        "caller_name": r"(?:my name|caller name)\s*(?:is|should be|to|=)?\s*(?P<value>[^?.!]+)",
        "callback_number": r"(?:callback number|phone number)\s*(?:is|should be|to|=)?\s*(?P<value>[^?.!]+)",
    }
    for slot_name, pattern in labeled_patterns.items():
        match = re.search(pattern, user_text, re.IGNORECASE)
        if match and slot_name in state.slots:
            value = _clean_correction_value(match.group("value"))
            if value:
                return [ProposedCorrection(name=slot_name, value=value)]

    provider_correction = re.search(
        r"(?:not|no)\s+(?:a|an)?\s*(?:hospital|clinic|doctor|dentist|pharmacy|vet|veterinary)[^?.!]*"
        r"(?:i mean|rather|instead)\s+(?:a|an)?\s*(?P<value>[^?.!]+)",
        user_text,
        re.IGNORECASE,
    )
    if provider_correction and "care_setting" in state.slots:
        value = _clean_correction_value(provider_correction.group("value"))
        if value:
            return [ProposedCorrection(name="care_setting", value=value)]

    implied_match = re.search(
        r"(?:actually\s*,?\s*)?(?:i meant|change that to|change it to)\s+(?P<value>[^?.!]+)",
        user_text,
        re.IGNORECASE,
    )
    if implied_match:
        value = _clean_correction_value(implied_match.group("value"))
        target = _infer_correction_target(value, state)
        if target and value:
            return [ProposedCorrection(name=target, value=value)]

    if "instead" in normalized:
        value_match = re.search(r"(?:actually\s*,?\s*)?(?:i want|i need|use)\s+(?P<value>[^?.!]+)", user_text, re.IGNORECASE)
        if value_match:
            value = _clean_correction_value(value_match.group("value"))
            target = _infer_correction_target(value, state)
            if target and value:
                return [ProposedCorrection(name=target, value=value)]
    return corrections


def _infer_correction_target(value: str, state: ConversationState) -> str | None:
    normalized = value.lower()
    provider_terms = ("clinic", "hospital", "doctor", "dentist", "pharmacy", "vet", "veterinary")
    if any(term in normalized for term in provider_terms) and "care_setting" in state.slots:
        return "care_setting"
    if "location" in state.slots:
        return "location"
    if "appointment_reason" in state.slots:
        return "appointment_reason"
    if "preferred_time" in state.slots:
        return "preferred_time"
    return None


def _clean_correction_value(value: str) -> str:
    return re.sub(r"\s+(?:instead|now)\s*$", "", value.strip(" .,;:"))


def _intent_name(value: str) -> IntentName:
    try:
        return IntentName(value)
    except ValueError:
        return IntentName.GENERAL_CONVERSATION
