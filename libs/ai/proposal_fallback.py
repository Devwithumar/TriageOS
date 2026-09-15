"""Conservative recovery proposals for transient structured-model failures."""

import re

from libs.ai.conversation_intelligence import detect_intent
from libs.conversation.contracts import DialogueAct, IntentName
from libs.conversation.domain import ConversationState, SlotSource, TaskName, WorkflowState
from libs.conversation.provider_matching import resolve_provider_reference
from libs.conversation.proposals import (
    ConversationProposal,
    ProposalConfidenceBand,
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
    value = _extract_value(user_text, expected)
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


def _extract_provider_lookup_slots(user_text: str) -> list[ProposedSlot]:
    normalized = " ".join(user_text.lower().split())
    provider_terms = (
        "emergency department",
        "primary care",
        "veterinary",
        "hospital",
        "clinic",
        "doctor",
        "dentist",
        "pharmacy",
        "provider",
        "vet",
    )
    care_setting = next((term for term in provider_terms if term in normalized), None)
    location_match = re.search(
        r"\b(?:near|in|around|at|close to)\s+([^?.!]+)",
        user_text,
        re.IGNORECASE,
    )
    location = location_match.group(1).strip(" ,") if location_match else None
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


def _intent_name(value: str) -> IntentName:
    try:
        return IntentName(value)
    except ValueError:
        return IntentName.GENERAL_CONVERSATION
