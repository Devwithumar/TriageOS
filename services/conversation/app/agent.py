import re

from libs.ai.conversation_intelligence import build_context, detect_intent
from libs.ai.llm import CompletionResult, complete_conversation
from libs.ai.model_router import route_model
from libs.conversation.contracts import (
    ConversationState,
    ExtractedSlot,
    IntentClassification,
    IntentName,
    ReceptionistState,
)
from libs.conversation.workflow import apply_receptionist_turn

SAFETY_REPLY = (
    "Chest pain can be serious. If this is happening now, call your local emergency "
    "services immediately or have someone take you to the nearest emergency department. "
    "Please do not drive yourself."
)

INTENT_MAP = {
    "greeting": IntentName.GREETING,
    "capabilities": IntentName.CAPABILITIES,
    "gratitude": IntentName.GRATITUDE,
    "practice_information": IntentName.PRACTICE_INFORMATION,
    "appointment_request": IntentName.APPOINTMENT_REQUEST,
    "appointment_change": IntentName.APPOINTMENT_CHANGE,
    "appointment_cancellation": IntentName.APPOINTMENT_CANCELLATION,
    "confirmation": IntentName.CONFIRMATION,
    "correction": IntentName.CORRECTION,
    "urgent_safety": IntentName.URGENT_SAFETY,
    "healthcare_request": IntentName.UNSUPPORTED_CLINICAL,
    "general_conversation": IntentName.GENERAL_CONVERSATION,
    "question": IntentName.GENERAL_CONVERSATION,
    "sharing": IntentName.GENERAL_CONVERSATION,
    "unclear": IntentName.UNKNOWN,
}

STATUS_MAP = {
    "collecting_details": ReceptionistState.COLLECTING_DETAILS,
    "reviewing_request": ReceptionistState.REVIEWING_REQUEST,
    "confirmed": ReceptionistState.CONFIRMED,
    "submitted": ReceptionistState.SUBMITTED,
    "completed": ReceptionistState.COMPLETED,
    "cancelled": ReceptionistState.CANCELLED,
    "correction_required": ReceptionistState.CORRECTION_REQUIRED,
    "submission_failed": ReceptionistState.SUBMISSION_FAILED,
}


def _contract_intent(intent_name: str, confidence: float, topic: str | None) -> IntentClassification:
    return IntentClassification(
        name=INTENT_MAP.get(intent_name, IntentName.UNKNOWN),
        confidence=confidence,
        topic=topic,
    )


def _conversation_state(session_id: str, structured_state: dict[str, object]) -> ConversationState:
    workflow_state = STATUS_MAP.get(
        str(structured_state.get("appointment_status", "idle")),
        ReceptionistState.IDLE,
    )
    slots = {
        field: ExtractedSlot(value=str(structured_state[field]), confidence=1, source="session_state")
        for field in ("caller_name", "callback_number", "preferred_time", "appointment_reason", "email")
        if structured_state.get(field)
    }
    workflow_name = "receptionist" if workflow_state != ReceptionistState.IDLE else "general"
    return ConversationState(
        session_id=session_id,
        workflow={"name": workflow_name, "state": workflow_state},
        slots=slots,
    )


def _extract_entities(
    user_text: str,
    intent_name: str,
    current_state: ConversationState,
) -> dict[str, ExtractedSlot]:
    entities: dict[str, ExtractedSlot] = {}
    name_match = re.search(
        r"\b(?:my name is|i am|i'm)\s+([A-Za-z][A-Za-z .'-]{1,50})",
        user_text,
        re.IGNORECASE,
    )
    if name_match and not any(char.isdigit() for char in name_match.group(1)):
        entities["caller_name"] = ExtractedSlot(
            value=name_match.group(1).strip(" ."), confidence=0.98, source="conversation"
        )

    phone_match = re.search(r"(?:\+?\d[\d ()-]{7,}\d)", user_text)
    if phone_match:
        entities["callback_number"] = ExtractedSlot(
            value=phone_match.group(0).strip(), confidence=0.98, source="conversation"
        )

    email_match = re.search(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b", user_text)
    if email_match:
        entities["email"] = ExtractedSlot(
            value=email_match.group(0).lower(), confidence=0.99, source="conversation"
        )

    lowered = user_text.lower()
    time_markers = (
        "morning", "afternoon", "evening", "tomorrow", "next week", "monday", "tuesday",
        "wednesday", "thursday", "friday", "saturday", "sunday",
    )
    if any(marker in lowered for marker in time_markers):
        preferred_time = re.sub(r"(?:\+?\d[\d ()-]{7,}\d)", "", user_text).strip(" ,.-")
        entities["preferred_time"] = ExtractedSlot(
            value=preferred_time, confidence=0.88, source="conversation"
        )

    if (
        current_state.workflow.state == ReceptionistState.COLLECTING_DETAILS
        and intent_name not in {"appointment_request", "appointment_change"}
        and not entities
        and user_text.strip()
    ):
        entities["appointment_reason"] = ExtractedSlot(
            value=user_text.strip(), confidence=0.72, source="conversation"
        )
    return entities


def _structured_state(decision_state: ConversationState, decision_next_action: str) -> dict[str, object]:
    return {
        **{name: slot.value for name, slot in decision_state.slots.items()},
        "appointment_status": decision_state.workflow.state.value,
        "workflow": decision_state.workflow.name.value,
        "next_action": decision_next_action,
        "missing_fields": decision_state.workflow.missing_fields,
    }


def generate_reply(
    user_text: str,
    recent_messages: list[dict[str, str]],
    structured_state: dict[str, object] | None = None,
    session_id: str = "conversation",
) -> dict[str, object]:
    route = route_model("conversation")
    intent = detect_intent(user_text)
    context = build_context(recent_messages, intent)
    current_state = _conversation_state(session_id, dict(structured_state or {}))
    contract_intent = _contract_intent(intent.name, intent.confidence, intent.topic)

    if intent.name == "urgent_safety":
        result = CompletionResult(
            text=SAFETY_REPLY,
            model="deterministic-safety",
            provider="guardrail",
            reason="urgent safety escalation guardrail",
        )
        updated_state = dict(structured_state or {})
    elif (
        intent.name in {
            "appointment_request",
            "appointment_change",
            "appointment_cancellation",
            "practice_information",
        }
        or current_state.workflow.state == ReceptionistState.COLLECTING_DETAILS
        or current_state.workflow.state == ReceptionistState.REVIEWING_REQUEST
        or (
            intent.name in {"confirmation", "correction"}
            and current_state.workflow.state != ReceptionistState.IDLE
        )
    ):
        entities = _extract_entities(user_text, intent.name, current_state)
        decision = apply_receptionist_turn(current_state, contract_intent, entities)
        result = CompletionResult(
            text=decision.response,
            model="deterministic-orchestrator",
            provider="workflow",
            reason=f"workflow transition: {decision.transition}",
        )
        updated_state = _structured_state(decision.state, decision.next_action)
        context["workflow_transition"] = decision.transition
        context["tool_call"] = decision.tool_call.model_dump() if decision.tool_call else None
    else:
        result = complete_conversation(user_text, recent_messages, context)
        updated_state = dict(structured_state or {})

    return {
        "reply": result.text,
        "model": result.model,
        "provider": result.provider,
        "reason": result.reason or route.reason,
        "context_messages": len(recent_messages),
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "intent": intent.as_dict(),
        "context": context,
        "structured_state": updated_state,
    }
