from collections.abc import Mapping

from libs.conversation.contracts import (
    ConversationState,
    ExtractedSlot,
    IntentClassification,
    IntentName,
    ReceptionistState,
    ToolRequest,
    WorkflowContext,
    WorkflowDecision,
    WorkflowName,
)

APPOINTMENT_FIELDS = (
    "caller_name",
    "callback_number",
    "preferred_time",
    "appointment_reason",
)

FIELD_QUESTIONS = {
    "caller_name": "What name should I put on the appointment request?",
    "callback_number": "What phone number should the practice use to reach you?",
    "preferred_time": "What day or time would you prefer?",
    "appointment_reason": "What would you like the appointment to be about?",
}


def apply_receptionist_turn(
    state: ConversationState,
    intent: IntentClassification,
    entities: Mapping[str, ExtractedSlot] | None = None,
) -> WorkflowDecision:
    next_state = state.model_copy(deep=True)
    next_state.turn_count += 1
    extracted = dict(entities or {})

    if intent.name == IntentName.URGENT_SAFETY:
        next_state.workflow = WorkflowContext(
            name=WorkflowName.SAFETY_ESCALATION,
            state=ReceptionistState.IDLE,
            next_action="escalate",
        )
        return WorkflowDecision(
            state=next_state,
            transition="safety_escalation",
            next_action="escalate",
            response="This needs urgent attention. Please contact your local emergency services immediately.",
        )

    if intent.name == IntentName.PRACTICE_INFORMATION:
        next_state.workflow = WorkflowContext(
            name=WorkflowName.RECEPTIONIST,
            state=ReceptionistState.IDLE,
            next_action="lookup_practice_profile",
        )
        return WorkflowDecision(
            state=next_state,
            transition="practice_information_requested",
            next_action="lookup_practice_profile",
            tool_call=ToolRequest(name="get_practice_profile"),
            response="I’ll check the practice information for you.",
        )

    if intent.name == IntentName.APPOINTMENT_CANCELLATION:
        next_state.workflow = WorkflowContext(
            name=WorkflowName.RECEPTIONIST,
            state=ReceptionistState.CANCELLED,
            next_action="end_workflow",
        )
        return WorkflowDecision(
            state=next_state,
            transition="appointment_cancelled",
            next_action="end_workflow",
            response="Understood. I won’t continue with that appointment request.",
        )

    if intent.name == IntentName.CORRECTION:
        next_state.workflow = WorkflowContext(
            name=WorkflowName.RECEPTIONIST,
            state=ReceptionistState.CORRECTION_REQUIRED,
            next_action="collect_correction",
        )
        return WorkflowDecision(
            state=next_state,
            transition="correction_requested",
            next_action="collect_correction",
            response="Of course. What would you like to change?",
        )

    if intent.name == IntentName.CONFIRMATION and state.workflow.state == ReceptionistState.REVIEWING_REQUEST:
        next_state.workflow = WorkflowContext(
            name=WorkflowName.RECEPTIONIST,
            state=ReceptionistState.CONFIRMED,
            next_action="submit_appointment_request",
        )
        arguments = {name: slot.value for name, slot in next_state.slots.items()}
        return WorkflowDecision(
            state=next_state,
            transition="appointment_request_confirmed",
            next_action="submit_appointment_request",
            tool_call=ToolRequest(
                name="create_appointment_request",
                arguments=arguments,
                requires_confirmation=False,
            ),
            response="Thanks. I’m submitting that appointment request now.",
        )

    if intent.name not in {IntentName.APPOINTMENT_REQUEST, IntentName.APPOINTMENT_CHANGE} and state.workflow.state != ReceptionistState.COLLECTING_DETAILS:
        return WorkflowDecision(
            state=next_state,
            transition="no_receptionist_transition",
            next_action="continue_conversation",
            response="How can I help with the practice today?",
        )

    next_state.workflow = WorkflowContext(
        name=WorkflowName.RECEPTIONIST,
        state=ReceptionistState.COLLECTING_DETAILS,
        next_action="ask_for_missing_field",
    )
    next_state.slots.update(extracted)
    missing_fields = [field for field in APPOINTMENT_FIELDS if field not in next_state.slots]
    if missing_fields:
        next_state.workflow = WorkflowContext(
            name=WorkflowName.RECEPTIONIST,
            state=ReceptionistState.COLLECTING_DETAILS,
            next_action="ask_for_missing_field",
            missing_fields=missing_fields,
        )
        return WorkflowDecision(
            state=next_state,
            transition="appointment_details_updated",
            next_action="ask_for_missing_field",
            missing_fields=missing_fields,
            response=FIELD_QUESTIONS[missing_fields[0]],
        )

    next_state.workflow = WorkflowContext(
        name=WorkflowName.RECEPTIONIST,
        state=ReceptionistState.REVIEWING_REQUEST,
        next_action="confirm_request",
    )
    summary = ", ".join(f"{field.replace('_', ' ')}: {slot.value}" for field, slot in next_state.slots.items())
    return WorkflowDecision(
        state=next_state,
        transition="appointment_details_complete",
        next_action="confirm_request",
        response=f"I have the appointment details as {summary}. Would you like me to submit this request?",
    )
