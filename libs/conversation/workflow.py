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

APPOINTMENT_CONTEXT_FIELDS = (
    "care_setting",
    "location",
    "appointment_reason",
)

APPOINTMENT_DETAIL_FIELDS = (
    "preferred_time",
    "caller_name",
    "callback_number",
)

APPOINTMENT_FIELDS = APPOINTMENT_CONTEXT_FIELDS + APPOINTMENT_DETAIL_FIELDS

FIELD_QUESTIONS = {
    "care_setting": "What kind of care or clinic are you looking for?",
    "location": "What area or location should I use?",
    "appointment_reason": "What would you like help with during the visit?",
    "preferred_time": "What day or time would you prefer?",
    "caller_name": "What name should I use for the appointment request?",
    "callback_number": "What phone number should the practice use to reach you?",
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
            response="The practice profile is not configured yet, so I cannot verify its services or location.",
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

    if state.workflow.state == ReceptionistState.COLLECTING_DETAILS and intent.name == IntentName.GENERAL_CONVERSATION and not extracted:
        next_action = state.workflow.next_action or "collect_appointment_context"
        expected_field = next_action.removeprefix("collect_")
        stage_fields = APPOINTMENT_CONTEXT_FIELDS
        if expected_field in APPOINTMENT_DETAIL_FIELDS:
            stage_fields = APPOINTMENT_DETAIL_FIELDS
        missing = [field for field in stage_fields if field not in next_state.slots]
        if next_action == "search_providers":
            response = "I’ll retry the provider search for that care type and location."
            return WorkflowDecision(
                state=next_state,
                transition="provider_search_retry",
                next_action="search_providers",
                tool_call=ToolRequest(
                    name="search_providers",
                    arguments={field: next_state.slots[field].value for field in APPOINTMENT_CONTEXT_FIELDS},
                    requires_confirmation=False,
                ),
                response=response,
            )
        if next_action == "select_provider":
            if next_action == "select_provider":
                response = "Please tell me the number or name of the provider you would like to use."
                next_action = "select_provider"
        else:
            missing_text = ", ".join(field.replace("_", " ") for field in missing)
            response = f"I’m still gathering the appointment context. I need your {missing_text}."
        return WorkflowDecision(
            state=next_state,
            transition="workflow_question",
            next_action=next_action,
            missing_fields=missing,
            response=response,
        )

    if state.workflow.state == ReceptionistState.COLLECTING_DETAILS and state.workflow.next_action == "select_provider":
        provider_name = extracted.get("provider_name")
        if not provider_name:
            response = (
                "Please tell me the number or name of the provider you would like to use."
            )
            return WorkflowDecision(
                state=next_state,
                transition="provider_selection_required",
                next_action="select_provider",
                response=response,
            )
        next_state.slots.update(extracted)
        missing_details = [field for field in APPOINTMENT_DETAIL_FIELDS if field not in next_state.slots]
        next_action = f"collect_{missing_details[0]}" if missing_details else "confirm_request"
        next_state.workflow = WorkflowContext(
            name=WorkflowName.RECEPTIONIST,
            state=ReceptionistState.COLLECTING_DETAILS if missing_details else ReceptionistState.REVIEWING_REQUEST,
            next_action=next_action,
            missing_fields=missing_details,
        )
        if missing_details:
            return WorkflowDecision(
                state=next_state,
                transition="provider_selected",
                next_action=next_action,
                missing_fields=missing_details,
                response=FIELD_QUESTIONS[missing_details[0]],
            )
        return WorkflowDecision(
            state=next_state,
            transition="provider_selected_request_complete",
            next_action=next_action,
            response="The request is ready to review. Would you like me to submit it?",
        )

    if state.workflow.state == ReceptionistState.COLLECTING_DETAILS and state.workflow.next_action == "search_providers":
        next_state.workflow = WorkflowContext(
            name=WorkflowName.RECEPTIONIST,
            state=ReceptionistState.COLLECTING_DETAILS,
            next_action="search_providers",
        )
        return WorkflowDecision(
            state=next_state,
            transition="provider_search_pending",
            next_action="search_providers",
            tool_call=ToolRequest(
                name="search_providers",
                arguments={field: next_state.slots[field].value for field in APPOINTMENT_CONTEXT_FIELDS},
                requires_confirmation=False,
            ),
            response=(
                "I have the care type, location, and reason for the visit. "
                "I cannot suggest or select a provider yet because no provider directory is connected. "
                "I won’t ask for your personal details until a provider is identified."
            ),
        )

    if intent.name not in {IntentName.APPOINTMENT_REQUEST, IntentName.APPOINTMENT_CHANGE} and state.workflow.state != ReceptionistState.COLLECTING_DETAILS:
        return WorkflowDecision(
            state=next_state,
            transition="no_receptionist_transition",
            next_action="continue_conversation",
            response="How can I help with the practice today?",
        )

    next_state.slots.update(extracted)
    missing_context = [field for field in APPOINTMENT_CONTEXT_FIELDS if field not in next_state.slots]
    if missing_context:
        next_state.workflow = WorkflowContext(
            name=WorkflowName.RECEPTIONIST,
            state=ReceptionistState.COLLECTING_DETAILS,
            next_action=f"collect_{missing_context[0]}",
            missing_fields=missing_context,
        )
        return WorkflowDecision(
            state=next_state,
            transition="appointment_context_updated",
            next_action=f"collect_{missing_context[0]}",
            missing_fields=missing_context,
            response=FIELD_QUESTIONS[missing_context[0]],
        )

    if state.workflow.next_action != "search_providers" and not any(
        field in extracted for field in APPOINTMENT_CONTEXT_FIELDS
    ) and not all(field in next_state.slots for field in APPOINTMENT_CONTEXT_FIELDS):
        missing_context = [field for field in APPOINTMENT_CONTEXT_FIELDS if field not in next_state.slots]
        next_state.workflow = WorkflowContext(
            name=WorkflowName.RECEPTIONIST,
            state=ReceptionistState.COLLECTING_DETAILS,
            next_action=f"collect_{missing_context[0]}",
            missing_fields=missing_context,
        )
        return WorkflowDecision(
            state=next_state,
            transition="appointment_context_required",
            next_action=f"collect_{missing_context[0]}",
            missing_fields=missing_context,
            response=FIELD_QUESTIONS[missing_context[0]],
        )

    if all(field in next_state.slots for field in APPOINTMENT_CONTEXT_FIELDS) and state.workflow.next_action != "search_providers":
        next_state.workflow = WorkflowContext(
            name=WorkflowName.RECEPTIONIST,
            state=ReceptionistState.COLLECTING_DETAILS,
            next_action="search_providers",
        )
        context_summary = ", ".join(
            f"{field.replace('_', ' ')}: {next_state.slots[field].value}"
            for field in APPOINTMENT_CONTEXT_FIELDS
        )
        return WorkflowDecision(
            state=next_state,
            transition="appointment_context_complete",
            next_action="search_providers",
            tool_call=ToolRequest(
                name="search_providers",
                arguments={field: next_state.slots[field].value for field in APPOINTMENT_CONTEXT_FIELDS},
                requires_confirmation=False,
            ),
            response=(
                f"I understand the request as {context_summary}. "
                "I need to identify a suitable provider before collecting your personal details. "
                "The provider directory is not connected yet, so I cannot responsibly suggest a clinic."
            ),
        )

    missing_fields = [field for field in APPOINTMENT_DETAIL_FIELDS if field not in next_state.slots]
    if missing_fields:
        next_state.workflow = WorkflowContext(
            name=WorkflowName.RECEPTIONIST,
            state=ReceptionistState.COLLECTING_DETAILS,
            next_action=f"collect_{missing_fields[0]}",
            missing_fields=missing_fields,
        )
        return WorkflowDecision(
            state=next_state,
            transition="appointment_details_updated",
            next_action=f"collect_{missing_fields[0]}",
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
