"""Untrusted LLM proposals and deterministic command validation."""

from enum import StrEnum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from libs.conversation.contracts import DialogueAct, IntentName
from libs.conversation.domain import (
    CaptureSlotCommand,
    CancelTaskCommand,
    ConversationState,
    CorrectSlotCommand,
    DomainCommand,
    DomainInvariantError,
    OperationName,
    RequestOperationCommand,
    SlotSource,
    StartWorkflowCommand,
    TaskName,
    WorkflowState,
    command_to_event,
    reduce_state,
)


class ProposalValidationError(DomainInvariantError):
    """Raised when an LLM proposal cannot be safely converted into commands."""


class ProposalConfidenceBand(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ProposedSlot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    value: str = Field(min_length=1)
    source: SlotSource
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="before")
    @classmethod
    def normalize_contract_name(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        name = normalized.get("name")
        if isinstance(name, str):
            normalized["name"] = _SLOT_ALIASES.get(name.strip().lower(), name.strip().lower())
        return normalized


class ProposedCorrection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    value: str = Field(min_length=1)
    confidence: float = Field(default=1, ge=0, le=1)

    @model_validator(mode="before")
    @classmethod
    def normalize_contract_name(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        name = normalized.get("name")
        if isinstance(name, str):
            normalized["name"] = _SLOT_ALIASES.get(name.strip().lower(), name.strip().lower())
        return normalized


class ToolSelectionProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: OperationName
    rationale: str | None = None


class ConversationProposal(BaseModel):
    """Structured model output; it has no authority to mutate state."""

    model_config = ConfigDict(extra="forbid")

    proposal_schema_version: int = Field(default=1, ge=1)
    proposal_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    session_id: str = Field(min_length=1)
    based_on_state_version: int = Field(ge=0)
    correlation_id: str = Field(min_length=1)
    intent: IntentName
    dialogue_act: DialogueAct
    confidence_band: ProposalConfidenceBand
    requested_task: TaskName = TaskName.NONE
    slots: list[ProposedSlot] = Field(default_factory=list)
    corrections: list[ProposedCorrection] = Field(default_factory=list)
    tool_selection: ToolSelectionProposal | None = None
    cancel_requested: bool = False
    response_draft: str | None = None


class ValidatedProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    session_id: str
    based_on_state_version: int = Field(ge=0)
    commands: list[DomainCommand] = Field(default_factory=list)
    response_draft: str | None = None


_KNOWN_SLOTS = {
    "care_setting",
    "location",
    "appointment_reason",
    "provider_id",
    "provider_name",
    "preferred_time",
    "caller_name",
    "callback_number",
    "email",
}
_EXPLICIT_ONLY_SLOTS = {
    "care_setting",
    "location",
    "appointment_reason",
    "provider_id",
    "provider_name",
    "preferred_time",
    "caller_name",
    "callback_number",
    "email",
}
_REQUIRED_PROVIDER_SEARCH_SLOTS = {"care_setting", "location"}
_REQUIRED_APPOINTMENT_SLOTS = {
    "provider_id",
    "care_setting",
    "location",
    "appointment_reason",
    "preferred_time",
    "caller_name",
    "callback_number",
}

_SLOT_ALIASES = {
    "specialty": "care_setting",
    "provider_type": "care_setting",
    "care_type": "care_setting",
    "clinic_type": "care_setting",
    "service_type": "care_setting",
    "provider": "provider_name",
    "reason": "appointment_reason",
    "visit_reason": "appointment_reason",
    "reason_for_visit": "appointment_reason",
    "appointment_reason_text": "appointment_reason",
    "purpose": "appointment_reason",
    "city": "location",
    "area": "location",
    "neighborhood": "location",
    "postal_code": "location",
    "zip": "location",
    "postcode": "location",
    "date": "preferred_time",
    "time": "preferred_time",
    "datetime": "preferred_time",
    "date_time": "preferred_time",
    "appointment_time": "preferred_time",
    "availability": "preferred_time",
    "name": "caller_name",
    "full_name": "caller_name",
    "phone": "callback_number",
    "phone_number": "callback_number",
}


def validate_proposal(
    state: ConversationState,
    proposal: ConversationProposal,
) -> ValidatedProposal:
    """Convert one model proposal into commands legal for the current state."""

    if proposal.session_id != state.session_id:
        raise ProposalValidationError("proposal session does not match current state")
    if proposal.based_on_state_version != state.state_version:
        raise ProposalValidationError("proposal was generated from a stale state version")
    if proposal.confidence_band == ProposalConfidenceBand.LOW:
        raise ProposalValidationError("low-confidence proposal requires clarification")
    cancellation_requested = (
        proposal.cancel_requested
        or proposal.intent == IntentName.APPOINTMENT_CANCELLATION
        or proposal.dialogue_act == DialogueAct.CANCEL
    )
    if cancellation_requested:
        if any((proposal.slots, proposal.corrections, proposal.tool_selection)) or proposal.requested_task != TaskName.NONE:
            raise ProposalValidationError("cancellation cannot be combined with other state changes")
        if not _has_cancellable_context(state):
            return ValidatedProposal(
                proposal_id=proposal.proposal_id,
                session_id=proposal.session_id,
                based_on_state_version=state.state_version,
                response_draft=proposal.response_draft,
            )
        command = _cancel_command(state, proposal)
        return ValidatedProposal(
            proposal_id=proposal.proposal_id,
            session_id=proposal.session_id,
            based_on_state_version=state.state_version,
            commands=[command],
            response_draft=proposal.response_draft,
        )

    commands: list[DomainCommand] = []
    working_state = state
    if proposal.requested_task != TaskName.NONE:
        if state.active_task != TaskName.NONE and proposal.requested_task != state.active_task:
            raise ProposalValidationError("cannot start a second task while one is active")
        if proposal.requested_task not in {TaskName.APPOINTMENT_REQUEST, TaskName.PROVIDER_LOOKUP}:
            raise ProposalValidationError("requested task is not enabled")
        if state.active_task == TaskName.NONE:
            command = StartWorkflowCommand(
                session_id=state.session_id,
                expected_state_version=working_state.state_version,
                correlation_id=proposal.correlation_id,
                task=proposal.requested_task,
            )
            commands.append(command)
            working_state = reduce_state(working_state, command_to_event(working_state, command))

    if working_state.active_task == TaskName.NONE and (proposal.slots or proposal.corrections or proposal.tool_selection):
        raise ProposalValidationError("state changes require an active task")

    for slot in proposal.slots:
        _validate_slot(slot)
        command = CaptureSlotCommand(
            session_id=working_state.session_id,
            expected_state_version=working_state.state_version,
            correlation_id=proposal.correlation_id,
            slot=slot.name,
            value=slot.value,
            source=slot.source,
            confidence=slot.confidence,
        )
        commands.append(command)
        working_state = reduce_state(working_state, command_to_event(working_state, command))

    for correction in proposal.corrections:
        if correction.name not in _KNOWN_SLOTS:
            raise ProposalValidationError(f"unknown correction slot: {correction.name}")
        if correction.name not in working_state.slots:
            raise ProposalValidationError(f"cannot correct uncaptured slot: {correction.name}")
        command = CorrectSlotCommand(
            session_id=working_state.session_id,
            expected_state_version=working_state.state_version,
            correlation_id=proposal.correlation_id,
            slot=correction.name,
            value=correction.value,
        )
        commands.append(command)
        working_state = reduce_state(working_state, command_to_event(working_state, command))

    if proposal.tool_selection:
        _validate_tool_selection(working_state, proposal.tool_selection)
        request_id = f"{proposal.session_id}:{working_state.state_version}:{proposal.tool_selection.operation}"
        command = RequestOperationCommand(
            session_id=working_state.session_id,
            expected_state_version=working_state.state_version,
            correlation_id=proposal.correlation_id,
            request_id=request_id,
            operation=proposal.tool_selection.operation,
            idempotency_key=request_id,
        )
        commands.append(command)
        reduce_state(working_state, command_to_event(working_state, command))

    return ValidatedProposal(
        proposal_id=proposal.proposal_id,
        session_id=proposal.session_id,
        based_on_state_version=state.state_version,
        commands=commands,
        response_draft=proposal.response_draft,
    )


def _validate_slot(slot: ProposedSlot) -> None:
    if slot.name not in _KNOWN_SLOTS:
        raise ProposalValidationError(f"unknown slot: {slot.name}")
    if slot.source == SlotSource.MODEL_INFERENCE and slot.name in _EXPLICIT_ONLY_SLOTS:
        raise ProposalValidationError(f"slot requires explicit user confirmation: {slot.name}")
    if slot.confidence < 0.7:
        raise ProposalValidationError(f"slot confidence is too low: {slot.name}")


def _validate_tool_selection(state: ConversationState, selection: ToolSelectionProposal) -> None:
    if state.pending_operation is not None:
        raise ProposalValidationError("cannot select a tool while another operation is pending")
    if selection.operation == OperationName.SEARCH_PROVIDERS:
        if state.active_task not in {TaskName.APPOINTMENT_REQUEST, TaskName.PROVIDER_LOOKUP}:
            raise ProposalValidationError("provider search requires a receptionist task")
        missing = _REQUIRED_PROVIDER_SEARCH_SLOTS - state.slots.keys()
        if missing:
            raise ProposalValidationError(f"provider search is missing slots: {sorted(missing)}")
    elif selection.operation == OperationName.CREATE_APPOINTMENT_REQUEST:
        missing = _REQUIRED_APPOINTMENT_SLOTS - state.slots.keys()
        if missing:
            raise ProposalValidationError(f"appointment request is missing slots: {sorted(missing)}")
        if state.workflow_state not in {
            WorkflowState.REVIEWING_REQUEST,
            WorkflowState.AWAITING_CONFIRMATION,
        }:
            raise ProposalValidationError("appointment request requires a reviewed request")


def _cancel_command(state: ConversationState, proposal: ConversationProposal) -> CancelTaskCommand:
    return CancelTaskCommand(
        command_type="cancel_task",
        session_id=state.session_id,
        expected_state_version=state.state_version,
        correlation_id=proposal.correlation_id,
        reason="user requested cancellation",
    )


def _has_cancellable_context(state: ConversationState) -> bool:
    return bool(
        state.active_task != TaskName.NONE
        or state.workflow_state not in {
            WorkflowState.IDLE,
            WorkflowState.CANCELLED,
        }
        or state.slots
        or state.provider_options
        or state.pending_operation
        or state.last_operation_result
    )
