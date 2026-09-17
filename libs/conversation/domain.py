"""Canonical conversation domain model and pure event reducer.

This module is intentionally separate from the current receptionist prototype.
It defines the contracts that the replacement orchestration layer must obey.
"""

from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Literal, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DomainInvariantError(ValueError):
    """Raised when a domain event would produce invalid conversation state."""


class StateConflictError(DomainInvariantError):
    """Raised when an event was created against an older state version."""


class StaleOperationResultError(DomainInvariantError):
    """Raised when an asynchronous result no longer belongs to active state."""


class TaskName(StrEnum):
    NONE = "none"
    APPOINTMENT_REQUEST = "appointment_request"
    PROVIDER_LOOKUP = "provider_lookup"
    PRACTICE_INFORMATION = "practice_information"


class WorkflowState(StrEnum):
    IDLE = "idle"
    COLLECTING_CONTEXT = "collecting_context"
    SEARCHING_PROVIDERS = "searching_providers"
    SELECTING_PROVIDER = "selecting_provider"
    COLLECTING_DETAILS = "collecting_details"
    REVIEWING_REQUEST = "reviewing_request"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    EXECUTING = "executing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class Channel(StrEnum):
    TEXT = "text"
    VOICE = "voice"


class SlotSource(StrEnum):
    USER_EXPLICIT = "user_explicit"
    USER_CORRECTION = "user_correction"
    TOOL_RESULT = "tool_result"
    PRACTICE_PROFILE = "practice_profile"
    MODEL_INFERENCE = "model_inference"


class OperationName(StrEnum):
    SEARCH_PROVIDERS = "search_providers"
    CREATE_APPOINTMENT_REQUEST = "create_appointment_request"
    GET_PRACTICE_PROFILE = "get_practice_profile"


class OperationStatus(StrEnum):
    REQUESTED = "requested"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    SUPERSEDED = "superseded"


class CommandName(StrEnum):
    START_WORKFLOW = "start_workflow"
    CAPTURE_SLOT = "capture_slot"
    CORRECT_SLOT = "correct_slot"
    REQUEST_OPERATION = "request_operation"
    START_OPERATION = "start_operation"
    COMPLETE_OPERATION = "complete_operation"
    FAIL_OPERATION = "fail_operation"
    TIMEOUT_OPERATION = "timeout_operation"
    CANCEL_OPERATION = "cancel_operation"
    SUPERSEDE_OPERATION = "supersede_operation"
    CANCEL_TASK = "cancel_task"


class SlotRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str = Field(min_length=1)
    source: SlotSource
    confidence: float = Field(ge=0, le=1)
    event_id: str = Field(min_length=1)
    captured_at: datetime


class PendingOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    operation: OperationName
    status: OperationStatus
    requested_state_version: int = Field(ge=0)
    idempotency_key: str = Field(min_length=1)
    correlation_id: str = Field(min_length=1)
    requested_at: datetime
    updated_at: datetime


class ConversationMemory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str | None = None
    recent_message_ids: list[str] = Field(default_factory=list)


class ProviderResultRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    address: str | None = None
    category: str = Field(min_length=1)
    distance_km: float | None = Field(default=None, ge=0)
    phone: str | None = None
    website: str | None = None


class ProviderSearchResultData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result_type: Literal["provider_search"] = "provider_search"
    location: str = Field(min_length=1)
    providers: list[ProviderResultRecord] = Field(default_factory=list)
    source: str = Field(min_length=1)
    error: str | None = None
    error_code: str | None = None


class AppointmentRequestResultData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result_type: Literal["appointment_request"] = "appointment_request"
    request_reference: str = Field(min_length=1)
    status: Literal["submitted", "failed"]
    error: str | None = None


class PracticeHoursRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    day: str = Field(min_length=1)
    opens_at: str | None = None
    closes_at: str | None = None
    closed: bool = False


class PracticeProfileResultData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result_type: Literal["practice_profile"] = "practice_profile"
    profile_id: str = Field(min_length=1)
    profile_version: int = Field(ge=1)
    display_name: str
    address: str | None = None
    phone: str | None = None
    website: str | None = None
    hours: list[PracticeHoursRecord] = Field(default_factory=list)
    services: list[str] = Field(default_factory=list)
    accepted_insurance: list[str] = Field(default_factory=list)
    source: str = Field(min_length=1)
    error: str | None = None
    error_code: str | None = None


OperationResultData = Annotated[
    Union[ProviderSearchResultData, AppointmentRequestResultData, PracticeProfileResultData],
    Field(discriminator="result_type"),
]


class ConversationSession(BaseModel):
    """Top-level consistency boundary for one conversation session."""

    model_config = ConfigDict(extra="forbid")

    session_schema_version: int = Field(default=1, ge=1)
    session_id: str = Field(min_length=1)
    state: "ConversationState"
    memory: ConversationMemory = Field(default_factory=ConversationMemory)
    event_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def session_matches_state(self) -> "ConversationSession":
        if self.state.session_id != self.session_id:
            raise DomainInvariantError("session and state IDs must match")
        return self

    @classmethod
    def create(cls, session_id: str) -> "ConversationSession":
        return cls(session_id=session_id, state=ConversationState(session_id=session_id))


class ConversationState(BaseModel):
    """The current projection of the conversation aggregate."""

    model_config = ConfigDict(extra="forbid")

    state_schema_version: int = Field(default=1, ge=1)
    session_id: str = Field(min_length=1)
    state_version: int = Field(default=0, ge=0)
    turn_count: int = Field(default=0, ge=0)
    active_task: TaskName = TaskName.NONE
    workflow_state: WorkflowState = WorkflowState.IDLE
    slots: dict[str, SlotRecord] = Field(default_factory=dict)
    provider_options: list[ProviderResultRecord] = Field(default_factory=list)
    last_operation_result: OperationResultData | None = None
    pending_operation: PendingOperation | None = None
    last_event_id: str | None = None

    @model_validator(mode="after")
    def enforce_invariants(self) -> "ConversationState":
        validate_state_invariants(self)
        return self


class EventBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    event_schema_version: int = Field(default=1, ge=1)
    aggregate_id: str = Field(min_length=1)
    aggregate_version: int = Field(ge=1)
    correlation_id: str = Field(min_length=1)
    causation_id: str | None = None
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TurnReceivedEvent(EventBase):
    event_type: Literal["conversation.turn.received"] = "conversation.turn.received"
    text: str = Field(min_length=1)
    channel: Channel


class WorkflowStartedEvent(EventBase):
    event_type: Literal["workflow.started"] = "workflow.started"
    task: TaskName


class SlotCapturedEvent(EventBase):
    event_type: Literal["slot.captured"] = "slot.captured"
    slot: str = Field(min_length=1)
    value: str = Field(min_length=1)
    source: SlotSource
    confidence: float = Field(ge=0, le=1)


class SlotCorrectedEvent(EventBase):
    event_type: Literal["slot.corrected"] = "slot.corrected"
    slot: str = Field(min_length=1)
    value: str = Field(min_length=1)
    confidence: float = Field(default=1, ge=0, le=1)


class OperationRequestedEvent(EventBase):
    event_type: Literal["operation.requested"] = "operation.requested"
    request_id: str = Field(min_length=1)
    operation: OperationName
    requested_state_version: int = Field(ge=0)
    idempotency_key: str = Field(min_length=1)


class OperationStartedEvent(EventBase):
    event_type: Literal["operation.started"] = "operation.started"
    request_id: str = Field(min_length=1)


class OperationSucceededEvent(EventBase):
    event_type: Literal["operation.succeeded"] = "operation.succeeded"
    request_id: str = Field(min_length=1)
    requested_state_version: int = Field(ge=0)
    result: OperationResultData | None = None


class OperationFailedEvent(EventBase):
    event_type: Literal["operation.failed"] = "operation.failed"
    request_id: str = Field(min_length=1)
    requested_state_version: int = Field(ge=0)
    error_code: str = Field(min_length=1)
    retryable: bool


class OperationTimedOutEvent(EventBase):
    event_type: Literal["operation.timed_out"] = "operation.timed_out"
    request_id: str = Field(min_length=1)
    requested_state_version: int = Field(ge=0)


class OperationCancelledEvent(EventBase):
    event_type: Literal["operation.cancelled"] = "operation.cancelled"
    request_id: str = Field(min_length=1)
    requested_state_version: int = Field(ge=0)


class OperationSupersededEvent(EventBase):
    event_type: Literal["operation.superseded"] = "operation.superseded"
    request_id: str = Field(min_length=1)
    requested_state_version: int = Field(ge=0)
    reason: str = Field(min_length=1)


class TaskCancelledEvent(EventBase):
    event_type: Literal["task.cancelled"] = "task.cancelled"
    reason: str = Field(min_length=1)


class OperationResultRejectedEvent(EventBase):
    event_type: Literal["operation.result.rejected"] = "operation.result.rejected"
    request_id: str = Field(min_length=1)
    requested_state_version: int = Field(ge=0)
    reason: str = Field(min_length=1)


class CommandBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    command_type: CommandName
    session_id: str = Field(min_length=1)
    expected_state_version: int = Field(ge=0)
    correlation_id: str = Field(min_length=1)
    issued_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StartWorkflowCommand(CommandBase):
    command_type: Literal[CommandName.START_WORKFLOW] = CommandName.START_WORKFLOW
    task: TaskName


class CaptureSlotCommand(CommandBase):
    command_type: Literal[CommandName.CAPTURE_SLOT] = CommandName.CAPTURE_SLOT
    slot: str = Field(min_length=1)
    value: str = Field(min_length=1)
    source: SlotSource
    confidence: float = Field(ge=0, le=1)


class CorrectSlotCommand(CommandBase):
    command_type: Literal[CommandName.CORRECT_SLOT] = CommandName.CORRECT_SLOT
    slot: str = Field(min_length=1)
    value: str = Field(min_length=1)


class RequestOperationCommand(CommandBase):
    command_type: Literal[CommandName.REQUEST_OPERATION] = CommandName.REQUEST_OPERATION
    request_id: str = Field(min_length=1)
    operation: OperationName
    idempotency_key: str = Field(min_length=1)


class StartOperationCommand(CommandBase):
    command_type: Literal[CommandName.START_OPERATION] = CommandName.START_OPERATION
    request_id: str = Field(min_length=1)


class CompleteOperationCommand(CommandBase):
    command_type: Literal[CommandName.COMPLETE_OPERATION] = CommandName.COMPLETE_OPERATION
    request_id: str = Field(min_length=1)
    requested_state_version: int = Field(ge=0)
    result: OperationResultData | None = None


class FailOperationCommand(CommandBase):
    command_type: Literal[CommandName.FAIL_OPERATION] = CommandName.FAIL_OPERATION
    request_id: str = Field(min_length=1)
    requested_state_version: int = Field(ge=0)
    error_code: str = Field(min_length=1)
    retryable: bool


class TimeoutOperationCommand(CommandBase):
    command_type: Literal[CommandName.TIMEOUT_OPERATION] = CommandName.TIMEOUT_OPERATION
    request_id: str = Field(min_length=1)
    requested_state_version: int = Field(ge=0)


class CancelOperationCommand(CommandBase):
    command_type: Literal[CommandName.CANCEL_OPERATION] = CommandName.CANCEL_OPERATION
    request_id: str = Field(min_length=1)
    requested_state_version: int = Field(ge=0)


class SupersedeOperationCommand(CommandBase):
    command_type: Literal[CommandName.SUPERSEDE_OPERATION] = CommandName.SUPERSEDE_OPERATION
    request_id: str = Field(min_length=1)
    requested_state_version: int = Field(ge=0)
    reason: str = Field(min_length=1)


class CancelTaskCommand(CommandBase):
    command_type: Literal[CommandName.CANCEL_TASK] = CommandName.CANCEL_TASK
    reason: str = Field(min_length=1)


DomainCommand = Annotated[
    Union[
        StartWorkflowCommand,
        CaptureSlotCommand,
        CorrectSlotCommand,
        RequestOperationCommand,
        StartOperationCommand,
        CompleteOperationCommand,
        FailOperationCommand,
        TimeoutOperationCommand,
        CancelOperationCommand,
        SupersedeOperationCommand,
        CancelTaskCommand,
    ],
    Field(discriminator="command_type"),
]


DomainEvent = Annotated[
    Union[
        TurnReceivedEvent,
        WorkflowStartedEvent,
        SlotCapturedEvent,
        SlotCorrectedEvent,
        OperationRequestedEvent,
        OperationStartedEvent,
        OperationSucceededEvent,
        OperationFailedEvent,
        OperationTimedOutEvent,
        OperationCancelledEvent,
        OperationSupersededEvent,
        TaskCancelledEvent,
        OperationResultRejectedEvent,
    ],
    Field(discriminator="event_type"),
]


def validate_state_invariants(state: ConversationState) -> None:
    if state.workflow_state == WorkflowState.IDLE and state.active_task != TaskName.NONE:
        raise DomainInvariantError("idle state cannot have an active task")
    if state.workflow_state not in {
        WorkflowState.IDLE,
        WorkflowState.COMPLETED,
        WorkflowState.CANCELLED,
        WorkflowState.FAILED,
    } and state.active_task == TaskName.NONE:
        raise DomainInvariantError("active workflow state requires an active task")
    if state.pending_operation and state.pending_operation.status not in {
        OperationStatus.REQUESTED,
        OperationStatus.RUNNING,
    }:
        raise DomainInvariantError("pending_operation cannot contain a terminal operation")
    if state.pending_operation and state.pending_operation.requested_state_version > state.state_version:
        raise DomainInvariantError("operation cannot target a future state version")


def reduce_state(state: ConversationState, event: DomainEvent) -> ConversationState:
    """Apply exactly one event to a state projection without side effects."""

    from libs.conversation.transitions import transition_for

    if event.aggregate_id != state.session_id:
        raise DomainInvariantError("event aggregate does not match session state")
    if event.aggregate_version != state.state_version + 1:
        raise StateConflictError(
            f"expected aggregate version {state.state_version + 1}, "
            f"received {event.aggregate_version}"
        )
    transition = transition_for(state, event)

    next_state = state.model_copy(deep=True)
    next_state.state_version = event.aggregate_version
    next_state.last_event_id = event.event_id

    if isinstance(event, TurnReceivedEvent):
        next_state.turn_count += 1
    elif isinstance(event, WorkflowStartedEvent):
        if next_state.workflow_state not in {
            WorkflowState.IDLE,
            WorkflowState.COMPLETED,
            WorkflowState.CANCELLED,
            WorkflowState.FAILED,
        }:
            raise DomainInvariantError("cannot start a workflow while another is active")
        next_state.active_task = event.task
        next_state.workflow_state = WorkflowState.COLLECTING_CONTEXT
    elif isinstance(event, (SlotCapturedEvent, SlotCorrectedEvent)):
        source = (
            SlotSource.USER_CORRECTION
            if isinstance(event, SlotCorrectedEvent)
            else event.source
        )
        next_state.slots[event.slot] = SlotRecord(
            value=event.value,
            source=source,
            confidence=event.confidence,
            event_id=event.event_id,
            captured_at=event.occurred_at,
        )
        if event.slot in {"care_setting", "location", "appointment_reason"}:
            next_state.provider_options = []
            next_state.last_operation_result = None
            if isinstance(event, SlotCorrectedEvent):
                next_state.slots.pop("provider_id", None)
                next_state.slots.pop("provider_name", None)
                next_state.workflow_state = WorkflowState.COLLECTING_CONTEXT
            if state.workflow_state == WorkflowState.SELECTING_PROVIDER:
                next_state.workflow_state = WorkflowState.COLLECTING_CONTEXT
        if (
            state.workflow_state == WorkflowState.SELECTING_PROVIDER
            and event.slot in {"provider_id", "provider_name"}
        ):
            next_state.workflow_state = WorkflowState.COLLECTING_DETAILS
    elif isinstance(event, OperationRequestedEvent):
        if next_state.pending_operation is not None:
            raise DomainInvariantError("cannot request an operation while one is pending")
        if event.requested_state_version != state.state_version:
            raise StateConflictError("operation was requested against a stale state version")
        next_state.pending_operation = PendingOperation(
            request_id=event.request_id,
            operation=event.operation,
            status=OperationStatus.REQUESTED,
            requested_state_version=event.requested_state_version,
            idempotency_key=event.idempotency_key,
            correlation_id=event.correlation_id,
            requested_at=event.occurred_at,
            updated_at=event.occurred_at,
        )
        next_state.workflow_state = (
            WorkflowState.SEARCHING_PROVIDERS
            if event.operation == OperationName.SEARCH_PROVIDERS
            else WorkflowState.EXECUTING
        )
    elif isinstance(event, OperationStartedEvent):
        operation = _require_pending_operation(next_state, event.request_id)
        operation.status = OperationStatus.RUNNING
        operation.updated_at = event.occurred_at
    elif isinstance(
        event,
        (
            OperationSucceededEvent,
            OperationFailedEvent,
            OperationTimedOutEvent,
            OperationCancelledEvent,
            OperationSupersededEvent,
        ),
    ):
        operation = _require_pending_operation(next_state, event.request_id)
        if event.requested_state_version != operation.requested_state_version:
            raise StaleOperationResultError("operation result belongs to an older request version")
        if isinstance(event, OperationSucceededEvent):
            next_state.last_operation_result = event.result
            if isinstance(event.result, ProviderSearchResultData):
                next_state.provider_options = event.result.providers
            next_state.workflow_state = (
                WorkflowState.SELECTING_PROVIDER
                if operation.operation == OperationName.SEARCH_PROVIDERS
                else WorkflowState.COMPLETED
            )
            if operation.operation == OperationName.GET_PRACTICE_PROFILE:
                next_state.active_task = TaskName.NONE
        elif isinstance(event, OperationSupersededEvent):
            next_state.workflow_state = WorkflowState.COLLECTING_CONTEXT
        else:
            next_state.workflow_state = WorkflowState.FAILED
        next_state.pending_operation = None
    elif isinstance(event, TaskCancelledEvent):
        next_state.pending_operation = None
        next_state.active_task = TaskName.NONE
        next_state.slots = {}
        next_state.provider_options = []
        next_state.last_operation_result = None
        next_state.workflow_state = WorkflowState.CANCELLED
    elif isinstance(event, OperationResultRejectedEvent):
        pass

    validate_state_invariants(next_state)
    if next_state.workflow_state != transition.target_state:
        raise DomainInvariantError(
            f"reducer produced {next_state.workflow_state} but transition table requires "
            f"{transition.target_state}"
        )
    return next_state


def _require_pending_operation(state: ConversationState, request_id: str) -> PendingOperation:
    operation = state.pending_operation
    if operation is None or operation.request_id != request_id:
        raise StaleOperationResultError("operation result does not match the pending operation")
    return operation


def reduce_session(session: ConversationSession, event: DomainEvent) -> ConversationSession:
    """Apply an event to the session projection and append its event ID."""

    if event.aggregate_id != session.session_id:
        raise DomainInvariantError("event aggregate does not match conversation session")
    next_state = reduce_state(session.state, event)
    next_session = session.model_copy(deep=True)
    next_session.state = next_state
    next_session.event_ids.append(event.event_id)
    next_session.updated_at = event.occurred_at
    return next_session


def command_to_event(state: ConversationState, command: DomainCommand) -> DomainEvent:
    """Validate a command against current state and produce one typed event."""

    if command.session_id != state.session_id:
        raise DomainInvariantError("command session does not match state session")
    if command.expected_state_version != state.state_version:
        raise StateConflictError("command was issued against a stale state version")

    event_defaults = {
        "aggregate_id": state.session_id,
        "aggregate_version": state.state_version + 1,
        "correlation_id": command.correlation_id,
        "causation_id": command.command_id,
        "occurred_at": command.issued_at,
    }
    if isinstance(command, StartWorkflowCommand):
        if state.workflow_state not in {
            WorkflowState.IDLE,
            WorkflowState.COMPLETED,
            WorkflowState.CANCELLED,
            WorkflowState.FAILED,
        }:
            raise DomainInvariantError("cannot start a workflow while another is active")
        return WorkflowStartedEvent(**event_defaults, task=command.task)
    if isinstance(command, CaptureSlotCommand):
        return SlotCapturedEvent(
            **event_defaults,
            slot=command.slot,
            value=command.value,
            source=command.source,
            confidence=command.confidence,
        )
    if isinstance(command, CorrectSlotCommand):
        return SlotCorrectedEvent(**event_defaults, slot=command.slot, value=command.value)
    if isinstance(command, RequestOperationCommand):
        if state.pending_operation is not None:
            raise DomainInvariantError("cannot request an operation while one is pending")
        return OperationRequestedEvent(
            **event_defaults,
            request_id=command.request_id,
            operation=command.operation,
            requested_state_version=state.state_version,
            idempotency_key=command.idempotency_key,
        )
    if isinstance(command, StartOperationCommand):
        _require_pending_operation(state, command.request_id)
        return OperationStartedEvent(**event_defaults, request_id=command.request_id)
    if isinstance(command, CompleteOperationCommand):
        operation = _require_pending_operation(state, command.request_id)
        _validate_operation_version(operation, command.requested_state_version)
        return OperationSucceededEvent(
            **event_defaults,
            request_id=command.request_id,
            requested_state_version=command.requested_state_version,
            result=command.result,
        )
    if isinstance(command, FailOperationCommand):
        operation = _require_pending_operation(state, command.request_id)
        _validate_operation_version(operation, command.requested_state_version)
        return OperationFailedEvent(
            **event_defaults,
            request_id=command.request_id,
            requested_state_version=command.requested_state_version,
            error_code=command.error_code,
            retryable=command.retryable,
        )
    if isinstance(command, TimeoutOperationCommand):
        operation = _require_pending_operation(state, command.request_id)
        _validate_operation_version(operation, command.requested_state_version)
        return OperationTimedOutEvent(
            **event_defaults,
            request_id=command.request_id,
            requested_state_version=command.requested_state_version,
        )
    if isinstance(command, CancelOperationCommand):
        operation = _require_pending_operation(state, command.request_id)
        _validate_operation_version(operation, command.requested_state_version)
        return OperationCancelledEvent(
            **event_defaults,
            request_id=command.request_id,
            requested_state_version=command.requested_state_version,
        )
    if isinstance(command, SupersedeOperationCommand):
        operation = _require_pending_operation(state, command.request_id)
        _validate_operation_version(operation, command.requested_state_version)
        return OperationSupersededEvent(
            **event_defaults,
            request_id=command.request_id,
            requested_state_version=command.requested_state_version,
            reason=command.reason,
        )
    if isinstance(command, CancelTaskCommand):
        return TaskCancelledEvent(**event_defaults, reason=command.reason)
    raise TypeError(f"unsupported domain command: {type(command).__name__}")


def _validate_operation_version(operation: PendingOperation, requested_state_version: int) -> None:
    if requested_state_version != operation.requested_state_version:
        raise StaleOperationResultError("operation result belongs to an older request version")
