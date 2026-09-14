"""Explicit workflow transition table for the replacement conversation engine."""

from dataclasses import dataclass
from typing import Any

from libs.conversation.domain import (
    DomainInvariantError,
    StaleOperationResultError,
    OperationCancelledEvent,
    OperationFailedEvent,
    OperationRequestedEvent,
    OperationStartedEvent,
    OperationSucceededEvent,
    OperationSupersededEvent,
    OperationTimedOutEvent,
    OperationResultRejectedEvent,
    SlotCapturedEvent,
    SlotCorrectedEvent,
    TaskCancelledEvent,
    TaskName,
    TurnReceivedEvent,
    WorkflowStartedEvent,
    WorkflowState,
)


@dataclass(frozen=True)
class TransitionRule:
    event_type: str
    source_states: frozenset[WorkflowState]
    target_state: WorkflowState | None
    guard: str


@dataclass(frozen=True)
class TransitionDecision:
    event_type: str
    source_state: WorkflowState
    target_state: WorkflowState
    guard: str


_ACTIVE_STATES = frozenset(
    {
        WorkflowState.COLLECTING_CONTEXT,
        WorkflowState.SELECTING_PROVIDER,
        WorkflowState.COLLECTING_DETAILS,
        WorkflowState.REVIEWING_REQUEST,
        WorkflowState.AWAITING_CONFIRMATION,
    }
)
_TERMINAL_STATES = frozenset(
    {WorkflowState.IDLE, WorkflowState.COMPLETED, WorkflowState.CANCELLED, WorkflowState.FAILED}
)
_ALL_STATES = frozenset(WorkflowState)


TRANSITION_TABLE = (
    TransitionRule(
        event_type="conversation.turn.received",
        source_states=_ALL_STATES,
        target_state=None,
        guard="session accepts the turn",
    ),
    TransitionRule(
        event_type="workflow.started",
        source_states=_TERMINAL_STATES,
        target_state=WorkflowState.COLLECTING_CONTEXT,
        guard="task is not none",
    ),
    TransitionRule(
        event_type="slot.captured",
        source_states=_ACTIVE_STATES,
        target_state=None,
        guard="slot is accepted by the active workflow",
    ),
    TransitionRule(
        event_type="slot.corrected",
        source_states=_ACTIVE_STATES,
        target_state=None,
        guard="correction targets a known workflow slot",
    ),
    TransitionRule(
        event_type="operation.requested",
        source_states=_ACTIVE_STATES,
        target_state=None,
        guard="no operation is pending and request targets current state",
    ),
    TransitionRule(
        event_type="operation.started",
        source_states=frozenset({WorkflowState.SEARCHING_PROVIDERS, WorkflowState.EXECUTING}),
        target_state=None,
        guard="matching operation is requested",
    ),
    TransitionRule(
        event_type="operation.succeeded",
        source_states=frozenset({WorkflowState.SEARCHING_PROVIDERS, WorkflowState.EXECUTING}),
        target_state=None,
        guard="matching current operation and request version",
    ),
    TransitionRule(
        event_type="operation.failed",
        source_states=frozenset({WorkflowState.SEARCHING_PROVIDERS, WorkflowState.EXECUTING}),
        target_state=WorkflowState.FAILED,
        guard="matching current operation and request version",
    ),
    TransitionRule(
        event_type="operation.timed_out",
        source_states=frozenset({WorkflowState.SEARCHING_PROVIDERS, WorkflowState.EXECUTING}),
        target_state=WorkflowState.FAILED,
        guard="matching current operation and request version",
    ),
    TransitionRule(
        event_type="operation.cancelled",
        source_states=frozenset({WorkflowState.SEARCHING_PROVIDERS, WorkflowState.EXECUTING}),
        target_state=WorkflowState.FAILED,
        guard="matching current operation and request version",
    ),
    TransitionRule(
        event_type="operation.superseded",
        source_states=frozenset({WorkflowState.SEARCHING_PROVIDERS, WorkflowState.EXECUTING}),
        target_state=WorkflowState.COLLECTING_CONTEXT,
        guard="matching current operation and request version",
    ),
    TransitionRule(
        event_type="operation.result.rejected",
        source_states=_ALL_STATES,
        target_state=None,
        guard="rejection is audit-only",
    ),
    TransitionRule(
        event_type="task.cancelled",
        source_states=_ALL_STATES,
        target_state=WorkflowState.CANCELLED,
        guard="cancellation is explicit",
    ),
)


def transition_for(state: Any, event: Any) -> TransitionDecision:
    """Resolve and validate the state + event transition without mutating state."""

    rule = next((candidate for candidate in TRANSITION_TABLE if candidate.event_type == event.event_type), None)
    if rule is None:
        raise DomainInvariantError(f"no transition rule for event {event.event_type}")
    if isinstance(
        event,
        (
            OperationSucceededEvent,
            OperationFailedEvent,
            OperationTimedOutEvent,
            OperationCancelledEvent,
            OperationSupersededEvent,
        ),
    ) and (
        state.pending_operation is None
        or state.pending_operation.request_id != event.request_id
    ):
        raise StaleOperationResultError("operation result does not match the active operation")
    if state.workflow_state not in rule.source_states:
        raise DomainInvariantError(
            f"event {event.event_type} is not allowed from {state.workflow_state}"
        )
    if isinstance(event, WorkflowStartedEvent) and event.task == TaskName.NONE:
        raise DomainInvariantError("cannot start a none task")
    if isinstance(event, OperationStartedEvent):
        _require_operation(state, event.request_id, {"requested"})
    if isinstance(
        event,
        (
            OperationSucceededEvent,
            OperationFailedEvent,
            OperationTimedOutEvent,
            OperationCancelledEvent,
            OperationSupersededEvent,
        ),
    ):
        operation = _require_operation(state, event.request_id, {"requested", "running"})
        if event.requested_state_version != operation.requested_state_version:
            raise DomainInvariantError("operation result targets a stale request version")
    target_state = rule.target_state or state.workflow_state
    if isinstance(event, OperationRequestedEvent):
        target_state = (
            WorkflowState.SEARCHING_PROVIDERS
            if event.operation.value == "search_providers"
            else WorkflowState.EXECUTING
        )
    if isinstance(event, (SlotCapturedEvent, SlotCorrectedEvent)):
        if state.workflow_state == WorkflowState.SELECTING_PROVIDER and event.slot in {
            "provider_id",
            "provider_name",
        }:
            target_state = WorkflowState.COLLECTING_DETAILS
        elif state.workflow_state == WorkflowState.SELECTING_PROVIDER and event.slot in {
            "care_setting",
            "location",
            "appointment_reason",
        }:
            target_state = WorkflowState.COLLECTING_CONTEXT
    if isinstance(event, OperationSucceededEvent):
        operation = _require_operation(state, event.request_id, {"requested", "running"})
        target_state = (
            WorkflowState.SELECTING_PROVIDER
            if operation.operation.value == "search_providers"
            else WorkflowState.COMPLETED
        )
    return TransitionDecision(
        event_type=event.event_type,
        source_state=state.workflow_state,
        target_state=target_state,
        guard=rule.guard,
    )


def _require_operation(state: Any, request_id: str, statuses: set[str]) -> Any:
    operation = state.pending_operation
    if operation is None or operation.request_id != request_id:
        raise StaleOperationResultError("event does not match the pending operation")
    if operation.status.value not in statuses:
        raise DomainInvariantError("operation is not in a valid state for this event")
    return operation
