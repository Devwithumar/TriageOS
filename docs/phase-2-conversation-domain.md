# Phase 2 Conversation Domain

This document defines the foundational implementation slices of the replacement conversation engine. It does not replace the frozen receptionist prototype yet.

## Source of truth

`ConversationSession` is the top-level consistency boundary for one conversation. The active workflow is a component of that session for the MVP, not a separate aggregate. `ConversationState` is the current projection inside the session. It is not an event log and it does not contain arbitrary model output.

```text
Typed domain event
        |
        v
Transition table + pure reducer
        |
        v
ConversationState projection
```

Events are append-only facts. State is rebuilt by applying events in aggregate-version order. A state update must never be performed without a corresponding typed event. The transition table defines the allowed state + event + guard combinations; the reducer enforces that specification.

PostgreSQL stores the durable event history and the current state projection in one transaction. Redis may provide cache, locks, and ephemeral execution coordination, but it is never the only copy of conversational truth.

Commands are validated before event creation:

```text
Command
  ↓ validate expected state version
Typed domain event
  ↓ pure reducer
New ConversationState
  ↓ one database transaction
Persist event + state projection
```

## State rules

- `state_version` starts at zero and increases exactly once per accepted event.
- `state_schema_version` and `event_schema_version` evolve independently.
- An event must target the next aggregate version.
- Every event carries an event ID, event schema version, aggregate ID, aggregate version, timestamp, correlation ID, and causation ID.
- Only one operation may be pending for a conversation aggregate.
- A pending operation may only be `requested` or `running` in the current state projection.
- A tool result must match the active `request_id` and its original requested state version.
- A stale or mismatched tool result is rejected rather than merged into current state.
- A rejected stale result may be recorded as an audit event, but it has no domain-state effect.
- Slots carry their source, confidence, event ID, and capture time.
- Inference is not equivalent to an explicit user statement.
- A workflow cannot be active without an active task.
- An idle workflow cannot retain an active task.

## Operation lifecycle

```text
requested -> running -> succeeded
                    -> failed
                    -> timed_out
                    -> cancelled
                    -> superseded
```

The current state stores only the active operation. Terminal operation facts remain in the event stream and audit projection.

## Implemented scope

The current foundation includes:

- the `ConversationSession` consistency boundary;
- versioned state, events, and commands;
- slot provenance and pending-operation contracts;
- a pure command-to-event validator;
- an explicit transition table;
- a pure reducer that enforces the table;
- stale-operation rejection and audit-only rejection events.

It does not yet:

- connect the new reducer to the live service;
- choose PostgreSQL tables or Redis keys;
- define provider-specific tool payloads;
- change user-facing responses.

Those are separate design and implementation slices.
