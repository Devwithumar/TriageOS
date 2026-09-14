# Phase 2 Tool and Persistence Contracts

This slice defines the boundary between deterministic conversation state and asynchronous tools. It does not connect the contracts to the frozen receptionist runtime.

## Tool boundary

```text
ConversationState
      |
      v
Typed ToolRequest
      |
      v
Tool adapter / worker
      |
      v
Typed ToolResult
```

Every request includes the session ID, operation, state version it was created against, correlation ID, request ID, and idempotency key. Every result repeats the correlation and requested state version so the domain layer can reject stale or misrouted results.

Successful results require typed output. Failed, timed-out, cancelled, or superseded results require a structured error. Neither request nor result permits arbitrary payload blobs.

## Persistence boundary

`SessionRepository.commit()` is the transaction seam. A PostgreSQL implementation must:

1. Lock or version-check the current session projection.
2. Verify the expected state version.
3. Append the domain event.
4. Write the resulting state projection.
5. Commit both writes atomically.

If any step fails, neither the event nor state projection is visible as committed. Duplicate event delivery with the same event ID and payload is idempotent. Reuse of an event ID with a different payload is a conflict.

Redis may provide locks, cache, and ephemeral worker coordination, but recovery must be possible from PostgreSQL state and event history alone.
