# Phase 2 Orchestration Pipeline

This slice connects the structured proposal boundary to the session aggregate without changing the live receptionist service.

## Turn transaction

Each turn follows one deterministic sequence:

```text
user turn
  -> conversation.turn.received
  -> state projection
  -> structured LLM proposal
  -> deterministic proposal validation
  -> typed domain commands
  -> typed domain events
  -> one atomic session commit
```

The model never mutates the session directly. The validator converts an accepted proposal into commands, and the reducer converts commands into events against explicit state versions.

## Consistency behavior

The repository commits the complete event batch inside one consistency boundary. If any event is invalid, no event from that batch is stored. The commit checks:

- The session is still at the expected state version.
- Every event belongs to the session aggregate.
- Event versions are sequential within the batch.
- A replayed complete batch has the same event payloads.
- A partial replay or event ID reuse is rejected.

If proposal generation or deterministic validation fails, the user turn is still committed as an audit fact, but no proposed workflow mutation is applied. The caller receives a typed orchestration result with the failure reason.

## Concurrency behavior

The orchestrator performs proposal work outside the repository lock. A concurrent turn can therefore advance the session while an LLM request is in flight. The final batch commit then fails its optimistic version check instead of overwriting newer state.

The orchestrator is intentionally isolated from the current runtime. The next integration slice will add an application-facing adapter, response policy, and asynchronous tool dispatch without allowing the old state machine and the new aggregate to share mutable state.
