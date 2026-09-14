# Phase 2 LLM Proposal Boundary

The language model is an interpreter, not a state manager.

```text
User turn + current session projection
              |
              v
       Structured proposal
              |
              v
   Deterministic proposal validator
              |
              v
       Domain commands
              |
              v
       Typed domain events
```

## Proposal rules

- Proposals carry the session ID and state version they were based on.
- A stale proposal is rejected before it can create a command.
- Unknown slots are rejected.
- Low-confidence proposals require clarification rather than state mutation.
- Location, provider identity, appointment time, and patient contact fields require explicit user confirmation; model inference alone cannot satisfy them.
- A proposal cannot start a second task while another task is active.
- A tool selection is rejected until the required validated slots exist and no operation is pending.
- Cancellation cannot be combined with other state mutations in one proposal.
- The response draft is non-authoritative and is never persisted as domain state.

The validator may emit multiple sequential commands for one turn. Each command carries the state version it expects, and the validator simulates each command through the pure reducer before returning the command list.

The live LLM adapter will be added later. It must produce only the `ConversationProposal` schema; it must not receive a state-mutation API.
