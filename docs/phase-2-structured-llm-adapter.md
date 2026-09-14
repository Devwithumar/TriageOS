# Phase 2 Structured LLM Adapter

The structured adapter is the only boundary through which the replacement orchestration layer may consume model interpretation.

## Contract

```text
ConversationState + bounded history + user turn
                    |
                    v
              LLM provider
                    |
                    v
             JSON object only
                    |
                    v
          ConversationProposal
                    |
                    v
        Deterministic validator
```

The adapter does not mutate state, execute tools, or render a response. It rejects malformed JSON, unknown fields, mismatched session IDs, stale state versions, and incorrect correlation IDs.

OpenAI-compatible providers use JSON-object response mode. Anthropic responses are parsed through the same strict Pydantic schema. The local stub follows the same output contract so tests do not depend on a network provider.

The model receives only a bounded recent-message window and a serialized current projection. User text and history are explicitly untrusted content. The model cannot change the session identity, state version, workflow state, provenance, or pending operation.

## Deliberate boundary

The adapter is not yet connected to the live receptionist service. Integration will be a separate slice that loads the session, requests a proposal, validates it, commits commands as events, and only then renders a grounded response.
