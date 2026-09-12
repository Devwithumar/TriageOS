# TriageOS Phase 2: Master Architecture

## Decision

The current receptionist implementation is frozen as a prototype. It demonstrates the desired interaction, but it is not the production workflow foundation. Future Phase 2 work must use the orchestrated design in this document.

## Product Boundary

Phase 2 is an AI receptionist, not a clinical assistant. It may answer configured practice questions, collect caller details, prepare appointment requests, and hand off to scheduling. It must not diagnose, triage symptoms, provide medical advice, or claim that an appointment was booked without a successful scheduling tool result.

## Turn Pipeline

```text
Voice or text transcript
        |
        v
Safety and policy gate
        |
        v
Intent and dialogue-act classification
        |
        v
Entity and slot extraction
        |
        v
Conversation state load
        |
        v
Workflow transition and validation
        |
        +--> Knowledge retrieval, when needed
        |
        +--> Tool execution, when authorized
        |
        v
Response policy and spoken rendering
        |
        v
State persistence, audit event, and metrics
```

## Layer Responsibilities

## Reusing the WhatsApp Architecture

The previous WhatsApp chatbot gives us a useful conceptual foundation, but the voice channel adds stricter latency, interruption, and safety requirements:

| WhatsApp layer | TriageOS equivalent |
| --- | --- |
| Intent classification | Intent and dialogue-act layer |
| Small talk | Small-Talk layer |
| Policy layer | Safety and Policy layer |
| RAG layer | Knowledge layer, added only for approved document-backed use cases |
| Bot response | Response layer after workflow validation |
| Chat memory | Canonical state plus bounded conversation history |

The key change is adding an explicit workflow and tool layer between classification and response. In TriageOS, a natural-language answer is not enough; the system must prove what state changed and what action actually completed.

### Safety and Policy Layer

Runs before normal receptionist behavior. It owns emergency escalation, privacy boundaries, unsupported clinical requests, confirmation requirements, and tool authorization. Safety decisions are deterministic and cannot be overridden by the language model.

### Intent and Dialogue Layer

Classifies what the caller is trying to do and what conversational action is needed. Initial intents include greeting, small talk, practice information, appointment request, appointment change, appointment cancellation, confirmation, correction, human handoff, and unsupported clinical request.

The classifier must return structured data and confidence. It must not generate the final response or mutate state.

### Small-Talk Layer

Handles greetings, gratitude, conversational transitions, and short general exchanges. It keeps the experience natural without allowing small talk to bypass policy or workflow state.

### State and Workflow Layer

Owns the canonical session state and valid transitions. The model may propose extracted values, but only this layer can apply validated state changes.

### Knowledge Layer

Answers configured practice questions from an approved practice profile. It should prefer structured data over free-form generation.

### Tool Layer

Owns scheduling, notifications, and future EHR operations. Every tool call must have a typed input, authorization rule, result status, and audit event.

### Response Layer

Turns the validated workflow result into a short, natural, voice-friendly response. It must describe only completed actions and must not invent availability, policies, or confirmations.

## Canonical Turn Contract

Every turn should converge on a structure equivalent to:

```json
{
  "intent": {
    "name": "appointment_request",
    "confidence": 0.96
  },
  "dialogue_act": "request_information",
  "entities": {
    "caller_name": "Alex Johnson",
    "callback_number": "08098765432",
    "preferred_time": "next Tuesday afternoon"
  },
  "workflow": {
    "name": "receptionist",
    "state": "collecting_details",
    "next_action": "ask_for_missing_field",
    "missing_fields": ["appointment_reason"]
  },
  "policy": {
    "decision": "allow",
    "requires_confirmation": false
  },
  "tool_call": null,
  "response": "What would you like the appointment to be about?"
}
```

The response is an output of the orchestration decision, not the source of truth.

## RAG Decision

RAG is not required for the first receptionist slice. We should use structured practice configuration for hours, location, contact information, services, and insurance policies.

RAG becomes appropriate later for approved, versioned knowledge such as practice documents, preparation instructions, or internal operating procedures. It should not be used as the first source for emergency guidance or clinical triage. Those workflows require explicit policy and clinical governance.

## State Machine

```text
IDLE
  -> COLLECTING_DETAILS
  -> ANSWERING_QUESTION
  -> HANDOFF_REQUESTED

COLLECTING_DETAILS
  -> REVIEWING_REQUEST
  -> CANCELLED
  -> SAFETY_ESCALATION

REVIEWING_REQUEST
  -> CONFIRMED
  -> CORRECTION_REQUIRED
  -> CANCELLED

CONFIRMED
  -> SUBMITTED
  -> SUBMISSION_FAILED

SUBMITTED
  -> COMPLETED
```

Any state may enter `SAFETY_ESCALATION` when the policy layer detects a potentially urgent symptom.
