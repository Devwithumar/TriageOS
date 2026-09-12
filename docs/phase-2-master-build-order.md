# TriageOS Phase 2: Master Build Order

The build order is intentionally conservative. Each stage must pass its regression suite before the next stage begins.

## Stage 0: Freeze and Baseline

- Freeze the current receptionist prototype.
- Do not add more intent-specific reply branches to it.
- Capture golden conversations for Phase 1 and the prototype receptionist.
- Record current behavior, known limitations, and safety expectations.

## Stage 1: Contracts and Schemas

Define typed schemas for:

- Conversation state
- Intent and confidence
- Dialogue act
- Extracted entities and slot provenance
- Policy decisions
- Workflow transitions
- Tool requests and results
- Assistant turn responses

No workflow code should pass untyped dictionaries across service boundaries after this stage.

## Stage 2: Deterministic Orchestrator

Build the workflow engine with stubbed classifiers and tools. Prove that valid state transitions, missing fields, corrections, cancellations, and confirmations work without an LLM.

## Stage 3: Policy and Safety Gates

Separate emergency handling, clinical deflection, privacy rules, confirmation requirements, and unsupported-request handling from the language model.

## Stage 4: Practice Knowledge

Add a versioned practice profile and deterministic lookup for hours, location, contact details, services, and insurance information.

## Stage 5: Mock Scheduling Tool

Add deterministic available slots, slot selection, review, confirmation, and request submission. The assistant must distinguish between `slot_reserved`, `appointment_created`, and `submission_failed`.

## Stage 6: Model Adapters

Use the LLM for classification, extraction, and response wording behind strict schemas. Add validation, confidence thresholds, fallback behavior, and cost tracking.

## Stage 7: Voice Integration

Expose the stable text workflow through the Voice Service. Validate interruptions, reconnects, latency, duplicate transcripts, and spoken response length.

## Stage 8: Evaluation and Observability

Add golden transcripts, adversarial conversations, latency metrics, tool success rates, fallback rates, policy violations, and token-cost measurements.

## Stage 9: Real Scheduling Integration

Only after the mocked workflow is reliable should we connect Google Calendar, Outlook, or a dedicated Scheduling Service.

## Stage 10: Later Clinical Expansion

Triage, symptom extraction, medical knowledge retrieval, FHIR, and EHR actions remain separate phases with separate policy and evaluation requirements.
