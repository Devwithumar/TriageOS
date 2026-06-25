# TriageOS Events

These event names define the product's event-driven backbone. Phase 1 only emits session-level events locally, but the names are stable enough to wire into NATS later.

## Voice Events

- `voice.session.started`
- `voice.audio.chunk`
- `voice.user.transcript.partial`
- `voice.user.transcript.final`
- `voice.assistant.response.created`
- `voice.interruption.detected`
- `voice.session.ended`

## Product Events

- `patient.registered`
- `appointment.created`
- `triage.completed`
- `summary.generated`
- `ehr.updated`

## Event Shape

```json
{
  "event": "voice.session.started",
  "session_id": "session_123",
  "timestamp": "2026-06-25T12:00:00Z",
  "payload": {}
}
```
