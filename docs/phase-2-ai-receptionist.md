# Phase 2: AI Receptionist

## Current Slice

The Conversation Service now supports a structured receptionist workflow alongside general conversation:

- Detects appointment requests and basic practice-information questions
- Collects caller name, callback number, and preferred time
- Persists structured caller state for the conversation session
- Returns a clear request-ready response without claiming that an appointment was booked
- Keeps clinical questions outside this phase
- Loads a versioned practice profile through a typed deterministic operation
- Answers hours, location, contact, services, and insurance questions only from that profile

## Deliberate Boundaries

This slice does not yet connect to a calendar, send notifications, provide medical advice, or perform triage. If `TRIAGEOS_PRACTICE_PROFILE_PATH` is unset, practice facts remain unavailable rather than being invented. Appointment availability and booking are represented by an explicitly opt-in mock scheduler for deterministic local tests; the default runtime does not expose demonstration slots or claim to submit requests until a real scheduling adapter is configured.

The profile file must contain the following versioned fields:

```json
{
  "profile_id": "practice:example",
  "profile_version": 1,
  "display_name": "Example Practice",
  "address": "Configured address",
  "phone": "Configured phone",
  "website": "https://example.test",
  "hours": [
    {"day": "Monday", "opens_at": "08:00", "closes_at": "17:00", "closed": false}
  ],
  "services": ["primary care"],
  "accepted_insurance": ["Example Health"]
}
```

The default path is `config/practice_profile.json`; production deployments should provide this path through configuration and keep organization-specific data out of source control.

## Next Milestones

1. Add a calendar integration behind the scheduling service boundary
2. Add provider-specific availability normalization and idempotent submission
3. Add notification delivery after confirmed scheduling
