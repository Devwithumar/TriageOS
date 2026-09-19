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
- Keeps scheduling behind a typed provider boundary with opt-in mock and Google Calendar adapters

## Deliberate Boundaries

This slice does not yet send notifications, provide medical advice, or perform triage. If `TRIAGEOS_PRACTICE_PROFILE_PATH` is unset, practice facts remain unavailable rather than being invented. Scheduling is fail-closed by default; the mock adapter is for deterministic local tests, and the Google Calendar adapter requires explicit credentials and calendar configuration.

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

1. Add OAuth token acquisition and refresh for calendar providers
2. Add provider-specific availability normalization and idempotent submission evaluation
3. Add notification delivery after confirmed scheduling
