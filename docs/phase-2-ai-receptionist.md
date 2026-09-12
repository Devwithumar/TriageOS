# Phase 2: AI Receptionist

## Current Slice

The Conversation Service now supports a structured receptionist workflow alongside general conversation:

- Detects appointment requests and basic practice-information questions
- Collects caller name, callback number, and preferred time
- Persists structured caller state for the conversation session
- Returns a clear request-ready response without claiming that an appointment was booked
- Keeps clinical questions outside this phase

## Deliberate Boundaries

This slice does not yet connect to a calendar, send notifications, provide medical advice, or perform triage. Appointment availability and booking will be added after the workflow is tested with a mocked scheduling tool.

## Next Milestones

1. Add a practice profile for hours, location, phone, and accepted insurance
2. Add a mocked scheduling tool with deterministic available slots
3. Add confirmation and correction handling
4. Add calendar integration behind the scheduling service boundary
