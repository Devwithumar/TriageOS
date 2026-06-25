# Phase 1: Voice Engine MVP

## Goal

Have a natural spoken conversation with an AI.

This phase deliberately excludes healthcare-specific logic:

- No patient intake
- No symptom triage
- No scheduling
- No EHR or FHIR writes

## Architecture

```text
Browser Voice Client
  |
  | WebSocket session events
  v
Voice Service
  |
  | HTTP turn request
  v
Conversation Service
  |
  | Redis-backed state
  v
Session Memory
```

## Responsibilities

### Voice Service

- Owns realtime WebSocket sessions
- Receives browser speech transcripts
- Buffers audio chunk metadata for future STT providers
- Handles interruption events
- Calls the Conversation Service for AI turns
- Emits assistant text and TTS instructions

### Conversation Service

- Maintains conversation state
- Builds compact context windows
- Produces structured assistant turns
- Tracks token-cost metadata placeholders
- Exposes a clean API for future LangGraph orchestration

## Future Upgrade Path

The first MVP uses browser STT and TTS to reduce cost and complexity. Later, the Voice Service can swap in:

- Deepgram or Whisper for server-side STT
- ElevenLabs or OpenAI TTS for server-side voice output
- WebRTC for lower-latency bidirectional audio
- NATS events for session lifecycle and analytics
