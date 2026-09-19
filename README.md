# TriageOS

TriageOS is an AI-powered voice operating system for healthcare providers. The long-term product automates patient intake, symptom triage, appointment scheduling, clinical documentation, and EHR integrations through natural conversations.

## Roadmap

1. Real-time voice conversation
2. AI receptionist
3. Triage engine
4. Doctor dashboard
5. FHIR integration
6. Analytics

## Phase 1: Voice Engine MVP

The first milestone is intentionally narrow:

- Talk to an AI using your voice
- Keep conversation state
- Stream realtime session events over WebSockets
- Avoid healthcare, triage, scheduling, and EHR logic for now

## Local Stack

- Web client: browser SpeechRecognition where available, with MediaRecorder fallback
- Voice Service: FastAPI WebSocket gateway
- Conversation Service: FastAPI stateful conversation brain
- Redis: session state and memory
- PostgreSQL with pgvector: future structured data and embeddings
- NATS: future event-driven workflows

## Quick Start

```powershell
docker-compose -f infra/docker-compose.yml up --build
```

Then open:

```text
http://localhost:8080
```

If Docker is not available yet, run the two services manually:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r services/conversation/requirements.txt
pip install -r services/voice/requirements.txt
$env:PYTHONPATH = (Get-Location).Path
uvicorn services.conversation.app.main:app --reload --port 8001
uvicorn services.voice.app.main:app --reload --port 8000
```

Then open:

```text
http://localhost:8000
```

To run the smoke tests against non-default local ports, set the service URLs before starting the test runner:

```powershell
$env:TRIAGEOS_VOICE_URL = "http://localhost:8000"
$env:TRIAGEOS_CONVERSATION_URL = "http://localhost:8001"
$env:TRIAGEOS_WS_BASE = "ws://localhost:8000"
python scripts/smoke_test.py
```

Firefox uses the MediaRecorder fallback. To turn recorded audio into text, set `STT_PROVIDER=deepgram` and add `DEEPGRAM_API_KEY` to `.env`. Chromium browsers continue using browser SpeechRecognition without this setting.

Provider lookup is protected by bounded retries, per-endpoint circuit breakers, and an operation deadline. Tune these with `PROVIDER_DIRECTORY_OPERATION_TIMEOUT`, `PROVIDER_DIRECTORY_MAX_ATTEMPTS`, `PROVIDER_DIRECTORY_RETRY_BACKOFF`, `PROVIDER_DIRECTORY_CIRCUIT_FAILURE_THRESHOLD`, and `PROVIDER_DIRECTORY_CIRCUIT_OPEN_SECONDS`. The conversation service remains available when the directory is degraded and will not substitute unverified providers. Inspect dependency state at `/ready`.

Scheduling is fail-closed by default. Set `SCHEDULING_PROVIDER=mock` only for deterministic local demonstrations; production scheduling adapters will be added behind the same service boundary.

## Repository Layout

```text
frontend/
  web/
services/
  voice/
  conversation/
libs/
  ai/
  events/
  observability/
infra/
docs/
```
