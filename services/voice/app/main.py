import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from libs.events import event_names
from libs.observability.logging import configure_logging
from services.voice.app.conversation_client import ConversationClient
from services.voice.app.schemas import ClientEvent, ServerEvent

configure_logging("voice-service")
logger = logging.getLogger(__name__)

app = FastAPI(title="TriageOS Voice Service", version="0.1.0")
conversation_client = ConversationClient()
frontend_path = Path(__file__).resolve().parents[3] / "frontend" / "web"

if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def web_client() -> FileResponse:
    return FileResponse(frontend_path / "index.html")


@app.websocket("/v1/voice/sessions/{session_id}/stream")
async def voice_stream(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    await send_event(
        websocket,
        ServerEvent(
            event=event_names.VOICE_SESSION_STARTED,
            session_id=session_id,
            payload={"message": "Voice session connected."},
        ),
    )

    try:
        while True:
            raw_event = await websocket.receive_json()
            try:
                client_event = ClientEvent.model_validate(raw_event)
            except ValidationError as exc:
                await send_event(
                    websocket,
                    ServerEvent(
                        event="voice.error",
                        session_id=session_id,
                        payload={"message": "Invalid event payload.", "details": exc.errors()},
                    ),
                )
                continue

            if client_event.event == event_names.VOICE_AUDIO_CHUNK:
                await send_event(
                    websocket,
                    ServerEvent(
                        event="voice.audio.chunk.ack",
                        session_id=session_id,
                        payload={"received": True},
                    ),
                )
                continue

            if client_event.event == event_names.VOICE_INTERRUPTION_DETECTED:
                await send_event(
                    websocket,
                    ServerEvent(
                        event=event_names.VOICE_INTERRUPTION_DETECTED,
                        session_id=session_id,
                        payload={"message": "Assistant speech interrupted."},
                    ),
                )
                continue

            if client_event.event == event_names.VOICE_TRANSCRIPT_PARTIAL:
                await send_event(
                    websocket,
                    ServerEvent(
                        event=event_names.VOICE_TRANSCRIPT_PARTIAL,
                        session_id=session_id,
                        payload=client_event.payload,
                    ),
                )
                continue

            if client_event.event == event_names.VOICE_TRANSCRIPT_FINAL:
                user_text = str(client_event.payload.get("text", "")).strip()
                if not user_text:
                    continue

                turn = await conversation_client.create_turn(session_id, user_text)
                await send_event(
                    websocket,
                    ServerEvent(
                        event=event_names.VOICE_ASSISTANT_RESPONSE_CREATED,
                        session_id=session_id,
                        payload={
                            "text": turn["reply"],
                            "tts": {"provider": "browser", "voice": "default"},
                            "usage": turn["usage"],
                        },
                    ),
                )
                continue

            if client_event.event == event_names.VOICE_SESSION_ENDED:
                await send_event(
                    websocket,
                    ServerEvent(
                        event=event_names.VOICE_SESSION_ENDED,
                        session_id=session_id,
                        payload={"message": "Voice session ended."},
                    ),
                )
                await websocket.close()
                return

    except WebSocketDisconnect:
        logger.info("Voice session disconnected: %s", session_id)


async def send_event(websocket: WebSocket, event: ServerEvent) -> None:
    await websocket.send_json(event.model_dump())
