import asyncio
import logging
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from libs.events import event_names
from libs.observability.logging import configure_logging
from services.voice.app.conversation_client import ConversationClient
from services.voice.app.schemas import ClientEvent, ServerEvent
from services.voice.app.session import VoiceSessionController

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


async def process_turn(
    websocket: WebSocket,
    session: VoiceSessionController,
    turn_id: int,
    user_text: str,
    stt_ms: int | None,
) -> None:
    turn_started_at = time.perf_counter()
    try:
        llm_started_at = time.perf_counter()
        turn = await conversation_client.create_turn(session.session_id, user_text)
        llm_ms = round((time.perf_counter() - llm_started_at) * 1000)

        if not session.is_turn_active(turn_id):
            logger.info(
                "turn_stale_dropped session=%s turn_id=%s llm_ms=%s",
                session.session_id,
                turn_id,
                llm_ms,
            )
            return

        total_ms = round((time.perf_counter() - turn_started_at) * 1000)
        latency = {
            "stt_ms": stt_ms,
            "llm_ms": llm_ms,
            "server_ms": total_ms,
        }
        logger.info(
            "turn_latency session=%s turn_id=%s stt_ms=%s llm_ms=%s server_ms=%s",
            session.session_id,
            turn_id,
            stt_ms,
            llm_ms,
            total_ms,
        )

        session.begin_speaking()
        await send_event(
            websocket,
            ServerEvent(
                event=event_names.VOICE_ASSISTANT_RESPONSE_CREATED,
                session_id=session.session_id,
                payload={
                    "turn_id": turn_id,
                    "text": turn["reply"],
                    "tts": {"provider": "browser", "voice": "default"},
                    "usage": turn["usage"],
                    "latency": latency,
                },
            ),
        )
    except asyncio.CancelledError:
        logger.info(
            "turn_task_cancelled session=%s turn_id=%s",
            session.session_id,
            turn_id,
        )
        raise
    except Exception:
        if session.is_turn_active(turn_id):
            session.return_to_listening()
        raise


@app.websocket("/v1/voice/sessions/{session_id}/stream")
async def voice_stream(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    session = VoiceSessionController(session_id)
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
                turn_seq = session.interrupt()
                await send_event(
                    websocket,
                    ServerEvent(
                        event=event_names.VOICE_INTERRUPTION_DETECTED,
                        session_id=session_id,
                        payload={
                            "message": "Assistant speech interrupted.",
                            "turn_seq": turn_seq,
                        },
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

                stt_raw = client_event.payload.get("stt_ms")
                stt_ms = int(stt_raw) if stt_raw is not None else None
                turn_id = session.begin_thinking()
                task = asyncio.create_task(
                    process_turn(websocket, session, turn_id, user_text, stt_ms)
                )
                session.set_active_task(task)
                continue

            if client_event.event == event_names.VOICE_SESSION_ENDED:
                session.interrupt()
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
        session.interrupt()
        logger.info("Voice session disconnected: %s", session_id)


async def send_event(websocket: WebSocket, event: ServerEvent) -> None:
    await websocket.send_json(event.model_dump())
