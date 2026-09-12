"""Phase 1 smoke tests — run with voice (8000) and conversation (8001) services up."""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from dataclasses import dataclass, field

import httpx
import websockets

from libs.conversation.contracts import (
    ConversationState,
    ExtractedSlot,
    IntentClassification,
    IntentName,
    ReceptionistState,
)
from libs.conversation.workflow import apply_receptionist_turn

VOICE_URL = "http://localhost:8000"
CONVERSATION_URL = "http://localhost:8001"
WS_BASE = "ws://localhost:8000"


@dataclass
class Results:
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def ok(self, name: str) -> None:
        self.passed.append(name)
        print(f"  PASS  {name}")

    def fail(self, name: str, detail: str) -> None:
        self.failed.append(f"{name}: {detail}")
        print(f"  FAIL  {name} — {detail}")

    def warn(self, name: str, detail: str) -> None:
        self.warnings.append(f"{name}: {detail}")
        print(f"  WARN  {name} — {detail}")


def test_workflow_contracts(results: Results) -> None:
    try:
        state = ConversationState(session_id="workflow_smoke")
        intent = IntentClassification(name=IntentName.APPOINTMENT_REQUEST, confidence=1)
        entities = {
            name: ExtractedSlot(value=value, confidence=1)
            for name, value in {
                "caller_name": "Alex Johnson",
                "callback_number": "08098765432",
                "preferred_time": "Tuesday afternoon",
                "appointment_reason": "general consultation",
            }.items()
        }
        review = apply_receptionist_turn(state, intent, entities)
        confirmed = apply_receptionist_turn(
            review.state,
            IntentClassification(name=IntentName.CONFIRMATION, confidence=1),
        )
        if (
            review.state.workflow.state == ReceptionistState.REVIEWING_REQUEST
            and confirmed.tool_call
            and confirmed.tool_call.name == "create_appointment_request"
        ):
            results.ok("workflow contract transitions")
        else:
            results.fail("workflow contract transitions", "unexpected workflow decision")
    except Exception as exc:
        results.fail("workflow contract transitions", str(exc))


async def recv_json(ws: websockets.ClientConnection, timeout: float = 5.0) -> dict:
    raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
    return json.loads(raw)


async def test_health(results: Results) -> None:
    async with httpx.AsyncClient() as client:
        for name, url in [("voice health", f"{VOICE_URL}/health"), ("conversation health", f"{CONVERSATION_URL}/health")]:
            try:
                response = await client.get(url, timeout=5)
                if response.status_code == 200 and response.json().get("status") == "ok":
                    results.ok(name)
                else:
                    results.fail(name, f"unexpected response: {response.status_code} {response.text}")
            except Exception as exc:
                results.fail(name, str(exc))


async def test_web_client(results: Results) -> None:
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{VOICE_URL}/", timeout=5)
            if response.status_code == 200 and "TriageOS" in response.text:
                results.ok("web client index")
            else:
                results.fail("web client index", f"status={response.status_code}")
        except Exception as exc:
            results.fail("web client index", str(exc))


async def test_conversation_turn(results: Results) -> None:
    session_id = f"smoke_{uuid.uuid4()}"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{CONVERSATION_URL}/v1/conversations/{session_id}/turn",
                json={"session_id": session_id, "text": "hello there"},
                timeout=10,
            )
            response.raise_for_status()
            body = response.json()
            if body.get("reply") and body.get("session_id") == session_id:
                results.ok("conversation turn (greeting)")
            else:
                results.fail("conversation turn (greeting)", f"unexpected body: {body}")

            response2 = await client.post(
                f"{CONVERSATION_URL}/v1/conversations/{session_id}/turn",
                json={"session_id": session_id, "text": "tell me about scheduling"},
                timeout=10,
            )
            response2.raise_for_status()
            state = response2.json().get("state", {})
            messages = state.get("recent_messages", [])
            if len(messages) >= 4:
                results.ok("conversation state persistence")
            else:
                results.fail("conversation state persistence", f"expected >=4 messages, got {len(messages)}")
        except Exception as exc:
            results.fail("conversation turn", str(exc))


async def test_urgent_safety_response(results: Results) -> None:
    session_id = f"safety_{uuid.uuid4()}"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{CONVERSATION_URL}/v1/conversations/{session_id}/turn",
                json={"session_id": session_id, "text": "I am having chest pains"},
                timeout=10,
            )
            response.raise_for_status()
            body = response.json()
            reply = body.get("reply", "").lower()
            usage = body.get("usage", {})
            intent = usage.get("intent", {}).get("name")
            if (
                intent == "urgent_safety"
                and usage.get("provider") == "guardrail"
                and "emergency" in reply
                and "do not drive" in reply
            ):
                results.ok("urgent safety escalation")
            else:
                results.fail("urgent safety escalation", f"unexpected response: {body}")
        except Exception as exc:
            results.fail("urgent safety escalation", str(exc))


async def test_receptionist_appointment_flow(results: Results) -> None:
    session_id = f"receptionist_{uuid.uuid4()}"
    turns = [
        "I need to book an appointment",
        "My name is Jane Doe",
        "08012345678 tomorrow morning",
        "general consultation",
        "yes",
    ]
    async with httpx.AsyncClient() as client:
        try:
            body = None
            for text in turns:
                response = await client.post(
                    f"{CONVERSATION_URL}/v1/conversations/{session_id}/turn",
                    json={"session_id": session_id, "text": text},
                    timeout=10,
                )
                response.raise_for_status()
                body = response.json()

            state = body.get("state", {}).get("structured_state", {})
            usage = body.get("usage", {})
            if (
                usage.get("provider") == "workflow"
                and usage.get("tool_call", {}).get("name") == "create_appointment_request"
                and state.get("appointment_status") == "confirmed"
                and state.get("caller_name") == "Jane Doe"
                and state.get("callback_number") == "08012345678"
            ):
                results.ok("receptionist appointment request")
            else:
                results.fail("receptionist appointment request", f"unexpected response: {body}")
        except Exception as exc:
            results.fail("receptionist appointment request", str(exc))


async def test_websocket_session_lifecycle(results: Results) -> None:
    session_id = f"smoke_{uuid.uuid4()}"
    uri = f"{WS_BASE}/v1/voice/sessions/{session_id}/stream"
    try:
        async with websockets.connect(uri) as ws:
            started = await recv_json(ws)
            if started.get("event") != "voice.session.started":
                results.fail("websocket session.started", f"got {started}")
                return
            if not started.get("timestamp"):
                results.fail("server event timestamp", "voice.session.started missing timestamp")
                return
            results.ok("websocket session.started")
            results.ok("server event timestamp")

            await ws.send(
                json.dumps(
                    {
                        "event": "voice.user.transcript.final",
                        "session_id": session_id,
                        "payload": {"text": "hello", "stt_ms": 420},
                    }
                )
            )

            response = await recv_json(ws)
            if response.get("event") != "voice.assistant.response.created":
                results.fail("websocket assistant response", f"got {response}")
                return

            payload = response.get("payload", {})
            latency = payload.get("latency", {})
            if (
                payload.get("text")
                and payload.get("turn_id", 0) > 0
                and payload.get("tts", {}).get("provider") == "browser"
                and latency.get("llm_ms") is not None
                and latency.get("stt_ms") == 420
            ):
                results.ok("websocket assistant response")
                results.ok("turn latency payload")
            else:
                results.fail("websocket assistant response", f"unexpected payload: {payload}")

            await ws.send(
                json.dumps(
                    {
                        "event": "voice.session.ended",
                        "session_id": session_id,
                        "payload": {},
                    }
                )
            )
            ended = await recv_json(ws)
            if ended.get("event") == "voice.session.ended":
                results.ok("websocket session.ended")
            else:
                results.fail("websocket session.ended", f"got {ended}")
    except Exception as exc:
        results.fail("websocket session lifecycle", str(exc))


async def test_websocket_partial_transcript(results: Results) -> None:
    session_id = f"smoke_{uuid.uuid4()}"
    uri = f"{WS_BASE}/v1/voice/sessions/{session_id}/stream"
    try:
        async with websockets.connect(uri) as ws:
            await recv_json(ws)  # session.started

            await ws.send(
                json.dumps(
                    {
                        "event": "voice.user.transcript.partial",
                        "session_id": session_id,
                        "payload": {"text": "hel"},
                    }
                )
            )
            partial = await recv_json(ws)
            if partial.get("event") == "voice.user.transcript.partial":
                results.ok("websocket partial transcript echo")
            else:
                results.fail("websocket partial transcript echo", f"got {partial}")
    except Exception as exc:
        results.fail("websocket partial transcript", str(exc))


async def test_websocket_audio_chunk_ack(results: Results) -> None:
    session_id = f"smoke_{uuid.uuid4()}"
    uri = f"{WS_BASE}/v1/voice/sessions/{session_id}/stream"
    try:
        async with websockets.connect(uri) as ws:
            await recv_json(ws)

            await ws.send(
                json.dumps(
                    {
                        "event": "voice.audio.chunk",
                        "session_id": session_id,
                        "payload": {"seq": 1, "bytes": 0},
                    }
                )
            )
            ack = await recv_json(ws)
            if ack.get("event") == "voice.audio.chunk.ack":
                results.ok("websocket audio chunk ack")
            else:
                results.fail("websocket audio chunk ack", f"got {ack}")
    except Exception as exc:
        results.fail("websocket audio chunk ack", str(exc))


async def test_interruption_layer(results: Results) -> None:
    """Interruption is foundation-critical — verify cancel propagates and turns stay consistent."""
    session_id = f"smoke_{uuid.uuid4()}"
    uri = f"{WS_BASE}/v1/voice/sessions/{session_id}/stream"
    try:
        async with websockets.connect(uri) as ws:
            await recv_json(ws)

            await ws.send(
                json.dumps(
                    {
                        "event": "voice.interruption.detected",
                        "session_id": session_id,
                        "payload": {},
                    }
                )
            )
            interruption = await recv_json(ws)
            if interruption.get("event") != "voice.interruption.detected":
                results.fail("interruption event echo", f"got {interruption}")
                return
            if interruption.get("payload", {}).get("turn_seq", 0) <= 0:
                results.fail("interruption turn_seq", f"missing turn_seq in {interruption}")
                return
            results.ok("interruption event echo (server acknowledges)")
            results.ok("interruption turn_seq in payload")

            await ws.send(
                json.dumps(
                    {
                        "event": "voice.user.transcript.final",
                        "session_id": session_id,
                        "payload": {"text": "wait stop, I meant something else", "stt_ms": 300},
                    }
                )
            )
            response = await recv_json(ws)
            if response.get("event") == "voice.assistant.response.created":
                results.ok("turn after interruption (no server block)")
            else:
                results.fail("turn after interruption", f"got {response}")

            results.warn(
                "interruption — automatic barge-in",
                "no always-on mic / VAD; user must press Start Talking to interrupt",
            )
    except Exception as exc:
        results.fail("interruption layer", str(exc))


async def test_stale_turn_superseded(results: Results) -> None:
    """A rapid second final must invalidate the first in-flight turn."""
    session_id = f"smoke_{uuid.uuid4()}"
    uri = f"{WS_BASE}/v1/voice/sessions/{session_id}/stream"
    try:
        async with websockets.connect(uri) as ws:
            await recv_json(ws)

            await ws.send(
                json.dumps(
                    {
                        "event": "voice.user.transcript.final",
                        "session_id": session_id,
                        "payload": {"text": "first utterance"},
                    }
                )
            )
            await ws.send(
                json.dumps(
                    {
                        "event": "voice.user.transcript.final",
                        "session_id": session_id,
                        "payload": {"text": "second utterance"},
                    }
                )
            )

            responses: list[dict] = []
            while len(responses) < 2:
                try:
                    message = await recv_json(ws, timeout=3.0)
                except TimeoutError:
                    break
                if message.get("event") == "voice.assistant.response.created":
                    responses.append(message)

            if len(responses) != 1:
                results.fail("stale turn superseded", f"expected 1 response, got {len(responses)}")
                return

            results.ok("stale turn superseded")
    except Exception as exc:
        results.fail("stale turn superseded", str(exc))


async def test_invalid_event(results: Results) -> None:
    session_id = f"smoke_{uuid.uuid4()}"
    uri = f"{WS_BASE}/v1/voice/sessions/{session_id}/stream"
    try:
        async with websockets.connect(uri) as ws:
            await recv_json(ws)
            await ws.send(json.dumps({"event": "voice.session.started", "session_id": session_id, "payload": {}}))
            # voice.session.started is valid ClientEventType but not handled — should be silently ignored
            # Send malformed payload instead
            await ws.send(json.dumps({"event": "not.a.real.event", "session_id": session_id, "payload": {}}))
            error = await recv_json(ws)
            if error.get("event") == "voice.error":
                results.ok("invalid event returns voice.error")
            else:
                results.fail("invalid event handling", f"got {error}")
    except Exception as exc:
        results.fail("invalid event handling", str(exc))


async def main() -> int:
    results = Results()
    print("\nTriageOS Voice and Receptionist Smoke Tests\n" + "=" * 44)

    test_workflow_contracts(results)
    await test_health(results)
    await test_web_client(results)
    await test_conversation_turn(results)
    await test_urgent_safety_response(results)
    await test_receptionist_appointment_flow(results)
    await test_websocket_session_lifecycle(results)
    await test_websocket_partial_transcript(results)
    await test_websocket_audio_chunk_ack(results)
    await test_interruption_layer(results)
    await test_stale_turn_superseded(results)
    await test_invalid_event(results)

    print("\n" + "=" * 32)
    print(f"Passed: {len(results.passed)}  Failed: {len(results.failed)}  Warnings: {len(results.warnings)}")

    if results.failed:
        print("\nFailures:")
        for item in results.failed:
            print(f"  - {item}")

    if results.warnings:
        print("\nInterruption / foundation gaps:")
        for item in results.warnings:
            print(f"  - {item}")

    return 1 if results.failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
