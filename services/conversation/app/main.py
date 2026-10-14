import os
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from libs.observability.logging import configure_logging
from libs.observability.metrics import MetricsRegistry
from libs.ai.config import load_llm_config
from services.conversation.app.agent import generate_reply
from services.conversation.app.canonical_engine import CanonicalConversationEngine
from services.conversation.app.state import ConversationStateStore, compact_state

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

configure_logging("conversation-service")

app = FastAPI(title="TriageOS Conversation Service", version="0.1.0")
metrics = MetricsRegistry()
state_store = ConversationStateStore()
canonical_engine = CanonicalConversationEngine()
conversation_engine = os.getenv("CONVERSATION_ENGINE", "canonical").strip().lower()


class ConversationTurnRequest(BaseModel):
    session_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class ConversationTurnResponse(BaseModel):
    session_id: str
    reply: str
    state: dict
    usage: dict


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics() -> PlainTextResponse:
    return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")


@app.get("/ready")
def ready() -> JSONResponse:
    llm = load_llm_config()
    llm_ready = llm.provider == "stub" or bool(llm.api_key)
    llm_check = {
        "status": "ready" if llm_ready else "not_ready",
        "provider": llm.provider,
        "model": llm.model,
    }

    provider_directory = canonical_engine.provider_directory.health()
    directory_configured = provider_directory.get("provider") == "osm"
    directory_check = {
        **provider_directory,
        "status": (
            provider_directory.get("status", "not_ready")
            if directory_configured
            else "not_ready"
        ),
    }
    state_check = state_store.readiness()
    state_ready = state_check["status"] in {"ready", "disabled"}
    persistence_check = canonical_engine.readiness()
    persistence_ready = persistence_check["status"] in {"ready", "disabled"}
    overall_ready = llm_ready and directory_configured and state_ready and persistence_ready
    payload = {
        "status": "ready" if overall_ready else "not_ready",
        "engine": conversation_engine,
        "dependencies": {
            "llm": llm_check,
            "provider_directory": directory_check,
            "state_store": state_check,
            "persistence": persistence_check,
        },
    }
    return JSONResponse(status_code=200 if overall_ready else 503, content=payload)


@app.post("/v1/conversations/{session_id}/turn", response_model=ConversationTurnResponse)
def create_turn(session_id: str, request: ConversationTurnRequest) -> ConversationTurnResponse:
    started_at = time.perf_counter()
    outcome = "error"
    provider = "unknown"
    try:
        if conversation_engine == "canonical":
            result = canonical_engine.handle_turn(session_id, request.text)
            provider = str(result.usage.get("provider", "unknown"))
            outcome = "success" if not result.usage.get("error") else "fallback"
            return ConversationTurnResponse(
                session_id=session_id,
                reply=result.reply,
                state=result.state,
                usage=result.usage,
            )
        recent_messages = state_store.get_recent_messages(session_id)
        structured_state = state_store.get_structured_state(session_id)
        agent_result = generate_reply(request.text, recent_messages, structured_state, session_id)
        provider = str(agent_result.get("provider", "stub"))
        outcome = "success"
        reply = str(agent_result["reply"])
        state_store.append_turn(session_id, request.text, reply)
        state_store.update_structured_state(session_id, agent_result.get("structured_state", structured_state))

        updated_messages = state_store.get_recent_messages(session_id)
        return ConversationTurnResponse(
            session_id=session_id,
            reply=reply,
            state=compact_state(updated_messages, agent_result.get("structured_state", structured_state)),
            usage={
                "provider": agent_result.get("provider", "stub"),
                "model": agent_result["model"],
                "route_reason": agent_result["reason"],
                "estimated_input_messages": agent_result["context_messages"],
                "prompt_tokens": agent_result.get("prompt_tokens"),
                "completion_tokens": agent_result.get("completion_tokens"),
                "intent": agent_result.get("intent", {}),
                "tool_call": agent_result.get("context", {}).get("tool_call"),
                "tool_result": agent_result.get("context", {}).get("tool_result"),
            },
        )
    finally:
        metrics.increment(
            "triageos_conversation_turns_total",
            {"outcome": outcome, "provider": provider},
        )
        metrics.observe(
            "triageos_conversation_turn_duration_seconds",
            time.perf_counter() - started_at,
            {"outcome": outcome},
        )
