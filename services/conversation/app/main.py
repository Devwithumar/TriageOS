import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from fastapi import FastAPI

from libs.observability.logging import configure_logging
from services.conversation.app.agent import generate_reply
from services.conversation.app.canonical_engine import CanonicalConversationEngine
from services.conversation.app.state import ConversationStateStore, compact_state

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

configure_logging("conversation-service")

app = FastAPI(title="TriageOS Conversation Service", version="0.1.0")
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


@app.get("/ready")
def ready() -> dict[str, object]:
    return {
        "status": "ready",
        "dependencies": {
            "provider_directory": canonical_engine.provider_directory.health(),
        },
    }


@app.post("/v1/conversations/{session_id}/turn", response_model=ConversationTurnResponse)
def create_turn(session_id: str, request: ConversationTurnRequest) -> ConversationTurnResponse:
    if conversation_engine == "canonical":
        result = canonical_engine.handle_turn(session_id, request.text)
        return ConversationTurnResponse(
            session_id=session_id,
            reply=result.reply,
            state=result.state,
            usage=result.usage,
        )
    recent_messages = state_store.get_recent_messages(session_id)
    structured_state = state_store.get_structured_state(session_id)
    agent_result = generate_reply(request.text, recent_messages, structured_state, session_id)
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
