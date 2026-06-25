from pydantic import BaseModel, Field
from fastapi import FastAPI

from libs.observability.logging import configure_logging
from services.conversation.app.agent import generate_reply
from services.conversation.app.state import ConversationStateStore, compact_state

configure_logging("conversation-service")

app = FastAPI(title="TriageOS Conversation Service", version="0.1.0")
state_store = ConversationStateStore()


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


@app.post("/v1/conversations/{session_id}/turn", response_model=ConversationTurnResponse)
def create_turn(session_id: str, request: ConversationTurnRequest) -> ConversationTurnResponse:
    recent_messages = state_store.get_recent_messages(session_id)
    agent_result = generate_reply(request.text, recent_messages)
    reply = str(agent_result["reply"])
    state_store.append_turn(session_id, request.text, reply)

    updated_messages = state_store.get_recent_messages(session_id)
    return ConversationTurnResponse(
        session_id=session_id,
        reply=reply,
        state=compact_state(updated_messages),
        usage={
            "model": agent_result["model"],
            "route_reason": agent_result["reason"],
            "estimated_input_messages": agent_result["context_messages"],
        },
    )
