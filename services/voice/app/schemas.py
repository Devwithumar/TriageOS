from typing import Any, Literal

from pydantic import BaseModel, Field


ClientEventType = Literal[
    "voice.session.started",
    "voice.audio.chunk",
    "voice.user.transcript.partial",
    "voice.user.transcript.final",
    "voice.interruption.detected",
    "voice.session.ended",
]


class ClientEvent(BaseModel):
    event: ClientEventType
    session_id: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)


class ServerEvent(BaseModel):
    event: str
    session_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
