from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


ClientEventType = Literal[
    # Server-only lifecycle event — listed for schema completeness, clients must not emit.
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
    timestamp: str = Field(default_factory=utc_timestamp)
    payload: dict[str, Any] = Field(default_factory=dict)
