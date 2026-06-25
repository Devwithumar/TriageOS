import json
import os
from typing import Any

import redis


class ConversationStateStore:
    def __init__(self) -> None:
        redis_url = os.getenv("REDIS_URL")
        self._memory: dict[str, list[dict[str, str]]] = {}
        self._redis = redis.from_url(redis_url, decode_responses=True) if redis_url else None

    def get_recent_messages(self, session_id: str, limit: int = 8) -> list[dict[str, str]]:
        messages = self._read_messages(session_id)
        return messages[-limit:]

    def append_turn(self, session_id: str, user_text: str, assistant_text: str) -> None:
        messages = self._read_messages(session_id)
        messages.extend(
            [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ]
        )
        self._write_messages(session_id, messages[-20:])

    def _read_messages(self, session_id: str) -> list[dict[str, str]]:
        if self._redis:
            raw_messages = self._redis.get(self._key(session_id))
            return json.loads(raw_messages) if raw_messages else []
        return self._memory.get(session_id, [])

    def _write_messages(self, session_id: str, messages: list[dict[str, str]]) -> None:
        if self._redis:
            self._redis.set(self._key(session_id), json.dumps(messages), ex=60 * 60 * 12)
            return
        self._memory[session_id] = messages

    @staticmethod
    def _key(session_id: str) -> str:
        return f"conversation:{session_id}:messages"


def compact_state(messages: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "recent_messages": messages,
        "summary": "Conversation summary placeholder for Phase 1.",
        "structured_state": {},
    }
