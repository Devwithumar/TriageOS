import json
import os
from typing import Any

import redis


class ConversationStateStore:
    def __init__(self) -> None:
        redis_url = os.getenv("REDIS_URL")
        self._memory: dict[str, list[dict[str, str]]] = {}
        self._structured_state: dict[str, dict[str, Any]] = {}
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

    def get_structured_state(self, session_id: str) -> dict[str, Any]:
        if self._redis:
            raw_state = self._redis.get(self._state_key(session_id))
            return json.loads(raw_state) if raw_state else {}
        return dict(self._structured_state.get(session_id, {}))

    def update_structured_state(self, session_id: str, state: dict[str, Any]) -> None:
        if self._redis:
            self._redis.set(self._state_key(session_id), json.dumps(state), ex=60 * 60 * 12)
            return
        self._structured_state[session_id] = dict(state)

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

    @staticmethod
    def _state_key(session_id: str) -> str:
        return f"conversation:{session_id}:structured-state"


def compact_state(messages: list[dict[str, str]], structured_state: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "recent_messages": messages,
        "summary": "Short-term conversation context is active.",
        "structured_state": structured_state or {},
    }
