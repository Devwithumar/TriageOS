import os

import httpx


class ConversationClient:
    def __init__(self) -> None:
        self._base_url = os.getenv("CONVERSATION_SERVICE_URL", "http://localhost:8001")

    async def create_turn(self, session_id: str, text: str) -> dict:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(
                f"{self._base_url}/v1/conversations/{session_id}/turn",
                json={"session_id": session_id, "text": text},
            )
            response.raise_for_status()
            return response.json()
