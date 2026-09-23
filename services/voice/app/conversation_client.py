import asyncio
import os

import httpx


class ConversationClient:
    def __init__(self) -> None:
        self._base_url = os.getenv("CONVERSATION_SERVICE_URL", "http://127.0.0.1:8001")

    async def create_turn(self, session_id: str, text: str) -> dict:
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=45.0) as client:
                    response = await client.post(
                        f"{self._base_url}/v1/conversations/{session_id}/turn",
                        json={"session_id": session_id, "text": text},
                    )
                    response.raise_for_status()
                    return response.json()
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
                if attempt == 2:
                    raise RuntimeError("Conversation service is temporarily unavailable. Please try again.") from exc
                await asyncio.sleep(0.5 * (2**attempt))

        raise RuntimeError("Conversation service is temporarily unavailable. Please try again.")
