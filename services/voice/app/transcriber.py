import base64
import os
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    provider: str


async def transcribe_audio(audio_base64: str, mime_type: str) -> TranscriptionResult | None:
    provider = os.getenv("STT_PROVIDER", "stub").strip().lower()
    if provider == "stub":
        return None

    if provider != "deepgram":
        raise RuntimeError(f"Unsupported STT_PROVIDER: {provider}")

    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPGRAM_API_KEY is required when STT_PROVIDER=deepgram")

    audio_bytes = base64.b64decode(audio_base64)
    params = {
        "model": os.getenv("DEEPGRAM_MODEL", "nova-3"),
        "smart_format": "true",
        "language": "en-US",
    }
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": mime_type or "application/octet-stream",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.deepgram.com/v1/listen",
            params=params,
            headers=headers,
            content=audio_bytes,
        )
        response.raise_for_status()
        body = response.json()

    channels = body.get("results", {}).get("channels", [{}])
    alternatives = channels[0].get("alternatives", [{}]) if channels else [{}]
    text = alternatives[0].get("transcript", "").strip() if alternatives else ""
    return TranscriptionResult(text=text, provider="deepgram")
