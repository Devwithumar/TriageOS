import logging
from dataclasses import dataclass
from typing import Any

import httpx

from libs.ai.config import LLMConfig, load_llm_config, resolve_api_key_env_name

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are TriageOS, a natural voice assistant for a healthcare product. "
    "Phase 1 is voice-only conversation — no medical advice, triage, scheduling, or EHR actions. "
    "Keep replies short and spoken-friendly (one to three sentences). "
    "Be warm, clear, and conversational."
)


@dataclass(frozen=True)
class CompletionResult:
    text: str
    model: str
    provider: str
    reason: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


def complete_conversation(user_text: str, recent_messages: list[dict[str, str]]) -> CompletionResult:
    config = load_llm_config()
    messages = _build_messages(user_text, recent_messages)

    if config.provider == "stub":
        return _stub_complete(user_text, recent_messages, config)

    if not config.api_key:
        env_name = resolve_api_key_env_name(config.provider)
        raise RuntimeError(f"{env_name} is required when LLM_PROVIDER={config.provider}")

    if config.provider in {"groq", "openai"}:
        return _openai_compatible_complete(config, messages)

    if config.provider == "anthropic":
        return _anthropic_complete(config, messages)

    raise RuntimeError(f"Unsupported LLM_PROVIDER: {config.provider}")


def _build_messages(user_text: str, recent_messages: list[dict[str, str]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(recent_messages)
    messages.append({"role": "user", "content": user_text.strip()})
    return messages


def _stub_complete(
    user_text: str,
    recent_messages: list[dict[str, str]],
    config: LLMConfig,
) -> CompletionResult:
    normalized_text = user_text.strip()
    if not normalized_text:
        reply = "I did not catch that. Could you say it once more?"
    elif any(greeting in normalized_text.lower() for greeting in ("hello", "hi", "hey")):
        reply = "Hey, I am TriageOS. I can hear you clearly. What would you like to try next?"
    else:
        reply = (
            "I heard you say: "
            f"{normalized_text}. For this first milestone, I am focused on keeping the voice conversation smooth."
        )

    return CompletionResult(
        text=reply,
        model=config.model,
        provider=config.provider,
        reason="stub provider for local development",
        prompt_tokens=None,
        completion_tokens=None,
    )


def _openai_compatible_complete(config: LLMConfig, messages: list[dict[str, str]]) -> CompletionResult:
    url = f"{config.base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": 0.6,
        "max_tokens": 256,
    }

    with httpx.Client(timeout=30.0) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        body = response.json()

    choice = body["choices"][0]["message"]["content"].strip()
    usage = body.get("usage", {})
    logger.info(
        "llm_complete provider=%s model=%s prompt_tokens=%s completion_tokens=%s",
        config.provider,
        config.model,
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
    )
    return CompletionResult(
        text=choice,
        model=config.model,
        provider=config.provider,
        reason=f"{config.provider} chat completion",
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
    )


def _anthropic_complete(config: LLMConfig, messages: list[dict[str, str]]) -> CompletionResult:
    system_text, anthropic_messages = _split_anthropic_messages(messages)
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": config.api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": config.model,
        "max_tokens": 256,
        "system": system_text,
        "messages": anthropic_messages,
    }

    with httpx.Client(timeout=30.0) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        body = response.json()

    text_blocks = [block["text"] for block in body.get("content", []) if block.get("type") == "text"]
    usage = body.get("usage", {})
    logger.info(
        "llm_complete provider=%s model=%s prompt_tokens=%s completion_tokens=%s",
        config.provider,
        config.model,
        usage.get("input_tokens"),
        usage.get("output_tokens"),
    )
    return CompletionResult(
        text=" ".join(text_blocks).strip(),
        model=config.model,
        provider=config.provider,
        reason="anthropic messages api",
        prompt_tokens=usage.get("input_tokens"),
        completion_tokens=usage.get("output_tokens"),
    )


def _split_anthropic_messages(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    system_parts: list[str] = []
    conversation: list[dict[str, str]] = []
    for message in messages:
        if message["role"] == "system":
            system_parts.append(message["content"])
            continue
        conversation.append({"role": message["role"], "content": message["content"]})
    return "\n\n".join(system_parts), conversation
