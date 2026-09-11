import logging
from dataclasses import dataclass
from typing import Any

import httpx

from libs.ai.config import LLMConfig, load_llm_config, resolve_api_key_env_name

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are TriageOS, a calm and capable voice assistant for a healthcare product. "
    "This is Phase 1, so have natural general conversations only. Do not provide medical advice, "
    "diagnoses, triage, scheduling, or EHR actions yet. Keep replies spoken-friendly: one or two "
    "short sentences, no markdown, no lists, and no unnecessary repetition. Acknowledge what the "
    "person said, answer when you can, and ask one useful follow-up question when appropriate. "
    "If a person asks for healthcare help, explain that those capabilities are coming later and "
    "offer to continue with a general conversation. Exception: if they mention a potentially "
    "life-threatening symptom such as chest pain or difficulty breathing, do not deflect. Tell "
    "them to contact local emergency services immediately or have someone take them to the nearest "
    "emergency department, and advise them not to drive themselves. Do not diagnose or reassure them."
)


@dataclass(frozen=True)
class CompletionResult:
    text: str
    model: str
    provider: str
    reason: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


def complete_conversation(
    user_text: str,
    recent_messages: list[dict[str, str]],
    conversation_context: dict[str, object] | None = None,
) -> CompletionResult:
    config = load_llm_config()
    messages = _build_messages(user_text, recent_messages, conversation_context or {})

    if config.provider == "stub":
        return _stub_complete(user_text, recent_messages, config, conversation_context or {})

    if not config.api_key:
        env_name = resolve_api_key_env_name(config.provider)
        raise RuntimeError(f"{env_name} is required when LLM_PROVIDER={config.provider}")

    if config.provider in {"groq", "openai"}:
        return _openai_compatible_complete(config, messages)

    if config.provider == "anthropic":
        return _anthropic_complete(config, messages)

    raise RuntimeError(f"Unsupported LLM_PROVIDER: {config.provider}")


def _build_messages(
    user_text: str,
    recent_messages: list[dict[str, str]],
    conversation_context: dict[str, object],
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({
        "role": "system",
        "content": f"Conversation context (use quietly, do not mention it): {conversation_context}",
    })
    messages.extend(recent_messages)
    messages.append({"role": "user", "content": user_text.strip()})
    return messages


def _stub_complete(
    user_text: str,
    recent_messages: list[dict[str, str]],
    config: LLMConfig,
    conversation_context: dict[str, object],
) -> CompletionResult:
    normalized_text = user_text.strip()
    lowered_text = normalized_text.lower()
    intent = str(conversation_context.get("current_intent", "general_conversation"))
    if not normalized_text:
        reply = "I didn’t catch that. Could you say it once more?"
    elif intent == "greeting":
        reply = "Hey, I’m TriageOS. I’m ready to chat. What would you like to talk about?"
    elif intent == "capabilities":
        reply = "I’m TriageOS, a voice assistant for healthcare teams. Right now I’m focused on having a smooth, natural conversation with you."
    elif intent == "gratitude":
        reply = "You’re welcome. What would you like to explore next?"
    elif intent == "goodbye":
        reply = "It was good talking with you. I’ll be here whenever you’re ready to continue."
    elif intent == "urgent_safety":
        reply = "Chest pain can be serious. If this is happening now, call your local emergency services immediately or have someone take you to the nearest emergency department. Please do not drive yourself."
    elif intent == "healthcare_request":
        reply = "That healthcare workflow is planned for a later TriageOS phase. For now, I can still keep you company or answer general questions."
    elif intent == "question":
        reply = "That’s a good question. I’m still in my conversation-first phase, but I can think it through with you. What matters most about it?"
    elif intent == "sharing" and recent_messages:
        reply = "I’m with you. What part of that feels most important right now?"
    elif intent == "sharing":
        reply = "Thanks for sharing that. What would you like to explore about it?"
    elif recent_messages:
        reply = "I’m following along. Would you like to tell me a little more, or switch to another topic?"
    else:
        reply = "Thanks for sharing that. I’m following along. Would you like to tell me more about it?"

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
