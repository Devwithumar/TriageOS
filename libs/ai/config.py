import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    api_key: str | None
    model: str
    base_url: str | None = None


def load_llm_config() -> LLMConfig:
    provider = os.getenv("LLM_PROVIDER", "stub").strip().lower()

    if provider == "groq":
        return LLMConfig(
            provider="groq",
            api_key=os.getenv("GROQ_API_KEY"),
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            base_url="https://api.groq.com/openai/v1",
        )

    if provider == "openai":
        return LLMConfig(
            provider="openai",
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )

    if provider == "anthropic":
        return LLMConfig(
            provider="anthropic",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"),
        )

    return LLMConfig(provider="stub", api_key=None, model="stub")


def resolve_api_key_env_name(provider: str) -> str:
    return {
        "groq": "GROQ_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }.get(provider, "LLM_API_KEY")
