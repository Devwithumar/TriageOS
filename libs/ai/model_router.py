import os

from dataclasses import dataclass

from libs.ai.config import load_llm_config


@dataclass(frozen=True)
class ModelRoute:
    task: str
    model: str
    provider: str
    reason: str


def route_model(task: str) -> ModelRoute:
    config = load_llm_config()
    cheap_tasks = {"intent", "routing", "extraction"}

    if config.provider != "stub":
        if task in cheap_tasks:
            fast_model = os.getenv("LLM_FAST_MODEL", config.model)
            return ModelRoute(
                task=task,
                model=fast_model,
                provider=config.provider,
                reason="low-complexity task",
            )
        return ModelRoute(
            task=task,
            model=config.model,
            provider=config.provider,
            reason="configured provider model",
        )

    if task in cheap_tasks:
        return ModelRoute(
            task=task,
            model="cheap-fast-model",
            provider="stub",
            reason="low-complexity task",
        )
    return ModelRoute(
        task=task,
        model="strong-reasoning-model",
        provider="stub",
        reason="complex conversational task",
    )
