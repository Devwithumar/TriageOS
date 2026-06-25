from dataclasses import dataclass


@dataclass(frozen=True)
class ModelRoute:
    task: str
    model: str
    reason: str


def route_model(task: str) -> ModelRoute:
    cheap_tasks = {"intent", "routing", "extraction"}
    if task in cheap_tasks:
        return ModelRoute(task=task, model="cheap-fast-model", reason="low-complexity task")
    return ModelRoute(task=task, model="strong-reasoning-model", reason="complex conversational task")
