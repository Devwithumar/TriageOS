from libs.ai.llm import complete_conversation
from libs.ai.model_router import route_model


def generate_reply(user_text: str, recent_messages: list[dict[str, str]]) -> dict[str, object]:
    route = route_model("conversation")
    result = complete_conversation(user_text, recent_messages)

    return {
        "reply": result.text,
        "model": result.model,
        "provider": result.provider,
        "reason": result.reason or route.reason,
        "context_messages": len(recent_messages),
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
    }
