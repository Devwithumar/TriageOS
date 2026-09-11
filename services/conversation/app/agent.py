from libs.ai.conversation_intelligence import build_context, detect_intent
from libs.ai.llm import CompletionResult, complete_conversation
from libs.ai.model_router import route_model

SAFETY_REPLY = (
    "Chest pain can be serious. If this is happening now, call your local emergency "
    "services immediately or have someone take you to the nearest emergency department. "
    "Please do not drive yourself."
)


def generate_reply(user_text: str, recent_messages: list[dict[str, str]]) -> dict[str, object]:
    route = route_model("conversation")
    intent = detect_intent(user_text)
    context = build_context(recent_messages, intent)
    if intent.name == "urgent_safety":
        result = CompletionResult(
            text=SAFETY_REPLY,
            model="deterministic-safety",
            provider="guardrail",
            reason="urgent safety escalation guardrail",
        )
    else:
        result = complete_conversation(user_text, recent_messages, context)

    return {
        "reply": result.text,
        "model": result.model,
        "provider": result.provider,
        "reason": result.reason or route.reason,
        "context_messages": len(recent_messages),
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "intent": intent.as_dict(),
        "context": context,
    }
