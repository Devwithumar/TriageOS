from libs.ai.model_router import route_model


def generate_reply(user_text: str, recent_messages: list[dict[str, str]]) -> dict[str, object]:
    route = route_model("conversation")
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

    return {
        "reply": reply,
        "model": route.model,
        "reason": route.reason,
        "context_messages": len(recent_messages),
    }
