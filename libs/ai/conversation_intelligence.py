from dataclasses import asdict, dataclass
import re


@dataclass(frozen=True)
class ConversationIntent:
    name: str
    confidence: float
    topic: str | None
    expects_question: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


HEALTHCARE_TERMS = (
    "appointment",
    "symptom",
    "diagnosis",
    "medical advice",
    "triage",
    "doctor",
    "patient",
    "prescription",
    "pain",
)

RECEPTIONIST_TERMS = (
    "appointment",
    "book a visit",
    "schedule",
    "reschedule",
    "cancel my appointment",
    "see a doctor",
    "available slot",
    "availability",
)

PRACTICE_INFO_TERMS = (
    "opening hours",
    "office hours",
    "where are you",
    "location",
    "address",
    "accept insurance",
    "insurance",
    "phone number",
)

URGENT_TERMS = (
    "chest pain",
    "chest pressure",
    "can't breathe",
    "cannot breathe",
    "difficulty breathing",
    "trouble breathing",
    "shortness of breath",
    "stroke",
    "face drooping",
    "weakness on one side",
    "severe bleeding",
    "unconscious",
    "passed out",
    "suicidal",
    "kill myself",
)


def detect_intent(text: str) -> ConversationIntent:
    normalized = " ".join(text.lower().split())
    topic = extract_topic(text)

    if not normalized:
        return ConversationIntent("unclear", 1.0, None, True)
    if any(term in normalized for term in URGENT_TERMS):
        return ConversationIntent("urgent_safety", 0.99, topic, False)
    if any(phrase in normalized for phrase in ("cancel my appointment", "cancel the appointment", "i want to cancel")):
        return ConversationIntent("appointment_cancellation", 0.97, topic, False)
    if any(phrase in normalized for phrase in ("reschedule", "change my appointment", "change the appointment")):
        return ConversationIntent("appointment_change", 0.96, topic, True)
    if any(phrase in normalized for phrase in ("yes", "confirm", "looks good", "that's correct", "that is correct")):
        return ConversationIntent("confirmation", 0.9, topic, False)
    if any(phrase in normalized for phrase in ("actually", "change that", "correct that", "i meant")):
        return ConversationIntent("correction", 0.88, topic, True)
    if any(term in normalized for term in RECEPTIONIST_TERMS):
        return ConversationIntent("appointment_request", 0.96, topic, True)
    if any(term in normalized for term in PRACTICE_INFO_TERMS):
        return ConversationIntent("practice_information", 0.94, topic, True)
    if any(greeting in normalized.split() for greeting in ("hello", "hi", "hey", "good morning", "good afternoon")):
        return ConversationIntent("greeting", 0.98, topic, True)
    if any(phrase in normalized for phrase in ("who are you", "what are you", "what can you do", "how do you work")):
        return ConversationIntent("capabilities", 0.96, topic, True)
    if any(phrase in normalized for phrase in ("thank you", "thanks", "appreciate it")):
        return ConversationIntent("gratitude", 0.96, topic, True)
    if any(phrase in normalized for phrase in ("goodbye", "good bye", "see you", "bye")):
        return ConversationIntent("goodbye", 0.96, topic, False)
    if any(term in normalized for term in HEALTHCARE_TERMS):
        return ConversationIntent("healthcare_request", 0.94, topic, True)
    if normalized.endswith("?") or normalized.startswith(("what ", "why ", "how ", "when ", "where ", "can ", "could ", "would ")):
        return ConversationIntent("question", 0.82, topic, True)
    if any(phrase in normalized for phrase in ("i feel", "i think", "i am", "i'm", "i want", "i like", "i need")):
        return ConversationIntent("sharing", 0.78, topic, True)
    return ConversationIntent("general_conversation", 0.65, topic, True)


def extract_topic(text: str) -> str | None:
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    stop_words = {"this", "that", "with", "about", "have", "from", "would", "could", "like", "just", "really"}
    meaningful = [word for word in words if len(word) > 3 and word not in stop_words]
    return " ".join(meaningful[:4]) or None


def build_context(recent_messages: list[dict[str, str]], intent: ConversationIntent) -> dict[str, object]:
    user_turns = [message["content"] for message in recent_messages if message.get("role") == "user"]
    return {
        "turn_count": len(user_turns) + 1,
        "previous_user_message": user_turns[-1] if user_turns else None,
        "recent_topics": [extract_topic(message) for message in user_turns[-3:] if extract_topic(message)],
        "current_intent": intent.name,
        "current_topic": intent.topic,
    }
