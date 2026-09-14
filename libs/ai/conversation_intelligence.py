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

CAPABILITY_ACTION_TERMS = (
    "what can",
    "how can you help",
    "what services",
    "what do you offer",
    "what do you help with",
    "what are you able to do",
)

PRACTICE_TARGET_TERMS = (
    "clinic",
    "hospital",
    "practice",
    "office",
    "provider",
    "facility",
)

PROVIDER_TERMS = (
    "clinic",
    "hospital",
    "doctor",
    "dentist",
    "dental",
    "veterinary",
    "vet",
    "pharmacy",
    "emergency department",
)

LOCAL_LOOKUP_MARKERS = (
    "nearby",
    " near ",
    "near me",
    "close to me",
    "closest",
    "nearest",
    "where is",
    "where are",
    "address",
    "directions",
    "how do i get",
    "how far",
    "how close",
    "drive",
    "zip code",
    "postcode",
)

UNSUPPORTED_LOCAL_TERMS = (
    "supermarket",
    "grocery",
    "fast food",
    "restaurant",
    "ice cream",
    "shopping mall",
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
    if any(term in normalized for term in UNSUPPORTED_LOCAL_TERMS) and any(
        marker in normalized for marker in LOCAL_LOOKUP_MARKERS
    ):
        return ConversationIntent("unsupported_local_search", 0.98, topic, True)
    if any(
        phrase in normalized
        for phrase in (
            "cancel my appointment",
            "cancel the appointment",
            "i want to cancel",
            "forget about the appointment",
            "forget the appointment",
            "never mind the appointment",
            "let's talk about something else",
            "lets talk about something else",
        )
    ):
        return ConversationIntent("appointment_cancellation", 0.97, topic, False)
    if any(phrase in normalized for phrase in ("reschedule", "change my appointment", "change the appointment")):
        return ConversationIntent("appointment_change", 0.96, topic, True)
    if any(phrase in normalized for phrase in ("yes", "confirm", "looks good", "that's correct", "that is correct")):
        return ConversationIntent("confirmation", 0.9, topic, False)
    if any(phrase in normalized for phrase in ("actually", "change that", "correct that", "i meant")):
        return ConversationIntent("correction", 0.88, topic, True)
    if any(term in normalized for term in RECEPTIONIST_TERMS):
        return ConversationIntent("appointment_request", 0.96, topic, True)
    if _is_capability_question(normalized):
        return ConversationIntent("capabilities", 0.96, topic, True)
    if _is_practice_information_question(normalized):
        return ConversationIntent("practice_information", 0.94, topic, True)
    if any(term in normalized for term in PROVIDER_TERMS) and any(
        marker in normalized for marker in LOCAL_LOOKUP_MARKERS
    ):
        return ConversationIntent("provider_lookup", 0.94, topic, True)
    if any(greeting in normalized.split() for greeting in ("hello", "hi", "hey", "good morning", "good afternoon")):
        return ConversationIntent("greeting", 0.98, topic, True)
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


def _is_capability_question(normalized: str) -> bool:
    if any(phrase in normalized for phrase in ("who are you", "what are you", "what can you do", "how do you work")):
        return True
    if not any(term in normalized for term in CAPABILITY_ACTION_TERMS):
        return False
    if any(term in normalized for term in PRACTICE_TARGET_TERMS):
        return False
    return "triageos" in normalized or "you" in normalized or "your" in normalized


def _is_practice_information_question(normalized: str) -> bool:
    if any(term in normalized for term in PRACTICE_INFO_TERMS):
        return True
    return any(term in normalized for term in PRACTICE_TARGET_TERMS) and any(
        term in normalized for term in ("service", "offer", "provide", "available")
    )


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
