import logging
import re

from libs.ai.conversation_intelligence import (
    LOCAL_LOOKUP_MARKERS,
    build_context,
    detect_intent,
)
from libs.ai.llm import CompletionResult, complete_conversation
from libs.ai.model_router import route_model
from libs.conversation.contracts import (
    ConversationState,
    ExtractedSlot,
    IntentClassification,
    IntentName,
    ReceptionistState,
    ToolResult,
    ToolRequest,
    WorkflowContext,
    WorkflowName,
)
from libs.conversation.workflow import (
    APPOINTMENT_CONTEXT_FIELDS,
    APPOINTMENT_DETAIL_FIELDS,
    apply_receptionist_turn,
)
from services.conversation.app.provider_directory import ProviderDirectory, ProviderSearchResult

logger = logging.getLogger(__name__)
provider_directory = ProviderDirectory()

SAFETY_REPLY = (
    "Chest pain can be serious. If this is happening now, call your local emergency "
    "services immediately or have someone take you to the nearest emergency department. "
    "Please do not drive yourself."
)

LLM_FALLBACK_REPLY = (
    "I’m having trouble reaching my conversation service right now. "
    "Please try again in a moment."
)

UNSUPPORTED_LOCAL_REPLY = (
    "I’m focused on healthcare-related help, so I can’t reliably look up supermarkets, "
    "restaurants, or other general local businesses. I can help find a healthcare provider or appointment."
)

UNVERIFIED_FACT_TERMS = (
    "our clinic",
    "our hospital",
    "directions to",
    "the address is",
    "located at",
    "phone number is",
    "minutes away",
    "minute drive",
    "well-known",
    "well-rated",
    "well rated",
)

INTENT_MAP = {
    "greeting": IntentName.GREETING,
    "capabilities": IntentName.CAPABILITIES,
    "gratitude": IntentName.GRATITUDE,
    "practice_information": IntentName.PRACTICE_INFORMATION,
    "provider_lookup": IntentName.PROVIDER_LOOKUP,
    "unsupported_local_search": IntentName.UNSUPPORTED_LOCAL_SEARCH,
    "appointment_request": IntentName.APPOINTMENT_REQUEST,
    "appointment_change": IntentName.APPOINTMENT_CHANGE,
    "appointment_cancellation": IntentName.APPOINTMENT_CANCELLATION,
    "confirmation": IntentName.CONFIRMATION,
    "correction": IntentName.CORRECTION,
    "urgent_safety": IntentName.URGENT_SAFETY,
    "healthcare_request": IntentName.UNSUPPORTED_CLINICAL,
    "general_conversation": IntentName.GENERAL_CONVERSATION,
    "question": IntentName.GENERAL_CONVERSATION,
    "sharing": IntentName.GENERAL_CONVERSATION,
    "unclear": IntentName.UNKNOWN,
}

STATUS_MAP = {
    "collecting_details": ReceptionistState.COLLECTING_DETAILS,
    "reviewing_request": ReceptionistState.REVIEWING_REQUEST,
    "confirmed": ReceptionistState.CONFIRMED,
    "submitted": ReceptionistState.SUBMITTED,
    "completed": ReceptionistState.COMPLETED,
    "cancelled": ReceptionistState.CANCELLED,
    "correction_required": ReceptionistState.CORRECTION_REQUIRED,
    "submission_failed": ReceptionistState.SUBMISSION_FAILED,
}


def _contract_intent(intent_name: str, confidence: float, topic: str | None) -> IntentClassification:
    return IntentClassification(
        name=INTENT_MAP.get(intent_name, IntentName.UNKNOWN),
        confidence=confidence,
        topic=topic,
    )


def _conversation_state(session_id: str, structured_state: dict[str, object]) -> ConversationState:
    workflow_state = STATUS_MAP.get(
        str(structured_state.get("appointment_status", "idle")),
        ReceptionistState.IDLE,
    )
    slots = {
        field: ExtractedSlot(value=str(structured_state[field]), confidence=1, source="session_state")
        for field in (
            *APPOINTMENT_CONTEXT_FIELDS,
            *APPOINTMENT_DETAIL_FIELDS,
            "provider_name",
            "provider_id",
            "email",
        )
        if structured_state.get(field)
    }
    workflow_name = "receptionist" if workflow_state != ReceptionistState.IDLE else "general"
    missing_fields = structured_state.get("missing_fields", [])
    if not isinstance(missing_fields, list):
        missing_fields = []
    provider_options = structured_state.get("provider_options", [])
    if not isinstance(provider_options, list):
        provider_options = []
    return ConversationState(
        session_id=session_id,
        workflow={
            "name": workflow_name,
            "state": workflow_state,
            "next_action": structured_state.get("next_action"),
            "missing_fields": missing_fields,
        },
        slots=slots,
        provider_options=provider_options,
    )


def _extract_entities(
    user_text: str,
    intent_name: str,
    current_state: ConversationState,
) -> dict[str, ExtractedSlot]:
    entities: dict[str, ExtractedSlot] = {}
    name_match = re.search(
        r"\b(?:my name is|i am|i'm|put the name|the name is|name is)\s+([A-Za-z][A-Za-z .'-]{1,50})",
        user_text,
        re.IGNORECASE,
    )
    if name_match and not any(char.isdigit() for char in name_match.group(1)):
        entities["caller_name"] = ExtractedSlot(
            value=name_match.group(1).strip(" ."), confidence=0.98, source="conversation"
        )

    phone_match = re.search(r"(?:\+?\d[\d ()-]{7,}\d)", user_text)
    if phone_match:
        entities["callback_number"] = ExtractedSlot(
            value=phone_match.group(0).strip(), confidence=0.98, source="conversation"
        )

    email_match = re.search(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b", user_text)
    if email_match:
        entities["email"] = ExtractedSlot(
            value=email_match.group(0).lower(), confidence=0.99, source="conversation"
        )

    lowered = user_text.lower()
    if current_state.workflow.next_action == "select_provider":
        selected_provider = _select_provider(user_text, current_state.provider_options)
        if selected_provider:
            entities["provider_name"] = ExtractedSlot(
                value=str(selected_provider["name"]), confidence=0.94, source="provider_directory"
            )
            entities["provider_id"] = ExtractedSlot(
                value=str(selected_provider["provider_id"]), confidence=0.99, source="provider_directory"
            )
            return entities

    time_markers = (
        "morning", "afternoon", "evening", "tomorrow", "next week", "monday", "tuesday",
        "wednesday", "thursday", "friday", "saturday", "sunday",
    )
    if any(marker in lowered for marker in time_markers):
        preferred_time = re.sub(r"(?:\+?\d[\d ()-]{7,}\d)", "", user_text).strip(" ,.-")
        entities["preferred_time"] = ExtractedSlot(
            value=preferred_time, confidence=0.88, source="conversation"
        )

    expected_slot = (current_state.workflow.next_action or "").removeprefix("collect_")
    expected_fields = set(APPOINTMENT_CONTEXT_FIELDS + APPOINTMENT_DETAIL_FIELDS)
    is_question = user_text.strip().endswith("?") or lowered.lstrip().startswith(
        ("what ", "why ", "how ", "which ", "can you ", "could you ")
    )
    if (
        current_state.workflow.state == ReceptionistState.COLLECTING_DETAILS
        and expected_slot in expected_fields
        and not entities
        and not is_question
        and user_text.strip()
    ):
        entities[expected_slot] = ExtractedSlot(
            value=user_text.strip(), confidence=0.72, source="conversation"
        )
    return entities


def _select_provider(text: str, options: list[dict[str, object]]) -> dict[str, object] | None:
    normalized = text.lower().strip()
    ordinal_indexes = {
        "first": 0,
        "1": 0,
        "one": 0,
        "second": 1,
        "2": 1,
        "two": 1,
        "third": 2,
        "3": 2,
        "three": 2,
    }
    for token, index in ordinal_indexes.items():
        if re.search(rf"\b{re.escape(token)}\b", normalized) and index < len(options):
            return options[index]
    for option in options:
        name = str(option.get("name", "")).lower()
        if name and name in normalized:
            return option
    return None


def _structured_state(decision_state: ConversationState, decision_next_action: str) -> dict[str, object]:
    return {
        **{name: slot.value for name, slot in decision_state.slots.items()},
        "provider_options": decision_state.provider_options,
        "appointment_status": decision_state.workflow.state.value,
        "workflow": decision_state.workflow.name.value,
        "next_action": decision_next_action,
        "missing_fields": decision_state.workflow.missing_fields,
    }


def generate_reply(
    user_text: str,
    recent_messages: list[dict[str, str]],
    structured_state: dict[str, object] | None = None,
    session_id: str = "conversation",
) -> dict[str, object]:
    route = route_model("conversation")
    intent = detect_intent(user_text)
    context = build_context(recent_messages, intent)
    current_state = _conversation_state(session_id, dict(structured_state or {}))
    contract_intent = _contract_intent(intent.name, intent.confidence, intent.topic)
    workflow_question = (
        current_state.workflow.state == ReceptionistState.COLLECTING_DETAILS
        and (
            user_text.strip().endswith("?")
            or user_text.lower().lstrip().startswith(("what ", "why ", "how ", "which ", "can you "))
        )
    )
    if workflow_question:
        contract_intent = IntentClassification(
            name=IntentName.GENERAL_CONVERSATION,
            confidence=intent.confidence,
            topic=intent.topic,
        )

    if intent.name == "urgent_safety":
        result = CompletionResult(
            text=SAFETY_REPLY,
            model="deterministic-safety",
            provider="guardrail",
            reason="urgent safety escalation guardrail",
        )
        updated_state = dict(structured_state or {})
    elif intent.name == "unsupported_local_search":
        result = CompletionResult(
            text=UNSUPPORTED_LOCAL_REPLY,
            model="deterministic-policy",
            provider="policy",
            reason="non-healthcare local search is outside the receptionist boundary",
        )
        updated_state = dict(structured_state or {})
    elif intent.name == "provider_lookup" or _is_local_follow_up(user_text, structured_state):
        result, updated_state, tool_request, tool_result = _handle_local_lookup(
            user_text,
            structured_state,
        )
        context["tool_call"] = tool_request.model_dump() if tool_request else None
        context["tool_result"] = tool_result.model_dump() if tool_result else None
    elif (
        intent.name in {
            "appointment_request",
            "appointment_change",
            "appointment_cancellation",
            "practice_information",
        }
        or current_state.workflow.state == ReceptionistState.COLLECTING_DETAILS
        or current_state.workflow.state == ReceptionistState.REVIEWING_REQUEST
        or (
            intent.name in {"confirmation", "correction"}
            and current_state.workflow.state != ReceptionistState.IDLE
        )
    ):
        entities = _extract_entities(user_text, intent.name, current_state)
        decision = apply_receptionist_turn(current_state, contract_intent, entities)
        tool_result = decision.tool_result
        if decision.tool_call and decision.tool_call.name == "search_providers":
            search_result = provider_directory.search(
                care_setting=str(decision.tool_call.arguments["care_setting"]),
                location=str(decision.tool_call.arguments["location"]),
                reason=str(decision.tool_call.arguments.get("appointment_reason", "")),
            )
            tool_result = ToolResult(
                name="search_providers",
                status="success" if search_result.providers else "empty",
                data=search_result.as_dict(),
                error=search_result.error,
            )
            decision = _apply_provider_search_result(decision, search_result)
        result = CompletionResult(
            text=decision.response,
            model="deterministic-orchestrator",
            provider="workflow",
            reason=f"workflow transition: {decision.transition}",
        )
        updated_state = _structured_state(decision.state, decision.next_action)
        context["workflow_transition"] = decision.transition
        context["tool_call"] = decision.tool_call.model_dump() if decision.tool_call else None
        context["tool_result"] = tool_result.model_dump() if tool_result else None
    else:
        try:
            result = complete_conversation(user_text, recent_messages, context)
            result = _guard_generated_reply(result, context)
        except Exception:
            logger.exception("conversation_provider_failed session_id=%s", session_id)
            result = CompletionResult(
                text=LLM_FALLBACK_REPLY,
                model="provider-fallback",
                provider="fallback",
                reason="conversation provider unavailable",
            )
        updated_state = dict(structured_state or {})

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
        "structured_state": updated_state,
    }


def _is_local_follow_up(user_text: str, structured_state: dict[str, object] | None) -> bool:
    local_search = (structured_state or {}).get("local_search")
    if not isinstance(local_search, dict):
        return False
    if local_search.get("status") == "pending":
        return True
    normalized = " ".join(user_text.lower().split())
    return any(marker in normalized for marker in LOCAL_LOOKUP_MARKERS)


def _handle_local_lookup(
    user_text: str,
    structured_state: dict[str, object] | None,
) -> tuple[CompletionResult, dict[str, object], ToolRequest | None, ToolResult | None]:
    next_state = dict(structured_state or {})
    saved_request = next_state.get("local_search")
    local_search = dict(saved_request) if isinstance(saved_request, dict) else {}
    provider_name = _extract_named_provider(user_text) or local_search.get("provider_name")
    care_setting = _care_setting_hint(user_text) or local_search.get("care_setting") or "general medical"
    location = _extract_location_hint(user_text) or local_search.get("location")
    if not location and local_search.get("status") == "pending" and not _looks_like_question(user_text):
        location = _clean_location_candidate(user_text)

    if not location:
        local_search.update(
            {
                "status": "pending",
                "care_setting": care_setting,
                "provider_name": provider_name,
            }
        )
        next_state["local_search"] = local_search
        if provider_name:
            reply = f"What city, neighborhood, or postal code should I use to verify {provider_name}?"
        else:
            reply = "What city, neighborhood, or postal code should I search?"
        return (
            CompletionResult(
                text=reply,
                model="deterministic-provider-directory",
                provider="provider_directory",
                reason="location required before factual provider lookup",
            ),
            next_state,
            None,
            None,
        )

    if local_search.get("status") == "completed" and not provider_name and not _extract_location_hint(user_text):
        providers = local_search.get("providers", [])
        reply = _format_follow_up_lookup(user_text, providers)
        return (
            CompletionResult(
                text=reply,
                model="deterministic-provider-directory",
                provider="provider_directory",
                reason="responded from previously verified provider results",
            ),
            next_state,
            None,
            None,
        )

    if provider_name:
        search_result = provider_directory.search_named_provider(provider_name, str(location))
        tool_request = ToolRequest(
            name="search_named_provider",
            arguments={"provider_name": provider_name, "location": location},
        )
    else:
        search_result = provider_directory.search(str(care_setting), str(location))
        tool_request = ToolRequest(
            name="search_providers",
            arguments={"care_setting": care_setting, "location": location},
        )
    tool_result = ToolResult(
        name=tool_request.name,
        status="success" if search_result.providers else "empty",
        data=search_result.as_dict(),
        error=search_result.error,
    )
    local_search.update(
        {
            "status": "completed",
            "care_setting": care_setting,
            "provider_name": provider_name,
            "location": location,
            "providers": [provider.as_dict() for provider in search_result.providers],
            "source": search_result.source,
            "error": search_result.error,
        }
    )
    next_state["local_search"] = local_search
    return (
        CompletionResult(
            text=_format_lookup_result(search_result, provider_name),
            model="deterministic-provider-directory",
            provider="provider_directory",
            reason="factual response rendered from provider directory result",
        ),
        next_state,
        tool_request,
        tool_result,
    )


def _extract_named_provider(text: str) -> str | None:
    match = re.search(
        r"\b(?:clinic|hospital|provider|vet|veterinary clinic)\s+(?:called|named)\s+([A-Za-z0-9][A-Za-z0-9 .'-]{1,80}?)(?:\?|\.|,|$)",
        text,
        re.IGNORECASE,
    )
    return match.group(1).strip() if match else None


def _extract_location_hint(text: str) -> str | None:
    postal_match = re.search(
        r"\b(?:zip(?:\s+code)?|postal(?:\s+code)?)\s*(?:is\s*)?"
        r"(?P<postal>[A-Za-z0-9-]{3,12})(?:,\s*(?P<area>[^.?;]+?))?"
        r"(?=\s*(?:[.;?]|$)|\s+(?:find|look|search)\b)",
        text,
        re.IGNORECASE,
    )
    if postal_match:
        postal_code = postal_match.group("postal").strip(" .,?")
        area = postal_match.group("area")
        candidate = f"{postal_code}, {area.strip(' .,?')}" if area else postal_code
        return _clean_location_candidate(candidate)
    match = re.search(
        r"\b(?:i\s+live|i\s+am|i'm|located|based|stay)\s+(?:at|in|within|around)\s+(.+?)(?:\?|\.|$)",
        text,
        re.IGNORECASE,
    )
    if match:
        return _clean_location_candidate(match.group(1))
    match = re.search(
        r"\b(?:in|at|around|near)\s+([A-Za-z][A-Za-z .,'-]{2,80}?)(?:\?|\.|$)",
        text,
        re.IGNORECASE,
    )
    return _clean_location_candidate(match.group(1)) if match else None


def _clean_location_candidate(value: str) -> str | None:
    candidate = value.strip(" .,?;:")
    candidate = re.split(
        r"\s*(?:,\s*)?(?:so|and)\s+(?:just\s+)?(?:find|look|search|somewhere|anything)\b",
        candidate,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    candidate = re.split(
        r"\s*(?:,\s*)?(?:find|look\s+up|search)\b",
        candidate,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    candidate = candidate.strip(" .,?;:")
    if candidate.lower() in {"me", "here", "there"}:
        return None
    return candidate or None


def _care_setting_hint(text: str) -> str | None:
    normalized = text.lower()
    for term in ("veterinary", "vet", "animal", "pet", "dental", "dentist", "emergency department", "hospital", "clinic"):
        if term in normalized:
            return term
    return None


def _looks_like_question(text: str) -> bool:
    normalized = text.lower().lstrip()
    return text.strip().endswith("?") or normalized.startswith(("what ", "where ", "which ", "how ", "can you "))


def _format_lookup_result(search_result: ProviderSearchResult, provider_name: str | None) -> str:
    if not search_result.providers:
        if provider_name:
            return f"I couldn’t verify {provider_name} from the connected provider directory for that location."
        if search_result.error:
            return "I couldn’t reach the provider directory right now, so I won’t invent a clinic or address."
        return "I couldn’t find a matching healthcare provider in that area from the connected directory."
    response_lines = [
        "I found these directory records. They are not ratings, endorsements, or appointment confirmations:"
    ]
    for index, provider in enumerate(search_result.providers, start=1):
        address = f", {provider.address}" if provider.address else ""
        response_lines.append(f"{index}. {provider.name}{address}, about {provider.distance_km:.1f} km away.")
    response_lines.append("Which result would you like to use, or would you like me to verify a specific one?")
    return " ".join(response_lines)


def _format_follow_up_lookup(text: str, providers: object) -> str:
    if not isinstance(providers, list) or not providers:
        return "I don’t have a verified provider result to use for that request."
    if any(marker in text.lower() for marker in ("direction", "how do i get", "drive", "route")):
        return "I can share the verified provider address, but a routing service is not connected yet. " + _format_lookup_options(providers)
    return _format_lookup_options(providers)


def _format_lookup_options(providers: list[object]) -> str:
    options = []
    for provider in providers:
        if not isinstance(provider, dict):
            continue
        address = f", {provider['address']}" if provider.get("address") else ""
        options.append(f"{provider.get('name', 'Unnamed provider')}{address}")
    return "Verified directory details: " + "; ".join(options) + "."


def _guard_generated_reply(result: CompletionResult, context: dict[str, object]) -> CompletionResult:
    if context.get("tool_result"):
        return result
    normalized = result.text.lower()
    if not any(term in normalized for term in UNVERIFIED_FACT_TERMS):
        return result
    logger.warning("suppressed_unverified_generated_claim provider=%s", result.provider)
    return CompletionResult(
        text=(
            "I can help with appointment requests and general receptionist conversation, "
            "but I don’t have verified practice location information available yet."
        ),
        model="deterministic-policy",
        provider="policy",
        reason="suppressed unverified practice or local claim",
    )


def _apply_provider_search_result(decision, search_result: ProviderSearchResult):
    next_state = decision.state.model_copy(deep=True)
    next_state.provider_options = [provider.as_dict() for provider in search_result.providers]
    if search_result.providers:
        next_state.workflow = WorkflowContext(
            name=WorkflowName.RECEPTIONIST,
            state=ReceptionistState.COLLECTING_DETAILS,
            next_action="select_provider",
        )
        response_lines = [
            "I found these nearby providers. These are directory results, not a booking or endorsement:"
        ]
        for index, provider in enumerate(search_result.providers, start=1):
            distance = f"{provider.distance_km:.1f} km away"
            address = f" — {provider.address}" if provider.address else ""
            response_lines.append(f"{index}. {provider.name}, {distance}{address}.")
        response_lines.append("Which provider would you like to use?")
        response = " ".join(response_lines)
        next_action = "select_provider"
    else:
        next_state.workflow = WorkflowContext(
            name=WorkflowName.RECEPTIONIST,
            state=ReceptionistState.COLLECTING_DETAILS,
            next_action="search_providers",
        )
        if search_result.error:
            response = (
                "I couldn’t search the provider directory right now. "
                "I won’t invent a clinic, so please try another location in a moment."
            )
        else:
            response = (
                "I couldn’t find a matching provider in that area. "
                "Would you like to try a nearby location?"
            )
        next_action = "search_providers"
    return decision.model_copy(
        update={
            "state": next_state,
            "transition": "providers_found" if search_result.providers else "providers_not_found",
            "next_action": next_action,
            "response": response,
        }
    )
