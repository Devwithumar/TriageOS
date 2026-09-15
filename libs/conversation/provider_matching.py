"""Deterministic matching of user references to verified provider options."""

import re

from libs.conversation.domain import ProviderResultRecord


_ORDINALS = {
    "first": 0,
    "1": 0,
    "one": 0,
    "second": 1,
    "2": 1,
    "two": 1,
    "third": 2,
    "3": 2,
    "three": 2,
    "fourth": 3,
    "4": 3,
    "four": 3,
    "fifth": 4,
    "5": 4,
    "five": 4,
}


def resolve_provider_reference(
    user_text: str,
    options: list[ProviderResultRecord],
) -> ProviderResultRecord | None:
    """Resolve an ordinal or meaningful name fragment to one known option."""

    normalized = " ".join(user_text.lower().split())
    ordinal_index = _ordinal_index(normalized)
    if ordinal_index is not None:
        return options[ordinal_index] if ordinal_index < len(options) else None

    query_tokens = _meaningful_tokens(normalized)
    if not query_tokens:
        return None
    matches = []
    for option in options:
        option_tokens = _meaningful_tokens(option.name.lower())
        compact_name = _compact(option.name)
        compact_query = _compact(user_text)
        token_match = any(token in compact_name for token in query_tokens)
        compact_match = len(compact_query) >= 4 and compact_query in compact_name
        if token_match or compact_match:
            matches.append(option)
    return matches[0] if len(matches) == 1 else None


def is_provider_details_request(user_text: str) -> bool:
    normalized = " ".join(user_text.lower().split())
    return any(
        phrase in normalized
        for phrase in (
            "details",
            "tell me more",
            "tell me about",
            "a bit about",
            "more about",
            "what can you tell me",
            "address",
            "where is",
            "located",
            "phone",
            "website",
            "contact",
            "information",
        )
    )


def is_provider_retry_request(user_text: str) -> bool:
    normalized = " ".join(user_text.lower().split())
    return any(
        phrase in normalized
        for phrase in (
            "try again",
            "search again",
            "look again",
            "check again",
            "retry",
            "run that again",
        )
    )


def _ordinal_index(normalized: str) -> int | None:
    match = re.search(
        r"\b(?:option|choice|number|no\.?|the)?\s*"
        r"(first|1|one|second|2|two|third|3|three|fourth|4|four|fifth|5|five)\b",
        normalized,
    )
    return _ORDINALS.get(match.group(1)) if match else None


def _meaningful_tokens(value: str) -> set[str]:
    ignored = {
        "about",
        "address",
        "and",
        "at",
        "can",
        "details",
        "get",
        "hospital",
        "information",
        "located",
        "me",
        "more",
        "please",
        "the",
        "there",
        "what",
        "where",
        "which",
        "would",
        "you",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if len(token) >= 4 and token not in ignored
    }


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())
