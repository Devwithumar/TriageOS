"""Versioned, deterministic practice-profile lookup."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from libs.conversation.domain import PracticeHoursRecord, PracticeProfileResultData


class PracticeProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_id: str = Field(min_length=1)
    profile_version: int = Field(ge=1)
    display_name: str = Field(min_length=1)
    address: str | None = None
    phone: str | None = None
    website: str | None = None
    hours: list[PracticeHoursRecord] = Field(default_factory=list)
    services: list[str] = Field(default_factory=list)
    accepted_insurance: list[str] = Field(default_factory=list)


class PracticeProfileLookup:
    """Load only explicitly configured practice data; never invent defaults."""

    def __init__(self, path: str | Path | None = None) -> None:
        configured_path = path or os.getenv("TRIAGEOS_PRACTICE_PROFILE_PATH")
        self._path = Path(configured_path) if configured_path else self._default_path()

    def lookup(self) -> PracticeProfileResultData:
        try:
            with self._path.open("r", encoding="utf-8") as profile_file:
                profile = PracticeProfile.model_validate(json.load(profile_file))
        except FileNotFoundError:
            return self._failure("misconfigured", "practice profile is not configured")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            return self._failure("invalid_profile", f"practice profile is invalid: {exc}")

        return PracticeProfileResultData(
            profile_id=profile.profile_id,
            profile_version=profile.profile_version,
            display_name=profile.display_name,
            address=profile.address,
            phone=profile.phone,
            website=profile.website,
            hours=profile.hours,
            services=profile.services,
            accepted_insurance=profile.accepted_insurance,
            source="configured_practice_profile",
        )

    @staticmethod
    def _default_path() -> Path:
        return Path(__file__).resolve().parents[3] / "config" / "practice_profile.json"

    @staticmethod
    def _failure(error_code: str, error: str) -> PracticeProfileResultData:
        return PracticeProfileResultData(
            profile_id="unconfigured",
            profile_version=1,
            display_name="",
            source="configured_practice_profile",
            error=error,
            error_code=error_code,
        )


def practice_profile_topic(text: str) -> Literal["hours", "location", "contact", "services", "insurance", "overview"]:
    normalized = " ".join(text.lower().split())
    if any(term in normalized for term in ("hour", "open", "close")):
        return "hours"
    if any(term in normalized for term in ("where", "location", "address")):
        return "location"
    if any(term in normalized for term in ("phone", "call", "contact")):
        return "contact"
    if any(term in normalized for term in ("insurance", "cover")):
        return "insurance"
    if any(term in normalized for term in ("service", "offer", "provide", "available")):
        return "services"
    return "overview"
