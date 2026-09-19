"""Deterministic mock scheduling service for the Phase 2 workflow."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
import hashlib
import os
from typing import Protocol
from urllib.parse import quote
from zoneinfo import ZoneInfo

import httpx

from libs.conversation.domain import AvailabilitySlotRecord


@dataclass(frozen=True)
class AvailabilityResult:
    provider_id: str
    slots: list[AvailabilitySlotRecord]
    source: str
    error: str | None = None
    error_code: str | None = None


@dataclass(frozen=True)
class AppointmentSubmissionResult:
    request_reference: str
    status: str
    source: str
    error: str | None = None


@dataclass(frozen=True)
class AppointmentDetails:
    provider_id: str
    preferred_time: str
    caller_name: str
    callback_number: str
    appointment_reason: str


class SchedulingService(Protocol):
    def get_availability(self, provider_id: str) -> AvailabilityResult:
        ...

    def submit_request(
        self,
        *,
        idempotency_key: str,
        preferred_time: str,
        details: AppointmentDetails | None = None,
    ) -> AppointmentSubmissionResult:
        ...


class UnavailableSchedulingService:
    """Fail closed until a real scheduling adapter is configured."""

    def __init__(self, error_code: str = "not_configured") -> None:
        self._error_code = error_code

    def get_availability(self, provider_id: str) -> AvailabilityResult:
        return AvailabilityResult(
            provider_id=provider_id,
            slots=[],
            source="scheduling_unconfigured",
            error="No scheduling provider is configured.",
            error_code=self._error_code,
        )

    def submit_request(
        self,
        *,
        idempotency_key: str,
        preferred_time: str,
        details: AppointmentDetails | None = None,
    ) -> AppointmentSubmissionResult:
        return AppointmentSubmissionResult(
            request_reference="not-created",
            status="failed",
            source="scheduling_unconfigured",
            error="No scheduling provider is configured.",
        )


class MockSchedulingService:
    """Provide configured demo slots and idempotent request submission."""

    def __init__(self, slots: list[AvailabilitySlotRecord] | None = None) -> None:
        self._slots = slots or [
            AvailabilitySlotRecord(
                slot_id="mock-slot-1",
                start_at="2026-10-01T09:00:00+01:00",
                end_at="2026-10-01T09:30:00+01:00",
                label="Thursday, October 1 at 9:00 AM",
            ),
            AvailabilitySlotRecord(
                slot_id="mock-slot-2",
                start_at="2026-10-01T14:00:00+01:00",
                end_at="2026-10-01T14:30:00+01:00",
                label="Thursday, October 1 at 2:00 PM",
            ),
        ]
        self._submissions: dict[str, AppointmentSubmissionResult] = {}

    def get_availability(self, provider_id: str) -> AvailabilityResult:
        return AvailabilityResult(
            provider_id=provider_id,
            slots=list(self._slots),
            source="mock_scheduling",
        )

    def submit_request(
        self,
        *,
        idempotency_key: str,
        preferred_time: str,
        details: AppointmentDetails | None = None,
    ) -> AppointmentSubmissionResult:
        existing = self._submissions.get(idempotency_key)
        if existing is not None:
            return existing
        slot = next(
            (candidate for candidate in self._slots if preferred_time in {candidate.start_at, candidate.label}),
            None,
        )
        if slot is None:
            return AppointmentSubmissionResult(
                request_reference="not-created",
                status="failed",
                source="mock_scheduling",
                error="selected time is not one of the available mock slots",
            )
        reference = "mock-" + hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()[:12]
        result = AppointmentSubmissionResult(
            request_reference=reference,
            status="submitted",
            source="mock_scheduling",
        )
        self._submissions[idempotency_key] = result
        return result


class GoogleCalendarSchedulingService:
    """Use Google Calendar free/busy and event APIs when explicitly configured."""

    def __init__(self, http_client_factory=httpx.Client, now=None) -> None:
        self._access_token = os.getenv("GOOGLE_CALENDAR_ACCESS_TOKEN", "").strip()
        self._calendar_id = os.getenv("GOOGLE_CALENDAR_ID", "").strip()
        self._base_url = os.getenv(
            "GOOGLE_CALENDAR_BASE_URL",
            "https://www.googleapis.com/calendar/v3",
        ).rstrip("/")
        self._timezone_name = os.getenv("SCHEDULING_TIMEZONE", "UTC").strip()
        try:
            self._timezone = ZoneInfo(self._timezone_name)
        except Exception:
            self._timezone = timezone.utc
            self._timezone_name = "UTC"
        self._window_days = max(1, int(os.getenv("SCHEDULING_WINDOW_DAYS", "14")))
        self._slot_minutes = max(5, int(os.getenv("SCHEDULING_SLOT_MINUTES", "30")))
        self._max_slots = max(1, int(os.getenv("SCHEDULING_MAX_SLOTS", "10")))
        self._workday_start = _parse_clock(os.getenv("SCHEDULING_WORKDAY_START", "09:00"))
        self._workday_end = _parse_clock(os.getenv("SCHEDULING_WORKDAY_END", "17:00"))
        self._event_summary = os.getenv(
            "GOOGLE_CALENDAR_EVENT_SUMMARY",
            "TriageOS appointment request",
        )
        self._request_timeout = float(os.getenv("SCHEDULING_REQUEST_TIMEOUT", "8"))
        self._http_client_factory = http_client_factory
        self._now = now or (lambda: datetime.now(timezone.utc))

    def get_availability(self, provider_id: str) -> AvailabilityResult:
        configuration_error = self._configuration_error(provider_id)
        if configuration_error:
            return configuration_error
        now = self._now().astimezone(self._timezone)
        window_end = now + timedelta(days=self._window_days)
        try:
            payload = self._request(
                "POST",
                f"{self._base_url}/freeBusy",
                json={
                    "timeMin": now.isoformat(),
                    "timeMax": window_end.isoformat(),
                    "timeZone": self._timezone_name,
                    "items": [{"id": self._calendar_id}],
                },
            )
            busy = payload.get("calendars", {}).get(self._calendar_id, {}).get("busy", [])
            return AvailabilityResult(
                provider_id=provider_id,
                slots=self._build_slots(now, busy)[: self._max_slots],
                source="google_calendar",
            )
        except httpx.TimeoutException:
            return AvailabilityResult(
                provider_id=provider_id,
                slots=[],
                source="google_calendar",
                error="Google Calendar availability timed out.",
                error_code="timeout",
            )
        except httpx.HTTPStatusError as exc:
            return AvailabilityResult(
                provider_id=provider_id,
                slots=[],
                source="google_calendar",
                error="Google Calendar rejected the availability request.",
                error_code=_google_error_code(exc),
            )
        except (httpx.HTTPError, KeyError, TypeError, ValueError):
            return AvailabilityResult(
                provider_id=provider_id,
                slots=[],
                source="google_calendar",
                error="Google Calendar availability is unavailable.",
                error_code="unavailable",
            )

    def submit_request(
        self,
        *,
        idempotency_key: str,
        preferred_time: str,
        details: AppointmentDetails | None = None,
    ) -> AppointmentSubmissionResult:
        if not self._access_token or not self._calendar_id:
            return AppointmentSubmissionResult(
                request_reference="not-created",
                status="failed",
                source="google_calendar",
                error="Google Calendar is not configured.",
            )
        if details is None:
            return AppointmentSubmissionResult(
                request_reference="not-created",
                status="failed",
                source="google_calendar",
                error="Appointment details are incomplete.",
            )
        try:
            start = datetime.fromisoformat(preferred_time.replace("Z", "+00:00"))
            if start.tzinfo is None:
                start = start.replace(tzinfo=self._timezone)
            start = start.astimezone(self._timezone)
            end = start + timedelta(minutes=self._slot_minutes)
            event_id = "triageos" + hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()[:32]
            calendar_id = quote(self._calendar_id, safe="")
            payload = self._request(
                "POST",
                f"{self._base_url}/calendars/{calendar_id}/events",
                params={"sendUpdates": "none"},
                json={
                    "id": event_id,
                    "summary": self._event_summary,
                    "description": (
                        f"Requested through TriageOS.\nProvider: {details.provider_id}\n"
                        f"Reason: {details.appointment_reason}\nCaller: {details.caller_name}\n"
                        f"Callback: {details.callback_number}"
                    ),
                    "start": {"dateTime": start.isoformat(), "timeZone": self._timezone_name},
                    "end": {"dateTime": end.isoformat(), "timeZone": self._timezone_name},
                    "extendedProperties": {
                        "private": {"triageos_idempotency_key": idempotency_key}
                    },
                },
            )
            reference = str(payload.get("id") or event_id)
            return AppointmentSubmissionResult(
                request_reference=f"google:{reference}",
                status="submitted",
                source="google_calendar",
            )
        except ValueError:
            return AppointmentSubmissionResult(
                request_reference="not-created",
                status="failed",
                source="google_calendar",
                error="The selected time is not a valid calendar timestamp.",
            )
        except httpx.TimeoutException:
            return AppointmentSubmissionResult(
                request_reference="not-created",
                status="failed",
                source="google_calendar",
                error="Google Calendar request timed out.",
            )
        except httpx.HTTPStatusError as exc:
            return AppointmentSubmissionResult(
                request_reference="not-created",
                status="failed",
                source="google_calendar",
                error="Google Calendar rejected the appointment request.",
            )
        except httpx.HTTPError:
            return AppointmentSubmissionResult(
                request_reference="not-created",
                status="failed",
                source="google_calendar",
                error="Google Calendar is unavailable.",
            )

    def _configuration_error(self, provider_id: str) -> AvailabilityResult | None:
        if self._access_token and self._calendar_id:
            return None
        return AvailabilityResult(
            provider_id=provider_id,
            slots=[],
            source="google_calendar",
            error="Google Calendar is not configured.",
            error_code="not_configured",
        )

    def _request(self, method: str, url: str, *, json: dict, params: dict | None = None) -> dict:
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        with self._http_client_factory(timeout=self._request_timeout, headers=headers) as client:
            response = client.request(method, url, params=params, json=json)
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Google Calendar returned an invalid response.")
        return payload

    def _build_slots(self, now: datetime, busy: list[dict]) -> list[AvailabilitySlotRecord]:
        busy_intervals = [
            (
                datetime.fromisoformat(item["start"].replace("Z", "+00:00")).astimezone(self._timezone),
                datetime.fromisoformat(item["end"].replace("Z", "+00:00")).astimezone(self._timezone),
            )
            for item in busy
        ]
        slots: list[AvailabilitySlotRecord] = []
        duration = timedelta(minutes=self._slot_minutes)
        for offset in range(self._window_days):
            day = (now + timedelta(days=offset)).date()
            if day.weekday() >= 5:
                continue
            cursor = datetime.combine(day, self._workday_start, self._timezone)
            day_end = datetime.combine(day, self._workday_end, self._timezone)
            while cursor + duration <= day_end:
                end = cursor + duration
                if cursor > now and not any(cursor < busy_end and end > busy_start for busy_start, busy_end in busy_intervals):
                    token = f"{self._calendar_id}:{cursor.isoformat()}"
                    slots.append(
                        AvailabilitySlotRecord(
                            slot_id="google-slot-" + hashlib.sha256(token.encode("utf-8")).hexdigest()[:16],
                            start_at=cursor.isoformat(),
                            end_at=end.isoformat(),
                            label=_slot_label(cursor),
                        )
                    )
                cursor += duration
        return slots


def build_scheduling_service() -> SchedulingService:
    provider = os.getenv("SCHEDULING_PROVIDER", "unconfigured").strip().lower()
    if provider == "mock":
        return MockSchedulingService()
    if provider == "google_calendar":
        return GoogleCalendarSchedulingService()
    if provider not in {"", "unconfigured"}:
        return UnavailableSchedulingService(error_code="misconfigured")
    return UnavailableSchedulingService()


def _parse_clock(value: str) -> time:
    hour, minute = (int(part) for part in value.split(":", 1))
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError("invalid scheduling workday time")
    return time(hour, minute)


def _slot_label(value: datetime) -> str:
    hour = value.hour % 12 or 12
    meridiem = "AM" if value.hour < 12 else "PM"
    return f"{value.strftime('%A, %B')} {value.day} at {hour}:{value.minute:02d} {meridiem}"


def _google_error_code(error: httpx.HTTPStatusError) -> str:
    if error.response.status_code in {401, 403}:
        return "not_authorized"
    if error.response.status_code == 429 or error.response.status_code >= 500:
        return "unavailable"
    return "invalid_request"
