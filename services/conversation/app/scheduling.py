"""Deterministic mock scheduling service for the Phase 2 workflow."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

from libs.conversation.domain import AvailabilitySlotRecord


@dataclass(frozen=True)
class AvailabilityResult:
    provider_id: str
    slots: list[AvailabilitySlotRecord]
    error: str | None = None


@dataclass(frozen=True)
class AppointmentSubmissionResult:
    request_reference: str
    status: str
    error: str | None = None


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
        return AvailabilityResult(provider_id=provider_id, slots=list(self._slots))

    def submit_request(
        self,
        *,
        idempotency_key: str,
        preferred_time: str,
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
                error="selected time is not one of the available mock slots",
            )
        reference = "mock-" + hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()[:12]
        result = AppointmentSubmissionResult(
            request_reference=reference,
            status="submitted",
        )
        self._submissions[idempotency_key] = result
        return result
