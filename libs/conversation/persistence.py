"""Persistence boundary for the conversation session aggregate."""

from dataclasses import dataclass
from threading import RLock
from typing import Protocol

from libs.conversation.domain import (
    ConversationSession,
    DomainEvent,
    StateConflictError,
    reduce_session,
)


class PersistenceConflictError(StateConflictError):
    """Raised when a session changed before an event could be committed."""


@dataclass(frozen=True)
class CommitResult:
    session: ConversationSession
    event: DomainEvent
    idempotent_replay: bool = False


@dataclass(frozen=True)
class BatchCommitResult:
    session: ConversationSession
    events: tuple[DomainEvent, ...]
    idempotent_replay: bool = False


class SessionRepository(Protocol):
    """Durable repository contract; implementations must commit atomically."""

    def create(self, session_id: str) -> ConversationSession:
        ...

    def load(self, session_id: str) -> ConversationSession | None:
        ...

    def commit(
        self,
        session_id: str,
        expected_state_version: int,
        event: DomainEvent,
    ) -> CommitResult:
        """
        Append the event and update the state projection atomically.

        A production implementation must use one PostgreSQL transaction with
        optimistic version checking. Redis may coordinate work but cannot be
        the source of truth.
        """
        ...

    def commit_batch(
        self,
        session_id: str,
        expected_state_version: int,
        events: list[DomainEvent],
    ) -> BatchCommitResult:
        """Append a complete turn projection in one atomic transaction."""
        ...

    def events(self, session_id: str) -> list[DomainEvent]:
        ...


class InMemorySessionRepository:
    """Deterministic repository used for contract and reducer tests."""

    def __init__(self) -> None:
        self._sessions: dict[str, ConversationSession] = {}
        self._events: dict[str, list[DomainEvent]] = {}
        self._event_payloads: dict[str, dict[str, object]] = {}
        self._lock = RLock()

    def create(self, session_id: str) -> ConversationSession:
        with self._lock:
            if session_id in self._sessions:
                return self._sessions[session_id]
            session = ConversationSession.create(session_id)
            self._sessions[session_id] = session
            self._events[session_id] = []
            return session

    def load(self, session_id: str) -> ConversationSession | None:
        with self._lock:
            session = self._sessions.get(session_id)
            return session.model_copy(deep=True) if session else None

    def commit(
        self,
        session_id: str,
        expected_state_version: int,
        event: DomainEvent,
    ) -> CommitResult:
        result = self.commit_batch(session_id, expected_state_version, [event])
        return CommitResult(
            session=result.session,
            event=event,
            idempotent_replay=result.idempotent_replay,
        )

    def commit_batch(
        self,
        session_id: str,
        expected_state_version: int,
        events: list[DomainEvent],
    ) -> BatchCommitResult:
        if not events:
            raise PersistenceConflictError("cannot commit an empty event batch")
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = self.create(session_id)
            if any(event.aggregate_id != session_id for event in events):
                raise PersistenceConflictError("event aggregate does not match session")
            incoming_ids = [event.event_id for event in events]
            if len(set(incoming_ids)) != len(incoming_ids):
                raise PersistenceConflictError("event batch contains duplicate event IDs")
            existing_ids = [event_id for event_id in incoming_ids if event_id in self._event_payloads]
            if existing_ids:
                if len(existing_ids) != len(events):
                    raise PersistenceConflictError("event batch partially overlaps a prior commit")
                for event in events:
                    if self._event_payloads[event.event_id] != event.model_dump(mode="json"):
                        raise PersistenceConflictError("event ID was reused with different payload")
                return BatchCommitResult(
                    session=session.model_copy(deep=True),
                    events=tuple(events),
                    idempotent_replay=True,
                )
            if session.state.state_version != expected_state_version:
                raise PersistenceConflictError(
                    f"expected state version {expected_state_version}, "
                    f"found {session.state.state_version}"
                )
            working_session = session.model_copy(deep=True)
            for offset, event in enumerate(events, start=1):
                if event.aggregate_version != expected_state_version + offset:
                    raise PersistenceConflictError("event version is not sequential in the batch")
                working_session = reduce_session(working_session, event)
            self._sessions[session_id] = working_session
            self._events[session_id].extend(events)
            for event in events:
                self._event_payloads[event.event_id] = event.model_dump(mode="json")
            return BatchCommitResult(
                session=working_session.model_copy(deep=True),
                events=tuple(events),
            )

    def events(self, session_id: str) -> list[DomainEvent]:
        with self._lock:
            return list(self._events.get(session_id, []))
