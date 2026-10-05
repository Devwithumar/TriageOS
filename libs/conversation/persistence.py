"""Persistence boundary for the conversation session aggregate."""

from dataclasses import dataclass
from threading import RLock
import json
from typing import Any, Callable, Protocol

from pydantic import TypeAdapter

from libs.conversation.domain import (
    ConversationSession,
    DomainEvent,
    StateConflictError,
    reduce_session,
)


class PersistenceConflictError(StateConflictError):
    """Raised when a session changed before an event could be committed."""


class PersistenceUnavailableError(RuntimeError):
    """Raised when the configured durable repository cannot be reached."""


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

    def readiness(self) -> dict[str, object]:
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

    def readiness(self) -> dict[str, object]:
        return {"status": "disabled", "mode": "memory"}


class PostgresSessionRepository:
    """PostgreSQL-backed event log and session projection repository."""

    _event_adapter = TypeAdapter(DomainEvent)

    def __init__(
        self,
        dsn: str,
        connection_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if not dsn.strip():
            raise ValueError("PostgreSQL repository requires a non-empty DSN")
        self._dsn = dsn
        self._connection_factory = connection_factory

    def _connect(self) -> Any:
        if self._connection_factory is not None:
            return self._connection_factory(self._dsn)
        try:
            import psycopg
        except ImportError as exc:
            raise PersistenceUnavailableError(
                "psycopg is required for the PostgreSQL conversation repository"
            ) from exc
        try:
            return psycopg.connect(self._dsn)
        except Exception as exc:
            raise PersistenceUnavailableError(
                f"PostgreSQL conversation repository unavailable: {type(exc).__name__}"
            ) from exc

    def initialize(self) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversation_sessions (
                        session_id TEXT PRIMARY KEY,
                        state_version BIGINT NOT NULL,
                        session_json JSONB NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversation_events (
                        event_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL REFERENCES conversation_sessions(session_id),
                        aggregate_version BIGINT NOT NULL,
                        event_type TEXT NOT NULL,
                        event_json JSONB NOT NULL,
                        occurred_at TIMESTAMPTZ NOT NULL,
                        UNIQUE (session_id, aggregate_version)
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS conversation_events_session_idx
                    ON conversation_events (session_id, aggregate_version)
                    """
                )

    def readiness(self) -> dict[str, object]:
        try:
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
            return {"status": "ready", "mode": "postgres"}
        except PersistenceUnavailableError as exc:
            return {"status": "not_ready", "mode": "postgres", "error": str(exc)}
        except Exception as exc:
            return {"status": "not_ready", "mode": "postgres", "error": type(exc).__name__}

    def create(self, session_id: str) -> ConversationSession:
        session = ConversationSession.create(session_id)
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO conversation_sessions
                        (session_id, state_version, session_json, created_at, updated_at)
                    VALUES (%s, %s, %s::jsonb, %s, %s)
                    ON CONFLICT (session_id) DO NOTHING
                    """,
                    (
                        session_id,
                        session.state.state_version,
                        _json_text(session),
                        session.created_at,
                        session.updated_at,
                    ),
                )
                stored = self._select_session(cursor, session_id, for_update=False)
        if stored is None:
            raise PersistenceUnavailableError("PostgreSQL session could not be created")
        return stored

    def load(self, session_id: str) -> ConversationSession | None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                return self._select_session(cursor, session_id, for_update=False)

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
        if any(event.aggregate_id != session_id for event in events):
            raise PersistenceConflictError("event aggregate does not match session")
        incoming_ids = [event.event_id for event in events]
        if len(set(incoming_ids)) != len(incoming_ids):
            raise PersistenceConflictError("event batch contains duplicate event IDs")

        with self._connect() as connection:
            with connection.cursor() as cursor:
                session = self._select_session(cursor, session_id, for_update=True)
                if session is None:
                    session = ConversationSession.create(session_id)
                    cursor.execute(
                        """
                        INSERT INTO conversation_sessions
                            (session_id, state_version, session_json, created_at, updated_at)
                        VALUES (%s, %s, %s::jsonb, %s, %s)
                        """,
                        (
                            session_id,
                            session.state.state_version,
                            _json_text(session),
                            session.created_at,
                            session.updated_at,
                        ),
                    )

                existing = self._existing_events(cursor, incoming_ids)
                if existing:
                    if len(existing) != len(events):
                        raise PersistenceConflictError(
                            "event batch partially overlaps a prior commit"
                        )
                    for event in events:
                        if existing[event.event_id] != event.model_dump(mode="json"):
                            raise PersistenceConflictError("event ID was reused with different payload")
                    return BatchCommitResult(
                        session=session,
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

                for event in events:
                    payload = event.model_dump(mode="json")
                    cursor.execute(
                        """
                        INSERT INTO conversation_events
                            (event_id, session_id, aggregate_version, event_type, event_json, occurred_at)
                        VALUES (%s, %s, %s, %s, %s::jsonb, %s)
                        """,
                        (
                            event.event_id,
                            session_id,
                            event.aggregate_version,
                            event.event_type,
                            json.dumps(payload),
                            event.occurred_at,
                        ),
                    )
                cursor.execute(
                    """
                    UPDATE conversation_sessions
                    SET state_version = %s, session_json = %s::jsonb, updated_at = %s
                    WHERE session_id = %s AND state_version = %s
                    """,
                    (
                        working_session.state.state_version,
                        _json_text(working_session),
                        working_session.updated_at,
                        session_id,
                        expected_state_version,
                    ),
                )
                if cursor.rowcount != 1:
                    raise PersistenceConflictError("conversation session changed during commit")
                return BatchCommitResult(
                    session=working_session,
                    events=tuple(events),
                )

    def events(self, session_id: str) -> list[DomainEvent]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT event_json
                    FROM conversation_events
                    WHERE session_id = %s
                    ORDER BY aggregate_version ASC
                    """,
                    (session_id,),
                )
                return [
                    self._event_adapter.validate_python(_json_value(row[0]))
                    for row in cursor.fetchall()
                ]

    @staticmethod
    def _select_session(cursor: Any, session_id: str, *, for_update: bool) -> ConversationSession | None:
        cursor.execute(
            "SELECT session_json FROM conversation_sessions WHERE session_id = %s" +
            (" FOR UPDATE" if for_update else ""),
            (session_id,),
        )
        row = cursor.fetchone()
        return ConversationSession.model_validate(_json_value(row[0])) if row else None

    @staticmethod
    def _existing_events(cursor: Any, event_ids: list[str]) -> dict[str, dict[str, object]]:
        placeholders = ", ".join("%s" for _ in event_ids)
        cursor.execute(
            f"SELECT event_id, event_json FROM conversation_events WHERE event_id IN ({placeholders})",
            tuple(event_ids),
        )
        return {event_id: _json_value(payload) for event_id, payload in cursor.fetchall()}


def _json_text(value: Any) -> str:
    return json.dumps(value.model_dump(mode="json"))


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def build_session_repository() -> SessionRepository:
    import os

    mode = os.getenv("CONVERSATION_REPOSITORY", "memory").strip().lower()
    if mode in {"memory", "inmemory"}:
        return InMemorySessionRepository()
    if mode != "postgres":
        raise ValueError(f"unsupported conversation repository: {mode}")
    dsn = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_DSN")
    if not dsn:
        raise ValueError("DATABASE_URL or POSTGRES_DSN is required for postgres persistence")
    repository = PostgresSessionRepository(dsn)
    repository.initialize()
    return repository
