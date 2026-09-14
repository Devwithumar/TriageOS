"""Transactional orchestration for the replacement conversation pipeline."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol
from uuid import uuid4

from libs.ai.proposal_adapter import ProposalCompletion, ProposalAdapterError
from libs.conversation.domain import (
    Channel,
    ConversationSession,
    ConversationState,
    DomainEvent,
    DomainInvariantError,
    TurnReceivedEvent,
    command_to_event,
    reduce_state,
)
from libs.conversation.persistence import BatchCommitResult, SessionRepository
from libs.conversation.proposals import (
    ConversationProposal,
    ProposalValidationError,
    ValidatedProposal,
    validate_proposal,
)


class ProposalSource(Protocol):
    def propose(
        self,
        user_text: str,
        state: ConversationState,
        recent_messages: list[dict[str, str]],
        correlation_id: str,
    ) -> ProposalCompletion:
        ...


@dataclass(frozen=True)
class OrchestrationResult:
    session: ConversationSession
    events: tuple[DomainEvent, ...]
    turn_event: TurnReceivedEvent
    proposal: ConversationProposal | None = None
    validated_proposal: ValidatedProposal | None = None
    proposal_completion: ProposalCompletion | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


class ConversationOrchestrator:
    """Run one user turn against a session with optimistic concurrency."""

    def __init__(self, repository: SessionRepository, proposal_source: ProposalSource) -> None:
        self._repository = repository
        self._proposal_source = proposal_source

    def process_turn(
        self,
        session_id: str,
        user_text: str,
        *,
        channel: Channel = Channel.TEXT,
        recent_messages: list[dict[str, str]] | None = None,
        correlation_id: str | None = None,
    ) -> OrchestrationResult:
        normalized_text = user_text.strip()
        if not normalized_text:
            raise ValueError("user turn cannot be empty")
        base_session = self._repository.load(session_id)
        if base_session is None:
            base_session = self._repository.create(session_id)
        correlation = correlation_id or str(uuid4())
        turn_event = TurnReceivedEvent(
            aggregate_id=session_id,
            aggregate_version=base_session.state.state_version + 1,
            correlation_id=correlation,
            text=normalized_text,
            channel=channel,
            occurred_at=datetime.now(timezone.utc),
        )
        turn_state = reduce_state(base_session.state, turn_event)
        history = list(recent_messages or [])
        try:
            completion = self._proposal_source.propose(
                normalized_text,
                turn_state,
                history,
                correlation,
            )
            validated = validate_proposal(turn_state, completion.proposal)
            events = self._build_events(turn_state, turn_event, validated)
            committed = self._repository.commit_batch(
                session_id,
                base_session.state.state_version,
                events,
            )
            return OrchestrationResult(
                session=committed.session,
                events=committed.events,
                turn_event=turn_event,
                proposal=completion.proposal,
                validated_proposal=validated,
                proposal_completion=completion,
            )
        except (ProposalAdapterError, ProposalValidationError, DomainInvariantError) as exc:
            committed = self._commit_turn_only(session_id, base_session, turn_event)
            return OrchestrationResult(
                session=committed.session,
                events=committed.events,
                turn_event=turn_event,
                error=str(exc),
            )

    @staticmethod
    def _build_events(
        turn_state: ConversationState,
        turn_event: TurnReceivedEvent,
        validated: ValidatedProposal,
    ) -> list[DomainEvent]:
        events: list[DomainEvent] = [turn_event]
        working_state = turn_state
        for command in validated.commands:
            event = command_to_event(working_state, command)
            events.append(event)
            working_state = reduce_state(working_state, event)
        return events

    def _commit_turn_only(
        self,
        session_id: str,
        base_session: ConversationSession,
        turn_event: TurnReceivedEvent,
    ) -> BatchCommitResult:
        return self._repository.commit_batch(
            session_id,
            base_session.state.state_version,
            [turn_event],
        )
