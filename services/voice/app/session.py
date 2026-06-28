import asyncio
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class SessionPhase(str, Enum):
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


class VoiceSessionController:
    """Tracks turn generation and cancels in-flight LLM work on barge-in."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.phase = SessionPhase.LISTENING
        self.turn_seq = 0
        self.active_turn_id: int | None = None
        self._active_task: asyncio.Task | None = None

    def interrupt(self) -> int:
        """Invalidate the current turn and cancel any in-flight conversation call."""
        self.turn_seq += 1
        cancelled_turn_id = self.active_turn_id
        self.active_turn_id = None
        self.phase = SessionPhase.LISTENING
        self._cancel_active_task()
        if cancelled_turn_id is not None:
            logger.info(
                "turn_cancelled session=%s turn_id=%s phase=%s",
                self.session_id,
                cancelled_turn_id,
                SessionPhase.THINKING.value,
            )
        return self.turn_seq

    def begin_thinking(self) -> int:
        """Start a new turn, cancelling any previous in-flight work."""
        self._cancel_active_task()
        self.turn_seq += 1
        turn_id = self.turn_seq
        self.active_turn_id = turn_id
        self.phase = SessionPhase.THINKING
        return turn_id

    def begin_speaking(self) -> None:
        self.phase = SessionPhase.SPEAKING

    def return_to_listening(self) -> None:
        self.phase = SessionPhase.LISTENING

    def is_turn_active(self, turn_id: int) -> bool:
        return self.active_turn_id == turn_id

    def set_active_task(self, task: asyncio.Task) -> None:
        self._active_task = task

    def _cancel_active_task(self) -> None:
        if self._active_task and not self._active_task.done():
            self._active_task.cancel()
        self._active_task = None
