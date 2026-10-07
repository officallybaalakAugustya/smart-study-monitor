"""Session Manager module for Smart Study Monitor.

Manages the lifecycle of an active study session, tracks runtime elapsed
duration using a monotonic clock, and logs behavioral events upon confirmed
state transitions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Callable, Optional

from behavior.behavior_engine import BehaviorResult, BehaviorState
from session.event_logger import EventLogger


class SessionStatus(str, Enum):
    """Lifecycle statuses for a study session."""

    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


@dataclass
class Session:
    """Represents a study monitoring session entity.

    Attributes:
        session_id: Unique database primary key.
        start_time: ISO 8601 wall-clock timestamp of session start.
        end_time: ISO 8601 wall-clock timestamp of session end, or None if active.
        duration_seconds: Total elapsed runtime in seconds.
        status: Current session status ("ACTIVE", "COMPLETED", "CANCELLED").
        created_at: ISO 8601 wall-clock timestamp when the session row was created.
    """

    session_id: int
    start_time: str
    end_time: Optional[str] = None
    duration_seconds: float = 0.0
    status: str = SessionStatus.ACTIVE.value
    created_at: Optional[str] = None


class SessionManager:
    """Coordinates study session lifecycle and behavioral event recording.

    Design invariants:
        - Only ONE active session is managed at a time.
        - Runtime elapsed duration is calculated exclusively via a monotonic clock.
        - Database timestamps use timezone-aware ISO 8601 wall-clock time.
        - Behavioral events are logged ONLY on state transitions (plus the first
          confirmed state of the session). No per-frame duplicate database insertions.
        - SessionManager never recalculates or inspects lower-level CV signals;
          BehaviorResult is the single source of truth.
    """

    def __init__(
        self,
        event_logger: Optional[EventLogger] = None,
        clock: Optional[Callable[[], float]] = None,
        wall_clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        """Initializes the SessionManager.

        Args:
            event_logger: EventLogger persistence instance. If None, creates a default.
            clock: Monotonic clock callable returning seconds (defaults to time.monotonic).
            wall_clock: Wall-clock callable returning datetime (defaults to local timezone-aware now).
        """
        self._logger = event_logger or EventLogger()
        self._clock = clock or time.monotonic
        self._wall_clock = wall_clock or (lambda: datetime.now().astimezone())

        self._current_session: Optional[Session] = None
        self._start_monotonic: float = 0.0
        self._last_logged_state: Optional[BehaviorState] = None
        self._total_transitions: int = 0

    @property
    def event_logger(self) -> EventLogger:
        """Returns the underlying EventLogger."""
        return self._logger

    def start_session(self) -> Session:
        """Starts a new study session.

        If a session is already active, safely concludes it first.

        Returns:
            The newly created Session object.
        """
        if self.is_active():
            self.end_session(status=SessionStatus.COMPLETED.value)
        else:
            self.recover_unfinished_sessions()

        wall_now = self._wall_clock()
        start_time_iso = wall_now.isoformat()
        mono_now = self._clock()

        session_id = self._logger.create_session(
            start_time=start_time_iso,
            status=SessionStatus.ACTIVE.value,
        )

        session = Session(
            session_id=session_id,
            start_time=start_time_iso,
            status=SessionStatus.ACTIVE.value,
            created_at=start_time_iso,
        )

        self._current_session = session
        self._start_monotonic = mono_now
        self._last_logged_state = None
        self._total_transitions = 0

        return session

    def update(self, behavior_result: BehaviorResult) -> Optional[int]:
        """Receives the latest BehaviorResult and logs events on state transitions.

        Rules:
            1. If session is not active, does nothing and returns None.
            2. If this is the first BehaviorResult of the session, records the initial state.
            3. If behavior_result.state != last_logged_state, logs a transition event.
            4. If behavior_result.state == last_logged_state, ignores to avoid duplicate writes.

        Args:
            behavior_result: Current BehaviorResult from BehaviorEngine.

        Returns:
            Event ID if an event was persisted, None otherwise.
        """
        if not self.is_active() or self._current_session is None:
            return None

        current_state = behavior_result.state
        session_id = self._current_session.session_id

        # Rule 1: First event of the session
        if self._last_logged_state is None:
            now_iso = self._wall_clock().isoformat()
            event_id = self._logger.log_behavior_event(
                session_id=session_id,
                timestamp=now_iso,
                state=current_state,
                state_duration=behavior_result.state_duration,
                confidence=behavior_result.confidence,
                reason=behavior_result.reason,
                signals=behavior_result.signals,
            )
            self._last_logged_state = current_state
            return event_id

        # Rule 2: State transition occurred
        if current_state != self._last_logged_state:
            now_iso = self._wall_clock().isoformat()
            event_id = self._logger.log_behavior_event(
                session_id=session_id,
                timestamp=now_iso,
                state=current_state,
                state_duration=behavior_result.state_duration,
                confidence=behavior_result.confidence,
                reason=behavior_result.reason,
                signals=behavior_result.signals,
            )
            self._last_logged_state = current_state
            self._total_transitions += 1
            return event_id

        # Rule 3: Steady state - do not write duplicate database rows
        return None

    def end_session(self, status: str = SessionStatus.COMPLETED.value) -> Optional[Session]:
        """Ends the active study session and persists final duration and status.

        Repeated calls when no session is active are safe and return None.

        Args:
            status: Final status ("COMPLETED" or "CANCELLED").

        Returns:
            The finalized Session object, or None if no session was active.
        """
        if not self.is_active() or self._current_session is None:
            return None

        end_wall_iso = self._wall_clock().isoformat()
        final_duration = max(0.0, self._clock() - self._start_monotonic)

        session_id = self._current_session.session_id
        self._logger.end_session(
            session_id=session_id,
            end_time=end_wall_iso,
            duration_seconds=final_duration,
            status=status,
        )

        completed_session = Session(
            session_id=session_id,
            start_time=self._current_session.start_time,
            end_time=end_wall_iso,
            duration_seconds=final_duration,
            status=status,
            created_at=self._current_session.created_at,
        )

        self._current_session = None
        self._last_logged_state = None
        self._start_monotonic = 0.0

        return completed_session

    def is_active(self) -> bool:
        """Returns True if there is currently an active study session."""
        return self._current_session is not None and self._current_session.status == SessionStatus.ACTIVE.value

    def get_elapsed_time(self) -> float:
        """Returns the current elapsed duration in seconds using the monotonic clock.

        Accurate regardless of FPS, frame drops, or loop speed.
        """
        if not self.is_active():
            return 0.0
        return max(0.0, self._clock() - self._start_monotonic)

    def get_current_session(self) -> Optional[Session]:
        """Returns a copy of the current active session, or None."""
        if self._current_session is None:
            return None
        return Session(
            session_id=self._current_session.session_id,
            start_time=self._current_session.start_time,
            end_time=self._current_session.end_time,
            duration_seconds=self.get_elapsed_time(),
            status=self._current_session.status,
            created_at=self._current_session.created_at,
        )

    def get_total_transitions(self) -> int:
        """Returns the total number of behavioral state transitions recorded."""
        return self._total_transitions

    def get_last_logged_state(self) -> Optional[BehaviorState]:
        """Returns the most recently logged behavior state."""
        return self._last_logged_state

    def recover_unfinished_sessions(self) -> list[int]:
        """Recovers any unfinished ACTIVE sessions lingering in the database from prior crashes."""
        return self._logger.recover_unfinished_sessions()

    def close(self) -> None:
        """Safely closes active resources."""
        if self.is_active():
            self.end_session(status=SessionStatus.COMPLETED.value)
        self._logger.close()
