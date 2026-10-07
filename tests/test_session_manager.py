"""Tests for session/session_manager.py.

Verifies session lifecycle, monotonic timer accuracy, event logging on state
transitions only (no per-frame duplicate events), clock injection, and persistence.
"""

from datetime import datetime, timezone
import pytest

from behavior.behavior_engine import BehaviorResult, BehaviorState, BehaviorReason
from database.database import Database
from session.event_logger import EventLogger
from session.session_manager import SessionManager, SessionStatus


class MockClock:
    """Mock monotonic clock for deterministic time progression testing."""

    def __init__(self, start_time: float = 1000.0) -> None:
        self.current_time = start_time

    def __call__(self) -> float:
        return self.current_time

    def advance(self, seconds: float) -> None:
        self.current_time += seconds


class MockWallClock:
    """Mock wall clock for deterministic ISO timestamps."""

    def __init__(self) -> None:
        self.current_time = datetime(2026, 10, 6, 12, 0, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.current_time

    def advance(self, seconds: float) -> None:
        from datetime import timedelta
        self.current_time += timedelta(seconds=seconds)


@pytest.fixture
def mock_session_env():
    """Provides a SessionManager with in-memory DB and injectable clocks."""
    db = Database(db_path=":memory:")
    logger = EventLogger(db=db)
    logger.initialize_database()
    clock = MockClock(start_time=100.0)
    wall_clock = MockWallClock()
    manager = SessionManager(event_logger=logger, clock=clock, wall_clock=wall_clock)
    return manager, logger, clock, wall_clock


class TestSessionLifecycle:
    def test_start_session(self, mock_session_env):
        manager, logger, clock, wall_clock = mock_session_env
        assert not manager.is_active()

        session = manager.start_session()
        assert manager.is_active()
        assert session.session_id > 0
        assert session.status == SessionStatus.ACTIVE.value
        assert session.start_time == "2026-10-06T12:00:00+00:00"
        assert manager.get_elapsed_time() == 0.0

    def test_elapsed_time_monotonic_independent_of_frames(self, mock_session_env):
        manager, logger, clock, wall_clock = mock_session_env
        manager.start_session()

        # Advance 20 minutes (1200 seconds) without any frame calls
        clock.advance(1200.0)
        assert manager.get_elapsed_time() == 1200.0

        current_session = manager.get_current_session()
        assert current_session is not None
        assert current_session.duration_seconds == 1200.0

    def test_end_session(self, mock_session_env):
        manager, logger, clock, wall_clock = mock_session_env
        session = manager.start_session()
        clock.advance(300.0)
        wall_clock.advance(300.0)

        ended = manager.end_session(status=SessionStatus.COMPLETED.value)
        assert ended is not None
        assert ended.session_id == session.session_id
        assert ended.status == SessionStatus.COMPLETED.value
        assert ended.duration_seconds == 300.0
        assert ended.end_time == "2026-10-06T12:05:00+00:00"
        assert not manager.is_active()
        assert manager.get_elapsed_time() == 0.0

        # Verify persisted state in database
        persisted = logger.get_session(session.session_id)
        assert persisted is not None
        assert persisted["status"] == "COMPLETED"
        assert persisted["duration_seconds"] == 300.0
        assert persisted["end_time"] == "2026-10-06T12:05:00+00:00"

    def test_safe_repeated_end_session(self, mock_session_env):
        manager, logger, clock, wall_clock = mock_session_env
        manager.start_session()
        assert manager.end_session() is not None
        # Second call returns None and does not error
        assert manager.end_session() is None

    def test_start_session_closes_previous_active(self, mock_session_env):
        manager, logger, clock, wall_clock = mock_session_env
        s1 = manager.start_session()
        clock.advance(50.0)
        s2 = manager.start_session()

        assert s1.session_id != s2.session_id
        # Previous session should be marked COMPLETED
        s1_row = logger.get_session(s1.session_id)
        assert s1_row["status"] == "COMPLETED"
        assert s1_row["duration_seconds"] == 50.0


class TestEventLoggingTransitions:
    def test_first_behavior_result_logged(self, mock_session_env):
        manager, logger, clock, wall_clock = mock_session_env
        session = manager.start_session()

        result = BehaviorResult(
            state=BehaviorState.FOCUSED,
            confidence=0.95,
            reason=BehaviorReason.FOCUSED,
            signals={"face_detected": True},
        )

        event_id = manager.update(result)
        assert event_id is not None
        assert manager.get_last_logged_state() == BehaviorState.FOCUSED
        assert manager.get_total_transitions() == 0

        events = logger.get_session_events(session.session_id)
        assert len(events) == 1
        assert events[0]["state"] == "FOCUSED"

    def test_steady_state_does_not_duplicate_events(self, mock_session_env):
        manager, logger, clock, wall_clock = mock_session_env
        session = manager.start_session()

        result = BehaviorResult(
            state=BehaviorState.FOCUSED,
            confidence=0.90,
            reason=BehaviorReason.FOCUSED,
        )

        # First frame logs initial state
        assert manager.update(result) is not None

        # Simulate 100 consecutive frames in the same state
        for _ in range(100):
            clock.advance(0.033)
            event_id = manager.update(result)
            assert event_id is None

        # DB must still contain only 1 event
        events = logger.get_session_events(session.session_id)
        assert len(events) == 1

    def test_state_transition_logging(self, mock_session_env):
        manager, logger, clock, wall_clock = mock_session_env
        session = manager.start_session()

        # 1. First event: FOCUSED
        res_focused = BehaviorResult(
            state=BehaviorState.FOCUSED,
            confidence=0.95,
            reason=BehaviorReason.FOCUSED,
        )
        assert manager.update(res_focused) is not None

        # 2. Stay FOCUSED for 50 frames (0 events)
        for _ in range(50):
            clock.advance(0.033)
            assert manager.update(res_focused) is None

        # 3. Transition to DISTRACTED (1 event)
        res_distracted = BehaviorResult(
            state=BehaviorState.DISTRACTED,
            confidence=0.85,
            reason=BehaviorReason.HEAD_AWAY,
        )
        assert manager.update(res_distracted) is not None
        assert manager.get_total_transitions() == 1

        # 4. Stay DISTRACTED for 30 frames (0 events)
        for _ in range(30):
            clock.advance(0.033)
            assert manager.update(res_distracted) is None

        # 5. Transition to PHONE_DETECTED (1 event)
        res_phone = BehaviorResult(
            state=BehaviorState.PHONE_DETECTED,
            confidence=0.88,
            reason=BehaviorReason.PHONE_DETECTED,
        )
        assert manager.update(res_phone) is not None
        assert manager.get_total_transitions() == 2

        # 6. Return to FOCUSED (1 event)
        assert manager.update(res_focused) is not None
        assert manager.get_total_transitions() == 3

        # Check all logged events in order
        events = logger.get_session_events(session.session_id)
        assert len(events) == 4
        states_recorded = [e["state"] for e in events]
        assert states_recorded == ["FOCUSED", "DISTRACTED", "PHONE_DETECTED", "FOCUSED"]

    def test_all_behavior_states_transitions(self, mock_session_env):
        manager, logger, clock, wall_clock = mock_session_env
        session = manager.start_session()

        all_states = [
            (BehaviorState.UNKNOWN, "insufficient_signal"),
            (BehaviorState.NO_FACE, "face_not_detected"),
            (BehaviorState.FOCUSED, "focused"),
            (BehaviorState.DISTRACTED, "head_away"),
            (BehaviorState.DROWSY, "eyes_closed"),
            (BehaviorState.PHONE_DETECTED, "phone_detected"),
            (BehaviorState.AWAY_FROM_DESK, "head_away"),
        ]

        for state, reason in all_states:
            res = BehaviorResult(state=state, confidence=0.8, reason=reason)
            eid = manager.update(res)
            assert eid is not None

        events = logger.get_session_events(session.session_id)
        assert len(events) == len(all_states)
        for logged, (exp_state, _) in zip(events, all_states):
            assert logged["state"] == exp_state.value

    def test_update_when_inactive_returns_none(self, mock_session_env):
        manager, logger, clock, wall_clock = mock_session_env
        res = BehaviorResult(state=BehaviorState.FOCUSED)
        assert manager.update(res) is None

    def test_start_session_recovers_lingering_crashed_session(self, mock_session_env):
        manager, logger, clock, wall_clock = mock_session_env

        # Simulate prior session that remained ACTIVE due to crash
        crashed_id = logger.create_session(start_time="2026-10-06T08:00:00+00:00", status="ACTIVE")
        logger.log_behavior_event(
            session_id=crashed_id,
            timestamp="2026-10-06T08:20:00+00:00",
            state=BehaviorState.FOCUSED,
            state_duration=1200.0,
            confidence=0.9,
            reason="focused",
        )

        # Starting a new session triggers recovery of the lingering crashed session
        new_session = manager.start_session()
        assert new_session.session_id != crashed_id
        assert manager.is_active()

        # Check recovered status of the crashed session
        crashed_row = logger.get_session(crashed_id)
        assert crashed_row["status"] == "CANCELLED"
        assert crashed_row["end_time"] == "2026-10-06T08:20:00+00:00"
        assert crashed_row["duration_seconds"] == 1200.0

