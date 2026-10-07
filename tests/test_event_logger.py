"""Tests for session/event_logger.py.

Verifies session creation, updating, event logging, data validation,
JSON serialization/deserialization, and session isolation.
"""

import pytest
from datetime import datetime, timezone
from pathlib import Path

from behavior.behavior_engine import BehaviorState, BehaviorReason
from database.database import Database
from session.event_logger import EventLogger


@pytest.fixture
def memory_logger() -> EventLogger:
    """Fixture providing an isolated in-memory EventLogger."""
    db = Database(db_path=":memory:")
    logger = EventLogger(db=db)
    logger.initialize_database()
    return logger


class TestEventLoggerSessionLifecycle:
    def test_create_session(self, memory_logger: EventLogger):
        now_iso = datetime.now(timezone.utc).isoformat()
        session_id = memory_logger.create_session(start_time=now_iso)
        assert session_id > 0

        session = memory_logger.get_session(session_id)
        assert session is not None
        assert session["id"] == session_id
        assert session["start_time"] == now_iso
        assert session["status"] == "ACTIVE"
        assert session["duration_seconds"] == 0.0
        assert session["end_time"] is None

    def test_create_session_invalid_status(self, memory_logger: EventLogger):
        now_iso = datetime.now(timezone.utc).isoformat()
        with pytest.raises(ValueError, match="Invalid status"):
            memory_logger.create_session(start_time=now_iso, status="INVALID")

    def test_create_session_invalid_timestamp(self, memory_logger: EventLogger):
        with pytest.raises(ValueError, match="not a valid ISO 8601"):
            memory_logger.create_session(start_time="not-a-timestamp")

    def test_end_session(self, memory_logger: EventLogger):
        start_iso = "2026-10-06T10:00:00+00:00"
        end_iso = "2026-10-06T10:30:00+00:00"
        session_id = memory_logger.create_session(start_time=start_iso)

        updated = memory_logger.end_session(
            session_id=session_id,
            end_time=end_iso,
            duration_seconds=1800.0,
            status="COMPLETED",
        )
        assert updated is True

        session = memory_logger.get_session(session_id)
        assert session is not None
        assert session["end_time"] == end_iso
        assert session["duration_seconds"] == 1800.0
        assert session["status"] == "COMPLETED"

    def test_end_session_invalid_duration(self, memory_logger: EventLogger):
        start_iso = "2026-10-06T10:00:00+00:00"
        session_id = memory_logger.create_session(start_time=start_iso)
        with pytest.raises(ValueError, match="duration_seconds must be >= 0.0"):
            memory_logger.end_session(
                session_id=session_id,
                end_time="2026-10-06T10:30:00+00:00",
                duration_seconds=-10.0,
            )

    def test_end_nonexistent_session(self, memory_logger: EventLogger):
        updated = memory_logger.end_session(
            session_id=999,
            end_time="2026-10-06T10:30:00+00:00",
            duration_seconds=50.0,
        )
        assert updated is False


class TestEventLoggerBehaviorEvents:
    def test_log_behavior_event_with_signals(self, memory_logger: EventLogger):
        session_id = memory_logger.create_session(start_time="2026-10-06T10:00:00+00:00")
        signals = {
            "face_detected": True,
            "eye_state": "OPEN",
            "smoothed_ear": 0.28,
            "head_direction": "CENTER",
            "yaw": 1.2,
            "pitch": -0.5,
            "phone_detected": False,
            "phone_confidence": None,
        }

        event_id = memory_logger.log_behavior_event(
            session_id=session_id,
            timestamp="2026-10-06T10:00:05+00:00",
            state=BehaviorState.FOCUSED,
            state_duration=5.0,
            confidence=0.92,
            reason=BehaviorReason.FOCUSED,
            signals=signals,
        )
        assert event_id > 0

        events = memory_logger.get_session_events(session_id)
        assert len(events) == 1
        ev = events[0]
        assert ev["id"] == event_id
        assert ev["session_id"] == session_id
        assert ev["state"] == "FOCUSED"
        assert ev["state_duration"] == 5.0
        assert ev["confidence"] == 0.92
        assert ev["reason"] == BehaviorReason.FOCUSED
        assert ev["signals"]["face_detected"] is True
        assert ev["signals"]["smoothed_ear"] == 0.28

    def test_log_event_all_states(self, memory_logger: EventLogger):
        session_id = memory_logger.create_session(start_time="2026-10-06T10:00:00+00:00")
        states = [
            BehaviorState.FOCUSED,
            BehaviorState.DISTRACTED,
            BehaviorState.DROWSY,
            BehaviorState.PHONE_DETECTED,
            BehaviorState.AWAY_FROM_DESK,
            BehaviorState.NO_FACE,
            BehaviorState.UNKNOWN,
        ]
        for idx, st in enumerate(states):
            eid = memory_logger.log_behavior_event(
                session_id=session_id,
                timestamp=f"2026-10-06T10:0{idx}:00+00:00",
                state=st,
                state_duration=float(idx),
                confidence=0.85,
                reason="test_reason",
            )
            assert eid > 0

        events = memory_logger.get_session_events(session_id)
        assert len(events) == len(states)
        logged_states = [e["state"] for e in events]
        expected_states = [s.value for s in states]
        assert logged_states == expected_states

    def test_log_event_invalid_confidence(self, memory_logger: EventLogger):
        session_id = memory_logger.create_session(start_time="2026-10-06T10:00:00+00:00")
        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            memory_logger.log_behavior_event(
                session_id=session_id,
                timestamp="2026-10-06T10:00:05+00:00",
                state=BehaviorState.FOCUSED,
                state_duration=1.0,
                confidence=1.5,
                reason="focused",
            )

    def test_log_event_invalid_state(self, memory_logger: EventLogger):
        session_id = memory_logger.create_session(start_time="2026-10-06T10:00:00+00:00")
        with pytest.raises(ValueError, match="Invalid behavior state"):
            memory_logger.log_behavior_event(
                session_id=session_id,
                timestamp="2026-10-06T10:00:05+00:00",
                state="NOT_A_VALID_STATE",
                state_duration=1.0,
                confidence=0.5,
                reason="focused",
            )

    def test_log_event_empty_reason(self, memory_logger: EventLogger):
        session_id = memory_logger.create_session(start_time="2026-10-06T10:00:00+00:00")
        with pytest.raises(ValueError, match="reason must be a non-empty string"):
            memory_logger.log_behavior_event(
                session_id=session_id,
                timestamp="2026-10-06T10:00:05+00:00",
                state=BehaviorState.FOCUSED,
                state_duration=1.0,
                confidence=0.5,
                reason="   ",
            )

    def test_log_event_nonexistent_session(self, memory_logger: EventLogger):
        with pytest.raises(ValueError, match="does not exist"):
            memory_logger.log_behavior_event(
                session_id=9999,
                timestamp="2026-10-06T10:00:05+00:00",
                state=BehaviorState.FOCUSED,
                state_duration=1.0,
                confidence=0.5,
                reason="focused",
            )


class TestEventLoggerIsolationAndQueries:
    def test_session_events_isolation(self, memory_logger: EventLogger):
        s1 = memory_logger.create_session(start_time="2026-10-06T10:00:00+00:00")
        s2 = memory_logger.create_session(start_time="2026-10-06T11:00:00+00:00")

        memory_logger.log_behavior_event(
            session_id=s1,
            timestamp="2026-10-06T10:01:00+00:00",
            state=BehaviorState.FOCUSED,
            state_duration=0.0,
            confidence=0.9,
            reason="focused",
        )
        memory_logger.log_behavior_event(
            session_id=s2,
            timestamp="2026-10-06T11:01:00+00:00",
            state=BehaviorState.DISTRACTED,
            state_duration=0.0,
            confidence=0.8,
            reason="head_away",
        )

        events_s1 = memory_logger.get_session_events(s1)
        events_s2 = memory_logger.get_session_events(s2)

        assert len(events_s1) == 1
        assert len(events_s2) == 1
        assert events_s1[0]["state"] == "FOCUSED"
        assert events_s2[0]["state"] == "DISTRACTED"

    def test_get_recent_sessions(self, memory_logger: EventLogger):
        for i in range(5):
            memory_logger.create_session(start_time=f"2026-10-06T1{i}:00:00+00:00")

        recent = memory_logger.get_recent_sessions(limit=3)
        assert len(recent) == 3
        # Should be newest first (id descending)
        assert recent[0]["id"] > recent[1]["id"] > recent[2]["id"]

    def test_get_completed_sessions(self, memory_logger: EventLogger):
        s1 = memory_logger.create_session(start_time="2026-10-06T10:00:00+00:00")
        s2 = memory_logger.create_session(start_time="2026-10-06T11:00:00+00:00")
        memory_logger.end_session(s1, end_time="2026-10-06T10:30:00+00:00", duration_seconds=1800.0)
        # s2 remains ACTIVE

        completed = memory_logger.get_completed_sessions()
        assert len(completed) == 1
        assert completed[0]["id"] == s1

    def test_recover_unfinished_sessions_with_events(self, memory_logger: EventLogger):
        start_ts = "2026-10-06T10:00:00+00:00"
        ev1_ts = "2026-10-06T10:05:00+00:00"
        ev2_ts = "2026-10-06T10:12:00+00:00"

        sid = memory_logger.create_session(start_time=start_ts, status="ACTIVE")
        memory_logger.log_behavior_event(
            session_id=sid,
            timestamp=ev1_ts,
            state=BehaviorState.FOCUSED,
            state_duration=300.0,
            confidence=0.9,
            reason="focused",
        )
        memory_logger.log_behavior_event(
            session_id=sid,
            timestamp=ev2_ts,
            state=BehaviorState.DISTRACTED,
            state_duration=420.0,
            confidence=0.85,
            reason="looking_away",
        )

        # Simulate crash recovery
        recovered = memory_logger.recover_unfinished_sessions(fallback_status="CANCELLED")
        assert recovered == [sid]

        session = memory_logger.get_session(sid)
        assert session["status"] == "CANCELLED"
        assert session["end_time"] == ev2_ts
        assert session["duration_seconds"] == 720.0  # 12 minutes = 720 seconds

        # Events preserved
        events = memory_logger.get_session_events(sid)
        assert len(events) == 2

    def test_recover_unfinished_sessions_without_events(self, memory_logger: EventLogger):
        start_ts = "2026-10-06T14:00:00+00:00"
        sid = memory_logger.create_session(start_time=start_ts, status="ACTIVE")

        recovered = memory_logger.recover_unfinished_sessions()
        assert recovered == [sid]

        session = memory_logger.get_session(sid)
        assert session["status"] == "CANCELLED"
        assert session["end_time"] == start_ts
        assert session["duration_seconds"] == 0.0

    def test_recover_unfinished_sessions_ignores_completed(self, memory_logger: EventLogger):
        start_ts = "2026-10-06T15:00:00+00:00"
        sid = memory_logger.create_session(start_time=start_ts, status="ACTIVE")
        memory_logger.end_session(sid, end_time="2026-10-06T15:30:00+00:00", duration_seconds=1800.0)

        recovered = memory_logger.recover_unfinished_sessions()
        assert recovered == []
        session = memory_logger.get_session(sid)
        assert session["status"] == "COMPLETED"
