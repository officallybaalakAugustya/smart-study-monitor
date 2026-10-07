"""Tests for analytics/session_analytics.py.

Verifies timeline reconstruction, final state extension to end_time,
chronological sorting of out-of-order events, transition counting,
cross-session isolation, and synthetic integration sessions.
"""

from datetime import datetime, timezone
import pytest

from analytics.focus_score import FocusScoreCalculator
from analytics.session_analytics import SessionAnalytics
from behavior.behavior_engine import BehaviorState, BehaviorReason
from database.database import Database
from session.event_logger import EventLogger


@pytest.fixture
def analytics_env():
    """Provides isolated SessionAnalytics and EventLogger on in-memory DB."""
    db = Database(db_path=":memory:")
    logger = EventLogger(db=db)
    logger.initialize_database()
    analytics = SessionAnalytics(event_logger=logger)
    return analytics, logger


class TestTimelineReconstruction:
    def test_single_event_extends_to_session_end(self, analytics_env):
        # TEST 1 equivalent: 60 minutes session with only FOCUSED
        analytics, logger = analytics_env
        s_id = logger.create_session(start_time="2026-10-07T10:00:00+00:00")
        logger.log_behavior_event(
            session_id=s_id,
            timestamp="2026-10-07T10:00:00+00:00",
            state=BehaviorState.FOCUSED,
            state_duration=0.0,
            confidence=0.9,
            reason=BehaviorReason.FOCUSED,
        )
        logger.end_session(
            session_id=s_id,
            end_time="2026-10-07T11:00:00+00:00",
            duration_seconds=3600.0,
        )

        durations = analytics.get_state_durations(s_id)
        assert durations.focused_seconds == 3600.0
        assert durations.total_seconds == 3600.0

        summary = analytics.get_session_summary(s_id)
        assert summary is not None
        assert summary.focus_score == 100.0
        assert summary.transition_count == 0

    def test_focused_distracted_focused(self, analytics_env):
        # TEST 6 & TEST 7: FOCUSED -> DISTRACTED -> FOCUSED
        # 10:00 - 10:25 FOCUSED (25m = 1500s)
        # 10:25 - 10:27 DISTRACTED (2m = 120s)
        # 10:27 - 11:00 FOCUSED (33m = 1980s) - extends to end_time
        analytics, logger = analytics_env
        s_id = logger.create_session(start_time="2026-10-07T10:00:00+00:00")

        logger.log_behavior_event(
            session_id=s_id,
            timestamp="2026-10-07T10:00:00+00:00",
            state=BehaviorState.FOCUSED,
            state_duration=0.0,
            confidence=0.9,
            reason=BehaviorReason.FOCUSED,
        )
        logger.log_behavior_event(
            session_id=s_id,
            timestamp="2026-10-07T10:25:00+00:00",
            state=BehaviorState.DISTRACTED,
            state_duration=0.0,
            confidence=0.85,
            reason=BehaviorReason.HEAD_AWAY,
        )
        logger.log_behavior_event(
            session_id=s_id,
            timestamp="2026-10-07T10:27:00+00:00",
            state=BehaviorState.FOCUSED,
            state_duration=0.0,
            confidence=0.9,
            reason=BehaviorReason.FOCUSED,
        )
        logger.end_session(
            session_id=s_id,
            end_time="2026-10-07T11:00:00+00:00",
            duration_seconds=3600.0,
        )

        durations = analytics.get_state_durations(s_id)
        assert durations.focused_seconds == 1500.0 + 1980.0  # 3480.0
        assert durations.distracted_seconds == 120.0
        assert durations.total_seconds == 3600.0

        assert analytics.get_transition_count(s_id) == 2

    def test_out_of_order_events_sorted_chronologically(self, analytics_env):
        # TEST 8: Events inserted with out-of-order timestamps
        analytics, logger = analytics_env
        s_id = logger.create_session(start_time="2026-10-07T10:00:00+00:00")

        # Insert 10:10 first, then 10:00
        logger.log_behavior_event(
            session_id=s_id,
            timestamp="2026-10-07T10:10:00+00:00",
            state=BehaviorState.DISTRACTED,
            state_duration=0.0,
            confidence=0.8,
            reason=BehaviorReason.HEAD_AWAY,
        )
        logger.log_behavior_event(
            session_id=s_id,
            timestamp="2026-10-07T10:00:00+00:00",
            state=BehaviorState.FOCUSED,
            state_duration=0.0,
            confidence=0.9,
            reason=BehaviorReason.FOCUSED,
        )
        logger.end_session(
            session_id=s_id,
            end_time="2026-10-07T10:15:00+00:00",
            duration_seconds=900.0,
        )

        durations = analytics.get_state_durations(s_id)
        # 10:00 to 10:10 FOCUSED (600s)
        # 10:10 to 10:15 DISTRACTED (300s)
        assert durations.focused_seconds == 600.0
        assert durations.distracted_seconds == 300.0
        assert durations.total_seconds == 900.0

    def test_multi_session_isolation(self, analytics_env):
        # TEST 9: Session 1 and Session 2 isolation
        analytics, logger = analytics_env
        s1 = logger.create_session(start_time="2026-10-07T10:00:00+00:00")
        s2 = logger.create_session(start_time="2026-10-07T11:00:00+00:00")

        logger.log_behavior_event(
            session_id=s1,
            timestamp="2026-10-07T10:00:00+00:00",
            state=BehaviorState.FOCUSED,
            state_duration=0.0,
            confidence=0.9,
            reason="focused",
        )
        logger.end_session(s1, end_time="2026-10-07T10:30:00+00:00", duration_seconds=1800.0)

        logger.log_behavior_event(
            session_id=s2,
            timestamp="2026-10-07T11:00:00+00:00",
            state=BehaviorState.DISTRACTED,
            state_duration=0.0,
            confidence=0.8,
            reason="head_away",
        )
        logger.end_session(s2, end_time="2026-10-07T11:30:00+00:00", duration_seconds=1800.0)

        sum1 = analytics.get_session_summary(s1)
        sum2 = analytics.get_session_summary(s2)

        assert sum1.focused_seconds == 1800.0
        assert sum1.distracted_seconds == 0.0
        assert sum1.focus_score == 100.0

        assert sum2.focused_seconds == 0.0
        assert sum2.distracted_seconds == 1800.0
        assert sum2.focus_score == 0.0

    def test_transition_count(self, analytics_env):
        # TEST 15: Transition count rule
        # 5 events -> 4 transitions
        analytics, logger = analytics_env
        s_id = logger.create_session(start_time="2026-10-07T10:00:00+00:00")
        events = [
            (BehaviorState.FOCUSED, "2026-10-07T10:00:00+00:00"),
            (BehaviorState.DISTRACTED, "2026-10-07T10:05:00+00:00"),
            (BehaviorState.FOCUSED, "2026-10-07T10:07:00+00:00"),
            (BehaviorState.PHONE_DETECTED, "2026-10-07T10:10:00+00:00"),
            (BehaviorState.FOCUSED, "2026-10-07T10:11:00+00:00"),
        ]
        for st, ts in events:
            logger.log_behavior_event(
                session_id=s_id,
                timestamp=ts,
                state=st,
                state_duration=0.0,
                confidence=0.9,
                reason="test",
            )
        assert analytics.get_transition_count(s_id) == 4

    def test_missing_and_empty_sessions(self, analytics_env):
        # TEST 16: Non-existent session and session with no events
        analytics, logger = analytics_env
        assert analytics.get_session_summary(999) is None
        assert analytics.get_state_durations(999).total_seconds == 0.0

        # Session with no events
        empty_id = logger.create_session(start_time="2026-10-07T10:00:00+00:00")
        logger.end_session(empty_id, end_time="2026-10-07T10:05:00+00:00", duration_seconds=300.0)
        summary = analytics.get_session_summary(empty_id)
        assert summary is not None
        assert summary.focus_score is None
        assert summary.transition_count == 0

    def test_controlled_synthetic_session_benchmark(self, analytics_env):
        # TEST 28 Manual verification benchmark:
        # FOCUSED: 10 min (600s) -> 10:00 to 10:10
        # DISTRACTED: 2 min (120s) -> 10:10 to 10:12
        # PHONE_DETECTED: 1 min (60s) -> 10:12 to 10:13
        # FOCUSED: 7 min (420s) -> 10:13 to 10:20
        # NO_FACE: 5 min (300s) -> 10:20 to 10:25 (end)
        # Total: 25 min (1500s)
        # Scored: 20 min (1200s)
        # Focused: 17 min (1020s)
        # Expected Focus Score: 17 / 20 * 100 = 85.0
        analytics, logger = analytics_env
        s_id = logger.create_session(start_time="2026-10-07T10:00:00+00:00")

        logger.log_behavior_event(s_id, "2026-10-07T10:00:00+00:00", BehaviorState.FOCUSED, 0.0, 0.9, "focused")
        logger.log_behavior_event(s_id, "2026-10-07T10:10:00+00:00", BehaviorState.DISTRACTED, 0.0, 0.85, "head_away")
        logger.log_behavior_event(s_id, "2026-10-07T10:12:00+00:00", BehaviorState.PHONE_DETECTED, 0.0, 0.9, "phone_detected")
        logger.log_behavior_event(s_id, "2026-10-07T10:13:00+00:00", BehaviorState.FOCUSED, 0.0, 0.9, "focused")
        logger.log_behavior_event(s_id, "2026-10-07T10:20:00+00:00", BehaviorState.NO_FACE, 0.0, 0.95, "face_not_detected")
        logger.end_session(s_id, end_time="2026-10-07T10:25:00+00:00", duration_seconds=1500.0)

        durations = analytics.get_state_durations(s_id)
        assert durations.focused_seconds == 1020.0  # 600 + 420
        assert durations.distracted_seconds == 120.0
        assert durations.phone_detected_seconds == 60.0
        assert durations.no_face_seconds == 300.0
        assert durations.total_seconds == 1500.0

        summary = analytics.get_session_summary(s_id)
        assert summary is not None
        assert pytest.approx(summary.focus_score, 0.001) == 85.0
        assert summary.transition_count == 4
        assert summary.duration_seconds == 1500.0

    def test_get_completed_session_summaries(self, analytics_env):
        analytics, logger = analytics_env
        s1 = logger.create_session(start_time="2026-10-07T10:00:00+00:00")
        s2 = logger.create_session(start_time="2026-10-07T11:00:00+00:00")
        logger.end_session(s1, end_time="2026-10-07T10:30:00+00:00", duration_seconds=1800.0)

        completed = analytics.get_completed_session_summaries()
        assert len(completed) == 1
        assert completed[0].session_id == s1

    def test_get_timeline_segments(self, analytics_env):
        analytics, logger = analytics_env
        s_id = logger.create_session(start_time="2026-10-07T10:00:00+00:00")
        logger.log_behavior_event(s_id, "2026-10-07T10:00:00+00:00", BehaviorState.FOCUSED, 0.0, 0.9, "focused")
        logger.log_behavior_event(s_id, "2026-10-07T10:15:00+00:00", BehaviorState.DISTRACTED, 0.0, 0.85, "head_away")
        logger.end_session(s_id, end_time="2026-10-07T10:20:00+00:00", duration_seconds=1200.0)

        segments = analytics.get_timeline_segments(s_id)
        assert len(segments) == 2
        assert segments[0]["state"] == "FOCUSED"
        assert segments[0]["duration_seconds"] == 900.0
        assert segments[1]["state"] == "DISTRACTED"
        assert segments[1]["duration_seconds"] == 300.0
