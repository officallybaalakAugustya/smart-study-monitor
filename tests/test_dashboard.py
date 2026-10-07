"""Tests for dashboard formatting and data preparation logic.

Verifies data transformations, formatting helpers, distribution calculations,
timeline segment transformations, and empty/malformed data resilience without
requiring a browser context.
"""

import pytest

from analytics.focus_score import calculate_focus_score
from analytics.models import FocusScoreResult, SessionSummary, StateDurations
from analytics.session_analytics import SessionAnalytics
from behavior.behavior_engine import BehaviorState, BehaviorReason
from dashboard.formatting import (
    format_duration,
    format_focus_score,
    format_iso_datetime,
    get_state_color,
    get_state_label,
    prepare_event_table_data,
    prepare_session_history_data,
    prepare_state_distribution_data,
)
from database.database import Database
from session.event_logger import EventLogger


@pytest.fixture
def test_env():
    """Provides an isolated database, logger, and analytics layer."""
    db = Database(db_path=":memory:")
    logger = EventLogger(db=db)
    logger.initialize_database()
    analytics = SessionAnalytics(event_logger=logger)
    return db, logger, analytics


class TestDashboardFormattingHelpers:
    def test_format_duration(self):
        assert format_duration(0.0) == "00:00:00"
        assert format_duration(65.0) == "00:01:05"
        assert format_duration(3600.0) == "01:00:00"
        assert format_duration(3665.0) == "01:01:05"
        assert format_duration(-10.0) == "00:00:00"

    def test_format_focus_score(self):
        assert format_focus_score(None) == "N/A"
        assert format_focus_score(85.0) == "85.0"
        assert format_focus_score(92.345) == "92.3"

    def test_format_iso_datetime_valid(self):
        iso_str = "2026-10-07T10:15:30+00:00"
        formatted = format_iso_datetime(iso_str, include_time=True)
        assert "Oct 07, 2026" in formatted or "Oct 7, 2026" in formatted
        assert "10:15:30" in formatted

    def test_format_iso_datetime_date_only(self):
        iso_str = "2026-10-07T10:15:30+00:00"
        formatted = format_iso_datetime(iso_str, include_time=False)
        assert "10:15:30" not in formatted

    def test_format_iso_datetime_invalid_and_none(self):
        assert format_iso_datetime(None) == "--"
        assert format_iso_datetime("") == "--"
        assert format_iso_datetime("not-a-date") == "not-a-date"

    def test_get_state_color_and_label(self):
        assert get_state_color("FOCUSED") == "#22C55E"
        assert get_state_color("NON_EXISTENT") == "#94A3B8"
        assert get_state_label("FOCUSED") == "Focused"
        assert get_state_label("PHONE_DETECTED") == "Phone Detected"


class TestDashboardDataPreparation:
    def test_empty_database(self, test_env):
        db, logger, analytics = test_env
        completed = analytics.get_completed_session_summaries()
        assert completed == []
        history_rows = prepare_session_history_data(completed)
        assert history_rows == []

    def test_one_completed_session(self, test_env):
        db, logger, analytics = test_env
        s_id = logger.create_session(start_time="2026-10-07T10:00:00+00:00")
        logger.log_behavior_event(s_id, "2026-10-07T10:00:00+00:00", BehaviorState.FOCUSED, 0.0, 0.9, "focused")
        logger.end_session(s_id, end_time="2026-10-07T10:30:00+00:00", duration_seconds=1800.0)

        completed = analytics.get_completed_session_summaries()
        assert len(completed) == 1
        history_rows = prepare_session_history_data(completed)
        assert len(history_rows) == 1
        row = history_rows[0]
        assert row["Session ID"] == s_id
        assert row["Focus Score"] == "100.0"
        assert row["Focused Time"] == "00:30:00"

    def test_multiple_completed_sessions_ordering(self, test_env):
        db, logger, analytics = test_env
        s1 = logger.create_session(start_time="2026-10-07T09:00:00+00:00")
        logger.end_session(s1, end_time="2026-10-07T09:30:00+00:00", duration_seconds=1800.0)

        s2 = logger.create_session(start_time="2026-10-07T10:00:00+00:00")
        logger.end_session(s2, end_time="2026-10-07T10:30:00+00:00", duration_seconds=1800.0)

        completed = analytics.get_completed_session_summaries()
        assert len(completed) == 2
        # Newest first
        assert completed[0].session_id == s2
        assert completed[1].session_id == s1

    def test_focus_score_none_handling(self, test_env):
        db, logger, analytics = test_env
        # Session < 60s -> focus score is None
        s_id = logger.create_session(start_time="2026-10-07T10:00:00+00:00")
        logger.log_behavior_event(s_id, "2026-10-07T10:00:00+00:00", BehaviorState.FOCUSED, 0.0, 0.9, "focused")
        logger.end_session(s_id, end_time="2026-10-07T10:00:30+00:00", duration_seconds=30.0)

        summary = analytics.get_session_summary(s_id)
        assert summary.focus_score is None

        history_rows = prepare_session_history_data([summary])
        assert history_rows[0]["Focus Score"] == "N/A"

    def test_zero_duration_states_filtering(self):
        durations = StateDurations(focused_seconds=1800.0, distracted_seconds=0.0)
        summary = SessionSummary(
            session_id=1,
            start_time="2026-10-07T10:00:00+00:00",
            end_time="2026-10-07T10:30:00+00:00",
            duration_seconds=1800.0,
            status="COMPLETED",
            state_durations=durations,
            transition_count=0,
            focus_score_result=calculate_focus_score(durations),
        )

        filtered = prepare_state_distribution_data(summary, include_zero_duration=False)
        assert len(filtered) == 1
        assert filtered[0]["state"] == "FOCUSED"
        assert filtered[0]["percentage"] == 100.0

        all_states = prepare_state_distribution_data(summary, include_zero_duration=True)
        assert len(all_states) == 7

    def test_event_table_data_preparation(self, test_env):
        db, logger, analytics = test_env
        s_id = logger.create_session(start_time="2026-10-07T10:00:00+00:00")
        logger.log_behavior_event(s_id, "2026-10-07T10:00:00+00:00", BehaviorState.FOCUSED, 0.0, 0.92, "focused")
        logger.log_behavior_event(s_id, "2026-10-07T10:20:00+00:00", BehaviorState.DISTRACTED, 0.0, 0.85, "head_away")
        logger.end_session(s_id, end_time="2026-10-07T10:25:00+00:00", duration_seconds=1500.0)

        segments = analytics.get_timeline_segments(s_id)
        assert len(segments) == 2

        table_rows = prepare_event_table_data(segments)
        assert len(table_rows) == 2
        assert table_rows[0]["State"] == "Focused"
        assert table_rows[0]["Duration"] == "00:20:00"
        assert table_rows[0]["Confidence"] == "0.92"
        assert table_rows[0]["Reason"] == "focused"

        assert table_rows[1]["State"] == "Distracted"
        assert table_rows[1]["Duration"] == "00:05:00"
        assert table_rows[1]["Confidence"] == "0.85"
        assert table_rows[1]["Reason"] == "head_away"
