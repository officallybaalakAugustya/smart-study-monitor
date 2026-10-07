"""Tests for analytics/models.py.

Verifies model instantiation, dictionary conversions, property access,
and serialization.
"""

from analytics.models import FocusScoreResult, SessionSummary, StateDurations
from behavior.behavior_engine import BehaviorState


class TestStateDurations:
    def test_defaults(self):
        dur = StateDurations()
        assert dur.focused_seconds == 0.0
        assert dur.distracted_seconds == 0.0
        assert dur.total_seconds == 0.0

        d = dur.to_dict()
        assert d[BehaviorState.FOCUSED.value] == 0.0
        assert len(d) == 7

    def test_from_dict_and_total(self):
        raw = {
            "FOCUSED": 1800.0,
            "DISTRACTED": 300.0,
            "PHONE_DETECTED": 60.0,
        }
        dur = StateDurations.from_dict(raw)
        assert dur.focused_seconds == 1800.0
        assert dur.distracted_seconds == 300.0
        assert dur.phone_detected_seconds == 60.0
        assert dur.drowsy_seconds == 0.0
        assert dur.total_seconds == 2160.0


class TestFocusScoreResult:
    def test_instantiation_and_properties(self):
        res = FocusScoreResult(
            score=85.5,
            focused_seconds=1800.0,
            scored_seconds=2000.0,
            excluded_seconds=200.0,
            total_duration_seconds=2200.0,
            is_valid=True,
        )
        assert res.score == 85.5
        assert res.is_valid is True
        assert res.reason is None


class TestSessionSummary:
    def test_properties_and_to_dict(self):
        durations = StateDurations(
            focused_seconds=1200.0,
            distracted_seconds=300.0,
            drowsy_seconds=60.0,
            phone_detected_seconds=40.0,
            away_from_desk_seconds=100.0,
            no_face_seconds=150.0,
            unknown_seconds=50.0,
        )
        score_res = FocusScoreResult(
            score=70.588,
            focused_seconds=1200.0,
            scored_seconds=1700.0,
            excluded_seconds=200.0,
            total_duration_seconds=1900.0,
            is_valid=True,
        )
        summary = SessionSummary(
            session_id=1,
            start_time="2026-10-07T10:00:00+00:00",
            end_time="2026-10-07T10:31:40+00:00",
            duration_seconds=1900.0,
            status="COMPLETED",
            state_durations=durations,
            transition_count=5,
            focus_score_result=score_res,
        )

        assert summary.session_id == 1
        assert summary.focus_score == 70.588
        assert summary.focused_seconds == 1200.0
        assert summary.distracted_seconds == 300.0
        assert summary.drowsy_seconds == 60.0
        assert summary.phone_detected_seconds == 40.0
        assert summary.away_from_desk_seconds == 100.0
        assert summary.no_face_seconds == 150.0
        assert summary.unknown_seconds == 50.0

        d = summary.to_dict()
        assert d["session_id"] == 1
        assert d["focus_score"] == 70.588
        assert d["transition_count"] == 5
        assert d["state_durations"]["FOCUSED"] == 1200.0
