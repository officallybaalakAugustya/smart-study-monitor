"""Tests for analytics/focus_score.py.

Verifies mathematical accuracy of Focus Score calculation, handling of
excluded states (NO_FACE, UNKNOWN), minimum duration thresholds, bounds clamping,
and custom weights.
"""

import pytest

from analytics.focus_score import FocusScoreCalculator, calculate_focus_score
from analytics.models import StateDurations
from config.config import AnalyticsConfig


class TestFocusScoreCalculation:
    def test_100_percent_focused(self):
        # TEST 1: 60 minutes focused -> Score = 100.0
        durations = {"FOCUSED": 3600.0}
        res = calculate_focus_score(durations)
        assert res.score == 100.0
        assert res.is_valid is True
        assert res.focused_seconds == 3600.0
        assert res.scored_seconds == 3600.0
        assert res.excluded_seconds == 0.0

    def test_focused_and_distracted(self):
        # TEST 2: 50 minutes focused, 10 minutes distracted -> Score = 83.333...
        durations = {
            "FOCUSED": 3000.0,
            "DISTRACTED": 600.0,
        }
        res = calculate_focus_score(durations)
        expected = (3000.0 / 3600.0) * 100.0
        assert pytest.approx(res.score, 0.001) == expected
        assert res.is_valid is True
        assert res.scored_seconds == 3600.0

    def test_no_face_excluded(self):
        # TEST 3: 50 minutes focused, 5 minutes distracted, 5 minutes NO_FACE
        # NO_FACE must be excluded from denominator -> 50 / 55 * 100 ≈ 90.909
        durations = {
            "FOCUSED": 3000.0,
            "DISTRACTED": 300.0,
            "NO_FACE": 300.0,
        }
        res = calculate_focus_score(durations)
        expected = (3000.0 / 3300.0) * 100.0
        assert pytest.approx(res.score, 0.001) == expected
        assert res.is_valid is True
        assert res.scored_seconds == 3300.0
        assert res.excluded_seconds == 300.0

    def test_only_no_face_returns_none(self):
        # TEST 4: Only NO_FACE -> focus_score = None
        durations = {"NO_FACE": 1800.0}
        res = calculate_focus_score(durations)
        assert res.score is None
        assert res.is_valid is False
        assert res.scored_seconds == 0.0
        assert res.excluded_seconds == 1800.0
        assert "excluded states" in res.reason.lower()

    def test_only_unknown_returns_none(self):
        # TEST 5: Only UNKNOWN -> focus_score = None
        durations = {"UNKNOWN": 1800.0}
        res = calculate_focus_score(durations)
        assert res.score is None
        assert res.is_valid is False
        assert res.scored_seconds == 0.0
        assert res.excluded_seconds == 1800.0

    def test_minimum_scored_duration_threshold(self):
        # TEST 10: Very short session (< 60s default) -> returns None
        durations = {"FOCUSED": 45.0, "DISTRACTED": 10.0}
        res = calculate_focus_score(durations)
        assert res.score is None
        assert res.is_valid is False
        assert "below minimum threshold" in res.reason

    def test_minimum_scored_duration_satisfied(self):
        # Exactly 60 seconds satisfies default threshold
        durations = {"FOCUSED": 60.0}
        res = calculate_focus_score(durations)
        assert res.score == 100.0
        assert res.is_valid is True

    def test_score_cannot_exceed_100(self):
        # TEST 11: Even with abnormal weights, score is clamped <= 100
        cfg = AnalyticsConfig(focused_weight=2.0)
        calc = FocusScoreCalculator(config=cfg)
        durations = {"FOCUSED": 120.0}
        res = calc.calculate(durations)
        assert res.score == 100.0

    def test_score_cannot_become_negative(self):
        # TEST 12: Score is non-negative
        durations = {"DISTRACTED": 120.0}
        res = calculate_focus_score(durations)
        assert res.score == 0.0

    def test_zero_duration_returns_none(self):
        # Zero duration session
        durations = StateDurations()
        res = calculate_focus_score(durations)
        assert res.score is None
        assert res.is_valid is False

    def test_all_behavior_states_with_custom_weights(self):
        # TEST 14: All states present
        durations = StateDurations(
            focused_seconds=1000.0,
            distracted_seconds=200.0,
            drowsy_seconds=100.0,
            phone_detected_seconds=100.0,
            away_from_desk_seconds=100.0,
            no_face_seconds=300.0,
            unknown_seconds=100.0,
        )
        res = calculate_focus_score(durations)
        # Scored: 1000 + 200 + 100 + 100 + 100 = 1500s
        # Focused: 1000s
        # Score: 1000 / 1500 * 100 = 66.666...
        expected = (1000.0 / 1500.0) * 100.0
        assert pytest.approx(res.score, 0.001) == expected
        assert res.scored_seconds == 1500.0
        assert res.excluded_seconds == 400.0
        assert res.total_duration_seconds == 1900.0
