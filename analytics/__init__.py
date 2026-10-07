"""Analytics package for Smart Study Monitor.

Provides Focus Score computation, state duration timeline reconstruction,
and session summaries.
"""

from analytics.models import FocusScoreResult, SessionSummary, StateDurations
from analytics.focus_score import FocusScoreCalculator, calculate_focus_score
from analytics.session_analytics import SessionAnalytics

__all__ = [
    "StateDurations",
    "FocusScoreResult",
    "SessionSummary",
    "FocusScoreCalculator",
    "calculate_focus_score",
    "SessionAnalytics",
]
