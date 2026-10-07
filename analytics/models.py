"""Data models for Smart Study Monitor analytics.

Defines structured models for behavioral state durations, Focus Score
calculation results, and session analytics summaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from behavior.behavior_engine import BehaviorState


@dataclass
class StateDurations:
    """Reconstructed durations in seconds for each behavioral state."""

    focused_seconds: float = 0.0
    distracted_seconds: float = 0.0
    drowsy_seconds: float = 0.0
    phone_detected_seconds: float = 0.0
    away_from_desk_seconds: float = 0.0
    no_face_seconds: float = 0.0
    unknown_seconds: float = 0.0

    @property
    def total_seconds(self) -> float:
        """Returns the sum of all state durations."""
        return (
            self.focused_seconds
            + self.distracted_seconds
            + self.drowsy_seconds
            + self.phone_detected_seconds
            + self.away_from_desk_seconds
            + self.no_face_seconds
            + self.unknown_seconds
        )

    def to_dict(self) -> Dict[str, float]:
        """Converts to a dictionary keyed by canonical BehaviorState string values."""
        return {
            BehaviorState.FOCUSED.value: self.focused_seconds,
            BehaviorState.DISTRACTED.value: self.distracted_seconds,
            BehaviorState.DROWSY.value: self.drowsy_seconds,
            BehaviorState.PHONE_DETECTED.value: self.phone_detected_seconds,
            BehaviorState.AWAY_FROM_DESK.value: self.away_from_desk_seconds,
            BehaviorState.NO_FACE.value: self.no_face_seconds,
            BehaviorState.UNKNOWN.value: self.unknown_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> StateDurations:
        """Constructs StateDurations from a dictionary of state string keys."""
        return cls(
            focused_seconds=float(data.get(BehaviorState.FOCUSED.value, 0.0)),
            distracted_seconds=float(data.get(BehaviorState.DISTRACTED.value, 0.0)),
            drowsy_seconds=float(data.get(BehaviorState.DROWSY.value, 0.0)),
            phone_detected_seconds=float(data.get(BehaviorState.PHONE_DETECTED.value, 0.0)),
            away_from_desk_seconds=float(data.get(BehaviorState.AWAY_FROM_DESK.value, 0.0)),
            no_face_seconds=float(data.get(BehaviorState.NO_FACE.value, 0.0)),
            unknown_seconds=float(data.get(BehaviorState.UNKNOWN.value, 0.0)),
        )


@dataclass
class FocusScoreResult:
    """Detailed breakdown and explainability result of a Focus Score calculation.

    Attributes:
        score: Computed Focus Score in [0.0, 100.0], or None if insufficient scorable data.
        focused_seconds: Duration in seconds classified as FOCUSED.
        scored_seconds: Total scorable duration (denominator of the score).
        excluded_seconds: Duration excluded from calculation (e.g. NO_FACE, UNKNOWN).
        total_duration_seconds: Total timeline duration of the session.
        state_durations: Dictionary mapping state names to durations in seconds.
        weights_used: Dictionary of state weights used in calculation.
        excluded_states: List of state names excluded from the scored denominator.
        is_valid: True if scorable duration met the minimum threshold.
        reason: Explanation if the score could not be calculated, or None.
    """

    score: Optional[float]
    focused_seconds: float
    scored_seconds: float
    excluded_seconds: float
    total_duration_seconds: float
    state_durations: Dict[str, float] = field(default_factory=dict)
    weights_used: Dict[str, float] = field(default_factory=dict)
    excluded_states: List[str] = field(default_factory=list)
    is_valid: bool = True
    reason: Optional[str] = None


@dataclass
class SessionSummary:
    """Comprehensive analytics summary of a study session.

    Suitable for consumption by future dashboards, reporting, and exports.
    """

    session_id: int
    start_time: str
    end_time: Optional[str]
    duration_seconds: float
    status: str
    state_durations: StateDurations
    transition_count: int
    focus_score_result: FocusScoreResult

    @property
    def focus_score(self) -> Optional[float]:
        """Convenience property returning the calculated Focus Score."""
        return self.focus_score_result.score

    @property
    def focused_seconds(self) -> float:
        return self.state_durations.focused_seconds

    @property
    def distracted_seconds(self) -> float:
        return self.state_durations.distracted_seconds

    @property
    def drowsy_seconds(self) -> float:
        return self.state_durations.drowsy_seconds

    @property
    def phone_detected_seconds(self) -> float:
        return self.state_durations.phone_detected_seconds

    @property
    def away_from_desk_seconds(self) -> float:
        return self.state_durations.away_from_desk_seconds

    @property
    def no_face_seconds(self) -> float:
        return self.state_durations.no_face_seconds

    @property
    def unknown_seconds(self) -> float:
        return self.state_durations.unknown_seconds

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the summary into a standard dictionary."""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "status": self.status,
            "state_durations": self.state_durations.to_dict(),
            "transition_count": self.transition_count,
            "focus_score": self.focus_score,
            "focus_score_details": {
                "score": self.focus_score_result.score,
                "is_valid": self.focus_score_result.is_valid,
                "reason": self.focus_score_result.reason,
                "scored_seconds": self.focus_score_result.scored_seconds,
                "excluded_seconds": self.focus_score_result.excluded_seconds,
                "weights_used": self.focus_score_result.weights_used,
            },
        }
