"""Focus Score calculation engine for Smart Study Monitor.

Provides a deterministic, transparent, weighted-duration model to quantify
study engagement from reconstructed behavioral state timelines.

Important Note:
    The Focus Score is a heuristic application-defined behavioral metric
    reflecting observed camera-based engagement states. It is NOT a medical,
    psychological, or scientifically validated measure of human productivity,
    intelligence, concentration, or cognitive performance.

Scoring Model:
    - Each behavioral state is assigned a non-negative weight (default: FOCUSED=1.0, others=0.0).
    - Unobservable or ambiguous states (NO_FACE, UNKNOWN) are excluded from the
      scorable duration denominator by default because the camera cannot verify
      what the user is doing when the face is unavailable.
    - Scored Duration = Sum of non-excluded state durations.
    - Focus Score = (Sum(weight_i * duration_i) / Scored Duration) * 100.0
    - If Scored Duration < minimum_scored_duration_sec (default 60s), the score
      is reported as None to prevent misleading scores on trivial durations.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Union

from analytics.models import FocusScoreResult, StateDurations
from behavior.behavior_engine import BehaviorState
from config.config import AnalyticsConfig, config as global_config


class FocusScoreCalculator:
    """Calculates explainable, deterministic Focus Scores from state durations."""

    def __init__(self, config: Optional[AnalyticsConfig] = None) -> None:
        """Initializes the calculator.

        Args:
            config: Analytics configuration container. If None, uses global default.
        """
        self._cfg = config or global_config.analytics

    @property
    def config(self) -> AnalyticsConfig:
        """Returns the active AnalyticsConfig."""
        return self._cfg

    def calculate(
        self,
        durations: Union[StateDurations, Dict[str, float]],
    ) -> FocusScoreResult:
        """Computes the Focus Score from behavioral state durations.

        Args:
            durations: StateDurations instance or dictionary mapping state names
                       to duration in seconds.

        Returns:
            FocusScoreResult containing the final score, durations, and transparency metadata.
        """
        if isinstance(durations, StateDurations):
            dur_dict = durations.to_dict()
        else:
            # Normalize dictionary with all known states defaulting to 0.0
            dur_dict = {
                state.value: float(durations.get(state.value, 0.0))
                for state in BehaviorState
            }

        # Determine state weights from configuration
        weights: Dict[str, float] = {
            BehaviorState.FOCUSED.value: float(self._cfg.focused_weight),
            BehaviorState.DISTRACTED.value: float(self._cfg.distracted_weight),
            BehaviorState.DROWSY.value: float(self._cfg.drowsy_weight),
            BehaviorState.PHONE_DETECTED.value: float(self._cfg.phone_detected_weight),
            BehaviorState.AWAY_FROM_DESK.value: float(self._cfg.away_from_desk_weight),
            BehaviorState.NO_FACE.value: float(self._cfg.no_face_weight),
            BehaviorState.UNKNOWN.value: float(self._cfg.unknown_weight),
        }

        # Identify excluded states
        excluded_states: List[str] = []
        if self._cfg.exclude_no_face:
            excluded_states.append(BehaviorState.NO_FACE.value)
        if self._cfg.exclude_unknown:
            excluded_states.append(BehaviorState.UNKNOWN.value)

        # Aggregate durations
        total_duration = sum(max(0.0, float(v)) for v in dur_dict.values())
        focused_seconds = max(0.0, float(dur_dict.get(BehaviorState.FOCUSED.value, 0.0)))

        scored_seconds = 0.0
        excluded_seconds = 0.0
        weighted_focus_seconds = 0.0

        for state_name, duration_sec in dur_dict.items():
            valid_duration = max(0.0, float(duration_sec))
            if state_name in excluded_states:
                excluded_seconds += valid_duration
            else:
                scored_seconds += valid_duration
                weight = weights.get(state_name, 0.0)
                weighted_focus_seconds += weight * valid_duration

        # Validate minimum scorable duration threshold
        min_duration = self._cfg.minimum_scored_duration_sec

        if scored_seconds <= 0.0:
            reason = (
                "No scorable duration recorded (all time was in excluded states or zero)"
                if total_duration > 0.0
                else "Zero session duration"
            )
            return FocusScoreResult(
                score=None,
                focused_seconds=focused_seconds,
                scored_seconds=0.0,
                excluded_seconds=excluded_seconds,
                total_duration_seconds=total_duration,
                state_durations=dur_dict,
                weights_used=weights,
                excluded_states=excluded_states,
                is_valid=False,
                reason=reason,
            )

        if scored_seconds < min_duration:
            return FocusScoreResult(
                score=None,
                focused_seconds=focused_seconds,
                scored_seconds=scored_seconds,
                excluded_seconds=excluded_seconds,
                total_duration_seconds=total_duration,
                state_durations=dur_dict,
                weights_used=weights,
                excluded_states=excluded_states,
                is_valid=False,
                reason=f"Scored duration ({scored_seconds:.1f}s) is below minimum threshold ({min_duration:.1f}s)",
            )

        # Calculate score: (weighted_focus_seconds / scored_seconds) * 100.0
        raw_score = (weighted_focus_seconds / scored_seconds) * 100.0

        # Safety clamp to guarantee 0.0 <= score <= 100.0
        if math.isnan(raw_score) or math.isinf(raw_score):
            score: Optional[float] = None
            is_valid = False
            reason = "Calculation produced non-finite value"
        else:
            score = max(0.0, min(100.0, float(raw_score)))
            is_valid = True
            reason = None

        return FocusScoreResult(
            score=score,
            focused_seconds=focused_seconds,
            scored_seconds=scored_seconds,
            excluded_seconds=excluded_seconds,
            total_duration_seconds=total_duration,
            state_durations=dur_dict,
            weights_used=weights,
            excluded_states=excluded_states,
            is_valid=is_valid,
            reason=reason,
        )


def calculate_focus_score(
    durations: Union[StateDurations, Dict[str, float]],
    config: Optional[AnalyticsConfig] = None,
) -> FocusScoreResult:
    """Convenience helper to compute Focus Score using FocusScoreCalculator."""
    return FocusScoreCalculator(config=config).calculate(durations)
