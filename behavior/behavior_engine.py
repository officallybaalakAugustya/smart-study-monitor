"""Behavior Engine for Smart Study Monitor.

Combines individual computer-vision signals (face detection, eye state,
head pose, phone detection) into stable, temporally-confirmed behavioral
states using a priority-based evaluation system with temporal confirmation
and hysteresis recovery.

This is a project-defined heuristic rule system, NOT a medical diagnostic
system. Thresholds and state transitions are tuned for a study-monitoring
context and are not intended for clinical use.

State Priority (Canonical Behavioral States):
    1. PHONE_DETECTED  — A mobile phone is visible in the frame.
    2. DROWSY          — Eyes have been closed for a sustained period.
    3. AWAY_FROM_DESK  — User is away from desk (sustained absence of face or sustained head-away).
    4. DISTRACTED      — Head is turned away from center for a sustained period.
    5. FOCUSED         — User is attentive (face detected, eyes open, head centered, no phone).

Special Availability / Validity States:
    - NO_FACE          — Face is currently not detected in frame (sensor availability condition).
    - UNKNOWN          — Available signals are insufficient or invalid for reliable classification.

Temporal Confirmation:
    All state transitions require sustained signal persistence before the
    behavior state actually changes. This prevents instant state flipping
    from momentary signal noise. Recovery back to FOCUSED also requires a
    cooldown period (hysteresis).

Timing:
    Uses time.monotonic() for all temporal logic. An injectable clock
    function enables deterministic unit testing without real-time delays.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Optional

from config.config import config as default_global_config
from vision.eye_tracker import EyeState, EyeTrackingResult
from vision.head_pose import HeadDirection, HeadPoseResult
from vision.face_detector import FaceDetectionResult
from detection.phone_detector import PhoneDetectionResult


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class BehaviorState(Enum):
    """High-level behavioral states for the study monitor."""

    FOCUSED = "FOCUSED"
    DISTRACTED = "DISTRACTED"
    DROWSY = "DROWSY"
    PHONE_DETECTED = "PHONE_DETECTED"
    AWAY_FROM_DESK = "AWAY_FROM_DESK"
    NO_FACE = "NO_FACE"
    UNKNOWN = "UNKNOWN"


class BehaviorReason:
    """Stable structured reason codes explaining why a behavioral state was assigned."""

    FOCUSED = "focused"
    PHONE_DETECTED = "phone_detected"
    EYES_CLOSED = "eyes_closed"
    HEAD_AWAY = "head_away"
    AWAY_FROM_DESK = "head_away"  # sustained head away maps to head_away
    FACE_NOT_DETECTED = "face_not_detected"
    INSUFFICIENT_SIGNAL = "insufficient_signal"
    INVALID_INPUT = "invalid_input"
    NORMAL = "focused"  # alias for backward compatibility
    NO_FACE = "face_not_detected"  # alias for backward compatibility
    INSUFFICIENT_DATA = "insufficient_signal"  # alias for backward compatibility


# ---------------------------------------------------------------------------
# Result Dataclass
# ---------------------------------------------------------------------------

@dataclass
class BehaviorResult:
    """Output of each BehaviorEngine.update() call.

    Stable public interface between Behavior Engine and downstream modules.

    Invariants:
        - state: Must always be a valid BehaviorState.
        - state_started_at: Monotonic timestamp (float) or None.
        - state_duration: Must be >= 0.0.
        - confidence: Float between 0.0 and 1.0 inclusive.
        - reason: Stable structured reason code (str), not a UI sentence.
        - signals: Dictionary containing relevant sensor and derived inputs.
    """

    state: BehaviorState = BehaviorState.UNKNOWN
    state_started_at: Optional[float] = None
    state_duration: float = 0.0
    confidence: float = 0.0
    reason: str = BehaviorReason.INSUFFICIENT_SIGNAL
    signals: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Behavior Engine
# ---------------------------------------------------------------------------

class BehaviorEngine:
    """Combines CV signals into temporally-confirmed behavioral states.

    Design Principles:
        - Pure signal processing: no CV inference, no I/O, no GPU usage.
        - Negligible per-frame overhead (simple comparisons and timestamps).
        - Injectable clock for deterministic unit testing.
        - All temporal logic uses monotonic time (not frame counting).

    Usage:
        engine = BehaviorEngine()
        result = engine.update(face_result, eye_result, pose_result, phone_result)
    """

    def __init__(
        self,
        clock: Optional[Callable[[], float]] = None,
        phone_confirmation_sec: Optional[float] = None,
        drowsiness_confirmation_sec: Optional[float] = None,
        head_away_confirmation_sec: Optional[float] = None,
        away_from_desk_confirmation_sec: Optional[float] = None,
        no_face_confirmation_sec: Optional[float] = None,
        state_recovery_sec: Optional[float] = None,
    ) -> None:
        """Initializes the BehaviorEngine.

        Args:
            clock: Callable returning monotonic time in seconds. Defaults to
                   time.monotonic. Inject a custom clock for testing.
            phone_confirmation_sec: Seconds phone must persist before PHONE_DETECTED.
            drowsiness_confirmation_sec: Seconds eyes must stay closed before DROWSY.
            head_away_confirmation_sec: Seconds head must stay turned before DISTRACTED.
            away_from_desk_confirmation_sec: Seconds head significantly away before AWAY_FROM_DESK.
            no_face_confirmation_sec: Seconds without a face before AWAY_FROM_DESK.
            state_recovery_sec: Cooldown before transitioning back to FOCUSED.
        """
        # Clock
        self._clock = clock or time.monotonic

        # Load defaults from global config, allow per-instance overrides
        bcfg = default_global_config.behavior

        self._phone_confirmation_sec = (
            phone_confirmation_sec if phone_confirmation_sec is not None
            else bcfg.phone_confirmation_sec
        )
        self._drowsiness_confirmation_sec = (
            drowsiness_confirmation_sec if drowsiness_confirmation_sec is not None
            else bcfg.drowsiness_confirmation_sec
        )
        self._head_away_confirmation_sec = (
            head_away_confirmation_sec if head_away_confirmation_sec is not None
            else bcfg.head_away_confirmation_sec
        )
        self._away_from_desk_confirmation_sec = (
            away_from_desk_confirmation_sec if away_from_desk_confirmation_sec is not None
            else bcfg.away_from_desk_confirmation_sec
        )
        self._no_face_confirmation_sec = (
            no_face_confirmation_sec if no_face_confirmation_sec is not None
            else bcfg.no_face_confirmation_sec
        )
        self._state_recovery_sec = (
            state_recovery_sec if state_recovery_sec is not None
            else bcfg.state_recovery_sec
        )

        # Confirmed state tracking
        self._confirmed_state: BehaviorState = BehaviorState.UNKNOWN
        self._confirmed_reason: str = BehaviorReason.INSUFFICIENT_SIGNAL
        self._confirmed_state_started_at: float = self._clock()

        # Candidate state tracking (pre-confirmation)
        self._candidate_state: Optional[BehaviorState] = None
        self._candidate_reason: Optional[str] = None
        self._candidate_started_at: float = 0.0

        # Recovery tracking
        self._recovery_started_at: Optional[float] = None

    def reset(self) -> None:
        """Resets all internal state to initial values."""
        now = self._clock()
        self._confirmed_state = BehaviorState.UNKNOWN
        self._confirmed_reason = BehaviorReason.INSUFFICIENT_SIGNAL
        self._confirmed_state_started_at = now
        self._candidate_state = None
        self._candidate_reason = None
        self._candidate_started_at = 0.0
        self._recovery_started_at = None

    @property
    def current_state(self) -> BehaviorState:
        """Returns the current confirmed behavioral state."""
        return self._confirmed_state

    def update(
        self,
        face_result: FaceDetectionResult,
        eye_result: EyeTrackingResult,
        pose_result: HeadPoseResult,
        phone_result: PhoneDetectionResult,
    ) -> BehaviorResult:
        """Evaluates all signals and returns the current behavioral state.

        This method should be called once per frame. It is extremely lightweight
        (no CV operations, pure Python comparisons and timestamp checks).

        Args:
            face_result:  Output from FaceDetector.process().
            eye_result:   Output from EyeTracker.process().
            pose_result:  Output from HeadPoseEstimator.process().
            phone_result: Output from PhoneDetector.detect().

        Returns:
            BehaviorResult with the current confirmed state, timing, and signals.
        """
        now = self._clock()

        # Build signals snapshot for debugging / logging
        signals = self._build_signals(face_result, eye_result, pose_result, phone_result)

        # Evaluate raw (instantaneous) state from signals using priority order
        raw_state, raw_reason = self._evaluate_raw_state(
            face_result, eye_result, pose_result, phone_result,
        )

        # Determine confirmation duration for this raw state
        confirmation_sec = self._get_confirmation_duration(raw_state)

        # --- Temporal confirmation logic ---

        if raw_state == self._confirmed_state:
            # Already in this confirmed state — clear any candidate and recovery
            # Check if sustained DISTRACTED or NO_FACE should transition to AWAY_FROM_DESK
            if self._confirmed_state == BehaviorState.DISTRACTED:
                if (now - self._confirmed_state_started_at) >= self._away_from_desk_confirmation_sec:
                    self._confirmed_state = BehaviorState.AWAY_FROM_DESK
                    self._confirmed_reason = BehaviorReason.HEAD_AWAY
                    self._confirmed_state_started_at = now
            elif self._confirmed_state == BehaviorState.NO_FACE:
                if (now - self._confirmed_state_started_at) >= self._no_face_confirmation_sec:
                    self._confirmed_state = BehaviorState.AWAY_FROM_DESK
                    self._confirmed_reason = BehaviorReason.FACE_NOT_DETECTED
                    self._confirmed_state_started_at = now

            self._candidate_state = None
            self._candidate_reason = None
            self._candidate_started_at = 0.0
            self._recovery_started_at = None

        elif self._confirmed_state == BehaviorState.AWAY_FROM_DESK and raw_state in (BehaviorState.DISTRACTED, BehaviorState.NO_FACE):
            # Remain in AWAY_FROM_DESK as long as the head remains away or no face
            self._candidate_state = None
            self._candidate_reason = None
            self._candidate_started_at = 0.0
            self._recovery_started_at = None

        elif raw_state == self._candidate_state:
            # Same candidate persisting — check if confirmation period elapsed
            elapsed = now - self._candidate_started_at
            if elapsed >= confirmation_sec:
                # Confirm the new state
                self._confirmed_state = raw_state
                self._confirmed_reason = raw_reason
                self._confirmed_state_started_at = now
                self._candidate_state = None
                self._candidate_reason = None
                self._candidate_started_at = 0.0
                self._recovery_started_at = None

        else:
            # New candidate — start tracking
            self._candidate_state = raw_state
            self._candidate_reason = raw_reason
            self._candidate_started_at = now
            if confirmation_sec <= 0.0:
                self._confirmed_state = raw_state
                self._confirmed_reason = raw_reason
                self._confirmed_state_started_at = now
                self._candidate_state = None
                self._candidate_reason = None
                self._candidate_started_at = 0.0
                self._recovery_started_at = None

        # --- Recovery hysteresis for returning to FOCUSED ---
        if raw_state == BehaviorState.FOCUSED and self._confirmed_state != BehaviorState.FOCUSED:
            if self._confirmed_state == BehaviorState.UNKNOWN:
                # Immediate transition from initial UNKNOWN to FOCUSED upon valid input
                self._confirmed_state = BehaviorState.FOCUSED
                self._confirmed_reason = BehaviorReason.FOCUSED
                self._confirmed_state_started_at = now
                self._recovery_started_at = None
                self._candidate_state = None
                self._candidate_reason = None
                self._candidate_started_at = 0.0
            elif self._recovery_started_at is None:
                self._recovery_started_at = now
            elif (now - self._recovery_started_at) >= self._state_recovery_sec:
                self._confirmed_state = BehaviorState.FOCUSED
                self._confirmed_reason = BehaviorReason.FOCUSED
                self._confirmed_state_started_at = now
                self._recovery_started_at = None
                self._candidate_state = None
                self._candidate_reason = None
                self._candidate_started_at = 0.0
        elif raw_state != BehaviorState.FOCUSED:
            self._recovery_started_at = None

        # Calculate confidence (0.0 to 1.0)
        confidence = self._calculate_confidence(
            self._confirmed_state, face_result, eye_result, pose_result, phone_result
        )

        # Build result adhering to exact BehaviorResult contract
        state_duration = max(0.0, now - self._confirmed_state_started_at)
        return BehaviorResult(
            state=self._confirmed_state,
            state_started_at=self._confirmed_state_started_at,
            state_duration=state_duration,
            confidence=confidence,
            reason=str(self._confirmed_reason),
            signals=signals,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_raw_state(
        self,
        face_result: FaceDetectionResult,
        eye_result: EyeTrackingResult,
        pose_result: HeadPoseResult,
        phone_result: PhoneDetectionResult,
    ) -> tuple[BehaviorState, BehaviorReason]:
        """Evaluates the instantaneous (unconfirmed) state from raw signals.

        Evaluation Order:
            1. Availability gating:
               - No face detected -> NO_FACE (sensor availability condition)
               - Incomplete / invalid sensors -> UNKNOWN (signal validity condition)
            2. Canonical Behavioral Priority:
               - PHONE_DETECTED  (highest)
               - DROWSY          (eyes closed)
               - DISTRACTED      (head turned away)
               - FOCUSED         (attentive posture)
               Note: AWAY_FROM_DESK is a confirmed state arising from sustained
               absence of face or sustained head-away in temporal confirmation.
        """
        # Sensor Availability Check: No face in frame
        if not face_result.face_detected:
            return BehaviorState.NO_FACE, BehaviorReason.FACE_NOT_DETECTED

        # Signal Validity Check: Incomplete or unknown sensor signals
        if eye_result.eye_state == EyeState.UNKNOWN or pose_result.direction == HeadDirection.UNKNOWN:
            return BehaviorState.UNKNOWN, BehaviorReason.INSUFFICIENT_SIGNAL

        # Canonical Priority 1: Phone detected
        if phone_result.phone_detected:
            return BehaviorState.PHONE_DETECTED, BehaviorReason.PHONE_DETECTED

        # Canonical Priority 2: Eyes closed (drowsiness candidate)
        if eye_result.eye_state == EyeState.CLOSED:
            return BehaviorState.DROWSY, BehaviorReason.EYES_CLOSED

        # Canonical Priority 3: Head turned away (distraction candidate)
        if pose_result.direction != HeadDirection.CENTER:
            return BehaviorState.DISTRACTED, BehaviorReason.HEAD_AWAY

        # Canonical Priority 4: Normal attentive posture
        return BehaviorState.FOCUSED, BehaviorReason.FOCUSED

    def _get_confirmation_duration(self, state: BehaviorState) -> float:
        """Returns the temporal confirmation duration for a given state."""
        durations = {
            BehaviorState.PHONE_DETECTED: self._phone_confirmation_sec,
            BehaviorState.DROWSY: self._drowsiness_confirmation_sec,
            BehaviorState.AWAY_FROM_DESK: self._away_from_desk_confirmation_sec,
            BehaviorState.NO_FACE: 0.0,  # NO_FACE transitions immediately as raw state candidate
            BehaviorState.DISTRACTED: self._head_away_confirmation_sec,
            BehaviorState.FOCUSED: self._state_recovery_sec,
            BehaviorState.UNKNOWN: 0.0,
        }
        return durations.get(state, 0.0)

    @staticmethod
    def _build_signals(
        face_result: FaceDetectionResult,
        eye_result: EyeTrackingResult,
        pose_result: HeadPoseResult,
        phone_result: PhoneDetectionResult,
    ) -> Dict[str, object]:
        """Builds a snapshot dict of the current raw CV signal values."""
        return {
            "face_detected": face_result.face_detected,
            "eye_state": eye_result.eye_state.value,
            "smoothed_ear": eye_result.smoothed_ear,
            "head_direction": pose_result.direction.value,
            "yaw": pose_result.yaw,
            "pitch": pose_result.pitch,
            "phone_detected": phone_result.phone_detected,
            "phone_confidence": phone_result.confidence,
            "phone_cached": phone_result.is_cached,
        }

    @staticmethod
    def _calculate_confidence(
        state: BehaviorState,
        face_result: FaceDetectionResult,
        eye_result: EyeTrackingResult,
        pose_result: HeadPoseResult,
        phone_result: PhoneDetectionResult,
    ) -> float:
        """Computes a project-defined heuristic confidence score in [0.0, 1.0]."""
        if state == BehaviorState.PHONE_DETECTED:
            if phone_result.confidence is not None:
                return float(max(0.0, min(1.0, phone_result.confidence)))
            return 0.90
        elif state == BehaviorState.DROWSY:
            return 0.85 if eye_result.eye_state == EyeState.CLOSED else 0.70
        elif state == BehaviorState.AWAY_FROM_DESK:
            return 0.90 if not face_result.face_detected else 0.80
        elif state == BehaviorState.NO_FACE:
            return 0.95 if not face_result.face_detected else 0.50
        elif state == BehaviorState.DISTRACTED:
            return 0.85 if pose_result.direction != HeadDirection.CENTER else 0.70
        elif state == BehaviorState.FOCUSED:
            return 0.90 if face_result.face_detected and eye_result.eye_state == EyeState.OPEN else 0.75
        elif state == BehaviorState.UNKNOWN:
            return 0.0
        return 0.50

