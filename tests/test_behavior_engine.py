"""Unit tests for behavior/behavior_engine.py.

Verifies priority evaluation, temporal confirmation, blink rejection,
sustained eye closure (drowsiness), head movement (temporary vs sustained),
phone detection (temporary vs sustained), recovery hysteresis, and handling
of invalid/missing sensor signals using an injectable deterministic clock.
"""

import pytest

from behavior.behavior_engine import (
    BehaviorEngine,
    BehaviorResult,
    BehaviorState,
    BehaviorReason,
)
from vision.face_detector import FaceDetectionResult
from vision.eye_tracker import EyeTrackingResult, EyeState
from vision.head_pose import HeadPoseResult, HeadDirection
from detection.phone_detector import PhoneDetectionResult, PhoneBoundingBox


class MockClock:
    """Deterministic injectable clock for testing without sleep or real-time delays."""

    def __init__(self, start_time: float = 100.0) -> None:
        self.time = start_time

    def __call__(self) -> float:
        return self.time

    def advance(self, seconds: float) -> None:
        self.time += seconds


def make_signals(
    face_detected: bool = True,
    eye_state: EyeState = EyeState.OPEN,
    average_ear: float = 0.28,
    head_direction: HeadDirection = HeadDirection.CENTER,
    yaw: float = 0.0,
    pitch: float = 0.0,
    phone_detected: bool = False,
    phone_confidence: float = 0.0,
):
    """Helper to generate synthetic sensor results."""
    face_res = FaceDetectionResult(face_detected=face_detected)
    eye_res = EyeTrackingResult(
        left_ear=average_ear,
        right_ear=average_ear,
        average_ear=average_ear,
        smoothed_ear=average_ear,
        eye_state=eye_state,
    )
    pose_res = HeadPoseResult(
        pitch=pitch,
        yaw=yaw,
        roll=0.0,
        direction=head_direction,
    )
    detections = (
        [PhoneBoundingBox(10, 10, 50, 50, phone_confidence)]
        if phone_detected
        else []
    )
    phone_res = PhoneDetectionResult(
        phone_detected=phone_detected,
        confidence=phone_confidence if phone_detected else None,
        bbox=(10, 10, 50, 50) if phone_detected else None,
        detections=detections,
        inference_time_ms=25.0,
        is_cached=False,
    )
    return face_res, eye_res, pose_res, phone_res


class TestBehaviorEngineInitAndReset:
    def test_initial_state_unknown(self):
        clock = MockClock()
        engine = BehaviorEngine(clock=clock)
        assert engine.current_state == BehaviorState.UNKNOWN

    def test_reset_clears_state(self):
        clock = MockClock()
        engine = BehaviorEngine(clock=clock)
        face, eye, pose, phone = make_signals()
        engine.update(face, eye, pose, phone)
        assert engine.current_state == BehaviorState.FOCUSED

        engine.reset()
        assert engine.current_state == BehaviorState.UNKNOWN


class TestNoFaceHandling:
    def test_no_face_classified_as_no_face(self):
        clock = MockClock()
        engine = BehaviorEngine(clock=clock)
        face, eye, pose, phone = make_signals(face_detected=False)

        res = engine.update(face, eye, pose, phone)
        assert res.state == BehaviorState.NO_FACE
        assert res.reason == BehaviorReason.FACE_NOT_DETECTED

    def test_distinction_between_no_face_and_away_from_desk(self):
        """NO_FACE is immediate sensor unavailability; AWAY_FROM_DESK requires sustained absence."""
        clock = MockClock()
        engine = BehaviorEngine(clock=clock, no_face_confirmation_sec=3.0)
        face, eye, pose, phone = make_signals(face_detected=False)

        # Immediate check: Must be NO_FACE, NOT AWAY_FROM_DESK
        res = engine.update(face, eye, pose, phone)
        assert res.state == BehaviorState.NO_FACE
        assert res.reason == BehaviorReason.FACE_NOT_DETECTED

        # Progress time to 2.0s (< 3.0s confirmation): Still NO_FACE
        clock.advance(2.0)
        res = engine.update(face, eye, pose, phone)
        assert res.state == BehaviorState.NO_FACE

        # Progress beyond 3.0s: Confirmed AWAY_FROM_DESK
        clock.advance(1.1)
        res = engine.update(face, eye, pose, phone)
        assert res.state == BehaviorState.AWAY_FROM_DESK
        assert res.reason == BehaviorReason.FACE_NOT_DETECTED

    def test_face_disappears_then_returns(self):
        """Test user leaving desk (face disappears) and returning (face detected again)."""
        clock = MockClock()
        engine = BehaviorEngine(
            clock=clock,
            no_face_confirmation_sec=2.0,
            state_recovery_sec=1.0,
        )

        # 1. Baseline FOCUSED
        res = engine.update(*make_signals())
        assert res.state == BehaviorState.FOCUSED

        # 2. Face disappears
        clock.advance(0.1)
        res = engine.update(*make_signals(face_detected=False))
        assert res.state == BehaviorState.NO_FACE

        # 3. Sustained disappearance -> AWAY_FROM_DESK
        clock.advance(2.5)
        res = engine.update(*make_signals(face_detected=False))
        assert res.state == BehaviorState.AWAY_FROM_DESK

        # 4. Face returns
        clock.advance(0.1)
        res = engine.update(*make_signals(face_detected=True))
        # Immediate return is in recovery period
        assert res.state == BehaviorState.AWAY_FROM_DESK

        # 5. Sustained return after recovery cooldown -> FOCUSED
        clock.advance(1.1)
        res = engine.update(*make_signals(face_detected=True))
        assert res.state == BehaviorState.FOCUSED
        assert res.reason == BehaviorReason.FOCUSED


class TestNormalPosture:
    def test_normal_signals_yield_focused(self):
        clock = MockClock()
        engine = BehaviorEngine(clock=clock)
        face, eye, pose, phone = make_signals(
            face_detected=True,
            eye_state=EyeState.OPEN,
            head_direction=HeadDirection.CENTER,
            phone_detected=False,
        )

        res = engine.update(face, eye, pose, phone)
        assert res.state == BehaviorState.FOCUSED
        assert res.reason == BehaviorReason.NORMAL
        assert res.state_duration == 0.0


class TestHeadMovementAndDistraction:
    def test_brief_head_turn_does_not_trigger_distracted(self):
        clock = MockClock()
        engine = BehaviorEngine(clock=clock, head_away_confirmation_sec=1.5)

        # Baseline FOCUSED
        face, eye, pose, phone = make_signals()
        res = engine.update(face, eye, pose, phone)
        assert res.state == BehaviorState.FOCUSED

        # Turn head left briefly
        clock.advance(0.1)
        face, eye, pose, phone = make_signals(head_direction=HeadDirection.LEFT, yaw=25.0)
        res = engine.update(face, eye, pose, phone)
        assert res.state == BehaviorState.FOCUSED  # Not yet confirmed

        # Advance less than confirmation duration
        clock.advance(0.8)
        res = engine.update(face, eye, pose, phone)
        assert res.state == BehaviorState.FOCUSED  # Still FOCUSED

    def test_sustained_head_turn_triggers_distracted(self):
        clock = MockClock()
        engine = BehaviorEngine(clock=clock, head_away_confirmation_sec=1.5)

        # Baseline FOCUSED
        face, eye, pose, phone = make_signals()
        engine.update(face, eye, pose, phone)

        # Turn head right continuously
        clock.advance(0.1)
        face, eye, pose, phone = make_signals(head_direction=HeadDirection.RIGHT, yaw=-25.0)
        engine.update(face, eye, pose, phone)

        clock.advance(1.5)
        res = engine.update(face, eye, pose, phone)
        assert res.state == BehaviorState.DISTRACTED
        assert res.reason == BehaviorReason.HEAD_AWAY

    def test_sustained_distraction_becomes_away_from_desk(self):
        clock = MockClock()
        engine = BehaviorEngine(
            clock=clock,
            head_away_confirmation_sec=1.0,
            away_from_desk_confirmation_sec=3.0,
        )

        # Establish DISTRACTED
        face, eye, pose, phone = make_signals()
        engine.update(face, eye, pose, phone)

        clock.advance(0.1)
        face, eye, pose, phone = make_signals(head_direction=HeadDirection.LEFT, yaw=30.0)
        engine.update(face, eye, pose, phone)

        clock.advance(1.0)
        res = engine.update(face, eye, pose, phone)
        assert res.state == BehaviorState.DISTRACTED

        # Advance past away_from_desk_confirmation_sec
        clock.advance(3.1)
        res = engine.update(face, eye, pose, phone)
        assert res.state == BehaviorState.AWAY_FROM_DESK
        assert res.reason == BehaviorReason.AWAY_FROM_DESK

    def test_return_center_recovers_to_focused_after_cooldown(self):
        clock = MockClock()
        engine = BehaviorEngine(
            clock=clock,
            head_away_confirmation_sec=1.0,
            state_recovery_sec=1.0,
        )

        # Put in DISTRACTED
        face, eye, pose, phone = make_signals()
        engine.update(face, eye, pose, phone)
        clock.advance(0.1)
        face, eye, pose, phone = make_signals(head_direction=HeadDirection.LEFT)
        engine.update(face, eye, pose, phone)
        clock.advance(1.1)
        res = engine.update(face, eye, pose, phone)
        assert res.state == BehaviorState.DISTRACTED

        # Return to center
        clock.advance(0.1)
        face, eye, pose, phone = make_signals(head_direction=HeadDirection.CENTER)
        res = engine.update(face, eye, pose, phone)
        assert res.state == BehaviorState.DISTRACTED  # Cooldown active

        # Advance past recovery duration
        clock.advance(1.0)
        res = engine.update(face, eye, pose, phone)
        assert res.state == BehaviorState.FOCUSED
        assert res.reason == BehaviorReason.NORMAL


class TestEyeClosureAndDrowsiness:
    def test_brief_blink_does_not_trigger_drowsy(self):
        clock = MockClock()
        engine = BehaviorEngine(clock=clock, drowsiness_confirmation_sec=1.5)

        # Baseline FOCUSED
        face, eye, pose, phone = make_signals()
        engine.update(face, eye, pose, phone)

        # Blink (eyes closed for 250ms)
        clock.advance(0.1)
        face, eye, pose, phone = make_signals(eye_state=EyeState.CLOSED, average_ear=0.12)
        res = engine.update(face, eye, pose, phone)
        assert res.state == BehaviorState.FOCUSED

        clock.advance(0.25)
        res = engine.update(face, eye, pose, phone)
        assert res.state == BehaviorState.FOCUSED

        # Reopen eyes
        clock.advance(0.05)
        face, eye, pose, phone = make_signals(eye_state=EyeState.OPEN, average_ear=0.28)
        res = engine.update(face, eye, pose, phone)
        assert res.state == BehaviorState.FOCUSED

    def test_sustained_eye_closure_triggers_drowsy(self):
        clock = MockClock()
        engine = BehaviorEngine(clock=clock, drowsiness_confirmation_sec=1.5)

        # Baseline FOCUSED
        face, eye, pose, phone = make_signals()
        engine.update(face, eye, pose, phone)

        # Close eyes continuously
        clock.advance(0.1)
        face, eye, pose, phone = make_signals(eye_state=EyeState.CLOSED, average_ear=0.12)
        engine.update(face, eye, pose, phone)

        clock.advance(1.5)
        res = engine.update(face, eye, pose, phone)
        assert res.state == BehaviorState.DROWSY
        assert res.reason == BehaviorReason.EYES_CLOSED


class TestPhoneDetectionBehavior:
    def test_brief_phone_does_not_immediately_trigger(self):
        clock = MockClock()
        engine = BehaviorEngine(clock=clock, phone_confirmation_sec=1.0)

        # Baseline FOCUSED
        face, eye, pose, phone = make_signals()
        engine.update(face, eye, pose, phone)

        # Flash phone for 300ms
        clock.advance(0.1)
        face, eye, pose, phone = make_signals(phone_detected=True, phone_confidence=0.88)
        res = engine.update(face, eye, pose, phone)
        assert res.state == BehaviorState.FOCUSED

        clock.advance(0.3)
        res = engine.update(face, eye, pose, phone)
        assert res.state == BehaviorState.FOCUSED

    def test_sustained_phone_triggers_phone_detected(self):
        clock = MockClock()
        engine = BehaviorEngine(clock=clock, phone_confirmation_sec=1.0)

        face, eye, pose, phone = make_signals()
        engine.update(face, eye, pose, phone)

        clock.advance(0.1)
        face, eye, pose, phone = make_signals(phone_detected=True, phone_confidence=0.88)
        engine.update(face, eye, pose, phone)

        clock.advance(1.0)
        res = engine.update(face, eye, pose, phone)
        assert res.state == BehaviorState.PHONE_DETECTED
        assert res.reason == BehaviorReason.PHONE_DETECTED

    def test_phone_removal_recovers_to_focused(self):
        clock = MockClock()
        engine = BehaviorEngine(clock=clock, phone_confirmation_sec=1.0, state_recovery_sec=1.0)

        face, eye, pose, phone = make_signals()
        engine.update(face, eye, pose, phone)

        clock.advance(0.1)
        face, eye, pose, phone = make_signals(phone_detected=True, phone_confidence=0.88)
        engine.update(face, eye, pose, phone)
        clock.advance(1.0)
        res = engine.update(face, eye, pose, phone)
        assert res.state == BehaviorState.PHONE_DETECTED

        # Remove phone
        clock.advance(0.1)
        face, eye, pose, phone = make_signals(phone_detected=False)
        res = engine.update(face, eye, pose, phone)
        assert res.state == BehaviorState.PHONE_DETECTED  # Cooldown active

        clock.advance(1.0)
        res = engine.update(face, eye, pose, phone)
        assert res.state == BehaviorState.FOCUSED


class TestPriorityResolution:
    def test_phone_over_drowsiness(self):
        """PHONE_DETECTED has higher priority than DROWSY."""
        clock = MockClock()
        engine = BehaviorEngine(clock=clock, phone_confirmation_sec=0.5, drowsiness_confirmation_sec=0.5)

        face, eye, pose, phone = make_signals(
            eye_state=EyeState.CLOSED,
            phone_detected=True,
            phone_confidence=0.92,
        )
        engine.update(face, eye, pose, phone)
        clock.advance(0.6)
        res = engine.update(face, eye, pose, phone)
        assert res.state == BehaviorState.PHONE_DETECTED

    def test_drowsiness_over_head_away(self):
        """DROWSY has higher priority than DISTRACTED."""
        clock = MockClock()
        engine = BehaviorEngine(clock=clock, drowsiness_confirmation_sec=0.5, head_away_confirmation_sec=0.5)

        face, eye, pose, phone = make_signals(
            eye_state=EyeState.CLOSED,
            head_direction=HeadDirection.LEFT,
        )
        engine.update(face, eye, pose, phone)
        clock.advance(0.6)
        res = engine.update(face, eye, pose, phone)
        assert res.state == BehaviorState.DROWSY

    def test_phone_over_head_away(self):
        """PHONE_DETECTED has higher priority than DISTRACTED."""
        clock = MockClock()
        engine = BehaviorEngine(clock=clock, phone_confirmation_sec=0.5, head_away_confirmation_sec=0.5)

        face, eye, pose, phone = make_signals(
            phone_detected=True,
            phone_confidence=0.85,
            head_direction=HeadDirection.RIGHT,
        )
        engine.update(face, eye, pose, phone)
        clock.advance(0.6)
        res = engine.update(face, eye, pose, phone)
        assert res.state == BehaviorState.PHONE_DETECTED


class TestInvalidAndUnknownInputs:
    def test_unknown_eye_state_yields_unknown(self):
        clock = MockClock()
        engine = BehaviorEngine(clock=clock)
        face, eye, pose, phone = make_signals(eye_state=EyeState.UNKNOWN)

        res = engine.update(face, eye, pose, phone)
        assert res.state == BehaviorState.UNKNOWN
        assert res.reason == BehaviorReason.INSUFFICIENT_DATA

    def test_unknown_head_direction_yields_unknown(self):
        clock = MockClock()
        engine = BehaviorEngine(clock=clock)
        face, eye, pose, phone = make_signals(head_direction=HeadDirection.UNKNOWN)

        res = engine.update(face, eye, pose, phone)
        assert res.state == BehaviorState.UNKNOWN
        assert res.reason == BehaviorReason.INSUFFICIENT_DATA

    def test_signals_dictionary_snapshot(self):
        clock = MockClock()
        engine = BehaviorEngine(clock=clock)
        face, eye, pose, phone = make_signals(average_ear=0.29, yaw=12.3, pitch=-4.5)

        res = engine.update(face, eye, pose, phone)
        assert "face_detected" in res.signals
        assert res.signals["face_detected"] is True
        assert res.signals["eye_state"] == "OPEN"
        assert res.signals["smoothed_ear"] == 0.29
        assert res.signals["head_direction"] == "CENTER"
        assert res.signals["yaw"] == 12.3
        assert res.signals["pitch"] == -4.5
        assert res.signals["phone_detected"] is False


class TestBehaviorResultContract:
    """Verifies that BehaviorResult satisfies all contract invariants specified for Phase 6."""

    def test_behavior_state_enum_exact_members(self):
        """BehaviorState must contain exactly: FOCUSED, DISTRACTED, DROWSY, PHONE_DETECTED, AWAY_FROM_DESK, NO_FACE, UNKNOWN."""
        expected_members = {
            "FOCUSED",
            "DISTRACTED",
            "DROWSY",
            "PHONE_DETECTED",
            "AWAY_FROM_DESK",
            "NO_FACE",
            "UNKNOWN",
        }
        actual_members = {member.name for member in BehaviorState}
        assert actual_members == expected_members

    def test_contract_invariants_across_all_states(self):
        """Contract invariants:
        - state must always be a valid BehaviorState
        - state_started_at must be float or None
        - state_duration must be >= 0
        - confidence must always be between 0.0 and 1.0
        - reason must be a stable reason code string, not a UI sentence
        - signals must be a dictionary
        """
        clock = MockClock(start_time=50.0)
        engine = BehaviorEngine(clock=clock, phone_confirmation_sec=0.1, drowsiness_confirmation_sec=0.1)

        valid_reason_codes = {
            BehaviorReason.FOCUSED,
            BehaviorReason.PHONE_DETECTED,
            BehaviorReason.EYES_CLOSED,
            BehaviorReason.HEAD_AWAY,
            BehaviorReason.FACE_NOT_DETECTED,
            BehaviorReason.INSUFFICIENT_SIGNAL,
            BehaviorReason.INVALID_INPUT,
        }

        # 1. Initial UNKNOWN state
        res_init = engine.update(*make_signals(eye_state=EyeState.UNKNOWN))
        assert isinstance(res_init.state, BehaviorState)
        assert res_init.state_started_at is not None and isinstance(res_init.state_started_at, float)
        assert res_init.state_duration >= 0.0
        assert 0.0 <= res_init.confidence <= 1.0
        assert res_init.reason in valid_reason_codes
        assert isinstance(res_init.signals, dict)

        # 2. FOCUSED state
        clock.advance(1.0)
        res_focused = engine.update(*make_signals())
        assert res_focused.state == BehaviorState.FOCUSED
        assert res_focused.state_duration >= 0.0
        assert 0.0 <= res_focused.confidence <= 1.0
        assert res_focused.reason in valid_reason_codes

        # 3. NO_FACE state
        clock.advance(1.0)
        res_noface = engine.update(*make_signals(face_detected=False))
        assert res_noface.state == BehaviorState.NO_FACE
        assert res_noface.state_duration >= 0.0
        assert 0.0 <= res_noface.confidence <= 1.0
        assert res_noface.reason in valid_reason_codes

        # 4. PHONE_DETECTED state
        clock.advance(1.0)
        engine.update(*make_signals(phone_detected=True, phone_confidence=0.88))
        clock.advance(0.2)
        res_phone = engine.update(*make_signals(phone_detected=True, phone_confidence=0.88))
        assert res_phone.state == BehaviorState.PHONE_DETECTED
        assert res_phone.state_duration >= 0.0
        assert 0.0 <= res_phone.confidence <= 1.0
        assert res_phone.reason in valid_reason_codes

        # 5. DROWSY state
        clock.advance(1.0)
        engine.update(*make_signals(eye_state=EyeState.CLOSED))
        clock.advance(0.2)
        res_drowsy = engine.update(*make_signals(eye_state=EyeState.CLOSED))
        assert res_drowsy.state == BehaviorState.DROWSY
        assert res_drowsy.state_duration >= 0.0
        assert 0.0 <= res_drowsy.confidence <= 1.0
        assert res_drowsy.reason in valid_reason_codes

