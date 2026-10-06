"""Tests for vision/eye_tracker.py.

All tests use synthetic landmark data and do NOT require a physical webcam
or the MediaPipe model file.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import EyeTrackerConfig
from vision.eye_tracker import (
    EyeState,
    EyeTracker,
    EyeTrackingResult,
    compute_ear,
    _euclidean,
    LEFT_EYE_EAR_INDICES,
    RIGHT_EYE_EAR_INDICES,
)
from vision.face_detector import FaceDetectionResult


# ---------------------------------------------------------------------------
# Helpers — create synthetic MediaPipe-style landmarks
# ---------------------------------------------------------------------------

def _make_landmark(x: float, y: float, z: float = 0.0):
    """Creates a minimal object mimicking NormalizedLandmark."""
    return SimpleNamespace(x=x, y=y, z=z)


def _make_landmarks_wide_open():
    """Returns a 468-element list of landmarks with wide-open eye geometry.

    The EAR keypoints are placed to produce an EAR of approximately 0.35–0.40
    (clearly above the default open threshold of 0.22).
    """
    lms = [_make_landmark(0.5, 0.5) for _ in range(468)]

    # Right eye: p1=33, p2=160, p3=158, p4=133, p5=153, p6=144
    lms[33] = _make_landmark(0.35, 0.40)   # p1 outer corner
    lms[133] = _make_landmark(0.45, 0.40)  # p4 inner corner
    lms[160] = _make_landmark(0.38, 0.36)  # p2 upper-outer
    lms[158] = _make_landmark(0.42, 0.36)  # p3 upper-inner
    lms[153] = _make_landmark(0.42, 0.44)  # p5 lower-inner
    lms[144] = _make_landmark(0.38, 0.44)  # p6 lower-outer

    # Left eye: p1=362, p2=385, p3=387, p4=263, p5=373, p6=380
    lms[362] = _make_landmark(0.55, 0.40)  # p1 outer corner
    lms[263] = _make_landmark(0.65, 0.40)  # p4 inner corner
    lms[385] = _make_landmark(0.58, 0.36)  # p2 upper-outer
    lms[387] = _make_landmark(0.62, 0.36)  # p3 upper-inner
    lms[373] = _make_landmark(0.62, 0.44)  # p5 lower-inner
    lms[380] = _make_landmark(0.58, 0.44)  # p6 lower-outer

    return lms


def _make_landmarks_closed():
    """Returns landmarks with nearly-closed eye geometry.

    Vertical distances are very small → EAR ≈ 0.05 (clearly below 0.18).
    """
    lms = [_make_landmark(0.5, 0.5) for _ in range(468)]

    # Right eye — very narrow vertical opening
    lms[33] = _make_landmark(0.35, 0.40)
    lms[133] = _make_landmark(0.45, 0.40)
    lms[160] = _make_landmark(0.38, 0.395)
    lms[158] = _make_landmark(0.42, 0.395)
    lms[153] = _make_landmark(0.42, 0.405)
    lms[144] = _make_landmark(0.38, 0.405)

    # Left eye — same narrow opening
    lms[362] = _make_landmark(0.55, 0.40)
    lms[263] = _make_landmark(0.65, 0.40)
    lms[385] = _make_landmark(0.58, 0.395)
    lms[387] = _make_landmark(0.62, 0.395)
    lms[373] = _make_landmark(0.62, 0.405)
    lms[380] = _make_landmark(0.58, 0.405)

    return lms


def _make_face_result(landmarks):
    """Wraps synthetic landmarks into a FaceDetectionResult-like structure."""
    raw = SimpleNamespace(face_landmarks=[landmarks])
    return FaceDetectionResult(face_detected=True, raw_result=raw)


# ---------------------------------------------------------------------------
# Tests: _euclidean
# ---------------------------------------------------------------------------

class TestEuclidean:
    def test_zero_distance(self):
        assert _euclidean(1.0, 2.0, 1.0, 2.0) == 0.0

    def test_horizontal(self):
        assert _euclidean(0.0, 0.0, 3.0, 0.0) == pytest.approx(3.0)

    def test_diagonal(self):
        assert _euclidean(0.0, 0.0, 3.0, 4.0) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Tests: compute_ear
# ---------------------------------------------------------------------------

class TestComputeEar:
    def test_open_eye_ear(self):
        lms = _make_landmarks_wide_open()
        ear = compute_ear(lms, RIGHT_EYE_EAR_INDICES)
        assert ear is not None
        assert 0.25 < ear < 1.0  # wide-open should give high EAR

    def test_closed_eye_ear(self):
        lms = _make_landmarks_closed()
        ear = compute_ear(lms, RIGHT_EYE_EAR_INDICES)
        assert ear is not None
        assert ear < 0.15  # nearly closed should give low EAR

    def test_open_higher_than_closed(self):
        open_ear = compute_ear(_make_landmarks_wide_open(), RIGHT_EYE_EAR_INDICES)
        closed_ear = compute_ear(_make_landmarks_closed(), RIGHT_EYE_EAR_INDICES)
        assert open_ear > closed_ear

    def test_left_and_right_independent(self):
        lms = _make_landmarks_wide_open()
        left_ear = compute_ear(lms, LEFT_EYE_EAR_INDICES)
        right_ear = compute_ear(lms, RIGHT_EYE_EAR_INDICES)
        assert left_ear is not None
        assert right_ear is not None
        # Both eyes are symmetrically placed so EAR values should be similar
        assert abs(left_ear - right_ear) < 0.05

    def test_none_landmarks(self):
        assert compute_ear(None, RIGHT_EYE_EAR_INDICES) is None

    def test_empty_landmarks(self):
        assert compute_ear([], RIGHT_EYE_EAR_INDICES) is None

    def test_too_few_landmarks(self):
        lms = [_make_landmark(0.5, 0.5) for _ in range(10)]
        assert compute_ear(lms, RIGHT_EYE_EAR_INDICES) is None

    def test_zero_horizontal_distance(self):
        lms = [_make_landmark(0.5, 0.5) for _ in range(468)]
        # p1 and p4 at the exact same point → horizontal = 0
        lms[33] = _make_landmark(0.4, 0.4)
        lms[133] = _make_landmark(0.4, 0.4)
        ear = compute_ear(lms, RIGHT_EYE_EAR_INDICES, min_width=0.001)
        assert ear is None  # should return None, not crash

    def test_very_small_horizontal_distance(self):
        lms = [_make_landmark(0.5, 0.5) for _ in range(468)]
        lms[33] = _make_landmark(0.4, 0.4)
        lms[133] = _make_landmark(0.4 + 0.0001, 0.4)  # below default min_width
        ear = compute_ear(lms, RIGHT_EYE_EAR_INDICES, min_width=0.001)
        assert ear is None


# ---------------------------------------------------------------------------
# Tests: EyeTracker.process — state classification
# ---------------------------------------------------------------------------

class TestEyeTrackerProcess:
    def test_no_face_returns_unknown(self):
        tracker = EyeTracker()
        result = tracker.process(FaceDetectionResult(face_detected=False))
        assert result.eye_state is EyeState.UNKNOWN
        assert result.left_ear is None
        assert result.right_ear is None
        assert result.average_ear is None

    def test_open_eyes_detected(self):
        tracker = EyeTracker()
        face_result = _make_face_result(_make_landmarks_wide_open())
        result = tracker.process(face_result)
        assert result.eye_state is EyeState.OPEN
        assert result.left_ear is not None
        assert result.right_ear is not None
        assert result.average_ear > 0.22

    def test_closed_eyes_detected(self):
        tracker = EyeTracker()
        face_result = _make_face_result(_make_landmarks_closed())
        result = tracker.process(face_result)
        assert result.eye_state is EyeState.CLOSED
        assert result.average_ear < 0.18

    def test_average_ear_is_mean_of_left_right(self):
        tracker = EyeTracker()
        face_result = _make_face_result(_make_landmarks_wide_open())
        result = tracker.process(face_result)
        expected = (result.left_ear + result.right_ear) / 2.0
        assert result.average_ear == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Tests: Hysteresis
# ---------------------------------------------------------------------------

class TestHysteresis:
    def test_hysteresis_retains_open_in_band(self):
        """EAR in the hysteresis band should retain the previous OPEN state."""
        cfg = EyeTrackerConfig(
            ear_close_threshold=0.18,
            ear_open_threshold=0.22,
            ear_smoothing_alpha=1.0,  # no smoothing for deterministic test
        )
        tracker = EyeTracker(tracker_config=cfg)

        # First: clearly open
        result = tracker.process(_make_face_result(_make_landmarks_wide_open()))
        assert result.eye_state is EyeState.OPEN

        # Now feed EAR in the hysteresis band (0.18 < EAR < 0.22)
        # Create landmarks with EAR ≈ 0.20
        lms = _make_landmarks_wide_open()
        # Reduce vertical opening to get ~0.20 EAR
        lms[160] = _make_landmark(0.38, 0.39)
        lms[158] = _make_landmark(0.42, 0.39)
        lms[153] = _make_landmark(0.42, 0.41)
        lms[144] = _make_landmark(0.38, 0.41)
        lms[385] = _make_landmark(0.58, 0.39)
        lms[387] = _make_landmark(0.62, 0.39)
        lms[373] = _make_landmark(0.62, 0.41)
        lms[380] = _make_landmark(0.58, 0.41)

        result2 = tracker.process(_make_face_result(lms))
        # Should retain OPEN because EAR is in the hysteresis band
        assert result2.eye_state is EyeState.OPEN

    def test_hysteresis_retains_closed_in_band(self):
        """EAR in the hysteresis band should retain the previous CLOSED state."""
        cfg = EyeTrackerConfig(
            ear_close_threshold=0.18,
            ear_open_threshold=0.22,
            ear_smoothing_alpha=1.0,
        )
        tracker = EyeTracker(tracker_config=cfg)

        # First: clearly closed
        result = tracker.process(_make_face_result(_make_landmarks_closed()))
        assert result.eye_state is EyeState.CLOSED

        # Feed EAR in hysteresis band (~0.20)
        lms = _make_landmarks_wide_open()
        lms[160] = _make_landmark(0.38, 0.39)
        lms[158] = _make_landmark(0.42, 0.39)
        lms[153] = _make_landmark(0.42, 0.41)
        lms[144] = _make_landmark(0.38, 0.41)
        lms[385] = _make_landmark(0.58, 0.39)
        lms[387] = _make_landmark(0.62, 0.39)
        lms[373] = _make_landmark(0.62, 0.41)
        lms[380] = _make_landmark(0.58, 0.41)

        result2 = tracker.process(_make_face_result(lms))
        assert result2.eye_state is EyeState.CLOSED


# ---------------------------------------------------------------------------
# Tests: EMA smoothing
# ---------------------------------------------------------------------------

class TestSmoothing:
    def test_alpha_1_no_smoothing(self):
        """With alpha=1.0, smoothed_ear should equal the raw average_ear."""
        cfg = EyeTrackerConfig(ear_smoothing_alpha=1.0)
        tracker = EyeTracker(tracker_config=cfg)
        face_result = _make_face_result(_make_landmarks_wide_open())
        result = tracker.process(face_result)
        assert result.smoothed_ear == pytest.approx(result.average_ear)

    def test_smoothing_dampens_change(self):
        """With alpha<1.0, the smoothed value should lag behind a sudden change."""
        cfg = EyeTrackerConfig(ear_smoothing_alpha=0.3)
        tracker = EyeTracker(tracker_config=cfg)

        # Several frames with open eyes to establish baseline
        open_result = None
        for _ in range(5):
            open_result = tracker.process(_make_face_result(_make_landmarks_wide_open()))

        # Sudden change to closed
        closed_result = tracker.process(_make_face_result(_make_landmarks_closed()))

        # Smoothed EAR should be higher than raw closed EAR (lag effect)
        assert closed_result.smoothed_ear > closed_result.average_ear


# ---------------------------------------------------------------------------
# Tests: Reset on face loss
# ---------------------------------------------------------------------------

class TestResetOnFaceLoss:
    def test_no_face_resets_state(self):
        tracker = EyeTracker()
        tracker.process(_make_face_result(_make_landmarks_wide_open()))
        # Now face disappears
        result = tracker.process(FaceDetectionResult(face_detected=False))
        assert result.eye_state is EyeState.UNKNOWN
        assert result.smoothed_ear is None

    def test_face_returns_after_loss(self):
        tracker = EyeTracker()
        tracker.process(_make_face_result(_make_landmarks_wide_open()))
        tracker.process(FaceDetectionResult(face_detected=False))
        # Face comes back
        result = tracker.process(_make_face_result(_make_landmarks_wide_open()))
        assert result.eye_state is EyeState.OPEN
        assert result.smoothed_ear is not None


# ---------------------------------------------------------------------------
# Tests: EyeTrackingResult dataclass
# ---------------------------------------------------------------------------

class TestEyeTrackingResult:
    def test_defaults(self):
        r = EyeTrackingResult()
        assert r.left_ear is None
        assert r.right_ear is None
        assert r.average_ear is None
        assert r.smoothed_ear is None
        assert r.eye_state is EyeState.UNKNOWN


# ---------------------------------------------------------------------------
# Tests: EyeTracker initialization and configuration validation
# ---------------------------------------------------------------------------

class TestEyeTrackerInit:
    def test_init_with_valid_config(self):
        cfg = EyeTrackerConfig(ear_close_threshold=0.15, ear_open_threshold=0.25)
        tracker = EyeTracker(tracker_config=cfg)
        assert tracker._cfg == cfg

    def test_init_with_invalid_config_raises(self):
        cfg = EyeTrackerConfig(ear_close_threshold=0.30, ear_open_threshold=0.20)
        with pytest.raises(ValueError):
            EyeTracker(tracker_config=cfg)

