"""Unit tests for vision/head_pose.py.

All tests use synthetic geometric and landmark coordinates.
They do NOT require a physical webcam or external model assets.
"""

from types import SimpleNamespace
import math
import cv2
import numpy as np
import pytest

from config.config import HeadPoseConfig
from vision.head_pose import (
    HeadDirection,
    HeadPoseEstimator,
    HeadPoseResult,
    build_camera_matrix,
    classify_direction,
    rotation_matrix_to_euler_angles,
    CANONICAL_FACE_MODEL_3D,
    HEAD_POSE_LANDMARK_INDICES,
)
from vision.face_detector import FaceDetectionResult


# ---------------------------------------------------------------------------
# Synthetic Landmark Helpers
# ---------------------------------------------------------------------------

def _make_lm(x: float, y: float, z: float = 0.0):
    return SimpleNamespace(x=x, y=y, z=z)


def _make_projected_landmarks(pitch_deg: float, yaw_deg: float, roll_deg: float = 0.0, width: int = 640, height: int = 480):
    """Generates synthetic landmarks by projecting the 3D canonical model with exact Euler rotation."""
    K = build_camera_matrix(width, height)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    tvec = np.array([[0.0], [0.0], [1000.0]], dtype=np.float64)

    # In camera coordinates, user-centric yaw > 0 corresponds to -yaw around Y axis
    rvec = np.array(
        [[np.radians(pitch_deg)], [-np.radians(yaw_deg)], [np.radians(roll_deg)]],
        dtype=np.float64,
    )
    pts_2d, _ = cv2.projectPoints(CANONICAL_FACE_MODEL_3D, rvec, tvec, K, dist_coeffs)
    pts_2d = pts_2d.reshape(-1, 2)

    lms = [_make_lm(0.5, 0.5, 0.0) for _ in range(468)]
    for i, idx in enumerate(HEAD_POSE_LANDMARK_INDICES):
        lms[idx] = _make_lm(pts_2d[i][0] / width, pts_2d[i][1] / height, 0.0)
    return lms


def _make_face_result(landmarks):
    raw = SimpleNamespace(face_landmarks=[landmarks])
    return FaceDetectionResult(face_detected=True, raw_result=raw)


# ---------------------------------------------------------------------------
# Tests: Camera Matrix & Euler Angles
# ---------------------------------------------------------------------------

class TestEulerAndGeometry:
    def test_build_camera_matrix(self):
        K = build_camera_matrix(640, 480)
        assert K.shape == (3, 3)
        assert K[0, 0] == 640.0
        assert K[1, 1] == 640.0
        assert K[0, 2] == 320.0
        assert K[1, 2] == 240.0
        assert K[2, 2] == 1.0

    def test_identity_rotation(self):
        R = np.eye(3, dtype=np.float64)
        pitch, yaw, roll = rotation_matrix_to_euler_angles(R)
        assert abs(pitch) < 1e-4
        assert abs(yaw) < 1e-4
        assert abs(roll) < 1e-4

    def test_pure_yaw_rotation(self):
        theta = math.radians(20.0)
        # In our user-centric convention, R around Y axis gives positive yaw
        R = np.array([
            [math.cos(theta), 0, math.sin(theta)],
            [0, 1, 0],
            [-math.sin(theta), 0, math.cos(theta)],
        ], dtype=np.float64)
        pitch, yaw, roll = rotation_matrix_to_euler_angles(R)
        # atan2(R[2, 0], sy) with R[2, 0] = -sin(theta)
        assert abs(yaw) == pytest.approx(20.0, abs=1e-2)
        assert abs(pitch) < 1e-2
        assert abs(roll) < 1e-2

    def test_pure_pitch_rotation(self):
        theta = math.radians(15.0)
        R = np.array([
            [1, 0, 0],
            [0, math.cos(theta), -math.sin(theta)],
            [0, math.sin(theta), math.cos(theta)],
        ], dtype=np.float64)
        pitch, yaw, roll = rotation_matrix_to_euler_angles(R)
        assert pitch == pytest.approx(15.0, abs=1e-2)
        assert abs(yaw) < 1e-2
        assert abs(roll) < 1e-2


# ---------------------------------------------------------------------------
# Tests: classify_direction
# ---------------------------------------------------------------------------

class TestClassifyDirection:
    @pytest.fixture
    def cfg(self):
        return HeadPoseConfig(
            yaw_left_threshold=15.0,
            yaw_right_threshold=-15.0,
            pitch_up_threshold=-15.0,
            pitch_down_threshold=15.0,
        )

    def test_center(self, cfg):
        assert classify_direction(0.0, 0.0, cfg) == HeadDirection.CENTER
        assert classify_direction(5.0, -3.0, cfg) == HeadDirection.CENTER

    def test_left(self, cfg):
        assert classify_direction(20.0, 0.0, cfg) == HeadDirection.LEFT

    def test_right(self, cfg):
        assert classify_direction(-20.0, 0.0, cfg) == HeadDirection.RIGHT

    def test_up(self, cfg):
        assert classify_direction(0.0, -20.0, cfg) == HeadDirection.UP

    def test_down(self, cfg):
        assert classify_direction(0.0, 20.0, cfg) == HeadDirection.DOWN

    def test_exact_threshold_boundaries(self, cfg):
        assert classify_direction(15.0, 0.0, cfg) == HeadDirection.CENTER
        assert classify_direction(15.01, 0.0, cfg) == HeadDirection.LEFT

        assert classify_direction(-15.0, 0.0, cfg) == HeadDirection.CENTER
        assert classify_direction(-15.01, 0.0, cfg) == HeadDirection.RIGHT

        assert classify_direction(0.0, -15.0, cfg) == HeadDirection.CENTER
        assert classify_direction(0.0, -15.01, cfg) == HeadDirection.UP

        assert classify_direction(0.0, 15.0, cfg) == HeadDirection.CENTER
        assert classify_direction(0.0, 15.01, cfg) == HeadDirection.DOWN

    def test_dominant_direction_yaw_over_pitch(self, cfg):
        # Yaw: 30 deg (ratio = 30 / 15 = 2.0)
        # Pitch: -18 deg (ratio = 18 / 15 = 1.2)
        assert classify_direction(30.0, -18.0, cfg) == HeadDirection.LEFT

    def test_dominant_direction_pitch_over_yaw(self, cfg):
        # Yaw: 18 deg (ratio = 18 / 15 = 1.2)
        # Pitch: -30 deg (ratio = 30 / 15 = 2.0)
        assert classify_direction(18.0, -30.0, cfg) == HeadDirection.UP


# ---------------------------------------------------------------------------
# Tests: HeadPoseEstimator
# ---------------------------------------------------------------------------

class TestHeadPoseEstimator:
    def test_no_face_detected(self):
        estimator = HeadPoseEstimator()
        result = estimator.process(FaceDetectionResult(face_detected=False), (480, 640))
        assert result.direction == HeadDirection.UNKNOWN
        assert result.pitch is None
        assert result.yaw is None
        assert result.roll is None

    def test_none_or_empty_landmarks(self):
        estimator = HeadPoseEstimator()
        raw = SimpleNamespace(face_landmarks=[])
        result = estimator.process(FaceDetectionResult(face_detected=True, raw_result=raw), (480, 640))
        assert result.direction == HeadDirection.UNKNOWN

    def test_insufficient_landmarks(self):
        estimator = HeadPoseEstimator()
        short_lms = [_make_lm(0.5, 0.5) for _ in range(10)]
        res = _make_face_result(short_lms)
        result = estimator.process(res, (480, 640))
        assert result.direction == HeadDirection.UNKNOWN

    def test_invalid_frame_dimensions(self):
        estimator = HeadPoseEstimator()
        res = _make_face_result(_make_projected_landmarks(0, 0, 0))
        result = estimator.process(res, (0, 0))
        assert result.direction == HeadDirection.UNKNOWN

    def test_synthetic_centered_face(self):
        estimator = HeadPoseEstimator()
        res = _make_face_result(_make_projected_landmarks(0.0, 0.0, 0.0))
        result = estimator.process(res, (480, 640))
        assert result.direction == HeadDirection.CENTER
        assert result.yaw == pytest.approx(0.0, abs=1.0)
        assert result.pitch == pytest.approx(0.0, abs=1.0)

    def test_synthetic_turn_left(self):
        estimator = HeadPoseEstimator()
        res = _make_face_result(_make_projected_landmarks(0.0, 25.0, 0.0))
        result = estimator.process(res, (480, 640))
        assert result.yaw is not None
        assert result.yaw == pytest.approx(25.0, abs=1.0)
        assert result.direction == HeadDirection.LEFT

    def test_synthetic_turn_right(self):
        estimator = HeadPoseEstimator()
        res = _make_face_result(_make_projected_landmarks(0.0, -25.0, 0.0))
        result = estimator.process(res, (480, 640))
        assert result.yaw is not None
        assert result.yaw == pytest.approx(-25.0, abs=1.0)
        assert result.direction == HeadDirection.RIGHT

    def test_synthetic_look_up(self):
        estimator = HeadPoseEstimator()
        res = _make_face_result(_make_projected_landmarks(-25.0, 0.0, 0.0))
        result = estimator.process(res, (480, 640))
        assert result.pitch is not None
        assert result.pitch == pytest.approx(-25.0, abs=1.0)
        assert result.direction == HeadDirection.UP

    def test_synthetic_look_down(self):
        estimator = HeadPoseEstimator()
        res = _make_face_result(_make_projected_landmarks(25.0, 0.0, 0.0))
        result = estimator.process(res, (480, 640))
        assert result.pitch is not None
        assert result.pitch == pytest.approx(25.0, abs=1.0)
        assert result.direction == HeadDirection.DOWN


# ---------------------------------------------------------------------------
# Tests: Smoothing & Temporal Handling
# ---------------------------------------------------------------------------

class TestHeadPoseSmoothing:
    def test_smoothing_alpha_one(self):
        cfg = HeadPoseConfig(pose_smoothing_alpha=1.0)
        estimator = HeadPoseEstimator(pose_config=cfg)
        res_c = _make_face_result(_make_projected_landmarks(0, 0, 0))
        r1 = estimator.process(res_c, (480, 640))
        assert r1.direction == HeadDirection.CENTER

        res_left = _make_face_result(_make_projected_landmarks(0, 25, 0))
        r2 = estimator.process(res_left, (480, 640))
        assert r2.direction == HeadDirection.LEFT

    def test_smoothing_dampens_sudden_change(self):
        cfg = HeadPoseConfig(pose_smoothing_alpha=0.1)
        estimator = HeadPoseEstimator(pose_config=cfg)

        # Establish centered baseline
        for _ in range(5):
            estimator.process(_make_face_result(_make_projected_landmarks(0, 0, 0)), (480, 640))

        # Sudden jump to left (+25 deg)
        r_jump = estimator.process(_make_face_result(_make_projected_landmarks(0, 25, 0)), (480, 640))
        # With alpha=0.1: 0.1 * 25 + 0.9 * 0 = 2.5 deg (well below 15 deg threshold)
        assert r_jump.yaw < 10.0
        assert r_jump.direction == HeadDirection.CENTER

    def test_reset_clears_history(self):
        estimator = HeadPoseEstimator()
        estimator.process(_make_face_result(_make_projected_landmarks(0, 0, 0)), (480, 640))
        assert estimator._smoothed_yaw is not None

        estimator.reset()
        assert estimator._smoothed_pitch is None
        assert estimator._smoothed_yaw is None
        assert estimator._smoothed_roll is None

    def test_face_loss_resets_history(self):
        estimator = HeadPoseEstimator()
        estimator.process(_make_face_result(_make_projected_landmarks(0, 0, 0)), (480, 640))
        r_loss = estimator.process(FaceDetectionResult(face_detected=False), (480, 640))
        assert r_loss.direction == HeadDirection.UNKNOWN
        assert estimator._smoothed_yaw is None


# ---------------------------------------------------------------------------
# Tests: Drawing & Visualization
# ---------------------------------------------------------------------------

class TestHeadPoseDrawing:
    def test_draw_no_face_does_not_mutate(self):
        estimator = HeadPoseEstimator()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        orig = frame.copy()
        estimator.draw(frame, FaceDetectionResult(face_detected=False), HeadPoseResult())
        np.testing.assert_array_equal(frame, orig)

    def test_draw_valid_pose_runs_safely(self):
        estimator = HeadPoseEstimator()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = _make_face_result(_make_projected_landmarks(0, 0, 0))
        pose_res = estimator.process(res, (480, 640))
        estimator.draw(frame, res, pose_res)
