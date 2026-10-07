"""Head pose estimation and looking-away direction classification module.

Uses MediaPipe face landmarks together with OpenCV solvePnP to estimate:
  - Yaw   (horizontal rotation: left / right)
  - Pitch (vertical rotation: up / down)
  - Roll  (in-plane tilt: clockwise / counter-clockwise)

And classifies the discrete head direction:
  - CENTER
  - LEFT
  - RIGHT
  - UP
  - DOWN
  - UNKNOWN (no face or computation failure)

Sign Conventions (from user's physical perspective):
  - Turning toward user's physical LEFT:  Yaw > 0  (exceeds yaw_left_threshold)
  - Turning toward user's physical RIGHT: Yaw < 0  (exceeds yaw_right_threshold)
  - Looking UP:                           Pitch < 0 (exceeds pitch_up_threshold)
  - Looking DOWN:                         Pitch > 0 (exceeds pitch_down_threshold)

Landmark Topology:
  Uses 6 stable 3D anatomical reference points from the 468-point mesh:
    1. Nose tip:                 Landmark 1
    2. Chin:                     Landmark 152
    3. Left eye outer corner:    Landmark 263 (subject's left)
    4. Right eye outer corner:   Landmark 33  (subject's right)
    5. Left mouth corner:        Landmark 291 (subject's left)
    6. Right mouth corner:       Landmark 61  (subject's right)

Camera Matrix Approximation:
  Uses pinhole camera model with focal length approximated by image width
  and optical center at image center:
    fx = fy = frame_width, cx = frame_width / 2, cy = frame_height / 2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np

from config.config import HeadPoseConfig, config
from vision.face_detector import FaceDetectionResult


# ---------------------------------------------------------------------------
# Facial Landmark Indices & 3D Canonical Model Points
# ---------------------------------------------------------------------------

# Indices in the 468/478-point MediaPipe face mesh
NOSE_TIP_IDX = 1
CHIN_IDX = 152
LEFT_EYE_OUTER_IDX = 263
RIGHT_EYE_OUTER_IDX = 33
LEFT_MOUTH_CORNER_IDX = 291
RIGHT_MOUTH_CORNER_IDX = 61

HEAD_POSE_LANDMARK_INDICES: Tuple[int, ...] = (
    NOSE_TIP_IDX,
    CHIN_IDX,
    LEFT_EYE_OUTER_IDX,
    RIGHT_EYE_OUTER_IDX,
    LEFT_MOUTH_CORNER_IDX,
    RIGHT_MOUTH_CORNER_IDX,
)

# Canonical 3D anthropometric face model points (in mm).
# Origin (0,0,0) is at the tip of the nose.
# +X: viewer's right / subject's physical left
# +Y: downwards (chin has +Y, eyes have -Y)
# +Z: into the face (nose is furthest towards camera at Z=0, eyes/mouth recessed at +Z)
CANONICAL_FACE_MODEL_3D: np.ndarray = np.array(
    [
        [0.0, 0.0, 0.0],          # Nose tip (landmark 1)
        [0.0, 330.0, 65.0],       # Chin (landmark 152)
        [225.0, -170.0, 135.0],   # Left eye outer corner (landmark 263)
        [-225.0, -170.0, 135.0],  # Right eye outer corner (landmark 33)
        [150.0, 150.0, 125.0],    # Left mouth corner (landmark 291)
        [-150.0, 150.0, 125.0],   # Right mouth corner (landmark 61)
    ],
    dtype=np.float64,
)

# 3D points for projecting a forward-facing gaze pointer line
AXIS_POINTS_3D: np.ndarray = np.array(
    [
        [0.0, 0.0, 0.0],         # Nose tip base
        [0.0, 0.0, -150.0],      # Vector pointing forward from face
    ],
    dtype=np.float64,
)


class HeadDirection(Enum):
    """Discrete head orientation classifications."""

    CENTER = "CENTER"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UP = "UP"
    DOWN = "DOWN"
    UNKNOWN = "UNKNOWN"


@dataclass
class HeadPoseResult:
    """Per-frame head pose output container."""

    pitch: Optional[float] = None
    yaw: Optional[float] = None
    roll: Optional[float] = None
    direction: HeadDirection = HeadDirection.UNKNOWN
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Pure Helper Functions (Stateless & Unit-Testable)
# ---------------------------------------------------------------------------

def build_camera_matrix(width: int, height: int) -> np.ndarray:
    """Builds an approximated camera intrinsic matrix for pinhole camera model."""
    focal_length = float(width)
    center_x = float(width) / 2.0
    center_y = float(height) / 2.0
    return np.array(
        [
            [focal_length, 0.0, center_x],
            [0.0, focal_length, center_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rotation_matrix_to_euler_angles(rmat: np.ndarray) -> Tuple[float, float, float]:
    """Decomposes a 3x3 rotation matrix into Euler angles (Pitch, Yaw, Roll) in degrees.

    Convention (user-centric):
      - Pitch: rotation around X axis (looking UP < 0, looking DOWN > 0)
      - Yaw:   rotation around Y axis (turning user LEFT > 0, turning user RIGHT < 0)
      - Roll:  rotation around Z axis (tilting head sideways)
    """
    sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(rmat[2, 1], rmat[2, 2])
        yaw = math.atan2(rmat[2, 0], sy)
        roll = math.atan2(rmat[1, 0], rmat[0, 0])
    else:
        pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
        yaw = math.atan2(rmat[2, 0], sy)
        roll = 0.0

    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)



def classify_direction(
    yaw: float,
    pitch: float,
    cfg: HeadPoseConfig,
) -> HeadDirection:
    """Deterministically classifies head orientation without overlapping states.

    If both Yaw and Pitch deviate significantly, the dominant direction is selected
    by comparing deviation magnitude relative to the respective threshold.
    """
    is_left = yaw > cfg.yaw_left_threshold
    is_right = yaw < cfg.yaw_right_threshold
    is_up = pitch < cfg.pitch_up_threshold
    is_down = pitch > cfg.pitch_down_threshold

    yaw_deviates = is_left or is_right
    pitch_deviates = is_up or is_down

    if not yaw_deviates and not pitch_deviates:
        return HeadDirection.CENTER

    if yaw_deviates and not pitch_deviates:
        return HeadDirection.LEFT if is_left else HeadDirection.RIGHT

    if pitch_deviates and not yaw_deviates:
        return HeadDirection.UP if is_up else HeadDirection.DOWN

    # Both deviate: select the dominant direction relative to threshold magnitude
    yaw_thresh = cfg.yaw_left_threshold if is_left else abs(cfg.yaw_right_threshold)
    pitch_thresh = abs(cfg.pitch_up_threshold) if is_up else cfg.pitch_down_threshold

    yaw_ratio = abs(yaw) / yaw_thresh
    pitch_ratio = abs(pitch) / pitch_thresh

    if yaw_ratio >= pitch_ratio:
        return HeadDirection.LEFT if is_left else HeadDirection.RIGHT
    else:
        return HeadDirection.UP if is_up else HeadDirection.DOWN


# ---------------------------------------------------------------------------
# HeadPoseEstimator Class
# ---------------------------------------------------------------------------

class HeadPoseEstimator:
    """Estimates 3D head pose and classifies looking-away direction using solvePnP.

    Reuses face landmarks produced by FaceDetector; does NOT re-initialize MediaPipe.
    Holds internal state for exponential moving average (EMA) smoothing.
    """

    def __init__(self, pose_config: Optional[HeadPoseConfig] = None) -> None:
        self._cfg = pose_config or config.head_pose
        self._cfg.validate()
        self._smoothed_pitch: Optional[float] = None
        self._smoothed_yaw: Optional[float] = None
        self._smoothed_roll: Optional[float] = None
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    def reset(self) -> None:
        """Resets temporal smoothing history."""
        self._smoothed_pitch = None
        self._smoothed_yaw = None
        self._smoothed_roll = None

    def process(
        self,
        face_result: FaceDetectionResult,
        frame_shape: Tuple[int, int],
    ) -> HeadPoseResult:
        """Estimates head pose angles and direction from face landmarks.

        Args:
            face_result: Output from FaceDetector.process().
            frame_shape: (height, width) of the captured video frame.

        Returns:
            HeadPoseResult with smoothed pitch, yaw, roll, and classified direction.
        """
        if not face_result.face_detected or face_result.raw_result is None:
            self.reset()
            return HeadPoseResult()

        landmarks_list = face_result.raw_result.face_landmarks
        if not landmarks_list or len(landmarks_list[0]) < max(HEAD_POSE_LANDMARK_INDICES) + 1:
            self.reset()
            return HeadPoseResult()

        primary_landmarks = landmarks_list[0]
        h, w = frame_shape[:2]

        if w <= 0 or h <= 0:
            self.reset()
            return HeadPoseResult()

        # Extract 2D image coordinates for the 6 canonical points
        try:
            image_points_2d = np.array(
                [
                    [primary_landmarks[idx].x * w, primary_landmarks[idx].y * h]
                    for idx in HEAD_POSE_LANDMARK_INDICES
                ],
                dtype=np.float64,
            )
        except (AttributeError, IndexError):
            self.reset()
            return HeadPoseResult()

        camera_matrix = build_camera_matrix(w, h)

        # Solve Perspective-n-Point for 3D rotation and translation vectors
        try:
            success, rvec, tvec = cv2.solvePnP(
                CANONICAL_FACE_MODEL_3D,
                image_points_2d,
                camera_matrix,
                self._dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            self.reset()
            return HeadPoseResult()

        if not success or rvec is None or tvec is None:
            self.reset()
            return HeadPoseResult()

        # Convert rotation vector to 3x3 rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)

        # Extract Euler angles in degrees
        raw_pitch, raw_yaw, raw_roll = rotation_matrix_to_euler_angles(rmat)

        # Apply Exponential Moving Average (EMA) smoothing
        pitch, yaw, roll = self._apply_ema(raw_pitch, raw_yaw, raw_roll)

        # Classify discrete head direction
        direction = classify_direction(yaw, pitch, self._cfg)

        return HeadPoseResult(
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            direction=direction,
            rvec=rvec,
            tvec=tvec,
        )

    def draw(
        self,
        bgr_frame: np.ndarray,
        face_result: FaceDetectionResult,
        pose_result: HeadPoseResult,
    ) -> None:
        """Renders lightweight 3D pose direction vector on the video frame."""
        if not self._cfg.draw_head_pose:
            return
        if not face_result.face_detected or pose_result.direction is HeadDirection.UNKNOWN:
            return
        if pose_result.rvec is None or pose_result.tvec is None:
            return

        h, w = bgr_frame.shape[:2]
        camera_matrix = build_camera_matrix(w, h)

        try:
            projected_pts, _ = cv2.projectPoints(
                AXIS_POINTS_3D,
                pose_result.rvec,
                pose_result.tvec,
                camera_matrix,
                self._dist_coeffs,
            )
            projected_pts = projected_pts.reshape(-1, 2)
            p_nose = (int(projected_pts[0][0]), int(projected_pts[0][1]))
            p_forward = (int(projected_pts[1][0]), int(projected_pts[1][1]))

            # Color: Green if CENTER, Red/Amber if looking away
            if pose_result.direction is HeadDirection.CENTER:
                color = (0, 255, 128)
            else:
                color = (0, 140, 255)

            # Draw nose anchor point
            cv2.circle(bgr_frame, p_nose, 4, color, -1, cv2.LINE_AA)

            # Draw directional 3D gaze pointer vector line
            cv2.line(bgr_frame, p_nose, p_forward, color, 2, cv2.LINE_AA)
        except (cv2.error, IndexError, ValueError):
            pass

    def _apply_ema(
        self,
        raw_pitch: float,
        raw_yaw: float,
        raw_roll: float,
    ) -> Tuple[float, float, float]:
        """Applies EMA smoothing to Euler angles."""
        alpha = self._cfg.pose_smoothing_alpha

        if self._smoothed_pitch is None:
            self._smoothed_pitch = raw_pitch
            self._smoothed_yaw = raw_yaw
            self._smoothed_roll = raw_roll
        else:
            self._smoothed_pitch = alpha * raw_pitch + (1.0 - alpha) * self._smoothed_pitch
            self._smoothed_yaw = alpha * raw_yaw + (1.0 - alpha) * self._smoothed_yaw
            self._smoothed_roll = alpha * raw_roll + (1.0 - alpha) * self._smoothed_roll

        return self._smoothed_pitch, self._smoothed_yaw, self._smoothed_roll
