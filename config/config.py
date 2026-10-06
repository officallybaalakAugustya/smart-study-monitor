"""Configuration module for Smart Study Monitor.

Centralizes all user-tunable parameters across video capture, display,
face landmark detection, and eye tracking. Includes validation to prevent
invalid configurations at startup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Camera Settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CameraConfig:
    """Settings for webcam capture and hardware interface."""

    device_index: int = 0
    backend: str = "DSHOW"  # "DSHOW", "MSMF", or "DEFAULT"
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 30
    fps_update_interval_sec: float = 0.5

    def validate(self) -> None:
        if self.device_index < 0:
            raise ValueError(f"device_index must be >= 0, got {self.device_index}")
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise ValueError(
                f"frame_width and frame_height must be positive, "
                f"got ({self.frame_width}x{self.frame_height})"
            )
        if self.target_fps <= 0:
            raise ValueError(f"target_fps must be > 0, got {self.target_fps}")
        if self.fps_update_interval_sec <= 0:
            raise ValueError(
                f"fps_update_interval_sec must be > 0, got {self.fps_update_interval_sec}"
            )


# ---------------------------------------------------------------------------
# Display & HUD Settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DisplayConfig:
    """Settings for GUI window and HUD rendering."""

    window_name: str = "Smart Study Monitor - Live Feed"
    window_width: int = 960
    window_height: int = 720
    window_x: int = 50
    window_y: int = 50
    force_foreground: bool = True
    heartbeat_interval_sec: float = 2.0
    show_hud: bool = True
    show_face_landmarks: bool = True
    show_eye_landmarks: bool = True
    show_head_pose: bool = True
    debug_mode: bool = False
    panel_opacity: float = 0.80

    def validate(self) -> None:
        if not self.window_name or not self.window_name.strip():
            raise ValueError("window_name cannot be empty")
        if self.window_width <= 0 or self.window_height <= 0:
            raise ValueError(
                f"window_width and window_height must be positive, "
                f"got ({self.window_width}x{self.window_height})"
            )
        if self.heartbeat_interval_sec <= 0:
            raise ValueError(
                f"heartbeat_interval_sec must be > 0, got {self.heartbeat_interval_sec}"
            )
        if not (0.0 <= self.panel_opacity <= 1.0):
            raise ValueError(
                f"panel_opacity must be in [0.0, 1.0], got {self.panel_opacity}"
            )


# ---------------------------------------------------------------------------
# Face Detection Settings (MediaPipe Tasks API)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FaceDetectorConfig:
    """Settings for MediaPipe FaceLandmarker detection."""

    model_path: str = str(BASE_DIR / "models" / "face_landmarker.task")
    num_faces: int = 1
    min_face_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    draw_tesselation: bool = False
    draw_contours: bool = True
    draw_irises: bool = True


    def validate(self) -> None:
        if self.num_faces <= 0:
            raise ValueError(f"num_faces must be > 0, got {self.num_faces}")
        for name, val in [
            ("min_face_detection_confidence", self.min_face_detection_confidence),
            ("min_face_presence_confidence", self.min_face_presence_confidence),
            ("min_tracking_confidence", self.min_tracking_confidence),
        ]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0.0, 1.0], got {val}")


# ---------------------------------------------------------------------------
# Eye Tracking & EAR Settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EyeTrackerConfig:
    """Settings for Eye Aspect Ratio calculation, smoothing, and hysteresis.

    Hysteresis logic:
        EAR <= ear_close_threshold  → CLOSED
        EAR >= ear_open_threshold   → OPEN
        Between the two             → retain previous state

    Calibration note:
        Thresholds may vary per user due to face anatomy, camera angle,
        lighting conditions, glasses, and camera-to-face distance.
    """

    ear_close_threshold: float = 0.18
    ear_open_threshold: float = 0.22
    ear_smoothing_alpha: float = 0.4
    min_eye_width: float = 0.001
    draw_eye_landmarks: bool = True

    def validate(self) -> None:
        if self.ear_close_threshold <= 0:
            raise ValueError(
                f"ear_close_threshold must be > 0, got {self.ear_close_threshold}"
            )
        if self.ear_open_threshold <= 0:
            raise ValueError(
                f"ear_open_threshold must be > 0, got {self.ear_open_threshold}"
            )
        if self.ear_close_threshold >= self.ear_open_threshold:
            raise ValueError(
                f"ear_close_threshold ({self.ear_close_threshold}) must be strictly less "
                f"than ear_open_threshold ({self.ear_open_threshold}) for hysteresis to work."
            )
        if not (0.0 < self.ear_smoothing_alpha <= 1.0):
            raise ValueError(
                f"ear_smoothing_alpha must be in (0.0, 1.0], got {self.ear_smoothing_alpha}"
            )
        if self.min_eye_width <= 0:
            raise ValueError(
                f"min_eye_width must be > 0, got {self.min_eye_width}"
            )


# ---------------------------------------------------------------------------
# Head Pose Estimation & Looking-Away Settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HeadPoseConfig:
    """Settings for Head Pose Estimation and Looking-Away Direction Classification.

    Yaw convention:
        Positive Yaw (> yaw_left_threshold)   -> Looking toward user's physical LEFT
        Negative Yaw (< yaw_right_threshold)  -> Looking toward user's physical RIGHT
    Pitch convention:
        Negative Pitch (< pitch_up_threshold) -> Looking UP
        Positive Pitch (> pitch_down_threshold) -> Looking DOWN
    """

    yaw_left_threshold: float = 15.0       # degrees (must be > 0)
    yaw_right_threshold: float = -15.0     # degrees (must be < 0)
    pitch_up_threshold: float = -15.0      # degrees (must be < 0)
    pitch_down_threshold: float = 15.0     # degrees (must be > 0)
    pose_smoothing_alpha: float = 0.3      # EMA smoothing factor (0 < alpha <= 1)
    draw_head_pose: bool = True            # Render 3D pose axis / pointer

    def validate(self) -> None:
        if self.yaw_left_threshold <= 0:
            raise ValueError(
                f"yaw_left_threshold must be > 0, got {self.yaw_left_threshold}"
            )
        if self.yaw_right_threshold >= 0:
            raise ValueError(
                f"yaw_right_threshold must be < 0, got {self.yaw_right_threshold}"
            )
        if self.pitch_up_threshold >= 0:
            raise ValueError(
                f"pitch_up_threshold must be < 0, got {self.pitch_up_threshold}"
            )
        if self.pitch_down_threshold <= 0:
            raise ValueError(
                f"pitch_down_threshold must be > 0, got {self.pitch_down_threshold}"
            )
        if not (0.0 < self.pose_smoothing_alpha <= 1.0):
            raise ValueError(
                f"pose_smoothing_alpha must be in (0.0, 1.0], got {self.pose_smoothing_alpha}"
            )


# ---------------------------------------------------------------------------
# Master Config Container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    """Master application configuration container."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    face_detector: FaceDetectorConfig = field(default_factory=FaceDetectorConfig)
    eye_tracker: EyeTrackerConfig = field(default_factory=EyeTrackerConfig)
    head_pose: HeadPoseConfig = field(default_factory=HeadPoseConfig)
    base_dir: Path = BASE_DIR
    assets_dir: Path = BASE_DIR / "assets"
    models_dir: Path = BASE_DIR / "models"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validates all sub-configurations."""
        self.camera.validate()
        self.display.validate()
        self.face_detector.validate()
        self.eye_tracker.validate()
        self.head_pose.validate()


# Global default configuration instance
config = Config()

