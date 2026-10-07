"""Configuration module for Smart Study Monitor.

Centralizes all user-tunable parameters across video capture, display,
face landmark detection, and eye tracking. Includes validation to prevent
invalid configurations at startup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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
    show_phone_detection: bool = True
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
# Phone Detection Settings (Ultralytics YOLO)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhoneDetectorConfig:
    """Settings for YOLO-based mobile phone detection."""

    model_path: str = str(BASE_DIR / "models" / "yolov8n.pt")
    confidence_threshold: float = 0.45
    iou_threshold: float = 0.45
    image_size: int = 320
    inference_interval: int = 4
    max_detections: int = 5
    cache_expiry_frames: int = 15
    enabled: bool = True
    device: str = "cpu"
    draw_boxes: bool = True

    def validate(self) -> None:
        if not self.model_path or not self.model_path.strip():
            raise ValueError("model_path cannot be empty")
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be in [0.0, 1.0], got {self.confidence_threshold}"
            )
        if not (0.0 <= self.iou_threshold <= 1.0):
            raise ValueError(
                f"iou_threshold must be in [0.0, 1.0], got {self.iou_threshold}"
            )
        if self.image_size <= 0:
            raise ValueError(
                f"image_size must be positive, got {self.image_size}"
            )
        if self.inference_interval <= 0:
            raise ValueError(
                f"inference_interval must be >= 1, got {self.inference_interval}"
            )
        if self.max_detections <= 0:
            raise ValueError(
                f"max_detections must be >= 1, got {self.max_detections}"
            )
        if self.cache_expiry_frames <= 0:
            raise ValueError(
                f"cache_expiry_frames must be >= 1, got {self.cache_expiry_frames}"
            )


# ---------------------------------------------------------------------------
# Behavior Engine Settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BehaviorConfig:
    """Settings for the Behavior Engine temporal confirmation and hysteresis.

    All durations are in seconds. The engine uses time.monotonic() internally.

    This is a project-defined heuristic rule system for study monitoring,
    NOT a medical diagnostic system.
    """

    phone_confirmation_sec: float = 1.0        # Phone must persist this long before PHONE_DETECTED
    drowsiness_confirmation_sec: float = 1.5   # Eyes closed this long before DROWSY
    head_away_confirmation_sec: float = 1.5    # Head turned away this long before DISTRACTED
    away_from_desk_confirmation_sec: float = 3.0 # Sustained head away / absence before AWAY_FROM_DESK
    no_face_confirmation_sec: float = 3.0      # Sustained no face before AWAY_FROM_DESK
    state_recovery_sec: float = 1.0            # Cooldown before returning to FOCUSED
    enabled: bool = True                       # Master enable/disable toggle

    def validate(self) -> None:
        for name, val in [
            ("phone_confirmation_sec", self.phone_confirmation_sec),
            ("drowsiness_confirmation_sec", self.drowsiness_confirmation_sec),
            ("head_away_confirmation_sec", self.head_away_confirmation_sec),
            ("away_from_desk_confirmation_sec", self.away_from_desk_confirmation_sec),
            ("no_face_confirmation_sec", self.no_face_confirmation_sec),
            ("state_recovery_sec", self.state_recovery_sec),
        ]:
            if val < 0:
                raise ValueError(f"{name} must be >= 0, got {val}")


# ---------------------------------------------------------------------------
# Alert & Audio Settings (Phase 7)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AlertConfig:
    """Settings for visual and audio alert notifications.

    Paths are resolved relative to BASE_DIR unless absolute.
    Cooldowns prevent spamming sounds on repeated or sustained frames.
    """

    alerts_enabled: bool = True
    sound_alerts_enabled: bool = True
    visual_alerts_enabled: bool = True
    alert_volume: float = 0.7
    visual_alert_duration_sec: float = 3.0

    # Cooldown durations in seconds
    phone_alert_cooldown_sec: float = 5.0
    drowsiness_alert_cooldown_sec: float = 5.0
    distraction_alert_cooldown_sec: float = 5.0
    away_alert_cooldown_sec: float = 10.0

    # Sound asset paths (relative to BASE_DIR or absolute)
    phone_alert_sound_path: str = str(BASE_DIR / "assets" / "sounds" / "phone_alert.mp3")
    distraction_alert_sound_path: str = str(BASE_DIR / "assets" / "sounds" / "distraction_alert.mp3")
    drowsiness_alert_sound_path: str = str(BASE_DIR / "assets" / "sounds" / "drowsiness_alert.mp3")

    # Away from desk sound behavior: "distraction", "none", or custom path
    away_alert_sound_mode: str = "distraction"  # "distraction", "none"

    def validate(self) -> None:
        if not (0.0 <= self.alert_volume <= 1.0):
            raise ValueError(
                f"alert_volume must be in [0.0, 1.0], got {self.alert_volume}"
            )
        for name, val in [
            ("visual_alert_duration_sec", self.visual_alert_duration_sec),
            ("phone_alert_cooldown_sec", self.phone_alert_cooldown_sec),
            ("drowsiness_alert_cooldown_sec", self.drowsiness_alert_cooldown_sec),
            ("distraction_alert_cooldown_sec", self.distraction_alert_cooldown_sec),
            ("away_alert_cooldown_sec", self.away_alert_cooldown_sec),
        ]:
            if val < 0.0:
                raise ValueError(f"{name} must be >= 0.0, got {val}")
        if self.away_alert_sound_mode not in ("distraction", "none"):
            raise ValueError(
                f"away_alert_sound_mode must be 'distraction' or 'none', got '{self.away_alert_sound_mode}'"
            )


# ---------------------------------------------------------------------------
# Database & Persistence Settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatabaseConfig:
    """Settings for SQLite database persistence."""

    db_path: str = str(BASE_DIR / "data" / "study_monitor.db")
    timeout_sec: float = 10.0

    def validate(self) -> None:
        if not self.db_path or not self.db_path.strip():
            raise ValueError("db_path cannot be empty")
        if self.timeout_sec <= 0:
            raise ValueError(f"timeout_sec must be > 0, got {self.timeout_sec}")


# ---------------------------------------------------------------------------
# Analytics & Focus Score Settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnalyticsConfig:
    """Settings for session analytics and Focus Score calculation."""

    minimum_scored_duration_sec: float = 60.0
    focused_weight: float = 1.0
    distracted_weight: float = 0.0
    drowsy_weight: float = 0.0
    phone_detected_weight: float = 0.0
    away_from_desk_weight: float = 0.0
    no_face_weight: float = 0.0
    unknown_weight: float = 0.0
    exclude_no_face: bool = True
    exclude_unknown: bool = True

    def validate(self) -> None:
        import math
        if self.minimum_scored_duration_sec < 0.0:
            raise ValueError(
                f"minimum_scored_duration_sec must be >= 0, got {self.minimum_scored_duration_sec}"
            )
        weights = [
            ("focused_weight", self.focused_weight),
            ("distracted_weight", self.distracted_weight),
            ("drowsy_weight", self.drowsy_weight),
            ("phone_detected_weight", self.phone_detected_weight),
            ("away_from_desk_weight", self.away_from_desk_weight),
            ("no_face_weight", self.no_face_weight),
            ("unknown_weight", self.unknown_weight),
        ]
        for name, val in weights:
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val) or val < 0.0:
                raise ValueError(f"{name} must be a non-negative finite number, got {val}")


# ---------------------------------------------------------------------------
# Dashboard Presentation Settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DashboardConfig:
    """Settings for Streamlit local analytics presentation dashboard."""

    page_title: str = "Smart Study Monitor - Analytics"
    page_icon: str = "📚"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"
    session_history_limit: int = 50

    def validate(self) -> None:
        if not self.page_title or not self.page_title.strip():
            raise ValueError("page_title cannot be empty")
        if self.session_history_limit <= 0:
            raise ValueError(f"session_history_limit must be > 0, got {self.session_history_limit}")


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
    phone_detector: PhoneDetectorConfig = field(default_factory=PhoneDetectorConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    base_dir: Path = BASE_DIR
    assets_dir: Path = BASE_DIR / "assets"
    models_dir: Path = BASE_DIR / "models"
    data_dir: Path = BASE_DIR / "data"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validates all sub-configurations."""
        self.camera.validate()
        self.display.validate()
        self.face_detector.validate()
        self.eye_tracker.validate()
        self.head_pose.validate()
        self.phone_detector.validate()
        self.behavior.validate()
        self.alerts.validate()
        self.database.validate()
        self.analytics.validate()
        self.dashboard.validate()


# Global default configuration instance
config = Config()

