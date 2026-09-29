"""Configuration module for Smart Study Monitor.

Centralizes all configurable thresholds, video capture settings,
detection parameters, and UI options to avoid magic numbers.
"""

from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class CameraConfig:
    device_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 30
    window_name: str = "Smart Study Monitor - Live Feed"


@dataclass(frozen=True)
class FaceDetectorConfig:
    model_path: str = str(BASE_DIR / "models" / "face_landmarker.task")
    num_faces: int = 1
    min_face_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    draw_tesselation: bool = True
    draw_contours: bool = True
    draw_irises: bool = True


@dataclass(frozen=True)
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    face_detector: FaceDetectorConfig = field(default_factory=FaceDetectorConfig)
    base_dir: Path = BASE_DIR
    assets_dir: Path = BASE_DIR / "assets"
    models_dir: Path = BASE_DIR / "models"


# Global default configuration instance
config = Config()
