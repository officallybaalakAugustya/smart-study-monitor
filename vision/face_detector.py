"""Face landmark detection module for Smart Study Monitor.

Wraps the MediaPipe Tasks API FaceLandmarker (mediapipe >= 0.10 / 1.0.x) to
provide a clean, testable interface that the rest of the application can use
without any MediaPipe-specific knowledge leaking out.

Primary user: one face.  If multiple faces are returned by MediaPipe the one
with the highest landmark confidence (first in the result list, as MediaPipe
returns faces in detection-confidence order) is selected.  This is documented
behaviour, not silent magic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config.config import FaceDetectorConfig, config

# -- MediaPipe Tasks imports (mediapipe >= 0.10 / 1.0.x) ----------------------
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    FaceLandmarkerResult,
    FaceLandmarksConnections,
    RunningMode,
    drawing_styles,
    drawing_utils,
)
from mediapipe.tasks.python.vision.core.image import Image as MpImage, ImageFormat

# Official Google-hosted model bundle for the Tasks API Face Landmarker.
_MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


@dataclass
class FaceDetectionResult:
    """Minimal, framework-agnostic container for one detection result.

    face_detected: True if at least one face was found in the frame.
    raw_result:    The raw FaceLandmarkerResult from MediaPipe (None when no
                   face is detected).  Kept for downstream modules that need
                   direct landmark access (e.g., Phase 3 EAR calculation).
    """

    face_detected: bool
    raw_result: Optional[FaceLandmarkerResult] = None


class FaceDetector:
    """Initialises MediaPipe's FaceLandmarker once and reuses it per frame.

    Usage
    -----
    detector = FaceDetector()        # may raise if model file is missing
    result   = detector.process(bgr_frame)
    detector.draw(bgr_frame, result) # mutates frame in-place
    detector.close()
    """

    def __init__(self, detector_config: Optional[FaceDetectorConfig] = None) -> None:
        self._cfg = detector_config or config.face_detector
        self._landmarker: Optional[FaceLandmarker] = None
        self._init_landmarker()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_landmarker(self) -> None:
        """Creates and configures the MediaPipe FaceLandmarker instance."""
        model_path = Path(self._cfg.model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe face-landmarker model not found at:\n"
                f"  {model_path}\n\n"
                "Download it by running:\n"
                f"  python -c \"import urllib.request; "
                f"urllib.request.urlretrieve('{_MODEL_DOWNLOAD_URL}', r'{model_path}')\"\n\n"
                "Or follow the instructions in models/README.md."
            )

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.IMAGE,
            num_faces=self._cfg.num_faces,
            min_face_detection_confidence=self._cfg.min_face_detection_confidence,
            min_face_presence_confidence=self._cfg.min_face_presence_confidence,
            min_tracking_confidence=self._cfg.min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        self._landmarker = FaceLandmarker.create_from_options(options)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, bgr_frame: np.ndarray) -> FaceDetectionResult:
        """Runs face landmark detection on a single BGR OpenCV frame.

        The frame is converted to RGB internally; the original frame is not
        mutated.

        Args:
            bgr_frame: H×W×3 uint8 numpy array in BGR channel order.

        Returns:
            FaceDetectionResult with face_detected flag and raw MediaPipe
            result available for downstream use.
        """
        if self._landmarker is None:
            return FaceDetectionResult(face_detected=False)

        if bgr_frame is None or bgr_frame.size == 0:
            return FaceDetectionResult(face_detected=False)

        # MediaPipe Tasks API requires an mp.Image (RGB channel order).
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = MpImage(image_format=ImageFormat.SRGB, data=rgb_frame)

        result: FaceLandmarkerResult = self._landmarker.detect(mp_image)

        face_detected = bool(result.face_landmarks)
        return FaceDetectionResult(face_detected=face_detected, raw_result=result)

    def draw(self, bgr_frame: np.ndarray, result: FaceDetectionResult) -> None:
        """Draws face landmarks directly onto bgr_frame (in-place mutation).

        When no face is detected, nothing is drawn.  Visualisation elements
        are controlled by the FaceDetectorConfig flags.

        Args:
            bgr_frame: The frame that will be annotated.
            result:    The FaceDetectionResult produced by process().
        """
        if not result.face_detected or result.raw_result is None:
            return

        # MediaPipe Tasks drawing_utils.draw_landmarks works directly on the
        # BGR numpy array; no additional colour conversion is required here.
        landmarks_list = result.raw_result.face_landmarks

        # Select only the primary face (index 0 — highest-confidence detection).
        primary_landmarks = landmarks_list[0]

        if self._cfg.draw_tesselation:
            drawing_utils.draw_landmarks(
                image=bgr_frame,
                landmark_list=primary_landmarks,
                connections=FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
            )

        if self._cfg.draw_contours:
            drawing_utils.draw_landmarks(
                image=bgr_frame,
                landmark_list=primary_landmarks,
                connections=FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
            )

        if self._cfg.draw_irises:
            drawing_utils.draw_landmarks(
                image=bgr_frame,
                landmark_list=primary_landmarks,
                connections=FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
            )
            drawing_utils.draw_landmarks(
                image=bgr_frame,
                landmark_list=primary_landmarks,
                connections=FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
            )

    def close(self) -> None:
        """Releases MediaPipe FaceLandmarker resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    # Context-manager support so callers can use `with FaceDetector() as fd:`
    def __enter__(self) -> "FaceDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
