"""Tests for vision/face_detector.py.

These tests do NOT require a physical webcam.  They verify initialisation
error paths, input validation, and that the FaceDetectionResult dataclass
behaves correctly.

Tests that require the actual model file are skipped automatically when the
model is absent (typical CI environment) so the test suite can still run.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path so imports work when tests/ is the CWD
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision.face_detector import FaceDetectionResult


# ---------------------------------------------------------------------------
# FaceDetectionResult dataclass
# ---------------------------------------------------------------------------

class TestFaceDetectionResult:
    def test_no_face_default(self):
        result = FaceDetectionResult(face_detected=False)
        assert result.face_detected is False
        assert result.raw_result is None

    def test_face_detected_true(self):
        result = FaceDetectionResult(face_detected=True, raw_result=object())
        assert result.face_detected is True
        assert result.raw_result is not None


# ---------------------------------------------------------------------------
# FaceDetector — model-absent path (always runs)
# ---------------------------------------------------------------------------

class TestFaceDetectorNoModel:
    def test_missing_model_raises_file_not_found(self, tmp_path):
        """FaceDetector must raise FileNotFoundError for a non-existent model."""
        from config.config import FaceDetectorConfig
        from vision.face_detector import FaceDetector

        cfg = FaceDetectorConfig(model_path=str(tmp_path / "no_such_model.task"))
        with pytest.raises(FileNotFoundError, match="face-landmarker model not found"):
            FaceDetector(detector_config=cfg)


# ---------------------------------------------------------------------------
# FaceDetector — model-present path (skipped when model is absent)
# ---------------------------------------------------------------------------

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"
requires_model = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="face_landmarker.task model not found — run `python download_models.py` first.",
)


@requires_model
class TestFaceDetectorWithModel:
    def test_initialises_without_error(self):
        from vision.face_detector import FaceDetector
        detector = FaceDetector()
        detector.close()

    def test_context_manager(self):
        from vision.face_detector import FaceDetector
        with FaceDetector() as detector:
            assert detector._landmarker is not None

    def test_process_black_frame_no_crash(self):
        """A completely black 480×640 frame should return face_detected=False."""
        from vision.face_detector import FaceDetector
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with FaceDetector() as detector:
            result = detector.process(black_frame)
        assert result.face_detected is False
        assert result.raw_result is not None  # result object always exists

    def test_process_none_frame_handled(self):
        """Passing None as frame must not raise; return face_detected=False."""
        from vision.face_detector import FaceDetector
        with FaceDetector() as detector:
            result = detector.process(None)  # type: ignore[arg-type]
        assert result.face_detected is False

    def test_process_empty_frame_handled(self):
        """Passing an empty ndarray must return face_detected=False."""
        from vision.face_detector import FaceDetector
        empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
        with FaceDetector() as detector:
            result = detector.process(empty_frame)
        assert result.face_detected is False

    def test_draw_no_face_does_not_mutate(self):
        """draw() with face_detected=False must leave the frame unchanged."""
        from vision.face_detector import FaceDetector
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        original = frame.copy()
        with FaceDetector() as detector:
            result = FaceDetectionResult(face_detected=False)
            detector.draw(frame, result)
        np.testing.assert_array_equal(frame, original)
