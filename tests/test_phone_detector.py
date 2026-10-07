"""Unit tests for detection/phone_detector.py.

Fast, deterministic tests using dependency injection and mock YOLO models.
Requires no webcam, GPU, or model weights download.
"""

from __future__ import annotations

import numpy as np
import pytest

from config.config import PhoneDetectorConfig
from detection.phone_detector import (
    PhoneDetector,
    PhoneDetectionResult,
    PhoneBoundingBox,
)


# ---------------------------------------------------------------------------
# Test Helpers & Mock YOLO Objects
# ---------------------------------------------------------------------------

class MockTensor:
    """Mock for PyTorch 1D/2D tensor operations used by Ultralytics."""

    def __init__(self, val):
        self._val = val

    def item(self):
        return self._val

    def tolist(self):
        return self._val

    def __getitem__(self, idx):
        return self


class MockBox:
    """Mock for Ultralytics Results.boxes element."""

    def __init__(self, xyxy, conf: float, cls_id: int):
        self.xyxy = MockTensor([xyxy])
        self.conf = MockTensor(conf)
        self.cls = MockTensor(cls_id)


class MockPredictionResult:
    """Mock for an individual element of model.predict(...) list."""

    def __init__(self, boxes=None):
        self.boxes = boxes if boxes is not None else []


class MockYOLOModel:
    """Mock YOLO model allowing controlled box generation without neural network compute."""

    def __init__(self, names=None, prediction_boxes=None):
        self.names = names if names is not None else {0: "person", 67: "cell phone"}
        self.prediction_boxes = prediction_boxes if prediction_boxes is not None else []
        self.predict_calls: int = 0
        self.last_predict_kwargs = {}

    def predict(self, *args, **kwargs):
        self.predict_calls += 1
        self.last_predict_kwargs = kwargs
        return [MockPredictionResult(self.prediction_boxes)]


# ---------------------------------------------------------------------------
# 1. Data Structures Tests
# ---------------------------------------------------------------------------

class TestPhoneDataStructures:
    def test_phone_bounding_box_properties(self):
        box = PhoneBoundingBox(x1=10, y1=20, x2=110, y2=220, confidence=0.88, class_name="cell phone")
        assert box.width == 100
        assert box.height == 200
        assert box.as_tuple == (10, 20, 110, 220)
        assert box.confidence == 0.88
        assert box.class_name == "cell phone"

    def test_phone_bounding_box_zero_or_negative(self):
        box = PhoneBoundingBox(x1=50, y1=50, x2=40, y2=30, confidence=0.5)
        assert box.width == 0
        assert box.height == 0

    def test_phone_detection_result_defaults(self):
        res = PhoneDetectionResult()
        assert res.phone_detected is False
        assert res.confidence is None
        assert res.bbox is None
        assert res.detections == []
        assert res.inference_time_ms == 0.0
        assert res.is_cached is False


# ---------------------------------------------------------------------------
# 2. Dynamic Class Resolution Tests
# ---------------------------------------------------------------------------

class TestPhoneDetectorClassResolution:
    def test_resolves_standard_coco_cell_phone(self):
        mock_model = MockYOLOModel(names={0: "person", 67: "cell phone"})
        detector = PhoneDetector(config=PhoneDetectorConfig(), model=mock_model)
        assert detector.phone_class_id == 67
        assert detector.class_name == "cell phone"

    def test_resolves_case_insensitive_and_whitespace(self):
        mock_model = MockYOLOModel(names={0: "person", 3: "  Cell Phone  "})
        detector = PhoneDetector(config=PhoneDetectorConfig(), model=mock_model)
        assert detector.phone_class_id == 3
        assert detector.class_name == "  Cell Phone  "

    def test_resolves_synonym_mobile_phone(self):
        mock_model = MockYOLOModel(names={0: "person", 15: "mobile phone"})
        detector = PhoneDetector(config=PhoneDetectorConfig(), model=mock_model)
        assert detector.phone_class_id == 15

    def test_resolves_synonym_telephone(self):
        mock_model = MockYOLOModel(names={0: "person", 8: "telephone"})
        detector = PhoneDetector(config=PhoneDetectorConfig(), model=mock_model)
        assert detector.phone_class_id == 8

    def test_resolves_from_names_list(self):
        mock_model = MockYOLOModel(names=["person", "bicycle", "cell phone"])
        detector = PhoneDetector(config=PhoneDetectorConfig(), model=mock_model)
        assert detector.phone_class_id == 2
        assert detector.class_name == "cell phone"

    def test_missing_phone_class_raises_value_error(self):
        mock_model = MockYOLOModel(names={0: "person", 1: "bicycle", 2: "car"})
        with pytest.raises(ValueError, match="does not contain a 'cell phone'"):
            PhoneDetector(config=PhoneDetectorConfig(), model=mock_model)

    def test_no_names_attribute_raises_value_error(self):
        class NoNamesModel:
            pass

        with pytest.raises(ValueError, match="does not have a 'names' attribute"):
            PhoneDetector(config=PhoneDetectorConfig(), model=NoNamesModel())

    def test_invalid_names_type_raises_value_error(self):
        class InvalidNamesModel:
            names = 12345

        with pytest.raises(ValueError, match="Unexpected type for model.names"):
            PhoneDetector(config=PhoneDetectorConfig(), model=InvalidNamesModel())


# ---------------------------------------------------------------------------
# 3. Confidence and Detection Filtering Tests
# ---------------------------------------------------------------------------

class TestConfidenceAndFiltering:
    @pytest.fixture
    def test_frame(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_confidence_above_threshold_detected(self, test_frame):
        box = MockBox([50, 60, 200, 300], conf=0.85, cls_id=67)
        mock_model = MockYOLOModel(prediction_boxes=[box])
        cfg = PhoneDetectorConfig(confidence_threshold=0.45)
        detector = PhoneDetector(config=cfg, model=mock_model)

        res = detector.detect(test_frame)
        assert res.phone_detected is True
        assert res.confidence == pytest.approx(0.85)
        assert res.bbox == (50, 60, 200, 300)
        assert len(res.detections) == 1
        assert res.is_cached is False

    def test_confidence_below_threshold_ignored(self, test_frame):
        box = MockBox([50, 60, 200, 300], conf=0.30, cls_id=67)
        mock_model = MockYOLOModel(prediction_boxes=[box])
        cfg = PhoneDetectorConfig(confidence_threshold=0.45)
        detector = PhoneDetector(config=cfg, model=mock_model)

        res = detector.detect(test_frame)
        assert res.phone_detected is False
        assert res.confidence is None
        assert res.bbox is None
        assert len(res.detections) == 0

    def test_multiple_detections_sorted_by_confidence(self, test_frame):
        box_low = MockBox([10, 10, 100, 150], conf=0.55, cls_id=67)
        box_high = MockBox([200, 200, 300, 400], conf=0.92, cls_id=67)
        mock_model = MockYOLOModel(prediction_boxes=[box_low, box_high])
        detector = PhoneDetector(config=PhoneDetectorConfig(), model=mock_model)

        res = detector.detect(test_frame)
        assert res.phone_detected is True
        assert res.confidence == pytest.approx(0.92)
        assert res.bbox == (200, 200, 300, 400)
        assert len(res.detections) == 2
        assert res.detections[0].confidence == pytest.approx(0.92)
        assert res.detections[1].confidence == pytest.approx(0.55)

    def test_different_class_id_ignored(self, test_frame):
        # Class 0 is person, not cell phone (67)
        box_person = MockBox([50, 50, 250, 350], conf=0.95, cls_id=0)
        mock_model = MockYOLOModel(prediction_boxes=[box_person])
        detector = PhoneDetector(config=PhoneDetectorConfig(), model=mock_model)

        res = detector.detect(test_frame)
        assert res.phone_detected is False
        assert len(res.detections) == 0

    def test_bounding_box_clamped_to_image_boundaries(self, test_frame):
        # Frame is 640x480
        out_of_bounds_box = MockBox([-50, -30, 800, 600], conf=0.80, cls_id=67)
        mock_model = MockYOLOModel(prediction_boxes=[out_of_bounds_box])
        detector = PhoneDetector(config=PhoneDetectorConfig(), model=mock_model)

        res = detector.detect(test_frame)
        assert res.phone_detected is True
        assert res.bbox == (0, 0, 640, 480)

    def test_zero_area_box_discarded(self, test_frame):
        inverted_box = MockBox([100, 100, 50, 50], conf=0.80, cls_id=67)
        mock_model = MockYOLOModel(prediction_boxes=[inverted_box])
        detector = PhoneDetector(config=PhoneDetectorConfig(), model=mock_model)

        res = detector.detect(test_frame)
        assert res.phone_detected is False
        assert len(res.detections) == 0


# ---------------------------------------------------------------------------
# 4. Temporal Caching & Inference Interval Tests
# ---------------------------------------------------------------------------

class TestTemporalCaching:
    @pytest.fixture
    def test_frame(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_inference_interval_skips_predict(self, test_frame):
        box = MockBox([50, 60, 200, 300], conf=0.85, cls_id=67)
        mock_model = MockYOLOModel(prediction_boxes=[box])
        cfg = PhoneDetectorConfig(inference_interval=3)
        detector = PhoneDetector(config=cfg, model=mock_model)

        # Frame 1: Runs inference
        res1 = detector.detect(test_frame)
        assert res1.phone_detected is True
        assert res1.is_cached is False
        assert mock_model.predict_calls == 1

        # Frame 2: Skips inference, reuses cache
        res2 = detector.detect(test_frame)
        assert res2.phone_detected is True
        assert res2.is_cached is True
        assert mock_model.predict_calls == 1

        # Frame 3: Skips inference, reuses cache
        res3 = detector.detect(test_frame)
        assert res3.phone_detected is True
        assert res3.is_cached is True
        assert mock_model.predict_calls == 1

        # Frame 4: Runs inference again
        res4 = detector.detect(test_frame)
        assert res4.phone_detected is True
        assert res4.is_cached is False
        assert mock_model.predict_calls == 2

    def test_cache_cleared_on_fresh_frame_when_phone_removed(self, test_frame):
        box = MockBox([50, 60, 200, 300], conf=0.85, cls_id=67)
        mock_model = MockYOLOModel(prediction_boxes=[box])
        cfg = PhoneDetectorConfig(inference_interval=2)
        detector = PhoneDetector(config=cfg, model=mock_model)

        # Frame 1: Phone present
        res1 = detector.detect(test_frame)
        assert res1.phone_detected is True

        # User removes phone from scene before Frame 3 (next inference frame)
        mock_model.prediction_boxes = []

        # Frame 2: Skipped frame returns cached detection
        res2 = detector.detect(test_frame)
        assert res2.phone_detected is True
        assert res2.is_cached is True

        # Frame 3: Fresh inference frame runs, phone is gone
        res3 = detector.detect(test_frame)
        assert res3.phone_detected is False
        assert res3.is_cached is False
        assert res3.confidence is None
        assert res3.bbox is None

    def test_cache_expiry_after_max_skipped_frames(self, test_frame):
        box = MockBox([50, 60, 200, 300], conf=0.85, cls_id=67)
        mock_model = MockYOLOModel(prediction_boxes=[box])
        # Force interval to 10 and expiry to 3
        cfg = PhoneDetectorConfig(inference_interval=10, cache_expiry_frames=3)
        detector = PhoneDetector(config=cfg, model=mock_model)

        # Frame 1: Inference frame
        res1 = detector.detect(test_frame)
        assert res1.phone_detected is True

        # Frames 2, 3, 4: Within 3 skipped frames
        for _ in range(3):
            res = detector.detect(test_frame)
            assert res.phone_detected is True
            assert res.is_cached is True

        # Frame 5: Exceeds 3 skipped frames -> cache expires
        res5 = detector.detect(test_frame)
        assert res5.phone_detected is False
        assert res5.confidence is None


# ---------------------------------------------------------------------------
# 5. Edge Cases & Robustness
# ---------------------------------------------------------------------------

class TestEdgeCasesAndRobustness:
    def test_none_frame_returns_empty_result(self):
        mock_model = MockYOLOModel()
        detector = PhoneDetector(config=PhoneDetectorConfig(), model=mock_model)
        res = detector.detect(None)
        assert res.phone_detected is False
        assert mock_model.predict_calls == 0

    def test_empty_frame_returns_empty_result(self):
        mock_model = MockYOLOModel()
        detector = PhoneDetector(config=PhoneDetectorConfig(), model=mock_model)
        empty_frame = np.empty((0, 0, 3), dtype=np.uint8)
        res = detector.detect(empty_frame)
        assert res.phone_detected is False
        assert mock_model.predict_calls == 0

    def test_disabled_detector_returns_empty_result(self):
        mock_model = MockYOLOModel()
        cfg = PhoneDetectorConfig(enabled=False)
        detector = PhoneDetector(config=cfg, model=mock_model)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = detector.detect(frame)
        assert res.phone_detected is False
        assert mock_model.predict_calls == 0

    def test_model_exception_trapped_gracefully(self):
        class ExplodingModel:
            names = {0: "person", 67: "cell phone"}
            def predict(self, *args, **kwargs):
                raise RuntimeError("CUDA out of memory or inference crash")

        detector = PhoneDetector(config=PhoneDetectorConfig(), model=ExplodingModel())
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = detector.detect(frame)
        assert res.phone_detected is False
        assert res.confidence is None


# ---------------------------------------------------------------------------
# 6. Visualization & Drawing Tests
# ---------------------------------------------------------------------------

class TestPhoneDrawing:
    def test_draw_no_phone_does_not_mutate_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        orig = frame.copy()
        detector = PhoneDetector(config=PhoneDetectorConfig(), model=MockYOLOModel())
        res = PhoneDetectionResult(phone_detected=False)

        detector.draw(frame, res)
        assert np.array_equal(frame, orig)

    def test_draw_with_detected_phone_mutates_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        orig = frame.copy()
        detector = PhoneDetector(config=PhoneDetectorConfig(), model=MockYOLOModel())
        box = PhoneBoundingBox(x1=50, y1=50, x2=200, y2=300, confidence=0.87)
        res = PhoneDetectionResult(
            phone_detected=True,
            confidence=0.87,
            bbox=(50, 50, 200, 300),
            detections=[box],
        )

        detector.draw(frame, res)
        # Verify bounding box and text were drawn
        assert not np.array_equal(frame, orig)

    def test_draw_boxes_disabled_does_not_mutate(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        orig = frame.copy()
        cfg = PhoneDetectorConfig(draw_boxes=False)
        detector = PhoneDetector(config=cfg, model=MockYOLOModel())
        box = PhoneBoundingBox(x1=50, y1=50, x2=200, y2=300, confidence=0.87)
        res = PhoneDetectionResult(phone_detected=True, detections=[box])

        detector.draw(frame, res)
        assert np.array_equal(frame, orig)

    def test_draw_near_top_edge_does_not_crash(self):
        # Bounding box starting at y1=0 (checks top edge clamping of label pill)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detector = PhoneDetector(config=PhoneDetectorConfig(), model=MockYOLOModel())
        box = PhoneBoundingBox(x1=10, y1=0, x2=100, y2=150, confidence=0.80)
        res = PhoneDetectionResult(phone_detected=True, detections=[box])

        detector.draw(frame, res)
        assert np.any(frame != 0)


# ---------------------------------------------------------------------------
# 7. Reset Tests
# ---------------------------------------------------------------------------

class TestPhoneDetectorReset:
    def test_reset_clears_counters_and_state(self):
        box = MockBox([50, 60, 200, 300], conf=0.85, cls_id=67)
        mock_model = MockYOLOModel(prediction_boxes=[box])
        detector = PhoneDetector(config=PhoneDetectorConfig(), model=mock_model)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        detector.detect(frame)
        assert detector._frame_count == 1
        assert detector._last_result.phone_detected is True

        detector.reset()
        assert detector._frame_count == 0
        assert detector._frames_since_inference == 0
        assert detector._last_result.phone_detected is False
