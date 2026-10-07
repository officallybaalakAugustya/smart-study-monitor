"""Real-time Mobile Phone Detection using Ultralytics YOLO.

Part of Smart Study Monitor Phase 5.
Detects visible mobile/cell phones in webcam frames using a lightweight
YOLO model (YOLOv8 nano) on CPU. Exposes structured detection telemetry
without making behavioral judgments or distraction assumptions.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from config.config import BASE_DIR, PhoneDetectorConfig, config as default_global_config

logger = logging.getLogger(__name__)

# Canonical synonyms for cell phone in standard detection datasets (COCO, etc.)
PHONE_CLASS_SYNONYMS = (
    "cell phone",
    "cellphone",
    "mobile phone",
    "telephone",
    "phone",
)


@dataclass(frozen=True)
class PhoneBoundingBox:
    """Bounding box and confidence for a single detected phone."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_name: str = "cell phone"

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class PhoneDetectionResult:
    """Structured phone detection result for application and future Behavior Engine."""

    phone_detected: bool = False
    confidence: Optional[float] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    detections: List[PhoneBoundingBox] = field(default_factory=list)
    inference_time_ms: float = 0.0
    is_cached: bool = False


class PhoneDetector:
    """Manages YOLO phone detection, class resolution, temporal caching, and rendering."""

    def __init__(
        self,
        config: Optional[PhoneDetectorConfig] = None,
        model: Optional[Any] = None,
    ) -> None:
        """Initializes PhoneDetector.

        Args:
            config: Optional configuration instance. Defaults to global config.phone_detector.
            model: Optional pre-instantiated YOLO model (for testing/mocking).
        """
        self.config = config or default_global_config.phone_detector
        self.config.validate()

        self._frame_count: int = 0
        self._frames_since_inference: int = 0
        self._last_result: PhoneDetectionResult = PhoneDetectionResult()

        if model is not None:
            self.model = model
        else:
            self.model = self._load_model(self.config.model_path)

        self.phone_class_id, self.class_name = self._resolve_phone_class()

        if model is None:
            self._warmup()

    def _warmup(self) -> None:
        """Executes a single dummy forward pass to warm up PyTorch and eliminate frame 1 latency."""
        try:
            dummy = np.zeros((self.config.image_size, self.config.image_size, 3), dtype=np.uint8)
            self.model.predict(
                source=dummy,
                classes=[self.phone_class_id],
                imgsz=self.config.image_size,
                device=self.config.device,
                verbose=False,
            )
        except Exception as exc:
            logger.debug("Warmup pass skipped or failed: %s", exc)

    def _load_model(self, model_path: str) -> Any:
        """Loads the Ultralytics YOLO model from disk or triggers auto-download."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "The 'ultralytics' package is required for phone detection. "
                "Install it via 'pip install ultralytics'."
            ) from exc

        p = Path(model_path)
        path = p if p.is_absolute() else (BASE_DIR / p).resolve()
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("YOLO model not found at '%s'. Attempting download...", path)

        try:
            # Ultralytics will locate existing path or download to target path
            return YOLO(str(path))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load YOLO model from '{model_path}': {exc}. "
                f"Run 'python download_models.py' to obtain required models."
            ) from exc

    def _resolve_phone_class(self) -> Tuple[int, str]:
        """Dynamically identifies the class index matching 'cell phone' in the loaded model.

        Raises:
            ValueError: If the loaded model does not contain a cell phone class.
        """
        names = getattr(self.model, "names", None)
        if names is None:
            raise ValueError("Loaded model does not have a 'names' attribute.")

        # `names` in Ultralytics is typically a dict {int: str} or a list of str
        if isinstance(names, list):
            names_dict = {i: name for i, name in enumerate(names)}
        elif isinstance(names, dict):
            names_dict = names
        else:
            raise ValueError(f"Unexpected type for model.names: {type(names)}")

        for class_id, raw_name in names_dict.items():
            normalized = str(raw_name).strip().lower()
            if normalized in PHONE_CLASS_SYNONYMS:
                return int(class_id), str(raw_name)

        available_sample = list(names_dict.values())[:10]
        raise ValueError(
            f"The loaded model does not contain a 'cell phone' or equivalent class. "
            f"Available sample classes: {available_sample}."
        )

    def detect(self, frame: np.ndarray) -> PhoneDetectionResult:
        """Detects mobile phones in the webcam frame with temporal caching.

        Args:
            frame: BGR video frame from OpenCV.

        Returns:
            PhoneDetectionResult with detection state, confidence, and bounding boxes.
        """
        if frame is None or frame.size == 0 or not self.config.enabled:
            return PhoneDetectionResult()

        self._frame_count += 1
        should_infer = (
            ((self._frame_count - 1) % self.config.inference_interval == 0)
            or (self._last_result is None)
        )

        if not should_infer:
            self._frames_since_inference += 1
            if self._frames_since_inference > self.config.cache_expiry_frames:
                # Caching duration exceeded without a new inference frame
                self._last_result = PhoneDetectionResult(
                    phone_detected=False,
                    confidence=None,
                    bbox=None,
                    detections=[],
                    inference_time_ms=0.0,
                    is_cached=False,
                )
                return self._last_result

            # Return latest valid detection with cached flag
            return PhoneDetectionResult(
                phone_detected=self._last_result.phone_detected,
                confidence=self._last_result.confidence,
                bbox=self._last_result.bbox,
                detections=list(self._last_result.detections),
                inference_time_ms=self._last_result.inference_time_ms,
                is_cached=True,
            )

        # Execute YOLO inference
        t0 = time.perf_counter()
        try:
            results = self.model.predict(
                source=frame,
                classes=[self.phone_class_id],
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                imgsz=self.config.image_size,
                max_det=self.config.max_detections,
                device=self.config.device,
                verbose=False,
            )
        except Exception as exc:
            logger.error("Error during YOLO phone detection inference: %s", exc)
            return PhoneDetectionResult(
                phone_detected=False,
                confidence=None,
                bbox=None,
                detections=[],
                inference_time_ms=0.0,
                is_cached=False,
            )

        inference_time_ms = (time.perf_counter() - t0) * 1000.0

        detections: List[PhoneBoundingBox] = []
        fh, fw = frame.shape[:2]

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                try:
                    conf = float(box.conf[0].item())
                    cls_id = int(box.cls[0].item())
                except (AttributeError, IndexError):
                    continue

                if cls_id == self.phone_class_id and conf >= self.config.confidence_threshold:
                    coords = box.xyxy[0].tolist() if hasattr(box.xyxy[0], "tolist") else list(box.xyxy[0])
                    if isinstance(coords[0], (list, tuple)):
                        coords = coords[0]
                    x1 = max(0, min(fw, int(round(coords[0]))))
                    y1 = max(0, min(fh, int(round(coords[1]))))
                    x2 = max(0, min(fw, int(round(coords[2]))))
                    y2 = max(0, min(fh, int(round(coords[3]))))

                    if x2 > x1 and y2 > y1:
                        detections.append(
                            PhoneBoundingBox(
                                x1=x1,
                                y1=y1,
                                x2=x2,
                                y2=y2,
                                confidence=conf,
                                class_name=self.class_name,
                            )
                        )

        # Sort detections by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)

        phone_detected = len(detections) > 0
        primary_conf = detections[0].confidence if phone_detected else None
        primary_bbox = detections[0].as_tuple if phone_detected else None

        result = PhoneDetectionResult(
            phone_detected=phone_detected,
            confidence=primary_conf,
            bbox=primary_bbox,
            detections=detections,
            inference_time_ms=inference_time_ms,
            is_cached=False,
        )

        self._last_result = result
        self._frames_since_inference = 0
        return result

    def draw(self, frame: np.ndarray, result: PhoneDetectionResult) -> None:
        """Renders bounding boxes and confidence badges for detected phones.

        Args:
            frame: BGR video frame to mutate in-place.
            result: Current phone detection result.
        """
        if (
            frame is None
            or not self.config.draw_boxes
            or not result.phone_detected
            or not result.detections
        ):
            return

        try:
            from ui.theme import UIColors
            box_color = UIColors.WARNING
            label_bg_color = UIColors.CARD_BG
            text_color = UIColors.TEXT_PRIMARY
        except Exception:
            box_color = (0, 200, 255)
            label_bg_color = (20, 24, 33)
            text_color = (255, 255, 255)

        for det in result.detections:
            # Draw primary bounding rectangle
            cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), box_color, 2, cv2.LINE_AA)

            # Format label: rounded percentage without excessive decimal noise
            label = f"CELL PHONE {int(round(det.confidence * 100))}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.44
            thickness = 1

            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Position label pill above bounding box if space permits, otherwise inside top
            pill_h = text_h + 8
            pill_w = text_w + 10

            if det.y1 - pill_h >= 0:
                pill_y1 = det.y1 - pill_h
                pill_y2 = det.y1
            else:
                pill_y1 = det.y1
                pill_y2 = det.y1 + pill_h

            pill_x1 = det.x1
            pill_x2 = min(frame.shape[1], det.x1 + pill_w)

            # Draw background pill and border
            cv2.rectangle(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), label_bg_color, -1)
            cv2.rectangle(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), box_color, 1, cv2.LINE_AA)

            # Draw label text
            text_x = pill_x1 + 5
            text_y = pill_y2 - 5
            cv2.putText(frame, label, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

    def reset(self) -> None:
        """Resets frame counters and cached detection states."""
        self._frame_count = 0
        self._frames_since_inference = 0
        self._last_result = PhoneDetectionResult()
