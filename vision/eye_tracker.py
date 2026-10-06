"""Eye tracking and Eye Aspect Ratio (EAR) module for Smart Study Monitor.

Uses MediaPipe face landmarks to calculate EAR for both eyes independently,
applies exponential moving average smoothing, and determines whether the
eyes are OPEN, CLOSED, or UNKNOWN (no face).

EAR Formula (per eye):
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

    p1, p4 = horizontal corners (lateral, medial)
    p2, p6 = first vertical pair  (upper-outer, lower-outer)
    p3, p5 = second vertical pair (upper-inner, lower-inner)

Hysteresis is used to prevent flickering when EAR hovers near a threshold:
    EAR <= ear_close_threshold  → CLOSED
    EAR >= ear_open_threshold   → OPEN
    Between the two             → retain previous state

MediaPipe 468-point Face Mesh landmark indices:
    Right eye (subject's right):
        p1=33  (outer corner)    p4=133 (inner corner)
        p2=160 (upper-outer)     p5=153 (lower-inner)
        p3=158 (upper-inner)     p6=144 (lower-outer)

    Left eye (subject's left):
        p1=362 (outer corner)    p4=263 (inner corner)
        p2=385 (upper-outer)     p5=373 (lower-inner)
        p3=387 (upper-inner)     p6=380 (lower-outer)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config.config import EyeTrackerConfig, config
from vision.face_detector import FaceDetectionResult


# ---------------------------------------------------------------------------
# Landmark indices for EAR (6 points per eye)
# ---------------------------------------------------------------------------

# Each tuple: (p1, p2, p3, p4, p5, p6)
RIGHT_EYE_EAR_INDICES: Tuple[int, ...] = (33, 160, 158, 133, 153, 144)
LEFT_EYE_EAR_INDICES: Tuple[int, ...] = (362, 385, 387, 263, 373, 380)


class EyeState(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    UNKNOWN = "UNKNOWN"


@dataclass
class EyeTrackingResult:
    """Per-frame eye tracking output."""

    left_ear: Optional[float] = None
    right_ear: Optional[float] = None
    average_ear: Optional[float] = None
    smoothed_ear: Optional[float] = None
    eye_state: EyeState = EyeState.UNKNOWN


# ---------------------------------------------------------------------------
# Pure helper functions (stateless, easily testable)
# ---------------------------------------------------------------------------

def _euclidean(
    x1: float, y1: float, x2: float, y2: float,
) -> float:
    """Euclidean distance between two 2-D points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def compute_ear(
    landmarks: List,
    indices: Tuple[int, ...],
    min_width: float = 0.001,
) -> Optional[float]:
    """Computes the Eye Aspect Ratio for one eye.

    Args:
        landmarks: List of NormalizedLandmark objects (468 points for primary face).
        indices:   Tuple of 6 landmark indices (p1, p2, p3, p4, p5, p6).
        min_width: Minimum horizontal eye distance below which EAR is not
                   computed (returns None to prevent division by near-zero).

    Returns:
        EAR as a float, or None if computation is invalid.
    """
    if landmarks is None or len(landmarks) < max(indices) + 1:
        return None

    p1_idx, p2_idx, p3_idx, p4_idx, p5_idx, p6_idx = indices

    try:
        p1 = landmarks[p1_idx]
        p2 = landmarks[p2_idx]
        p3 = landmarks[p3_idx]
        p4 = landmarks[p4_idx]
        p5 = landmarks[p5_idx]
        p6 = landmarks[p6_idx]
    except (IndexError, TypeError, AttributeError):
        return None

    horizontal = _euclidean(p1.x, p1.y, p4.x, p4.y)
    if horizontal < min_width:
        return None

    vert_a = _euclidean(p2.x, p2.y, p6.x, p6.y)
    vert_b = _euclidean(p3.x, p3.y, p5.x, p5.y)

    ear = (vert_a + vert_b) / (2.0 * horizontal)
    return ear


# ---------------------------------------------------------------------------
# EyeTracker class
# ---------------------------------------------------------------------------

class EyeTracker:
    """Calculates EAR, applies smoothing, and determines eye state.

    Initialised once and reused per frame. Holds internal EMA state
    and the previous eye state for hysteresis.
    """

    def __init__(self, tracker_config: Optional[EyeTrackerConfig] = None) -> None:
        self._cfg = tracker_config or config.eye_tracker
        self._cfg.validate()
        self._smoothed_ear: Optional[float] = None
        self._prev_state: EyeState = EyeState.UNKNOWN

    def reset(self) -> None:
        """Resets internal smoothing and state history."""
        self._smoothed_ear = None
        self._prev_state = EyeState.UNKNOWN

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, face_result: FaceDetectionResult) -> EyeTrackingResult:
        """Computes EAR and eye state from the face detection result."""
        if not face_result.face_detected or face_result.raw_result is None:
            self.reset()
            return EyeTrackingResult()

        landmarks_list = face_result.raw_result.face_landmarks
        if not landmarks_list:
            self.reset()
            return EyeTrackingResult()

        primary_landmarks = landmarks_list[0]

        left_ear = compute_ear(
            primary_landmarks, LEFT_EYE_EAR_INDICES, self._cfg.min_eye_width,
        )
        right_ear = compute_ear(
            primary_landmarks, RIGHT_EYE_EAR_INDICES, self._cfg.min_eye_width,
        )

        if left_ear is None or right_ear is None:
            self.reset()
            return EyeTrackingResult()

        avg_ear = (left_ear + right_ear) / 2.0

        # Exponential moving average smoothing
        smoothed = self._apply_ema(avg_ear)

        # Eye state with hysteresis
        eye_state = self._classify_state(smoothed)

        return EyeTrackingResult(
            left_ear=left_ear,
            right_ear=right_ear,
            average_ear=avg_ear,
            smoothed_ear=smoothed,
            eye_state=eye_state,
        )

    def draw(
        self,
        bgr_frame: np.ndarray,
        face_result: FaceDetectionResult,
        eye_result: EyeTrackingResult,
    ) -> None:
        """Draws the 6 EAR keypoints and contours for each eye onto the frame."""
        if not self._cfg.draw_eye_landmarks:
            return
        if not face_result.face_detected or face_result.raw_result is None:
            return
        if eye_result.eye_state is EyeState.UNKNOWN:
            return

        landmarks_list = face_result.raw_result.face_landmarks
        if not landmarks_list:
            return

        landmarks = landmarks_list[0]
        h, w = bgr_frame.shape[:2]

        color = (0, 255, 0) if eye_result.eye_state is EyeState.OPEN else (0, 0, 255)

        for indices in (LEFT_EYE_EAR_INDICES, RIGHT_EYE_EAR_INDICES):
            pts = []
            for idx in indices:
                lm = landmarks[idx]
                px = int(lm.x * w)
                py = int(lm.y * h)
                pts.append((px, py))
                cv2.circle(bgr_frame, (px, py), 2, color, -1, cv2.LINE_AA)

            # Draw connecting lines between the 6 points to outline the eye contour
            for i in range(len(pts)):
                cv2.line(
                    bgr_frame, pts[i], pts[(i + 1) % len(pts)],
                    color, 1, cv2.LINE_AA,
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_ema(self, raw_ear: float) -> float:
        """Applies exponential moving average to the raw EAR value."""
        alpha = self._cfg.ear_smoothing_alpha
        if self._smoothed_ear is None:
            self._smoothed_ear = raw_ear
        else:
            self._smoothed_ear = alpha * raw_ear + (1.0 - alpha) * self._smoothed_ear
        return self._smoothed_ear

    def _classify_state(self, smoothed_ear: float) -> EyeState:
        """Classifies eye state using hysteresis thresholds."""
        if smoothed_ear <= self._cfg.ear_close_threshold:
            self._prev_state = EyeState.CLOSED
        elif smoothed_ear >= self._cfg.ear_open_threshold:
            self._prev_state = EyeState.OPEN
        # In between: retain previous state (hysteresis band)
        return self._prev_state
