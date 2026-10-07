"""Visual alert module for Smart Study Monitor.

Part of Phase 7.
Handles non-blocking, temporary visual banners displayed on the video feed
using the centralized design system. Does NOT block or flash.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from ui.theme import UIColors, UITypography


@dataclass
class VisualAlertState:
    """State of an active or expired visual alert."""

    title: str
    message: str
    accent_color: Tuple[int, int, int]
    start_time: float
    duration_sec: float

    def is_active(self, current_time: float) -> bool:
        return (current_time - self.start_time) < self.duration_sec


class VisualAlert:
    """Manages temporary non-blocking visual alert banners on the video frame."""

    def __init__(
        self,
        default_duration_sec: float = 3.0,
        enabled: bool = True,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.default_duration_sec = max(0.0, float(default_duration_sec))
        self.enabled = bool(enabled)
        self._clock = clock or time.monotonic
        self._current_alert: Optional[VisualAlertState] = None

    def trigger(
        self,
        title: str,
        message: str,
        accent_color: Tuple[int, int, int] = UIColors.WARNING,
        duration_sec: Optional[float] = None,
    ) -> None:
        """Triggers a new visual alert, replacing any existing active banner."""
        if not self.enabled:
            return

        dur = duration_sec if duration_sec is not None else self.default_duration_sec
        now = self._clock()
        self._current_alert = VisualAlertState(
            title=title,
            message=message,
            accent_color=accent_color,
            start_time=now,
            duration_sec=dur,
        )

    def clear(self) -> None:
        """Clears the active visual alert."""
        self._current_alert = None

    @property
    def is_active(self) -> bool:
        """Checks if there is a currently active visual alert banner."""
        if not self.enabled or self._current_alert is None:
            return False
        return self._current_alert.is_active(self._clock())

    @property
    def active_alert(self) -> Optional[VisualAlertState]:
        """Returns the active visual alert if not expired."""
        if not self.is_active:
            return None
        return self._current_alert

    def draw(self, frame: np.ndarray) -> None:
        """Renders the alert banner at the top center of the video frame."""
        if not self.is_active or self._current_alert is None or frame is None or frame.size == 0:
            return

        h, w = frame.shape[:2]
        banner_w = min(420, int(w * 0.65))
        banner_h = 44
        x = (w - banner_w) // 2
        y = 52  # Just beneath the top header bar (header height: 44 + 8)

        # Alpha-blend panel background
        roi = frame[y : y + banner_h, x : x + banner_w]
        bg = np.full_like(roi, UIColors.CARD_BG, dtype=np.uint8)
        cv2.addWeighted(bg, 0.90, roi, 0.10, 0, roi)

        # Border with accent color
        color = self._current_alert.accent_color
        cv2.rectangle(frame, (x, y), (x + banner_w - 1, y + banner_h - 1), color, 1, cv2.LINE_AA)

        # Left accent status bar
        cv2.rectangle(frame, (x, y), (x + 5, y + banner_h - 1), color, -1)

        # Title and message
        title_text = self._current_alert.title
        msg_text = self._current_alert.message

        cv2.putText(
            frame,
            title_text,
            (x + 16, y + 18),
            UITypography.FONT,
            UITypography.SCALE_BADGE,
            color,
            UITypography.THICK_BADGE,
            UITypography.LINE_TYPE,
        )

        cv2.putText(
            frame,
            msg_text,
            (x + 16, y + 36),
            UITypography.FONT,
            UITypography.SCALE_CAPTION,
            UIColors.TEXT_SECONDARY,
            1,
            UITypography.LINE_TYPE,
        )
