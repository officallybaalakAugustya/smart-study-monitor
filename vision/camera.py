"""Camera capture module for Smart Study Monitor.

Provides a clean, testable abstraction over OpenCV's VideoCapture
supporting configurable resolutions, backends, frame rates, and safe resource handling.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np

from config.config import CameraConfig, config


class Camera:
    """Manages webcam capture, hardware settings, and real-time FPS calculation."""

    def __init__(self, camera_config: Optional[CameraConfig] = None) -> None:
        self.config = camera_config or config.camera
        self.config.validate()
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 0.0
        self._prev_frame_time: float = 0.0
        self._frame_count: int = 0
        self._last_fps_update: float = 0.0

    def start(self) -> bool:
        """Initializes and opens the camera capture device."""
        if self._cap is not None and self._cap.isOpened():
            return True

        cap = self._create_capture()
        if cap is None or not cap.isOpened():
            self._cap = None
            return False

        # Configure frame dimensions and target FPS
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.config.frame_width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.config.frame_height))
        cap.set(cv2.CAP_PROP_FPS, float(self.config.target_fps))

        self._cap = cap
        self._prev_frame_time = time.perf_counter()
        self._last_fps_update = self._prev_frame_time
        self._frame_count = 0
        return True

    def _create_capture(self) -> Optional[cv2.VideoCapture]:
        """Creates cv2.VideoCapture respecting configured backend."""
        backend_upper = self.config.backend.upper()
        if backend_upper == "DSHOW":
            cap = cv2.VideoCapture(self.config.device_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                # Fallback to default backend if DirectShow is unavailable
                cap = cv2.VideoCapture(self.config.device_index)
            return cap
        elif backend_upper == "MSMF":
            return cv2.VideoCapture(self.config.device_index, cv2.CAP_MSMF)
        else:
            return cv2.VideoCapture(self.config.device_index)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reads a single frame from the webcam and updates FPS metrics.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success_status, frame)
        """
        if self._cap is None or not self._cap.isOpened():
            return False, None

        try:
            ret, frame = self._cap.read()
        except Exception:
            return False, None

        if not ret or frame is None:
            return False, None

        self._update_fps()
        return True, frame

    def _update_fps(self) -> None:
        """Calculates smoothed frame rate per second from actual frame timings."""
        current_time = time.perf_counter()
        self._frame_count += 1
        elapsed = current_time - self._last_fps_update

        if elapsed >= self.config.fps_update_interval_sec:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._last_fps_update = current_time

    @property
    def fps(self) -> float:
        """Returns the current measured FPS."""
        return self._fps

    def is_opened(self) -> bool:
        """Checks if the camera device is currently active and opened."""
        return self._cap is not None and self._cap.isOpened()

    def release(self) -> None:
        """Safely releases the webcam capture device."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            finally:
                self._cap = None

    def __enter__(self) -> "Camera":
        if not self.start():
            raise RuntimeError(
                f"Failed to open webcam at index {self.config.device_index}."
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
