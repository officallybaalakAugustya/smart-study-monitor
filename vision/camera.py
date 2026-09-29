"""Camera capture module for Smart Study Monitor.

Provides a clean and robust abstraction over OpenCV's VideoCapture
supporting resolution configuration, FPS measurement, and safe resource handling.
"""

import time
from typing import Optional, Tuple
import cv2
import numpy as np
from config.config import CameraConfig, config


class Camera:
    """Manages webcam capture, properties, and FPS tracking."""

    def __init__(self, camera_config: Optional[CameraConfig] = None) -> None:
        self.config = camera_config or config.camera
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 0.0
        self._prev_frame_time: float = 0.0
        self._frame_count: int = 0
        self._fps_update_interval: float = 0.5
        self._last_fps_update: float = 0.0

    def start(self) -> bool:
        """Initializes and opens the camera capture device."""
        if self._cap is not None and self._cap.isOpened():
            return True

        # Try opening with DirectShow on Windows for faster initialization, fallback to default backend
        cap = cv2.VideoCapture(self.config.device_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.config.device_index)

        if not cap.isOpened():
            self._cap = None
            return False

        # Configure frame dimensions
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)

        self._cap = cap
        self._prev_frame_time = time.perf_counter()
        self._last_fps_update = self._prev_frame_time
        self._frame_count = 0
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reads a single frame from the webcam and updates FPS metrics.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success_status, frame)
        """
        if self._cap is None or not self._cap.isOpened():
            return False, None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            return False, None

        self._update_fps()
        return True, frame

    def _update_fps(self) -> None:
        """Calculates smoothed frame rate per second."""
        current_time = time.perf_counter()
        self._frame_count += 1
        elapsed = current_time - self._last_fps_update

        if elapsed >= self._fps_update_interval:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._last_fps_update = current_time

    @property
    def fps(self) -> float:
        """Returns the current measured FPS."""
        return self._fps

    def is_opened(self) -> bool:
        """Checks if the camera is currently active and opened."""
        return self._cap is not None and self._cap.isOpened()

    def release(self) -> None:
        """Safely releases the webcam capture device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        if not self.start():
            raise RuntimeError(
                f"Failed to open webcam at index {self.config.device_index}."
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
