"""Unit tests for vision/camera.py.

Verifies Camera class initialization, configuration handling, context manager,
and error handling without requiring a physical camera.
"""

import pytest
from config.config import CameraConfig
from vision.camera import Camera


class TestCameraUnit:
    def test_camera_init_with_valid_config(self):
        cfg = CameraConfig(device_index=0, frame_width=640, frame_height=480, target_fps=30)
        camera = Camera(camera_config=cfg)
        assert camera.config == cfg
        assert not camera.is_opened()
        assert camera.fps == 0.0

    def test_camera_init_with_invalid_config_raises(self):
        cfg = CameraConfig(device_index=-1)
        with pytest.raises(ValueError):
            Camera(camera_config=cfg)

    def test_camera_release_when_not_started(self):
        camera = Camera()
        camera.release()
        assert not camera.is_opened()

    def test_camera_read_when_not_opened_returns_false_none(self):
        camera = Camera()
        ret, frame = camera.read()
        assert ret is False
        assert frame is None

    def test_camera_context_manager_on_unavailable_device(self):
        # Index 99999 is typically unavailable
        cfg = CameraConfig(device_index=99999)
        with pytest.raises(RuntimeError, match="Failed to open webcam at index 99999"):
            with Camera(camera_config=cfg) as _:
                pass
