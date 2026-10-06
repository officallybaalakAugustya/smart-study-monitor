"""Tests for config/config.py.

Verifies default values, custom instantiation, and validation error paths.
"""

import pytest
from config.config import (
    CameraConfig,
    DisplayConfig,
    FaceDetectorConfig,
    EyeTrackerConfig,
    HeadPoseConfig,
    Config,
)



class TestCameraConfig:
    def test_defaults(self):
        cfg = CameraConfig()
        assert cfg.device_index == 0
        assert cfg.backend == "DSHOW"
        assert cfg.frame_width == 640
        assert cfg.frame_height == 480
        assert cfg.target_fps == 30
        cfg.validate()

    def test_negative_device_index(self):
        cfg = CameraConfig(device_index=-1)
        with pytest.raises(ValueError, match="device_index must be >= 0"):
            cfg.validate()

    def test_invalid_dimensions(self):
        cfg = CameraConfig(frame_width=0, frame_height=480)
        with pytest.raises(ValueError, match="frame_width and frame_height must be positive"):
            cfg.validate()

    def test_invalid_fps(self):
        cfg = CameraConfig(target_fps=0)
        with pytest.raises(ValueError, match="target_fps must be > 0"):
            cfg.validate()

    def test_invalid_fps_interval(self):
        cfg = CameraConfig(fps_update_interval_sec=-0.5)
        with pytest.raises(ValueError, match="fps_update_interval_sec must be > 0"):
            cfg.validate()


class TestDisplayConfig:
    def test_defaults(self):
        cfg = DisplayConfig()
        assert cfg.window_width == 960
        assert cfg.window_height == 720
        assert cfg.window_name == "Smart Study Monitor - Live Feed"
        cfg.validate()

    def test_empty_window_name(self):
        cfg = DisplayConfig(window_name="   ")
        with pytest.raises(ValueError, match="window_name cannot be empty"):
            cfg.validate()

    def test_invalid_dimensions(self):
        cfg = DisplayConfig(window_width=-100)
        with pytest.raises(ValueError, match="window_width and window_height must be positive"):
            cfg.validate()

    def test_invalid_heartbeat(self):
        cfg = DisplayConfig(heartbeat_interval_sec=0)
        with pytest.raises(ValueError, match="heartbeat_interval_sec must be > 0"):
            cfg.validate()

    def test_invalid_panel_opacity(self):
        cfg = DisplayConfig(panel_opacity=-0.1)
        with pytest.raises(ValueError, match="panel_opacity must be in"):
            cfg.validate()

        cfg2 = DisplayConfig(panel_opacity=1.5)
        with pytest.raises(ValueError, match="panel_opacity must be in"):
            cfg2.validate()



class TestFaceDetectorConfig:
    def test_defaults(self):
        cfg = FaceDetectorConfig()
        assert cfg.num_faces == 1
        assert cfg.min_face_detection_confidence == 0.5
        cfg.validate()

    def test_invalid_num_faces(self):
        cfg = FaceDetectorConfig(num_faces=0)
        with pytest.raises(ValueError, match="num_faces must be > 0"):
            cfg.validate()

    def test_invalid_confidence(self):
        cfg = FaceDetectorConfig(min_face_detection_confidence=1.5)
        with pytest.raises(ValueError, match="min_face_detection_confidence must be in"):
            cfg.validate()


class TestEyeTrackerConfig:
    def test_defaults(self):
        cfg = EyeTrackerConfig()
        assert cfg.ear_close_threshold == 0.18
        assert cfg.ear_open_threshold == 0.22
        assert cfg.ear_smoothing_alpha == 0.4
        cfg.validate()

    def test_close_greater_than_open_threshold(self):
        """Hysteresis requires close_threshold < open_threshold."""
        cfg = EyeTrackerConfig(ear_close_threshold=0.25, ear_open_threshold=0.20)
        with pytest.raises(ValueError, match="ear_close_threshold.*must be strictly less than ear_open_threshold"):
            cfg.validate()

    def test_equal_thresholds(self):
        cfg = EyeTrackerConfig(ear_close_threshold=0.20, ear_open_threshold=0.20)
        with pytest.raises(ValueError, match="ear_close_threshold.*must be strictly less than ear_open_threshold"):
            cfg.validate()

    def test_negative_threshold(self):
        cfg = EyeTrackerConfig(ear_close_threshold=-0.1)
        with pytest.raises(ValueError, match="ear_close_threshold must be > 0"):
            cfg.validate()

    def test_invalid_smoothing_alpha_zero(self):
        cfg = EyeTrackerConfig(ear_smoothing_alpha=0.0)
        with pytest.raises(ValueError, match="ear_smoothing_alpha must be in"):
            cfg.validate()

    def test_invalid_smoothing_alpha_greater_than_one(self):
        cfg = EyeTrackerConfig(ear_smoothing_alpha=1.2)
        with pytest.raises(ValueError, match="ear_smoothing_alpha must be in"):
            cfg.validate()

    def test_invalid_min_eye_width(self):
        cfg = EyeTrackerConfig(min_eye_width=0.0)
        with pytest.raises(ValueError, match="min_eye_width must be > 0"):
            cfg.validate()


class TestHeadPoseConfig:
    def test_defaults(self):
        cfg = HeadPoseConfig()
        assert cfg.yaw_left_threshold == 15.0
        assert cfg.yaw_right_threshold == -15.0
        assert cfg.pitch_up_threshold == -15.0
        assert cfg.pitch_down_threshold == 15.0
        assert cfg.pose_smoothing_alpha == 0.3
        assert cfg.draw_head_pose is True
        cfg.validate()

    def test_invalid_yaw_left_threshold(self):
        cfg = HeadPoseConfig(yaw_left_threshold=-5.0)
        with pytest.raises(ValueError, match="yaw_left_threshold must be > 0"):
            cfg.validate()

    def test_invalid_yaw_right_threshold(self):
        cfg = HeadPoseConfig(yaw_right_threshold=5.0)
        with pytest.raises(ValueError, match="yaw_right_threshold must be < 0"):
            cfg.validate()

    def test_invalid_pitch_up_threshold(self):
        cfg = HeadPoseConfig(pitch_up_threshold=5.0)
        with pytest.raises(ValueError, match="pitch_up_threshold must be < 0"):
            cfg.validate()

    def test_invalid_pitch_down_threshold(self):
        cfg = HeadPoseConfig(pitch_down_threshold=-5.0)
        with pytest.raises(ValueError, match="pitch_down_threshold must be > 0"):
            cfg.validate()

    def test_invalid_pose_smoothing_alpha(self):
        cfg = HeadPoseConfig(pose_smoothing_alpha=0.0)
        with pytest.raises(ValueError, match="pose_smoothing_alpha must be in"):
            cfg.validate()

        cfg_high = HeadPoseConfig(pose_smoothing_alpha=1.5)
        with pytest.raises(ValueError, match="pose_smoothing_alpha must be in"):
            cfg_high.validate()


class TestMasterConfig:
    def test_master_config_validation(self):
        cfg = Config()
        cfg.validate()

    def test_master_config_propagates_invalid_child(self):
        with pytest.raises(ValueError):
            Config(eye_tracker=EyeTrackerConfig(ear_close_threshold=0.3, ear_open_threshold=0.2))

    def test_master_config_propagates_invalid_head_pose(self):
        with pytest.raises(ValueError):
            Config(head_pose=HeadPoseConfig(yaw_left_threshold=-1.0))

