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
    PhoneDetectorConfig,
    BehaviorConfig,
    AlertConfig,
    DatabaseConfig,
    AnalyticsConfig,
    DashboardConfig,
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


class TestPhoneDetectorConfig:
    def test_defaults(self):
        cfg = PhoneDetectorConfig()
        assert cfg.confidence_threshold == 0.45
        assert cfg.iou_threshold == 0.45
        assert cfg.image_size == 320
        assert cfg.inference_interval == 4
        assert cfg.max_detections == 5
        assert cfg.cache_expiry_frames == 15
        assert cfg.enabled is True
        assert cfg.device == "cpu"
        assert cfg.draw_boxes is True
        cfg.validate()

    def test_empty_model_path(self):
        cfg = PhoneDetectorConfig(model_path="")
        with pytest.raises(ValueError, match="model_path cannot be empty"):
            cfg.validate()

    def test_invalid_confidence_negative(self):
        cfg = PhoneDetectorConfig(confidence_threshold=-0.1)
        with pytest.raises(ValueError, match="confidence_threshold must be in"):
            cfg.validate()

    def test_invalid_confidence_greater_than_one(self):
        cfg = PhoneDetectorConfig(confidence_threshold=1.5)
        with pytest.raises(ValueError, match="confidence_threshold must be in"):
            cfg.validate()

    def test_invalid_iou_negative(self):
        cfg = PhoneDetectorConfig(iou_threshold=-0.1)
        with pytest.raises(ValueError, match="iou_threshold must be in"):
            cfg.validate()

    def test_invalid_image_size(self):
        cfg = PhoneDetectorConfig(image_size=0)
        with pytest.raises(ValueError, match="image_size must be positive"):
            cfg.validate()

    def test_invalid_inference_interval(self):
        cfg = PhoneDetectorConfig(inference_interval=0)
        with pytest.raises(ValueError, match="inference_interval must be >= 1"):
            cfg.validate()

    def test_invalid_max_detections(self):
        cfg = PhoneDetectorConfig(max_detections=0)
        with pytest.raises(ValueError, match="max_detections must be >= 1"):
            cfg.validate()

    def test_invalid_cache_expiry_frames(self):
        cfg = PhoneDetectorConfig(cache_expiry_frames=0)
        with pytest.raises(ValueError, match="cache_expiry_frames must be >= 1"):
            cfg.validate()


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

    def test_master_config_propagates_invalid_phone_detector(self):
        with pytest.raises(ValueError):
            Config(phone_detector=PhoneDetectorConfig(confidence_threshold=-0.5))

    def test_master_config_propagates_invalid_behavior(self):
        with pytest.raises(ValueError):
            Config(behavior=BehaviorConfig(phone_confirmation_sec=-1.0))

    def test_master_config_propagates_invalid_alerts(self):
        with pytest.raises(ValueError):
            Config(alerts=AlertConfig(alert_volume=2.5))


class TestAlertConfig:
    def test_defaults(self):
        cfg = AlertConfig()
        assert cfg.alerts_enabled is True
        assert cfg.sound_alerts_enabled is True
        assert cfg.visual_alerts_enabled is True
        assert cfg.alert_volume == 0.7
        assert cfg.visual_alert_duration_sec == 3.0
        assert cfg.phone_alert_cooldown_sec == 5.0
        assert cfg.drowsiness_alert_cooldown_sec == 5.0
        assert cfg.distraction_alert_cooldown_sec == 5.0
        assert cfg.away_alert_cooldown_sec == 10.0
        assert cfg.away_alert_sound_mode == "distraction"
        cfg.validate()

    def test_invalid_volume(self):
        with pytest.raises(ValueError, match="alert_volume must be in"):
            AlertConfig(alert_volume=-0.1).validate()
        with pytest.raises(ValueError, match="alert_volume must be in"):
            AlertConfig(alert_volume=1.5).validate()

    def test_invalid_cooldown(self):
        with pytest.raises(ValueError, match="phone_alert_cooldown_sec must be >= 0.0"):
            AlertConfig(phone_alert_cooldown_sec=-1.0).validate()

    def test_invalid_away_sound_mode(self):
        with pytest.raises(ValueError, match="away_alert_sound_mode must be"):
            AlertConfig(away_alert_sound_mode="unknown_mode").validate()


class TestBehaviorConfig:
    def test_defaults(self):
        cfg = BehaviorConfig()
        assert cfg.phone_confirmation_sec == 1.0
        assert cfg.drowsiness_confirmation_sec == 1.5
        assert cfg.head_away_confirmation_sec == 1.5
        assert cfg.away_from_desk_confirmation_sec == 3.0
        assert cfg.no_face_confirmation_sec == 3.0
        assert cfg.state_recovery_sec == 1.0
        assert cfg.enabled is True
        cfg.validate()

    def test_invalid_phone_confirmation(self):
        cfg = BehaviorConfig(phone_confirmation_sec=-0.5)
        with pytest.raises(ValueError, match="phone_confirmation_sec must be >= 0"):
            cfg.validate()

    def test_invalid_drowsiness_confirmation(self):
        cfg = BehaviorConfig(drowsiness_confirmation_sec=-1.0)
        with pytest.raises(ValueError, match="drowsiness_confirmation_sec must be >= 0"):
            cfg.validate()

    def test_invalid_head_away_confirmation(self):
        cfg = BehaviorConfig(head_away_confirmation_sec=-0.1)
        with pytest.raises(ValueError, match="head_away_confirmation_sec must be >= 0"):
            cfg.validate()

    def test_invalid_away_from_desk_confirmation(self):
        cfg = BehaviorConfig(away_from_desk_confirmation_sec=-2.0)
        with pytest.raises(ValueError, match="away_from_desk_confirmation_sec must be >= 0"):
            cfg.validate()

    def test_invalid_no_face_confirmation(self):
        cfg = BehaviorConfig(no_face_confirmation_sec=-1.0)
        with pytest.raises(ValueError, match="no_face_confirmation_sec must be >= 0"):
            cfg.validate()

    def test_invalid_state_recovery(self):
        cfg = BehaviorConfig(state_recovery_sec=-0.5)
        with pytest.raises(ValueError, match="state_recovery_sec must be >= 0"):
            cfg.validate()


class TestDatabaseConfig:
    def test_defaults(self):
        cfg = DatabaseConfig()
        assert "study_monitor.db" in cfg.db_path
        assert cfg.timeout_sec == 10.0
        cfg.validate()

    def test_empty_db_path(self):
        cfg = DatabaseConfig(db_path="")
        with pytest.raises(ValueError, match="db_path cannot be empty"):
            cfg.validate()

    def test_whitespace_db_path(self):
        cfg = DatabaseConfig(db_path="   ")
        with pytest.raises(ValueError, match="db_path cannot be empty"):
            cfg.validate()

    def test_invalid_timeout(self):
        cfg = DatabaseConfig(timeout_sec=0)
        with pytest.raises(ValueError, match="timeout_sec must be > 0"):
            cfg.validate()

    def test_master_config_includes_database(self):
        master = Config()
        assert hasattr(master, "database")
        assert master.database.timeout_sec == 10.0
        assert master.data_dir.name == "data"
        master.validate()


class TestAnalyticsConfig:
    def test_defaults(self):
        cfg = AnalyticsConfig()
        assert cfg.minimum_scored_duration_sec == 60.0
        assert cfg.focused_weight == 1.0
        assert cfg.distracted_weight == 0.0
        assert cfg.drowsy_weight == 0.0
        assert cfg.phone_detected_weight == 0.0
        assert cfg.away_from_desk_weight == 0.0
        assert cfg.no_face_weight == 0.0
        assert cfg.unknown_weight == 0.0
        assert cfg.exclude_no_face is True
        assert cfg.exclude_unknown is True
        cfg.validate()

    def test_invalid_minimum_duration(self):
        cfg = AnalyticsConfig(minimum_scored_duration_sec=-1.0)
        with pytest.raises(ValueError, match="minimum_scored_duration_sec must be >= 0"):
            cfg.validate()

    def test_negative_weight(self):
        cfg = AnalyticsConfig(focused_weight=-0.5)
        with pytest.raises(ValueError, match="focused_weight must be a non-negative finite number"):
            cfg.validate()

    def test_nan_weight(self):
        cfg = AnalyticsConfig(distracted_weight=float("nan"))
        with pytest.raises(ValueError, match="distracted_weight must be a non-negative finite number"):
            cfg.validate()

    def test_master_config_includes_analytics(self):
        master = Config()
        assert hasattr(master, "analytics")
        assert master.analytics.minimum_scored_duration_sec == 60.0
        master.validate()


class TestDashboardConfig:
    def test_defaults(self):
        cfg = DashboardConfig()
        assert cfg.page_title == "Smart Study Monitor - Analytics"
        assert cfg.session_history_limit == 50
        cfg.validate()

    def test_empty_page_title(self):
        cfg = DashboardConfig(page_title="")
        with pytest.raises(ValueError, match="page_title cannot be empty"):
            cfg.validate()

    def test_invalid_history_limit(self):
        cfg = DashboardConfig(session_history_limit=0)
        with pytest.raises(ValueError, match="session_history_limit must be > 0"):
            cfg.validate()

    def test_master_config_includes_dashboard(self):
        master = Config()
        assert hasattr(master, "dashboard")
        assert master.dashboard.session_history_limit == 50
        master.validate()

