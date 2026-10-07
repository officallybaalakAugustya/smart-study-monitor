"""Failure injection and edge-case resiliency tests for Smart Study Monitor.

Part of Phase 11 hardening.
Verifies graceful handling of:
- Missing models and invalid paths
- Invalid, empty, or None frame inputs
- Audio system / sound file failures
- Database and event logger error handling
- Malformed data in analytics and dashboard formatting
- Out-of-bounds configuration inputs
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest

from alerts.alert_manager import AlertManager
from alerts.sound_alert import SoundAlert
from behavior.behavior_engine import BehaviorEngine, BehaviorResult, BehaviorState, BehaviorReason
from config.config import (
    AlertConfig,
    BehaviorConfig,
    CameraConfig,
    DatabaseConfig,
    EyeTrackerConfig,
    FaceDetectorConfig,
    HeadPoseConfig,
    PhoneDetectorConfig,
)
from dashboard.formatting import (
    format_duration,
    format_focus_score,
    format_iso_datetime,
    get_state_color,
    get_state_label,
    prepare_event_table_data,
    prepare_session_history_data,
    prepare_state_distribution_data,
)
from database.database import Database
from detection.phone_detector import PhoneDetectionResult, PhoneDetector
from session.event_logger import EventLogger
from session.session_manager import SessionManager
from vision.eye_tracker import EyeState, EyeTracker, EyeTrackingResult
from vision.face_detector import FaceDetectionResult, FaceDetector
from vision.head_pose import HeadDirection, HeadPoseEstimator, HeadPoseResult


# ==============================================================================
# 1. Missing Model & Asset Failures
# ==============================================================================

class TestMissingAssetsAndModels:
    """Verifies that missing model weights or sound files produce safe errors."""

    def test_face_detector_missing_model_raises_filenotfound(self):
        bogus_cfg = FaceDetectorConfig(model_path="nonexistent_face_detector.task")
        with pytest.raises(FileNotFoundError, match="MediaPipe face-landmarker model not found"):
            FaceDetector(detector_config=bogus_cfg)

    def test_phone_detector_missing_model_raises_runtime_or_filenotfound(self, tmp_path):
        bogus_path = tmp_path / "sub" / "nonexistent_yolo.pt"
        bogus_cfg = PhoneDetectorConfig(model_path=str(bogus_path))
        with pytest.raises(RuntimeError, match="Failed to load YOLO model"):
            PhoneDetector(config=bogus_cfg)

    def test_sound_alert_missing_file_returns_false_without_crash(self):
        sound = SoundAlert(enabled=True)
        # Attempt to load a completely non-existent sound file
        success = sound.load_sound("missing_alert", "nonexistent/sounds/test_beep.mp3")
        assert success is False
        assert sound.play("missing_alert") is False

    def test_sound_alert_uninitialized_mixer_returns_false(self):
        sound = SoundAlert(mixer_module=None)
        sound._mixer_available = False
        assert sound.load_sound("test", "some/path.mp3") is False
        assert sound.play("test") is False


# ==============================================================================
# 2. Corrupted & Invalid Input Frames
# ==============================================================================

class TestCorruptedFrameInputs:
    """Verifies that CV processing modules handle corrupted or None frames safely."""

    def test_face_detector_handles_none_frame(self):
        # We can test with a mocked or initialized landmarker
        detector = FaceDetector.__new__(FaceDetector)
        detector._cfg = FaceDetectorConfig()
        detector._landmarker = MagicMock()

        result = detector.process(None)  # type: ignore
        assert isinstance(result, FaceDetectionResult)
        assert result.face_detected is False
        assert result.raw_result is None

    def test_face_detector_handles_empty_frame(self):
        detector = FaceDetector.__new__(FaceDetector)
        detector._cfg = FaceDetectorConfig()
        detector._landmarker = MagicMock()

        empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
        result = detector.process(empty_frame)
        assert isinstance(result, FaceDetectionResult)
        assert result.face_detected is False

    def test_phone_detector_handles_none_frame(self):
        mock_model = MagicMock()
        mock_model.names = {0: "person", 67: "cell phone"}
        detector = PhoneDetector(model=mock_model)

        res = detector.detect(None)  # type: ignore
        assert isinstance(res, PhoneDetectionResult)
        assert res.phone_detected is False

    def test_phone_detector_handles_empty_frame(self):
        mock_model = MagicMock()
        mock_model.names = {0: "person", 67: "cell phone"}
        detector = PhoneDetector(model=mock_model)

        empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
        res = detector.detect(empty_frame)
        assert isinstance(res, PhoneDetectionResult)
        assert res.phone_detected is False

    def test_eye_tracker_handles_empty_or_none_landmarks(self):
        tracker = EyeTracker()
        # Case A: face_detected is False
        res = tracker.process(FaceDetectionResult(face_detected=False, raw_result=None))
        assert res.eye_state == EyeState.UNKNOWN
        assert res.average_ear is None
        assert res.smoothed_ear is None

        # Case B: raw_result has empty landmarks list
        mock_raw = MagicMock()
        mock_raw.face_landmarks = []
        res = tracker.process(FaceDetectionResult(face_detected=True, raw_result=mock_raw))
        assert res.eye_state == EyeState.UNKNOWN

    def test_head_pose_handles_empty_or_none_landmarks(self):
        pose = HeadPoseEstimator()
        # Case A: face_detected is False
        res = pose.process(FaceDetectionResult(face_detected=False, raw_result=None), (480, 640))
        assert res.direction == HeadDirection.UNKNOWN
        assert res.yaw is None
        assert res.pitch is None
        assert res.roll is None

        # Case B: raw_result has empty landmarks list
        mock_raw = MagicMock()
        mock_raw.face_landmarks = []
        res = pose.process(FaceDetectionResult(face_detected=True, raw_result=mock_raw), (480, 640))
        assert res.direction == HeadDirection.UNKNOWN


# ==============================================================================
# 3. Behavior Engine & Alert Manager Fault Resiliency
# ==============================================================================

class TestPipelineFaultResiliency:
    """Verifies that downstream consumers handle unexpected signals safely."""

    def test_behavior_engine_handles_none_or_partial_signals(self):
        engine = BehaviorEngine()
        # Missing landmark signals
        face = FaceDetectionResult(face_detected=False)
        eye = EyeTrackingResult()
        pose = HeadPoseResult()
        phone = PhoneDetectionResult()

        res = engine.update(face, eye, pose, phone)
        assert isinstance(res, BehaviorResult)
        assert res.state in (BehaviorState.NO_FACE, BehaviorState.UNKNOWN)
        assert 0.0 <= res.confidence <= 1.0

    def test_alert_manager_with_disabled_or_missing_sounds_never_raises(self):
        alert_cfg = AlertConfig(
            alerts_enabled=True,
            sound_alerts_enabled=True,
            phone_alert_sound_path="nonexistent/phone.mp3",
            distraction_alert_sound_path="nonexistent/distraction.mp3",
            drowsiness_alert_sound_path="nonexistent/drowsy.mp3",
        )
        alert_mgr = AlertManager(alert_config=alert_cfg)

        # Trigger every possible alert state
        states = [
            (BehaviorState.PHONE_DETECTED, BehaviorReason.PHONE_DETECTED),
            (BehaviorState.DROWSY, BehaviorReason.EYES_CLOSED),
            (BehaviorState.DISTRACTED, BehaviorReason.HEAD_AWAY),
            (BehaviorState.AWAY_FROM_DESK, BehaviorReason.HEAD_AWAY),
            (BehaviorState.FOCUSED, BehaviorReason.FOCUSED),
            (BehaviorState.NO_FACE, BehaviorReason.FACE_NOT_DETECTED),
            (BehaviorState.UNKNOWN, BehaviorReason.INSUFFICIENT_SIGNAL),
        ]

        for state, reason in states:
            b_res = BehaviorResult(
                state=state,
                state_started_at=100.0,
                state_duration=1.0,
                confidence=0.9,
                reason=reason,
                signals={},
            )
            event = alert_mgr.handle_behavior(b_res)
            # Must return None or AlertEvent without throwing
            assert event is None or hasattr(event, "alert_type")


# ==============================================================================
# 4. Database & Logger Edge Cases
# ==============================================================================

class TestDatabaseAndLoggerEdgeCases:
    """Verifies database constraints and error handling."""

    def test_database_foreign_key_enforcement(self):
        db = Database(db_path=":memory:")
        logger = EventLogger(db=db)
        logger.initialize_database()

        # Attempt to insert event for nonexistent session_id 999
        import sqlite3
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                """
                INSERT INTO behavior_events (session_id, timestamp, state, state_duration, confidence, reason, signals_json)
                VALUES (999, '2026-10-07T00:00:00Z', 'FOCUSED', 5.0, 1.0, 'focused', '{}');
                """
            )

    def test_database_safe_reinitialization(self):
        db = Database(db_path=":memory:")
        # Repeated initialize calls should be idempotent
        db.initialize()
        db.initialize()
        db.initialize()
        tables = db.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        table_names = [t["name"] for t in tables]
        assert "sessions" in table_names
        assert "behavior_events" in table_names

    def test_session_manager_update_when_inactive_is_noop(self):
        db = Database(db_path=":memory:")
        logger = EventLogger(db=db)
        mgr = SessionManager(event_logger=logger)
        assert not mgr.is_active()

        # Updating while session is not started must not raise
        b_res = BehaviorResult(
            state=BehaviorState.FOCUSED,
            state_started_at=1.0,
            state_duration=1.0,
            confidence=1.0,
            reason="focused",
            signals={},
        )
        mgr.update(b_res)
        assert mgr.get_current_session() is None


# ==============================================================================
# 5. Dashboard Formatting Resiliency
# ==============================================================================

class TestDashboardFormattingResiliency:
    """Verifies formatting utilities on None, malformed, or boundary values."""

    def test_format_iso_datetime_robustness(self):
        assert format_iso_datetime(None) == "--"
        assert format_iso_datetime("") == "--"
        assert format_iso_datetime("not-a-datetime") == "not-a-datetime"
        assert format_iso_datetime("2026-10-07T12:00:00Z") == "Oct 07, 2026, 12:00:00"

    def test_format_duration_negative_or_zero(self):
        assert format_duration(0.0) == "00:00:00"
        assert format_duration(-10.0) == "00:00:00"
        assert format_duration(3665.0) == "01:01:05"

    def test_format_focus_score_none_or_valid(self):
        assert format_focus_score(None) == "N/A"
        assert format_focus_score(88.42) == "88.4"

    def test_formatting_with_empty_collections(self):
        assert prepare_session_history_data([]) == []
        assert prepare_event_table_data([]) == []
        assert prepare_state_distribution_data(None) == []

    def test_state_color_and_label_unknown_fallback(self):
        # Unknown state string returns fallback color and original string
        assert get_state_color("NON_EXISTENT_STATE") == "#94A3B8"
        assert get_state_label("NON_EXISTENT_STATE") == "NON_EXISTENT_STATE"


# ==============================================================================
# 6. Configuration Validation Boundaries
# ==============================================================================

class TestConfigValidationBoundaries:
    """Verifies configuration dataclass validation rejects invalid bounds."""

    def test_camera_config_invalid_fps(self):
        with pytest.raises(ValueError, match="target_fps must be > 0"):
            CameraConfig(target_fps=0).validate()

    def test_face_detector_config_invalid_confidence(self):
        with pytest.raises(ValueError, match="min_face_detection_confidence must be in"):
            FaceDetectorConfig(min_face_detection_confidence=1.5).validate()

    def test_eye_tracker_config_invalid_threshold(self):
        with pytest.raises(ValueError):
            EyeTrackerConfig(ear_close_threshold=0.30, ear_open_threshold=0.20).validate()

    def test_head_pose_config_invalid_yaw(self):
        with pytest.raises(ValueError, match="yaw_left_threshold must be > 0"):
            HeadPoseConfig(yaw_left_threshold=-10.0).validate()  # must be > 0

    def test_phone_detector_config_invalid_threshold(self):
        with pytest.raises(ValueError, match="confidence_threshold"):
            PhoneDetectorConfig(confidence_threshold=1.5).validate()

    def test_behavior_config_negative_confirmation(self):
        with pytest.raises(ValueError, match="phone_confirmation_sec"):
            BehaviorConfig(phone_confirmation_sec=-0.5).validate()

    def test_alert_config_invalid_volume(self):
        with pytest.raises(ValueError):
            AlertConfig(alert_volume=1.5).validate()

    def test_database_config_empty_path(self):
        with pytest.raises(ValueError, match="db_path cannot be empty"):
            DatabaseConfig(db_path="").validate()
