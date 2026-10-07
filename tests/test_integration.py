"""Integration tests verifying contracts across module boundaries.

Tests data flow between:
Camera -> FaceDetector -> EyeTracker & HeadPose -> PhoneDetector ->
BehaviorEngine -> AlertManager & SessionManager -> EventLogger ->
SQLite -> SessionAnalytics -> Dashboard Formatting.
"""

import numpy as np
import pytest

from alerts.alert_manager import AlertManager
from analytics.session_analytics import SessionAnalytics
from behavior.behavior_engine import BehaviorEngine, BehaviorResult, BehaviorState, BehaviorReason
from dashboard.formatting import prepare_event_table_data, prepare_state_distribution_data
from database.database import Database
from detection.phone_detector import PhoneDetectionResult
from session.event_logger import EventLogger
from session.session_manager import SessionManager
from vision.eye_tracker import EyeState, EyeTracker, EyeTrackingResult
from vision.face_detector import FaceDetectionResult
from vision.head_pose import HeadDirection, HeadPoseEstimator, HeadPoseResult


@pytest.fixture
def integrated_pipeline():
    """Sets up an end-to-end integrated environment with isolated in-memory DB."""
    db = Database(db_path=":memory:")
    logger = EventLogger(db=db)
    logger.initialize_database()

    engine = BehaviorEngine(
        phone_confirmation_sec=0.01,
        drowsiness_confirmation_sec=0.01,
        head_away_confirmation_sec=0.01,
        away_from_desk_confirmation_sec=0.02,
        no_face_confirmation_sec=0.02,
        state_recovery_sec=0.01,
    )
    alert_mgr = AlertManager()
    session_mgr = SessionManager(event_logger=logger)
    analytics = SessionAnalytics(event_logger=logger)

    return {
        "db": db,
        "logger": logger,
        "engine": engine,
        "alert_mgr": alert_mgr,
        "session_mgr": session_mgr,
        "analytics": analytics,
    }


class TestModuleBoundaryIntegration:
    def test_vision_to_behavior_boundary(self):
        """Tests that vision data structures flow cleanly into BehaviorEngine."""
        face_result = FaceDetectionResult(face_detected=True)
        eye_result = EyeTrackingResult(eye_state=EyeState.OPEN, smoothed_ear=0.30)
        pose_result = HeadPoseResult(direction=HeadDirection.CENTER, yaw=0.0, pitch=0.0, roll=0.0)
        phone_result = PhoneDetectionResult(phone_detected=False)

        engine = BehaviorEngine()
        behavior_result = engine.update(face_result, eye_result, pose_result, phone_result)

        assert isinstance(behavior_result, BehaviorResult)
        assert behavior_result.state in (BehaviorState.FOCUSED, BehaviorState.UNKNOWN)
        assert "face_detected" in behavior_result.signals
        assert behavior_result.signals["face_detected"] is True
        assert behavior_result.signals["eye_state"] == "OPEN"
        assert behavior_result.signals["head_direction"] == "CENTER"

    def test_behavior_to_alert_and_session_boundaries(self, integrated_pipeline):
        """Tests BehaviorResult consumed concurrently by AlertManager and SessionManager."""
        p = integrated_pipeline
        session = p["session_mgr"].start_session()

        result = BehaviorResult(
            state=BehaviorState.PHONE_DETECTED,
            confidence=0.95,
            reason=BehaviorReason.PHONE_DETECTED,
            signals={"phone_detected": True},
        )

        # 1. Alert Manager receives BehaviorResult
        p["alert_mgr"].handle_behavior(result)
        # Visual alert should be active
        assert p["alert_mgr"].visual.is_active is True

        # 2. Session Manager receives BehaviorResult
        event_id = p["session_mgr"].update(result)
        assert event_id is not None

        # Verify event logged in SQLite via EventLogger
        events = p["logger"].get_session_events(session.session_id)
        assert len(events) == 1
        assert events[0]["state"] == "PHONE_DETECTED"
        assert events[0]["confidence"] == 0.95

    def test_session_manager_to_sqlite_to_analytics_boundary(self, integrated_pipeline):
        """Tests the full data flow from SessionManager through SQLite into SessionAnalytics."""
        p = integrated_pipeline
        session = p["session_mgr"].start_session()

        res_focused = BehaviorResult(state=BehaviorState.FOCUSED, confidence=0.9, reason="focused")
        p["session_mgr"].update(res_focused)

        res_distracted = BehaviorResult(state=BehaviorState.DISTRACTED, confidence=0.85, reason="head_away")
        p["session_mgr"].update(res_distracted)

        ended = p["session_mgr"].end_session()
        assert ended is not None

        # Analytics consumes data from SQLite via EventLogger
        summary = p["analytics"].get_session_summary(session.session_id)
        assert summary is not None
        assert summary.session_id == session.session_id
        assert summary.transition_count == 1
        assert summary.status == "COMPLETED"
        assert summary.state_durations.total_seconds >= 0.0

    def test_analytics_to_dashboard_formatting_boundary(self, integrated_pipeline):
        """Tests that SessionAnalytics models feed directly into dashboard data preparation."""
        p = integrated_pipeline
        s_id = p["logger"].create_session(start_time="2026-10-07T10:00:00+00:00")
        p["logger"].log_behavior_event(s_id, "2026-10-07T10:00:00+00:00", BehaviorState.FOCUSED, 0.0, 0.9, "focused")
        p["logger"].log_behavior_event(s_id, "2026-10-07T10:15:00+00:00", BehaviorState.DISTRACTED, 0.0, 0.85, "head_away")
        p["logger"].end_session(s_id, end_time="2026-10-07T10:20:00+00:00", duration_seconds=1200.0)

        summary = p["analytics"].get_session_summary(s_id)
        segments = p["analytics"].get_timeline_segments(s_id)

        # Dashboard data preparation
        dist_data = prepare_state_distribution_data(summary)
        event_data = prepare_event_table_data(segments)

        assert len(dist_data) >= 1
        assert dist_data[0]["label"] == "Focused"
        assert len(event_data) == 2
        assert event_data[0]["State"] == "Focused"
        assert event_data[1]["State"] == "Distracted"
