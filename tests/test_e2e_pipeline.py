"""End-to-end synthetic pipeline test for Smart Study Monitor.

Simulates the complete study monitoring lifecycle from session start through
all behavioral transitions to session conclusion and analytics generation
without requiring a physical webcam or display.
"""

from datetime import datetime, timezone
import pytest

from alerts.alert_manager import AlertManager
from analytics.session_analytics import SessionAnalytics
from behavior.behavior_engine import BehaviorEngine, BehaviorState, BehaviorReason
from config.config import AlertConfig
from dashboard.formatting import (
    prepare_event_table_data,
    prepare_session_history_data,
    prepare_state_distribution_data,
)
from database.database import Database
from detection.phone_detector import PhoneDetectionResult
from session.event_logger import EventLogger
from session.session_manager import SessionManager, SessionStatus
from vision.eye_tracker import EyeState, EyeTrackingResult
from vision.face_detector import FaceDetectionResult
from vision.head_pose import HeadDirection, HeadPoseResult


class SteppableClock:
    """Controllable monotonic clock for deterministic end-to-end simulation."""

    def __init__(self, start: float = 100.0) -> None:
        self.time = start

    def __call__(self) -> float:
        return self.time

    def step(self, seconds: float) -> None:
        self.time += seconds


class SteppableWallClock:
    """Controllable wall clock for deterministic ISO timestamps."""

    def __init__(self) -> None:
        self.time = datetime(2026, 10, 7, 9, 0, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.time

    def step(self, seconds: float) -> None:
        from datetime import timedelta
        self.time += timedelta(seconds=seconds)


class TestEndToEndPipeline:
    def test_full_study_session_lifecycle(self):
        # 1. Setup isolated in-memory DB and test clocks
        db = Database(db_path=":memory:")
        logger = EventLogger(db=db)
        logger.initialize_database()

        mono_clock = SteppableClock(start=100.0)
        wall_clock = SteppableWallClock()

        # Engine with 1.0s confirmation thresholds
        engine = BehaviorEngine(
            clock=mono_clock,
            phone_confirmation_sec=1.0,
            drowsiness_confirmation_sec=1.0,
            head_away_confirmation_sec=1.0,
            away_from_desk_confirmation_sec=15.0,
            no_face_confirmation_sec=2.0,
            state_recovery_sec=1.0,
        )

        alert_mgr = AlertManager(
            alert_config=AlertConfig(sound_alerts_enabled=False),
            clock=mono_clock,
        )
        session_mgr = SessionManager(
            event_logger=logger,
            clock=mono_clock,
            wall_clock=wall_clock,
        )
        analytics = SessionAnalytics(event_logger=logger)

        # Synthetic signal fixtures
        face_on = FaceDetectionResult(face_detected=True)
        face_off = FaceDetectionResult(face_detected=False)

        eye_open = EyeTrackingResult(eye_state=EyeState.OPEN, smoothed_ear=0.30)
        eye_closed = EyeTrackingResult(eye_state=EyeState.CLOSED, smoothed_ear=0.14)

        pose_center = HeadPoseResult(direction=HeadDirection.CENTER, yaw=0.0, pitch=0.0, roll=0.0)
        pose_left = HeadPoseResult(direction=HeadDirection.LEFT, yaw=-35.0, pitch=0.0, roll=0.0)

        phone_no = PhoneDetectionResult(phone_detected=False)
        phone_yes = PhoneDetectionResult(phone_detected=True, confidence=0.92)

        def run_frames(face, eye, pose, phone, duration_sec: float, fps: float = 30.0):
            """Simulates processing frames over a duration."""
            frame_dt = 1.0 / fps
            steps = int(duration_sec * fps)
            for _ in range(steps):
                mono_clock.step(frame_dt)
                wall_clock.step(frame_dt)
                res = engine.update(face, eye, pose, phone)
                alert_mgr.handle_behavior(res)
                session_mgr.update(res)

        # --------------------------------------------------------------
        # Simulation Scenario:
        # START SESSION
        # -> FOCUSED (30s)
        # -> DISTRACTED (10s)
        # -> FOCUSED (20s)
        # -> PHONE_DETECTED (10s)
        # -> FOCUSED (20s)
        # -> DROWSY (10s)
        # -> FOCUSED (20s)
        # -> NO_FACE -> AWAY_FROM_DESK (15s)
        # -> SESSION END
        # --------------------------------------------------------------

        # A. Start Session
        session = session_mgr.start_session()
        assert session_mgr.is_active()

        # B. Stage 1: FOCUSED
        run_frames(face_on, eye_open, pose_center, phone_no, duration_sec=30.0)
        assert session_mgr.get_last_logged_state() == BehaviorState.FOCUSED

        # C. Stage 2: DISTRACTED (head turned away for 10s)
        run_frames(face_on, eye_open, pose_left, phone_no, duration_sec=10.0)
        assert session_mgr.get_last_logged_state() == BehaviorState.DISTRACTED

        # D. Stage 3: Return to FOCUSED (20s)
        run_frames(face_on, eye_open, pose_center, phone_no, duration_sec=20.0)
        assert session_mgr.get_last_logged_state() == BehaviorState.FOCUSED

        # E. Stage 4: PHONE_DETECTED (10s)
        run_frames(face_on, eye_open, pose_center, phone_yes, duration_sec=10.0)
        assert session_mgr.get_last_logged_state() == BehaviorState.PHONE_DETECTED

        # F. Stage 5: Return to FOCUSED (20s)
        run_frames(face_on, eye_open, pose_center, phone_no, duration_sec=20.0)
        assert session_mgr.get_last_logged_state() == BehaviorState.FOCUSED

        # G. Stage 6: DROWSY (eyes closed for 10s)
        run_frames(face_on, eye_closed, pose_center, phone_no, duration_sec=10.0)
        assert session_mgr.get_last_logged_state() == BehaviorState.DROWSY

        # H. Stage 7: Return to FOCUSED (20s)
        run_frames(face_on, eye_open, pose_center, phone_no, duration_sec=20.0)
        assert session_mgr.get_last_logged_state() == BehaviorState.FOCUSED

        # I. Stage 8: NO_FACE -> sustained into AWAY_FROM_DESK (15s)
        run_frames(face_off, eye_open, pose_center, phone_no, duration_sec=15.0)
        assert session_mgr.get_last_logged_state() == BehaviorState.AWAY_FROM_DESK

        # J. End Session
        ended = session_mgr.end_session(status=SessionStatus.COMPLETED.value)
        assert ended is not None
        assert not session_mgr.is_active()
        assert ended.duration_seconds >= 134.0

        # --------------------------------------------------------------
        # Verification & Analytics
        # --------------------------------------------------------------

        # 1. Database records check
        db_session = logger.get_session(session.session_id)
        assert db_session is not None
        assert db_session["status"] == "COMPLETED"
        assert db_session["duration_seconds"] >= 134.0

        db_events = logger.get_session_events(session.session_id)
        assert len(db_events) >= 8  # FOCUSED, DISTRACTED, FOCUSED, PHONE, FOCUSED, DROWSY, FOCUSED, NO_FACE, AWAY

        # 2. Timeline and Durations reconstruction
        durations = analytics.get_state_durations(session.session_id)
        assert durations.focused_seconds > 0.0
        assert durations.distracted_seconds > 0.0
        assert durations.phone_detected_seconds > 0.0
        assert durations.drowsy_seconds > 0.0
        assert durations.away_from_desk_seconds > 0.0

        # Total reconstructed duration must closely match session duration
        assert pytest.approx(durations.total_seconds, abs=0.5) == ended.duration_seconds

        # 3. Focus Score Calculation
        summary = analytics.get_session_summary(session.session_id)
        assert summary is not None
        assert summary.focus_score is not None
        assert 0.0 <= summary.focus_score <= 100.0

        # 4. Dashboard Data Preparation (Consumability check)
        dist_data = prepare_state_distribution_data(summary)
        assert len(dist_data) >= 5

        segments = analytics.get_timeline_segments(session.session_id)
        assert len(segments) == len(db_events)
        event_rows = prepare_event_table_data(segments)
        assert len(event_rows) == len(segments)

        history_rows = prepare_session_history_data([summary])
        assert len(history_rows) == 1
        assert history_rows[0]["Focus Score"] != "N/A"
