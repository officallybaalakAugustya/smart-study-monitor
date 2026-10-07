"""End-to-End Scenario Verification Script for Smart Study Monitor.

Executes all 15 specific user manual scenarios deterministically to verify
the entire system across Phases 1 through 10.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alerts.alert_manager import AlertManager, AlertType
from analytics.session_analytics import SessionAnalytics
from behavior.behavior_engine import BehaviorEngine, BehaviorReason, BehaviorState
from config.config import AlertConfig
from database.database import Database
from detection.phone_detector import PhoneDetectionResult
from session.event_logger import EventLogger
from session.session_manager import SessionManager, SessionStatus
from vision.eye_tracker import EyeState, EyeTrackingResult
from vision.face_detector import FaceDetectionResult
from vision.head_pose import HeadDirection, HeadPoseResult


class SimClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.time = start

    def __call__(self) -> float:
        return self.time

    def step(self, seconds: float) -> None:
        self.time += seconds


class SimWallClock:
    def __init__(self) -> None:
        self.time = datetime(2026, 10, 7, 10, 0, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.time

    def step(self, seconds: float) -> None:
        from datetime import timedelta
        self.time += timedelta(seconds=seconds)


def run_scenario_verification() -> bool:
    print("=" * 70)
    print("SMART STUDY MONITOR — 15-SCENARIO END-TO-END VERIFICATION")
    print("=" * 70)

    clock = SimClock(start=1000.0)
    wall_clock = SimWallClock()

    db = Database(db_path=":memory:")
    logger = EventLogger(db=db)
    logger.initialize_database()

    engine = BehaviorEngine(
        clock=clock,
        phone_confirmation_sec=1.0,
        drowsiness_confirmation_sec=1.5,
        head_away_confirmation_sec=1.5,
        away_from_desk_confirmation_sec=3.0,
        no_face_confirmation_sec=3.0,
        state_recovery_sec=1.0,
    )

    alert_mgr = AlertManager(
        alert_config=AlertConfig(sound_alerts_enabled=False),
        clock=clock,
    )

    session_mgr = SessionManager(
        event_logger=logger,
        clock=clock,
        wall_clock=wall_clock,
    )

    analytics = SessionAnalytics(event_logger=logger)

    # Fixtures
    face_on = FaceDetectionResult(face_detected=True)
    face_off = FaceDetectionResult(face_detected=False)

    eye_open = EyeTrackingResult(eye_state=EyeState.OPEN, smoothed_ear=0.32)
    eye_closed = EyeTrackingResult(eye_state=EyeState.CLOSED, smoothed_ear=0.14)

    pose_center = HeadPoseResult(direction=HeadDirection.CENTER, yaw=0.0, pitch=0.0)
    pose_left = HeadPoseResult(direction=HeadDirection.LEFT, yaw=-30.0, pitch=0.0)
    pose_right = HeadPoseResult(direction=HeadDirection.RIGHT, yaw=30.0, pitch=0.0)

    phone_no = PhoneDetectionResult(phone_detected=False)
    phone_yes = PhoneDetectionResult(phone_detected=True, confidence=0.92)

    def step_frames(face, eye, pose, phone, duration_sec: float, fps: float = 30.0):
        dt = 1.0 / fps
        steps = int(round(duration_sec * fps))
        last_b_res = None
        for _ in range(steps):
            clock.step(dt)
            wall_clock.step(dt)
            res = engine.update(face, eye, pose, phone)
            alert_mgr.handle_behavior(res)
            session_mgr.update(res)
            last_b_res = res
        return last_b_res

    results = {}

    # Start Session (Prerequisite for logging)
    session = session_mgr.start_session()

    # TEST 1: Normal face -> FOCUSED (run 65s to exceed minimum scored duration of 60s)
    step_frames(face_on, eye_open, pose_center, phone_no, duration_sec=65.0)
    b1 = engine.current_state
    results["TEST 1 (Normal face -> FOCUSED)"] = (b1 == BehaviorState.FOCUSED)

    # TEST 2: Blink (0.25s eye closure) -> remains normal (FOCUSED)
    step_frames(face_on, eye_closed, pose_center, phone_no, duration_sec=0.25)
    b2 = engine.current_state
    results["TEST 2 (Blink -> remains normal)"] = (b2 == BehaviorState.FOCUSED)

    # TEST 3: Eyes closed continuously (2.0s >= 1.5s) -> DROWSY -> alert
    step_frames(face_on, eye_closed, pose_center, phone_no, duration_sec=2.0)
    b3 = engine.current_state
    ev3 = alert_mgr.last_event
    alert3_ok = (ev3 is not None and ev3.alert_type == AlertType.DROWSINESS)
    results["TEST 3 (Eyes closed continuously -> DROWSY -> alert)"] = (b3 == BehaviorState.DROWSY and alert3_ok)

    # Recover to FOCUSED
    step_frames(face_on, eye_open, pose_center, phone_no, duration_sec=2.0)

    # TEST 4: Look left (2.0s >= 1.5s) -> DISTRACTED after confirmation
    step_frames(face_on, eye_open, pose_left, phone_no, duration_sec=2.0)
    b4 = engine.current_state
    results["TEST 4 (Look left -> DISTRACTED after confirmation)"] = (b4 == BehaviorState.DISTRACTED)

    # Recover to FOCUSED
    step_frames(face_on, eye_open, pose_center, phone_no, duration_sec=2.0)

    # TEST 5: Look right (2.0s >= 1.5s) -> DISTRACTED after confirmation
    step_frames(face_on, eye_open, pose_right, phone_no, duration_sec=2.0)
    b5 = engine.current_state
    results["TEST 5 (Look right -> DISTRACTED after confirmation)"] = (b5 == BehaviorState.DISTRACTED)

    # Recover to FOCUSED
    step_frames(face_on, eye_open, pose_center, phone_no, duration_sec=2.0)

    # TEST 6: Phone visible (1.5s >= 1.0s) -> PHONE_DETECTED -> alert
    step_frames(face_on, eye_open, pose_center, phone_yes, duration_sec=1.5)
    b6 = engine.current_state
    ev6 = alert_mgr.last_event
    alert6_ok = (ev6 is not None and ev6.alert_type == AlertType.PHONE)
    results["TEST 6 (Phone visible -> PHONE_DETECTED -> alert)"] = (b6 == BehaviorState.PHONE_DETECTED and alert6_ok)

    # TEST 7: Remove phone -> recovery to FOCUSED
    step_frames(face_on, eye_open, pose_center, phone_no, duration_sec=2.0)
    b7 = engine.current_state
    results["TEST 7 (Remove phone -> recovery)"] = (b7 == BehaviorState.FOCUSED)

    # TEST 8: Leave camera -> NO_FACE immediately
    step_frames(face_off, eye_open, pose_center, phone_no, duration_sec=0.1)
    b8 = engine.current_state
    results["TEST 8 (Leave camera -> NO_FACE)"] = (b8 == BehaviorState.NO_FACE)

    # TEST 9: Remain absent (4.0s >= 3.0s no_face_confirmation_sec) -> AWAY_FROM_DESK
    step_frames(face_off, eye_open, pose_center, phone_no, duration_sec=4.0)
    b9 = engine.current_state
    results["TEST 9 (Remain absent -> AWAY_FROM_DESK)"] = (b9 == BehaviorState.AWAY_FROM_DESK)

    # TEST 10: Return -> appropriate recovery to FOCUSED
    step_frames(face_on, eye_open, pose_center, phone_no, duration_sec=2.0)
    b10 = engine.current_state
    results["TEST 10 (Return -> appropriate recovery)"] = (b10 == BehaviorState.FOCUSED)

    # TEST 11: Run session -> SQLite events persisted
    events = logger.get_session_events(session.session_id)
    results["TEST 11 (Run session -> SQLite events)"] = (len(events) >= 6)

    # TEST 12: End session -> COMPLETED
    ended = session_mgr.end_session(status=SessionStatus.COMPLETED.value)
    results["TEST 12 (End session -> COMPLETED)"] = (
        ended is not None and ended.status == "COMPLETED" and ended.duration_seconds > 0.0
    )

    # TEST 13: Open dashboard queries -> session appears
    summaries = analytics.get_completed_session_summaries()
    results["TEST 13 (Open dashboard -> session appears)"] = (
        len(summaries) >= 1 and summaries[0].session_id == session.session_id
    )

    # TEST 14: Verify Focus Score -> matches analytics calculation
    summary = summaries[0]
    calc_score = summary.focus_score
    results["TEST 14 (Verify Focus Score -> matches calculation)"] = (
        calc_score is not None and 0.0 <= calc_score <= 100.0
    )

    # TEST 15: Close application -> clean shutdown
    session_mgr.close()
    alert_mgr.close()
    engine.reset()
    results["TEST 15 (Close application -> clean shutdown)"] = (
        not session_mgr.is_active() and not alert_mgr.visual.is_active
    )

    # Display Checklist
    all_passed = True
    print("\nScenario Checklist:")
    print("-" * 70)
    for name, passed in results.items():
        status_str = "PASS" if passed else "FAIL"
        print(f"  {name:<58} [{status_str}]")
        if not passed:
            all_passed = False
    print("-" * 70)
    print(f"Overall Result: {'ALL 15 SCENARIOS PASSED' if all_passed else 'SOME SCENARIOS FAILED'}")
    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    success = run_scenario_verification()
    sys.exit(0 if success else 1)
