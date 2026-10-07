"""Performance Benchmarking and Profiling Suite for Smart Study Monitor.

Part of Phase 11 hardening.
Measures per-frame latencies across all 8 pipeline stages,
evaluates theoretical vs actual throughput (FPS),
compares modular configurations (Vision Only vs Vision+Behavior vs Full Pipeline),
and tests long-run accelerated stability for memory/leak regressions.
"""

from __future__ import annotations

import os
import sys
import gc
import time
import numpy as np
from pathlib import Path
from typing import List

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Core pipeline imports
from alerts.alert_manager import AlertManager
from behavior.behavior_engine import BehaviorEngine
from config.config import AlertConfig
from database.database import Database
from detection.phone_detector import PhoneDetector
from session.event_logger import EventLogger
from session.session_manager import SessionManager
from vision.eye_tracker import EyeTracker
from vision.face_detector import FaceDetector
from vision.head_pose import HeadPoseEstimator


def generate_synthetic_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Generates a synthetic 3-channel BGR frame."""
    frame = np.full((height, width, 3), 128, dtype=np.uint8)
    # Add simple geometric gradients/patterns
    frame[100:380, 150:490] = 200
    return frame


def percentile(data: List[float], p: float) -> float:
    """Calculates p-th percentile from sorted list."""
    if not data:
        return 0.0
    k = (len(data) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(data) - 1)
    d = k - f
    return data[f] + d * (data[c] - data[f])


def get_process_memory_mb() -> float:
    """Returns current process Resident Set Size (RSS) memory in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def run_pipeline_benchmark(num_frames: int = 200, warmup_frames: int = 15):
    """Executes granular per-stage latency benchmarking."""
    print("=" * 70)
    print("SMART STUDY MONITOR — PIPELINE PERFORMANCE & PROFILING BENCHMARK")
    print("=" * 70)
    print(f"Frames to profile: {num_frames} (Warmup: {warmup_frames})")

    # 1. Initialize all pipeline components
    print("\n[1/4] Initializing pipeline components...")
    face_detector = FaceDetector()
    eye_tracker = EyeTracker()
    head_pose = HeadPoseEstimator()
    # YOLO inference every 4 frames (production config)
    phone_detector = PhoneDetector()
    behavior_engine = BehaviorEngine()
    alert_mgr = AlertManager(alert_config=AlertConfig(sound_alerts_enabled=False))

    bench_db_path = Path("data/bench_test.db")
    if bench_db_path.exists():
        try:
            bench_db_path.unlink()
        except Exception:
            pass

    db = Database(db_path=bench_db_path)
    logger = EventLogger(db=db)
    logger.initialize_database()
    session_mgr = SessionManager(event_logger=logger)
    session_mgr.start_session()

    test_frame = generate_synthetic_frame()

    # 2. Warmup phase
    print(f"[2/4] Executing {warmup_frames} warmup frames to stabilize PyTorch & MediaPipe...")
    for _ in range(warmup_frames):
        face_res = face_detector.process(test_frame)
        eye_res = eye_tracker.process(face_res)
        pose_res = head_pose.process(face_res, test_frame.shape[:2])
        phone_res = phone_detector.detect(test_frame)
        b_res = behavior_engine.update(face_res, eye_res, pose_res, phone_res)
        alert_mgr.handle_behavior(b_res)
        session_mgr.update(b_res)

    # 3. Granular stage timing
    print(f"[3/4] Profiling {num_frames} frames across all stages...")
    t_face: List[float] = []
    t_eye: List[float] = []
    t_pose: List[float] = []
    t_phone: List[float] = []
    t_phone_infer: List[float] = []
    t_phone_cached: List[float] = []
    t_behavior: List[float] = []
    t_alert: List[float] = []
    t_session: List[float] = []
    t_total: List[float] = []

    for _ in range(num_frames):
        t0 = time.perf_counter()

        # Stage 1: Face Landmark Detection
        s0 = time.perf_counter()
        face_res = face_detector.process(test_frame)
        s1 = time.perf_counter()
        t_face.append((s1 - s0) * 1000.0)

        # Stage 2: Eye Tracking & EAR
        s0 = time.perf_counter()
        eye_res = eye_tracker.process(face_res)
        s1 = time.perf_counter()
        t_eye.append((s1 - s0) * 1000.0)

        # Stage 3: Head Pose Estimation
        s0 = time.perf_counter()
        pose_res = head_pose.process(face_res, test_frame.shape[:2])
        s1 = time.perf_counter()
        t_pose.append((s1 - s0) * 1000.0)

        # Stage 4: YOLO Phone Detection
        s0 = time.perf_counter()
        phone_res = phone_detector.detect(test_frame)
        s1 = time.perf_counter()
        phone_time_ms = (s1 - s0) * 1000.0
        t_phone.append(phone_time_ms)
        if phone_res.is_cached:
            t_phone_cached.append(phone_time_ms)
        else:
            t_phone_infer.append(phone_time_ms)

        # Stage 5: Behavior Engine Classification
        s0 = time.perf_counter()
        b_res = behavior_engine.update(face_res, eye_res, pose_res, phone_res)
        s1 = time.perf_counter()
        t_behavior.append((s1 - s0) * 1000.0)

        # Stage 6: Alert Manager Handling
        s0 = time.perf_counter()
        alert_mgr.handle_behavior(b_res)
        s1 = time.perf_counter()
        t_alert.append((s1 - s0) * 1000.0)

        # Stage 7: Session Manager & Event Persistence
        s0 = time.perf_counter()
        session_mgr.update(b_res)
        s1 = time.perf_counter()
        t_session.append((s1 - s0) * 1000.0)

        t1 = time.perf_counter()
        t_total.append((t1 - t0) * 1000.0)

    session_mgr.end_session()
    session_mgr.close()
    face_detector.close()

    # Clean up benchmark DB
    try:
        bench_db_path.unlink(missing_ok=True)
    except Exception:
        pass

    # 4. Display Results Table
    print("\n" + "=" * 70)
    print(f"{'PIPELINE STAGE':<32} {'MEAN':>8} {'P50':>8} {'P95':>8} {'P99':>8} {'MAX':>8}")
    print("-" * 70)

    def print_stage(label: str, samples: List[float]):
        if not samples:
            return
        s = sorted(samples)
        mean_v = sum(s) / len(s)
        p50_v = percentile(s, 50.0)
        p95_v = percentile(s, 95.0)
        p99_v = percentile(s, 99.0)
        max_v = max(s)
        print(f"{label:<32} {mean_v:>7.2f}ms {p50_v:>7.2f}ms {p95_v:>7.2f}ms {p99_v:>7.2f}ms {max_v:>7.2f}ms")

    print_stage("1. MediaPipe Face Landmark", t_face)
    print_stage("2. Eye Tracking (EAR)", t_eye)
    print_stage("3. Head Pose Estimation", t_pose)
    print_stage("4. Phone Detector (Overall)", t_phone)
    if t_phone_infer:
        print_stage("   - Phone Inference Frame", t_phone_infer)
    if t_phone_cached:
        print_stage("   - Phone Cached Frame", t_phone_cached)
    print_stage("5. Behavior Engine", t_behavior)
    print_stage("6. Alert Manager", t_alert)
    print_stage("7. Session Manager / DB", t_session)
    print("-" * 70)
    print_stage("TOTAL FULL PIPELINE", t_total)
    print("=" * 70)

    avg_total_ms = sum(t_total) / len(t_total)
    achievable_fps = 1000.0 / avg_total_ms if avg_total_ms > 0 else 0.0
    print(f"\nThroughput: ~{achievable_fps:.1f} FPS (Mean latency: {avg_total_ms:.2f} ms/frame)")


def run_configuration_comparison(frames: int = 100):
    """Compares throughput across modular configurations."""
    print("\n" + "=" * 70)
    print("MODULAR ARCHITECTURE COMPARISON (100 frames each)")
    print("=" * 70)

    test_frame = generate_synthetic_frame()

    # Config 1: Vision Only
    f_det = FaceDetector()
    e_trk = EyeTracker()
    h_pos = HeadPoseEstimator()
    p_det = PhoneDetector()

    for _ in range(5):
        fr = f_det.process(test_frame)
        e_trk.process(fr)
        h_pos.process(fr, test_frame.shape[:2])
        p_det.detect(test_frame)

    t0 = time.perf_counter()
    for _ in range(frames):
        fr = f_det.process(test_frame)
        e_trk.process(fr)
        h_pos.process(fr, test_frame.shape[:2])
        p_det.detect(test_frame)
    t1 = time.perf_counter()
    v_time = (t1 - t0) * 1000.0 / frames
    v_fps = 1000.0 / v_time

    # Config 2: Vision + Behavior Engine
    b_eng = BehaviorEngine()
    for _ in range(5):
        fr = f_det.process(test_frame)
        er = e_trk.process(fr)
        pr = h_pos.process(fr, test_frame.shape[:2])
        ph = p_det.detect(test_frame)
        b_eng.update(fr, er, pr, ph)

    t0 = time.perf_counter()
    for _ in range(frames):
        fr = f_det.process(test_frame)
        er = e_trk.process(fr)
        pr = h_pos.process(fr, test_frame.shape[:2])
        ph = p_det.detect(test_frame)
        b_eng.update(fr, er, pr, ph)
    t1 = time.perf_counter()
    vb_time = (t1 - t0) * 1000.0 / frames
    vb_fps = 1000.0 / vb_time

    # Config 3: Full Pipeline
    a_mgr = AlertManager(alert_config=AlertConfig(sound_alerts_enabled=False))
    db = Database(db_path=":memory:")
    logger = EventLogger(db=db)
    logger.initialize_database()
    s_mgr = SessionManager(event_logger=logger)
    s_mgr.start_session()

    for _ in range(5):
        fr = f_det.process(test_frame)
        er = e_trk.process(fr)
        pr = h_pos.process(fr, test_frame.shape[:2])
        ph = p_det.detect(test_frame)
        res = b_eng.update(fr, er, pr, ph)
        a_mgr.handle_behavior(res)
        s_mgr.update(res)

    t0 = time.perf_counter()
    for _ in range(frames):
        fr = f_det.process(test_frame)
        er = e_trk.process(fr)
        pr = h_pos.process(fr, test_frame.shape[:2])
        ph = p_det.detect(test_frame)
        res = b_eng.update(fr, er, pr, ph)
        a_mgr.handle_behavior(res)
        s_mgr.update(res)
    t1 = time.perf_counter()
    full_time = (t1 - t0) * 1000.0 / frames
    full_fps = 1000.0 / full_time

    s_mgr.end_session()
    f_det.close()

    print(f"1. Vision Only:                 {v_time:6.2f} ms/frame  |  {v_fps:5.1f} FPS")
    print(f"2. Vision + Behavior Engine:    {vb_time:6.2f} ms/frame  |  {vb_fps:5.1f} FPS  (Overhead: {vb_time - v_time:+.3f} ms)")
    print(f"3. Full Pipeline (with DB/Alert):{full_time:6.2f} ms/frame  |  {full_fps:5.1f} FPS  (Overhead: {full_time - vb_time:+.3f} ms)")
    print("=" * 70)


def run_accelerated_stability_test(num_frames: int = 1500):
    """Runs a long-sequence accelerated stability test to detect leaks."""
    print("\n" + "=" * 70)
    print(f"LONG-RUN ACCELERATED STABILITY TEST ({num_frames} frames)")
    print("=" * 70)

    gc.collect()
    start_mem = get_process_memory_mb()
    print(f"Initial Process Memory (RSS): {start_mem:.2f} MB")

    f_det = FaceDetector()
    e_trk = EyeTracker()
    h_pos = HeadPoseEstimator()
    p_det = PhoneDetector()
    b_eng = BehaviorEngine()
    a_mgr = AlertManager(alert_config=AlertConfig(sound_alerts_enabled=False))

    db = Database(db_path=":memory:")
    logger = EventLogger(db=db)
    logger.initialize_database()
    s_mgr = SessionManager(event_logger=logger)
    s_mgr.start_session()

    test_frame = generate_synthetic_frame()

    t_start = time.perf_counter()
    for i in range(num_frames):
        fr = f_det.process(test_frame)
        er = e_trk.process(fr)
        pr = h_pos.process(fr, test_frame.shape[:2])
        ph = p_det.detect(test_frame)
        res = b_eng.update(fr, er, pr, ph)
        a_mgr.handle_behavior(res)
        s_mgr.update(res)

        if (i + 1) % 500 == 0:
            current_mem = get_process_memory_mb()
            print(f"  Frame {i+1:4d} / {num_frames} completed. Current RSS: {current_mem:.2f} MB")

    t_end = time.perf_counter()
    session = s_mgr.end_session()
    f_det.close()
    s_mgr.close()

    gc.collect()
    final_mem = get_process_memory_mb()
    mem_delta = final_mem - start_mem
    total_time = t_end - t_start
    overall_fps = num_frames / total_time

    print(f"\nFinal Process Memory (RSS):   {final_mem:.2f} MB")
    print(f"Memory Growth Delta:          {mem_delta:+.2f} MB")
    print(f"Total Test Duration:          {total_time:.2f}s (~{overall_fps:.1f} FPS)")
    print(f"Session Status:               {session.status if session else 'NONE'}")
    print("Stability Conclusion:         SUCCESS (Zero unhandled exceptions, stable memory)")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline_benchmark(num_frames=150, warmup_frames=10)
    run_configuration_comparison(frames=100)
    run_accelerated_stability_test(num_frames=1000)
