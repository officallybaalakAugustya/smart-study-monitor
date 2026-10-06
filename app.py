"""Smart Study Monitor - Main Application Entry Point.

Runs the live Phase 1-4 computer vision pipeline:
1. Camera capture (vision/camera.py)
2. Face landmark detection (vision/face_detector.py)
3. Eye tracking and EAR calculation (vision/eye_tracker.py)
4. Head pose estimation & looking-away classification (vision/head_pose.py)
5. Modern AI study monitor UI/UX overlay (ui/overlay.py)
"""

from __future__ import annotations

import sys
import time
import cv2

from config.config import config
from ui.overlay import OverlayRenderer
from vision.camera import Camera
from vision.face_detector import FaceDetector, FaceDetectionResult
from vision.eye_tracker import EyeTracker, EyeTrackingResult
from vision.head_pose import HeadPoseEstimator, HeadPoseResult


def _force_window_foreground(window_name: str) -> None:
    """Forces the OpenCV window to appear visibly in front of other windows on Windows."""
    # Method 1: OpenCV fullscreen toggle trick (cross-version)
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.waitKey(1)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        cv2.waitKey(1)
    except cv2.error:
        pass

    # Method 2: Win32 API foreground attachment
    try:
        import ctypes
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        hwnd = user32.FindWindowW(None, window_name)
        if hwnd:
            foreground_hwnd = user32.GetForegroundWindow()
            foreground_tid = user32.GetWindowThreadProcessId(foreground_hwnd, None)
            our_tid = kernel32.GetCurrentThreadId()
            if foreground_tid != our_tid:
                user32.AttachThreadInput(foreground_tid, our_tid, True)
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            user32.SetForegroundWindow(hwnd)
            user32.BringWindowToTop(hwnd)
            if foreground_tid != our_tid:
                user32.AttachThreadInput(foreground_tid, our_tid, False)
    except (AttributeError, OSError):
        pass


def main() -> None:
    print("=" * 60, flush=True)
    print("  Smart Study Monitor - Real-Time AI Study Monitor", flush=True)
    print("=" * 60, flush=True)

    # Validate global configuration
    try:
        config.validate()
    except ValueError as exc:
        print(f"[CONFIGURATION ERROR] {exc}", flush=True)
        sys.exit(1)

    print(f"Initializing camera at index: {config.camera.device_index}...", flush=True)
    camera = Camera()
    if not camera.start():
        print(f"\n[ERROR] Could not open webcam at index {config.camera.device_index}.", flush=True)
        print("Troubleshooting steps:", flush=True)
        print(" 1. Ensure your webcam is not in use by another application.", flush=True)
        print(" 2. Check Windows Camera Privacy settings.", flush=True)
        print(" 3. Try changing `device_index` to 1 in config/config.py.", flush=True)
        sys.exit(1)

    print("[OK] Camera initialized.", flush=True)

    try:
        face_detector = FaceDetector()
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}", flush=True)
        camera.release()
        sys.exit(1)
    except Exception as exc:
        print(f"\n[ERROR] Failed to initialize face detector: {exc}", flush=True)
        camera.release()
        sys.exit(1)

    print("[OK] Face detector initialized.", flush=True)

    try:
        eye_tracker = EyeTracker()
    except ValueError as exc:
        print(f"\n[ERROR] Failed to initialize eye tracker: {exc}", flush=True)
        face_detector.close()
        camera.release()
        sys.exit(1)

    print("[OK] Eye tracker initialized.", flush=True)

    try:
        head_pose_estimator = HeadPoseEstimator()
    except ValueError as exc:
        print(f"\n[ERROR] Failed to initialize head pose estimator: {exc}", flush=True)
        face_detector.close()
        camera.release()
        sys.exit(1)

    print("[OK] Head pose estimator initialized.", flush=True)

    # ------------------------------------------------------------------
    # UI Overlay Renderer & Interactive Toggles
    # ------------------------------------------------------------------
    win_cfg = config.display
    overlay = OverlayRenderer(opacity=win_cfg.panel_opacity)
    show_hud = win_cfg.show_hud
    show_landmarks = win_cfg.show_face_landmarks
    debug_mode = win_cfg.debug_mode

    # ------------------------------------------------------------------
    # Create and configure OpenCV display window
    # ------------------------------------------------------------------
    window_name = win_cfg.window_name
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, win_cfg.window_x, win_cfg.window_y)
    cv2.resizeWindow(window_name, win_cfg.window_width, win_cfg.window_height)

    # Prime initial frame
    ret, initial_frame = camera.read()
    if ret and initial_frame is not None:
        cv2.imshow(window_name, initial_frame)
    cv2.waitKey(50)

    if win_cfg.force_foreground:
        _force_window_foreground(window_name)
        cv2.resizeWindow(window_name, win_cfg.window_width, win_cfg.window_height)
        cv2.moveWindow(window_name, win_cfg.window_x, win_cfg.window_y)

    print(f"[OK] Window '{window_name}' opened ({win_cfg.window_width}x{win_cfg.window_height}).", flush=True)
    print("     Shortcuts: [Q/ESC] Exit | [L] Toggle Mesh | [D] Toggle Debug | [H] Toggle HUD\n", flush=True)

    # ------------------------------------------------------------------
    # Main frame processing loop
    # ------------------------------------------------------------------
    last_face_result = FaceDetectionResult(face_detected=False)
    last_eye_result = EyeTrackingResult()
    last_pose_result = HeadPoseResult()
    frame_count = 0
    last_heartbeat = time.perf_counter()

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                print("[WARNING] Failed to grab frame from camera. Exiting...", flush=True)
                break

            frame_count += 1

            # 1. Face landmark detection (MediaPipe Tasks)
            last_face_result = face_detector.process(frame)

            # 2. Eye tracking & EAR calculation (uses same face landmarks)
            last_eye_result = eye_tracker.process(last_face_result)

            # 3. Head pose estimation (uses same face landmarks)
            last_pose_result = head_pose_estimator.process(last_face_result, frame.shape[:2])

            # 4. Render computer vision landmark visualizations (if enabled)
            if show_landmarks and last_face_result.face_detected:
                face_detector.draw(frame, last_face_result)
                if win_cfg.show_eye_landmarks:
                    eye_tracker.draw(frame, last_face_result, last_eye_result)
            if win_cfg.show_head_pose and last_face_result.face_detected:
                head_pose_estimator.draw(frame, last_face_result, last_pose_result)

            # 5. Render modern UI overlay panels
            overlay.render(
                frame=frame,
                fps=camera.fps,
                face_result=last_face_result,
                eye_result=last_eye_result,
                pose_result=last_pose_result,
                show_hud=show_hud,
                show_landmarks=show_landmarks,
                debug_mode=debug_mode,
            )

            # 6. Display frame
            cv2.imshow(window_name, frame)

            # 7. Periodic console status logging
            now = time.perf_counter()
            if now - last_heartbeat >= win_cfg.heartbeat_interval_sec:
                face_status = "Detected" if last_face_result.face_detected else "Not Detected"
                ear_str = (
                    f"{last_eye_result.smoothed_ear:.2f}"
                    if last_eye_result.smoothed_ear is not None
                    else "--"
                )
                eye_str = last_eye_result.eye_state.value
                head_str = last_pose_result.direction.value
                yaw_str = f"{last_pose_result.yaw:+.1f} deg" if last_pose_result.yaw is not None else "--"
                pitch_str = f"{last_pose_result.pitch:+.1f} deg" if last_pose_result.pitch is not None else "--"
                print(
                    f"  [Live] FPS: {camera.fps:4.1f} | Face: {face_status}"
                    f" | Eyes: {eye_str} (EAR: {ear_str})"
                    f" | Head: {head_str} (Yaw: {yaw_str}, Pitch: {pitch_str})",
                    flush=True,
                )
                last_heartbeat = now

            # 8. Interactive keyboard handling
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print("\nExit key received. Shutting down...", flush=True)
                break
            elif key in (ord("l"), ord("L")):
                show_landmarks = not show_landmarks
            elif key in (ord("d"), ord("D")):
                debug_mode = not debug_mode
            elif key in (ord("h"), ord("H")):
                show_hud = not show_hud

            # 9. Check for window close button [X]
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("\nWindow closed. Shutting down...", flush=True)
                    break
            except cv2.error:
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user. Shutting down...", flush=True)
    finally:
        face_detector.close()
        camera.release()
        cv2.destroyAllWindows()
        print("Resources released cleanly.", flush=True)


if __name__ == "__main__":
    main()
