"""Smart Study Monitor - Main Application Entry Point.

Phase 2: Webcam + OpenCV + MediaPipe Face Landmark Detection.
"""

import sys
import cv2
from config.config import config
from vision.camera import Camera
from vision.face_detector import FaceDetector, FaceDetectionResult


def draw_hud(
    frame,
    fps: float,
    face_result: FaceDetectionResult,
) -> None:
    """Draws the Phase 2 HUD overlay on the video frame."""
    h, w = frame.shape[:2]

    # Top banner background
    cv2.rectangle(frame, (0, 0), (w, 45), (20, 20, 20), -1)

    # Title
    cv2.putText(
        frame,
        "Smart Study Monitor | Phase 2: Face Landmarks",
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # FPS counter (green >= 20 FPS, amber below)
    fps_text = f"FPS: {fps:4.1f}"
    fps_color = (0, 255, 128) if fps >= 20.0 else (0, 180, 255)
    cv2.putText(
        frame,
        fps_text,
        (w - 130, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        fps_color,
        2,
        cv2.LINE_AA,
    )

    # Face status (below title bar on left side)
    if face_result.face_detected:
        face_label = "Face: Detected"
        face_color = (0, 255, 128)
    else:
        face_label = "Face: Not Detected"
        face_color = (0, 100, 255)

    cv2.putText(
        frame,
        face_label,
        (15, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        face_color,
        2,
        cv2.LINE_AA,
    )

    # Bottom helper bar
    cv2.rectangle(frame, (0, h - 30), (w, h), (20, 20, 20), -1)
    cv2.putText(
        frame,
        "Press 'q' or 'ESC' to exit",
        (15, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    print("=" * 60)
    print("  Smart Study Monitor - Phase 2 (Face Landmark Detection)")
    print("=" * 60)
    print(f"Initializing camera at index: {config.camera.device_index}...")

    camera = Camera()
    if not camera.start():
        print(f"\n[ERROR] Could not open webcam at index {config.camera.device_index}.")
        print("Troubleshooting steps:")
        print(" 1. Ensure your webcam is not in use by another app.")
        print(" 2. Check Windows Camera Privacy settings.")
        print(" 3. Try changing `device_index` to 1 in config/config.py.")
        sys.exit(1)

    print("Camera initialized successfully.")

    try:
        face_detector = FaceDetector()
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        camera.release()
        sys.exit(1)
    except Exception as exc:
        print(f"\n[ERROR] Failed to initialize face detector: {exc}")
        camera.release()
        sys.exit(1)

    print("Face detector initialized successfully. Streaming live feed...")
    print("Focus the video window and press 'q' or 'ESC' to quit.\n")

    window_name = config.camera.window_name
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, config.camera.frame_width, config.camera.frame_height)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    print(f"[INFO] Pop-up window '{window_name}' created.", flush=True)
    print("[INFO] Streaming live feed. Check your taskbar or Alt+Tab if hidden.\n", flush=True)

    # Provide a default empty result so the HUD renders before the first detection
    last_face_result = FaceDetectionResult(face_detected=False)

    import time
    frame_count = 0
    last_heartbeat = time.perf_counter()

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                print("[WARNING] Failed to grab frame from camera stream. Exiting...", flush=True)
                break

            frame_count += 1

            # Run face landmark detection
            last_face_result = face_detector.process(frame)

            # Draw face mesh onto frame (in-place, before HUD)
            face_detector.draw(frame, last_face_result)

            # Draw HUD on top of the annotated frame
            draw_hud(frame, camera.fps, last_face_result)

            cv2.imshow(window_name, frame)

            # Periodic terminal status update every 2 seconds
            now = time.perf_counter()
            if now - last_heartbeat >= 2.0:
                face_status = "Detected" if last_face_result.face_detected else "Not Detected"
                print(
                    f"[Streaming] Frames: {frame_count:5d} | FPS: {camera.fps:4.1f} | Face: {face_status}",
                    flush=True,
                )
                last_heartbeat = now

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print("\nExit key received. Shutting down...", flush=True)
                break

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("\nWindow closed by user. Shutting down...", flush=True)
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user. Shutting down...")
    finally:
        face_detector.close()
        camera.release()
        cv2.destroyAllWindows()
        print("Resources released cleanly.")


if __name__ == "__main__":
    main()
