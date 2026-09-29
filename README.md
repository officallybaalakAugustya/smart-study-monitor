# Smart Study Monitor — AI-Based Study Focus & Distraction Detection System

An intelligent, modular computer vision system designed to monitor a student's study sessions in real time. It detects focus, looking-away, drowsiness, desk absence, and mobile phone usage using facial landmark analysis and lightweight object detection.

---

## ✅ Current Milestone: Phase 2 — MediaPipe Face Landmark Detection

Phase 2 adds real-time facial landmark detection onto the Phase 1 webcam pipeline using the **MediaPipe Tasks API** (`FaceLandmarker`).

### What Phase 2 does

- Detects a human face in each captured frame.
- Locates **468 facial landmarks** (the full MediaPipe Face Mesh).
- Draws the mesh **tesselation**, **contours**, and **iris** connections over the face.
- Updates the HUD to show `Face: Detected` or `Face: Not Detected`.
- Gracefully handles frames where no face is visible — no crash, just a status change.
- Correctly cleans up all MediaPipe resources on exit.

### What Phase 2 does NOT do

> Phase 2 is strictly landmark detection and visualization.
> EAR, eye-state, blink detection, drowsiness, head pose, YOLO, phone detection,
> alerts, focus scoring, database, and dashboard belong to later phases.

---

## 📂 Project Structure

```text
smart-study-monitor/
├── app.py                     # Main entry point (Phase 2)
├── download_models.py         # One-command model downloader
├── requirements.txt
├── README.md
├── .gitignore
│
├── config/
│   ├── __init__.py
│   └── config.py              # CameraConfig + FaceDetectorConfig
│
├── vision/
│   ├── __init__.py
│   ├── camera.py              # Camera abstraction with FPS tracking
│   └── face_detector.py       # MediaPipe FaceLandmarker wrapper
│
├── models/
│   ├── README.md              # Model acquisition instructions
│   └── face_landmarker.task   # Downloaded by download_models.py
│
└── tests/
    ├── __init__.py
    └── test_face_detector.py  # Phase 2 unit tests
```

---

## 🛠️ Setup & Installation

### 1. Prerequisites
- Python 3.10+ (tested on 3.13)
- A working webcam

### 2. Create and activate a virtual environment
```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download required model
```bash
python download_models.py
```

This downloads `face_landmarker.task` (~5.4 MB) from Google's official MediaPipe
model repository and saves it to `models/face_landmarker.task`.

> See [models/README.md](models/README.md) for manual download instructions.

---

## ▶️ Running Phase 2

```bash
python app.py
```

### Expected behavior

1. The webcam opens and displays a live video window.
2. A green mesh of facial landmarks appears over your face.
3. The HUD at the top shows:
   ```
   Smart Study Monitor | Phase 2: Face Landmarks        FPS: 28.4
   Face: Detected
   ```
4. Moving out of frame changes the status to `Face: Not Detected`.
5. Moving back into frame restores the mesh instantly.
6. Press **`q`** or **`ESC`** to exit cleanly.

---

## 🧪 Running Tests

```bash
python -m pytest tests/test_face_detector.py -v
```

Tests cover:
- `FaceDetectionResult` dataclass correctness.
- `FileNotFoundError` raised when the model is missing.
- FaceDetector initialization via context manager.
- Black/empty/None frame handling without crash.
- No mutation of frame when no face is detected.

> Tests that require the model file skip automatically if `face_landmarker.task`
> is absent (e.g., in a CI environment).

---

## 🧠 How Face Landmark Detection Works

MediaPipe uses a two-stage pipeline:

1. **Face detection** — a fast, lightweight model finds the bounding box of any face in the frame.
2. **Landmark regression** — a second model predicts the precise 3D positions of 478 keypoints on the face (468 mesh points + 10 iris points).

All processing is done **locally on-device**. No video frames leave your machine.

### MediaPipe API used

This project uses the **MediaPipe Tasks API** (`mediapipe >= 0.10 / 1.0.x`):

```python
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
```

The older `mediapipe.solutions.face_mesh` API was deliberately avoided because
it is deprecated in `mediapipe >= 0.10`.

### Multiple faces

If more than one face is detected, the primary face (index 0, highest detection
confidence as ranked by MediaPipe) is used for landmark drawing. Future phases
may handle multi-user scenarios.

---

## ⚙️ Configuration

All tunable parameters live in [`config/config.py`](config/config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `device_index` | `0` | Webcam device index |
| `frame_width` | `640` | Capture resolution width |
| `frame_height` | `480` | Capture resolution height |
| `num_faces` | `1` | Max faces to detect |
| `min_face_detection_confidence` | `0.5` | Minimum detection confidence threshold |
| `draw_tesselation` | `True` | Draw full face mesh triangles |
| `draw_contours` | `True` | Draw face oval and feature outlines |
| `draw_irises` | `True` | Draw iris landmarks |

---

## 🔒 Privacy

- All video processing is performed **locally** on your machine.
- No video frames, images, or personal data are sent to any external server.
- No raw video is stored to disk by default.
- Only derived statistics and events will be stored (in later phases).

---

## ⚠️ Known Limitations (Phase 2)

- Landmark detection accuracy decreases significantly in poor lighting conditions.
- The system currently supports a single primary face.
- No temporal logic yet — each frame is processed independently.
- No downstream use of landmarks yet (EAR, head pose, etc. — Phase 3 onwards).

---

## 🗺️ Development Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Webcam + OpenCV pipeline + FPS | ✅ Done |
| 2 | MediaPipe face landmark detection | ✅ Done |
| 3 | Eye Aspect Ratio + blink/drowsiness | 🔜 Next |
| 4 | Head pose / looking-away detection | ⏳ Planned |
| 5 | YOLO mobile phone detection | ⏳ Planned |
| 6 | Behavior engine + state classification | ⏳ Planned |
| 7 | Alert system (visual + sound) | ⏳ Planned |
| 8 | Study session timer + event logging | ⏳ Planned |
| 9 | Focus score calculation | ⏳ Planned |
| 10 | SQLite database integration | ⏳ Planned |
| 11 | Streamlit analytics dashboard | ⏳ Planned |
| 12 | Testing, optimization, UI polish | ⏳ Planned |
