# Smart Study Monitor — AI-Based Study Focus & Distraction Detection System

An intelligent, modular computer vision system designed to monitor a student's study sessions in real time. It detects focus, looking-away, drowsiness, desk absence, and mobile phone usage using facial landmark analysis and lightweight object detection.

---

## 📌 Implementation Status: Phase 1–4 Complete + UI/UX Overhaul

| Phase | Description | Status |
|---|---|---|
| **Phase 1** | Webcam capture & OpenCV live video pipeline with real-time FPS | ✅ Completed |
| **Phase 2** | MediaPipe FaceLandmarker detection (468-point facial mesh) | ✅ Completed |
| **Phase 3** | Eye landmark extraction, Eye Aspect Ratio (EAR), EMA smoothing & eye-state classification | ✅ Completed |
| **Phase 4** | 3D Head Pose estimation (`cv2.solvePnP`), Euler angles (Yaw/Pitch/Roll) & looking-away classification | ✅ Completed |
| **UI/UX Pass** | Dark translucent HUD cards, centralized design system, hotkey toggles, non-obtrusive layout | ✅ Completed |
| **Phase 5** | YOLO mobile phone detection | ⏳ Planned |
| **Phase 6** | Behavior & attention classification engine | ⏳ Planned |
| **Phase 7** | Real-time warning & alert system (visual + sound) | ⏳ Planned |
| **Phase 8** | Study session timer & event logging | ⏳ Planned |
| **Phase 9** | Focus score calculation engine | ⏳ Planned |
| **Phase 10** | SQLite database integration | ⏳ Planned |
| **Phase 11** | Analytics dashboard (Streamlit) | ⏳ Planned |
| **Phase 12** | Testing, optimization & final polish | ⏳ Planned |

> **Important Architectural Distinctions**:
> 1. **Head Pose vs Eye Gaze**: Phase 4 measures **3D head orientation**. It does **NOT** measure eye-gaze tracking (where the pupils are focused on screen).
> 2. **Orientation vs Distraction**: Phase 4 exposes discrete orientation (`CENTER`, `LEFT`, `RIGHT`, `UP`, `DOWN`, `UNKNOWN`). It does **NOT** classify sustained looking-away as distraction or trigger alarms yet. Distraction analysis belongs to the future behavior engine.
> 3. **No False Productivity Claims**: The UI reports what the computer vision system actually detects (`FACE DETECTED`, `EYES: OPEN`, `HEAD: CENTER`). It does not display fake productivity scores or arbitrary focus percentages.

---

## 📂 Project Structure

```text
smart-study-monitor/
├── app.py                     # Main application entry point (pipeline orchestration)
├── download_models.py         # One-command MediaPipe model downloader
├── requirements.txt           # Project dependencies
├── README.md                  # Documentation and setup guide
├── .gitignore                 # Python gitignore
│
├── config/
│   ├── __init__.py
│   └── config.py              # Centralized configuration & strict validation
│
├── ui/
│   ├── __init__.py            # UI package exports
│   ├── theme.py               # Color palette, typography scales, layout constants
│   └── overlay.py             # Translucent cards, badges, header, and footer renderer
│
├── vision/
│   ├── __init__.py
│   ├── camera.py              # Camera abstraction with FPS tracking
│   ├── face_detector.py       # MediaPipe FaceLandmarker wrapper
│   └── eye_tracker.py         # Eye landmark extraction, EAR & hysteresis
│   └── head_pose.py           # 3D Head Pose estimation via solvePnP & direction classifier
│
├── models/
│   ├── README.md              # Model documentation
│   └── face_landmarker.task   # MediaPipe Tasks model bundle (downloaded)
│
├── assets/
│   ├── icons/
│   └── sounds/
│
└── tests/
    ├── __init__.py
    ├── test_camera.py         # Camera unit tests
    ├── test_config.py         # Configuration validation unit tests
    ├── test_face_detector.py  # Face detector unit tests
    ├── test_eye_tracker.py    # EAR math, smoothing & hysteresis unit tests
    ├── test_head_pose.py      # Head pose 3D geometry, Euler angles & classifier tests
    └── test_ui.py             # UI design system, card rendering, and layout tests
```

---

## 🎨 UI/UX Design System

The application features a modern dark-translucent HUD design inspired by AI productivity monitors rather than raw debugging overlays:

* **Unobstructed Webcam Feed**: Floating cards dock at the bottom corners, keeping the student's face and upper body in the center 100% visible and unblocked.
* **Translucent Panels**: Sub-region alpha blending ($O(\text{panel})$ instead of full-frame copies) renders subtle dark obsidian cards (`#141519`) with metallic slate borders (`#323744`).
* **Visual Hierarchy & Typography**:
  * **Top Header**: Application title + active status badge (`● MONITORING ACTIVE`).
  * **Face & Eye Card (Bottom-Left)**: Face presence status, bold Eye openness badge (`EYES OPEN` / `EYES CLOSED`), and cleaned 2-decimal average EAR (`0.31`).
  * **Head Pose Card (Bottom-Right)**: Bold directional orientation badge (`HEAD CENTERED` / `LOOKING LEFT` / `LOOKING RIGHT` / `LOOKING UP` / `LOOKING DOWN`) and discrete Euler angles (`Yaw`, `Pitch`, `Roll`).
  * **Bottom Footer**: Current measured FPS and live interactive keyboard controls.
* **Color-Blind Accessibility**: State is never communicated by color alone; every indicator combines text badges, geometric symbols (filled vs hollow circles), and semantic color accents.
* **Interactive Hotkeys**:
  * **`Q` / `ESC`**: Cleanly exit application.
  * **`L`**: Toggle facial landmark mesh on/off.
  * **`D`**: Toggle developer debug metrics (raw multi-eye EAR, roll, state names).
  * **`H`**: Toggle HUD overlay on/off (clean full-screen webcam view).

---

## 🧠 Core Computer Vision Concepts (Phases 1–4)

### 1. MediaPipe FaceLandmarker (Phase 2)
Utilizes Google's **MediaPipe Tasks API** (`FaceLandmarker`) with `face_landmarker.task`. Runs locally to extract 468 3D normalized facial landmarks. When multiple faces are visible, the primary face (index 0, highest confidence) is tracked.

### 2. Eye Landmark Topology & Eye Aspect Ratio (EAR) (Phase 3)
For each eye, six key topological landmarks are extracted from the 468-point mesh:

* **Right Eye** (subject's right): $p_1=33, p_2=160, p_3=158, p_4=133, p_5=153, p_6=144$
* **Left Eye** (subject's left): $p_1=362, p_2=385, p_3=387, p_4=263, p_5=373, p_6=380$

The Eye Aspect Ratio is computed as:

$$\text{EAR} = \frac{\|\mathbf{p}_2 - \mathbf{p}_6\| + \|\mathbf{p}_3 - \mathbf{p}_5\|}{2 \cdot \|\mathbf{p}_1 - \mathbf{p}_4\|}$$

A dual-threshold hysteresis system avoids rapid flickering:
* $\text{EAR}_{\text{smoothed}} \le \text{ear\_close\_threshold}$ (default: `0.18`) $\rightarrow$ **`CLOSED`**
* $\text{EAR}_{\text{smoothed}} \ge \text{ear\_open\_threshold}$ (default: `0.22`) $\rightarrow$ **`OPEN`**
* In-between $\rightarrow$ **Retain Previous State**

### 3. 3D Head Pose Estimation via solvePnP (Phase 4)
Head pose estimation computes 3D rotation of the student's head relative to the camera coordinate system using OpenCV's `solvePnP` (Perspective-n-Point):

#### 3D Canonical Face Model Points:
1. **Nose tip** (`Landmark 1`): $(0.0, 0.0, 0.0)$
2. **Chin** (`Landmark 152`): $(0.0, 330.0, 65.0)$
3. **Left eye outer corner** (`Landmark 263`): $(225.0, -170.0, 135.0)$
4. **Right eye outer corner** (`Landmark 33`): $(-225.0, -170.0, 135.0)$
5. **Left mouth corner** (`Landmark 291`): $(150.0, 150.0, 125.0)$
6. **Right mouth corner** (`Landmark 61`): $(-150.0, 150.0, 125.0)$

#### Euler Angle Extraction:
`cv2.solvePnP` solves for rotation vector $\mathbf{r}$ and translation vector $\mathbf{t}$. $\mathbf{R} = \text{Rodrigues}(\mathbf{r})$ is decomposed into decoupled Euler angles:
* **Yaw** (horizontal rotation): turning user-left ($> 0$) or user-right ($< 0$)
* **Pitch** (vertical rotation): looking up ($< 0$) or down ($> 0$)
* **Roll** (in-plane tilt): tilting head sideways

---

## ⚙️ Configuration Reference

All settings are centralized in [`config/config.py`](config/config.py):

| Section | Parameter | Default | Description |
|---|---|---|---|
| **Camera** | `device_index` | `0` | Webcam index (0 for default built-in, 1 for external) |
| | `backend` | `"DSHOW"` | Capture backend (`"DSHOW"`, `"MSMF"`, or `"DEFAULT"`) |
| | `frame_width` | `640` | Capture resolution width |
| | `frame_height` | `480` | Capture resolution height |
| | `target_fps` | `30` | Target camera frame rate |
| | `fps_update_interval_sec` | `0.5` | Smoothing window for FPS measurement |
| **Display** | `window_name` | `"Smart Study Monitor - Live Feed"` | OpenCV window title |
| | `window_width` | `960` | Display window width |
| | `window_height` | `720` | Display window height |
| | `force_foreground` | `True` | Forces window visible in foreground on launch |
| | `heartbeat_interval_sec`| `2.0` | Terminal status logging frequency |
| | `show_hud` | `True` | Render HUD overlay cards |
| | `show_face_landmarks` | `True` | Render facial landmark mesh outlines |
| | `show_eye_landmarks` | `True` | Highlight eye keypoints & contours |
| | `show_head_pose` | `True` | Render 3D gaze pointer vector |
| | `debug_mode` | `False` | Display detailed developer diagnostics |
| | `panel_opacity` | `0.80` | Card alpha blending opacity ($0.0 \dots 1.0$) |
| **Face** | `num_faces` | `1` | Maximum faces to detect |
| | `min_face_detection_confidence` | `0.5` | Minimum face detection confidence threshold |
| | `draw_tesselation` | `False` | Render full 468-point mesh triangles |
| | `draw_contours` | `True` | Render face oval and feature contours |
| | `draw_irises` | `True` | Render iris points |
| **Eye Tracker**| `ear_close_threshold` | `0.18` | EAR below which eyes are classified as CLOSED |
| | `ear_open_threshold` | `0.22` | EAR above which eyes are classified as OPEN |
| | `ear_smoothing_alpha` | `0.4` | EMA smoothing weight (0 < $\alpha \le 1$) |
| | `min_eye_width` | `0.001` | Minimum horizontal distance for safe division |
| | `draw_eye_landmarks` | `True` | Draw eye keypoints & contour on feed |
| **Head Pose** | `yaw_left_threshold` | `15.0` | Positive yaw angle (deg) threshold for LEFT |
| | `yaw_right_threshold`| `-15.0` | Negative yaw angle (deg) threshold for RIGHT |
| | `pitch_up_threshold` | `-15.0` | Negative pitch angle (deg) threshold for UP |
| | `pitch_down_threshold`| `15.0` | Positive pitch angle (deg) threshold for DOWN |
| | `pose_smoothing_alpha`| `0.3` | EMA smoothing weight for Euler angles |
| | `draw_head_pose` | `True` | Draw 3D gaze pointer vector on feed |

---

## 🛠️ Setup & Installation

### 1. Prerequisites
* Python 3.10+ (tested on Python 3.11 and 3.13)
* Local webcam

### 2. Environment Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate environment (Windows)
.venv\Scripts\activate

# Activate environment (Linux/macOS)
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Model Bundle
```bash
python download_models.py
```
*Downloads `face_landmarker.task` (~5.4 MB) from Google's official MediaPipe repository.*

---

## ▶️ Running the Application

```bash
python app.py
```

### Controls:
* **`q`** or **`ESC`**: Safely exit the application.
* **`l`**: Toggle facial landmark mesh outlines on/off.
* **`d`**: Toggle developer debug mode (raw EARs, angles, states).
* **`h`**: Toggle HUD overlay on/off (clean full-screen webcam).
* **`[X]` Window Close Button**: Safely close window and release webcam.

---

## 🧪 Running Unit Tests

Run the complete test suite with:

```bash
python -m pytest tests/ -v
```

All 109 tests execute in pure software with synthetic geometry and do not require a physical camera.

---

## 🔒 Privacy & On-Device Processing

* **100% Local**: All video capture, landmark extraction, EAR calculations, and pose estimations are processed entirely on-device.
* **No Video Recording**: Raw camera frames are processed in memory and never written to disk or uploaded to any external server.
