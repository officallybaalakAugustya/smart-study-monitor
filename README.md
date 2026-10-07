# Smart Study Monitor — AI-Based Study Focus & Distraction Detection System

An intelligent, modular computer vision and release-ready desktop application designed to monitor a student's study sessions in real time. It detects focus, distraction (looking away), drowsiness (prolonged eye closure), desk absence, and mobile phone usage using facial landmark analysis and lightweight object detection.

---

## 1. Project Overview

The **Smart Study Monitor** is an on-device student engagement tracking system built for personal productivity and academic evaluation. It processes webcam video locally in real time, converts low-level computer vision signals into temporally confirmed behavioral states, enforces anti-spam notifications, records state-transition events to SQLite, and presents actionable focus analytics via an interactive local dashboard.

### Core Architectural Distinctions:
1. **Sensors vs. Behavior Decisions**: Raw sensors report *what is observed* (`EYES: CLOSED`, `HEAD: LEFT`, `PHONE: DETECTED`). The `BehaviorEngine` infers *what state the user is in* (`DROWSY`, `DISTRACTED`, `PHONE_DETECTED`, `AWAY_FROM_DESK`) only after temporal grace periods expire.
2. **Deterministic Heuristic System, NOT Medical Diagnosis**: The monitor estimates study engagement using explicit geometric and temporal rules. It is **not** a clinical diagnostic tool, neurological fatigue sensor, or psychological evaluation system.
3. **No False Productivity Claims**: Focus Scores quantify observable, on-screen study duration without fabricating opaque cognitive metrics.

---

## 2. Features

- **Live Camera Pipeline**: High-performance OpenCV capture with FPS monitoring and graceful camera disconnection handling.
- **MediaPipe Facial Landmark Tracking**: 468-point 3D facial mesh detection via Google MediaPipe Tasks API.
- **Blink-Resistant Eye Tracking**: Eye Aspect Ratio (EAR) calculation with Exponential Moving Average (EMA) smoothing and dual-threshold hysteresis.
- **3D Head Pose & Gaze Direction**: Perspective-n-Point (`cv2.solvePnP`) 3D head rotation decomposition into Yaw, Pitch, and Roll angles.
- **YOLOv8 Mobile Phone Detection**: Lightweight CPU inference (`yolov8n.pt`) with temporal caching (1 inference per 4 frames) and dynamic COCO class resolution.
- **Behavior Engine with Temporal Confirmation**: Resolves multi-signal conflicts using a canonical priority hierarchy; eliminates single-frame blinks and accidental head twitches.
- **Dual-Mode Alert System**: Non-blocking visual HUD banners combined with audio alerts via `pygame.mixer` and independent cooldown timers.
- **Study Session Lifecycle & Crash Recovery**: Monotonic runtime tracking independent of frame drops; automatically detects and recovers unclosed active sessions from prior crashes.
- **State-Transition SQLite Logging**: Only logs events when behavioral states change, preventing database write spam during steady-state study.
- **Weighted-Duration Focus Score**: Deterministic 4-component formula with scorable duration accounting and 60-second minimum session gating.
- **Streamlit Analytics Dashboard**: Local offline presentation layer featuring session selectors, KPI cards, state distributions, timeline Gantt charts, and event logs.

---

## 3. Architecture

```text
                               Webcam Capture (vision/camera.py)
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │      Vision & Detection Layer   │
                             ├─────────────────────────────────┤
                             │ • MediaPipe FaceLandmarker      │
                             │ • EyeTracker & EAR Calculation  │
                             │ • HeadPose 3D solvePnP Solver   │
                             │ • YOLOv8 Mobile Phone Detector  │
                             └────────────────┬────────────────┘
                                              │
                                              ▼
                                Behavior Engine (behavior/behavior_engine.py)
                             [Canonical Priority + Temporal Confirmation]
                                              │
                               ┌──────────────┴──────────────┐
                               ▼                             ▼
                    Alert Manager (alerts/)       Session Manager (session/)
                    ├── Visual HUD Banners                   │
                    └── Pygame Sound Playback                ▼
                                                    Event Logger (session/event_logger.py)
                                                             │
                                                             ▼
                                                    SQLite DB (data/study_monitor.db)
                                                             │
                                                             ▼
                                                Session Analytics (analytics/)
                                                             │
                                                             ▼
                                                Streamlit Dashboard (dashboard/app.py)
```

---

## 4. Technology Stack

- **Language**: Python 3.10–3.13 (verified on Python 3.13.1, 64-bit Windows)
- **Computer Vision**: OpenCV (`opencv-contrib-python`), MediaPipe Tasks (`mediapipe`)
- **Deep Learning Inference**: Ultralytics YOLOv8 (`ultralytics`, PyTorch CPU)
- **Audio Synthesis & Playback**: Pygame (`pygame.mixer`, SDL2)
- **Data Persistence**: SQLite 3 (Python standard library with WAL mode)
- **Analytics & Math**: NumPy, Pandas, Plotly
- **Web Dashboard**: Streamlit
- **Testing & Diagnostics**: pytest, psutil

---

## 5. Project Structure

```text
smart-study-monitor/
├── app.py                     # Main live monitoring application
├── download_models.py         # One-command model downloader
├── requirements.txt           # Production runtime dependencies
├── requirements-dev.txt       # Development & testing dependencies
├── README.md                  # Complete project documentation
├── .gitignore                 # Standard Python / model / db exclusions
│
├── config/
│   ├── __init__.py
│   └── config.py              # Centralized dataclass configurations & validation
│
├── vision/
│   ├── camera.py              # Camera capture & FPS estimation
│   ├── face_detector.py       # MediaPipe FaceLandmarker wrapper
│   ├── eye_tracker.py         # Eye landmark extraction, EAR & hysteresis
│   └── head_pose.py           # 3D Head Pose estimation via solvePnP & Euler angles
│
├── detection/
│   └── phone_detector.py      # Ultralytics YOLOv8n phone detector with caching
│
├── behavior/
│   └── behavior_engine.py     # Priority-based state machine & temporal confirmation
│
├── alerts/
│   ├── alert_manager.py       # Alert coordinator & cooldown enforcement
│   ├── sound_alert.py         # Pygame audio playback & safe degradation
│   └── visual_alert.py        # Non-blocking HUD banner overlay
│
├── session/
│   ├── session_manager.py     # Session lifecycle, monotonic timer & crash recovery
│   └── event_logger.py        # State-transition behavioral event logger
│
├── database/
│   └── database.py            # SQLite schema initialization, WAL mode & execution
│
├── analytics/
│   ├── models.py              # Strongly-typed dataclasses for analytics
│   ├── focus_score.py         # Weighted-duration Focus Score engine
│   └── session_analytics.py   # Historical analytics & timeline reconstruction
│
├── dashboard/
│   ├── app.py                 # Streamlit analytics presentation layer
│   └── components.py          # Decoupled visualization helpers & chart data prep
│
├── ui/
│   ├── theme.py               # Color palette, typography scales, layout constants
│   └── overlay.py             # Translucent cards, badges, header, and footer renderer
│
├── models/
│   ├── README.md              # Model setup documentation
│   ├── face_landmarker.task   # MediaPipe Tasks model bundle
│   └── yolov8n.pt             # Ultralytics YOLOv8 Nano weights
│
├── assets/
│   └── sounds/                # Optional audio assets (phone, distraction, drowsiness)
│       └── README.md          # Audio documentation & graceful degradation
│
├── docs/
│   ├── USER_GUIDE.md          # Practical end-user operating manual
│   ├── DEVELOPER_GUIDE.md     # Architectural and developer maintenance guide
│   └── FINAL_TEST_REPORT.md   # Complete QA, benchmark, and test evidence report
│
├── scripts/
│   ├── benchmark_pipeline.py  # Precision per-stage benchmarking & profiling tool
│   └── verify_scenario.py     # 15-scenario end-to-end integration verification runner
│
└── tests/                     # 320 automated unit and integration tests
```

---

## 6. Requirements

- **Operating System**: Windows 10/11, macOS, or Ubuntu/Debian Linux.
- **Python**: Version 3.10 to 3.13.
- **Camera**: Standard USB or built-in webcam supporting 640x480 resolution.
- **Compute**: Minimum dual-core CPU, 4 GB RAM. (GPU not required; all models run efficiently on CPU).

---

## 7. Installation

Clone the repository and set up a clean Python virtual environment:

```powershell
# 1. Clone repository
git clone https://github.com/officallybaalakAugustya/smart-study-monitor.git
cd smart-study-monitor

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
# (On macOS/Linux: source .venv/bin/activate)

# 4. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 8. Model Setup

The application requires two pre-trained neural network models:
1. `models/face_landmarker.task` (~5.4 MB)
2. `models/yolov8n.pt` (~6.2 MB)

Download both models automatically with a single command:

```powershell
python download_models.py
```

If you prefer manual download, follow the instructions in [`models/README.md`](models/README.md).

---

## 9. Audio Setup

Optional sound chimes reside in `assets/sounds/`:
- `phone_alert.mp3`
- `distraction_alert.mp3`
- `drowsiness_alert.mp3`

**Safe Audio Degradation**: If these sound files are omitted or audio hardware is missing (e.g. running headlessly), the application logs an informational notice and runs silently with visual banners intact. See [`assets/sounds/README.md`](assets/sounds/README.md).

---

## 10. Running the Monitor

To start real-time study tracking:

```powershell
python app.py
```

### Hotkey Controls in Video Window:
- **`Q`** or **`ESC`**: Conclude session, commit logs, and exit cleanly.
- **`H`**: Toggle HUD cards visibility on/off.
- **`L`**: Toggle 468-point facial mesh wireframes on/off.
- **`P`**: Toggle phone detection bounding boxes on/off.
- **`D`**: Toggle developer debug mode (raw EARs, angles, inference times).

---

## 11. Running the Analytics Dashboard

To inspect historical study sessions, focus scores, and timeline charts:

```powershell
streamlit run dashboard/app.py
```

The dashboard opens in your browser at `http://localhost:8501`.

*Note: The dashboard is strictly read-only and never runs computer-vision inference or writes to the database.*

---

## 12. Running Tests

Run the complete test suite:

```powershell
python -m pytest tests/ -v
```

Execute the 15-scenario end-to-end acceptance runner:

```powershell
python scripts/verify_scenario.py
```

Execute the 1,000-frame performance and memory benchmark:

```powershell
python scripts/benchmark_pipeline.py
```

---

## 13. Configuration

All application parameters are centralized and validated in [`config/config.py`](config/config.py). Key parameters include:

| Domain | Parameter | Default | Purpose |
| :--- | :--- | :---: | :--- |
| **Camera** | `device_index` | `0` | Webcam index (change to 1 for external USB camera) |
| **Eye Tracker** | `ear_close_threshold` | `0.18` | EAR below which eyes are classified as CLOSED |
| | `ear_open_threshold` | `0.22` | EAR above which eyes are classified as OPEN |
| **Head Pose** | `yaw_left_threshold` | `15.0°` | Yaw angle threshold for looking left |
| | `yaw_right_threshold` | `-15.0°` | Yaw angle threshold for looking right |
| **Phone** | `inference_interval` | `4` | Run YOLO inference every 4 frames (temporal caching) |
| **Behavior** | `phone_confirmation_sec` | `1.0s` | Required persistence before `PHONE_DETECTED` |
| | `drowsiness_confirmation_sec` | `1.5s` | Continuous eye closure before `DROWSY` (ignores blinks) |
| | `head_away_confirmation_sec` | `1.5s` | Continuous head-away before `DISTRACTED` |
| | `away_from_desk_confirmation_sec` | `3.0s` | Sustained head-away or absence before `AWAY_FROM_DESK` |
| **Alerts** | `phone_alert_cooldown_sec` | `5.0s` | Cooldown between repeated phone alerts |
| | `drowsiness_alert_cooldown_sec` | `5.0s` | Cooldown between repeated drowsiness alerts |
| **Analytics** | `minimum_scored_duration_sec` | `60.0s` | Minimum scorable duration required for Focus Score |

---

## 14. Database

Study history is stored locally in SQLite at `data/study_monitor.db` with WAL mode enabled:
- **`sessions` table**: Records session ID, start/end wall-clock timestamps (ISO 8601), monotonic duration, and status (`ACTIVE`, `COMPLETED`, `CANCELLED`).
- **`behavior_events` table**: Records confirmed state transitions with timestamp, state, duration, confidence, reason code, and raw sensor readings.
- **Crash Recovery**: If the system is force-killed while active, the next launch automatically marks the abandoned session as `CANCELLED` and sets its duration to its last recorded event.

---

## 15. Focus Score

The Focus Score measures observable attentiveness during a study session:

$$\text{Focus Score} = \left( \frac{\text{Focused Seconds}}{\text{Scored Seconds}} \right) \times 100$$

- **Observable States**: `FOCUSED` ($w=1.0$), `DISTRACTED` ($w=0.0$), `DROWSY` ($w=0.0$), `PHONE_DETECTED` ($w=0.0$), `AWAY_FROM_DESK` ($w=0.0$).
- **Excluded States**: `NO_FACE` and `UNKNOWN` are excluded from the scorable duration denominator by default because the camera cannot observe user activity when no face is present.
- **Minimum Duration Gate**: Sessions shorter than 60 scorable seconds return `None` (`is_valid = False`) to prevent misleading scores on trivial tests.

---

## 16. Privacy

- **100% On-Device Processing**: All computer vision, facial analysis, and phone detection execute locally on your CPU.
- **No Video Recording**: Raw webcam frames, face crops, and audio feeds are processed in memory and never saved to disk or transmitted over the internet.
- **Metadata Only**: Only high-level numerical events (timestamps, state names, EAR values, angles) are persisted to the local SQLite database.

---

## 17. Troubleshooting

| Symptom | Cause | Resolution |
| :--- | :--- | :--- |
| **`[ERROR] Could not open webcam`** | Webcam in use or incorrect index | Close Zoom/Teams/OBS. In `config/config.py`, change `device_index` from `0` to `1`. |
| **`FileNotFoundError: models/...`** | Models not downloaded | Run `python download_models.py` from project root. |
| **No alert sound playing** | Missing files or missing audio output | Check `assets/sounds/` or volume in `config/config.py`. Visual alerts remain functional. |
| **Dashboard shows "Database Not Found"** | No session recorded yet | Run `python app.py` for at least 1 minute and exit with `Q` to record a session. |
| **Low FPS on older hardware** | High inference load | In `config/config.py`, increase `inference_interval` from `4` to `6` or `8`. |

---

## 18. Limitations

1. **Ambient Lighting**: Extreme low-light or strong backlighting can impair MediaPipe facial mesh landmarks.
2. **Extreme Camera Angles**: Mounting the webcam high above eye level (> 45° tilt) can bias pitch calculations. Keep the camera centered.
3. **Audio on Headless Linux**: In headless server environments, Pygame requires a dummy audio driver (`export SDL_AUDIODRIVER=dummy`).

---

## 19. Testing & Acceptance Status

- **Automated Unit & Integration Tests**: **322 passed**, 0 failed, 0 skipped (`pytest tests/ -v` in 14.57s).
- **15 End-to-End Acceptance Scenarios**: **15 passed**, 0 failed (`python scripts/verify_scenario.py`).
- **Throughput SLA**: **~84.0 FPS** (11.90 ms mean latency, target $\ge 30$ FPS).
- **Memory Stability**: Zero memory leaks across 1,000 continuous frames (+0.04 MB flatline).
- **Verdict**: **READY FOR DEMONSTRATION**.

For detailed architecture docs, see [`docs/DEVELOPER_GUIDE.md`](docs/DEVELOPER_GUIDE.md) and [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md).
