# Smart Study Monitor — User Guide

A practical, step-by-step operating guide for students and evaluators using the **Smart Study Monitor**.

---

## 1. Installation & Environment Setup

### Prerequisites
- **Python**: Version 3.10 to 3.13 (verified on Python 3.13.1, 64-bit).
- **Operating System**: Windows 10/11, macOS, or Linux.
- **Hardware**: Standard integrated or USB webcam, CPU with 2+ cores, 4 GB+ RAM.

### Setup Instructions
1. Open your terminal or PowerShell and navigate to the project directory:
   ```powershell
   cd smart-study-monitor
   ```
2. Create and activate a Python virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   *(On macOS/Linux: `source .venv/bin/activate`)*
3. Upgrade pip and install required dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 2. Model Setup

The system uses two lightweight, pre-trained AI models:
- **MediaPipe Face Landmarker** (`face_landmarker.task`): 468-point facial mesh.
- **YOLOv8 Nano** (`yolov8n.pt`): Lightweight phone object detection.

Run the automatic model downloader once:
```powershell
python download_models.py
```
This verifies both models in the `models/` folder and downloads any missing files directly from official sources.

---

## 3. Audio Setup (Optional)

Audio alerts use `pygame.mixer` to provide audible cues when you get distracted, drowsy, or pick up your phone.

- Audio files reside in `assets/sounds/`:
  - `phone_alert.mp3`: Warning chime when a mobile phone is detected.
  - `distraction_alert.mp3`: Gentle reminder when you look away from the screen.
  - `drowsiness_alert.mp3`: Alert tone when prolonged eye closure occurs.
- **Graceful Audio Degradation**: If these sound files are not present or your computer does not have an active audio output, the system simply runs silently with full on-screen visual banners—it will never crash.

---

## 4. Starting the Monitor

To launch the real-time study monitoring window:

```powershell
python app.py
```

### What to Expect on Startup:
1. The terminal displays component initialization messages.
2. If any previous study session was interrupted abruptly (e.g. power cut), the system automatically recovers it cleanly.
3. A live OpenCV webcam window opens with an AI Head-Up Display (HUD) overlay showing:
   - **Header**: Live study timer, current FPS, and session ID.
   - **Status Card**: Current behavioral state (e.g., `FOCUSED`).
   - **Vision Cards**: Eye Aspect Ratio (EAR), looking direction (Yaw/Pitch), and phone detection status.
   - **Footer**: Interactive hotkey reminders.

---

## 5. Keyboard Shortcuts & Display Controls

While the monitoring window is active, press these keys:

| Key | Action | Description |
| :---: | :--- | :--- |
| `H` | **Toggle HUD** | Shows or hides the entire analytics overlay. |
| `L` | **Toggle Landmarks** | Visualizes or hides the facial mesh and eye contour points. |
| `P` | **Toggle Phone Box** | Toggles bounding boxes around detected phones. |
| `D` | **Toggle Debug Mode** | Displays detailed mathematical metrics (raw EAR, yaw angles). |
| `Q` or `ESC` | **Quit & Save** | Concludes the study session, saves all logs, and cleanly exits. |

---

## 6. Understanding Behavioral States

The monitor uses temporal confirmation to prevent false triggers (e.g., a normal blink will never trigger drowsiness):

| Behavioral State | Color Indicator | Meaning & Trigger Criteria |
| :--- | :---: | :--- |
| **FOCUSED** | 🟢 Emerald Green | Student is facing the screen with open eyes. Normal blinks are ignored. |
| **DISTRACTED** | 🟡 Warm Amber | Student has turned their head away (left/right/down) continuously for $\ge 1.5$ seconds. |
| **DROWSY** | 🔴 Coral Red | Eyes remain continuously closed for $\ge 1.5$ seconds (longer than natural blinks). |
| **PHONE DETECTED** | 🟠 Neon Orange | Mobile phone is visible in the webcam view continuously for $\ge 1.0$ seconds. |
| **AWAY FROM DESK** | 🟣 Purple | Student has stepped away from the desk or camera view for $\ge 3.0$ seconds. |
| **NO FACE** | ⚪ Gray | Immediate indicator when no face is present in the frame. |

---

## 7. Understanding Alerts & Cooldowns

- **Visual Alerts**: When a non-focused state is confirmed, a high-contrast banner appears at the top of the video feed.
- **Audio Chimes**: An alert sound plays asynchronously.
- **Smart Cooldowns**: To prevent irritating sound spam, alerts have built-in cooldowns (e.g., phone alert sounds at most once every 5.0 seconds while the phone remains visible).

---

## 8. Concluding a Study Session

To end your study session:
1. Press `Q` or `ESC` on the webcam window (or close the window with `X`).
2. The system concludes the session, finalizes the monotonic runtime, computes state statistics, and commits all event records to `data/study_monitor.db`.
3. The terminal confirms the session was successfully saved with status `COMPLETED`.

---

## 9. Opening the Analytics Dashboard

To inspect historical study sessions, focus trends, and detailed state breakdowns, launch the local dashboard:

```powershell
streamlit run dashboard/app.py
```

The dashboard will open automatically in your web browser (default: `http://localhost:8501`).

### Dashboard Features:
- **Session Selector**: Pick any completed study session from the sidebar dropdown.
- **Summary KPIs**: Review Total Duration, Focus Score, Time Focused, and Distraction count.
- **Focus Score Breakdown**: Transparent look into how your score was calculated.
- **State Time Distribution**: Bar and pie charts detailing minutes spent in each state.
- **Session Timeline**: Chronological, color-coded visual bar illustrating your session flow.
- **Event Audit Log**: Paginated, filterable table listing every state transition, timestamp, and confidence rating.

---

## 10. Understanding the Focus Score

The Focus Score is calculated strictly on **observable study time**:

$$\text{Focus Score} = \left( \frac{\text{Focused Seconds}}{\text{Scored Seconds}} \right) \times 100$$

- **Scored Seconds**: Sum of time spent in active states (`FOCUSED`, `DISTRACTED`, `DROWSY`, `PHONE_DETECTED`, `AWAY_FROM_DESK`).
- **Excluded States**: Ambiguous states (`NO_FACE` and `UNKNOWN`) are excluded from the scorable denominator by default because the camera cannot confirm what you were doing.
- **Minimum Duration**: Sessions under 60 seconds are marked as *Short Session* without a Focus Score to avoid mathematically distorted ratings.

---

## 11. Troubleshooting Common Issues

| Problem | Root Cause | Solution |
| :--- | :--- | :--- |
| **Webcam fails to open** | Device occupied by Zoom/Teams or camera index mismatch | Close other camera apps. In `config/config.py`, change `device_index: int = 0` to `1`. |
| **Missing Model Error** | Model files have not been downloaded | Run `python download_models.py` from the project root. |
| **No audio sounds** | Audio files missing or muted | Verify files in `assets/sounds/` or check volume settings in `config/config.py`. Visual alerts remain active. |
| **Dashboard says "Database Not Found"** | No study session has been run yet | Run `python app.py` for at least 1 minute and exit with `Q` to record your first session. |
