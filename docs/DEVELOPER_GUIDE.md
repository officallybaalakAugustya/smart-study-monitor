# Smart Study Monitor — Developer & Architecture Guide

A comprehensive architectural reference and developer manual for maintaining, extending, and testing the **Smart Study Monitor**.

---

## 1. System Architecture

The pipeline follows a strict unidirectional data flow. Higher layers ingest data from lower layers, preserving loose coupling and clean separation of concerns:

```text
Webcam (vision/camera.py)
   │
   ▼
Vision Analysis Layer
   ├── Face Landmark Mesh (vision/face_detector.py)
   ├── Eye Tracking & EAR (vision/eye_tracker.py)
   └── Head Pose 3D Estimation (vision/head_pose.py)
   │
   ▼
Object Detection Layer
   └── YOLO Mobile Phone Detector (detection/phone_detector.py)
   │
   ▼
Behavior Decision Engine (behavior/behavior_engine.py)
   │   [Produces immutable BehaviorResult]
   ├───┬────────────────────────────────────────┐
   │   ▼                                        ▼
   │  Alert Coordinator (alerts/alert_manager.py)  Session Manager (session/session_manager.py)
   │   ├── Visual Banner (alerts/visual_alert.py)       │
   │   └── Audio Playback (alerts/sound_alert.py)       ▼
   ▼                                           Event Logger (session/event_logger.py)
HUD Renderer (ui/overlay.py)                            │
   │                                                    ▼
   ▼                                           SQLite Database (database/database.py)
OpenCV Window                                           │
                                                        ▼
                                               Session Analytics (analytics/session_analytics.py)
                                                        │
                                                        ▼
                                               Streamlit Dashboard (dashboard/app.py)
```

---

## 2. Core Architectural Contracts

### A. The `BehaviorResult` Contract
The `BehaviorResult` dataclass is the **single authoritative public contract** between the `BehaviorEngine` and all downstream consumers (`AlertManager`, `SessionManager`, `OverlayRenderer`).

```python
@dataclass
class BehaviorResult:
    state: BehaviorState
    state_started_at: float | None
    state_duration: float
    confidence: float
    reason: str
    signals: dict
```

#### Contract Invariants:
- `state`: Must always be an instance of `BehaviorState`.
- `state_started_at`: Monotonic timestamp (`time.monotonic()`) when the current state began, or `None`.
- `state_duration`: Active elapsed duration in seconds ($\ge 0.0$).
- `confidence`: Bounded normalized confidence float in $[0.0, 1.0]$.
- `reason`: Stable, machine-readable reason code (e.g., `focused`, `phone_detected`, `eyes_closed`, `head_away`, `face_not_detected`).
- `signals`: Dictionary containing raw sensor measurements (e.g., `ear`, `yaw`, `pitch`, `phone_confidence`).

### B. Canonical `BehaviorState` Priority Ladder
The `BehaviorEngine` resolves multi-signal conflicts using a canonical behavioral priority hierarchy:

$$\text{PHONE\_DETECTED} > \text{DROWSY} > \text{AWAY\_FROM\_DESK} > \text{DISTRACTED} > \text{FOCUSED}$$

- **`NO_FACE`**: Returned immediately when the face detector reports zero faces.
- **`UNKNOWN`**: Returned when signals are contradictory, corrupted, or insufficient.
- Both `NO_FACE` and `UNKNOWN` represent sensor availability states and do not participate in ordinary behavioral escalation.

---

## 3. Module Responsibilities

| Module | Core Responsibility | Downstream Guarantees |
| :--- | :--- | :--- |
| `vision/camera.py` | OpenCV video capture, FPS smoothing, frame resizing. | Thread-safe capture, returns `(False, None)` on drop. |
| `vision/face_detector.py` | MediaPipe FaceLandmarker Tasks API wrapper. | Returns 468 normalized 3D landmarks or `None`. |
| `vision/eye_tracker.py` | EAR computation, EMA smoothing, open/closed hysteresis. | Deterministic blink resistance, EAR in $[0.0, 1.0]$. |
| `vision/head_pose.py` | 3D PnP perspective solver, Euler angle decomposition. | Yaw, Pitch, Roll angles and direction classification. |
| `detection/phone_detector.py`| YOLOv8n object detection with inference frame-skipping. | Dynamic COCO class resolution, bounding boxes. |
| `behavior/behavior_engine.py`| Multi-sensor state fusion and temporal confirmation. | Sole authority for behavior classification. |
| `alerts/alert_manager.py` | Dispatches visual banners and non-blocking sound playback. | Cooldown enforcement, non-blocking execution. |
| `session/session_manager.py`| Tracks session runtime via monotonic clock, logs events. | Zero duplicate steady-state logging; crash recovery. |
| `session/event_logger.py` | SQLite DAO for sessions and behavioral events. | ISO 8601 timestamps, JSON signal serialization. |
| `database/database.py` | SQLite schema initialization, WAL mode, foreign keys. | Idempotent table creation, connection pooling. |
| `analytics/session_analytics.py`| Reconstructs behavioral timelines and durations. | Non-overlapping time accounting, summaries. |
| `analytics/focus_score.py` | Computes weighted Focus Scores from state durations. | Deterministic scoring; 60s minimum threshold. |
| `dashboard/app.py` | Streamlit presentation layer. | Strictly read-only; zero CV or DB mutation. |

---

## 4. SQLite Database Schema

The database is stored locally at `data/study_monitor.db` with SQLite WAL (Write-Ahead Logging) enabled.

### Table: `sessions`
```sql
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    start_time TEXT NOT NULL,
    end_time TEXT,
    duration_seconds REAL,
    status TEXT NOT NULL CHECK(status IN ('ACTIVE', 'COMPLETED', 'CANCELLED')),
    created_at TEXT NOT NULL
);
```

### Table: `behavior_events`
```sql
CREATE TABLE IF NOT EXISTS behavior_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    state TEXT NOT NULL,
    state_duration REAL NOT NULL,
    confidence REAL NOT NULL,
    reason TEXT NOT NULL,
    signals_json TEXT,
    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
```

### Crash Recovery Strategy:
On startup, `SessionManager.start_session()` invokes `EventLogger.recover_unfinished_sessions()`. Any lingering sessions with `status = 'ACTIVE'` (e.g. from an abrupt process termination or power loss) have their `end_time` set to their last recorded event timestamp and status updated to `'CANCELLED'`. All recorded events remain intact.

---

## 5. Analytics & Focus Score Mathematics

The Focus Score quantifies study engagement over observable time:

$$\text{Focus Score} = \left( \frac{\sum w_i \cdot \text{duration}_i}{\text{Scored Seconds}} \right) \times 100$$

### Default State Weights ($w_i$):
- $\text{FOCUSED} = 1.0$
- $\text{DISTRACTED} = 0.0$
- $\text{DROWSY} = 0.0$
- $\text{PHONE\_DETECTED} = 0.0$
- $\text{AWAY\_FROM\_DESK} = 0.0$

### Scorable Duration:
$$\text{Scored Seconds} = \sum_{s \in \text{Observable States}} \text{duration}_s$$

- `NO_FACE` and `UNKNOWN` are excluded from the scorable duration denominator by default.
- If $\text{Scored Seconds} < 60.0\text{ seconds}$, `FocusScoreResult.score` returns `None` (`is_valid = False`), preventing misleading ratings on trivial sessions.

---

## 6. Testing Strategy & Principles

The repository adheres to strict testing standards:
1. **Isolated In-Memory Databases**: All database tests run against `:memory:` instances, leaving user databases untouched.
2. **Clock Injection**: Time progression is tested deterministically using injectable `MockClock` and `MockWallClock` fixtures.
3. **Synthetic Frame Fixtures**: Vision tests inject mathematical face/eye/head geometries without relying on live hardware or external cameras.
4. **No Trivial Tests**: Every test verifies real behavioral contracts, edge cases, failure recovery, or error handling.

To run the full suite:
```powershell
python -m pytest tests/ -v
```
To run end-to-end scenario verification:
```powershell
python scripts/verify_scenario.py
```
To run the performance profiler:
```powershell
python scripts/benchmark_pipeline.py
```
