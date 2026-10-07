# Models Directory

This directory stores pretrained model files required by Smart Study Monitor.

---

## Phase 2 — Face Landmark Detection

### Model: `face_landmarker.task`

**What it is:** A self-contained MediaPipe Tasks API model bundle that performs
468-point facial landmark detection on a single image or video stream.  It is
the official pre-trained model provided by Google for use with the MediaPipe
`FaceLandmarker` Tasks API (mediapipe >= 0.10 / 1.0.x).

**Why it is not committed to git:** The `.task` bundle is a binary file of
approximately 5.4 MB.  Binary model weights must not be committed to source
control (`models/*.task` is listed in `.gitignore`).

---

### Setup: one-command download

From the project root directory, run:

```bash
python download_models.py
```

This script downloads `face_landmarker.task` from the official Google storage
bucket and saves it to `models/face_landmarker.task`.

---

### Manual download

If the automated script fails, download the file manually:

**URL:**
```
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

Place the downloaded file at:
```
smart-study-monitor/models/face_landmarker.task
```

---

## Phase 5 — Mobile Phone Detection

### Model: `yolov8n.pt`

**What it is:** The official pretrained YOLOv8 Nano (`yolov8n.pt`) neural network
weights from Ultralytics, trained on the COCO dataset containing 80 object classes.

**Model Size:** Approximately 6.2 MB.

**Purpose:** Real-time mobile phone detection on CPU webcam feeds. Identifies whether
a cell phone is visible within the user's study environment without making behavioral
judgments or distraction assertions.

**Why this model was selected:**
- **Inference Latency:** Optimized for lightweight CPU inference (~38–40 ms on CPU at 320x320 image size).
- **Handheld Accuracy:** High detection sensitivity for standard handheld smartphone form factors across portrait, landscape, and angled orientations.
- **Dynamic Class Mapping:** Contains the standard COCO `cell phone` class, which the `PhoneDetector` dynamically resolves by class name rather than hardcoded IDs.

**How to obtain:**
1. **Automated script:** Run `python download_models.py` from the project root.
2. **Auto-download on startup:** If `models/yolov8n.pt` is missing, the Ultralytics loader will automatically download it to `models/yolov8n.pt` on first startup.
3. **Manual download:** Download directly from [Ultralytics Assets v8.4.0](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt) and place at `models/yolov8n.pt`.

**Version Control Exclusion:**
The binary weight file `yolov8n.pt` is intentionally excluded from Git via `models/*.pt` in `.gitignore`.

