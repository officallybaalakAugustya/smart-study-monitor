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

**What it is:** The pretrained YOLOv8 Nano weights from Ultralytics, used for
object detection (including the `cell phone` class).

**How to obtain:** The Ultralytics library automatically downloads `yolov8n.pt`
the first time it is used.  You can also download it manually from
[https://github.com/ultralytics/assets/releases](https://github.com/ultralytics/assets/releases)
and place it here.

> Phase 5 is not yet implemented.
