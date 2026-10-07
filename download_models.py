"""Model download helper for Smart Study Monitor.

Run this script once before launching the application to download the required
MediaPipe model bundle.

Usage:
    python download_models.py
"""

import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"

DOWNLOADS = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    ),
    "yolov8n.pt": (
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt"
    ),
}


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded / total_size * 100)
        bar_len = 40
        filled = int(bar_len * pct / 100)
        bar = "=" * filled + "-" * (bar_len - filled)
        print(f"\r  [{bar}] {pct:5.1f}%", end="", flush=True)


def download_models() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    all_ok = True
    for filename, url in DOWNLOADS.items():
        dest = MODELS_DIR / filename
        if dest.exists():
            print(f"[OK] {filename} already exists — skipping download.")
            continue

        print(f"Downloading {filename} ...")
        try:
            urllib.request.urlretrieve(url, dest, reporthook=_progress_hook)
            print(f"\n[OK] Saved to: {dest}")
        except Exception as exc:
            print(f"\n[ERROR] Failed to download {filename}: {exc}")
            all_ok = False

    if not all_ok:
        sys.exit(1)
    print("\nAll models ready.")


if __name__ == "__main__":
    download_models()
