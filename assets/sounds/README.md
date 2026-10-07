# Audio Assets Directory (`assets/sounds/`)

This directory holds optional sound notification assets for the Smart Study Monitor alert system (Phase 7).

---

## Expected Sound Assets

| Alert Type | Default Config Path | Purpose | Recommended Tone |
| :--- | :--- | :--- | :--- |
| **Phone Detected** | `assets/sounds/phone_alert.mp3` | Notification when phone presence is confirmed | Double chime / high warning beep |
| **Distraction** | `assets/sounds/distraction_alert.mp3` | Notification when user looks away past threshold | Soft reminder ping |
| **Drowsiness** | `assets/sounds/drowsiness_alert.mp3` | Urgent notification for sustained eye closure | Ascending alert pulse |

---

## Safe Degradation Design Invariant

The audio system (`alerts/sound_alert.py`) is designed with strict graceful degradation:

1. **Missing Files**: If any sound file is missing from this directory, `SoundAlert.load_sound()` logs an `INFO` message and cleanly disables audio playback for that specific alert. The visual banner overlay remains 100% active.
2. **Audio Device Unavailable**: If running in a headless environment, Docker container, or an OS without active audio drivers, `pygame.mixer` initialization failure is caught, logged as a warning, and the video pipeline continues uninterrupted at full frame rate.
3. **Format Support**: `pygame.mixer` natively supports standard `.mp3`, `.wav`, and `.ogg` audio formats. Paths can be customized in `config/config.py` under `AlertConfig`.
