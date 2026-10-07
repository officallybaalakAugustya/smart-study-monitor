"""Alert & Audio System Package for Smart Study Monitor.

Part of Phase 7.
Translates Behavior Engine states (BehaviorResult) into controlled visual
banners and audio cues, respecting cooldowns and transition triggers.
"""

from alerts.alert_manager import AlertManager, AlertType, AlertEvent
from alerts.sound_alert import SoundAlert
from alerts.visual_alert import VisualAlert

__all__ = [
    "AlertManager",
    "AlertType",
    "AlertEvent",
    "SoundAlert",
    "VisualAlert",
]
