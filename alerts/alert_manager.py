"""Alert Manager module for Smart Study Monitor.

Part of Phase 7.
Coordinates visual and audio alert triggers based on BehaviorResult inputs.
Strictly respects behavior states, transition entries, real-time cooldowns,
and separation of concerns (does NOT classify behavior).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Optional

from alerts.sound_alert import SoundAlert
from alerts.visual_alert import VisualAlert
from behavior.behavior_engine import BehaviorResult, BehaviorState
from config.config import AlertConfig, config as default_global_config
from ui.theme import UIColors

logger = logging.getLogger(__name__)


class AlertType(Enum):
    NONE = "none"
    PHONE = "phone"
    DROWSINESS = "drowsiness"
    DISTRACTION = "distraction"
    AWAY = "away"


@dataclass
class AlertEvent:
    """Internal lightweight representation of an active or triggered alert."""

    alert_type: AlertType
    behavior_state: BehaviorState
    timestamp: float
    message: str
    sound_played: bool
    visual_shown: bool


class AlertManager:
    """Consumes BehaviorResult and triggers non-blocking sound and visual alerts."""

    def __init__(
        self,
        alert_config: Optional[AlertConfig] = None,
        sound_alert: Optional[SoundAlert] = None,
        visual_alert: Optional[VisualAlert] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        """Initializes AlertManager.

        Args:
            alert_config: Alert settings. Defaults to global config.alerts.
            sound_alert: Optional injected SoundAlert instance (e.g. for testing).
            visual_alert: Optional injected VisualAlert instance.
            clock: Monotonic clock callable (injectable for testing).
        """
        self.config = alert_config or default_global_config.alerts
        self.config.validate()
        self._clock = clock or time.monotonic

        # Subsystems
        if sound_alert is not None:
            self.sound = sound_alert
        else:
            self.sound = SoundAlert(
                volume=self.config.alert_volume,
                enabled=self.config.alerts_enabled and self.config.sound_alerts_enabled,
            )
        self._load_sounds()

        if visual_alert is not None:
            self.visual = visual_alert
        else:
            self.visual = VisualAlert(
                default_duration_sec=self.config.visual_alert_duration_sec,
                enabled=self.config.alerts_enabled and self.config.visual_alerts_enabled,
                clock=self._clock,
            )

        # State tracking for transition triggers and cooldowns
        self._last_state: BehaviorState = BehaviorState.UNKNOWN
        self._last_alert_times: Dict[AlertType, float] = {
            AlertType.PHONE: -999.0,
            AlertType.DROWSINESS: -999.0,
            AlertType.DISTRACTION: -999.0,
            AlertType.AWAY: -999.0,
        }
        self._last_event: Optional[AlertEvent] = None

    def _load_sounds(self) -> None:
        """Loads configured sound assets once."""
        self.sound.load_sound("phone", self.config.phone_alert_sound_path)
        self.sound.load_sound("drowsiness", self.config.drowsiness_alert_sound_path)
        self.sound.load_sound("distraction", self.config.distraction_alert_sound_path)

    @property
    def last_event(self) -> Optional[AlertEvent]:
        """Returns the most recent alert event."""
        return self._last_event

    def handle_behavior(self, behavior_result: BehaviorResult) -> Optional[AlertEvent]:
        """Evaluates BehaviorResult and determines whether to trigger alerts.

        Args:
            behavior_result: Current BehaviorResult from Phase 6 Behavior Engine.

        Returns:
            Optional[AlertEvent] if an alert was triggered, None otherwise.
        """
        if not self.config.alerts_enabled:
            self._last_state = behavior_result.state
            return None

        current_state = behavior_result.state
        state_entered = current_state != self._last_state
        self._last_state = current_state

        now = self._clock()

        # Map state to AlertType
        alert_type = self._map_state_to_alert_type(current_state)
        if alert_type is AlertType.NONE:
            return None

        # Check cooldown
        cooldown = self._get_cooldown(alert_type)
        last_triggered = self._last_alert_times.get(alert_type, -999.0)
        cooldown_elapsed = (now - last_triggered) >= cooldown

        # Trigger only on state entry OR when sustained state exceeds cooldown
        should_alert = state_entered or cooldown_elapsed
        if not should_alert:
            return None

        # Execute alerts
        title, message, sound_name, accent = self._get_alert_details(alert_type)

        sound_played = False
        if self.config.sound_alerts_enabled and sound_name is not None:
            sound_played = self.sound.play(sound_name)

        visual_shown = False
        if self.config.visual_alerts_enabled:
            self.visual.trigger(title=title, message=message, accent_color=accent)
            visual_shown = True

        self._last_alert_times[alert_type] = now

        event = AlertEvent(
            alert_type=alert_type,
            behavior_state=current_state,
            timestamp=now,
            message=message,
            sound_played=sound_played,
            visual_shown=visual_shown,
        )
        self._last_event = event
        return event

    def _map_state_to_alert_type(self, state: BehaviorState) -> AlertType:
        """Centralized mapping of BehaviorState to AlertType."""
        if state == BehaviorState.PHONE_DETECTED:
            return AlertType.PHONE
        elif state == BehaviorState.DROWSY:
            return AlertType.DROWSINESS
        elif state == BehaviorState.DISTRACTED:
            return AlertType.DISTRACTION
        elif state == BehaviorState.AWAY_FROM_DESK:
            return AlertType.AWAY
        return AlertType.NONE

    def _get_cooldown(self, alert_type: AlertType) -> float:
        """Retrieves configured cooldown in seconds for an alert type."""
        if alert_type == AlertType.PHONE:
            return self.config.phone_alert_cooldown_sec
        elif alert_type == AlertType.DROWSINESS:
            return self.config.drowsiness_alert_cooldown_sec
        elif alert_type == AlertType.DISTRACTION:
            return self.config.distraction_alert_cooldown_sec
        elif alert_type == AlertType.AWAY:
            return self.config.away_alert_cooldown_sec
        return 0.0

    def _get_alert_details(self, alert_type: AlertType) -> tuple[str, str, Optional[str], tuple[int, int, int]]:
        """Returns (title, message, sound_name, accent_color) for an alert type."""
        if alert_type == AlertType.PHONE:
            return (
                "PHONE DETECTED",
                "Mobile device visible in study area.",
                "phone",
                UIColors.WARNING,
            )
        elif alert_type == AlertType.DROWSINESS:
            return (
                "DROWSINESS DETECTED",
                "Prolonged eye closure detected.",
                "drowsiness",
                UIColors.DANGER,
            )
        elif alert_type == AlertType.DISTRACTION:
            return (
                "DISTRACTION DETECTED",
                "Head turned away from study workspace.",
                "distraction",
                UIColors.WARNING,
            )
        elif alert_type == AlertType.AWAY:
            sound_name = "distraction" if self.config.away_alert_sound_mode == "distraction" else None
            return (
                "AWAY FROM DESK",
                "No active study posture observed.",
                sound_name,
                UIColors.NEUTRAL,
            )
        return ("", "", None, UIColors.NEUTRAL)

    def reset(self) -> None:
        """Resets state history and cooldowns."""
        self._last_state = BehaviorState.UNKNOWN
        for k in self._last_alert_times:
            self._last_alert_times[k] = -999.0
        self._last_event = None
        self.visual.clear()
        self.sound.stop_all()

    def close(self) -> None:
        """Cleans up alert resources."""
        self.reset()
        self.sound.close()
