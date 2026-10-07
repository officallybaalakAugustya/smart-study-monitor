"""Unit tests for alerts package (Phase 7).

Verifies SoundAlert, VisualAlert, and AlertManager with mock hardware,
synthetic BehaviorResult objects, cooldown enforcement, state-entry triggering,
and error handling.
"""

import numpy as np
import pytest

from alerts.alert_manager import AlertManager, AlertType, AlertEvent
from alerts.sound_alert import SoundAlert
from alerts.visual_alert import VisualAlert
from behavior.behavior_engine import BehaviorResult, BehaviorState, BehaviorReason
from config.config import AlertConfig


class MockClock:
    def __init__(self, start_time: float = 100.0) -> None:
        self.time = start_time

    def __call__(self) -> float:
        return self.time

    def advance(self, seconds: float) -> None:
        self.time += seconds


class MockMixerSound:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.volume = 1.0
        self.played = False
        self.play_count = 0

    def set_volume(self, volume: float) -> None:
        self.volume = volume

    def play(self) -> None:
        self.played = True
        self.play_count += 1


class MockMixerModule:
    def __init__(self) -> None:
        self.initialized = True
        self.sounds = {}
        self.stopped = False

    def get_init(self) -> bool:
        return self.initialized

    def init(self) -> None:
        self.initialized = True

    def Sound(self, file_path: str) -> MockMixerSound:
        sound = MockMixerSound(file_path)
        self.sounds[file_path] = sound
        return sound

    def stop(self) -> None:
        self.stopped = True


class TestSoundAlert:
    def test_load_and_play_with_mock_mixer(self, tmp_path):
        sound_file = tmp_path / "test.mp3"
        sound_file.write_bytes(b"dummy audio data")

        mock_mixer = MockMixerModule()
        sound_alert = SoundAlert(volume=0.8, enabled=True, mixer_module=mock_mixer)
        assert sound_alert.load_sound("test_sound", sound_file) is True

        assert sound_alert.play("test_sound") is True
        assert mock_mixer.sounds[str(sound_file.resolve())].played is True

    def test_missing_file_does_not_crash(self, tmp_path):
        mock_mixer = MockMixerModule()
        sound_alert = SoundAlert(volume=0.8, enabled=True, mixer_module=mock_mixer)
        missing_file = tmp_path / "non_existent.mp3"
        assert sound_alert.load_sound("missing", missing_file) is False
        assert sound_alert.play("missing") is False

    def test_disabled_sound_does_not_play(self, tmp_path):
        sound_file = tmp_path / "test.mp3"
        sound_file.write_bytes(b"dummy")

        mock_mixer = MockMixerModule()
        sound_alert = SoundAlert(volume=0.8, enabled=False, mixer_module=mock_mixer)
        sound_alert.load_sound("test_sound", sound_file)
        assert sound_alert.play("test_sound") is False


class TestVisualAlert:
    def test_trigger_and_expiration(self):
        clock = MockClock(10.0)
        visual = VisualAlert(default_duration_sec=3.0, enabled=True, clock=clock)
        assert visual.is_active is False

        visual.trigger("TEST", "Message")
        assert visual.is_active is True
        assert visual.active_alert.title == "TEST"

        clock.advance(2.5)
        assert visual.is_active is True

        clock.advance(0.6)  # Total 3.1s > 3.0s
        assert visual.is_active is False
        assert visual.active_alert is None

    def test_draw_does_not_crash(self):
        clock = MockClock(10.0)
        visual = VisualAlert(default_duration_sec=3.0, enabled=True, clock=clock)
        visual.trigger("PHONE DETECTED", "Please put phone away.")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        visual.draw(frame)
        assert np.count_nonzero(frame) > 0


class TestAlertManager:
    def _create_manager(self, tmp_path, clock):
        p_sound = tmp_path / "phone_alert.mp3"
        d_sound = tmp_path / "drowsiness_alert.mp3"
        dist_sound = tmp_path / "distraction_alert.mp3"
        for s in (p_sound, d_sound, dist_sound):
            s.write_bytes(b"audio")

        cfg = AlertConfig(
            alerts_enabled=True,
            sound_alerts_enabled=True,
            visual_alerts_enabled=True,
            phone_alert_cooldown_sec=5.0,
            drowsiness_alert_cooldown_sec=5.0,
            distraction_alert_cooldown_sec=5.0,
            away_alert_cooldown_sec=10.0,
            phone_alert_sound_path=str(p_sound),
            drowsiness_alert_sound_path=str(d_sound),
            distraction_alert_sound_path=str(dist_sound),
        )

        mock_mixer = MockMixerModule()
        sound_alert = SoundAlert(volume=0.7, enabled=True, mixer_module=mock_mixer)
        visual_alert = VisualAlert(default_duration_sec=3.0, enabled=True, clock=clock)

        manager = AlertManager(
            alert_config=cfg,
            sound_alert=sound_alert,
            visual_alert=visual_alert,
            clock=clock,
        )
        return manager, mock_mixer

    def test_focused_triggers_no_alert(self, tmp_path):
        clock = MockClock()
        manager, _ = self._create_manager(tmp_path, clock)
        res = BehaviorResult(state=BehaviorState.FOCUSED, reason=BehaviorReason.FOCUSED)
        event = manager.handle_behavior(res)
        assert event is None
        assert manager.visual.is_active is False

    def test_phone_state_entry_triggers_alert(self, tmp_path):
        clock = MockClock()
        manager, mixer = self._create_manager(tmp_path, clock)

        # Baseline FOCUSED
        manager.handle_behavior(BehaviorResult(state=BehaviorState.FOCUSED))

        # Enter PHONE_DETECTED
        clock.advance(0.1)
        event = manager.handle_behavior(BehaviorResult(state=BehaviorState.PHONE_DETECTED))
        assert event is not None
        assert event.alert_type == AlertType.PHONE
        assert event.sound_played is True
        assert event.visual_shown is True

    def test_persistent_state_respects_cooldown(self, tmp_path):
        clock = MockClock()
        manager, mixer = self._create_manager(tmp_path, clock)

        # Enter PHONE_DETECTED at t=100
        event1 = manager.handle_behavior(BehaviorResult(state=BehaviorState.PHONE_DETECTED))
        assert event1 is not None

        # Repeat PHONE_DETECTED at t=101 (within 5.0s cooldown)
        clock.advance(1.0)
        event2 = manager.handle_behavior(BehaviorResult(state=BehaviorState.PHONE_DETECTED))
        assert event2 is None  # Blocked by cooldown

        # Advance past cooldown to t=106
        clock.advance(4.5)
        event3 = manager.handle_behavior(BehaviorResult(state=BehaviorState.PHONE_DETECTED))
        assert event3 is not None  # Allowed after cooldown

    def test_drowsiness_alert_trigger(self, tmp_path):
        clock = MockClock()
        manager, mixer = self._create_manager(tmp_path, clock)
        event = manager.handle_behavior(BehaviorResult(state=BehaviorState.DROWSY))
        assert event is not None
        assert event.alert_type == AlertType.DROWSINESS
        assert event.sound_played is True
        # Verify mapped drowsiness_alert.mp3 was played
        d_sound = str((tmp_path / "drowsiness_alert.mp3").resolve())
        assert mixer.sounds[d_sound].played is True

    def test_distraction_alert_trigger(self, tmp_path):
        clock = MockClock()
        manager, mixer = self._create_manager(tmp_path, clock)
        event = manager.handle_behavior(BehaviorResult(state=BehaviorState.DISTRACTED))
        assert event is not None
        assert event.alert_type == AlertType.DISTRACTION
        assert event.sound_played is True
        # Verify mapped distraction_alert.mp3 was played
        dist_sound = str((tmp_path / "distraction_alert.mp3").resolve())
        assert mixer.sounds[dist_sound].played is True

    def test_no_face_and_unknown_trigger_no_alert(self, tmp_path):
        clock = MockClock()
        manager, mixer = self._create_manager(tmp_path, clock)
        assert manager.handle_behavior(BehaviorResult(state=BehaviorState.NO_FACE)) is None
        assert manager.handle_behavior(BehaviorResult(state=BehaviorState.UNKNOWN)) is None
        assert manager.handle_behavior(BehaviorResult(state=BehaviorState.FOCUSED)) is None
        # Verify no audio played for neutral/availability states
        assert not any(s.played for s in mixer.sounds.values())

    def test_master_switch_disables_all_alerts(self, tmp_path):
        clock = MockClock()
        manager, _ = self._create_manager(tmp_path, clock)
        manager.config = AlertConfig(alerts_enabled=False)

        event = manager.handle_behavior(BehaviorResult(state=BehaviorState.PHONE_DETECTED))
        assert event is None
        assert manager.visual.is_active is False

    def test_audio_overlap_prevention(self, tmp_path):
        clock = MockClock()
        manager, mixer = self._create_manager(tmp_path, clock)

        # Trigger distraction
        ev1 = manager.handle_behavior(BehaviorResult(state=BehaviorState.DISTRACTED))
        assert ev1 is not None
        assert mixer.stopped is True  # stop_all was called before playback

        # Advance clock and trigger phone (higher priority)
        clock.advance(0.5)
        mixer.stopped = False
        ev2 = manager.handle_behavior(BehaviorResult(state=BehaviorState.PHONE_DETECTED))
        assert ev2 is not None
        assert mixer.stopped is True  # stop_all called again to cancel previous sound

    def test_real_audio_assets_exist_and_loadable(self):
        from pathlib import Path
        from config.config import BASE_DIR

        sounds_dir = BASE_DIR / "assets" / "sounds"
        expected_files = ["phone_alert.mp3", "distraction_alert.mp3", "drowsiness_alert.mp3"]

        for filename in expected_files:
            file_path = sounds_dir / filename
            assert file_path.exists(), f"Expected audio asset missing: {file_path}"
            assert file_path.stat().st_size > 0, f"Audio file is empty: {file_path}"

        # Test loading real files with mock mixer
        mock_mixer = MockMixerModule()
        sound_alert = SoundAlert(enabled=True, mixer_module=mock_mixer)
        for filename in expected_files:
            key = filename.replace("_alert.mp3", "")
            loaded = sound_alert.load_sound(key, sounds_dir / filename)
            assert loaded is True, f"Failed to load real audio asset {filename}"
            assert sound_alert.is_loaded(key) is True
