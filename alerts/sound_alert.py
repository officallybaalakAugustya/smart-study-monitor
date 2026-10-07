"""Sound alert module for Smart Study Monitor.

Part of Phase 7.
Handles non-blocking audio playback using pygame.mixer.
Initializes audio once and loads configured sound assets once.
Gracefully handles missing audio files or unavailable audio devices without crashing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

from config.config import BASE_DIR

logger = logging.getLogger(__name__)


class SoundAlert:
    """Manages audio playback, mixer lifecycle, and sound asset caching."""

    def __init__(
        self,
        volume: float = 0.7,
        enabled: bool = True,
        mixer_module: Optional[object] = None,
    ) -> None:
        """Initializes SoundAlert.

        Args:
            volume: Master volume in [0.0, 1.0].
            enabled: Master sound enabled flag.
            mixer_module: Optional injectable mixer module (for unit testing/mocking).
        """
        self.volume = max(0.0, min(1.0, float(volume)))
        self.enabled = bool(enabled)
        self._sounds: Dict[str, object] = {}
        self._mixer_available = False

        if mixer_module is not None:
            self._mixer = mixer_module
            self._mixer_available = True
        else:
            self._init_mixer()

    def _init_mixer(self) -> None:
        """Initializes pygame.mixer once."""
        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            self._mixer = pygame.mixer
            self._mixer_available = True
            logger.info("Pygame audio mixer initialized successfully.")
        except Exception as exc:
            self._mixer_available = False
            self._mixer = None
            logger.warning("Audio device or pygame mixer unavailable: %s. Audio alerts disabled.", exc)

    def load_sound(self, name: str, file_path: str | Path) -> bool:
        """Loads a sound file into memory once.

        Args:
            name: Logical sound identifier (e.g. 'phone', 'drowsiness').
            file_path: Absolute or relative path to the sound file.

        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        if not self._mixer_available or self._mixer is None:
            return False

        p = Path(file_path)
        path = (BASE_DIR / p).resolve() if not p.is_absolute() else p.resolve()
        if not path.exists():
            logger.warning("Audio file not found: '%s'. Disabling sound '%s'.", path.name, name)
            return False

        try:
            sound = self._mixer.Sound(str(path))
            sound.set_volume(self.volume)
            self._sounds[name] = sound
            logger.info("Loaded %s alert sound from %s", name, path.name)
            return True
        except Exception as exc:
            logger.warning("Failed to load sound file '%s': %s", path.name, exc)
            return False

    def is_loaded(self, name: str) -> bool:
        """Returns True if the specified sound identifier is loaded into cache."""
        return name in self._sounds

    def loaded_sounds(self) -> list[str]:
        """Returns a list of all currently loaded sound identifiers."""
        return list(self._sounds.keys())

    def play(self, name: str) -> bool:
        """Plays a loaded sound asset asynchronously (non-blocking).

        Stops any currently playing alert audio first to prevent overlapping sounds.

        Args:
            name: Logical sound identifier.

        Returns:
            bool: True if sound was played, False otherwise.
        """
        if not self.enabled or not self._mixer_available or self._mixer is None:
            return False

        sound = self._sounds.get(name)
        if sound is None:
            return False

        try:
            # Prevent audio overlap by stopping previous alert sound before triggering new one
            self.stop_all()
            sound.play()
            return True
        except Exception as exc:
            logger.warning("Playback failure for sound '%s': %s", name, exc)
            return False

    def set_volume(self, volume: float) -> None:
        """Updates master playback volume for all loaded sounds."""
        self.volume = max(0.0, min(1.0, float(volume)))
        for sound in self._sounds.values():
            try:
                sound.set_volume(self.volume)
            except Exception:
                pass

    def stop_all(self) -> None:
        """Stops all currently playing sounds."""
        if self._mixer_available and self._mixer is not None:
            try:
                self._mixer.stop()
            except Exception:
                pass

    def close(self) -> None:
        """Cleans up audio resources on shutdown."""
        self.stop_all()
        self._sounds.clear()
