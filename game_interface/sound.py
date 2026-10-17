"""Jump/duck sound effects, synthesized with numpy so no asset files are needed."""
import numpy as np
import pygame

RATE = 22050


def _sweep(f0: float, f1: float, seconds: float, volume: float = 0.35) -> pygame.mixer.Sound:
    n = int(RATE * seconds)
    freq = np.linspace(f0, f1, n)
    phase = 2 * np.pi * np.cumsum(freq) / RATE
    fade = np.linspace(1.0, 0.0, n) ** 1.5
    wave = (np.sin(phase) * fade * volume * 32767).astype(np.int16)
    return pygame.mixer.Sound(buffer=wave.tobytes())


class Sounds:
    """Falls back to silence (with one warning) if there's no audio device."""

    def __init__(self):
        self.jump = self.duck = None
        try:
            pygame.mixer.init(frequency=RATE, size=-16, channels=1)
            self.jump = _sweep(300, 800, 0.18)    # rising chirp
            self.duck = _sweep(500, 180, 0.18)    # falling tone
        except pygame.error as e:
            print(f"[sound disabled] no audio device available: {e}")

    def play(self, name):
        sound = {"jump": self.jump, "duck": self.duck}.get(name)
        if sound:
            sound.play()
