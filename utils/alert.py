import sys
import time

try:
    import winsound  # type: ignore
except Exception:
    winsound = None


class AlertManager:
    def __init__(self, enabled: bool = True, cooldown: float = 2.0):
        self.enabled = enabled
        self.cooldown = cooldown
        self._last_trigger = 0.0

    def trigger(self):
        if not self.enabled:
            return
        now = time.time()
        if now - self._last_trigger < self.cooldown:
            return
        self._last_trigger = now
        _beep()


def _beep():
    if winsound is not None:
        try:
            winsound.Beep(1200, 200)
            return
        except Exception:
            pass

    try:
        import numpy as np
        import simpleaudio as sa  # type: ignore

        frequency = 1200
        duration = 0.2
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = np.sin(frequency * t * 2 * np.pi)
        audio = (tone * 32767).astype(np.int16)
        sa.play_buffer(audio, 1, 2, sample_rate)
        return
    except Exception:
        pass

    # Terminal bell fallback (may be silent depending on system settings)
    sys.stdout.write("\a")
    sys.stdout.flush()
