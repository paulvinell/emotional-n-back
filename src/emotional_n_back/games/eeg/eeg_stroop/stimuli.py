
import pygame
import numpy as np
from typing import Dict

def make_beep(
    frequency: int = 880, duration_ms: int = 120, volume: float = 0.5
) -> pygame.mixer.Sound:
    """Generate a sine beep as a pygame Sound. Assumes mixer is initialized."""
    init = pygame.mixer.get_init()
    if init is None:
        raise RuntimeError("pygame.mixer not initialized")
    sample_rate, _fmt, channels = init

    n_samples = int(sample_rate * (duration_ms / 1000.0))
    t = np.linspace(
        0.0, duration_ms / 1000.0, n_samples, endpoint=False, dtype=np.float64
    )
    wave = 0.5 * np.sin(2.0 * np.pi * float(frequency) * t)
    mono = (wave * (2**15 - 1)).astype(np.int16, copy=False)

    if channels == 1:
        pcm = mono
    elif channels == 2:
        pcm = np.column_stack((mono, mono))
    else:
        raise ValueError(f"Unsupported mixer channels: {channels}")

    pcm = np.ascontiguousarray(pcm)
    snd = pygame.sndarray.make_sound(pcm)
    snd.set_volume(max(0.0, min(1.0, float(volume))))
    return snd

def load_fit_image(path: str, box: pygame.Rect, img_cache: Dict[str, pygame.Surface]) -> pygame.Surface:
    if path in img_cache:
        return img_cache[path]
    img = pygame.image.load(path).convert_alpha()
    iw, ih = img.get_width(), img.get_height()
    scale = min(box.w / iw, box.h / ih)
    surf = pygame.transform.smoothscale(img, (int(iw * scale), int(ih * scale)))
    img_cache[path] = surf
    return surf
