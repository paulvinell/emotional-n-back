import pygame
from pygame import Rect
from typing import Optional

from emotional_n_back.data import (
    KDEFSentimentLoader,
    MAVSentimentLoader,
)
from .reward_modules import Sentiment


class ResourceManager:
    def __init__(self):
        self.kdef_loader = KDEFSentimentLoader()
        self.mav_loader = MAVSentimentLoader()
        self.sentiments = self.mav_loader.sentiments
        self._img_cache: dict[str, pygame.Surface] = {}

    def get_random_sentiment(self) -> Sentiment:
        # Assuming sentiments are compatible with Sentiment enum or we map them
        # The original code used self.mav_loader.sentiments which are likely strings or enums
        # Let's check game.py imports. It imports KDEFSentimentLoader, MAVSentimentLoader.
        # And in game.py: self.visual_sentiment = random.choice(self.sentiments)
        # We need to make sure we return what the game expects.
        # For now, let's expose the list.
        import random
        return random.choice(self.sentiments)

    def get_random_image_path(self, sentiment) -> str:
        return str(self.kdef_loader.get_random_image(sentiment))

    def get_random_audio_path(self, sentiment) -> str:
        return str(self.mav_loader.get_random_audio(sentiment))

    def load_audio(self, path: str) -> pygame.mixer.Sound:
        sound = pygame.mixer.Sound(path)
        if sound is None:
            raise RuntimeError(f"Failed to load audio file: {path}")
        return sound

    def load_fit_image(self, path: str, box: Rect) -> pygame.Surface:
        if path in self._img_cache:
            return self._img_cache[path]
        
        try:
            img = pygame.image.load(path).convert_alpha()
        except pygame.error as e:
            # Fallback or re-raise? Original code didn't handle it explicitly but pygame would raise.
            raise RuntimeError(f"Failed to load image: {path}") from e

        iw, ih = img.get_width(), img.get_height()
        scale = min(box.w / iw, box.h / ih)
        surf = pygame.transform.smoothscale(img, (int(iw * scale), int(ih * scale)))
        self._img_cache[path] = surf
        return surf
