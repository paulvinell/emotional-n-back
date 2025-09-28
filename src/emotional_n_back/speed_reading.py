# speed_reading.py
import random
from typing import Optional

import pygame
from pygame import Rect

from emotional_n_back.data import (
    KDEFSentimentLoader,
    MAVLoader,
    TextLoader,
)


class SpeedReadingGame:
    """
    A speed reading game with emotional distractions.
    Text scrolls from right to left. The user's goal is to read it.
    Random visual (faces) and audio (vocalizations) distractions appear.
    """

    def __init__(
        self,
        *,
        language: str = "en",
        scroll_speed: int = 150,  # pixels per second
        audio_distraction_freq: float = 0.2,  # events per second
        visual_distraction_freq: float = 0.3,  # events per second
        visual_distraction_duration_ms: int = 700,
        seed: Optional[int] = None,
        window_size=(1200, 600),
    ):
        if seed is not None:
            random.seed(seed)

        self.scroll_speed = scroll_speed
        self.audio_distraction_freq = audio_distraction_freq
        self.visual_distraction_freq = visual_distraction_freq
        self.visual_distraction_duration_ms = visual_distraction_duration_ms

        # Pygame
        pygame.init()
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Speed Reading Game")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 60)

        # Data Loaders
        self.text_loader = TextLoader(language=language)
        self.word_generator = self.text_loader.word_generator()
        self.kdef_loader = KDEFSentimentLoader()
        self.mav_loader = MAVLoader()
        self._img_cache: dict[str, pygame.Surface] = {}

        # UI Layout
        W, H = self.screen.get_size()
        self.distraction_area = Rect(0, 0, W, H // 2)
        self.text_y = H * 3 // 4

        # Game state
        self.text_surfaces = [self._create_text_surface(), self._create_text_surface()]
        self.current_surface_idx = 0
        self.text_x = float(self.screen.get_width())

        self.visual_distraction_surface: Optional[pygame.Surface] = None
        self.visual_distraction_end_time: int = 0
        self.visual_distraction_centerx: int = 0

    def _load_fit_image(self, path: str, box: Rect) -> pygame.Surface:
        if path in self._img_cache:
            return self._img_cache[path]
        img = pygame.image.load(path).convert_alpha()
        iw, ih = img.get_width(), img.get_height()
        scale = min(box.w / iw, box.h / ih)
        surf = pygame.transform.smoothscale(img, (int(iw * scale), int(ih * scale)))
        self._img_cache[path] = surf
        return surf

    def _create_text_surface(self) -> pygame.Surface:
        num_words = 15
        words = [next(self.word_generator) for _ in range(num_words)]
        text = " ".join(words) + "   "
        return self.font.render(text, True, (230, 230, 230))

    def run(self):
        running = True
        while running:
            delta_time_ms = self.clock.tick(60)
            delta_time_s = delta_time_ms / 1000.0
            now = pygame.time.get_ticks()

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    running = False

            # --- Update text scrolling ---
            self.text_x -= self.scroll_speed * delta_time_s

            current_surface = self.text_surfaces[self.current_surface_idx]

            if self.text_x < -current_surface.get_width():
                self.text_x += current_surface.get_width()
                self.current_surface_idx = (self.current_surface_idx + 1) % len(
                    self.text_surfaces
                )
                self.text_surfaces[
                    (self.current_surface_idx - 1 + len(self.text_surfaces))
                    % len(self.text_surfaces)
                ] = self._create_text_surface()

            # --- Handle distractions ---
            if self.visual_distraction_surface and now > self.visual_distraction_end_time:
                self.visual_distraction_surface = None

            if (
                not self.visual_distraction_surface
                and random.random() < self.visual_distraction_freq * delta_time_s
            ):
                sentiment = random.choice(self.kdef_loader.sentiments)
                image_path = self.kdef_loader.get_random_image(sentiment)

                distraction_box = Rect(
                    0,
                    0,
                    self.distraction_area.width * 0.8,
                    self.distraction_area.height * 0.95,
                )
                self.visual_distraction_surface = self._load_fit_image(
                    str(image_path), distraction_box
                )
                self.visual_distraction_end_time = (
                    now + self.visual_distraction_duration_ms
                )
                W = self.screen.get_width()
                distraction_w = self.visual_distraction_surface.get_width()
                if distraction_w < W:
                    self.visual_distraction_centerx = random.randint(
                        distraction_w // 2, W - distraction_w // 2
                    )
                else:
                    self.visual_distraction_centerx = W // 2

            if random.random() < self.audio_distraction_freq * delta_time_s:
                emotion = random.choice(self.mav_loader.emotions)
                audio_path = self.mav_loader.get_random_audio(emotion)
                pygame.mixer.Sound(str(audio_path)).play()

            # --- Drawing ---
            self.screen.fill((20, 22, 26))

            # Draw text
            text_y = self.text_y

            current_surface = self.text_surfaces[self.current_surface_idx]
            self.screen.blit(
                current_surface, (self.text_x, text_y - current_surface.get_height() // 2)
            )

            next_surface_idx = (self.current_surface_idx + 1) % len(self.text_surfaces)
            next_surface = self.text_surfaces[next_surface_idx]
            self.screen.blit(
                next_surface,
                (
                    self.text_x + current_surface.get_width(),
                    text_y - next_surface.get_height() // 2,
                ),
            )

            # Draw visual distraction
            if self.visual_distraction_surface:
                distraction_rect = self.visual_distraction_surface.get_rect(
                    center=(
                        self.visual_distraction_centerx,
                        self.distraction_area.centery + 30,
                    )
                )
                self.screen.blit(self.visual_distraction_surface, distraction_rect)

            pygame.display.flip()

        pygame.quit()
