# stroop.py
import random
from typing import Optional

import numpy as np
import pygame
from pygame import Rect

from emotional_n_back.data import KDEFSentimentLoader, MAVSentimentLoader


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


class SentimentStroopGame:
    """
    A sentiment-based Stroop game.

    In each trial:
    1. A face with a random sentiment (positive, neutral, negative) is shown.
    2. After a delay, an audio clip with a random sentiment is played.
    3. The user must identify the sentiment of the AUDIO clip, ignoring the face.

    Keys:
    - 1: Positive
    - 2: Neutral
    - 3: Negative
    """

    def __init__(
        self,
        *,
        length: int = 30,
        seed: Optional[int] = None,
        visual_intro_ms: int = 500,  # Time face is shown before audio
        response_window_ms: int = 2000,  # Time to answer after audio starts
        feedback_ms: int = 500,
        isi_ms: int = 300,
        window_size=(900, 650),
    ):
        if seed is not None:
            random.seed(seed)
        self.length = length
        self.visual_intro_ms = visual_intro_ms
        self.response_window_ms = response_window_ms
        self.feedback_ms = feedback_ms
        self.isi_ms = isi_ms

        # Pygame
        pygame.init()
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Sentiment Stroop Game")
        self.clock = pygame.time.Clock()
        self.font_big = pygame.font.SysFont(None, 48)
        self.font_small = pygame.font.SysFont(None, 24)

        # Beeps
        self.beep_correct = make_beep(1100, 130, 0.6)
        self.beep_incorrect = make_beep(550, 180, 0.6)

        # Data Loaders
        self.kdef_loader = KDEFSentimentLoader()
        self.mav_loader = MAVSentimentLoader()
        self.sentiments = self.mav_loader.sentiments  # ['positive', 'neutral', 'negative']
        self.key_map = {
            pygame.K_1: "positive",
            pygame.K_2: "neutral",
            pygame.K_3: "negative",
        }

        # UI Layout
        W, H = self.screen.get_size()
        self.stimulus_rect = Rect(W // 2 - 220, 100, 440, 440)

        # Game State
        self.trial_num = 0
        self.score = 0
        self._img_cache: dict[str, pygame.Surface] = {}

    def _load_fit_image(self, path: str, box: Rect) -> pygame.Surface:
        if path in self._img_cache:
            return self._img_cache[path]
        img = pygame.image.load(path).convert_alpha()
        iw, ih = img.get_width(), img.get_height()
        scale = min(box.w / iw, box.h / ih)
        surf = pygame.transform.smoothscale(img, (int(iw * scale), int(ih * scale)))
        self._img_cache[path] = surf
        return surf

    def _draw_header(self):
        hdr = self.font_big.render(
            f"Trial {self.trial_num + 1}/{self.length}", True, (235, 235, 235)
        )
        self.screen.blit(hdr, (24, 24))
        tip = self.font_small.render(
            "What is the sentiment of the audio? (1: Pos, 2: Neu, 3: Neg)",
            True,
            (210, 210, 210),
        )
        self.screen.blit(tip, (24, 60))

    def _draw_scorebar(self):
        s_txt = self.font_small.render(
            f"Score: {self.score}/{self.trial_num}", True, (200, 200, 200)
        )
        self.screen.blit(s_txt, (24, self.screen.get_height() - 30))

    def _draw_stimulus_box(self, image_surface: Optional[pygame.Surface] = None):
        pygame.draw.rect(
            self.screen, (60, 60, 65), self.stimulus_rect, border_radius=12
        )
        pygame.draw.rect(
            self.screen,
            (160, 160, 170),
            self.stimulus_rect,
            width=2,
            border_radius=12,
        )
        if image_surface:
            dst = image_surface.get_rect(center=self.stimulus_rect.center)
            self.screen.blit(image_surface, dst)

    def run(self):
        running = True
        while running and self.trial_num < self.length:
            # --- 1. Prepare Trial ---
            visual_sentiment = random.choice(self.sentiments)
            audio_sentiment = random.choice(self.sentiments)

            image_path = self.kdef_loader.get_random_image(visual_sentiment)
            image_surface = self._load_fit_image(str(image_path), self.stimulus_rect)

            audio_path = self.mav_loader.get_random_audio(audio_sentiment)
            audio_sound = pygame.mixer.Sound(str(audio_path))

            answered_this_trial = False
            correct_answer = False

            # --- 2. Visual Intro Phase ---
            intro_t0 = pygame.time.get_ticks()
            while pygame.time.get_ticks() - intro_t0 < self.visual_intro_ms:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                    ):
                        running = False
                if not running:
                    break

                self.screen.fill((20, 22, 26))
                self._draw_header()
                self._draw_stimulus_box(image_surface)
                self._draw_scorebar()
                pygame.display.flip()
                self.clock.tick(120)
            if not running:
                break

            # --- 3. Response Phase (Audio plays) ---
            audio_sound.play()
            response_t0 = pygame.time.get_ticks()
            while pygame.time.get_ticks() - response_t0 < self.response_window_ms:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key in self.key_map and not answered_this_trial:
                            answered_this_trial = True
                            user_sentiment = self.key_map[event.key]
                            correct_answer = user_sentiment == audio_sentiment
                            if correct_answer:
                                self.beep_correct.play()
                                self.score += 1
                            else:
                                self.beep_incorrect.play()
                if not running or answered_this_trial:
                    break

                self.screen.fill((20, 22, 26))
                self._draw_header()
                self._draw_stimulus_box(image_surface)
                self._draw_scorebar()
                pygame.display.flip()
                self.clock.tick(120)
            if not running:
                break

            self.trial_num += 1

            # --- 4. Feedback Phase ---
            feedback_t0 = pygame.time.get_ticks()
            while pygame.time.get_ticks() - feedback_t0 < self.feedback_ms:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                    ):
                        running = False
                if not running:
                    break

                self.screen.fill((20, 22, 26))
                self._draw_header()
                # Show feedback color
                overlay = pygame.Surface(self.stimulus_rect.size, pygame.SRCALPHA)
                if answered_this_trial:
                    fill = (40, 160, 90, 140) if correct_answer else (180, 60, 60, 140)
                else: # Timed out
                    fill = (180, 60, 60, 140)
                overlay.fill(fill)

                self._draw_stimulus_box(image_surface)
                self.screen.blit(overlay, self.stimulus_rect.topleft)
                self._draw_scorebar()
                pygame.display.flip()
                self.clock.tick(120)
            if not running:
                break

            # --- 5. Inter-trial Interval (ISI) ---
            isi_t0 = pygame.time.get_ticks()
            while pygame.time.get_ticks() - isi_t0 < self.isi_ms:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                    ):
                        running = False
                if not running:
                    break

                self.screen.fill((20, 22, 26))
                self._draw_header()
                self._draw_stimulus_box() # Blank box
                self._draw_scorebar()
                pygame.display.flip()
                self.clock.tick(120)
            if not running:
                break

        # --- Final screen ---
        if running:
            self.show_final_screen()
        pygame.quit()

    def show_final_screen(self):
        self.screen.fill((20, 22, 26))
        final_trials = max(1, self.trial_num)
        acc = 100.0 * (self.score / final_trials)
        summary = f"Done! Score: {self.score}/{final_trials} ({acc:.1f}%)"
        s_surf = self.font_big.render(summary, True, (255, 255, 255))
        self.screen.blit(
            s_surf,
            s_surf.get_rect(
                center=(self.screen.get_width() // 2, self.screen.get_height() // 2)
            ),
        )
        tip = self.font_small.render(
            "Press Esc or close the window to exit.", True, (220, 220, 220)
        )
        self.screen.blit(
            tip,
            tip.get_rect(
                center=(
                    self.screen.get_width() // 2,
                    self.screen.get_height() // 2 + 40,
                )
            ),
        )
        pygame.display.flip()

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    waiting = False
            self.clock.tick(60)
