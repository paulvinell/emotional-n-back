import random
from typing import Optional

import numpy as np
import pygame
from pygame import Rect

from emotional_n_back.data import KDEFLoader, MAVLoader
from emotional_n_back.nback import NBackSequence


def make_beep(
    frequency: int = 880, duration_ms: int = 120, volume: float = 0.5
) -> pygame.mixer.Sound:
    """
    Generate a sine beep as a pygame Sound that matches the current mixer config.
    Expects pygame.mixer to be initialized. No try/except here by request.
    """
    init = pygame.mixer.get_init()
    if init is None:
        raise RuntimeError(
            "pygame.mixer is not initialized; call pygame.mixer.init() first."
        )
    sample_rate, _fmt, channels = init

    # time base
    n_samples = int(sample_rate * (duration_ms / 1000.0))
    t = np.linspace(
        0.0, duration_ms / 1000.0, n_samples, endpoint=False, dtype=np.float64
    )

    # mono waveform in float, then to int16 PCM
    wave = 0.5 * np.sin(2.0 * np.pi * float(frequency) * t)  # [-0.5, 0.5]
    mono = (wave * (2**15 - 1)).astype(np.int16, copy=False)

    # match mixer channels
    if channels == 1:
        pcm = mono  # shape (N,)
    elif channels == 2:
        pcm = np.column_stack((mono, mono))  # shape (N, 2)
    else:
        raise ValueError(f"Unsupported channel count from mixer: {channels}")

    # ensure C-contiguous for sndarray
    pcm = np.ascontiguousarray(pcm)
    snd = pygame.sndarray.make_sound(pcm)
    snd.set_volume(max(0.0, min(1.0, float(volume))))
    return snd


class EmotionalDualNBack:
    """
    Visual = KDEF emotion class (image shown centered).
    Auditory = MAV emotion class (random wav for that emotion is played).
    Matching rule: current emotion == emotion n steps back (per modality).

    Keys:
      1 -> Image match (press independently any time during the trial)
      2 -> Audio match (press independently any time during the trial)
      ESC -> Quit
    """

    def __init__(
        self,
        *,
        length: int = 30,
        n: int = 2,
        repeat_probability: float = 0.25,
        seed: int | None = None,
        stim_ms: int = 1000,
        isi_ms: int = 500,
        feedback_ms: int = 300,  # kept for compatibility; not used for an overlay
        window_size=(900, 650),
        show_debug_labels: bool = False,
        show_help_labels: bool = False,
    ):
        if seed is not None:
            random.seed(seed)

        self.length = length
        self.n = n
        self.stim_ms = stim_ms
        self.isi_ms = isi_ms
        self.feedback_ms = feedback_ms
        self.show_debug_labels = show_debug_labels
        self.show_help_labels = show_help_labels

        # Load datasets
        self.kdef = KDEFLoader()
        self.mav = MAVLoader()
        if not self.kdef.emotions:
            raise RuntimeError("KDEF emotions list is empty.")
        if not self.mav.emotions:
            raise RuntimeError("MAV emotions list is empty.")

        # Sequences over emotion indices (may yield (val, truth)); normalize to ints
        self.vis_seq_idx = [
            x[0] if isinstance(x, (tuple, list)) else x
            for x in NBackSequence(
                length,
                n,
                repeat_probability=repeat_probability,
                distinct_items=len(self.kdef.emotions),
            )
        ]
        self.aud_seq_idx = [
            x[0] if isinstance(x, (tuple, list)) else x
            for x in NBackSequence(
                length,
                n,
                repeat_probability=repeat_probability,
                distinct_items=len(self.mav.emotions),
            )
        ]

        # Pygame setup
        pygame.init()
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption(f"Emotional Dual N-Back (n={n})")
        self.clock = pygame.time.Clock()
        self.font_big = pygame.font.SysFont(None, 48)
        self.font_small = pygame.font.SysFont(None, 24)

        # Layout
        self.image_rect = Rect(100, 90, 400, 400)  # fixed image box
        self.status_img_rect = Rect(540, 140, 300, 90)
        self.status_aud_rect = Rect(540, 250, 300, 90)

        # Feedback beeps
        self.beep_double = make_beep(1100, 130, 0.6)  # double-correct
        self.beep_half = make_beep(650, 130, 0.6)  # half-correct

        # Stats
        self.t = 0
        self.v_hist: list[int] = []  # emotion index history (visual)
        self.a_hist: list[int] = []  # emotion index history (audio)
        self.v_correct = 0
        self.a_correct = 0

        # Simple image cache
        self._img_cache: dict[str, pygame.Surface] = {}

    # ---------- helpers ----------
    @staticmethod
    def nback_match(history: list[int], n: int) -> bool:
        return len(history) > n and history[-1] == history[-(n + 1)]

    def _load_image_fit(self, path) -> pygame.Surface:
        key = str(path)
        if key in self._img_cache:
            return self._img_cache[key]
        img = pygame.image.load(key).convert_alpha()
        iw, ih = img.get_width(), img.get_height()
        box_w, box_h = self.image_rect.w, self.image_rect.h
        scale = min(box_w / iw, box_h / ih)
        surf = pygame.transform.smoothscale(img, (int(iw * scale), int(ih * scale)))
        self._img_cache[key] = surf
        return surf

    def _draw_status_box(
        self,
        rect: pygame.Rect,
        label: str,
        answered: bool,
        correct: bool,
        subtitle: str | None = None,
    ):
        # Panel and border (gray)
        pygame.draw.rect(self.screen, (40, 42, 48), rect, border_radius=12)
        pygame.draw.rect(self.screen, (160, 160, 170), rect, width=2, border_radius=12)

        # Fill only if answered: green/red with transparency
        if answered:
            fill = (40, 160, 90, 140) if correct else (180, 60, 60, 140)
            overlay = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
            overlay.fill(fill)
            self.screen.blit(overlay, rect.topleft)

        # Main label
        lbl = self.font_big.render(label, True, (235, 235, 235))
        if subtitle and self.show_help_labels:
            lbl_rect = lbl.get_rect(center=(rect.centerx, rect.y + rect.h * 0.40))
            self.screen.blit(lbl, lbl_rect)

            # Subtitle (shown once answered, persists until next trial)
            sub = self.font_small.render(subtitle, True, (235, 235, 235))
            sub_rect = sub.get_rect(center=(rect.centerx, rect.y + rect.h * 0.72))
            self.screen.blit(sub, sub_rect)
        else:
            lbl_rect = lbl.get_rect(center=rect.center)
            self.screen.blit(lbl, lbl_rect)

    def _draw_frame(
        self,
        image: pygame.Surface | None,
        *,
        answered_v: bool,
        correct_v_now: bool,
        subtitle_v: str | None,
        answered_a: bool,
        correct_a_now: bool,
        subtitle_a: str | None,
        labels: dict[str, str] | None = None,
    ):
        self.screen.fill((20, 22, 26))

        # Header & tips
        hdr = self.font_big.render(
            f"Trial {self.t + 1}/{self.length}   n={self.n}", True, (235, 235, 235)
        )
        self.screen.blit(hdr, (24, 24))
        tip = self.font_small.render(
            "Press 1 for Image match, 2 for Audio match (independent).",
            True,
            (210, 210, 210),
        )
        self.screen.blit(tip, (24, 60))

        # Image area
        pygame.draw.rect(self.screen, (60, 60, 65), self.image_rect, border_radius=12)
        pygame.draw.rect(
            self.screen, (160, 160, 170), self.image_rect, width=2, border_radius=12
        )
        if image:
            rect = image.get_rect(center=self.image_rect.center)
            self.screen.blit(image, rect)

        # Status boxes
        self._draw_status_box(
            self.status_img_rect, "Image", answered_v, correct_v_now, subtitle_v
        )
        self._draw_status_box(
            self.status_aud_rect, "Audio", answered_a, correct_a_now, subtitle_a
        )

        # Optional debug labels
        if labels and self.show_debug_labels:
            y = self.status_aud_rect.bottom + 12
            for k, v in labels.items():
                s = self.font_small.render(f"{k}: {v}", True, (220, 220, 220))
                self.screen.blit(s, (24, y))
                y += 20

        # Scorebar (accurate after each completed trial)
        s_txt = self.font_small.render(
            f"Image: {self.v_correct}/{self.t}    Audio: {self.a_correct}/{self.t}",
            True,
            (200, 200, 200),
        )
        self.screen.blit(s_txt, (24, self.screen.get_height() - 30))

        pygame.display.flip()

    # ---------- main loop ----------
    def run(self):
        running = True
        while running and self.t < self.length:
            # Current emotion indices (1-based from sequences)
            v_idx = self.vis_seq_idx[self.t]
            a_idx = self.aud_seq_idx[self.t]

            # Map to emotion names
            v_emotion = self.kdef.get_emotion(v_idx - 1)  # loaders are 0-based
            a_emotion = self.mav.get_emotion(a_idx - 1)

            # Pick concrete stimuli
            img_path = self.kdef.get_random_image(v_emotion)
            aud_path = self.mav.get_random_audio(a_emotion)

            # Load image; play audio
            image = self._load_image_fit(img_path)
            try:
                sound = pygame.mixer.Sound(str(aud_path))
            except Exception:
                sound = None

            # Append to histories for GT
            self.v_hist.append(v_idx)
            self.a_hist.append(a_idx)
            gt_v = self.nback_match(self.v_hist, self.n)
            gt_a = self.nback_match(self.a_hist, self.n)

            # Per-trial input state (independent)
            answered_v = False
            answered_a = False
            correct_v_now = False  # correctness for the press (if answered)
            correct_a_now = False
            subtitle_v = (
                None  # "This was: <emotion>" after answer, persists until next trial
            )
            subtitle_a = None

            # --- Stimulus phase ---
            if sound:
                sound.play()
            stim_start = pygame.time.get_ticks()
            while pygame.time.get_ticks() - stim_start < self.stim_ms:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_1 and not answered_v:
                            answered_v = True
                            correct_v_now = gt_v
                            subtitle_v = f"This was: {v_emotion}"
                            if correct_v_now:
                                if answered_a and correct_a_now and self.beep_double:
                                    self.beep_double.play()
                                elif self.beep_half:
                                    self.beep_half.play()
                        elif event.key == pygame.K_2 and not answered_a:
                            answered_a = True
                            correct_a_now = gt_a
                            subtitle_a = f"This was: {a_emotion}"
                            if correct_a_now:
                                if answered_v and correct_v_now and self.beep_double:
                                    self.beep_double.play()
                                elif self.beep_half:
                                    self.beep_half.play()

                self._draw_frame(
                    image,
                    answered_v=answered_v,
                    correct_v_now=correct_v_now,
                    subtitle_v=subtitle_v,
                    answered_a=answered_a,
                    correct_a_now=correct_a_now,
                    subtitle_a=subtitle_a,
                    labels={
                        "Image emotion": v_emotion,
                        "Audio emotion": a_emotion,
                        "GT Image match?": "Yes" if gt_v else "No",
                        "GT Audio match?": "Yes" if gt_a else "No",
                    },
                )
                self.clock.tick(120)

            # --- ISI (blank), still accept independent inputs ---
            isi_start = pygame.time.get_ticks()
            while pygame.time.get_ticks() - isi_start < self.isi_ms:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_1 and not answered_v:
                            answered_v = True
                            correct_v_now = gt_v
                            subtitle_v = f"This was: {v_emotion}"
                            if correct_v_now:
                                if answered_a and correct_a_now and self.beep_double:
                                    self.beep_double.play()
                                elif self.beep_half:
                                    self.beep_half.play()
                        elif event.key == pygame.K_2 and not answered_a:
                            answered_a = True
                            correct_a_now = gt_a
                            subtitle_a = f"This was: {a_emotion}"
                            if correct_a_now:
                                if answered_v and correct_v_now and self.beep_double:
                                    self.beep_double.play()
                                elif self.beep_half:
                                    self.beep_half.play()

                self._draw_frame(
                    None,
                    answered_v=answered_v,
                    correct_v_now=correct_v_now,
                    subtitle_v=subtitle_v,
                    answered_a=answered_a,
                    correct_a_now=correct_a_now,
                    subtitle_a=subtitle_a,
                    labels=None,
                )
                self.clock.tick(120)

            # --- Score this trial ---
            # “Pressed == match” is a correct response; unpressed on non-match also correct
            press_v = answered_v
            press_a = answered_a
            correct_v = press_v == gt_v
            correct_a = press_a == gt_a
            if correct_v:
                self.v_correct += 1
            if correct_a:
                self.a_correct += 1

            # Advance to next trial
            self.t += 1

        # Final screen
        self.screen.fill((20, 22, 26))
        acc_v = 100.0 * (self.v_correct / max(1, self.t))
        acc_a = 100.0 * (self.a_correct / max(1, self.t))
        summary = f"Done!  Image: {self.v_correct}/{self.t} ({acc_v:.1f}%)   Audio: {self.a_correct}/{self.t} ({acc_a:.1f}%)"
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

        # Wait for exit
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    waiting = False
            self.clock.tick(60)
        pygame.quit()
