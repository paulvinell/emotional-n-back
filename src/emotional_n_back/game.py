import random
from typing import Optional

import pygame
from pygame import Rect

from emotional_n_back.data import KDEFLoader, MAVLoader
from emotional_n_back.nback import NBackSequence


# --------------------------------
# Minimal Pygame Dual N-Back Driver
# --------------------------------
class DualNBackPygame:
    def __init__(
        self,
        length: int = 30,
        n: int = 2,
        *,
        visual_items: int = 9,  # 3x3 grid positions
        auditory_items: int = 8,  # letters A..H
        repeat_probability: float = 0.2,
        seed: int | None = None,
        stim_ms: int = 1000,  # stimulus on screen (ms)
        isi_ms: int = 500,  # blank / response window (ms)
        show_feedback_ms: int = 350,  # brief ✓/✗ overlay after each trial
        window_size: tuple[int, int] = (700, 500),
    ):
        if seed is not None:
            random.seed(seed)
        self.length = length
        self.n = n
        self.stim_ms = stim_ms
        self.isi_ms = isi_ms
        self.show_feedback_ms = show_feedback_ms

        # Sequences (two modalities)
        self.visual_seq = [
            v
            for (v, _) in NBackSequence(
                length,
                n,
                repeat_probability=repeat_probability,
                distinct_items=visual_items,
            )
        ]
        self.audio_seq = [
            v
            for (v, _) in NBackSequence(
                length,
                n,
                repeat_probability=repeat_probability,
                distinct_items=auditory_items,
            )
        ]

        # History buffers to compute ground truth on the interleaved stream
        self.v_hist: list[int] = []
        self.a_hist: list[int] = []

        # Scoring
        self.v_correct = 0
        self.a_correct = 0
        self.trial = 0

        # Pygame setup
        pygame.init()
        pygame.display.set_caption(f"Dual N-Back (n={n})")
        self.screen = pygame.display.set_mode(window_size)
        self.clock = pygame.time.Clock()
        self.font_big = pygame.font.SysFont(None, 96)
        self.font_med = pygame.font.SysFont(None, 36)
        self.font_small = pygame.font.SysFont(None, 24)

        self.grid_rect = Rect(50, 80, 300, 300)  # left area for 3x3 grid
        self.audio_rect = Rect(420, 150, 230, 160)  # right area for big letter

    # ------- visuals -------
    def draw_grid(self, pos_idx: int | None):
        # 3x3 positions 1..9
        cols = rows = 3
        cell_w = self.grid_rect.w // cols
        cell_h = self.grid_rect.h // rows

        for r in range(rows):
            for c in range(cols):
                cell = Rect(
                    self.grid_rect.x + c * cell_w + 4,
                    self.grid_rect.y + r * cell_h + 4,
                    cell_w - 8,
                    cell_h - 8,
                )
                pygame.draw.rect(self.screen, (200, 200, 200), cell, width=2)

        if pos_idx is not None:
            idx = pos_idx - 1
            r = idx // cols
            c = idx % cols
            hi = Rect(
                self.grid_rect.x + c * cell_w + 10,
                self.grid_rect.y + r * cell_h + 10,
                cell_w - 20,
                cell_h - 20,
            )
            pygame.draw.rect(self.screen, (80, 180, 80), hi, border_radius=8)

        label = self.font_small.render("Visual (grid)", True, (220, 220, 220))
        self.screen.blit(label, (self.grid_rect.x, self.grid_rect.y - 24))

    def draw_audio_item(self, item_idx: int | None):
        # Map 1..8 to letters A..H
        letter = chr(ord("A") + (item_idx - 1)) if item_idx is not None else ""
        pygame.draw.rect(self.screen, (40, 40, 40), self.audio_rect, border_radius=10)
        pygame.draw.rect(
            self.screen, (200, 200, 200), self.audio_rect, width=2, border_radius=10
        )
        if letter:
            surf = self.font_big.render(letter, True, (200, 200, 255))
            rect = surf.get_rect(center=self.audio_rect.center)
            self.screen.blit(surf, rect)
        label = self.font_small.render("Auditory (letter)", True, (220, 220, 220))
        self.screen.blit(label, (self.audio_rect.x, self.audio_rect.y - 24))

    def draw_header(self):
        hdr = self.font_med.render(
            f"Trial {self.trial + 1}/{self.length}   n={self.n}", True, (230, 230, 230)
        )
        self.screen.blit(hdr, (20, 20))
        help_txt = self.font_small.render(
            "Press 1=Visual, 2=Auditory, b=Both, or do nothing for None",
            True,
            (190, 190, 190),
        )
        self.screen.blit(help_txt, (20, 50))

    def draw_feedback(self, correct_v: bool, correct_a: bool):
        txt = f"S1 {'✓' if correct_v else '✗'}   S2 {'✓' if correct_a else '✗'}"
        surf = self.font_med.render(txt, True, (255, 255, 255))
        rect = surf.get_rect(
            center=(self.screen.get_width() // 2, self.screen.get_height() - 40)
        )
        self.screen.blit(surf, rect)

    def draw_scorebar(self):
        txt = f"S1: {self.v_correct}/{self.trial}   S2: {self.a_correct}/{self.trial}"
        surf = self.font_small.render(txt, True, (200, 200, 200))
        self.screen.blit(surf, (20, self.screen.get_height() - 28))

    # ------- logic -------
    def gt(self, stream: list[int], n: int) -> bool:
        return len(stream) >= n + 1 and stream[-1] == stream[-(n + 1)]

    def run(self):
        running = True
        while running and self.trial < self.length:
            # Current stimuli
            v_item = self.visual_seq[self.trial]
            a_item = self.audio_seq[self.trial]

            # Append first, then compute GT against n-back
            self.v_hist.append(v_item)
            self.a_hist.append(a_item)
            gt_v = self.gt(self.v_hist, self.n)
            gt_a = self.gt(self.a_hist, self.n)

            # Collect responses for this trial
            want_v = False
            want_a = False

            # ---- Stimulus phase ----
            t_start = pygame.time.get_ticks()
            while pygame.time.get_ticks() - t_start < self.stim_ms:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_1:
                            want_v = True
                        elif event.key == pygame.K_2:
                            want_a = True
                        elif event.key == pygame.K_b:
                            want_v = True
                            want_a = True
                        elif event.key == pygame.K_ESCAPE:
                            running = False
                self.screen.fill((25, 25, 28))
                self.draw_header()
                self.draw_grid(v_item)  # show visual
                self.draw_audio_item(a_item)  # show auditory (letter)
                self.draw_scorebar()
                pygame.display.flip()
                self.clock.tick(120)

            # ---- ISI phase (blank, still accept input) ----
            t_blank = pygame.time.get_ticks()
            while pygame.time.get_ticks() - t_blank < self.isi_ms:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_1:
                            want_v = True
                        elif event.key == pygame.K_2:
                            want_a = True
                        elif event.key == pygame.K_b:
                            want_v = True
                            want_a = True
                        elif event.key == pygame.K_SPACE:
                            pass  # explicit "none"
                        elif event.key == pygame.K_ESCAPE:
                            running = False
                self.screen.fill((25, 25, 28))
                self.draw_header()
                self.draw_grid(None)
                self.draw_audio_item(None)
                self.draw_scorebar()
                pygame.display.flip()
                self.clock.tick(120)

            # Score this trial
            correct_v = want_v == gt_v
            correct_a = want_a == gt_a
            self.v_correct += 1 if correct_v else 0
            self.a_correct += 1 if correct_a else 0

            # Brief feedback overlay
            t_fb = pygame.time.get_ticks()
            while pygame.time.get_ticks() - t_fb < self.show_feedback_ms:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        running = False
                self.screen.fill((25, 25, 28))
                self.draw_header()
                self.draw_grid(None)
                self.draw_audio_item(None)
                self.draw_feedback(correct_v, correct_a)
                self.draw_scorebar()
                pygame.display.flip()
                self.clock.tick(120)

            self.trial += 1

        # Final summary screen
        self.screen.fill((25, 25, 28))
        acc_v = 100.0 * (self.v_correct / max(1, self.trial))
        acc_a = 100.0 * (self.a_correct / max(1, self.trial))
        summary = self.font_med.render(
            f"Done! S1: {self.v_correct}/{self.trial} ({acc_v:.1f}%)   "
            f"S2: {self.a_correct}/{self.trial} ({acc_a:.1f}%)",
            True,
            (255, 255, 255),
        )
        self.screen.blit(
            summary,
            summary.get_rect(
                center=(self.screen.get_width() // 2, self.screen.get_height() // 2)
            ),
        )
        tip = self.font_small.render(
            "Press Esc or close the window to exit.", True, (200, 200, 200)
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

        # Idle loop until exit
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
            self.clock.tick(60)


def make_beep(
    frequency: int = 880, duration_ms: int = 120, volume: float = 0.5
) -> Optional[pygame.mixer.Sound]:
    """
    Generate a sine beep as a pygame Sound. Requires numpy. Returns None if synth fails.
    """
    try:
        import numpy as np

        sample_rate = 44100
        t = np.linspace(
            0, duration_ms / 1000.0, int(sample_rate * duration_ms / 1000.0), False
        )
        wave = 0.5 * np.sin(2 * np.pi * frequency * t)
        audio = (wave * (2**15 - 1)).astype(np.int16)
        snd = pygame.sndarray.make_sound(audio)
        snd.set_volume(max(0.0, min(1.0, volume)))
        return snd
    except Exception:
        return None


class EmotionalDualNBack:
    """
    Visual = KDEF emotion class (image shown centered).
    Auditory = MAV emotion class (random wav for that emotion is played).
    Matching rule: current emotion == emotion n steps back (per modality).
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
        feedback_ms: int = 300,  # kept for compatibility; no big overlay used now
        window_size=(900, 650),
        show_debug_labels: bool = False,
    ):
        if seed is not None:
            random.seed(seed)

        self.length = length
        self.n = n
        self.stim_ms = stim_ms
        self.isi_ms = isi_ms
        self.feedback_ms = feedback_ms
        self.show_debug_labels = show_debug_labels

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
        # status boxes (no “letter box” anymore)
        self.status_img_rect = Rect(540, 140, 300, 90)
        self.status_aud_rect = Rect(540, 250, 300, 90)

        # Feedback beeps
        self.beep_double = make_beep(1100, 130, 0.6)  # double-correct
        self.beep_half = make_beep(650, 130, 0.6)  # half-correct

        # Stats
        self.t = 0
        self.v_hist = []  # emotion index history (visual)
        self.a_hist = []  # emotion index history (audio)
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
        self, rect: pygame.Rect, label: str, answered: bool, correct: bool
    ):
        # border (gray)
        pygame.draw.rect(
            self.screen, (40, 42, 48), rect, border_radius=12
        )  # subtle panel bg
        pygame.draw.rect(self.screen, (160, 160, 170), rect, width=2, border_radius=12)

        # fill only if answered: green/red with transparency
        if answered:
            fill = (40, 160, 90, 140) if correct else (180, 60, 60, 140)
            overlay = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
            overlay.fill(fill)
            self.screen.blit(overlay, rect.topleft)

        # label centered
        lbl = self.font_big.render(label, True, (235, 235, 235))
        self.screen.blit(lbl, lbl.get_rect(center=rect.center))

    def _draw_frame(
        self,
        image: pygame.Surface | None,
        *,
        answered_v: bool,
        correct_v_now: bool,
        answered_a: bool,
        correct_a_now: bool,
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

        # Status boxes (always visible)
        self._draw_status_box(self.status_img_rect, "Image", answered_v, correct_v_now)
        self._draw_status_box(self.status_aud_rect, "Audio", answered_a, correct_a_now)

        # Optional debug labels (under status boxes)
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
            # correctness “now” only shown if answered; remains from last press
            correct_v_now = False
            correct_a_now = False

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
                            # beep logic on correct press
                            if correct_v_now:
                                if answered_a and correct_a_now and self.beep_double:
                                    self.beep_double.play()
                                elif self.beep_half:
                                    self.beep_half.play()
                        elif event.key == pygame.K_2 and not answered_a:
                            answered_a = True
                            correct_a_now = gt_a
                            if correct_a_now:
                                if answered_v and correct_v_now and self.beep_double:
                                    self.beep_double.play()
                                elif self.beep_half:
                                    self.beep_half.play()

                self._draw_frame(
                    image,
                    answered_v=answered_v,
                    correct_v_now=correct_v_now,
                    answered_a=answered_a,
                    correct_a_now=correct_a_now,
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
                            if correct_v_now:
                                if answered_a and correct_a_now and self.beep_double:
                                    self.beep_double.play()
                                elif self.beep_half:
                                    self.beep_half.play()
                        elif event.key == pygame.K_2 and not answered_a:
                            answered_a = True
                            correct_a_now = gt_a
                            if correct_a_now:
                                if answered_v and correct_v_now and self.beep_double:
                                    self.beep_double.play()
                                elif self.beep_half:
                                    self.beep_half.play()

                self._draw_frame(
                    None,
                    answered_v=answered_v,
                    correct_v_now=correct_v_now,
                    answered_a=answered_a,
                    correct_a_now=correct_a_now,
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
