# nback_refactor.py
import math
import random
from abc import ABC, abstractmethod
from typing import Iterable, Optional

import numpy as np
import pygame
from pygame import Rect

from emotional_n_back.data import KDEFLoader, MAVLoader
from emotional_n_back.nback import NBackSequence


# -------------------- Utilities --------------------
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


def nback_match(history: list[int], n: int) -> bool:
    """True iff last value equals the one n steps back in the shown stream."""
    return len(history) > n and history[-1] == history[-(n + 1)]


# -------------------- Abstract Modality --------------------
class Modality(ABC):
    """
    Encapsulates one modality:
      - label & key binding
      - stimulus selection / preloading
      - drawing its own status panel
      - per-trial state (answered?, correct?, subtitle?)
    The game drives: next_stimulus -> (stim phase/isi) -> score.
    """

    def __init__(self, label: str, trigger_key: int):
        self.label = label
        self.trigger_key = trigger_key
        # Per-trial state
        self.answered = False
        self.correct_on_press = False
        self.subtitle: Optional[str] = None
        # Ground-truth for current trial (computed by Base game)
        self.gt_is_match_nback = False
        # The “truth_prev” from NBackSequence (same-as-previous step, for debug)
        self.truth_prev: Optional[bool] = None
        # UI
        self.panel_rect: Optional[Rect] = None

    # ---- lifecycle ----
    def reset_trial_state(self):
        self.answered = False
        self.correct_on_press = False
        self.subtitle = None
        self.gt_is_match_nback = False
        self.truth_prev = None

    @abstractmethod
    def next_stimulus(self, idx_value: int) -> None:
        """Prepare concrete stimulus for this trial (choose image/audio)."""

    @abstractmethod
    def play_stimulus(self) -> None:
        """Kick off any immediate playback (e.g., audio)."""

    @abstractmethod
    def draw_stimulus(self, screen: pygame.Surface, rect: Rect) -> None:
        """Draw the main stimulus (visual draws image; audio draws nothing)."""

    def try_answer(self) -> bool:
        """
        Register a single press. Returns True if this was the first press.
        The correctness is evaluated immediately vs current gt.
        """
        if self.answered:
            return False
        self.answered = True
        self.correct_on_press = self.gt_is_match_nback
        self.subtitle = self.build_subtitle()
        return True

    @abstractmethod
    def build_subtitle(self) -> Optional[str]:
        """E.g. 'This was: <emotion>' after press, persistent until next trial."""

    def current_debug(self) -> dict[str, str]:
        """Optional debug info (like emotion names, truth_prev)."""
        return {}

    # Panel rendering is shared; subclasses may override for custom visuals if needed.
    def draw_panel(self, screen: pygame.Surface, font_big, font_small):
        assert self.panel_rect is not None, "panel_rect not set via layout"
        rect = self.panel_rect

        # Panel + border
        pygame.draw.rect(screen, (40, 42, 48), rect, border_radius=12)
        pygame.draw.rect(screen, (160, 160, 170), rect, width=2, border_radius=12)

        # Fill once answered
        if self.answered:
            fill = (40, 160, 90, 140) if self.correct_on_press else (180, 60, 60, 140)
            overlay = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
            overlay.fill(fill)
            screen.blit(overlay, rect.topleft)

        # Main label
        lbl = font_big.render(self.label, True, (235, 235, 235))
        # If subtitle exists (help mode), position the label higher
        if self.subtitle:
            lbl_rect = lbl.get_rect(center=(rect.centerx, rect.y + rect.h * 0.40))
            screen.blit(lbl, lbl_rect)
            sub = font_small.render(self.subtitle, True, (235, 235, 235))
            sub_rect = sub.get_rect(center=(rect.centerx, rect.y + rect.h * 0.72))
            screen.blit(sub, sub_rect)
        else:
            lbl_rect = lbl.get_rect(center=rect.center)
            screen.blit(lbl, lbl_rect)


# -------------------- Visual Modality --------------------
class VisualModality(Modality):
    def __init__(self, kdef: KDEFLoader, trigger_key: int):
        super().__init__("Image", trigger_key)
        self.kdef = kdef
        self.current_emotion_name: Optional[str] = None
        self.current_image_surface: Optional[pygame.Surface] = None
        self._img_cache: dict[str, pygame.Surface] = {}

    def _load_fit(self, path: str, box: Rect) -> pygame.Surface:
        if path in self._img_cache:
            return self._img_cache[path]
        img = pygame.image.load(path).convert_alpha()
        iw, ih = img.get_width(), img.get_height()
        scale = min(box.w / iw, box.h / ih)
        surf = pygame.transform.smoothscale(img, (int(iw * scale), int(ih * scale)))
        self._img_cache[path] = surf
        return surf

    def next_stimulus(self, idx_value: int) -> None:
        # Map class index -> emotion name (0-based in loader, our idx_value is 1-based)
        self.current_emotion_name = self.kdef.get_emotion(idx_value - 1)
        img_path = self.kdef.get_random_image(self.current_emotion_name)
        # Surface is scaled at draw time (needs target rect); keep raw path cached
        self.current_image_surface = None  # will set in draw_stimulus using box

        # stash the chosen path so draw can load/scale once we know the rect
        self._current_image_path = str(img_path)

    def play_stimulus(self) -> None:
        # Nothing to “play” immediately for visuals; drawing happens in draw_stimulus
        pass

    def draw_stimulus(self, screen: pygame.Surface, rect: Rect) -> None:
        if self.current_image_surface is None:
            self.current_image_surface = self._load_fit(self._current_image_path, rect)
        if self.current_image_surface:
            dst = self.current_image_surface.get_rect(center=rect.center)
            screen.blit(self.current_image_surface, dst)

    def build_subtitle(self) -> Optional[str]:
        return (
            f"This was: {self.current_emotion_name}"
            if self.current_emotion_name
            else None
        )

    def current_debug(self) -> dict[str, str]:
        return {
            "Image emotion": self.current_emotion_name or "",
            "Same-as-previous (seq truth)": "Yes" if self.truth_prev else "No",
        }


# -------------------- Audio Modality --------------------
class AudioModality(Modality):
    def __init__(self, mav: MAVLoader, trigger_key: int):
        super().__init__("Audio", trigger_key)
        self.mav = mav
        self.current_emotion_name: Optional[str] = None
        self.current_sound: Optional[pygame.mixer.Sound] = None
        self._current_audio_path: Optional[str] = None

    def next_stimulus(self, idx_value: int) -> None:
        self.current_emotion_name = self.mav.get_emotion(idx_value - 1)
        aud_path = self.mav.get_random_audio(self.current_emotion_name)
        try:
            self.current_sound = pygame.mixer.Sound(str(aud_path))
        except Exception:
            self.current_sound = None
        self._current_audio_path = str(aud_path)

    def play_stimulus(self) -> None:
        if self.current_sound:
            self.current_sound.play()

    def draw_stimulus(self, screen: pygame.Surface, rect: Rect) -> None:
        # No explicit audio drawing; leave area blank or let game show placeholder if desired.
        pass

    def build_subtitle(self) -> Optional[str]:
        return (
            f"This was: {self.current_emotion_name}"
            if self.current_emotion_name
            else None
        )

    def current_debug(self) -> dict[str, str]:
        return {
            "Audio emotion": self.current_emotion_name or "",
            "Same-as-previous (seq truth)": "Yes" if self.truth_prev else "No",
        }


# -------------------- Base Game --------------------
class BaseNBackGame(ABC):
    """
    Core loop/timing/scoring. Subclasses specify:
      - which modalities to include (build_modalities)
      - layout of panels & stimulus area (layout_ui)
      - help text & key bindings
    """

    def __init__(
        self,
        *,
        length: int = 30,
        n: int = 2,
        repeat_probability: float = 0.25,
        seed: Optional[int] = None,
        stim_ms: int = 1000,
        isi_ms: int = 500,
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
        self.show_debug_labels = show_debug_labels
        self.show_help_labels = show_help_labels

        # Pygame
        pygame.init()
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption(self.window_title())
        self.clock = pygame.time.Clock()
        self.font_big = pygame.font.SysFont(None, 48)
        self.font_small = pygame.font.SysFont(None, 24)

        # Beeps
        self.beep_double = make_beep(1100, 130, 0.6)
        self.beep_half = make_beep(650, 130, 0.6)

        # Build modalities (and datasets internally if needed)
        self.modalities: list[Modality] = self.build_modalities()
        assert len(self.modalities) in (1, 2), "Only single or dual supported"

        # Layout rects (panel rects assigned here)
        self.stimulus_rect, panel_rects = self.layout_ui(self.screen.get_size())
        for mod, rect in zip(self.modalities, panel_rects):
            mod.panel_rect = rect

        # Sequences per modality (we also keep history to check n-back truth)
        self.seq_values: list[list[tuple[int, bool]]] = []
        self.histories: list[list[int]] = [[] for _ in self.modalities]
        for i, mod in enumerate(self.modalities):
            distinct = self.distinct_items_for(mod)
            seq_iter = NBackSequence(
                self.length,
                self.n,
                repeat_probability=repeat_probability,
                distinct_items=distinct,
            )
            # Materialize sequence as [(value, truth_prev), ...]
            seq = [
                (v, t) if isinstance((v, t), tuple) else (v, False)
                for (v, t) in seq_iter
            ]
            self.seq_values.append(seq)

        # Stats
        self.t = 0
        self.correct_counts = [0 for _ in self.modalities]

    # ---- abstract/overridable ----
    @abstractmethod
    def build_modalities(self) -> list[Modality]: ...

    @abstractmethod
    def distinct_items_for(self, modality: Modality) -> int: ...

    @abstractmethod
    def layout_ui(self, window_size: tuple[int, int]) -> tuple[Rect, list[Rect]]:
        """Return (stimulus_rect, panel_rects)."""

    @abstractmethod
    def window_title(self) -> str: ...

    @abstractmethod
    def help_text(self) -> str: ...

    # ---- rendering helpers ----
    def _draw_header(self):
        hdr = self.font_big.render(
            f"Trial {self.t + 1}/{self.length}   n={self.n}", True, (235, 235, 235)
        )
        self.screen.blit(hdr, (24, 24))
        tip = self.font_small.render(self.help_text(), True, (210, 210, 210))
        self.screen.blit(tip, (24, 60))

    def _draw_panels(self):
        for mod in self.modalities:
            mod.draw_panel(self.screen, self.font_big, self.font_small)

    def _draw_debug(self):
        if not self.show_debug_labels:
            return
        y = self.stimulus_rect.bottom + 10
        for mod in self.modalities:
            for k, v in mod.current_debug().items():
                s = self.font_small.render(
                    f"{mod.label} {k}: {v}", True, (220, 220, 220)
                )
                self.screen.blit(s, (24, y))
                y += 18

    def _draw_scorebar(self):
        parts = []
        for i, mod in enumerate(self.modalities):
            parts.append(f"{mod.label}: {self.correct_counts[i]}/{self.t}")
        s_txt = self.font_small.render("   ".join(parts), True, (200, 200, 200))
        self.screen.blit(s_txt, (24, self.screen.get_height() - 30))

    # ---- input routing ----
    def _handle_keydown(self, key: int):
        # Route to the modality that owns this key
        for i, mod in enumerate(self.modalities):
            if key == mod.trigger_key and not mod.answered:
                first = mod.try_answer()
                if first and mod.correct_on_press:
                    # If another modality already answered correctly => double beep
                    other_correct = any(
                        m is not mod and m.answered and m.correct_on_press
                        for m in self.modalities
                    )
                    if other_correct and self.beep_double:
                        self.beep_double.play()
                    elif self.beep_half:
                        self.beep_half.play()

    # ---- main loop ----
    def run(self):
        running = True
        while running and self.t < self.length:
            # Fetch sequence step for each modality (value + truth_prev from generator)
            for i, mod in enumerate(self.modalities):
                value, truth_prev = self.seq_values[i][self.t]
                mod.reset_trial_state()
                mod.truth_prev = truth_prev  # from NBackSequence (same-as-previous)
                mod.next_stimulus(value)
                # Append for true n-back ground truth
                self.histories[i].append(value)
                mod.gt_is_match_nback = nback_match(self.histories[i], self.n)

            # Stimulus phase
            for mod in self.modalities:
                mod.play_stimulus()

            stim_t0 = pygame.time.get_ticks()
            while pygame.time.get_ticks() - stim_t0 < self.stim_ms:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        else:
                            self._handle_keydown(event.key)

                # Frame
                self.screen.fill((20, 22, 26))
                self._draw_header()
                # Stimulus drawing (visual shows image; audio draws nothing)
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
                for mod in self.modalities:
                    mod.draw_stimulus(self.screen, self.stimulus_rect)
                self._draw_panels()
                self._draw_debug()
                self._draw_scorebar()
                pygame.display.flip()
                self.clock.tick(120)

            # ISI phase (blank; still capture input)
            isi_t0 = pygame.time.get_ticks()
            while pygame.time.get_ticks() - isi_t0 < self.isi_ms:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        else:
                            self._handle_keydown(event.key)

                self.screen.fill((20, 22, 26))
                self._draw_header()
                # Blank stimulus box
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
                self._draw_panels()
                self._draw_debug()
                self._draw_scorebar()
                pygame.display.flip()
                self.clock.tick(120)

            # Score each modality: pressed == n-back-match
            for i, mod in enumerate(self.modalities):
                pressed = mod.answered
                correct = pressed == mod.gt_is_match_nback
                if correct:
                    self.correct_counts[i] += 1

            self.t += 1

        # Final screen
        self.screen.fill((20, 22, 26))
        parts = []
        for i, mod in enumerate(self.modalities):
            acc = 100.0 * (self.correct_counts[i] / max(1, self.t))
            parts.append(f"{mod.label}: {self.correct_counts[i]}/{self.t} ({acc:.1f}%)")
        summary = "Done!  " + "   ".join(parts)
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


# -------------------- Concrete Games --------------------
class VisualNBackGame(BaseNBackGame):
    def __init__(self, *, window_size=(900, 650), **kwargs):
        self._kdef = KDEFLoader()
        super().__init__(window_size=window_size, **kwargs)

    def build_modalities(self) -> list[Modality]:
        # Single mode uses key "1"
        return [VisualModality(self._kdef, pygame.K_1)]

    def distinct_items_for(self, modality: Modality) -> int:
        return len(self._kdef.emotions)

    def layout_ui(self, window_size: tuple[int, int]) -> tuple[Rect, list[Rect]]:
        W, H = window_size
        stim = Rect(W // 2 - 220, 100, 440, 440)  # centered larger image
        panel = Rect(W // 2 - 160, stim.bottom + 20, 320, 90)
        return stim, [panel]

    def window_title(self) -> str:
        return "Emotional N-Back — Visual Only"

    def help_text(self) -> str:
        return "Press 1 for Image match (n-back)."


class AudioNBackGame(BaseNBackGame):
    def __init__(self, *, window_size=(900, 650), **kwargs):
        self._mav = MAVLoader()
        super().__init__(window_size=window_size, **kwargs)

    def build_modalities(self) -> list[Modality]:
        # Single mode uses key "1"
        return [AudioModality(self._mav, pygame.K_1)]

    def distinct_items_for(self, modality: Modality) -> int:
        return len(self._mav.emotions)

    def layout_ui(self, window_size: tuple[int, int]) -> tuple[Rect, list[Rect]]:
        W, H = window_size
        stim = Rect(
            W // 2 - 200, 140, 400, 220
        )  # placeholder box (audio has no visual)
        panel = Rect(W // 2 - 160, stim.bottom + 40, 320, 90)
        return stim, [panel]

    def window_title(self) -> str:
        return "Emotional N-Back — Audio Only"

    def help_text(self) -> str:
        return "Press 1 for Audio match (n-back)."


class EmotionalDualNBack(BaseNBackGame):
    def __init__(self, *, window_size=(1000, 650), **kwargs):
        self._kdef = KDEFLoader()
        self._mav = MAVLoader()
        super().__init__(window_size=window_size, **kwargs)

    def build_modalities(self) -> list[Modality]:
        # Dual mode: 1 -> Image, 2 -> Audio
        return [
            VisualModality(self._kdef, pygame.K_1),
            AudioModality(self._mav, pygame.K_2),
        ]

    def distinct_items_for(self, modality: Modality) -> int:
        if isinstance(modality, VisualModality):
            return len(self._kdef.emotions)
        return len(self._mav.emotions)

    def layout_ui(self, window_size: tuple[int, int]) -> tuple[Rect, list[Rect]]:
        W, H = window_size
        stim = Rect(80, 100, 420, 420)  # visual area on the left
        right_x = stim.right + 40
        panel_img = Rect(right_x, 140, 360, 100)
        panel_aud = Rect(right_x, 260, 360, 100)
        return stim, [panel_img, panel_aud]

    def window_title(self) -> str:
        return "Emotional Dual N-Back"

    def help_text(self) -> str:
        return "Press 1 for Image match, 2 for Audio match (independent)."
