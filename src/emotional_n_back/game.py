import random

import pygame
from pygame import Rect

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


if __name__ == "__main__":
    game = DualNBackPygame(
        length=20,
        n=2,
        visual_items=9,
        auditory_items=8,
        repeat_probability=0.3,
        seed=42,
        stim_ms=900,
        isi_ms=500,
        show_feedback_ms=300,
    )
    game.run()
