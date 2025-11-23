from typing import Optional

import pygame

from .reward import Reward
from dataclasses import dataclass
from .reward import Reward
from .state import GameState


@dataclass
class RenderState:
    state: GameState
    display_text: str
    score: int
    scoreable_trial_num: int
    is_calibrating: bool
    stimulus_rect: pygame.Rect
    image_surface: Optional[pygame.Surface]
    reward: Reward
    show_fs: bool
    fs: float
    continuous_fs_est: float
    initial_calibration_trials: int


class GameRenderer:
    def __init__(self, window_size=(900, 650)):
        pygame.init()
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        pygame.font.init()
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("EEG Stroop Game")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        self.large_font = pygame.font.SysFont(None, 48)

    def render_game(self, render_state: RenderState):
        self.screen.fill((20, 22, 26))

        if render_state.state == GameState.WAIT_EEG:
            self.draw_waiting_eeg()
        elif render_state.state == GameState.INTRO:
            self.draw_intro(render_state)
        elif render_state.state in [
            GameState.STIMULUS,
            GameState.RESPONSE,
            GameState.FEEDBACK,
        ]:
            self.draw_trial(render_state)
        elif render_state.state == GameState.FINAL_SCREEN:
            self.draw_final_screen(render_state)
        pygame.display.flip()

    def draw_trial(self, state: RenderState):
        self.screen.fill((20, 22, 26))
        self.draw_header(state)
        self.draw_stimulus_box(state.stimulus_rect, state.image_surface)
        if state.reward is not None:
            self.draw_feedback_overlay(state.stimulus_rect, state.reward)
        self.draw_scorebar(state.score, state.scoreable_trial_num)
        self.draw_fs(state)

    def draw_intro(self, state: RenderState):
        self.screen.fill((20, 22, 26))
        self.draw_header(state)
        self.draw_stimulus_box(state.stimulus_rect)
        self.draw_scorebar(state.score, state.scoreable_trial_num)
        self.draw_fs(state)
        pygame.display.flip()

    def draw_waiting_eeg(self):
        text = self.large_font.render("Waiting for EEG stream...", True, (255, 255, 255))
        text_rect = text.get_rect(center=self.screen.get_rect().center)
        self.screen.fill((20, 22, 26))
        self.screen.blit(text, text_rect)
        pygame.display.flip()

    def draw_header(self, state: RenderState):
        hdr = self.large_font.render(state.display_text, True, (235, 235, 235))
        self.screen.blit(hdr, (24, 24))

    def draw_scorebar(self, score: int, total: int):
        s_txt = self.font.render(f"Score: {score}/{total}", True, (200, 200, 200))
        self.screen.blit(s_txt, (24, self.screen.get_height() - 30))

    def draw_fs(self, state: RenderState):
        if not state.show_fs:
            return
        fs = state.continuous_fs_est
        fs_text = self.font.render(f"fs: {fs:.1f} Hz", True, (200, 200, 200))
        text_rect = fs_text.get_rect()
        text_rect.bottomright = self.screen.get_rect().bottomright
        text_rect.x -= 10
        text_rect.y -= 10
        self.screen.blit(fs_text, text_rect)

    def draw_stimulus_box(self, rect: pygame.Rect, image_surface: Optional[pygame.Surface] = None):
        pygame.draw.rect(self.screen, (60, 60, 65), rect, border_radius=12)
        pygame.draw.rect(self.screen, (160, 160, 170), rect, width=2, border_radius=12)
        if image_surface:
            dst = image_surface.get_rect(center=rect.center)
            self.screen.blit(image_surface, dst)

    def draw_feedback_overlay(self, rect: pygame.Rect, reward: Reward):
        if reward == Reward.NONE:
            return

        overlay = pygame.Surface(rect.size, pygame.SRCALPHA)
        fill = (40, 160, 90, 140) if reward == Reward.SUCCESS else (180, 60, 60, 140)
        overlay.fill(fill)
        self.screen.blit(overlay, rect.topleft)

    def draw_final_screen(self, state: RenderState):
        self.screen.fill((20, 22, 26))
        final_trials = max(1, state.trial_num)
        acc = 100.0 * (state.score / final_trials)
        summary = f"Done! Score: {state.score}/{final_trials} ({acc:.1f}%)"
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
