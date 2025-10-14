import pygame
from typing import Optional
from .reward import Reward

from .state import GameState

class GameRenderer:
    def __init__(self, window_size=(900, 650)):
        pygame.init()
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("EEG Stroop Game")
        self.clock = pygame.time.Clock()
        self.font_big = pygame.font.SysFont(None, 48)
        self.font_small = pygame.font.SysFont(None, 24)

    def render_game(self, game):
        self.screen.fill((20, 22, 26))
        if game.state == GameState.WAIT_EEG:
            self.draw_waiting_eeg()
        elif game.state == GameState.INTRO:
            self.draw_intro(game.get_trial_data())
        elif game.state in [GameState.STIMULUS, GameState.RESPONSE, GameState.FEEDBACK]:
            self.draw_trial(game.get_trial_data())
        elif game.state == GameState.FINAL_SCREEN:
            self.draw_final_screen(game.get_final_screen_data())
        pygame.display.flip()

    def draw_trial(self, trial_data):
        self.screen.fill((20, 22, 26))
        self.draw_header(trial_data["trial_num"], trial_data["is_calibrating"])
        self.draw_stimulus_box(trial_data["stimulus_rect"], trial_data.get("image_surface"))
        if trial_data.get("reward") is not None:
            self.draw_feedback_overlay(trial_data["stimulus_rect"], trial_data["reward"])
        self.draw_scorebar(trial_data["score"], trial_data["scoreable_trial_num"])
        pygame.display.flip()

    def draw_intro(self, trial_data):
        self.screen.fill((20, 22, 26))
        self.draw_header(trial_data["trial_num"], trial_data["is_calibrating"])
        self.draw_stimulus_box(trial_data["stimulus_rect"])
        self.draw_scorebar(trial_data["score"], trial_data["scoreable_trial_num"])
        pygame.display.flip()

    def draw_waiting_eeg(self):
        text = self.font_big.render("Waiting for EEG stream...", True, (255, 255, 255))
        text_rect = text.get_rect(center=self.screen.get_rect().center)
        self.screen.fill((20, 22, 26))
        self.screen.blit(text, text_rect)
        pygame.display.flip()

    def draw_header(self, trial_idx: int, calibrating: bool):
        hdr = self.font_big.render(f"Trial {trial_idx + 1}", True, (235, 235, 235))
        self.screen.blit(hdr, (24, 24))

        if calibrating:
            calib_text = self.font_small.render("Calibrating...", True, (255, 255, 255))
            self.screen.blit(calib_text, (24, 60))

    def draw_scorebar(self, score: int, total: int):
        s_txt = self.font_small.render(f"Score: {score}/{total}", True, (200, 200, 200))
        self.screen.blit(s_txt, (24, self.screen.get_height() - 30))

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
        
    def draw_final_screen(self, final_screen_data):
        self.screen.fill((20, 22, 26))
        final_trials = max(1, final_screen_data['trial_num'])
        acc = 100.0 * (final_screen_data['score'] / final_trials)
        summary = f"Done! Score: {final_screen_data['score']}/{final_trials} ({acc:.1f}%)"
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