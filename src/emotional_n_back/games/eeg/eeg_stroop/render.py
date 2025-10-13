
import pygame
from typing import Optional

from .reward import Reward

def draw_waiting_eeg(screen: pygame.Surface, font: pygame.font.Font):
    text = font.render("Waiting for EEG stream...", True, (255, 255, 255))
    text_rect = text.get_rect(center=screen.get_rect().center)
    screen.fill((20, 22, 26))
    screen.blit(text, text_rect)
    pygame.display.flip()

def draw_header(screen: pygame.Surface, font_big: pygame.font.Font, font_small: pygame.font.Font, trial_idx: int, calibrating: bool):
    hdr = font_big.render(f"Trial {trial_idx + 1}", True, (235, 235, 235))
    screen.blit(hdr, (24, 24))

    if calibrating:
        calib_text = font_small.render("Calibrating...", True, (255, 255, 255))
        screen.blit(calib_text, (24, 60))

def draw_scorebar(screen: pygame.Surface, font_small: pygame.font.Font, score: int, total: int):
    s_txt = font_small.render(f"Score: {score}/{total}", True, (200, 200, 200))
    screen.blit(s_txt, (24, screen.get_height() - 30))

def draw_stimulus_box(screen: pygame.Surface, rect: pygame.Rect, image_surface: Optional[pygame.Surface] = None):
    pygame.draw.rect(screen, (60, 60, 65), rect, border_radius=12)
    pygame.draw.rect(screen, (160, 160, 170), rect, width=2, border_radius=12)
    if image_surface:
        dst = image_surface.get_rect(center=rect.center)
        screen.blit(image_surface, dst)

def draw_feedback_overlay(screen: pygame.Surface, rect: pygame.Rect, reward: Reward):
    if reward == Reward.NONE:
        return
    
    overlay = pygame.Surface(rect.size, pygame.SRCALPHA)
    fill = (40, 160, 90, 140) if reward == Reward.SUCCESS else (180, 60, 60, 140)
    overlay.fill(fill)
    screen.blit(overlay, rect.topleft)
