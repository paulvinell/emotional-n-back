import random
import threading
from collections import deque
from enum import Enum, auto

import pygame
from pythonosc import dispatcher, osc_server

from emotional_n_back.stroop import SentimentStroopGame, make_beep

class Reward(Enum):
    SUCCESS = auto()
    FAILURE = auto()
    NONE = auto()

class OSCReader:
    def __init__(self, ip="127.0.0.1", port=5005, buffer_size=256 * 2):
        self.ip = ip
        self.port = port
        self.buffer = deque(maxlen=buffer_size)
        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.map("/*", self._handler)
        self.server = osc_server.ThreadingOSCUDPServer((self.ip, self.port), self.dispatcher)
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True

    def _handler(self, address, *args):
        # In the future, we might want to do more complex processing here
        # For now, just append the first argument to the buffer
        if args:
            self.buffer.append(args[0])

    def start(self):
        self.server_thread.start()
        print(f"Serving on {self.server.server_address}")

    def stop(self):
        self.server.shutdown()

    def get_buffer(self):
        return list(self.buffer)

class EEGStroopGame(SentimentStroopGame):
    """
    An EEG-integrated version of the sentiment Stroop game.
    - No user input is required.
    - After the audio stimulus, a reward is determined based on EEG data.
    - Audio feedback (success/failure) is provided based on the reward.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.osc_reader = OSCReader()
        self.beep_success = make_beep(1300, 100, 0.5)
        self.beep_failure = make_beep(440, 200, 0.5)

    def _get_reward(self) -> Reward:
        """
        Determines the reward based on EEG data.
        This is a placeholder for the actual ERP analysis.
        """
        # TODO: Implement ERP analysis here.
        # The analysis should be based on the data in self.osc_reader.get_buffer()
        # For now, we return a random reward.
        return random.choice([Reward.SUCCESS, Reward.FAILURE, Reward.NONE])

    def run(self):
        self.osc_reader.start()
        running = True
        while running and self.trial_num < self.length:
            # --- 1. Prepare Trial ---
            visual_sentiment = random.choice(self.sentiments)
            audio_sentiment = random.choice(self.sentiments)

            image_path = self.kdef_loader.get_random_image(visual_sentiment)
            image_surface = self._load_fit_image(str(image_path), self.stimulus_rect)

            audio_path = self.mav_loader.get_random_audio(audio_sentiment)
            audio_sound = pygame.mixer.Sound(str(audio_path))

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
            # Wait for the audio to finish playing before getting the reward
            while pygame.mixer.get_busy() and (pygame.time.get_ticks() - response_t0 < self.response_window_ms):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                    ):
                        running = False
                if not running:
                    break
                self.clock.tick(120)
            
            if not running:
                break

            # --- 4. Reward Phase ---
            reward = self._get_reward()
            if reward == Reward.SUCCESS:
                self.beep_success.play()
                self.score += 1
            elif reward == Reward.FAILURE:
                self.beep_failure.play()

            # --- 5. Feedback Phase ---
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
                overlay = pygame.Surface(self.stimulus_rect.size, pygame.SRCALPHA)
                if reward == Reward.SUCCESS:
                    fill = (40, 160, 90, 140)
                elif reward == Reward.FAILURE:
                    fill = (180, 60, 60, 140)
                else:
                    fill = (0,0,0,0) # No feedback
                overlay.fill(fill)

                self._draw_stimulus_box(image_surface)
                self.screen.blit(overlay, self.stimulus_rect.topleft)
                self._draw_scorebar()
                pygame.display.flip()
                self.clock.tick(120)
            if not running:
                break

            self.trial_num += 1

            # --- 6. Inter-trial Interval (ISI) ---
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

        self.osc_reader.stop()
        if running:
            self.show_final_screen()
        pygame.quit()
