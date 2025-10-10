import random
from enum import Enum, auto
from typing import Optional

import pygame

from emotional_n_back.eeg.erp import OscErpServer
from emotional_n_back.stroop import SentimentStroopGame, make_beep


class Reward(Enum):
    SUCCESS = auto()
    FAILURE = auto()
    NONE = auto()


class EEGStroopGame(SentimentStroopGame):
    """
    An EEG-integrated version of the sentiment Stroop game.
    - No user input is required.
    - After the audio stimulus, a reward is determined based on EEG data.
    - Audio feedback (success/failure) is provided based on the reward.
    """

    def __init__(
        self,
        *args,
        fs_fallback: float = 256.0,
        p300_threshold: float = 2.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.p300_threshold = p300_threshold
        self.last_erp_update = None
        self.erp_server = OscErpServer(
            host="127.0.0.1",
            port=5005,
            fs_fallback=fs_fallback,
            on_update=self._handle_erp_update,
        )

        self.beep_success = make_beep(1300, 100, 0.5)
        self.beep_failure = make_beep(440, 200, 0.5)

    def _handle_erp_update(self, update: dict):
        """Callback to receive ERP updates."""
        # This is where you could add more complex logic, e.g. storing
        # a history of ERPs for different conditions.
        self.last_erp_update = update

    def _get_reward(self) -> Reward:
        """
        Determines the reward based on EEG data.
        This is a placeholder for more sophisticated ERP analysis.
        """
        # TODO: Implement more sophisticated ERP analysis here.
        # This is a simple example that rewards high P300 amplitude.
        if self.last_erp_update is None:
            return Reward.NONE

        p300_amp = self.last_erp_update.get("components", {}).get("P300", {}).get("amp")

        if p300_amp is None:
            return Reward.NONE

        if p300_amp > self.p300_threshold:
            print(f"Success! P300 amp: {p300_amp:.2f} > {self.p300_threshold:.2f}")
            return Reward.SUCCESS
        else:
            print(f"Failure. P300 amp: {p300_amp:.2f} <= {self.p300_threshold:.2f}")
            return Reward.FAILURE

    def run(self):
        self.erp_server.start()
        running = True
        while running and self.trial_num < self.length:
            # --- 1. Prepare Trial ---
            visual_sentiment = random.choice(self.sentiments)
            audio_sentiment = random.choice(self.sentiments)
            is_congruent = visual_sentiment == audio_sentiment
            event_code = "congruent" if is_congruent else "incongruent"

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
            # Ingest event directly
            self.erp_server.ingest_event(event_code)
            print(f"Ingested event: {event_code}")

            response_t0 = pygame.time.get_ticks()
            # Wait for the audio to finish playing before getting the reward
            while pygame.mixer.get_busy() and (
                pygame.time.get_ticks() - response_t0 < self.response_window_ms
            ):
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
            # The reward is based on the last ERP update, which is handled
            # by the on_update callback in a separate thread.
            # We add a small delay to allow the ERP to be processed.
            pygame.time.wait(500) # Wait for LPP to be captured
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
                    fill = (0, 0, 0, 0)  # No feedback
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
                self._draw_stimulus_box()  # Blank box
                self._draw_scorebar()
                pygame.display.flip()
                self.clock.tick(120)
            if not running:
                break

        self.erp_server.shutdown()
        if running:
            self.show_final_screen()
        pygame.quit()
