import random
import threading
import time
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
        
        # Thread-safe mechanism for ERP updates
        self.erp_updates = {}
        self.erp_lock = threading.Lock()

        self.erp_server = OscErpServer(
            host="127.0.0.1",
            port=5005,
            fs_fallback=fs_fallback,
            on_update=self._handle_erp_update,
        )

        self.beep_success = make_beep(1300, 100, 0.5)
        self.beep_failure = make_beep(440, 200, 0.5)

    def _handle_erp_update(self, update: dict):
        """Callback to receive ERP updates in a thread-safe manner."""
        with self.erp_lock:
            self.erp_updates[update['code']] = update

    def _get_reward(self, event_code: str) -> Reward:
        """
        Determines the reward based on EEG data for a specific event.
        Waits for a short period for the ERP result to become available.
        """
        # Wait for the ERP result for the specific event_code
        update = None
        wait_start_t = time.monotonic()
        while time.monotonic() - wait_start_t < 1.0: # 1-second timeout
            with self.erp_lock:
                if event_code in self.erp_updates:
                    update = self.erp_updates.pop(event_code) # Pop to avoid reuse
                    break
            time.sleep(0.01)

        if update is None:
            print(f"No ERP update received for event: {event_code}")
            return Reward.NONE

        # TODO: Implement more sophisticated ERP analysis here.
        # This is a simple example that rewards high P300 amplitude.
        p300_amp = update.get("components", {}).get("P300", {}).get("amp")

        if p300_amp is None:
            return Reward.NONE

        if p300_amp > self.p300_threshold:
            print(f"Success! P300 amp for {event_code}: {p300_amp:.2f} > {self.p300_threshold:.2f}")
            return Reward.SUCCESS
        else:
            print(f"Failure. P300 amp for {event_code}: {p300_amp:.2f} <= {self.p300_threshold:.2f}")
            return Reward.FAILURE

    def run(self):
        self.erp_server.start()
        running = True
        while running and self.trial_num < self.length:
            # --- 1. Prepare Trial ---
            visual_sentiment = random.choice(self.sentiments)
            audio_sentiment = random.choice(self.sentiments)
            is_congruent = visual_sentiment == audio_sentiment
            # Create a unique event code for this trial
            event_code = f"{'congruent' if is_congruent else 'incongruent'}_{self.trial_num}"

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
            # Wait for the audio to finish playing
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
            # The reward is now fetched specifically for the event of this trial
            reward = self._get_reward(event_code)
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