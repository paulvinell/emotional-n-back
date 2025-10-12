import random
import threading
import time
from enum import Enum, auto
from typing import Optional

import numpy as np
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
        initial_calibration_trials: int = 10,
        recalibration_interval: int = 10,
        **kwargs,
    ):
        super().__init__(*args, length=None, **kwargs)
        self.p300_threshold: Optional[float] = None  # Set after calibration
        self.calibration_data = []
        self.initial_calibration_trials = initial_calibration_trials
        self.recalibration_interval = recalibration_interval

        # Thread-safe mechanism for ERP updates
        self.erp_updates = {}
        self.erp_lock = threading.Lock()
        self.eeg_started = threading.Event()

        self.erp_server = OscErpServer(
            host="127.0.0.1",
            port=5005,
            fs_fallback=fs_fallback,
            on_update=self._handle_erp_update,
            eeg_started=self.eeg_started,
        )

        self.beep_success = make_beep(1300, 100, 0.5)
        self.beep_failure = make_beep(440, 200, 0.5)

        self.scoreable_trial_num = 0  # Only incremented after calibration

    def _draw_header(self):
        hdr = self.font_big.render(f"Trial {self.trial_num + 1}", True, (235, 235, 235))
        self.screen.blit(hdr, (24, 24))

        if self.p300_threshold is None:
            calib_text = self.font_small.render("Calibrating...", True, (255, 255, 255))
            self.screen.blit(calib_text, (24, 60))

    def _draw_scorebar(self):
        s_txt = self.font_small.render(
            f"Score: {self.score}/{self.scoreable_trial_num}", True, (200, 200, 200)
        )
        self.screen.blit(s_txt, (24, self.screen.get_height() - 30))

    def _handle_erp_update(self, update: dict):
        """Callback to receive ERP updates in a thread-safe manner."""
        # We are only interested in P300 for reward
        if "P300" in update.get("component", {}):
            with self.erp_lock:
                self.erp_updates[update["code"]] = update

    def _get_erp_update(self, event_code: str) -> Optional[dict]:
        """
        Retrieves the ERP update for a specific event.
        Waits for a short period for the result to become available.
        """
        update = None
        wait_start_t = time.monotonic()
        while time.monotonic() - wait_start_t < 1.0:  # 1-second timeout
            with self.erp_lock:
                if event_code in self.erp_updates:
                    update = self.erp_updates.pop(event_code)  # Pop to avoid reuse
                    break
            time.sleep(0.01)

        if update is None:
            print(f"No ERP update received for event: {event_code}")

        return update

    def run(self):
        self.erp_server.start()

        # Wait for EEG stream to start
        font = pygame.font.Font(None, 48)
        text = font.render("Waiting for EEG stream...", True, (255, 255, 255))
        text_rect = text.get_rect(center=self.screen.get_rect().center)
        while not self.eeg_started.is_set():
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    self.erp_server.shutdown()
                    pygame.quit()
                    return
            self.screen.fill((20, 22, 26))
            self.screen.blit(text, text_rect)
            pygame.display.flip()
            self.clock.tick(10)

        running = True
        while running:
            # --- 1. Prepare Trial ---
            visual_sentiment = random.choice(self.sentiments)
            audio_sentiment = random.choice(self.sentiments)
            is_congruent = visual_sentiment == audio_sentiment
            # Create a unique event code for this trial
            event_code = (
                f"{'congruent' if is_congruent else 'incongruent'}_{self.trial_num}"
            )

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
            erp_update = self._get_erp_update(event_code)
            reward = Reward.NONE

            if erp_update:
                component = erp_update.get("component", {})
                p300_amp = component.get("P300", {}).get("amp")
                p300_lat = component.get("P300", {}).get("lat")

                if p300_amp is not None and p300_lat is not None:
                    # --- Reward Determination (if we have a threshold) ---
                    if self.p300_threshold is not None:
                        if p300_amp > self.p300_threshold:
                            print(
                                f"Success! P300 amp for {event_code}: {p300_amp:.2f} > {self.p300_threshold:.2f}"
                            )
                            reward = Reward.SUCCESS
                        else:
                            print(
                                f"Failure. P300 amp for {event_code}: {p300_amp:.2f} <= {self.p300_threshold:.2f}"
                            )
                            reward = Reward.FAILURE

                    self.calibration_data.append((p300_amp, p300_lat))

                    # --- Calibration and Recalibration ---
                    is_initial_cal = self.p300_threshold is None
                    trials_needed = (
                        self.initial_calibration_trials
                        if is_initial_cal
                        else self.recalibration_interval
                    )
                    is_calibration_time = len(self.calibration_data) >= trials_needed

                    if is_calibration_time:
                        if self.calibration_data:
                            amps, lats = zip(*self.calibration_data)
                            mean_amp = np.mean(amps)
                            std_amp = np.std(amps)
                            mean_lat = np.mean(lats)
                            std_lat = np.std(lats)

                            self.p300_threshold = mean_amp  # Set threshold to mean
                            print(
                                f"\n--- Recalibrating ---"
                                f"\nNew P300 Amp Threshold: {self.p300_threshold:.2f}"
                                f"\nStats (last {len(self.calibration_data)} trials):"
                                f"  Amp: μ={mean_amp:.2f}, σ={std_amp:.2f}"
                                f"  Lat: μ={mean_lat:.2f}, σ={std_lat:.2f}"
                                f"\n---------------------"
                            )
                            # Reset for the next batch
                            self.calibration_data = []

            # --- Reward sound and score update (only if not calibrating) ---
            if self.p300_threshold is not None:
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
            if self.p300_threshold is not None:
                self.scoreable_trial_num += 1

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
