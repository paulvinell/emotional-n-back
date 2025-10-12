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
        outlier_std_devs: Optional[float] = 3.0,
        erp_component: str = "P300",
        trial_duration_ms: Optional[int] = 4000,
        **kwargs,
    ):
        super().__init__(*args, length=None, **kwargs)
        self.erp_component = erp_component
        self.trial_duration_ms = trial_duration_ms
        self.erp_threshold: Optional[float] = None  # Set after calibration
        self.mean_amp: Optional[float] = None
        self.std_amp: Optional[float] = None
        self.mean_lat: Optional[float] = None
        self.std_lat: Optional[float] = None
        self.calibration_data = []
        self.initial_calibration_trials = initial_calibration_trials
        if recalibration_interval < 2:
            raise ValueError("recalibration_interval must be at least 2")
        self.recalibration_interval = recalibration_interval
        self.outlier_std_devs = outlier_std_devs

        # Thread-safe mechanism for ERP updates
        self.erp_updates = {}
        self.erp_lock = threading.Lock()
        self.erp_cond = threading.Condition(self.erp_lock)
        self.eeg_started = threading.Event()

        self.erp_server = OscErpServer(
            host="127.0.0.1",
            port=5005,
            fs_fallback=fs_fallback,
            on_update=self._handle_erp_update,
            eeg_started=self.eeg_started,
            components_to_calculate=[self.erp_component],
        )

        self.beep_success = make_beep(1300, 100, 0.5)
        self.beep_failure = make_beep(440, 200, 0.5)

        self.scoreable_trial_num = 0  # Only incremented after calibration

    def _draw_header(self):
        hdr = self.font_big.render(f"Trial {self.trial_num + 1}", True, (235, 235, 235))
        self.screen.blit(hdr, (24, 24))

        if self.erp_threshold is None:
            calib_text = self.font_small.render("Calibrating...", True, (255, 255, 255))
            self.screen.blit(calib_text, (24, 60))

    def _draw_scorebar(self):
        s_txt = self.font_small.render(
            f"Score: {self.score}/{self.scoreable_trial_num}", True, (200, 200, 200)
        )
        self.screen.blit(s_txt, (24, self.screen.get_height() - 30))

    def _handle_erp_update(self, update: dict):
        """Callback to receive ERP updates in a thread-safe manner."""
        if self.erp_component in update.get("component", {}):
            with self.erp_cond:
                self.erp_updates[update["code"]] = update
                self.erp_cond.notify()

    def _get_erp_update(self, event_code: str) -> Optional[dict]:
        """
        Retrieves the ERP update for a specific event.
        Waits for a short period for the result to become available.
        """
        update = None
        with self.erp_cond:
            self.erp_cond.wait_for(lambda: event_code in self.erp_updates, timeout=1.0)
            if event_code in self.erp_updates:
                update = self.erp_updates.pop(event_code)

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
            trial_start_t = pygame.time.get_ticks()
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
                amp = component.get(self.erp_component, {}).get("amp")
                lat = component.get(self.erp_component, {}).get("lat")

                if amp is not None and lat is not None:
                    # --- Reward Determination (z-score based) ---
                    if (
                        self.mean_amp is not None
                        and self.std_amp is not None
                        and self.mean_lat is not None
                        and self.std_lat is not None
                    ):
                        if self.std_amp > 1e-6 and self.std_lat > 1e-6:
                            z_amp = (amp - self.mean_amp) / self.std_amp
                            z_lat = (self.mean_lat - lat) / self.std_lat
                            avg_z = (z_amp + z_lat) / 2

                            if avg_z > 0.5:
                                reward = Reward.SUCCESS
                            elif avg_z < -0.5:
                                reward = Reward.FAILURE
                            else:
                                reward = Reward.NONE
                        else:
                            reward = Reward.NONE
                    else:
                        reward = Reward.NONE

                    self.calibration_data.append((amp, lat))

                    # --- Calibration and Recalibration ---
                    is_initial_cal = self.erp_threshold is None
                    trials_needed = (
                        self.initial_calibration_trials
                        if is_initial_cal
                        else self.recalibration_interval
                    )
                    is_calibration_time = len(self.calibration_data) >= trials_needed

                    if is_calibration_time:
                        if self.calibration_data:
                            mean_amp_cal = np.mean(
                                [d[0] for d in self.calibration_data]
                            )
                            std_amp_cal = np.std([d[0] for d in self.calibration_data])

                            if self.outlier_std_devs is not None:
                                filtered_data = [
                                    d
                                    for d in self.calibration_data
                                    if abs(d[0] - mean_amp_cal)
                                    <= self.outlier_std_devs * std_amp_cal
                                ]
                            else:
                                filtered_data = self.calibration_data

                            if (
                                len(filtered_data) < 2
                                and len(self.calibration_data) >= 2
                            ):
                                deviations = [
                                    (d, abs(d[0] - mean_amp_cal))
                                    for d in self.calibration_data
                                ]
                                deviations.sort(key=lambda x: x[1])
                                final_data = [d[0] for d in deviations[:2]]
                            else:
                                final_data = filtered_data

                            if final_data:
                                amps, lats = zip(*final_data)
                                self.mean_amp = np.mean(amps)
                                self.std_amp = np.std(amps)
                                self.mean_lat = np.mean(lats)
                                self.std_lat = np.std(lats)
                                self.erp_threshold = self.mean_amp

                                print(
                                    f"\n--- Recalibrating ---"
                                    f"\nNew {self.erp_component} Amp Threshold: {self.erp_threshold:.2f}"
                                    f"\nStats (last {len(final_data)} trials):"
                                    f"  Amp: μ={self.mean_amp:.2f}, σ={self.std_amp:.2f}"
                                    f"  Lat: μ={self.mean_lat:.2f}, σ={self.std_lat:.2f}"
                                    f"\n---------------------"
                                )
                                # Reset for the next batch
                                self.calibration_data = []

            # --- Reward sound and score update (only if not calibrating) ---
            if self.erp_threshold is not None:
                if reward == Reward.SUCCESS:
                    self.beep_success.play()
                    self.score += 1
                elif reward == Reward.FAILURE:
                    self.beep_failure.play()

            self.trial_num += 1
            if self.erp_threshold is not None:
                self.scoreable_trial_num += 1

            # --- Inter-trial Interval (ISI) with integrated feedback ---
            if self.trial_duration_ms is not None:
                elapsed_ms = pygame.time.get_ticks() - trial_start_t
                wait_ms = self.trial_duration_ms - elapsed_ms
                if wait_ms > 0:
                    isi_t0 = pygame.time.get_ticks()
                    while pygame.time.get_ticks() - isi_t0 < wait_ms:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT or (
                                event.type == pygame.KEYDOWN
                                and event.key == pygame.K_ESCAPE
                            ):
                                running = False
                        if not running:
                            break

                        self.screen.fill((20, 22, 26))
                        self._draw_header()
                        # Draw feedback overlay on stimulus
                        overlay = pygame.Surface(
                            self.stimulus_rect.size, pygame.SRCALPHA
                        )
                        if reward == Reward.SUCCESS:
                            fill = (40, 160, 90, 140)
                        elif reward == Reward.FAILURE:
                            fill = (180, 60, 60, 140)
                        else:
                            fill = (0, 0, 0, 0)
                        overlay.fill(fill)

                        self._draw_stimulus_box(image_surface)
                        self.screen.blit(overlay, self.stimulus_rect.topleft)

                        self._draw_scorebar()
                        pygame.display.flip()
                        self.clock.tick(120)
            else:
                # Fallback to original feedback + ISI logic
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
                    overlay = pygame.Surface(
                        self.stimulus_rect.size, pygame.SRCALPHA
                    )
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

                # --- 6. Inter-trial Interval (ISI) ---
                isi_t0 = pygame.time.get_ticks()
                while pygame.time.get_ticks() - isi_t0 < self.isi_ms:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or (
                            event.type == pygame.KEYDOWN
                            and event.key == pygame.K_ESCAPE
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
