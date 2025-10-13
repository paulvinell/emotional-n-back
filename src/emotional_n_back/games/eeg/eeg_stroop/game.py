import random
from typing import Optional

import pygame

from emotional_n_back.games.regular.stroop import SentimentStroopGame
from .erp_adapter import ErpAdapter
from .reward import Calibration, ZScorePolicy, Reward, Stats
from . import render
from . import stimuli


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
        self.initial_calibration_trials = initial_calibration_trials
        if recalibration_interval < 2:
            raise ValueError("recalibration_interval must be at least 2")
        self.recalibration_interval = recalibration_interval
        self.outlier_std_devs = outlier_std_devs

        self.erp_adapter = ErpAdapter(
            erp_component=self.erp_component,
            host="127.0.0.1",
            port=5005,
            fs_fallback=fs_fallback,
        )

        self.calibration = Calibration()
        self.reward_policy = ZScorePolicy()
        self.stats: Optional[Stats] = None

        self.beep_success = stimuli.make_beep(1300, 100, 0.5)
        self.beep_failure = stimuli.make_beep(440, 200, 0.5)

        self.scoreable_trial_num = 0  # Only incremented after calibration

    def run(self):
        self.erp_adapter.start()

        # Wait for EEG stream to start
        font = pygame.font.Font(None, 48)
        while not self.erp_adapter.eeg_started.is_set():
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    self.erp_adapter.shutdown()
                    pygame.quit()
                    return
            render.draw_waiting_eeg(self.screen, font)
            self.clock.tick(10)

        running = True
        while running:
            trial_start_t = pygame.time.get_ticks()
            # --- 1. Prepare Trial ---
            visual_sentiment = random.choice(self.sentiments)
            audio_sentiment = random.choice(self.sentiments)
            # Create a unique event code for this trial
            event_code = f"trial_{self.trial_num}"

            image_path = self.kdef_loader.get_random_image(visual_sentiment)
            image_surface = stimuli.load_fit_image(str(image_path), self.stimulus_rect, self._img_cache)

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
                render.draw_header(self.screen, self.font_big, self.font_small, self.trial_num, self.stats is None)
                render.draw_stimulus_box(self.screen, self.stimulus_rect, image_surface)
                render.draw_scorebar(self.screen, self.font_small, self.score, self.scoreable_trial_num)
                pygame.display.flip()
                self.clock.tick(120)
            if not running:
                break

            # --- 3. Response Phase (Audio plays) ---
            audio_sound.play()
            # Ingest event directly
            self.erp_adapter.ingest_event(event_code)
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
            erp_update = None
            reward_phase_start_t = pygame.time.get_ticks()
            while pygame.time.get_ticks() - reward_phase_start_t < 1000: # poll for 1 second
                erp_update = self.erp_adapter.poll_update(event_code)
                if erp_update:
                    break
                pygame.time.wait(10)

            reward = Reward.NONE

            if erp_update and erp_update.amp is not None and erp_update.lat is not None:
                self.calibration.update(erp_update.amp, erp_update.lat)
                reward = self.reward_policy.decide(erp_update.amp, erp_update.lat, self.stats)

                if self.calibration.ready(
                    self.initial_calibration_trials, self.recalibration_interval, self.stats is not None
                ):
                    self.stats = self.calibration.compute(self.outlier_std_devs)
                    self.calibration.reset_batch()
                    if self.stats:
                        print(
                            f"\n--- Recalibrating ---"
                            f"\nNew {self.erp_component} Amp Threshold: {self.stats.threshold_amp:.2f}"
                            f"\nStats (last {len(self.calibration.calibration_data)} trials):"
                            f"  Amp: μ={self.stats.mean_amp:.2f}, σ={self.stats.std_amp:.2f}"
                            f"  Lat: μ={self.stats.mean_lat:.2f}, σ={self.stats.std_lat:.2f}"
                            f"\n---------------------"
                        )

            # --- Reward sound and score update (only if not calibrating) ---
            if self.stats is not None:
                if reward == Reward.SUCCESS:
                    self.beep_success.play()
                    self.score += 1
                elif reward == Reward.FAILURE:
                    self.beep_failure.play()

            self.trial_num += 1
            if self.stats is not None:
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
                        render.draw_header(self.screen, self.font_big, self.font_small, self.trial_num, self.stats is None)
                        render.draw_stimulus_box(self.screen, self.stimulus_rect, image_surface)
                        render.draw_feedback_overlay(self.screen, self.stimulus_rect, reward)
                        render.draw_scorebar(self.screen, self.font_small, self.score, self.scoreable_trial_num)
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
                    render.draw_header(self.screen, self.font_big, self.font_small, self.trial_num, self.stats is None)
                    render.draw_stimulus_box(self.screen, self.stimulus_rect, image_surface)
                    render.draw_feedback_overlay(self.screen, self.stimulus_rect, reward)
                    render.draw_scorebar(self.screen, self.font_small, self.score, self.scoreable_trial_num)
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
                    render.draw_header(self.screen, self.font_big, self.font_small, self.trial_num, self.stats is None)
                    render.draw_stimulus_box(self.screen, self.stimulus_rect)
                    render.draw_scorebar(self.screen, self.font_small, self.score, self.scoreable_trial_num)
                    pygame.display.flip()
                    self.clock.tick(120)
            if not running:
                break

        self.erp_adapter.shutdown()
        if running:
            self.show_final_screen()
        pygame.quit()