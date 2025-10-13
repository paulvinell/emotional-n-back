import random
import time
from enum import Enum, auto
from typing import Optional

import numpy as np
import pygame
from pygame import Rect

from emotional_n_back.data import (
    KDEFSentimentLoader,
    MAVSentimentLoader,
)

from .erp_adapter import ErpAdapter
from .reward import Calibration, Reward, Stats, ZScorePolicy
from .state import GameState


def make_beep(
    frequency: int = 880, duration_ms: int = 120, volume: float = 0.5
) -> pygame.mixer.Sound:
    """Generate a sine beep as a pygame Sound. Assumes mixer is initialized."""
    init = pygame.mixer.get_init()
    if init is None:
        raise RuntimeError("pygame.mixer not initialized")
    sample_rate, _fmt, channels = init

    n_samples = int(sample_rate * (duration_ms / 1000.0))
    t = np.linspace(
        0.0, duration_ms / 1000.0, n_samples, endpoint=False, dtype=np.float64
    )
    wave = 0.5 * np.sin(2.0 * np.pi * float(frequency) * t)
    mono = (wave * (2**15 - 1)).astype(np.int16, copy=False)

    if channels == 1:
        pcm = mono
    elif channels == 2:
        pcm = np.column_stack((mono, mono))
    else:
        raise ValueError(f"Unsupported mixer channels: {channels}")

    pcm = np.ascontiguousarray(pcm)
    snd = pygame.sndarray.make_sound(pcm)
    snd.set_volume(max(0.0, min(1.0, float(volume))))
    return snd


class EEGStroopGame:
    def __init__(
        self,
        seed: Optional[int] = None,
        visual_intro_ms: int = 500,
        response_window_ms: int = 2000,
        feedback_ms: int = 500,
        isi_ms: int = 300,
        window_size=(900, 650),
        fs_fallback: float = 256.0,
        initial_calibration_trials: int = 10,
        recalibration_interval: int = 10,
        outlier_std_devs: Optional[float] = 3.0,
        erp_component: str = "P300",
        trial_duration_ms: Optional[int] = 2500,
    ):
        if seed is not None:
            random.seed(seed)
        self.visual_intro_ms = visual_intro_ms
        self.response_window_ms = response_window_ms
        self.feedback_ms = feedback_ms
        self.isi_ms = isi_ms

        # Data Loaders
        self.kdef_loader = KDEFSentimentLoader()
        self.mav_loader = MAVSentimentLoader()
        self.sentiments = self.mav_loader.sentiments

        # UI Layout
        W, H = window_size
        self.stimulus_rect = Rect(W // 2 - 220, 100, 440, 440)

        # Game State
        self.trial_num = 0
        self.score = 0
        self._img_cache: dict[str, pygame.Surface] = {}
        self.state = GameState.WAIT_EEG
        self.trial_start_t = 0
        self.image_surface = None
        self.audio_sound = None
        self.event_code = ""
        self.reward = Reward.NONE

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

        self.beep_success = make_beep(1300, 100, 0.5)
        self.beep_failure = make_beep(440, 200, 0.5)

        self.scoreable_trial_num = 0
        self.recompute_stats_next_trial = False

    def start(self):
        self.erp_adapter.start()

    def update(self):
        if self.state == GameState.WAIT_EEG:
            self.state = self._wait_eeg()
        elif self.state == GameState.PREPARE_TRIAL:
            self.state = self._prepare_trial()
        elif self.state == GameState.INTRO:
            self.state = self._intro()
        elif self.state == GameState.STIMULUS:
            self.state = self._stimulus()
        elif self.state == GameState.RESPONSE:
            self.state = self._response()
        elif self.state == GameState.FEEDBACK:
            self.state = self._feedback()
        elif self.state == GameState.ISI:
            self.state = self._isi()

    def _wait_eeg(self):
        if self.erp_adapter.eeg_started.is_set():
            return GameState.PREPARE_TRIAL
        return GameState.WAIT_EEG

    def _prepare_trial(self):
        if self.recompute_stats_next_trial:
            self.stats = self.calibration.compute(self.outlier_std_devs)
            self.calibration.reset_batch()
            self.recompute_stats_next_trial = False

        self.reward = Reward.NONE
        visual_sentiment = random.choice(self.sentiments)
        audio_sentiment = random.choice(self.sentiments)
        self.event_code = f"trial_{self.trial_num}"

        image_path = self.kdef_loader.get_random_image(visual_sentiment)
        self.image_surface = self._load_fit_image(str(image_path), self.stimulus_rect)

        audio_path = self.mav_loader.get_random_audio(audio_sentiment)
        self.audio_sound = pygame.mixer.Sound(str(audio_path))
        if self.audio_sound is None:
            raise RuntimeError(f"Failed to load audio file: {audio_path}")

        self.trial_start_t = pygame.time.get_ticks()
        return GameState.INTRO

    def _intro(self):
        if pygame.time.get_ticks() - self.trial_start_t > self.visual_intro_ms:
            self.trial_start_t = pygame.time.get_ticks()
            return GameState.STIMULUS
        return GameState.INTRO

    def _stimulus(self):
        self.audio_sound.play()
        self.erp_adapter.ingest_event(self.event_code)
        print(f"Ingested event: {self.event_code}")
        self.trial_start_t = pygame.time.get_ticks()
        return GameState.RESPONSE

    def _response(self):
        erp_update = self.erp_adapter.poll_update(self.event_code)
        if erp_update:
            self._process_erp_update(erp_update)
            self.trial_start_t = pygame.time.get_ticks()
            return GameState.FEEDBACK

        if pygame.time.get_ticks() - self.trial_start_t > self.response_window_ms:
            # Timeout, check one last time
            erp_update = self.erp_adapter.poll_update(self.event_code)
            if erp_update:
                self._process_erp_update(erp_update)

            self.trial_start_t = pygame.time.get_ticks()
            return GameState.FEEDBACK

        return GameState.RESPONSE

    def _process_erp_update(self, erp_update):
        if erp_update and erp_update.amp is not None and erp_update.lat is not None:
            self.calibration.update(erp_update.amp, erp_update.lat)
            self.reward = self.reward_policy.decide(
                erp_update.amp, erp_update.lat, self.stats
            )

            if self.recompute_stats_next_trial is False and self.calibration.ready(
                self.initial_calibration_trials,
                self.recalibration_interval,
                self.stats is not None,
            ):
                self.recompute_stats_next_trial = True

            if self.stats is not None:
                if self.reward == Reward.SUCCESS:
                    self.beep_success.play()
                    self.score += 1
                elif self.reward == Reward.FAILURE:
                    self.beep_failure.play()

    def _feedback(self):
        if pygame.time.get_ticks() - self.trial_start_t > self.feedback_ms:
            self.trial_start_t = pygame.time.get_ticks()
            return GameState.ISI
        return GameState.FEEDBACK

    def _isi(self):
        if self.trial_duration_ms is not None:
            wait_ms = self.trial_duration_ms - (
                pygame.time.get_ticks() - self.trial_start_t
            )

            if wait_ms < 0:
                self.trial_start_t = pygame.time.get_ticks()
                self.trial_num += 1
                if self.stats is not None:
                    self.scoreable_trial_num += 1
                return GameState.PREPARE_TRIAL

        elif pygame.time.get_ticks() - self.trial_start_t > self.isi_ms:
            self.trial_start_t = pygame.time.get_ticks()
            self.trial_num += 1
            if self.stats is not None:
                self.scoreable_trial_num += 1
            return GameState.PREPARE_TRIAL

        return GameState.ISI

    def _load_fit_image(self, path: str, box: Rect) -> pygame.Surface:
        if path in self._img_cache:
            return self._img_cache[path]
        img = pygame.image.load(path).convert_alpha()
        iw, ih = img.get_width(), img.get_height()
        scale = min(box.w / iw, box.h / ih)
        surf = pygame.transform.smoothscale(img, (int(iw * scale), int(ih * scale)))
        self._img_cache[path] = surf
        return surf

    def get_trial_data(self):
        return {
            "trial_num": self.trial_num,
            "is_calibrating": self.stats is None,
            "stimulus_rect": self.stimulus_rect,
            "image_surface": self.image_surface,
            "reward": self.reward,
            "score": self.score,
            "scoreable_trial_num": self.scoreable_trial_num,
        }

    def get_final_screen_data(self):
        return {
            "trial_num": self.trial_num,
            "score": self.score,
        }

    def shutdown(self):
        self.erp_adapter.shutdown()
