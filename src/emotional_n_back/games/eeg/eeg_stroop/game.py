import random
from typing import Optional

import pygame
from pygame import Rect

from .erp_adapter import ErpAdapter
from .reward import Reward
from .reward_modules import ModularReward, Sentiment
from .protocols import ProtocolFactory
from .state import GameState
from .trial_manager import TrialManager
from .resources import ResourceManager
from .render import RenderState


def make_beep(
    frequency: int = 880, duration_ms: int = 120, volume: float = 0.5
) -> pygame.mixer.Sound:
    """Generate a sine beep as a pygame Sound. Assumes mixer is initialized."""
    import numpy as np
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
        stimulus_intro_ms: int = 500,
        window_size=(900, 650),
        fs: Optional[float] = None,
        initial_calibration_trials: int = 10,
        recalibration_interval: int = 10,
        outlier_std_devs: Optional[float] = 3.0,
        erp_component: str = "P300",
        trial_duration_ms: Optional[int] = 2500,
        show_fs: bool = False,
    ):
        if seed is not None:
            random.seed(seed)
        self.stimulus_intro_ms = stimulus_intro_ms
        self.show_fs = show_fs

        # Managers
        self.trial_manager = TrialManager(initial_calibration_trials)
        self.resource_manager = ResourceManager()

        # UI Layout
        W, H = window_size
        self.stimulus_rect = Rect(W // 2 - 220, 100, 440, 440)

        # Game State
        self.state = GameState.WAIT_EEG
        self.trial_start_t = 0
        self.image_surface = None
        self.audio_sound = None
        self.event_code = ""
        self.reward = Reward.NONE
        
        self.visual_sentiment: Optional[Sentiment] = None
        self.audio_sentiment: Optional[Sentiment] = None

        self.erp_component = erp_component
        self.trial_duration_ms = trial_duration_ms
        self.recalibration_interval = recalibration_interval
        self.outlier_std_devs = outlier_std_devs
        
        if recalibration_interval < 2:
            raise ValueError("recalibration_interval must be at least 2")

        self.erp_adapter = ErpAdapter(
            erp_component=self.erp_component,
            host="127.0.0.1",
            port=5005,
            fs_target=fs,
        )

        # Use the "Original" protocol by default, which replicates the old behavior
        # but using the new data-driven architecture.
        self.modular_reward = ProtocolFactory.create_protocol(
            "Original",
            initial_calibration_trials=initial_calibration_trials,
            recalibration_interval=self.recalibration_interval,
            outlier_std_devs=self.outlier_std_devs,
        )
        self.success_threshold = 0.5
        self.failure_threshold = -0.5

        self.beep_success = make_beep(1300, 100, 0.5)
        self.beep_failure = make_beep(440, 200, 0.5)

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

    def _wait_eeg(self):
        if self.erp_adapter.eeg_started.is_set():
            return GameState.PREPARE_TRIAL
        return GameState.WAIT_EEG

    def _prepare_trial(self):
        self.modular_reward.recalibrate_modules()
        
        self.trial_manager.check_calibration_start(self.modular_reward.is_calibrated())

        self.reward = Reward.NONE
        self.visual_sentiment = self.resource_manager.get_random_sentiment()
        self.audio_sentiment = self.resource_manager.get_random_sentiment()
        self.event_code = f"trial_{self.trial_manager.trial_num}"

        image_path = self.resource_manager.get_random_image_path(self.visual_sentiment)
        self.image_surface = self.resource_manager.load_fit_image(image_path, self.stimulus_rect)

        audio_path = self.resource_manager.get_random_audio_path(self.audio_sentiment)
        self.audio_sound = self.resource_manager.load_audio(audio_path)

        self.trial_start_t = pygame.time.get_ticks()
        return GameState.INTRO

    def _intro(self):
        if pygame.time.get_ticks() - self.trial_start_t > self.stimulus_intro_ms:
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
            return GameState.FEEDBACK

        return GameState.RESPONSE

    def _process_erp_update(self, erp_update):
        # Do not process or give feedback on trials with artifacts
        if not erp_update.clean or erp_update.amp is None or erp_update.lat is None:
            self.reward = Reward.NONE
            return

        erp_data = {
            self.erp_component: {
                "amp": erp_update.amp,
                "lat": erp_update.lat,
            }
        }

        # This now internally handles updating the calibrators
        total_reward = self.modular_reward.calculate_total_reward(
            visual_sentiment=self.visual_sentiment,
            audio_sentiment=self.audio_sentiment,
            erp_data=erp_data,
        )

        # Discretize the final reward
        if total_reward > self.success_threshold:
            self.reward = Reward.SUCCESS
        elif total_reward < self.failure_threshold:
            self.reward = Reward.FAILURE
        else:
            self.reward = Reward.NONE

        # Update score and provide feedback if the system is calibrated
        # We delegate the score update to TrialManager
        is_calibrated = self.modular_reward.is_calibrated()
        self.trial_manager.on_trial_complete(self.reward, is_calibrated)
        
        if is_calibrated:
            if self.reward == Reward.SUCCESS:
                self.beep_success.play()
            elif self.reward == Reward.FAILURE:
                self.beep_failure.play()

    def _feedback(self):
        if self.trial_duration_ms is not None:
            if pygame.time.get_ticks() - self.trial_start_t > self.trial_duration_ms:
                self.trial_manager.increment_trial_num()
                return GameState.PREPARE_TRIAL
        return GameState.FEEDBACK

    def get_render_state(self) -> RenderState:
        trial_state = self.trial_manager.get_state()
        
        return RenderState(
            state=self.state,
            display_text=trial_state.display_text,
            score=trial_state.score,
            scoreable_trial_num=trial_state.scoreable_trial_num,
            is_calibrating=trial_state.is_calibrating,
            stimulus_rect=self.stimulus_rect,
            image_surface=self.image_surface if self.state != GameState.INTRO else None,
            reward=self.reward,
            show_fs=self.show_fs,
            fs=self.erp_adapter.effective_fs,
            continuous_fs_est=self.erp_adapter.continuous_fs_est,
            initial_calibration_trials=self.trial_manager.initial_calibration_trials,
        )

    def get_final_screen_data(self):
        return {
            "trial_num": self.trial_manager.trial_num,
            "score": self.trial_manager.score,
        }

    def shutdown(self):
        self.erp_adapter.shutdown()
