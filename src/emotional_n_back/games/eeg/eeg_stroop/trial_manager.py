from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from .reward import Reward


class Phase(ABC):
    @abstractmethod
    def get_display_text(self) -> str:
        pass

    @abstractmethod
    def is_calibrating(self) -> bool:
        pass


@dataclass
class CalibrationPhase(Phase):
    trial_idx: int
    
    def get_display_text(self) -> str:
        return f"Calibration Trial {self.trial_idx + 1}"

    def is_calibrating(self) -> bool:
        return True


@dataclass
class RegularPhase(Phase):
    trial_idx: int
    
    def get_display_text(self) -> str:
        return f"Trial {self.trial_idx + 1}"

    def is_calibrating(self) -> bool:
        return False


@dataclass
class TrialState:
    display_text: str
    score: int
    scoreable_trial_num: int
    is_calibrating: bool


class TrialManager:
    def __init__(self, initial_calibration_trials: int = 10):
        self.score = 0
        self.scoreable_trial_num = 0
        self.initial_calibration_trials = initial_calibration_trials
        
        # Start in calibration phase
        self.current_phase: Phase = CalibrationPhase(trial_idx=0)
        
        # We track total trials internally if needed, but the phase handles the relative index
        self._total_trials = 0

    @property
    def trial_num(self) -> int:
        return self._total_trials

    def on_trial_complete(self, reward: Reward, is_calibrated: bool):
        """Called when a trial is completed (after feedback)."""
        # Update score immediately if applicable
        if isinstance(self.current_phase, RegularPhase):
            if reward == Reward.SUCCESS:
                self.score += 1
            self.scoreable_trial_num += 1

    def increment_trial_num(self):
        """Called when moving to the next trial."""
        self._total_trials += 1
        
        if isinstance(self.current_phase, CalibrationPhase):
            self.current_phase.trial_idx += 1
        elif isinstance(self.current_phase, RegularPhase):
            self.current_phase.trial_idx += 1

    def check_calibration_start(self, is_calibrated: bool):
        """Checks if we should transition to RegularPhase."""
        if isinstance(self.current_phase, CalibrationPhase):
            # We only switch if the system reports it is calibrated
            if is_calibrated:
                # Transition to RegularPhase starting at trial 0
                self.current_phase = RegularPhase(trial_idx=0)

    def get_state(self) -> TrialState:
        return TrialState(
            display_text=self.current_phase.get_display_text(),
            score=self.score,
            scoreable_trial_num=self.scoreable_trial_num,
            is_calibrating=self.current_phase.is_calibrating(),
        )
