from dataclasses import dataclass
from typing import Optional

from .reward import Reward


@dataclass
class TrialState:
    trial_num: int
    score: int
    scoreable_trial_num: int
    is_calibrating: bool
    regular_trials_start_idx: Optional[int]


class TrialManager:
    def __init__(self, initial_calibration_trials: int = 10):
        self.trial_num = 0
        self.score = 0
        self.scoreable_trial_num = 0
        self.initial_calibration_trials = initial_calibration_trials
        self.regular_trials_start_idx: Optional[int] = None

    def on_trial_complete(self, reward: Reward, is_calibrated: bool):
        """Called when a trial is completed (after feedback)."""
        # Update score immediately if applicable
        if is_calibrated:
            if reward == Reward.SUCCESS:
                self.score += 1
            
            # We count scoreable trials here
            self.scoreable_trial_num += 1

    def increment_trial_num(self):
        """Called when moving to the next trial."""
        self.trial_num += 1

    def check_calibration_start(self, is_calibrated: bool):
        """Checks and sets the start index for regular trials if calibration is done."""
        if is_calibrated and self.regular_trials_start_idx is None:
            self.regular_trials_start_idx = self.trial_num

    def get_state(self, is_calibrated: bool) -> TrialState:
        return TrialState(
            trial_num=self.trial_num,
            score=self.score,
            scoreable_trial_num=self.scoreable_trial_num,
            is_calibrating=not is_calibrated,
            regular_trials_start_idx=self.regular_trials_start_idx,
        )
