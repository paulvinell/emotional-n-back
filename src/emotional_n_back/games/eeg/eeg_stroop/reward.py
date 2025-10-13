
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np

class Reward(Enum):
    SUCCESS = auto()
    FAILURE = auto()
    NONE = auto()

@dataclass
class Stats:
    mean_amp: float
    std_amp: float
    mean_lat: float
    std_lat: float
    threshold_amp: float

class Calibration:
    def __init__(self):
        self.calibration_data = []

    def update(self, amp: float, lat: float) -> None:
        self.calibration_data.append((amp, lat))

    def ready(self, initial_trials: int, interval: int, have_threshold: bool) -> bool:
        trials_needed = initial_trials if not have_threshold else interval
        return len(self.calibration_data) >= trials_needed

    def compute(self, outlier_std_devs: Optional[float]) -> Optional[Stats]:
        if not self.calibration_data:
            return None

        mean_amp_cal = np.mean([d[0] for d in self.calibration_data])
        std_amp_cal = np.std([d[0] for d in self.calibration_data])

        if outlier_std_devs is not None:
            filtered_data = [
                d
                for d in self.calibration_data
                if abs(d[0] - mean_amp_cal) <= outlier_std_devs * std_amp_cal
            ]
        else:
            filtered_data = self.calibration_data

        if len(filtered_data) < 2 and len(self.calibration_data) >= 2:
            deviations = [
                (d, abs(d[0] - mean_amp_cal))
                for d in self.calibration_data
            ]
            deviations.sort(key=lambda x: x[1])
            final_data = [d[0] for d in deviations[:2]]
        else:
            final_data = filtered_data

        if not final_data:
            return None

        amps, lats = zip(*final_data)
        mean_amp = np.mean(amps)
        std_amp = np.std(amps)
        mean_lat = np.mean(lats)
        std_lat = np.std(lats)
        threshold_amp = mean_amp

        return Stats(mean_amp, std_amp, mean_lat, std_lat, threshold_amp)

    def reset_batch(self) -> None:
        self.calibration_data = []

class ZScorePolicy:
    def __init__(self, success=0.5, failure=-0.5):
        self.success_threshold = success
        self.failure_threshold = failure

    def decide(self, amp: float, lat: float, stats: Optional[Stats]) -> Reward:
        if stats is None or stats.std_amp <= 1e-6 or stats.std_lat <= 1e-6:
            return Reward.NONE

        z_amp = (amp - stats.mean_amp) / stats.std_amp
        z_lat = (stats.mean_lat - lat) / stats.std_lat
        avg_z = (z_amp + z_lat) / 2

        if avg_z > self.success_threshold:
            return Reward.SUCCESS
        elif avg_z < self.failure_threshold:
            return Reward.FAILURE
        else:
            return Reward.NONE
