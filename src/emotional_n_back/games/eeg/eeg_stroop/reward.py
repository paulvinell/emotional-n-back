from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np


class Reward(Enum):
    SUCCESS = auto()
    FAILURE = auto()
    NONE = auto()


@dataclass
class SimpleStats:
    mean: float
    std: float


class ZScorer:
    def __init__(
        self,
        initial_calibration_trials: int = 10,
        recalibration_interval: int = 10,
        outlier_std_devs: Optional[float] = 3.0,
    ):
        self.initial_calibration_trials = initial_calibration_trials
        self.recalibration_interval = recalibration_interval
        self.outlier_std_devs = outlier_std_devs

        self._calibration_data: List[float] = []
        self.stats: Optional[SimpleStats] = None
        self._recalibrate_needed = False

    def update(self, value: float):
        self._calibration_data.append(value)
        if not self._recalibrate_needed:
            is_ready = len(self._calibration_data) >= (
                self.initial_calibration_trials
                if self.stats is None
                else self.recalibration_interval
            )
            if is_ready:
                self._recalibrate_needed = True

    def recalibrate_if_ready(self):
        if not self._recalibrate_needed:
            return

        if not self._calibration_data:
            self._recalibrate_needed = False
            return

        data = np.array(self._calibration_data)
        if self.outlier_std_devs is not None and len(data) > 2:
            mean = np.mean(data)
            std = np.std(data)
            if std > 1e-6:
                filtered_data = data[np.abs(data - mean) <= self.outlier_std_devs * std]
                if len(filtered_data) >= 2:
                    data = filtered_data

        self.stats = SimpleStats(mean=np.mean(data), std=np.std(data))

        self._calibration_data = []
        self._recalibrate_needed = False

    def get_z_score(self, value: float) -> float:
        if self.stats is None or self.stats.std <= 1e-6:
            return 0.0
        return (value - self.stats.mean) / self.stats.std

    def is_calibrated(self) -> bool:
        return self.stats is not None
