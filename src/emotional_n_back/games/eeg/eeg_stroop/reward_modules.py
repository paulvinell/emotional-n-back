from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Dict, List, Optional

from .reward import ZScorer


class Sentiment(Enum):
    POSITIVE = auto()
    NEGATIVE = auto()
    NEUTRAL = auto()


class RewardModule(ABC):
    def __init__(self, required_erps: List[str]):
        self.required_erps = required_erps

    @abstractmethod
    def is_trial_type_applicable(
        self,
        visual_sentiment: Sentiment,
        audio_sentiment: Sentiment,
    ) -> bool:
        """Determines if the module is applicable to the current trial type."""
        pass

    @abstractmethod
    def calculate_reward(self, erp_data: Dict[str, Dict[str, float]]) -> float:
        """Calculates the reward based on ERP data."""
        pass

    def has_required_erps(self, available_erps: List[str]) -> bool:
        """Checks if all required ERPs are available."""
        return all(erp in available_erps for erp in self.required_erps)


# Protocol A
class FluentEngageModule(RewardModule):
    """Boost P300 on congruent targets."""

    def __init__(self):
        super().__init__(required_erps=["P300"])

    def is_trial_type_applicable(
        self,
        visual_sentiment: Sentiment,
        audio_sentiment: Sentiment,
    ) -> bool:
        is_congruent = visual_sentiment == audio_sentiment
        return is_congruent

    def calculate_reward(self, erp_data: Dict[str, Dict[str, float]]) -> float:
        # Reward larger P300 amplitude
        return erp_data.get("P300", {}).get("amp", 0.0)


# Protocol B
class ClearAndLetGoModule(RewardModule):
    """Reduce late LPP on negative incongruent trials."""

    def __init__(self):
        super().__init__(required_erps=["LPP_late"])

    def is_trial_type_applicable(
        self,
        visual_sentiment: Sentiment,
        audio_sentiment: Sentiment,
    ) -> bool:
        is_incongruent = visual_sentiment != audio_sentiment
        is_negative_present = (
            visual_sentiment == Sentiment.NEGATIVE
            or audio_sentiment == Sentiment.NEGATIVE
        )
        return is_incongruent and is_negative_present

    def calculate_reward(self, erp_data: Dict[str, Dict[str, float]]) -> float:
        # Reward smaller LPP_late amplitude (inverse relationship)
        amp = erp_data.get("LPP_late", {}).get("amp", 0.0)
        return 1.0 / (amp + 1e-6)  # Add epsilon to avoid division by zero


# Protocol C
class FastConflictDetectModule(RewardModule):
    """Enhance N200 on incongruent trials."""

    def __init__(self):
        super().__init__(required_erps=["N200"])

    def is_trial_type_applicable(
        self,
        visual_sentiment: Sentiment,
        audio_sentiment: Sentiment,
    ) -> bool:
        is_incongruent = visual_sentiment != audio_sentiment
        return is_incongruent

    def calculate_reward(self, erp_data: Dict[str, Dict[str, float]]) -> float:
        # N200 is negative, so a more negative value is better.
        return -erp_data.get("N200", {}).get("amp", 0.0)


# Protocol D
class BidirectionalPositiveLPPModule(RewardModule):
    """Increase LPP for positive-congruent trials."""

    def __init__(self):
        super().__init__(required_erps=["LPP_early"])

    def is_trial_type_applicable(
        self,
        visual_sentiment: Sentiment,
        audio_sentiment: Sentiment,
    ) -> bool:
        is_congruent_positive = (
            visual_sentiment == Sentiment.POSITIVE
            and audio_sentiment == Sentiment.POSITIVE
        )
        return is_congruent_positive

    def calculate_reward(self, erp_data: Dict[str, Dict[str, float]]) -> float:
        return erp_data.get("LPP_early", {}).get("amp", 0.0)


class BidirectionalNegativeLPPModule(RewardModule):
    """Decrease LPP for negative-incongruent trials."""

    def __init__(self):
        super().__init__(required_erps=["LPP_late"])

    def is_trial_type_applicable(
        self,
        visual_sentiment: Sentiment,
        audio_sentiment: Sentiment,
    ) -> bool:
        is_incongruent = visual_sentiment != audio_sentiment
        is_negative_present = (
            visual_sentiment == Sentiment.NEGATIVE
            or audio_sentiment == Sentiment.NEGATIVE
        )
        return is_incongruent and is_negative_present

    def calculate_reward(self, erp_data: Dict[str, Dict[str, float]]) -> float:
        amp = erp_data.get("LPP_late", {}).get("amp", 0.0)
        return 1.0 / (amp + 1e-6)


# Protocol E
class EfficientProcessingIndexModule(RewardModule):
    """High P300, Low LPP-late on task-relevant (emotional) trials."""

    def __init__(self):
        super().__init__(required_erps=["P300", "LPP_late"])

    def is_trial_type_applicable(
        self,
        visual_sentiment: Sentiment,
        audio_sentiment: Sentiment,
    ) -> bool:
        is_emotional = (
            visual_sentiment != Sentiment.NEUTRAL
            or audio_sentiment != Sentiment.NEUTRAL
        )
        return is_emotional

    def calculate_reward(self, erp_data: Dict[str, Dict[str, float]]) -> float:
        p300_amp = erp_data.get("P300", {}).get("amp", 0.0)
        lpp_late_amp = erp_data.get("LPP_late", {}).get("amp", 0.0)
        # Rough index: P300 amp - LPP_late amp
        return p300_amp - lpp_late_amp


class P300ZScoreRewardModule(RewardModule):
    """
    Replicates the original game's reward system by calculating a reward
    based on the combined z-scores of P300 amplitude and latency.
    """

    def __init__(
        self,
        initial_calibration_trials: int = 10,
        recalibration_interval: int = 10,
        outlier_std_devs: Optional[float] = 3.0,
    ):
        super().__init__(required_erps=["P300"])
        self.amp_zscorer = ZScorer(
            initial_calibration_trials,
            recalibration_interval,
            outlier_std_devs,
        )
        self.lat_zscorer = ZScorer(
            initial_calibration_trials,
            recalibration_interval,
            outlier_std_devs,
        )

    def is_trial_type_applicable(
        self, visual_sentiment: Sentiment, audio_sentiment: Sentiment
    ) -> bool:
        # Active on all trials, like the original system
        return True

    def update_calibrators(self, erp_data: Dict[str, Dict[str, float]]):
        if "P300" in erp_data:
            amp = erp_data["P300"].get("amp")
            lat = erp_data["P300"].get("lat")
            if amp is not None:
                self.amp_zscorer.update(amp)
            if lat is not None:
                self.lat_zscorer.update(lat)

    def recalibrate(self):
        self.amp_zscorer.recalibrate_if_ready()
        self.lat_zscorer.recalibrate_if_ready()

    def is_calibrated(self) -> bool:
        return self.amp_zscorer.is_calibrated() and self.lat_zscorer.is_calibrated()

    def calculate_reward(self, erp_data: Dict[str, Dict[str, float]]) -> float:
        if not self.amp_zscorer.is_calibrated() or not self.lat_zscorer.is_calibrated():
            return 0.0

        p300_data = erp_data.get("P300", {})
        amp = p300_data.get("amp")
        lat = p300_data.get("lat")

        if amp is None or lat is None:
            return 0.0

        # Higher amplitude is better
        z_amp = self.amp_zscorer.get_z_score(amp)
        # Lower latency is better, so we invert the z-score
        z_lat = -self.lat_zscorer.get_z_score(lat)

        avg_z = (z_amp + z_lat) / 2

        return avg_z


class ModularReward:
    """A class that manages a list of reward modules."""

    def __init__(self, modules: List[RewardModule]):
        self.modules = modules

    def update_calibrators(self, erp_data: Dict[str, Dict[str, float]]):
        """
        Calls the 'update_calibrators' method on any module that has it.
        """
        for module in self.modules:
            if hasattr(module, "update_calibrators"):
                module.update_calibrators(erp_data)

    def recalibrate_modules(self):
        """
        Calls the 'recalibrate' method on any module that has it.
        """
        for module in self.modules:
            if hasattr(module, "recalibrate"):
                module.recalibrate()

    def is_calibrated(self) -> bool:
        """Checks if all calibrating modules are ready."""
        return all(
            module.is_calibrated()
            for module in self.modules
            if hasattr(module, "is_calibrated")
        )

    def calculate_total_reward(
        self,
        visual_sentiment: Sentiment,
        audio_sentiment: Sentiment,
        erp_data: Dict[str, Dict[str, float]],
    ) -> float:
        """Calculates the total reward from all active modules based on ERP events."""
        available_erps = list(erp_data.keys())

        # Update any modules that use calibration
        self.update_calibrators(erp_data)

        total_reward = 0.0
        for module in self.modules:
            if module.is_trial_type_applicable(
                visual_sentiment, audio_sentiment
            ) and module.has_required_erps(available_erps):
                total_reward += module.calculate_reward(erp_data)

        return total_reward
