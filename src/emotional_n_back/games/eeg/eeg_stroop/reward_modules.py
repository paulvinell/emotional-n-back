from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Dict, List, Optional

from .reward import ZScorer


class Sentiment(Enum):
    POSITIVE = auto()
    NEGATIVE = auto()
    NEUTRAL = auto()


class RewardModule(ABC):
    def __init__(
        self,
        name: str,
        required_erps: List[str],
        enable_z_scoring: bool = True,
        initial_calibration_trials: int = 10,
        recalibration_interval: int = 10,
        outlier_std_devs: Optional[float] = 3.0,
    ):
        self.name = name
        self.enabled = True
        self.required_erps = required_erps
        self.z_scorer = (
            ZScorer(
                initial_calibration_trials=initial_calibration_trials,
                recalibration_interval=recalibration_interval,
                outlier_std_devs=outlier_std_devs,
            )
            if enable_z_scoring
            else None
        )

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
        """Calculates the raw reward based on ERP data."""
        pass

    def has_required_erps(self, available_erps: List[str]) -> bool:
        """Checks if all required ERPs are available."""
        return all(erp in available_erps for erp in self.required_erps)

    def update_calibrators(
        self,
        erp_data: Dict[str, Dict[str, float]],
        visual_sentiment: Optional[Sentiment] = None,
        audio_sentiment: Optional[Sentiment] = None,
    ):
        """Updates the internal z-scorer if enabled and applicable."""
        if self.z_scorer:
            # Only update if the trial type is applicable
            if visual_sentiment and audio_sentiment:
                if not self.is_trial_type_applicable(visual_sentiment, audio_sentiment):
                    return

            # Check if we have required ERPs
            available_erps = list(erp_data.keys())
            if not self.has_required_erps(available_erps):
                return

            raw_reward = self.calculate_reward(erp_data)
            self.z_scorer.update(raw_reward)

    def recalibrate(self):
        """Recalibrates the z-scorer if enabled."""
        if self.z_scorer:
            self.z_scorer.recalibrate_if_ready()

    def is_calibrated(self) -> bool:
        """Checks if the z-scorer is calibrated."""
        if self.z_scorer:
            return self.z_scorer.is_calibrated()
        return True  # If no z-scoring, we are always "calibrated" (or it doesn't matter)

    def calculate_normalized_reward(
        self, erp_data: Dict[str, Dict[str, float]]
    ) -> float:
        """Calculates the z-scored reward."""
        raw_reward = self.calculate_reward(erp_data)
        if self.z_scorer:
            if self.z_scorer.is_calibrated():
                return self.z_scorer.get_z_score(raw_reward)
            else:
                return 0.0
        return raw_reward


# Protocol A
class FluentEngageModule(RewardModule):
    """Boost P300 on congruent targets."""

    def __init__(self):
        super().__init__(name="FluentEngage", required_erps=["P300"])

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
        super().__init__(name="ClearAndLetGo", required_erps=["LPP_late"])

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
        super().__init__(name="FastConflictDetect", required_erps=["N200"])

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
        super().__init__(name="BidirectionalPositiveLPP", required_erps=["LPP_early"])

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
        super().__init__(name="BidirectionalNegativeLPP", required_erps=["LPP_late"])

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
        super().__init__(name="EfficientProcessingIndex", required_erps=["P300", "LPP_late"])

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
        super().__init__(name="P300ZScore", required_erps=["P300"], enable_z_scoring=False)
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

    def update_calibrators(
        self,
        erp_data: Dict[str, Dict[str, float]],
        visual_sentiment: Optional[Sentiment] = None,
        audio_sentiment: Optional[Sentiment] = None,
    ):
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

    def __init__(
        self,
        modules: List[RewardModule],
    ):
        self.modules = modules

    def get_module(self, name: str) -> Optional[RewardModule]:
        for module in self.modules:
            if module.name == name:
                return module
        return None

    def enable_module(self, name: str):
        module = self.get_module(name)
        if module:
            module.enabled = True

    def disable_module(self, name: str):
        module = self.get_module(name)
        if module:
            module.enabled = False

    def set_active_modules(self, names: List[str]):
        """Enables only the modules in the list, disables others."""
        for module in self.modules:
            module.enabled = module.name in names

    def update_calibrators(
        self,
        erp_data: Dict[str, Dict[str, float]],
        visual_sentiment: Optional[Sentiment] = None,
        audio_sentiment: Optional[Sentiment] = None,
    ):
        """
        Calls the 'update_calibrators' method on any module that has it.
        """
        for module in self.modules:
            if not module.enabled:
                continue
            if hasattr(module, "update_calibrators"):
                # Check signature to see if it accepts sentiments (for backward compatibility if needed, 
                # though we updated the base class)
                # But P300ZScoreRewardModule overrides it.
                module.update_calibrators(
                    erp_data,
                    visual_sentiment=visual_sentiment,
                    audio_sentiment=audio_sentiment,
                )

    def recalibrate_modules(self):
        """
        Calls the 'recalibrate' method on any module that has it.
        """
        for module in self.modules:
            if not module.enabled:
                continue
            if hasattr(module, "recalibrate"):
                module.recalibrate()

    def is_calibrated(self) -> bool:
        """Checks if all calibrating modules are ready."""
        all_modules_calibrated = all(
            module.is_calibrated()
            for module in self.modules
            if module.enabled and hasattr(module, "is_calibrated")
        )
        return all_modules_calibrated

    def calculate_total_reward(
        self,
        visual_sentiment: Sentiment,
        audio_sentiment: Sentiment,
        erp_data: Dict[str, Dict[str, float]],
    ) -> float:
        """Calculates the mean normalized reward from all active modules."""
        available_erps = list(erp_data.keys())

        # Update any modules that use calibration
        self.update_calibrators(
            erp_data,
            visual_sentiment=visual_sentiment,
            audio_sentiment=audio_sentiment,
        )

        total_reward = 0.0
        active_count = 0
        
        for module in self.modules:
            if not module.enabled:
                continue
            if module.is_trial_type_applicable(
                visual_sentiment, audio_sentiment
            ) and module.has_required_erps(available_erps):
                total_reward += module.calculate_normalized_reward(erp_data)
                active_count += 1

        if active_count == 0:
            return 0.0
            
        return total_reward / active_count

    def calculate_normalized_total_reward(
        self,
        visual_sentiment: Sentiment,
        audio_sentiment: Sentiment,
        erp_data: Dict[str, Dict[str, float]],
    ) -> float:
        """
        Returns the total reward. 
        Since modules are already normalized and we average them, 
        this is equivalent to calculate_total_reward.
        """
        return self.calculate_total_reward(
            visual_sentiment, audio_sentiment, erp_data
        )
