import math
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

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
            if math.isfinite(raw_reward):
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


@dataclass
class TrialFilter:
    """Defines conditions for when a reward module applies."""
    sentiments: Optional[Set[Sentiment]] = None
    congruence: Optional[bool] = None  # True=Congruent, False=Incongruent, None=Any

    def __call__(self, visual_sentiment: Sentiment, audio_sentiment: Sentiment) -> bool:
        # Check congruence
        if self.congruence is not None:
            is_congruent = visual_sentiment == audio_sentiment
            if self.congruence != is_congruent:
                return False

        # Check sentiments (if any sentiment matches)
        if self.sentiments:
            if (visual_sentiment not in self.sentiments) and (
                audio_sentiment not in self.sentiments
            ):
                return False

        return True


@dataclass
class FeatureExtractor:
    """Defines what to measure from the ERP data."""
    component: str
    metric: str  # "amp" or "lat"
    inverse: bool = False  # True if smaller is better (e.g. latency, or negative N200)

    def __call__(self, erp_data: Dict[str, Dict[str, float]]) -> float:
        val = erp_data.get(self.component, {}).get(self.metric)
        if val is None:
            return 0.0
        
        if self.inverse:
            # For N200 (negative), we want more negative, so we invert.
            # For latency, we want smaller, so we invert (conceptually).
            # But wait, standard z-scoring assumes higher is better.
            # If we want "more negative" (e.g. -10 is better than -5), 
            # we should actually just negate it? -(-10) = 10, -(-5) = 5. Yes.
            # If we want "smaller latency" (e.g. 300 is better than 400),
            # we should negate it? -300 > -400. Yes.
            return -val
        return val


class GenericRewardModule(RewardModule):
    """A data-driven reward module configured by filter and extractor."""

    def __init__(
        self,
        name: str,
        trial_filter: TrialFilter,
        extractor: FeatureExtractor,
        enable_z_scoring: bool = True,
        initial_calibration_trials: int = 10,
        recalibration_interval: int = 10,
        outlier_std_devs: Optional[float] = 3.0,
    ):
        super().__init__(
            name=name,
            required_erps=[extractor.component],
            enable_z_scoring=enable_z_scoring,
            initial_calibration_trials=initial_calibration_trials,
            recalibration_interval=recalibration_interval,
            outlier_std_devs=outlier_std_devs,
        )
        self.trial_filter = trial_filter
        self.extractor = extractor

    def is_trial_type_applicable(
        self,
        visual_sentiment: Sentiment,
        audio_sentiment: Sentiment,
    ) -> bool:
        return self.trial_filter(visual_sentiment, audio_sentiment)

    def calculate_reward(self, erp_data: Dict[str, Dict[str, float]]) -> float:
        return self.extractor(erp_data)


# Special case for composite modules (like Protocol E's index)
# We can implement them as GenericRewardModule with a custom extractor 
# or keep them as subclasses if they are truly complex.
# Protocol E: Index = z(P300) - z(LPP_late). This requires z-scoring *before* combination?
# Or just raw combination? The protocol says "Index = z(P300) - z(LPP)".
# This implies we need access to z-scores inside the calculation.
# Our current architecture z-scores the *result* of calculate_reward.
# So we might need a CompositeRewardModule or just keep specific classes for complex logic.
# Let's keep EfficientProcessingIndexModule and P300ZScoreRewardModule as they are complex.
# But we can replace the simple ones.

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
        # Note: This is a raw difference, z-scoring happens after.
        return p300_amp - lpp_late_amp




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
