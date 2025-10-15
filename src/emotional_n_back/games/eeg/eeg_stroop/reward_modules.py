from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Dict, List


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


class ModularReward:
    """A class that manages a list of reward modules."""

    def __init__(self, modules: List[RewardModule]):
        self.modules = modules

    def calculate_total_reward(
        self,
        visual_sentiment: Sentiment,
        audio_sentiment: Sentiment,
        erp_events: List[Dict],
    ) -> float:
        """Calculates the total reward from all active modules based on ERP events."""

        repackaged_erps: Dict[str, Dict[str, float]] = {}
        for event in erp_events:
            if event.get("type") == "erp_update" and "component" in event:
                for comp_name, comp_data in event["component"].items():
                    repackaged_erps[comp_name] = comp_data

        available_erps = list(repackaged_erps.keys())

        total_reward = 0.0
        for module in self.modules:
            if module.is_trial_type_applicable(
                visual_sentiment, audio_sentiment
            ) and module.has_required_erps(available_erps):
                total_reward += module.calculate_reward(repackaged_erps)
        return total_reward
