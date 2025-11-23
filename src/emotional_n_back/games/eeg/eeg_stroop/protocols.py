from typing import List, Dict, Callable

from .reward_modules import (
    EfficientProcessingIndexModule,
    FeatureExtractor,
    GenericRewardModule,
    ModularReward,
    GenericRewardModule,
    ModularReward,
    RewardModule,
    Sentiment,
    TrialFilter,
)


# Protocol Definitions
# Protocol Definitions
# Now lambdas accept kwargs to configure the modules
PROTOCOL_CONFIGS: Dict[str, Callable[..., List[RewardModule]]] = {
    "A": lambda **kwargs: [
        GenericRewardModule(
            name="FluentEngage",
            trial_filter=TrialFilter(congruence=True),
            extractor=FeatureExtractor(component="P300", metric="amp"),
            **kwargs,
        )
    ],
    "B": lambda **kwargs: [
        GenericRewardModule(
            name="ClearAndLetGo",
            trial_filter=TrialFilter(
                congruence=False, sentiments={Sentiment.NEGATIVE}
            ),
            extractor=FeatureExtractor(component="LPP_late", metric="amp", inverse=True),
            **kwargs,
        )
    ],
    "C": lambda **kwargs: [
        GenericRewardModule(
            name="FastConflictDetect",
            trial_filter=TrialFilter(congruence=False),
            extractor=FeatureExtractor(component="N200", metric="amp", inverse=True),
            **kwargs,
        )
    ],
    "D": lambda **kwargs: [
        GenericRewardModule(
            name="BidirectionalPositiveLPP",
            trial_filter=TrialFilter(
                congruence=True, sentiments={Sentiment.POSITIVE}
            ),
            extractor=FeatureExtractor(component="LPP_early", metric="amp"),
            **kwargs,
        ),
        GenericRewardModule(
            name="BidirectionalNegativeLPP",
            trial_filter=TrialFilter(
                congruence=False, sentiments={Sentiment.NEGATIVE}
            ),
            extractor=FeatureExtractor(component="LPP_late", metric="amp", inverse=True),
            **kwargs,
        ),
    ],
    "E": lambda **kwargs: [EfficientProcessingIndexModule(**kwargs)],
    "Original": lambda **kwargs: [
        GenericRewardModule(
            name="P300Amplitude",
            trial_filter=TrialFilter(),  # Applies to all trials
            extractor=FeatureExtractor(component="P300", metric="amp"),
            **kwargs,
        ),
        GenericRewardModule(
            name="P300Latency",
            trial_filter=TrialFilter(),  # Applies to all trials
            extractor=FeatureExtractor(component="P300", metric="lat", inverse=True),
            **kwargs,
        ),
    ],
}


class ProtocolFactory:
    @staticmethod
    def create_protocol(
        protocol_name: str,
        initial_calibration_trials: int = 10,
        recalibration_interval: int = 10,
        outlier_std_devs: float = 3.0,
    ) -> ModularReward:
        """
        Creates a ModularReward instance configured for the specified protocol.
        """
        if protocol_name not in PROTOCOL_CONFIGS:
            raise ValueError(f"Unknown protocol: {protocol_name}")

        config = {
            "initial_calibration_trials": initial_calibration_trials,
            "recalibration_interval": recalibration_interval,
            "outlier_std_devs": outlier_std_devs,
        }

        modules = PROTOCOL_CONFIGS[protocol_name](**config)
        return ModularReward(modules=modules)

    @staticmethod
    def get_available_protocols() -> List[str]:
        return list(PROTOCOL_CONFIGS.keys())
