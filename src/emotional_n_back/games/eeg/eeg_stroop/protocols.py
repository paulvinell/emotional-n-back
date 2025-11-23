from typing import List

from .reward_modules import (
    BidirectionalNegativeLPPModule,
    BidirectionalPositiveLPPModule,
    ClearAndLetGoModule,
    EfficientProcessingIndexModule,
    FastConflictDetectModule,
    FluentEngageModule,
    ModularReward,
    P300ZScoreRewardModule,
    RewardModule,
)


class ProtocolFactory:
    @staticmethod
    def create_protocol(protocol_name: str) -> ModularReward:
        """
        Creates a ModularReward instance configured for the specified protocol.
        """
        modules: List[RewardModule] = []

        if protocol_name == "A":
            # Protocol A: Fluent Engage
            modules = [FluentEngageModule()]
        elif protocol_name == "B":
            # Protocol B: Clear & Let Go
            modules = [ClearAndLetGoModule()]
        elif protocol_name == "C":
            # Protocol C: Fast Conflict Detect
            modules = [FastConflictDetectModule()]
        elif protocol_name == "D":
            # Protocol D: Bidirectional Emotional Control
            modules = [
                BidirectionalPositiveLPPModule(),
                BidirectionalNegativeLPPModule(),
            ]
        elif protocol_name == "E":
            # Protocol E: Efficient Processing Index
            modules = [EfficientProcessingIndexModule()]
        elif protocol_name == "Original":
            # Original P300 Z-Score
            modules = [P300ZScoreRewardModule()]
        else:
            raise ValueError(f"Unknown protocol: {protocol_name}")

        # Create ModularReward
        # Note: z-scoring is now handled by individual modules, and the aggregator
        # simply averages the normalized results.
        return ModularReward(modules=modules)

    @staticmethod
    def get_available_protocols() -> List[str]:
        return ["A", "B", "C", "D", "E", "Original"]
