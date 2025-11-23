import pytest
import math
from emotional_n_back.games.eeg.eeg_stroop.reward_modules import (
    GenericRewardModule,
    TrialFilter,
    FeatureExtractor,
    Sentiment,
)

def test_nan_handling_in_calibration():
    # Create a module
    module = GenericRewardModule(
        name="TestModule",
        trial_filter=TrialFilter(),
        extractor=FeatureExtractor(component="P300", metric="amp"),
        initial_calibration_trials=3,
        recalibration_interval=3,
    )
    
    # Feed it some valid data
    erp_valid = {"P300": {"amp": 10.0}}
    module.update_calibrators(erp_valid, Sentiment.NEUTRAL, Sentiment.NEUTRAL)
    
    # Feed it some NaN data
    erp_nan = {"P300": {"amp": float("nan")}}
    module.update_calibrators(erp_nan, Sentiment.NEUTRAL, Sentiment.NEUTRAL)
    
    # Feed more valid data
    module.update_calibrators(erp_valid, Sentiment.NEUTRAL, Sentiment.NEUTRAL)
    module.update_calibrators(erp_valid, Sentiment.NEUTRAL, Sentiment.NEUTRAL)
    
    # Should be ready to calibrate (3 valid trials)
    # The NaN trial should have been ignored
    module.recalibrate()
    
    assert module.is_calibrated()
    assert module.z_scorer.stats is not None
    assert not math.isnan(module.z_scorer.stats.mean)
    assert module.z_scorer.stats.mean == 10.0

def test_nan_handling_runtime():
    # Create a module and calibrate it manually
    module = GenericRewardModule(
        name="TestModule",
        trial_filter=TrialFilter(),
        extractor=FeatureExtractor(component="P300", metric="amp"),
    )
    # Mock calibration
    module.z_scorer.stats = type("Stats", (), {"mean": 10.0, "std": 2.0})()
    
    # Calculate reward with NaN input
    erp_nan = {"P300": {"amp": float("nan")}}
    reward = module.calculate_normalized_reward(erp_nan)
    
    # Should return NaN (runtime propagation is allowed/expected, 
    # as the game handles it by giving no reward)
    assert math.isnan(reward)
