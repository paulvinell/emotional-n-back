import pytest
from unittest.mock import MagicMock
from emotional_n_back.games.eeg.eeg_stroop.reward_modules import RewardModule, ModularReward, Sentiment
from emotional_n_back.games.eeg.eeg_stroop.reward import ZScorer

class MockRewardModule(RewardModule):
    def __init__(self, required_erps=["P300"], **kwargs):
        super().__init__(required_erps=required_erps, **kwargs)

    def is_trial_type_applicable(self, visual_sentiment, audio_sentiment):
        return True

    def calculate_reward(self, erp_data):
        return erp_data.get("P300", {}).get("amp", 0.0)

def test_reward_module_initialization():
    module = MockRewardModule()
    assert module.required_erps == ["P300"]
    assert module.z_scorer is None

def test_reward_module_with_z_scoring():
    module = MockRewardModule(enable_z_scoring=True)
    assert isinstance(module.z_scorer, ZScorer)
    assert not module.is_calibrated()

def test_update_calibrators():
    module = MockRewardModule(enable_z_scoring=True, initial_calibration_trials=2)
    erp_data = {"P300": {"amp": 10.0}}
    
    module.update_calibrators(erp_data)
    module.update_calibrators({"P300": {"amp": 12.0}})
    module.recalibrate()
    assert module.is_calibrated()

def test_calculate_normalized_reward():
    module = MockRewardModule(enable_z_scoring=True, initial_calibration_trials=2)
    
    # Calibrate: Mean 15, Std 5
    module.update_calibrators({"P300": {"amp": 10.0}})
    module.update_calibrators({"P300": {"amp": 20.0}})
    module.recalibrate()
    
    erp_data = {"P300": {"amp": 20.0}}
    reward = module.calculate_normalized_reward(erp_data)
    assert reward == pytest.approx(1.0)

def test_calculate_normalized_reward_not_calibrated():
    module = MockRewardModule(enable_z_scoring=True)
    erp_data = {"P300": {"amp": 20.0}}
    reward = module.calculate_normalized_reward(erp_data)
    assert reward == 0.0

def test_modular_reward_z_scoring():
    # Setup two modules that return 10.0 and 20.0 respectively
    m1 = MockRewardModule()
    m2 = MockRewardModule()
    
    # Enable z-scoring for the aggregator
    modular_reward = ModularReward(
        modules=[m1, m2],
        enable_z_scoring=True,
        initial_calibration_trials=2
    )
    
    erp_data = {"P300": {"amp": 10.0}} # m1 returns 10, m2 returns 10 -> total 20
    
    # Calibration trial 1: Total 20
    modular_reward.calculate_normalized_total_reward(Sentiment.NEUTRAL, Sentiment.NEUTRAL, erp_data)
    
    erp_data_2 = {"P300": {"amp": 20.0}} # m1 returns 20, m2 returns 20 -> total 40
    # Calibration trial 2: Total 40
    modular_reward.calculate_normalized_total_reward(Sentiment.NEUTRAL, Sentiment.NEUTRAL, erp_data_2)
    
    modular_reward.recalibrate_modules()
    
    # Mean 30, Std 10
    # Test with total 40 -> (40 - 30) / 10 = 1.0
    
    reward = modular_reward.calculate_normalized_total_reward(Sentiment.NEUTRAL, Sentiment.NEUTRAL, erp_data_2)
    assert reward == pytest.approx(1.0)

def test_modular_reward_not_calibrated():
    m1 = MockRewardModule()
    modular_reward = ModularReward(modules=[m1], enable_z_scoring=True)
    erp_data = {"P300": {"amp": 10.0}}
    reward = modular_reward.calculate_normalized_total_reward(Sentiment.NEUTRAL, Sentiment.NEUTRAL, erp_data)
    assert reward == 0.0
