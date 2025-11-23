import pytest
from emotional_n_back.games.eeg.eeg_stroop.protocols import ProtocolFactory
from emotional_n_back.games.eeg.eeg_stroop.reward_modules import GenericRewardModule

def test_protocol_factory_config_propagation():
    # Test with custom configuration
    initial_trials = 5
    recal_interval = 20
    outlier_devs = 4.0
    
    modular_reward = ProtocolFactory.create_protocol(
        "Original",
        initial_calibration_trials=initial_trials,
        recalibration_interval=recal_interval,
        outlier_std_devs=outlier_devs,
    )
    
    # Check all modules have the correct config
    for module in modular_reward.modules:
        assert isinstance(module, GenericRewardModule)
        assert module.z_scorer is not None
        assert module.z_scorer.initial_calibration_trials == initial_trials
        assert module.z_scorer.recalibration_interval == recal_interval
        assert module.z_scorer.outlier_std_devs == outlier_devs

def test_protocol_factory_defaults():
    # Test with defaults
    modular_reward = ProtocolFactory.create_protocol("Original")
    
    for module in modular_reward.modules:
        assert module.z_scorer.initial_calibration_trials == 10
        assert module.z_scorer.recalibration_interval == 10
        assert module.z_scorer.outlier_std_devs == 3.0
