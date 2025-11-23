import pytest
from emotional_n_back.games.eeg.eeg_stroop.reward_modules import (
    ModularReward,
    FluentEngageModule,
    ClearAndLetGoModule,
    Sentiment,
)
from emotional_n_back.games.eeg.eeg_stroop.protocols import ProtocolFactory

def test_enable_disable_module():
    m1 = FluentEngageModule()
    m2 = ClearAndLetGoModule()
    modular_reward = ModularReward(modules=[m1, m2])
    
    assert m1.enabled
    assert m2.enabled
    
    modular_reward.disable_module("FluentEngage")
    assert not m1.enabled
    assert m2.enabled
    
    modular_reward.enable_module("FluentEngage")
    assert m1.enabled

def test_set_active_modules():
    m1 = FluentEngageModule()
    m2 = ClearAndLetGoModule()
    modular_reward = ModularReward(modules=[m1, m2])
    
    modular_reward.set_active_modules(["ClearAndLetGo"])
    assert not m1.enabled
    assert m2.enabled
    
    modular_reward.set_active_modules(["FluentEngage", "ClearAndLetGo"])
    assert m1.enabled
    assert m2.enabled

def test_disabled_module_no_contribution():
    m1 = FluentEngageModule() # P300
    # Note: enable_z_scoring defaults to True now, but for this test we want raw values
    # or we need to mock the z-scorer.
    # Let's disable z-scoring on the module for this test to verify aggregation logic
    m1.z_scorer = None 
    
    modular_reward = ModularReward(modules=[m1])
    
    erp_data = {"P300": {"amp": 10.0}}
    
    # Enabled
    reward = modular_reward.calculate_total_reward(Sentiment.POSITIVE, Sentiment.POSITIVE, erp_data)
    assert reward == 10.0
    
    # Disabled
    modular_reward.disable_module("FluentEngage")
    reward = modular_reward.calculate_total_reward(Sentiment.POSITIVE, Sentiment.POSITIVE, erp_data)
    assert reward == 0.0

def test_averaging_logic():
    m1 = FluentEngageModule()
    m1.z_scorer = None # Disable z-scoring for deterministic testing
    
    m2 = ClearAndLetGoModule()
    m2.z_scorer = None
    
    modular_reward = ModularReward(modules=[m1, m2])
    
    # P300 = 10, LPP_late = 0.5 -> reward = 1/0.5 = 2.0
    erp_data = {
        "P300": {"amp": 10.0},
        "LPP_late": {"amp": 0.5}
    }
    
    # Both applicable (hack: force applicability for test)
    # Actually, let's use a trial type where both are applicable?
    # FluentEngage: Congruent
    # ClearAndLetGo: Incongruent + Negative
    # They are mutually exclusive by definition!
    # So we need to mock is_trial_type_applicable or use a different set of modules.
    
    # Let's mock
    m1.is_trial_type_applicable = lambda v, a: True
    m2.is_trial_type_applicable = lambda v, a: True
    
    reward = modular_reward.calculate_total_reward(Sentiment.POSITIVE, Sentiment.POSITIVE, erp_data)
    
    # m1 reward: 10.0
    # m2 reward: 2.0 (approx)
    # average: 6.0 (approx)
    assert reward == pytest.approx(6.0, rel=1e-4)

def test_protocol_factory():
    pA = ProtocolFactory.create_protocol("A")
    assert len(pA.modules) == 1
    assert isinstance(pA.modules[0], FluentEngageModule)
    
    pD = ProtocolFactory.create_protocol("D")
    assert len(pD.modules) == 2
    names = [m.name for m in pD.modules]
    assert "BidirectionalPositiveLPP" in names
    assert "BidirectionalNegativeLPP" in names

def test_protocol_factory_unknown():
    with pytest.raises(ValueError):
        ProtocolFactory.create_protocol("Unknown")
