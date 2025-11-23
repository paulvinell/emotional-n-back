import pytest
from emotional_n_back.games.eeg.eeg_stroop.reward_modules import (
    ModularReward,
    GenericRewardModule,
    TrialFilter,
    FeatureExtractor,
    Sentiment,
)
from emotional_n_back.games.eeg.eeg_stroop.protocols import ProtocolFactory

def create_fluent_engage_module():
    return GenericRewardModule(
        name="FluentEngage",
        trial_filter=TrialFilter(congruence=True),
        extractor=FeatureExtractor(component="P300", metric="amp"),
    )

def create_clear_and_let_go_module():
    return GenericRewardModule(
        name="ClearAndLetGo",
        trial_filter=TrialFilter(congruence=False, sentiments={Sentiment.NEGATIVE}),
        extractor=FeatureExtractor(component="LPP_late", metric="amp", inverse=True),
    )

def test_enable_disable_module():
    m1 = create_fluent_engage_module()
    m2 = create_clear_and_let_go_module()
    modular_reward = ModularReward(modules=[m1, m2])
    
    assert m1.enabled
    assert m2.enabled
    
    modular_reward.disable_module("FluentEngage")
    assert not m1.enabled
    assert m2.enabled
    
    modular_reward.enable_module("FluentEngage")
    assert m1.enabled

def test_set_active_modules():
    m1 = create_fluent_engage_module()
    m2 = create_clear_and_let_go_module()
    modular_reward = ModularReward(modules=[m1, m2])
    
    modular_reward.set_active_modules(["ClearAndLetGo"])
    assert not m1.enabled
    assert m2.enabled
    
    modular_reward.set_active_modules(["FluentEngage", "ClearAndLetGo"])
    assert m1.enabled
    assert m2.enabled

def test_disabled_module_no_contribution():
    m1 = create_fluent_engage_module() # P300
    # Note: enable_z_scoring defaults to True now, but for this test we want raw values
    # or we need to mock the z-scorer.
    # Let's disable z-scoring on the module for this test to verify aggregation logic
    m1.z_scorer = None 
    
    modular_reward = ModularReward(modules=[m1])
    
    erp_data = {"P300": {"amp": 10.0}}
    
    # Enabled
    # Congruent trial required for FluentEngage
    reward = modular_reward.calculate_total_reward(Sentiment.POSITIVE, Sentiment.POSITIVE, erp_data)
    assert reward == 10.0
    
    # Disabled
    modular_reward.disable_module("FluentEngage")
    reward = modular_reward.calculate_total_reward(Sentiment.POSITIVE, Sentiment.POSITIVE, erp_data)
    assert reward == 0.0

def test_averaging_logic():
    m1 = create_fluent_engage_module()
    m1.z_scorer = None # Disable z-scoring for deterministic testing
    
    m2 = create_clear_and_let_go_module()
    m2.z_scorer = None
    
    modular_reward = ModularReward(modules=[m1, m2])
    
    # P300 = 10, LPP_late = 0.5 -> reward = 1/0.5 = 2.0
    erp_data = {
        "P300": {"amp": 10.0},
        "LPP_late": {"amp": 0.5}
    }
    
    # Let's mock is_trial_type_applicable to force both to be active
    # (Since they are mutually exclusive in reality)
    m1.is_trial_type_applicable = lambda v, a: True
    m2.is_trial_type_applicable = lambda v, a: True
    
    reward = modular_reward.calculate_total_reward(Sentiment.POSITIVE, Sentiment.POSITIVE, erp_data)
    
    # m1 reward: 10.0
    # m2 reward: -0.5 (linear inverse: -0.5)
    # average: (10.0 - 0.5) / 2 = 4.75
    assert reward == pytest.approx(4.75, rel=1e-4)

def test_protocol_factory():
    pA = ProtocolFactory.create_protocol("A")
    assert len(pA.modules) == 1
    assert pA.modules[0].name == "FluentEngage"
    
    pD = ProtocolFactory.create_protocol("D")
    assert len(pD.modules) == 2
    names = [m.name for m in pD.modules]
    assert "BidirectionalPositiveLPP" in names
    assert "BidirectionalNegativeLPP" in names

def test_protocol_factory_unknown():
    with pytest.raises(ValueError):
        ProtocolFactory.create_protocol("Unknown")
