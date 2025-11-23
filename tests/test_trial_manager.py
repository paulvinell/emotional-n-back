import pytest
from emotional_n_back.games.eeg.eeg_stroop.trial_manager import TrialManager, CalibrationPhase, RegularPhase
from emotional_n_back.games.eeg.eeg_stroop.reward import Reward

def test_trial_manager_initialization():
    tm = TrialManager(initial_calibration_trials=5)
    assert tm.trial_num == 0
    assert tm.score == 0
    assert tm.scoreable_trial_num == 0
    assert isinstance(tm.current_phase, CalibrationPhase)
    assert tm.current_phase.trial_idx == 0

def test_increment_trial_num():
    tm = TrialManager()
    tm.increment_trial_num()
    assert tm.trial_num == 1
    assert tm.current_phase.trial_idx == 1

def test_check_calibration_start():
    tm = TrialManager()
    
    # Not calibrated yet
    tm.check_calibration_start(is_calibrated=False)
    assert isinstance(tm.current_phase, CalibrationPhase)
    
    # Calibrated now
    tm.check_calibration_start(is_calibrated=True)
    assert isinstance(tm.current_phase, RegularPhase)
    assert tm.current_phase.trial_idx == 0
    
    # Increment trial in regular phase
    tm.increment_trial_num()
    assert tm.current_phase.trial_idx == 1

def test_on_trial_complete_calibrated():
    tm = TrialManager()
    tm.current_phase = RegularPhase(trial_idx=0)
    
    # Success
    tm.on_trial_complete(Reward.SUCCESS, is_calibrated=True)
    assert tm.score == 1
    assert tm.scoreable_trial_num == 1
    
    # Failure
    tm.on_trial_complete(Reward.FAILURE, is_calibrated=True)
    assert tm.score == 1 # Score shouldn't decrease
    assert tm.scoreable_trial_num == 2
    
    # None
    tm.on_trial_complete(Reward.NONE, is_calibrated=True)
    assert tm.score == 1
    assert tm.scoreable_trial_num == 3

def test_on_trial_complete_not_calibrated():
    tm = TrialManager()
    # In CalibrationPhase by default
    
    tm.on_trial_complete(Reward.SUCCESS, is_calibrated=False)
    assert tm.score == 0
    assert tm.scoreable_trial_num == 0

def test_get_state():
    tm = TrialManager()
    tm.current_phase = RegularPhase(trial_idx=5)
    tm.score = 2
    tm.scoreable_trial_num = 3
    
    state = tm.get_state()
    
    assert state.display_text == "Trial 6"
    assert state.score == 2
    assert state.scoreable_trial_num == 3
    assert state.is_calibrating == False
