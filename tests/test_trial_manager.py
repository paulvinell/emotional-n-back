import pytest
from emotional_n_back.games.eeg.eeg_stroop.trial_manager import TrialManager
from emotional_n_back.games.eeg.eeg_stroop.reward import Reward

def test_trial_manager_initialization():
    tm = TrialManager(initial_calibration_trials=5)
    assert tm.trial_num == 0
    assert tm.score == 0
    assert tm.scoreable_trial_num == 0
    assert tm.initial_calibration_trials == 5
    assert tm.regular_trials_start_idx is None

def test_increment_trial_num():
    tm = TrialManager()
    tm.increment_trial_num()
    assert tm.trial_num == 1

def test_check_calibration_start():
    tm = TrialManager()
    
    # Not calibrated yet
    tm.check_calibration_start(is_calibrated=False)
    assert tm.regular_trials_start_idx is None
    
    # Calibrated now
    tm.trial_num = 3
    tm.check_calibration_start(is_calibrated=True)
    assert tm.regular_trials_start_idx == 3
    
    # Still calibrated, shouldn't change
    tm.trial_num = 4
    tm.check_calibration_start(is_calibrated=True)
    assert tm.regular_trials_start_idx == 3

def test_on_trial_complete_calibrated():
    tm = TrialManager()
    
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
    
    tm.on_trial_complete(Reward.SUCCESS, is_calibrated=False)
    assert tm.score == 0
    assert tm.scoreable_trial_num == 0

def test_get_state():
    tm = TrialManager()
    tm.trial_num = 5
    tm.score = 2
    tm.scoreable_trial_num = 3
    tm.regular_trials_start_idx = 2
    
    state = tm.get_state(is_calibrated=True)
    
    assert state.trial_num == 5
    assert state.score == 2
    assert state.scoreable_trial_num == 3
    assert state.is_calibrating == False
    assert state.regular_trials_start_idx == 2
