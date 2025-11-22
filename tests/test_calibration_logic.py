import pytest
from unittest.mock import MagicMock, patch
from emotional_n_back.games.eeg.eeg_stroop.game import EEGStroopGame
from emotional_n_back.games.eeg.eeg_stroop.erp_adapter import ErpUpdate
from emotional_n_back.games.eeg.eeg_stroop.reward import Reward

class MockErpAdapter:
    def __init__(self, *args, **kwargs):
        self.effective_fs = 256.0
        self.continuous_fs_est = 256.0
        self.eeg_started = MagicMock()
        self.eeg_started.is_set.return_value = True

    def start(self): pass
    def shutdown(self): pass
    def ingest_event(self, code): pass
    def poll_update(self, code, timeout=0): return None

@pytest.fixture(autouse=True)
def mock_pygame_and_beep():
    with patch('emotional_n_back.games.eeg.eeg_stroop.game.make_beep'), \
         patch('emotional_n_back.games.eeg.eeg_stroop.game.pygame'):
        yield

def test_calibration_with_artifacts():
    # Mock ErpAdapter to avoid networking
    with patch('emotional_n_back.games.eeg.eeg_stroop.game.ErpAdapter', MockErpAdapter):
        game = EEGStroopGame(initial_calibration_trials=5, recalibration_interval=5)
        
        # Simulate 5 clean trials
        for i in range(5):
            game.modular_reward.recalibrate_modules() # Called at start of trial
            update = ErpUpdate(
                code=f"trial_{i}",
                clean=True,
                amp=10.0,
                lat=0.3,
                raw={}
            )
            game._process_erp_update(update)
            
        # Recalibrate one last time to process the data from the 5th trial
        game.modular_reward.recalibrate_modules()
        
        # Should be calibrated now
        assert game.modular_reward.is_calibrated()
        
def test_calibration_with_mixed_artifacts():
    with patch('emotional_n_back.games.eeg.eeg_stroop.game.ErpAdapter', MockErpAdapter):
        game = EEGStroopGame(initial_calibration_trials=5, recalibration_interval=5)
        
        # 3 clean trials
        for i in range(3):
            game.modular_reward.recalibrate_modules()
            update = ErpUpdate(code=f"trial_{i}", clean=True, amp=10.0, lat=0.3, raw={})
            game._process_erp_update(update)
            
        game.modular_reward.recalibrate_modules()
        assert not game.modular_reward.is_calibrated()
        
        # 2 artifact trials
        for i in range(3, 5):
            game.modular_reward.recalibrate_modules()
            update = ErpUpdate(code=f"trial_{i}", clean=False, amp=None, lat=None, raw={})
            game._process_erp_update(update)
            
        game.modular_reward.recalibrate_modules()
        assert not game.modular_reward.is_calibrated()
        
        # 2 more clean trials (total 5 clean)
        for i in range(5, 7):
            game.modular_reward.recalibrate_modules()
            update = ErpUpdate(code=f"trial_{i}", clean=True, amp=10.0, lat=0.3, raw={})
            game._process_erp_update(update)
            
        game.modular_reward.recalibrate_modules()
        assert game.modular_reward.is_calibrated()
