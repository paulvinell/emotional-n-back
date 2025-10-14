from enum import Enum, auto


class GameState(Enum):
    ESTIMATING_FS = auto()
    ESTIMATION_COMPLETE = auto()
    WAIT_EEG = auto()
    PREPARE_TRIAL = auto()
    INTRO = auto()
    STIMULUS = auto()
    RESPONSE = auto()
    FEEDBACK = auto()
    FINAL_SCREEN = auto()
