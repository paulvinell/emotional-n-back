# EEG Stroop Game TODO

This document outlines the current architecture of the EEG-integrated Stroop game and provides a list of potential future improvements.

## Current Architecture

The system is composed of two main modules: `eeg_stroop.py` (the game itself) and `eeg/erp.py` (the ERP processing engine).

### `eeg_stroop.py` - The Game

-   **`EEGStroopGame`**: This class manages the game logic, display, and stimulus presentation. It inherits from the standard `SentimentStroopGame` to reuse the UI and basic structure.
-   **Game Flow**: For each trial, the game displays a face (visual stimulus) and plays a word (audio stimulus).
-   **ERP Integration**:
    -   It runs an `OscErpServer` in a background thread to listen for incoming single-channel EEG data from an external application.
    -   When the audio stimulus is presented, the game directly calls `erp_server.ingest_event()` to notify the ERP engine, passing a unique, trial-specific event code (e.g., `"congruent_12"`).
    -   It uses a thread-safe callback (`_handle_erp_update`) to receive processed ERP data from the engine. Updates are stored in a dictionary keyed by the unique event code.
-   **Reward Mechanism**:
    -   After a trial, the game calls `_get_reward()`, which waits for the specific ERP result for that trial's event to become available.
    -   The current reward logic is a simple proof-of-concept: if the P300 amplitude is above a hard-coded threshold, the trial is considered a "success."
    -   The player receives audio feedback (success/failure beep) based on the outcome.

### `eeg/erp.py` - The ERP Engine

-   **`OscErpServer`**: A class that runs a `ThreadingOSCUDPServer` to receive raw EEG samples via `/eeg` OSC messages. It manages the data ingestion queue and the processing lifecycle.
-   **`StreamEpocher`**: The core processing class. It takes the raw EEG stream and:
    1.  Applies a zero-phase band-pass filter.
    2.  Maintains a ring buffer of recent EEG data.
    3.  When an event is ingested, it waits for a complete data window (`tmin` to `tmax`) around the event timestamp.
    4.  Extracts the epoch, performs baseline correction, and updates a running average of the ERP for that event type.
    5.  Calculates the amplitude and latency for pre-configured neural components (P1, N1, N200, P300, LPP).
-   **Communication**: The engine communicates back to the game by invoking a callback (`on_update`) with a dictionary containing the latest running ERP average and component scores for a given event.

## Future Improvements / TODO

### Core Gameplay & Reward Logic

-   [ ] **Sophisticated Reward Conditions**: The current P300 threshold logic is very basic. Implement more meaningful reward criteria:
    -   Reward based on the *difference* in P300 amplitude between congruent and incongruent trials.
    -   Incorporate other components (e.g., N200 for conflict monitoring, LPP for sustained attention).
    -   Use a moving baseline or adaptive threshold for rewards.
-   [ ] **Dynamic Difficulty**: Adjust game parameters (e.g., stimulus speed, ISI) based on ERP-derived metrics of player engagement or cognitive load.

### Configuration & Data Handling

-   [ ] **External Configuration**: Move hard-coded parameters (`p300_threshold`, `fs_fallback`, ERP component windows) to a configuration file (e.g., YAML or TOML) or expose them as command-line arguments in `main.py`.
-   [ ] **Robust Data Logging**: The ERP engine currently prints JSON to stdout. Implement proper file logging for all ERP updates (e.g., to a `.jsonl` file) to make data collection and offline analysis easier. This could be a flag on the `eeg-stroop` command.
-   [ ] **Save/Load Game State**: Allow pausing and resuming the game, saving the state of the `StreamEpocher`'s running averages.

### Visualization & User Interface

-   [ ] **In-Game ERP Visualization**: Add an optional real-time plot to the game window. This could show:
    -   The running ERP for different conditions (e.g., congruent vs. incongruent).
    -   A bar chart of the latest component amplitudes.
-   [ ] **Better Feedback**: Go beyond simple beeps. Use the feedback phase of the trial to visually indicate the strength of the detected ERP response.

### Architecture & Extensibility

-   [ ] **Abstract the EEG Game Logic**: Create a base `EEGGame` class that encapsulates the `OscErpServer` integration, event handling, and reward loop. This would make it much easier to create new games that are also driven by ERPs.
-   [ ] **Multi-channel Support**: The `erp.py` engine is currently single-channel. A significant future step would be to extend it to handle multi-channel EEG data, which would allow for more advanced spatial analysis.
-   [ ] **More Robust Concurrency**: The `_get_reward` function currently has a simple polling loop with a fixed timeout. This could be replaced with a more robust inter-thread signaling mechanism, like `threading.Event` or `queue.Queue`, to wait for results.
