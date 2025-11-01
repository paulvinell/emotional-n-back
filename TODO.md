# Rewards
- We want different reward functions for different ERP components.
- We probably want different rewards for incongruent and congruent trials.
- We want some notion of averaged rewards over a sliding window of trials and something that indicates amplitude and latency separately.

# Miscellaneous thoughts
Consider what happens when there are multiple peaks? Pick first one?

Add rewards for low/high variability?

Add an elo system for the user and sets of stimuli 

We are no longer marking congruent/incrongruent trials in the game.

Add tests that compare ERP processing with another library like MNE?

include ssvep training?

# Modular rewards
For continuous rewards, keep one long average (+ stddev) and one short average (+ stddev) for each component of interest. You can then visualize the short term trend against the long term trend.

Recalibration should respect how often a module is used. If a module is not used often, it should recalibrate less often.

The RewardModule class should have its own z-scoring for each set of reward modules.