Protocol A — “Fluent Engage” (Boost P300 on congruent targets)

Goal: Increase efficient, goal-directed attention and reduce rumination by reinforcing clean stimulus evaluation when the world makes sense.

Site: Pz

Contingency: only congruent trials (e.g., pos+pos, neg+neg, neu+neu) counted for reward.

Metric: mean P300 amplitude (300–500 ms) and optional latency (time-to-peak 280–520 ms).

Reinforcement rule: reward if (Amplitude > threshold) and (Latency < threshold). Weight amplitude 70%, latency 30%.

Feedback mapping: raise score bar / speed up a calm animation after each block if the block’s composite index exceeds criterion.

Shaping: gradually tighten latency criterion (−5 ms) once amplitude is stable; then inch amplitude criterion up (+0.1 µV).

Block composition: 60–70% congruent, 30–40% incongruent (incongruent withheld from reward to keep the signal “pure”).

Why it works: larger/earlier P300 tracks fluent evaluation and resource allocation; reinforcing it on congruent trials trains a clean “engage” mode.

Protocol B — “Clear & Let Go” (Reduce late LPP on negative incongruent)

Goal: Soften over-processing of emotionally conflicting negatives (a common driver of stickiness).

Site: Pz

Contingency: negative incongruent trials only (e.g., angry face + non-angry audio, or vice versa).

Metric: LPP-late amplitude (600–900 ms).

Reinforcement rule: reward lower-than-threshold LPP-late while accuracy and RT remain within limits (e.g., error ≤20% and median RT not slower than baseline +1 SD).

Feedback mapping: “cool-down meter” fills more when LPP-late is smaller and performance intact.

Shaping: reduce threshold by 0.1 µV after two successful blocks; if accuracy drops, freeze threshold for a block.

Why it works: late LPP indexes sustained emotional elaboration; down-training specifically when emotion conflicts with task demands trains disengagement without harming performance.

Protocol C — “Fast Conflict Detect” (Enhance N200 on incongruent)

Goal: Sharpen early conflict monitoring so resolution happens earlier and cleaner.

Site: Cz

Contingency: incongruent trials only.

Metric: N200 amplitude (mean negativity 180–280 ms). Optional peak latency (earlier is better).

Reinforcement rule: reward more negative N200 (relative to baseline) and/or earlier peak provided errors don’t rise.

Feedback mapping: brief auditory token after each good trial + block score.

Shaping: once amplitude stabilizes, introduce a small latency bonus (e.g., 10% of score if peak < baseline −10 ms).

Why it works: N200 reflects ACC-linked conflict detection; strengthening it can reduce later processing costs (improves P300 later, often reduces RT variance).

Protocol D — “Bidirectional Emotional Control” (Differential LPP: up for positive-congruent, down for negative-incongruent)

Goal: Train flexible affect: savor coherent positives, de-amplify conflicted negatives.

Site: Pz

Contingency: two tracked channels inside one session:

Channel 1 (reward up): positive-congruent, target LPP-early (400–600 ms) ↑

Channel 2 (reward down): negative-incongruent, target LPP-late (600–900 ms) ↓

Reinforcement rule: composite = z(LPP_pos_early) − z(LPP_neg_late). Reward when composite > threshold, with minimum accuracy constraints.

Feedback mapping: single bar that grows with the composite differential, to avoid juggling two meters.

Shaping: first stabilize each channel in isolation (2–3 blocks each), then run mixed blocks using the composite.

Why it works: teaches the system to engage with coherent positives and release conflicted negatives—i.e., controlled approach/withdrawal.

Protocol E — “Efficient Processing Index” (High P300, Low LPP-late on task-relevant trials)

Goal: General “engage → evaluate → disengage” efficiency on whatever trials matter most to you (e.g., all emotional targets).

Site: Pz

Contingency: user-defined “task-relevant” trials (e.g., any emotional target, regardless of congruence).

Metric: Index = z(P300 amplitude 300–500 ms) − z(LPP-late 600–900 ms).

Reinforcement rule: reward when Index > threshold and accuracy/RT within limits.

Feedback mapping: progress ring fills faster with higher index.

Shaping: once stable, add a small bonus for earlier P300 peak.

Why it works: encodes a compact “on-then-off” control policy: strong evaluation but minimal perseveration.