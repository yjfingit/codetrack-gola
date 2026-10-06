# CodeTrack Research Ratchet

This file records the rules used for autonomous experiment progression. The thresholds are
fixed by the repository validators and are not relaxed to make a run pass.

1. Every hypothesis starts with a deterministic, fixed-real-batch causal probe.
2. A mechanism advances only if the probe changes the intended metric in the intended direction.
3. The first training allocation is four isolated GPU arms with a small, identical update budget.
4. A smoke arm advances to 400 updates only when its terminal window improves over its initial
   window and no NaN, OOM, or calibration failure occurred.
5. S1 passes only with `gain_total > 0.005`, `q_std > 0.001`, and diagnosis AUROC above chance.
6. S2-S4 stay blocked until S1 passes. Training completion alone is never counted as success.
7. Every run keeps its config, command, log, checkpoint, and machine-readable summary.
8. Failed hypotheses remain recorded; code changes require a fresh structural preflight (104/104).
9. GPU ownership for this campaign is 1,2,3,4. GPUs 0,5,6 are not used.
10. A failed launch is diagnosed from its existing log and process handle before any relaunch.
