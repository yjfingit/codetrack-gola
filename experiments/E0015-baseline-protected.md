# E0015: Baseline-protected inference

The E0013 candidate was evaluated with `accept_enabled=true`: a decoded frame is committed only
when the original GOLA score-map peak does not decrease. Each sequence used its own isolated view
and ran in strict frame order. The run used commit `b0642c4` and the `causal_clip_accept` mixin.

## Result

- PR: **57.926%**.
- SR: **47.964%**.
- NPR: **55.338%**.
- Paired GOLA: PR 58.075%, SR 48.153%, NPR 55.675%.

The gate recovers a small amount relative to E0014 but remains below GOLA. It is not the final
model. The sequence-level damage is reduced only slightly, so the next change must improve q/SATR
selectivity rather than add another global score gate.

## Parallel correctness

The 10 sequences were run with two independent processes per GPU batch. Every sequence had a
private `LasHeR_single` view, and the per-sequence metrics were stable across reruns. The earlier
shared-view race was fixed before this run.
