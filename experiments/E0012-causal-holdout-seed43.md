# E0012: Independent seed confirmation

E0011 was repeated with `--seed 43` and the same strict seven-train / three-validation split.
The run used commit `c7fb232` and the same 5% SATR trust region.

## Result

- Held-out q-AUROC: **0.8527**.
- Held-out bad-token feature recovery gain: **+0.000372**.
- Held-out tracking-loss gain: **+0.05529**.
- Held-out healthy-token drift: **0.000180**.
- Best step: 75 of 100.
- Checkpoint: `_probe/continuous_clip_holdout100_seed43.safetensors`.

The signs and magnitudes agree with E0011. The result is still a probe metric, not a LasHeR-test
PR/SR result; the next gate is an actual 10-clip tracking comparison using the saved checkpoint.
