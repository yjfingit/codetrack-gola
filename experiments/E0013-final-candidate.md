# E0013: Standalone checkpoint and all-clip probe

This rerun uses commit `4c900e7`, which persists `H` and `template_pool` in addition to the
diagnosis, SATR, and motion parameters. A separate model construction successfully loaded all 53
checkpoint tensors, so the artifact is reproducible outside the training process.

## Results

- Held-out q-AUROC: 0.876.
- All 10 clips q-AUROC: **0.7465**.
- All 10 bad-token feature recovery gain: **+0.000542**.
- All 10 tracking-loss gain: **+0.04577**.
- All 10 healthy-token drift: **0.000109**.
- Checkpoint: `_probe/continuous_clip_final_candidate.safetensors`.

The candidate is retained for the next production-style 10-sequence comparison. It is not yet a
LasHeR PR/SR result.
