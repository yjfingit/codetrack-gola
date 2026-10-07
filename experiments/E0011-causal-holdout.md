# E0011: Causal clip training with strict sequence holdout

## Question

Does the revised recipe generalize when both the diagnosis and SATR modules are trained only on
seven clips and the checkpoint is selected on three completely unseen clips?

## Protocol

```text
CUDA_VISIBLE_DEVICES=4 python tools/continuous_clip_probe.py \
  --steps 100 --diag-steps 200 --clip-length 4 --val-count 3 \
  --output _probe/continuous_clip_holdout100_v2.json \
  --save-checkpoint _probe/continuous_clip_holdout100_v2.safetensors \
  --abstain-threshold 0.25
```

Training clips are the first seven entries of the fixed 10-clip manifest; validation clips are
`theleftestrunningboy`, `rightbackcup`, and `whitebetweenblackandblue`. The validation set is
never used for gradients. The best checkpoint is selected by validation tracking gain plus
bad-token recovery and a healthy-drift penalty.

## Result

- Causal target strong-positive fraction: 4.46%.
- Held-out q-AUROC: **0.876**.
- Held-out bad-token feature recovery gain: **+0.000380**.
- Held-out tracking-loss gain: **+0.06249**.
- Held-out healthy-token drift: **0.000180**.
- Active route fraction: 6.90%.
- Best step: 75 of 100.
- Checkpoint: `_probe/continuous_clip_holdout100_v2.safetensors`.

## Decision

Retain this recipe as the first promising candidate. It is evidence that causal q supervision and
separate SATR fitting can generalize beyond the training clips. It is not yet a benchmark result:
the next gate is a second sequence split or seed, followed by the 10-clip tracking comparison.
