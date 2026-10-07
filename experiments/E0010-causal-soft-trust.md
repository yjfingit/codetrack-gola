# E0010: Causal soft-route and trust-region probe

## Question

Does the corrected training path avoid the previous failure where hard Top-K and joint q/SATR
updates produced a large training gain but no held-out recovery?

## Changes

- q supervision uses the frozen GOLA head's per-token causal tracking-impact target.
- Training uses a dense soft route; deployment remains hard Top-K.
- q is frozen while SATR learns its correction direction.
- A residual trust region clips each token update to 5% of the received token norm.
- Clean paired frames are explicit q=0 examples; natural frames are not assigned a fake label.

## Run

```text
CUDA_VISIBLE_DEVICES=4 python tools/continuous_clip_probe.py \
  --steps 40 --diag-steps 100 --clip-length 4 \
  --output _probe/continuous_clip_causal_trust40.json \
  --save-checkpoint _probe/continuous_clip_causal_trust40.safetensors \
  --abstain-threshold 0.25
```

## Result

- Causal target strong-positive fraction: 4.46% of tokens.
- Training tracking gain: +0.223 loss units at step 39.
- Learned-route held-out feature gain: `-1.26e-6` (effectively zero).
- Healthy-token drift: `1.15e-4` (small but non-zero).
- q mean: 0.193; hard-route active fraction: 2.36%.

## Decision

The numerical instability is fixed and the route no longer writes broadly, but the learned
repair direction still does not generalize. This is a diagnostic success and a model-training
failure. Do not run full LasHeR-test from this checkpoint. The next experiment should train the
repair direction on held-out corruption locations/sequences and select by learned-route gain,
not by training tracking loss.
