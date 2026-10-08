# H38 Decision: DROP

## Hypothesis

Make `s` a check-health posterior (`1 - H @ error`) so the existing negative vote
`q_logits = -H^T s + prior` agrees with the syndrome target.

## Run

- One seed: 42
- Diagnosis/q only; refiner, denoiser, motion and memory frozen/disabled by the S1 setup
- 100 optimizer updates (`800` micro-batches, accumulation 8)
- TIR-only `quadrant_mix`, 16/256 damaged tokens, K=64
- Structural preflight: `104/104`

## Fixed real-batch result

| checkpoint | q std | q AUROC | Top-K overlap | syndrome std |
|---|---:|---:|---:|---:|
| H33 q-pretrain baseline | 0.007369 | 0.679866 | 0.478747 | 0.254266 |
| H38 health-syndrome | 0.000309 | 0.443685 | 0.122344 | 0.002586 |

The H38 training log also showed q std collapsing from about `0.007` to `0.0002-0.0003`.

## Decision

DROP. The target-only sign change worsens diagnosis calibration and is not promoted to
refiner or multi-seed training. The code change was reverted. The next hypothesis must
alter the message/abstention contract or the pre-aggregation evidence path, not repeat a
syndrome sign sweep.
