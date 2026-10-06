# S1 Decision: Partial / NO-GO

Date: 2026-10-04

## Decision

S1 is not robust enough to unlock S2. The selected 50-update recovery-only recipe passed all
fixed-batch hard gates for 3 of 4 sampler seeds. The failed seed is retained as a failure; the
mean is not used to override a per-seed hard gate.

| seed | q AUROC | q std | fixed damage gain | decision |
|---:|---:|---:|---:|:---|
| 42 | 0.735613 | 0.006582 | 0.006887 | PASS |
| 43 | 0.735353 | 0.006200 | 0.002403 | FAIL |
| 44 | 0.735416 | 0.005708 | 0.005332 | PASS |
| 45 | 0.736410 | 0.006826 | 0.006127 | PASS |

Hard gates: AUROC > 0.5, q std > 0.001, fixed damage gain > 0.005.

## Ratchet Evidence

- Oracle routing at initialization did not improve recovery, so q calibration alone was not the
  bottleneck.
- A fixed-batch capacity study showed that the denoiser and joint recovery path can learn large
  improvements, ruling out insufficient architecture capacity.
- The grid mixin previously changed only the optimizer default LR while per-parameter groups kept
  1e-4. The list-path operator and LR generator were fixed and verified against the resolved config.
- The criterion builder previously dropped `lambda_gain`; it is now passed through and verified.
- Split learning rates prevented a high recovery LR from collapsing the diagnosis head.
- Four 400-update replications overfit: training-batch gain stayed high, but fixed-batch damage
  gain fell to 0.002441-0.003757 and q std collapsed below 0.001 for every seed.
- A 50/100/150/200-update curve with all non-recovery modules frozen showed that 50 updates was
  the best generalization point. Longer training was not more robust.
- The final aggressive 50-update confirmation improved 3 of 4 seeds, but seed 43 remained below
  the recovery threshold.

## Next Allowed Step

Do not start S2-S4. The next S1 hypothesis must target cross-seed recovery generalization, using
the retained fixed-batch probes as the selection gate. Thresholds must not be relaxed.
