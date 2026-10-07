# CodeTrack causal training status

## Current baseline

- GOLA-DINOv2-B: PR 77.19, SR 61.79, NPR 73.76.
- Best existing SATR: PR 77.34, SR 61.76. It does not pass the SR requirement.

## What was changed

The training code now has an opt-in causal q target. For token `i`, it replaces only that
token with the clean counterpart and measures the reduction in the frozen GOLA head output
error. The target is therefore tracking impact, rather than the synthetic injector mask.
The target is sparse: a token must provide at least `causal_q_target_min_ratio` of the largest
positive gain in its frame. Clean frames produce an all-zero target.

The deterministic `grid` H layout is also available. It uses 64 overlapping local checks on
the 16x16 search lattice, 12 variables per check, column degree 3, rank 64, and no random free
edges. The historical seeded random layout remains the default for backward compatibility.

## Evidence

1. Real 10-sequence causal target wiring:
   `_probe/causal_target_grid.json`
   - target shape 10x256, finite;
   - target mean 0.0341;
   - nonzero fraction 6.80%;
   - diagnosis gradient finite and nonzero.

2. Causal q generalisation with three training corruption locations and one held-out location:
   `_probe/causal_q_generalization_grid.json`
   - best held-out AUROC 0.808;
   - training AUROC about 0.99;
   - later iterations overfit, so early stopping is required.

3. Causal-mask SATR with learned routing and q>=0.6 abstention on the fixed recovery probe:
   `_probe/recovery_grid_causal_q06.json`
   - learned feature gain 0.00154;
   - healthy drift 0.00129;
   - learned route is usable but still too destructive for immediate full training.

4. Random-H comparison with the same causal-mask protocol:
   `_probe/recovery_causal_mask_q06.json`
   - learned gain 0.00130;
   - healthy drift 0.00053.

## Decision

The causal target is the correct supervision concept, but the current SATR residual direction is
not yet safe enough for a full LasHeR run. Do not start full training from these probes. The next
candidate should keep two Tanner rounds, use early-stopped causal q, and train/validate an explicit
identity penalty until healthy drift is below the acceptance threshold while retaining positive
learned-route gain. Motion and memory should only be reintroduced after this spatial gate passes
on the same sequence split.
