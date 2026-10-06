# CodeTrack Autonomous Experiment Status

Date: 2026-10-04

## Execution

- All training grids used local RTX 4090 GPUs 1, 2, 3, and 4. GPUs 0, 5, and 6 were left to
  the other workload.
- Every long run used an independent tmux session, a per-arm log, a machine-readable plan, and
  a checkpoint directory. No run was restarted after an observation timeout.
- All active training and probe processes are finished; GPU 1-4 are released.

## Engineering Gates

- DDP syndrome calibration, staged update stopping, and initialization flow are repaired.
- Mixin list-index replacement and effective per-parameter LR routing are repaired and witnessed.
- `lambda_gain` is now passed through the criterion builder and witnessed.
- q sign and q prior semantics are corrected: only syndrome evidence is sign-inverted; the prior
  bias is preserved.
- TIR-only corruption now preserves RGB auxiliary evidence, matching the recovery contract.
- Diffusion-disabled probes and K64 probes now load and report their effective configuration.
- Final structural preflight: `104/104` checks pass (`_probe/final_preflight_h14.log`).

## Scientific Decision

S1 remains **Partial / NO-GO**. Do not unlock S2.

- The 50-update H14 confirmation passed fixed diagnosis/recovery gates for 3/4 seeds, but seed43
  failed (`gain_damage=0.002403`).
- Four 400-update replications overfit: fixed damage gain fell to `0.002441-0.003757` and q
  variance collapsed.
- Recovery-only capacity on the exact fixed batch improved by about `0.028-0.029`, proving the
  refiner can learn when the damage and route are fixed.
- Recovery checkpoint soups passed the center probe (`gain_damage=0.007760`) but failed spatial
  robustness: top-left/top-right/bottom-right gains were negative.
- Correcting q prior semantics reduced peripheral damage, but sparse routing alone produced only
  `0.00018-0.00046` gain under random training.
- Explicit rank loss restored diagnosis quality (fixed AUROC about `0.625-0.735`, q std above
  `0.006`), but did not produce recovery gain above `0.005`.
- TIR-only corruption, centered corruption, diffusion removal, aligned token ratio, frozen
  diagnosis, and recovery LR/gain sweeps all remained below the fixed recovery gate. The best
  recent fixed gain was `0.004463` before the final push and then regressed.

The current evidence supports a new research hypothesis around the recovery target/conditioning
contract, not another blind LR sweep. Existing thresholds were not relaxed and S2-S4 were not
started.
