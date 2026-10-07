# E0017: Causal motion and moving search-crop coordinates

- Date: 2026-10-07.
- Hypothesis: an unreprojected Kalman posterior and delayed observations prevent continuous-clip training from learning a deployable motion prior. Fix geometry and timing before changing correction capacity.
- Starting implementation: `8058fb5` (snapshot of inherited source changes).
- Change: optional `search_crop_params` moves the posterior, velocities and covariance between normalised crop coordinates. In this mode the prior advances once, and the current accepted head box updates the posterior after recovery. Sequence reset clears innovation statistics.
- Scope: new causal replay probe opts in. The existing production evaluator does not yet provide crop parameters; deploying this interface also needs per-sequence state ownership and the evaluator's accepted, Hann-window-decoded box.
- Checks: `tools/verify_motion_coordinates.py`: four CPU physical-coordinate/transition/reset witnesses pass. `tools/preflight_acceptance.py`: 97/97 checks pass on GPU4; this is the current validator count, not the historical 104.
- PR/SR: none; this is a state-correctness change, not a performance claim.
- Checkpoint: none.
- Next: E0018 tests train-sequence-held-out recovery with baseline-predicted crops, real template foreground masks, online template updates and mid-video windows.

```bash
source scripts/00_env.sh
"$PYTHON" tools/verify_motion_coordinates.py
CUDA_VISIBLE_DEVICES=4 "$PYTHON" tools/preflight_acceptance.py
```
