# E0018: Causal crops versus GT-centred training control

- Date: 2026-10-07.
- Status: prepared, not yet a result.
- Hypothesis: tracking-loss gains in GT-centred short clips overestimate useful correction under predicted crops. A matched causal replay test will decide whether the current q/SATR recipe deserves any additional training.
- Source: implementation commit is recorded by the launcher before execution. Starting snapshot `8058fb5`.
- Data: versioned `experiments/train10.txt`, LasHeR-train only. Two contiguous 4-frame windows at 25% and 75% of each video; seven train sequences, last three entirely held out. One seed, 42.
- Arm A: frozen GOLA baseline trajectory chooses later input crops. Only window initialisation receives GT. Template foreground masks, Hann-window postprocessing and online template update at score > 0.84 use repository implementations.
- Arm B: same windows, seed, loss, updates and state interface; input crops use current GT as an explicitly labelled oracle control.
- Frozen: DINOv2, LoRA, GOLA head, H and template pool. Two stages: 120 q updates from token-replacement tracking impact, then 100 SATR/motion-prior updates with learned q frozen. K=4, lr=5e-4, residual norm bound=5%, q threshold=.25; inherited representative settings, no search.
- Loss: tracking + .5 harmful-tracking margin + .2 recovery + healthy preservation. q is trained separately with impact BCE/rank loss. Paired faults are TIR pixel blackout in two frames per window; unedited natural faults have no known repair reference. Synthetic causal q labels do not prove natural-fault detection.
- Motion: crop-aware posterior rebase; exactly one transition per frame; accepted current head prediction is observed after repair. Current annotation never enters Kalman state. Motion-off evaluation uses the same selected checkpoint.
- GPU: dynamically query before each launch; use only idle GPU4 at preparation (the other seven cards have external jobs or allocations).
- Outputs: `_probe/E0018/{baseline,gt_centered}.{json,safetensors,log}` and per-frame `.trajectory.json` crop provenance. Large files remain outside Git.
- Promotion gate: positive held-out tracking and selected-box IoU gains, bounded healthy drift, no large clip failures; then run actual closed-loop LasHeR-train validation. This cache is off-policy and never counts as PR/SR. Negative results reject the recipe; no seed or threshold sweep follows.
- PR/SR / checkpoint / conclusion: pending.

```bash
source scripts/00_env.sh
CUDA_VISIBLE_DEVICES=4 "$PYTHON" tools/causal_replay_probe.py \
  --crop-policy baseline --clips-per-sequence 2 --clip-length 4 --val-count 3 \
  --diag-steps 120 --steps 100 --seed 42 \
  --output _probe/E0018/baseline.json \
  --save-checkpoint _probe/E0018/baseline.safetensors
# Run the matched gt_centered control using the same arguments and its own outputs.
```
