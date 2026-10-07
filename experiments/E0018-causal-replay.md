# E0018: Causal crops versus GT-centred training control

- Date: 2026-10-07.
- Status: first attempt failed at validation after diagnosis and the first SATR update; corrected before retry.
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
- Causal arm completed at `58b1b2a`: best validation checkpoint is step 0. Held-out q-AUROC 0.62414, tracking-loss gain **-0.00007460**, selected-box IoU gain **-0.00014762**, healthy drift 1.90e-7. Motion-off loss gain is -0.00007522; the difference is too small to establish useful motion correction. At step 50 training improves by 0.02905 while validation worsens by 0.002279 and IoU falls by 0.005449. Reject this recipe for further scale-up; no seed/threshold sweep.
- Diagnostic checkpoint: `_probe/E0018/baseline.safetensors`, 53 persisted tensors including H/template pool. It is not a final tracker or the project's best checkpoint.
- PR/SR: not evaluated; this is off-policy feature evidence. The matched GT-centred control is still pending and will determine how much of the failure comes from crop policy.
- First launch source: `67ed7ac`, GPU4 PID 3411599. Terminal log: `_probe/E0018/baseline_attempt1.log`. Failure: AUROC helper imported unavailable `sklearn`; no selected checkpoint exists. Replaced it with tie-aware rank AUROC in PyTorch. A 12-trial identical-token witness with the actual pretrained head passed (zero false positive labels; head batch differences up to 9.54e-7). Explicit no-op masking now makes this invariant independent of batch-kernel numerics. The retry also reports unedited-frame label density.

```bash
source scripts/00_env.sh
CUDA_VISIBLE_DEVICES=4 "$PYTHON" tools/causal_replay_probe.py \
  --crop-policy baseline --clips-per-sequence 2 --clip-length 4 --val-count 3 \
  --diag-steps 120 --steps 100 --seed 42 \
  --output _probe/E0018/baseline.json \
  --save-checkpoint _probe/E0018/baseline.safetensors
# Run the matched gt_centered control using the same arguments and its own outputs.
```
