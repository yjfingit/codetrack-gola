# E0029: Live error-code ablation

- Date: 2026-10-07.
- Hypothesis: the observed native tracking improvement must depend on useful syndrome information, not merely a second frozen GOLA word or unary classification. Remove BP messages or shuffle check locations while preserving words, reference-quality estimator, checkpoint, motion, ordered state and thresholds.
- Data: same complete held-out 200-frame native sequence `theleftestrunningboy`.
- Reference: native GOLA PR .55 / SR .35214287; learned code E0028 PR .92 / SR .62380952.
- Arms: `unary` uses channel posterior only; `shuffled` rolls check likelihoods by 17 before the exact same sparse decoder. Every arm restarts at frame 0 and owns private state/output.
- Checkpoint: `_probe/E0027/train/decoder.safetensors`, no new fitting or seed/threshold sweep. Source commit captured before execution.
- Decision: compare exact complete-sequence PR/SR and correction spans. If learned check information does not help, the communication mechanism is not established even if the reference words are useful.
- Outputs/status: `_probe/E0029/{unary,shuffled}`; pending.
- Unary arm completes at `feee219`: 200/200 ordered frames, PR **.58000**, SR **.35999998**. Learned code E0028 is .92000 / .62380952, a **+34.00 / +26.38 pp** difference on this one sequence with the same reference-quality network, observations and checkpoint. Shuffle verification remains pending; this does not by itself establish full-test mechanism benefits.

```bash
source scripts/00_env.sh
CUDA_VISIBLE_DEVICES=4 "$PYTHON" tools/mine_native_failures.py \
  --sequences experiments/manifests/E0028-leftboy.txt \
  --decoder-checkpoint _probe/E0027/train/decoder.safetensors --decoder-ablation unary \
  --output _probe/E0029/unary --workers 4 --prefetch 8 --verify-prefetch 0 --events-per-sequence 0
# Repeat in a fresh state/output with --decoder-ablation shuffled.
```
