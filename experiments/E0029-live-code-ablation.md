# E0029: Live error-code ablation

- Date: 2026-10-07.
- Hypothesis: the observed native tracking improvement must depend on useful syndrome information, not merely a second frozen GOLA word or unary classification. Remove BP messages or shuffle check locations while preserving words, reference-quality estimator, checkpoint, motion, ordered state and thresholds.
- Data: same complete held-out 200-frame native sequence `theleftestrunningboy`.
- Reference: native GOLA PR .55 / SR .35214287; learned code E0028 PR .92 / SR .62380952.
- Arms: `unary` uses channel posterior only; `shuffled` rolls check likelihoods by 17 before the exact same sparse decoder. Every arm restarts at frame 0 and owns private state/output.
- Checkpoint: `_probe/E0027/train/decoder.safetensors`, no new fitting or seed/threshold sweep. Source commit captured before execution.
- Decision: compare exact complete-sequence PR/SR and correction spans. If learned check information does not help, the communication mechanism is not established even if the reference words are useful.
- Outputs: `_probe/E0029/{unary,shuffled}`; both complete 200/200 ordered frames.
- Unary arm completes at `feee219`: PR **.58000**, SR **.35999998**. Shuffled checks complete at `0a44e90`: PR **.92000**, SR **.62595242**. The learned-code arm is .92000 / .62380952. Thus unary removal loses **34.00 / 26.38 pp**, while shuffling the check likelihoods does not reduce this single-sequence score. The BP contribution is therefore not isolated by this witness; a longer, paired ablation is required before claiming that learned check placement improves tracking.
- The result also exposed a separate safety failure in the pre-safety E0027 live arm: the 8,574-frame `rightbackcup` rollout wrote on 2,739 frames and scored PR **.31257** / SR **.25170**, below its paired native GOLA PR **.44332** / SR **.33945**. Those writes used the old uncalibrated word-quality argmax. Do not promote that checkpoint.

```bash
source scripts/00_env.sh
CUDA_VISIBLE_DEVICES=4 "$PYTHON" tools/mine_native_failures.py \
  --sequences experiments/manifests/E0028-leftboy.txt \
  --decoder-checkpoint _probe/E0027/train/decoder.safetensors --decoder-ablation unary \
  --output _probe/E0029/unary --workers 4 --prefetch 8 --verify-prefetch 0 --events-per-sequence 0
# Repeat in a fresh state/output with --decoder-ablation shuffled.
```
