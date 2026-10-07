# E0005: Half-Strength Residual, Selected-10 Evaluation

- Date: 2026-10-07
- Git commit / dirty state: `0280b14095123a0e894fd1f184732b75964a07b3`, dirty worktree
- Status: complete
- Hypothesis: reducing correction magnitude preserves useful direction while limiting catastrophic over-correction.
- Structure and config: identical to E0002; inference override `CODETRACK_RECOVERY_SCALE=0.5`; no parameter retraining.
- Training recipe and sampling: no training; same official tracker pipeline.
- Data and split: same selected-10 LasHeR-test manifest as E0001.
- Seed: 42.
- GPU: GPU4.
- Checkpoint: `_probe/causal_taskimpact20/merged_full.safetensors`.
- Training log: not applicable.
- Evaluation log: `_probe/paired_eval10_causal_scale05.log`.
- Reproduction command:

```bash
CUDA_VISIBLE_DEVICES=4 TRACKIT_CONSTS_PATH="$PWD/_probe/causal_eval_consts.yaml" \
CODETRACK_RECOVERY_SCALE=0.5 bash -c 'source scripts/00_env.sh; "$PYTHON" main.py GOLA codetrack_eval \
  --eval --mixin_config causal_taskimpact_eval10 --distributed_nproc_per_node 1 --device cuda \
  --disable_wandb --weight_path _probe/causal_taskimpact20/merged_full.safetensors \
  --output_dir _probe/paired_eval10_causal_scale05'
```

- Results: PR 59.35 / SR 48.63 / NPR 56.76; paired GOLA was 58.07 / 48.15 / 55.68.
- Mechanism evidence: sequence deltas became milder; `catbrownback2bush` improved vs baseline by 2.48 SR pp and `rightbottlecomes` by 4.18 SR pp. Full-test confirmation is E0007.
- Conclusion: continue validation; not accepted until complete LasHeR-test finishes.
- Failure analysis and next action: scaling is a blunt safety control, not a learned confidence estimate. If full-test passes, use it as a baseline safety factor and learn the decision on real clips.
