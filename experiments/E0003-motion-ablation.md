# E0003: Motion Module Activation Ablation

- Date: 2026-10-07
- Git commit / dirty state: `0280b14095123a0e894fd1f184732b75964a07b3`, dirty worktree
- Status: complete
- Hypothesis: enabling Kalman motion should alter candidate routing or recovery and improve tracking on the selected set.
- Structure and config: same E0002 checkpoint and selected-10 recipe; only `motion_enabled` changed from false to true; memory remains disabled.
- Training recipe and sampling: no training; one-stream evaluation.
- Data and split: same LasHeR-test selected ten.
- Seed: 42.
- GPU: GPU4.
- Checkpoint: `_probe/causal_taskimpact20/merged_full.safetensors`.
- Training log: not applicable.
- Evaluation log: `_probe/paired_eval10_causal_motion.log`.
- Reproduction command:

```bash
CUDA_VISIBLE_DEVICES=4 TRACKIT_CONSTS_PATH="$PWD/_probe/causal_eval_consts.yaml" \
bash -c 'source scripts/00_env.sh; "$PYTHON" main.py GOLA codetrack_eval --eval \
  --mixin_config causal_taskimpact_eval10_motion --distributed_nproc_per_node 1 --device cuda \
  --disable_wandb --weight_path _probe/causal_taskimpact20/merged_full.safetensors \
  --output_dir _probe/paired_eval10_causal_motion'
```

- Results: PR 58.07 / SR 48.15 / NPR 55.67, effectively identical to paired GOLA.
- Mechanism evidence: sequence-level outputs matched baseline across all ten; current q abstention prevented writes in this setting.
- Conclusion: reject the claim that the current motion module contributes. Its mere presence is insufficient.
- Failure analysis and next action: motion must be trained and shown to alter target-aware routing or abstention; test it separately after safe spatial correction passes full-test.
