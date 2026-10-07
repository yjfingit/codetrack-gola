# E0002: Causal-Impact Spatial SATR Probe

- Date: 2026-10-06 to 2026-10-07
- Git commit / dirty state: `0280b14095123a0e894fd1f184732b75964a07b3`, dirty worktree
- Status: complete
- Hypothesis: supervise q using whether replacing a search token with its clean counterpart reduces frozen GOLA-head tracking error, then route sparse SATR correction.
- Structure and config: deterministic grid H; neural BP; 2 SATR rounds; motion, memory, and template protection disabled; top-k cap 8; q abstention threshold 0.6; GOLA head/backbone frozen in evaluation. Candidate checkpoint `_probe/causal_taskimpact20/merged_full.safetensors`.
- Training recipe and sampling: LasHeR-train Siamese pair short probe, configured budget 20 optimizer updates, 2 warmup updates, batch sampling 320 examples/epoch, causal-q target enabled, causal rank weight 0.5. This is a short probe, not a full/converged training.
- Data and split: train probe on LasHeR-train; selected-10 evaluation on LasHeR-test.
- Seed: 42.
- GPU: training GPU2 according to run context; selected-10 evaluations on GPU4.
- Checkpoint: `_probe/causal_taskimpact20/merged_full.safetensors`; source epoch files are in `_probe/causal_taskimpact20/GOLA-codetrack_s1-mixin-causal_s1_unlocked20-2026.10.06-23.09.18-713947/checkpoint/`.
- Training log: training stdout log was not retained in the run directory; only epoch_06/epoch_07 checkpoints and the saved config are available. Do not treat convergence as verified.
- Evaluation log: `_probe/paired_eval10_causal_candidate.log`.
- Reproduction command:

```bash
CUDA_VISIBLE_DEVICES=4 TRACKIT_CONSTS_PATH="$PWD/_probe/causal_eval_consts.yaml" \
bash -c 'source scripts/00_env.sh; "$PYTHON" main.py GOLA codetrack_eval --eval \
  --mixin_config causal_taskimpact_eval10 --distributed_nproc_per_node 1 --device cuda \
  --disable_wandb --weight_path _probe/causal_taskimpact20/merged_full.safetensors \
  --output_dir _probe/paired_eval10_causal_candidate'
```

- Results: PR 58.99 / SR 48.21 / NPR 56.12; paired GOLA was 58.07 / 48.15 / 55.68.
- Mechanism evidence: earlier probe `_probe/causal_q_generalization_grid.json` reports held-out spatial-block q AUROC 0.808; this does not establish natural-failure localization or safe correction. Full-strength corrections caused severe drops on some full-test sequences.
- Conclusion: retain as a diagnostic checkpoint, reject full-strength deployment.
- Failure analysis and next action: q identifies tracking-sensitive locations better than injector-mask labels, but does not guarantee a safe correction direction or magnitude.
