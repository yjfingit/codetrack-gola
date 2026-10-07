# E0007: Full LasHeR-Test, Half-Strength Residual

- Date: 2026-10-07
- Git commit / dirty state: `0280b14095123a0e894fd1f184732b75964a07b3`, dirty worktree
- Status: interrupted/incomplete
- Hypothesis: the half-strength residual from E0005 will retain useful improvements and avoid the large sequence failures in E0004.
- Structure and config: E0002 spatial-only configuration, `CODETRACK_RECOVERY_SCALE=0.5`; no motion, memory, or template protection.
- Training recipe and sampling: no new training; official closed-loop full LasHeR-test evaluation.
- Data and split: LasHeR-test, 245 sequences.
- Seed: 42.
- GPU: GPU4; approximately 75-100 frames/s during the last-sequence portion. Other GPUs were occupied by unrelated jobs; none were touched.
- Checkpoint: `_probe/causal_taskimpact20/merged_full.safetensors`.
- Training log: source probe has no retained stdout training log; see E0002.
- Evaluation log: `_probe/full_eval_causal_spatial_scale05.log`.
- Reproduction command:

```bash
CUDA_VISIBLE_DEVICES=4 CODETRACK_RECOVERY_SCALE=0.5 bash -c 'source scripts/00_env.sh; "$PYTHON" main.py GOLA codetrack_eval \
  --eval --mixin_config causal_taskimpact_full_spatial --distributed_nproc_per_node 1 \
  --device cuda --disable_wandb --weight_path _probe/causal_taskimpact20/merged_full.safetensors \
  --output_dir _probe/full_eval_causal_spatial_scale05'
```

- Results: invalid/incomplete; log stopped after 238/245 sequences, process no longer existed, and no `results.zip` or final metric summary was produced.
- Mechanism evidence: pending full paired sequence and attribute analysis.
- Conclusion: no metric claim can be made from this run.
- Failure analysis and next action: rerun with a new output directory and preserve the process/log until the official summary and result archive appear. Do not infer completion from GPU memory being released.
