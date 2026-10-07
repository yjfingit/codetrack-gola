# E0008: Full LasHeR-Test Retry, Half-Strength Residual

- Date: 2026-10-07
- Git commit / dirty state: `0280b14095123a0e894fd1f184732b75964a07b3`, dirty worktree
- Status: interrupted/incomplete; live process audit on 2026-10-07 found no CodeTrack worker.
- Hypothesis: the half-strength residual from E0005 will retain useful improvements and avoid the large sequence failures in E0004.
- Structure and config: E0002 spatial-only configuration, `CODETRACK_RECOVERY_SCALE=0.5`; no motion, memory, or template protection.
- Training recipe and sampling: no new training; official closed-loop full LasHeR-test evaluation.
- Data and split: LasHeR-test, 245 sequences.
- Seed: 42.
- GPU: GPU4 was idle at launch; other GPUs had external jobs.
- Checkpoint: `_probe/causal_taskimpact20/merged_full.safetensors`.
- Training log: source probe has no retained stdout training log; see E0002.
- Evaluation log: `_probe/full_eval_causal_spatial_scale05_retry.log`.
- Reproduction command:

```bash
CUDA_VISIBLE_DEVICES=4 CODETRACK_RECOVERY_SCALE=0.5 bash -c 'source scripts/00_env.sh; "$PYTHON" main.py GOLA codetrack_eval \
  --eval --mixin_config causal_taskimpact_full_spatial --distributed_nproc_per_node 1 \
  --device cuda --disable_wandb --weight_path _probe/causal_taskimpact20/merged_full.safetensors \
  --output_dir _probe/full_eval_causal_spatial_scale05_retry'
```

- Results: invalid/incomplete. The retained log stops at 36/245 completed sequences and has no official final summary or result archive. Do not aggregate these partial sequences as a full-test score or restart because an old status says running.
- Mechanism evidence: pending full paired sequence and attribute analysis.
- Conclusion: pending.
- Failure analysis and next action: compare all 245 per-sequence rows with `_probe/eval_gola_identity_full.log`; if both PR and SR are at least baseline, evaluate RGBT234. Otherwise use the paired attribute deltas to design the next training experiment.
