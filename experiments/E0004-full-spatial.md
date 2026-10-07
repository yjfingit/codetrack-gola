# E0004: Full LasHeR-Test, Full-Strength Causal Spatial SATR

- Date: 2026-10-07
- Git commit / dirty state: `0280b14095123a0e894fd1f184732b75964a07b3`, dirty worktree
- Status: complete
- Hypothesis: the selected-10 improvement from E0002 generalizes to the complete test split.
- Structure and config: E0002 spatial-only recipe, full residual scale, 8-token cap, q threshold 0.6; no motion or memory.
- Training recipe and sampling: no new training; official closed-loop tracker evaluation.
- Data and split: all 245 LasHeR-test sequences.
- Seed: 42.
- GPU: GPU4.
- Checkpoint: `_probe/causal_taskimpact20/merged_full.safetensors`.
- Training log: see E0002; training stdout was not retained.
- Evaluation log: `_probe/full_eval_causal_spatial.log`.
- Reproduction command:

```bash
CUDA_VISIBLE_DEVICES=4 bash -c 'source scripts/00_env.sh; "$PYTHON" main.py GOLA codetrack_eval \
  --eval --mixin_config causal_taskimpact_full_spatial --distributed_nproc_per_node 1 \
  --device cuda --disable_wandb --weight_path _probe/causal_taskimpact20/merged_full.safetensors \
  --output_dir _probe/full_eval_causal_spatial'
```

- Results: PR 77.16 / SR 61.79 / NPR 73.70. GOLA reference: PR 77.19 / SR 61.79 / NPR 73.76.
- Mechanism evidence: per-sequence report `_probe/full_causal_spatial_attribute_analysis.md`; examples include `rightblkfatboyleftwhite` SR +29.67 pp and `whiteridingbike` -34.32 pp. Challenge groups HO and AIV improved on average; MB, CM, HI, DEF declined.
- Conclusion: reject as final; SR ties baseline but PR is 0.03 pp below it.
- Failure analysis and next action: rare destructive correction cancels gains. Test residual magnitude scaling, then train an explicit safe-write rule on continuous clips if it generalizes.
