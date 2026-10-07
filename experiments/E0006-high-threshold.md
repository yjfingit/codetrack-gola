# E0006: High q-Abstention Threshold Probe

- Date: 2026-10-07
- Git commit / dirty state: `0280b14095123a0e894fd1f184732b75964a07b3`, dirty worktree
- Status: stopped early
- Hypothesis: a much higher absolute q threshold could suppress false positive corrections.
- Structure and config: E0002 recipe with q abstention threshold raised from 0.6 to 0.8.
- Training recipe and sampling: no training; same selected-10 evaluation.
- Data and split: LasHeR-test; run stopped during sequence 8/10 after early sequences reverted to baseline under the high threshold.
- Seed: 42.
- GPU: GPU4.
- Checkpoint: `_probe/causal_taskimpact20/merged_full.safetensors`.
- Training log: not applicable.
- Evaluation log: `_probe/paired_eval10_causal_threshold08.log` (partial; no aggregate metric).
- Reproduction command:

```bash
CUDA_VISIBLE_DEVICES=4 TRACKIT_CONSTS_PATH="$PWD/_probe/causal_eval_consts.yaml" \
bash -c 'source scripts/00_env.sh; "$PYTHON" main.py GOLA codetrack_eval --eval \
  --mixin_config causal_taskimpact_eval10_threshold08 --distributed_nproc_per_node 1 --device cuda \
  --disable_wandb --weight_path _probe/causal_taskimpact20/merged_full.safetensors \
  --output_dir _probe/paired_eval10_causal_threshold08'
```

- Results: incomplete; aggregate PR/SR intentionally not reported.
- Mechanism evidence: early sequence behavior was close to baseline because candidate tokens did not pass q>=0.8.
- Conclusion: stop threshold sweep; do not retain as a candidate.
- Failure analysis and next action: high threshold suppresses useful corrections as well as harmful ones. Control correction direction/magnitude and learn when to abstain from real clips.
