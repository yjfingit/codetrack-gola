# E0001: Paired GOLA Baseline on Selected LasHeR Sequences

- Date: 2026-10-06 to 2026-10-07
- Git commit / dirty state: `0280b14095123a0e894fd1f184732b75964a07b3`, dirty worktree
- Status: complete
- Hypothesis: establish a fair reference on the exact ten sequences used for the current causal candidate.
- Structure and config: GOLA-DINOv2-B checkpoint; CodeTrack route disabled with `CODETRACK_ABLATE_SATR=1`; no CodeTrack output changes.
- Training recipe and sampling: no training; official one-stream tracker evaluation.
- Data and split: LasHeR-test, sequence manifest `_probe/LasHeR_test10/testingsetList.txt` (10 sequences, 13,464 frames).
- Seed: evaluation seed from default command (42).
- GPU: GPU4; evaluation ran alone on this project GPU.
- Checkpoint: `weights/gola_b224.bin`.
- Training log: not applicable.
- Evaluation log: `_probe/paired_eval10_gola_baseline.log`.
- Reproduction command:

```bash
CUDA_VISIBLE_DEVICES=4 TRACKIT_CONSTS_PATH="$PWD/_probe/causal_eval_consts.yaml" \
CODETRACK_ABLATE_SATR=1 bash -c 'source scripts/00_env.sh; "$PYTHON" main.py GOLA codetrack_eval \
  --eval --mixin_config causal_taskimpact_eval10 --distributed_nproc_per_node 1 \
  --device cuda --disable_wandb --weight_path weights/gola_b224.bin \
  --output_dir _probe/paired_eval10_gola_baseline'
```

- Results: PR 58.07 / SR 48.15 / NPR 55.68.
- Mechanism evidence: reference only.
- Conclusion: retain as the paired ten-sequence control.
- Failure analysis and next action: these ten sequences are a diagnostic subset, not an acceptance test.
