#!/usr/bin/env bash
# One-shot validation of the grid machinery: generate a mixin the same way
# grid_run.sh does, launch it exactly the same way, and print the verdict.
#
# Usage: bash grid_selftest.sh [gpu]
set -u

PROJ=/home/yangjuanfeng/lab/projects/gola-CodeTrack
PY=/home/yangjuanfeng/lab/envs/gola/bin/python
MIXIN_DIR="$PROJ/config/GOLA/_mixin"
GPU="${1:-2}"

cd "$PROJ" || exit 9
"$PY" _probe/grid/grid_make_mixin.py "$MIXIN_DIR/_grid_selftest.yaml" \
  "max_updates=3,t_initial_updates=3,warmup_updates=0,sched_warmup_updates=0,samples_per_epoch=8,global_batch_size=2,grad_accumulation_steps=1,lr=1e-4" \
  || exit 3
# the update budget above is far below the epoch, so stop the sampler early
cp "$MIXIN_DIR/_grid_selftest.yaml" /tmp/grid_selftest_mixin.yaml
cat /tmp/grid_selftest_mixin.yaml

source scripts/00_env.sh > /dev/null 2>&1
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

"$PY" main.py GOLA codetrack_s1 \
  --distributed_nproc_per_node 1 \
  --mixin_config "_grid_selftest" \
  --disable_wandb \
  --weight_path "$PROJ/weights/gola_b224.bin" \
  --output_dir "$PROJ/_probe/grid/_selftest" 2>&1 | tail -60

rm -f "$MIXIN_DIR/_grid_selftest.yaml"
