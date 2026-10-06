#!/usr/bin/env bash
set -u

ROOT=/home/yangjuanfeng/lab/projects/gola-CodeTrack
PY=/home/yangjuanfeng/lab/envs/gola/bin/python
CKPT="$ROOT/_probe/grid/s1_upscale_400/out/gpu2_up005/GOLA-codetrack_s1-mixin-_grid_gpu2_up005-2026.10.04-16.20.18-049725/checkpoint/epoch_09/model.bin"
OUT="$ROOT/_probe/denoiser_topk_h4"
mkdir -p "$OUT"
source "$ROOT/scripts/00_env.sh"
export PYTHONUNBUFFERED=1

ks=(8 16 32 64)
gpus=(1 2 3 4)
pids=()
for i in "${!ks[@]}"; do
  k="${ks[$i]}"; gpu="${gpus[$i]}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PY" "$ROOT/tools/denoiser_write_probe.py" --mode topk --topk "$k" \
      --checkpoint "$CKPT" --seed 0 --output "$OUT/topk${k}.json"
  ) > "$OUT/topk${k}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then rc=0; else rc=$?; failed=$((failed + 1)); fi
  printf '%s\n' "$rc" > "$OUT/topk${ks[$i]}.exit"
done
exit "$failed"
