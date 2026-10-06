#!/usr/bin/env bash
set -u

ROOT=/home/yangjuanfeng/lab/projects/gola-CodeTrack
PY=/home/yangjuanfeng/lab/envs/gola/bin/python
CKPT="$ROOT/_probe/grid/s1_upscale_400/out/gpu2_up005/GOLA-codetrack_s1-mixin-_grid_gpu2_up005-2026.10.04-16.20.18-049725/checkpoint/epoch_09/model.bin"
OUT="$ROOT/_probe/denoiser_write_h3"
mkdir -p "$OUT"

source "$ROOT/scripts/00_env.sh"
export PYTHONUNBUFFERED=1

modes=(current topk oracle_soft oracle_unit)
gpus=(1 2 3 4)
pids=()
for i in "${!modes[@]}"; do
  mode="${modes[$i]}"
  gpu="${gpus[$i]}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PY" "$ROOT/tools/denoiser_write_probe.py" \
      --mode "$mode" --checkpoint "$CKPT" --seed 0 \
      --output "$OUT/$mode.json"
  ) > "$OUT/$mode.log" 2>&1 &
  pids+=("$!")
  printf '[%s] %s started on gpu%s\n' "$(date +%H:%M:%S)" "$mode" "$gpu"
done

failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then rc=0; else rc=$?; failed=$((failed + 1)); fi
  printf '%s\n' "$rc" > "$OUT/${modes[$i]}.exit"
  printf '[%s] %s finished rc=%s\n' "$(date +%H:%M:%S)" "${modes[$i]}" "$rc"
done
exit "$failed"
