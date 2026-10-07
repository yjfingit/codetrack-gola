#!/usr/bin/env bash
# Parallel independent sequence evaluation on one GPU.
#
# Example:
#   GPU=4 NPROC=4 WEIGHT=/path/model.safetensors \
#   MIXIN_CONFIG=causal_clip_candidate \
#   SEQ_LIST=_probe/LasHeR_test10/testingsetList.txt \
#   OUT=_probe/eval_parallel bash scripts/codetrack_eval_parallel.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPU="${GPU:-0}"
NPROC="${NPROC:-4}"
SEQ_LIST="${SEQ_LIST:-_probe/LasHeR_test10/testingsetList.txt}"
OUT="${OUT:-$PWD/_probe/eval_parallel}"
WEIGHT="${WEIGHT:-$PWD/weights/gola_b224.bin}"
MIXIN_CONFIG="${MIXIN_CONFIG:-}"

if ! [[ "$NPROC" =~ ^[1-9][0-9]*$ ]]; then
  echo "NPROC must be a positive integer" >&2
  exit 2
fi
[[ -f "$SEQ_LIST" ]] || { echo "missing sequence list: $SEQ_LIST" >&2; exit 2; }
mkdir -p "$OUT"

mapfile -t SEQUENCES < <(sed '/^[[:space:]]*$/d' "$SEQ_LIST")
running=0
batch=()
run_one() {
  local seq="$1"
  local log="$OUT/$seq.log"
  local dir="$OUT/$seq"
  mkdir -p "$dir"
  echo "[parallel-eval] GPU=$GPU seq=$seq"
  CUDA_VISIBLE_DEVICES="$GPU" WEIGHT="$WEIGHT" MIXIN_CONFIG="$MIXIN_CONFIG" \
    SEQ="$seq" OUT="$dir" bash scripts/codetrack_eval_single.sh >"$log" 2>&1
}

for seq in "${SEQUENCES[@]}"; do
  run_one "$seq" &
  batch+=("$!")
  running=$((running + 1))
  if (( running >= NPROC )); then
    failed=0
    for pid in "${batch[@]}"; do
      wait "$pid" || failed=1
    done
    (( failed == 0 )) || { echo "one or more sequence evaluations failed" >&2; exit 1; }
    batch=()
    running=0
  fi
done

if (( running > 0 )); then
  failed=0
  for pid in "${batch[@]}"; do
    wait "$pid" || failed=1
  done
  (( failed == 0 )) || { echo "one or more sequence evaluations failed" >&2; exit 1; }
fi

echo "[parallel-eval] completed ${#SEQUENCES[@]} sequences"
for seq in "${SEQUENCES[@]}"; do
  rg -o "^${seq}: success [0-9.]+, prec [0-9.]+, norm_pre [0-9.]+" "$OUT/$seq.log" || true
done
