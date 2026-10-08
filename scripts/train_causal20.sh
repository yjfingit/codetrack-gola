#!/usr/bin/env bash
# Corrected curriculum. Explicit command, no automatic relaunch of the completed pair run.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/00_env.sh
unset NCCL_P2P_DISABLE
ulimit -n 65535
export CUDA_VISIBLE_DEVICES=3,4
export CAUSAL_GPU_SET=3,4
export CAUSAL_WORLD_SIZE=2
export CODETRACK_BF16=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
RUN_OUTPUT="${RUN_OUTPUT:-$PWD/outputs/causal20_full}"
RESUME_ARGS=()
if [[ -n "${CAUSAL_RESUME:-}" ]]; then
  [[ -f "$CAUSAL_RESUME/model.safetensors" && -f "$CAUSAL_RESUME/state.pth" ]] || {
    echo "Incomplete resume checkpoint: $CAUSAL_RESUME" >&2; exit 2;
  }
  RESUME_ARGS=(--resume "$CAUSAL_RESUME")
elif [[ -e "$RUN_OUTPUT" ]]; then
  echo "Output already exists: $RUN_OUTPUT. Choose a new RUN_OUTPUT or use explicit --resume." >&2
  exit 2
fi
# Require both requested GPUs to be available before taking them.
gpu_memory=$(nvidia-smi --id=3,4 --query-gpu=memory.used --format=csv,noheader,nounits)
while read -r memory; do
  if (( memory > 500 )); then
    echo "GPU 3/4 are not both idle; refusing to start on another user's allocations." >&2
    exit 2
  fi
done <<< "$gpu_memory"
mkdir -p "$(dirname "$RUN_OUTPUT")"
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 --max_restarts=0 \
  tools/train_final20_causal.py --local-clips 4 --sampling frame_coverage --output "$RUN_OUTPUT" "${RESUME_ARGS[@]}" "$@" \
  2>&1 | tee -a "${RUN_OUTPUT}.log"
for arg in "$@"; do
  if [[ "$arg" == --smoke ]]; then exit 0; fi
done
# Test only after validation selection and 20 complete resumable epochs.
for arm in baseline candidate; do
  pids=()
  for shard in 0 1; do
    physical_gpu=$((shard + 3))
    CUDA_VISIBLE_DEVICES="$physical_gpu" "$PYTHON" tools/evaluate_final20_causal.py \
      --run "$RUN_OUTPUT" --output "${RUN_OUTPUT}_test" --arm "$arm" \
      --shard "$shard" --shards 2 > "${RUN_OUTPUT}_${arm}_${shard}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  if ((failed)); then echo "Evaluation arm failed: $arm" >&2; exit 1; fi
done
"$PYTHON" - "$RUN_OUTPUT" "${RUN_OUTPUT}_test" <<'PY'
import json,sys
from tools.evaluate_final20_causal import compare_test_shards
print(json.dumps(compare_test_shards(sys.argv[1],sys.argv[2]),indent=2))
PY
