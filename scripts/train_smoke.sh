#!/usr/bin/env bash
# Train GOLA-B (dinov2, 224) on LasHeR from the repo root.
#
#   bash scripts/train_smoke.sh              # tiny 8-sample run, verifies the loop
#   BS_MIXIN=bs_8 bash scripts/train_smoke.sh  # normal batch, still short
#
# Real training: omit the smoke mixin and pass the upstream bs_128 mixin.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck source=/dev/null
source scripts/00_env.sh

OUT="${OUT:-$PWD/outputs/train_smoke}"
mkdir -p "$OUT"

"$PYTHON" main.py GOLA dinov2 \
  --mixin_config disable_torch_compile \
  --mixin_config "${BS_MIXIN:-smoke_train}" \
  --distributed_nproc_per_node "${NPROC:-1}" \
  --disable_wandb \
  --weight_path "$WEIGHT" \
  --output_dir="$OUT" \
  "$@"
