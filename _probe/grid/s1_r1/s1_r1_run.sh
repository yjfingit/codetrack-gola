#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# CodeTrack S1 Round-1: one trial per GPU, 4 GPUs concurrently.
#
#   GPU1  gpu0_baseline  lr=1e-4  gain=0.2  alpha=0.5   (true baseline)
#   GPU2  gpu1_lr5e5     lr=5e-5  gain=0.4  alpha=0.5
#   GPU3  gpu2_gain04    lr=1e-4  gain=0.4  alpha=0.5
#   GPU4  gpu3_alpha07   lr=1e-4  gain=0.4  alpha=0.7
#
# All four: micro 16 x accum 8 = 128 effective, warmup 128, max_updates 2048.
# (GPU 0/5/6 belong to another user and are NOT touched.)
#
# Fixes vs the first attempt
# -------------------------
# * Uses grid_make_mixin2.py, which emits real YAML floats.  The original wrote
#   `5e-05`, which PyYAML read back as the STRING '5e-05'; AdamW then died with
#   "TypeError: '<=' not supported between instances of 'float' and 'str'".
# * Trial names are whitespace-stripped, so `name |kv` no longer produces
#   `_grid_name .yaml` with an embedded space.
#
# Usage:
#   bash s1_r1_run.sh <grid_name> [--gpus 1,2,3,4]
# ---------------------------------------------------------------------------
set -u

PROJ=/home/yangjuanfeng/lab/projects/gola-CodeTrack
PY=/home/yangjuanfeng/lab/envs/gola/bin/python
MIXIN_DIR="$PROJ/config/GOLA/_mixin"
GEN="$PROJ/_probe/grid/grid_make_mixin2.py"

GRID_NAME="${1:?grid name}"
shift || true
GPUS="1,2,3,4"
BASE="codetrack_s1"
WEIGHT="$PROJ/weights/gola_b224.bin"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)   GPUS="$2"; shift 2 ;;
    --base)   BASE="$2"; shift 2 ;;
    --weight) WEIGHT="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

ROOT="$PROJ/_probe/grid/$GRID_NAME"
PLAN="$ROOT/plan.txt"
mkdir -p "$ROOT/logs" "$ROOT/mixins" "$ROOT/out"
[[ -f "$PLAN" ]] || { echo "ERROR: plan not found: $PLAN"; exit 2; }

# strip CR so a plan edited on Windows still parses
sed -i 's/\r$//' "$PLAN"

IFS=',' read -ra GPU_ARR <<< "$GPUS"
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
log "S1 round-1 grid '$GRID_NAME'  base=$BASE  gpus=${GPU_ARR[*]}"

run_trial() {
  local gpu="$1" trial="$2" overrides="$3"
  local mixin_in_tree="$MIXIN_DIR/_grid_$trial.yaml"
  local logf="$ROOT/logs/$trial.log"
  local outd="$ROOT/out/$trial"

  local -a kv=()
  IFS=',' read -ra PAIRS <<< "$overrides"
  for p in "${PAIRS[@]}"; do [[ -n "$p" ]] && kv+=("$p"); done

  if ! "$PY" "$GEN" "$mixin_in_tree" "${kv[@]}" > "$logf.gen" 2>&1; then
    log "gpu$gpu  $trial  CONFIG-FAIL"; cat "$logf.gen" >&2
    echo 3 > "$ROOT/logs/$trial.exit"; return 3
  fi
  cp "$mixin_in_tree" "$ROOT/mixins/$trial.yaml"

  log "gpu$gpu  $trial  start"
  local t0=$SECONDS
  (
    cd "$PROJ" || exit 9
    # shellcheck disable=SC1091
    source scripts/00_env.sh > /dev/null 2>&1
    export CUDA_VISIBLE_DEVICES="$gpu"
    export PYTHONUNBUFFERED=1
    "$PY" -u main.py GOLA "$BASE" \
      --distributed_nproc_per_node 1 \
      --mixin_config "_grid_$trial" \
      --disable_wandb \
      --weight_path "$WEIGHT" \
      --output_dir "$outd"
  ) > "$logf" 2>&1
  local rc=$?
  echo "$rc" > "$ROOT/logs/$trial.exit"
  log "gpu$gpu  $trial  done rc=$rc  $((SECONDS - t0))s"
  rm -f "$mixin_in_tree"
  return $rc
}

i=0; pids=(); names=()
while IFS='|' read -r trial overrides; do
  trial="${trial#"${trial%%[![:space:]]*}"}"; trial="${trial%"${trial##*[![:space:]]}"}"
  [[ -z "$trial" ]] && continue
  case "$trial" in \#*) continue ;; esac
  gpu="${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}"
  i=$((i + 1))
  run_trial "$gpu" "$trial" "$overrides" &
  pids+=($!); names+=("$trial@gpu$gpu")
  sleep 20
done < "$PLAN"

log "launched ${#pids[@]} trial(s): ${names[*]}"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
log "grid done, $fail non-zero exit(s)"
echo "logs: $ROOT/logs"
