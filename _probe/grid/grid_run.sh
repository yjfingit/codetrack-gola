#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 4-GPU parallel grid search over CodeTrack hyper-parameters.
#
#   * Each trial = a `--mixin_config` override set, run for a short fixed update
#     budget on ONE dedicated GPU.  Up to `#gpus` trials run concurrently.
#   * Single process per trial (no DDP): the verified cross-rank calibration
#     path stays idle, so trials are directly comparable.
#   * A GPU lock guarantees two trials never share a card.
#
# Two framework facts this script has to work around:
#   1. `--mixin_config` resolves relative to `config/<method>/_mixin/` and strips
#      a leading '/', so the mixin file MUST live inside the repo config tree.
#      We write them to config/GOLA/_mixin/_grid_<name>.yaml and clean up after.
#   2. PyTurboJPEG cannot locate libturbojpeg without LD_LIBRARY_PATH, so
#      scripts/00_env.sh must be sourced in every trial subshell.
#
# Usage:
#   bash grid_run.sh <grid_name> [--gpus 2,3,4,7] [--base codetrack_s1]
#
# Plan is read from <grid_root>/plan.txt, one line per trial:
#   <trial_name>|<k=v,k=v,...>
# ---------------------------------------------------------------------------
set -u

PROJ=/home/yangjuanfeng/lab/projects/gola-CodeTrack
PY=/home/yangjuanfeng/lab/envs/gola/bin/python
MIXIN_DIR="$PROJ/config/GOLA/_mixin"

GRID_NAME="${1:?grid name}"
shift || true
GPUS="2,3,4,7"
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
LOCKROOT="$ROOT/locks"

[[ -f "$PLAN" ]] || { echo "ERROR: plan not found: $PLAN"; exit 2; }

mkdir -p "$ROOT/logs" "$ROOT/mixins" "$ROOT/out" "$LOCKROOT"

IFS=',' read -ra GPU_ARR <<< "$GPUS"
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

log "grid '$GRID_NAME'  base=$BASE  gpus=${GPU_ARR[*]}"

run_trial() {
  local gpu="$1" trial="$2" overrides="$3"
  local lock="$LOCKROOT/gpu$gpu.lock"
  local mixin_in_tree="$MIXIN_DIR/_grid_$trial.yaml"
  local logf="$ROOT/logs/$trial.log"
  local outd="$ROOT/out/$trial"

  while ! mkdir "$lock" 2>/dev/null; do sleep 3; done

  local -a kv=()
  IFS=',' read -ra PAIRS <<< "$overrides"
  for p in "${PAIRS[@]}"; do [[ -n "$p" ]] && kv+=("$p"); done

  # generate into the repo mixin dir (framework requires it there), keep a copy
  if ! "$PY" "$PROJ/_probe/grid/grid_make_mixin.py" "$mixin_in_tree" "${kv[@]}" \
        > "$logf.gen" 2>&1; then
    log "gpu$gpu  $trial  CONFIG-FAIL"
    cat "$logf.gen" >&2
    rmdir "$lock" 2>/dev/null
    echo 3 > "$ROOT/logs/$trial.exit"
    return 3
  fi
  cp "$mixin_in_tree" "$ROOT/mixins/$trial.yaml"

  log "gpu$gpu  $trial  start  [${overrides}]"
  local t0=$SECONDS
  (
    cd "$PROJ" || exit 9
    # shellcheck disable=SC1091
    source scripts/00_env.sh > /dev/null 2>&1
    export CUDA_VISIBLE_DEVICES="$gpu"
    export PYTHONUNBUFFERED=1
    "$PY" main.py GOLA "$BASE" \
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
  rmdir "$lock" 2>/dev/null
  return $rc
}

i=0; pids=()
while IFS='|' read -r trial overrides; do
  [[ -z "${trial// }" ]] && continue
  case "$trial" in \#*) continue ;; esac
  gpu="${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}"
  i=$((i + 1))
  run_trial "$gpu" "$trial" "$overrides" &
  pids+=($!)
  sleep 25          # stagger: avoid 979-sequence dataset scans colliding
done < "$PLAN"

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done

log "grid done, $fail non-zero exit(s)"
echo "logs: $ROOT/logs"
