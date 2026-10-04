#!/usr/bin/env bash
# Reorganise the workspace into a proper GOLA-based working repository.
#
#   root/            working GOLA instance (trackit + config), as LoRAT/DTPTrack do
#   third_party/GOLA read-only upstream reference @ 339c737
#   scripts/         our tooling (shell + python), grouped by purpose
#   configs/         CodeTrack-specific configs (mixin + experiment)
#   weights/         checkpoints
#   refs/            papers + reference repos
#   docs/            design & experiment docs
#   outputs/         run artefacts
#   data/            local data views (single-sequence eval lists)
#
# Idempotent: safe to re-run. Pass --dry-run to only print actions.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

run() { if [[ $DRY == 1 ]]; then echo "  DRY: $*"; else eval "$@"; fi; }

echo "== workspace: $ROOT"

# ---------------------------------------------------------------- 0. sanity
if [[ ! -d third_party/GOLA/trackit ]]; then
  echo "FATAL: third_party/GOLA is not a pristine GOLA checkout" >&2; exit 1
fi
GOLA_HEAD="$(git -C third_party/GOLA rev-parse HEAD)"
echo "== third_party/GOLA @ $GOLA_HEAD (must stay clean)"

# ------------------------------------------------- 1. working instance -> root
echo "== [1] materialise working GOLA instance at repo root"
for item in main.py profile_model.py evaluation.py trackit config assets; do
  if [[ -e "$item" ]]; then
    echo "  skip (exists): $item"
  else
    run "cp -a third_party/GOLA/$item ./"
  fi
done

# ------------------------------------------------- 2. directory skeleton
echo "== [2] create directory skeleton"
for d in scripts/train scripts/infer scripts/data scripts/env \
         configs/experiment configs/mixin \
         weights outputs data docs refs; do
  run "mkdir -p '$d'"
done

# ------------------------------------------------- 3. requirements / metadata
echo "== [3] requirements + metadata"
if [[ ! -e requirements.txt ]]; then
  run "cp -a third_party/GOLA/requirements.txt ./"
fi
run "touch weights/.gitkeep outputs/.gitkeep data/.gitkeep"

# ------------------------------------------------- 4. consts.yaml from template
echo "== [4] consts.yaml"
if [[ ! -e consts.yaml ]]; then
  if [[ -e third_party/GOLA/consts.yaml ]]; then
    run "cp -a third_party/GOLA/consts.yaml ./consts.yaml"
  fi
fi

# ------------------------------------------------- 5. report
echo
echo "== resulting top level =="
ls -1
echo
echo "== next steps (manual, once) =="
cat <<'EOF'
  1. edit ./consts.yaml -> LasHeR_PATH / RGBT234_PATH point at /root/autodl-tmp/lab/dataset/*
  2. export LD_LIBRARY_PATH=/root/autodl-tmp/lab/tools   # PyTurboJPEG
  3. run from repo root: python main.py GOLA dinov2 ...
EOF
