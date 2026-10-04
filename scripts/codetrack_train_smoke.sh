#!/usr/bin/env bash
# Short joint training of GOLA-B + CodeTrack on ONE LasHeR sequence.
#
#   bash scripts/codetrack_train_smoke.sh
#
# The training source is an `!include` resolved while parsing the config, and the mixin
# machinery cannot replace it.  TRACKIT_CONSTS_PATH points this process at a temporary constants
# file, so the repository-wide consts.yaml is never mutated and train/eval smoke jobs may coexist.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck source=/dev/null
source scripts/00_env.sh

SEQ="${SEQ:-10crosswhite}"
VIEW="$PWD/data/LasHeR_train_single"
CONFIG="${CONFIG:-codetrack_smoke}"
OUT="${OUT:-$PWD/outputs/codetrack_smoke}"
CONSTS="$PWD/consts.yaml"

mkdir -p "$VIEW" "$OUT"
DS="$(sed -n "s|^LasHeR_PATH: '\(.*\)'|\1|p" "$CONSTS")"
DS="${DS%/}"
[[ -d "$DS/trainingset/$SEQ" ]] || { echo "FATAL: no such sequence: $DS/trainingset/$SEQ" >&2; exit 1; }
printf '%s\n' "$SEQ" > "$VIEW/trainingsetList.txt"
printf '%s\n' "$(head -1 "$DS/testingsetList.txt")" > "$VIEW/testingsetList.txt"
for item in trainingset testingset annos AttriSeqsTxt Attributes_order.txt; do
  [[ -e "$DS/$item" ]] && ln -sfn "$DS/$item" "$VIEW/$item"
done

RUN_CONSTS="$(mktemp "$OUT/consts.XXXXXX.yaml")"
trap 'rm -f "$RUN_CONSTS"' EXIT
cp -a "$CONSTS" "$RUN_CONSTS"
"$PYTHON" - "$RUN_CONSTS" "$VIEW" <<'PY'
import sys
path, view = sys.argv[1], sys.argv[2]
s = open(path).read()
out, hit = [], 0
for line in s.splitlines(keepends=True):
    if line.startswith('LasHeR_PATH:'):
        line = f"LasHeR_PATH: '{view}/'\n"; hit += 1
    out.append(line)
assert hit == 1, f"expected exactly 1 LasHeR_PATH line, found {hit}"
open(path, 'w').write(''.join(out))
PY
export TRACKIT_CONSTS_PATH="$RUN_CONSTS"
echo "[view] TRACKIT_CONSTS_PATH=$RUN_CONSTS (LasHeR_PATH -> $VIEW/)"
echo "[run ] CodeTrack joint training on sequence: $SEQ"

"$PYTHON" main.py GOLA "$CONFIG" \
  --distributed_nproc_per_node 1 \
  --disable_wandb \
  --weight_path "$WEIGHT" \
  --output_dir="$OUT" \
  "$@"
