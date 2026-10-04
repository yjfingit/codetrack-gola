#!/usr/bin/env bash
# Short joint training of GOLA-B + CodeTrack on ONE LasHeR sequence.
#
#   bash scripts/codetrack_train_smoke.sh
#
# Why the consts swap: the training source is an `!include` resolved while parsing the
# config, and the mixin machinery can only replace *existing* keys, so the dataset cannot be
# swapped from a mixin.  Instead consts.LasHeR_PATH is pointed at a one-line view
# (data/LasHeR_train_single) for the duration of the run and restored afterwards.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck source=/dev/null
source scripts/00_env.sh

SEQ="${SEQ:-10crosswhite}"
VIEW="$PWD/data/LasHeR_train_single"
CONFIG="${CONFIG:-codetrack_smoke}"
OUT="${OUT:-$PWD/outputs/codetrack_smoke}"
CONSTS="$PWD/consts.yaml"
BACKUP="$CONSTS.codetrack_backup"

mkdir -p "$VIEW" "$OUT"
DS="$(sed -n "s|^LasHeR_PATH: '\(.*\)'|\1|p" "$CONSTS")"
DS="${DS%/}"
[[ -d "$DS/trainingset/$SEQ" ]] || { echo "FATAL: no such sequence: $DS/trainingset/$SEQ" >&2; exit 1; }
printf '%s\n' "$SEQ" > "$VIEW/trainingsetList.txt"
printf '%s\n' "$(head -1 "$DS/testingsetList.txt")" > "$VIEW/testingsetList.txt"
for item in trainingset testingset annos AttriSeqsTxt Attributes_order.txt; do
  [[ -e "$DS/$item" ]] && ln -sfn "$DS/$item" "$VIEW/$item"
done

restore() {
  if [[ -f "$BACKUP" ]]; then mv -f "$BACKUP" "$CONSTS"; echo "[restored] consts.yaml"; fi
}
trap restore EXIT

cp -a "$CONSTS" "$BACKUP"
"$PYTHON" - "$CONSTS" "$VIEW" <<'PY'
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
echo "[swap] consts.LasHeR_PATH -> $VIEW/"
echo "[run ] CodeTrack joint training on sequence: $SEQ"

"$PYTHON" main.py GOLA "$CONFIG" \
  --distributed_nproc_per_node 1 \
  --disable_wandb \
  --weight_path "$WEIGHT" \
  --output_dir="$OUT" \
  "$@"
