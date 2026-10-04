#!/usr/bin/env bash
# Single-sequence inference with the CodeTrack model (official eval pipeline).
#
#   bash scripts/codetrack_eval_single.sh
#
# Point consts.LasHeR_PATH at the one-line view for the duration of the run, then restore.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck source=/dev/null
source scripts/00_env.sh

SEQ="${SEQ:-10runone}"
CONFIG="${CONFIG:-codetrack_eval}"
VIEW="$PWD/data/LasHeR_single"
OUT="${OUT:-$PWD/outputs/codetrack_eval_$SEQ}"
CONSTS="$PWD/consts.yaml"
BACKUP="$CONSTS.eval_backup"

mkdir -p "$VIEW" "$OUT"
printf '%s\n' "$SEQ" > "$VIEW/testingsetList.txt"
DS="$(sed -n "s|^LasHeR_PATH: '\(.*\)'|\1|p" "$CONSTS")"
DS="${DS%/}"
[[ -d "$DS/testingset/$SEQ" ]] || { echo "FATAL: no such sequence: $DS/testingset/$SEQ" >&2; exit 1; }
for item in trainingset testingset annos AttriSeqsTxt Attributes_order.txt; do
  [[ -e "$DS/$item" ]] && ln -sfn "$DS/$item" "$VIEW/$item"
done

restore() { if [[ -f "$BACKUP" ]]; then mv -f "$BACKUP" "$CONSTS"; echo "[restored] consts.yaml"; fi; }
trap restore EXIT

cp -a "$CONSTS" "$BACKUP"
"$PYTHON" - "$CONSTS" "$VIEW" <<'PY'
import sys
path, view = sys.argv[1], sys.argv[2]
s = open(path).read(); out, hit = [], 0
for line in s.splitlines(keepends=True):
    if line.startswith('LasHeR_PATH:'):
        line = f"LasHeR_PATH: '{view}/'\n"; hit += 1
    out.append(line)
assert hit == 1, f"expected 1 LasHeR_PATH line, found {hit}"
open(path, 'w').write(''.join(out))
PY
echo "[swap] consts.LasHeR_PATH -> $VIEW/"
echo "[run ] CodeTrack inference on sequence: $SEQ"

ANS="$OUT/answer.bin"
if [[ -f "$ANS" ]]; then
  WEIGHT_ARG=(--weight_path "$ANS")
  echo "[init] using the fine-tuned checkpoint $ANS"
else
  WEIGHT_ARG=(--weight_path "$WEIGHT")
  echo "[init] using the pretrained GOLA checkpoint $WEIGHT"
fi

"$PYTHON" main.py GOLA "$CONFIG" \
  --eval \
  --distributed_nproc_per_node 1 \
  --device cuda \
  --disable_wandb \
  "${WEIGHT_ARG[@]}" \
  --output_dir="$OUT"
