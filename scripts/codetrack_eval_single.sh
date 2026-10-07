#!/usr/bin/env bash
# Single-sequence inference with the CodeTrack model (official eval pipeline).
#
#   bash scripts/codetrack_eval_single.sh
#
# Point this process at a run-local constants file; never mutate the shared consts.yaml.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck source=/dev/null
source scripts/00_env.sh

SEQ="${SEQ:-10runone}"
CONFIG="${CONFIG:-codetrack_eval}"
MIXIN_CONFIG="${MIXIN_CONFIG:-}"
VIEW="$PWD/data/LasHeR_single"
OUT="${OUT:-$PWD/outputs/codetrack_eval_$SEQ}"
CONSTS="$PWD/consts.yaml"

mkdir -p "$VIEW" "$OUT"
printf '%s\n' "$SEQ" > "$VIEW/testingsetList.txt"
DS="$(sed -n "s|^LasHeR_PATH: '\(.*\)'|\1|p" "$CONSTS")"
DS="${DS%/}"
[[ -d "$DS/testingset/$SEQ" ]] || { echo "FATAL: no such sequence: $DS/testingset/$SEQ" >&2; exit 1; }
for item in trainingset testingset annos AttriSeqsTxt Attributes_order.txt; do
  [[ -e "$DS/$item" ]] && ln -sfn "$DS/$item" "$VIEW/$item"
done

RUN_CONSTS="$(mktemp "$OUT/consts.XXXXXX.yaml")"
trap 'rm -f "$RUN_CONSTS"' EXIT
cp -a "$CONSTS" "$RUN_CONSTS"
"$PYTHON" - "$RUN_CONSTS" "$VIEW" <<'PY'
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
export TRACKIT_CONSTS_PATH="$RUN_CONSTS"
echo "[view] TRACKIT_CONSTS_PATH=$RUN_CONSTS (LasHeR_PATH -> $VIEW/)"
echo "[run ] CodeTrack inference on sequence: $SEQ"

ANS="$OUT/answer.bin"
if [[ -f "$ANS" ]]; then
  WEIGHT_ARG=(--weight_path "$ANS")
  echo "[init] using the fine-tuned checkpoint $ANS"
else
  WEIGHT_ARG=(--weight_path "$WEIGHT")
  echo "[init] using the pretrained GOLA checkpoint $WEIGHT"
fi

MIXIN_ARGS=()
if [[ -n "$MIXIN_CONFIG" ]]; then
  MIXIN_ARGS+=(--mixin_config "$MIXIN_CONFIG")
fi

"$PYTHON" main.py GOLA "$CONFIG" \
  --eval \
  --distributed_nproc_per_node 1 \
  --device cuda \
  --disable_wandb \
  "${MIXIN_ARGS[@]}" \
  "${WEIGHT_ARG[@]}" \
  --output_dir="$OUT"
