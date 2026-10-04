#!/usr/bin/env bash
# Official GOLA eval pipeline over ONE LasHeR sequence.
#
#   bash scripts/single_seq_eval.sh            # sequence 10runone (default)
#   SEQ=11leftboy bash scripts/single_seq_eval.sh
#
# trackit reads the .txt sequence list from <LasHeR_PATH> directly
# (trackit/datasets/MMOT/datasets/LasHeR.py), and consts.yaml is loaded globally rather
# than through the config tree, so a mixin cannot redirect it. TRACKIT_CONSTS_PATH gives this
# process an isolated constants copy containing a one-line testingsetList.txt view.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck source=/dev/null
source scripts/00_env.sh

SEQ="${SEQ:-10runone}"
VIEW="$PWD/data/LasHeR_single"
OUT="${OUT:-$PWD/outputs/eval_single_$SEQ}"
CONSTS="$PWD/consts.yaml"

# ---- build the one-sequence view (read-only symlinks; dataset untouched) -------
mkdir -p "$VIEW" "$OUT"
printf '%s\n' "$SEQ" > "$VIEW/testingsetList.txt"
DS="$(sed -n "s|^LasHeR_PATH: '\(.*\)'|\1|p" "$CONSTS")"
DS="${DS%/}"
[[ -d "$DS" ]] || { echo "FATAL: LasHeR_PATH not a directory: $DS" >&2; exit 1; }
[[ -d "$DS/testingset/$SEQ" ]] || { echo "FATAL: no such sequence: $DS/testingset/$SEQ" >&2; exit 1; }
for item in testingset annos AttriSeqsTxt Attributes_order.txt; do
  [[ -e "$DS/$item" ]] && ln -sfn "$DS/$item" "$VIEW/$item"
done

# ---- make a process-local constants file for the single-sequence view ----------
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
echo "[run ] evaluating sequence: $SEQ"

# ---- official eval path ------------------------------------------------------
"$PYTHON" main.py GOLA dinov2 \
  --eval \
  --mixin_config evaluation \
  --mixin_config disable_torch_compile \
  --mixin_config eval_bs_8 \
  --distributed_nproc_per_node 1 \
  --device cuda \
  --disable_wandb \
  --weight_path "$WEIGHT" \
  --output_dir="$OUT"
