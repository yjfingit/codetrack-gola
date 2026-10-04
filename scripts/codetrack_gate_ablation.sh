#!/usr/bin/env bash
# Residual-gate ablation: three arms, same seed, same fixed batch pool, 600 updates each.
#
#   bash scripts/codetrack_gate_ablation.sh [UPDATES] [OUTDIR]
#
# Arms
#   A  gate_init -8, gate trainable   (the current stage config)
#   B  gate_init -5, gate trainable   (the reviewer's suggestion)
#   C  gate_init -5, gate FROZEN      (control: isolates "gate pushed open" from
#                                      "the correction branch itself moves tokens away")
#
# Why arm C matters: the write-back is scaled by sigmoid(residual_gate), which starts at 3.4e-4.
# A measured d_after / d_input ratio of ~4.8 therefore cannot come from the gate staying closed --
# either the gate has opened (armed by Loss/rec, which unlike a gain term has no anchor to the
# input) or the branch moves tokens regardless.  Arm C answers which, and it is the arm that
# decides whether the fix belongs in the initialisation or in the objective.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck source=/dev/null
source scripts/00_env.sh

UPDATES="${1:-600}"
OUT="${2:-$PWD/outputs/gate_ablation}"
mkdir -p "$OUT"
echo "[ablation] updates=$UPDATES  out=$OUT"

run_arm() {
  local name="$1"; shift
  echo "[ablation] === arm $name ==="
  "$PYTHON" tools/recovery_report.py "$@" --updates "$UPDATES" --report-every 50 \
    > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "[ablation] arm $name exit=$rc"
  tail -3 "$OUT/$name.log"
}

run_arm A_neg8         --gates -8
run_arm B_neg5         --gates -5
run_arm C_neg5_frozen  --gates -5 --freeze-gate

echo
echo "===================== SUMMARY ====================="
for f in A_neg8 B_neg5 C_neg5_frozen; do
  echo "--- $f ---"
  # the milestone table plus the verdict line
  grep -E "^ *[0-9]+ +[0-9]|VERDICT|RESULT" "$OUT/$f.log" | tail -14
done
echo "==================================================="
echo "[ablation] logs in $OUT"
