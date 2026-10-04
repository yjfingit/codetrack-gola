#!/usr/bin/env bash
# Wait until the preflight training log reaches TARGET micro-iterations, then summarise.
#   bash scripts/codetrack_wait_progress.sh <log> <target_iters> [max_minutes]
set -uo pipefail
LOG="${1:?usage: $0 <log> <target_iters> [max_minutes]}"
TARGET="${2:?}"
MAXMIN="${3:-60}"
DEADLINE=$(( $(date +%s) + MAXMIN * 60 ))
while :; do
  N=$(grep -c "^Epoch" "$LOG" 2>/dev/null || echo 0)
  if [[ "$N" -ge "$TARGET" ]]; then
    echo "REACHED $N >= $TARGET micro-iterations ($(( N / 16 )) optimizer updates)"
    break
  fi
  if [[ $(date +%s) -gt $DEADLINE ]]; then
    echo "TIMEOUT after ${MAXMIN}m at $N/$TARGET micro-iterations"
    break
  fi
  sleep 20
done
echo "=== last line ==="
grep "^Epoch" "$LOG" | tail -1
echo "=== metrics at milestones (single-batch value; the (x) is the running EMA) ==="
# the parsing is easier to keep correct in one place than in bash string surgery
"${PYTHON:-python3}" - "$LOG" <<'PY'
import re, sys
keys = ["lr", "grad_norm", "loss", "Loss/cls", "Loss/diag", "Loss/rec", "Loss/gain",
        "Loss/align", "Loss/track_corr", "Error/q_auroc", "Error/d_before", "Error/d_after",
        "Error/gain", "Error/corrupted_fraction"]
rows = []
for line in open(sys.argv[1], errors="replace"):
    m = re.match(r"Epoch: \[\d+\] \[ *(\d+)/\d+\]", line)
    if not m:
        continue
    vals = {}
    for k in keys:
        mm = re.search(re.escape(k) + r": (-?[\d.]+|nan)", line)
        if mm:
            vals[k] = mm.group(1)
    rows.append((int(m.group(1)), vals))
print("logged micro-iterations:", len(rows), "-> optimizer updates:", len(rows) // 16)
print(f"{'iter':>6} {'upd':>5} " + " ".join(f"{k.split('/')[-1]:>12}" for k in keys))
for milestone in (0, 400, 800, 1600, 2400, 3200, 4000, 4800):
    pick = next((r for r in rows if r[0] >= milestone), None)
    if pick is None:
        continue
    it, v = pick
    print(f"{it:>6} {it // 16:>5} " + " ".join(f"{v.get(k, '-'):>12}" for k in keys))
PY
