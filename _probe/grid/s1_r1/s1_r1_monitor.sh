#!/usr/bin/env bash
# CodeTrack S1 round-1: quick status + hard-metric readout.
#
#   bash s1_r1_monitor.sh <grid_name> [step]     # default step = latest
#
# Prints, per arm: last step reached, and the values AT (or nearest below) the given
# optimizer step for the three hard metrics:
#     Error/gain_total      need  > 0.005
#     Error/q_error_spearman  want  > 0.25
#     loss / grad_norm        need  stable (no NaN, not exploding)
#
# The trainer logs one line per MICRO step; 1 optimizer update = accum 8 micro steps.
set -u
PROJ=/home/yangjuanfeng/lab/projects/gola-CodeTrack
GRID="${1:?grid name}"
AT="${2:-0}"
ROOT="$PROJ/_probe/grid/$GRID"

printf '%-16s %8s %8s %10s %10s %10s %10s %8s\n' \
  trial micro last_upd gain_total spearman q_std d_after grad_norm
for f in "$ROOT"/logs/*.log; do
  [[ -f "$f" ]] || continue
  t=$(basename "$f" .log)
  awk -v T="$t" -v AT="$AT" '
    /Epoch:/ {
      if (match($0, /\[ *[0-9]+\/[0-9]+\]/)) {
        s=substr($0, RSTART, RLENGTH); gsub(/[^0-9\/]/,"",s); split(s,a,"/"); micro=a[1]+0
      }
      if (match($0, /gain_total: *[-+0-9.eE]+/)) { v=substr($0, RSTART, RLENGTH); sub(/.*: */,"",v); g=v+0 }
      if (match($0, /q_error_spearman: *[-+0-9.eE]+/)) { v=substr($0, RSTART, RLENGTH); sub(/.*: */,"",v); sp=v+0 }
      if (match($0, /q_std: *[-+0-9.eE]+/)) { v=substr($0, RSTART, RLENGTH); sub(/.*: */,"",v); qs=v+0 }
      if (match($0, /d_after: *[-+0-9.eE]+/)) { v=substr($0, RSTART, RLENGTH); sub(/.*: */,"",v); da=v+0 }
      if (match($0, /grad_norm: *[-+0-9.eE]+/)) { v=substr($0, RSTART, RLENGTH); sub(/.*: */,"",v); gn=v+0 }
      upd=int(micro/8)
      if (AT==0 || upd<=AT) { LG=g; LSP=sp; LQS=qs; LDA=da; LGN=gn; LU=upd }
      LM=micro
    }
    END { printf "%-16s %8d %8d %10.5f %10.4f %10.3e %10.4f %8.4f\n", T, LM, LU, LG, LSP, LQS, LDA, LGN }
  ' "$f" 2>/dev/null
done
echo
echo "exit codes:"; for e in "$ROOT"/logs/*.exit; do [[ -f "$e" ]] && echo "  $(basename "$e" .exit) = $(cat "$e")"; done
echo
echo "running procs:"; ps -u yangjuanfeng -o pid,etime,args | grep '[m]ain.py GOLA' | head
