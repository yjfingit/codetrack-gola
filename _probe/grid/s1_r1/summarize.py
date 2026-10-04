#!/usr/bin/env python3
"""Summarise the S1 round-1 grid: last-window metrics per trial."""
import glob
import os
import re
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "."
win = int(sys.argv[2]) if len(sys.argv) > 2 else 10

PAT = {
    "gain": r"Error/gain_total: *([-+0-9.eE]+)",
    "spr": r"q_error_spearman: *([-+0-9.eE]+)",
    "d_in": r"Error/d_input: *([-+0-9.eE]+)",
    "d_aft": r"Error/d_after: *([-+0-9.eE]+)",
    "auroc": r"Error/q_auroc_mask: *([-+0-9.eE]+)",
    "diag": r"Loss/diag: *([-+0-9.eE]+)",
    "qstd": r"Error/q_std: *([-+0-9.eE]+)",
    "gn": r"grad_norm: *([-+0-9.eE]+)",
}
RE_UPD = re.compile(r"\[ *(\d+)/\d+\]")


def col(lines, key):
    rx = re.compile(PAT[key])
    out = []
    for ln in lines:
        m = rx.search(ln)
        if m:
            out.append(float(m.group(1)))
    return out


hdr = f"{'trial':<16}{'upd':>6}{'gain':>10}{'spearman':>10}{'d_input':>9}{'d_after':>9}{'auroc':>8}{'q_std':>9}"
print(hdr)
print("-" * len(hdr))
rows = []
for f in sorted(glob.glob(os.path.join(root, "gpu*.log"))):
    lines = [l for l in open(f, errors="replace") if l.startswith("Epoch:")]
    if not lines:
        continue
    m = RE_UPD.search(lines[-1])
    upd = int(m.group(1)) // 8 if m else -1

    def avg(k):
        v = col(lines, k)[-win:]
        return sum(v) / len(v) if v else float("nan")

    name = os.path.basename(f)[:-4]
    r = dict(name=name, upd=upd, gain=avg("gain"), spr=avg("spr"),
             d_in=avg("d_in"), d_aft=avg("d_aft"), auroc=avg("auroc"),
             qstd=avg("qstd"), diag=avg("diag"), gn=avg("gn"))
    rows.append(r)
    print(f"{name:<16}{upd:>6}{r['gain']:>10.5f}{r['spr']:>10.4f}"
          f"{r['d_in']:>9.4f}{r['d_aft']:>9.4f}{r['auroc']:>8.4f}{r['qstd']:>9.2e}")

print()
print("=== ranking ===")
print("by gain   :", ", ".join(x["name"] for x in sorted(rows, key=lambda z: -z["gain"])))
print("by spearman:", ", ".join(x["name"] for x in sorted(rows, key=lambda z: -z["spr"])))
print()
print("gate: gain > 0.005, spearman > 0.25")
for r in sorted(rows, key=lambda z: -z["gain"]):
    ok_g = "PASS" if r["gain"] > 0.005 else "FAIL"
    ok_s = "PASS" if r["spr"] > 0.25 else "FAIL"
    print(f"  {r['name']:<16} gain {r['gain']:.5f} [{ok_g}]   spearman {r['spr']:.4f} [{ok_s}]")
