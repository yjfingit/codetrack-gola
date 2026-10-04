"""Rank grid-search trials: parse each trial log, extract the final training
loss and the CodeTrack diagnostic quantities, then sort.

Usage:
    grid_report.py <grid_root>

Reads   <grid_root>/logs/*.log
Writes  <grid_root>/summary.json  and  <grid_root>/summary.md
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

NUM = r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"

# "step: 123 ... total: 1.2345"  -> keep the LAST one
RE_STEP = re.compile(rf"step[:\s]+(\d+)")
RE_TOTAL = re.compile(rf"total[:\s]+({NUM})")
RE_LOSS = re.compile(rf"(?:^|\s)loss[:\s]+({NUM})")

# CodeTrack diagnostic keys, e.g. "Error/d_input: 0.1813"
DIAG_KEYS = [
    "Loss/diag", "Loss/rec", "Loss/gain", "Loss/gate", "Loss/mem",
    "Error/q_std", "Error/d_input", "Error/d_after", "Error/gain_total",
    "Error/syndrome_std", "Error/q_mean",
]


def scan(path: Path):
    rec = {"trial": path.stem, "steps": 0, "loss_last": None,
           "loss_min": None, "loss_first": None, "diag": {}, "nan": 0,
           "grad_norm_last": None}
    txt = path.read_text(errors="replace")
    for line in txt.splitlines():
        m = RE_STEP.search(line)
        if m:
            rec["steps"] = max(rec["steps"], int(m.group(1)))
        for rx, key in ((RE_TOTAL, "loss"), (RE_LOSS, "loss2")):
            mm = rx.search(line)
            if mm:
                try:
                    v = float(mm.group(1))
                except ValueError:
                    continue
                if key == "loss":
                    rec["loss_last"] = v
                    rec["loss_min"] = v if rec["loss_min"] is None else min(rec["loss_min"], v)
                    if rec["loss_first"] is None:
                        rec["loss_first"] = v
        if "nan" in line.lower() and ("grad_norm" in line or "loss" in line):
            rec["nan"] += 1
        for k in DIAG_KEYS:
            mm = re.search(rf"{re.escape(k)}[:\s=]+({NUM})", line)
            if mm:
                rec["diag"][k] = float(mm.group(1))
    return rec


def main() -> int:
    root = Path(sys.argv[1])
    logs = sorted((root / "logs").glob("*.log"))
    if not logs:
        print(f"no logs under {root/'logs'}")
        return 2

    rows = [scan(p) for p in logs]

    def keyf(r):
        # lower final loss is better; NaN-containing runs go last
        v = r["loss_last"]
        return (r["nan"] > 0, v if v is not None else 9e9)

    rows.sort(key=keyf)

    (root / "summary.json").write_text(json.dumps(rows, indent=2))

    md = ["# Grid search results", "",
          f"trials: {len(rows)}", "",
          "| rank | trial | steps | final loss | min loss | NaN lines | d_input | d_after | gain |",
          "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        d = r["diag"]
        md.append(
            f"| {i} | `{r['trial']}` | {r['steps']} | "
            f"{r['loss_last'] if r['loss_last'] is not None else '-'} | "
            f"{r['loss_min'] if r['loss_min'] is not None else '-'} | "
            f"{r['nan']} | "
            f"{d.get('Error/d_input', '-')} | "
            f"{d.get('Error/d_after', '-')} | "
            f"{d.get('Error/gain_total', '-')} |")
    (root / "summary.md").write_text("\n".join(md) + "\n")

    print("\n".join(md))
    print(f"\nwrote {root/'summary.json'} and {root/'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
