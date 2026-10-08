#!/usr/bin/env python3
"""Compare LasHeR tracking metrics by the dataset's official attributes.

The evaluator prints one ``success/prec/norm_pre`` row per sequence.  LasHeR ships the
official sequence attributes as one binary row in ``AttriSeqsTxt``; this script joins the
two sources and reports paired GOLA-versus-CodeTrack differences without inventing a new
split or re-running inference.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


ROW = re.compile(
    r"^([^:]+): success ([0-9.]+), prec ([0-9.]+), norm_pre ([0-9.]+)$"
)

# Names used by the official LasHeR release.  Keep the original abbreviations in every
# report; expanded labels are only for readability and do not alter the grouping.
KNOWN = {
    "NO": "no occlusion",
    "PO": "partial occlusion",
    "TO": "total occlusion",
    "HO": "heavy occlusion",
    "MB": "motion blur",
    "LI": "low illumination",
    "HI": "high illumination",
    "AIV": "abrupt illumination variation",
    "LR": "low resolution",
    "DEF": "deformation",
    "BC": "background clutter",
    "SA": "similar appearance",
    "CM": "camera motion",
    "TC": "thermal crossover",
    "FL": "fast locomotion",
    "OV": "out of view",
    "FM": "fast motion",
    "SV": "scale variation",
    "ARV": "aspect-ratio variation",
}


def read_metrics(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for line in path.read_text(errors="replace").splitlines():
        match = ROW.match(line.strip())
        if match:
            out[match.group(1)] = np.asarray(tuple(float(match.group(i)) for i in (2, 3, 4)))
    return out


def read_attrs(dataset: Path, names: list[str], attrs: list[str]) -> dict[str, np.ndarray]:
    out = {}
    for name in names:
        path = dataset / "AttriSeqsTxt" / f"{name}.txt"
        if not path.exists():
            continue
        raw = "".join(path.read_text(errors="replace").split())
        # Releases differ only in whether commas are present.
        bits = [int(c) for c in raw if c in "01"]
        if len(bits) != len(attrs):
            raise ValueError(f"{path}: expected {len(attrs)} flags, got {len(bits)}")
        out[name] = np.asarray(bits, dtype=bool)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("/home/yangjuanfeng/lab/dataset/LasHeR"))
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    attrs = [x.strip() for x in (args.dataset / "Attributes_order.txt").read_text().split(",") if x.strip()]
    base = read_metrics(args.baseline)
    cand = read_metrics(args.candidate)
    names = sorted(set(base) & set(cand))
    flags = read_attrs(args.dataset, names, attrs)
    rows = []
    for name in names:
        if name not in flags:
            continue
        rows.append({"name": name, "flags": flags[name], "baseline": base[name], "candidate": cand[name]})

    report = {
        "baseline_log": str(args.baseline),
        "candidate_log": str(args.candidate),
        "dataset": str(args.dataset),
        "sequence_count": len(rows),
        "metrics": ["success", "precision", "normalized_precision"],
        "attributes": attrs,
        "attribute_names": {a: KNOWN.get(a, "official LasHeR attribute") for a in attrs},
        "groups": [],
    }
    for index, attr in enumerate(attrs):
        group = [r for r in rows if bool(r["flags"][index])]
        if not group:
            continue
        b = np.stack([r["baseline"] for r in group])
        c = np.stack([r["candidate"] for r in group])
        report["groups"].append({
            "attribute": attr,
            "meaning": KNOWN.get(attr, "official LasHeR attribute"),
            "count": len(group),
            "baseline": b.mean(0).tolist(),
            "candidate": c.mean(0).tolist(),
            "delta": (c.mean(0) - b.mean(0)).tolist(),
        })
    all_b = np.stack([r["baseline"] for r in rows])
    all_c = np.stack([r["candidate"] for r in rows])
    report["overall_paired_mean"] = {
        "baseline": all_b.mean(0).tolist(),
        "candidate": all_c.mean(0).tolist(),
        "delta": (all_c.mean(0) - all_b.mean(0)).tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
