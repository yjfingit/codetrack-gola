#!/usr/bin/env python3
"""Produce a paired, official LasHeR challenge-attribute analysis.

The LasHeR attributes overlap by design.  This report therefore keeps the paired
per-sequence differences and reports both mean and improvement rate instead of
pretending that the groups form a partition.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


ROW = re.compile(r"^([^:]+): success ([0-9.]+), prec ([0-9.]+), norm_pre ([0-9.]+)$")
MEANING = {
    "NO": "无遮挡", "PO": "部分遮挡", "TO": "完全遮挡", "HO": "重遮挡",
    "MB": "运动模糊", "LI": "低照度", "HI": "高照度", "AIV": "照度突变",
    "LR": "低分辨率", "DEF": "形变", "BC": "背景杂乱", "SA": "相似外观",
    "CM": "摄像机运动", "TC": "热交叉", "FL": "快速行进", "OV": "出视野",
    "FM": "快速运动", "SV": "尺度变化", "ARV": "宽高比变化",
}


def read_metrics(path: Path) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for line in path.read_text(errors="replace").splitlines():
        match = ROW.match(line.strip())
        if match:
            result[match.group(1)] = np.asarray(
                [float(match.group(i)) for i in (2, 3, 4)], dtype=np.float64
            )
    return result


def read_flags(dataset: Path, names: list[str], attrs: list[str]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name in names:
        path = dataset / "AttriSeqsTxt" / f"{name}.txt"
        if not path.exists():
            continue
        bits = [int(char) for char in "".join(path.read_text(errors="replace").split()) if char in "01"]
        if len(bits) == len(attrs):
            result[name] = np.asarray(bits, dtype=bool)
    return result


def bootstrap_mean(values: np.ndarray, seed: int = 0, rounds: int = 4000) -> list[float]:
    if len(values) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(values), size=(rounds, len(values)))
    means = values[sample].mean(axis=1)
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def make_report(dataset: Path, baseline: Path, candidate: Path) -> dict:
    attrs = [x.strip() for x in (dataset / "Attributes_order.txt").read_text().split(",") if x.strip()]
    base = read_metrics(baseline)
    cand = read_metrics(candidate)
    names = sorted(set(base) & set(cand))
    flags = read_flags(dataset, names, attrs)
    rows = [
        {"name": name, "flags": flags[name].tolist(), "delta": (cand[name] - base[name]).tolist()}
        for name in names if name in flags
    ]

    groups = []
    for index, attr in enumerate(attrs):
        selected = [row for row in rows if row["flags"][index]]
        if not selected:
            continue
        delta = np.asarray([row["delta"] for row in selected], dtype=np.float64)
        sr, pr, npr = delta.T
        groups.append({
            "attribute": attr,
            "meaning": MEANING.get(attr, attr),
            "count": len(selected),
            "delta_sr_pp": float(sr.mean() * 100),
            "delta_pr_pp": float(pr.mean() * 100),
            "delta_npr_pp": float(npr.mean() * 100),
            "median_delta_sr_pp": float(np.median(sr) * 100),
            "sr_std_pp": float(sr.std(ddof=1) * 100) if len(sr) > 1 else 0.0,
            "sr_improved_fraction": float(np.mean(sr > 0)),
            "sr_worsened_fraction": float(np.mean(sr < 0)),
            "sr_ci95_pp": [x * 100 for x in bootstrap_mean(sr)],
        })
    groups.sort(key=lambda row: row["delta_sr_pp"], reverse=True)
    overall = np.asarray([row["delta"] for row in rows], dtype=np.float64)
    report = {
        "dataset": str(dataset),
        "baseline_log": str(baseline),
        "candidate_log": str(candidate),
        "sequence_count": len(rows),
        "attributes_are_overlapping": True,
        "metrics": ["success (SR)", "prec (PR)", "norm_pre (NPR)"],
        "groups_ranked_by_sr": groups,
        "overall_paired_delta_pp": {
            "sr": float(overall[:, 0].mean() * 100),
            "pr": float(overall[:, 1].mean() * 100),
            "npr": float(overall[:, 2].mean() * 100),
        },
        "sequence_delta": sorted(
            [{"name": row["name"], "delta_sr_pp": row["delta"][0] * 100,
              "delta_pr_pp": row["delta"][1] * 100, "delta_npr_pp": row["delta"][2] * 100}
             for row in rows],
            key=lambda row: row["delta_sr_pp"], reverse=True,
        ),
    }
    return report


def write_markdown(report: dict, output: Path) -> None:
    lines = [
        "# LasHeR 挑战属性配对分析",
        "",
        f"- 配对序列：{report['sequence_count']} 条",
        f"- 整体逐序列均值差：SR {report['overall_paired_delta_pp']['sr']:+.2f} pp，"
        f"PR {report['overall_paired_delta_pp']['pr']:+.2f} pp，"
        f"NPR {report['overall_paired_delta_pp']['npr']:+.2f} pp",
        "- 属性是官方定义的重叠标签，一条序列可同时属于多个挑战；不能把各行相加。",
        "- `CI95` 是对逐序列 SR 差值均值的 bootstrap 区间；样本数很少的属性只作诊断。",
        "",
        "| 属性 | 含义 | 序列数 | SR差 | PR差 | NPR差 | SR改善比例 | SR恶化比例 | SR CI95 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["groups_ranked_by_sr"]:
        ci = row["sr_ci95_pp"]
        lines.append(
            f"| `{row['attribute']}` | {row['meaning']} | {row['count']} | "
            f"{row['delta_sr_pp']:+.2f} | {row['delta_pr_pp']:+.2f} | {row['delta_npr_pp']:+.2f} | "
            f"{row['sr_improved_fraction'] * 100:.1f}% | {row['sr_worsened_fraction'] * 100:.1f}% | "
            f"[{ci[0]:+.2f}, {ci[1]:+.2f}] |"
        )
    lines += ["", "## SR 最明显改善的序列", "", "| 序列 | SR差 | PR差 |", "|---|---:|---:|"]
    for row in report["sequence_delta"][:12]:
        lines.append(f"| `{row['name']}` | {row['delta_sr_pp']:+.2f} | {row['delta_pr_pp']:+.2f} |")
    lines += ["", "## SR 最明显退化的序列", "", "| 序列 | SR差 | PR差 |", "|---|---:|---:|"]
    for row in report["sequence_delta"][-12:][::-1]:
        lines.append(f"| `{row['name']}` | {row['delta_sr_pp']:+.2f} | {row['delta_pr_pp']:+.2f} |")
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("/home/yangjuanfeng/lab/dataset/LasHeR"))
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    report = make_report(args.dataset, args.baseline, args.candidate)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    write_markdown(report, args.markdown)
    print(json.dumps({"sequence_count": report["sequence_count"], "groups": len(report["groups_ranked_by_sr"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
