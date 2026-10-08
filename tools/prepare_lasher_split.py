#!/usr/bin/env python3
"""Create a deterministic sequence-disjoint LasHeR train/validation view.

LasHeR's loader reads ``trainingsetList.txt`` from the dataset root.  The loader also
accepts an explicit ``path`` in a dataset config, so two lightweight roots containing
different lists let train and validation coexist in one DDP job without changing the
global ``consts.yaml``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def sha256_lines(lines: list[str]) -> str:
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20261008)
    args = ap.parse_args()
    root = args.dataset_root.resolve()
    source = root / "trainingsetList.txt"
    if not source.is_file():
        raise FileNotFoundError(source)
    names = [x.strip() for x in source.read_text().splitlines() if x.strip()]
    if len(set(names)) != len(names):
        raise ValueError("trainingsetList.txt contains duplicate sequence names")
    if not 0 < args.val_fraction < 0.5:
        raise ValueError("val-fraction must be in (0, 0.5)")

    shuffled = names[:]
    random.Random(args.seed).shuffle(shuffled)
    n_val = max(1, round(len(names) * args.val_fraction))
    val = sorted(shuffled[:n_val])
    train = sorted(shuffled[n_val:])
    if set(train) & set(val) or set(train) | set(val) != set(names):
        raise AssertionError("split is not a disjoint complete partition")

    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for label, subset in (("train", train), ("val", val)):
        view = out / label
        view.mkdir(parents=True, exist_ok=True)
        list_path = view / "trainingsetList.txt"
        list_path.write_text("\n".join(subset) + "\n")
        # The constructor expects these entries under the root.  Symlinks avoid duplicating
        # the image corpus while preserving the exact upstream LasHeR layout.
        for entry in ("trainingset", "testingset", "annos", "AttriSeqsTxt", "Attributes_order.txt"):
            target = root / entry
            link = view / entry
            if target.exists() and not link.exists():
                link.symlink_to(target, target_is_directory=target.is_dir())
        (view / "trainingsetList.sha256").write_text(sha256_lines(subset) + "\n")

    manifest = {
        "dataset_root": str(root),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "total_sequences": len(names),
        "train_sequences": len(train),
        "val_sequences": len(val),
        "train_sha256": sha256_lines(train),
        "val_sha256": sha256_lines(val),
        "all_sha256": sha256_lines(sorted(names)),
        "disjoint": True,
        "train_list": str(out / "train" / "trainingsetList.txt"),
        "val_list": str(out / "val" / "trainingsetList.txt"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
