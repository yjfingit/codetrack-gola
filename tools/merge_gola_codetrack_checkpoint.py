"""Merge a sparse CodeTrack training checkpoint over the complete GOLA baseline."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = load_file(str(args.base), device="cpu")
    trained = load_file(str(args.trained), device="cpu")
    unexpected = sorted(set(trained) - (set(base) | {k for k in trained if k.startswith("codetrack.")}))
    if unexpected:
        raise ValueError(f"unexpected trained keys: {unexpected[:12]}")
    missing = sorted(set(base) - set(trained))
    if missing:
        raise ValueError(f"trained checkpoint omits baseline keys: {missing[:12]}")

    merged = {**base, **trained}
    bad = [key for key, value in merged.items()
           if value.is_floating_point() and not bool(torch.isfinite(value).all())]
    if bad:
        raise ValueError(f"non-finite checkpoint tensors: {bad[:12]}")

    metadata = {}
    with safe_open(str(args.base), framework="pt", device="cpu") as handle:
        metadata.update(handle.metadata() or {})
    metadata.update({"merged_base": str(args.base), "trained_delta": str(args.trained)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(args.output), metadata=metadata)
    print({"output": str(args.output), "base_keys": len(base),
           "trained_keys": len(trained), "merged_keys": len(merged),
           "codetrack_keys": sum(key.startswith("codetrack.") for key in merged)})


if __name__ == "__main__":
    main()
