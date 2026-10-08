"""Drop explicitly named tensor prefixes from a safetensors checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drop-prefix", action="append", required=True)
    args = parser.parse_args()

    state = load_file(str(args.input), device="cpu")
    prefixes = tuple(args.drop_prefix)
    kept = {key: value for key, value in state.items() if not key.startswith(prefixes)}
    removed = sorted(set(state) - set(kept))
    if not removed:
        raise ValueError(f"no tensors matched prefixes {prefixes}")
    with safe_open(str(args.input), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
    metadata.update({"filtered_from": str(args.input), "dropped_prefixes": "|".join(prefixes)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(kept, str(args.output), metadata=metadata)
    print(f"wrote {args.output}: kept {len(kept)}, removed {len(removed)} tensors")


if __name__ == "__main__":
    main()
