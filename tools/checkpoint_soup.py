"""Average a declared subset of tensors across compatible safetensors checkpoints."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


DEFAULT_PREFIXES = (
    "codetrack.refiner.",
    "codetrack.condition_proj.",
    "codetrack.denoiser.",
    "codetrack.meanvar.",
)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", action="append", default=None)
    args = parser.parse_args()

    if len(args.input) < 2:
        raise ValueError("checkpoint soup requires at least two --input checkpoints")
    prefixes = tuple(args.prefix or DEFAULT_PREFIXES)
    states = [load_file(str(path), device="cpu") for path in args.input]
    keys = tuple(states[0])
    for path, state in zip(args.input[1:], states[1:]):
        if tuple(state) != keys:
            raise ValueError(f"checkpoint keys differ: {path}")
        for key in keys:
            if state[key].shape != states[0][key].shape or state[key].dtype != states[0][key].dtype:
                raise ValueError(f"incompatible tensor {key!r} in {path}")

    selected = [key for key in keys if key.startswith(prefixes)]
    if not selected:
        raise ValueError(f"no tensors matched prefixes {prefixes}")
    output = {key: value.clone() for key, value in states[0].items()}
    for key in selected:
        if not states[0][key].is_floating_point():
            if not all(torch.equal(states[0][key], state[key]) for state in states[1:]):
                raise ValueError(f"selected non-floating tensor differs: {key}")
            continue
        output[key] = torch.stack([state[key].to(torch.float64) for state in states]).mean(0).to(
            states[0][key].dtype)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(output, str(args.output), metadata={
        "method": "equal_weight_subset_soup",
        "sources": "|".join(str(path) for path in args.input),
        "prefixes": "|".join(prefixes),
    })
    print(f"wrote {args.output}: averaged {len(selected)}/{len(keys)} tensors from "
          f"{len(states)} checkpoints")


if __name__ == "__main__":
    main()
