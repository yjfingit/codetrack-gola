"""Causal probe for the dense CodeTrack denoiser write-back support.

Loads a trained S1 checkpoint on top of the official GOLA weights, holds one real LasHeR
batch fixed, and changes only the per-token denoiser write gate.  This is an observation-only
probe: no parameters are updated and the checkpoint is never modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--mode", required=True,
                        choices=("current", "topk", "oracle_soft", "oracle_unit"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--topk", type=int, default=None,
                        help="override the hard support size in topk mode")
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    from trackit.models import ModelImplSuggestions
    from trackit.models.methods.GOLA.builder import build_GOLA_model
    from tools.preflight_acceptance import load_stage_config
    from tools.verify_codetrack_initialization import real_batch

    torch.manual_seed(args.seed)
    view = ROOT / "_probe/codetrack_flow_validation/LasHeR_curated10"
    names = (view / "trainingsetList.txt").read_text().splitlines()
    data = real_batch(view, names)

    cfg = load_stage_config(str(ROOT / "config/GOLA/codetrack_s1/config.yaml"))
    if args.topk is not None:
        cfg["model"]["codetrack"]["topk_tokens"] = int(args.topk)
    cfg["model"]["codetrack"]["corruption_enabled"] = False
    model = build_GOLA_model(cfg, ModelImplSuggestions()).cuda().eval()
    model.load_state_dict(load_file(str(ROOT / "weights/gola_b224.bin")), strict=False)
    checkpoint_state = load_file(str(args.checkpoint))
    loaded = model.load_state_dict(checkpoint_state, strict=False)
    ct = model.codetrack

    captured = []
    capture = ct.register_forward_pre_hook(
        lambda _module, _inputs, kwargs: captured.append(kwargs["F_L"].detach().clone()),
        with_kwargs=True)
    with torch.no_grad():
        model.reset_sequence()
        model(**data)
        clean = captured[-1]
        damaged_data = dict(data)
        damaged_data["x"] = data["x"].clone()
        damaged_data["x"][:, 3:, 84:140, 84:140] = 0
        model.reset_sequence()
        model(**damaged_data)
        damaged = captured[-1]
    capture.remove()

    teacher = ct._split(clean)["X_TIR"].detach()
    x_input = ct._split(damaged)["X_TIR"].detach()
    input_error = (1 - F.cosine_similarity(x_input, teacher, dim=-1)).clamp(min=0)
    damage_mask = input_error > (input_error.mean(dim=1, keepdim=True)
                                 + input_error.std(dim=1, keepdim=True))

    observed = {}

    def intervene(_module, _inputs, kwargs):
        kw = dict(kwargs)
        q = kw["token_error"].detach()
        if args.mode == "topk":
            k = int(args.topk if args.topk is not None else ct.cfg.topk_tokens)
            if not 1 <= k <= q.shape[1]:
                raise ValueError(f"topk must be in [1, {q.shape[1]}], got {k}")
            index = q.squeeze(-1).topk(k, dim=1).indices
            support = torch.zeros_like(q, dtype=torch.bool)
            support.scatter_(1, index.unsqueeze(-1), True)
            gate = q * support
        elif args.mode == "oracle_soft":
            support = damage_mask.unsqueeze(-1)
            gate = q * support
        elif args.mode == "oracle_unit":
            support = damage_mask.unsqueeze(-1)
            gate = support.to(q.dtype)
        else:
            support = torch.ones_like(q, dtype=torch.bool)
            gate = q
        observed["q"] = q.squeeze(-1).clone()
        observed["gate"] = gate.squeeze(-1).clone()
        observed["support"] = support.squeeze(-1).clone()
        kw["token_error"] = gate
        kw["noise_gate"] = gate
        return _inputs, kw

    hook = ct.denoiser.register_forward_pre_hook(intervene, with_kwargs=True)
    with torch.no_grad():
        model.reset_sequence()
        output = model(**damaged_data)
    hook.remove()

    extras = output["codetrack_extras"]
    x_rec = output.get("X_rec", extras["recovered_pre_denoise"])
    x_final = output.get("X_final", extras["recovered"])
    before_error = (1 - F.cosine_similarity(x_rec, teacher, dim=-1)).clamp(min=0)
    final_error = (1 - F.cosine_similarity(x_final, teacher, dim=-1)).clamp(min=0)
    selected = observed["support"]
    mask = damage_mask
    payload = {
        "probe": "denoiser_write_support_v1",
        "mode": args.mode,
        "topk": args.topk,
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "batch": len(names),
        "checkpoint_tensors": len(checkpoint_state),
        "unexpected_keys": loaded.unexpected_keys,
        "damage_fraction": float(mask.float().mean()),
        "support_fraction": float(selected.float().mean()),
        "support_damage_recall": float((selected & mask).float().sum()
                                        / mask.float().sum().clamp(min=1)),
        "q_mean": float(observed["q"].mean()),
        "q_std": float(observed["q"].std()),
        "gate_mean": float(observed["gate"].mean()),
        "d_input": float(input_error[mask].mean()),
        "d_refiner": float(before_error[mask].mean()),
        "d_final": float(final_error[mask].mean()),
        "gain_refiner": float((input_error[mask] - before_error[mask]).mean()),
        "gain_final": float((input_error[mask] - final_error[mask]).mean()),
        "denoiser_delta_l2_damage": float((x_final - x_rec).norm(dim=-1)[mask].mean()),
        "denoiser_delta_l2_clean": float((x_final - x_rec).norm(dim=-1)[~mask].mean()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
