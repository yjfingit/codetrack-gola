"""Fixed-batch oracle gate probe for CodeTrack recovery.

Compares learned centered-q routing, exact feature-error gate, and a soft oracle gate. No
parameters are updated. The purpose is to distinguish a diagnosis calibration problem from a
recovery-capacity problem.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--topk", type=int, default=None)
    ap.add_argument("--disable-diffusion", action="store_true", help="legacy no-op")
    ap.add_argument("--damage-region", type=int, nargs=4, metavar=("Y0", "Y1", "X0", "X1"),
                    default=(84, 140, 84, 140))
    args = ap.parse_args()
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    from trackit.models import ModelImplSuggestions
    from trackit.models.methods.GOLA.builder import build_GOLA_model
    from tools.preflight_acceptance import load_stage_config
    from tools.verify_codetrack_initialization import real_batch

    view = ROOT / "_probe/codetrack_flow_validation/LasHeR_curated10"
    names = (view / "trainingsetList.txt").read_text().splitlines()
    data = real_batch(view, names)
    cfg = load_stage_config(str(ROOT / "config/GOLA/codetrack_s1/config.yaml"))
    if args.topk is not None:
        cfg["model"]["codetrack"]["topk_tokens"] = int(args.topk)
    cfg["model"]["codetrack"]["corruption_enabled"] = False
    model = build_GOLA_model(cfg, ModelImplSuggestions()).cuda().eval()
    model.load_state_dict(load_file(str(ROOT / "weights/gola_b224.bin")), strict=False)
    if args.checkpoint is not None:
        model.load_state_dict(load_file(str(args.checkpoint)), strict=False)
    ct = model.codetrack

    captured = []
    hook = ct.register_forward_pre_hook(
        lambda _m, _i, kwargs: captured.append(kwargs["F_L"].detach().clone()),
        with_kwargs=True)
    with torch.no_grad():
        model.reset_sequence(); model(**data); clean = captured[-1]
        damaged_data = dict(data)
        damaged_data["x"] = data["x"].clone()
        y0, y1, x0, x1 = args.damage_region
        h, w = damaged_data["x"].shape[-2:]
        if not (0 <= y0 < y1 <= h and 0 <= x0 < x1 <= w):
            raise ValueError(f"invalid damage region {args.damage_region} for {h}x{w} input")
        damaged_data["x"][:, 3:, y0:y1, x0:x1] = 0
        model.reset_sequence(); model(**damaged_data); damaged = captured[-1]
    hook.remove()
    teacher = ct._split(clean)["X_TIR"].detach()
    xin = ct._split(damaged)["X_TIR"].detach()
    err = (1 - F.cosine_similarity(xin, teacher, dim=-1)).clamp(min=0)
    # Continuous oracle gate: preserve relative severity but cap to [0, 1].
    soft = ((err - err.mean(dim=-1, keepdim=True)) /
            (err.std(dim=-1, keepdim=True) + 1e-6)).sigmoid()
    hard = (err > err.mean(dim=-1, keepdim=True) + err.std(dim=-1, keepdim=True)).float()

    outputs = {}
    for name, gate in (("learned", None), ("oracle_soft", soft), ("oracle_hard", hard)):
        def intervene(_m, inputs, kwargs, gate=gate):
            if gate is None:
                return inputs, kwargs
            kw = dict(kwargs)
            kw["q"] = gate
            return inputs, kw
        h = ct.satr.register_forward_pre_hook(intervene, with_kwargs=True)
        with torch.no_grad():
            ct.reset_sequence(); out = model(**damaged_data)
        h.remove()
        final = out.get("X_final")
        if final is None and isinstance(out.get("recovered"), torch.Tensor):
            final = out["recovered"]
        if final is None and isinstance(out.get("codetrack_extras"), dict):
            final = out["codetrack_extras"].get("recovered")
        if final is None:
            raise KeyError(f"X_final missing; output keys={list(out.keys())}")
        d_final = (1 - F.cosine_similarity(final, teacher, dim=-1)).clamp(min=0)
        outputs[name] = {
            "d_input": float(err.mean()),
            "d_final": float(d_final.mean()),
            "gain_total": float((err - d_final).mean()),
            "d_input_damage": float(err[hard.bool()].mean()),
            "d_final_damage": float(d_final[hard.bool()].mean()),
            "gain_damage": float((err[hard.bool()] - d_final[hard.bool()]).mean()),
        }
    payload = {"probe": "oracle_gate_v1", "checkpoint": str(args.checkpoint),
               "topk": int(ct.cfg.topk_tokens),
               "satr_rounds": int(ct.cfg.satr_rounds),
               "damage_region": list(args.damage_region),
               "damage_fraction": float(hard.mean()), "results": outputs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
