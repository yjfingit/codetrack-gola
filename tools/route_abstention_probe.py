"""Causal probe for ECC confidence abstention on a fixed real batch.

The production route currently always selects K tokens.  This read-only probe keeps the
learned diagnosis fixed and masks low relative-q tokens at the SATR boundary.  SATR may
still receive K indices, but zero-gated entries take an exact identity write,
which isolates whether abstention itself reduces collateral damage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--disable-diffusion", action="store_true")
    ap.add_argument("--gpu", type=int, default=1)
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
    cfg["model"]["codetrack"]["topk_tokens"] = int(args.topk)
    cfg["model"]["codetrack"]["corruption_enabled"] = False
    if args.disable_diffusion:
        cfg["model"]["codetrack"]["diffusion_enabled"] = False
    model = build_GOLA_model(cfg, ModelImplSuggestions()).cuda().eval()
    model.load_state_dict(load_file(str(ROOT / "weights/gola_b224.bin")), strict=False)
    model.load_state_dict(load_file(str(args.checkpoint)), strict=False)
    ct = model.codetrack

    regions = {
        "center": (84, 140, 84, 140),
        "top_left": (28, 84, 28, 84),
        "top_right": (28, 84, 140, 196),
        "bottom_right": (140, 196, 140, 196),
    }
    thresholds = (-1.0, 0.0, 0.5, 1.0, 1.5)
    rows = []
    for region_name, (y0, y1, x0, x1) in regions.items():
        damaged = dict(data)
        damaged["x"] = data["x"].clone()
        damaged["x"][:, 3:, y0:y1, x0:x1] = 0
        captured = []
        hook = ct.register_forward_pre_hook(
            lambda _m, _i, kwargs: captured.append(kwargs["F_L"].detach().clone()),
            with_kwargs=True)
        with torch.no_grad():
            model.reset_sequence(); model(**data); clean = captured[-1]
            model.reset_sequence(); model(**damaged); corrupted = captured[-1]
        hook.remove()
        teacher = ct._split(clean)["X_TIR"].detach()
        xin = ct._split(corrupted)["X_TIR"].detach()
        d_in = (1 - F.cosine_similarity(xin, teacher, dim=-1)).clamp(min=0)
        hard = d_in > d_in.mean(dim=-1, keepdim=True) + d_in.std(dim=-1, keepdim=True)

        for threshold in (None,) + thresholds:
            def intervene(_m, inputs, kwargs, threshold=threshold):
                kw = dict(kwargs)
                q = kw["q"]
                if threshold is not None:
                    mu = q.mean(dim=-1, keepdim=True)
                    sd = q.std(dim=-1, keepdim=True, unbiased=False).clamp(min=1e-6)
                    active = ((q - mu) / sd) >= threshold
                    kw["q"] = torch.where(active, q, torch.zeros_like(q))
                return inputs, kw

            rh = ct.satr.register_forward_pre_hook(intervene, with_kwargs=True)
            with torch.no_grad():
                ct.reset_sequence()
                out = model(**damaged)
            rh.remove()
            final = out["codetrack_extras"]["recovered"]
            d_final = (1 - F.cosine_similarity(final, teacher, dim=-1)).clamp(min=0)
            rows.append({
                "region": region_name,
                "threshold": "learned" if threshold is None else float(threshold),
                "active_fraction": float("nan") if threshold is None else float(
                    (((out["codetrack_extras"]["q"] - out["codetrack_extras"]["q"].mean(-1, keepdim=True)) /
                      out["codetrack_extras"]["q"].std(-1, keepdim=True, unbiased=False).clamp(min=1e-6)) >= threshold).float().mean()),
                "gain_total": float((d_in - d_final).mean()),
                "gain_damage": float((d_in[hard] - d_final[hard]).mean()),
                "d_input_damage": float(d_in[hard].mean()),
                "d_final_damage": float(d_final[hard].mean()),
            })

    payload = {"probe": "route_abstention_v1", "checkpoint": str(args.checkpoint),
               "topk": int(args.topk), "thresholds": list(thresholds),
               "regions": list(regions), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
