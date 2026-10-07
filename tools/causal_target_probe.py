"""Run the opt-in causal q target on one real LasHeR batch.

This is a wiring/health probe, not a training run.  It forces a token corruption so the
counterfactual target is populated, then checks target/q shapes, finiteness, and gradients.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "config/GOLA/codetrack_s1/config.yaml")
    args = parser.parse_args()

    import torch
    from safetensors.torch import load_file
    from trackit.models import ModelImplSuggestions
    from trackit.models.methods.GOLA.builder import build_GOLA_model
    from tools.preflight_acceptance import load_stage_config
    from tools.verify_codetrack_initialization import prepare_view, real_batch

    dataset = Path("/home/yangjuanfeng/lab/dataset/LasHeR")
    view = ROOT / "_probe/codetrack_flow_validation/LasHeR_curated10"
    names = prepare_view(dataset, view)
    data = real_batch(dataset, names)
    cfg = load_stage_config(str(args.config))
    ct_cfg = cfg["model"]["codetrack"]
    ct_cfg.update({
        "causal_q_target_enabled": True,
        "causal_q_target_tokens": 256,
        "causal_q_target_min_ratio": 0.25,
        "h_layout": "grid",
        "h_free_edge_frac": 0.0,
        "corruption_enabled": True,
        "corruption_token_prob": 1.0,
        "corruption_image_prob": 0.0,
        "corruption_token_ratio": 0.10,
        "motion_enabled": False,
        "memory_enabled": False,
    })
    model = build_GOLA_model(cfg, ModelImplSuggestions()).cuda().float()
    model.load_state_dict(load_file(str(ROOT / "weights/gola_b224.bin")), strict=False)
    model.train(True)
    model.reset_sequence()
    torch.manual_seed(17)
    out = model(**data)
    extras = out["codetrack_extras"]
    target = extras["causal_q_target"]
    q = extras["q"]
    finite = bool(torch.isfinite(target).all() and torch.isfinite(q).all())
    loss = target.mean() * 0.0 + q.mean()
    loss.backward()
    grad_norm = 0.0
    for p in model.codetrack.diagnosis.parameters():
        if p.grad is not None:
            grad_norm += float(p.grad.detach().float().norm())
    result = {
        "probe": "causal_target_v1",
        "sequences": names,
        "target_shape": list(target.shape),
        "q_shape": list(q.shape),
        "target_mean": float(target.mean().detach()),
        "target_max": float(target.max().detach()),
        "target_nonzero_fraction": float((target > 1e-6).float().mean().detach()),
        "q_mean": float(q.mean().detach()),
        "finite": finite,
        "diagnosis_grad_norm": grad_norm,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
