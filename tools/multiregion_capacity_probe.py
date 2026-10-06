"""Train a recovery refiner on one fixed real batch repeated across four damage regions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


REGIONS = {
    "center": (84, 140, 84, 140),
    "top_left": (28, 84, 28, 84),
    "top_right": (28, 84, 140, 196),
    "bottom_right": (140, 196, 140, 196),
}


def main() -> None:
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--topk", type=int, default=64)
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
    cfg["model"]["codetrack"]["corruption_enabled"] = False
    cfg["model"]["codetrack"]["diffusion_enabled"] = False
    cfg["model"]["codetrack"]["topk_tokens"] = args.topk
    model = build_GOLA_model(cfg, ModelImplSuggestions()).cuda().eval()
    state = load_file(str(args.checkpoint))
    state = {key: value for key, value in state.items()
             if not key.startswith("codetrack.denoiser.")}
    model.load_state_dict(state, strict=False)
    ct = model.codetrack

    captured = []
    hook = ct.register_forward_pre_hook(
        lambda _m, _i, kwargs: captured.append(kwargs["F_L"].detach().clone()),
        with_kwargs=True)
    with torch.no_grad():
        model.reset_sequence()
        model(**data)
        clean = captured[-1]
        damaged_features = []
        region_ids = []
        for region_name, (y0, y1, x0, x1) in REGIONS.items():
            damaged_data = dict(data)
            damaged_data["x"] = data["x"].clone()
            damaged_data["x"][:, 3:, y0:y1, x0:x1] = 0
            model.reset_sequence()
            model(**damaged_data)
            damaged_features.append(captured[-1])
            region_ids.extend([region_name] * len(names))
    hook.remove()

    teacher_one = ct._split(clean)["X_TIR"].detach()
    damaged = torch.cat(damaged_features, dim=0)
    teacher = teacher_one.repeat(len(REGIONS), 1, 1)
    x_input = ct._split(damaged)["X_TIR"].detach()
    input_error = (1 - F.cosine_similarity(x_input, teacher, dim=-1)).clamp(min=0)
    mask = input_error > input_error.mean(dim=1, keepdim=True) + input_error.std(dim=1, keepdim=True)

    ct.train()
    params = [p for name, p in ct.named_parameters()
              if name.startswith("refiner.") and p.requires_grad]
    if not params:
        raise RuntimeError("no refiner parameters selected")
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0)

    def run():
        ct.reset_sequence()
        return ct(F_L=damaged)

    def error(output):
        return (1 - F.cosine_similarity(output["X_final"], teacher, dim=-1))[mask].mean()

    def region_metrics(output):
        final_error = (1 - F.cosine_similarity(output["X_final"], teacher, dim=-1)).detach()
        result = {}
        n = len(names)
        for index, region_name in enumerate(REGIONS):
            sl = slice(index * n, (index + 1) * n)
            result[region_name] = {
                "d_input": float(input_error[sl][mask[sl]].mean()),
                "d_final": float(final_error[sl][mask[sl]].mean()),
                "gain": float((input_error[sl][mask[sl]] - final_error[sl][mask[sl]]).mean()),
            }
        return result

    history = []
    with torch.no_grad():
        initial = run()
    history.append({"step": 0, "loss": float(error(initial)), "regions": region_metrics(initial)})
    for step in range(1, args.steps + 1):
        output = run()
        loss = error(output)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step == args.steps:
            with torch.no_grad():
                evaluated = run()
            row = {"step": step, "loss": float(loss), "regions": region_metrics(evaluated)}
            history.append(row)
            print(json.dumps(row), flush=True)

    payload = {
        "probe": "multiregion_capacity_v1",
        "checkpoint": str(args.checkpoint),
        "steps": args.steps,
        "lr": args.lr,
        "topk": args.topk,
        "regions": list(REGIONS),
        "batch_per_region": len(names),
        "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
