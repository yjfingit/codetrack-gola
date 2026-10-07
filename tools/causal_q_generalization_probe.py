"""Train q on some curated LasHeR sequences and test it on held-out sequences/locations."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _auc(scores, labels):
    import torch
    labels = labels.to(torch.bool).reshape(-1)
    scores = scores.reshape(-1)
    if not bool(labels.any()) or not bool((~labels).any()):
        return None
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores, dtype=torch.float32)
    ranks[order] = torch.arange(1, scores.numel() + 1, device=scores.device, dtype=torch.float32)
    p = labels.sum().float(); n = (~labels).sum().float()
    return float(((ranks[labels].sum() - p * (p + 1) / 2) / (p * n)).detach())


def _features(model, data, top: int, left: int):
    import torch
    captured = []
    hook = model.codetrack.register_forward_pre_hook(
        lambda _m, _i, kwargs: captured.append(kwargs["F_L"].detach()), with_kwargs=True)
    bad = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in data.items()}
    bad["x"][:, 3:, top:top + 56, left:left + 56] = 0.0
    with torch.no_grad():
        model.reset_sequence(); model(**data); clean_f = captured[-1]
        model.reset_sequence(); model(**bad); bad_f = captured[-1]
    hook.remove()
    return model.codetrack._split(clean_f), model.codetrack._split(bad_f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--min-ratio", type=float, default=0.25)
    parser.add_argument("--save-diagnosis", type=Path, default=None)
    parser.add_argument("--save-bundle", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    from trackit.models import ModelImplSuggestions
    from trackit.models.methods.GOLA.builder import build_GOLA_model
    from tools.preflight_acceptance import load_stage_config
    from tools.verify_codetrack_initialization import prepare_view, real_batch

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dataset = Path("/home/yangjuanfeng/lab/dataset/LasHeR")
    view = ROOT / "_probe/codetrack_flow_validation/LasHeR_causal_split"
    names = prepare_view(dataset, view)
    train_names, val_names = names[:7], names[7:]
    cfg = load_stage_config(str(ROOT / "config/GOLA/codetrack_s1/config.yaml"))
    cfg["model"]["codetrack"].update({
        "corruption_enabled": False,
        "motion_enabled": False,
        "memory_enabled": False,
        "h_layout": "grid",
        "h_free_edge_frac": 0.0,
    })
    model = build_GOLA_model(cfg, ModelImplSuggestions()).cuda().float().eval()
    model.load_state_dict(load_file(str(ROOT / "weights/gola_b224.bin")), strict=False)
    ct = model.codetrack
    train_data = real_batch(dataset, train_names)
    val_data = real_batch(dataset, val_names)
    # Train on three spatial locations and hold out a fourth.  A single fixed block can be
    # memorised by the learned symbol encoder even when the graph looks structured.
    train_positions = [(28, 28), (28, 140), (140, 28)]
    val_position = (140, 140)
    train_tir, train_rgb, train_ctx, train_targets = [], [], [], []
    with torch.no_grad():
        hook = ct.register_forward_pre_hook(lambda _m, _i, kwargs: train_ctx.append(kwargs["F_L"].detach()), with_kwargs=True)
        model.reset_sequence(); model(**train_data)
        hook.remove()
    clean_ctx_parts = ct._split(train_ctx[-1])
    clean_context = ct.template_pool(torch.cat([clean_ctx_parts["Z_RGB"], clean_ctx_parts["Z_TIR"], clean_ctx_parts["Z_on"], clean_ctx_parts["D_TIR"]], 1).mean(1)).detach()
    for top, left in train_positions:
        clean_p, bad_p = _features(model, train_data, top, left)
        clean_t, bad_t = clean_p["X_TIR"].detach(), bad_p["X_TIR"].detach()
        clean_head, bad_head = model.head(clean_t), model.head(bad_t)
        train_tir.append(bad_t); train_rgb.append(bad_p["X_RGB"].detach())
        train_targets.append(model._causal_token_impact_target(
            bad_t, clean_t, bad_head, clean_head, min_ratio=args.min_ratio))
    train_bad = torch.cat(train_tir, 0)
    train_rgb = torch.cat(train_rgb, 0)
    train_target = torch.cat(train_targets, 0)
    train_ctx_batch = clean_context.repeat(len(train_positions), 1)

    val_clean_p, val_bad_p = _features(model, val_data, *val_position)
    val_clean, val_bad = val_clean_p["X_TIR"].detach(), val_bad_p["X_TIR"].detach()
    with torch.no_grad():
        val_clean_head, val_bad_head = model.head(val_clean), model.head(val_bad)
    val_target = model._causal_token_impact_target(
        val_bad, val_clean, val_bad_head, val_clean_head, min_ratio=args.min_ratio)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in ct.diagnosis.parameters():
        p.requires_grad_(True)
    H = ct.H.matrix().detach()
    opt = torch.optim.AdamW(ct.diagnosis.parameters(), lr=5e-4, weight_decay=1e-5)
    history = []
    best_val = -float("inf")
    for step in range(args.steps):
        out = ct.diagnosis(train_bad, train_rgb, H, template_context=train_ctx_batch)
        loss = F.binary_cross_entropy_with_logits(out["q_logits"].float(), train_target.float())
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(ct.diagnosis.parameters(), 1.0); opt.step()
        if step % 50 == 0 or step == args.steps - 1:
            with torch.no_grad():
                tq = ct.diagnosis(train_bad, train_rgb, H, template_context=train_ctx_batch)["q"]
                # Held-out RGB/template context is not mixed into training.
                val_contexts = []
                h2 = ct.register_forward_pre_hook(lambda _m, _i, kwargs: val_contexts.append(kwargs["F_L"].detach()), with_kwargs=True)
                model.reset_sequence(); model(**val_data)
                h2.remove(); vp = ct._split(val_contexts[-1])
                vctx = ct.template_pool(torch.cat([vp["Z_RGB"], vp["Z_TIR"], vp["Z_on"], vp["D_TIR"]], 1).mean(1)).detach()
                vq = ct.diagnosis(val_bad, vp["X_RGB"].detach(), H, template_context=vctx)["q"]
                history.append({"step": step, "loss": float(loss.detach()),
                                "train_auc": _auc(tq, train_target > 0),
                                "val_auc": _auc(vq, val_target > 0),
                                "val_target_fraction": float((val_target > 0).float().mean())})
                if history[-1]["val_auc"] is not None and history[-1]["val_auc"] > best_val:
                    best_val = history[-1]["val_auc"]
                    if args.save_diagnosis is not None:
                        from safetensors.torch import save_file
                        args.save_diagnosis.parent.mkdir(parents=True, exist_ok=True)
                        save_file({k: v.detach().cpu() for k, v in ct.diagnosis.state_dict().items()},
                                  str(args.save_diagnosis))
                    if args.save_bundle is not None:
                        from safetensors.torch import save_file
                        bundle = {f"diagnosis.{k}": v.detach().cpu()
                                  for k, v in ct.diagnosis.state_dict().items()}
                        bundle.update({f"template_pool.{k}": v.detach().cpu()
                                       for k, v in ct.template_pool.state_dict().items()})
                        bundle["H.H"] = ct.H.H.detach().cpu()
                        args.save_bundle.parent.mkdir(parents=True, exist_ok=True)
                        save_file(bundle, str(args.save_bundle))
    result = {"probe": "causal_q_generalization_v1", "train": train_names,
              "validation": val_names, "history": history}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
