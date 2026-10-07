"""Stage-1 reliability diagnosis probe.

The GOLA trunk is frozen.  Only the ECC diagnosis/BP parameters are trained on paired
clean/corrupted feature streams.  Corruption regions are split spatially: three regions are
used for fitting and the fourth is held out, so a low loss cannot be explained by memorising one
fixed token block.  This is a diagnosis gate, not a tracking benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _rank(score, label):
    import torch
    s = score.reshape(-1).float()
    y = label.reshape(-1).bool()
    if not bool(y.any()) or not bool((~y).any()):
        return float("nan")
    order = torch.argsort(s)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, s.numel() + 1, device=s.device, dtype=torch.float32)
    n1, n0 = y.sum().float(), (~y).sum().float()
    return float(((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)).detach())


def _tokens(ct, fused):
    import torch
    toks = ct._split(fused)
    x_t, x_aux = toks["X_TIR"], toks["X_RGB"]
    template = torch.cat([toks["Z_RGB"], toks["Z_TIR"], toks["Z_on"], toks["D_TIR"]], dim=1)
    template_ctx = ct.template_pool(template.mean(dim=1))
    return x_t.detach(), x_aux.detach(), template_ctx.detach()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--save-checkpoint", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file, save_file
    from trackit.models import ModelImplSuggestions
    from trackit.models.methods.GOLA.builder import build_GOLA_model
    from tools.preflight_acceptance import load_stage_config
    from tools.verify_codetrack_initialization import real_batch

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    view = ROOT / "_probe/codetrack_flow_validation/LasHeR_curated10"
    names = (view / "trainingsetList.txt").read_text().splitlines()
    data = real_batch(view, names)
    cfg = load_stage_config(str(ROOT / "config/GOLA/codetrack_s1/config.yaml"))
    ct_cfg = cfg["model"]["codetrack"]
    ct_cfg["corruption_enabled"] = False
    ct_cfg["freeze_vote_bias"] = True
    ct_cfg["center_bp_logits"] = True
    model = build_GOLA_model(cfg, ModelImplSuggestions()).cuda().float().eval()
    model.load_state_dict(load_file(str(ROOT / "weights/gola_b224.bin")), strict=False)
    ct = model.codetrack

    captured = []
    hook = ct.register_forward_pre_hook(
        lambda _m, _i, kwargs: captured.append(kwargs["F_L"].detach().clone()),
        with_kwargs=True)
    with torch.no_grad():
        model.reset_sequence()
        model(**data)
        clean = captured[-1]
        regions = {
            "center": (84, 140, 84, 140),
            "top_left": (28, 84, 28, 84),
            "top_right": (28, 84, 140, 196),
            "bottom_right": (140, 196, 140, 196),
        }
        corrupted = {}
        corruption_kinds = ("tir_zero", "rgb_zero", "tir_noise", "tir_blur_like")
        for name, (y0, y1, x0, x1) in regions.items():
            for kind in corruption_kinds:
                damaged = dict(data)
                damaged["x"] = data["x"].clone()
                if kind == "tir_zero":
                    damaged["x"][:, 3:, y0:y1, x0:x1] = 0
                elif kind == "rgb_zero":
                    damaged["x"][:, :3, y0:y1, x0:x1] = 0
                elif kind == "tir_noise":
                    damaged["x"][:, 3:, y0:y1, x0:x1] += \
                        torch.randn_like(damaged["x"][:, 3:, y0:y1, x0:x1]) * 0.35
                elif kind == "tir_blur_like":
                    patch = damaged["x"][:, 3:, y0:y1, x0:x1]
                    damaged["x"][:, 3:, y0:y1, x0:x1] = patch.mean(dim=(-2, -1), keepdim=True)
                model.reset_sequence()
                model(**damaged)
                corrupted[f"{name}:{kind}"] = captured[-1]
    hook.remove()

    # Labels describe the injected image region, not a post-transform feature threshold.
    # This tests whether BP can localise a known transmission error consistently.
    n = clean.shape[0]
    grid = ct.cfg.grid
    samples = []
    for sample_name, fused in corrupted.items():
        region_name, _kind = sample_name.split(":", 1)
        mask = torch.zeros(n, grid * grid, device=clean.device, dtype=torch.bool)
        y0, y1, x0, x1 = regions[region_name]
        gy0, gy1 = y0 // 14, (y1 + 13) // 14
        gx0, gx1 = x0 // 14, (x1 + 13) // 14
        for yy in range(gy0, min(gy1, grid)):
            for xx in range(gx0, min(gx1, grid)):
                mask[:, yy * grid + xx] = True
        samples.append((region_name, sample_name, *_tokens(ct, fused), mask))
    clean_parts = _tokens(ct, clean)
    clean_mask = torch.zeros(n, grid * grid, device=clean.device, dtype=torch.bool)

    train_names = {"center", "top_left", "top_right"}
    train = [s for s in samples if s[0] in train_names]
    valid = [s for s in samples if s[0] not in train_names]

    for p in model.parameters():
        p.requires_grad_(False)
    selected = []
    for p in ct.diagnosis.parameters():
        p.requires_grad_(True)
        selected.append(p)
    optimizer = torch.optim.AdamW(selected, lr=args.lr, weight_decay=1e-5)
    H = ct.H.matrix().detach()
    ct.diagnosis.train()

    def diagnose(batch):
        x_t = torch.cat([x[2] for x in batch], dim=0)
        x_a = torch.cat([x[3] for x in batch], dim=0)
        t = torch.cat([x[4] for x in batch], dim=0)
        y = torch.cat([x[5] for x in batch], dim=0)
        return ct.diagnosis(x_t, x_a, H, template_context=t), y

    history = []
    best = None
    for step in range(args.steps):
        out, y = diagnose(train)
        # Healthy frames are included in every update.  The weak prior anchor prevents the
        # token posterior from becoming an all-zero solution while the vote bias is frozen.
        clean_out = ct.diagnosis(clean_parts[0], clean_parts[1], H,
                                 template_context=clean_parts[2])
        logits = out["q_logits"].float()
        l_bad = F.binary_cross_entropy_with_logits(logits, y.float())
        l_clean = F.binary_cross_entropy_with_logits(clean_out["q_logits"].float(), clean_mask.float())
        # Pairwise margin uses only the injected support and keeps ranking useful even when the
        # positive fraction changes with region size.
        pos = logits[y].mean()
        neg = logits[~y].mean()
        l_rank = F.relu(0.5 - pos + neg)
        loss = l_bad + 0.5 * l_clean + 0.25 * l_rank
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(selected, 1.0)
        optimizer.step()

        if step % 25 == 0 or step == args.steps - 1:
            with torch.no_grad():
                valid_rows = []
                for region_name, sample_name, x_t, x_a, t, yv in valid:
                    vo = ct.diagnosis(x_t, x_a, H, template_context=t)
                    valid_rows.append({
                        "region": sample_name,
                        "auroc": _rank(vo["q"], yv),
                        "q_pos": float(vo["q"][yv].mean()),
                        "q_neg": float(vo["q"][~yv].mean()),
                    })
                clean_q = clean_out["q"]
                row = {
                    "step": step, "loss": float(loss.detach()),
                    "bad": float(l_bad.detach()), "clean": float(l_clean.detach()),
                    "rank": float(l_rank.detach()), "grad": float(grad),
                    "train_auroc": _rank(out["q"], y),
                    "valid": valid_rows,
                    "clean_q_mean": float(clean_q.mean()),
                    "clean_q_p95": float(torch.quantile(clean_q, 0.95)),
                }
                history.append(row)
                # Select by held-out AUROC with a hard clean false-positive penalty.
                val_auc = sum(r["auroc"] for r in valid_rows) / len(valid_rows)
                score = val_auc - max(0.0, float(torch.quantile(clean_q, 0.95)) - 0.5)
                if best is None or score > best["score"]:
                    best = {"score": score, "step": step,
                            "state": {k: v.detach().cpu().clone()
                                      for k, v in ct.diagnosis.state_dict().items()}}
                print(json.dumps(row), flush=True)

    if best is None:
        raise RuntimeError("no diagnosis checkpoint produced")
    args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_file(best["state"], str(args.save_checkpoint))
    # Evaluate corruption families that were not used to fit the detector.  This is the
    # anti-injector gate: a detector that only recognises zeroed TIR blocks is not useful for
    # natural RGB-T degradation.
    ct.diagnosis.load_state_dict(best["state"], strict=True)
    ct.diagnosis.eval()

    def capture_variant(kind):
        variant = dict(data)
        variant["x"] = data["x"].clone()
        y0, y1, x0, x1 = regions["bottom_right"]
        if kind == "rgb_zero":
            variant["x"][:, :3, y0:y1, x0:x1] = 0
        elif kind == "tir_noise":
            noise = torch.randn_like(variant["x"][:, 3:, y0:y1, x0:x1]) * 0.35
            variant["x"][:, 3:, y0:y1, x0:x1] += noise
        elif kind == "tir_blur_like":
            patch = variant["x"][:, 3:, y0:y1, x0:x1]
            variant["x"][:, 3:, y0:y1, x0:x1] = patch.mean(dim=(-2, -1), keepdim=True)
        else:
            raise ValueError(kind)
        captured_variant = []
        h = ct.register_forward_pre_hook(
            lambda _m, _i, kwargs: captured_variant.append(kwargs["F_L"].detach().clone()),
            with_kwargs=True)
        with torch.no_grad():
            model.reset_sequence()
            model(**variant)
        h.remove()
        return captured_variant[-1]

    y_eval = valid[0][5]
    unseen = {}
    with torch.no_grad():
        for kind in ("rgb_zero", "tir_noise", "tir_blur_like"):
            fused = capture_variant(kind)
            x_t, x_a, t = _tokens(ct, fused)
            vo = ct.diagnosis(x_t, x_a, H, template_context=t)
            qv = vo["q"]
            unseen[kind] = {
                "auroc": _rank(qv, y_eval),
                "q_pos": float(qv[y_eval].mean()),
                "q_neg": float(qv[~y_eval].mean()),
                "clean_q_p95": float(torch.quantile(clean_out["q"], 0.95)),
            }
    report = {
        "probe": "diagnosis_stage1_v1", "seed": args.seed,
        "steps": args.steps, "lr": args.lr, "sequences": names,
        "train_regions": sorted(train_names), "valid_regions": [x[0] for x in valid],
        "best_step": best["step"], "best_score": best["score"],
        "unseen_corruption": unseen,
        "history": history, "checkpoint": str(args.save_checkpoint),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
