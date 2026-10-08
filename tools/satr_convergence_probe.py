#!/usr/bin/env python3
"""Fixed-feature convergence test for the active SATR path.

The probe first runs GOLA once on the curated LasHeR train-10 view and caches clean and
region-corrupted fused features.  It then trains only the real ECC path (H, syndrome
diagnosis, and SATR) on those features.  This isolates whether the correction mechanism can
actually learn to locate and repair a transmission error before spending time on tracking
evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _capture_features(model, data):
    captured = []
    h = model.codetrack.register_forward_pre_hook(
        lambda _m, _i, kwargs: captured.append(kwargs["F_L"].detach().clone()),
        with_kwargs=True,
    )
    with __import__("torch").no_grad():
        model.reset_sequence()
        model(**data)
    h.remove()
    if not captured:
        raise RuntimeError("CodeTrack pre-hook did not capture fused features")
    return captured[-1]


def _region_data(data, y0, y1, x0, x1):
    damaged = dict(data)
    damaged["x"] = data["x"].clone()
    # The Siamese tuple has RGB then TIR channels; retain RGB and damage the TIR crop.
    damaged["x"][:, 3:, y0:y1, x0:x1] = 0
    return damaged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--checkpoint", type=Path,
                    default=ROOT / "weights/codetrack_satr_learned_best.safetensors")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--save-checkpoint", type=Path, required=True)
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
    cfg["model"]["codetrack"]["corruption_enabled"] = False
    cfg["model"]["codetrack"]["topk_tokens"] = int(args.topk)
    model = build_GOLA_model(cfg, ModelImplSuggestions()).cuda().float()
    model.load_state_dict(load_file(str(ROOT / "weights/gola_b224.bin")), strict=False)
    model.load_state_dict(load_file(str(args.checkpoint)), strict=False)
    model.eval()

    # Multiple spatial failure modes make the fixed-feature test less likely to overfit one
    # location.  These are the same centre/corner perturbations used by the ten-sequence probes.
    regions = {
        "center": (84, 140, 84, 140),
        "top_left": (28, 84, 28, 84),
        "top_right": (28, 84, 140, 196),
        "bottom_right": (140, 196, 140, 196),
    }
    clean_parts, damage_parts = [], []
    for box in regions.values():
        clean_parts.append(_capture_features(model, data))
        damage_parts.append(_capture_features(model, _region_data(data, *box)))
    clean = torch.cat(clean_parts, dim=0)
    damaged = torch.cat(damage_parts, dim=0)
    ct = model.codetrack
    clean_tok = ct._split(clean)["X_TIR"].detach()
    input_tok = ct._split(damaged)["X_TIR"].detach()
    d_input = (1.0 - F.cosine_similarity(input_tok, clean_tok, dim=-1, eps=1e-6)).clamp(0, 2)
    # A transmission error is a token whose feature changed materially after the image fault.
    # The threshold is fixed from the batch, so it is not an oracle during optimisation.
    mask = d_input > (d_input.mean(dim=-1, keepdim=True)
                      + 0.75 * d_input.std(dim=-1, keepdim=True, unbiased=False))
    # Guarantee both classes for BCE even on an unusually easy crop.
    if not bool(mask.any()) or not bool((~mask).any()):
        flat = d_input.flatten(1)
        mask = torch.zeros_like(d_input, dtype=torch.bool)
        mask.scatter_(1, flat.topk(min(38, flat.shape[1]), dim=1).indices, True)

    # Freeze the GOLA trunk and the tracking head. Only the explicit ECC route is trained.
    for p in model.parameters():
        p.requires_grad_(False)
    train_modules = [ct.H, ct.diagnosis, ct.satr]
    selected = []
    for module in train_modules:
        for p in module.parameters():
            p.requires_grad_(True)
            selected.append(p)
    opt = torch.optim.AdamW(selected, lr=float(args.lr), weight_decay=1e-5)
    ct.train()
    history = []
    best = {"score": -float("inf"), "step": -1, "state": None}

    def forward_loss():
        ct.reset_sequence()
        out = ct(F_L=damaged, corruption_mask=mask,
                 was_corrupted=torch.ones(damaged.shape[0], device=damaged.device),
                 update_state=False)
        q_logits = out["q_logits"].float()
        q_target = mask.float()
        # Use exact damage labels for localisation and clean features for the recovery target.
        l_diag = F.binary_cross_entropy_with_logits(q_logits, q_target)
        x_final = out["X_final"]
        d_after = (1.0 - F.cosine_similarity(x_final, clean_tok, dim=-1, eps=1e-6)).clamp(0, 2)
        l_rec = d_after[mask].mean()
        gain = d_input[mask].mean() - d_after[mask].mean()
        l_gain = F.relu(0.5 * d_input[mask].mean().detach() - gain)
        # Code-space consistency: a valid correction should reduce the measured parity/check
        # residual, but this term is paired with recovery and preservation losses so the decoder
        # cannot minimize syndrome by destroying target identity.
        syn_before = out.get("syndrome_energy_before")
        syn_after = out.get("syndrome_energy_after")
        if syn_before is not None and syn_after is not None:
            l_post = F.relu(syn_after - 0.9 * syn_before).mean()
        else:
            l_post = d_after.mean() * 0.0
        # The code promises an identity path for reliable tokens. Penalise drift there.
        l_pres = (x_final[~mask] - input_tok[~mask]).square().mean()
        loss = l_diag + 2.0 * l_rec + 1.0 * l_gain + 0.1 * l_pres + 0.1 * l_post
        with torch.no_grad():
            q = out["q"]
            top = q.topk(min(args.topk, q.shape[-1]), dim=-1).indices
            picked = torch.zeros_like(mask).scatter(1, top, True)
            overlap = (picked & mask).float().sum() / mask.float().sum().clamp_min(1)
            # A rank-based score avoids a threshold-dependent stopping decision.
            pos = q[mask].mean()
            neg = q[~mask].mean()
            auroc = float("nan")
            try:
                from sklearn.metrics import roc_auc_score
                auroc = float(roc_auc_score(mask.detach().cpu().numpy().reshape(-1),
                                            q.detach().cpu().numpy().reshape(-1)))
            except Exception:
                pass
            row = {
                "loss": float(loss.detach()), "diag": float(l_diag.detach()),
                "rec": float(l_rec.detach()), "gain": float(gain.detach()),
                "pres": float(l_pres.detach()), "d_input": float(d_input[mask].mean()),
                "d_after": float(d_after[mask].mean()), "gain_total": float(gain),
                "post_syndrome": float(l_post.detach()),
                "topk_overlap": float(overlap), "q_pos": float(pos),
                "q_neg": float(neg), "q_auroc": auroc,
            }
        return loss, row

    for step in range(int(args.steps)):
        opt.zero_grad(set_to_none=True)
        loss, row = forward_loss()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite SATR loss at step {step}: {row}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(selected, 1.0)
        if not bool(torch.isfinite(grad_norm)):
            raise RuntimeError(f"non-finite SATR gradient at step {step}")
        opt.step()
        row["step"] = step
        row["grad_norm"] = float(grad_norm)
        if step % 25 == 0 or step == args.steps - 1:
            history.append(row)
            print(json.dumps(row), flush=True)
            # Prefer positive recovery gain, then localisation and preservation.
            score = row["gain_total"] + 0.05 * row["topk_overlap"] - 0.01 * row["pres"]
            if score > best["score"]:
                best = {"score": score, "step": step,
                        "state": {k: v.detach().cpu().clone() for k, v in ct.state_dict().items()}}

    if best["state"] is None:
        raise RuntimeError("no finite SATR checkpoint was produced")
    args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_file(best["state"], str(args.save_checkpoint))
    report = {
        "probe": "satr_convergence_v1", "seed": args.seed, "steps": args.steps,
        "lr": args.lr, "topk": args.topk, "sequences": names,
        "regions": list(regions), "damage_fraction": float(mask.float().mean()),
        "best_step": best["step"], "best_score": best["score"],
        "history": history, "checkpoint": str(args.save_checkpoint),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
