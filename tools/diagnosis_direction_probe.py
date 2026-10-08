"""Read-only diagnosis-direction and check-correlation probe on curated LasHeR crops.

The probe never updates parameters. It compares candidate routing scores against the same
feature-error target used by the existing TopK probe, and reports whether the current q sign,
its negation, q logits, or syndrome-derived scores contain useful localization signal.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def rank_auroc(score, label):
    import torch
    s = score.reshape(-1).float()
    y = label.reshape(-1).bool()
    if not bool(y.any()) or bool(y.all()):
        return None
    order = torch.argsort(s)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, s.numel() + 1, device=s.device, dtype=torch.float32)
    n1 = y.sum().float()
    n0 = (~y).sum().float()
    u = ranks[y].sum() - n1 * (n1 + 1) / 2
    return float((u / (n1 * n0)).detach())


def candidate_stats(score, label, k):
    import torch
    selected = score.topk(k, dim=-1).indices
    route = torch.zeros_like(label, dtype=torch.bool)
    route.scatter_(1, selected, True)
    overlap = (route & label).float().sum(-1) / label.float().sum(-1).clamp(min=1)
    return {
        "auroc": rank_auroc(score, label),
        "overlap": float(overlap.mean()),
        "topk_gap": float(score.topk(k, dim=-1).values.mean() - score.mean()),
        "score_std": float(score.std()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--topk", type=int, default=None)
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
    label = err > (err.mean(dim=1, keepdim=True) + err.std(dim=1, keepdim=True))

    captured_diag = []
    dh = ct.diagnosis.register_forward_hook(
        lambda _m, _i, out: captured_diag.append({k: out[k].detach().clone()
                                                   for k in ("q", "q_logits", "s")
                                                   if k in out}))
    with torch.no_grad():
        ct.reset_sequence(); model(**data)
        clean_diag = captured_diag[-1]
        ct.reset_sequence(); model(**damaged_data)
    dh.remove()
    d = captured_diag[-1]
    q, qlog, s = d["q"], d["q_logits"], d["s"]
    Hbar = ct.H.matrix().detach()
    # Candidate syndrome-to-token projections, independent of the refiner and denoiser.
    s_to_token = torch.einsum("mn,bm->bn", Hbar, s)
    candidates = {
        "q": q,
        "neg_q": -q,
        "q_logits": qlog,
        "neg_q_logits": -qlog,
        "s_to_token": s_to_token,
        "neg_s_to_token": -s_to_token,
    }
    k = int(ct.cfg.topk_tokens)
    report = {
        "probe": "diagnosis_direction_v1",
        "checkpoint": str(args.checkpoint),
        "damage_region": list(args.damage_region),
        "sequences": names,
        "tokens": int(err.shape[-1]),
        "topk": k,
        "damage_fraction": float(label.float().mean()),
        "error_mean": float(err.mean()),
        "error_std": float(err.std()),
        "q_mean": float(q.mean()), "q_std": float(q.std()),
        "clean_q_mean": float(clean_diag["q"].mean()),
        "clean_q_std": float(clean_diag["q"].std()),
        "clean_q_topk_gap": float(
            clean_diag["q"].topk(int(ct.cfg.topk_tokens), dim=-1).values.mean()
            - clean_diag["q"].mean()),
        "q_logits_std": float(qlog.std()), "s_std": float(s.std()),
        "candidates": {name: candidate_stats(score, label, k)
                        for name, score in candidates.items()},
    }
    s_center = s - s.mean(dim=-1, keepdim=True)
    corr = torch.corrcoef(s_center.reshape(-1, s_center.shape[-1]).T.float())
    off = ~torch.eye(corr.shape[0], dtype=torch.bool, device=corr.device)
    report["check_corr"] = {
        "mean_offdiag": float(corr[off].mean()),
        "median_offdiag": float(corr[off].median()),
        "abs_mean_offdiag": float(corr[off].abs().mean()),
        "max_abs_offdiag": float(corr[off].abs().max()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
