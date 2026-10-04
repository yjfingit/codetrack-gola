"""Answer the one question ``Loss/rec`` cannot: is the recovery *better*, or just plausible?

``Loss/rec`` measures ``1 - cos(X_final, X_clean)`` and at initialisation it is ~1e-8.  That
number on its own is uninformative: the residual gate is ``sigmoid(-8) ~ 3.4e-4`` by design, so
"recovering accurately" and "changing nothing" are the *same* measurement.  Worse, averaging over
a batch that is ~86% undamaged data hides whatever the damaged minority does.

This script reports the quantities that separate the two hypotheses, restricted to the tokens the
corruption actually touched:

    d_before   = ||X_rec      - X_clean|| / ||X_clean||   (post-refiner,  pre-denoise)
    d_after    = ||X_final    - X_clean|| / ||X_clean||   (post-denoiser)
    gain       = d_before - d_after
    q_auroc    = rank AUROC of the diagnosis q against the measured token error

Run:
    LD_LIBRARY_PATH=/root/autodl-tmp/lab/tools python tools/recovery_report.py

It is deliberately independent of the optimizer: it only builds the model, runs one batch with
forced corruption, and prints the above for each residual-gate initialisation you ask for, so the
``-8 vs -5`` ablation is a single command rather than a training run.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safetensors.torch import load_file  # noqa: E402
from trackit.models.backbone.builder import build_backbone  # noqa: E402
from trackit.models.methods.GOLA.gola import GOLA_DINOv2  # noqa: E402
from codetrack.criteria import CodeTrackCriteria, _rank_auroc  # noqa: E402

WEIGHT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "weights/gola_b224.bin")

BASE_CFG = dict(enabled=True, z_len=64, x_len=256, dim=768, grid=16, mid_dim=128,
                num_checks=64, h_links_per_check=12, h_min_col_degree=3, h_locality_window=5,
                topk_tokens=32, num_neighbours=8, refiner_hidden=256, diffusion_steps=2,
                motion_enabled=False, memory_enabled=False, template_protection=False)


def build(cfg: dict):
    torch.manual_seed(0)
    bb = build_backbone({"type": "DINOv2", "parameters": {"name": "ViT-B/14", "acc": "default"}},
                        torch_jit_trace_compatible=False)
    m = GOLA_DINOv2(bb, (8, 8), (16, 16), 64, 64.0, 0.0, False, codetrack_config=cfg)
    m.load_state_dict(load_file(WEIGHT), strict=False)
    return m


def batch(b: int = 4, seed: int = 7):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return dict(
        z=torch.rand(b, 6, 112, 112, generator=g, device="cuda") * 0.4 + 0.3,
        x=torch.rand(b, 6, 224, 224, generator=g, device="cuda") * 0.4 + 0.3,
        d=torch.rand(b, 6, 112, 112, generator=g, device="cuda") * 0.4 + 0.3,
        z_feat_mask=torch.ones(b, 8, 8, dtype=torch.long, device="cuda"),
        d_feat_mask=torch.ones(b, 8, 8, dtype=torch.long, device="cuda"),
    )


def targets(b: int, device="cuda"):
    return {"num_positive_samples": torch.tensor(float(b), device=device),
            "boxes": torch.tensor([[0.5, 0.48, 0.22, 0.28]] * b, device=device),
            "positive_sample_batch_dim_indices": torch.arange(b, device=device),
            "positive_sample_map_dim_indices": torch.full((b,), 136, device=device)}


def report(gate_init: float) -> None:
    cfg = dict(BASE_CFG, residual_gate_init=float(gate_init))
    m = build(cfg).train().cuda()
    # force token-level corruption so every batch element carries a measurable target
    m.codetrack_cfg.corruption_enabled = True
    m.codetrack_cfg.corruption_token_prob = 1.0
    m.codetrack_cfg.corruption_image_prob = 0.0

    data = batch()
    out = m(**data)
    ex = out["codetrack_extras"]
    clean = ex["clean_tokens"]
    x_rec, x_final = ex["recovered_pre_denoise"], ex["recovered"]
    mask = ex["corruption_mask"].to(torch.bool)

    def rel(x):
        return (x - clean).norm(dim=-1) / clean.norm(dim=-1).clamp(min=1e-6)

    d_before_map, d_after_map = rel(x_rec), rel(x_final)
    d_before = float(d_before_map[mask].mean())
    d_after = float(d_after_map[mask].mean())

    # healthy tokens are the control: they must NOT move (selective recovery)
    healthy = d_after_map[~mask].mean() if bool((~mask).any()) else torch.tensor(float("nan"))
    damaged = d_after_map[mask].mean() if bool(mask.any()) else torch.tensor(float("nan"))

    q, err = ex["q"], ex["error_target"]
    # The threshold must sit between the target on damaged tokens and the target on untouched
    # ones; the measured values are printed as well so a threshold that no longer separates them
    # is visible instead of silently turning AUROC into 0.5.
    thr = float(os.environ.get("CODETRACK_AUROC_THR", "0.25"))
    auroc = _rank_auroc(q.detach().float(), err.detach().float(), thr=thr)

    gate = m.codetrack.denoiser.residual_gate.detach().reshape(())
    w = m.codetrack.denoiser.write_weight
    pos = err.detach() > thr
    err_pos = float(err.detach()[pos].mean()) if bool(pos.any()) else float("nan")
    err_neg = float(err.detach()[~pos].mean()) if bool((~pos).any()) else float("nan")
    print(f"\n--- residual_gate_init = {gate_init:+.1f} ---")
    print(f"  sigmoid(gate)              = {float(torch.sigmoid(gate)):.6f}")
    print(f"  write schedule             = {m.codetrack.denoiser.write_schedule} "
          f"weights={[round(float(v), 4) for v in w]}")
    print(f"  corrupted token fraction   = {float(mask.to(torch.float32).mean()):.4f}")
    print(f"  d_before (corrupted only)  = {d_before:.6f}")
    print(f"  d_after  (corrupted only)  = {d_after:.6f}")
    print(f"  gain = before - after      = {d_before - d_after:+.6f}")
    print(f"  ||delta|| damaged/healthy  = "
          f"{float(damaged / healthy.clamp(min=1e-12)):.3f}x")
    print(f"  error_target  pos/neg      = {err_pos:.4f} / {err_neg:.4f}  (thr={thr})")
    print(f"  q AUROC vs measured error  = "
          f"{'n/a (no split)' if auroc is None else f'{auroc:.4f}'}")
    print(f"  q mean / error mean        = {float(q.mean()):.6f} / {float(err.mean()):.6f}")

    crit = CodeTrackCriteria().cuda()
    co = crit(out, targets(ex["q"].shape[0]))
    metrics = {k: v for k, v in co.metrics.items()
               if k.startswith("Error/") or k.startswith("Loss/")}
    for k in sorted(metrics):
        print(f"  {k:<28} = {metrics[k]:.6f}")


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA is unavailable; this report needs the real model on GPU.", file=sys.stderr)
        return 1
    gates = [float(v) for v in (sys.argv[1:] or ["-8", "-5"])]
    for g in gates:
        report(g)
    print("\nReading: `gain > 0` on corrupted tokens means the branch repairs; `AUROC` well above "
          "0.5 means the diagnosis ranks damage correctly. A gate that stays at sigmoid(-8) while "
          "`gain <= 0` is the signal to try the weaker initialisation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
