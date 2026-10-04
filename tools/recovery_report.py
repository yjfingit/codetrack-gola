"""Residual-gate ablation: does the recovery branch actually reduce the input's error?

The question this answers
------------------------
``Loss/rec`` is ``1 - cos(X_final, X_clean)``.  At a residual gate of ``sigmoid(-8) ~ 3.4e-4``
that number is ~1e-8 for a branch that changes nothing, so it cannot distinguish "recovered
accurately" from "did nothing".  The quantity that can is

    d_input = 1 - cos(X_t,     X_clean)      what CodeTrack was handed (student input tokens)
    d_before= 1 - cos(X_rec,   X_clean)      after H-routed evidence routing
    d_after = 1 - cos(X_final, X_clean)      after the noise-modulated correction

all restricted to the tokens the injector actually damaged, and all measured with the same
angular metric the training criterion uses.  **The paper-level claim is ``d_after < d_input``**;
comparing only ``X_rec`` against clean proves "the denoiser improves on the refiner", not
"CodeTrack improves on its input".

Why this is a *training* run and not a forward report
----------------------------------------------------
A single forward pass cannot answer it: at initialisation the gate is deliberately closed, so
both arms look identical and both gains are ~0.  The two arms below are therefore each trained
for the same number of optimizer updates with the same seed and the same pre-generated batches,
so the only difference is ``residual_gate_init``.

Usage
-----
    LD_LIBRARY_PATH=/root/autodl-tmp/lab/tools python tools/recovery_report.py            # -8 vs -5, 600 updates
    ... python tools/recovery_report.py --gate-init -8 --updates 600
    ... python tools/recovery_report.py --gates -8,-5,-4 --updates 300 --lr 1e-4

Only the ``codetrack.*`` parameters are optimised, matching stage S1.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safetensors.torch import load_file  # noqa: E402
from trackit.models.backbone.builder import build_backbone  # noqa: E402
from trackit.models.methods.GOLA.gola import GOLA_DINOv2  # noqa: E402
from codetrack.criteria import CodeTrackCriteria  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHT = os.path.join(ROOT, "weights/gola_b224.bin")

BASE_CFG = dict(enabled=True, z_len=64, x_len=256, dim=768, grid=16, mid_dim=128,
                num_checks=64, h_links_per_check=12, h_min_col_degree=3, h_locality_window=5,
                topk_tokens=32, num_neighbours=8, refiner_hidden=256, diffusion_steps=2,
                motion_enabled=False, memory_enabled=False, template_protection=False)

# Metrics reported per milestone, in print order.
KEYS = ["d_input", "d_before", "d_after", "gain_total", "gain_refiner",
        "q_auroc_mask", "q_auprc_mask", "q_spearman",
        "Loss/pres", "Loss/rec", "Loss/cls_clean", "gate"]


def build(gate_init: float):
    cfg = dict(BASE_CFG, residual_gate_init=float(gate_init))
    torch.manual_seed(0)
    bb = build_backbone({"type": "DINOv2", "parameters": {"name": "ViT-B/14", "acc": "default"}},
                        torch_jit_trace_compatible=False)
    m = GOLA_DINOv2(bb, (8, 8), (16, 16), 64, 64.0, 0.0, False, codetrack_config=cfg)
    m.load_state_dict(load_file(WEIGHT), strict=False)
    return m


def batch_pool(n_batches: int, b: int = 8, seed: int = 7, device: str = "cuda"):
    """A fixed pool of synthetic batches.

    Reused every step so the two arms see *identical* data: with fresh random draws the
    between-arm difference would be indistinguishable from draw-to-draw noise.
    """
    pool = []
    for i in range(n_batches):
        g = torch.Generator(device=device).manual_seed(seed + i)

        def r(*shape):
            return torch.rand(*shape, generator=g, device=device) * 0.4 + 0.3

        pool.append((dict(z=r(b, 6, 112, 112), x=r(b, 6, 224, 224), d=r(b, 6, 112, 112),
                          z_feat_mask=torch.ones(b, 8, 8, dtype=torch.long, device=device),
                          d_feat_mask=torch.ones(b, 8, 8, dtype=torch.long, device=device)),
                     {"num_positive_samples": torch.tensor(float(b), device=device),
                      "boxes": torch.tensor([[0.5, 0.48, 0.22, 0.28]] * b, device=device),
                      "positive_sample_batch_dim_indices": torch.arange(b, device=device),
                      "positive_sample_map_dim_indices": torch.full((b,), 136, device=device)}))
    return pool


def milestone(metrics: dict, m, gate_init: float) -> dict:
    gate = m.codetrack.denoiser.residual_gate.detach().reshape(())
    return {
        "d_input": metrics.get("Error/d_input"),
        "d_before": metrics.get("Error/d_before"),
        "d_after": metrics.get("Error/d_after"),
        "gain_total": metrics.get("Error/gain_total"),
        "gain_refiner": metrics.get("Error/gain_refiner"),
        "q_auroc_mask": metrics.get("Error/q_auroc_mask"),
        "q_auprc_mask": metrics.get("Error/q_auprc_mask"),
        "q_spearman": metrics.get("Error/q_error_spearman"),
        "Loss/pres": metrics.get("Loss/pres"),
        "Loss/rec": metrics.get("Loss/rec"),
        "Loss/cls_clean": metrics.get("Loss/cls_clean"),
        # the raw parameter; read its sigmoid to see how far the gate has opened
        "gate": float(gate),
    }


def run_arm(gate_init: float, updates: int, lr: float, pool, report_every: int = 100,
            min_gain: float = 5e-3, freeze_gate: bool = False) -> dict:
    m = build(gate_init).cuda().train()
    crit = CodeTrackCriteria().cuda()
    # force token-level corruption so every batch carries a measurable per-token target
    m.codetrack_cfg.corruption_enabled = True
    m.codetrack_cfg.corruption_token_prob = 1.0
    m.codetrack_cfg.corruption_image_prob = 0.0
    m._corruption_schedule = None

    params = [p for n, p in m.named_parameters() if p.requires_grad and n.startswith("codetrack.")]
    if freeze_gate:
        # Control arm: the gate cannot move, so any change in d_after must come from the rest of
        # the correction branch (refiner/denoiser/conditioning) rather than from the gate opening.
        # This is what separates "the gate is being pushed open by Loss/rec" from
        # "the branch itself moves tokens away from the input".
        params = [p for p in params
                  if p is not m.codetrack.denoiser.residual_gate
                  and p is not m.codetrack.refiner.residual_gate]
    assert params, "no CodeTrack parameters to train"
    opt = torch.optim.AdamW(params, lr=lr)

    print(f"\n===== residual_gate_init = {gate_init:+.1f} | {updates} updates | "
          f"lr={lr:g} | {len(params)} CodeTrack tensors | freeze_gate={freeze_gate} =====")
    print(f"  write schedule = {m.codetrack.denoiser.write_schedule} "
          f"weights={[round(float(v), 4) for v in m.codetrack.denoiser.write_weight]}")
    print(f"{'upd':>5} " + " ".join(f"{k:>12}" for k in KEYS))

    history = []
    for step in range(1, updates + 1):
        data, tg = pool[(step - 1) % len(pool)]
        opt.zero_grad(set_to_none=True)
        out = m(**data)
        co = crit(out, tg)
        co.loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % report_every == 0 or step == 1:
            row = milestone(co.metrics, m, gate_init)
            history.append(row)
            print(f"{step:>5} " + " ".join(
                f"{('-' if row[k] is None else f'{row[k]:.6f}'):>12}" for k in KEYS), flush=True)

    first, last = history[0], history[-1]
    # ``gain_total`` at initialisation is fp32 rounding noise around 0 (~1e-7).  A gain counts
    # only if it is BOTH positive and larger than the noise floor, otherwise the "improving"
    # verdict would fire on the numerical dust produced by a branch that changes nothing.
    ok_gain = (last["gain_total"] is not None and last["gain_total"] > min_gain
               and first["gain_total"] is not None and last["gain_total"] > first["gain_total"])
    ok_diag = last["q_auroc_mask"] is not None and last["q_auroc_mask"] > 0.5
    clean_drift = None
    if first["Loss/cls_clean"] and last["Loss/cls_clean"]:
        clean_drift = last["Loss/cls_clean"] / first["Loss/cls_clean"] - 1.0
    ok_clean = clean_drift is None or clean_drift < 0.10
    print(f"  VERDICT gate={gate_init:+.1f}: gain_total {first['gain_total']} -> {last['gain_total']} "
          f"({'improving' if ok_gain else 'NOT improving'}); "
          f"q_auroc_mask {last['q_auroc_mask']} ({'ok' if ok_diag else 'not above chance'}); "
          f"clean drift {clean_drift if clean_drift is None else round(clean_drift, 4)} "
          f"({'ok' if ok_clean else 'WORSE'}); "
          f"gate param {first['gate']:.4f} -> {last['gate']:.4f} "
          f"(sigmoid {torch.sigmoid(torch.tensor(last['gate'])):.4f})")
    return {"gate_init": gate_init, "history": history,
            "go": bool(ok_gain and ok_diag and ok_clean)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gates", default="-8,-5",
                    help="comma-separated residual_gate_init values (default -8,-5)")
    ap.add_argument("--updates", type=int, default=600,
                    help="optimizer updates per arm (default 600)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--pool", type=int, default=8, help="number of fixed batches to cycle")
    ap.add_argument("--report-every", type=int, default=100)
    ap.add_argument("--freeze-gate", action="store_true",
                    help="control arm: exclude both residual_gate parameters from the optimizer")
    ap.add_argument("--min-gain", type=float, default=5e-3,
                    help="minimum d_input - d_after (angular units) to count as a real gain; "
                         "below this the change is fp32 noise around zero")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is unavailable; this report needs the real model on GPU.", file=sys.stderr)
        return 1

    gates = [float(v) for v in args.gates.split(",") if v.strip()]
    pool = batch_pool(args.pool, b=args.batch)
    results = []
    for g in gates:
        results.append(run_arm(g, args.updates, args.lr, pool, args.report_every,
                               min_gain=args.min_gain, freeze_gate=args.freeze_gate))

    print("\n===== summary =====")
    print(f"{'gate':>6} {'gain_total(first)':>18} {'gain_total(last)':>17} {'q_auroc_mask':>13} "
          f"{'go':>4}")
    for r in results:
        f, l = r["history"][0], r["history"][-1]
        print(f"{r['gate_init']:>6.1f} {str(f['gain_total']):>18} {str(l['gain_total']):>17} "
              f"{str(l['q_auroc_mask']):>13} {'YES' if r['go'] else 'no':>4}")
    any_go = any(r["go"] for r in results)
    print(f"\nGO condition: gain_total > {args.min_gain:g} and increasing, "
          "q_auroc_mask > 0.5, clean tracking not more than 10% worse than at the start of the arm.")
    print("RESULT:", "GO" if any_go else "NO-GO")
    return 0 if any_go else 1


if __name__ == "__main__":
    raise SystemExit(main())
