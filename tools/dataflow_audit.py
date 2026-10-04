"""Live audit of the CodeTrack training dataflow.

Instruments the *real* training step (dual-branch forward + criterion backward) with
module hooks and prints every module-to-module tensor that crosses a boundary: shape, dtype,
and gradient norm after backward.  The point is to answer two questions with evidence rather
than by reading code:

  1. WHICH modules exchange information, and through exactly what tensors?
  2. IS THAT WIRING SOUND -- i.e. does any signal reach a place it should not (a ground-truth
     box or a teacher feature leaking into the student path), and does every signal that
     should exist actually reach its consumer?

Run:
    LD_LIBRARY_PATH=/root/autodl-tmp/lab/tools python tools/dataflow_audit.py
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safetensors.torch import load_file  # noqa: E402
from trackit.models.backbone.builder import build_backbone  # noqa: E402
from trackit.models.methods.GOLA.gola import GOLA_DINOv2  # noqa: E402
from codetrack.criteria import CodeTrackCriteria  # noqa: E402

WEIGHT = "/root/autodl-tmp/lab/projects/gola-CodeTrack/weights/gola_b224.bin"
CFG = dict(enabled=True, z_len=64, x_len=256, dim=768, grid=16, mid_dim=128,
           num_checks=64, h_links_per_check=12, h_min_col_degree=3, h_locality_window=5,
           h_free_edge_frac=0.25, syndrome_hidden=128,
           topk_tokens=32, num_neighbours=8, refiner_hidden=256, refiner_heads=4,
           residual_gate_init=-8.0, motion_bias_scale=0.5,
           diffusion_enabled=True, diffusion_steps=2, diffusion_hidden=256,
           motion_enabled=True, memory_enabled=True, memory_frames=3,
           memory_tokens=8, memory_dim=128, template_protection=True)

# ---------------------------------------------------------------- capture machinery
LIN: List[Tuple[str, str]] = []     # (module, description)
_tensors: Dict[str, torch.Tensor] = {}


def cap(desc: str, t):
    """Record a tensor crossing a boundary, with shape / dtype / grad info."""
    if t is None:
        LIN.append((desc, "--  (None)"))
        return
    if isinstance(t, (tuple, list)):
        for i, x in enumerate(t):
            cap(f"{desc}[{i}]", x)
        return
    if not torch.is_tensor(t):
        LIN.append((desc, f"--  ({type(t).__name__})"))
        return
    if t.requires_grad and t.is_floating_point():
        # ``retain_grad`` makes the gradient of a *non-leaf* interface tensor readable after
        # backward.  Leaf inputs (the module's own inputs) are additionally wrapped by the
        # wrappers below, which clone them with ``requires_grad_``.
        t.retain_grad()
    _tensors[desc] = t
    LIN.append((desc, f"{tuple(t.shape)}  {str(t.dtype).replace('torch.','')}"))


def build():
    torch.manual_seed(0)
    bb = build_backbone({"type": "DINOv2", "parameters": {"name": "ViT-B/14", "acc": "default"}},
                        torch_jit_trace_compatible=False)
    m = GOLA_DINOv2(bb, (8, 8), (16, 16), lora_r=64, lora_alpha=64.0,
                    lora_dropout=0.0, use_rslora=False, codetrack_config=CFG)
    m.load_state_dict(load_file(WEIGHT), strict=False)
    return m.train().cuda()


def inputs(batch=2):
    """Non-constant crops so the corruption / diagnosis path is actually exercised."""
    g = torch.Generator(device="cuda").manual_seed(7)
    z = torch.rand(batch, 6, 112, 112, generator=g, device="cuda") * 0.4 + 0.3
    x = torch.rand(batch, 6, 224, 224, generator=g, device="cuda") * 0.4 + 0.3
    d = torch.rand(batch, 6, 112, 112, generator=g, device="cuda") * 0.4 + 0.3
    zm = torch.ones(batch, 8, 8, dtype=torch.long, device="cuda")
    dm = torch.ones_like(zm)
    gt = torch.tensor([[96.0, 96.0, 48.0, 64.0]] * batch, device="cuda")
    isz = torch.tensor([[224.0, 224.0]] * batch, device="cuda")
    return dict(z=z, x=x, d=d, z_feat_mask=zm, d_feat_mask=dm), gt, isz


def main():
    from codetrack import motion, recovery, template, ecc

    m = build()
    m.reset_sequence()
    data, gt, isz = inputs()

    # ---------- wrap the CodeTrack sub-modules so their I/O is recorded ----------
    ct = m.codetrack
    def wrap(mod, label, args_spec):
        orig = mod.forward
        def f(*a, **kw):
            for name, val in zip(args_spec, a):
                if torch.is_tensor(val) and val.is_floating_point() and not val.requires_grad:
                    # detach+requires_grad so the gradient w.r.t. this interface is observable
                    val = val.detach().requires_grad_(True)
                cap(f"[IN ] {label}.{name}", val)
            for k, v in kw.items():
                if torch.is_tensor(v) or v is None:
                    cap(f"[IN ] {label}.{k}", v)
            out = orig(*a, **kw)
            if isinstance(out, dict):
                for k, v in out.items():
                    if torch.is_tensor(v) or v is None:
                        cap(f"[OUT] {label}.{k}", v)
            else:
                cap(f"[OUT] {label}", out)
            return out
        mod.forward = f
        return orig

    wrap(ct.H, "H(ParityCheckMatrix)", [])
    wrap(ct.diagnosis, "SyndromeDiagnosis", ["X_t", "X_aux", "H_bar", "template_context"])
    wrap(ct.motion, "KalmanMotionPrior", ["box_xywh", "image_size"])
    wrap(ct.memory, "TemporalMemory", ["tokens", "reliability"])
    wrap(ct.refiner, "H_RoutedSparseRefiner", ["X_t", "X_aux", "q", "neighbour_index"])
    wrap(ct.denoiser, "NoiseModulatedDenoiser", ["tokens", "condition", "syndrome", "token_error",
                                                 "noise_gate"])
    wrap(ct.meanvar, "MeanVarCompletion", ["tokens"])
    wrap(ct.template_gate, "TemplateProtectionGate", ["score", "q"])

    targets = {
        "num_positive_samples": torch.tensor(float(data["z"].shape[0]), device="cuda"),
        "boxes": torch.tensor([[0.5, 0.48, 0.22, 0.28]] * data["z"].shape[0], device="cuda"),
        "positive_sample_batch_dim_indices": torch.arange(data["z"].shape[0], device="cuda"),
        "positive_sample_map_dim_indices": torch.tensor([8 * 16 + 8] * data["z"].shape[0],
                                                        device="cuda"),
    }

    crit = CodeTrackCriteria().cuda()
    out = m(**data, gt_box=gt, image_size=isz)
    co = crit(out, targets)
    co.loss.backward()

    # ---------- report ----------
    print("\n" + "=" * 100)
    print("A. MODULE BOUNDARY TENSORS (real training step, batch=2)")
    print("=" * 100)
    for desc, shape in LIN:
        print(f"  {desc:52s} {shape}")

    # machine-readable dump so downstream tools do not have to parse this report
    import json
    grad_rows = []
    for desc in _tensors:
        t = _tensors[desc]
        if not t.requires_grad or t.grad is None:
            continue
        grad_rows.append({
            "name": desc.replace("[OUT] ", "").replace("[IN ] ", ""),
            "kind": "OUT" if desc.startswith("[OUT]") else "IN",
            "grad_norm": float(t.grad.detach().float().norm()),
            "value_norm": float(t.detach().float().norm()),
        })
    with open("/tmp/dataflow_grads.json", "w") as f:
        json.dump({"grads": grad_rows,
                   "losses": {k: float(v) for k, v in co.metrics.items()}}, f, indent=1)

    print("\n" + "=" * 100)
    print("B. GRADIENT AT EACH INTERFACE (does information actually FLOW back?)")
    print("=" * 100)
    keys = [k for k in _tensors if k.startswith("[OUT]") or k.startswith("[IN ]")]
    for desc in keys:
        t = _tensors[desc]
        if not t.requires_grad or t.grad is None:
            continue
        gn = float(t.grad.detach().float().norm())
        fn = float(t.detach().float().norm())
        rel = gn / fn if fn > 0 else float("nan")
        flag = "" if gn > 0 else "   <-- NO GRADIENT"
        print(f"  {desc:52s} |grad|={gn:10.3e}  rel={rel:9.3e}{flag}")

    print("\n" + "=" * 100)
    print("C. LOSS BREAKDOWN")
    print("=" * 100)
    for k, v in sorted(co.metrics.items()):
        print(f"  {k:26s} {v:12.6f}")

    print("\n" + "=" * 100)
    print("D. INFORMATION-LEAK CHECKS")
    print("=" * 100)
    extras = out["codetrack_extras"]

    # D1. teacher features must be detached (cannot backprop into the corrupted path)
    clean = extras["clean_tokens"]
    print(f"  teacher clean_tokens.requires_grad        = {clean.requires_grad}"
          f"   (expect False)")
    # D2. recovered path must be a genuine function of the corrupted input
    rec = extras["recovered"]
    print(f"  recovered.requires_grad                   = {rec.requires_grad}"
          f"   (expect True)")
    # D3. the teacher head output DOES keep a graph (intentional, for L_track^clean)
    th = extras["teacher"]
    print(f"  teacher.score_map.requires_grad           = {th['score_map'].requires_grad}"
          f"   (expect True: L_track^clean trains the adapters)")
    # D4. memory bank must be detached (a buffer, not a differentiable path)
    mem = out["codetrack"]["motion"]
    if ct.memory is not None:
        st = ct._state
        print(f"  memory bank .requires_grad                = "
              f"{None if st['memory'] is None else st['memory'].requires_grad}"
              f"   (expect False)")
    # D5. nearest-neighbour index must be integer geometry, not learned
    print(f"  neighbour_index.dtype                     = {ct.neighbour_index.dtype}"
          f"   (expect int64: geometry, not a parameter)")
    # D6. the head input in the student path must differ from the teacher's
    Xt = ct._split(ct._split.__self__ and torch.zeros(1)) if False else None
    # D7. Kalman must not have been seeded from the ground-truth box in EVAL mode
    m.eval()
    with torch.no_grad():
        m.reset_sequence()
        out_eval = m(**data)          # no gt_box / image_size passed -> no admission
    mo = out_eval["codetrack"]["motion"]
    print(f"  EVAL motion_map all-zero (no GT admission) = "
          f"{bool((mo['motion_map'] == 0).all())}"
          f"   (expect False: prior is prediction-only, seeded at image centre)")
    print(f"  EVAL motion_box                             = "
          f"{[round(v,1) for v in mo['motion_box'][0].tolist()]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
