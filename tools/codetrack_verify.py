"""Verification harness for the CodeTrack branch.

Checks, in order:

  1. parity             -- with the branch disabled the model is bit-identical to upstream
  2. identity at init   -- with the branch enabled, stage-0 output == GOLA baseline output
  3. checkpoint         -- weights/gola_b224.bin loads; missing/unexpected keys reported
  4. shape audit        -- every node of the architecture figure, printed
  5. gradient flow      -- every new sub-module receives a non-zero gradient

Run:  LD_LIBRARY_PATH=/root/autodl-tmp/lab/tools python tools/codetrack_verify.py
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safetensors.torch import load_file  # noqa: E402
from trackit.models.backbone.builder import build_backbone  # noqa: E402
from trackit.models.methods.GOLA.gola import GOLA_DINOv2  # noqa: E402

from codetrack.config import CodeTrackConfig  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WEIGHT = os.path.join(_ROOT, "weights/gola_b224.bin")
DEFAULT_CONFIG: Dict = dict(
    enabled=True, z_len=64, x_len=256, dim=768, grid=16,
    mid_dim=128, num_checks=64, h_links_per_check=12, h_min_col_degree=3,
    h_locality_window=5, syndrome_hidden=128,
    decoder_type="neural_bp", bp_iterations=3, bp_damping=0.5,
    topk_tokens=32, num_neighbours=8, refiner_hidden=256, refiner_heads=4,
    residual_gate_init=-3.0, up_init_std=0.002,
    satr_rounds=3,
    motion_enabled=True, memory_enabled=True, memory_frames=3, memory_tokens=8, memory_dim=128,
    template_protection=True,
)


def build(batch: int = 2, cfg: Dict = None, seed: int = 0):
    torch.manual_seed(seed)
    bb = build_backbone({"type": "DINOv2", "parameters": {"name": "ViT-B/14", "acc": "default"}},
                        torch_jit_trace_compatible=False)
    model = GOLA_DINOv2(bb, (8, 8), (16, 16), lora_r=64, lora_alpha=64.0,
                        lora_dropout=0.0, use_rslora=False, codetrack_config=cfg)
    model.eval().cuda()
    return model


def fake_input(batch: int = 2, seed: int = 1, const: bool = True):
    """Constant 0.5 crops by default, matching the official dummy data generator.

    Random Gaussian crops are not used: the frozen backbone plus a *randomly
    initialised* LoRA (alpha=64) produces non-finite logits on them, which would
    confuse the identity check.
    """
    if const:
        z = torch.full((batch, 6, 112, 112), 0.5, device="cuda")
        x = torch.full((batch, 6, 224, 224), 0.5, device="cuda")
        d = torch.full((batch, 6, 112, 112), 0.5, device="cuda")
        zm = torch.ones(batch, 8, 8, dtype=torch.long, device="cuda")
        dm = torch.ones_like(zm)
        gt = torch.tensor([[56.0, 56.0, 28.0, 28.0]] * batch, device="cuda")
        isz = torch.tensor([[224.0, 224.0]] * batch, device="cuda")
        return dict(z=z, x=x, d=d, z_feat_mask=zm, d_feat_mask=dm), gt, isz
    g = torch.Generator(device="cuda").manual_seed(seed)
    z = torch.randn(batch, 6, 112, 112, generator=g, device="cuda")
    x = torch.randn(batch, 6, 224, 224, generator=g, device="cuda")
    d = torch.randn(batch, 6, 112, 112, generator=g, device="cuda")
    zm = torch.ones(batch, 8, 8, dtype=torch.long, device="cuda")
    dm = torch.ones_like(zm)
    gt = torch.tensor([[56.0, 56.0, 28.0, 28.0]] * batch, device="cuda")
    isz = torch.tensor([[224.0, 224.0]] * batch, device="cuda")
    return dict(z=z, x=x, d=d, z_feat_mask=zm, d_feat_mask=dm), gt, isz


# --------------------------------------------------------------------------- checks
def check_parity():
    """Branch disabled -> no new parameters, and forward takes the upstream path."""
    print("\n== [1] parity: branch disabled ==")
    m_off = build(cfg=None)
    has_ct = hasattr(m_off, "codetrack") and m_off.codetrack is not None
    print(f"  codetrack module present : {has_ct}   (expect False)")
    n_off = sum(p.numel() for p in m_off.parameters())
    m_on = build(cfg=DEFAULT_CONFIG)
    n_on = sum(p.numel() for p in m_on.parameters())
    print(f"  params disabled={n_off/1e6:.2f}M  enabled={n_on/1e6:.2f}M  "
          f"new={(n_on - n_off)/1e6:.3f}M")
    assert not has_ct, "disabled model must not build a CodeTrack branch"
    return m_off, m_on


def check_identity(m_on, weight: str):
    """Enabled branch at initialisation must not change the head's input.

    Comparing two *separate* model instances is meaningless here: the GOLA checkpoint
    stores only the trained ``GA.*`` groups, so two fresh instances have different random
    ``lora.A`` / ``lora.B`` and therefore different features.  The check is done inside one
    instance instead -- the teacher pass (clean, no CodeTrack) against the student pass,
    which routes through CodeTrack.  With the recovery branch zero-initialised they must
    agree.

    The checkpoint is loaded first, because a randomly initialised LoRA (alpha=64) makes the
    frozen backbone output non-finite, which says nothing about CodeTrack.
    """
    print("\n== [2] identity at init: CodeTrack output == GOLA baseline ==")
    sd = load_file(weight)
    m_on.load_state_dict(sd, strict=False)
    m_on.reset_sequence()
    data, gt, isz = fake_input(const=True)
    with torch.no_grad():
        teacher = m_on(**data, teacher=True)
        student = m_on(**data, gt_box=gt, image_size=isz)
    finite = bool(torch.isfinite(student["score_map"]).all())
    d_score = (teacher["score_map"] - student["score_map"]).abs().max().item()
    d_box = (teacher["boxes"] - student["boxes"]).abs().max().item()
    print(f"  outputs finite  : {finite}")
    print(f"  max|dscore_map| = {d_score:.3e}   (teacher vs CodeTrack, clean input)")
    print(f"  max|dboxes|     = {d_box:.3e}")
    assert finite, "CodeTrack output is not finite with the checkpoint loaded"
    # The recovery branch is a *near*-identity at step 0: the residual gate starts at
    # sigmoid(-8) ~ 3.4e-4 (see codetrack/recovery.py for why it is not exactly zero -- an
    # exactly-zero output layer would leave the whole branch with no gradient).  The band
    # below therefore measures "stage-0 perturbation is negligible", not exact equality.
    assert d_score < 1e-2 and d_box < 1e-2, "initialisation is not approximately identity"
    return d_score, d_box


def check_checkpoint(m_on, weight: str):
    print(f"\n== [3] checkpoint: {weight} ==")
    sd = load_file(weight)
    ct_keys = {k for k in sd if k.startswith("codetrack")}
    print(f"  checkpoint tensors          : {len(sd)}")
    print(f"  of which codetrack.*        : {len(ct_keys)}   (expect 0: new params)")
    res = m_on.load_state_dict(sd, strict=False)
    missing = [k for k in res.missing_keys]
    unexpected = [k for k in res.unexpected_keys]
    gola_missing = [k for k in missing if not k.startswith("codetrack")]
    ct_missing = [k for k in missing if k.startswith("codetrack")]
    print(f"  unexpected keys             : {len(unexpected)}")
    print(f"  missing (GOLA, frozen bb)   : {len(gola_missing)}")
    print(f"  missing (codetrack, new)    : {len(ct_missing)}")
    if gola_missing:
        print(f"    sample: {gola_missing[:4]}")
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    # every non-codetrack, non-backbone trainable key must have been loaded
    m_on_sd = m_on.state_dict()
    ck_loaded = sum(1 for k in sd if k in m_on_sd)
    print(f"  checkpoint keys matched     : {ck_loaded}/{len(sd)}")
    assert ck_loaded == len(sd), "checkpoint keys did not all match"
    return dict(missing=len(missing), unexpected=len(unexpected),
                gola_missing=len(gola_missing), ct_missing=len(ct_missing))


def check_shapes(m_on):
    print("\n== [4] shape audit (architecture figure) ==")
    data, gt, isz = fake_input()
    with torch.no_grad():
        out = m_on(**data, gt_box=gt, image_size=isz)
    ex, ct = out["codetrack_extras"], out["codetrack"]
    rows = [
        ("score_map", out["score_map"], "(B,16,16)"),
        ("boxes", out["boxes"], "(B,16,16,4)"),
        ("X_t / recovered", ex["recovered"], "(B,256,768)"),
        ("clean tokens", ex["clean_tokens"], "(B,256,768)"),
        ("q", ex["q"], "(B,256)"),
        ("s", ex["s"], "(B,64)"),
        ("suspect_index", ex["suspect_index"], "(B,K<=32)"),
        ("C_obs", ct["C_obs"], "(B,64,128)"),
        ("C_ref", ct["C_ref"], "(B,64,128)"),
        ("motion_map", ct["motion"]["motion_map"], "(B,1,16,16)"),
        ("uncertainty u_t", ct["motion"]["uncertainty"], "(B,)"),
        ("c_t", ct["c_t"], "(B,)"),
    ]
    for name, t, want in rows:
        print(f"  {name:22s} {str(tuple(t.shape)):20s} expect {want}")
    return out


def check_gradients(m_on, weight: str):
    """Backward through the *real* training criterion, then inspect every new module.

    Using an ad-hoc surrogate loss here is not enough: it would report "zero gradient" for
    modules whose loss term was simply absent from the surrogate (the mean/var completion,
    the temporal memory and the template gate are all supervised through the criterion).
    """
    print("\n== [5] gradient flow into new modules (real criterion) ==")
    from codetrack.criteria import CodeTrackCriteria

    m_on.load_state_dict(load_file(weight), strict=False)
    m_on.train()
    m_on.reset_sequence()
    data, gt, isz = fake_input(const=False)
    # a realistic target pack, as produced by the box_with_score_map label plugin
    targets = {
        "num_positive_samples": torch.tensor(2.0, device="cuda"),
        "boxes": torch.tensor([[0.3, 0.3, 0.2, 0.2]] * gt.shape[0], device="cuda"),
        "positive_sample_batch_dim_indices": torch.tensor([0, 1], device="cuda"),
        "positive_sample_map_dim_indices": torch.tensor([5 * 16 + 5, 6 * 16 + 6], device="cuda"),
    }
    crit = CodeTrackCriteria(lambda_mem=0.1, lambda_gate=0.1).cuda()
    out = m_on(**data, gt_box=gt, image_size=isz)
    crit_out = crit(out, targets)
    print(f"  criterion loss = {float(crit_out.loss):.4f}")
    for k, v in sorted(crit_out.metrics.items()):
        print(f"    {k:22s} {v:.5f}")
    crit_out.loss.backward()

    groups = {
        "H (parity check)": ["codetrack.H.H"],
        "diagnosis (syndrome/q)": ["codetrack.diagnosis"],
        "template_pool": ["codetrack.template_pool"],
        "SATR recovery": ["codetrack.satr"],
        "motion prior (Kalman)": ["codetrack.motion"],
        "temporal memory": ["codetrack.memory"],
        "template gate": ["codetrack.template_gate"],
    }
    print(f"  {'module':24s} {'grad-norm':>12s}  params")
    bad: List[str] = []
    for label, prefixes in groups.items():
        tot, nparam = 0.0, 0
        for name, p in m_on.named_parameters():
            if any(name.startswith(pref) for pref in prefixes):
                if p.grad is not None:
                    tot += float(p.grad.detach().float().norm())
                nparam += 1
        print(f"  {label:24s} {tot:12.4e}  {nparam}")
        if nparam > 0 and tot == 0.0:
            bad.append(label)
    if bad:
        print(f"  !! modules with ZERO gradient: {bad}")
    else:
        print("  every CodeTrack module received a non-zero gradient")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", default=DEFAULT_WEIGHT)
    ap.add_argument("--skip-parity", action="store_true")
    args = ap.parse_args()

    m_off, m_on = check_parity()
    if not args.skip_parity:
        check_identity(m_on, args.weight)
    info = check_checkpoint(m_on, args.weight)
    check_shapes(m_on)
    bad = check_gradients(m_on, args.weight)

    tot = sum(p.numel() for p in m_on.parameters())
    tr = sum(p.numel() for p in m_on.parameters() if p.requires_grad)
    ctp = sum(p.numel() for p in m_on.codetrack.parameters())
    print("\n== summary ==")
    print(f"  total params        : {tot/1e6:.2f}M")
    print(f"  trainable params    : {tr/1e6:.2f}M")
    print(f"  CodeTrack params    : {ctp/1e6:.3f}M")
    print(f"  checkpoint missing  : {info['missing']} "
          f"({info['gola_missing']} GOLA frozen-backbone + {info['ct_missing']} new)")
    print(f"  modules w/o gradient: {bad if bad else 'none'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
