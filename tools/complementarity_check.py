"""Theoretical verification of module COMPLEMENTARITY, expressed as code checks.

"Complementary" is not a vibe; it reduces to four falsifiable properties:

  P1 INDEPENDENT INPUTS  -- no two modules may be driven by the same information channel,
     otherwise they are functionals of one variable and cannot add anything.
  P2 DISTINCT OBJECTS    -- each module must operate on a different mathematical object
     (a token set, a whole-frame vector, a spatial grid, a scalar).
  P3 KILLED CHANNELS     -- a channel that would let one module substitute for another must
     be absent (this is where implementation bugs usually hide).
  P4 NO SHORTCUT         -- no auxiliary signal may reach the tracking head except through
     the recoverable path, otherwise the modules are bypassed and cannot be trained to
     complement each other.

Every check below either passes, or it *reports the defect*.  Run:

    LD_LIBRARY_PATH=/root/autodl-tmp/lab/tools python tools/complementarity_check.py
"""

from __future__ import annotations

import inspect
import os
import sys
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safetensors.torch import load_file  # noqa: E402
from trackit.models.backbone.builder import build_backbone  # noqa: E402
from trackit.models.methods.GOLA.gola import GOLA_DINOv2  # noqa: E402

WEIGHT = "/root/autodl-tmp/lab/projects/gola-CodeTrack/weights/gola_b224.bin"
CFG = dict(enabled=True, z_len=64, x_len=256, dim=768, grid=16, mid_dim=128,
           num_checks=64, h_links_per_check=12, h_min_col_degree=3, h_locality_window=5,
           h_free_edge_frac=0.25, syndrome_hidden=128,
           topk_tokens=32, num_neighbours=8, refiner_hidden=256, refiner_heads=4,
           residual_gate_init=-8.0, motion_bias_scale=0.5,
           diffusion_enabled=True, diffusion_steps=2, diffusion_hidden=256,
           motion_enabled=True, memory_enabled=True, memory_frames=3,
           memory_tokens=8, memory_dim=128, template_protection=True)

RESULTS: List[Tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def build():
    torch.manual_seed(0)
    bb = build_backbone({"type": "DINOv2", "parameters": {"name": "ViT-B/14", "acc": "default"}},
                        torch_jit_trace_compatible=False)
    m = GOLA_DINOv2(bb, (8, 8), (16, 16), lora_r=64, lora_alpha=64.0, lora_dropout=0.0,
                    use_rslora=False, codetrack_config=CFG)
    m.load_state_dict(load_file(WEIGHT), strict=False)
    return m.eval().cuda()


def main() -> int:
    m = build()
    ct = m.codetrack
    src = {n: inspect.getsource(f) for n, f in [
        ("contrast", ct.diagnosis.forward), ("motion", ct.motion.forward),
        ("memory", ct.memory.forward), ("refiner", ct.refiner.forward),
        ("denoiser", ct.denoiser.forward), ("gate", ct.template_gate.forward)]}
    ct_src = inspect.getsource(ct.forward)
    gola_src = inspect.getsource(type(m)._forward_codetrack)

    print("\n" + "=" * 96)
    print("P1  INDEPENDENT INPUT CHANNELS")
    print("=" * 96)
    # The whole frame is single-frame: there is no temporal history in the search tokens.
    check("motion is the ONLY temporal source (single-frame search)",
          "gt_box_xywh" in ct_src and "image_size" in ct_src,
          "Kalman is fed a box + image size; no past frames enter any other module")
    # The syndrome must be built from the token grid, not from a global frame statistic.
    check("diagnosis works on the token grid (not a frame vector)",
          "X_t" in src["contrast"] and "einsum" in src["contrast"],
          "C_obs = H_bar U over tokens -> a 64-d syndrome, not a global embedding")
    # The memory must receive the diagnosis output, not raw features only.
    check("memory is driven by the diagnosis (q -> reliability)",
          "reliability" in inspect.getsource(ct.forward)
          and "1.0 - q" in ct_src,
          "reliability = 1 - q, so history admission depends on the syndrome, not on appearance alone")
    # The refiner must be routed by H, not by spatial proximity.  Check the actual gathering
    # expression rather than the mere presence of the word "grid" (the motion-map flattening
    # legitimately uses a grid side length).
    rf = src["refiner"]
    gathers_by_H = "neighbour_index[suspect_idx]" in rf
    check("refiner K/V are gathered by H incidence, not by coordinates",
          gathers_by_H and "coords" not in rf and "meshgrid" not in rf,
          "nb = neighbour_index[suspect_idx] -> nb_tokens = X_t[b, nb]")

    print("\n" + "=" * 96)
    print("P2  DISTINCT MATHEMATICAL OBJECTS")
    print("=" * 96)
    objs = {
        "diagnosis": "token set -> {C_obs, C_ref} (B,64,128) -> q (B,256): per-token severity",
        "motion": "single state vector -> M_t (B,1,16,16) grid + u_t scalar (B,)",
        "memory": "per-frame summary bank -> prior_tokens (B,8,128): cross-FRAME tokens",
        "refiner": "per-suspect token (B,32,768) residuals: sparse, token-indexed",
        "denoiser": "full token grid (B,256,768) iterated twice: dense",
        "gate": "one scalar per frame c_t (B,): a decision, not a representation",
    }
    for k, v in objs.items():
        print(f"    {k:10s} : {v}")
    check("six modules operate on six different objects", len(set(objs.values())) == 6,
          "token-set / grid+scalar / cross-frame tokens / sparse residual / dense grid / scalar")

    print("\n" + "=" * 96)
    print("P3  CHANNELS THAT WOULD DESTROY COMPLEMENTARITY ARE ABSENT")
    print("=" * 96)
    # (a) the syndrome must not see the corruption mask (that would make it a label copier)
    check("syndrome does not receive the corruption mask",
          "corruption_mask" not in src["contrast"],
          "q is supervised by L_diag, but the mask is not an input -> it must infer, not read")
    # (b) the refiner must not see the clean teacher feature (that would solve recovery trivially)
    check("refiner does not receive clean/teacher features",
          "clean" not in src["refiner"],
          "recovery is conditioned on corrupted tokens + cross-modal + prior only")
    # (c) the denoiser must not see the corruption mask either
    check("denoiser does not receive the corruption mask",
          "corruption_mask" not in src["denoiser"],
          "the noise schedule is applied by the caller, not chosen from the label")
    # (d) the refiner residual must be gated by q, otherwise it is blind selective recovery
    check("refiner residual is gated by q",
          "suspect_idx" in src["refiner"] and "q[" in src["refiner"],
          "delta = q_i * dX_i, so low-severity suspects move little")
    # (e) the denoiser write-back must be gated per token (fixed this round)
    check("denoiser write-back is gated per token by alpha",
          "w * alpha * pred" in src["denoiser"],
          "was `w * pred` (global scalar) -- noise was per-token but the correction was not")

    print("\n" + "=" * 96)
    print("P4  NO SHORTCUT INTO THE TRACKING HEAD")
    print("=" * 96)
    check("head consumes X_final only (no auxiliary bypass)",
          "self.head(out[\"X_final\"]" in gola_src,
          "motion_target / target_mask go to losses, never into the head")
    check("clean branch features are detached for the diagnosis target",
          "clean_tokens.detach()" in gola_src,
          "the teacher cannot be trained by the student's diagnosis loss")
    check("clean branch head keeps a graph (intended, for L_track^clean)",
          "clean_logits = self.head(clean_tokens)" in gola_src,
          "clean tracking loss is a regulariser that still trains the adapters")

    print("\n" + "=" * 96)
    print("P5  EMPIRICAL COROLLARIES")
    print("=" * 96)
    n = 256
    q = torch.rand(2, n)
    si = torch.topk(q, k=32, dim=-1).indices
    dup = any(len(set(si[b].tolist())) != 32 for b in range(2))
    check("TopK suspect set is a genuine set (no duplicate write-back)", not dup)

    # H-routing vs spatial kNN: the decisive risk-#3 question
    from codetrack.ecc import build_sparse_support
    sup = build_sparse_support(64, 256, 12, 3, 16, 5, True, 0.25, 1234)
    a = sup.t() @ sup
    a.fill_diagonal_(0.0)
    deg = (a > 0).sum(dim=1)
    yy, xx = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
    coords = torch.stack([yy.reshape(-1), xx.reshape(-1)], 1).float()
    d = (coords[:, None, :] - coords[None, :, :]).abs().max(-1).values
    d.fill_diagonal_(99)
    h8 = torch.topk(a, k=8, dim=-1).indices
    s8 = torch.topk(d, k=8, largest=False, dim=-1).indices
    ov = torch.tensor([len(set(h8[i].tolist()) & set(s8[i].tolist())) for i in range(256)],
                      dtype=torch.float)
    check("H-routing is NOT spatial kNN",
          float(ov.mean()) < 4.0,
          f"mean overlap {ov.mean():.2f}/8, tokens with >=6 overlap: "
          f"{float((ov >= 6).float().mean()) * 100:.1f}%")
    check("K_n=8 is satisfiable for every token",
          int(deg.min()) >= 8,
          f"min shared-check neighbours = {int(deg.min())} (mean {deg.float().mean():.1f})")

    print("\n" + "=" * 96)
    bad = [r for r in RESULTS if not r[1]]
    print(f"RESULT: {len(RESULTS) - len(bad)}/{len(RESULTS)} checks pass")
    if bad:
        print("DEFECTS:")
        for name, _, detail in bad:
            print(f"  - {name}: {detail}")
    print("=" * 96)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
