"""CodeTrack: does the recovery branch do anything measurable, and can it ever?

Two experiments, both on real GOLA-B weights + a 10-sequence LasHeR view is NOT needed here
because the question is architectural, not dataset-level: the branch operates on the fused
token tensor F_L, and we inject a KNOWN corruption into the input images so that
X_clean is well-defined.

E1  corrupted-input recovery: corrupt the search image, then measure
      d_input / d_before / d_after  restricted to the tokens the corruption touched.
    A branch that works must show d_after < d_input.

E2  gate sweep: rebuild with residual_gate_init in {-8,-5,-2,0,3} and report the branch's
    total effect ||X_final - X_t|| and the recovered d_after, so we can see whether the
    gate is the binding constraint or whether something else is.
"""
import os
import sys

os.chdir("/home/yangjuanfeng/lab/projects/gola-CodeTrack")
sys.path.insert(0, os.getcwd())

import torch
import torch.nn.functional as F  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from trackit.models.backbone.builder import build_backbone  # noqa: E402
from trackit.models.methods.GOLA.gola import GOLA_DINOv2  # noqa: E402

BASE = dict(enabled=True, z_len=64, x_len=256, dim=768, grid=16, mid_dim=128,
            num_checks=64, h_links_per_check=12, h_min_col_degree=3, h_locality_window=5,
            topk_tokens=32, num_neighbours=8, refiner_hidden=256, diffusion_steps=2,
            motion_enabled=True, memory_enabled=True, template_protection=True)


def build(gate=None):
    cfg = dict(BASE)
    if gate is not None:
        cfg["residual_gate_init"] = float(gate)
    torch.manual_seed(0)
    bb = build_backbone({"type": "DINOv2",
                         "parameters": {"name": "ViT-B/14", "acc": "default"}},
                        torch_jit_trace_compatible=False)
    m = GOLA_DINOv2(bb, (8, 8), (16, 16), 64, 64.0, 0.0, False, codetrack_config=cfg)
    m.load_state_dict(load_file("weights/gola_b224.bin"), strict=False)
    return m.cuda().eval()


B = 4
gg = torch.Generator(device="cuda").manual_seed(4)


def mkdata(corrupt: bool):
    x = torch.rand(B, 6, 224, 224, generator=gg, device="cuda") * .4 + .3
    if corrupt:
        # zero out a 32x32 patch in the RGB image -> a localised, known corruption
        x = x.clone()
        x[:, 0:3, 96:128, 96:128] = 0.0
    return dict(
        z=torch.rand(B, 6, 112, 112, generator=gg, device="cuda") * .4 + .3,
        x=x,
        d=torch.rand(B, 6, 112, 112, generator=gg, device="cuda") * .4 + .3,
        z_feat_mask=torch.ones(B, 8, 8, dtype=torch.long, device="cuda"),
        d_feat_mask=torch.ones(B, 8, 8, dtype=torch.long, device="cuda"),
    )


GT = torch.tensor([[.5, .48, .22, .28]] * B, device="cuda")
ISZ = torch.tensor([[224., 224.]] * B, device="cuda")


def run(m, data, teacher=False):
    m.reset_sequence()
    with torch.no_grad():
        return m(**data, gt_box=GT, image_size=ISZ, teacher=teacher)


print("=" * 100)
print("E1. CORRUPTED INPUT: does the branch move the damaged tokens toward clean?")
print("=" * 100)
m = build()
clean = mkdata(False)
corrupt = mkdata(True)
Xc = run(m, clean)["codetrack_extras"]["input_tokens"]          # clean reference
o = run(m, corrupt)
ex = o["codetrack_extras"]
Xt, Xr, Xf = ex["input_tokens"], ex["recovered_pre_denoise"], ex["recovered"]

# which search tokens did the corruption touch?  use the per-token distance to clean as proxy
d_in = 1 - F.cosine_similarity(Xt, Xc, dim=-1)     # (B,256)
thr = d_in.mean() + d_in.std()
dmg = d_in > thr
print("  corruption severity: mean d_input=%.6f  std=%.6f  #(d>mu+sd)=%d"
      % (d_in.mean(), d_in.std(), int(dmg.sum())))
d_b = 1 - F.cosine_similarity(Xr, Xc, dim=-1)
d_a = 1 - F.cosine_similarity(Xf, Xc, dim=-1)
print("  ALL      tokens: d_input=%.6f  d_before=%.6f  d_after=%.6f"
      % (d_in.mean(), d_b.mean(), d_a.mean()))
print("  DAMAGED  tokens: d_input=%.6f  d_before=%.6f  d_after=%.6f"
      % (d_in[dmg].mean(), d_b[dmg].mean(), d_a[dmg].mean()))
print("  => gain (damaged) refine=%+.6f  total=%+.6f  (negative == recovery worked)"
      % (d_b[dmg].mean() - d_in[dmg].mean(), d_a[dmg].mean() - d_in[dmg].mean()))
print("  ||X_final - X_t||max = %.4e   (X_t absmax %.4e)"
      % ((Xf - Xt).abs().max(), Xt.abs().max()))

print()
print("=" * 100)
print("E2. GATE SWEEP  (does opening the gate help? / is the gate the binding constraint?)")
print("=" * 100)
print("  %-8s %-10s %-12s %-12s %-12s" % ("gate", "sigmoid", "|dX|max", "d_after(dmg)", "d_score"))
for gv in (-8.0, -5.0, -2.0, 0.0, 3.0):
    m2 = build(gv)
    o2 = run(m2, corrupt)
    e2 = o2["codetrack_extras"]
    Xt2, Xf2 = e2["input_tokens"], e2["recovered"]
    d_in2 = 1 - F.cosine_similarity(Xt2, Xc, dim=-1)
    dmg2 = d_in2 > (d_in2.mean() + d_in2.std())
    d_a2 = 1 - F.cosine_similarity(Xf2, Xc, dim=-1)
    import math
    print("  %-8.1f %-10.3e %-12.3e %-12.6f %-12.3e"
          % (gv, 1 / (1 + math.exp(-gv)), (Xf2 - Xt2).abs().max(),
             d_a2[dmg2].mean(), o2["score_map"].abs().max()))
