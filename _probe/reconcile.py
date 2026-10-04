"""Reconcile the two contradictory readings.

flow_probe B  reported  d_input == d_before == d_after == 0.000000  (identical to 6 dp)
learn_probe F reported  d_input = 0.1092  ->  d_after = 0.0057

Both cannot be right unless the difference is WHERE X_clean comes from.

flow_probe  compared the CodeTrack branch against the SAME model's "teacher" pass on the
SAME (clean) input.  Because the teacher pass and the student pass share the frozen trunk,
X_clean == X_t by construction, so every d collapses to 0.  That makes the metric vacuous:
it cannot see a recovery because there is nothing to recover FROM.

learn_probe compared against the token tensor the trunk produces for a DIFFERENT (corrupted)
image, so d_input is a real 0.109 and the branch's effect is measurable.

This script states the correct metric and reports both readings side by side, so nobody
repeats the vacuous version.
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

CFG = dict(enabled=True, z_len=64, x_len=256, dim=768, grid=16, mid_dim=128,
           num_checks=64, h_links_per_check=12, h_min_col_degree=3, h_locality_window=5,
           topk_tokens=32, num_neighbours=8, refiner_hidden=256, diffusion_steps=2,
           motion_enabled=True, memory_enabled=True, template_protection=True)

torch.manual_seed(0)
bb = build_backbone({"type": "DINOv2", "parameters": {"name": "ViT-B/14", "acc": "default"}},
                    torch_jit_trace_compatible=False)
m = GOLA_DINOv2(bb, (8, 8), (16, 16), 64, 64.0, 0.0, False, codetrack_config=CFG)
m.load_state_dict(load_file("weights/gola_b224.bin"), strict=False)
m.cuda().eval()

B = 4
gg = torch.Generator(device="cuda").manual_seed(4)


def mk(corrupt):
    x = torch.rand(B, 6, 224, 224, generator=gg, device="cuda") * .4 + .3
    if corrupt:
        x = x.clone()
        x[:, 0:3, 96:128, 96:128] = 0.0
    return dict(z=torch.rand(B, 6, 112, 112, generator=gg, device="cuda") * .4 + .3, x=x,
                d=torch.rand(B, 6, 112, 112, generator=gg, device="cuda") * .4 + .3,
                z_feat_mask=torch.ones(B, 8, 8, dtype=torch.long, device="cuda"),
                d_feat_mask=torch.ones(B, 8, 8, dtype=torch.long, device="cuda"))


GT = torch.tensor([[.5, .48, .22, .28]] * B, device="cuda")
ISZ = torch.tensor([[224., 224.]] * B, device="cuda")


def run(data, teacher=False):
    m.reset_sequence()
    with torch.no_grad():
        return m(**data, gt_box=GT, image_size=ISZ, teacher=teacher)


def trunk_tokens(data):
    """Fused token tensor the trunk produces for this input (CodeTrack-independent)."""
    return run(data)["codetrack_extras"]["input_tokens"]


CL = mk(False)
CO = mk(True)

print("=" * 96)
print("G. METRIC RECONCILIATION — 'X_clean' must come from a DIFFERENT input")
print("=" * 96)
Xc = trunk_tokens(CL)             # trunk on the CLEAN image -> the reference
Xow = trunk_tokens(CO)            # trunk on the CORRUPT image, CodeTrack OFF path
m.reset_sequence()
with torch.no_grad():
    o = m(**CO, gt_box=GT, image_size=ISZ)
ex = o["codetrack_extras"]
Xt = ex["input_tokens"]           # the corrupted student token tensor CodeTrack receives
Xr, Xf = ex["recovered_pre_denoise"], ex["recovered"]

d = lambda a, b: (1 - F.cosine_similarity(a, b, dim=-1))
din = d(Xt, Xc)
dmg = din > (din.mean() + din.std())
print("  X_t vs trunk-on-corrupt-input (sanity, must be ~0): %.6f" % d(Xt, Xow).mean())
print("  #tokens where the corruption landed: %d / %d" % (int(dmg.sum()), dmg.numel()))
print()
print("  (i)  VACUOUS definition  X_clean := X_t itself")
print("       d_input=%.6f  d_before=%.6f  d_after=%.6f     <-- says nothing"
      % (d(Xt, Xt).mean(), d(Xr, Xt).mean(), d(Xf, Xt).mean()))
print()
print("  (ii) VALID definition  X_clean := trunk output on the CLEAN image")
print("       d_input =%.6f   (how far the corruption pushed the tokens)"
      % din[dmg].mean())
print("       d_before=%.6f   d_after=%.6f   gain_total=%+.6f  (negative == recovery)"
      % (d(Xr, Xc)[dmg].mean(), d(Xf, Xc)[dmg].mean(),
         d(Xf, Xc)[dmg].mean() - din[dmg].mean()))
