"""CodeTrack learnability test: can the recovery branch EVER reduce d_after?

The static probes show the branch cannot recover a known corruption at init, and that opening
the residual gate makes things WORSE (d_after 0.1201 -> 0.1256).  Two explanations remain:

  H1  the branch is mis-wired (gradient cannot flow to the place that would fix it)
  H2  the branch can learn but needs a different optimiser setting / more capacity

To separate them we OPTIMISE THE BRANCH DIRECTLY on a single fixed batch with a
recovery-only objective:  L = 1 - cos(X_final, X_clean)  restricted to the damaged tokens,
optimising ONLY codetrack.* parameters.  If a branch that cannot improve this is trained with
its own objective, then the problem is not "not enough training" -- it is the wiring.

Two arms:
  A. gate FREE      (residual_gate is a trainable scalar)
  B. gate FROZEN at 0 (sigmoid(0)=0.5, so the branch is fully open; only the up/down
     projections can move) -- isolates "gate can't move" from "everything can't move".
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
    return m.cuda()


B = 4
gg = torch.Generator(device="cuda").manual_seed(4)


def mkdata(corrupt):
    x = torch.rand(B, 6, 224, 224, generator=gg, device="cuda") * .4 + .3
    if corrupt:
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
CLEAN = mkdata(False)
CORRUPT = mkdata(True)


def train_arm(freeze_gate, gate_init=0.0, steps=300, lr=1e-3, tag=""):
    m = build(gate_init)
    m.train()
    m.reset_sequence()
    with torch.no_grad():
        Xc = m(**CLEAN, gt_box=GT, image_size=ISZ, teacher=True)
        Xc = None
    # clean teacher tokens: run clean input through the trunk only
    with torch.no_grad():
        m.reset_sequence()
        tc = m(**CLEAN, gt_box=GT, image_size=ISZ)["codetrack_extras"]["input_tokens"]

    if freeze_gate:
        for p in [m.codetrack.refiner.residual_gate, m.codetrack.denoiser.residual_gate]:
            p.requires_grad_(False)

    params = [p for n, p in m.named_parameters()
              if n.startswith("codetrack.") and p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)

    m.reset_sequence()
    with torch.no_grad():
        o0 = m(**CORRUPT, gt_box=GT, image_size=ISZ)
    Xt0 = o0["codetrack_extras"]["input_tokens"]
    d_in = 1 - F.cosine_similarity(Xt0, tc, dim=-1)
    dmg = d_in > (d_in.mean() + d_in.std())

    hist = []
    for i in range(steps):
        m.reset_sequence()
        o = m(**CORRUPT, gt_box=GT, image_size=ISZ)
        Xf = o["codetrack_extras"]["recovered"]
        loss = (1 - F.cosine_similarity(Xf, tc, dim=-1))[dmg].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if i % 50 == 0 or i == steps - 1:
            with torch.no_grad():
                m.reset_sequence()
                oo = m(**CORRUPT, gt_box=GT, image_size=ISZ)
                Xff = oo["codetrack_extras"]["recovered"]
                da = (1 - F.cosine_similarity(Xff, tc, dim=-1))[dmg].mean().item()
                gb = float(torch.sigmoid(m.codetrack.refiner.residual_gate))
            hist.append((i, loss.item(), da, gb))

    print("  [%s] damaged-token d  (lower = better recovery)" % tag)
    print("    %-6s %-12s %-12s %-12s" % ("step", "train_loss", "d_after", "sigmoid(gate)"))
    for i, l, da, gb in hist:
        print("    %-6d %-12.6f %-12.6f %-12.6f" % (i, l, da, gb))
    print("    d_input(reference) = %.6f" % d_in[dmg].mean().item())
    return hist


print("=" * 100)
print("F. DIRECT LEARNABILITY: optimise codetrack.* on a recovery-only loss")
print("=" * 100)
print()
print("--- arm A: gate FREE (starts at 0.0, sigmoid 0.5) ---")
train_arm(False, 0.0, 300, 1e-3, "gate-free @0.0")
print()
print("--- arm B: gate FREE (starts at -8.0 as the project ships) ---")
train_arm(False, -8.0, 300, 1e-3, "gate-free @-8.0")
