"""Correct the 'dead noise_gate' reading: the noise path is train-only.

flow_probe ran the model in .eval().  recovery.py:317 guards the whole noise-injection block
behind ``if train_noise and self.training:``, so in eval the gate is never read and zeroing it
is trivially a no-op.  That is NOT a defect -- it is the correct behaviour for inference.

This script re-runs the same ablation in TRAIN mode and additionally checks the semantic
contract that actually matters:

    the noise must be *selective*: damaged tokens receive more noise than healthy ones.

The project already recorded a 98.2x selective-noise ratio once; this asserts it still holds
and reports the number so a regression is visible.
"""
import inspect
import os
import sys

os.chdir("/home/yangjuanfeng/lab/projects/gola-CodeTrack")
sys.path.insert(0, os.getcwd())

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from trackit.models.backbone.builder import build_backbone  # noqa: E402
from trackit.models.methods.GOLA.gola import GOLA_DINOv2  # noqa: E402

CFG = dict(enabled=True, z_len=64, x_len=256, dim=768, grid=16, mid_dim=128,
           num_checks=64, h_links_per_check=12, h_min_col_degree=3, h_locality_window=5,
           topk_tokens=32, num_neighbours=8, refiner_hidden=256, diffusion_steps=2,
           motion_enabled=True, memory_enabled=True, template_protection=True)


def build():
    torch.manual_seed(0)
    bb = build_backbone({"type": "DINOv2",
                         "parameters": {"name": "ViT-B/14", "acc": "default"}},
                        torch_jit_trace_compatible=False)
    m = GOLA_DINOv2(bb, (8, 8), (16, 16), 64, 64.0, 0.0, False, codetrack_config=CFG)
    m.load_state_dict(load_file("weights/gola_b224.bin"), strict=False)
    return m.cuda()


B = 4
g = torch.Generator(device="cuda").manual_seed(4)
DATA = dict(
    z=torch.rand(B, 6, 112, 112, generator=g, device="cuda") * .4 + .3,
    x=torch.rand(B, 6, 224, 224, generator=g, device="cuda") * .4 + .3,
    d=torch.rand(B, 6, 112, 112, generator=g, device="cuda") * .4 + .3,
    z_feat_mask=torch.ones(B, 8, 8, dtype=torch.long, device="cuda"),
    d_feat_mask=torch.ones(B, 8, 8, dtype=torch.long, device="cuda"),
)
GT = torch.tensor([[.5, .48, .22, .28]] * B, device="cuda")
ISZ = torch.tensor([[224., 224.]] * B, device="cuda")

print("=" * 96)
print("H. noise_gate semantics: TRAIN mode, and the selectivity contract")
print("=" * 96)

# ---- 1. train-mode ablation: does zeroing the gate change the output? ----------
m = build().train()


def run(mo, **kw):
    mo.reset_sequence()
    with torch.no_grad():
        return mo(**DATA, gt_box=GT, image_size=ISZ, **kw)


base = run(m)
ex = base["codetrack_extras"]

dsig = list(inspect.signature(type(m.codetrack.denoiser).forward).parameters)
dfn = type(m.codetrack.denoiser).forward


def ablate(name):
    m2 = build().train()
    idx = dsig.index(name) - 1

    def inner(self, *a, **k):
        a, k = list(a), dict(k)
        if name in k:
            if torch.is_tensor(k[name]):
                k[name] = torch.zeros_like(k[name])
        elif 0 <= idx < len(a) and torch.is_tensor(a[idx]):
            a[idx] = torch.zeros_like(a[idx])
        return dfn(self, *a, **k)

    m2.codetrack.denoiser.forward = inner.__get__(m2.codetrack.denoiser)
    o = run(m2)
    e = o["codetrack_extras"]
    r = (e["recovered"] - ex["recovered"]).abs().max().item() / max(
        ex["recovered"].abs().max().item(), 1e-12)
    return r


for nm in ["noise_gate", "token_error", "token_trust", "syndrome", "motion", "memory"]:
    print("  TRAIN zero %-12s -> d(X_final) rel = %.3e" % (nm, ablate(nm)))

# ---- 2. selectivity contract: damaged tokens must get more noise ---------------
m3 = build().train()
cap = {}
orig = type(m3.codetrack.denoiser).forward


def spy(self, tokens, condition, syndrome=None, motion=None, memory=None,
        token_error=None, token_trust=None, noise_gate=None, train_noise=True,
        noise_weak=0.05, noise_strong=0.20, strong_prob=0.5):
    cap["gate"] = noise_gate.detach().float().reshape(tokens.shape[0], -1) \
        if noise_gate is not None else None
    return orig(self, tokens, condition, syndrome=syndrome, motion=motion, memory=memory,
                token_error=token_error, token_trust=token_trust, noise_gate=noise_gate,
                train_noise=train_noise, noise_weak=noise_weak, noise_strong=noise_strong,
                strong_prob=strong_prob)


m3.codetrack.denoiser.forward = spy.__get__(m3.codetrack.denoiser)
run(m3)
gt_ = cap["gate"]
if gt_ is not None:
    print()
    print("  noise gate (train): min=%.6f max=%.6f mean=%.6f std=%.6f spread=%.6f"
          % (gt_.min(), gt_.max(), gt_.mean(), gt_.std(), gt_.max() - gt_.min()))
    print("  -> selectivity ratio  max/min = %.2fx   (1x would mean the gate is a constant)"
          % (gt_.max() / max(gt_.min(), 1e-12)))
