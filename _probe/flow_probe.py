"""CodeTrack inter-module information-flow audit.

Measures, in one forward pass on the real GOLA-B checkpoint:
  A. per-tensor magnitudes on the forward path (information budget)
  B. the identity test d_input / d_before / d_after restricted to damaged tokens
  C. hook-exact ablation: zero a module's argument at the module boundary and measure
     how much the DOWNSTREAM tensors move.  Zero downstream movement == severed link.
  D. residual-gate arithmetic (what the gate actually multiplies)
  E. q / s dynamic range (is the diagnosis saturated?)
"""
import inspect
import math
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


def build():
    torch.manual_seed(0)
    bb = build_backbone({"type": "DINOv2",
                         "parameters": {"name": "ViT-B/14", "acc": "default"}},
                        torch_jit_trace_compatible=False)
    m = GOLA_DINOv2(bb, (8, 8), (16, 16), 64, 64.0, 0.0, False, codetrack_config=CFG)
    m.load_state_dict(load_file("weights/gola_b224.bin"), strict=False)
    return m.cuda().eval()


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


def run(m):
    m.reset_sequence()
    with torch.no_grad():
        return m(**DATA, gt_box=GT, image_size=ISZ)


def rel(a, b):
    a, b = a.detach().float(), b.detach().float()
    return (a - b).abs().max().item() / max(a.abs().max().item(), 1e-12)


m = build()
base = run(m)
ex, ct = base["codetrack_extras"], base["codetrack"]

print("=" * 104)
print("A. FORWARD-PATH INFORMATION BUDGET (magnitudes at init, real checkpoint)")
print("=" * 104)
for name, t in [("X_t / X_clean (student, teacher)", ex["input_tokens"]),
                ("X_rec   (refiner output)", ex["recovered_pre_denoise"]),
                ("X_final (denoiser output)", ex["recovered"]),
                ("delta   (routed evidence, B,32,768)", ct["delta"]),
                ("q       (error prob)", ex["q"]),
                ("q_logits", ex["q_logits"]),
                ("s       (syndrome)", ex["s"]),
                ("motion_map_norm", ex["motion_map_norm"]),
                ("frame_reliability", ex["frame_reliability"])]:
    t = t.detach().float()
    print("  %-36s absmax=%.4e  std=%.4e  mean=%.4e"
          % (name, t.abs().max(), t.std(), t.mean()))

print()
print("=" * 104)
print("B. IDENTITY / RECOVERY METRIC  (d = 1 - cos(., X_clean), damaged tokens only)")
print("=" * 104)
Xt, Xr, Xf, Xc = (ex["input_tokens"], ex["recovered_pre_denoise"],
                  ex["recovered"], ex["clean_tokens"])
dm = torch.zeros(B, 256, dtype=torch.bool, device="cuda")
dm[:, :16] = True  # 6% of 256 ~ 16 damaged tokens
for tag, a in [("d_input  (X_t)", Xt), ("d_before (X_rec)", Xr), ("d_after  (X_final)", Xf)]:
    v = (1 - F.cosine_similarity(a, Xc, dim=-1))
    print("  %-22s all=%.6f   damaged=%.6f" % (tag, v.mean(), v[dm].mean()))
print("  ||X_rec  - X_t  ||max = %.4e  (rel %.3e)" % ((Xr - Xt).abs().max(), rel(Xr, Xt)))
print("  ||X_final- X_rec||max = %.4e  (rel %.3e)" % ((Xf - Xr).abs().max(), rel(Xf, Xr)))
print("  ||X_final- X_t  ||max = %.4e  (rel %.3e)   <-- TOTAL branch effect"
      % ((Xf - Xt).abs().max(), rel(Xf, Xt)))

print()
print("=" * 104)
print("C. HOOK-EXACT ABLATION  (zero one argument at the refiner boundary)")
print("=" * 104)
sig = list(inspect.signature(type(m.codetrack.refiner).forward).parameters)
print("  refiner.forward args:", sig)
ARGS = [a for a in sig[1:-1] if a != "neighbour_index"]  # skip self and int topk


def zero_ablation(nm):
    """Rebuild, patch the refiner's forward to zero ``nm`` (positional or keyword)."""
    m2 = build()
    fn = type(m2.codetrack.refiner).forward
    idx = sig.index(nm) - 1  # positional index in (*args)

    def inner(self, *a, **k):
        a = list(a)
        k = dict(k)
        if nm in k:
            if torch.is_tensor(k[nm]):
                k[nm] = torch.zeros_like(k[nm])
        elif 0 <= idx < len(a) and torch.is_tensor(a[idx]):
            a[idx] = torch.zeros_like(a[idx])
        return fn(self, *a, **k)

    m2.codetrack.refiner.forward = inner.__get__(m2.codetrack.refiner)
    return m2


for name in ARGS:
    m2 = zero_ablation(name)
    o = run(m2)
    print("  zero %-16s -> d(X_rec)=%.3e  d(X_final)=%.3e  d(score)=%.3e"
          % (name,
             rel(o["codetrack_extras"]["recovered_pre_denoise"], ex["recovered_pre_denoise"]),
             rel(o["codetrack_extras"]["recovered"], ex["recovered"]),
             rel(o["score_map"], base["score_map"])))

print()
print("=" * 104)
print("C2. HOOK-EXACT ABLATION  (zero the DENOISER's conditioning inputs)")
print("=" * 104)
dsig = list(inspect.signature(type(m.codetrack.denoiser).forward).parameters)
print("  denoiser.forward args:", dsig)
DZ = [a for a in dsig[1:] if a in ("syndrome", "motion", "memory", "token_error",
                                   "noise_gate", "condition")]
dfn = type(m.codetrack.denoiser).forward
for name in DZ:
    m2 = build()
    idx = dsig.index(name) - 1

    def inner(self, *a, **k):
        a = list(a)
        k = dict(k)
        if name in k:
            if torch.is_tensor(k[name]):
                k[name] = torch.zeros_like(k[name])
        elif 0 <= idx < len(a) and torch.is_tensor(a[idx]):
            a[idx] = torch.zeros_like(a[idx])
        return dfn(self, *a, **k)

    m2.codetrack.denoiser.forward = inner.__get__(m2.codetrack.denoiser)
    o = run(m2)
    print("  zero %-14s -> d(X_final)=%.3e   d(score)=%.3e"
          % (name,
             rel(o["codetrack_extras"]["recovered"], ex["recovered"]),
             rel(o["score_map"], base["score_map"])))

print()
print("=" * 104)
print("D. RESIDUAL-GATE ARITHMETIC")
print("=" * 104)
for gv in (-8.0, -5.0, -2.0, 0.0, 3.0):
    s = 1 / (1 + math.exp(-gv))
    eff = s * 0.23
    print("  gate=%-6.1f sigmoid=%.4e  x q(0.2334) = %.4e   / X_t.absmax(20.13) = %.3e"
          % (gv, s, eff, eff / 20.13))

print()
print("=" * 104)
print("E. q / s DYNAMIC RANGE")
print("=" * 104)
q = ex["q"].detach().float()
s = ex["s"].detach().float()
print("  q : min=%.6f max=%.6f mean=%.6f std=%.6f spread=%.6f"
      % (q.min(), q.max(), q.mean(), q.std(), q.max() - q.min()))
print("  s : min=%.6f max=%.6f mean=%.6f std=%.6f spread=%.6f"
      % (s.min(), s.max(), s.mean(), s.std(), s.max() - s.min()))
ql = ex["q_logits"].detach().float()
print("  q_logits: min=%.3f max=%.3f std=%.3f" % (ql.min(), ql.max(), ql.std()))
si = ex["suspect_index"]
print("  TopK q: mean=%.6f std=%.6f   vs all-token mean=%.6f"
      % (torch.gather(q, 1, si).mean(), torch.gather(q, 1, si).std(), q.mean()))
