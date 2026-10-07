"""Does the LoRA adapter actually train?

``torch.no_grad()`` around the student trunk silently made every LoRA parameter
untrainable: the parameters never entered the autograd graph, so they received no gradient
at all and no optimizer step could move them.  This is easy to miss because the *head* and
the CodeTrack branch both move normally, so the run looks healthy.

This script runs real AdamW steps and reports, per parameter group, how much moved.  It also
verifies the two invariants that must hold simultaneously:

  * LoRA moves            (joint PEFT actually happens)
  * DINOv2 base does NOT  (the frozen backbone stays frozen)

Run:
    LD_LIBRARY_PATH=/root/autodl-tmp/lab/tools python tools/lora_grad_check.py
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safetensors.torch import load_file  # noqa: E402
from trackit.models.backbone.builder import build_backbone  # noqa: E402
from trackit.models.methods.GOLA.gola import GOLA_DINOv2  # noqa: E402
from codetrack.criteria import CodeTrackCriteria  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHT = os.path.join(_ROOT, "weights/gola_b224.bin")
CFG = dict(enabled=True, z_len=64, x_len=256, dim=768, grid=16, mid_dim=128,
           num_checks=64, h_links_per_check=12, h_min_col_degree=3, h_locality_window=5,
           topk_tokens=32, num_neighbours=8, refiner_hidden=256,
           memory_enabled=True, template_protection=True)

STEPS = 5
# LoRA must move; anything below this fraction of its own magnitude is a failure.
MIN_REL_DISPLACEMENT = 1e-6


def group_of(name: str) -> str:
    if name.startswith("blocks") and ".lora." in name:
        return "LoRA (blocks)"
    if name.startswith("blocks"):
        return "blocks: DINOv2 base (MUST stay frozen)"
    if name.startswith("head"):
        return "head"
    if name.startswith("codetrack"):
        return "CodeTrack"
    return name.split(".")[0]


def build():
    torch.manual_seed(0)
    bb = build_backbone({"type": "DINOv2", "parameters": {"name": "ViT-B/14", "acc": "default"}},
                        torch_jit_trace_compatible=False)
    m = GOLA_DINOv2(bb, (8, 8), (16, 16), lora_r=64, lora_alpha=64.0, lora_dropout=0.0,
                    use_rslora=False, codetrack_config=CFG)
    m.load_state_dict(load_file(WEIGHT), strict=False)
    return m


def main() -> int:
    m = build().train().cuda()
    b = 2
    g = torch.Generator(device="cuda").manual_seed(11)
    z = torch.rand(b, 6, 112, 112, generator=g, device="cuda") * 0.4 + 0.3
    x = torch.rand(b, 6, 224, 224, generator=g, device="cuda") * 0.4 + 0.3
    d = torch.rand(b, 6, 112, 112, generator=g, device="cuda") * 0.4 + 0.3
    zm = torch.ones(b, 8, 8, dtype=torch.long, device="cuda")
    dm = torch.ones_like(zm)
    gt = torch.tensor([[96.0, 96.0, 48.0, 64.0]] * b, device="cuda")
    isz = torch.tensor([[224.0, 224.0]] * b, device="cuda")
    tgt = {"num_positive_samples": torch.tensor(float(b), device="cuda"),
           "boxes": torch.tensor([[0.5, 0.48, 0.22, 0.28]] * b, device="cuda"),
           "positive_sample_batch_dim_indices": torch.arange(b, device="cuda"),
           "positive_sample_map_dim_indices": torch.tensor([136] * b, device="cuda")}

    trainable = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    print(f"trainable tensors: {len(trainable)}  "
          f"({sum(p.numel() for _, p in trainable) / 1e6:.3f} M params)")
    before: Dict[str, torch.Tensor] = {n: p.detach().clone() for n, p in trainable}
    # also snapshot the frozen base so we can prove it did not drift
    frozen = {n: p.detach().clone() for n, p in m.named_parameters() if not p.requires_grad}

    crit = CodeTrackCriteria().cuda()
    opt = torch.optim.AdamW([p for _, p in trainable], lr=1e-4)
    losses = []
    grad_norms = []
    for _ in range(STEPS):
        m.reset_sequence()
        out = m(z=z, x=x, d=d, z_feat_mask=zm, d_feat_mask=dm, gt_box=gt, image_size=isz)
        co = crit(out, tgt)
        opt.zero_grad(set_to_none=True)
        co.loss.backward()
        gn = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1e9)
        grad_norms.append(float(gn))
        opt.step()
        losses.append(float(co.loss))

    print(f"loss: {losses[0]:.4f} -> {losses[-1]:.4f}   "
          f"grad_norm: {min(grad_norms):.2f}..{max(grad_norms):.2f}")

    agg: Dict[str, list] = {}
    for n, p in trainable:
        gname = group_of(n)
        disp = float((p.detach() - before[n]).norm())
        base = float(before[n].norm())
        rec = agg.setdefault(gname, [0.0, 0.0, 0, 0, 0])
        rec[0] += disp
        rec[1] += base
        rec[3] += 1
        if disp > 0:
            rec[2] += 1
        if p.grad is None or float(p.grad.norm()) == 0:
            rec[4] += 1

    print(f"\n  {'group':44s} {'rel.disp':>11s} {'moved':>9s} {'grad==0':>8s}")
    print("  " + "-" * 76)
    ok = True
    for gname, (disp, base, moved, total, nog) in sorted(agg.items()):
        rel = disp / base if base > 0 else float("nan")
        print(f"  {gname:44s} {rel:11.3e} {f'{moved}/{total}':>9s} {f'{nog}/{total}':>8s}")
        if gname.startswith("LoRA"):
            if rel < MIN_REL_DISPLACEMENT or moved == 0:
                print("      ^^ FAIL: LoRA did not train -- joint PEFT is a no-op")
                ok = False
        if "MUST stay frozen" in gname and moved != 0:
            print("      ^^ FAIL: a frozen backbone parameter moved")
            ok = False

    # independent check on the frozen set (these are not in the optimizer, so any change
    # would indicate an in-place op rather than an optimizer step)
    drift = 0
    for n, p in m.named_parameters():
        if not p.requires_grad and n in frozen:
            if not torch.equal(p.detach(), frozen[n]):
                drift += 1
    print(f"\n  frozen params that changed value: {drift}  (expect 0)")

    print("\n" + ("PASS: LoRA trains, backbone stays frozen" if ok and drift == 0
                  else "FAIL"))
    return 0 if (ok and drift == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
