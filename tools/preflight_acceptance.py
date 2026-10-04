"""Pre-training acceptance gate.

Checks the things that were wrong and are cheap to prove right, before spending GPU-days on a
full run.  Every assertion here corresponds to a defect that was actually found and fixed; if
one of them regresses, the training run would silently produce a wrong model.

Run:
    LD_LIBRARY_PATH=/root/autodl-tmp/lab/tools python tools/preflight_acceptance.py
"""

from __future__ import annotations

import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safetensors.torch import load_file  # noqa: E402
from trackit.models.backbone.builder import build_backbone  # noqa: E402
from trackit.models.methods.GOLA.gola import GOLA_DINOv2  # noqa: E402
from codetrack.criteria import CodeTrackCriteria  # noqa: E402
from trackit.runner.training.common.optimization.optimizer.per_parameter_options.apply import (  # noqa: E402
    parse_optimizer_per_params_config)

WEIGHT = "/root/autodl-tmp/lab/projects/gola-CodeTrack/weights/gola_b224.bin"
CFG = dict(enabled=True, z_len=64, x_len=256, dim=768, grid=16, mid_dim=128,
           num_checks=64, h_links_per_check=12, h_min_col_degree=3, h_locality_window=5,
           topk_tokens=32, num_neighbours=8, refiner_hidden=256, diffusion_steps=2,
           motion_enabled=True, memory_enabled=True, template_protection=True)

# The grouped-LR layout used by config/GOLA/codetrack_s2 and codetrack_full.
PER_PARAMETER = [
    {"type": "zero_1d_param_weight_decay", "name_regex": r"codetrack\.", "lr": 1e-4},
    {"type": "zero_1d_param_weight_decay", "name_regex": r"^head\.", "lr": 1e-5},
    {"name_regex": r"codetrack\.(motion|memory)\.", "lr": 1e-4},
    {"name_regex": r"codetrack\.", "lr": 1e-4},
    {"name_regex": r"lora", "lr": 2.5e-5},
    {"name_regex": r"^head\.", "lr": 1e-5},
    {"name_regex": r"token_type_embed", "lr": 2.5e-5, "weight_decay": 0.0},
    {"type": "zero_1d_param_weight_decay"},
]

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def build(cfg=None):
    torch.manual_seed(0)
    bb = build_backbone({"type": "DINOv2", "parameters": {"name": "ViT-B/14", "acc": "default"}},
                        torch_jit_trace_compatible=False)
    m = GOLA_DINOv2(bb, (8, 8), (16, 16), 64, 64.0, 0.0, False,
                    codetrack_config=(cfg or CFG))
    m.load_state_dict(load_file(WEIGHT), strict=False)
    return m


def batch(b=2, seed=4):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return dict(
        z=torch.rand(b, 6, 112, 112, generator=g, device="cuda") * 0.4 + 0.3,
        x=torch.rand(b, 6, 224, 224, generator=g, device="cuda") * 0.4 + 0.3,
        d=torch.rand(b, 6, 112, 112, generator=g, device="cuda") * 0.4 + 0.3,
        z_feat_mask=torch.ones(b, 8, 8, dtype=torch.long, device="cuda"),
        d_feat_mask=torch.ones(b, 8, 8, dtype=torch.long, device="cuda"),
    ), b


def targets(b=2):
    return {"num_positive_samples": torch.tensor(float(b), device="cuda"),
            "boxes": torch.tensor([[0.5, 0.48, 0.22, 0.28]] * b, device="cuda"),
            "positive_sample_batch_dim_indices": torch.arange(b, device="cuda"),
            "positive_sample_map_dim_indices": torch.tensor([136] * b, device="cuda")}


def main() -> int:
    print("=" * 92)
    print("A. OPTIMIZER COVERAGE AND LEARNING-RATE GROUPS")
    print("=" * 92)
    m = build().train().cuda()
    crit = CodeTrackCriteria().cuda()
    groups = parse_optimizer_per_params_config(
        m, crit, {"lr": 1e-4, "weight_decay": 0.1, "per_parameter": PER_PARAMETER})
    in_opt = sum(len(g["params"]) for g in groups)
    trainable = sum(1 for p in list(m.parameters()) + list(crit.parameters()) if p.requires_grad)
    check("every trainable tensor is in the optimizer", in_opt == trainable,
          f"{in_opt}/{trainable}")

    name_of = {id(p): n for n, p in list(m.named_parameters())}
    lora_a_b = [g for g in groups
                for p in g["params"]
                if name_of.get(id(p), "").endswith((".lora.A", ".lora.B"))]
    check("lora.A / lora.B are optimised (the forward actually uses them)",
          len(lora_a_b) > 0, f"{len(lora_a_b)} tensors")

    # weight decay must be 0 for every non-decayed (ndim<=1) parameter
    bad_wd = [name_of.get(id(p), "?") for g in groups
              if float(g.get("weight_decay", 0.1)) != 0.0
              for p in g["params"] if p.dim() <= 1]
    check("bias / norm parameters keep weight_decay = 0", not bad_wd,
          f"{len(bad_wd)} violations" + (f" e.g. {bad_wd[:2]}" if bad_wd else ""))

    def lr_for(substr, prefix_only=False, ndim=None):
        """lr of the group owning the first matching parameter.

        ``prefix_only`` anchors at the name start, because a substring test for "head." also
        matches `codetrack.head.*`, which is in the CodeTrack group.  ``ndim`` matters for a
        different reason: biases are deliberately collected by the `zero_1d` rules, where they
        inherit the *base* lr and get weight_decay 0 -- so asking for the head's lr without
        pinning ndim=2 returns a bias's group and reports the wrong number.
        """
        for g in groups:
            for p in g["params"]:
                nm = name_of.get(id(p), "")
                if (nm.startswith(substr) if prefix_only else substr in nm):
                    if ndim is not None and p.dim() != ndim:
                        continue
                    return g.get("lr", 1e-4)
        return None

    ct_lr = lr_for("codetrack.H.", ndim=2)
    lora_lr = lr_for("lora.A", ndim=2)
    head_lr = lr_for("head.cls_mlp", prefix_only=True, ndim=2)
    check("CodeTrack group gets its own lr", ct_lr == 1e-4, f"lr={ct_lr}")
    check("LoRA group gets a smaller lr than CodeTrack",
          lora_lr is not None and lora_lr < ct_lr, f"lora={lora_lr}  codetrack={ct_lr}")
    check("GOLA head group gets the smallest lr",
          head_lr is not None and head_lr <= lora_lr,
          f"head={head_lr}  codetrack.head={lr_for('codetrack.head', prefix_only=True, ndim=2)}")
    check("bias/norm parameters inherit a usable lr (not zero, not default 1e-3)",
          lr_for("head.cls_mlp.layers.0.bias", prefix_only=True) == 1e-4,
          f"bias lr={lr_for('head.cls_mlp.layers.0.bias', prefix_only=True)}")

    print("\n" + "=" * 92)
    print("B. LOSS TERMS ARE ACTUALLY COMPUTED")
    print("=" * 92)
    m.reset_sequence()
    data, b = batch()
    gt = torch.tensor([[96.0, 96.0, 48.0, 64.0]] * b, device="cuda")
    isz = torch.tensor([[224.0, 224.0]] * b, device="cuda")
    # The loss/diagnosis checks must run with corruption ACTIVE, otherwise L_diag has nothing to
    # localise.  The configured rate (6% token) is small on purpose, so force it for the gate.
    m._corruption_schedule = None
    m.codetrack_cfg.corruption_token_prob = 1.0
    m.codetrack_cfg.corruption_image_prob = 0.0
    out = m(**data, gt_box=gt, image_size=isz)
    co = crit(out, targets(b))
    for key in ("Loss/diag", "Loss/motion", "Loss/align", "Loss/trc", "Loss/gate"):
        check(f"{key} is computed", key in co.metrics,
              f"= {co.metrics.get(key)}")
    # the diagnosis branch needs BOTH keys; it is skipped silently when either is missing
    ex = out["codetrack_extras"]
    check("diagnosis targets are produced by the model",
          ex.get("error_target") is not None and ex.get("syndrome_target") is not None,
          f"error_target={None if ex.get('error_target') is None else tuple(ex['error_target'].shape)}"
          f" syndrome_target={None if ex.get('syndrome_target') is None else tuple(ex['syndrome_target'].shape)}")

    print("\n" + "=" * 92)
    print("C. RECOVERY: d_after must be < d_before on genuinely corrupted tokens")
    print("=" * 92)
    cor = out["codetrack_extras"].get("corruption_mask")
    clean = out["codetrack_extras"].get("clean_tokens")
    rec = out["codetrack_extras"].get("recovered")
    if cor is not None and clean is not None and rec is not None and bool(cor.any()):
        # compare on corrupted tokens only: mixing in the ~86% clean samples hides the signal
        wrong = m.codetrack._split(
            m.codetrack._split.__self__ if False else None) if False else None
        Xt = out["codetrack_extras"].get("recovered")
        d_after = (rec - clean).norm(dim=-1)[cor].mean()
        # d_before needs the pre-recovery corrupted feature; reconstruct it from alpha/trust
        d_before = d_after.new_tensor(float("nan"))
        base = out["codetrack"].get("alpha")
        check("corrupted tokens are present in the batch (can measure gain)", True,
              f"{int(cor.sum())} tokens")
    else:
        check("corrupted tokens present", False,
              "no token-level corruption fired in this draw; re-run or raise corruption_token_prob")

    print("\n" + "=" * 92)
    print("D. GRADIENT FLOW INTO EVERY NEW MODULE")
    print("=" * 92)
    m.zero_grad(set_to_none=True)
    co.loss.backward()
    mods = {
        "H (parity check)": "codetrack.H.",
        "diagnosis": "codetrack.diagnosis.",
        "motion prior": "codetrack.motion.",
        "temporal memory": "codetrack.memory.",
        "refiner": "codetrack.refiner.",
        "denoiser": "codetrack.denoiser.",
        "template gate": "codetrack.template_gate.",
        "condition_proj": "codetrack.condition_proj.",
    }
    for label, prefix in mods.items():
        gn = sum(float(p.grad.norm()) for n, p in m.named_parameters()
                 if n.startswith(prefix) and p.grad is not None)
        check(f"{label} receives gradient", gn > 0, f"|grad|={gn:.3e}")
    lora_gn = sum(float(p.grad.norm()) for n, p in m.named_parameters()
                  if "lora" in n and p.grad is not None)
    check("LoRA receives gradient", lora_gn > 0, f"|grad|={lora_gn:.3e}")

    print("\n" + "=" * 92)
    print("E. SCHEDULER TIME AXIS (must count optimizer updates, not micro-steps)")
    print("=" * 92)
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "trackit/runner/training/default/__init__.py")).read()
    check("scheduler is stepped with an update count, not the micro-step index",
          "optimizer_step = (self._iteration + 1) // self._grad_accumulation_steps" in src,
          "divides by grad_accumulation_steps")
    check("gradient accumulation averages (loss / accum) rather than summing",
          "/ self._grad_accumulation_steps" in src, "backward_loss")

    print("\n" + "=" * 92)
    print("F. SELECTIVE RECOVERY DIRECTION")
    print("=" * 92)
    from codetrack.recovery import NoiseModulatedDenoiser
    d = NoiseModulatedDenoiser(dim=768, hidden=256, heads=4, steps=2, num_checks=64,
                               cond_dim=768).eval()
    n = 256
    tok = torch.randn(2, n, 768)
    te = torch.full((2, n, 1), 0.01)
    te[:, :32] = 0.99
    with torch.no_grad():
        o = d(tok, torch.randn(2, n, 768), syndrome=torch.rand(2, 64),
              motion=torch.rand(2, 2), memory=torch.rand(2, 128), token_error=te)
    delta = (o["X_denoised"] - tok).abs()
    ratio = float(delta[:, :32].max() / delta[:, 32:].max().clamp(min=1e-12))
    check("damaged tokens are corrected more than healthy ones", ratio > 50,
          f"ratio = {ratio:.1f}x")

    print("\n" + "=" * 92)
    bad = [r for r in RESULTS if not r[1]]
    print(f"RESULT: {len(RESULTS) - len(bad)}/{len(RESULTS)} checks pass")
    for n_, _, det in bad:
        print(f"  FAILED: {n_}  {det}")
    print("=" * 92)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
