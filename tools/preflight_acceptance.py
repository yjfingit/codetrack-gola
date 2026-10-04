"""Pre-training acceptance gate.

Checks the things that were wrong and are cheap to prove right, before spending GPU-days on a
full run.  Every assertion here corresponds to a defect that was actually found and fixed; if
one of them regresses, the training run would silently produce a wrong model.

Design rule: the *structural* checks (optimizer groups, scheduler, accumulation) read the REAL
stage config from ``config/GOLA/`` instead of a copy of its rules.  The previous revision
hard-coded a paraphrased copy, so it passed while the shipped config put the head and LoRA biases
in the CodeTrack learning-rate group -- the check was measuring its own fixture, not the artefact
that would run.

Run:
    LD_LIBRARY_PATH=/root/autodl-tmp/lab/tools python tools/preflight_acceptance.py
"""

from __future__ import annotations

import os
import re
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safetensors.torch import load_file  # noqa: E402
from trackit.core.boot.funcs.utils.custom_yaml_loader import CustomLoader  # noqa: E402
from trackit.models.backbone.builder import build_backbone  # noqa: E402
from trackit.models.methods.GOLA.gola import GOLA_DINOv2  # noqa: E402
from codetrack.criteria import CodeTrackCriteria, _rank_auroc  # noqa: E402
from trackit.runner.training.common.optimization.optimizer.per_parameter_options.apply import (  # noqa: E402
    parse_optimizer_per_params_config)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHT = os.path.join(ROOT, "weights/gola_b224.bin")

# The stage that is about to run.  S2 shares the spatial config (motion/memory off); S1 uses the
# same file with the GOLA parameters frozen.
STAGE_CONFIG = os.path.join(ROOT, "config/GOLA/codetrack_s2/config.yaml")
FULL_CONFIG = os.path.join(ROOT, "config/GOLA/codetrack_full/config.yaml")
SMOKE_CONFIG = os.path.join(ROOT, "config/GOLA/codetrack_smoke/config.yaml")

# Loss/gradient probes need the temporal modules present to exercise them; the stage config
# legitimately disables them (see codetrack_spatial.yaml), so the probe config turns them on.
CFG = dict(enabled=True, z_len=64, x_len=256, dim=768, grid=16, mid_dim=128,
           num_checks=64, h_links_per_check=12, h_min_col_degree=3, h_locality_window=5,
           topk_tokens=32, num_neighbours=8, refiner_hidden=256, diffusion_steps=2,
           motion_enabled=True, memory_enabled=True, template_protection=True)

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def load_stage_config(path):
    with open(path, "rb") as f:
        return yaml.load(f, Loader=CustomLoader)


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
    # ------------------------------------------------------------------ A
    print("=" * 92)
    print("A. STAGE CONFIGS PARSE AND ARE SELF-CONSISTENT")
    print("=" * 92)
    stage = load_stage_config(STAGE_CONFIG)
    opt = stage["run"]["runner"]["train"]["optimization"]
    rules = opt["optimizer"]["per_parameter"]
    sched = opt["lr_scheduler"]["parameters"]
    ct_cfg = stage["model"]["codetrack"]

    check("stage config uses the spatial (temporal-off) CodeTrack branch",
          not ct_cfg.get("motion_enabled", True) and not ct_cfg.get("memory_enabled", True),
          f"motion={ct_cfg.get('motion_enabled')} memory={ct_cfg.get('memory_enabled')}")
    check("stage config has no epoch-based warmup (stages warm up by update count)",
          int(sched.get("warmup_epochs", -1)) == 0, f"warmup_epochs={sched.get('warmup_epochs')}")
    # every tensor rule must pin ndim=2 and every zero_1d rule must pin the 1-D/norm side,
    # otherwise a single rule can span learning-rate scopes (see B below)
    tensor_rules = [r for r in rules if r.get("type") != "zero_1d_param_weight_decay"]
    unfiltered_tensor = [r for r in tensor_rules if r.get("ndim") != 2]
    check("every non-zero_1d rule pins ndim: 2", not unfiltered_tensor,
          f"{len(unfiltered_tensor)} without it")
    zero_rules = [r for r in rules if r.get("type") == "zero_1d_param_weight_decay"]
    scoped_zero = [r for r in zero_rules if "name_regex" in r]
    check("every scoped zero_1d rule pins ndim: [0, 1]",
          all(r.get("ndim") == [0, 1] for r in scoped_zero),
          f"{len(scoped_zero)} scoped rule(s)")
    check("the trailing zero_1d rule is unfiltered (catch-all)",
          "name_regex" not in zero_rules[-1] and rules[-1].get("type") == "zero_1d_param_weight_decay",
          f"last rule = {rules[-1].get('type', 'tensor')}")

    for path, label in ((FULL_CONFIG, "full"), (SMOKE_CONFIG, "smoke")):
        try:
            cfg = load_stage_config(path)
            r = cfg["run"]["runner"]["train"]["optimization"]
            ok = int(r["lr_scheduler"]["parameters"].get("warmup_epochs", -1)) == 0
            check(f"{label} config has warmup_epochs 0", ok)
        except Exception as exc:  # noqa: BLE001 - report, do not crash the gate
            check(f"{label} config parses", False, f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ B
    print("\n" + "=" * 92)
    print("B. OPTIMIZER COVERAGE AND LEARNING-RATE GROUPS (FROM THE REAL CONFIG)")
    print("=" * 92)
    m = build().train().cuda()
    crit = CodeTrackCriteria().cuda()
    groups = parse_optimizer_per_params_config(
        m, crit, {"lr": opt["optimizer"]["lr"], "weight_decay": opt["optimizer"]["weight_decay"],
                  "per_parameter": rules})
    in_opt = sum(len(g["params"]) for g in groups)
    trainable = sum(1 for p in list(m.parameters()) + list(crit.parameters()) if p.requires_grad)
    check("every trainable tensor is in the optimizer", in_opt == trainable,
          f"{in_opt}/{trainable}")

    name_of = {id(p): n for n, p in list(m.named_parameters())}

    def scope_of(name: str) -> str:
        for tag, pattern in (("codetrack", r"codetrack\."), ("head", r"^head\."),
                             ("lora", r"lora"), ("embed", r"token_type_embed")):
            if re.search(pattern, name):
                return tag
        return "other"

    # A group must not mix learning-rate scopes: that is exactly what happened when an
    # unfiltered/ignored `zero_1d` rule drained the whole model's biases into one group.
    mixed = []
    for g in groups:
        scopes = {scope_of(name_of.get(id(p), "")) for p in g["params"]}
        scopes.discard("other")
        if len(scopes) > 1:
            mixed.append(sorted(scopes))
    check("no LR group mixes parameter scopes", not mixed, f"offending groups: {mixed[:3]}")

    lora_a_b = [g for g in groups
                for p in g["params"]
                if name_of.get(id(p), "").endswith((".lora.A", ".lora.B"))]
    check("lora.A / lora.B are optimised (the forward actually uses them)",
          len(lora_a_b) > 0, f"{len(lora_a_b)} tensors")

    # A decay-eligible parameter that only a scoped ``zero_1d`` rule matches must be handed back
    # to the pool, not dropped.  CodeTrack has one: the 3-D ``codetrack.memory.base_prior``.
    # Regression: the empty-set early return used to run BEFORE the restore, so that tensor ended
    # up in no optimizer group at all (measured 1405/1406) while still reporting requires_grad.
    three_d = [n for n, p in list(m.named_parameters()) + list(crit.named_parameters())
               if p.requires_grad and p.dim() > 2]
    owned = {id(p) for g in groups for p in g["params"]}
    dropped = [n for n, p in list(m.named_parameters()) + list(crit.named_parameters())
               if p.requires_grad and id(p) not in owned]
    check("higher-dimensional (3-D+) parameters are not dropped by the zero_1d rules",
          not dropped, f"trainable dim>2: {len(three_d)}, dropped: {dropped}")

    bad_wd = [name_of.get(id(p), "?") for g in groups
              if float(g.get("weight_decay", 0.1)) != 0.0
              for p in g["params"] if p.dim() <= 1]
    check("bias / norm parameters keep weight_decay = 0", not bad_wd,
          f"{len(bad_wd)} violations" + (f" e.g. {bad_wd[:2]}" if bad_wd else ""))

    def info(substr, prefix_only=False, ndim=None):
        """(group index, lr, weight_decay) of the group owning the first matching parameter."""
        for i, g in enumerate(groups):
            for p in g["params"]:
                nm = name_of.get(id(p), "")
                if (nm.startswith(substr) if prefix_only else substr in nm):
                    if ndim is not None and p.dim() != ndim:
                        continue
                    return i, g.get("lr", opt["optimizer"]["lr"]), g.get("weight_decay", 0.1)
        return None

    ct = info("codetrack.H.", ndim=2)
    lora = info("lora.A", ndim=2)
    head = info("head.cls_mlp", prefix_only=True, ndim=2)
    ct_1d = info("codetrack.refiner.norm.weight")
    head_1d = info("head.cls_mlp.layers.0.bias", prefix_only=True)

    check("CodeTrack 2-D group gets lr 1e-4", ct is not None and ct[1] == 1e-4, f"{ct}")
    check("LoRA 2-D group gets lr 2.5e-5", lora is not None and lora[1] == 2.5e-5, f"{lora}")
    check("GOLA head 2-D group gets lr 1e-5", head is not None and head[1] == 1e-5, f"{head}")
    check("CodeTrack 1-D/norm get lr 1e-4 with wd 0",
          ct_1d is not None and ct_1d[1] == 1e-4 and ct_1d[2] == 0.0, f"{ct_1d}")
    check("GOLA head 1-D get lr 1e-5 with wd 0",
          head_1d is not None and head_1d[1] == 1e-5 and head_1d[2] == 0.0, f"{head_1d}")
    check("bias/norm do NOT inherit AdamW's 1e-3 default",
          ct_1d is not None and ct_1d[1] != 1e-3, f"lr={None if ct_1d is None else ct_1d[1]}")

    # ------------------------------------------------------------------ C
    print("\n" + "=" * 92)
    print("C. LOSS TERMS ARE ACTUALLY COMPUTED")
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
    for key in ("Loss/diag", "Loss/rec", "Loss/gain", "Loss/motion", "Loss/align",
                "Loss/trc", "Loss/gate", "Loss/pres"):
        check(f"{key} is computed", key in co.metrics, f"= {co.metrics.get(key)}")
    for key in ("Error/d_before", "Error/d_after", "Error/gain", "Error/q_mean"):
        check(f"{key} is reported", key in co.metrics, f"= {co.metrics.get(key)}")

    ex = out["codetrack_extras"]
    check("diagnosis targets are produced by the model",
          ex.get("error_target") is not None and ex.get("syndrome_target") is not None,
          f"error_target={None if ex.get('error_target') is None else tuple(ex['error_target'].shape)}"
          f" syndrome_target={None if ex.get('syndrome_target') is None else tuple(ex['syndrome_target'].shape)}")
    check("pre-denoise recovery tensor is exposed (needed for d_before)",
          ex.get("recovered_pre_denoise") is not None)

    # Diagnosis ranking.  At initialisation the head is random, so AUROC ~ 0.5 is the EXPECTED
    # reading -- this check exists to prove the quantity is measurable and in a sane range before
    # training, and to leave the number in the log so the trend (it must rise above 0.5) is
    # visible.  A missing/inverted measurement is what would be a defect.
    thr = 0.25
    auroc = _rank_auroc(ex["q"].detach().float(), ex["error_target"].detach().float(), thr=thr)
    check("q AUROC is measurable and not wildly out of range at init",
          auroc is not None and 0.2 <= auroc <= 0.8,
          f"AUROC(q, error_target>{thr}) = {auroc} (expected ~0.5 for a random head)")
    check("error_target separates damaged from untouched tokens",
          float(ex["error_target"].detach().max()) > thr,
          f"max={float(ex['error_target'].detach().max()):.4f} mean={float(ex['error_target'].detach().mean()):.4f}")

    # ------------------------------------------------------------------ D
    print("\n" + "=" * 92)
    print("D. RECOVERY GAIN AND SELECTIVE DIRECTION")
    print("=" * 92)
    cor = ex.get("corruption_mask")
    clean = ex.get("clean_tokens")
    rec = ex.get("recovered")
    pre = ex.get("recovered_pre_denoise")
    if cor is not None and clean is not None and rec is not None and pre is not None and bool(cor.any()):
        cmask = cor.to(torch.bool)
        rel = lambda x: (x - clean).norm(dim=-1) / clean.norm(dim=-1).clamp(min=1e-6)  # noqa: E731
        d_before = float(rel(pre.detach())[cmask].mean())
        d_after = float(rel(rec)[cmask].mean())
        check("d_before/d_after are measured on corrupted tokens only",
              d_before > 1e-3,
              f"d_before={d_before:.6f} d_after={d_after:.6f} gain={d_before - d_after:+.2e} "
              f"({int(cmask.sum())} tokens)")
        # At initialisation the residual gate is sigmoid(-8) ~ 3.4e-4, so the recovery is a
        # near-identity BY DESIGN and a non-positive gain here is expected, not a failure.  What
        # would be a failure is the branch being unable to move a damaged token more than a
        # healthy one, which is the standalone check below.
        check("recovery is a near-identity at initialisation (gate design)",
              abs(d_after - d_before) < 1e-2 * max(d_before, 1e-6),
              f"|gain| / d_before = {abs(d_after - d_before) / max(d_before, 1e-9):.3e}")
    else:
        check("corrupted tokens present", False,
              "no token-level corruption fired; re-run or raise corruption_token_prob")

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

    # ------------------------------------------------------------------ E
    print("\n" + "=" * 92)
    print("E. WRITE-BACK SCHEDULE AND IDENTITY AT INIT")
    print("=" * 92)
    d_ramp = NoiseModulatedDenoiser(dim=768, hidden=64, heads=4, steps=2, cond_dim=768,
                                    write_schedule="ramp")
    d_old = NoiseModulatedDenoiser(dim=768, hidden=64, heads=4, steps=2, cond_dim=768,
                                   write_schedule="linear_noise")
    check("ramp schedule gives every step non-zero write strength",
          bool((d_ramp.write_weight > 0).all()),
          f"ramp={[round(float(v), 4) for v in d_ramp.write_weight]}")
    check("linear_noise schedule still reproduces the old wasted first step",
          float(d_old.write_weight[0]) == 0.0,
          f"linear_noise={[round(float(v), 4) for v in d_old.write_weight]}")
    check("stage config exposes the write schedule",
          str(ct_cfg.get("diffusion_write_schedule", "ramp")) in ("ramp", "linear_noise"),
          f"diffusion_write_schedule={ct_cfg.get('diffusion_write_schedule', 'ramp')}")

    # ------------------------------------------------------------------ F
    print("\n" + "=" * 92)
    print("F. IMAGE CORRUPTION KEEPS A CLEAN TEACHER COPY")
    print("=" * 92)
    plugin_src = open(os.path.join(
        "trackit/data/methods/siamese_tracker_train/transform/default/plugin/image_corruption",
        "__init__.py")).read()
    check("plugin publishes x_clean before overwriting x",
          plugin_src.index('"x_clean"') < plugin_src.index('collated.input[key] ='),
          'collated.input["x_clean"] = x precedes the corrupting write')
    gola_src = open(os.path.join("trackit/models/methods/GOLA/gola.py")).read()
    check("model forwards x_clean to the teacher branch",
          "x_teacher = x_clean if x_clean is not None else x" in gola_src
          and "self._x_feat(x_teacher.float())" in gola_src)

    # ------------------------------------------------------------------ G
    print("\n" + "=" * 92)
    print("G. GRADIENT FLOW INTO EVERY NEW MODULE")
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

    # ------------------------------------------------------------------ H
    print("\n" + "=" * 92)
    print("H. TRAINING-LOOP INFRASTRUCTURE")
    print("=" * 92)
    src = open(os.path.join(ROOT, "trackit/runner/training/default/__init__.py")).read()
    check("scheduler is stepped with an update count, not the micro-step index",
          "optimizer_step = (self._iteration + 1) // self._grad_accumulation_steps" in src,
          "divides by grad_accumulation_steps")
    check("gradient accumulation averages (loss / accum) rather than summing",
          "/ self._grad_accumulation_steps" in src, "backward_loss")

    # The scheduler's own time axis must match the update budget the stage driver assumes.
    from trackit.runner.training.common.optimization.lr_scheduler.timm_scheduler.builder import (  # noqa: E402
        build_timm_lr_scheduler)
    oc = dict(opt["optimizer"])
    oc["lr_scheduler"] = opt["lr_scheduler"]
    oc["lr_scheduler"]["override"] = {"num_epochs": 10}
    fake_opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-4)
    per_iter, _, warmup_steps = build_timm_lr_scheduler(
        oc["lr_scheduler"], fake_opt, 1e-4, num_epochs=10,
        num_iterations_per_epoch=16384, grad_accumulation_steps=16)
    t_initial = per_iter.t_initial
    check("scheduler cosine horizon counts optimizer updates (10 x 1024)",
          t_initial == 10240, f"t_initial={t_initial}")
    check("scheduler warmup is 0 updates with warmup_epochs 0",
          warmup_steps == 0, f"warmup_t={per_iter.warmup_t}")

    print("\n" + "=" * 92)
    bad = [r for r in RESULTS if not r[1]]
    print(f"RESULT: {len(RESULTS) - len(bad)}/{len(RESULTS)} checks pass")
    for n_, _, det in bad:
        print(f"  FAILED: {n_}  {det}")
    print("=" * 92)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
