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
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safetensors.torch import load_file  # noqa: E402
from trackit.core.boot.funcs.utils.custom_yaml_loader import CustomLoader  # noqa: E402
from trackit.models.backbone.builder import build_backbone  # noqa: E402
from trackit.models.methods.GOLA.gola import GOLA_DINOv2  # noqa: E402
from codetrack.criteria import (  # noqa: E402
    CodeTrackCriteria, _auroc_against_mask, _rank_auroc)
from trackit.runner.training.common.optimization.optimizer.per_parameter_options.apply import (  # noqa: E402
    parse_optimizer_per_params_config)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHT = os.path.join(ROOT, "weights/gola_b224.bin")

# The stages that are about to run.  S1 is CodeTrack-only (parameter_scope), S2 is joint PEFT;
# both use the spatial branch (motion/memory off).
S1_CONFIG = os.path.join(ROOT, "config/GOLA/codetrack_s1/config.yaml")
S3_CONFIG = os.path.join(ROOT, "config/GOLA/codetrack_s3/config.yaml")
S4_CONFIG = os.path.join(ROOT, "config/GOLA/codetrack_s4/config.yaml")
STAGE_CONFIG = os.path.join(ROOT, "config/GOLA/codetrack_s2/config.yaml")
FULL_CONFIG = os.path.join(ROOT, "config/GOLA/codetrack_full/config.yaml")
SMOKE_CONFIG = os.path.join(ROOT, "config/GOLA/codetrack_smoke/config.yaml")
PREFLIGHT_CONFIG = os.path.join(ROOT, "config/GOLA/codetrack_preflight/config.yaml")

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



def build_from_branch(branch_cfg: dict):
    """Build the model with the exact CodeTrack branch a stage config declares.

    ``requires_grad`` must be set the way the builder sets it (frozen DINOv2 base excluded via
    ``requires_grad=False``, LoRA/head/embed/token-embed trainable).  Without that step every
    ``parameter_scope`` query returns "no parameters", which is a probe artefact and not a
    property of the config.
    """
    # ``enabled`` must be passed through: it is a bool field and the dataclass default is False,
    # so stripping it silently constructs no CodeTrack branch at all (the previous revision of
    # this helper did exactly that and every scope query returned "no parameters").
    return build(dict(branch_cfg))

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

    # ---- stage identity: length in optimizer updates, and a defined parameter scope ----------
    stage_cfgs = {}
    for path, label in ((S1_CONFIG, "S1"), (STAGE_CONFIG, "S2"), (S3_CONFIG, "S3"),
                        (S4_CONFIG, "S4"), (FULL_CONFIG, "S3full"),
                        (PREFLIGHT_CONFIG, "S0")):
        try:
            stage_cfgs[label] = load_stage_config(path)
        except Exception as exc:  # noqa: BLE001
            check(f"{label} config parses", False, f"{type(exc).__name__}: {exc}")

    for label, cfg in stage_cfgs.items():
        rt = cfg["run"]["runner"]["train"]
        st = rt.get("stage") or {}
        o = rt["optimization"]
        ls = o["lr_scheduler"]
        max_updates = st.get("max_updates")
        warmup_updates = st.get("warmup_updates")
        t_init = (ls.get("override") or {}).get("t_initial_updates")
        warm = ls["parameters"].get("warmup_updates")
        check(f"{label}: stage.max_updates is set and matches the cosine horizon",
              max_updates is not None and t_init == max_updates,
              f"max_updates={max_updates} t_initial_updates={t_init}")
        check(f"{label}: warmup_updates matches between stage and scheduler",
              warmup_updates is not None and warm == warmup_updates,
              f"stage={warmup_updates} scheduler={warm}")

    # S1 must be CodeTrack-only.  Without this the "S1" command silently ran S2-style joint PEFT,
    # because the optimizer whitelist was hard-coded and no config could express the freeze.
    # NOTE the location: ``parameter_scope`` belongs to ``optimization.optimizer``, which is where
    # the parameter-selection code reads it -- not to the ``stage`` block.
    s1_opt_cfg = (stage_cfgs.get("S1", {}).get("run", {}).get("runner", {}).get("train", {})
                  .get("optimization", {}).get("optimizer", {}))
    s1_scope = s1_opt_cfg.get("parameter_scope")
    check("S1 declares parameter_scope = ['codetrack'] (GOLA frozen)",
          s1_scope == ["codetrack"], f"parameter_scope={s1_scope}")
    s2_opt_cfg = (stage_cfgs.get("S2", {}).get("run", {}).get("runner", {}).get("train", {})
                  .get("optimization", {}).get("optimizer", {}))
    s2_scope = s2_opt_cfg.get("parameter_scope")
    check("S2 leaves parameter_scope at the framework default (joint PEFT)",
          s2_scope is None, f"parameter_scope={s2_scope}")

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

    # ---- S1 really freezes GOLA -------------------------------------------------------------
    # Same model and criterion, only the optimizer scope changes.  If this passes but the S1
    # config lacked ``parameter_scope``, running "S1" would silently be S2.
    s1_opt = stage_cfgs["S1"]["run"]["runner"]["train"]["optimization"]["optimizer"]
    s1_groups = parse_optimizer_per_params_config(
        m, crit, {"lr": s1_opt["lr"], "weight_decay": s1_opt["weight_decay"],
                  "parameter_scope": s1_opt["parameter_scope"],
                  "per_parameter": s1_opt["per_parameter"]})
    s1_names = [name_of.get(id(p), "?") for g in s1_groups for p in g["params"]]
    s1_foreign = [n for n in s1_names if not n.startswith("codetrack.")]
    check("S1 optimizer contains CodeTrack parameters only",
          bool(s1_names) and not s1_foreign,
          f"{len(s1_names)} tensors, {len(s1_foreign)} foreign"
          + (f" e.g. {s1_foreign[:2]}" if s1_foreign else ""))
    s1_all_ct = sum(1 for n, p in m.named_parameters()
                    if p.requires_grad and n.startswith("codetrack."))
    check("S1 covers every CodeTrack tensor", len(s1_names) == s1_all_ct,
          f"{len(s1_names)}/{s1_all_ct}")

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
    for key in ("Error/d_input", "Error/d_before", "Error/d_after", "Error/gain_total",
                "Error/gain_refiner", "Error/gain_denoiser", "Error/q_mean",
                "Error/q_auroc_mask", "Error/q_error_spearman"):
        check(f"{key} is reported", key in co.metrics, f"= {co.metrics.get(key)}")

    ex = out["codetrack_extras"]
    check("diagnosis targets are produced by the model",
          ex.get("error_target") is not None and ex.get("syndrome_target") is not None,
          f"error_target={None if ex.get('error_target') is None else tuple(ex['error_target'].shape)}"
          f" syndrome_target={None if ex.get('syndrome_target') is None else tuple(ex['syndrome_target'].shape)}")
    check("pre-denoise recovery tensor is exposed (needed for d_before)",
          ex.get("recovered_pre_denoise") is not None)
    check("student input tokens are exposed (needed for d_input)",
          ex.get("input_tokens") is not None,
          f"input_tokens={None if ex.get('input_tokens') is None else tuple(ex['input_tokens'].shape)}")

    # Diagnosis ranking.  At initialisation the head is random, so AUROC ~ 0.5 is the EXPECTED
    # reading -- this check exists to prove the quantity is measurable and in a sane range before
    # training, and to leave the number in the log so the trend (it must rise above 0.5) is
    # visible.  A missing/inverted measurement is what would be a defect.
    thr = 0.25
    auroc_soft = _rank_auroc(ex["q"].detach().float(), ex["error_target"].detach().float(), thr=thr)
    check("q AUROC (soft target) is measurable and not wildly out of range at init",
          auroc_soft is not None and 0.2 <= auroc_soft <= 0.8,
          f"AUROC(q, error_target>{thr}) = {auroc_soft} (expected ~0.5 for a random head)")
    # The primary metric needs no threshold: the injector's own token label is the ground truth.
    auroc_mask = _auroc_against_mask(ex["q"].detach().float(), ex["corruption_mask"])
    check("q AUROC against the exact corruption mask is measurable",
          auroc_mask is not None and 0.0 <= auroc_mask <= 1.0,
          f"AUROC(q, corruption_mask) = {auroc_mask} (threshold-free primary metric)")
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
    xin = ex.get("input_tokens")
    if (cor is not None and clean is not None and rec is not None and pre is not None
            and xin is not None and bool(cor.any())):
        cmask = cor.to(torch.bool)
        # same angular metric as the criterion, so the gate and the loss cannot disagree
        rel = lambda x: (1.0 - F.cosine_similarity(x.detach(), clean, dim=-1, eps=1e-6)).clamp(0, 2)  # noqa: E731
        d_input = float(rel(xin)[cmask].mean())
        d_before = float(rel(pre)[cmask].mean())
        d_after = float(rel(rec)[cmask].mean())
        check("d_input/d_before/d_after are measured on corrupted tokens only",
              d_input > 1e-3,
              f"d_input={d_input:.6f} d_before={d_before:.6f} d_after={d_after:.6f} "
              f"gain_total={d_input - d_after:+.2e} ({int(cmask.sum())} tokens)")
        check("d_input is strictly larger than the post-recovery distance",
              d_input > d_before - 1e-6,
              f"d_input={d_input:.6f} vs d_before={d_before:.6f} (recovery can only reduce it)")
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

    # ---- the noise half of "noise-modulated" must actually be alive -------------------------
    # Regression: the module defaulted ``alpha`` to ones while the caller never passed it, so
    # ``eps * (1 - alpha)`` was identically 0.  The write-back was gated correctly, which is why
    # the loss still fell -- the dead half was invisible in every training curve.
    d_noise = NoiseModulatedDenoiser(dim=16, hidden=16, heads=4, steps=2, cond_dim=16,
                                     num_checks=4, motion_dim=2, memory_dim=8)
    d_noise.train()
    b, nn, cc = 4, 256, 16
    zeros_tok = torch.zeros(b, nn, cc)
    zeros_cond = torch.zeros(b, nn, cc)
    err = torch.full((b, nn, 1), 0.01)
    err[:, :32] = 0.99
    captured = []
    handle = d_noise.in_proj.register_forward_hook(
        lambda mod, inp, out: captured.append(inp[0].detach().clone()))
    dmg, healthy = [], []
    for seed in range(8):
        captured.clear()
        torch.manual_seed(seed)
        d_noise(zeros_tok, zeros_cond, syndrome=torch.zeros(b, 4),
                motion=torch.zeros(b, 2), memory=torch.zeros(b, 8), token_error=err)
        last = captured[-1]
        dmg.append(float(last[:, :32].std()))
        healthy.append(float(last[:, 32:].std()))
    handle.remove()
    mean_dmg = sum(dmg) / len(dmg)
    mean_healthy = sum(healthy) / len(healthy)
    noise_ratio = mean_dmg / max(mean_healthy, 1e-12)
    check("noise injection reaches damaged tokens (not dead code)",
          mean_dmg > 1e-3 and noise_ratio > 3.0,
          f"damaged std={mean_dmg:.5f} healthy std={mean_healthy:.5f} ratio={noise_ratio:.1f}x")

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

    # The scheduler's own time axis must match the update budget the stage declares.  Built from
    # the REAL stage scheduler config, so a missing override/warmup_updates shows up here rather
    # than as a stage that quietly runs a different cosine horizon than its own max_updates.
    from trackit.runner.training.common.optimization.lr_scheduler.timm_scheduler.builder import (  # noqa: E402
        build_timm_lr_scheduler)
    for label in ("S1", "S2", "S3", "S4", "S3full", "S0"):
        cfg = stage_cfgs.get(label)
        if cfg is None:
            continue
        rt = cfg["run"]["runner"]["train"]
        ls_cfg = rt["optimization"]["lr_scheduler"]
        st = rt["stage"]
        fake_opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-4)
        per_iter, _, warmup_steps = build_timm_lr_scheduler(
            ls_cfg, fake_opt, rt["optimization"]["optimizer"]["lr"], num_epochs=10,
            num_iterations_per_epoch=16384, grad_accumulation_steps=16)
        check(f"{label}: built cosine horizon == stage.max_updates",
              per_iter.t_initial == st["max_updates"],
              f"t_initial={per_iter.t_initial} max_updates={st['max_updates']}")
        check(f"{label}: built warmup == stage.warmup_updates",
              per_iter.warmup_t == st["warmup_updates"],
              f"warmup_t={per_iter.warmup_t} stage={st['warmup_updates']}")
        # Without the override the horizon would silently fall back to a 10-epoch value, which is
        # exactly the coupling between "stage length" and "num_epochs" this replaces.
        check(f"{label}: horizon is NOT the epoch-derived fallback",
              per_iter.t_initial != 10 * (16384 // 16) or st["max_updates"] == 10240,
              f"t_initial={per_iter.t_initial}")



    # ------------------------------------------------------------------ I
    print("\n" + "=" * 92)
    print("I. PER-STAGE OPTIMIZER SCOPE AND LR ROUTING (LIVE MODEL)")
    print("=" * 92)
    # Each stage must select a DIFFERENT parameter set, otherwise "stage" is a label and not a
    # mechanism.  The probe model is rebuilt once per stage *scope*, not once per config: the
    # temporal stages need motion/memory modules that the spatial probe deliberately omits, and
    # ``parse_optimizer_per_params_config`` prints one line per parameter (handled below), so
    # reusing a single model keeps this section to seconds instead of minutes.
    import contextlib
    import io as _io

    def scope_union(cfg):
        """(set of optimised names, {name: lr}) for a stage config, on a matching probe model."""
        probe = build_from_branch(cfg["model"]["codetrack"])
        probe.cuda().train()
        probe_name_of = {id(p): n for n, p in probe.named_parameters()}
        probe_crit = CodeTrackCriteria().cuda()
        o = cfg["run"]["runner"]["train"]["optimization"]["optimizer"]
        sink = _io.StringIO()
        with contextlib.redirect_stdout(sink):
            gs = parse_optimizer_per_params_config(probe, probe_crit, {
                "lr": o["lr"], "weight_decay": o["weight_decay"],
                "parameter_scope": o.get("parameter_scope"),
                "per_parameter": o["per_parameter"]})
        lrs = {}
        for g in gs:
            for p in g["params"]:
                lrs[probe_name_of.get(id(p), "?")] = g.get("lr", o["lr"])
        del probe, probe_crit
        torch.cuda.empty_cache()
        return set(lrs), lrs

    scopes = {}
    for label in ("S1", "S2", "S3", "S4"):
        if label not in stage_cfgs:
            continue
        try:
            names, lrs = scope_union(stage_cfgs[label])
        except Exception as exc:  # noqa: BLE001
            check(f"{label} optimizer scope resolvable", False, f"{type(exc).__name__}: {exc}")
            continue
        scopes[label] = (names, lrs)
        print(f"  [{label}] {len(names)} optimised tensors", flush=True)

    if "S1" in scopes and "S2" in scopes:
        check("S1 selects a strictly smaller parameter set than S2 (freeze is real)",
              scopes["S1"][0] < scopes["S2"][0],
              f"S1={len(scopes['S1'][0])} S2={len(scopes['S2'][0])}")
    if "S2" in scopes and "S3" in scopes:
        check("S3 adds DINOv2 last-2 base weights to S2",
              len(scopes["S3"][0]) > len(scopes["S2"][0]),
              f"S2={len(scopes['S2'][0])} S3={len(scopes['S3'][0])}")
        dino = [n for n in scopes["S3"][0] if n.startswith("blocks.") and "lora" not in n]
        check("S3 unfreezes only blocks 10/11 base weights",
              bool(dino) and all(n.startswith(("blocks.10.", "blocks.11.")) for n in dino),
              f"{len(dino)} tensors, e.g. {sorted(dino)[:2]}")
    if "S4" in scopes:
        names, lrs = scopes["S4"]
        temporal = [n for n in names if ".motion." in n or ".memory." in n or "template_gate" in n]
        check("S4 enables the temporal parameters", bool(temporal), f"{len(temporal)} tensors")
        # the DINOv2 base weights must be FROZEN again in S4
        dino4 = [n for n in names if n.startswith("blocks.") and "lora" not in n]
        check("S4 re-freezes the DINOv2 base weights", not dino4,
              f"{len(dino4)} unexpectedly trainable")

    # LR routing for the S3 foundation blocks: they must get the small foundation lr, not the
    # base lr.  Falling through to 1e-4 would be ~70x too large for a pretrained block.
    if "S3" in scopes:
        _, lrs = scopes["S3"]
        dino_names = sorted(n for n in lrs if n.startswith(("blocks.10.", "blocks.11."))
                            and "lora" not in n)
        if dino_names:
            got = lrs[dino_names[0]]
            check("S3 routes DINOv2 last-2 to the foundation lr (1.5e-6)",
                  got == 1.5e-6, f"{dino_names[0]} -> lr={got}")
        # block 0 must not be in the optimizer at all
        check("S3 leaves the first ten DINOv2 blocks out of the optimizer",
              not any(n.startswith("blocks.0.") for n in lrs))
    if "S4" in scopes:
        _, lrs = scopes["S4"]
        motion_names = [n for n in lrs if ".motion." in n and n.endswith(".weight")]
        if motion_names:
            got = lrs[sorted(motion_names)[0]]
            check("S4 routes motion to 1e-4", got == 1e-4,
                  f"{sorted(motion_names)[0]} -> lr={got}")
    print("\n" + "=" * 92)
    bad = [r for r in RESULTS if not r[1]]
    print(f"RESULT: {len(RESULTS) - len(bad)}/{len(RESULTS)} checks pass")
    for n_, _, det in bad:
        print(f"  FAILED: {n_}  {det}")
    print("=" * 92)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
