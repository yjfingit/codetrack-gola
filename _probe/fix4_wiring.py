"""Apply fix #4 -- wire the new knobs through config, and fix the S3 freeze bug.

(4a) config plumbing.  ``up_init_std`` / ``motion_bias_normalise`` / ``syndrome_logit_gain``
     are added to ``CodeTrackConfig`` and passed to the modules.  Defaults keep the *new*
     behaviour, because the old defaults are the measured defect.

(4b) S3 / S4 backbone freeze.  ``trackit/.../per_parameter_options/apply.py:53`` reads
     ``optimizer_config['backbone_scope']``; the S3 and S4 configs only list literal
     ``"blocks.10."`` / ``"blocks.11."`` entries in ``parameter_scope``.  That list is used as
     a *substring filter over parameters that already have requires_grad=True*, and
     ``_unfreeze_backbone_scope`` has not run yet at that point, so the trunk parameters are
     still False and are filtered out.  Result: S3 runs for 2.6 h without training the last two
     DINOv2 blocks, and the framework's "rule must be effective" assertion fires on the
     ``^blocks\\.1[01]\\.`` rule because it matches nothing.

     The fix is to declare the same scope in the place the optimiser actually reads.
"""
import re

ROOT = "/home/yangjuanfeng/lab/projects/gola-CodeTrack"

# ---------------------------------------------------------------- (4a) config.py
P = ROOT + "/codetrack/config.py"
src = open(P).read()
orig = src

old = """    motion_bias_scale: float = 0.5
"""
new = """    motion_bias_scale: float = 0.5
    # Centre and standardise the log-space motion bias before scaling it.  Without this the
    # bias is a near-constant (measured logit spread 0.05 nats over 8 neighbours, because the
    # unit-mass motion map sits at 1/256 +- 2e-3), so the Kalman prior cannot influence
    # attention at all.  False reproduces the pre-2026-10-04 numerics exactly.
    motion_bias_normalise: bool = True
    # Init std of the residual predictor's output layer inside BOTH the refiner and the
    # denoiser.  Replaces the old "large negative residual gate" idiom: a small nonzero
    # ``up`` keeps step-0 output near identity while leaving the conditioning paths their
    # gradient.  See codetrack/recovery.py for the measurements behind this.
    up_init_std: float = 0.02
"""
assert old in src, "motion_bias_scale anchor not found"
src = src.replace(old, new, 1)

old = """    detection_prior: float = 0.2          # syndrome sigmoid bias init
"""
new = """    detection_prior: float = 0.2          # syndrome sigmoid bias init
    # Initial gain on the raw syndrome logit.  The stock head produces s_raw with a spread of
    # only ~0.2 over 64 checks, so s = sigmoid(s_raw) is a point mass at 0.794 and every
    # consumer of q degenerates (constant q -> random TopK routing, non-selective noise gate,
    # one frame-reliability value per batch).  ``calibrate_syndrome_gain`` sets this from the
    # measured logit std; 1.0 disables the correction.
    syndrome_logit_gain: float = 1.0
"""
assert old in src, "detection_prior anchor not found"
src = src.replace(old, new, 1)
open(P, "w").write(src)
print("config.py: %d -> %d bytes" % (len(orig), len(src)))

# ---------------------------------------------------------------- (4a) codetrack.py
P = ROOT + "/codetrack/codetrack.py"
src = open(P).read()
orig = src

old = """            detection_prior=cfg.detection_prior, use_cos=cfg.syndrome_cos)"""
new = """            detection_prior=cfg.detection_prior, use_cos=cfg.syndrome_cos,
            syndrome_logit_gain=cfg.syndrome_logit_gain)"""
assert old in src, "diagnosis ctor anchor not found"
src = src.replace(old, new, 1)

old = """            dropout=cfg.refiner_dropout, residual_gate_init=cfg.residual_gate_init,
            motion_bias_scale=cfg.motion_bias_scale)"""
new = """            dropout=cfg.refiner_dropout, residual_gate_init=cfg.residual_gate_init,
            motion_bias_scale=cfg.motion_bias_scale, up_init_std=cfg.up_init_std,
            motion_bias_normalise=cfg.motion_bias_normalise)"""
assert old in src, "refiner ctor anchor not found"
src = src.replace(old, new, 1)

old = """            residual_gate_init=cfg.residual_gate_init) if cfg.diffusion_enabled else None"""
new = """            residual_gate_init=cfg.residual_gate_init,
            up_init_std=cfg.up_init_std) if cfg.diffusion_enabled else None"""
assert old in src, "denoiser ctor anchor not found"
src = src.replace(old, new, 1)
open(P, "w").write(src)
print("codetrack.py: %d -> %d bytes" % (len(orig), len(src)))

# ---------------------------------------------------------------- (4b) S3 / S4 configs
for name in ("codetrack_s3", "codetrack_s4"):
    P = "%s/config/GOLA/%s/config.yaml" % (ROOT, name)
    src = open(P).read()
    orig = src
    anchor = '          parameter_scope: ["lora.A", "lora.B", "lora.GA", "lora.GB", "embed", "head", "codetrack", "blocks.10.", "blocks.11."]'
    if anchor not in src:
        print("  !! %s: parameter_scope anchor not found, skipped" % name)
        continue
    add = anchor + """
          # ADDED 2026-10-04.  ``parameter_scope`` is applied as a substring filter over
          # parameters that ALREADY have requires_grad=True, and the trunk blocks only get
          # that flag from ``GOLA/builder._unfreeze_backbone_scope`` -- which reads
          # ``codetrack.backbone_scope``.  The optimiser reads ``optimizer.backbone_scope``
          # separately (apply.py:53).  With only the literal entries above, the last two
          # DINOv2 blocks were filtered out before they could be unlocked, so this stage ran
          # without training them and the ``^blocks\\.1[01]\\.`` rule matched nothing
          # ("rule must be effective").  Declaring the scope here is what the optimiser reads.
          backbone_scope: ["blocks.10.", "blocks.11."]"""
    src = src.replace(anchor, add, 1)
    open(P, "w").write(src)
    print("%s/config.yaml: %d -> %d bytes" % (name, len(orig), len(src)))
