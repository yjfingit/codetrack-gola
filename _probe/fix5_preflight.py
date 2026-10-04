"""Apply fix #5 -- repair the two links that keep S3 from training the DINOv2 tail.

The defect has THREE broken links, not one:

  (a) ``tools/preflight_acceptance.py::scope_union`` builds the optimiser config dict by hand
      and forgets to carry ``backbone_scope`` across.  It therefore measures a fixture that
      is not the real config -- the exact "testing your own fixture" failure the project's own
      discipline note warns about.  The live runner reads ``optimizer.backbone_scope``; the
      probe never passed it, so the probe could never pass even once the config was fixed.

  (b) ``build_from_branch`` constructs the model via ``build(dict(branch_cfg))`` where
      ``branch_cfg`` is only ``cfg['model']['codetrack']``.  ``GOLA_DINOv2`` freezes every
      trunk parameter in ``__init__`` and nothing calls ``_unfreeze_backbone_scope`` on this
      path, so the probe's trunk is always fully frozen regardless of the config.  The live
      builder calls the unwrap at ``builder.py:75``; the probe must mirror it.

  (c) With (a) and (b) fixed, the config-side ``optimizer.backbone_scope`` (added by fix #4)
      finally reaches the code that consumes it.

After this patch the S3 assertion can pass, AND it passes for the right reason: the probe now
exercises the same builder + optimiser path the runner does.
"""
ROOT = "/home/yangjuanfeng/lab/projects/gola-CodeTrack"

# ------------------------------------------------------------ (b) unfreeze in the probe
P = ROOT + "/tools/preflight_acceptance.py"
src = open(P).read()
orig = src

old = """def build_from_branch(branch_cfg: dict):"""
# nothing to replace above; patch the body and the scope_union call site below.
old_call = """        probe = build_from_branch(cfg["model"]["codetrack"])"""
new_call = """        # The runner unlocks the trunk through ``codetrack.backbone_scope`` before the
        # optimiser ever sees the parameters (GOLA/builder.py:75).  This probe must do the
        # same, or a stage that is supposed to fine-tune the last DINOv2 blocks looks like it
        # trains nothing -- which is what made this check report a defect that was real in the
        # runner but invisible here, for the opposite reason.  ``build`` stores the branch
        # config on the module, so the unlock is applied inside ``build_from_branch`` via the
        # explicit flag below.
        probe = build_from_branch(cfg["model"]["codetrack"])"""
assert old_call in src, "scope_union probe call anchor not found"
src = src.replace(old_call, new_call, 1)

old_opt = """            gs = parse_optimizer_per_params_config(probe, probe_crit, {
                "lr": o["lr"], "weight_decay": o["weight_decay"],
                "parameter_scope": o.get("parameter_scope"),
                "per_parameter": o["per_parameter"]})"""
new_opt = """            # ``backbone_scope`` MUST be carried across: it is how the live optimiser learns
            # that the trunk blocks are in scope (apply.py:53).  Omitting it here made the
            # probe test a stricter fixture than the runner and produced a false FAIL -- the
            # same class of error as the earlier "measured the replica instead of the real
            # config" incident.
            gs = parse_optimizer_per_params_config(probe, probe_crit, {
                "lr": o["lr"], "weight_decay": o["weight_decay"],
                "parameter_scope": o.get("parameter_scope"),
                "backbone_scope": o.get("backbone_scope"),
                "per_parameter": o["per_parameter"]})"""
assert old_opt in src, "scope_union opt anchor not found"
src = src.replace(old_opt, new_opt, 1)

# make build_from_branch perform the unlock, mirroring the live builder
old_b = '''    # ``enabled`` must be passed through: it is a bool field and the dataclass default is False,
    # so stripping it silently constructs no CodeTrack branch at all (the previous revision of
    # this helper did exactly that and every scope query returned "no parameters").
    return build(dict(branch_cfg))'''
new_b = '''    # ``enabled`` must be passed through: it is a bool field and the dataclass default is False,
    # so stripping it silently constructs no CodeTrack branch at all (the previous revision of
    # this helper did exactly that and every scope query returned "no parameters").
    branch = dict(branch_cfg)
    m = build(branch)
    # Mirror GOLA/builder.py:75.  ``GOLA_DINOv2.__init__`` freezes the whole trunk, so a stage
    # that declares ``backbone_scope`` (S3: the last two DINOv2 blocks) is only distinguishable
    # from S2 once the same unlock runs here.
    scope = branch.get("backbone_scope") or []
    if scope:
        from trackit.models.methods.GOLA.builder import _unfreeze_backbone_scope
        _unfreeze_backbone_scope(m, branch)
    return m'''
assert old_b in src, "build_from_branch body anchor not found"
src = src.replace(old_b, new_b, 1)

open(P, "w").write(src)
print("preflight_acceptance.py: %d -> %d bytes" % (len(orig), len(src)))
