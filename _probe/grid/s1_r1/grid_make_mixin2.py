#!/usr/bin/env python3
"""Emit a trackit `--mixin_config` file holding the grid overrides.

Patched for the S1 round-1 grid (2026-10-04).  Two fixes over the original
`_probe/grid/grid_make_mixin.py`:

1. FLOAT FORMATTING BUG.  The original wrote `f"{float(raw):.10g}"`, which renders
   `5e-5` as the bare token `5e-05`.  PyYAML parses `5e-05` as a STRING (it needs a
   dot: `5.0e-05`), so `optimizer.lr` arrived as `'5e-05'` and AdamW crashed with
   `TypeError: '<=' not supported between instances of 'float' and 'str'`.
   Fixed by using `repr(float(raw))`, which always emits a form PyYAML reads as float.
   Every value is additionally written through `yaml.safe_dump` so quoting is decided
   by a real YAML emitter rather than by hand.

2. The generator now also accepts `--dry` to print the file without writing.

Usage:
    grid_make_mixin2.py <out.yaml> key=value [key=value ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

PATHS = {
    "max_updates": "run.runner.train.stage.max_updates",
    "warmup_updates": "run.runner.train.stage.warmup_updates",
    "t_initial_updates": "run.runner.train.optimization.lr_scheduler.override.t_initial_updates",
    "sched_warmup_updates": "run.runner.train.optimization.lr_scheduler.parameters.warmup_updates",
    "lr": [
        "run.runner.train.optimization.optimizer.lr",
        "run.runner.train.optimization.optimizer.per_parameter.0.lr",
        "run.runner.train.optimization.optimizer.per_parameter.1.lr",
        "run.runner.train.optimization.optimizer.per_parameter.2.lr",
        "run.runner.train.optimization.optimizer.per_parameter.3.lr",
        "run.runner.train.optimization.optimizer.per_parameter.4.lr",
        "run.runner.train.optimization.optimizer.per_parameter.5.lr",
    ],
    "recovery_lr": [
        "run.runner.train.optimization.optimizer.per_parameter.0.lr",
        "run.runner.train.optimization.optimizer.per_parameter.3.lr",
    ],
    "diagnosis_lr": [
        "run.runner.train.optimization.optimizer.per_parameter.1.lr",
        "run.runner.train.optimization.optimizer.per_parameter.4.lr",
    ],
    "other_lr": [
        "run.runner.train.optimization.optimizer.per_parameter.2.lr",
        "run.runner.train.optimization.optimizer.per_parameter.5.lr",
    ],
    "lr_min": "run.runner.train.optimization.lr_scheduler.parameters.lr_min",
    "global_batch_size": "run.data.train.global_batch_size",
    "samples_per_epoch": "run.data.train.sampler.samples_per_epoch",
    "sampler_seed": "run.data.train.sampler.seed",
    "checkpoint_max_to_keep": "run.checkpoint.0.max_to_keep",
    "grad_accumulation_steps": "run.runner.train.optimization.grad_accumulation_steps",
    "max_grad_norm": "run.runner.train.optimization.max_grad_norm",
    "topk_tokens": "model.codetrack.topk_tokens",
    "up_init_std": "model.codetrack.up_init_std",
    "diffusion_enabled": "model.codetrack.diffusion_enabled",
    "corruption_image_prob": "model.codetrack.corruption_image_prob",
    "corruption_token_prob": "model.codetrack.corruption_token_prob",
    "corruption_token_ratio": "model.codetrack.corruption_token_ratio",
    "corruption_centered": "model.codetrack.corruption_centered",
    "corruption_spatial_mode": "model.codetrack.corruption_spatial_mode",
    "lambda_diag": "run.runner.train.criteria.lambda_diag",
    "lambda_diag_rank": "run.runner.train.criteria.lambda_diag_rank",
    "diag_rank_margin": "run.runner.train.criteria.diag_rank_margin",
    "lambda_rec": "run.runner.train.criteria.lambda_rec",
    "lambda_gain": "run.runner.train.criteria.lambda_gain",
    "gain_margin": "run.runner.train.criteria.gain_margin",
    "lambda_align": "run.runner.train.criteria.lambda_align",
    "lambda_pres": "run.runner.train.criteria.lambda_pres",
    "lambda_mem": "run.runner.train.criteria.lambda_mem",
    "lambda_gate": "run.runner.train.criteria.lambda_gate",
    "lambda_motion": "run.runner.train.criteria.lambda_motion",
    "lambda_post_syndrome": "run.runner.train.criteria.lambda_post_syndrome",
    "post_syndrome_margin": "run.runner.train.criteria.post_syndrome_margin",
    "w_track_corr": "run.runner.train.criteria.w_track_corr",
    "w_track_clean": "run.runner.train.criteria.w_track_clean",
    "diagnosis_alpha": "run.runner.train.criteria.diagnosis_alpha",
}

INT_KEYS = {"max_updates", "warmup_updates", "t_initial_updates",
            "sched_warmup_updates", "global_batch_size", "samples_per_epoch",
            "sampler_seed", "checkpoint_max_to_keep", "grad_accumulation_steps", "topk_tokens"}
BOOL_KEYS = {"diffusion_enabled", "corruption_centered"}
STRING_KEYS = {"corruption_spatial_mode"}


def coerce(key: str, raw: str):
    """Return a REAL Python float/int so the YAML emitter picks the right scalar type."""
    if key in BOOL_KEYS:
        value = raw.strip().lower()
        if value not in {"true", "false"}:
            raise ValueError(f"{key} must be true or false, got {raw!r}")
        return value == "true"
    if key in STRING_KEYS:
        return raw.strip()
    if key in INT_KEYS:
        return int(float(raw))
    return float(raw)          # never a string -- this is the bug that was fixed


def main() -> int:
    import yaml

    argv = [a for a in sys.argv[1:] if a != "--dry"]
    out = Path(argv[0])
    raw_edits = []
    for arg in argv[1:]:
        raw_edits.extend(p for p in arg.split(",") if p.strip())

    rules, unknown = [], []
    for kv in raw_edits:
        k, _, v = kv.partition("=")
        k, v = k.strip(), v.strip()
        if not k:
            continue
        if k not in PATHS:
            unknown.append(k)
            continue
        rules.append({"path": PATHS[k], "value": coerce(k, v)})
    if unknown:
        print(f"ERROR unknown keys: {unknown}")
        return 2
    if not rules:
        print("ERROR no edits parsed")
        return 2

    text = yaml.safe_dump(rules, sort_keys=False, default_flow_style=False)
    if "--dry" in sys.argv:
        print(text)
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# generated by grid_make_mixin2.py\n" + text)

    # self-check: re-parse and assert the types survived the round trip
    back = yaml.safe_load(text)
    for r in back:
        if r["path"] not in PATHS.values():
            raise AssertionError(f"unknown emitted path: {r}")
        if r["path"] not in (PATHS["corruption_spatial_mode"],):
            assert not isinstance(r["value"], str), f"value stayed a string: {r}"
    print(f"WROTE {out} ({len(rules)} rules, all values typed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
