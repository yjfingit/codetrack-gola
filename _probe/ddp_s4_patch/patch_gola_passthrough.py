#!/usr/bin/env python3
"""Pass the pending syndrome calibration out of the model so the training loop can
all-reduce it across ranks.  Idempotent; backs up first."""
import sys, pathlib, hashlib
p = pathlib.Path("trackit/models/methods/GOLA/gola.py")
src = p.read_text(encoding="utf-8"); orig = src

if "syndrome_pending_calibration" in src:
    print("ALREADY_PATCHED"); sys.exit(0)

old = '''            "motion_map_norm": out.get("motion_map_norm"),
            "motion_target": out.get("motion_target"),
        }
        result = {"score_map": head_out["score_map"], "boxes": head_out["boxes"],
                  "codetrack_extras": extras,'''
new = '''            "motion_map_norm": out.get("motion_map_norm"),
            "motion_target": out.get("motion_target"),
            # One-shot syndrome calibration.  The diagnosis head exports the RAW pre-sigmoid
            # syndrome on its very first training forward; the training loop all-reduces the
            # statistics across ranks and applies the gain/offset once.  Letting each rank
            # calibrate locally would give the ranks different parameter values, and the first
            # DDP all-reduce would then average four inconsistent models.
            "syndrome_pending_calibration": out.get("syndrome_pending_calibration"),
        }
        result = {"score_map": head_out["score_map"], "boxes": head_out["boxes"],
                  "codetrack_extras": extras,'''
assert src.count(old) == 1, "anchor not unique"
src = src.replace(old, new)
assert src != orig
p.with_suffix(".py.bak-ddp20261004").write_text(orig, encoding="utf-8")
p.write_text(src, encoding="utf-8")
print("PATCHED gola.py")
print("md5 before:", hashlib.md5(orig.encode()).hexdigest())
print("md5 after :", hashlib.md5(src.encode()).hexdigest())
