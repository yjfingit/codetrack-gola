#!/usr/bin/env python3
"""Hook the one-shot cross-rank syndrome calibration into the training loop."""
import sys, pathlib, hashlib
p = pathlib.Path("trackit/runner/training/default/__init__.py")
src = p.read_text(encoding="utf-8"); orig = src

if "_consume_syndrome_calibration" in src:
    print("ALREADY_PATCHED"); sys.exit(0)

# ---- 1) import + state flag ---------------------------------------------------
old_import = "from .utils import criterion_has_parameters"
if old_import not in src:
    # fall back: find the last 'from .utils' style import
    raise SystemExit("import anchor not found")

new_import = old_import + """


def _consume_syndrome_calibration(module: nn.Module, extras) -> bool:
    \"\"\"Apply the one-shot, CROSS-RANK syndrome gain/offset calibration.

    The diagnosis head exports the raw pre-sigmoid syndrome on its first training
    forward (``codetrack_extras['syndrome_pending_calibration']``).  We aggregate the
    statistics over all ranks here -- OUTSIDE the model -- and apply the result once.

    Why not inside ``forward``: by the time a forward runs, DDP has already built its
    reducer buckets, so a collective there can deadlock.  Why not per-rank: each rank
    would derive its own gain from its own shard, and the first all-reduce would then
    mix four inconsistent parameter sets.

    Returns True when a calibration was applied.
    \"\"\"
    if extras is None:
        return False
    pending = extras.get("syndrome_pending_calibration")
    if pending is None:
        return False
    diagnosis = getattr(module, "diagnosis", None)
    if diagnosis is None:
        return False
    try:
        from codetrack.ddp_calibration import consume_pending_calibration
    except Exception:
        return False
    gain = consume_pending_calibration(diagnosis, pending)
    if gain is not None:
        rank = 0
        try:
            import torch.distributed as _dist
            if _dist.is_available() and _dist.is_initialized():
                rank = _dist.get_rank()
        except Exception:
            pass
        print(f"[rank {rank}] syndrome calibration applied: "
              f"gain={gain:.6f} offset={float(diagnosis.syndrome_logit_offset.detach().item()):.6f}",
              flush=True)
        return True
    return False"""
src = src.replace(old_import, new_import, 1)

# ---- 2) call it right after the criterion runs --------------------------------
old_call = """                if criterion_output.extra_metrics is not None:
                    metrics.update(criterion_output.extra_metrics)
"""
new_call = """                if criterion_output.extra_metrics is not None:
                    metrics.update(criterion_output.extra_metrics)

                # One-shot cross-rank syndrome calibration: the diagnosis head hands us the
                # raw syndrome on its first training forward.  Consume it here, before the
                # first backward, so every rank shares one calibrated gain/offset.
                if self.is_train:
                    _consume_syndrome_calibration(self._model, criterion_output.extra_metrics)
"""
assert src.count(old_call) == 1, "call anchor not unique"
src = src.replace(old_call, new_call)

assert src != orig
p.with_suffix(".py.bak-ddp20261004").write_text(orig, encoding="utf-8")
p.write_text(src, encoding="utf-8")
print("PATCHED runner/__init__.py")
print("md5 before:", hashlib.md5(orig.encode()).hexdigest())
print("md5 after :", hashlib.md5(src.encode()).hexdigest())
