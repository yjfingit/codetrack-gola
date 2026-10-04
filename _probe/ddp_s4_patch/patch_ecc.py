#!/usr/bin/env python3
"""Apply the DDP-sync patch to codetrack/ecc.py.  Idempotent; writes .bak first."""
import sys, pathlib, hashlib

p = pathlib.Path("codetrack/ecc.py")
src = p.read_text(encoding="utf-8")
orig = src

if "syndrome_calibration_statistics" in src:
    print("ALREADY_PATCHED"); sys.exit(0)

# ---- 1) add the two new methods right before calibrate_syndrome_gain ----------
anchor = "    @torch.no_grad()\n    def calibrate_syndrome_gain(self"
new_methods = '''    @torch.no_grad()
    def syndrome_calibration_statistics(self, s_raw: torch.Tensor):
        """Return (count, sum, sum_sq, min, max) of the PRE-sigmoid syndrome.

        Collecting instead of applying keeps this point free of any collective
        communication: the caller aggregates across ranks and then calls
        ``apply_syndrome_calibration`` exactly once, OUTSIDE the model.  Doing the
        all-reduce here, inside ``forward``, is unsafe -- the DDP reducer buckets
        are already built by this point and an unmatched collective hangs the job.
        """
        raw = s_raw.detach().float().reshape(-1)
        raw = raw[torch.isfinite(raw)]
        if raw.numel() == 0:
            return 0, 0.0, 0.0, 0.0, 0.0
        return (int(raw.numel()), float(raw.sum()),
                float((raw * raw).sum()), float(raw.min()), float(raw.max()))

    @torch.no_grad()
    def apply_syndrome_calibration(self, mean: float, std: float,
                                   target_std: float = 1.0) -> float:
        """Apply a CROSS-RANK calibrated gain/offset.

        ``mean`` and ``std`` must already describe the *global* batch (see
        ``codetrack/ddp_calibration.py``), not this rank's shard.
        """
        if not math.isfinite(target_std) or target_std <= 0:
            raise ValueError("target_std must be finite and positive")
        if not (math.isfinite(mean) and math.isfinite(std)) or std <= 1e-8:
            return float(self.syndrome_logit_gain)
        gain = float(min(max(target_std / std, 1.0), 1e3))
        self.syndrome_logit_offset.fill_(mean)
        self.syndrome_logit_gain.fill_(gain)
        self._syndrome_gain_calibrated = True
        return gain

'''
assert src.count(anchor) == 1, "anchor for new methods not unique"
src = src.replace(anchor, new_methods + anchor)

# ---- 2) forward: export instead of calibrating in-place -----------------------
old_fwd = """        if self.training and self.syndrome_gain_calibration and not self._syndrome_gain_calibrated:
            self.calibrate_syndrome_gain(s_raw)
"""
new_fwd = """        # Export the raw syndrome instead of calibrating in place.  The training loop
        # all-reduces the statistics across ranks and then calls
        # ``apply_syndrome_calibration`` once, outside the model.  Calibrating here
        # would make each rank adopt a DIFFERENT gain/offset (its own shard's std),
        # and the first DDP all-reduce would then mix four inconsistent parameter sets.
        _pending_calibration = (self.training and self.syndrome_gain_calibration
                                and not self._syndrome_gain_calibrated)
"""
assert src.count(old_fwd) == 1, "forward anchor not unique"
src = src.replace(old_fwd, new_fwd)

# ---- 3) expose it on the output dict -----------------------------------------
old_out = """        out = {"q": q, "q_logits": q_logits, "s": s, "s_logits": s_raw,
               "C_obs": C_obs, "C_ref": C_ref, "U": U, "R": R}"""
new_out = """        out = {"q": q, "q_logits": q_logits, "s": s, "s_logits": s_raw,
               "C_obs": C_obs, "C_ref": C_ref, "U": U, "R": R}
        if _pending_calibration:
            out["syndrome_pending_calibration"] = s_raw.detach()"""
assert src.count(old_out) == 1, "out-dict anchor not unique"
src = src.replace(old_out, new_out)

assert src != orig
p.with_suffix(".py.bak-ddp20261004").write_text(orig, encoding="utf-8")
p.write_text(src, encoding="utf-8")
print("PATCHED ecc.py")
print("md5 before:", hashlib.md5(orig.encode()).hexdigest())
print("md5 after :", hashlib.md5(src.encode()).hexdigest())
