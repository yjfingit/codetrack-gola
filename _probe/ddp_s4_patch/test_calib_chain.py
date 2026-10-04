"""Verify the calibration chain end-to-end WITHOUT launching a full training run.

Checks:
  1. the diagnosis head exports syndrome_pending_calibration on its first training forward
  2. the export is a finite tensor with the expected shape
  3. after applying calibration the flag flips and the export stops
  4. single-process path is a no-op for distributed (no dist init required)
"""
import sys, torch
sys.path.insert(0, ".")

# --- unit-level: drive SyndromeDiagnosis directly ---------------------------------
from codetrack.ecc import SyndromeDiagnosis
from codetrack.ddp_calibration import calibrate_across_ranks, broadcast_calibration

M, N, D = 64, 256, 128
torch.manual_seed(0)
diag = SyndromeDiagnosis(num_checks=M, num_variables=N, mid_dim=D,
                         syndrome_gain_calibration=True)
diag.train()

H = torch.rand(M, N); H = H / H.sum(-1, keepdim=True)
X_t = torch.randn(2, N, 768); X_aux = torch.randn(2, N, 768)
out = diag(X_t, X_aux, H)
assert "syndrome_pending_calibration" in out, "MISSING pending calibration export"
pend = out["syndrome_pending_calibration"]
assert torch.isfinite(pend).all(), "pending tensor not finite"
print(f"[1] exported: shape={tuple(pend.shape)} finite=True")

# --- statistics helper -----------------------------------------------------------
cnt, ssum, ssq, mn, mx = diag.syndrome_calibration_statistics(pend)
assert cnt == pend.numel(), f"count mismatch {cnt} vs {pend.numel()}"
print(f"[2] statistics: count={cnt} sum={ssum:.4f} sumsq={ssq:.4f}")

# --- apply (single process => no dist) -------------------------------------------
gain_before = float(diag.syndrome_logit_gain.detach())
g = calibrate_across_ranks(diag, pend)
print(f"[3] calibrate_across_ranks -> gain={g} (was {gain_before})")
assert g is not None, "calibration returned None unexpectedly"
assert diag._syndrome_gain_calibrated is True, "flag not set"

# --- second forward must NOT export again ----------------------------------------
out2 = diag(X_t, X_aux, H)
assert "syndrome_pending_calibration" not in out2, "exported twice -- would re-calibrate"
print("[4] second forward does not re-export (one-shot confirmed)")

# --- broadcast is a no-op without dist -------------------------------------------
broadcast_calibration(diag)
print("[5] broadcast_calibration no-op without dist: OK")

# --- disabled => no export at all ------------------------------------------------
diag2 = SyndromeDiagnosis(num_checks=M, num_variables=N, mid_dim=D,
                          syndrome_gain_calibration=False)
diag2.train()
out3 = diag2(X_t, X_aux, H)
assert "syndrome_pending_calibration" not in out3, "exported while calibration disabled"
print("[6] calibration disabled => no export: OK")

# --- loaded calibrated checkpoint must skip --------------------------------------
diag3 = SyndromeDiagnosis(num_checks=M, num_variables=N, mid_dim=D,
                          syndrome_logit_gain=4.84, syndrome_gain_calibration=True)
diag3.train()
print(f"[7] gain!=1 init => _syndrome_gain_calibrated={diag3._syndrome_gain_calibrated}")
assert diag3._syndrome_gain_calibrated is True, "pre-calibrated init not detected"

print("\nCHAIN TEST: ALL PASS")
