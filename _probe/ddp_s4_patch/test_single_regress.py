"""Single-process regression: with nproc=1 the patched path must reproduce the
calibration the OLD in-forward path produced.  We drive the real S1 config."""
import sys, json, torch
sys.path.insert(0, ".")
from codetrack.ecc import SyndromeDiagnosis
from codetrack.ddp_calibration import consume_pending_calibration

torch.manual_seed(20261004)
M, N, D = 64, 256, 128
diag = SyndromeDiagnosis(num_checks=M, num_variables=N, mid_dim=D,
                         syndrome_gain_calibration=True)
diag.train()
H = torch.rand(M, N); H = H / H.sum(-1, keepdim=True)

# --- what the OLD implementation would have computed ---------------------------
s_raw_snapshot = None
X_t = torch.randn(2, N, 768); X_aux = torch.randn(2, N, 768)
out = diag(X_t, X_aux, H)
pend = out["syndrome_pending_calibration"]

# OLD semantics: gain = clamp(1/std, 1, 1e3), offset = mean  (single shard)
old_raw = pend.detach().float().reshape(-1)
old_std = float(old_raw.std(unbiased=False)); old_mean = float(old_raw.mean())
old_gain = min(max(1.0 / old_std, 1.0), 1e3)
print(f"OLD (in-forward, local) gain={old_gain:.6f} offset={old_mean:.6f}")

# --- NEW implementation, single process ----------------------------------------
new_gain = consume_pending_calibration(diag, pend)
print(f"NEW (loop-side, 1 proc) gain={new_gain:.6f} offset={float(diag.syndrome_logit_offset.item()):.6f}")

same_gain = abs(old_gain - new_gain) < 1e-9
same_off = abs(old_mean - float(diag.syndrome_logit_offset.item())) < 1e-9
print(f"\ngain  bit-identical: {same_gain}")
print(f"offset bit-identical: {same_off}")
print("SINGLE-PROC REGRESSION: " + ("PASS" if (same_gain and same_off) else "FAIL"))
