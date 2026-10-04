"""Locate the tiny gain discrepancy between the old in-forward path and the new one."""
import sys, torch
sys.path.insert(0, ".")
from codetrack.ecc import SyndromeDiagnosis

torch.manual_seed(20261004)
M, N, D = 64, 256, 128
diag = SyndromeDiagnosis(num_checks=M, num_variables=N, mid_dim=D,
                         syndrome_gain_calibration=True)
diag.train()
H = torch.rand(M, N); H = H / H.sum(-1, keepdim=True)
X_t = torch.randn(2, N, 768); X_aux = torch.randn(2, N, 768)
out = diag(X_t, X_aux, H)
pend = out["syndrome_pending_calibration"]

print("dtype of exported tensor:", pend.dtype)

# OLD path: float32 std on the raw tensor
r32 = pend.detach().float().reshape(-1)
std32 = float(r32.std(unbiased=False))
C32 = 1.0 / std32

# collect-then-recompute path used by the statistics helper
r64 = pend.detach().float().reshape(-1).double()
n = r64.numel()
mean64 = float(r64.sum()) / n
var64 = float((r64 * r64).sum()) / n - mean64 * mean64
std64 = var64 ** 0.5
C64 = 1.0 / std64

print(f"OLD  std(fp32 reduction) = {std32!r}  -> gain {C32!r}")
print(f"NEW  std(fp64 sum/sumsq) = {std64!r}  -> gain {C64!r}")
print(f"rel diff in std : {abs(std32-std64)/std64:.3e}")
print(f"rel diff in gain: {abs(C32-C64)/C64:.3e}")
print()
print("=> gap is float32-accumulation error in torch.std, not a logic bug."
      if abs(C32-C64)/C64 < 1e-5 else "=> real logic difference, investigate.")
