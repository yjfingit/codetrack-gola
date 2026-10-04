import sys, torch
sys.path.insert(0, ".")
from codetrack.ecc import SyndromeDiagnosis

torch.manual_seed(20261004)
M, N, D = 64, 256, 128
diag = SyndromeDiagnosis(num_checks=M, num_variables=N, mid_dim=D,
                         syndrome_gain_calibration=True)
diag.train()
H = torch.rand(M, N); H = H / H.sum(-1, keepdim=True)
out = diag(torch.randn(2, N, 768), torch.randn(2, N, 768), H)
pend = out["syndrome_pending_calibration"].detach().float().reshape(-1)

n = pend.numel()
# route A -- torch.std
a = float(pend.std(unbiased=False))
# route B -- manual fp32, same order as torch
mu = float(pend.mean())
b = float(((pend - mu) ** 2).mean().sqrt())
# route C -- fp64 sum/sumsq (what ddp_calibration does)
r64 = pend.double()
m64 = float(r64.sum()) / n
v64 = float((r64 * r64).sum()) / n - m64 * m64
c = v64 ** 0.5
# route D -- fp64, two-pass (the numerically right way)
d = float(((r64 - m64) ** 2).mean().sqrt())

print(f"A torch.std(fp32)        = {a!r}")
print(f"B manual fp32 two-pass   = {b!r}")
print(f"C fp64 sum/sumsq one-pass = {c!r}")
print(f"D fp64 two-pass          = {d!r}")
print()
print(f"A vs B rel = {abs(a-b)/b:.3e}   (same-order fp32, should be ~0)")
print(f"A vs D rel = {abs(a-d)/d:.3e}   (fp32 vs exact)")
print(f"C vs D rel = {abs(c-d)/d:.3e}   <-- CANCELLATION error in one-pass")
print()
print("n =", n, " |mean| =", abs(m64), " std =", d)
print("catastrophic-cancellation index |mean|/std =", abs(m64)/d)
