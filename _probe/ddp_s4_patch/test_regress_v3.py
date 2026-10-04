"""Single-process regression, CORRECT tolerance and CORRECT reference.

Root cause of the earlier "FAIL": my tolerance assertion compared the two routes to
each other with tol 1e-9, but they legitimately differ by fp32-vs-fp64 rounding.

Measured on real data (n=128, |mean|/std ~ 11):
    old route  torch.std(fp32)     std = 0.012417505495250225   rel err 4.77e-09 vs exact
    new route  fp64 sum/sum_sq     std = 0.01241750543599846    rel err 8.38e-16 vs exact
    exact      fp64 two-pass       std = 0.01241750543599845

The new route is CLOSER to the exact value than the one it replaces: aggregating in
fp64 on the wire removes the fp32 rounding that the old in-forward call carried.  So
there is no regression -- there is a small accuracy improvement.

This test asserts the invariant that actually matters: both routes agree far below
any effect on training, and the new route is not worse than the old one.
"""
import sys, torch
sys.path.insert(0, ".")
from codetrack.ecc import SyndromeDiagnosis
from codetrack.ddp_calibration import consume_pending_calibration

worst_gain, worst_off = 0.0, 0.0
for seed in (20261004, 1, 42, 777, 12345, 99):
    torch.manual_seed(seed)
    M, N, D = 64, 256, 128
    diag = SyndromeDiagnosis(num_checks=M, num_variables=N, mid_dim=D,
                             syndrome_gain_calibration=True)
    diag.train()
    H = torch.rand(M, N); H = H / H.sum(-1, keepdim=True)
    out = diag(torch.randn(2, N, 768), torch.randn(2, N, 768), H)
    pend = out["syndrome_pending_calibration"]
    r = pend.detach().float().reshape(-1)

    old_std = float(r.std(unbiased=False)); old_mean = float(r.mean())
    old_gain = min(max(1.0 / old_std, 1.0), 1e3)

    r64 = r.double(); n = r64.numel()
    m64 = float(r64.sum()) / n
    exact_std = float(((r64 - m64) ** 2).mean().sqrt())
    exact_gain = min(max(1.0 / exact_std, 1.0), 1e3)

    new_gain = consume_pending_calibration(diag, pend)
    new_off = float(diag.syndrome_logit_offset.item())

    err_old = abs(old_gain - exact_gain) / exact_gain
    err_new = abs(new_gain - exact_gain) / exact_gain
    dg = abs(old_gain - new_gain) / old_gain
    do = abs(old_mean - new_off)
    worst_gain = max(worst_gain, dg); worst_off = max(worst_off, do)

    print(f"seed={seed:>8}  |old-exact|={err_old:.2e}  |new-exact|={err_new:.2e}  "
          f"|old-new|={dg:.2e}  off={do:.2e}")
    assert err_new <= err_old * 1.001, f"new route LESS accurate at seed {seed}"
    assert dg < 1e-6, f"routes disagree too much at seed {seed}: {dg}"

print(f"\nworst |old-new|: gain {worst_gain:.2e} (rel), offset {worst_off:.2e} (abs)")
print("SINGLE-PROC REGRESSION: ALL PASS")
print("The new route is at least as accurate as the old one on every seed,")
print("and the two agree to <1e-6 relative -- orders below any training effect.")
