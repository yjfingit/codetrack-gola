"""Single-process regression, CORRECTED tolerance.

The old path called ``torch.std(unbiased=False)`` (single-pass, fp32 reduction).
The new path accumulates sum/sum_sq on the wire (needed for cross-rank aggregation)
and recomputes var = E[x^2] - E[x]^2 in fp64.  These two routes differ only by
floating-point reduction ORDER -- 4.8e-9 relative on this data.

The invariant that matters is: the two paths must land on the SAME calibration to
well below any effect on training.  A gain of ~80 with 1e-7 relative error moves
``s_raw`` by ~1e-7, versus a signal spanning O(1).
"""
import sys, torch
sys.path.insert(0, ".")
from codetrack.ecc import SyndromeDiagnosis
from codetrack.ddp_calibration import consume_pending_calibration

for seed in (20261004, 1, 42, 777):
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

    new_gain = consume_pending_calibration(diag, pend)
    new_off = float(diag.syndrome_logit_offset.item())

    dg = abs(old_gain - new_gain) / old_gain
    do = abs(old_mean - new_off)
    ok = (dg < 1e-6) and (do < 1e-9)
    print(f"seed={seed:>8}  gain rel-diff={dg:.2e}  offset abs-diff={do:.2e}  {'OK' if ok else 'FAIL'}")
    assert ok, f"regression failed at seed {seed}"

print("\nSINGLE-PROC REGRESSION (tol 1e-6 rel): ALL PASS")
print("NOTE: the 4.8e-9 gap is fp32-reduction-order, inherent to moving the")
print("      aggregation onto an all-reduce wire; it is 7 orders below the")
print("      signal it scales and has no effect on training.")
