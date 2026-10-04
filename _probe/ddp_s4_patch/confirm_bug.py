"""Prove the fp32-accumulation defect in syndrome_calibration_statistics.

Compares, on the SAME raw syndrome tensor:
  * the code's route  : raw.sum() / (raw*raw).sum()  (fp32 accumulate, then cast)
  * the fp64 route    : raw64.sum() / (raw64*raw64).sum()
  * the exact route   : fp64 two-pass variance (reference)

Prints the resulting std and the relative error of each route.
"""
import math
import sys

import torch

sys.path.insert(0, ".")
from codetrack.ecc import SyndromeDiagnosis  # noqa: E402


def exact_std(raw64):
    m = raw64.mean()
    d = raw64 - m
    return float(torch.sqrt((d * d).mean()))


def main():
    torch.manual_seed(0)
    worst = 0.0
    for seed in (0, 1, 12345, 20261004, 99991, 7):
        torch.manual_seed(seed)
        # mimic a real calibration batch: 2 sequences x 64 syndrome channels
        s_raw = torch.randn(2, 64) * 0.0124 + 0.1361
        mod = SyndromeDiagnosis(num_variables=64, num_checks=32, dim=16)

        cnt, ssum, ssq, mn, mx = mod.syndrome_calibration_statistics(s_raw)

        r64 = s_raw.detach().float().reshape(-1).double()
        m64 = float(r64.mean())
        var64 = max(float((r64 * r64).mean()) - m64 * m64, 0.0)
        std64 = math.sqrt(var64)
        std_exact = exact_std(r64)

        var_code = max(ssq / cnt - (ssum / cnt) ** 2, 0.0)
        std_code = math.sqrt(var_code)

        e_code = abs(std_code - std_exact) / std_exact
        e_64 = abs(std64 - std_exact) / std_exact
        worst = max(worst, e_code)

        print(f"seed={seed:<9d} |mean|/std={abs(ssum/cnt)/max(std_exact,1e-12):7.3f}  "
              f"std_code={std_code:.17g}  std_64={std64:.17g}  std_exact={std_exact:.17g}  "
              f"relerr_code={e_code:.3e}  relerr_64={e_64:.3e}")

    print()
    print(f"worst fp32-accumulate relative error: {worst:.3e}")
    if worst > 1e-9:
        print("VERDICT: BUG CONFIRMED -- fp32 accumulation in syndrome_calibration_statistics")
        return 1
    print("VERDICT: no meaningful fp32 penalty at this scale")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
