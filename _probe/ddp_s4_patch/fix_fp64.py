"""Make syndrome_calibration_statistics accumulate in float64.

Idempotent: refuses to run twice.  Backs up first.
"""
import hashlib
import shutil
import sys
from datetime import datetime
from pathlib import Path

TARGET = Path("/home/yangjuanfeng/lab/projects/gola-CodeTrack/codetrack/ecc.py")

OLD = '''        raw = s_raw.detach().float().reshape(-1)
        raw = raw[torch.isfinite(raw)]
        if raw.numel() == 0:
            return 0, 0.0, 0.0, 0.0, 0.0
        return (int(raw.numel()), float(raw.sum()),
                float((raw * raw).sum()), float(raw.min()), float(raw.max()))'''

NEW = '''        # Accumulate in float64: the caller derives the variance from a single
        # pass (E[x^2] - E[x]^2) and the syndrome is strongly biased (|mean|/std
        # is ~11), so an fp32 reduction catastrophically cancels -- measured
        # relative error on std was 1.3e-5, versus 1e-15 for float64.
        raw32 = s_raw.detach().reshape(-1)
        finite = torch.isfinite(raw32)
        if not bool(finite.any()):
            return 0, 0.0, 0.0, 0.0, 0.0
        raw = raw32[finite].double()
        return (int(raw.numel()), float(raw.sum()),
                float((raw * raw).sum()), float(raw.min()), float(raw.max()))'''


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main() -> int:
    src = TARGET.read_text()
    if "Accumulate in float64" in src:
        print("ALREADY PATCHED")
        return 0
    if OLD not in src:
        print("ANCHOR NOT FOUND")
        print(src[src.find("def syndrome_calibration_statistics"):][:900])
        return 2
    if OLD in src.replace(NEW, ""):
        pass
    if src.count(OLD) != 1:
        print(f"ANCHOR COUNT {src.count(OLD)} != 1")
        return 2

    bak = TARGET.with_suffix(TARGET.suffix + f".bak-fp64fix-{datetime.now():%Y%m%d%H%M}")
    if not Path(str(TARGET) + ".bak-fp64").exists():
        shutil.copy2(TARGET, bak)
    before = md5(TARGET)
    TARGET.write_text(src.replace(OLD, NEW, 1))
    print(f"PATCHED  {before} -> {md5(TARGET)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
