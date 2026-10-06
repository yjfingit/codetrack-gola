#!/usr/bin/env python3
"""Monitor a CodeTrack training log, retain the best epoch, and stop on a plateau."""
import argparse
import os
import re
import shutil
import signal
import time
from pathlib import Path


METRIC_RE = re.compile(r"(?:^|\s)(Loss/track_corr|Error/q_auroc_mask|Error/gain_total)\s*:?[ \t]+([-+0-9.eE]+)")
EPOCH_RE = re.compile(r"Epoch:\s*\[(\d+)\]\s+summary metrics:")


def checkpoint_model(root: Path):
    candidates = [p for p in root.rglob("model.bin") if p.parent.name.startswith("epoch_")]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime)
    src = candidates[-1]
    best = root / "best" / "model.bin"
    best.parent.mkdir(parents=True, exist_ok=True)
    tmp = best.with_suffix(".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, best)
    shutil.copy2(src, root / "best" / "model.safetensors")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    args = ap.parse_args()
    log_path = Path(args.log)
    output = Path(args.output)
    pos = 0
    current_epoch = None
    metrics = {}
    best_score = float("-inf")
    bad_epochs = 0
    seen_epochs = set()
    pending_best = False
    while True:
        if log_path.exists():
            with log_path.open("r", errors="replace") as f:
                f.seek(pos)
                lines = f.readlines()
                pos = f.tell()
            for line in lines:
                # A non-finite gradient or loss invalidates all later weights.  Stop at the
                # first such line and leave the last finite best checkpoint untouched.
                # ``grad_norm`` is intentionally blank/NaN on accumulation micro-steps before
                # an optimizer update. Only a non-finite scalar loss is an immediate failure.
                if re.search(r"loss:\s*(?:nan|inf)\b", line, re.IGNORECASE):
                    print(f"[early-stop] non-finite training signal; stopping pid={args.pid}", flush=True)
                    try:
                        os.killpg(os.getpgid(args.pid), signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    return
                m = EPOCH_RE.search(line)
                if m:
                    current_epoch = int(m.group(1))
                    metrics = {}
                    continue
                m = METRIC_RE.match(line)
                if m and current_epoch is not None:
                    metrics[m.group(1)] = float(m.group(2))
                if current_epoch is not None and len(metrics) == 3 and current_epoch not in seen_epochs:
                    seen_epochs.add(current_epoch)
                    # Lower tracking loss is better; higher diagnosis AUROC and recovery gain help.
                    score = (-metrics["Loss/track_corr"] +
                             0.50 * metrics["Error/q_auroc_mask"] +
                             5.0 * metrics["Error/gain_total"])
                    if score > best_score + args.min_delta:
                        best_score = score
                        bad_epochs = 0
                        best = checkpoint_model(output)
                        pending_best = best is None
                        print(f"[early-stop] epoch={current_epoch} score={score:.6f} best={best}", flush=True)
                    else:
                        bad_epochs += 1
                        print(f"[early-stop] epoch={current_epoch} score={score:.6f} bad={bad_epochs}/{args.patience}", flush=True)
                    if bad_epochs >= args.patience:
                        print(f"[early-stop] stopping pid={args.pid} after epoch {current_epoch}", flush=True)
                        try:
                            os.killpg(os.getpgid(args.pid), signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        return
            if pending_best:
                best = checkpoint_model(output)
                if best is not None:
                    pending_best = False
                    print(f"[early-stop] checkpoint became available: {best}", flush=True)
        try:
            os.kill(args.pid, 0)
        except ProcessLookupError:
            # Make sure a final checkpoint is available if training ended naturally.
            if not (output / "best" / "model.bin").exists():
                checkpoint_model(output)
            return
        time.sleep(2.0)


if __name__ == "__main__":
    main()
