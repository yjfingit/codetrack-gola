"""Stage metrics: log parsing, JSONL persistence, and the objective numbers a stage is judged on.

Why a separate module
---------------------
The training harness writes a human-readable log line per iteration.  A staged pipeline needs to
turn those lines into machine-checkable numbers so that ``validate_stageN`` can decide
GO / WARN / NO-GO.  Keeping the parsing in one place means the decision logic and the report
cannot disagree about what a metric was.

Two rules that the earlier rounds got wrong and this module enforces:

* Every distance is ``1 - cosine_similarity``.  The criterion, the preflight gate and the
  ablation report all use the same definition; a Euclidean "relative distance" reads ~0 change
  whenever the token norm is preserved, which is exactly what happens here.
* The diagnosis metric is ``AUROC(q, corruption_mask)`` (and AUPRC), never a hard-coded
  threshold on the soft ``error_target``: measured target values are ~0.14 on untouched tokens
  and ~0.22 on damaged ones, so a threshold at 0.25 has almost no positives left.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

# ``name: value (ema)`` -- the harness prints a single-batch value and, in parentheses, the
# running window average.  We take the single-batch value for the milestone tables and the EMA
# only when explicitly asked, so a stage report never silently mixes the two.
_SCALAR = re.compile(r"(?<![\w/])([A-Za-z][\w./]*): (-?[\d.]+(?:e[-+]?\d+)?|nan|inf)(?: \((-?[\d.]+(?:e[-+]?\d+)?|nan|inf)\))?")
_ITER = re.compile(r"Epoch: \[(\d+)\] \[ *(\d+)/\d+\]")


class StageLog:
    """Parsed view of one stage's training log."""

    def __init__(self, path: str):
        self.path = path
        self.rows: List[dict] = []

    def parse(self) -> "StageLog":
        with open(self.path, errors="replace") as fh:
            for line in fh:
                m = _ITER.search(line)
                if not m:
                    continue
                row: Dict[str, object] = {"epoch": int(m.group(1)), "micro_step": int(m.group(2))}
                for key, val, ema in _SCALAR.findall(line):
                    try:
                        row[key] = float(val)
                    except ValueError:
                        continue
                    if ema:
                        try:
                            row[key + "@ema"] = float(ema)
                        except ValueError:
                            pass
                self.rows.append(row)
        return self

    # ---------------------------------------------------------------- accessors
    @property
    def updates_logged(self) -> int:
        """Optimizer updates seen so far.

        The harness logs every micro-step, and one optimizer update consumes
        ``grad_accumulation_steps`` micro-steps.  The count is therefore derived from the number
        of logged iterations, not from the raw iteration index (which restarts per epoch).
        """
        return len(self.rows) // 16

    def series(self, key: str, ema: bool = False) -> List[float]:
        k = key + "@ema" if ema else key
        return [r[k] for r in self.rows if k in r and isinstance(r[k], float)]

    def first_last(self, key: str, ema: bool = False):
        s = self.series(key, ema)
        if not s:
            return None, None
        return s[0], s[-1]

    def at_fraction(self, key: str, frac: float, ema: bool = True) -> Optional[float]:
        s = self.series(key, ema) or self.series(key, False)
        if not s:
            return None
        idx = min(len(s) - 1, max(0, int(round(frac * (len(s) - 1)))))
        return s[idx]

    def peak(self, key: str) -> Optional[float]:
        s = self.series(key)
        return max(s) if s else None

    def has_any_nan(self) -> bool:
        for r in self.rows:
            for k in ("loss", "grad_norm"):
                v = r.get(k)
                if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
                    return True
        return False

    def learning_rates(self) -> Dict[str, float]:
        """First and last observed ``lr``: the cosine horizon must actually be traversed."""
        s = self.series("lr")
        if not s:
            return {}
        return {"lr_first": s[0], "lr_last": s[-1],
                "lr_min_seen": min(s), "lr_max_seen": max(s)}


def write_jsonl(path: str, records: List[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def parameter_deltas(before: Dict[str, "object"], after: Dict[str, "object"],
                     groups: Dict[str, str]) -> Dict[str, float]:
    """``max|delta|`` per parameter group between two state dicts.

    ``groups`` maps a group name to a *name prefix*; the result reports, for each group, the
    largest absolute change across its parameters.  ``0.0`` means "this group did not move",
    which is the assertion a frozen group must satisfy.
    """
    import torch

    out: Dict[str, float] = {}
    for gname, prefix in groups.items():
        best = 0.0
        for name, before_t in before.items():
            if not name.startswith(prefix):
                continue
            after_t = after.get(name)
            if after_t is None or not torch.is_tensor(before_t) or not torch.is_tensor(after_t):
                continue
            if before_t.shape != after_t.shape:
                continue
            d = float((after_t.detach().float() - before_t.detach().float()).abs().max())
            best = max(best, d)
        out[gname] = best
    return out
