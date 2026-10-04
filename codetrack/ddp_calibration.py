"""Cross-rank calibration of the syndrome gain/offset.

Called ONCE at the start of training, BEFORE the first optimizer step, from the
training loop -- never from inside ``forward``.  Doing the all-reduce inside the
forward pass is unsafe: by then the DDP reducer buckets are already built, and a
collective that is not matched by an identically-ordered collective on every rank
hangs the job.

``nproc_per_node=1`` is a no-op fast path, so the single-GPU behaviour is
bit-for-bit what it was before.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.distributed as dist


def _dist_ready() -> bool:
    return bool(dist.is_available() and dist.is_initialized())


def _all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    if _dist_ready():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def calibrate_across_ranks(diagnosis_module,
                           s_raw: torch.Tensor,
                           target_std: float = 1.0) -> Optional[float]:
    """Aggregate (count, sum, sum_sq) over all ranks, then apply once.

    Returns the applied gain, or ``None`` when calibration was skipped
    (disabled, already calibrated, or no finite statistics).
    """
    if not bool(getattr(diagnosis_module, "syndrome_gain_calibration", False)):
        return None
    if bool(getattr(diagnosis_module, "_syndrome_gain_calibrated", False)):
        return None

    count, ssum, ssq, _mn, _mx = diagnosis_module.syndrome_calibration_statistics(s_raw)
    stats = torch.tensor([float(count), float(ssum), float(ssq)],
                         dtype=torch.float64, device=s_raw.device)
    stats = _all_reduce_sum(stats)

    total = float(stats[0])
    if total <= 0.0:
        return None
    mean = float(stats[1]) / total
    var = max(float(stats[2]) / total - mean * mean, 0.0)
    std = math.sqrt(var)

    gain = diagnosis_module.apply_syndrome_calibration(mean, std, target_std=target_std)
    return gain


def broadcast_calibration(diagnosis_module) -> None:
    """Belt-and-braces: after calibration force every rank to hold identical
    gain/offset.  The all-reduce above already makes the statistics identical,
    but each rank owns its own ``nn.Parameter`` objects."""
    if not _dist_ready():
        return
    with torch.no_grad():
        dist.broadcast(diagnosis_module.syndrome_logit_gain, src=0)
        dist.broadcast(diagnosis_module.syndrome_logit_offset, src=0)


def consume_pending_calibration(diagnosis_module, s_raw: torch.Tensor,
                                target_std: float = 1.0) -> Optional[float]:
    """Convenience wrapper used by the training loop: calibrate + broadcast."""
    gain = calibrate_across_ranks(diagnosis_module, s_raw, target_std=target_std)
    if gain is not None:
        broadcast_calibration(diagnosis_module)
    return gain
