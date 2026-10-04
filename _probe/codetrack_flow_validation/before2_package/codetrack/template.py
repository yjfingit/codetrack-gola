"""Block 6 -- online template protection.

Architecture figure:

    score_t ----------------------------+
    meanTopK(q) ------------------------+--> MLP --> c_t in [0,1]
    max(q) -----------------------------+                |
    motion uncertainty u_t -------------+                v
    recovery confidence r_rec ----------+     score_t > 0.84  AND  c_t > tau_c
                                                          |
                                              yes: Z_online^{t+1} = new crop
                                              no : Z_online^{t+1} = Z_online^t

The official GOLA updater replaces the online template whenever the tracking score
exceeds 0.84.  That is exactly the mechanism by which a corrupted frame's appearance
gets baked into the template and then contaminates every later frame -- the error
propagation the paper is about.  The gate adds the CodeTrack evidence (how bad the
diagnosis says this frame is, how uncertain the motion is, how confident the recovery
is) and blocks the update when the frame should not be trusted.

The gate is *learned*.  Its bias starts at **-1.0** (c ~ 0.27, i.e. conservative: do not
replace the template unless the evidence says the frame is trustworthy).  A permissive
start (bias +2 -> c ~ 0.88) suffers the same failure mode documented for the syndrome head
in ``docs/setup.md``: the sigmoid saturates, the gradient vanishes, and the gate spends the
run unlearning its own initialisation instead of learning the decision.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemplateProtectionGate(nn.Module):
    def __init__(self, hidden: int = 64, init_bias: float = -1.0):
        super().__init__()
        # inputs: score, meanTopK(q), max(q), motion uncertainty, recovery confidence
        self.mlp = nn.Sequential(
            nn.Linear(5, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )
        with torch.no_grad():
            self.mlp[-1].bias.fill_(float(init_bias))

    def forward(self, score: torch.Tensor, q: torch.Tensor,
                uncertainty: Optional[torch.Tensor] = None,
                recovery_confidence: Optional[torch.Tensor] = None,
                topk: int = 32) -> Dict[str, torch.Tensor]:
        """``score`` (B,); ``q`` is either

        * ``(B, N)``  the full token-level error severity -- then ``meanTopK(q)`` and
          ``max(q)`` are computed here, exactly as the figure's "更新可信度计算" panel shows; or
        * ``(B,)``    an already reduced corruption summary (what the differentiable training
          path passes, so the gate is reachable from the loss).

        Returns ``c_t`` (B,) and the two summary statistics used.
        """
        b = score.shape[0]
        q = q.reshape(b, -1)
        if q.shape[1] == 1:
            # already a reduced summary: use it for both statistics
            mean_topk = q[:, 0]
            max_q = q[:, 0]
        else:
            k = min(int(topk), q.shape[1])
            mean_topk = torch.topk(q, k=k, dim=-1).values.mean(dim=-1)   # (B,)
            max_q = q.max(dim=-1).values
        if uncertainty is None:
            uncertainty = q.new_zeros(b)
        if recovery_confidence is None:
            recovery_confidence = q.new_zeros(b)
        feats = torch.stack([score.to(q.dtype).reshape(b),
                             mean_topk, max_q,
                             uncertainty.to(q.dtype).reshape(b),
                             recovery_confidence.to(q.dtype).reshape(b)], dim=-1)
        c_t = torch.sigmoid(self.mlp(feats)).squeeze(-1)             # (B,)
        return {"c_t": c_t, "mean_topk_q": mean_topk, "max_q": max_q}

    @staticmethod
    def should_update(c_t: torch.Tensor, score: torch.Tensor,
                      score_threshold: float = 0.84,
                      c_threshold: float = 0.5) -> torch.Tensor:
        """The figure's rule, evaluated as a differentiable-free boolean."""
        return (score > score_threshold) & (c_t > c_threshold)
