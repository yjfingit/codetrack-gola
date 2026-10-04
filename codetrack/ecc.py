"""Block 3 / 3.1 -- ECC-inspired structured error diagnosis.

Architecture figure:

    X_t   (B, 256, 768)  search tokens, TIR position   (baseline GOLA feature)
    X_aux (B, 256, 768)  search tokens, RGB position   (missing-modality evidence)
        |
        |   W_x : 768 -> 128           W_r : 768 -> 128
        v
    U (B, 256, 128)                R (B, 256, 128) + TemplateContext
        |                               |
        +---------- H_bar in R^{M x N} -+
        v                               v
    C_obs = H U  (B, 64, 128)      C_ref = H R  (B, 64, 128)
        \\____________  ____________/
                      \\/
        s_j = MLP([ dC_j , |dC_j| , cos(C_obs_j, C_ref_j) ])   -> s in [0,1]^64
                      |
                      v
        q = sigmoid( H_bar^T s + b )                            -> q in [0,1]^256

This is **LDPC-inspired structured redundancy checking**, not a real GF(2) code:
vision features do not satisfy ``H x = 0``.  We therefore *define* the syndrome as the
discrepancy between the observation aggregated through ``H`` and a reliability
reference aggregated through the *same* ``H``, exactly as the figure shows.

``H`` has a **fixed sparse support and learnable edge weights** (GAT-style):

    H_bar[j, i] = H0[j, i] * softplus(w[j, i]) / sum_k H0[j, k] * softplus(w[j, k])

The support is geometrically constructed (each check watches a local window of the
16x16 token grid, plus a share of free edges), and column degrees are repaired to the
requested minimum so every token is watched by at least ``h_min_col_degree`` checks.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def sparsified_softmax(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Softmax keeping only the ``top_k`` entries per row."""
    if top_k >= logits.shape[-1]:
        return logits.softmax(dim=-1)
    kth = torch.topk(logits, k=top_k, dim=-1).values[..., -1:]
    masked = logits.masked_fill(logits < kth, float("-inf"))
    return masked.softmax(dim=-1)


def build_sparse_support(num_checks: int, num_variables: int,
                         links_per_check: int, min_col_degree: int,
                         grid: int, locality_window: int = 0,
                         locality_wrap: bool = True,
                         free_edge_frac: float = 0.25,
                         seed: int = 1234) -> torch.Tensor:
    """Geometrically construct the fixed ``M x N`` parity-check support.

    Each check draws most of its edges from a 2-D window on the token grid (a
    locality prior: a parity check that watches one region is what makes a syndrome
    an *address*).  Column degrees below ``min_col_degree`` are repaired by swapping
    edges in, which keeps every row's degree exactly ``links_per_check``.
    """
    g = torch.Generator().manual_seed(int(seed))
    support = torch.zeros(num_checks, num_variables)
    grid = int(grid)
    spatial = grid * grid == num_variables

    if spatial and locality_window > 0:
        yy, xx = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
        coords = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1).float()
        half = max(1, int(locality_window) // 2)
        n_free = int(round(free_edge_frac * links_per_check))
        inside_mask = torch.zeros(num_checks, num_variables, dtype=torch.bool)
        for row in range(num_checks):
            cy = int(torch.randint(0, grid, (1,), generator=g))
            cx = int(torch.randint(0, grid, (1,), generator=g))
            dy = (coords[:, 0] - cy).abs()
            dx = (coords[:, 1] - cx).abs()
            if locality_wrap:
                dy = torch.minimum(dy, grid - dy)
                dx = torch.minimum(dx, grid - dx)
            inside = torch.nonzero((dy <= half) & (dx <= half), as_tuple=False).flatten()
            inside_mask[row] = False
            inside_mask[row, inside] = True
            n_local = links_per_check - n_free
            pool = inside if inside.numel() >= n_local else torch.arange(num_variables)
            picked = pool[torch.randperm(pool.numel(), generator=g)[:n_local]]
            support[row, picked] = 1.0
            free = torch.nonzero(support[row] == 0).flatten()
            if n_free > 0 and free.numel() > 0:
                picked = free[torch.randperm(free.numel(), generator=g)[:n_free]]
                support[row, picked] = 1.0
    else:
        for row in range(num_checks):
            picked = torch.randperm(num_variables, generator=g)[:links_per_check]
            support[row, picked] = 1.0

    # ---- repair column degrees, preserving row degrees -----------------------
    col_degree = support.sum(dim=0)
    if support.sum() < min_col_degree * num_variables:
        raise ValueError(
            f"support too sparse for min_col_degree={min_col_degree}: "
            f"{int(support.sum())} edges < {min_col_degree * num_variables}")
    for col in torch.nonzero(col_degree < min_col_degree, as_tuple=False).flatten().tolist():
        guard = 0
        while int(col_degree[col]) < min_col_degree and guard < 1000:
            guard += 1
            rows = torch.nonzero(support[:, col] == 0, as_tuple=False).flatten()
            rows = rows[torch.randperm(rows.numel(), generator=g)]
            for row_t in rows:
                row = int(row_t)
                donors = torch.nonzero(
                    support[row].bool() & (col_degree > min_col_degree), as_tuple=False).flatten()
                if donors.numel() == 0:
                    continue
                donor = int(donors[torch.randint(donors.numel(), (1,), generator=g)])
                support[row, donor] = 0.0
                support[row, col] = 1.0
                col_degree[donor] -= 1.0
                col_degree[col] += 1.0
                break
            else:
                break
    return support


class ParityCheckMatrix(nn.Module):
    """``H_bar``: fixed sparse support, learnable non-negative edge weights."""

    def __init__(self, num_checks: int, num_variables: int, links_per_check: int = 12,
                 min_col_degree: int = 3, grid: int = 16, locality_window: int = 0,
                 locality_wrap: bool = True, free_edge_frac: float = 0.25,
                 seed: int = 1234, learn_weights: bool = True):
        super().__init__()
        support = build_sparse_support(
            num_checks, num_variables, links_per_check, min_col_degree,
            grid, locality_window, locality_wrap, free_edge_frac, seed)
        # ``persistent=False``: upstream ``GOLA_DINOv2.state_dict`` calls
        # ``get_parameter`` on every key, which raises for an unknown buffer.  The support
        # is a deterministic function of (M, N, d_c, d_v, grid, seed), so it is rebuilt
        # identically on load instead of being written into the checkpoint.
        self.register_buffer("H_support", support, persistent=False)
        # edge logits, initialised so softplus(w)=1 -> all edges start equal
        init = torch.full_like(support, math.log(math.e - 1.0))
        self.H = nn.Parameter(init, requires_grad=bool(learn_weights))
        self.learn_weights = bool(learn_weights)
        self.num_checks = num_checks
        self.num_variables = num_variables

    def matrix(self) -> torch.Tensor:
        """Row-normalised non-negative ``M x N`` matrix."""
        if self.learn_weights:
            w = F.softplus(self.H) * self.H_support
        else:
            w = self.H_support
        return w / w.sum(dim=1, keepdim=True).clamp(min=1e-6)

    def connectivity(self) -> torch.Tensor:
        return self.H_support

    def neighbour_index(self, num_neighbours: int) -> torch.Tensor:
        """``N x K_n`` index of each variable's most-related variables.

        Relationship is the shared-incidence sparsity pattern of ``H^T H`` (with the
        diagonal removed): "which other tokens share redundancy checks with me".
        """
        with torch.no_grad():
            h = self.H_support
            a = h.t() @ h                                   # (N, N) shared-check counts
            a.fill_diagonal_(0.0)
            k = min(int(num_neighbours), a.shape[1] - 1)
            idx = torch.topk(a, k=k, dim=-1).indices       # (N, k)
        return idx


class SyndromeDiagnosis(nn.Module):
    """Blocks 3 and 3.1: ``X_t`` / ``X_aux`` -> ``s`` -> ``q``."""

    def __init__(self, dim: int = 768, mid_dim: int = 128, num_checks: int = 64,
                 num_variables: int = 256, syndrome_hidden: int = 128,
                 template_dim: Optional[int] = None,
                 support: Optional[torch.Tensor] = None,
                 detection_prior: float = 0.2,
                 use_cos: bool = True):
        super().__init__()
        self.dim = dim
        self.mid_dim = mid_dim
        self.num_checks = num_checks
        self.num_variables = num_variables
        self.use_cos = bool(use_cos)

        self.W_x = nn.Linear(dim, mid_dim)
        self.W_r = nn.Linear(dim, mid_dim)
        # TemplateContext: initial + online template are pooled and broadcast
        self.template_ctx = nn.Linear(dim, mid_dim)

        # Block 3.1 computes three quantities with *different* shapes:
        #   delta  = C_obs - C_ref     (B, M, mid)   per-channel discrepancy
        #   |delta|                    (B, M, mid)
        #   cos(C_obs, C_ref)          (B, M)        one scalar per check
        # so they cannot be concatenated directly.  ``delta``/``|delta|`` go through the
        # per-channel MLP, and the per-check cosine is projected on its own and added,
        # which is the shape-safe form of the figure's "[delta, |delta|, cos] -> MLP".
        self.mlp_s = nn.Sequential(
            nn.Linear(mid_dim * 2, syndrome_hidden), nn.GELU(),
            nn.Linear(syndrome_hidden, 1),
        )
        self.cos_proj = nn.Sequential(
            nn.Linear(1, syndrome_hidden), nn.GELU(), nn.Linear(syndrome_hidden, 1),
        ) if self.use_cos else None
        # explicit scale on the raw discrepancy, so the head does not have to learn it.
        # NOTE: deliberately shape (1,) rather than a 0-d scalar.  The training framework's
        # gradient-norm helper calls ``torch.linalg.norm(p.grad, ord=2.0)`` with no ``dim``,
        # which raises "input must be 1D or 2D. Got 0D" for a 0-d parameter.  Keeping one
        # dimension avoids that without changing the maths.
        self.residual_scale = nn.Parameter(torch.ones(1))
        self.check_scale = nn.Parameter(torch.ones(num_checks))

        # q = sigmoid(H^T s + b)
        self.vote_bias = nn.Parameter(torch.zeros(num_variables))
        # biased start at the expected corruption density (a 0 bias starts at 0.5,
        # whose soft-BCE is ln2 and fights the target)
        prior = min(max(float(detection_prior), 1e-3), 1 - 1e-3)
        with torch.no_grad():
            self.vote_bias.fill_(float(torch.logit(torch.tensor(prior))))

    def forward(self, X_t: torch.Tensor, X_aux: torch.Tensor,
                H_bar: torch.Tensor,
                template_context: Optional[torch.Tensor] = None,
                return_checks: bool = False) -> Dict[str, torch.Tensor]:
        """``X_t``/``X_aux``: (B, N, C); ``H_bar``: (M, N).

        Returns ``q`` (B, N) error severity, ``s`` (B, M) syndrome, and the
        ``C_obs`` / ``C_ref`` check-node tensors.
        """
        U = self.W_x(X_t)                                       # (B, N, mid)
        R = self.W_r(X_aux)                                     # (B, N, mid)
        if template_context is not None:
            R = R + self.template_ctx(template_context).unsqueeze(1)

        C_obs = torch.einsum("mn,bnd->bmd", H_bar, U)           # (B, M, mid)
        C_ref = torch.einsum("mn,bnd->bmd", H_bar, R)           # (B, M, mid)

        delta = C_obs - C_ref
        s_raw = self.mlp_s(torch.cat([delta, delta.abs()], dim=-1)).squeeze(-1)   # (B, M)
        if self.cos_proj is not None:
            cos = F.cosine_similarity(C_obs, C_ref, dim=-1, eps=1e-6).unsqueeze(-1)
            s_raw = s_raw + self.cos_proj(cos).squeeze(-1)
        # explicit magnitude term (a small difference is otherwise invisible to an MLP)
        s_raw = s_raw + self.residual_scale * delta.pow(2).mean(dim=-1).sqrt()
        s_raw = s_raw * self.check_scale
        s = torch.sigmoid(s_raw)                                # (B, M)

        q_logits = torch.einsum("mn,bm->bn", H_bar, s) + self.vote_bias
        q = torch.sigmoid(q_logits)                             # (B, N)

        out = {"q": q, "q_logits": q_logits, "s": s, "s_logits": s_raw,
               "C_obs": C_obs, "C_ref": C_ref, "U": U, "R": R}
        if return_checks:
            out["H_bar"] = H_bar
        return out
