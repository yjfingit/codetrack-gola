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
                         seed: int = 1234,
                         layout: str = "random") -> torch.Tensor:
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

    if layout == "binary_cycles":
        if not spatial or num_checks != num_variables or links_per_check != 4:
            raise ValueError("binary_cycles requires one four-variable check per grid cell")
        for y in range(grid):
            for x in range(grid):
                corners = [y * grid + x, y * grid + (x + 1) % grid,
                           ((y + 1) % grid) * grid + x,
                           ((y + 1) % grid) * grid + (x + 1) % grid]
                support[y * grid + x, corners] = 1.
    elif spatial and locality_window > 0 and layout == "grid":
        side = int(round(num_checks ** 0.5))
        if side * side != num_checks or grid % side != 0:
            raise ValueError("grid layout requires square checks dividing token grid")
        stride = grid // side
        row_span, col_span = 3, 4
        for row in range(side):
            for col in range(side):
                cy, cx = row * stride, col * stride
                chosen = []
                for dy in range(-1, 2):
                    for dx in range(-2, 2):
                        yy = (cy + dy) % grid if locality_wrap else min(max(cy + dy, 0), grid - 1)
                        xx = (cx + dx) % grid if locality_wrap else min(max(cx + dx, 0), grid - 1)
                        chosen.append(yy * grid + xx)
                support[row * side + col, torch.tensor(chosen, dtype=torch.long)] = 1.0
    elif spatial and locality_window > 0:
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
                 seed: int = 1234, learn_weights: bool = True,
                 layout: str = "random"):
        super().__init__()
        support = build_sparse_support(
            num_checks, num_variables, links_per_check, min_col_degree,
            grid, locality_window, locality_wrap, free_edge_frac, seed, layout)
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
                 use_cos: bool = True,
                 syndrome_logit_gain: float = 1.0,
                 syndrome_gain_calibration: bool = True):
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

        # ---- syndrome saturation guard (added 2026-10-04) ----------------------
        # MEASURED DEFECT.  With the stock init, ``s_raw`` lands at ~0.2, so
        # ``s = sigmoid(s_raw)`` sits at 0.794 +- 0.0097 over 64 checks.  The damage
        # propagates through ``q_logits = H^T s + vote_bias``: the transmitted term has a
        # spread of only 2e-3 over 256 tokens, so ``q`` is constant to 3e-4 and, measured on
        # the same checkpoint,
        #   * ``TopK(q)`` selects a set 4.1e-4 away from the population mean -> near-random;
        #   * the denoiser noise gate spans max/min = 1.01x over 256 tokens (not selective);
        #   * ``frame_reliability`` is one value per batch (std 3.3e-5).
        # A point-mass head cannot be trained out of the point mass by the ordinary loss,
        # because every token sees the same gradient.  The spread has to exist at
        # initialisation.  This learnable scalar rescales the raw logit; its initial value is
        # set by ``calibrate_syndrome_gain`` below from the measured logit std.
        self.syndrome_logit_gain = nn.Parameter(
            torch.full((1,), float(syndrome_logit_gain)))
        # Scaling a nonzero mean alone saturates sigmoid.  Persist the calibrated centre
        # as a parameter: GOLA checkpoints deliberately discard buffers.
        self.syndrome_logit_offset = nn.Parameter(torch.zeros(1))
        self.syndrome_gain_calibration = bool(syndrome_gain_calibration)
        self._syndrome_gain_calibrated = float(syndrome_logit_gain) != 1.0

        # q = sigmoid(H^T s + b)
        self.vote_bias = nn.Parameter(torch.zeros(num_variables))
        # biased start at the expected corruption density (a 0 bias starts at 0.5,
        # whose soft-BCE is ln2 and fights the target)
        prior = min(max(float(detection_prior), 1e-3), 1 - 1e-3)
        with torch.no_grad():
            self.vote_bias.fill_(float(torch.logit(torch.tensor(prior))))

    @torch.no_grad()
    def syndrome_calibration_statistics(self, s_raw: torch.Tensor):
        """Return (count, sum, sum_sq, min, max) of the PRE-sigmoid syndrome.

        Collecting instead of applying keeps this point free of any collective
        communication: the caller aggregates across ranks and then calls
        ``apply_syndrome_calibration`` exactly once, OUTSIDE the model.  Doing the
        all-reduce here, inside ``forward``, is unsafe -- the DDP reducer buckets
        are already built by this point and an unmatched collective hangs the job.
        """
        # Accumulate in float64: the caller derives the variance from a single
        # pass (E[x^2] - E[x]^2) and the syndrome is strongly biased (|mean|/std
        # is ~11), so an fp32 reduction catastrophically cancels -- measured
        # relative error on std was 1.3e-5, versus 1e-15 for float64.
        raw32 = s_raw.detach().reshape(-1)
        finite = torch.isfinite(raw32)
        if not bool(finite.any()):
            return 0, 0.0, 0.0, 0.0, 0.0
        raw = raw32[finite].double()
        return (int(raw.numel()), float(raw.sum()),
                float((raw * raw).sum()), float(raw.min()), float(raw.max()))

    @torch.no_grad()
    def apply_syndrome_calibration(self, mean: float, std: float,
                                   target_std: float = 1.0) -> float:
        """Apply a CROSS-RANK calibrated gain/offset.

        ``mean`` and ``std`` must already describe the *global* batch (see
        ``codetrack/ddp_calibration.py``), not this rank's shard.
        """
        if not math.isfinite(target_std) or target_std <= 0:
            raise ValueError("target_std must be finite and positive")
        if not (math.isfinite(mean) and math.isfinite(std)) or std <= 1e-8:
            return float(self.syndrome_logit_gain)
        gain = float(min(max(target_std / std, 1.0), 1e3))
        self.syndrome_logit_offset.fill_(mean)
        self.syndrome_logit_gain.fill_(gain)
        self._syndrome_gain_calibrated = True
        return gain

    @torch.no_grad()
    def calibrate_syndrome_gain(self, s_raw: torch.Tensor, target_std: float = 1.0) -> float:
        """Set ``syndrome_logit_gain`` so the *pre-sigmoid* syndrome spans ``target_std``.

        Called on the first training batch, before applying gain/offset.  Centre the logits
        as well: multiplying their mean by a large gain would saturate the sigmoid again.
        Loaded calibrated checkpoints skip this initialisation.
        """
        if not math.isfinite(target_std) or target_std <= 0:
            raise ValueError("target_std must be finite and positive")
        raw = s_raw.detach().float()
        cur = float(raw.std(unbiased=False))
        if not bool(torch.isfinite(raw).all()) or cur <= 1e-8:
            return float(self.syndrome_logit_gain)
        gain = float(min(max(target_std / cur, 1.0), 1e3))
        self.syndrome_logit_offset.copy_(raw.mean().reshape_as(self.syndrome_logit_offset))
        self.syndrome_logit_gain.fill_(gain)
        self._syndrome_gain_calibrated = True
        return gain

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)
        # Legacy gain-only checkpoints still need centring.  A complete calibrated state
        # must never be overwritten when entering the next stage or resuming training.
        self._syndrome_gain_calibrated = (
            prefix + "syndrome_logit_gain" in state_dict
            and prefix + "syndrome_logit_offset" in state_dict)

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
        # See ``syndrome_logit_gain`` in ``__init__``: without this scale the 64 checks
        # collapse to a point mass and every downstream consumer of ``q`` degenerates.
        # Export the raw syndrome instead of calibrating in place.  The training loop
        # all-reduces the statistics across ranks and then calls
        # ``apply_syndrome_calibration`` once, outside the model.  Calibrating here
        # would make each rank adopt a DIFFERENT gain/offset (its own shard's std),
        # and the first DDP all-reduce would then mix four inconsistent parameter sets.
        _pending_calibration = (self.training and self.syndrome_gain_calibration
                                and not self._syndrome_gain_calibrated)
        s_raw = (s_raw - self.syndrome_logit_offset) * self.syndrome_logit_gain
        s = torch.sigmoid(s_raw)                                # (B, M)

        # The transmitted check evidence is an error score.  With the original positive
        # incidence map, damaged tokens were assigned *lower* q (measured AUROC 0.258 and
        # TopK overlap 0.016 on the fixed LasHeR batch).  Negating the projection makes q's
        # semantics match every downstream consumer: larger q means more likely damaged.
        # Reverse only the transmitted evidence. ``vote_bias`` is already logit(prior), so
        # negating it as well changes a 0.2 error prior into a 0.8 prior and makes every token
        # look damaged even when the syndrome carries no evidence.
        q_logits = -torch.einsum("mn,bm->bn", H_bar, s) + self.vote_bias
        q = torch.sigmoid(q_logits)                             # (B, N), error probability

        out = {"q": q, "q_logits": q_logits, "s": s, "s_logits": s_raw,
               "C_obs": C_obs, "C_ref": C_ref, "U": U, "R": R}
        if _pending_calibration:
            out["syndrome_pending_calibration"] = s_raw.detach()
        if return_checks:
            out["H_bar"] = H_bar
        return out


class NeuralBPSyndromeDiagnosis(nn.Module):
    """Unfolded neural belief propagation over the fixed Tanner graph.

    The feature encoder emits a signed symbol LLR for each search token.  Each BP
    iteration then exchanges extrinsic messages along the sparse parity-check support:
    variable-to-check messages add all other check messages, and check-to-variable
    messages use a differentiable sum-product update.  The graph therefore performs
    localization through redundancy instead of a dense MLP backprojection.

    The output convention is explicit throughout: positive logits mean ``damaged``.
    The final q is initialized to the configured corruption prior, so a fresh CodeTrack
    checkpoint remains an identity wrapper around the pretrained tracker.
    """

    def __init__(self, dim: int = 768, mid_dim: int = 128, num_checks: int = 64,
                 num_variables: int = 256, syndrome_hidden: int = 128,
                 detection_prior: float = 0.2, bp_iterations: int = 3,
                 bp_damping: float = 0.5, explicit_syndrome_weight: float = 0.0,
                 center_logits: bool = False):
        super().__init__()
        self.num_checks = int(num_checks)
        self.num_variables = int(num_variables)
        self.bp_iterations = int(bp_iterations)
        self.bp_damping = float(bp_damping)
        self.explicit_syndrome_weight = float(explicit_syndrome_weight)
        self.center_logits = bool(center_logits)
        self.W_x = nn.Linear(dim, mid_dim)
        self.W_r = nn.Linear(dim, mid_dim)
        self.template_ctx = nn.Linear(dim, mid_dim)
        self.symbol = nn.Sequential(
            nn.Linear(mid_dim * 2, syndrome_hidden), nn.GELU(),
            nn.Linear(syndrome_hidden, 1))
        # A zero final layer gives a deterministic prior-only decoder at step 0 while
        # retaining gradients through the complete encoder and BP graph.
        nn.init.normal_(self.symbol[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.symbol[-1].bias)
        self.edge_gain = nn.Parameter(torch.zeros(num_checks, num_variables))
        self.check_gain = nn.Parameter(torch.ones(num_checks))
        self.variable_gain = nn.Parameter(torch.ones(num_variables))
        self.vote_bias = nn.Parameter(torch.full((num_variables,),
                                  float(torch.logit(torch.tensor(
                                      min(max(float(detection_prior), 1e-3), 1 - 1e-3))))))

    def forward(self, X_t: torch.Tensor, X_aux: torch.Tensor, H_bar: torch.Tensor,
                template_context: Optional[torch.Tensor] = None,
                return_checks: bool = False) -> Dict[str, torch.Tensor]:
        b, n, _ = X_t.shape
        U = self.W_x(X_t)
        R = self.W_r(X_aux)
        if template_context is not None:
            R = R + self.template_ctx(template_context).unsqueeze(1)
        diff = U - R
        # Keep the observable check space explicit.  The post-recovery diagnostic consumes
        # these tensors; exporting BP messages here made ``syndrome_before`` live in a
        # different space from ``syndrome_after`` and produced a meaningless jump after a
        # near-identity write-back.
        C_obs = torch.einsum("mn,bnd->bmd", H_bar, U)
        C_ref = torch.einsum("mn,bnd->bmd", H_bar, R)
        symbol_logits = self.symbol(torch.cat([diff, diff.abs()], dim=-1)).squeeze(-1)
        support = (H_bar > 0).to(symbol_logits.dtype)
        # Keep learned edge scales bounded and zero at initialization.  The fixed H
        # still defines every message route; scales only tune useful checks.
        edge = H_bar * (1.0 + 0.25 * torch.tanh(self.edge_gain)) * support
        edge = edge / edge.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        # A real visual parity check: H aggregates the cross-modal residual energy.
        # This is a syndrome of the observation pair, rather than a free MLP score.
        check_residual = torch.sqrt(
            torch.einsum("mn,bn->bm", edge, diff.square().mean(dim=-1)).clamp_min(1e-8))
        check_z = (check_residual - check_residual.mean(dim=-1, keepdim=True)) / \
            check_residual.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        v2c = symbol_logits.unsqueeze(1).expand(-1, self.num_checks, -1)
        c2v = torch.zeros_like(v2c)
        damping = min(max(self.bp_damping, 0.0), 1.0)
        check_logits = torch.zeros(b, self.num_checks, device=X_t.device, dtype=X_t.dtype)
        for _ in range(self.bp_iterations):
            total = (edge.unsqueeze(0) * c2v).sum(dim=-1, keepdim=True)
            v2c_new = symbol_logits.unsqueeze(1) + total - c2v
            # Sum-product check update. tanh/atanh is stable after clamping and is
            # fully differentiable, unlike a hard parity decision.
            tanh_msg = torch.tanh(0.5 * v2c_new).clamp(-0.999, 0.999)
            signed = torch.sign(tanh_msg + 1e-8)
            log_abs = torch.log(tanh_msg.abs().clamp_min(1e-6))
            sum_log = (support.unsqueeze(0) * log_abs).sum(dim=-1, keepdim=True)
            prod_excl = torch.exp(sum_log - log_abs).clamp(1e-6, 0.999)
            # For +/-1 signs, product(all) * sign(i) is the product excluding i.
            # The sign is a routing cue; the magnitude remains differentiable.
            sign_excl = signed.prod(dim=-1, keepdim=True) * signed
            c2v_new = 2.0 * torch.atanh((sign_excl * prod_excl).clamp(-0.999, 0.999))
            c2v_new = c2v_new * self.check_gain.view(1, -1, 1) * edge.unsqueeze(0)
            c2v = damping * c2v + (1.0 - damping) * c2v_new
            check_logits = (edge.unsqueeze(0) * v2c_new).sum(dim=-1)
            v2c = v2c_new
        var_logits = symbol_logits * self.variable_gain.view(1, -1) + c2v.sum(dim=1)
        parity_vote = torch.einsum("mn,bm->bn", edge, check_z)
        if self.center_logits:
            # A fixed channel prior should set the frame-average reliability; BP should only
            # provide relative token evidence. Centering prevents a global negative message
            # from collapsing every q while preserving all token-to-token ordering.
            var_logits = var_logits - var_logits.mean(dim=-1, keepdim=True)
        q_logits = var_logits + self.vote_bias + self.explicit_syndrome_weight * parity_vote
        q = torch.sigmoid(q_logits)
        s_logits = check_logits + self.explicit_syndrome_weight * check_z
        s = torch.sigmoid(s_logits)
        out = {"q": q, "q_logits": q_logits, "s": s, "s_logits": s_logits,
               "C_obs": C_obs, "C_ref": C_ref, "U": U, "R": R,
               "symbol_logits": symbol_logits, "bp_messages": c2v,
               "check_residual": check_residual, "syndrome_residual": check_z}
        if return_checks:
            out["H_bar"] = H_bar
        return out
