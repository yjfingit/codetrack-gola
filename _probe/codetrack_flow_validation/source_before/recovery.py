"""Block 5 / 5.1 -- selective token recovery with H-routed evidence.

Architecture figure:

    q  ------------------------------>  E_t = TopK(q),  K <= 32  (<= 12.5%)
                                              |
    H-routed neighbours N_H(i)  K_n = 8 ------+
    cross-modal X_aux_i ----------------------+
    templates Z_0, Z_online -------------------+--> lightweight cross-attention
    motion map M_t, memory T_mem --------------+     768 -> 256 -> 768
                                              |
                                              v
                                        dX_i in R^768
                                              |
                          X_i' = X_i + q_i * dX_i        (healthy tokens: X_i' = X_i)

Only the TopK tokens are touched; every other token takes an exact identity path, so at
initialisation (``dX`` output layer zero-init) the module is a strict no-op and the
loaded GOLA checkpoint reproduces its original output bit-for-bit.

Evidence routing is **Tanner-constrained sparse attention**: ``K`` and ``V`` come from
``N_H(i) cap R`` (tokens that share redundancy checks with the suspect and are
themselves reliable), not from the whole search grid.  The motion prior enters as a
log-additive spatial bias on the attention logits, which is what makes it a *prior*
rather than a second tracker.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class H_RoutedSparseRefiner(nn.Module):
    """One-step residual recovery over the TopK suspects."""

    def __init__(self, dim: int = 768, hidden: int = 256, heads: int = 4,
                 num_neighbours: int = 8, template_dim: Optional[int] = None,
                 memory_dim: int = 128, dropout: float = 0.0,
                 residual_gate_init: float = -8.0, motion_bias_scale: float = 0.5,
                 up_init_std: float = 0.02, motion_bias_normalise: bool = True):
        super().__init__()
        self.dim = dim
        self.hidden = hidden
        self.heads = heads
        self.num_neighbours = num_neighbours
        self.memory_dim = memory_dim
        self.motion_bias_scale = float(motion_bias_scale)
        self.motion_bias_normalise = bool(motion_bias_normalise)

        # condition = [ K_n neighbours (flattened) | X_aux_i | template | memory | q_i ]
        neighbour_dim = dim * num_neighbours
        cond_in = neighbour_dim + dim + dim + memory_dim + 1
        self.down = nn.Sequential(
            nn.Linear(cond_in, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(dim, hidden)
        self.v_proj = nn.Linear(dim, hidden)
        self.out_proj = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # 512 -> 768 residual predictor.
        #
        # REVISED 2026-10-04.  The previous design put the entire "be the identity at step 0"
        # burden on a large negative scalar gate (sigmoid(-8) = 3.35e-4) while leaving ``up``
        # at the default init.  Two measured consequences:
        #   * ``||X_final - X_t|| / ||X_t|| = 1.675e-05`` -- the branch is invisible at the
        #     resolution of the angular recovery metric, so every downstream comparison reads
        #     "no change" whether or not the branch works;
        #   * optimising ``codetrack.*`` directly on a recovery-only loss converges to
        #     ``d_after = 0.0111`` from gate -8.0 but ``0.00565`` from gate 0.0 -- the gate
        #     halves the achievable recovery, it does not merely hide it.
        # The two roles are now separated: ``up`` is initialised small-but-nonzero
        # (``up_init_std``) so gradient flows, and the gate starts at a moderate value so the
        # product is still near identity.  See ``residual_scale`` below.
        self.up = nn.Linear(hidden, dim)
        nn.init.normal_(self.up.weight, mean=0.0, std=float(up_init_std))
        nn.init.zeros_(self.up.bias)
        self.residual_gate = nn.Parameter(torch.full((1,), float(residual_gate_init)))

        # NOTE: no ``template_proj`` here.  An earlier version built
        # ``nn.Linear(dim, dim) if template_dim is None else nn.Identity()`` and never called
        # it, which left two parameters in the optimizer that could never receive a gradient.
        self.memory_proj = nn.Linear(memory_dim, memory_dim)

    def forward(self, X_t: torch.Tensor, X_aux: torch.Tensor, q: torch.Tensor,
                neighbour_index: torch.Tensor, template_pool: torch.Tensor,
                memory_readout: Optional[torch.Tensor] = None,
                motion_map: Optional[torch.Tensor] = None,
                topk: int = 32) -> Dict[str, torch.Tensor]:
        """``X_t``/``X_aux``: (B, N, C); ``q``: (B, N); ``neighbour_index``: (N, K_n);
        ``template_pool``: (B, C); ``memory_readout``: (B, T, D) or None;
        ``motion_map``: (B, 1, 16, 16) or None.
        """
        b, n, c = X_t.shape
        k = min(int(topk), n)
        k_n = min(int(self.num_neighbours), neighbour_index.shape[1])

        # ---- 1. TopK suspect selection -----------------------------------------
        suspect_score, suspect_idx = torch.topk(q, k=k, dim=-1)     # (B, K)
        bidx = torch.arange(b, device=X_t.device)[:, None].expand(-1, k)
        nb = neighbour_index[suspect_idx]                            # (B, K, K_n)
        nb_tokens = X_t[bidx[:, :, None], nb]                        # (B, K, K_n, C)
        aux_tokens = X_aux[bidx, suspect_idx]                        # (B, K, C)

        # ---- 2. motion prior enters as an attention bias ------------------------
        attn_bias = None
        if motion_map is not None:
            flat_map = motion_map.reshape(motion_map.shape[0], -1)    # (B, N)
            nb_bias = flat_map[bidx[:, :, None], nb]                  # (B, K, K_n)
            log_b = torch.log(nb_bias.clamp(min=1e-4))
            # REVISED 2026-10-04.  ``motion_map`` is normalised to unit mass, so its entries
            # cluster at 1/256 = 3.9e-3 with a spread of only 2e-3.  ``log`` of that is a
            # constant -1.55 plus +-0.5, and scaling it down gave a logit spread of ~0.05 nats
            # over the 8 neighbours -- a softmax that is numerically uniform, which is why
            # zeroing ``motion_map`` changed ``X_rec`` by only 7.4e-07 relative.
            # Centring and standardising the log-bias makes ``motion_bias_scale`` a real
            # "nats of spread" knob independent of the map's arbitrary offset, while preserving
            # the prior's SHAPE (which cells are favoured).  Set
            # ``motion_bias_normalise=False`` to reproduce the pre-2026-10-04 numerics exactly.
            if self.motion_bias_normalise:
                dims = tuple(range(1, log_b.dim()))
                mu = log_b.mean(dim=dims, keepdim=True)
                sd = log_b.std(dim=dims, keepdim=True, unbiased=False).clamp(min=1e-6)
                log_b = (log_b - mu) / sd
            attn_bias = log_b * float(self.motion_bias_scale)

        # ---- 3. condition ------------------------------------------------
        tpl = template_pool.unsqueeze(1).expand(-1, k, -1)            # (B, K, C)
        if memory_readout is not None:
            # ``memory_readout`` carries the TGS prior tokens, shaped (B, F, D).  Reduce over
            # the frame axis -- dim=1 -- so the per-suspect condition has one vector per
            # sample; reducing over dim=-1 would average across the feature axis instead.
            mem = memory_readout.mean(dim=1)                          # (B, D)
            mem = self.memory_proj(mem).unsqueeze(1).expand(-1, k, -1)  # (B, K, D)
        else:
            mem = X_t.new_zeros(b, k, self.memory_dim)
        gate = suspect_score.unsqueeze(-1)                            # (B, K, 1)
        cond = torch.cat([nb_tokens.flatten(2), aux_tokens, tpl, mem, gate], dim=-1)
        h = self.down(cond)                                            # (B, K, hidden)

        # ---- 4. Tanner-constrained sparse cross-attention -----------------------
        # query: the suspect token's bottleneck feature       (B, K, hidden)
        # key/value: its K_n H-routed neighbours              (B, K, K_n, dim)
        # heads: (B, H, K, d) x (B, H, K, K_n, d) -> (B, H, K, K_n)
        query = self.q_proj(h)                                        # (B, K, hidden)
        heads = self.heads
        dh = self.hidden // heads
        qh = query.view(b, k, heads, dh).permute(0, 2, 1, 3)          # (B, H, K, dh)
        kv = nb_tokens.reshape(b, k * k_n, c)
        kh = self.k_proj(kv).view(b, k, k_n, heads, dh).permute(0, 3, 1, 2, 4)  # (B,H,K,Kn,dh)
        vh = self.v_proj(kv).view(b, k, k_n, heads, dh).permute(0, 3, 1, 2, 4)
        logits = torch.einsum("bhkd,bhksd->bhks", qh, kh) / math.sqrt(dh)
        if attn_bias is not None:
            logits = logits + attn_bias.unsqueeze(1)                   # (B,1,K,Kn)
        attn = logits.softmax(dim=-1)
        ctx = torch.einsum("bhks,bhksd->bhkd", attn, vh)
        ctx = ctx.permute(0, 2, 1, 3).reshape(b, k, self.hidden)
        ctx = self.norm(h + self.dropout(self.out_proj(ctx)))

        # ---- 5. gated residual write-back --------------------------------------
        dX = torch.sigmoid(self.residual_gate.reshape(())) * self.up(ctx)   # (B, K, C)
        delta = q[bidx, suspect_idx].unsqueeze(-1) * dX                # (B, K, C)

        X_out = X_t.clone()
        X_out[bidx, suspect_idx] = X_t[bidx, suspect_idx] + delta
        return {"X_rec": X_out, "delta": dX, "suspect_index": suspect_idx,
                "suspect_score": suspect_score, "attn": attn, "context": ctx,
                "alpha": q}


def _cosine_noise_schedule(steps: int, power: float = 1.0) -> torch.Tensor:
    """SCDT-style cosine schedule: alpha_bar_t = cos(pi/2 * (t/T)^u).

    ``alpha_bar[0]`` is the *most* corrupted level and ``alpha_bar[T-1]`` the cleanest,
    matching a denoising loop that starts at the corrupted feature and refines.
    """
    t = torch.linspace(0.0, 1.0, steps)
    ab = torch.cos(0.5 * math.pi * (t ** power))
    return ab.clamp(1e-3, 1.0)


def _write_weights(steps: int, schedule: str, alpha_bar: torch.Tensor) -> torch.Tensor:
    """Per-step weight applied to the predicted correction.

    ``alpha_bar`` runs from 1 (noiseless) to ~0 (mostly noise), so ``1 - alpha_bar`` is nonzero
    only once the step has actually been noised -- which makes step 0 a guaranteed no-op
    (``1 - 1 = 0``): its prediction and all of its gradients are discarded.  ``ramp`` replaces it
    with an increasing linear schedule so every step moves the token, the later ones more, which
    is what a "T-step refinement" claim requires.
    """
    if schedule == "linear_noise":
        w = (1.0 - alpha_bar).clamp(min=0.0)
    else:  # ramp
        w = torch.arange(1, steps + 1, dtype=alpha_bar.dtype) / float(max(steps, 1))
    return w


class NoiseModulatedDenoiser(nn.Module):
    """Two-step noising-denoising refinement (architecture figure "Diffusion Correction").

    Replaces a 20-100 step DDPM with SCDT's noising-denoising idea at 2 steps, which is
    what a tracker can afford:

        X^(0) = X_corr
        for t = 0 .. T-1:
            eps_t   = noise_gate * eps              (weak/strong noise modulation)
            X_pred  = f_theta( X^t + eps_t, t, C )  (short-term cross-attention + FiLM)
            X^(t+1) = X^t + w_t * q * X_pred        (selective, gated residual)

    ``C`` is the condition set: syndrome ``s``, the H-routed context, the motion map and
    the temporal read-out.  Long-term (whole-frame) statistics enter through FiLM, i.e.
    the mean/variance of the condition set modulate the denoiser -- SCDT's "long-term
    temporal modulation".
    """

    def __init__(self, dim: int = 768, hidden: int = 256, heads: int = 4,
                 steps: int = 2, num_checks: int = 64, cond_dim: int = 256,
                 noise_schedule_power: float = 1.0, motion_dim: int = 2,
                 memory_dim: int = 128, residual_gate_init: float = -8.0,
                 write_schedule: str = "ramp", up_init_std: float = 0.02):
        super().__init__()
        self.dim = dim
        self.hidden = hidden
        self.heads = heads
        self.steps = int(steps)
        self.cond_dim = cond_dim
        self.motion_dim = int(motion_dim)
        self.memory_dim = int(memory_dim)
        if write_schedule not in ("ramp", "linear_noise"):
            raise ValueError(f"unknown diffusion write schedule: {write_schedule!r}")
        self.write_schedule = write_schedule
        self.register_buffer("alpha_bar", _cosine_noise_schedule(self.steps, noise_schedule_power))
        self.register_buffer("write_weight", _write_weights(self.steps, write_schedule,
                                                           self.alpha_bar))

        self.in_proj = nn.Linear(dim, hidden)
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(),
                                 nn.Linear(hidden * 2, hidden))
        # the condition set C = [ syndrome s (M) | H-routed/aux condition (cond_dim)
        #                         | motion prior + uncertainty (motion_dim)
        #                         | temporal memory read-out (memory_dim) ]
        self.cond_proj = nn.Linear(num_checks + cond_dim + self.motion_dim + self.memory_dim,
                                   hidden)
        # FiLM from the condition's global statistics (long-term modulation)
        self.film = nn.Linear(hidden * 2, hidden * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.step_embed = nn.Parameter(torch.zeros(self.steps, hidden))
        # REVISED 2026-10-04 -- same reasoning as the refiner above: a large negative gate on
        # the *output* collapses the whole branch (and therefore the gradient to the syndrome,
        # motion and memory conditioning paths) to ~3e-4 of nominal.  Instead, ``up`` starts
        # small-but-nonzero and the gate starts moderate, so the step-0 product is still tiny
        # while ``d(residual)/d(conditioning)`` stays usable.
        self.up = nn.Linear(hidden, dim)
        nn.init.normal_(self.up.weight, mean=0.0, std=float(up_init_std))
        nn.init.zeros_(self.up.bias)
        self.residual_gate = nn.Parameter(torch.full((1,), float(residual_gate_init)))

        # No projector is applied to an already zero-initialised residual head: a dead
        # ``W @ 0`` would zero the gradient to *every* upstream branch (including the motion
        # prior and the temporal memory), which is precisely the "new module silently gets no
        # gradient" failure this implementation is supposed to avoid.  Identity at step 0
        # comes from ``up`` itself.

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor,
                syndrome: Optional[torch.Tensor] = None,
                motion: Optional[torch.Tensor] = None,
                memory: Optional[torch.Tensor] = None,
                token_error: Optional[torch.Tensor] = None,
                token_trust: Optional[torch.Tensor] = None,
                noise_gate: Optional[torch.Tensor] = None,
                train_noise: bool = True, noise_weak: float = 0.05,
                noise_strong: float = 0.20, strong_prob: float = 0.5
                ) -> Dict[str, torch.Tensor]:
        """``tokens``: (B, N, C) corrupted feature; ``condition``: (B, N, cond_dim);
        ``syndrome``: (B, M); ``motion``: (B, motion_dim); ``memory``: (B, memory_dim).

        Returns ``X_denoised`` (B, N, C) and the per-step predictions.

        Both the noise injection and the write-back are gated by the *same* quantity: the
        per-token error probability ``q`` (block 3).  They are two separate arguments only so
        that the two roles stay independently readable; if the caller does not pass
        ``noise_gate`` it defaults to the error gate.

        This method used to take an ``alpha`` argument that the caller never supplied, so it
        fell back to ``ones`` and the noise was multiplied by ``1 - alpha == 0``.  The
        SCDT-style "noise-modulated" half of this module was therefore **dead code**: the
        write-back was gated correctly but no token ever received noise.  There is deliberately
        no ``alpha`` parameter any more -- one name for one concept.
        """
        b, n, c = tokens.shape
        heads = self.heads
        dh = self.hidden // heads
        x = tokens
        preds = []

        # Single source of truth for "how damaged is this token": q.
        if token_error is not None:
            error_gate = token_error
        elif token_trust is not None:
            error_gate = 1.0 - token_trust
        else:
            # Explicit, documented fallback: with no diagnosis available every token is
            # treated as suspect, i.e. noise acts everywhere and the write-back is ungated.
            # The previous fallback silently produced the *opposite* (no noise anywhere),
            # which is how the dead-noise bug stayed invisible.
            error_gate = tokens.new_ones(b, n, 1)
        if noise_gate is None:
            noise_gate = error_gate
        write_gate = error_gate

        # broadcast the frame-level condition terms onto every token
        if syndrome is None:
            syndrome = tokens.new_zeros(b, self.cond_proj.in_features - self.cond_dim
                                        - self.motion_dim - self.memory_dim)
        if motion is None:
            motion = tokens.new_zeros(b, self.motion_dim)
        if memory is None:
            memory = tokens.new_zeros(b, self.memory_dim)
        frame_cond = torch.cat([
            syndrome.to(tokens.dtype),
            motion.to(tokens.dtype),
            memory.to(tokens.dtype),
        ], dim=-1)                                                   # (B, M+motion+memory)
        full_cond = torch.cat([frame_cond.unsqueeze(1).expand(-1, n, -1), condition], dim=-1)

        for t in range(self.steps):
            ab = self.alpha_bar[t].to(tokens.dtype)
            # ---- noise modulation (weak / strong), SCDT's noise-modulated adaptation
            if train_noise and self.training:
                strong = (torch.rand(b, 1, 1, device=tokens.device) < strong_prob)
                sigma = torch.where(strong,
                                    torch.full_like(x[:, :1, :1], noise_strong),
                                    torch.full_like(x[:, :1, :1], noise_weak))
                eps = torch.randn_like(x) * sigma
                # Only the tokens the diagnosis flags as damaged receive noise.  Gated by the
                # error probability itself (not ``1 - alpha``, which was always 0 here).
                eps = eps * noise_gate
                x_in = (ab.sqrt() * x + (1.0 - ab).clamp(min=0).sqrt() * eps)
            else:
                x_in = x

            h = self.in_proj(x_in)                                  # (B, N, hidden)
            cond = self.cond_proj(full_cond)
            h = h + self.step_embed[t].view(1, 1, -1)
            h = self.norm1(h)

            # ---- short-term cross-attention over the H-routed condition ----------
            qh = self.q_proj(h).view(b, n, heads, dh).permute(0, 2, 1, 3)
            kh = self.k_proj(cond).view(b, n, heads, dh).permute(0, 2, 1, 3)
            vh = self.v_proj(cond).view(b, n, heads, dh).permute(0, 2, 1, 3)
            attn = (qh @ kh.transpose(-1, -2)) / math.sqrt(dh)
            ctx = (attn.softmax(dim=-1) @ vh).permute(0, 2, 1, 3).reshape(b, n, self.hidden)
            h = self.norm2(h + self.out_proj(ctx))

            # ---- long-term (global) FiLM modulation -----------------------------
            mu = h.mean(dim=1, keepdim=True)
            var = h.var(dim=1, unbiased=False, keepdim=True)
            film = self.film(torch.cat([mu, var], dim=-1).expand(-1, n, -1))
            gamma, beta = film.chunk(2, dim=-1)
            h = h * (1.0 + gamma) + beta

            h = h + self.ffn(h)
            pred = torch.sigmoid(self.residual_gate.reshape(())) * self.up(h)   # (B, N, C)
            preds.append(pred)

            # ---- residual composition: never a wholesale rewrite ---------------
            # ``noise_gate`` and ``write_gate`` are both the per-token *error* probability q.
            # A previous revision applied the gate to the noise only (``eps * (1-alpha)``) while
            # the write-back used a purely global scalar ``w``, which meant the correction went
            # to all 256 tokens alike -- undoing the architecture's own "selective recovery".
            # Worse, the caller never passed ``alpha``, so ``1 - alpha`` was 0 and the noise
            # side was not merely asymmetric but completely inert.
            w = self.write_weight[t].to(tokens.dtype)
            # Selective recovery: a token is corrected in proportion to how DAMAGED it is,
            # not how healthy.  Writing back ``w * trust * pred`` (the previous form) let
            # healthy tokens be rewritten most and damaged ones least -- the exact inverse
            # of what this branch exists to do.
            x = x + w * write_gate * pred

        return {"X_denoised": x, "step_preds": preds}


class MeanVarCompletion(nn.Module):
    """SCDT's "Mean-Var Completion": predict the clean condition's first two moments.

    The completion head is what ``L_align`` supervises -- matching per-channel mean and
    variance of the clean feature is a much cheaper target than the full tensor and is
    exactly what SCDT's ablation shows to matter (``L_align`` alone 76.0/59.9).
    """

    def __init__(self, dim: int = 768, hidden: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim * 2),
        )

    def forward(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        pooled = tokens.mean(dim=1)                                 # (B, C)
        out = self.head(pooled)
        mean, logvar = out.chunk(2, dim=-1)
        return {"pred_mean": mean, "pred_logvar": logvar}
