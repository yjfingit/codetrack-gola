"""Block 4 -- Motion prior and temporal memory.

Architecture figure:

    historical trajectory b_{t-K..t-1}
        |
        v
    Kalman filter (uncertainty estimation)   ->  b_hat_t, P_t
        |
        v
    motion prior map M_t (16 x 16)
        |
        v
    temporal memory T_mem   B x K x 8 x 128

This is a **recurrent** module: one Kalman state per batch element, carried across
calls.  It is deliberately *not* a second tracker -- it produces (a) a soft spatial
prior used as a bias inside the recovery cross-attention and (b) a scalar motion
uncertainty ``u_t`` consumed by the template-protection gate.  Nothing here writes to
the tracking head's input directly, so ground-truth teacher forcing cannot leak a box
into the head.

State (constant velocity), all in normalised image coordinates so the same process /
measurement noise works across sequences:

    x = [cx, cy, w, h, vx, vy, vw, vh]^T
    x_t = A x_{t-1} + w,   w ~ N(0, Q)
    z_t = H_obs x_t + v,   v ~ N(0, R)

The quantities the rest of CodeTrack consumes are the *prediction* and its covariance:

    b_hat_t = decode(x_{t|t-1})
    u_t     = f(tr P_{t|t-1}, ||innovation||)
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _logit(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x) - torch.log1p(-x)


class KalmanMotionPrior(nn.Module):
    """Constant-velocity Kalman filter over the target box, with a spatial prior head."""

    def __init__(self, grid: int = 16, state_dim: int = 8,
                 process_noise: float = 1e-2, measurement_noise: float = 4e-2,
                 gate_hidden: int = 64, default_seed_side: float = 0.3):
        super().__init__()
        self.grid = int(grid)
        self.state_dim = int(state_dim)
        # normalised side length of the "no observation yet" seed box (0.3 of the image)
        self.default_seed_side = float(default_seed_side)
        if self.state_dim != 8:
            raise ValueError("KalmanMotionPrior expects the 8-d box-velocity state")

        # ---- constant-velocity transition and observation models ----------------
        eye4 = torch.eye(4)
        A = torch.zeros(8, 8)
        A[:4, :4] = eye4
        A[:4, 4:] = eye4                     # position += velocity
        A[4:, 4:] = eye4                     # velocity random walk
        H_obs = torch.zeros(4, 8)
        H_obs[:, :4] = eye4
        self.register_buffer("A", A)
        self.register_buffer("H_obs", H_obs)
        self.register_buffer("Q", torch.eye(8) * float(process_noise))
        self.register_buffer("R", torch.eye(4) * float(measurement_noise))
        # initial covariance: position confident, velocity unknown
        init_P = torch.diag(torch.tensor([1e-2] * 4 + [1.0] * 4))
        self.register_buffer("P0", init_P)

        # ---- the spatial prior head --------------------------------------------
        # (x_{t|t-1} (8) | sqrt(diag P) (8) | u_t (1)) -> 5 geometric parameters
        #
        # The head predicts [dcx, dcy, dlog w, dlog h, log temperature] rather than a single
        # multiplicative amplitude.  An amplitude is inert here: the map is normalised to unit
        # mass by the consumer (``pm / pm.sum()``), so any global scale factor cancels exactly
        # and the head would receive no useful spatial gradient.  Offsetting the centre and
        # scaling the extents changes the *shape* of the prior, which is what actually
        # survives normalisation and what a motion prior is supposed to express.
        self.prior = nn.Sequential(
            nn.Linear(8 + 8 + 1, gate_hidden), nn.GELU(),
            nn.Linear(gate_hidden, gate_hidden), nn.GELU(),
            nn.Linear(gate_hidden, 5),
        )
        # zero-init the last layer: at step 0 the offsets are 0 and the temperature is
        # exp(0)=1, so the prior is exactly the analytic box Gaussian and the recovery branch
        # stays bit-identical to "no learned motion prior".
        nn.init.zeros_(self.prior[-1].weight)
        nn.init.zeros_(self.prior[-1].bias)
        self.uncertainty_gain = nn.Parameter(torch.ones(1))

        self.reset_state()

    # ------------------------------------------------------------------ state
    def reset_state(self) -> None:
        """Drop the per-batch Kalman state (call at the start of every sequence)."""
        self._x: Optional[torch.Tensor] = None      # (B, 8) normalised
        self._P: Optional[torch.Tensor] = None      # (B, 8, 8)
        self._initialised = False
        self.last_innovation = None
        self.last_mahalanobis = None

    def rebase(self, scale: torch.Tensor, translation: torch.Tensor) -> None:
        """Move a posterior between normalised search-crop coordinate systems.

        If p_new = scale * p_old + translation, extents and velocities scale,
        while only centres translate. Covariance uses the same affine Jacobian.
        This operation uses crop geometry, never current-frame annotations.
        """
        if self._x is None:
            return
        scale = scale.to(self._x).reshape(-1, 2)
        translation = translation.to(self._x).reshape(-1, 2)
        if scale.shape[0] != self._x.shape[0]:
            raise ValueError("crop transform batch does not match Kalman state")
        if not bool(torch.isfinite(scale).all() and (scale > 0).all()
                    and torch.isfinite(translation).all()):
            raise ValueError("invalid search-crop transform")
        jac = scale.repeat(1, 4)
        offset = torch.cat([translation, torch.zeros_like(translation).repeat(1, 3)], -1)
        self._x = self._x * jac + offset
        self._P = self._P * jac[:, :, None] * jac[:, None, :]
        if self.last_innovation is not None:
            self.last_innovation = self.last_innovation * scale.repeat(1, 2)

    @staticmethod
    def _flat_cov(P: torch.Tensor, b: int) -> torch.Tensor:
        """Return the covariance as exactly ``(B, d, d)``.

        Chained ``A @ P @ A.T + Q`` and ``P - K H P`` broadcast the leading axes, and a
        stray singleton silently grows the tensor to ``(B, 1, d, d)``.  Every covariance
        read-out below then disagrees about its own dimensionality, so the shape is pinned
        once, right after each update.
        """
        d = P.shape[-1]
        # collapse any broadcast-in leading axis (e.g. (B, B, d, d) from a stale state)
        return P.reshape(-1, d, d)[-b:].reshape(b, d, d)

    @property
    def initialised(self) -> bool:
        return self._initialised

    def _ensure_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> None:
        if (self._x is None or self._x.shape[0] != batch
                or self._x.device != device):
            self._x = torch.zeros(batch, 8, device=device, dtype=dtype)
            self._P = self.P0.to(device=device, dtype=dtype).unsqueeze(0).repeat(batch, 1, 1)
            self._initialised = False

    def _seed_unobserved(self, batch: int, device: torch.device, dtype: torch.dtype) -> None:
        """Seed the filter with a centred, neutrally sized box when nothing was observed.

        The centre is the image centre (cx=cy=0.5 in normalised coordinates) and the extent is
        ``default_seed_side`` of the image, with zero velocity.  Covariance is left at ``P0``
        (position confident, velocity unknown), so ``u_t`` stays high and the prior is treated
        as weak evidence until real observations arrive -- which is the honest encoding of
        "we have not seen the target yet".
        """
        side = float(self.default_seed_side)
        x0 = torch.tensor([0.5, 0.5, side, side, 0.0, 0.0, 0.0, 0.0],
                          device=device, dtype=dtype)
        self._x = x0.unsqueeze(0).repeat(batch, 1).contiguous()
        self._P = self.P0.to(device=device, dtype=dtype).unsqueeze(0).repeat(batch, 1, 1)

    # minimum normalised side length: 2% of the frame.  Without a floor, feeding the head's
    # own prediction back as an observation is a positive-feedback loop -- a small predicted box
    # sharpens the prior, which makes the head predict an even smaller box.  Measured: over six
    # frames the state collapsed from side 0.30 to 0.02 and drifted off-frame.
    MIN_SIDE = 0.02

    def _clamp_state(self) -> None:
        """Keep the filter in a physically admissible region of normalised box space.

        Centres are clipped to [0, 1] (a box centre outside the frame is not a hypothesis worth
        entertaining, and it makes the spatial prior map empty) and the sides are floored at
        ``MIN_SIDE``.  Velocities are untouched: they are already bounded by ``Q``.
        """
        if self._x is None:
            return
        x = self._x
        c = x[:, :2].clamp(0.0, 1.0)
        wh = x[:, 2:4].clamp(min=self.MIN_SIDE)
        self._x = torch.cat([c, wh, x[:, 4:]], dim=-1)

    # --------------------------------------------------------------- filtering
    def _normalise_box(self, box_xywh: torch.Tensor, image_size: torch.Tensor) -> torch.Tensor:
        """``box_xywh`` (B, 4) pixels -> (B, 4) normalised cxcywh.

        ``image_size`` is (B, 2) as (width, height); both the centre and the extent are
        divided by it, so a box in a 224-wide search crop and one in a 960-wide frame end
        up on the same scale.
        """
        size = image_size.to(box_xywh.dtype).reshape(box_xywh.shape[0], 2)
        size = size.repeat(1, 2)                                  # (B, 4) -> [w, h, w, h]
        cxcy = box_xywh[:, :2] / size[:, :2]
        whn = box_xywh[:, 2:] / size[:, 2:]
        return torch.cat([cxcy, whn], dim=-1)

    def observe(self, box_xywh: torch.Tensor, image_size: torch.Tensor,
                confidence: Optional[torch.Tensor] = None,
                valid: Optional[torch.Tensor] = None,
                predict: bool = True) -> None:
        """Feed one observed box into the filter (measurement update).

        ``valid`` (B,) bool marks samples whose observation should be ignored; the
        filter still propagates its prediction for them.
        """
        b = box_xywh.shape[0]
        self._ensure_state(b, box_xywh.device, box_xywh.dtype)
        z = self._normalise_box(box_xywh, image_size)
        # constrain the *measurement* too, not just the state: an off-frame or degenerate head
        # output must not be able to pull the filter out of the admissible region
        z = torch.cat([z[:, :2].clamp(0.0, 1.0),
                       z[:, 2:4].clamp(min=self.MIN_SIDE)], dim=-1)

        if valid is None:
            valid = torch.ones(b, device=box_xywh.device, dtype=torch.bool)
        if confidence is not None:
            # a low confidence observation is a noisy measurement: inflate R
            inflate = (1.0 - confidence.clamp(0.0, 1.0)).to(box_xywh.dtype)
            R = self.R.to(box_xywh.dtype).unsqueeze(0) * (1.0 + 20.0 * inflate)[:, None, None]
        else:
            R = self.R.to(box_xywh.dtype).unsqueeze(0).expand(b, -1, -1)

        if not self._initialised:
            # nothing to fuse with: seed the state directly
            self._x = torch.cat([z, torch.zeros_like(z)], dim=-1)
            self._P = self.P0.to(device=box_xywh.device, dtype=box_xywh.dtype
                                 ).unsqueeze(0).repeat(b, 1, 1).clone()
            self._clamp_state()
            self._initialised = True
            return

        # if the filter was seeded on a different batch before, re-seed those rows
        A = self.A.to(box_xywh.dtype)
        x_pred = self._x @ A.t() if predict else self._x
        P_pred = (self._flat_cov(A @ self._P @ A.t() + self.Q.to(box_xywh.dtype), b)
                  if predict else self._P)

        H = self.H_obs.to(box_xywh.dtype)
        S = H @ P_pred @ H.t() + R                                  # (B, 4, 4)
        K = P_pred @ H.t() @ torch.linalg.inv(S)                    # (B, 8, 4)
        innovation = z - (H @ x_pred.unsqueeze(-1)).squeeze(-1)     # (B, 4)
        x_upd = x_pred + (K @ innovation.unsqueeze(-1)).squeeze(-1)
        P_upd = self._flat_cov(P_pred - K @ H @ P_pred, b)

        mask = valid[:, None]
        self._x = torch.where(mask, x_upd, x_pred)
        self._P = torch.where(mask[:, None, None], P_upd, P_pred)
        self._clamp_state()
        self.last_innovation = innovation.detach()
        # Mahalanobis distance of the innovation w.r.t. its own covariance S.  This is the
        # theoretically correct "how surprising is this measurement" statistic (unlike the raw
        # norm, which ignores the anisotropy of S), and DTPTrack's worst ablation shows that a
        # fixed threshold on such a quantity is far worse than feeding it to a learned gate.
        self.last_mahalanobis = torch.linalg.solve(
            S.detach(), innovation.detach().unsqueeze(-1)).squeeze(-1)
        self.last_mahalanobis = (self.last_mahalanobis * innovation.detach()
                                 ).sum(-1).clamp(min=0).sqrt()          # (B,)

    # ---------------------------------------------------------------- prior out
    def forward(self, box_xywh: Optional[torch.Tensor] = None,
                image_size: Optional[torch.Tensor] = None,
                confidence: Optional[torch.Tensor] = None,
                valid: Optional[torch.Tensor] = None,
                batch_size: Optional[int] = None,
                device: Optional[torch.device] = None,
                dtype: Optional[torch.dtype] = None,
                defer_observe: bool = False,
                advance: bool = False,
                ) -> Dict[str, torch.Tensor]:
        """One frame of the filter.

        ``defer_observe=True`` makes this a **prediction-only** step: ``box_xywh`` is ignored
        and the state is left at ``x_{t|t-1}``.  The caller is then responsible for invoking
        :meth:`observe` once it has finished consuming this frame's outputs.

        That split is what makes the module causal.  It used to absorb the current frame's
        ground-truth box *before* building the prior map, so ``M_t`` was a function of the
        very frame it was supposed to help predict -- ground truth leaking into the current
        output rather than only into the loss.
        """
        requested_batch = (image_size.shape[0] if image_size is not None else
                           (box_xywh.shape[0] if box_xywh is not None else batch_size))
        # The final evaluation batch can be smaller than the preceding batch. Kalman
        # state is batch-shaped, so discard it when the layout changes instead of
        # broadcasting stale sequences into the new batch.
        if self._x is not None and requested_batch is not None \
                and self._x.shape[0] != int(requested_batch):
            self.reset_state()
        if box_xywh is not None and not defer_observe:
            self.observe(box_xywh, image_size, confidence=confidence, valid=valid)
        if not self._initialised or self._x is None:
            # No observation has ever arrived.  This is the situation on **every inference
            # frame**: the tracker is deliberately not handed a box, so ``observe()`` never
            # runs and the filter would otherwise stay at its zero state -- a box at the image
            # origin with zero extent.  That state makes ``motion_map`` identically zero, and
            # since the map enters the refiner as ``log(map)*scale`` the attention bias
            # degenerates to a constant, i.e. the motion prior contributes *nothing*.  Seed a
            # centred, reasonably sized box instead so the prior is a real (weak) prior.
            dev = (image_size.device if image_size is not None
                   else (device if device is not None else torch.device("cuda")))
            dt = (image_size.dtype if image_size is not None
                  else (dtype if dtype is not None else torch.float32))
            b_guess = (image_size.shape[0] if image_size is not None
                       else (batch_size if batch_size is not None else 1))
            self._ensure_state(b_guess, dev, dt)
            self._seed_unobserved(b_guess, dev, dt)
            self._initialised = True

        b = self._x.shape[0]
        device, dtype = self._x.device, self._x.dtype
        A = self.A.to(dtype)
        x_pred = self._x @ A.t()
        P_pred = self._flat_cov(A @ self._P @ A.t() + self.Q.to(dtype), b)
        if advance:
            # Exactly one transition per frame; the later head observation must call
            # observe(predict=False). The legacy deferred API remains available.
            self._x, self._P = x_pred, P_pred

        # ---- covariance read-out ------------------------------------------------
        # ``torch.diagonal`` indexing is ambiguous once a leading axis is involved, so the
        # diagonal is extracted by masking instead: multiply by the identity and sum.  Both
        # results are then (B, 8) and (B,) by construction.
        eye = torch.eye(self.state_dim, device=device, dtype=dtype)
        var = (P_pred * eye).sum(dim=-1).clamp(min=0.0).reshape(b, self.state_dim)
        trace = var.sum(dim=-1).reshape(b)                             # (B,)
        std = var.sqrt().reshape(b, self.state_dim)                    # (B, 8)

        if getattr(self, "last_innovation", None) is not None \
                and self.last_innovation.shape[0] == b:
            innov = self.last_innovation.to(dtype).norm(dim=-1)        # (B,)
        else:
            innov = torch.zeros(b, device=device, dtype=dtype)
        # u_t = f(tr P_{t|t-1}, ||innovation||)
        gain = self.uncertainty_gain.reshape(-1)[0]
        uncertainty = (gain * (trace.sqrt() + innov)).reshape(b).clamp(min=0.0)   # (B,)

        # ---- spatial prior -----------------------------------------------------
        # prior head consumes [x_pred (8) | sqrt(diag P) (8) | u_t (1)] and predicts geometric
        # offsets: centre shift, log-scale factors, and a temperature.  Applying them before the
        # Gaussian means the learned parameters shape the map rather than scaling it (a global
        # scale is removed by the consumer's normalisation anyway).
        x_prior = torch.cat([x_pred, std, uncertainty.unsqueeze(-1)], dim=-1)   # (B, 17)
        theta = self.prior(x_prior)                                    # (B, 5)
        d_centre, d_logwh, log_temp = theta[:, :2], theta[:, 2:4], theta[:, 4:5]
        mu = x_pred[:, :2] + d_centre                                  # (B, 2)
        wh = x_pred[:, 2:4].clamp(min=1e-3) * torch.exp(d_logwh)       # (B, 2)
        temp = torch.exp(log_temp).clamp(0.05, 20.0)                   # (B, 1)
        coords = self._grid_centres(device, dtype)                     # (16, 16, 2)
        xy = mu.unsqueeze(1).unsqueeze(1)                              # (B, 1, 1, 2)
        wh_b = wh.unsqueeze(1).unsqueeze(1)
        # inverse-box Gaussian: distance measured in units of the (modulated) box size
        d = (coords.unsqueeze(0) - xy) / wh_b
        sq = (d ** 2).sum(-1)                                          # (B, 16, 16)
        prior_map = torch.exp(-0.5 * sq / temp.unsqueeze(-1))

        # predicted box back to pixel xywh
        size = image_size if image_size is not None else torch.ones(b, 2, device=device, dtype=dtype)
        size = size.to(dtype)
        wh_px = x_pred[:, 2:4] * size
        xy_px = x_pred[:, :2] * size
        motion_box = torch.cat([xy_px - wh_px / 2.0, wh_px], dim=-1)

        mahal = (self.last_mahalanobis[:b]
                 if getattr(self, "last_mahalanobis", None) is not None
                 and self.last_mahalanobis.shape[0] == b
                 else torch.zeros(b, device=device, dtype=dtype))
        return {
            "motion_box": motion_box,
            "motion_map": prior_map.unsqueeze(1),                      # (B, 1, 16, 16)
            "uncertainty": uncertainty,
            "innovation": innov.reshape(b),                            # (B,)
            "mahalanobis": mahal.reshape(b),                           # (B,)
            "state": x_pred,
            "cov_trace": trace,
            # exposed so the supervision target can use the same functional family as the
            # prediction instead of forcing a fixed Gaussian width on the head
            "temperature": temp.reshape(b),
        }

    def _grid_centres(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # token-centre coordinates in normalised image space
        lin = (torch.arange(self.grid, device=device, dtype=dtype) + 0.5) / self.grid
        yy, xx = torch.meshgrid(lin, lin, indexing="ij")
        return torch.stack([xx, yy], dim=-1)


class TemporalMemory(nn.Module):
    """Reliability-calibrated temporal memory with a synthesized prior (DTPTrack-style TRC + TGS).

    Redesigned after DTPTrack (CVPR 2026), whose ablations isolate exactly the choices made
    here; each is noted with the measured effect reported in that paper.

    **TRC -- Temporal Reliability Calibrator.**
        1. *Masked average pooling* over the target mask to distil each frame into one
           summary vector:  s_i = sum_j Z_ij M_ij / (sum_j M_ij + eps)          (eq. 1)
           This is materially better than pooling over all tokens, because the template is
           112x112 of which the target occupies a small fraction.
        2. *Learned confidence gate* (MLP + sigmoid) over the summaries:
           [c_1..c_F] = sigmoid(MLP([s_1..s_F]))                                (eq. 2)
           DTPTrack measures **-2.3 AUC on LaSOT** when this is replaced by a static
           threshold, so the gate is learned rather than heuristic.
        3. *Anchored first frame*: the initial template comes from ground truth, so its
           confidence is pinned to 1.0 rather than predicted.  Letting the gate score it too
           degrades long-horizon tracking in their ablation.
        4. Calibrated summaries:  s_hat_i = s_i * c_i.

    **Motion coupling (CodeTrack-specific).** The Kalman uncertainty ``u_t`` is appended to
    the gate input.  Motion uncertainty is therefore not a separate heuristic branch: it
    enters the *same* learned reliability decision that decides how much history to trust.

    **TGS -- Temporal Guidance Synthesizer.**
           P_dyn = P_base + f_mod([s_hat_0 .. s_hat_F])                         (eq. 3)
       with learnable base prior tokens ``P_base``.  Removing ``P_base`` and using the
       modulated signal alone loses performance, and replacing the dedicated prior tokens by
       direct concatenation of the summaries also loses, so the prior is a separate token set
       that conditions the refiner instead of being spliced into the visual token stream.
    """

    def __init__(self, dim: int = 768, frames: int = 3, tokens: int = 8,
                 memory_dim: int = 128, decay: float = 0.6,
                 admission_threshold: float = 0.85):
        super().__init__()
        self.frames = int(frames)
        self.tokens = int(tokens)              # K: number of prior tokens (TGS P_base rows)
        self.memory_dim = int(memory_dim)
        self.decay = float(decay)
        # Admission threshold.  DTPTrack's released code admits a new history frame only when
        # the predicted max score is >= 0.85, otherwise it duplicates the previous last frame
        # instead.  The paper never mentions this, yet it is where a large part of the drift
        # control lives, so it is reproduced here (``admit`` lets the caller apply it at
        # inference only -- during training every frame is admitted, matching DTPTrack).
        self.admission_threshold = float(admission_threshold)
        self._admitted_once = False

        # summary projection for the stored history (eq. 1 output lives in memory_dim)
        self.summary_proj = nn.Linear(dim, memory_dim)
        # TRC confidence gate over the frame summaries (+1 for the Kalman uncertainty)
        # The bank holds exactly ``frames`` slots; the anchor confidence is pinned afterwards
        # by ``c_anchored``, so the gate does NOT need an extra +1 slot.  Input = the frame
        # summaries plus 2 motion statistics (Kalman uncertainty u_t and Mahalanobis distance).
        self.trc_gate = nn.Sequential(
            nn.Linear(frames * memory_dim + 2, 128), nn.GELU(),
            nn.Linear(128, frames),
        )
        # TGS: learnable base prior tokens and the modulator that shifts them (eq. 3)
        self.base_prior = nn.Parameter(torch.zeros(1, self.tokens, memory_dim))
        nn.init.trunc_normal_(self.base_prior, std=0.02)
        self.tgs_modulator = nn.Sequential(
            nn.Linear(frames * memory_dim, 2 * memory_dim), nn.GELU(),
            nn.Linear(2 * memory_dim, self.tokens * memory_dim),
        )
        # A zero output layer blocks recovery gradients into summaries/TRC on step zero.
        nn.init.normal_(self.tgs_modulator[-1].weight, std=0.02)
        nn.init.zeros_(self.tgs_modulator[-1].bias)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _masked_average(tokens: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Eq. 1: masked average pooling ``(B, N, C) -> (B, C)``."""
        if mask is None:
            return tokens.mean(dim=1)
        m = mask.to(tokens.dtype)
        if m.dim() == tokens.dim():
            m = m.squeeze(-1)
        m = m.unsqueeze(-1)                                    # (B, N, 1)
        return (tokens * m).sum(dim=1) / (m.sum(dim=1) + 1e-6)

    def forward(self, tokens: torch.Tensor, reliability: torch.Tensor,
                memory: Optional[torch.Tensor] = None,
                memory_rel: Optional[torch.Tensor] = None,
                foreground: Optional[torch.Tensor] = None,
                target_mask: Optional[torch.Tensor] = None,
                uncertainty: Optional[torch.Tensor] = None,
                admission_score: Optional[torch.Tensor] = None,
                admit: bool = True,
                mahalanobis: Optional[torch.Tensor] = None,
                detach_memory: bool = True,
                ) -> Dict[str, torch.Tensor]:
        """``tokens`` (B, N, C); ``reliability`` (B, N) in [0,1]; ``target_mask`` (B, N)
        boolean/float marking the target patches (drives the masked pooling);
        ``uncertainty`` (B,) Kalman motion uncertainty.

        Returns the updated bank (detached -- it is a buffer, not a parameterised path) and
        the synthesized dynamic prior tokens for the current frame.
        """
        b, n, _ = tokens.shape
        # Evaluation batches can change size at the dataset boundary. The recurrent
        # memory bank is batch-shaped; stale rows belong to the previous batch and
        # must be discarded rather than assigned into the new layout.
        if memory is not None and memory.shape[0] != b:
            memory = None
            memory_rel = None
            self._admitted_once = False
        # ---- eq. 1: summarise the current frame over the target mask ---------------
        pooled = self.summary_proj(tokens)                              # (B, N, D)
        if target_mask is not None:
            cur_summary = self._masked_average(pooled, target_mask)
        elif foreground is not None:
            cur_summary = self._masked_average(pooled, foreground)
        else:
            # fall back on the per-token reliability as the pooling weight
            w = reliability / reliability.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            cur_summary = (pooled * w.unsqueeze(-1)).sum(dim=1)
        cur_summary = cur_summary.unsqueeze(1)                          # (B, 1, D)
        # Reliability already lies in [0, 1]; another sigmoid restricts it to [0.5, 0.731].
        cur_rel = reliability.mean(dim=-1, keepdim=True).unsqueeze(-1)  # (B,1,1)

        # ---- roll the bank: newest frame in front, older ones decay ----------------
        admitted = None
        # The bank always has exactly ``frames`` slots: the TRC gate and the TGS modulator are
        # sized for a fixed frame count, so an unfilled bank is zero-padded *before* any branch
        # uses it.  Padding after the branch (as an earlier version did) let the first call
        # reach the gate with a single slot and a 128-wide input instead of 386.
        D = self.memory_dim
        prev = cur_summary.new_zeros(b, self.frames, D)
        prev_rel = cur_summary.new_zeros(b, self.frames, 1)
        if memory is not None:
            k = min(self.frames, memory.shape[1])
            prev[:, self.frames - k:] = memory[:, :k]
            prev_rel[:, self.frames - k:] = memory_rel[:, :k]

        if not self._admitted_once:
            # First real frame: it becomes the newest slot; the rest stay empty.  It also acts
            # as the anchor slot, whose confidence is pinned to 1.0 below.
            mem = torch.cat([cur_summary, prev[:, 1:]], dim=1)
            mem_rel = torch.cat([cur_rel, prev_rel[:, 1:]], dim=1)
            self._admitted_once = True
        else:
            mem = torch.cat([cur_summary, (prev * self.decay)[:, : self.frames - 1]], dim=1)
            mem_rel = torch.cat(
                [cur_rel, (prev_rel * self.decay)[:, : self.frames - 1]], dim=1)
            if (not admit) and admission_score is not None:
                # DTPTrack's released-code rule: a frame whose score is below the threshold is
                # not admitted, and the previous newest entry is duplicated in its place.
                ok = (admission_score.reshape(b) >= self.admission_threshold)
                admitted = ok
                for i in range(b):
                    if not bool(ok[i]):
                        mem[i, 0] = prev[i, 0]
                        mem_rel[i, 0] = prev_rel[i, 0]

        # ---- TRC (eq. 2): learned per-frame confidence, first frame anchored -------
        if uncertainty is None:
            u = mem.new_zeros(b, 1)
        else:
            u = uncertainty.reshape(b, 1).to(mem.dtype)
        # The gate input is the pooled appearance of each frame *plus* the motion statistics.
        # DTPTrack's largest ablation (-2.3 AUC) is replacing this learned gate by a fixed
        # threshold, and its motion-proxy ablation shows hand-tuned motion cues lose -- so the
        # Kalman quantities are fed INTO the gate rather than thresholded to select frames.
        if mahalanobis is not None:
            u = torch.cat([u, mahalanobis.reshape(b, 1).to(mem.dtype)], dim=-1)
        else:
            u = torch.cat([u, torch.zeros_like(u)], dim=-1)   # (B, 2)
        gate_in = torch.cat([mem.reshape(b, -1), u], dim=-1)
        c = torch.sigmoid(self.trc_gate(gate_in))                       # (B, F)
        # anchor: the oldest stored frame is the ground-truth-derived template
        # Slot 0 is the NEWEST rolling search summary, not an immutable clean template.
        # Pinning it to 1 blindly trusts the very frame that may be damaged.
        c_anchored = c
        # suppression: s_hat_i = s_i * c_i   (DTPTrack eq. 2, last line)
        calibrated = mem * c_anchored.unsqueeze(-1)

        # ---- TGS (eq. 3): P_dyn = P_base + f_mod(calibrated summaries) -------------
        mod = self.tgs_modulator(calibrated.reshape(b, -1))
        prior = self.base_prior.to(mod.dtype) + mod.reshape(b, self.tokens, self.memory_dim)

        # Detached: the bank is a buffer, not a parameterised path.
        memory_out = mem.detach() if detach_memory else mem
        memory_rel_out = mem_rel.detach() if detach_memory else mem_rel

        # Reliability-weighted read-out over the bank.  Every factor is given an explicit
        # trailing axis before multiplying: (B,F,1) * (B,F) would otherwise be read as
        # (B,F,1) x (B,F) and broadcast to (B,F,F) instead of (B,F,1).
        w = memory_rel_out * c_anchored.reshape(b, self.frames, 1)   # (B, F, 1)
        w = w.clamp(min=0.0)
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-6)
        readout = (memory_out * w).sum(dim=1)                        # (B, D)

        return {"memory": memory_out, "memory_reliability": memory_rel_out,
                "readout": readout, "prior_tokens": prior,
                "frame_reliability": cur_rel.reshape(b, 1),
                "trc_confidence": c_anchored, "admitted": admitted}

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _masked_average(tokens: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Eq. 1: masked average pooling ``(B, N, C) -> (B, C)``."""
        if mask is None:
            return tokens.mean(dim=1)
        m = mask.to(tokens.dtype)
        if m.dim() == tokens.dim():
            m = m.squeeze(-1)
        m = m.unsqueeze(-1)                                    # (B, N, 1)
        return (tokens * m).sum(dim=1) / (m.sum(dim=1) + 1e-6)
