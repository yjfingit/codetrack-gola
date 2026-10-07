"""CodeTrack -- full assembly (architecture figure blocks 3-6).

Wiring, exactly as the figure:

    GOLA blocks -> final LayerNorm -> F_L
        |
        +-- Z_RGB  = F_L[:, 0*z_len : 1*z_len]        initial template, visible
        +-- X_RGB  = F_L[:, 1*z_len : 2*z_len]        search, visible   <- X_aux
        +-- Z_TIR  = F_L[:, 2*z_len : 3*z_len]        initial template, infrared
        +-- X_TIR  = F_L[:, 3*z_len : 4*z_len]        search, infrared  <- X_t (baseline)
        +-- Z_on   = F_L[:, 4*z_len : 5*z_len]        online template
        +-- D_TIR  = F_L[:, 5*z_len : 6*z_len]        online template, infrared
        |
        +-- (4) Motion prior  -> M_t (B,1,16,16) + u_t + Mahalanobis distance
        |       TRC/TGS temporal memory (DTPTrack) -> reliability-weighted prior tokens
        |
        +-- (3) ECC diagnosis   X_t, X_aux -> H_bar -> s (B,64) -> q (B,256)
        |
        +-- (5) Selective recovery  TopK(q) -> H-routed sparse attention -> X''
        |       then noise-modulated 2-step denoising -> X_t'
        |
        +-- (6) Template protection  c_t gates the score>0.84 rule
        v
    original GOLA anchor-free head(X_t') -> score_map, boxes

``X_t`` is byte-for-byte the tensor upstream GOLA hands to its head, so with the
recovery branch zero-initialised the loaded checkpoint reproduces the original output at
step 0.  (The figure draws an extra LayerNorm before CodeTrack; upstream ``_fusion``
already applies ``self.norm`` to ``F_L``, so inserting another normalisation would change
the head's input scale and break ``X_t' = X_t`` at initialisation.  Per the stated
priority -- real GOLA constraints over the figure -- CodeTrack consumes the
already-normalised ``F_L`` directly.  Recorded in docs/setup.md.)
"""

from __future__ import annotations

import math
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CodeTrackConfig
from .ecc import ParityCheckMatrix, SyndromeDiagnosis, NeuralBPSyndromeDiagnosis
from .motion import KalmanMotionPrior, TemporalMemory
from .recovery import SATRRecovery
from .template import TemplateProtectionGate


def _merge_corruption_flags(token_flag: Optional[torch.Tensor],
                            image_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Frame-level "this input was damaged" flag, from token *or* image corruption.

    ``token_flag`` is per-sample bool from the token-level corruption draw; ``image_mask`` is
    (B, 6) from the data plugin.  Either can be absent.  Returning ``None`` when both are
    absent preserves the previous behaviour (the reliability losses are simply skipped).
    """
    flags = None
    if token_flag is not None:
        flags = token_flag.reshape(token_flag.shape[0]).to(torch.bool)
    if image_mask is not None and torch.is_tensor(image_mask):
        img = (image_mask.reshape(image_mask.shape[0], -1).abs().sum(dim=-1) > 0)
        flags = img if flags is None else (flags | img.to(flags.device))
    return flags


class CodeTrack(nn.Module):
    """Per-frame CodeTrack branch, driven by the GOLA forward pass."""

    def __init__(self, cfg: CodeTrackConfig, head: Optional[nn.Module] = None):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.dim = cfg.dim
        self.z_len = cfg.z_len
        self.x_len = cfg.x_len
        self.grid = cfg.grid

        # ---- block 3: parity-check matrix + syndrome --------------------------
        self.H = ParityCheckMatrix(
            num_checks=cfg.num_checks, num_variables=cfg.x_len,
            links_per_check=cfg.h_links_per_check,
            min_col_degree=cfg.h_min_col_degree, grid=cfg.grid,
            locality_window=cfg.h_locality_window, locality_wrap=cfg.h_locality_wrap,
            free_edge_frac=cfg.h_free_edge_frac, seed=cfg.h_seed,
            layout=cfg.h_layout)
        diagnosis_cls = (NeuralBPSyndromeDiagnosis
                         if cfg.decoder_type == "neural_bp" else SyndromeDiagnosis)
        diagnosis_kwargs = dict(
            dim=cfg.dim, mid_dim=cfg.mid_dim, num_checks=cfg.num_checks,
            num_variables=cfg.x_len, syndrome_hidden=cfg.syndrome_hidden,
            detection_prior=cfg.detection_prior)
        if diagnosis_cls is SyndromeDiagnosis:
            diagnosis_kwargs.update(use_cos=cfg.syndrome_cos,
                                    syndrome_logit_gain=cfg.syndrome_logit_gain,
                                    syndrome_gain_calibration=cfg.syndrome_gain_calibration)
        else:
            diagnosis_kwargs.update(bp_iterations=cfg.bp_iterations,
                                    bp_damping=cfg.bp_damping,
                                    explicit_syndrome_weight=cfg.explicit_syndrome_weight,
                                    center_logits=bool(getattr(cfg, "center_bp_logits", False)))
        self.diagnosis = diagnosis_cls(**diagnosis_kwargs)
        if bool(getattr(cfg, "freeze_vote_bias", False)) and hasattr(self.diagnosis, "vote_bias"):
            # Keep the channel prior fixed while learning syndrome evidence. Otherwise the
            # decoder can lower every q by moving one global bias instead of learning which
            # tokens are unreliable.
            self.diagnosis.vote_bias.requires_grad_(False)
        # variable -> variable relation used for H-routing (from the same incidence)
        # ``persistent=False`` on purpose: upstream ``GOLA_DINOv2.state_dict`` walks every
        # key and calls ``get_parameter`` on it, which raises for a persistent buffer it does
        # not know about.  The index is a pure function of ``H_support`` (itself a persistent
        # buffer), so it is rebuilt on load rather than stored.
        self.register_buffer("neighbour_index",
                             self.H.neighbour_index(cfg.num_neighbours), persistent=False)

        # ---- block 4: motion prior + temporal memory --------------------------
        self.motion = KalmanMotionPrior(
            grid=cfg.grid, state_dim=cfg.motion_state_dim,
            process_noise=cfg.motion_process_noise,
            measurement_noise=cfg.motion_measurement_noise,
            gate_hidden=cfg.motion_gate_hidden) if cfg.motion_enabled else None
        self.memory = TemporalMemory(
            dim=cfg.dim, frames=cfg.memory_frames, tokens=cfg.memory_tokens,
            memory_dim=cfg.memory_dim) if cfg.memory_enabled else None

        # template pooling: initial (v+i) and online (v, i)
        self.template_pool = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim), nn.GELU(), nn.Linear(cfg.dim, cfg.dim))

        # ---- block 5: SATR Tanner recovery ------------------------------------
        self.satr = SATRRecovery(
            dim=cfg.dim, hidden=cfg.refiner_hidden, heads=cfg.refiner_heads,
            num_neighbours=cfg.num_neighbours, memory_dim=cfg.memory_dim,
            dropout=cfg.refiner_dropout, residual_gate_init=cfg.residual_gate_init,
            motion_bias_scale=cfg.motion_bias_scale, up_init_std=cfg.up_init_std,
            motion_bias_normalise=cfg.motion_bias_normalise, rounds=cfg.satr_rounds,
            residual_clip_ratio=cfg.residual_clip_ratio,
            cross_modal_anchor_scale=cfg.cross_modal_anchor_scale)

        # SATR owns all correction rounds. There is no dense denoiser/diffusion path.

        # ---- block 6: template protection -------------------------------------
        self.template_gate = TemplateProtectionGate(
            hidden=cfg.template_gate_hidden) if cfg.template_protection else None

        # ``head`` is the *original* GOLA head, borrowed (not copied) so that the
        # CodeTrack path and the baseline path share exactly one head instance.
        self.head = head
        self._last_decision: Optional[Dict[str, torch.Tensor]] = None
        # per-sequence state (also cleared by ``reset_sequence``)
        self._state: Dict[str, Optional[torch.Tensor]] = {"memory": None, "memory_rel": None}
        self._prev_score: Optional[torch.Tensor] = None
        self._prev_box: Optional[torch.Tensor] = None
        # Previous frame's OWN decoded box/score.  At inference there is no ground truth,
        # so this is the only legitimate observation the Kalman filter can be given, and
        # using the previous frame (not the current prediction) keeps it causal.
        self._prev_admitted = False
        self._gate_inputs = None

        self._motion_crop_params = None
        self._motion_crop_size = None
        self._motion_observe_current = False

    # ------------------------------------------------------------------ helpers
    def reset_sequence(self) -> None:
        """Clear per-sequence recurrent state (Kalman filter + memory bank)."""
        if self.motion is not None:
            self.motion.reset_state()
        self._state: Dict[str, Optional[torch.Tensor]] = {"memory": None, "memory_rel": None}
        self._last_decision = None
        # score of the *previous* accepted frame, used for the memory-admission rule below
        self._prev_score: Optional[torch.Tensor] = None
        self._prev_box = None
        self._prev_admitted = False
        self._gate_inputs = None
        if self.memory is not None:
            self.memory._admitted_once = False
        self._motion_crop_params = None
        self._motion_crop_size = None
        self._motion_observe_current = False

    def _split(self, F_L: torch.Tensor) -> Dict[str, torch.Tensor]:
        z, x = self.z_len, self.x_len
        if F_L.shape[1] != 4 * z + 2 * x:
            raise ValueError(f"Expected {4 * z + 2 * x} fused tokens, got {F_L.shape[1]}")
        return {
            "Z_RGB": F_L[:, 0 * z: 1 * z],
            "X_RGB": F_L[:, 1 * z: 1 * z + x],
            "Z_TIR": F_L[:, z + x: 2 * z + x],
            "X_TIR": F_L[:, 2 * z + x: 2 * z + 2 * x],
            "Z_on": F_L[:, 2 * z + 2 * x: 3 * z + 2 * x],
            "D_TIR": F_L[:, 3 * z + 2 * x: 4 * z + 2 * x],
        }

    def _search_target_mask(self, gt_box_xywh: torch.Tensor, image_size: torch.Tensor,
                            batch: int) -> torch.Tensor:
        """Rasterise the observed box onto the ``grid x grid`` search-token lattice.

        The SiamFC search crop is centred on the observation and scaled so that the crop
        side equals ``sqrt(area_factor * box_area)`` with ``area_factor = 4``.  Because the
        crop is centred on the box, the box centre lands on the lattice centre and its extent
        in grid cells follows directly from the crop side.  The result is used **only** as
        the masked-pooling mask of DTPTrack eq. 1; it never reaches the tracking head.
        """
        size = image_size.to(gt_box_xywh.dtype).reshape(batch, 2)         # (B, 2) W,H
        whn = (gt_box_xywh[:, 2:] / size).clamp(min=1e-4)                 # (B, 2) normalised
        # crop side (normalised) and the box extent measured in grid cells
        crop_side = torch.sqrt(4.0 * whn[:, 0] * whn[:, 1]).clamp(min=1e-6)   # (B,)
        gx = torch.full_like(whn[:, 0], self.grid * 0.5)                  # (B,)
        gy = torch.full_like(whn[:, 1], self.grid * 0.5)
        bw = whn[:, 0] / crop_side * self.grid
        bh = whn[:, 1] / crop_side * self.grid

        x1 = (gx - bw / 2).floor().clamp(0, self.grid - 1)
        y1 = (gy - bh / 2).floor().clamp(0, self.grid - 1)
        x2 = (gx + bw / 2).ceil().clamp(1, self.grid)
        y2 = (gy + bh / 2).ceil().clamp(1, self.grid)

        lin = torch.arange(self.grid, device=gt_box_xywh.device, dtype=gt_box_xywh.dtype)
        yy, xx = torch.meshgrid(lin, lin, indexing="ij")
        xx = xx.reshape(1, -1)
        yy = yy.reshape(1, -1)
        inside = ((xx >= x1.unsqueeze(-1)) & (xx < x2.unsqueeze(-1))
                  & (yy >= y1.unsqueeze(-1)) & (yy < y2.unsqueeze(-1)))
        inside = inside.to(gt_box_xywh.dtype)                            # (B, grid*grid)

        # A degenerate box would yield an all-zero mask that the pooling cannot normalise;
        # fall back to the single lattice cell nearest the centre.
        empty = inside.sum(dim=-1, keepdim=True) == 0
        if bool(empty.any()):
            centre_idx = ((gy.floor().clamp(0, self.grid - 1)) * self.grid
                          + gx.floor().clamp(0, self.grid - 1)).long().unsqueeze(-1)
            ones = torch.ones_like(centre_idx, dtype=inside.dtype)
            inside = inside.scatter(1, centre_idx, ones)
        return inside

    def notify_tracking_score(self, score: torch.Tensor,
                              box_xywh: Optional[torch.Tensor] = None) -> None:
        """Publish this frame's head output for the *next* frame to consume.

        ``score`` drives memory admission; ``box_xywh`` additionally becomes the Kalman
        observation on the next frame.  Both are detached: they are inference-time selection
        statistics, not loss terms.  They are recorded rather than used immediately because the
        head runs *after* CodeTrack, so neither is available while CodeTrack is executing.
        """
        self._prev_score = score.detach().reshape(score.shape[0])
        if box_xywh is not None:
            self._prev_box = box_xywh.detach().reshape(box_xywh.shape[0], 4)
            if self._motion_observe_current and self.motion is not None:
                # The caller supplies the accepted current-frame prediction AFTER
                # recovery. The next prior includes it without a one-frame lag.
                self.motion.observe(self._prev_box, self._motion_crop_size,
                                    confidence=self._prev_score, predict=False)
                self._motion_observe_current = False
        # The current score is only available AFTER recovery and the tracking head.
        # Refresh the evaluation decision now, using this frame's diagnostic evidence.
        if not self.training and self.template_gate is not None and self._gate_inputs is not None:
            with torch.no_grad():
                decision = self.template_gate(score=self._prev_score, **self._gate_inputs)
                self._last_decision = {
                    "c_t": decision["c_t"], "score": self._prev_score,
                    # The updater multiplies score by quality; a binary quality implements
                    # the intended conjunction score > threshold AND c_t > tau exactly.
                    "confidence": (decision["c_t"] > self.cfg.template_threshold).to(score.dtype),
                    "update": TemplateProtectionGate.should_update(
                        decision["c_t"], self._prev_score,
                        self.cfg.gola_update_threshold, self.cfg.template_threshold),
                }

    # ------------------------------------------------------------------ forward
    def forward(self, F_L: torch.Tensor,
                gt_box_xywh: Optional[torch.Tensor] = None,
                image_size: Optional[torch.Tensor] = None,
                tracking_score: Optional[torch.Tensor] = None,
                box_confidence: Optional[torch.Tensor] = None,
                eval_observe: bool = False,
                corruption_mask: Optional[torch.Tensor] = None,
                was_corrupted: Optional[torch.Tensor] = None,
                image_corruption_mask: Optional[torch.Tensor] = None,
                observe_motion: bool = True,
                update_state: bool = True,
                preserve_state: bool = False,
                route_q_override: Optional[torch.Tensor] = None,
                search_crop_params: Optional[torch.Tensor] = None,
                **_: object) -> Dict[str, torch.Tensor]:
        """``F_L``: (B, 768, 768) normalised fused tokens from the GOLA forward.

        Supervision-side arguments (all optional, all detached inside):

        ``gt_box_xywh``  (B, 4) ground-truth box, used to drive the Kalman filter during
                         training.  ``box_confidence`` decides how much the filter trusts
                         it, so a ground-truth box cannot dictate a confident motion prior.
        ``corruption_mask`` (B, N) bool, the corruption the data pipeline applied.
        """
        b = F_L.shape[0]
        # Training pairs are sampled independently, so carrying Kalman/memory state from one
        # random batch into the next creates physically meaningless transitions and can make the
        # covariance solve unstable.  Recurrent state is reserved for causal sequence inference;
        # training still exercises the motion and memory heads on the current frame.
        # Pair training keeps the historical reset-on-call behavior.  A causal clip sampler
        # passes ``preserve_state=True`` so Kalman and temporal memory survive between adjacent
        # frames from the same sequence; the explicit boundary reset is then performed by the
        # sampler at clip start.
        if self.training and update_state and not preserve_state:
            self.reset_sequence()
        # Dynamic evaluation scheduling may emit a shorter final batch. All recurrent
        # and template-gate state is batch-shaped; carrying the previous layout causes
        # stale scores/boxes to be reshaped against the new batch.
        if self._prev_score is not None and self._prev_score.shape[0] != b:
            self.reset_sequence()
        toks = self._split(F_L)
        X_t, X_aux = toks["X_TIR"], toks["X_RGB"]        # baseline feature / aux evidence
        ablate_aux = os.environ.get("CODETRACK_ABLATE_AUX", "0") == "1"
        ablate_motion = os.environ.get("CODETRACK_ABLATE_MOTION", "0") == "1"
        ablate_memory = os.environ.get("CODETRACK_ABLATE_MEMORY", "0") == "1"
        ablate_bp = os.environ.get("CODETRACK_ABLATE_BP", "0") == "1"
        ablate_satr = os.environ.get("CODETRACK_ABLATE_SATR", "0") == "1"
        if ablate_aux:
            # Counterfactual: remove cross-modal side information while keeping the tracked
            # TIR stream and all other state identical.
            X_aux = torch.zeros_like(X_aux)
        # Optional controlled side-information attenuation.  Scaling around X_t keeps the
        # diagnosis input in the same feature distribution as a real modality, unlike setting
        # RGB to zero (which is an out-of-distribution intervention and can move the whole q
        # prior).  This is used for reliability ablations; the default path is unchanged.
        aux_blend = os.environ.get("CODETRACK_AUX_BLEND")
        if aux_blend is not None and not ablate_aux:
            blend = float(aux_blend)
            X_aux = X_t + blend * (X_aux - X_t)

        # ---- template context (initial + online, both modalities) --------------
        template_tokens = torch.cat(
            [toks["Z_RGB"], toks["Z_TIR"], toks["Z_on"], toks["D_TIR"]], dim=1)  # (B, 4z, C)
        template_ctx = self.template_pool(template_tokens.mean(dim=1))           # (B, C)

        # ---- block 4: motion prior -------------------------------------------
        motion_out: Dict[str, torch.Tensor] = {}
        pending_obs: Optional[torch.Tensor] = None
        pending_conf: Optional[torch.Tensor] = None
        if self.motion is not None and not ablate_motion:
            crop_aware = search_crop_params is not None
            if crop_aware:
                if image_size is None:
                    raise ValueError("crop-aware motion requires image_size")
                params = search_crop_params.detach().to(X_t).reshape(b, 2, 2)
                if self._motion_crop_params is not None:
                    old = self._motion_crop_params.to(params)
                    old_size = self._motion_crop_size.to(image_size)
                    ratio = params[:, 0] / old[:, 0]
                    scale = ratio * old_size / image_size
                    shift = (params[:, 1] - ratio * old[:, 1]) / image_size
                    self.motion.rebase(scale, shift)
                self._motion_crop_params = params
                self._motion_crop_size = image_size.detach()
                self._motion_observe_current = True
            # ---- D2: predict-only, then observe AFTER this frame is consumed --------
            # The order here is what makes the block causal.  ``observe()`` used to run
            # *before* the prior map was built, so M_t was a function of the current frame's
            # ground-truth box -- ground truth entering the current output instead of only the
            # loss.  Now the filter predicts from x_{t|t-1} (strictly past information), the
            # prior is built from that prediction, and the observation is absorbed at the end
            # of the forward so the state is ready for frame t+1.
            admitted = None
            admitted_conf = box_confidence
            if observe_motion and gt_box_xywh is not None and image_size is not None:
                admitted = gt_box_xywh
            elif box_confidence is not None and gt_box_xywh is not None:
                admitted = gt_box_xywh
            elif eval_observe and self._prev_box is not None and not crop_aware:
                # Inference: the previous frame's own prediction is the only legitimate
                # observation available (there is no ground truth while tracking).
                admitted = self._prev_box
                admitted_conf = self._prev_score
            pending_obs, pending_conf = admitted, admitted_conf
            motion_out = self.motion(
                box_xywh=admitted, image_size=image_size,
                confidence=admitted_conf, valid=None,
                batch_size=b, device=X_t.device, dtype=X_t.dtype,
                defer_observe=True, advance=crop_aware)
        motion_map = motion_out.get("motion_map")
        uncertainty = motion_out.get("uncertainty")

        # Normalise the motion prior to unit mass.  The map is consumed both as a log-space
        # attention bias and as a distribution for the KL term, and normalising here means the
        # prior head cannot smuggle information through a global scale factor (which the
        # consumer would cancel anyway).
        #
        # The *supervision target* is deliberately NOT built here: the training wrapper passes
        # ``targets`` only to the criterion, so this branch never has a ground-truth box on a
        # real step and used to yield ``None`` -- which is why ``Loss/motion`` never appeared.
        # The criterion builds it instead (``codetrack/criteria.py::_motion_target_map``).
        motion_target = None
        if motion_map is not None:
            pm = motion_map.reshape(b, -1)
            pm = pm / pm.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            motion_map = pm.reshape(b, 1, self.grid, self.grid)

        # ---- block 3: ECC diagnosis ------------------------------------------
        H_bar = self.H.matrix()
        diag = self.diagnosis(X_t, X_aux, H_bar, template_context=template_ctx)
        out_s = diag["s"]
        q = diag["q"]
        if route_q_override is not None:
            if route_q_override.shape != q.shape:
                raise ValueError(
                    f"route_q_override shape {tuple(route_q_override.shape)} != q shape {tuple(q.shape)}")
            # Oracle routing is a diagnostic intervention only. It changes the SATR route while
            # leaving the learned syndrome/q outputs available for comparison and never becomes
            # part of a production checkpoint.
            q = route_q_override.to(dtype=q.dtype, device=q.device)
        if ablate_bp and "symbol_logits" in diag:
            # Counterfactual decoder: retain the learned per-token symbol evidence but remove
            # Tanner check-to-variable messages. This isolates BP from the syndrome encoder.
            q = torch.sigmoid(diag["symbol_logits"] + self.diagnosis.vote_bias.view(1, -1))
            diag["q"] = q
            diag["bp_messages"] = None

        # ---- block 4b: temporal memory ---------------------------------------
        # `reliability` = 1 - q: a token the diagnosis considers healthy is reliable.
        reliability = 1.0 - q
        memory_readout = None
        prior_tokens = None
        if self.memory is not None and not ablate_memory:
            mem_state = getattr(self, "_state", {"memory": None, "memory_rel": None})
            # Target mask on the search grid, derived from the observed box.  It is only a
            # *pooling* mask for eq. 1, never a loss target, and a ground-truth box cannot
            # leak into the head through it (the head sees X_final, which the pooled summary
            # reaches only through the reliability gate).
            target_mask = None
            # Recovery must see only observable frame/history features.  The target box
            # remains available to the criterion as a label, but never controls this readout.
            target_mask = None
            # Without an observed box, pool using predicted reliability inside memory.
            # The injector's mask is a supervision label, never a recovery input.
            # Admission uses the *previous* frame's tracking score.  The head has not run
            # yet for this frame, so this is the causally correct signal -- and it matches the
            # evaluation pipeline, where the memory decision is taken after the previous frame
            # has been scored.  During training every frame is admitted, so the memory bank
            # sees the true temporal order instead of a partially frozen one.
            admit = bool(self.training) or self._prev_score is None
            mem = self.memory(X_t, reliability, mem_state.get("memory"),
                              mem_state.get("memory_rel"),
                              target_mask=target_mask, uncertainty=uncertainty,
                              admission_score=self._prev_score, admit=admit,
                              mahalanobis=motion_out.get("mahalanobis"))
            if update_state:
                self._state = {"memory": mem["memory"], "memory_rel": mem["memory_reliability"]}
            memory_readout = mem["readout"]
            prior_tokens = mem["prior_tokens"]
            memory_info = mem
        else:
            memory_info = {}

        # ---- block 5a: H-routed sparse recovery ------------------------------
        # The temporal guidance enters as dedicated *prior tokens* (DTPTrack's TGS output),
        # not as a pooled vector: their ablation shows a decoupled prior-token channel beats
        # concatenating the summaries into the visual token stream (-0.9 AUC), because it
        # guides without contaminating the raw features.
        # Recovery is a relative localization task. The calibrated probability level is useful
        # to the diagnosis loss, but its clean and damaged frame means are nearly identical;
        # subtracting a scalar prior either writes everywhere or nowhere. Standardize within
        # each frame and restrict writes to the declared TopK support. This preserves q's raw
        # probability semantics while making the recovery gate depend only on evidence ranking.
        q_mu = q.mean(dim=-1, keepdim=True)
        q_sd = q.std(dim=-1, keepdim=True, unbiased=False).clamp(min=1e-6)
        q_relative = torch.sigmoid((q - q_mu) / q_sd)
        soft_route = bool(self.training and getattr(self.cfg, "soft_route_training", False))
        if os.environ.get("CODETRACK_SOFT_ROUTE", "0") == "1":
            soft_route = bool(self.training)
        route_k = q.shape[-1] if soft_route else min(
            int(os.environ.get("CODETRACK_TOPK_TOKENS", self.cfg.topk_tokens)), q.shape[-1])
        route_score = q
        motion_route_scale = float(os.environ.get("CODETRACK_MOTION_ROUTE_SCALE", str(self.cfg.motion_route_scale)))
        if (not self.training) and motion_map is not None and motion_route_scale != 0.0:
            mp = motion_map.reshape(b, -1)
            mp = mp / mp.amax(dim=-1, keepdim=True).clamp_min(1e-6)
            route_score = q * (1.0 + motion_route_scale * mp)
        if soft_route:
            route_idx = torch.arange(q.shape[-1], device=q.device).view(1, -1).expand(b, -1)
        else:
            route_idx = route_score.topk(route_k, dim=-1).indices
        # Optional target-cell redundancy route.  The original GOLA head supplies the current
        # best signal location; including it guarantees that the ECC branch can actually affect
        # the token consumed by the tracking decision when syndrome ranking misses that cell.
        # This is disabled by default and is evaluated as a causal routing ablation.
        if (not self.training) and os.environ.get("CODETRACK_INCLUDE_BASE_CELL", "0") == "1" \
                and self.head is not None:
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                base_map = self.head(X_t.float())["score_map"].float().flatten(1)
                base_cell = base_map.argmax(dim=-1)
            if route_k > 0:
                route_idx[:, -1] = base_cell
        # Recompute route scores after the optional union so gather/scatter use the same support.
        route_support = torch.zeros_like(q)
        route_support.scatter_(1, route_idx, 1.0)
        # Inference uses an abstention rule from reliable communication: if the
        # detector has no absolute evidence of an error, decode nothing and preserve
        # the original GOLA token exactly. Training keeps soft Top-K routing so the
        # detector and SATR receive gradients.
        abstain_enabled = bool(self.cfg.abstain_enabled) or os.environ.get(
            "CODETRACK_ABSTAIN_THRESHOLD") is not None
        abstain_threshold = float(os.environ.get(
            "CODETRACK_ABSTAIN_THRESHOLD", self.cfg.abstain_threshold))
        if (not self.training) and abstain_enabled:
            route_support = route_support * (q >= abstain_threshold).to(q.dtype)
        adaptive_route = bool(getattr(self.cfg, "adaptive_route_enabled", False)) or \
            os.environ.get("CODETRACK_ADAPTIVE_ROUTE", "0") == "1"
        if (not self.training) and adaptive_route:
            # q is a posterior, but its absolute calibration can drift by sequence.  The second
            # cut is therefore relative to the current frame: only tokens that are both above
            # the absolute reliability threshold and z standard deviations above the frame mean
            # are decoded.  `route_k` remains a hard safety budget, not a fixed write count.
            z = float(os.environ.get("CODETRACK_ADAPTIVE_ROUTE_Z",
                                    str(getattr(self.cfg, "adaptive_route_z", 1.0))))
            q_cut = torch.maximum(
                q.new_full((b, 1), abstain_threshold),
                q.mean(dim=-1, keepdim=True) + z * q.std(dim=-1, keepdim=True, unbiased=False))
            route_support = route_support * (q >= q_cut).to(q.dtype)
        # Optional out-of-view safeguard.  A Kalman prediction whose box is close to the
        # search-region boundary is not reliable side information: carrying history into that
        # frame can manufacture a target after it has left the image.  The margin is expressed
        # as a fraction of the search-region width/height and is disabled by default.
        edge_margin = float(os.environ.get(
            "CODETRACK_EDGE_REJECT_MARGIN", str(self.cfg.edge_reject_margin)))
        if (not self.training) and edge_margin > 0.0 and image_size is not None \
                and motion_out.get("motion_box") is not None:
            mb = motion_out["motion_box"]
            sz = image_size.to(mb.dtype)
            x1, y1 = mb[:, 0], mb[:, 1]
            x2, y2 = x1 + mb[:, 2], y1 + mb[:, 3]
            margin_x, margin_y = sz[:, 0] * edge_margin, sz[:, 1] * edge_margin
            edge_risk = (x1 < margin_x) | (y1 < margin_y) | \
                        (x2 > sz[:, 0] - margin_x) | (y2 > sz[:, 1] - margin_y)
            route_support = route_support * (~edge_risk).to(q.dtype).unsqueeze(-1)

        # Reuse the trained template-trust gate as an optional frame-level decode gate.  This
        # keeps one reliability estimator for both template admission and ECC decoding.  The
        # current-frame recovery confidence is unavailable before SATR, so it is set to zero;
        # q, the previous score, and motion uncertainty remain causal inputs.  Disabled unless
        # explicitly requested for an ablation or a calibrated deployment run.
        decode_gate = None
        decode_gate_threshold = os.environ.get("CODETRACK_DECODE_GATE_THRESHOLD")
        decode_gate_enabled = bool(self.cfg.decode_gate_enabled) or decode_gate_threshold is not None
        if (not self.training) and decode_gate_enabled and self.template_gate is not None:
            gate_score = tracking_score if tracking_score is not None else self._prev_score
            if gate_score is None:
                gate_score = q.new_zeros(b)
            decode_gate = self.template_gate(
                score=gate_score.detach(), q=q, uncertainty=uncertainty,
                recovery_confidence=q.new_zeros(b), topk=self.cfg.topk_tokens)["c_t"]
            route_support = route_support * (
                decode_gate >= float(self.cfg.decode_gate_threshold if decode_gate_threshold is None
                                     else decode_gate_threshold)).to(q.dtype).unsqueeze(-1)
        qmax_threshold = os.environ.get("CODETRACK_QMAX_GATE_THRESHOLD")
        if (not self.training) and qmax_threshold is not None:
            frame_has_evidence = q.max(dim=-1).values >= float(qmax_threshold)
            route_support = route_support * frame_has_evidence.to(q.dtype).unsqueeze(-1)
        # Optional target-existence safeguard.  A parity violation is not sufficient evidence
        # for decoding when the base tracker no longer sees a target: history/Kalman can still
        # explain the syndrome after an object leaves the search image.  This gate is deliberately
        # based on the *unmodified* GOLA response and is only enabled for calibrated inference
        # probes, so it cannot leak labels into training or silently change the default recipe.
        response_threshold = os.environ.get("CODETRACK_RESPONSE_GATE_THRESHOLD")
        base_peak_for_route = None
        if (not self.training) and response_threshold is not None and self.head is not None:
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                base_for_route = self.head(X_t.float())
                base_peak_for_route = base_for_route["score_map"].float().flatten(1).sigmoid().amax(dim=-1)
            route_support = route_support * (
                base_peak_for_route >= float(response_threshold)).to(q.dtype).unsqueeze(-1)
        # Soft training uses calibrated probabilities directly.  Relative standardisation is
        # useful for a fixed inference budget but destroys absolute q semantics during fitting.
        if soft_route:
            # The configured detection prior is the no-error operating point.  During fitting,
            # routing the raw q would write a nonzero residual into every token at initialization
            # (q starts at the prior, usually 0.2).  A quadratic soft gate keeps that path close
            # to identity while retaining a dense gradient; inference still uses the calibrated
            # q-relative Top-K route below.
            q_route = q.square() * route_support
        else:
            q_route = q_relative * route_support
        if ablate_satr:
            rec = {"X_rec": X_t, "suspect_index": route_idx,
                   "suspect_score": q_route.gather(1, route_idx),
                   "delta": torch.zeros(b, route_k, self.dim, device=X_t.device, dtype=X_t.dtype),
                   "alpha": q_route.gather(1, route_idx)}
        else:
            reliability_scale = float(os.environ.get("CODETRACK_TANNER_RELIABILITY_SCALE", "0.0"))
            rec = self.satr(
                X_t=X_t, X_aux=X_aux, q=q_route, neighbour_index=self.neighbour_index,
                H_bar=H_bar, bp_messages=diag.get("bp_messages"), syndrome=out_s,
                template_pool=template_ctx, memory_readout=prior_tokens,
                motion_map=motion_map, topk=route_k,
                neighbour_q=q if reliability_scale != 0.0 else None,
                reliability_bias_scale=reliability_scale)
        X_rec = rec["X_rec"]
        # Optional Tanner propagation write: distribute a small fraction of each decoded
        # residual to variables sharing a check with the suspect. This is disabled by default;
        # it tests whether the head-relevant error is a propagated token rather than the q-ranked
        # source itself.
        spread = float(os.environ.get("CODETRACK_TANNER_SPREAD", "0.0"))
        if (not self.training) and spread != 0.0 and not ablate_satr:
            support = (H_bar > 0).to(X_t.dtype)
            shared = (support.t() @ support) > 0
            shared.fill_diagonal_(False)
            for jj in range(rec["suspect_index"].shape[1]):
                src = rec["suspect_index"][:, jj]
                src_delta = rec["delta"][:, jj] * spread
                for bb in range(b):
                    nb_idx = torch.where(shared[src[bb]])[0]
                    if nb_idx.numel() == 0:
                        continue
                    X_rec[bb, nb_idx] = X_rec[bb, nb_idx] + src_delta[bb] / float(nb_idx.numel())
        # Optional token-level repair gate.  It evaluates each sparse candidate against the
        # frozen GOLA response before committing it.  This is the inference counterpart of the
        # causal repair-gain probe: a frame-level peak check can hide one bad token behind one
        # good token, whereas this gate only writes candidates that do not lower the baseline
        # response. Disabled by default because it costs K extra head calls.
        token_accept = os.environ.get("CODETRACK_TOKEN_ACCEPT", "0") == "1"
        token_accept_mask = None
        if (not self.training) and token_accept and self.head is not None and not ablate_satr:
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                base_pred = self.head(X_t.float())
                base_map = base_pred["score_map"].float().sigmoid()
                base_flat = base_map.flatten(1)
                base_cell = base_flat.argmax(dim=-1)
                base_score = base_flat.gather(1, base_cell[:, None]).squeeze(1)
                base_boxes = base_pred["boxes"].float().reshape(b, -1, 4)
                base_box = base_boxes.gather(1, base_cell[:, None, None].expand(-1, 1, 4)).squeeze(1)
                gated = X_t.clone()
                token_accept_mask = torch.zeros_like(rec["suspect_score"], dtype=torch.bool)
                for jj in range(rec["suspect_index"].shape[1]):
                    cand = gated.clone()
                    bj = torch.arange(b, device=X_t.device)
                    ij = rec["suspect_index"][:, jj]
                    cand[bj, ij] = X_rec[bj, ij]
                    cand_pred = self.head(cand.float())
                    cand_map = cand_pred["score_map"].float().sigmoid().flatten(1)
                    cand_score = cand_map.gather(1, base_cell[:, None]).squeeze(1)
                    cand_box = cand_pred["boxes"].float().reshape(b, -1, 4).gather(
                        1, base_cell[:, None, None].expand(-1, 1, 4)).squeeze(1)
                    # Target-cell gate: preserve the causal location selected by the original
                    # tracker and reject geometric jumps. This avoids the failure of comparing
                    # only the global peak, which can move to a distractor after one token write.
                    box_delta = (cand_box - base_box).abs().mean(dim=-1)
                    take = (cand_score >= base_score) & (box_delta <= 0.05)
                    gated[bj, ij] = torch.where(take[:, None], X_rec[bj, ij], gated[bj, ij])
                    token_accept_mask[:, jj] = take
                X_rec = gated
        recovery_scale = float(os.environ.get("CODETRACK_RECOVERY_SCALE", "1.0"))
        if (not self.training) and recovery_scale != 1.0:
            X_rec = X_t + recovery_scale * (X_rec - X_t)
        # Selective correction safeguard inspired by selective propagation: accept a decoded
        # feature only when the original tracking head's peak response does not decrease.
        # This is inference-only and disabled by default; it prevents a low-confidence SATR
        # update from replacing a usable baseline prediction.
        accept_delta = os.environ.get("CODETRACK_ACCEPT_SCORE_DELTA")
        accept_enabled = bool(self.cfg.accept_enabled) or accept_delta is not None
        if (not self.training) and accept_enabled and self.head is not None:
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                base_head = (self.head(X_t.float()) if base_peak_for_route is None
                             else base_for_route)
                rec_head = self.head(X_rec.float())
                base_peak = base_head["score_map"].float().flatten(1).sigmoid().amax(dim=-1)
                rec_peak = rec_head["score_map"].float().flatten(1).sigmoid().amax(dim=-1)
                accept = rec_peak >= base_peak + float(
                    self.cfg.accept_score_delta if accept_delta is None else accept_delta)
            X_rec = torch.where(accept.view(b, 1, 1), X_rec, X_t)

        # SATR has already completed its sparse Tanner message rounds. No dense
        # denoiser or diffusion rewrite follows the correction.
        suspect = rec["suspect_index"]
        X_final = X_rec
        den = {"X_denoised": X_final, "step_preds": []}
        mv = None

        # Post-decode parity evidence.  This is the visual analogue of checking whether a
        # channel decoder's output is back in the code space.  It is exposed for diagnostics and
        # can be enabled as a training loss; default inference behavior is unchanged.
        check_before = (diag["C_obs"] - diag["C_ref"]).pow(2).mean(dim=-1).sqrt()
        U_after = self.diagnosis.W_x(X_final)
        R_after = self.diagnosis.W_r(X_aux)
        if template_ctx is not None:
            R_after = R_after + self.diagnosis.template_ctx(template_ctx).unsqueeze(1)
        C_after = torch.einsum("mn,bnd->bmd", H_bar, U_after)
        C_ref_after = torch.einsum("mn,bnd->bmd", H_bar, R_after)
        check_after = (C_after - C_ref_after).pow(2).mean(dim=-1).sqrt()

        # ---- identity preservation on the tokens recovery was NOT asked to touch ----------
        # The anchor used to be ``~corruption_mask`` (the injector's ground truth), which does not
        # correspond to anything the branch actually does: the refiner rewrites only
        # ``TopK(q)`` suspects, so the two sets overlap only partially.  Measured consequence at
        # 300 optimizer updates: no token stayed put, ``Loss/pres`` rose from 5.6e-5 to 0.95, and
        # ``d_final`` (1-cos vs clean) went from 0.055 to 0.143 while ``d_input`` stayed at 0.117.
        # Anchoring on "the tokens the recovery did **not** select" makes the penalty describe the
        # branch's own contract: untouched tokens must come out untouched.
        preserve = None
        if X_final is not None:
            keep = None
            if suspect is not None and suspect.numel() > 0:
                bidx_p = torch.arange(b, device=X_t.device)[:, None].expand_as(suspect)
                keep = torch.ones(b, X_t.shape[1], dtype=torch.bool, device=X_t.device)
                keep[bidx_p, suspect] = False
            elif corruption_mask is not None:
                # fall back to the injector's mask when there is no routing decision to honour
                keep = ~corruption_mask
            if keep is not None and bool(keep.any()):
                preserve = (X_final[keep] - X_t[keep]).abs().mean()

        # ---- block 6: template protection ------------------------------------
        gate_out: Dict[str, torch.Tensor] = {}
        disable_template_gate = os.environ.get("CODETRACK_ABLATE_TEMPLATE_GATE", "0") == "1"
        if self.template_gate is not None and not disable_template_gate:
            score = tracking_score
            if score is None:
                score = self._prev_score
            if score is None:
                score = torch.zeros(b, device=X_t.device, dtype=X_t.dtype)
            # Fraction of tokens the diagnosis flags: a frame-level corruption summary that
            # is *differentiable*, so L_mem below can actually train the gate.  A detached
            # scalar here would leave the gate as dead parameters.
            corruption_fraction = q.mean(dim=-1).clamp(1e-6, 1 - 1e-6)
            recovery_conf = (1.0 - F.cosine_similarity(X_final, X_t, dim=-1, eps=1e-6)
                             ).mean(dim=-1)
            gate_out = self.template_gate(
                score=score.detach(), q=q, uncertainty=uncertainty,
                recovery_confidence=recovery_conf, topk=self.cfg.topk_tokens)
            if update_state:
                self._gate_inputs = dict(q=q.detach(),
                    uncertainty=None if uncertainty is None else uncertainty.detach(),
                    recovery_confidence=recovery_conf.detach(), topk=self.cfg.topk_tokens)
                self._last_decision = {
                    "c_t": gate_out["c_t"].detach(),
                    "score": score.detach(),
                    # ``confidence`` is the quantity the *evaluation pipeline* can consume.
                    # ``SimpleTemplateUpdater.update`` only receives a scalar confidence and
                    # compares it against the official 0.84 threshold, so exposing ``c_t``
                    # alone would be unusable.  ``mean_topk_q`` is 1 - (mean severity of the
                    # most suspicious tokens): it is high exactly when the frame's evidence is
                    # intact, which is the same notion of trust as ``c_t`` but on the same
                    # 0..1 scale as a tracking score, and it is a real output of block 3.
                    "confidence": (gate_out["c_t"] > self.cfg.template_threshold).to(score.dtype).detach(),
                    "update": TemplateProtectionGate.should_update(
                        gate_out["c_t"], score,
                        self.cfg.gola_update_threshold, self.cfg.template_threshold).detach(),
                }

        # ---- D2: absorb this frame's observation LAST ---------------------------
        # Everything above (the prior map, the recovery routing, the denoiser, the gate) has
        # already consumed x_{t|t-1}.  Updating the filter now leaves the state at x_{t|t} so
        # the *next* frame predicts from a posterior that includes this frame -- the correct
        # causal ordering.  ``pending_conf`` is the observation confidence, which inflates the
        # measurement noise, so a low-confidence box moves the state only slightly.
        if self.motion is not None and pending_obs is not None and image_size is not None:
            self.motion.observe(pending_obs, image_size,
                                confidence=pending_conf, valid=None,
                                predict=(search_crop_params is None))

        return {
            "X_final": X_final,
            "X_rec": X_rec,
            "q": q, "s": diag["s"], "s_logits": diag["s_logits"], "q_logits": diag["q_logits"],
            "q_prior": q.new_tensor(float(self.cfg.detection_prior)),
            "syndrome_pending_calibration": diag.get("syndrome_pending_calibration"),
            "C_obs": diag["C_obs"], "C_ref": diag["C_ref"],
            "syndrome_energy_before": check_before,
            "syndrome_energy_after": check_after,
            "U": diag["U"], "R": diag["R"], "H_bar": H_bar,
            "suspect_index": suspect, "suspect_score": rec["suspect_score"],
            "delta": rec["delta"], "alpha": rec["alpha"],
            "tokens": toks, "template_ctx": template_ctx,
            "motion": motion_out, "memory": memory_info, "memory_readout": memory_readout,
            "corruption_fraction": (q.mean(dim=-1) if self.template_gate is not None else None),
            # ``was_corrupted`` is the frame-level trust signal consumed by the memory
            # reliability head and the template gate.  Image-level corruption must count: the
            # data plugin corrupts the search crop, so a frame damaged only at image level was
            # previously reported as trustworthy, and every supervised module was told the wrong
            # thing about it.
            "was_corrupted": _merge_corruption_flags(was_corrupted, image_corruption_mask),
            "motion_map_norm": motion_map,
            "decode_gate": decode_gate,
            "motion_target": motion_target,
            "preserve": preserve,
            "token_accept_mask": token_accept_mask,
            "denoise_steps": den["step_preds"],
            **gate_out,
        }
