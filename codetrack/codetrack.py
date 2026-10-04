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
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CodeTrackConfig
from .ecc import ParityCheckMatrix, SyndromeDiagnosis
from .motion import KalmanMotionPrior, TemporalMemory
from .recovery import H_RoutedSparseRefiner, MeanVarCompletion, NoiseModulatedDenoiser
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
            free_edge_frac=cfg.h_free_edge_frac, seed=cfg.h_seed)
        self.diagnosis = SyndromeDiagnosis(
            dim=cfg.dim, mid_dim=cfg.mid_dim, num_checks=cfg.num_checks,
            num_variables=cfg.x_len, syndrome_hidden=cfg.syndrome_hidden,
            detection_prior=cfg.detection_prior, use_cos=cfg.syndrome_cos)
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

        # ---- block 5: recovery + denoising ------------------------------------
        self.refiner = H_RoutedSparseRefiner(
            dim=cfg.dim, hidden=cfg.refiner_hidden, heads=cfg.refiner_heads,
            num_neighbours=cfg.num_neighbours, memory_dim=cfg.memory_dim,
            dropout=cfg.refiner_dropout, residual_gate_init=cfg.residual_gate_init,
            motion_bias_scale=cfg.motion_bias_scale)

        # condition for the denoiser = [H-routed context (bottleneck) | aux tokens],
        # then projected back to the token dim so it can also be scattered into token
        # space for the *other* tokens (which is how the denoiser sees the evidence).
        self.condition_proj = nn.Linear(cfg.refiner_hidden + cfg.dim, cfg.dim)
        self.denoiser = NoiseModulatedDenoiser(
            dim=cfg.dim, hidden=cfg.diffusion_hidden, heads=cfg.diffusion_heads,
            steps=cfg.diffusion_steps, num_checks=cfg.num_checks, cond_dim=cfg.dim,
            noise_schedule_power=cfg.noise_schedule_power,
            residual_gate_init=cfg.residual_gate_init) if cfg.diffusion_enabled else None
        self.meanvar = MeanVarCompletion(dim=cfg.dim, hidden=cfg.diffusion_hidden)

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
        self._prev_box: Optional[torch.Tensor] = None
        self._prev_admitted = False

    # ------------------------------------------------------------------ helpers
    def reset_sequence(self) -> None:
        """Clear per-sequence recurrent state (Kalman filter + memory bank)."""
        if self.motion is not None:
            self.motion.reset_state()
        self._state: Dict[str, Optional[torch.Tensor]] = {"memory": None, "memory_rel": None}
        self._last_decision = None
        # score of the *previous* accepted frame, used for the memory-admission rule below
        self._prev_score: Optional[torch.Tensor] = None
        self._prev_admitted = False

    def _split(self, F_L: torch.Tensor) -> Dict[str, torch.Tensor]:
        z, x = self.z_len, self.x_len
        return {
            "Z_RGB": F_L[:, 0 * z: 1 * z],
            "X_RGB": F_L[:, 1 * z: 1 * z + x],
            "Z_TIR": F_L[:, 2 * z: 2 * z + z],
            "X_TIR": F_L[:, 2 * z + z: 2 * z + z + x],
            "Z_on": F_L[:, 3 * z + x: 3 * z + x + z],
            "D_TIR": F_L[:, 4 * z + x: 4 * z + x + z],
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
                **_: object) -> Dict[str, torch.Tensor]:
        """``F_L``: (B, 768, 768) normalised fused tokens from the GOLA forward.

        Supervision-side arguments (all optional, all detached inside):

        ``gt_box_xywh``  (B, 4) ground-truth box, used to drive the Kalman filter during
                         training.  ``box_confidence`` decides how much the filter trusts
                         it, so a ground-truth box cannot dictate a confident motion prior.
        ``corruption_mask`` (B, N) bool, the corruption the data pipeline applied.
        """
        b = F_L.shape[0]
        toks = self._split(F_L)
        X_t, X_aux = toks["X_TIR"], toks["X_RGB"]        # baseline feature / aux evidence

        # ---- template context (initial + online, both modalities) --------------
        template_tokens = torch.cat(
            [toks["Z_RGB"], toks["Z_TIR"], toks["Z_on"], toks["D_TIR"]], dim=1)  # (B, 4z, C)
        template_ctx = self.template_pool(template_tokens.mean(dim=1))           # (B, C)

        # ---- block 4: motion prior -------------------------------------------
        motion_out: Dict[str, torch.Tensor] = {}
        pending_obs: Optional[torch.Tensor] = None
        pending_conf: Optional[torch.Tensor] = None
        if self.motion is not None:
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
            elif eval_observe and self._prev_box is not None:
                # Inference: the previous frame's own prediction is the only legitimate
                # observation available (there is no ground truth while tracking).
                admitted = self._prev_box
                admitted_conf = self._prev_score
            pending_obs, pending_conf = admitted, admitted_conf
            motion_out = self.motion(
                box_xywh=admitted, image_size=image_size,
                confidence=admitted_conf, valid=None,
                batch_size=b, device=X_t.device, dtype=X_t.dtype,
                defer_observe=True)
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

        # ---- block 4b: temporal memory ---------------------------------------
        # `reliability` = 1 - q: a token the diagnosis considers healthy is reliable.
        reliability = (1.0 - q).detach()
        memory_readout = None
        prior_tokens = None
        if self.memory is not None:
            mem_state = getattr(self, "_state", {"memory": None, "memory_rel": None})
            # Target mask on the search grid, derived from the observed box.  It is only a
            # *pooling* mask for eq. 1, never a loss target, and a ground-truth box cannot
            # leak into the head through it (the head sees X_final, which the pooled summary
            # reaches only through the reliability gate).
            target_mask = None
            if gt_box_xywh is not None and image_size is not None:
                target_mask = self._search_target_mask(gt_box_xywh, image_size, b)
            elif corruption_mask is not None:
                # no box available: pool over the tokens the diagnosis considers healthy
                target_mask = (~corruption_mask).to(X_t.dtype)
            # Admission uses the *previous* frame's tracking score.  The head has not run
            # yet for this frame, so this is the causally correct signal -- and it matches the
            # evaluation pipeline, where the memory decision is taken after the previous frame
            # has been scored.  During training every frame is admitted, so the memory bank
            # sees the true temporal order instead of a partially frozen one.
            admit = bool(self.training) or self._prev_score is not None
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
        rec = self.refiner(
            X_t=X_t, X_aux=X_aux, q=q, neighbour_index=self.neighbour_index,
            template_pool=template_ctx, memory_readout=prior_tokens,
            motion_map=motion_map, topk=self.cfg.topk_tokens)
        X_rec = rec["X_rec"]

        # ---- block 5b: noise-modulated denoising -----------------------------
        # condition = [H-routed context scattered into token space | aux tokens].
        # ``context`` is the refiner's 256-d bottleneck, so it is projected by the same
        # layer that maps [context | aux] to the token dim, and only the suspect rows
        # carry a non-zero context (healthy tokens get no routed evidence).
        suspect = rec["suspect_index"]
        bidx = torch.arange(b, device=X_t.device)[:, None].expand_as(suspect)
        ctx_scatter = torch.zeros_like(X_t)
        # dtype must match the destination: under AMP autocast the projection returns fp16
        # while ``X_t`` may still be fp32 (or vice versa), and an in-place scatter demands
        # an exact match rather than casting.
        ctx_proj = self.condition_proj(
            torch.cat([rec["context"], X_aux[bidx, suspect]], dim=-1))
        ctx_scatter[bidx, suspect] = ctx_proj.to(ctx_scatter.dtype)
        ctx_zero = torch.zeros(b, X_t.shape[1], rec["context"].shape[-1],
                               device=X_t.device, dtype=X_t.dtype)
        condition = self.condition_proj(torch.cat([ctx_zero, X_aux], dim=-1)) + ctx_scatter

        # Naming is deliberately explicit: ``q`` is the per-token ERROR probability (block 3),
        # so ``trust = 1 - q`` is how healthy a token is.  The previous single name ``alpha``
        # carried "trust" but was multiplied into the denoiser's *write-back*, which inverted
        # the selective-recovery semantics (healthy tokens got corrected most, damaged ones
        # least).  Two names make that mistake impossible to repeat.
        token_error = q
        token_trust = (1.0 - q)
        if self.denoiser is not None:
            # frame-level condition terms: syndrome s (M), motion (map mean + u_t),
            # temporal memory read-out (pooled to memory_dim)
            motion_cond = None
            if motion_map is not None:
                # (B, 1): mean of the spatial prior map, plus the scalar uncertainty
                mm = motion_map.reshape(b, -1).mean(dim=-1, keepdim=True)
                uu = (uncertainty.reshape(b, 1) if uncertainty is not None
                      else torch.zeros(b, 1, device=X_t.device, dtype=X_t.dtype))
                motion_cond = torch.cat([mm, uu], dim=-1)             # (B, 2)
            mem_cond = None
            if prior_tokens is not None:
                # (B, F, memory_dim) -> (B, memory_dim): the frame axis is reduced, and the
                # result stays at memory_dim so it matches the denoiser's condition width.
                mem_cond = prior_tokens.mean(dim=1)
            den = self.denoiser(X_rec, condition, syndrome=out_s,
                                motion=motion_cond, memory=mem_cond,
                                token_error=token_error.unsqueeze(-1),
                                token_trust=token_trust.unsqueeze(-1),
                                noise_weak=self.cfg.noise_weak_std,
                                noise_strong=self.cfg.noise_strong_std,
                                strong_prob=self.cfg.noise_strong_prob)
            X_final = den["X_denoised"]
        else:
            den = {"X_denoised": X_rec, "step_preds": []}
            X_final = X_rec

        # identity preservation on reliable tokens (exact, not learned)
        preserve = None
        if corruption_mask is not None:
            keep = ~corruption_mask
            if keep.any():
                preserve = (X_final[keep] - X_t[keep]).abs().mean()

        mv = self.meanvar(X_final)

        # ---- block 6: template protection ------------------------------------
        gate_out: Dict[str, torch.Tensor] = {}
        if self.template_gate is not None:
            score = tracking_score
            if score is None:
                score = torch.zeros(b, device=X_t.device, dtype=X_t.dtype)
            # Fraction of tokens the diagnosis flags: a frame-level corruption summary that
            # is *differentiable*, so L_mem below can actually train the gate.  A detached
            # scalar here would leave the gate as dead parameters.
            corruption_fraction = q.mean(dim=-1).clamp(1e-6, 1 - 1e-6)
            recovery_conf = (1.0 - F.cosine_similarity(X_final, X_t, dim=-1, eps=1e-6)
                             ).mean(dim=-1)
            gate_out = self.template_gate(
                score=score.detach(), q=corruption_fraction, uncertainty=uncertainty,
                recovery_confidence=recovery_conf, topk=self.cfg.topk_tokens)
            if update_state:
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
                    "confidence": gate_out["mean_topk_q"].detach(),
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
                                confidence=pending_conf, valid=None)

        return {
            "X_final": X_final,
            "X_rec": X_rec,
            "q": q, "s": diag["s"], "s_logits": diag["s_logits"],
            "C_obs": diag["C_obs"], "C_ref": diag["C_ref"],
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
            "motion_target": motion_target,
            "meanvar": mv, "preserve": preserve,
            "denoise_steps": den["step_preds"],
            **gate_out,
        }
