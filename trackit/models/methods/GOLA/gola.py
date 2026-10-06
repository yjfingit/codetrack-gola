# Modified by Zekai Shao
# Licensed under Apache-2.0: http://www.apache.org/licenses/LICENSE-2.0
# Add support for RGB-T dataset

from typing import Tuple, List, Optional, Mapping, Any
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from timm.models.layers import trunc_normal_
from trackit.models.backbone.dinov2 import DinoVisionTransformer, interpolate_pos_encoding
from .modules.patch_embed import PatchEmbedNoSizeCheck
from .modules.gola.apply import find_all_frozen_nn_linear_names, apply_lora
from .modules.head.mlp import MlpAnchorFreeHead, Mlp
from codetrack.corruption import CorruptionSchedule


class GOLA_DINOv2(nn.Module):
    def __init__(self, vit: DinoVisionTransformer,
                 template_feat_size: Tuple[int, int],
                 search_region_feat_size: Tuple[int, int],
                 lora_r: int, lora_alpha: float, lora_dropout: float, use_rslora: bool = False,
                 codetrack_config: Optional[dict] = None):
        super().__init__()
        assert template_feat_size[0] <= search_region_feat_size[0] and template_feat_size[1] <= search_region_feat_size[
            1]
        self.z_size = template_feat_size
        self.d_size = template_feat_size
        self.x_size = search_region_feat_size

        self.patch_embed = PatchEmbedNoSizeCheck.build(vit.patch_embed)
        self.blocks = vit.blocks
        self.norm = vit.norm
        self.embed_dim = vit.embed_dim

        self.pos_embed = nn.Parameter(torch.empty(1, self.x_size[0] * self.x_size[1], self.embed_dim))
        self.pos_embed.data.copy_(interpolate_pos_encoding(vit.pos_embed.data[:, 1:, :],
                                                           self.x_size,
                                                           vit.patch_embed.patches_resolution,
                                                           num_prefix_tokens=0, interpolate_offset=0))

        self.lora_alpha = lora_alpha
        self.use_rslora = use_rslora

        for param in self.parameters():
            param.requires_grad = False

        self.token_type_embed = nn.Parameter(torch.empty(5, self.embed_dim))
        trunc_normal_(self.token_type_embed, std=.02)

        for i_layer, block in enumerate(self.blocks):
            linear_names = find_all_frozen_nn_linear_names(block)
            apply_lora(block, linear_names, lora_r, lora_alpha, lora_dropout, use_rslora)

        # self.fuse_search = Mlp(self.embed_dim * 2, out_features=self.embed_dim, num_layers=3)

        self.head = MlpAnchorFreeHead(self.embed_dim, self.x_size)

        # ------------------------------------------------------------------
        # CodeTrack branch (architecture figure blocks 3-6).
        #
        # Additive: when ``codetrack_config`` is absent or disabled, no module is
        # constructed, no parameter is added, and ``forward`` takes the original path
        # bit-for-bit.  When enabled, the branch consumes the already-normalised fused
        # tokens (``self.norm`` output) and the *original* head is shared between the
        # baseline and CodeTrack paths, so the loaded GOLA checkpoint still covers head.
        # ------------------------------------------------------------------
        self.codetrack = None
        self.codetrack_cfg = None
        if codetrack_config:
            from codetrack.config import CodeTrackConfig
            from codetrack.codetrack import CodeTrack
            cfg = CodeTrackConfig.from_dict(codetrack_config)
            if cfg.enabled:
                cfg.validate()
                self.codetrack_cfg = cfg
                self.codetrack = CodeTrack(cfg, head=self.head)

    # ------------------------------------------------------------------ helpers
    def _codetrack_split(self, feat: torch.Tensor):
        """Slice fused tokens into (Z_RGB, X_RGB, Z_TIR, X_TIR, Z_on, D_TIR)."""
        return self.codetrack._split(feat)

    def _corruption_patch(self, x: torch.Tensor, apply: torch.Tensor, kind: str,
                          ratio: float, severity: float):
        """Erase/noise patches of the *search* image for the selected samples.

        Returns the modified image and records the resulting (B, N) token mask, which is
        what ``L_diag`` is supervised against.  Samples not selected are returned untouched
        so the batch keeps a clean subset.
        """
        import torch.nn.functional as F
        x_out = x.clone()
        b = x.shape[0]
        # patch grid of the search crop
        n_side = int(round((x.shape[-1] // self.patch_embed.patch_size[0])))
        n_tokens = n_side * n_side
        p = self.patch_embed.patch_size[0]
        # CodeTrack recovers X_TIR and consumes X_RGB as auxiliary evidence. Corrupting all six
        # channels destroyed the auxiliary modality at exactly the locations it was supposed to
        # help reconstruct, while the supervision mask still described an X_TIR target. Keep the
        # RGB half intact and apply synthetic token damage only to the TIR half.
        target_channels = slice(x.shape[1] // 2, None)
        mask_b = torch.zeros(b, n_tokens, dtype=torch.bool, device=x.device)
        k = max(1, int(round(ratio * n_tokens)))
        for i in range(b):
            if not bool(apply[i]):
                continue
            if kind in ("block_erase", "burst_erase", "modality_drop"):
                side = max(1, int(round(k ** 0.5))) if kind == "block_erase" else k
                if kind == "block_erase":
                    centered = bool(getattr(self.codetrack_cfg, "corruption_centered", False))
                    spatial_mode = str(getattr(self.codetrack_cfg, "corruption_spatial_mode", "random"))
                    if spatial_mode == "quadrant_mix":
                        # Sample one of four spatial quadrants and keep the whole block in it.
                        # The random draw is per selected frame, so a batch covers all quadrants
                        # over time without coupling the corruption to the object box.
                        q = int(torch.randint(0, 4, (1,)).item())
                        half = n_side // 2
                        row0 = 0 if q < 2 else half
                        col0 = 0 if q % 2 == 0 else half
                        span = max(1, half - side + 1)
                        top = row0 + int(torch.randint(0, span, (1,)).item())
                        left = col0 + int(torch.randint(0, span, (1,)).item())
                    elif centered:
                        top = max(0, (n_side - side) // 2)
                        left = max(0, (n_side - side) // 2)
                    else:
                        top = int(torch.randint(0, max(1, n_side - side + 1), (1,)).item())
                        left = int(torch.randint(0, max(1, n_side - side + 1), (1,)).item())
                    idx = [(top + dy) * n_side + (left + dx)
                           for dy in range(side) for dx in range(side)
                           if top + dy < n_side and left + dx < n_side]
                    idx = idx[:k]
                else:
                    start = int(torch.randint(0, max(1, n_tokens - side), (1,)).item())
                    idx = list(range(start, min(start + side, n_tokens)))
                for t in idx:
                    row, col = divmod(t, n_side)
                    # zero the crop region in *normalised* space (0 == channel mean per
                    # modality), which is what a real erasure looks like to the backbone
                    x_out[i, target_channels,
                          row * p:(row + 1) * p, col * p:(col + 1) * p] = 0.0
                    mask_b[i, t] = True
            else:  # feat_noise: additive noise over a random token subset
                idx = torch.randperm(n_tokens)[:k]
                for t in idx:
                    row, col = divmod(int(t), n_side)
                    patch = x_out[i, target_channels,
                                  row * p:(row + 1) * p, col * p:(col + 1) * p]
                    patch.add_(torch.randn_like(patch) * severity)
                    mask_b[i, t] = True
        self._last_token_mask = mask_b
        return x_out, mask_b

    def reset_sequence(self) -> None:
        """Clear per-sequence recurrent state.  Call at the start of each sequence."""
        if self.codetrack is not None:
            self.codetrack.reset_sequence()

    def forward(self, z: torch.Tensor, x: torch.Tensor, d: torch.Tensor,
                z_feat_mask: torch.Tensor, d_feat_mask: torch.Tensor,
                gt_box: Optional[torch.Tensor] = None,
                image_size: Optional[torch.Tensor] = None,
                teacher: bool = False,
                image_corruption_mask: Optional[torch.Tensor] = None,
                x_clean: Optional[torch.Tensor] = None,
                **kwargs):
        if self.codetrack is None:
            z_feat_v, z_feat_i = self._z_feat(z, z_feat_mask)
            d_feat_v, d_feat_i = self._d_feat(d, d_feat_mask)
            x_feat_v, x_feat_i = self._x_feat(x)
            x_feat = self._fusion(z_feat_v, x_feat_v, z_feat_i, x_feat_i, d_feat_v, d_feat_i)
            return self.head(x_feat)
        return self._forward_codetrack(z, x, d, z_feat_mask, d_feat_mask,
                                       image_corruption_mask=image_corruption_mask,
                                       x_clean=x_clean,
                                       gt_box=gt_box, image_size=image_size,
                                       teacher=teacher, **kwargs)

    def _forward_codetrack(self, z, x, d, z_feat_mask, d_feat_mask,
                           image_corruption_mask=None,
                           x_clean=None,
                           gt_box=None, image_size=None, teacher=False, **kwargs):
        """One CodeTrack training/inference step.

        ``teacher=True`` (used only for the clean branch of the clean-teacher residual)
        returns ``score_map``/``boxes`` with gradients enabled through the head so that
        ``L_track^clean`` can train the GOLA adapters, while the *diagnosis target*
        ``clean_tokens`` is returned detached.
        """
        # ---- token-level corruption (architecture figure "block erase") ------
        # Applied to the *search* crop before the blocks, so the corruption is a genuine
        # token-grid erasure that CodeTrack's H/syndrome machinery has to locate.  A
        # per-sample Bernoulli draw keeps clean frames in the batch, which is what stops the
        # diagnosis head from degenerating into "always predict corrupted".
        token_mask = None
        drop_token = None
        x_cor = x
        if self.training and self.codetrack_cfg.corruption_enabled:
            if getattr(self, "_corruption_schedule", None) is None:
                self._corruption_schedule = CorruptionSchedule(
                    image_prob=self.codetrack_cfg.corruption_image_prob,
                    token_prob=getattr(self.codetrack_cfg, 'corruption_token_prob', 0.2),
                    token_ratio=self.codetrack_cfg.corruption_token_ratio,
                    severity=self.codetrack_cfg.corruption_severity,
                    enabled=True)
            sched = self._corruption_schedule
            b = x.shape[0]
            kind, sev, apply = sched.draw_token(b, x.device)
            drop_token = apply
            if bool(apply.any()):
                x_cor, token_mask = self._corruption_patch(
                    x, apply, kind, sched.token_ratio, sev)

        # ---- run both branches ------------------------------------------------
        # The teacher sees the *clean* crop and is used only to build the residual target
        # (e_i* = alpha e_feat + (1-alpha) e_task).  Its features are detached, but its head
        # output keeps its graph so L_track^clean can still train the GOLA adapters.  The
        # student sees the corrupted crop and carries the whole CodeTrack branch.
        # The frozen-backbone forward is run in **fp32** even under AMP.  Two forward passes
        # per step make the fp16 activation budget tight, and an overflowing fp16 activation
        # turns into ``inf`` which the head then propagates into every adapter gradient
        # (observed: inf grad-norms on the LoRA/head tensors while every loss term stayed
        # finite).  Running the trunk in fp32 costs a little memory and removes the overflow
        # entirely; the LoRA matmuls still benefit from the AMP autocast on the graph inputs.
        # In evaluation there is no corruption and no clean-teacher residual, so the trunk
        # is run once.  Only training needs the paired (clean, corrupted) forwards.
        needs_teacher = self.training or teacher or (x_cor is not x)

        # ---- STUDENT branch: gradients MUST flow through the blocks ------------
        # This used to be wrapped in ``torch.no_grad()`` together with the teacher, which
        # silently made the LoRA adapters untrainable: their parameters never entered the
        # autograd graph at all, so S2-style "joint PEFT" was a no-op.  Measured before the
        # fix: 0 of 1296 LoRA tensors moved after 5 AdamW steps, while head and CodeTrack
        # moved normally.
        #
        # The frozen DINOv2 base is excluded via ``requires_grad=False`` on its own
        # parameters (set by the builder), *not* by a blanket ``no_grad`` -- the LoRA
        # delta ``base(x) + lora_A(x) @ lora_B(x)`` is an addition, so leaving the graph
        # enabled lets the delta receive gradients while the base stays constant.
        #
        # Still fp32: an overflowing fp16 activation becomes ``inf``, which the head then
        # propagates into every adapter gradient.
        with torch.autocast('cuda', enabled=False):
            z_v, z_i = self._z_feat(z.float(), z_feat_mask)
            d_v, d_i = self._d_feat(d.float(), d_feat_mask)
            x_v_c, x_i_c = self._x_feat(x_cor.float())       # CORRUPTED (== clean in eval)
            cor_fused = torch.cat((z_v, x_v_c, z_i, x_i_c, d_v, d_i), dim=1)
            for block in self.blocks:
                cor_fused = block(cor_fused)
            cor_fused = self.norm(cor_fused)

        # Baseline measurement mode: keep the training-class GOLA weight loader and its
        # LoRA parameter mapping, but bypass the optional CodeTrack branch at inference.
        # This avoids comparing the legacy merged inference class (whose checkpoint key
        # layout differs) against a training-class CodeTrack model.
        if (not self.training and bool(getattr(self.codetrack_cfg, "extra", {}).get(
                "identity_only", False))):
            baseline_tokens = self._fuse_search(
                cor_fused, z_v.shape[1], x_v_c.shape[1])
            return self.head(baseline_tokens)

        # ---- CLEAN TEACHER branch: no gradient ---------------------------------
        # The teacher is only a *target* for the diagnosis residual; letting it receive the
        # student's gradients would make the target move with the prediction.  Its features
        # are detached below as a second line of defence.
        #
        # ``x_clean`` is the untouched search crop written by the ``image_corruption`` data
        # plugin.  Without it the teacher saw the SAME damaged crop as the student for every
        # image-level sample, so ``e_feat``/``e_aux`` -- and therefore ``L_diag`` -- were exactly
        # zero for precisely the corruption that plugin creates.  At inference the plugin does not
        # run, the key is absent, and this falls back to ``x`` (identical behaviour to before).
        x_teacher = x_clean if x_clean is not None else x
        if needs_teacher:
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                x_v, x_i = self._x_feat(x_teacher.float())   # CLEAN search
                clean_fused = torch.cat((z_v, x_v, z_i, x_i, d_v, d_i), dim=1)
                for block in self.blocks:
                    clean_fused = block(clean_fused)
                clean_fused = self.norm(clean_fused)
        else:
            clean_fused = cor_fused

        clean_tokens = self._codetrack_split(clean_fused)["X_TIR"]
        with torch.set_grad_enabled(bool(needs_teacher)), torch.autocast('cuda', enabled=False):
            clean_logits = self.head(clean_tokens)

        if teacher:
            return {"score_map": clean_logits["score_map"],
                    "boxes": clean_logits["boxes"],
                    "codetrack_extras": {"clean_tokens": clean_tokens.detach()}}

        # ---- diagnosis targets -------------------------------------------------
        # ``L_diag`` was previously dead code: the criterion looks for ``error_target`` and
        # ``syndrome_target``, and nothing produced them, so the diagnosis head had NO explicit
        # supervision at all -- it could only learn indirectly through the tracking/recovery
        # gradient, which is far too weak for the claim that the syndrome localises the damaged
        # tokens.
        #
        # The target is derived from the *measured discrepancy* between the clean reference
        # feature and the corrupted one, not from the token mask the injector applied.  Handing
        # the head the injector's mask would teach it to read a label instead of to detect
        # damage, and it would not transfer to the natural degradations LasHeR already contains.
        #
        # ``e_feat``   : per-token normalised feature deviation of the tracked modality (TIR)
        # ``e_aux``    : the same deviation on the RGB search tokens -- auxiliary evidence
        # ``alpha``    : how much the cross-modal signal is allowed to *attenuate* e_feat
        # ``X_clean`` is detached -- it is a target, and the teacher must not receive the
        # student's gradients.
        #
        # The combination is multiplicative on purpose.  The previous form was
        # ``alpha*e_feat + (1-alpha)*(0.5*(e_feat+e_aux)) = 0.75*e_feat + 0.25*e_aux`` at
        # alpha=0.5: the RGB term diluted the TIR term by 25% for no stated reason, and the
        # weights no longer matched the ``alpha`` the config exposes.  Here ``alpha`` is a
        # genuine mixing coefficient between the pure-TIR target and the both-modalities target.
        diag_targets = {}
        if needs_teacher:
            c_xt = self._codetrack_split(clean_fused)["X_TIR"].detach()
            c_xa = self._codetrack_split(clean_fused)["X_RGB"].detach()
            k_xt = self._codetrack_split(cor_fused)["X_TIR"]
            k_xa = self._codetrack_split(cor_fused)["X_RGB"]
            alpha = float(getattr(self.codetrack_cfg, "diagnosis_alpha", 0.5))

            # feature-side error: cosine deviation, scale-free so bright/dark frames compare
            e_feat = (1.0 - F.cosine_similarity(k_xt, c_xt, dim=-1, eps=1e-6))
            e_feat = e_feat.clamp(0.0, 1.0)
            # cross-modal error: the RGB search tokens are the auxiliary evidence; if they also
            # deviate, the frame is damaged in both modalities rather than in one
            e_aux = (1.0 - F.cosine_similarity(k_xa, c_xa, dim=-1, eps=1e-6)).clamp(0.0, 1.0)

            err = (alpha * e_feat + (1.0 - alpha) * e_aux).detach().clamp(1e-4, 1.0 - 1e-4)
            diag_targets["error_target"] = err
            # the two components are kept for the diagnostic report (Error/auroc_tir vs _rgb)
            diag_targets["error_target_tir"] = e_feat.detach()
            diag_targets["error_target_rgb"] = e_aux.detach()

            # syndrome target: apply the same parity check to the error field, i.e. what the
            # checks *should* read if the per-token error were as measured.  ``H_bar`` is
            # (M, N) and ``err`` is (B, N) -> (B, M).
            H_bar = self.codetrack.H.matrix().to(err.dtype)
            s_tgt = torch.einsum("mn,bn->bm", H_bar, err)
            diag_targets["syndrome_target"] = s_tgt.detach()

        # ---- student / CodeTrack pass ---------------------------------------
        # Inference-time motion observation.  ``image_size`` is now supplied by the evaluation
        # pipeline, and ``eval_observe`` enables feeding the *previous* frame's own decoded box
        # back into the Kalman filter (see CodeTrack.forward).  Without this the filter is
        # seeded once and never updated, so the motion prior is a constant.
        eval_observe = (not self.training) and image_size is not None
        # Optional diagnostic head output before any ECC write-back. This is only computed for
        # explicitly requested mechanism visualisation and never affects the tracking path.
        pre_recovery_head = None
        if os.environ.get("CODETRACK_DIAG_PREPOST", "0") == "1":
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                pre_tokens = self._codetrack_split(cor_fused)["X_TIR"]
                pre_recovery_head = self.head(pre_tokens.float())

        out = self.codetrack(
            F_L=cor_fused,
            gt_box_xywh=gt_box if kwargs.get("observe_motion", True) else None,
            image_size=image_size,
            eval_observe=eval_observe,
            tracking_score=None,
            box_confidence=kwargs.get("box_confidence"),
            corruption_mask=token_mask,
            was_corrupted=(drop_token if drop_token is not None else None),
            image_corruption_mask=image_corruption_mask,
            update_state=True,
        )
        with torch.autocast('cuda', enabled=False):
            head_out = self.head(out["X_final"].float())
            # Training-only selective-update target.  A frame is trustworthy for correction when
            # it was not synthetically damaged and the corrected head preserves most of the
            # clean-teacher response.  This teaches the shared reliability gate the actual
            # contract of selective propagation instead of equating every clean input with a
            # universally safe write-back.
            gate_target = None
            if self.training and needs_teacher and isinstance(clean_logits, dict):
                clean_peak = clean_logits["score_map"].detach().float().flatten(1).sigmoid().amax(dim=-1)
                corr_peak = head_out["score_map"].detach().float().flatten(1).sigmoid().amax(dim=-1)
                damaged_flag = None
                if drop_token is not None:
                    damaged_flag = drop_token.reshape(drop_token.shape[0]).to(torch.bool)
                if image_corruption_mask is not None and torch.is_tensor(image_corruption_mask):
                    image_flag = (image_corruption_mask.reshape(image_corruption_mask.shape[0], -1)
                                  .abs().sum(dim=-1) > 0)
                    damaged_flag = image_flag if damaged_flag is None else (
                        damaged_flag | image_flag.to(damaged_flag.device))
                if damaged_flag is None:
                    damaged_flag = torch.zeros_like(clean_peak, dtype=torch.bool)
                else:
                    damaged_flag = damaged_flag.to(device=clean_peak.device)
                gate_target = ((~damaged_flag.to(torch.bool)) &
                               (corr_peak >= 0.9 * clean_peak)).to(clean_peak.dtype)
            # Feed this frame's max classification score to CodeTrack so the *next* frame's
            # memory admission (DTPTrack's update_criteria-style rule) has a causally valid
            # signal.  Detached: it is a selection statistic, not a loss term.
            if self.codetrack is not None:
                score = head_out["score_map"].detach().flatten(1).amax(dim=-1).sigmoid()
                # Also publish the decoded box so the *next* frame's Kalman filter has a real
                # observation.  The decode follows the official
                # ``PostProcessing_BoxWithScoreMap`` exactly -- sigmoid scores, argmax over the
                # flattened map, gather that cell's corner-format box, multiply by the search
                # region size -- so the observation is in the same coordinates the evaluator
                # uses.  Only needed at inference; during training the ground-truth box drives
                # the filter.
                box_xywh = None
                if not self.training and image_size is not None:
                    sm = head_out["score_map"].detach().float().sigmoid()
                    n, h, w = sm.shape
                    best = sm.reshape(n, h * w).argmax(dim=-1)
                    bx = head_out["boxes"].detach().float().reshape(n, h * w, 4)
                    corners = bx.gather(1, best.view(n, 1, 1).expand(-1, -1, 4)).squeeze(1)
                    scale = image_size.to(corners.dtype).reshape(n, 2)
                    corners = (corners.reshape(n, 2, 2) * scale[:, None, :]).reshape(n, 4)
                    x1, y1, x2, y2 = corners.unbind(-1)
                    box_xywh = torch.stack(
                        [(x1 + x2) * 0.5, (y1 + y2) * 0.5,
                         (x2 - x1).clamp(min=1e-3), (y2 - y1).clamp(min=1e-3)], dim=-1)
                self.codetrack.notify_tracking_score(score, box_xywh=box_xywh)

        extras = {
            "teacher": clean_logits,
            "clean_tokens": clean_tokens.detach(),
            # L_diag supervision: without these two keys the criterion's diagnosis branch never
            # fires (it tests ``q is not None and error_target is not None``), which is why
            # Loss/diag never appeared in any run.
            **diag_targets,
            "recovered": out["X_final"],
            # The pre-denoise (post-refiner) tensor.  ``L_gain`` and the ``Error/d_before``
            # statistic need the *before* state of the recovery, and ``X_rec`` is exactly that:
            # the refiner's output before the diffusion-correction block touches it.
            "recovered_satr": out["X_final"],
            # The student's *input* token stream X_t (the corrupted search representation, before
            # any recovery).  ``d_input = d(X_t, X_clean)`` is the honest denominator for the whole
            # recovery claim: ``L_gain`` used to compare only ``X_rec`` against clean, which proves
            # "the denoiser improves on the refiner" and not "CodeTrack improves on what it was
            # given".  The paper-level criterion is ``d_final < d_input``.
            "input_tokens": out["tokens"]["X_TIR"],
            "q": out["q"], "s": out["s"],
            "syndrome_energy_before": out.get("syndrome_energy_before"),
            "syndrome_energy_after": out.get("syndrome_energy_after"),
            # raw pre-sigmoid evidence: the diagnosis loss must be the autocast-safe
            # ``*_with_logits`` form, because ``binary_cross_entropy`` on a sigmoid output aborts
            # under AMP ("unsafe to autocast").
            "q_logits": out.get("q_logits"), "s_logits": out.get("s_logits"),
            "suspect_index": out["suspect_index"],
            "preserve": out["preserve"],
            # supervision signals for the temporal memory and the template gate
            "corruption_mask": token_mask,
            "was_corrupted": out.get("was_corrupted"),
            "corruption_fraction": out.get("corruption_fraction"),
            "frame_reliability": (out.get("memory") or {}).get("frame_reliability"),
            # TRC per-frame confidences (B, F).  Supervised below: the learned gate has NO
            # other gradient path -- it only multiplies the frame summaries, and nothing in
            # the tracking loss says what c_i should be, so without a direct target its
            # parameters stay at exactly zero gradient.
            "trc_confidence": (out.get("memory") or {}).get("trc_confidence"),
            "c_t": out.get("c_t"),
            "gate_target": gate_target,
            "motion_map_norm": out.get("motion_map_norm"),
            "motion_target": out.get("motion_target"),
            # One-shot syndrome calibration.  The diagnosis head exports the RAW pre-sigmoid
            # syndrome on its very first training forward; the training loop all-reduces the
            # statistics across ranks and applies the gain/offset once.  Letting each rank
            # calibrate locally would give the ranks different parameter values, and the first
            # DDP all-reduce would then average four inconsistent models.
            "syndrome_pending_calibration": out.get("syndrome_pending_calibration"),
        }
        result = {"score_map": head_out["score_map"], "boxes": head_out["boxes"],
                  "codetrack_extras": extras,
                  "codetrack": {k: v for k, v in out.items()
                                if k in ("q", "s", "C_obs", "C_ref", "motion",
                                         "alpha", "delta", "suspect_index", "suspect_score", "c_t",
                                         "uncertainty", "mean_topk_q", "max_q", "decode_gate",
                                         "syndrome_energy_before", "syndrome_energy_after")},
                  "codetrack_pre": pre_recovery_head}
        return result

    def _z_feat(self, z: torch.Tensor, z_feat_mask: torch.Tensor):
        z_v = self.patch_embed(z[:, :3])
        z_i = self.patch_embed(z[:, 3:])

        z_W, z_H = self.z_size
        z_v = z_v + self.pos_embed.view(1, self.x_size[1], self.x_size[0], self.embed_dim)[:, : z_H, : z_W, :].reshape(
            1, z_H * z_W, self.embed_dim)
        z_v = z_v + self.token_type_embed[:2][z_feat_mask.flatten(1)]

        z_i = z_i + self.pos_embed.view(1, self.x_size[1], self.x_size[0], self.embed_dim)[:, : z_H, : z_W, :].reshape(
            1, z_H * z_W, self.embed_dim)
        z_i = z_i + self.token_type_embed[:2][z_feat_mask.flatten(1)]

        return z_v, z_i

    def _d_feat(self, d: torch.Tensor, d_feat_mask: torch.Tensor):
        d_v = self.patch_embed(d[:, :3])
        d_i = self.patch_embed(d[:, 3:])

        d_W, d_H = self.d_size
        d_v = d_v + self.pos_embed.view(1, self.x_size[1], self.x_size[0], self.embed_dim)[:, : d_H, : d_W, :].reshape(
            1, d_H * d_W, self.embed_dim)
        d_v = d_v + self.token_type_embed[2:4][d_feat_mask.flatten(1)]

        d_i = d_i + self.pos_embed.view(1, self.x_size[1], self.x_size[0], self.embed_dim)[:, : d_H, : d_W, :].reshape(
            1, d_H * d_W, self.embed_dim)
        d_i = d_i + self.token_type_embed[2:4][d_feat_mask.flatten(1)]

        return d_v, d_i

    def _x_feat(self, x: torch.Tensor):
        x_v = self.patch_embed(x[:, :3])
        x_i = self.patch_embed(x[:, 3:])

        x_v = x_v + self.pos_embed
        x_v = x_v + self.token_type_embed[4].view(1, 1, self.embed_dim)
        x_i = x_i + self.pos_embed
        x_i = x_i + self.token_type_embed[4].view(1, 1, self.embed_dim)
        return x_v, x_i

    def _fusion(self, z_feat_v: torch.Tensor, x_feat_v: torch.Tensor, z_feat_i: torch.Tensor, x_feat_i: torch.Tensor,
                d_feat_v: torch.Tensor, d_feat_i: torch.Tensor):
        fusion_feat = torch.cat((z_feat_v, x_feat_v, z_feat_i, x_feat_i, d_feat_v, d_feat_i), dim=1)
        for i in range(len(self.blocks)):
            fusion_feat = self.blocks[i](fusion_feat)
        fusion_feat = self.norm(fusion_feat)
        return self._fuse_search(fusion_feat, z_feat_v.shape[1], x_feat_v.shape[1])

    def _fuse_search(self, feat, z_len, x_len):
        # search_v = feat[:, z_len:z_len + x_len, :]
        search_i = feat[:, 2 * z_len + x_len:2 * (z_len + x_len), :]
        # search = torch.cat([search_v, search_i], dim=2)
        # return self.fuse_search(search)
        return search_i

    def state_dict(self, **kwargs):
        """Only *trainable* parameters are persisted (the frozen backbone is reloaded from
        the DINOv2 pretrained checkpoint, and buffers such as the positional embedding or a
        CodeTrack geometry buffer are reconstructed from the config).

        Buffers are skipped explicitly: ``get_parameter`` raises for them, and a persistent
        CodeTrack buffer would otherwise abort every checkpoint save.  For a plain GOLA model
        (no buffers beyond the upstream ones) the returned dict is unchanged.
        """
        state_dict = super().state_dict(**kwargs)
        prefix = kwargs.get('prefix', '')
        param_names = {name for name, _ in self.named_parameters()}
        for key in list(state_dict.keys()):
            name = key[len(prefix):]
            if name not in param_names or not self.get_parameter(name).requires_grad:
                state_dict.pop(key)
        if self.lora_alpha != 1.:
            state_dict[prefix + 'lora_alpha'] = torch.as_tensor(self.lora_alpha)
            state_dict[prefix + 'use_rslora'] = torch.as_tensor(self.use_rslora)
        return state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], **kwargs):
        if 'lora_alpha' in state_dict:
            state_dict = OrderedDict(**state_dict)
            self.lora_alpha = state_dict['lora_alpha'].item()
            self.use_rslora = state_dict['use_rslora'].item()
            del state_dict['lora_alpha']
            del state_dict['use_rslora']
        return super().load_state_dict(state_dict, **kwargs)
