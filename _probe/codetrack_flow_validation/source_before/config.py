"""Configuration for the CodeTrack branch.

Defaults are the numbers annotated on the architecture figure:

    N = 256 variable nodes      (= 16 x 16 search tokens)
    C = 768                       (DINOv2 ViT-B/14 embedding dim)
    M = 64 check nodes
    discrepancy / projection dim  = 128      (block 3.1 "768 -> 128")
    K = 32 TopK suspicious tokens (= 12.5%)
    K_n = 8 H-route neighbours
    recovery refiner              768 -> 256 -> 768
    T_mem = 3..4 frames x 8 tokens x 128-d
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CodeTrackConfig:
    # ---- enabled / wiring -----------------------------------------------------
    enabled: bool = False
    # GOLA fused-token layout: [z_v, x_v, z_i, x_i, d_v, d_i] with
    # z_len=64 (8x8 template) and x_len=256 (16x16 search).
    z_len: int = 64
    x_len: int = 256
    dim: int = 768                        # C
    grid: int = 16                        # search grid side

    # ---- block 3: ECC error diagnosis ---------------------------------------
    mid_dim: int = 128                    # W_x / W_r output dim
    num_checks: int = 64                  # M
    h_links_per_check: int = 12           # d_c  (M*d_c = 768 = N*d_v)
    h_min_col_degree: int = 3             # d_v
    h_locality_window: int = 0            # 0 = unrestricted sampling
    # share of each check's edges drawn from the global pool instead of its local window.
    # 0 = purely local (a syndrome can only report "something near this check"),
    # 1 = purely random (no addressability).  0.25 was the starting value.
    h_free_edge_frac: float = 0.25
    h_seed: int = 1234
    syndrome_hidden: int = 128
    # Block 3.1 detail: s_j = MLP([delta, |delta|, cos(obs, ref)])
    syndrome_cos: bool = True
    # Location prior on the check support: each check prefers a spatial window.
    h_locality_wrap: bool = True

    # ---- block 4: motion prior ----------------------------------------------
    motion_enabled: bool = True
    motion_state_dim: int = 8             # [cx, cy, log w, log h, vx, vy, vw, vh]
    motion_process_noise: float = 1e-2
    motion_measurement_noise: float = 4e-2
    motion_gate_hidden: int = 64

    # ---- block 5: selective token recovery ----------------------------------
    topk_tokens: int = 32                 # K_max, 12.5% of N
    num_neighbours: int = 8               # K_n H-route neighbours
    refiner_hidden: int = 256             # 768 -> 256 -> 768
    refiner_heads: int = 4
    refiner_dropout: float = 0.0
    # Residual-gate bias for both the refiner and the denoiser.  The correction is written
    # back as ``sigmoid(gate) * delta``, so this single scalar controls BOTH how close
    # stage 0 is to the GOLA baseline and how much gradient reaches the upstream branches.
    # sigmoid(-12)=6e-6 makes the branch effectively untrainable (measured grad 3.1e-07 to
    # the temporal memory); sigmoid(-8)=3.4e-4 keeps stage-0 output within ~2e-6 of the
    # baseline while giving 1.9e-05.  -8 is the recommended value.
    residual_gate_init: float = -8.0
    # Scale on the log-space motion attention bias inside the refiner:
    # ``bias = log(motion_map) * motion_bias_scale``.  This is the only knob governing how
    # strongly the motion prior steers evidence routing (0 = motion ignored by the refiner).
    motion_bias_scale: float = 0.5
    # Centre and standardise the log-space motion bias before scaling it.  Without this the
    # bias is a near-constant (measured logit spread 0.05 nats over 8 neighbours, because the
    # unit-mass motion map sits at 1/256 +- 2e-3), so the Kalman prior cannot influence
    # attention at all.  False reproduces the pre-2026-10-04 numerics exactly.
    motion_bias_normalise: bool = True
    # Init std of the residual predictor's output layer inside BOTH the refiner and the
    # denoiser.  Replaces the old "large negative residual gate" idiom: a small nonzero
    # ``up`` keeps step-0 output near identity while leaving the conditioning paths their
    # gradient.  See codetrack/recovery.py for the measurements behind this.
    up_init_std: float = 0.02

    # ---- block 5: diffusion / noising-denoising correction ------------------
    # SCDT-style noising-denoising: weak/strong noise on the *available* features,
    # a short-term cross-attention denoiser, mean-var completion and 2 refinement
    # steps.  ``diffusion_steps`` is the number of iterative refinement steps.
    diffusion_enabled: bool = True
    diffusion_steps: int = 2
    diffusion_hidden: int = 256
    diffusion_heads: int = 4
    noise_weak_std: float = 0.05
    noise_strong_std: float = 0.20
    noise_strong_prob: float = 0.5
    # cosine noise schedule coefficient: alpha_bar_t = cos(pi/2 * (t/T)^u)
    noise_schedule_power: float = 1.0
    # How strongly each step's prediction is written back:
    #   "ramp"         w_t = (t + 1) / T                -> every step contributes, later ones more
    #   "linear_noise" w_t = 1 - alpha_bar_t            -> the old behaviour, w_0 == 0
    #
    # "linear_noise" is kept as an option because it is what the previous revision used, but it
    # wastes the first step: with the cosine buffer at T=2 the weights are
    # ``[1 - cos(0), 1 - cos(pi/2)] = [0.0, 0.999]``, so step 0 computes a prediction (and its
    # gradients) and then multiplies it by zero.  The module was advertised as "2-step
    # refinement" while only one step could move a token.  ``ramp`` gives [0.5, 1.0] at T=2.
    # Identity at initialisation is unaffected either way: the whole prediction is still scaled
    # by ``sigmoid(residual_gate)``.
    diffusion_write_schedule: str = "ramp"
    # keep the input alive: the correction is a gated residual, never a rewrite
    residual_clip: float = 0.0            # 0 = unbounded

    # ---- block 6: online template protection --------------------------------
    template_protection: bool = True
    template_gate_hidden: int = 64
    template_threshold: float = 0.5       # tau_c
    gola_update_threshold: float = 0.84   # official score threshold

    # ---- block 4/5: temporal memory -----------------------------------------
    memory_enabled: bool = True
    memory_frames: int = 3
    memory_tokens: int = 8
    memory_dim: int = 128
    # RESERVED / CURRENTLY UNUSED: placeholder for a reliability-weighted blend of the
    # mean-TopK and recovery-confidence terms in the template gate.  The gate currently
    # takes both as separate inputs and lets its MLP weight them, so this knob does
    # nothing.  Kept so the name is not silently reintroduced with different semantics.
    memory_pull_weight: Optional[float] = None

    # ---- loss weights (architecture figure panel) ---------------------------
    w_track_corr: float = 1.0
    w_track_clean: float = 0.25
    lambda_diag: float = 0.5
    lambda_rec: float = 0.2
    # L_gain supervises the *improvement* the recovery must produce, not its similarity to the
    # teacher: ReLU(d_after - 0.8 * d_before).  Without it the recovery branch can be trained
    # towards "output something plausible" instead of "output something better".
    lambda_gain: float = 0.2
    lambda_align: float = 0.2
    lambda_pres: float = 0.01
    lambda_mem: float = 0.1
    # TRC reliability gate.  Without this the gate has zero gradient: nothing else in the
    # objective says what a frame's reliability should be (see codetrack/criteria.py).
    lambda_trc: float = 0.1
    lambda_motion: float = 0.2

    # ---- temporal training (stage 3+) ---------------------------------------
    # Number of frames a clip spans when the causal-clip sampler is active.
    clip_length: int = 8
    # Truncated backprop-through-time window.  0 keeps the historical behaviour of detaching
    # the memory bank every frame (no cross-frame credit assignment); >=1 lets gradients flow
    # across that many frames before detaching.
    memory_tbptt_steps: int = 0
    # Scheduled sampling: probability of using the model's OWN previous prediction as the
    # Kalman observation instead of the ground-truth box.  0 = pure teacher forcing (the old
    # behaviour, which never exposes the filter to its own error); 1 = fully autoregressive.
    scheduled_sampling_prob: float = 0.0
    # Observation safety gate (4-d measurement).  ``mahalanobis`` is returned as sqrt(NIS), so
    # the chi-square 0.999 quantile 16.27 corresponds to sqrt(16.27) ~ 4.03; 3.64 is the 0.998
    # quantile and is the value recommended for tracker gating.
    nis_threshold: float = 3.64
    score_reject_threshold: float = 0.4

    # ---- training-time supervision ------------------------------------------
    corruption_enabled: bool = True
    corruption_image_prob: float = 0.06
    # Explicit token-level attack rate, decoupled from corruption_image_prob so the fraction
    # of a batch left as *naturally captured* data is directly controllable.
    #
    # Both rates are deliberately small.  LasHeR already contains the appearance challenges in
    # abundance (of 1224 sequences: partial occlusion 85.5%, total occlusion 29.7%, motion blur
    # 31.5%, low illumination 16.5%, over-exposure 9.1%, deformation 16.2%, low resolution
    # 28.1%; only 4.6% are free of any degradation).  Synthetic attacks therefore must NOT be
    # what teaches the diagnosis head about those conditions -- at 6% + 6% roughly 86% of every
    # batch is real captured data.  Their only remaining job is to supply a *dense, per-patch*
    # error target, which the per-sequence real labels cannot provide.
    corruption_token_prob: float = 0.06
    corruption_token_ratio: float = 0.4
    corruption_severity: float = 0.4
    diagnosis_alpha: float = 0.5          # e* = a*e_feat + (1-a)*e_task
    detection_prior: float = 0.2          # syndrome sigmoid bias init
    # Initial gain on the raw syndrome logit.  The stock head produces s_raw with a spread of
    # only ~0.2 over 64 checks, so s = sigmoid(s_raw) is a point mass at 0.794 and every
    # consumer of q degenerates (constant q -> random TopK routing, non-selective noise gate,
    # one frame-reliability value per batch).  ``calibrate_syndrome_gain`` sets this from the
    # measured logit std; 1.0 disables the correction.
    syndrome_logit_gain: float = 1.0

    # ---- stage-specific DINOv2 unfreezing (S3) ------------------------------
    # Name prefixes of otherwise-frozen backbone parameters whose gradient is re-enabled.  Empty
    # for every stage except S3, where it is ``["blocks.10.", "blocks.11."]`` (the last two
    # DINOv2 transformer blocks).  Consumed by ``trackit.../GOLA/builder.py``; the optimizer
    # picks the parameters up through ``optimizer.backbone_scope``.
    backbone_scope: List[str] = field(default_factory=list)

    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ helpers
    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "CodeTrackConfig":
        if not raw:
            return cls(enabled=False)
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        kwargs: Dict[str, Any] = {}
        extra: Dict[str, Any] = {}
        for k, v in raw.items():
            if k in known:
                kwargs[k] = v
            else:
                extra[k] = v
        cfg = cls(**kwargs)
        cfg.extra = extra
        return cfg

    def validate(self) -> None:
        if self.x_len != self.grid * self.grid:
            raise ValueError(
                f"x_len ({self.x_len}) must equal grid^2 ({self.grid ** 2})")
        edges = self.num_checks * self.h_links_per_check
        need = self.x_len * self.h_min_col_degree
        if edges < need:
            raise ValueError(
                f"parity-check budget too small: M*d_c={edges} < N*d_v={need}")
        if self.topk_tokens > self.x_len:
            raise ValueError(
                f"topk_tokens ({self.topk_tokens}) exceeds x_len ({self.x_len})")
        if self.num_neighbours > self.x_len:
            raise ValueError("num_neighbours exceeds x_len")
        if self.mid_dim <= 0 or self.refiner_hidden <= 0:
            raise ValueError("mid_dim and refiner_hidden must be positive")
