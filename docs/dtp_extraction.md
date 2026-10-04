# DTPTrack — Implementable Mechanism Extraction

Sources: `docs/paper/03_S_DTPTrack_DriftResilientTemporalPriors_CVPR2026.pdf` (10 pp., text layer clean), `docs/paper/03b_S_DTPTrack_arXiv2604.02654.pdf` (13 pp., has Supp. Mat.), code `repo/DTPTrack/`.
Code files: `trackit/models/methods/DTPTrack/DTPTrack.py` (254 lines — all DTPTrack logic in L51–146), `.../modules/frame_wise_causal_mlora/attention.py`, `.../block.py`, `trackit/criteria/methods/DTPTrack/__init__.py`, `trackit/runners/evaluation/.../pipelines/DTPTrack/__init__.py`, `config/DTPTrack/{Base,Large}/run.yaml`, `config/DTPTrack/{Base,Large}/config.yaml`.

Notation: B batch, D embed dim (768 ViT-B / 1024 ViT-L), 4 provider frames = z0 (GT template) + z1,z2,z3 (dynamic refs); x0 = current search. B-224/B-378: template 112/196 px → 8×8 / 14×14 tokens (N=64/196); search 224/378 px → 16×16 / 27×27 (256/729).

## 1. Temporal Reliability Calibrator (TRC)

- **Inputs**: per-frame token embeddings `Z_i ∈ R^{N_i×D}` plus a binary mask `M_i ∈ {0,1}^{N_i}` (1 = patch overlaps target bbox). Paper: masks are rasterizations of the bbox onto the ViT patch grid (p.4, Sec. 3.2.1). In code the mask is produced by `template_foreground_indicating_mask_generation` / `..._all_template` plugins: `template_mask[bbox_y0:y1, x0:x2] = 1` on the 8×8 (or 14×14) grid — i.e. a **box-filled block**, not true patch-overlap (a small deviation).
- **Summary (masked average pooling)**, Eq. 1 p.4: `s_i = Σ_j Z_ij·M_ij / (Σ_j M_ij + ε)`, ε = 1e-6 in code, `s_i ∈ R^D`. Code: `_get_target_summary`, `summary_feat = (feat*mask_float).sum(1)/(mask_float.sum(1)+1e-6)`.
- **Reliability score**, Eq. 2 p.4: `[c1,c2,c3] = Sigmoid(MLP([s1,s2,s3]))`, `c_i ∈ [0,1]`. Gate `f_gate` = `Linear(D, D/4) → ReLU → Linear(D/4, 1) → Sigmoid` (code L68–73). Stacked shape `(B,4,D) → (B,4,1)`.
- **Anchoring**: `c_0 ≡ 1.0` (z0 is GT-derived) — code L132–135 sets `trc_final_weights[:,0,:]=1`, i.e. it **computes the gate on all 4 frames then discards c0** (paper's Eq. 2 takes only `[s1,s2,s3]` — minor discrepancy).
- **Suppression**: element-wise, broadcast over D: `ŝ_i = s_i · c_i` → `(B,4,D)` (code L137).
- **What makes a frame "reliable"**: purely **learned and unsupervised** — there is no reliability/IoU target, no auxiliary loss, no detach. The gate sees only the pooled target-foreground feature of that frame. Supp. Fig. 3 p.12 shows learned weights of 1.0, 0.71, 0.21, 0.95 (seq-A) and 1.0, 0.56, 0.19, 0.0 (seq-B) — occlusion/artifact frames get low scores.

## 2. Temporal Guidance Synthesizer (TGS)

- **No cross-attention, no FiLM, no gating.** It is an **additive modulation of learnable base tokens**.
- `P_base ∈ R^{1×K×D}` learnable, `N(0, 1e-6)` init; `K = num_prior_provider_frames × num_prior_tokens_per_frame = 4 × 1 = 4` (code L52–58). The paper calls K "a small hyperparameter" (p.4); the code **hard-wires K = number of provider frames** (default 1 token/frame, never overridden in any config).
- `f_mod`: `Linear(D, D/4) → ReLU → Linear(D/4, D)`, applied to `Ŝ ∈ R^{B×4×D}` → `M ∈ R^{B×4×D}` (code L61–65).
- Eq. 3 p.4: `P_dyn = P_base + f_mod([ŝ0,ŝ1,ŝ2,ŝ3])`; code L143–144 additionally adds `tgs_pos_embed` and `tgs_token_type_embed` → `P_dyn ∈ R^{B×4×D}`.
- **Injection** (Sec. 3.3 p.4–5, Eq. 4): token sequence = `[cls?, reg?, P_dyn, Z_0, Z_1, Z_2, Z_3, X_0]` (`_fusion`, L194–210). Chunk sizes = `[prefix+4+64, 64, 64, 64, 256]` for B-224, i.e. **P_dyn and Z_0 share the first chunk** exactly as the paper states. FWCA: each chunk attends to the concatenated K/V of itself + *all preceding* chunks (causal), so every later frame sees the priors (`attention.py` L83–93). `past_kv_enable_bits=[False]+[True]*4`.
- Paper: priors "guide the tracker's attention … without directly contaminating its raw visual feature processing" — i.e. they are never fused into `X_0`; they only act as extra K/V.

## 3. Motion / velocity / displacement modelling

**None. Explicitly absent.** No velocity, no Kalman filter, no trajectory/flow predictor, no displacement loss exists in the paper's method section (Sec. 3, pp. 3–5) or in the code — `grep -iE "momentum|kalman|velocity|optical_flow|displacement"` over `trackit/`+`config/` returns only optimizer-momentum code. Motion is only *implicit*, via the 3 dynamic reference frames + causal attention over them.
Two motion-proxy facts:
- Table 6 p.8 explicitly ablates **heuristic motion priors as TGS replacements**: momentum-based prior (LaSOT 73.8 AUC / VastTrack 40.1 AUC) and optical-flow-based prior (73.2 / 39.4) both lose to the learned TGS (74.3 / 40.7). So the authors tried explicit motion cues and rejected them as *replacements*.
- Inference reference frames are chosen by a hand-designed temporal-segmentation rule `select_memory_frames` (pipeline `__init__.py` L306–316): for `t < 2(n-1)=6`: `[0] + range(1,t)[:3]` (padded by duplicating the earliest); else `dur = t//3`, frames `[0, dur/2, 3·dur/2, 5·dur/2]`. Paper calls this "adapted from SPMTrack… extended to three dynamic reference frames" (p.5).

## 4. Training recipe

- **Sequence**: 5 frames/chunk — `num_template_frames: 4` (z0,z1,z2,z3) + `num_search_frames: 1` (x0) (`Base/run.yaml` L113–114, p.5).
- **Sampling**: `sample_mode: "interval"`, `max_gaps: 200`, `MAX_SAMPLE_INTERVAL: 400`, `auto_extend: {step: 5, max_retry_count: 20}`; datasets LaSOT + TrackingNet + GOT-10k + COCO-2017, all sampling weights 1.0; `samples_per_epoch: 131072`.
- **Teacher forcing**: yes for *inputs* — training templates/search and the TRC masks all come from **ground-truth** boxes/labels; at inference they come from predicted boxes. **No teacher forcing and no supervision for the reliability gate itself.** **No `detach()` anywhere** in the model (only loss-logging `.detach().cpu()` in the criteria) — summaries are differentiable; ViT `patch_embed` + all backbone params are frozen (`requires_grad=False`).
- **Losses** (Supp. Table 8 p.11; `Base/run.yaml` criteria): BCE-with-logits, **IoU-aware soft target** (`iou_aware_classification_score: true`, target = IoU(GT, predicted box)), weight 1.0; GIoU weight 1.0, `warmup_epochs: 0` (the ×10 regression-loss warmup exists in `SPMTrackCriteria` but is disabled). Because `enable_aux_output=True, aux_output_multi_head=True`, losses are **summed over all 4 outputs** (z1,z2,z3,x0) — per-frame deep supervision.
- **Optimisation**: AdamW lr 1e-4, wd 0.1 (no wd on 1-D/embed params), grad-clip 1.0, cosine schedule, 2-epoch linear warmup from 1e-7, 170 epochs, global batch 128, AMP fp16, `torch_compile` on.
- **Warm-up / curriculum**: only LR warmup (2 epochs). **No reliability curriculum and no scheduled history-length curriculum.**
- **Inference extras not in the paper**: `update_criteria: 0.85` — a new memory frame is admitted only if the predicted max score ≥ 0.85, else the last memory frame is duplicated; `template_area_factor: 2.0`; `window_penalty: 0.4`.
- **Code vs paper numeric mismatches**: Hann window penalty **0.4 (code) vs 0.45 (Table 8)**; `lr_min` **1e-6 (code) vs 5e-6 (Table 8)**; `num_epochs` 170 for B (matches) but **180 in `Large/run.yaml`** vs 170 in Table 8. `drop_path` 0.0 (B) / 0.1 in `Large/config.yaml` (matches Table 8).

## 5. Ablations that matter

All ViT-B/224, LaSOT / VastTrack, Table 5 p.8 (full model **(f) 74.3 / 40.7 AUC**):
- **(a) Fixed threshold instead of learned gate** — LaSOT 72.0 AUC / 77.7 PNorm, VastTrack 38.2 / 37.0 → **−2.3** AUC. *This is the reliability-estimation ablation; learned gating is worth 2.3 pts.*
- **(b) Fully gated z0** (stop anchoring c0=1.0) — 73.2 / 79.3, 40.1 / 39.4 → **−1.1**. *The anchor is essential.*
- **(c) w/o base token** (use only f_mod output) — 72.7 / 78.4, 39.0 / 38.2 → **−1.6**. *TGS aggregation ablation.*
- **(d) Concat fusion** (concatenate summary vectors instead of dedicated prior tokens) — 73.4 / 79.7, 40.3 / 39.8 → **−0.9**. *Injection-mechanism ablation.*
- **(e) Baseline**: extended 5-frame LoRATv2 **without** DTPTrack — 73.3 / 82.8, 40.1 / 39.9 → full model **+1.0 AUC**.
- **History length** (Table 7 p.8): 2F 72.0/81.3 (VastTrack 39.1/38.7); 3F 73.2/82.3 (39.7/39.2); 4F 73.7/83.1 (40.3/39.8); 5F (ours) 74.3/83.6 (40.7/40.2) — monotone gain, +2.3 AUC from 2→5. ⚠️ Caveats: Table 7 rows are reported as *total* sequence length (2F AUC 72.0 equals the plain 2-frame LoRATv2-B224 baseline in Table 1), and row (a) is numerically **identical to Table 5(a) Fixed Threshold** (72.0/81.3, 39.1/38.7) — likely a copy error in the paper; treat the 2F point as unreliable.
- **Motion-prior alternatives** (Table 6 p.8): momentum 73.8/80.3 (40.1/39.5); optical flow 73.2/79.3 (39.4/38.7); TGS 74.3/83.6 (40.7/40.2).
- **Plug-in generalisability** (Table 4 p.8): OSTrack-256 69.1→70.1 (+1.0) LaSOT, 33.6→35.4 (+1.8) VastTrack; ODTrack-B 73.2→73.7 (+0.5); LoRAT-B224 71.1→71.9 (+0.8); overhead +0.32–0.51 G MACs, +0.54–1.28 M trainable params.
- **Cost** (Table 3 p.6): DTPTrack-B224 53.8 G MACs, 128.0 M params, 16.5 FPS on A100; L378 581 G, 407 M, 6.1 FPS. SOTA claims: LaSOT 77.5 AUC, GOT-10k 80.3 AO (L378, Table 1 p.6).

## Code ↔ paper agreement

**Good agreement on the core mechanism**: masked-avg-pool → sigmoid-gate MLP → c0=1.0 anchor → element-wise reweight → additive modulation of learnable base priors → prepend to token sequence with priors+template in the first FWCA chunk. All equations map 1:1 to `DTPTrack.py` L95–146 and L194–218 (quoted in §1–2 above).
**Deviations**: (i) gate MLP input includes s0 (discarded after); (ii) K fixed at 4 = #provider frames rather than a free hyperparameter; (iii) summaries are pooled *after* pos-embed + token-type-embed, not at the raw patch-embed output as the paper implies; (iv) box-filled mask rasterization vs. patch-overlap; (v) 0.4 vs 0.45 Hann penalty, 1e-6 vs 5e-6 lr_min, 180 vs 170 epochs (Large); (vi) `update_criteria: 0.85` is code-only. Note the repo is a **derived** LoRATv2/SPMTrack codebase (classes still named `SPMTrackCriteria`, `SPMTrack_EvaluationPipeline`, `SPMTrack_TrainingTupletSampler`); git history is a single squashed commit, so paper-vs-released-code drift cannot be dated.

## WHAT IS DIRECTLY REUSABLE FOR A GOLA-BASED RGB-T TRACKER WITH A KALMAN MOTION PRIOR

1. **TRC as a drop-in per-reference-frame gate for your Kalman-updated reference states.** Masked-average-pool each reference frame's tokens into `s_i ∈ R^D` (cheap: one sum over N tokens), pass a 2-layer MLP `D→D/4→1` + sigmoid, force `c_0=1.0` on the GT template, and multiply `s_i *= c_i`. For RGB-T you can pool per modality and concatenate (`[s_i^RGB, s_i^T]`) or average the two masks' pooled features. Cost ≈ 2·D²/4 params — the ablation (a) shows a *learned* gate beats a fixed threshold by **2.3 AUC** (Table 5 p.8), and (b) that not anchoring c0 costs **1.1 AUC**.
2. **Feed the Kalman innovation into the gate rather than thresholding it.** The code's gate sees only pooled appearance; the paper's fixed-threshold variant (the natural analogue of a hand-tuned Kalman/score threshold) is the worst ablation. Concretely: append the Kalman innovation norm / Mahalanobis distance / predicted score to `s_i` before the gate MLP. **Do not replace TGS with a Kalman/momentum prior**: Table 6 p.8 shows hand-crafted motion priors score 73.8 (momentum) and 73.2 (optical flow) vs 74.3 for the learned synthesizer — motion must be a *feature*, not the output.
3. **TGS as a near-free guidance channel: 4 prior tokens, additive modulation, no cross-attention.** `P_dyn = P_base(4×D) + f_mod(Ŝ) + pos_embed + token_type_embed`, prepended to the token sequence. Reported overhead is only **+0.32–0.51 G MACs / +0.54–1.28 M params for +0.5 to +1.8 AUC** (Table 4 p.8). For your tracker, extend `f_mod`'s input to include the Kalman state (position, velocity, log-variance) so the priors become motion-aware guidance without becoming a hard motion model.
4. **Chunk placement rule if you use any causal/one-stream attention**: group the prior tokens with the **initial template** in the first chunk so all later frames attend to them causally (`_fusion` L194–210; `attention.py` L83–93). For a two-stream RGB-T fusion block, the equivalent is prepending priors as extra K/V to the template stream only, never merging them into the search tokens.
5. **Memory admission with a score threshold + temporal segmentation, which is where much of the drift control actually lives.** Inference uses `num_templates=4`, `update_criteria=0.85` (admit a new memory frame only if predicted max score ≥ 0.85, otherwise duplicate the last), and `select_memory_frames` (`[0]` + 3 frames at `dur/2, 3dur/2, 5dur/2` with `dur=t//3`) — none of this is in the paper and all of it is directly portable. For RGB-T, replace the single score with a fused (or per-modality) score.
6. **Training recipe to copy verbatim**: 5-frame tuplet (`num_template_frames=4` + 1 search), GT boxes for templates (teacher forcing) and GT masks for TRC — but **no supervision on the gate and no detach**, so the gate is learned end-to-end from BCE(IoU-aware target)+GIoU only (weights 1.0/1.0, summed over all 4 per-frame auxiliary heads). Backbone frozen, LoRA r=64 on all attn/MLP projections, AdamW 1e-4 / wd 0.1, clip 1.0, cosine, 2-epoch warmup, batch 128, 170 epochs. Budget ≥4 reference frames: Table 7 p.8 shows monotone gains to 5 total frames.
