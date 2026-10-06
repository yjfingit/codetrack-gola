# CodeTrack Architecture Audit (2026-10-05)

## Bottom line

The repeated S1 failures are primarily architectural-contract failures, not a lack of
refiner capacity. The ECC idea is promising, but the current implementation is an
ECC-shaped detector followed by a learned residual network, not a differentiable ECC
decoder. Several interfaces then give the detector contradictory targets or give the
recovery branch information at training time that is unavailable at inference time.

S2 remains locked. No training was started after this audit was requested.

## Evidence from this repository

| Observation | Evidence | Interpretation |
|---|---|---|
| Refiner has capacity | `_probe/multiregion_capacity_h35/lr1e3.json`: fixed four-region training reached gains `0.03687, 0.00969, 0.00502, 0.00277` (center to bottom-right) | The module can learn several locations when routing/data are fixed. |
| Production H34 does not generalize | `_probe/grid/s1_h34_seed_validation/`: all four seeds improve only `0.00011-0.00169` across four regions | The bottleneck is diagnosis/conditioning/supervision, not raw capacity. |
| Diagnosis can rank damage, but recovery still fails | `_probe/diagnosis_direction_h7_final.json`: q AUROC `0.742`, q std `0.00133`; `_probe/oracle_gate_h8.json`: learned gain `-0.000277`, oracle hard gain `-0.000643` on the same fixed crop | The current refiner contract is not guaranteed to improve even when a useful damage ranking exists. |
| H36 launcher did not inherit q pretraining | H37 logs show strict-load failure on denoiser keys; H34 `start/seed42.bin` is byte-identical in diagnosis tensors to H33 q checkpoint, while H36 used the default base weight | Checkpoint composition is part of the architecture protocol and is currently fragile. |

## Structural defects

### 1. `H` is not yet an error-correcting decoder

`codetrack/ecc.py:132-161` creates a positive, row-normalised incidence matrix. The
forward path (`codetrack/ecc.py:334-370`) computes

```
C_obs = H W_x(X_TIR),  C_ref = H W_r(X_RGB + template),
s = MLP(C_obs - C_ref),  q = sigmoid(-H^T s + bias).
```

There are no codewords, parity constraints, signed symbol log-likelihoods, variable-node
messages, check-node updates, extrinsic messages, or iterations. `H^T s` is therefore a
linear backprojection of 64 scalar scores to 256 tokens. A 64-to-256 projection cannot
uniquely decode arbitrary token errors; different damage patterns have the same checks.
The random/free edges also make the inverse spatial map seed-dependent. This explains why
center damage can work while peripheral locations do not.

The ECC contribution should remain the central idea, but the core needs to become a
small, explicit neural BP / normalized-min-sum decoder on the Tanner graph.

### 2. The syndrome and token posterior have contradictory signs

`trackit/models/methods/GOLA/gola.py:329-334` constructs a positive syndrome target with
`H @ error_target` (`gola.py:329-334`). The criterion also pushes `s` toward that positive
target (`codetrack/criteria.py:341-345`). But `ecc.py:369` computes `q_logits = -H^T s + bias`,
so a larger supervised syndrome lowers the error probability q. The later q-sign fix made
the fixed probe rank better, but it did not remove this loss-level contradiction. The q BCE,
syndrome SmoothL1, and optional rank loss can ask the same parameters to move in opposite
directions.

Use one signed convention throughout. For example, let positive variable LLR mean
"damaged" and let every check message use that convention; then `q = sigmoid(LLR_error)`.
Do not train `s` as a positive error average and negate it again at the variable output.

### 3. The current syndrome is formed after destructive mixing

The code first aggregates tokens with `H` and only then compares observation and reference
(`C_obs - C_ref`). Errors from multiple tokens can cancel or be averaged inside a check,
so the check value is not a faithful sufficient statistic for each token. A better order is:

1. compute per-token soft evidence / error LLR from TIR, RGB and temporal context;
2. pass those symbol messages through differentiable check-node updates;
3. decode variable posteriors with extrinsic messages.

The graph then supplies structured redundancy instead of asking one MLP to rediscover a
decoder from already-collapsed check vectors.

There is also a redundancy-budget mismatch: the current graph has `N=256` variables and
`M=64` checks (`h_links_per_check=12`, minimum column degree 3). Even if all checks were
independent, this is only 64 scalar constraints for 256 token variables. The experiments
route up to `K=64` tokens, while the synthetic attack damages 16 tokens. Nothing in the
repository measures the graph's rank, stopping sets, minimum distance, or erasure-correction
radius. Before claiming error correction, calculate these quantities for the fixed H and
choose K below a measured correction regime; otherwise call the graph a learned spatial
prior rather than an ECC decoder.

For the current seed/config I measured real rank **64**, GF(2) rank **64**, exactly 768
edges, and column degree exactly 3. That confirms the graph is full row-rank but does not
establish a useful minimum distance or decoding radius.

### 4. Training uses privileged target-region information

During training `CodeTrack.forward` passes `gt_box_xywh` into
`_search_target_mask` (`codetrack/codetrack.py:346-360`). `TemporalMemory` then pools the
current feature only inside that ground-truth target mask (`codetrack/motion.py:457-467`),
and the resulting prior tokens condition the refiner and denoiser. The fixed inference
probes do not provide this ground-truth box. This is a direct train/eval conditioning gap,
even though the box is not fed to the tracking head as a tensor.

Training must use the same observable inputs as inference: no target mask in the recovery
path. A box may remain a loss target for the motion prior, but the memory readout used by
recovery must be pooled from predicted reliability / previous-frame state only.

### 5. Forced Top-K has no abstention mode

`codetrack/codetrack.py:380-391` standardises q per frame and always selects exactly K
tokens. Even an intact frame is forced to undergo a recovery write. This is unsafe for a
tracker whose dominant case is “nothing is damaged”. A calibrated ECC decoder should output
both an error posterior and an erasure/abstain decision. Route only tokens satisfying a
posterior threshold (with an optional K cap), and take the exact identity path when no token
passes it.

### 6. The refiner cannot perform a real correction step

`codetrack/recovery.py:107-186` gives each suspect only fixed H-neighbours, the same-position
RGB token, pooled template, and pooled memory. It does not exchange variable-to-check or
check-to-variable messages, and it does not expose RGB neighbours to the correction context.
If the damaged TIR token's local H-neighbours are also degraded, the refiner is asked to
reconstruct from contaminated evidence. The fixed graph should determine the message route,
but the decoder should pass *extrinsic clean evidence* rather than raw neighbours alone.

### 7. The denoiser is not a trained diffusion objective

`codetrack/recovery.py:216-412` injects random feature noise and performs two gated residual
writes, but there is no noise-level prediction target or score-matching / denoising target.
The loss only asks the final feature to beat the input. This makes the denoiser a second
unconstrained residual network behind the refiner, so it can undo a good correction. It
should be removed from S1 and only reintroduced after the ECC decoder + refiner pass a
four-location identity/harm gate. If reintroduced, train it with an explicit conditional
denoising target and a monotonic harm guard (`d_final <= d_refiner` per batch/region).

### 8. The target is not one well-defined notion of error

The diagnosis target is a post-transform cosine discrepancy between clean and corrupted
tokens (`gola.py:316-324`), while `lambda_diag_rank` uses the injector's exact pre-transform
mask (`criteria.py:353-364`). A transformer can spread a patch corruption to neighbouring
tokens, so these are different labels. At the same time, `L_rec` uses clean-feature
similarity on the selected tokens and `L_gain` uses a separate relative margin
(`criteria.py:432-479`). The model is simultaneously asked to predict propagated feature
error, exact erased patches, and tracking improvement. Keep the exact mask as an auxiliary
label, but make the primary decoder target a clearly defined *latent error posterior* and
use one sign/metric for the recovery target.

## What the literature suggests

1. **Nachmani et al., “Learning to Decode Linear Codes Using Deep Learning,” arXiv:1607.04793.**
   Neural BP learns weights on the messages of a real Tanner graph while retaining the
   variable/check update structure. This is the closest principled replacement for the
   current `H^T s` backprojection.
   https://arxiv.org/abs/1607.04793

2. **Buchberger et al., “Learned Decimation for Neural Belief Propagation Decoders,”
   arXiv:2011.02161.** Learned decimation explicitly identifies unreliable variables and
   then reruns decoding. This supports adding an ECC-native erasure/abstention stage rather
   than forcing a fixed Top-K write.
   https://arxiv.org/abs/2011.02161

3. **Buchberger et al., “Pruning and Quantizing Neural Belief Propagation Decoders,”
   arXiv:2011.13594.** Check-node importance can be learned and pruned. It supports keeping
   a sparse, spatially meaningful graph while learning which checks matter, instead of
   learning arbitrary dense MLP compensation after aggregation.
   https://arxiv.org/abs/2011.13594

4. **Li et al., “RGB-T Object Tracking: Benchmark and Baseline,” arXiv:1805.08982.**
   RGB-T tracking benefits from explicit modality reliability and evaluates occlusion
   sensitivity. The benchmark's lesson is directly relevant: complementary modalities are
   not interchangeable, so the recovery path needs modality-aware reliability rather than
   treating RGB as a generic same-space codeword.
   https://arxiv.org/abs/1805.08982

5. **Zhu et al., “FANet: Quality-Aware Feature Aggregation Network for Robust RGB-T
   Tracking,” arXiv:1811.09855.** Quality-aware modality aggregation is a safer auxiliary
   design than assuming RGB and TIR projected features have identical semantics at every
   token.
   https://arxiv.org/abs/1811.09855

6. **Wang et al., “MFGNet: Dynamic Modality-Aware Filter Generation for RGB-T Tracking,”
   arXiv:2107.10433.** Dynamic modality-aware filters and direction-aware context address
   heavy occlusion and motion; they motivate a learned RGB-to-TIR adapter/context block,
   while leaving the ECC decoder as the central innovation.
   https://arxiv.org/abs/2107.10433

7. **Zhang et al., “Jointly Modeling Motion and Appearance Cues for Robust RGB-T Tracking,”
   arXiv:2007.02041.** Motion is useful when appearance is unreliable, but it is a separate
   cue and should not be injected through a privileged ground-truth target mask during
   training.
   https://arxiv.org/abs/2007.02041

8. **Xu et al., “Towards Effective and Efficient Adversarial Defense with Diffusion Models
   for Robust Visual Tracking,” arXiv:2506.00325.** Diffusion-style tracking defenses use
   explicit reconstruction/semantic/structural objectives. This is evidence against the
   current two-step random-noise residual block being called diffusion without a matching
   denoising objective.
   https://arxiv.org/abs/2506.00325

## Recommended redesigns

### Preferred: ECC-native neural BP + abstaining refiner

Keep the fixed sparse Tanner graph and the ECC claim, but replace `SyndromeDiagnosis` with:

1. a per-token symbol encoder producing a signed error LLR from `(X_TIR, X_RGB,
   template/memory)`;
2. 2-4 unfolded weighted min-sum or sum-product iterations on the fixed H graph, with
   learned damping/normalisation and extrinsic messages;
3. a variable posterior `q_error` and a separate `q_confidence` / abstain head;
4. a refiner that consumes check-to-variable messages plus RGB/TIR neighbour features,
   writes only where `q_error > threshold`, and has an exact no-write path;
5. no denoiser in S1.

Losses should be:

```
L = L_track(corrected) + 0.25 L_track(clean)
  + lambda_q BCEWithLogits(error_LLR, exact_damage_mask)
  + lambda_soft Huber(q_error, post_transform_error)
  + lambda_rec * d(corrected, clean) on damaged tokens
  + lambda_id * d(corrected, input) on non-damaged tokens
  + lambda_parity * ||check_messages - H(error_LLR)||.
```

Do not add a separate positive syndrome regression with the opposite sign. Train and validate
the decoder with no ground-truth box in memory/recovery inputs.

### Lower-risk salvage: keep modules, repair contracts first

If a full unfolded decoder is too invasive, the minimum credible architecture experiment is:

- remove the `-` sign conflict by defining one q/s convention;
- compute token evidence before H aggregation and use H only for structured message mixing;
- remove target-mask pooling from training;
- replace forced Top-K with threshold-plus-cap abstention;
- train refiner with oracle q first, then learned q, and retain only if learned routing
  matches oracle routing;
- keep denoiser disabled;
- make checkpoint loading filter optional denoiser keys automatically when the effective
  config disables it.

This is a diagnostic repair, not the final ECC contribution. It should be used to test whether
the graph contract is now learnable before adding any temporal/diffusion component.

### Alternative: ECC detector plus modality adapter

Keep ECC responsible only for error localisation. Add a small RGB-to-TIR conditional adapter
(cross-modal attention or FiLM) to predict a clean TIR residual. This is likely more robust
when RGB/TIR features are not aligned, but it weakens the claim that ECC itself performs the
correction. It is a good ablation and fallback, not the preferred main architecture.

## Decision

Do not spend another round on learning-rate, Top-K, or denoiser sweeps. First implement and
unit-test the ECC-native message contract and remove privileged target-mask conditioning. Then
run only deterministic fixed-batch probes: (1) q direction/AUROC, (2) oracle-vs-learned routing,
(3) four-region refiner gain, and (4) identity harm on clean inputs. Only a positive causal
probe justifies a new three-seed S1 run.
