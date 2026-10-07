# Training-Paradigm Comparison Plan

## Current Evidence

The short causal-impact candidate has q supervision based on a token replacement intervention and
was trained with random Siamese samples. It raises selected-10 scores a little, but full-strength
write-back creates a few catastrophic sequence failures. Half-strength write-back improved the
same selected-10 result, but that set belongs to LasHeR-test and has already influenced this
choice. The ongoing full-test run is therefore confirmation after test exposure, not an unbiased
final estimate.

The existing `continuous_clip_probe.py` is not a valid learned-routing comparison: SATR training
uses the injected-region mask as an oracle q override, and current-frame ground-truth boxes feed
the motion filter. Its learned-q evaluation shows a substantial oracle-to-learned gap. It proves
that the refiner can move corrupted features under oracle conditions, not that a deployable model
learns when or where to repair.

## Next Controlled Comparison

Use one LasHeR-train sequence split for training and a disjoint LasHeR-train validation set.
Choose ten representative validation sequences by challenge coverage. Do not tune on LasHeR-test.
Run one seed first, with matched optimizer updates, frames, corruption draws, and initialization.

**A: Pairwise joint baseline**

- Random GOLA Siamese pairs; no carried state.
- Train q, SATR, and the bounded write gate jointly with frozen GOLA head/backbone.
- On paired synthetic views, define a harmful token by whether replacing it with a reliable clean counterpart reduces the frozen GOLA tracking loss. Do not equate all injector-mask tokens with harmful tokens.
- On clean and natural frames without a paired reference, train only tracking and identity/preservation behavior; unknown error labels are masked.

**B: Two-stage error curriculum**

- Stage 1: same clean/corrupted pairs and same frozen tracker; train q and SATR on causal tracking-impact supervision plus a small clean-feature recovery term on known corrupted positions. Optimize the actual learned-q route, with q detached only where needed to prevent route collapse. Do not use oracle q for the main SATR update.
- Stage 2: initialize from Stage 1 and train on contiguous 4-8 frame LasHeR-train clips. At clip start reset all state. During the clip, motion and memory may consume only prior predictions; current ground truth is used only for tracking loss and counterfactual labels. Mix mostly natural clips with a small fraction of paired synthetic faults. Train the write/abstain decision on whether candidate output reduces the frozen-head tracking loss; label ambiguous or out-of-view cases as abstain.
- Freeze DINOv2, LoRA, and GOLA head in both stages initially. Train only q/BP and SATR/write-gate parameters. Unfreeze only if the frozen-tracker candidate passes the validation gate and residual errors clearly come from feature mismatch rather than correction.

Use the smallest interpretable objective:

```text
L = L_track(output)
  + lambda_q * L_causal_detection
  + lambda_rec * L_bad_feature_recovery
  + lambda_id * L_healthy_identity
```

`L_causal_detection` is masked on uncertain natural frames. `L_bad_feature_recovery` is used only
where a paired clean target exists. `L_healthy_identity` penalizes changes to trusted tokens and
clean frames. Motion receives no separate auxiliary loss unless it demonstrably changes a causal
route and improves validation tracking.

## Gates

- q: held-out-sequence AUROC/AUPRC against causal tracking-impact labels, clean false-positive rate, and calibration.
- recovery: compare learned q, shuffled q, oracle q, and no repair; measure target-region tracking loss as well as feature distance.
- preservation: clean-frame no-write fraction and healthy-token residual norm.
- temporal causality: assert no current GT box enters current-frame recovery; compare motion on/off with identical learned routes.
- promote to full training only if learned q has positive held-out tracking gain, oracle recovery is also positive, and no large per-sequence regressions appear.
- final acceptance: full LasHeR-test PR and SR both at least paired GOLA; then RGBT234. Select no thresholds from final test data.
