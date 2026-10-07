# CodeTrack Experiment Summary

All PR/SR/NPR values below are percentages unless marked as fractions. LasHeR's challenge
attributes overlap. The official full-test GOLA reference is PR 77.19 / SR 61.79 / NPR 73.76.
The worktree was dirty at commit `0280b14095123a0e894fd1f184732b75964a07b3`; experiment records
identify the exact config, checkpoint, and log used.

| ID | Change | Data / protocol | PR | SR | Delta vs paired/reference GOLA | Decision | Checkpoint / record |
|---|---|---|---:|---:|---|---|---|
| E0001 | GOLA reference, SATR write disabled | LasHeR-test selected 10 | 58.07 | 48.15 | paired reference | control | [record](E0001-paired-baseline.md) |
| E0002 | 20-update causal-impact CodeTrack, spatial SATR, scale 1.0 | same 10 | 58.99 | 48.21 | +0.92 / +0.06 pp | promising locally; fails full test | [record](E0002-causal-spatial.md) |
| E0003 | E0002 with motion module enabled | same 10 | 58.07 | 48.15 | 0 / 0 pp | reject: no measurable motion effect | [record](E0003-motion-ablation.md) |
| E0004 | E0002, full LasHeR-test | 245 sequences | 77.16 | 61.79 | -0.03 / 0.00 pp vs GOLA full test | reject as-is | [record](E0004-full-spatial.md) |
| E0005 | E0002 with inference residual scale 0.5 | same 10 | 59.35 | 48.63 | +1.28 / +0.48 pp | continue full-test validation | [record](E0005-scale-half.md) |
| E0006 | E0002 with q abstention threshold 0.8 | stopped before completion | pending | pending | early sequences reverted to baseline | stop low-value sweep | [record](E0006-high-threshold.md) |
| E0007 | E0005, full LasHeR-test | 245 sequences | n/a | n/a | interrupted at 238/245; no result archive | invalid/incomplete | [record](E0007-full-scale-half.md) |
| E0008 | E0005, full LasHeR-test retry | stopped at 36/245; no live worker | n/a | n/a | no complete archive | invalid/incomplete | [record](E0008-full-scale-half-retry.md) |
| E0009 | Legacy continuous-clip SATR probes | 10 LasHeR-train clips, length 4 | n/a | n/a | feature diagnostics only; oracle q used for SATR training | not evidence of learned correction timing | [record](E0009-legacy-clip-probes.md) |
| E0010 | Causal clip recipe: soft training route, frozen q during SATR, residual trust-region clip | 10 LasHeR-train clips, length 4, 1 seed | n/a | n/a | learned-route feature gain -0.0000013; healthy drift 0.000115 | diagnostic pass for stability, not a tracking result; no full evaluation | [record](E0010-causal-soft-trust.md) |
| E0011 | Same recipe with strict 7/3 sequence holdout and validation checkpoint selection | 7 train + 3 unseen LasHeR-train clips, length 4, 1 seed | n/a | n/a | held-out q-AUROC 0.876; bad-token recovery +0.00038; tracking-loss gain +0.0625; healthy drift 0.000180 | retain as first promising training recipe; needs split/seed confirmation | [record](E0011-causal-holdout.md) |
| E0012 | E0011 repeated with seed 43 | same 7/3 holdout, length 4 | n/a | n/a | held-out q-AUROC 0.853; bad-token recovery +0.000372; tracking-loss gain +0.0553; healthy drift 0.000180 | retain; proceed to 10-clip head comparison | [record](E0012-causal-holdout-seed43.md) |
| E0013 | E0011 final candidate with standalone checkpoint and all-clip probe | 7/3 holdout + all 10 probe clips, seed 42 | n/a | n/a | all-clip q-AUROC 0.747; bad-token recovery +0.000542; tracking-loss gain +0.0458; healthy drift 0.000109 | retain as checkpoint candidate; next run production 10-sequence PR/SR | [record](E0013-final-candidate.md) |
| E0014 | Production-style 10-sequence evaluation of E0013 candidate | LasHeR-test selected 10, serial, GPU4 | 57.882 | 47.929 | -0.193 / -0.224 pp vs paired GOLA (PR/SR) | reject candidate as final; midredboy and boyunder2baskets are damaging | [record](E0014-candidate-10seq.md) |
| E0015 | Frame-level baseline-protected write-back, isolated parallel evaluation | LasHeR-test selected 10, GPU4, NPROC=2 | 57.926 | 47.964 | -0.149 / -0.189 pp vs paired GOLA (PR/SR) | reject as final; slight improvement but still below baseline | [record](E0015-baseline-protected.md) |
| E0016 | CPU prefetch: 8 eval workers + 4 I/O threads | `bike2left` serial correctness check, GPU4 | 94.55 | 81.34 | exact metric match to serial; single-sequence wall time 22.99s | retain as optional full-eval setting; single short sequence is slower due startup overhead | [record](E0016-cpu-prefetch.md) |
| E0017 | Correct moving-crop Kalman geometry and observation timing | 4 physical/state witnesses; structural 97/97 | n/a | n/a | state correctness only | opt-in replay path; production integration still required | [record](E0017-causal-motion-coordinates.md) |
| E0018 | Causal predicted-crop replay versus GT-centred control | train10, 7/3 sequence holdout, mid-video clips, seed 42 | n/a | n/a | causal q-AUC .624 vs oracle-crop .760; both selected-box IoU gains negative | reject legacy recipe; no scale-up | [record](E0018-causal-replay.md) |
| E0019 | No-op-safe causal labels | exactly matched 20 real clip trajectories | n/a | n/a | strong label density 24.92% -> 1.64% | old mechanism diagnostics need remeasurement | [record](E0019-causal-label-audit.md) |
| E0020 | Noisy GF(2) error-syndrome sum-product decoder | five exact-posterior/causality/gradient witnesses | n/a | n/a | decoder primitive, not tracking result | pass mathematics; visual likelihood integration pending | [record](E0020-exact-syndrome-decoder.md) |
| E0021 | Visual parity likelihood + exact syndrome BP + motion evidence | same train10 causal crops, 7/3 sequence holdout, seed 42 | n/a | n/a | q AUC .916 vs unary .907 / shuffled .900; probability calibration and selected-box gates fail | retain verified decoder; reject recipe for scale-up | [record](E0021-visual-syndrome.md) |
| E0022 | Detached clean-reference tracking quality target | worse-box loss counterexample + gradient witness | n/a | n/a | old loss rewards worse IoU; fixed target removes this escape | pass objective witness; image-level training comparison next | [record](E0022-fixed-tracking-quality.md) |
| E0023 | Physical-image fault formation, qualified clean repair reference and matched student conditions | train10; 7/3 sequence split; held-out impairment families + native observations | n/a | n/a | held-out q AUC .926 but bad-token recovery 0; native tracking loss worsens .000569 | reject recipe; preserve condition controls | [record](E0023-physical-observation-training.md) |
| E0024 | Native hard-frame/history mining and target discovery | 10 complete native train sequences; 26,119 frames | 66.413 | 52.214 | train research reference; 72 natural failure contexts | complete; target discovery next | [record](E0024-native-target-mining.md) |
| E0025 | Initial-template / motion / temporal-modality reference words | 72 native failures, GT only for target verification | n/a | n/a | available observation/template sources; no fake student token faults | prepared | [record](E0025-native-reference-words.md) |
| E0026 | One-stage native observed-word reliability + exact syndrome decoding | native hard frames + actual healthy contexts, 7/3 sequence split | pending | pending | all offered references are student-visible; GT selects supervision only | prepared | [record](E0026-native-word-training.md) |
| E0027 | Conditional error likelihood + joint reference/error erasure | same native export and seed; no threshold sweep | pending | pending | fixes probability semantics before deployment | prepared | [record](E0027-conditional-word-erasure.md) |
| E0028 | Live native word/Kalman/template rollout | complete held-out native 200-frame sequence | 92.00 | 62.381 | +37.00 / +27.17 pp vs paired native sequence | one-sequence transfer passes; other native sequences running next | [record](E0028-live-native-word-rollout.md) |

## Current Read

The causal q detector and the correction vector solve different problems. Current full-strength
write-back gives large gains on some sequences and catastrophic losses on a few others. Scaling
the residual by 0.5 improves the selected-10 aggregate and is being checked on all LasHeR-test.
Enabling motion without a measurable change is not evidence that motion is helping; it must later
be connected to a trained, causal routing or abstention decision and tested separately.

No full RGBT234 evaluation or final converged CodeTrack training has yet been completed.
The selected-10 threshold and residual-scale exploration used LasHeR-test sequences. Treat these
as exploratory diagnostics with test exposure; future hyperparameter selection must use a
LasHeR-train held-out validation split, leaving the official LasHeR-test for final confirmation.
