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
| E0008 | E0005, full LasHeR-test retry | 245 sequences | pending | pending | GPU4 retry; pending | pending | [record](E0008-full-scale-half-retry.md) |
| E0009 | Legacy continuous-clip SATR probes | 10 LasHeR-train clips, length 4 | n/a | n/a | feature diagnostics only; oracle q used for SATR training | not evidence of learned correction timing | [record](E0009-legacy-clip-probes.md) |
| E0010 | Causal clip recipe: soft training route, frozen q during SATR, residual trust-region clip | 10 LasHeR-train clips, length 4, 1 seed | n/a | n/a | learned-route feature gain -0.0000013; healthy drift 0.000115 | diagnostic pass for stability, not a tracking result; no full evaluation | [record](E0010-causal-soft-trust.md) |
| E0011 | Same recipe with strict 7/3 sequence holdout and validation checkpoint selection | 7 train + 3 unseen LasHeR-train clips, length 4, 1 seed | n/a | n/a | held-out q-AUROC 0.876; bad-token recovery +0.00038; tracking-loss gain +0.0625; healthy drift 0.000180 | retain as first promising training recipe; needs split/seed confirmation | [record](E0011-causal-holdout.md) |
| E0012 | E0011 repeated with seed 43 | same 7/3 holdout, length 4 | n/a | n/a | held-out q-AUROC 0.853; bad-token recovery +0.000372; tracking-loss gain +0.0553; healthy drift 0.000180 | retain; proceed to 10-clip head comparison | [record](E0012-causal-holdout-seed43.md) |

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
