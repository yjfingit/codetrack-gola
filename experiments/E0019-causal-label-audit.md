# E0019: No-op interventions contaminated causal labels

- Date: 2026-10-07.
- Hypothesis: normalising tiny head-batch differences by the largest positive gain can make an identical-token replacement appear harmful.
- Evidence: E0018 attempt 1 (`67ed7ac`) and corrected attempt (`58b1b2a`) have exactly identical `.trajectory.json` files, including all crop parameters and baseline predictions. Their causal-label positive fractions are **25.9570% versus 2.6758%**; strong fractions are **24.9170% versus 1.6357%**. The target calculation is the only change affecting these statistics.
- Fix: return zero for an entirely identical input/reference pair, and explicitly zero the impact of identical individual tokens before relative-gain normalisation. The meaning follows the intervention itself; no threshold was tuned.
- Independent witness: `tools/verify_causal_noop.py` runs the actual pretrained GOLA head over 12 seeded random inputs. It observes up to 9.54e-7 singleton/batched output disagreement, although those particular fixtures already had zero labels before the fix. The real feature trajectory comparison establishes the defect, not the random fixture alone.
- Corrected E0018 unedited-frame positive fraction: **0**.
- Commit: `58b1b2a`.
- Checkpoint: none; E0018 supplies the corrected experiment.
- Consequence: historical causal-q AUROC and bad/healthy recovery diagnostics require remeasurement with no-op-safe labels. Retain their recorded tracking PR/SR, which were measured directly, but do not promote a mechanism based on the old label diagnostics.
- Next: evaluate genuine noisy-syndrome decoding and real-fault labels, with independent selected-box tracking evidence.
