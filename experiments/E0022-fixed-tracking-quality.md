# E0022: Fixed reference quality for correction training

- Date: 2026-10-07.
- Hypothesis: candidate-dependent IoU classification labels allow a worse predicted box to reduce the tracking loss. Fixing the quality target should make the recovery objective a more useful definition of a harmful token and repair success.
- Independent counterexample: with classification confidence .1, a perfect box has the old loss 2.30259. Enlarging the box reduces IoU to .25 but lowers the old total loss to 1.40467. With the same fixed reference quality target, the worse box instead increases loss to 3.05259.
- Change: `_tracking_loss` optionally accepts a detached `score_quality_map`. The paired clean teacher supplies quality at GT-positive locations. Candidate updates cannot change this classification target. Counterfactual label evaluation repeats the same target across token interventions.
- Scope: opt-in correction training; the upstream GOLA objective and default tracking evaluation are unchanged.
- Verification: two CPU witnesses prove the counterexample and its correction, per-sample consistency, detached teacher target and finite student gradients. Pass.
- Bad-token construction: no token corruption; this changes supervision only. The next experiment generates degraded observations at the physical image layer and runs the complete normal backbone.
- Checkpoint / PR/SR: none; next is an image-formation/repair-target experiment.

```bash
source scripts/00_env.sh
"$PYTHON" tools/verify_tracking_quality.py
```
