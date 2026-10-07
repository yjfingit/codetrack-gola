# E0020: Exact binary-error syndrome decoding core

- Date: 2026-10-07.
- Hypothesis: the existing neural BP does not implement extrinsic sparse parity propagation faithfully, so a trained unary detector cannot be presented as evidence that ECC helps.
- Code audit of `NeuralBPSyndromeDiagnosis.forward`: variable-to-check totals sum variables within a check instead of other checks at the receiver; sign products include absent edges; the channel prior is added after message propagation. The visual residual-energy score also supplies no calibrated GF(2) parity-bit observation.
- New core: `codetrack/syndrome_bp.py` implements error variables e_i in {0,1}, check bits XOR over H support, channel LLR including its prior, extrinsic messages over the correct axes, and uncertain check likelihoods. A syndrome probability of 0.5 sends zero message and preserves the channel posterior exactly.
- Scope: a mathematical decoder primitive. It is not yet connected to the GOLA likelihood encoder or deployed in a checkpoint. No claim of tracking improvement or natural-fault detection.
- Verification: five CPU checks compare a single factor and a tree to exhaustive posterior enumeration, prove nonedge isolation and uncertain-check identity, and check a finite nonzero gradient through a zero-valued symbol. All pass.
- PR/SR: none.
- Checkpoint: none.
- Next implementation: a position-shared visual likelihood encoder trained on causal error labels and their exact XOR check targets, with motion evidence from E0017. Compare learned syndrome, shuffled syndrome, uncertain syndrome and exact oracle syndrome. A learned parity likelihood must pass held-out calibration before any tracking promotion.

```bash
source scripts/00_env.sh
"$PYTHON" tools/verify_syndrome_bp.py
```
