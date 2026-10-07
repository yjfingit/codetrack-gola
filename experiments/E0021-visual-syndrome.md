# E0021: Learned noisy error-syndrome likelihood and exact BP

- Date: 2026-10-07.
- Status: completed at `c358c8e`; no promotion to full training.
- Hypothesis: explicitly supervising XOR check bits and decoding their calibrated likelihood can improve harmful-token detection beyond a shared unary channel, while an uncertain syndrome should cause no correction. Causal motion should supply target-location evidence directly to detection.
- Source commit: captured before launch in `_probe/E0021/source_commit.txt`.
- Data/sampling: same versioned train10, 7/3 complete sequence holdout, two middle-video contiguous clips per sequence, length 4, baseline-predicted crop policy. No LasHeR-test access for this choice.
- Check construction: one overlapping 2x2 binary error-parity check per 16x16 cell (256 checks, degree four); H support alone defines GF(2) parity. Row-normalised H is used only to aggregate visual features and route recovery neighbours.
- Likelihoods: position-shared unary head consumes received TIR, pooled target identity, causal motion log prior and uncertainty. A shared check head consumes RGB/TIR discrepancy at its four variables and predicts P(XOR harmful bits=1). These visual likelihoods overlap statistically; a factor approximation requires held-out calibration and is not an independence theorem.
- Labels: harmful bit = no-op-safe causal replacement impact >= .5. Weak positive labels below .5 are unknown; checks touching an unknown bit do not receive XOR supervision. Natural unedited frames have zero *additional paired-fault impact*, not a claim that they are naturally healthy. Natural-fault detection remains an open requirement.
- Detection loss: channel BCE + decoded posterior BCE + known-check XOR BCE, with no positive-class reweighting. Initial channel prior .05; initial syndrome logit zero (uncertain). No spatial vote-bias vectors or learned edge gains.
- Training: two stages, 120 detection updates (including motion prior), then 100 SATR updates with likelihoods/motion frozen. Frozen DINOv2, LoRA, GOLA head, H and template pool throughout. Tracking/recovery/preservation objective retains E0018 settings. K=4; posterior correction threshold .5 (more likely harmful than healthy); three BP rounds, no damping.
- Checks before launch: existing preflight plus six exact-posterior/sparse-isolation/uncertainty/gradient/integrated-identity tests and four physical motion-coordinate tests.
- Mechanism gate: compare held-out q AUROC/Brier against unary channel, shuffled check likelihoods and exact oracle syndrome; compare check Brier against constant-prevalence prediction. Motion-off uses the same weights. Do not claim ECC helps if only unary classification improves or if the likelihood encoder cannot predict parity.
- Tracking gate: positive held-out tracking and selected-box IoU gains with bounded healthy drift, then independent closed-loop LasHeR-train validation. Feature-cache metrics never count as final PR/SR.
- Outputs/checkpoint: `_probe/E0021/visual.{json,log,safetensors}` plus trajectory and source metadata. No large output enters Git.
- Held-out best step: 0; q-AUROC .91602, unary .90671, shuffled syndrome .89994, oracle syndrome .97826. Check parity AUROC .92396.
- Probability calibration: q Brier .02376 versus unary .02218; check Brier .01121 versus constant prevalence .01063. The rankings improve, but the calibrated-likelihood gate fails.
- Recovery: tracking-loss gain +0.00004287, bad-token feature gain +0.00000140, healthy drift 7.41e-8; selected-box IoU gain **-0.00002529**. Later SATR updates overfit. Reject as a final tracker or scale-up candidate.
- Motion off: tracking gain +0.00001994, selected-box IoU gain +0.00027988 and q-AUROC .91368. This does not establish beneficial motion correction.
- Checkpoint: `_probe/E0021/visual.safetensors`, diagnostic only. PR/SR not evaluated.
- Next decision under the updated task: image-level realistic impairment, trustworthy clean repair targets and matched student observation/history conditions take priority over further decoder tuning.

```bash
source scripts/00_env.sh
CUDA_VISIBLE_DEVICES=4 "$PYTHON" tools/causal_replay_probe.py \
  --decoder-type syndrome_bp --crop-policy baseline --abstain-threshold .5 \
  --clips-per-sequence 2 --clip-length 4 --val-count 3 \
  --diag-steps 120 --steps 100 --seed 42 \
  --output _probe/E0021/visual.json \
  --save-checkpoint _probe/E0021/visual.safetensors
```
