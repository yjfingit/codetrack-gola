# E0023: Physical observation faults, trustworthy repair targets and matched student conditions

- Date: 2026-10-07.
- Status: completed at `57f81f5`; rejected for scale-up. No PR/SR result.
- Hypothesis: the earlier detector/recovery gaps partly arise from unrealistic image blackout, clean-controlled crop/template history, unreliable teacher targets and different training/inference routes. A physically formed student stream, qualified repair reference and the exact deployment route should teach useful correction that transfers to unseen impairments and natural observations. The result decides whether this training paradigm deserves closed-loop validation or needs a different repair target.
- Source commit: captured before execution in `_probe/E0023/source_commit.txt`.
- Data: versioned train10, seven training and three entirely held-out LasHeR-train sequences; two contiguous 4-frame middle-video windows per sequence, seed 42. Twenty main clips plus six native held-out clips. No LasHeR-test tuning.

## Bad-token formation

Student input starts as physical six-channel RGB-T intensities. Training impairments are RGB lowlight with noise, directional motion blur and TIR contrast compression with shared greyscale noise. Each affects one real modality before cropping, normalisation and the complete frozen GOLA backbone. No student token is masked, zeroed or replaced. Held-out sequence impairments are RGB background-patch occlusion and TIR geometric misregistration; neither family is used to fit the model. Native validation uses the same held-out sequences with no augmentation.

The augmentation returns only an image, never a fault mask. Severity is .5/.85 in the two impaired clip frames; the first/last frames receive no new impairment. Earlier impaired online templates can still affect a later unmodified observation, which is deliberately represented in the paired target.

## Repair target and labels

The teacher runs the original image through the same frozen backbone, in the same student-chosen crop. Its online template uses the clean counterpart of the region selected by the student's own prior prediction. None of this teacher stream enters student inference conditions.

The reference is trusted for feature supervision only if its selected box has GT IoU >= .5 and confidence >= .5. Otherwise feature recovery/detection labels are unknown. A harmful-token label measures whether a single-token reference intervention reduces fixed-GT tracking classification/regression loss; replacement is a detached label computation, never a student observation fault. Weak impact values are unknown. Known error bits produce exact XOR check labels. Unknown checks have a uniform syndrome likelihood target (probability .5), teaching an erasure rather than fabricated parity evidence.

Native frames have no paired reference for naturally occurring faults. Their q AUROC is therefore not claimed as a native-fault detector score. Their baseline/candidate tracking difference, trusted-frame false writes and preservation remain measurable. A native failure vetoes scale-up even if image-augmentation metrics pass.

## Training/inference contract

- Clip initialisation uses its first GT box, as tracker initialisation does. Later crop geometry, online-template updates and padding mean come from the degraded student's own frozen GOLA predictions.
- CodeTrack receives only naturally produced fused student tokens, initial/current student template context and causal prediction/motion state. Runtime hooks reject current GT, clean tokens, corruption masks and oracle routes.
- Train and inference use identical K=4 support, q >= .5 posterior abstention and absolute-posterior residual gating. No soft all-token training route. The scalar posterior threshold expresses "more likely harmful than healthy" and is not swept.
- Motion posterior rebases with each student crop and observes the accepted current head prediction after recovery. No current/next GT enters this state.
- This experiment still caches a frozen-baseline trajectory: it is an off-policy representation probe, not candidate closed-loop PR/SR. Passing candidates must next rerun the live image/backbone/candidate feedback path on held-out LasHeR-train.

## Loss, freezing and checks

Two stages: 120 detection updates, then 100 repair updates. Frozen DINOv2, LoRA, GOLA head, H and template pool. Detection fits the position-shared channel/XOR likelihoods and motion prior; repair freezes these and trains SATR. The verified degree-four error-syndrome BP decoder remains fixed (three rounds, no damping).

Detection teaches harmful-token probability and reliable/erased XOR evidence. Recovery moves known harmful tokens toward qualified teacher features. Preservation keeps trusted zero-impact and unknown-reference tokens near the received representation. Tracking uses fixed GT-positive classification targets plus box regression, with a harmful-correction margin. Current GT is supervision only. Optimiser lr=5e-4 and the inherited 5% residual bound remain fixed; no parameter search.

Prelaunch checks: three physical image/modality/route-state witnesses, two tracking-quality counterexamples/gradient checks, six exact BP/integrated checks, four physical motion-coordinate checks, and the existing structural validator. Save impairment examples and per-frame crop/prediction/reference provenance.

## Decision gate and outputs

Report actual backbone feature deviations, qualified-label coverage, q AUROC/Brier, learned/unary/shuffled/oracle syndrome comparisons, correction-direction cosine, bad-token feature recovery, trusted-token drift, actual write fraction and selected-box IoU/tracking gains. Measure native observations separately.

Require held-out correction and selected-box gains, bounded preservation, useful calibrated syndrome evidence and nonnegative native tracking transfer before live candidate closed-loop validation. Neither a green unit test nor an augmentation-only gain is a final result.

Outputs: `_probe/E0023/physical.{json,log,safetensors,trajectory.json}`, native trajectory/evaluation, and `observation_examples/*.png`. Checkpoints/raw output stay outside Git.

## Results and decision

- Real image formation: mean backbone feature deviations are .02380 lowlight, .03034 motion blur, .04163 thermal contrast/noise, .03179 occlusion, .02267 misregistration. Occlusion and misregistration reduce teacher/student selected IoU by .06866/.06497. Thermal contrast/noise instead raises mean selected IoU .01286: changed features alone are not harmful-token labels.
- Qualified reference coverage: 96.15%; strong causal label density 2.06%. No-op label density remains zero. 0.53% positive labels on frames with no *new* impairment reflect corrupted historical online templates, not mask labels.
- Best repair checkpoint: step 99, `_probe/E0023/physical.safetensors` (diagnostic only).
- Held-out-family q-AUROC .92580 versus channel .92763, shuffled syndrome .91452 and oracle .94732. q Brier .02799 versus channel .02949. Check Brier .01512 fails the constant-prevalence reference .00901.
- Training repair: direction cosine +.280, feature gain +.000544, selected-box IoU gain +.001474. Held-out-family repair: feature gain **0**, no written supervised bad tokens, selected-box IoU gain **0**. High ranking AUROC does not establish useful routing or a transferable repair direction.
- Native observations: tracking-loss gain **-.0005692**, selected-box IoU gain **0**, trusted-token drift 5.14e-6; actual writes cover .163% of tokens. No native-fault q AUROC is claimed because there is no clean native counterpart.
- Decision: reject this training recipe. The useful construction/condition controls stay, but do not scale the image augmentation recipe or tune its q threshold. E0024 mines actual native failure frames and asks whether available past/cross-modal information can supply a verifiable repair target.

```bash
source scripts/00_env.sh
CUDA_VISIBLE_DEVICES=4 "$PYTHON" tools/causal_replay_probe.py \
  --decoder-type syndrome_bp --tracking-quality fixed_gt \
  --observation-recipe physical_mix --validation-degradation heldout \
  --native-validation --save-observation-examples \
  --crop-policy baseline --abstain-threshold .5 \
  --clips-per-sequence 2 --clip-length 4 --val-count 3 \
  --diag-steps 120 --steps 100 --seed 42 \
  --output _probe/E0023/physical.json \
  --save-checkpoint _probe/E0023/physical.safetensors
```
