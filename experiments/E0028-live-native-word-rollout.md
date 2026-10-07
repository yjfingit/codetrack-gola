# E0028: Live ordered native image/word/state rollout

- Date: 2026-10-07.
- Hypothesis: E0027's positive native feature pilot survives live crop/template/Kalman feedback. If its small receiver edits amplify into trajectory damage, the candidate is rejected before any LasHeR-test run.
- Source commit: captured before launch.
- Initial sequence: the held-out native `theleftestrunningboy`, all 200 frames from frame 0 in strict order. Paired native GOLA reference from E0024: PR .55000 / SR .35214287. No sequence is divided.
- Inputs: current original RGB/TIR, initial supplied template and confidence-admitted *past* raw observations. Motion warp uses the before-current Kalman prediction. GT enters only initialization and after-tracking evaluation; the tracking call accepts an image, not current GT.
- Mechanism: enumerate current/past-modality observation words, encode through normal frozen GOLA, infer reference reliability and exact error-syndrome posterior, jointly decode qualified symbols. No GT-selected word, clean frame, token corruption mask or future frame. Repaired raw frames do not automatically update/poison online templates.
- Erasure witness: an untrained/uncertain decoder on a real 4-frame prefix returns exactly the original baseline boxes. The original AMP head result is returned directly whenever no symbol is accepted.
- E0027 pilot outcome: 44 held-out native cached frames, mean selected-box IoU gain **+.019808**, three improve and none regress. Reliability still falsely authorizes some unrepairable words, so this live test is discovery/validation, not a final deliverable or calibrated-mechanism claim.
- Sampling/training/checkpoint: no new fitting; `_probe/E0027/train/decoder.safetensors`, seed 42, best update 100. Original frozen GOLA checkpoint is unchanged.
- Output: live ordered per-frame predictions/correction telemetry and official OPE-style train-sequence PR/SR under `_probe/E0028/`. Final LasHeR-test / RGBT234 remain outstanding.
- First complete live result at `50f3670`: **PR .92000 / SR .62380952**, paired native GOLA .55000 / .35214287; **+37.00 / +27.17 pp** on this one held-out train sequence. All 200 frames completed, 12 corrected frames, mean four offered words, wall time 28.71s. This is a strong native transfer result on one sequence, not a final test score or proof of global nonregression.
- Next: the remaining two held-out native sequences, `rightbackcup` and `whitebetweenblackandblue`, in independent ordered processes with unique state/output paths. Inspect refusal/correction telemetry and full paired per-sequence PR/SR before full LasHeR-test promotion.
- Extended validation: `whitebetweenblackandblue` completes **928/928** ordered frames, PR **.63577586** / SR **.54135883** versus paired native GOLA **.47844827 / .42025861**, gains **+15.73 / +12.11 pp**. Wall time 434.09s. `rightbackcup` remains a live ordered job; its longer full-sequence score is not available yet.

```bash
source scripts/00_env.sh
CUDA_VISIBLE_DEVICES=2 "$PYTHON" tools/mine_native_failures.py \
  --sequences experiments/manifests/E0028-leftboy.txt \
  --decoder-checkpoint _probe/E0027/train/decoder.safetensors \
  --output _probe/E0028/live --workers 4 --prefetch 8 --verify-prefetch 0 \
  --events-per-sequence 0
```
