# E0025: Verifiable repair directions from native observations

- Date: 2026-10-07.
- Hypothesis: a native failed feature should be repaired toward an alternative representation built from *available* reliable template/history/motion information, rather than an arbitrary learned residual. Reference candidates must improve actual GT tracking before they are used as targets.
- Source commit: captured before each launch in `_probe/E0025/`.
- Data: E0024's 72 naturally failed frames and causal recorded histories, from the same train10. Current fused tokens come directly from the ordinary frozen backbone on native images; no image augmentation or token corruption creates these examples.
- Template arm (GPU2 if still free): current image/current recorded crop, ordinary initial template replacing the online template; additionally inspect the current image in a past-prediction Kalman crop with online/initial templates. All candidate input choices are available at inference.
- Temporal arm (GPU4 if still free): current image with one modality supplied by a motion-warped **past observation of the same modality**, normal backbone and either current or initial template. The bank contains the provided first observation plus confidence-admitted past observations; no future frame or current GT determines the warp.
- Target qualification (supervision only): candidate selected-box IoU >= .5, IoU improves by at least .02, and same-crop fixed-GT tracking loss improves by >1e-4. Different-crop motion candidates are judged as observation/geometry recovery, not falsely reported as a same-grid token target.
- Label: exact per-token reference intervention through the frozen head measures tracking harm; this is label computation, never a negative student token example. Record oracle sparse token recovery to test whether the proposed target supports the eventual decoder. Current GT only evaluates/selects training targets after encoding.
- Decision: if a target family has useful native coverage and oracle recovery gains, train a reliability/syndrome decision around that available word and validate native closed-loop transfer. If it fails, do not spend another recovery training allocation on it.
- Metrics/checkpoint: reference coverage, actual selected IoU, GT tracking loss, oracle error-bit recovery; no candidate PR/SR or trained checkpoint is claimed at this discovery stage.
- Outputs: `_probe/E0025/{template,temporal}/report.json` and qualified native target tensors. Frozen GOLA weights remain the only model checkpoint.
- First discovery run at `ab87581` found apparent target coverage, but **is not accepted as evidence**: some re-encoded observations differ from the recorded native tensors by up to 4.224. The pixel-aligned cropper's returned affine is not generally a safe repeat request because integer rasterisation rounds again. Replay must reconstruct the original requested crop from the preceding recorded prediction. The corrected run asserts the adjusted geometry is identical and maximum fused difference <= .01 before evaluating targets.
- The corrected run also compares the inherited frame-max-relative labels with all numerically significant positive task gains. The frozen GOLA head is pointwise and fixed-GT classification/box loss is additive; verify this decomposition numerically. A native token with a useful box correction must not be discarded just because another background token has a larger classification gain.

```bash
source scripts/00_env.sh
CUDA_VISIBLE_DEVICES=2 "$PYTHON" tools/probe_native_reference_words.py \
  --events _probe/E0024/native_part0 _probe/E0024/native_part1 \
  --mode template --output _probe/E0025/template
CUDA_VISIBLE_DEVICES=4 "$PYTHON" tools/probe_native_reference_words.py \
  --events _probe/E0024/native_part0 _probe/E0024/native_part1 \
  --mode temporal --output _probe/E0025/temporal
```
