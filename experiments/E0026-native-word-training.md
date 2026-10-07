# E0026: One-stage native word-reliability and syndrome decoder

- Date: 2026-10-07.
- Hypothesis: the validated available observation words already supply a useful native repair direction; learning a free SATR residual wastes that direction. Train a small reliability/error-code decoder to choose an offered word and replace only confidently harmful receiver symbols. Test whether native target evidence transfers without any GT-picked inference input.
- Source commit: captured before export/train in `_probe/E0026/`.
- Receiver/bad-token formation: actual E0024 native images through normal frozen GOLA; no augmentation or token masking. Add actual good frames from the same recorded causal contexts. Past banks for a control are rebuilt using only frames strictly before that control.
- Repair words: current other modality plus motion-warped past observations of the same modality, normal frozen GOLA, current/initial templates. Export **all offered words**, including bad/unhelpful ones. GT never selects which word is passed to the student.
- Supervision: useful-word label requires verified GT selected-IoU improvement and fixed-GT loss improvement; conditional harmful-symbol labels are all numerically significant positive task gains for a qualified word. Thus q measures actionable harm given that offered reference, not an augmentation footprint or a claim that a failed reference makes the original image healthy.
- Architecture: shared low-dimensional projections, native word-quality head, channel/error likelihood, exact degree-four XOR syndrome BP (three rounds), and direct observed-word correction. No free learned residual or denoiser. Motion enters the observed word construction, symbol likelihood and word quality.
- Forward: a symbol is replaced only when both word reliability and decoded error posterior exceed .5. Otherwise the original symbol is preserved exactly. Training uses the identical hard forward rule with a straight-through backward path. Verify rejection, both-condition acceptance, exact deployment/training equivalence and tracking gradients.
- Training: one stage, 200 updates, batch 32, lr 5e-4, one seed 42. Freeze original backbone/head. Detection fits word probability, harmful-bit probability and XOR likelihood; recovery teaches movement toward the offered qualified word, preservation keeps zero-impact/unhelpful proposals at identity, tracking checks fixed-GT classification/regression and penalises harmful proposals.
- Split: first seven complete native train10 sequences fit; last three sequences select checkpoint and validate. No LasHeR-test use. Cached native representation metrics remain a pilot; passing decoder needs a live ordered image/bank/template/motion rollout.
- Gate: held-out word selection and real selected-box improvement with healthy/unknown preservation. Then compare learned BP against unary/shuffled/oracle; only promote to live validation after useful native improvement. Do not call an oracle result or identity-only checkpoint a final tracker.
- Status/checkpoint/PR/SR: prepared.
- Export completed at `651e73a`: **149 real native frames** (72 failure endpoints plus 77 actual good context controls); 66 have a qualified same-crop target. Control banks use strictly earlier observations. All offered words and their negative alternatives are exported; the student receives no GT-selected reference.

```bash
source scripts/00_env.sh
CUDA_VISIBLE_DEVICES=2 "$PYTHON" tools/probe_native_reference_words.py \
  --events _probe/E0024/native_part0 _probe/E0024/native_part1 \
  --mode temporal --export-words --include-controls --output _probe/E0026/words
CUDA_VISIBLE_DEVICES=2 "$PYTHON" tools/train_native_word_decoder.py \
  --dataset-report _probe/E0026/words/report.json --output _probe/E0026/train \
  --steps 200 --batch-size 32 --seed 42
```
