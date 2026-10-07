# E0027: Conditional error likelihood and joint reliability erasure

- Date: 2026-10-07.
- Hypothesis: separately exceeding .5 for word reliability and symbol error does not establish a likely successful correction. Fit P(error | useful word) only where a useful reference is known, and decode only when P(useful word) * P(error | useful word) >= .5. This is a probability-contract correction, not a threshold search.
- Data/receiver/reference: identical E0026 native word export, 149 native frames, complete 7/3 sequence split. No new augmentation, data split or seed. All reference inputs remain inference-available; GT only establishes supervised reference qualification and signed task gains.
- Supervision: word BCE uses all offered words. Conditional channel/decoded-bit BCE uses GT-qualified words only; an unqualified reference does not manufacture a healthy-token label. Its syndrome target is uniform (.5), expressing erased check information. Recovery/preservation/tracking retain the four-purpose objective.
- Architecture/change: native observed-word correction, same three exact BP rounds and frozen GOLA. Joint probability controls both training and inference hard forward. Exact original/current or observed/reference values are copied with torch.where; no free residual or denoiser.
- Witness: two .6 scores jointly authorize no change (.36 < .5); uncertain words preserve receiver exactly, accepted symbols copy actual offered word exactly, training/deployment forward is identical, and tracking gradients exist. Four CPU witnesses pass.
- Training: single stage, 200 updates, batch 32, lr 5e-4, seed 42. Source commit captured before launch.
- Gate: improve held-out native selection with calibrated joint evidence and refuse unrepairable/unknown frames before live closed-loop validation. No cached gain counts as final PR/SR.
- Pilot completed at `804f7e9`: best update 100, 44 held-out native frames, mean selected-box IoU gain **+.019808**, three improved and zero regressed; seven frames write. Word Brier .17062. The unqualified-reference conditional bit Brier must not be treated as a calibrated error score. Several unrepairable words still get false writes; this candidate has not passed final refusal/calibration gates. E0028 checks live state amplification on a short complete held-out sequence.

```bash
source scripts/00_env.sh
CUDA_VISIBLE_DEVICES=2 "$PYTHON" tools/train_native_word_decoder.py \
  --dataset-report _probe/E0026/words/report.json --output _probe/E0027/train \
  --steps 200 --batch-size 32 --seed 42 --conditional-errors
```
