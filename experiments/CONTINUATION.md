# Current CodeTrack continuation state

Latest objective is the 2026-10-07 attachment ending `85990379...`: real image/native observation formation, trustworthy repair targets, matching training/inference conditions, genuine syndrome/BP and motion, final LasHeR-test PR/SR >= GOLA and RGBT234. **Goal remains active; no final acceptance claim.**

## Authoritative completed work

- Native GOLA mining E0024: all ten complete train sequences, 26,119 frames, 72 actual failure contexts; no augmentation/GT-after-init model inputs. Canonical JPEG decode and ordered producers match serial 128-frame predictions and PR/SR exactly.
- E0025 exact native replay: all 72 reproduce the original fused features exactly. Historical same-modality observations warped by past-prediction motion supply 24 qualified same-crop targets. Positive *absolute* task-gain bits improve 23/24 oracle cases, mean IoU .31992 -> .71791. Old frame-relative thresholds discard important box-correction symbols.
- E0026 exports 149 native frames (72 hard, 77 genuine good contexts) and all offered words. GT only supplies target labels. Seven sequence training / three complete-sequence holdout, one seed. No teacher-picked reference is fed to the decoder.
- NativeWordDecoder replaces only symbols jointly authorised by reference reliability and three-round exact noisy-XOR BP. Actual offered reference words determine direction; there is no free SATR residual or denoiser. Unqualified conditional error labels are erased instead of called healthy.
- E0027 diagnostic checkpoint `_probe/E0027/train/decoder.safetensors`, best update 100. Cached held-out IoU +.019808, three improvements, zero box regressions; still false writes on some unrepairable words. This is not final/calibrated/nonregression proof.
- E0028 live `theleftestrunningboy`: 200/200 ordered frames, PR 92.00 / SR 62.381 versus paired native GOLA 55.00 / 35.214. Twelve corrected frames, no current GT model input, mean four words, 28.71s. Single held-out LasHeR-train sequence, not LasHeR-test.

## Live work to revalidate first

E0028 full native `rightbackcup` and `whitebetweenblackandblue` were launched at source `e58c3ab`, private per-sequence state/output, GPU2/GPU4. `whitebetweenblackandblue` is now terminal: 928 complete frames, PR .63577586 / SR .54135883 versus paired GOLA .47844827 / .42025861 (+15.73 / +12.11 pp). `rightbackcup` was revalidated live at PID 3665766, tool session 76057; do not restart it based on an observation timeout. Inspect actual process/log/report before deciding to wait/restart; do not infer liveness from this note.

- `_probe/E0028/cup.log`, `_probe/E0028/cup/report.json`
- `_probe/E0028/whitebetween.log`, `_probe/E0028/whitebetween/report.json`

Paired baseline is in E0024 native shard reports/traces. Compare full per-sequence PR/SR, error/correction spans, unknown/reference refusal, image/bank/template/Kalman feedback and state safety. If native transfer holds, implement proper BP/unary/shuffled and preservation/temporal ablations before full LasHeR-test/RGBT234 promotion. If it regresses, identify actual state/target failure; do not launch threshold/seed/LR sweeps.

## Material outstanding requirements

Full LasHeR-test acceptance, RGBT234, architecture/training-paradigm ablations, calibrated learned detection/refusal, efficient production integration and final best/tag/commands are **not done**. Original full-test reference remains PR 77.19 / SR 61.79; historical candidate 77.34 / 61.76 fails SR. Existing native word enumeration can cost several backbone calls; current work prioritises correct ordered state and native repair evidence.

Important implementation: `tools/observed_tracker.py`, `codetrack/word_decoder.py`, `codetrack/syndrome_bp.py`, `tools/probe_native_reference_words.py`, `tools/train_native_word_decoder.py`. Source/experiment records E0023-E0028 contain exact contracts, failures and commands. Four dirty `_probe/grid/...` files predate this work; leave unrelated modifications alone.
