# E0024: Native hard-frame and repair-target discovery

- Date: 2026-10-07.
- Hypothesis: first/middle short windows and paired artificial faults do not teach what goes wrong in actual long-running GOLA tracking. Full native trajectories can expose recoverable representation failures and their available reliable past observations; these determine whether to use temporal/cross-modal counterfactual targets or reject an unrepairable frame.
- Source commit: recorded before execution in `_probe/E0024/source_commit.txt`.
- Data: same ten curated LasHeR-train sequences. Every sequence starts at frame 0 and is consumed completely in strict order; no sequence is divided between workers. No augmentation, token manipulation, clean image or future observation.
- Student policy: ordinary frozen GOLA backbone/head, repository Hann postprocessor and simple online-template updater, with the canonical JPEG decoder. Initial GT sets the tracker once; `track(image)` accepts no current GT. Kalman observes actual previous outputs and predicts from past state, but does not alter baseline tracking boxes during mining.
- Hard example: GT-valid IoU below .5, or invalid/unknown visibility. GT is evaluated after tracking solely for research selection/targets. Save up to eight nonoverlapping 8-frame contexts per sequence, actual frozen-backbone tokens and their initial/current online-template state. Store observed history from confidence-only admissions plus the supplied initial reference. GT quality tags remain supervision metadata.
- Teacher discovery next: run candidate current/past-modality image observations through the normal backbone in the recorded crop; use current GT only to verify tracking improvement and select a reliable target. A physical-image teacher target must prove a useful direction before training another recovery branch.
- CPU/GPU path: four decode producers, ordered prefetch queue (8), pinned memory, non-blocking transfer, one GPU consumer. First verify the same 128-frame prefix against serial decode with exact predictions and PR/SR; restart at frame 0 for whole-sequence mining.
- Configuration/source/checkpoint/log: `_probe/E0024/`, frozen `weights/gola_b224.bin`, commit + command below. Event tensors remain outside Git. Full JSON traces and a report prove sequence completion; an in-progress status never counts as completion.
- PR/SR scope: LasHeR-train target discovery, not LasHeR-test or a final candidate score.
- First launch (`1c367c8`) stopped at the first event serialization: clip/history views shared tensors, which safetensors refuses to save. No complete sequence report was produced. Preserve `_probe/E0024/native.log`; retry clones independent archive tensors before save.
- Verified before the stop: the 128-frame `biketurndark` serial/prefetch predictions were exactly identical (maximum pixel difference 0), PR .984375 / SR .71428573 in both paths.
- Retry scheduling: GPU2 became free in the live query. Two versioned manifests distribute whole sequences by frame count across GPU2/GPU4, with unique outputs and no intra-sequence splitting. Auxiliary whole-image Kalman noise/floor uses full-frame units rather than crop-normalised extent limits; it does not affect baseline predictions.
- Completed retry at `17ac08f`: both reports are terminal, all 10 distinct sequences complete, **26,119 frames**, **72 native failure contexts**. Part0 has 13,004 frames / 11 events, part1 13,115 frames / 61 events. Among event endpoints the annotated target is fully in the search crop for 55, partly in it for 8, and absent for 9; a repair target cannot simply assume every crop still contains the target.
- Both shard prefetch witnesses pass exact 128-frame box equality with maximum difference zero and matching PR/SR. The full-sequence macro reference over the ten native train sequences is PR **66.413%**, SR **52.214%**; this is a training research baseline, not the official LasHeR-test score.
- Outputs: `_probe/E0024/native_part{0,1}/report.json`, complete per-frame traces, atomic event provenance and frozen normal-backbone tensors. E0025 tests whether inference-available template/history/motion reference words provide an actual useful repair direction.

```bash
source scripts/00_env.sh
CUDA_VISIBLE_DEVICES=4 "$PYTHON" tools/mine_native_failures.py \
  --output _probe/E0024/native --workers 4 --prefetch 8 \
  --verify-prefetch 128 --events-per-sequence 8 --clip-length 8
```
