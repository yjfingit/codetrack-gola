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

```bash
source scripts/00_env.sh
CUDA_VISIBLE_DEVICES=4 "$PYTHON" tools/mine_native_failures.py \
  --output _probe/E0024/native --workers 4 --prefetch 8 \
  --verify-prefetch 128 --events-per-sequence 8 --clip-length 8
```
