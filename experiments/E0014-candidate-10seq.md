# E0014: E0013 candidate on 10 LasHeR-test sequences

The standalone candidate was evaluated with the official single-sequence pipeline on GPU4.
The run used commit `f6a8ff3` for the candidate configuration and checkpoint
`_probe/codetrack_clip_candidate_full.safetensors`.

## Aggregate

| Metric | Candidate | Paired GOLA | Delta |
|---|---:|---:|---:|
| PR | 57.882 | 58.075 | -0.193 pp |
| SR | 47.929 | 48.153 | -0.224 pp |
| NPR | 55.295 | 55.675 | -0.380 pp |

The candidate is below the paired baseline and is rejected as a final model.

## Main sequence deltas

- `midredboy`: SR -4.09 pp, PR -4.00 pp.
- `boyunder2baskets`: SR -1.74 pp, PR -2.28 pp.
- `bike2left`: SR +1.09 pp.
- `rightbottlecomes`: SR +0.96 pp, PR +2.30 pp.
- `midboyNo_9`: SR +0.92 pp, PR +1.13 pp.

The errors are concentrated rather than a global collapse. A frame-level baseline-protection gate
is being tested next. The result must still be checked on a fresh split before any full LasHeR run.
