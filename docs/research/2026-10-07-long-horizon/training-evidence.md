# Long-horizon training and evidence audit

Audit date: 2026-10-07. This audit reads existing code and completed artifacts; it does not run inference, modify model code, touch running jobs, or treat a proposed experiment as a result. Paths below are repository-relative. Numbers newly computed here use saved JSON only.

## 1. What the completed evidence establishes

The native reference-word route can materially improve two held-out training sequences, but it regresses on the 8,574-frame `rightbackcup`. There is no completed full-test result for this route and no demonstrated long-horizon correction from the later E0030 safeguards. This supports redesigning sequential state estimation and training exposure, rather than claiming the current gates have solved accumulation.

| Complete paired sequence | Frames including initialization | Native GOLA PR / SR | E0027 decoder live PR / SR | Delta in percentage points |
|---|---:|---:|---:|---:|
| theleftestrunningboy | 200 | 55.000 / 35.214 | 92.000 / 62.381 | +37.000 / +27.167 |
| whitebetweenblackandblue | 928 | 47.845 / 42.026 | 63.578 / 54.136 | +15.733 / +12.110 |
| rightbackcup | 8,574 | 44.332 / 33.945 | 31.257 / 25.170 | −13.074 / −8.775 |

Provenance: `experiments/E0028-live-native-word-rollout.md:6,13,15`; `experiments/E0029-live-code-ablation.md:12`; `_probe/E0028/{live,whitebetween,cup}/report.json:6`. Cup terminal completion is explicitly recorded at `_probe/E0028/cup/report.json:10-15`. Older continuation text still describing that job as live is stale.

The three-sequence macro PR/SR improves from 49.059/37.062 to 62.278/47.229. However, weighting these same per-sequence scores by sequence length gives 44.888/34.744 to 35.601/28.707. The latter is an explanatory aggregation, not the official benchmark aggregate. Report both per-sequence outcomes and the official macro metric: otherwise a severe long-sequence regression is hidden by two short gains.

This comparison does **not** isolate elapsed time as a causal variable: sequence identity and difficulty are confounded with length. Cup is already difficult near its start, and both systems repeatedly lose and recover overlap. The accurate claim is observed long-sequence instability and worsened state trajectories, not a demonstrated monotonic error law on every video.

### Error-code mechanism remains unestablished

On the same complete 200-frame sequence, learned BP obtains PR/SR .92000/.62381; unary obtains .58000/.36000; shuffled check likelihoods obtain .92000/.62595. Removing message passing changes the trajectory, but shuffling learned check locations does not damage it. This is insufficient to attribute gains to correct learned parity placement. Source: `experiments/E0029-live-code-ablation.md:5-11`, terminal `_probe/E0029/{unary,shuffled}/report.json:6-15`.

### Latest E0030 artifacts are cached pilots

| Artifact | Best update | Cached held-out IoU gain | Write rate | Word Brier |
|---|---:|---:|---:|---:|
| E0030/train | 25 | 0 | 0/44 | .147619 |
| E0030/train_v2 | 0 | 0 | 0/44 | .164327 |
| E0030/seed42 | 0 | 0 | 0/44 | .164327 |
| E0030/seed43 | 0 | 0 | 0/44 | .164320 |
| E0030/seed44 | 0 | 0 | 0/44 | .164325 |
| E0030/group42b | 375 | +.006096 | 1/44 | .397724 |

Each source is `_probe/E0030/<artifact>/report.json:2-25`; each report explicitly calls its scope a cached native representation pilot. `group42b` has zero false writes among **one** selected frame, which is not evidence of a low deployment error rate. There is no complete E0030 live rollout in these artifact directories. Step 0 is the first evaluated training update, not necessarily untouched random initialization (`tools/train_native_word_decoder.py:221-226`).

## 2. Important version separation

The live E0028 results used the pre-safety E0027 checkpoint and older tracker behavior. Recorded sources are `50f36705fed30cf3acf29754af4e1e3f64bc9cb0` for the first sequence and `e58c3abcbad2b0af0cd3082037b4005ee7c96181` for extended sequences, in `_probe/E0028/source_commit.txt:1` and `_probe/E0028/extended_source_commit.txt:1`.

The **current** `tools/observed_tracker.py:164-182` explicitly treats repaired output as provisional: state uses the original baseline box/confidence, repaired frames do not update template/history, and motion consumes the state box. Consequently, historical E0028 deterioration is not a measured result of the current provisional-feedback implementation. The current code also adds abstention/utility gates (`codetrack/word_decoder.py:80-100`) that the old E0027 checkpoint did not train. An architecture proposal must evaluate this current state contract separately and must not present an already implemented isolation rule as a new fix.

## 3. Saved cup trajectory diagnostics

Inputs: baseline `_probe/E0024/native_part1/seq00_trace.json:1` and decoder `_probe/E0028/cup/seq00_trace.json:1`. Each is a one-line JSON array of frames 1–8,573, excluding supplied initialization; all 8,573 annotations have positive dimensions. All metrics below are descriptive post-tracking analyses, not information available to the tracker.

| Diagnostic | Baseline | E0028 decoder |
|---|---:|---:|
| Frames with any accepted token | 0 | 2,739 (31.95% of post-init frames) |
| Target fully inside actual search crop | 7,330 | 6,530 |
| Target partly inside actual search crop | 788 | 1,341 |
| Target wholly outside actual search crop | 455 | 702 |
| Token writes while target wholly outside crop | n/a | 224 |
| IoU < .1 and reported confidence > .84 | 0 | 405 |
| Longest contiguous IoU ≤ .1 span | 251 frames | 270 frames |
| Longest contiguous IoU ≤ .5 span | 252 frames | 326 frames |

First decoder write and first prediction difference both occur at frame 59. Of the 405 confident near-total localization errors, the target is fully inside the crop on 287 frames, partially inside on 72, and outside on 46. Therefore crop loss is material but cannot explain all failures; wrong-target evidence can also be confident within a valid crop. High repaired confidence is not an independent correctness certificate. These measurements are consistent with feedback amplification but do not by themselves identify which internal update caused each failure.

Crop classification was computed by applying saved affine parameters to the GT rectangle, then intersecting it with `[0,224] × [0,224]`: zero-area intersection is absent, complete containment is full, and the remainder is partial.

| Cup frame range | Baseline mean IoU | Decoder mean IoU | Decoder writes |
|---|---:|---:|---:|
| 1–200 | .1712 | .1711 | 44 |
| 201–500 | .4228 | .3766 | 37 |
| 501–1,000 | .3849 | .3508 | 78 |
| 1,001–2,000 | .2465 | .1571 | 363 |
| 2,001–3,000 | .6295 | .3789 | 272 |
| 3,001–4,000 | .4009 | .2739 | 157 |
| 4,001–5,000 | .4031 | .3970 | 308 |
| 5,001–6,000 | .3564 | .2104 | 312 |
| 6,001–7,000 | .2079 | .1492 | 465 |
| 7,001–8,000 | .1965 | .1646 | 369 |
| 8,001–8,573 | .1837 | .1319 | 334 |

Mean IoU is computed directly from trace values and differs from official SR, which averages a success curve over thresholds. These windows show repeated loss/recovery and increasing exposure to false decisions, not monotonically worsening IoU.

## 4. Actual training exposure is much smaller than the mined runtime

E0024 runs all ten sequences, totaling 26,119 frames, but saves only 72 native failure endpoints and 8-frame contexts (`experiments/E0024-native-target-mining.md:16-18`). Collection caps the number of events per sequence and records the first qualifying, sufficiently spaced failures (`tools/mine_native_failures.py:113,127-151`), rather than distributing examples throughout elapsed time. After the cap, the remaining trajectory is evaluated but contributes no new captured failure context.

E0026 adds 77 healthy context controls, for 149 total native frames; 66 have some qualified same-crop reference (`_probe/E0026/words/report.json:4-5`). The controls are selected from the earliest/latest healthy locations of those same short contexts; their banks use earlier observations only (`tools/probe_native_reference_words.py:45-82`).

| Split | Unique frames | Offered-reference rows | Qualified rows | Qualified-row prevalence |
|---|---:|---:|---:|---:|
| First seven train sequences | 105 | 1,540 | 399 | 25.91% |
| Last three validation sequences | 44 | 428 | 77 | 17.99% |

These counts were recomputed from `_probe/E0026/words/report.json` using the split in `experiments/train10.txt:1-10`. Reference rows sharing a frame are highly dependent and are not 1,540 independent trajectories. There are no duplicate `(sequence, frame)` pairs.

Most importantly, the 8,574-frame cup sequence contributes only **10 validation frames, all between frames 31 and 150**. The 928-frame whitebetween sequence contributes 12 frames between 214 and 369. Training ``whiteboy`head`` has 12,782 frames but only 22 exported frames between 5 and 1,492. Full-sequence mining therefore did not translate into broad late-state training or validation exposure.

The current training script loads cached tensors and samples independent frame groups; all words of a selected frame are trained together (`tools/train_native_word_decoder.py:31-42,155-163`). With default batch size 32 it samples only two frame groups per update, even though the resulting number of word rows varies. It does not advance images, crop, template, history, or motion under the current learned policy. This is a one-step, off-policy training distribution, not long-horizon training or even a causal multi-step unroll.

## 5. Loss and probability contracts that need redesign

1. **One-step full-reference labels are narrower than deployment utility.** A word is qualified if its complete replacement has IoU ≥ .5, improves IoU by ≥ .02, and improves fixed-GT loss by > 1e−4 (`tools/probe_native_reference_words.py:200-209,269-278`). Bit labels are positive pointwise loss gains masked by this full-word qualification. Such labels can exclude an unqualified full word with useful partial edits and can favor a current improvement that damages future state. Neither label directly supervises identity, target presence, delayed confirmation, or long-run state damage.

2. **The current gain target is not signed damage.** Training sets `utility = label × (0.1 + 0.9 tanh(max(raw_gain,0)))`, assigning zero to every unqualified word (`tools/train_native_word_decoder.py:73-87`). This removes the distinction between mildly useless and catastrophically harmful proposals. `codetrack/word_decoder.py:38-39` still describes signed task-loss improvement, so that comment overstates the actual target.

3. **Mean minus learned standard deviation is not a calibrated confidence bound.** Gaussian NLL fits expected utility and a standard deviation (`tools/train_native_word_decoder.py:176-178`), then deployment computes `gain_lcb = expected_gain - gain_std` (`codetrack/word_decoder.py:84-92`). There is no coverage level, held-out residual quantile, epistemic uncertainty estimate, or distribution-shift guarantee. Calling this quantity an LCB does not make it a statistically valid lower bound.

4. **Quality and abstention duplicate evidence.** Quality BCE and abstention BCE use complementary versions of the same useful-word label; abstention additionally weights useful-word rejection by four (`tools/train_native_word_decoder.py:167-175`). The token write probability multiplies `q × quality × (1−abstain)` (`codetrack/word_decoder.py:91`). This is a gating score, not an established calibrated joint probability: the last two factors are neither distinct calibrated conditional events nor independent evidence. Weighted BCE also changes the posterior represented by the abstention output.

5. **Post-selection risk matters.** Training evaluates all references but chooses the maximum quality score among eligible words (`tools/train_native_word_decoder.py:120-138`), matching live top-1 selection (`tools/observed_tracker.py:241-253`). Row-level Brier or BCE is insufficient to calibrate the chosen maximum, especially when candidate counts and dependencies vary. Calibration must run the complete candidate-generation and selection policy, including no-op, and condition on or account for candidate count/search area. Adding more candidates can increase false acceptance even at unchanged row-level calibration.

6. **The false-write measure is a proxy.** It checks whether the selected full-word qualification label is false (`tools/train_native_word_decoder.py:138`), not whether actual partial correction causes immediate or future regression. Conversely, a qualified word may still produce harmful partial edits. The .25 validation false-write cutoff (`tools/train_native_word_decoder.py:230-236`) is not a time-uniform safety budget. A single successful selected validation example gives almost no evidence about repeated deployment decisions.

7. **Reported bit Brier mixes unknown and known conditional labels.** Conditional bit fitting uses qualified words only and erases unqualified syndrome targets (`tools/train_native_word_decoder.py:194-202`), but evaluation concatenates all q values and all exported bits (`tools/train_native_word_decoder.py:139-147`). Unqualified bits were exported as zero. That aggregate cannot be interpreted as calibration of `P(error | useful word)`.

8. **The frozen head loss is useful but not a sufficient sequence objective.** Fixed GT quality targets avoid a previous self-referential target loophole and pointwise decomposition is explicitly checked (`tools/probe_native_reference_words.py:189-197,241-246`). However, the training loss still penalizes immediate pointwise BCE/GIoU, feature recovery, preservation, and immediate regression (`tools/train_native_word_decoder.py:208-218`). It omits future visibility, reacquisition latency, persistent identity switches, and memory admission damage.

For context, the empirical validation useful-word prevalence is .17991, whose prevalence-only constant predictor has Brier .14754. E0027 word Brier .17062 and E0030 group42b .39772 do not establish calibrated selection. This prevalence-only value is a descriptive comparator fitted to the validation prevalence, not a deployable trained baseline.

## 6. Training redesign constraints and concrete interventions

### Data and state distribution

- Split complete identities/sequences into training, calibration, model-selection validation, and a final untouched test set. The current three repeatedly consulted validation sequences are development data, not a fresh acceptance test.
- Retain native video and causal initialization. Sample uniformly over elapsed-time strata as well as failure/recovery events; include natural healthy operation, occlusion, out-of-crop targets, distractor takeover, modality degradation, and delayed reacquisition. Replace the first-eight-event cap with reservoir/time-stratified collection under a defined storage budget.
- Aggregate states generated by the current policy, plus baseline and prior checkpoints. Save crop, template, memory, motion, candidate set, update actions, and relevant randomness. This exposes training to the actual states its own past decisions create. GT remains post-observation training supervision only.
- Use continuous forward state with a curriculum of 32/128/512-frame rollout segments and long full-sequence validation. Truncate gradients at chunk boundaries if needed, but carry and detach the actual state rather than resetting to GT. Frozen backbone features can be cached only when the image/crop/template that produced them still matches the sampled state.

### Separate counterfactual supervision from calibration

For a state snapshot `s_t` and offered action `a` (including no-op, local candidate, larger search, and proposed state/template admission), branch training-only copies of the state. Continue each branch on the same future raw frames under the declared follow-up policy. Define a **signed** horizon target such as

`U_H(s_t,a) = Σ[k=0..H−1] γ^k (IoU_after_a(t+k) − IoU_after_noop(t+k)) − λ_id · identity_error − λ_lost · extra_lost_frames − λ_cost · extra_compute`.

Record the actual partial-output action and the state-update/admission action separately. Short horizons can label many candidate actions; a subset of long branches estimates delayed failures. Such future labels are legitimate training-only counterfactual outcomes, provided no future frame or GT enters the online decision. They are not proof that a learned utility estimator is calibrated.

Fit the action/utility model on training sequences. Then freeze model and candidate generation and calibrate **the final selected action** on separate continuous calibration rollouts. Report selective risk versus coverage, false accepts per 1,000 frames, false memory admissions, lost duration, and recovery time. If using empirical/conformal bounds, explicitly state the exchangeability or dependence assumptions and horizon covered; temporal dependence, repeated selection, adaptive memories, and distribution shift prevent casually claiming independent-sample or time-uniform guarantees.

### Objectives and acceptance evidence

- Add direct target identity/presence supervision and target-vs-clutter comparisons. Use localization losses only when the target is visible/in support; use absent-target/no-update supervision otherwise. A confident regression head score alone is not identity evidence.
- Train setwise action choice with an explicit no-op, signed immediate and future utility, and a state-admission loss. Preserve frozen GOLA initially to identify whether the new evidence/state policy works; enable small adapters only after this is measurable.
- Keep any XOR/BP route optional until learned checks beat unary, shuffled, and uniform-check controls on multiple full trajectories under matched candidates and computation. Feature repair, reference selection, and sequential detection should each have separable ablations.
- Evaluate the current provisional-feedback code against the historical feedback code, then independently vary confirmation/admission, search recovery, training state distribution, and decision calibration. Use paired state intervention rollouts to identify causation rather than inferring it from confidence traces alone.
- Report official PR/SR, per-sequence deltas, long-duration strata, risk/coverage, target absence behavior, time to reacquire, duration of false lock, memory contamination, and wall-clock cost. Include full trajectories and multiple seeds. Do not promote all-abstain behavior as solved tracking; require both useful correction and bounded measured failure exposure.

## 7. Reproduction notes

All report totals above can be obtained with Python's standard `json`, `collections`, `math`, and `statistics` modules. No GPU is required. Frame/word counts use each record in `_probe/E0026/words/report.json`, `same_crop` candidates, and the exact qualification condition implemented at exporter lines 269–270. Cup trace windows use the saved `iou` field and `correction.written_tokens > 0`; longest failure spans use consecutive saved frame indices. Source and checkpoint paths come from terminal reports, not assumptions about current processes.

No source code or existing experiment artifact was modified by this audit. This Markdown file is the sole audit output.
