# S1 round-1 four-GPU sweep — results and the blocker it exposed

**Date:** 2026-10-04
**Machine:** 175 (`4090server175`), 4× RTX 4090 24 GB, GPUs 1/2/3/4
**Method config:** `config/GOLA/codetrack_s1/config.yaml`
**Purpose:** find an S1 (`parameter_scope: ["codetrack"]`, GOLA frozen) setting
that makes the ECC recovery branch produce a **positive** gain, then screen the
best 1–2 arms to 2048 updates.

Result: **all four arms failed both hard gates**, and the failure is
structural, not a hyper-parameter issue. `q` — the per-token error
probability that drives `TopK(q) = 32` sparse routing — is flat to 1e-4 across
all 256 search tokens, so the routing selects a near-random subset.

---

## 1. What was run

Four arms, one per GPU, launched by `s1_r1_run.sh` from `plan.txt`.
Common to all four: `global_batch_size: 16`, `grad_accumulation_steps: 8`
(effective batch 128), `max_updates: 2048`, `warmup_updates: 128`,
`t_initial_updates: 2048`.

| trial | GPU | differs from baseline | lr | lambda_gain | lambda_align | lambda_pres | diagnosis_alpha |
|---|---|---|---|---|---|---|---|
| `gpu0_baseline` | 1 | — | 1e-4 | 0.2 | 0.1 | 0.02 | 0.5 |
| `gpu1_lr5e5` | 2 | **lr 1e-4 → 5e-5** | 5e-5 | 0.4 | 0.1 | 0.02 | 0.5 |
| `gpu2_gain04` | 3 | **lambda_gain 0.2 → 0.4** | 1e-4 | 0.4 | 0.1 | 0.02 | 0.5 |
| `gpu3_alpha07` | 4 | **diagnosis_alpha 0.5 → 0.7** | 1e-4 | 0.4 | 0.1 | 0.02 | 0.7 |

Note: relative to the *shipped* `codetrack_s1/config.yaml` (which has
`lambda_gain: 0.2`, `lambda_align: 0.2`, `lambda_pres: 0.01`), every arm also
carried the global loss change `lambda_align 0.2 → 0.1` and
`lambda_pres 0.01 → 0.02`. So `gpu0_baseline` is the baseline *for this round*,
not the shipped config.

Stopped early at ~400 updates (scheduled first screen was 400), because the
first screen already showed both gates failing by a wide margin and the
trajectories were flat.

## 2. Results at ~400 updates (30-step trailing window)

Reported values are the mean over the last 30 printed steps of each arm's
final segment. `—` = this script did not extract that field.

| trial | upd | **gain_total** | **q_error_spearman** | d_input | d_after | **q_auroc_mask** | **q_std** | q_mean | q_pos_mean | q_neg_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| `gpu0_baseline` | 410 | 0.00127 | 0.0549 | 0.1391 | 0.1360 | 0.6595 | 1.0e-04 | 0.1950 | 0.1950 | 0.1950 |
| **`gpu1_lr5e5`** | 414 | **0.00352** | **0.0606** | 0.1916 | 0.1923 | **0.6667** | 1.0e-04 | 0.1949 | 0.1950 | 0.1949 |
| `gpu2_gain04` | 404 | 0.00162 | 0.0387 | 0.1094 | 0.1056 | 0.6574 | 1.0e-04 | 0.1950 | 0.1951 | 0.1950 |
| `gpu3_alpha07` | 401 | 0.00190 | 0.0481 | 0.1134 | 0.1110 | **0.5801** | 1.0e-04 | 0.1951 | 0.1951 | 0.1951 |

Gates used by `tools/stage_validate.py`:

```python
GAIN_TOTAL_MIN  = 5e-3    # d_input - d_final
DIAG_AUROC_SOFT = 0.60    # AUROC(q, corruption_mask)
DIAG_Q_STD_MIN  = 1e-3    # q must not collapse to a constant
```

Reading of the table:

* **`gain_total` fails by 1.4×–3.9×** on every arm (best 0.00352 vs gate 5e-3).
* **`q_error_spearman` fails by 4×–6×** (best 0.0606 vs target 0.25).
* **`q_std` is 1.0e-04 on all four arms** — 10× below the `DIAG_Q_STD_MIN`
  floor. This is the headline finding.
* `q_auroc_mask` passes 0.60 on three of four arms — but see §2.1: this does
  **not** mean `q` carries usable signal.
* **`diagnosis_alpha 0.5 → 0.7` makes AUROC worse** (0.5801 vs 0.6595 on the
  baseline arm) — the only arm to fall below the AUROC gate. Raising the
  weighting on the diagnosis path degraded the diagnosis.

### 2.1 The decisive numbers: `q` is flat *and* correctly ordered

The `q_pos_mean` / `q_neg_mean` columns above are the key. Across all four
arms the two are equal to **within 1e-4**:

| arm | q_pos_mean − q_neg_mean | q_std | q_auroc_mask |
|---|---|---|---|
| `gpu0_baseline` | **0.0000** | 1e-04 | 0.6595 |
| `gpu1_lr5e5` | **0.0001** | 1e-04 | 0.6667 |
| `gpu2_gain04` | **0.0001** | 1e-04 | 0.6574 |
| `gpu3_alpha07` | **0.0000** | 1e-04 | 0.5801 |

The mean `q` over *corrupted* tokens and over *clean* tokens is the same to
4 decimals, yet AUROC reaches 0.66. These two facts are compatible, and the
way they fit together is the whole diagnosis:

* `_auroc_against_mask` (`codetrack/criteria.py:163`) is a pure **rank**
  statistic (Mann-Whitney U via average ranks, ties handled correctly). It
  measures *ordering* and is completely insensitive to *magnitude*.
* `q_pos_mean` / `q_neg_mean` / `q_std` are printed at 4 decimals
  (`float(qd.mean())` rendered as `0.1950`). A true spread of `1e-4` inside a
  mean of `0.1950` disappears at that precision.

So: **`q` orders the 256 tokens better than chance, but all 256 values are
squeezed into a band ~1e-4 wide around 0.1950.** The ordering is real; the
separation is unusable.

Note also that `q_mean ≈ 0.1950` is just `sigmoid(vote_bias)` with
`detection_prior: 0.2` — i.e. the reported `q` is essentially the *prior*.
The syndrome contribution `H_bar^T s` is too small to move it.

That is why `TopK(q, 32)` fails downstream. TopK is a rank operation, so it
*does* pick slightly-better-than-random tokens — but the "signal" it hands
downstream is a 1e-4 difference, and the downstream consumers are not
sensitive to that. This is the precise mechanism behind the project's own
note that `TopK(q)` selects a set 4.1e-4 from the population mean and the
denoiser noise gate spans only 1.01×.

**Consequence for the fix:** the problem is not "the ordering is wrong"
(training that harder, or adding a ranking loss, will not help) and not
"TopK is discrete" (it is, but it is selecting correctly). The problem is
**the amplitude**. Anything that widens the `q` band is on target; anything
that only sharpens the ordering is not.

Conclusions that survive:

1. **`lr = 5e-5` beats `lr = 1e-4`** on both gated metrics and on AUROC.
2. **`lambda_gain 0.2 → 0.4` is ineffective**: gpu2 (0.00162) vs gpu0 (0.00127)
   differ by only 2.7e-4, far inside run-to-run noise. Raising the weight on a
   loss term whose target is unreachable does not move the metric.
3. **`diagnosis_alpha 0.5 → 0.7` actively hurts** the diagnosis AUROC
   (0.5801 vs 0.6595), though the gated metrics stay mid-pack.
4. Rankings inside 400 updates are not stable — an earlier read at ~370
   updates had gpu1 last on both metrics and it finished first. Do not draw
   conclusions before the scheduled screen.

## 3. The structural blocker: `q` is flat

`q_std ≈ 1e-4` means `TopK(q, 32)` picks ~32 of 256 tokens with essentially no
discrimination. Everything downstream (H-routed sparse recovery, the denoiser
noise gate, `frame_reliability`) consumes `q`, so a flat `q` makes the whole
recovery branch a no-op — which is exactly what `d_after ≈ d_before ≈ d_input`
in §2 shows.

### 3.1 Where the variance is lost

The path, from `codetrack/ecc.py`:

```
X_t   (B, N, C)  ->  U = W_x(X_t)                       (B, N, mid)
X_aux (B, N, C)  ->  R = W_r(X_aux)                     (B, N, mid)
        C_obs = einsum("mn,bnd->bmd", H_bar, U)          (B, M, mid)   M=64
        C_ref = einsum("mn,bnd->bmd", H_bar, R)          (B, M, mid)
        delta = C_obs - C_ref
      s_raw  = mlp_s([delta, |delta|]) + cos_proj(cos) + residual_scale*||delta||
      s_raw  = (s_raw - offset) * gain                   # calibration
        s    = sigmoid(s_raw)                            (B, 64)
 q_logits    = einsum("mn,bm->bn", H_bar, s) + vote_bias (B, 256)
        q    = sigmoid(q_logits)                         (B, 256)
```

Three compressive stages stacked. The provenance of each number is given, so
nothing here has to be taken on trust:

| # | quantity | std | provenance |
|---|---|---|---|
| 1 | `s_raw` before calibration | **0.2066** | measured, this run's log (`raw logit std` line) |
| 2 | `s_raw` after calibration | **≈ 1.0** by construction | `calibrate_syndrome_gain(target_std=1.0)`; measured auto gain **4.840315** = 1.0 / 0.2066, so it fired |
| 3 | `s` = sigmoid(`s_raw`) | **≈ 0.0097** over 64 checks | **measured**, quoted in the source comment at `codetrack/ecc.py` ~L223 (`0.794 ± 0.0097`) |
| 4 | `q_logits` | **≈ 0.002** over 256 tokens | **measured**, same source comment ("the transmitted term has a spread of only 2e-3 over 256 tokens") |
| 5 | `q` | **≈ 3e-04** | **measured**, same source comment ("`q` is constant to 3e-4"); this run logs `1.0e-04` |

Note stage 3: the source's measured `0.0097` is far *smaller* than the
`0.25 × std(s_raw)` bound you would get from the sigmoid slope alone. That is
itself informative — it means the 64 checks were already highly correlated
before the calibration, i.e. the syndrome carries ~0.01 of spread, not ~0.22.
The calibration multiplies `s_raw` by 4.84, and since the check-to-check
correlation is preserved by a rank-1 rescale, the *useful* spread stays small:
rescaling a vector does not decorrelate its components.

**Stage 4 is the single largest loss: 0.0097 → 0.002, ≈ 5×.**
`H_bar` is a hard binary matrix with `h_links_per_check: 12`,
`h_min_col_degree: 3`, so the 64 → 256 map sums only ~3 syndrome entries per
token. Those 3 entries are near-equal (they are all functions of the same
`U`/`R`, and the 64 checks are correlated), so the sum is close to
`3 × mean(s)` — a near-constant. A linear map cannot create variance that is
not there, and with only ~3 co-varying inputs per output it cannot even
preserve the little there is.

Stage 5 halves it again (`0.002 → ~1e-3..1e-4`): `q_logits` sits near
`logit(0.2) = -1.386`, where `sigmoid'(-1.386) = 0.2 × 0.8 = 0.16`, so
`sigmoid` scales the spread by ~0.16. That is a **~6× loss**, comparable to
stage 4.

So the two dominant losses are **stage 4 (the `H_bar^T` map, ~5×)** and
**stage 5 (the second `sigmoid`, ~6×) — roughly equal.** Both must be
addressed; fixing only one buys ~5–6×, which turns `1e-4` into `~6e-4`, still
below the `1e-3` floor.

The source itself documents the same defect, measured on a real checkpoint
(`codetrack/ecc.py`, comment above `syndrome_logit_gain`, lines ~223–238):

> With the stock init, `s_raw` lands at ~0.2, so `s = sigmoid(s_raw)` sits at
> `0.794 ± 0.0097` over 64 checks. The damage propagates through
> `q_logits = H^T s + vote_bias`: the transmitted term has a spread of only
> 2e-3 over 256 tokens, so `q` is constant to 3e-4 and, measured on the same
> checkpoint,
> * `TopK(q)` selects a set 4.1e-4 away from the population mean -> near-random;
> * the denoiser noise gate spans max/min = 1.01× over 256 tokens (not selective);
> * `frame_reliability` is one value per batch (std 3.3e-5).
>
> A point-mass head cannot be trained out of the point mass by the ordinary
> loss, because every token sees the same gradient. The spread has to exist at
> initialisation.

So the diagnosis head is *aware* of the problem and added a learnable
`syndrome_logit_gain` (`nn.Parameter(torch.full((1,), syndrome_logit_gain))`)
plus `syndrome_logit_offset`, with `syndrome_gain_calibration: true` in
`config/GOLA/dinov2/codetrack_spatial.yaml`. The calibration does fire
(measured: raw logit std 0.206598 → auto gain 4.840315), but it only fixes
stage 2. Stages 3–5 are untouched, and the second `sigmoid` at stage 5 is
never addressed.

### 3.2 Why the calibration clamp is also suspect

`calibrate_syndrome_gain` (`codetrack/ecc.py` ~line 296):

```python
gain = float(min(max(target_std / cur, 1.0), 1e3))
```

The `max(..., 1.0)` lower clamp means **the gain can never shrink**, only
grow. If a checkpoint ever produces `std(s_raw) > target_std`, the calibration
silently returns 1.0 and leaves an over-wide syndrome alone. This is probably
not what killed this run (std was 0.2 < 1.0, so gain grew to 4.84), but it is
an asymmetry worth flagging.

### 3.3 Note on the `d_*` metrics being normalisation-sensitive

`_relative_error` (`codetrack/criteria.py` ~line 290) is an angular distance:

```python
return (1.0 - F.cosine_similarity(x, clean, dim=-1, eps=1e-6)).clamp(0.0, 2.0)
```

`d_input`, `d_before` and `d_after` are each measured on tensors that pass
through **different** LayerNorms. A pure rescale of a tensor leaves its
cosine distance unchanged, so a *normalisation-layer scale change* can make
`d_before == d_input` to 5 decimals without any real token rotation happening.
The project handoff (`_handoff/HANDOFF.md` §4.1) flags this as the leading
alternative explanation and asks for a **norm-only baseline** — measure the
distance after applying the same normalisation path with no refinement, and
report it alongside. **That baseline has not been implemented.** Until it is,
`gain_total` cannot be fully trusted, although the *flat `q`* finding above is
independent of it (`q` is a probability, not a cosine distance).

## 4. What was tried already and did not help

| change | source | outcome |
|---|---|---|
| `syndrome_gain_calibration: true` + learnable `syndrome_logit_gain`/`_offset` | `codetrack/ecc.py` | fires correctly (gain 4.84), does not fix stages 3–5 |
| `residual_gate_init: 0.0` | `config/GOLA/dinov2/codetrack_spatial.yaml` | removes an initialisation bias, not the flat-`q` cause |
| refiner/denoiser `up` init `N(0, 0.02)` | same | "fix dead denoiser noise" commit; gate still spans 1.01× |
| `lambda_gain 0.2 → 0.4` | this sweep | no measurable effect (2.7e-4 inside noise) |
| `diagnosis_alpha 0.5 → 0.7` | this sweep | no effect |
| `lr 1e-4 → 5e-5` | this sweep | **best arm**, but still 1.4× short of the gain gate |

## 5. Candidate directions (NOT yet implemented, no decision made)

The §3 numbers say the amplitude is lost in **two roughly equal steps**
(stage 4 `H_bar^T` ~5×, stage 5 second `sigmoid` ~6×), so a fix that only
addresses one of them lands at `~6e-4` — still under the `1e-3` floor. Plan
for both.

| # | idea | layer | why it could work | risk |
|---|---|---|---|---|
| A | **Drop the second `sigmoid`** — export `q_logits` as the routing score instead of `q = sigmoid(q_logits)`; return both so callers can migrate. `l_diag` already uses `binary_cross_entropy_with_logits(q_logits, ...)`, so the *loss* is already logit-space; only the exported `q` is squashed. | 5 | removes a ~6× compression at a point where the logit is far from 0 (`logit(0.2)`, slope 0.16) | downstream assumes `q ∈ [0,1]` and uses it as a probability (noise gate, `frame_reliability`) |
| B | **Widen the `H_bar^T` map** — either a learnable continuous `H_soft` initialised at the binary graph, or a learned `Linear(64 → 256)` residual branch added to `q_logits`. | 4 | attacks the other ~5× loss directly; a learned map can decorrelate the syndrome entries | weakens the "this is an LDPC parity check" story; `H_bar` also feeds `C_obs`/`C_ref`, so a change there is not local — a *residual* branch is the safer form |
| C | **Decorrelation regulariser on the 64 checks.** Stage 3's measured `0.0097` instead of `~0.22` says the checks are strongly correlated before any sigmoid. Penalise `offdiag(corr(s))`, or make `check_scale` (already `nn.Parameter(torch.ones(num_checks))`) actually differentiate checks — currently a rank-1 `(x-offset)*gain` rescale is applied *after* it, which cannot create per-check spread. | 3 | if the 64 checks are the reason `H_bar^T s` is flat, this is the root fix, and it also improves stage 4 | needs a measurement of the 64×64 correlation matrix first; the calibration step may fight it |
| D | **Soft / straight-through TopK + a ranking or coverage loss outside TopK**, keeping the 32-token hard budget at eval. | routing | the project's own defect report lists this as P2 | **does not address the measured problem.** §2.1 shows the ordering is already right (AUROC 0.66) and TopK selects correctly; only the amplitude is too small. A ranking loss sharpens the ordering, which is not the bottleneck |

Ordering by the evidence: **A first (cheapest, removes ~6×), then B as a
residual branch (~5×), then C only if the correlation measurement justifies
it.** D is listed because it is the project's own recorded suggestion, but the
§2.1 measurement argues against prioritising it.

**A measurement is missing before committing to any of these:** the direct
`std(s_raw)`, `std(s)`, `std(q_logits)`, `std(q)`, the 64×64 correlation of
`s`, and the column-degree statistics of `H_bar`, all captured on one real
batch. Stages 3–5 in §3.1 are quoted from the source's own comment, not
re-measured on this checkpoint, and stage 3's small value is what motivates
direction C. Running that probe is read-only and takes seconds; it should come
before any code edit.

The project's own defect report
(`训练过程问题分析报告-2026-10-04.md`) independently flags hard TopK as P2 and
recommends direction D, while the handoff §4.2 asks for a decision probe that
measures the `d_*` quantities **plus a norm-only baseline** before any code
change. Neither of those documents contains the `q_pos_mean == q_neg_mean`
observation from §2.1, which is what moves the priority from D to A/B.

## 6. How to reproduce

```bash
cd /home/yangjuanfeng/lab/projects/gola-CodeTrack

export LD_LIBRARY_PATH=/home/yangjuanfeng/lab/tools/libjpeg-turbo/root/usr/lib/x86_64-linux-gnu
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NCCL_P2P_DISABLE=1
PY=/home/yangjuanfeng/lab/envs/gola/bin/python

# regenerate the four mixins and launch one arm per GPU (see plan.txt)
bash _probe/grid/s1_r1/s1_r1_run.sh

# watch progress
bash _probe/grid/s1_r1/s1_r1_monitor.sh

# archive + rank + gate-check the finished run
$PY _probe/grid/s1_r1/summarize.py
```

`grid_make_mixin2.py` exists because the original `grid_make_mixin.py` emits
`5e-05` as a bare YAML token, which PyYAML parses as a **string**, so
`optimizer.lr` arrives as `'5e-05'` and AdamW dies with
`TypeError: '<=' not supported between instances of 'float' and 'str'`.
The `2` version coerces to a real float and self-checks the round trip.

---

## Files in this directory

| file | role |
|---|---|
| `plan.txt` | the four arms and their overrides, one per line, `name|key=value,...` |
| `s1_r1_run.sh` | generates each arm's mixin, launches one process per GPU, cleans up |
| `s1_r1_monitor.sh` | prints per-arm micro / update / gain / spearman / q_std / grad_norm |
| `summarize.py` | reads the finished logs, averages a trailing window, ranks, gate-checks |
| `grid_make_mixin2.py` | mixin generator with the YAML scalar-type fix |
| `mixins/*.yaml` | the exact mixin each arm ran with (kept for the record) |
| `logs/` | raw stdout, 4 MB per arm (git-ignored) |
| `out/` | per-arm spool directory (git-ignored) |
