# CodeTrack — GOLA-based working repository

Working instance of **[GOLA](https://github.com/MelanTech/GOLA)** (AAAI 2026, Group Orthogonal
Low-Rank Adaptation for RGB-T Tracking), used as the code base for the CodeTrack research line.

Design and experiment plans live in `docs/`:
`docs/CodeTrack-方案文档.md` (method/architecture) and `docs/CodeTrack-实验文档.md` (protocols, ablations, go/no-go).

---

## Layout

```text
.
├── main.py, profile_model.py, evaluation.py   # upstream entry points (from GOLA)
├── trackit/                                   # upstream framework (from GOLA) — the working copy
├── config/                                    # upstream config tree; our mixins live in config/GOLA/_mixin/
├── consts.yaml                                # dataset paths (local, machine-specific)
├── requirements.txt                           # upstream deps
│
├── scripts/                                   # OUR tooling (run these)
│   ├── 00_env.sh                              # env fixes: LD_LIBRARY_PATH, CUBLAS_WORKSPACE_CONFIG
│   ├── preflight.sh                           # verify env/weights/datasets before a long run
│   ├── train_smoke.sh                         # short training run (verifies the train loop)
│   ├── single_seq_eval.sh                     # official eval pipeline on ONE sequence
│   └── reorg.sh                               # idempotent repo scaffolding
│
├── third_party/GOLA/                          # READ-ONLY upstream reference @ 339c737
├── weights/                                   # checkpoints (gola_b224.bin = GOLA-B, the paper's model)
├── data/                                      # local data views (single-sequence eval lists)
├── docs/
│   ├── CodeTrack-方案文档.md                    # method / architecture
│   ├── CodeTrack-实验文档.md                    # protocols, ablations, go/no-go
│   ├── setup.md                               # env fixes + GOLA code verification notes
│   └── paper/                                 # 12 reference papers (PDF) + manifest
├── repo/                                      # 13 reference code repositories + paper↔repo manifest
├── refs/                                      # (freed; papers/repos now under docs/paper and repo/)
└── outputs/                                   # run artefacts
```

Reference material:

- **`docs/paper/`** — the papers CodeTrack borrows from, named by priority (S/A/B). See
  `docs/paper/README.md` for what each one contributes. Two S-tier papers (MoKA-HP, the TPAMI
  template-updating extension) are paywalled with no open version and are documented as missing.
- **`repo/`** — official implementations of those papers. See `repo/README.md` for the paper↔repo
  mapping and licence notes. `SCDT` is not open-sourced.

### Why the working copy sits at the root

LoRAT and DTPTrack (the projects GOLA builds on) both fork `trackit/` to the repository root and keep
`config/`, `consts.yaml`, `main.py` beside it. We follow the same shape so that upstream diffs stay
readable and upstream instructions apply unchanged.

`third_party/GOLA/` is a **pristine clone** of the upstream repository. Never edit it — it exists so
you can diff our working copy against the exact upstream commit:

```bash
diff -rq third_party/GOLA/trackit trackit          # what we changed
git -C third_party/GOLA rev-parse HEAD             # upstream commit: 339c737
```

Any local modification that upstream does not have is either a documented environment fix or a
CodeTrack addition; both are listed in `docs/setup.md`.

---

## Quick start

```bash
source scripts/00_env.sh      # required: PyTurboJPEG + deterministic-CuBLAS fixes
bash scripts/preflight.sh     # checks torch/cuda, turbojpeg, weights, backbone cache, datasets

# 1. training loop smoke run (tiny: batch 4, 8 samples, 1 epoch)
bash scripts/train_smoke.sh

# 2. inference smoke run (official eval pipeline, one LasHeR sequence)
bash scripts/single_seq_eval.sh
```

Model profiling (forward-pass sanity + FLOPs, no dataset needed):

```bash
source scripts/00_env.sh && "$PYTHON" profile_model.py GOLA dinov2 --device cuda
```

---

## Environment notes (this box)

| Item | Value |
|---|---|
| Python env | `/root/autodl-tmp/lab/envs/gola/bin/python` (3.10.8, torch 2.5.1+cu124) |
| GPU | RTX 4090 D 24 GB |
| Datasets | `consts.yaml` -> `/root/autodl-tmp/lab/dataset/{LasHeR,RGBT234}` |
| libturbojpeg | `/root/autodl-tmp/lab/tools/libturbojpeg.so.0` (needs `LD_LIBRARY_PATH`) |
| Backbone | DINOv2 ViT-B/14, cached at `~/.cache/torch/hub/checkpoints/` |
| Checkpoint | `weights/gola_b224.bin` — GOLA-B, the model used for the paper's LasHeR results |

`scripts/00_env.sh` encodes the two non-obvious fixes (turbojpeg discovery, `CUBLAS_WORKSPACE_CONFIG`
because the framework enables deterministic algorithms). See `docs/setup.md`.

---

## CodeTrack branch (this repository's own work)

An error-correcting spatio-temporal representation recovery network on top of frozen GOLA.
Implemented in `codetrack/`; the only changes to upstream `trackit/` are four additive files
(see `docs/implementation.md` §3 for the exact diff).

**Data flow** — `F_L (B,768,768)` from the GOLA trunk is split into template/search tokens, then:

| Block | Module | File | In -> Out |
|---|---|---|---|
| 3 | parity-check diagnosis | `codetrack/ecc.py` | `X_t,X_aux (B,256,768) -> s (B,64), q (B,256)` |
| 4 | Kalman motion prior | `codetrack/motion.py` | `box -> M_t (B,1,16,16), u_t (B,)` |
| 4b | temporal memory (TRC/TGS) | `codetrack/motion.py` | `-> prior_tokens (B,8,128)` |
| 5a | H-routed sparse recovery | `codetrack/recovery.py` | `TopK(q)=32 -> X_rec (B,256,768)` |
| 5b | noise-modulated refinement | `codetrack/recovery.py` | `2 steps -> X_final` |
| 6 | template protection | `codetrack/template.py` | `-> c_t (B,)` |
| 7 | original GOLA anchor-free head | (upstream) | `-> score_map (B,16,16), boxes (B,16,16,4)` |

**Entry points**

```bash
source scripts/00_env.sh                     # required: turbojpeg + CuBLAS determinism
bash scripts/preflight.sh                    # env / weights / dataset self-check
"$PYTHON" tools/codetrack_verify.py          # identity @ step 0, checkpoint, shape audit, gradients
"$PYTHON" tools/dataflow_audit.py            # instrumented real training step: who talks to whom
"$PYTHON" tools/complementarity_check.py     # 16 assertions that the modules cannot collapse into each other
bash scripts/codetrack_train_smoke.sh        # single-sequence joint training
bash scripts/codetrack_eval_single.sh        # single-sequence inference (official eval pipeline)
```

**Verified state** (details and numbers in `docs/implementation.md`)

- initialisation is a near-identity: `max|dscore_map| = 7.2e-5` vs the loaded GOLA checkpoint
- checkpoint: 1311/1311 keys matched, 0 unexpected
- all 10 new modules receive a non-zero gradient from the real criterion
- single-sequence joint training: loss 23.6 -> 22.2, no NaN/Inf, GOLA adapters co-update

**Known open items** (full list in `docs/dataflow_audit.md` §5 and `docs/hyperparameters_reference.md` §13)

1. Training uses upstream pair sampling, so the temporal modules never see a real frame
   sequence -- the single most important gap for making Kalman/memory learn. A causal-clip
   sampler is needed.
2. `Loss/rec` is measured at exactly 0, i.e. recovery quality is currently unsupervised.
3. Evaluation panics on `torch.isfinite(score)` for untrained weights, because feeding the
   model's own prediction back as a Kalman observation self-amplifies when the head is random.


---

## Upstream

- GOLA: <https://github.com/MelanTech/GOLA> (commit `339c737`)
- Built on LoRAT's `trackit` framework: <https://github.com/LitingLin/LoRAT>
- MMLoRAT (same authors, RGB-X extensions): <https://github.com/MelanTech/MMLoRAT>
