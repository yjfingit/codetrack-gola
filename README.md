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
"$PYTHON" tools/preflight_acceptance.py      # 86 structural checks against the REAL stage configs
"$PYTHON" tools/codetrack_verify.py          # identity @ step 0, checkpoint, shape audit, gradients
"$PYTHON" tools/dataflow_audit.py            # instrumented real training step: who talks to whom
"$PYTHON" tools/complementarity_check.py     # 16 assertions that the modules cannot collapse into each other
"$PYTHON" tools/causality_check.py           # the current frame's GT must not reach the current output
"$PYTHON" tools/lora_grad_check.py           # LoRA actually trains, backbone does not drift
"$PYTHON" tools/recovery_report.py            # real -8 vs -5 training: gain_total / AUROC
bash scripts/codetrack_train_smoke.sh        # single-sequence joint training
bash scripts/codetrack_eval_single.sh        # single-sequence inference (official eval pipeline)
```

**Training stages**

Each stage is defined by an **optimizer-update budget** (`stage.max_updates`), not by epochs, and
by an explicit `optimizer.parameter_scope`. S1 trains CodeTrack only and leaves GOLA frozen;
S2 unfreezes the LoRA adapters, the head and the token-type embedding.

```bash
# S0 -- 512 updates, CodeTrack-only. Must pass before any long run.
"$PYTHON" main.py GOLA codetrack_preflight --distributed_nproc_per_node 1 --disable_wandb \
  --weight_path "$WEIGHT" --output_dir="$PWD/outputs/s0"

# S1 -- 1500 updates, parameter_scope = ["codetrack"]  (GOLA frozen)
"$PYTHON" main.py GOLA codetrack_s1 --distributed_nproc_per_node 1 --disable_wandb \
  --weight_path "$WEIGHT" --output_dir="$PWD/outputs/s1"

# S2 -- 8000 updates, joint PEFT (framework default scope)
"$PYTHON" main.py GOLA codetrack_s2 --distributed_nproc_per_node 1 --disable_wandb \
  --weight_path "$WEIGHT" --output_dir="$PWD/outputs/s2"

# residual-gate ablation, 600 updates per arm, only variable = residual_gate_init
"$PYTHON" main.py GOLA codetrack_s1 --mixin_config codetrack_gate_neg8 --disable_wandb \
  --distributed_nproc_per_node 1 --weight_path "$WEIGHT" --output_dir="$PWD/outputs/abl_neg8"
"$PYTHON" main.py GOLA codetrack_s1 --mixin_config codetrack_gate_neg5 --disable_wandb \
  --distributed_nproc_per_node 1 --weight_path "$WEIGHT" --output_dir="$PWD/outputs/abl_neg5"

# compare the two arms (prints gain_total / q_auroc_mask / clean drift and a machine-readable GO)
"$PYTHON" tools/recovery_report.py --gates -8,-5 --updates 600
```

S3/S4 use `codetrack_full` (temporal branch on).

**Verified state** (details and numbers in `docs/implementation.md`)

- initialisation is a near-identity: `max|dscore_map| = 1.15e-4` vs the loaded GOLA checkpoint
- checkpoint: 1311/1311 keys matched, 0 unexpected
- all 10 new modules receive a non-zero gradient from the real criterion
- optimizer owns **1406/1406** trainable tensors, with per-scope lr routing and `wd=0` on every
  bias/norm; LoRA 1296/1296 moves, frozen backbone drifts by 0
- **S1 freeze is real**: with `parameter_scope: ["codetrack"]` the optimizer holds 97 CodeTrack
  tensors and zero `blocks.*` / `head.*` / `lora` tensors
- **the noise half of block 5b is alive**: injected noise is 98.2x stronger on tokens the
  diagnosis flags as damaged (it was identically zero before)
- 50 real optimizer steps on a fixed batch move `codetrack.H.H` by 3.6e-3 and the refiner/denoiser
  by ~6e-3, i.e. the recovery branch is trainable even though the residual gate starts at
  `sigmoid(-8) ~ 3.4e-4`

**Known open items** (full list in `docs/implementation.md` §15 and `docs/hyperparameters_reference.md` §13)

1. Training uses upstream pair sampling, so the temporal modules never see a real frame
   sequence -- the single most important gap for making Kalman/memory learn. A causal-clip
   sampler is needed (S3), together with scheduled sampling, TBPTT, D4 (TRC reliability range
   `[0.5, 0.73]`), D5 (per-slot trust labels) and the current-frame GT target-mask leak.
2. Evaluation panics on `torch.isfinite(score)` for untrained weights, because feeding the
   model's own prediction back as a Kalman observation self-amplifies when the head is random.


---

## Upstream

- GOLA: <https://github.com/MelanTech/GOLA> (commit `339c737`)
- Built on LoRAT's `trackit` framework: <https://github.com/LitingLin/LoRAT>
- MMLoRAT (same authors, RGB-X extensions): <https://github.com/MelanTech/MMLoRAT>
