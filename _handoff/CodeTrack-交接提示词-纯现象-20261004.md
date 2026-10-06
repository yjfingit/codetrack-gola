# CodeTrack 训练问题交接 —— 现象描述

本文档**只描述现象和可验证的事实**，不含根因推断、不含修改方案。请自行读代码判断。

---

## 一、项目是什么

**CodeTrack**：在 RGB-T 单目标跟踪器 **GOLA**（AAAI 2026，DINOv2 ViT-B/14 backbone + group-orthogonal LoRA）之上，加一条**纠错码（LDPC 风格）启发的恢复支路**。

恢复支路的意图：给 search token 注入损坏 → 用稀疏校验矩阵诊断出哪些 token 坏了 → 对坏 token 做稀疏恢复 → 恢复后的表示应该**更接近干净参考**。

仓库：
```
https://github.com/yjfingit/codetrack-gola
commit 964cf328b58486f75df92bd7138fa9c7f4bf42b9   (branch: main)
```

关键源码文件：
```
codetrack/ecc.py            # 校验矩阵 H_bar → syndrome s → 逐 token 错误概率 q
codetrack/criteria.py       # 损失与指标（注意 _auroc_against_mask 在 163 行，_relative_error 在 290 行）
trackit/criteria/builder.py # 构造 CodeTrackCriteria（注意 gain_margin 未转发，见 §五）
config/GOLA/codetrack_s1/config.yaml
```

---

## 二、跑了什么

**2026-10-04，175 机（4× RTX 4090 24GB），四个 GPU 各跑一路 S1 筛选**，共约 400 个优化器更新后手动停止。

四路配置（都从同一个 GOLA 权重出发，`parameter_scope: ["codetrack"]`，GOLA 冻结）：

| 路 | 与其他三路的差异 |
|---|---|
| `gpu0_baseline` | — （基准）|
| `gpu1_lr5e5` | `lr: 1e-4 → 5e-5` |
| `gpu2_gain04` | `lambda_gain: 0.2 → 0.4` |
| `gpu3_alpha07` | `diagnosis_alpha: 0.5 → 0.7` |

**四路共有**（相对仓库里 shipped 的 `codetrack_s1/config.yaml`）：`lambda_align: 0.2 → 0.1`、`lambda_pres: 0.01 → 0.02`、`global_batch_size: 16`、`grad_accumulation_steps: 8`（有效 batch 128）、`max_updates: 2048`、`warmup_updates: 128`。

启动脚本、配置、复现命令：仓库 `_probe/grid/s1_r1/`（`plan.txt`、`s1_r1_run.sh`、`mixins/*.yaml`、`README.md`）。

---

## 三、判据门槛

项目自己的 `tools/stage_validate.py` 写死了这几个数字：

```python
GAIN_TOTAL_MIN  = 5e-3     # d_input - d_after，恢复增益下限
DIAG_AUROC_SOFT = 0.60     # AUROC(q, corruption_mask) 下限
DIAG_Q_STD_MIN  = 1e-3     # q 不能塌成常数
```

另有项目文档里提到的期望值：`Error/q_error_spearman > 0.25`（越高越好，这是"预测的 syndrome 严重度"与"真实误差"的秩相关）。

---

## 四、现象（原始数字）

### 4.1 四路末尾 30 步的均值

| 路 | `gain_total` | `q_error_spearman` | `q_auroc_mask` | **`q_std`** |
|---|---|---|---|---|
| `gpu0_baseline` | 0.00127 | 0.0549 | 0.6595 | **1.0e-04** |
| `gpu1_lr5e5` | **0.00352** | **0.0606** | **0.6667** | **1.0e-04** |
| `gpu2_gain04` | 0.00162 | 0.0387 | 0.6574 | **1.0e-04** |
| `gpu3_alpha07` | 0.00190 | 0.0481 | **0.5801** | **1.0e-04** |

- 四路的 `gain_total` 全部低于门槛 `5e-3`（差 1.4× 到 3.9×）。
- 四路的 `q_error_spearman` 全部远低于期望的 0.25（差 4× 到 6×）。
- 四路的 `q_std` 全部等于 `1.0e-04`，低于门槛 `1e-3`（差 10×）。

### 4.2 一组看起来互相矛盾的字段

从 `gpu0_baseline` 日志取原始行（`Epoch: [0] [3280/8192]` 那一步，括号内是滑动平均）：

```
Error/q_auroc_mask:   0.6595 (0.6361)
Error/q_auprc_mask:   0.0545 (0.1119)
Error/q_std:          0.0001 (0.0003)
Error/q_pos_mean:     0.1950 (0.2008)
Error/q_neg_mean:     0.1950 (0.2007)
Error/q_mean:         0.1950 (0.2007)
Error/q_auroc_target: 0.7130 (0.7818)
```

**注意 `q_pos_mean` 与 `q_neg_mean` 相等（都是 0.1950），而 `q_auroc_mask` 却是 0.6595。**

四路全部呈现这个模式：

| 路 | `q_pos_mean` | `q_neg_mean` | 两者之差 | `q_auroc_mask` |
|---|---|---|---|---|
| `gpu0_baseline` | 0.1950 | 0.1950 | 0.0000 | 0.6595 |
| `gpu1_lr5e5` | 0.1950 | 0.1949 | 0.0001 | 0.6667 |
| `gpu2_gain04` | 0.1951 | 0.1950 | 0.0001 | 0.6574 |
| `gpu3_alpha07` | 0.1951 | 0.1951 | 0.0000 | 0.5801 |

相关字段说明（`codetrack/criteria.py` 里对每个量都有注释）：
- `q_pos_mean` = `q` 在**损坏 token** 上的均值；`q_neg_mean` = `q` 在**干净 token** 上的均值。
- `q_auroc_mask` = 用 `q` 对**精确的 token 损坏标签**做排序的 AUROC。
- `q_auroc_target` = 用 `q` 对**软目标** `err` 做排序的 AUROC（阈值相关，注释里说是次要指标）。
- `q_mean` = `q` 全体均值。

### 4.3 全部 `Error/` 字段的原始行（同一步，完整）

```
Error/q_auroc_mask: 0.6595 (0.6361)   Error/q_auprc_mask: 0.0545 (0.1119)
Error/q_std: 0.0001 (0.0003)          Error/q_pos_mean: 0.1950 (0.2008)
Error/q_neg_mean: 0.1950 (0.2007)     Error/q_auroc_target: 0.7130 (0.7818)
Error/q_error_spearman: 0.0697 (0.1073)  Error/q_mean: 0.1950 (0.2007)
Error/d_input: 0.1305 (0.1737)        Error/d_before: 0.1301 (0.1735)
Error/d_after: 0.1241 (0.1726)        Error/gain_total: 0.0012 (0.0010)
Error/gain_refiner: 0.0003 (0.0002)   Error/gain_denoiser: 0.0009 (0.0009)
Error/corrupted_fraction: 0.0249 (0.0374)
Error/d_input_suspect: 0.1385 (0.2063)
Error/d_before_suspect: 0.1376 (0.2052)
Error/d_after_suspect: 0.1456 (0.2019)
Error/gain_suspect: 0.0035 (0.0044)
```

同一行的相关 `Loss/` 与全局字段：

```
loss: 4.5995 (6.2491)   grad_norm: 0.0459   loss_scale: 65536.00
Loss/diag: 0.2222 (0.2507)      Loss/rec: 0.0071 (0.0249)
Loss/gain: 0.0468 (0.0683)      Loss/gain_refiner: 0.0257 (0.0346)
Loss/gain_final: 0.0226 (0.0337)  Loss/align: 0.3150 (0.9618)
Loss/pres: 0.0192 (0.0413)      Loss/track_corr: 0.8885 (0.9145)
Loss/cls: 0.7650 (0.7937)       Loss/box: 0.1200 (0.1209)
Loss/cls_clean: 0.7358 (0.7596) Loss/box_clean: 0.1096 (0.1133)
orth: 0.0572 (0.0571)
```

### 4.4 其他可观察事实

- **`Loss/diag` 在下降**：训练早期约 0.45 降到末段约 0.22（滑动平均 0.25）。
- **`q_auroc_mask` 在上升**：早期约 0.31 升到末段 0.58~0.72（滑动平均约 0.64）。
- **`q_std` 全程不动**：始终 `0.0001`（滑动平均 `0.0003`）。
- **`q_mean` 全程不动**：始终 `0.1950` 左右。
- **`Error/corrupted_fraction: 0.0249`**，即每 batch 只有约 2.5% 的 token 被注入损坏。
- **`d_input` 在训练中变化很大**（0.069 → 0.156），但同一时刻的 `d_before`、`d_after` 都极其接近 `d_input`（差值 ≤1e-2）。
- **`gain_suspect`（0.0035）比 `gain_total`（0.0012）大约 3 倍**。
- `loss`、`grad_norm`、`loss_scale` 全程有限，无 NaN、无 inf。

### 4.5 两条相关的源码注释（原文，是之前测出来的，不是我这次的测量）

`codetrack/ecc.py` 在 `syndrome_logit_gain` 上方有一段注释，记录了在**另一个** checkpoint 上的测量：

> With the stock init, `s_raw` lands at ~0.2, so `s = sigmoid(s_raw)` sits at `0.794 ± 0.0097` over 64 checks. The damage propagates through `q_logits = H^T s + vote_bias`: the transmitted term has a spread of only 2e-3 over 256 tokens, so `q` is constant to 3e-4 and, measured on the same checkpoint,
> * `TopK(q)` selects a set 4.1e-4 away from the population mean -> near-random;
> * the denoiser noise gate spans max/min = 1.01× over 256 tokens (not selective);
> * `frame_reliability` is one value per batch (std 3.3e-5).
>
> A point-mass head cannot be trained out of the point mass by the ordinary loss, because every token sees the same gradient. The spread has to exist at initialisation.

另有一个文件 `_probe/grid/s1_r1/README.md`（在仓库里）记录了我对这轮数据的整理。

---

## 五、已知的代码事实（可自行核实，非推断）

1. **`trackit/criteria/builder.py` 的 `codetrack` 分支不转发 `gain_margin`**（`lambda_trc` 同样未转发）。
   `CodeTrackCriteria.__init__` 有 `gain_margin: float = 0.8` 参数，但构造时没从 config 读。
   同时 `config/GOLA/codetrack_s{1,2,3,4}/config.yaml` 的 `criteria:` 段**没有** `gain_margin` 这一行。
   ⇒ config 和 mixin 都无法设置它。

2. **`calibrate_syndrome_gain` 的增益有单侧下限**（`codetrack/ecc.py`）：
   ```python
   gain = float(min(max(target_std / cur, 1.0), 1e3))
   ```
   `max(..., 1.0)` 意味着增益只增不减。

3. **`_relative_error` 是余弦距离**（`codetrack/criteria.py`），尺度不变：
   ```python
   return (1.0 - F.cosine_similarity(x, clean, dim=-1, eps=1e-6)).clamp(0.0, 2.0)
   ```
   而 `d_input` / `d_before` / `d_after` 三个量各自经过**不同**的 LayerNorm。

4. **`H_bar` 是稀疏二值矩阵**，由 config 决定：`num_checks: 64`、`h_links_per_check: 12`、`h_min_col_degree: 3`（见 `config/GOLA/dinov2/codetrack_spatial.yaml`）。

5. **`l_diag` 用的是 logit**（`codetrack/criteria.py` ~330 行）：
   ```python
   l_diag = F.binary_cross_entropy_with_logits(q_logits.float(), err.float())
   ```
   而上报的 `q` 是 `sigmoid(q_logits)`，`q_logits = H_bar^T s + vote_bias`。

6. **`q_mean ≈ 0.1950` 与 config 里的 `detection_prior: 0.2` 数值接近**，而 `vote_bias` 初始化为 `logit(detection_prior)`。

---

## 六、复现方式

```bash
cd /home/yangjuanfeng/lab/projects/gola-CodeTrack

export LD_LIBRARY_PATH=/home/yangjuanfeng/lab/tools/libjpeg-turbo/root/usr/lib/x86_64-linux-gnu
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NCCL_P2P_DISABLE=1
PY=/home/yangjuanfeng/lab/envs/gola/bin/python

# 一键跑四路（每卡一路）
bash _probe/grid/s1_r1/s1_r1_run.sh

# 看进度
bash _probe/grid/s1_r1/s1_r1_monitor.sh

# 汇总排名 + 门槛判定
$PY _probe/grid/s1_r1/summarize.py
```

单路手工启动（便于改参数）：
```bash
$PY -u main.py GOLA codetrack_s1 \
    --distributed_nproc_per_node 1 \
    --mixin_config _grid_gpu0_baseline \
    --disable_wandb \
    --weight_path weights/gola_b224.bin \
    --output_dir <out>
```
mixin 文件须放在 `config/GOLA/_mixin/`，`--mixin_config` 的值**不带 `.yaml`**。
mixin 只能替换**已存在**的 key，新 key 会被静默跳过。

---

## 七、请遵守的项目约定

1. **不要改训练策略参数**（lr、epoch、更新步数、损失权重、阈值）—— 这些由用户本人决定。只提**架构/实现**层面的改动。
2. **不要直接改仓库代码**。提出方案 + 可执行步骤，由用户自己执行。`repo/` 目录是只读的上游克隆。
3. 一切结论要能回溯到**原文或源码行号**。区分「论文写什么」与「代码做什么」。
4. 未经核实的引用要显式标注证据等级；**没有 assertion 和具体数字不要说"已验证"**。
5. `_handoff/HANDOFF.md` §4.2 提到过一个"判定探针"（含 norm-only baseline），**至今未实现**——里面的建议是"改代码前先跑它"。
