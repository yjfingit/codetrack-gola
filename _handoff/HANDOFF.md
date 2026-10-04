# CodeTrack 项目交接文档

> 交接人：AI 编码会话（2026-10-04）
> 来源机器：单卡 RTX 4090 D
> 主仓库：`gola-CodeTrack/`，提交 `26c63e2`（工作区干净）
> 早期仓库：`CodeTrack/`

---

## 0. 一句话

**在完全冻结的 GOLA（AAAI 2026，DINOv2 ViT-B/14 + group-orthogonal LoRA，RGB-T 跟踪）之上，实现一个约 10M 参数的"可纠错时空表征恢复"分支 CodeTrack，让五个模块真正互相作用并联合训练。**

主人的原始意图，按优先级（违反任何一条都算失败）：

1. **复现架构。** 冲突时优先级：**架构图 > GOLA 真实 tensor 约束 > 原论文 > 参考项目实现**
2. **不许占位/identity 假实现。** 每个模块必须真的在做事。
3. **严禁用 `.detach()` / `no_grad()` 切断梯度。** 必须是联合训练。
4. **最重要的顺序：先用小数据把"理论正确"验出来；参数之类的之后再调。**

---

## 1. 两个仓库的关系（重要）

| | `gola-CodeTrack/` | `CodeTrack/` |
|---|---|---|
| 定位 | **主仓库**（当前工作面） | **早期独立项目**（另一套架构） |
| backbone | GOLA 的 **DINOv2 ViT-B/14**（`patch 14`） | 共享 **ViT-B/16** + 独立 RGB/TIR patch_embed |
| 适配器 | **group-orthogonal LoRA**（1296 个张量） | **无 LoRA**（LoRA key 数 = 0） |
| 纠错机制 | ECC parity check `H̄(64,256)` + `q=σ(H̄ᵀs+b)` + `TopK(q)` 路由 + 2 步噪声调制精修 | Target **Codebook**（K=16 身份 token + M=16 稀疏 parity token）+ **神经置信传播解码器** `models/decoder/bp.py` + Reliability-Adaptive Tanner Graph |
| 预训练权重 | `weights/gola_b224.bin`、`gola_l224.bin` | `checkpoints/pretrained/OSTrack_ep0300.pth.tar` |
| 规模 | 总 106.37M / 可训练 20.65M / CodeTrack 10.03M | 可训练 27.527M |

**两套权重互不兼容**：`CodeTrack/outputs/*/final.pth` 的 key 是
`backbone.pos_embed_z`、`codebook.ecc_identity.weight`、`syndrome.energy_gain`——
在新仓库里一个都对不上。**这些中间 checkpoint 已排除（77 GB），不要试图加载。**

**但 `CodeTrack/` 的源码必须带走**：它的 `models/decoder/bp.py`（神经置信传播）、
`models/syndrome/`、`models/reliability/`、`models/codebook/` 是"复现架构"时最有价值的对照实现，
而且是**另一套完全不同的技术路线**，可能比当前路线更容易让恢复分支真正起作用。

---

## 2. 架构与数据流（`gola-CodeTrack` 的复现对象）

```
冻结的 GOLA trunk → F_L (B, 768, 768) = [z_v|x_v|z_i|x_i|d_v|d_i]
                                        64  256  64  256  64   64  tokens
  z = template 112×112 → 8×8,  x = search 224×224 → 16×16,  C = 768,  d = online template

块 3   ECC 结构化诊断    H̄(64,256) 固定稀疏支撑 + 可学习边权
                        s = H̄·err,  q = σ(H̄ᵀs + b) ∈ [0,1]^256   ← 每 token 出错概率
块 4   卡尔曼运动先验    M_t(B,1,16,16) + 不确定度 u_t
      时序记忆 TRC/TGS   → prior_tokens (B, F, 128)
块 5a  H 路由稀疏恢复    TopK(q)=32 个可疑 token，从 K_n=8 个校验邻居路由证据 → X_rec
块 5b  噪声调制精修 2 步  noise_gate = write_gate = q，写回 w·q·pred → X_final
块 6   在线模板保护      门控 c_t
      ↓
冻结的 GOLA anchor-free head → score_map(B,16,16), boxes(B,16,16,4)
```

**双分支训练**：student 看被破坏的搜索图；clean teacher（`no_grad`）提供诊断残差目标。
诊断标签来自**真实特征偏差** `e* = α·e_feat + (1-α)·e_aux`，而不是注入器写下的 mask
（避免网络学会"读标签"）。

**⚠ 与架构图不同、已按真实代码改掉、不要改回去**：

| 项 | 图里 | 真实 | 理由 |
|---|---|---|---|
| 输入尺寸 | 128/256 | **112/224** | ViT-B/14：`112/14=8`、`224/14=16` → 恰好 64/256 token |
| 第 4 槽 `d` | 前一帧 | **online template** | 不是时序前驱，直接影响时序模块怎么接 |
| CodeTrack 前 | 再加一层 LayerNorm | 上游 `_fusion` **已有** `self.norm` | 再加会破坏 step-0 恒等性 |
| check 数 M | 16 | **64** | 度数设计目标 |

---

## 3. 确认修好的（每条都有断言，不是自述）

先跑这个，期望 **101 项全绿**：

```bash
source scripts/00_env.sh && "$PYTHON" tools/preflight_acceptance.py
```

| 项 | 断言 / 证据 |
|---|---|
| 五模块真实落地 | `tools/codetrack_verify.py`：10/10 模块非零梯度、checkpoint 1311/1311、恒等性 `1.15e-4` |
| **数据流真的在交互** | 50 步固定 batch 参数位移：`H` 3.63e-3、`refiner` 5.79e-3、`denoiser` 6.25e-3、`diagnosis` 4.69e-3 |
| 联合训练（未切梯度） | `tools/lora_grad_check.py`：LoRA **1296/1296** 移动、冻结主干 **0 漂移** |
| 因果性 | `tools/causality_check.py`：**6/6** |
| 互补性 | `tools/complementarity_check.py`：**17/17** |
| 优化器覆盖 | **1406/1406**，8 组，1 维参数 `wd=0` 且 LR 路由正确 |
| **S1 真冻结 GOLA** | 真跑 optimizer 只有 **73–97** 个 `codetrack.*`，**0** 个 `blocks.*`/`head.*`/`lora` |
| update 作为时间轴 | 真跑打印 `[stage] update budget reached; stopping after epoch 9` |
| cosine 完整走完 | S0 最终 `lr = 1.000e-06`（= `lr_min`） |
| 四阶段 scope 各不相同 | preflight 实测：S1=73 / S2=1382 / S3 含 `blocks.10/11` / S4=1406 含 motion 24 个 |
| S4 重新冻结主干 | S4 optimizer 里 `blocks.*` 非 LoRA 参数 **0 个** |
| AMP 下不崩 | BCE 改 `*_with_logits`；`q_logits` 通路补齐 |
| `Loss/pres` 锚点口径 | **0.95 → 0.0001**（锚在"恢复未选中的 token"） |
| 判据可用 | 无阈值 `AUROC/AUPRC(q, mask)` + `q_std/q_pos_mean/q_neg_mean` + `d_input/d_before/d_after` |
| denoiser 噪声不是死码 | 坏 token 噪声 std / 健康 token = **98.2×**（此前恒为 0） |

---

## 4. ★ 唯一未解决的主线问题：恢复分支产不出正向效果

**定义**（"恢复有效"的唯一判据）：`d_final < d_input`，其中 `d_* = 1 − cos(·, X_clean)`，
**只在 `corruption_mask` 为真的 token 上统计**。

三个独立实验，结论一致：

| 实验 | 规模 | `d_input` | `d_before` | `d_after` | `gain_total` |
|---|---|---:|---:|---:|---:|
| 10 序列快速验证 | 20 updates | 0.0462 | 0.0462 | 0.0462 | **0.0000** |
| 600-update 消融（合成固定 batch） | 600 updates | 0.119 | 0.119 | 0.141 | **−0.022** |
| S0 真跑（全量 LasHeR） | 512 updates | 0.0179 | — | 0.2472 | **−0.2303** |

同时**诊断头确实在学**：`q_auroc_mask` 0.56 → 0.69、`Loss/diag` 0.43 → 0.23。
**问题只在恢复（refiner）与精修（denoiser），不在诊断。**

**gate 已排除为主因**：600 步里 `residual_gate` 只从 `−8.0001` 走到 `−7.9613`
（`sigmoid` 仍 `3.5e-4`，等于没开），所以 0.19 的角距离增量**不可能来自 gate 放大**。

### 4.1 ★ 新的线索：很可能是"归一化层造成的 scale 假象"，不是分支没动

`CodeTrack/`（早期项目）在同一个问题上留下了记录，`codetrack/engine/losses.py`：

> *"The decoder ends in a learnable LayerNorm (`out_norm`) that neither the corrupted input nor
> the clean teacher has passed through. So part of `e_after` is a pure **scale change, not message
> passing**. `tools/recovery_probe.py` reports the **norm-only baseline** that separates the two;
> without it a negative recovery_gain cannot be attributed to the message updates."*

**这直接命中当前的症状。** 在 `gola-CodeTrack` 里，三个被比较的量经过的归一化层**互不相同**：

| 量 | 经过的归一化 |
|---|---|
| `d_input`（`X_t`） | `_fusion` 的 `self.norm` |
| `d_before`（`X_rec`） | `self.norm` → `refiner.norm`（`nn.LayerNorm`） |
| `d_after`（`X_final`） | `self.norm` → `refiner.norm` → `denoiser.norm1` / `norm2` |

所以 `d_before == d_input` 到五位小数**可能根本不是"分支没动"**，而是：
refiner 的 `LayerNorm` 与 `residual_gate` 的组合造成一个近似纯尺度变换，
在 `1 − cos` 这个**尺度不变**的度量下读出来就是"完全相同"。

**正确的判据**（早期项目给出的方法）：加一个 **norm-only baseline**——
把 `X_t` 只过一遍 refiner 的 `norm`（不做任何消息传递），再算 `d_norm_only`。
只有 `d_after < d_norm_only` 才能说明**消息传递**真的在起作用。

同样的教训也适用于 `d_after`：denoiser 末端有 LayerNorm，必须先排除它的尺度效应。

### 4.2 四种可能，判定探针（**接手后第一件事，不要先改代码**）

扫 `residual_gate ∈ {−8, −5, −2, 0, 3}`，记录：
`||X_final − X_t||max`、`||X_rec − X_t||max`、三个 `d_*`、**以及 norm-only baseline**。

| 观察 | 结论 | 修法 |
|---|---|---|
| 随 gate **单调放大** | gate 没开，改动被 fp32 精度吞掉 | 修在**初始化或目标函数** |
| **完全不随 gate 变** | 分支真的断了（有 detach / 被覆盖） | 修在**接线**（用 §6 的扫描器揪） |
| 只在 1e-6 量级变 | 度量精度问题 | 改**度量口径**（直接报 `‖Δ‖` 或按维度归一化） |
| `d_after ≈ d_norm_only` 且都 ≈ `d_input` | **尺度假象**（§4.1） | 加 norm-only baseline 并改判据，**不要动模型** |

### 4.3 另一个必须检查的实现陷阱（早期项目踩过）

`CodeTrack/codetrack/engine/losses.py` 的另一条记录：

> *"PER SAMPLE. The `dim` argument is not optional: without it `(err_r * m_r).sum()` collapses the
> whole batch into one scalar, so every 'distribution' statistic computed downstream (active
> fraction, p95) silently reads 0 because a scalar has `numel() == 1`. **That is exactly the bug
> that made a dead hinge look measured.**"*

**要求**：接入任何新指标前，检查所有 `.sum()` / `.mean()` 的 `dim` 参数，
确认 per-sample 统计没有被塌成标量。`gola-CodeTrack/codetrack/criteria.py` 里的
`_auprc_against_mask` 用了 `.sum()`（无 dim），需要复核。

---

## 5. 工程纪律（本项目的 bug 反复出自这两类，务必遵守）

### 纪律一：验证"真正会运行的东西"，不是自己的摘要

**真实事故**：`tools/preflight_acceptance.py` 曾**复刻**一份优化器规则表去测 → 全绿；
而真正会跑的 `config/GOLA/*/config.yaml` 把 head/LoRA 的 1 维参数全塞进了 CodeTrack 组。
**测试测的是自己的 fixture。**

→ 凡"配置声明"与"代码消费"分两侧的，必须读**真实配置**并断言两侧一致。

### 纪律二：改签名后 grep 全部调用点，并为"默认值会不会让信号变 0"写**行为**断言

**真实事故**（denoiser 死噪声）：

```python
if alpha is None: alpha = torch.ones(...)   # 调用方从来不传 alpha → 恒为 1
eps = eps * (1.0 - alpha)                   # → 噪声恒为 0
```

写回方向修对了（测了 91.8×），**但同一函数另一行被同一个默认值变成死代码**，
而"验证"方式竟是 grep 变量名在不在源码里。

→ 断言**行为**，不断言源码字符串。正确测试是"坏 token 噪声 std / 健康 token = 98.2×"。

### 纪律三：不说"已验证"，除非能指出是哪条断言、跑出什么数

本项目多次出现"上一轮说已修好、实际没修好"（`1406/1406` 那次最典型：看的是复刻配置的运行结果）。

---

## 6. 已知的"声明了但代码从不读取"的字段

| 字段 | 位置 | 影响 |
|---|---|---|
| `scheduled_sampling_prob` | `codetrack/config.py` | S4 的 exposure bias 处理是空的 |
| `memory_tbptt_steps` | `codetrack/config.py` | 跨帧信用分配是空的（当前无条件 `mem.detach()`） |

全仓库 grep 命中数 = 2（只出现在 `config.py`）。

**框架缺口**：`--resume` 被 argparse 接受、存进 `runtime_vars.resume`、传给 `DefaultApplication`
的 `application_state_file`，但**从不被读取**。所以框架层面的 resume 实际不生效。
编排器的 resume 走的是"上一阶段 checkpoint 作 `--weight_path`" + "阶段边界与 epoch 边界对齐"。

**建议先写一个"配置字段 vs 实际读取字段"扫描器**，一次性列全这类问题。

---

## 7. 怎么跑

```bash
source scripts/00_env.sh                     # 必需：libturbojpeg 路径 + CuBLAS 确定性

# 静态综合门（101 项，读真实配置）
"$PYTHON" tools/preflight_acceptance.py

# 其它验证器
"$PYTHON" tools/causality_check.py            # 因果性 6/6
"$PYTHON" tools/lora_grad_check.py            # LoRA 训练 / 主干不漂移
"$PYTHON" tools/codetrack_verify.py           # 恒等性 + checkpoint + 梯度连通
"$PYTHON" tools/complementarity_check.py      # 17 项互补性

# 小数据快速验证（需先把 consts.yaml 指向 data/LasHeR_train10/）
"$PYTHON" main.py GOLA codetrack_fixverify --distributed_nproc_per_node 1 --disable_wandb \
  --weight_path "$WEIGHT" --output_dir="$PWD/outputs/fixverify"

# 四阶段一键（编排器 + 运行时验收门 + 自动 GO/NO-GO + S3 自动回滚）
"$PYTHON" tools/train_codetrack_staged.py --stages s1,s2,s3,s4 --resume auto \
  --output-root outputs/staged
# 状态机 smoke（每阶段 10 updates，约 3 分钟）
"$PYTHON" tools/train_codetrack_staged.py --stages s1,s2,s3,s4 --smoke \
  --output-root outputs/staged_smoke
```

| 入口 | 作用 |
|---|---|
| `tools/train_codetrack_staged.py` | 四阶段编排器（子进程 + 状态机 + resume + S3 回滚） |
| `tools/stage_validate.py` | 每阶段验收门（HARD 失败必须停；SOFT 只 WARN） |
| `tools/stage_metrics.py` | 日志解析、JSONL、参数位移 |
| `tools/preflight_acceptance.py` | 静态综合门（读**真实**配置） |
| `tools/recovery_report.py` | `-8` vs `-5` 真训练对照（需 600 updates 起；20 updates 尺度下必然 NO-GO，别误用） |

---

## 8. 四阶段配方与时间（实测）

实测吞吐 **0.30 s/micro-step**（8 样本）→ **26.7 样本/秒** → **4.80 s/optimizer update**（128 样本）。
`data` 等待只占 **4%**，GPU 是瓶颈。一个 epoch（1024 updates）= **1.41 h**。

| 阶段 | updates | warmup | trainable | 时间 |
|---|---:|---:|---|---:|
| S1 spatial recovery | 1500 | 128 | `codetrack.*` only | 2.0 h |
| S2 joint PEFT | 8000 | 410 | +LoRA/head/embed | 11.0 h |
| S3 DINOv2 last-2 | 1800 | 256 | +`blocks.10/11` | 2.6 h |
| S4 temporal | 3000 | 48 | +motion/memory/gate | ~4 h |
| **S1+S2+S3** | 11300 | | | **15.7 h** |
| **S1..S4** | 14300 | | | **约 19.7 h** |

### ★ 何时进行完整实验

**只有在 §4.2 判定探针通过、`Error/gain_total > 0`（且经 norm-only baseline 校正）之后才启动。**
否则会用一个"恢复无效"的配置跑掉十几个小时。这一步是主人的核心验收。

---

## 9. S4 还没实现的部分（独立一大块，纯实现，不涉及调参）

| 缺什么 | 位置 | 说明 |
|---|---|---|
| causal clip sampler | 需新建 | 采样连续 `{t-3,t-2,t-1,t}`；当前只有 `random` pair sampler |
| `reset_sequence()` 接线 | `GOLA_DINOv2.reset_sequence()` 已存在，**训练路径从不调用** | 防序列 A 的 memory 进序列 B |
| TBPTT | `memory_tbptt_steps` 从不被读 | 替换无条件 `mem.detach()` |
| scheduled sampling | `scheduled_sampling_prob` 从不被读 | 只作用于卡尔曼观测与 memory 状态，**不要改 SiamFC crop** |
| D4 reliability 值域 | `codetrack/motion.py`: `cur_rel = sigmoid(mean(1-q))` → 压在 `[0.50, 0.73]` | 必须能覆盖 `[0,1]` |
| D5 历史 slot 标签 | 历史 slot 共用当前帧 trust | 每 slot 存自己的 label |
| 当前帧 GT 影响未来帧 | `target_mask` 来自当前 `gt_box_xywh` | read-before-write 保证不影响当前帧输出，但写进 memory 后影响**未来帧** → train/test mismatch |

---

## 10. 环境复建

| 项 | 值 |
|---|---|
| Python | 3.10.8 |
| torch | 2.5.1+cu124（CUDA 12.4，cuDNN 9.1） |
| GPU（原机器） | RTX 4090 D 24 GB（micro_batch 8 峰值 4.4–5.7 GiB） |
| `LD_LIBRARY_PATH` | 含 libturbojpeg 的目录（原机器 `/root/autodl-tmp/lab/tools`） |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8`（框架开确定性算法，**必须**） |
| `NCCL_P2P_DISABLE` | `1` |
| 依赖 | `pip_freeze_gola_env.txt`（107 包） |

```bash
conda create -n gola python=3.10.8 -y && conda activate gola
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r gola-CodeTrack/requirements.txt
pip install -r pip_freeze_gola_env.txt
```

**`scripts/00_env.sh` 里有硬编码路径**（`PYTHON`、libturbojpeg 目录），迁移后必须改。

### 数据（**不在包里**，需新机器重新准备）

| 数据 | 原路径 | 体积 | 说明 |
|---|---|---:|---|
| LasHeR | `/root/autodl-tmp/lab/dataset/LasHeR` | **214 GB** | train 979 序列 / test 245 序列；`trainingsetList.txt`、`testingsetList.txt`、`annos/`、`AttriSeqsTxt/` |
| RGBT234 | `/root/autodl-tmp/lab/dataset/RGBT234` | — | 次要 |

`data/LasHeR_*` 里是**绝对路径符号链接**，迁移后失效，必须重建：

```bash
DS=/path/to/LasHeR; VIEW=$PWD/data/LasHeR_train10
mkdir -p "$VIEW"; head -10 "$DS/trainingsetList.txt" > "$VIEW/trainingsetList.txt"
head -3 "$DS/testingsetList.txt" > "$VIEW/testingsetList.txt"
for i in trainingset testingset annos AttriSeqsTxt Attributes_order.txt; do ln -sfn "$DS/$i" "$VIEW/$i"; done
```

**`consts.yaml` 的 `LasHeR_PATH` 必须改。** 注意：交接时它被临时指向了 10 序列视图
（`data/LasHeR_train10/`），**正式训练前要指回全量数据集**。

---

## 11. 已排除的内容与理由

| 排除项 | 体积 | 理由 | 恢复方式 |
|---|---:|---|---|
| `CodeTrack/outputs/*/{last,final}.pth` | **77 GB** | 架构不兼容（ViT-B/16 + Codebook + BP 解码器），key 与新仓库对不上，**加载不了** | 无意义，不必恢复 |
| `gola-CodeTrack/repo/*` | 144 MB | 11 个第三方参考检出，各自带 `.git` | 按 `repo/README.md` 记录的 URL `git clone` |
| `gola-CodeTrack/third_party/GOLA` | 11 MB | 上游只读参考 @ `339c737` | `git clone https://github.com/MelanTech/GOLA && git checkout 339c737` |
| `**/__pycache__`、`*.pyc`、`.pytest_cache` | — | 可重建 | 自动生成 |
| LasHeR / RGBT234 | 214 GB | 太大，有独立下载渠道 | 见 §10 |

**包内包含**：两个仓库的**全部代码**、**全部文本产物**（`CodeTrack/outputs/` 的 759 个
log/json/md/txt/csv，仅 5.8 MB）、**预训练权重**（`OSTrack_ep0300.pth.tar`、
`gola_b224.bin`、`gola_l224.bin`）、`.git` 完整历史、22 篇论文 PDF。

---

## 12. 给接手者的第一条建议

**先跑 §4.2 的判定探针（含 norm-only baseline），再决定改什么。**

本项目已经因为"在没搞清根因时动手"浪费过多轮时间。而 §4.1 那个来自早期项目的线索
（**归一化层造成的 scale 假象**）是目前最有可能的解释——**先验证它，再动模型**。

---

## 附：提交链

```
3dfbb58  CodeTrack 分支首次落地
0100a75  修 LoRA 未训练 / 当前帧 GT 泄漏 / 运动头幅度
49d8377  记录全量 LasHeR 就绪审计（L_motion 从未计算）
a205ada  让 L_motion 在真实数据上计算；修 0-d 参数崩溃与 LR 规则顺序
e8ee251  修累积语义、scheduler 时间轴、死 L_diag、写回方向
d39552f  修优化器规则作用域、clean teacher 污染、加恢复增益指标
0e2669c  修死噪声、落地真正的 S1 冻结与阶段预算、四阶段编排器
26c63e2  记录第六轮诚实状态：恢复尚未有效
```
