# CodeTrack 实验文档

> **版本**：v1.0（MVP-first 执行计划）
> **状态**：Stage 0 / Stage 1 待执行
> **配套文档**：`docs/CodeTrack-方案文档.md`（架构、算法、训练配置、风险登记表）
> **唯一目标**：先证明 **Syndrome > Confidence、Selective > Global、Joint > GOLA-FT**，把论文骨架立住；之后才谈 motion / 2-step generative recovery / memory protection。

---

## 0. 执行原则

1. **不通过 Stage 0，不做任何后续实验。** baseline 复现失败时，所有下游结论都不可信。
2. **每个阶段有明确的 go/no-go 判据。** 达不到就停下来改假设，不靠后续模块"补救"。
3. **所有阈值是工程筛选线，不是文献标准。** 论文中必须如此表述。
4. **corruption mask 不作为唯一 error target**；诊断监督一律使用 clean teacher residual。
5. **所有 FPS / FLOPs / 显存数字必须同卡实测**；设计目标不得当作结果。

---

## 1. 资源与流程约束

| 项 | 设定 |
|---|---|
| 单卡显存 | 24 GB |
| 主 batch | $8\sim16$ |
| gradient accumulation | 至 effective batch $=64$ |
| AMP | 开 |
| teacher 前向 | `torch.no_grad()`，不反传 |
| 随机种子 | 每实验 $\ge 3$ seeds（关键结论），探索期可 1 seed |
| 报告规范 | 均值 $\pm$ 标准差 |

> ⚠️ 因为多了 clean teacher forward，显存/时间约为单次前向的 $1.5\times$。先小 batch 跑通再放大。

> ⚠️ **多进程 / 多 worker 场景**（DataLoader workers、多卡评测、teacher cache 生成）在启动前先加载 `limit-process-threads` skill 完成线程数与 worker 数的设定与验证，避免 OpenMP/MKL/OpenCV 线程超订导致 GPU 利用率虚低。

---

## 2. Stage 0 — Baseline reproduction（不做训练）

### 2.1 目标

复现：

$$
\text{GOLA-B}\approx 77.5\ /\ 73.9\ /\ 61.6
\quad(\text{LasHeR PR / NPR / SR})
$$

### 2.2 检查清单

- [ ] 官方 GOLA-B checkpoint 正常加载，无 missing / unexpected keys。
- [ ] 输入尺寸确认：**template $112\times112$ / search $224\times224$**（不是 128/256）。
- [ ] token 数确认：$64 / 256$，$C=768$，concat 后 $768$ tokens。
- [ ] `_fuse_search()` 实际取的是 **TIR-position 的 256 个 token**（读代码确认，不假设）。
- [ ] 同时能取出 **RGB-position 的 256 个 token** 作为 $X_t^{aux}$（这是 CodeTrack 的必要前置能力）。
- [ ] 参数 / FLOPs / FPS 与论文同量级（$\sim99\text{M}$、$\sim85\text{G}$、$\sim125$ FPS，同卡实测为准）。

### 2.3 Go / No-Go

| 判据 | 通过线 |
|---|---|
| LasHeR SR 复现差距 | $\le$ 论文值 $-0.5$ |
| 已发布 checkpoint 数量 | $\ge 2$ 个（用于后续交叉验证） |

**不通过 $\Rightarrow$ 停下来查数据/评测脚本/预处理，禁止继续。**

---

## 3. Stage 1 — CodeTrack stabilization

### 3.1 目标

验证 diagnosis + 1-step recovery 的**机制正确性**，不追求最终性能。

### 3.2 配置

| 项 | 值 |
|---|---|
| steps | $500\sim1000$ |
| GOLA 状态 | **frozen**（warm-up，非最终模型，不违反硬约束） |
| 训练部分 | Diagnosis + 1-step Refiner |

### 3.3 必须通过的 sanity checks

| # | 检查 | 期望 |
|---|---|---|
| S1 | step 0 时 $X_t'$ 与 $X_t$ 的差异 | $=0$（zero-init 生效） |
| S2 | step 0 的 SR | 与 Stage 0 baseline 完全一致（$\Delta<0.05$） |
| S3 | 健康 token 是否被改动 | 严格无改动（identity bypass） |
| S4 | $q$ 分布 | 非退化（非常数、非全 0/1） |
| S5 | $\bar H$ 行和 | $=1$（归一化正确） |
| S6 | teacher 支路梯度 | 无梯度流入 frozen 参数 |
| S7 | $K_{\max}$ 实际触发率 | 记录每 batch 中 $q_i>\tau$ 的 token 比例 |

**S1/S2 不通过 $\Rightarrow$ 插入点或初始化有 bug，必须修好再做任何训练。**

---

## 4. Stage 2 — 主训练（Joint PEFT）

### 4.1 配置

| 项 | 值 |
|---|---|
| steps | $6000\sim8000$ optimizer steps |
| 训练部分 | $\boxed{\text{GOLA adapters} + \text{head} + \text{CodeTrack}}$ |
| 冻结 | DINOv2 foundation weights |
| LR | CodeTrack $1e\text{-}4$；GOLA low-rank $2\sim3e\text{-}5$；head $1e\text{-}5$ |
| Corruption 配比 | clean 30% / image 45% / feature 20% / compound 5% |

### 4.2 关键梯度路径验证

$$
X'\rightarrow \text{Head}\rightarrow L_{track}
$$

必须反传进 **CodeTrack 和 GOLA adapters**。

- [ ] 用一次 backward 检查 CodeTrack 与 GOLA low-rank 参数的 grad norm 均 $>0$。
- [ ] 检查 frozen DINOv2 参数 `grad is None`。
- [ ] 检查 teacher 支路不贡献梯度。

这才是论文中真正的 **learning error-correctable tracking representations**。

---

## 5. Stage 3 — 可选 DINO 解冻

| 项 | 值 |
|---|---|
| 解冻范围 | DINOv2 **last 2** blocks only |
| steps | $1500\sim2000$ |
| LR | $1\sim2\times10^{-6}$ |

**如果没有提升，就不要放进最终模型。** 不建议默认解冻 last 4/6 blocks。

---

## 6. Stage 4 — Temporal extension

只有前三阶段成功以后，再加入：

$$
\text{KF} + \text{temporal memory} + \text{memory protection}
$$

此时改 **causal clip training**（官方 pair sampling 不满足时序因果假设，必须先新写 4-frame causal sampler）。

---

## 7. MVP 实验（现在最该跑的）

### 7.1 范围

**先完全删掉：**

- diffusion；
- motion；
- temporal memory；
- template protection。

只留：

$$
\boxed{
\text{GOLA}
+
\text{Syndrome Diagnosis}
+
\text{H-routed Sparse Refiner}
}
$$

数据流：

$$
X_t^{RGB},\ X_t^{TIR}
\rightarrow
H\text{-based syndrome}
\rightarrow
q\in\mathbb R^{256}
\rightarrow
\text{Top-32 suspect tokens}
\rightarrow
\text{H-neighbor} + \text{other modality}
\rightarrow
\text{one-step residual recovery}
\rightarrow
\text{原 GOLA head}
$$

### 7.2 第一轮只需要 4 个模型

| 模型 | 回答的问题 |
|---|---|
| **GOLA pretrained** | baseline |
| **GOLA-FT same budget / corruption** | 单纯 fine-tuning 能否解释提升 |
| **Confidence + Sparse Refiner** | 普通置信检测够不够 |
| **Syndrome + H Sparse Refiner** | CodeTrack 核心假设 |

**参数量匹配要求**：`Confidence + Sparse Refiner` 与 `Syndrome + H Sparse Refiner` 的**可训练参数量必须对齐**（$\pm5\%$），否则"syndrome 更强"会被质疑成"参数更多"。

### 7.3 Go / No-Go 判据（工程筛选线）

| 指标 | 阈值 |
|---|---|
| $AUROC_{seen}$ | $>0.75$ |
| $AUROC_{held\text{-}out}$ | $>0.65$ |
| $SR_{robust}$ 相对 GOLA-FT | $+1\sim2$ 个百分点 |
| clean SR 下降 | $<0.3$ |
| 核心关系 | **Syndrome > Confidence** |

> 达到 $\Rightarrow$ 骨架立住，进入 Stage 3/4。
> 达不到 $\Rightarrow$ **先不要做 diffusion 和 motion。**

> **本轮一旦能证明 Syndrome > Confidence、Selective > Global、Joint > GOLA-FT，论文核心骨架基本就立住了。** 之后 motion、2-step generative recovery 和 memory protection 属于把核心故事往完整系统推进，而不是拿来救一个还没验证的基础假设。

---

## 8. Ablation Tree

### 8.1 主线（按论文逻辑一层一层证明）

$$
\text{A0 GOLA}
$$
$$
\downarrow
$$
$$
\text{A1 GOLA-FT}
$$
$$
\downarrow
$$
$$
\text{A2 GOLA-FT} + \text{same corruption}
$$
$$
\downarrow
$$
$$
\text{B confidence detector} + \text{sparse recovery}
$$
$$
\downarrow
$$
$$
\text{C syndrome detector} + \text{sparse recovery}
$$
$$
\downarrow
$$
$$
\text{D syndrome} + \text{spatial-neighbor recovery}
$$
$$
\downarrow
$$
$$
\boxed{\text{E syndrome} + \text{H-routed recovery}}
$$
$$
\downarrow
$$
$$
\text{F} + \text{2-step refinement}
$$
$$
\downarrow
$$
$$
\text{G} + \text{motion / temporal}
$$
$$
\downarrow
$$
$$
\boxed{\text{H} + \text{memory protection} = \text{Full CodeTrack}}
$$

**不要几十种排列组合。** 按这条链逐层证明。

### 8.2 额外三行（训练范式对比）

| 行 | 配置 | 论证目的 |
|---|---|---|
| 额外 1 | **Frozen CodeTrack** | 消融：证明 joint training 必要（注意：不能作为主方法） |
| 额外 2 | **Joint PEFT CodeTrack** | 主方法 |
| 额外 3 | **Joint PEFT + last-2-DINO** | 增强实验 |

### 8.3 H3：$H$ 的结构消融（单独一组）

$$
\text{Random } H
$$
$$
\text{Shuffled } H
$$
$$
\text{Spatial kNN}
$$
$$
\text{Dense attention}
$$
$$
\text{Learned graph}
$$
$$
\boxed{\text{Sparse } H + \text{learned edge}}
$$

**关键对照是 D vs E 与 `Spatial kNN` vs `Sparse H`**：如果 H-routing 不比 spatial kNN 强，就不硬吹 Tanner routing —— 保留 syndrome diagnosis，把 H routing 降级，改写为"结构化先验"。这直接对应风险 #3。

### 8.4 超参消融

| 维度 | 取值 |
|---|---|
| $M$（check 数） | $\{32,\ 64,\ 96\}$ |
| $K_{\max}$ | $\{16,\ 32,\ 64\}$ |
| 邻居数 $n_{nb}$ | $\{4,\ 8\}$ |
| $\lambda_{rec}$ | $\{0,\ 0.1,\ 0.25,\ 0.5\}$ |
| $\alpha$（诊断目标混合） | $\{0.3,\ 0.5,\ 0.7\}$ |

---

## 9. 评测协议

### 9.1 四层评测

#### 层 1：普通 benchmark

LasHeR：$PR / NPR / SR$。

再补：RGBT234、RGBT210、GTOT。

#### 层 2：Error diagnosis

$$
AUROC,\quad AUPRC,\quad Recall@K,\quad ECE / Brier
$$

#### 层 3：Robustness

分别报告：

$$
Clean
\quad
Seen\ corruption
\quad
Held\text{-}out\ corruption
\quad
Natural\ challenges
$$

LasHeR 重点看属性：

- LI（低照度）；
- PO（部分遮挡）；
- TO（全遮挡）；
- TC（热交叉）；
- FM（快速运动）；
- distractor / similar appearance。

GOLA 本身的 attribute results 已覆盖很多这类困难条件，因此**非常适合做差分分析**。

#### 层 4：Memory drift（让 Protect Memory 成为真贡献）

$$
WrongUpdateRate
=
\frac{\#\text{bad template updates}}{\#\text{updates}}
$$

以及：

- first failure frame；
- recovery after failure；
- continuous failure length；
- long-horizon SR；
- template contamination duration。

### 9.2 三层 corruption 协议（必须保留）

$$
\boxed{\text{Seen}\ \rightarrow\ \text{Held-out}\ \rightarrow\ \text{Natural}}
$$

| 层 | 定义 | 用途 |
|---|---|---|
| **Seen** | 训练中出现过的 corruption 类型 | 上界参考，不能单独作为卖点 |
| **Held-out** | 训练中**从未出现**的 corruption 类型 | 泛化性的真正证据 |
| **Natural** | 数据集原生困难属性（LI/PO/TO/TC/FM…） | 现实价值证据 |

> ⚠️ 只在 Seen 上报提升，审稿人会直接判为"过拟合到人工 corruption"。

### 9.3 诊断质量的特殊实验：按 tracking confidence 分桶

这是 H1 的核心设计。

分桶：$[0,0.25),\ [0.25,0.5),\ [0.5,0.75),\ [0.75,1.0]$。

在每一个桶中单独计算 error-detection AUROC。

**期望结论**：在 $score>0.75$ 时，syndrome 仍能检测出大量 corrupted token，而 confidence detector 失效。

$$
\Rightarrow
\boxed{\text{syndrome}\neq\text{confidence}}
$$

这个实验会很漂亮，也直接封堵风险 #1。

> 对照设置：`Syndrome Detector` vs `Direct MLP Error Predictor` vs `Tracking Confidence`，**参数量尽量匹配**。

### 9.4 结果表模板

**主表（LasHeR 等）**

| Method | Params | FLOPs | FPS | PR | NPR | SR |
|---|---:|---:|---:|---:|---:|---:|
| GOLA-B (repro) | | | | | | |
| + GOLA-FT | | | | | | |
| + Confidence + Sparse Recovery | | | | | | |
| **+ Syndrome + H Sparse Recovery (MVP)** | | | | | | |

**鲁棒性表**

| Method | Clean SR | Seen SR | Held-out SR | Natural SR |
|---|---:|---:|---:|---:|
| | | | | |

**诊断表**

| Detector | Params | AUROC(seen) | AUROC(held-out) | AUPRC | Recall@32 | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Tracking Confidence | | | | | | |
| Direct MLP Error Predictor | | | | | | |
| **Syndrome (ours)** | | | | | | |

**Confidence-stratified AUROC 表**

| Confidence bucket | Confidence detector AUROC | Syndrome AUROC |
|---|---:|---:|
| 0.00–0.25 | | |
| 0.25–0.50 | | |
| 0.50–0.75 | | |
| **0.75–1.00** | | |

**Memory drift 表（Stage 4）**

| Method | WrongUpdateRate $\downarrow$ | First failure frame $\uparrow$ | Recovery after failure $\uparrow$ | Contamination duration $\downarrow$ |
|---|---:|---:|---:|---:|
| GOLA (score>0.84) | | | | |
| **+ Reliability gate (ours)** | | | | |

---

## 10. 成本与效率评测

### 10.1 必须同卡 profile

| 指标 | 目标（设计目标，非结果） |
|---|---|
| bypass frame FPS | $\ge 95\%$ of GOLA |
| corrected frame FPS | $75\sim85\%$ of GOLA |
| average FPS | 最好 $\ge 100$ FPS |
| 新增可训练参数 | $2\sim3\text{M}$ |
| 新增 FLOPs | $<1\sim2\text{ G}$ |

### 10.2 必测项

- [ ] 纯 diagnosis（无 recovery）的额外开销；
- [ ] $K_{\max}=16/32/64$ 的开销-收益曲线；
- [ ] $q_i$ 触发稀疏时的实际 bypass 比例（统计所有评测帧）；
- [ ] peak memory。

---

## 11. 实验记录规范

每个 run 必须记录：

```yaml
run_id:            # 唯一标识
stage:             # 0 / 1 / 2 / 3 / 4 / MVP
git_commit:
seed:
config_hash:       # 完整配置文件的哈希
trainable_params:  # 总 + 分组
dataset_split:
corruption_mix:    # clean/image/feature/compound 实际比例
steps / wall_time:
peak_mem / avg_fps:
metrics:
  lasher: {PR, NPR, SR}
  robustness: {clean, seen, held_out, natural}
  diagnosis: {AUROC_seen, AUROC_heldout, AUPRC, Recall@32, ECE}
  stratified_auroc: {0-.25, .25-.5, .5-.75, .75-1}
notes:             # 异常观察、失败原因
```

**失败实验也必须记录** —— 风险登记表里的 6 条风险，每一条的结论都要有对应 run_id 支撑。

---

## 12. 决策树

```
Stage 0 复现
  ├─ 失败 ──> 修数据/评测，禁止继续
  └─ 通过
       └─> Stage 1 stabilization (S1–S7 sanity)
             ├─ S1/S2 失败 ──> 修插入点/初始化
             └─ 通过
                  └─> MVP 四模型对比
                        ├─ Syndrome ≤ Confidence ──> 重设计诊断（风险 #1）
                        ├─ SR 无提升 ──> 降 λ_rec，查 recovery 任务价值（风险 #2）
                        └─ 全部通过（AUROC 达标 + SR +1~2 + clean 掉 <0.3）
                             └─> Stage 2 主训练 (Joint PEFT)
                                   ├─ clean SR 掉 >0.3 ──> 降 LR / 增 clean 比例（风险 #6）
                                   └─ 通过
                                        ├─> Stage 3 (last-2 DINO，可选)
                                        └─> Stage 4 (KF + memory + protection)
                                              └─> 全量 ablation → 论文
```

### 12.1 提前终止条件

出现以下任一情况，**立即停止当前分支**：

1. Stage 0 复现失败；
2. S1/S2 未通过而继续训练；
3. Syndrome 与 Confidence 参数量未对齐就下结论；
4. Held-out corruption 上提升为 0 而只报告 Seen；
5. 2-step flow $\le$ 1-step refiner（删除 flow，见风险 #4）；
6. gated motion 后 FM 属性仍下降（删除 motion，见风险 #5）。

---

## 13. 论文级结论清单（实验要能支撑的句子）

按可证伪程度排列，每条都要有对应 run：

1. **结构化 syndrome 诊断优于同参数量的直接误差回归与 tracking confidence**（分层 2 + 9.3 分桶实验）。
2. **诊断监督使用 clean teacher residual 优于 binary corruption mask**（消融：$e^*$ vs mask）。
3. **H-routed 证据选择优于空间 kNN 与 dense attention**（8.3 的 D vs E）。
4. **选择性恢复优于全局恢复**（$K_{\max}$ 扫描 + Global recovery 对照）。
5. **联合训练优于 Frozen CodeTrack**（8.2 额外三行）。
6. **可靠性门控降低 WrongUpdateRate 并提升长时 SR**（层 4）。
7. **效率代价可控**（$\ge 100$ FPS、$<2$G 新增 FLOPs）。

> 若第 3 条不成立，第 1、4、5、6 条仍足以支撑一篇完整论文，只需把叙述从 "ECC/Tanner routing" 调整为 "structured error diagnosis"。

---

## 14. 时间线（相对顺序，非绝对日期）

| 阶段 | 依赖 | 预估 |
|---|---|---|
| Stage 0 复现 | — | 1–2 天 |
| Stage 1 stabilization | Stage 0 | 1 天 |
| MVP 四模型 + 诊断评测 | Stage 1 | 3–5 天 |
| Stage 2 主训练 | MVP 通过 | 1–2 周 |
| Stage 3 可选解冻 | Stage 2 | 3–5 天 |
| Stage 4 temporal + protection | Stage 2/3 | 1–2 周 |
| 全量 ablation + 写作 | 全部 | 2–3 周 |

> 投稿窗口请以目标会议当届的官方 deadline 为准（当前为 2026 年 10 月，最近窗口为 AAAI 2027 / CVPR 2027 一档，具体日期需按官网核实）。

---

*文档结束。架构、算法、训练配置与风险登记表见 `docs/CodeTrack-方案文档.md`。*
