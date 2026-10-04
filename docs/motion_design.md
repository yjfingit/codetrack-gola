# 运动先验与时序建模 — 文献依据与实现决策

本文档记录 CodeTrack 运动/时序模块的**文献依据**与**据此做出的具体改动**，便于写 Method 时引用。

---

## 1. 检索结论：哪些论文真正提供了可用的运动建模

| 论文 | 是否有显式运动模型 | 我们采纳的内容 |
|---|---|---|
| **DTPTrack** (CVPR 2026) | **没有** Kalman/速度/轨迹模型，运动只能由参考帧隐式表达 | **采纳其 TRC + TGS 机制**（见 §2），这是本次改动的主体 |
| **MoKA-HP** (Neurocomputing 2026) | 标题含 "Motion-aware KAdaptation"，但**全文闭源** | **不采纳、不引用其内部设计**（见 §4） |
| **SCDT** (CVPR 2026) | 无运动模型，但有时空条件去噪 | 其 FiLM / mean-var 已用于恢复模块，本次不改 |
| **RTracker** (CVPR 2024) | 官方仓库无代码 | 无 |

### DTPTrack 的两个关键消融（决定了我们的设计）

**Table 6：用启发式运动先验替换 TGS 会掉点。**

| TGS 的替代方案 | LaSOT AUC | VastTrack AUC |
|---|---:|---:|
| momentum 先验 | 73.8 | 40.1 |
| optical flow 先验 | 73.2 | 39.4 |
| **学习式 TGS（原文）** | **74.3** | **40.7** |

> **结论（决定了我们的架构）：运动信息必须作为"特征/输入"进入学习模块，而不能作为"运动模型的输出"去硬性替代时序聚合。**
> 这正是 CodeTrack 现有的两处用法，我们**保留**而不是改成"用 Kalman 预测框去引导/替换特征"：
> - Kalman 的 `M_t (B,1,16,16)` 作为 recovery cross-attention 的**加性对数偏置**；
> - Kalman 的 `u_t` 与 Mahalanobis 距离作为 **TRC 门控的输入特征**。

**Table 5：TRC/TGS 各设计的独立贡献**（ViT-B/224，LaSOT AUC，full = 74.3）。

| 变体 | LaSOT AUC | Δ |
|---|---:|---:|
| (a) 静态阈值替代学习式门控 | 72.0 | **−2.3** |
| (b) 首帧也参与门控（不锚定 `c_0=1`） | 73.2 | −1.1 |
| (c) 去掉可学习 base prior | 72.7 | −1.6 |
| (d) 直接拼接 summary 而非独立 prior token | 73.4 | −0.9 |
| (e) 基线（无 DTPTrack） | 73.3 | — |

---

## 2. 据此实现的改动

### 2.1 TRC — Temporal Reliability Calibrator

```
s_i   = Σ_j Z_ij M_ij / (Σ_j M_ij + ε)                      (eq.1, masked average pooling)
c     = sigmoid( MLP( [s_0..s_F], u_t, Mahalanobis ) )       (eq.2, 学习式门控)
c_0   = 1.0                                                  (锚定：首帧来自 GT 模板)
ŝ_i   = s_i · c_i                                            (抑制)
```

实现位置：`codetrack/motion.py::TemporalMemory`。

| 设计点 | 实现 | 依据 |
|---|---|---|
| masked average pooling | `_masked_average`，掩码来自 `_search_target_mask`（把观测框栅格化到 16×16 搜索网格） | eq.1；消融显示池化方式影响显著 |
| **学习式**门控 | 2 层 MLP + sigmoid | (a) −2.3 |
| **锚定首帧** | `c_anchored = [1, c[:, 1:]]` | (b) −1.1 |
| Kalman 量作为**门控输入** | `[u_t, Mahalanobis]` 拼进 MLP 输入 | Table 6；固定阈值是最差的消融 |
| **Mahalanobis 距离** | `sqrt(νᵀ S⁻¹ ν)`，`S = H P Hᵀ + R` | 比裸 `‖ν‖` 更有依据，且考虑了协方差各向异性 |

> 注：DTPTrack 原文 eq.2 只用 `[s_1..s_3]`，但其**代码**把 4 个 summary 全喂进 MLP 再把 `c_0` 丢掉。我们采用"喂入全部、随后锚定"的一致做法。

### 2.2 TGS — Temporal Guidance Synthesizer

```
P_dyn = P_base + f_mod( [ŝ_0..ŝ_F] )                          (eq.3)
```

实现：`base_prior (1, K, memory_dim)` 为**可学习**参数，`tgs_modulator` 输出加性调制。

| 设计点 | 实现 | 依据 |
|---|---|---|
| 可学习 base prior | `self.base_prior`（trunc-normal 初始化） | (c) −1.6 |
| **独立 prior token**，不拼进视觉 token 流 | prior tokens 送入 refiner 与 denoiser 作为**条件**，绝不改写 `X_t` | (d) −0.9；原文"guides attention without contaminating raw features" |

### 2.3 记忆准入（论文未提，全在代码里）

DTPTrack 的推理代码用 `update_criteria: 0.85`：预测最大分数低于阈值时**不接纳**该帧，而是复制上一帧。这是漂移控制的重要一环，但论文完全没写。

我们在 `TemporalMemory` 中实现为 `admission_threshold=0.85`，并把**上一帧**的跟踪分数作为准入信号（head 尚未运行，这是因果正确的信号）。训练时全部接纳，保持真实时序。

---

## 3. 一个重要的实现陷阱（记录以免重犯）

恢复分支的输出层**不能零初始化**：

- 若 `up.weight = 0`，则 `pred = 0`，而残差以 `gate * pred` 形式注入，梯度 `∂/∂(上游) ∝ up.weight = 0` → **整个分支（含运动先验、时序记忆、条件投影）梯度恒为零**，参数永不被更新。
- 因此改用**小而非零**的残差门 `sigmoid(-8) ≈ 3.4e-4`：输出近似恒等（`max|Δ score_map| ≈ 2.4e-6`），同时梯度可达。
- 实测（同一模型、同一 loss）：门 = −12 时到达时序记忆的梯度为 `3.1e-7`，门 = −8 时为 `1.7e-5`（**提升约 50×**）。−12 时分支"名义上连通但实际训不动"。

`tools/codetrack_verify.py` 第 5 项专门检查每个新模块的梯度是否非零，就是为了拦住这类问题。

---

## 4. MoKA-HP：明确放弃，不作任何断言

MoKA-HP（Neurocomputing 2026, DOI `10.1016/j.neucom.2025.132163`）是本项目最初的"运动模块参考"首选，但检索结论是**完全无法获取**：

- **Unpaywall**：`is_oa: false`，`oa_locations: []`；**OpenAlex** `W4416662262`：`is_oa: false`、**无摘要**；**Semantic Scholar**：`abstract: null`、无开放 PDF。
- 出版社页面全部反爬（ScienceDirect 403 + captcha、ACM DL 403、literatum 403）。
- **无 arXiv 预印本**，无 ResearchGate 副本，作者 ORCID（Si Chen, 0000-0002-5631-7942）无预印本链接。
- **无任何公开代码**（GitHub 搜索 + 5 个候选仓库 URL 全部失败；同名仓库 `ams-v-livers/amainu_moka_hp` 是一个无关的网页小游戏）。
- 引用它的 3 篇论文同样闭源，无法从二手描述还原方法。

因此：**KFMM 的状态向量、F/H/Q/R、注入方式、不确定性用法，全部 UNKNOWN。本实现不对 MoKA-HP 内部设计做任何假设，也不引用它作为方法依据。**

若日后拿到原文，值得与我们的设计对照的三点：
1. KFMM 如何把 Kalman 输出注入 tracker 特征（token 拼接？attention 偏置？—— 我们用的是**偏置 + 门控输入**）；
2. 它如何使用协方差/不确定性（我们同时用作门控输入与模板保护信号）；
3. "historical prompts" 的缓冲长度与更新规则（我们：`frames=3` 的可靠性加权 bank + 0.85 准入阈值）。

获取途径建议：机构订阅 / ScienceDirect TDM API / 向通讯作者索取。

---

## 5. 复现与验证

```bash
source scripts/00_env.sh
bash scripts/preflight.sh                       # 环境自检
"$PYTHON" tools/codetrack_verify.py             # 形状审计 + checkpoint + 梯度连通
bash scripts/codetrack_train_smoke.sh           # 单序列联合训练
```

`tools/codetrack_verify.py` 的第 5 项会打印每个新模块的梯度范数；**任何模块为零即视为失败**，因为那意味着该模块虽然结构上存在但实际不参与训练。
