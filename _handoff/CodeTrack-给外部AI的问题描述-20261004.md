# CodeTrack 问题描述 —— 给外部 AI 的启动提示词

## 你要做什么

一个 RGB-T 单目标跟踪项目（CodeTrack，建立在 AAAI 2026 的 GOLA 之上）的**纠错码恢复分支完全不工作**。我已经定位到具体症状并做了测量，需要你判断**根因**并给出**改动方案**。请先读代码再下结论，不要凭直觉建议。

## 仓库

```
https://github.com/yjfingit/codetrack-gola
commit: 964cf328b58486f75df92bd7138fa9c7f4bf42b9  (branch main)
```

**先读这一个文件，它包含全部实测数据和我的分析：**

```
_probe/grid/s1_r1/README.md
```

然后读源码：

```
codetrack/ecc.py          # 奇偶校验诊断 → syndrome s → 逐 token 错误概率 q
codetrack/criteria.py     # 损失 + 指标（_auroc_against_mask:163, _relative_error:290）
trackit/criteria/builder.py           # 注意：gain_margin 有 bug，不转发（见下）
config/GOLA/codetrack_s1/config.yaml  # S1 配置
```

## 背景：这条分支想做什么（30 秒版）

GOLA 是 DINOv2 ViT-B/14 + group-orthogonal LoRA 的 RGB-T 跟踪器。CodeTrack 在它上面加了一条**纠错码启发的恢复支路**：

1. 用 LDPC 风格的稀疏二值校验矩阵 `H_bar`(64×256) 把 256 个 search token 压成 **64 个 syndrome 检查**
2. 每个检查算"观测 vs 参考"的差异 `delta`，经 MLP 得到 syndrome logit `s_raw`，`s = sigmoid(s_raw)`
3. 用 `q_logits = H_bar^T s + vote_bias` 算出**每个 token 的错误概率 `q`**（256 维）
4. `TopK(q, 32)` 选出最可疑的 32 个 token，做 H 路由稀疏恢复 + 噪声调制精修
5. 期望：恢复后的 token 表示比输入的更接近干净参考，即 **`d_after < d_input`**

## 症状：恢复效果为零，且 `q` 是常数

2026-10-04 跑了四卡 S1 筛选（每卡一个超参组合，~400 updates）。**四路全部失败**：

| trial | gain_total | q_error_spearman | q_auroc_mask | **q_std** |
|---|---|---|---|---|
| baseline | 0.00127 | 0.0549 | 0.6595 | 1.0e-04 |
| lr=5e-5 | 0.00352 | 0.0606 | 0.6667 | 1.0e-04 |
| λ_gain=0.4 | 0.00162 | 0.0387 | 0.6574 | 1.0e-04 |
| α=0.7 | 0.00190 | 0.0481 | **0.5801** | 1.0e-04 |

硬门槛（`tools/stage_validate.py`）：
```python
GAIN_TOTAL_MIN  = 5e-3     # d_input - d_after
DIAG_AUROC_SOFT = 0.60     # AUROC(q, corruption_mask)
DIAG_Q_STD_MIN  = 1e-3     # q 不能塌成常数
```

`gain_total` 差 1.4~3.9 倍，`q_std` 差 10 倍。

## ★ 关键发现：`q` 的"排序对了，但幅度完全死了"

这是最反直觉、也最重要的一组数字。四路的 `q` 在**损坏 token** 和**干净 token** 上的均值：

| trial | q_pos_mean − q_neg_mean | q_std | q_auroc_mask |
|---|---|---|---|
| baseline | **0.0000** | 1e-04 | 0.6595 |
| lr=5e-5 | **0.0001** | 1e-04 | 0.6667 |
| λ_gain=0.4 | **0.0001** | 1e-04 | 0.6574 |
| α=0.7 | **0.0000** | 1e-04 | 0.5801 |

**正负样本均值差 ≈ 0，但 AUROC = 0.66。**

两者不矛盾，因为：`_auroc_against_mask` 是**纯排序**统计量（Mann-Whitney U），对幅度完全不敏感；而 `q_pos_mean`/`q_neg_mean` 是按 4 位小数打印的（`0.1950` 这种），1e-4 的差异在打印精度里消失了。

**结论：`q` 把 256 个 token 的排序做得比随机好（AUROC 0.66），但所有值被挤在 0.1950 ± 1e-4 这条窄带里。**

所以 `TopK(q,32)` **选得是对的**——它确实挑出了稍好的 token——但它递给下游的信号只有 1e-4 的差异，下游（噪声门控、稀疏恢复）对这个量级根本不敏感。`q_mean ≈ 0.1950` 本身就是 `sigmoid(vote_bias)`（`detection_prior: 0.2`），也就是说**上报的 `q` 几乎就是先验，syndrome 项 `H_bar^T s` 小到推不动它**。

## 我的分析：方差在两级被压掉，量级相当

从 `codetrack/ecc.py` 的 forward：

```
s_raw  = MLP([delta, |delta|]) + cos_proj(cos) + residual_scale*||delta||
s_raw  = (s_raw - offset) * gain            # 校准
s      = sigmoid(s_raw)                     # (B, 64)
q_logits = H_bar^T s + vote_bias            # (B, 256)
q      = sigmoid(q_logits)                  # (B, 256)
```

| # | 量 | std | 来源 |
|---|---|---|---|
| 1 | `s_raw` 校准前 | 0.2066 | 本轮日志实测 |
| 2 | `s_raw` 校准后 | ≈1.0（按设计）| 校准增益实测 4.840315，说明**校准生效了** |
| 3 | `s` | **≈0.0097** | **源码注释实测**（ecc.py ~L223：`0.794 ± 0.0097`）|
| 4 | `q_logits` | **≈0.002** | **源码注释实测**（"spread of only 2e-3 over 256 tokens"）|
| 5 | `q` | ≈1e-4 | 本轮日志实测 |

**注意第 3 行**：`0.0097` 远小于"sigmoid 斜率 0.25 × 0.2"给出的直觉值。这说明**64 个检查在校准之前就已经高度相关**——校准是 `(x-offset)*gain` 这种秩-1 缩放，**不能把相关的分量去相关**。所以放大之后有用的方差还是很小。

**两级主要损失，量级相当：**
- **stage 4**：`H_bar^T` 把 0.0097 → 0.002（约 5×）。`H_bar` 是稀疏二值矩阵，每列度数 ~3，每个 token 的 logit 只是 3 个几乎相等的 syndrome 项之和 ⇒ 近似常数。线性映射变不出方差。
- **stage 5**：第二次 `sigmoid`。`q_logits` 中心在 `logit(0.2) = -1.386`，那里 `sigmoid' = 0.16` ⇒ 约 6× 压缩。

**只修一个不够**：修掉一个只能拿到 5~6 倍，1e-4 → 6e-4，还在 1e-3 门槛下面。**两个都要动。**

## 请重点评估的候选方向

| # | 方案 | 打哪一层 | 我的评估 |
|---|---|---|---|
| A | **去掉第二次 sigmoid**，把 `q_logits` 直接当路由分数输出（同时返回 `q` 供兼容）| stage 5 | 最便宜。注意 `l_diag` 本来就用 `binary_cross_entropy_with_logits(q_logits, ...)`，**损失已经是 logit 空间**，只有上报的 `q` 被压过 |
| B | **加宽 `H_bar^T` 映射**：可学连续 `H_soft`（用二值图初始化），或在 `q_logits` 上加一条 `Linear(64→256)` 残差支路 | stage 4 | 直接打另一个 5× 损失。注意 `H_bar` 同时喂 `C_obs`/`C_ref`，所以**残差支路比改 H 本身更安全**（后者会破坏"这是 LDPC 校验"的故事）|
| C | **给 64 个检查去相关**：对 `s` 的 off-diagonal 相关加正则；或让已有的 `check_scale`（`nn.Parameter(torch.ones(num_checks))`）真正区分检查——目前它是被后面的秩-1 `(x-offset)*gain` 覆盖的 | stage 3 | 第 3 行实测 0.0097 说明检查强相关，如果这是 `H_bar^T s` 平掉的根因，这才是**根治**。但要先测 64×64 相关矩阵 |
| D | **soft / straight-through TopK + TopK 外的 ranking/coverage loss**，评估时仍保持 32-token 硬预算 | 路由 | **项目自己的缺陷报告（P2）推荐这个。但我的测量反对优先做它** —— 排序已经对了（AUROC 0.66），TopK 也选得对，缺的是**幅度**。ranking loss 只能让排序更锐，治不了病 |

**我的排序：A 先做（最便宜，解掉 6×），B 作为残差支路（解另一个 5×），C 只在相关性实测支持时才做。D 优先级最低。**

## ⚠️ 请先做这个测量，再提改动

以下量需要**在一个真实 batch 上直接测**，而不是从源码注释引用：

1. `std(s_raw)`（校准前/后）、`std(s)`、`std(q_logits)`、`std(q)`
2. **`s` 的 64×64 相关矩阵** —— 这决定方向 C 是否值得做
3. `H_bar` 的列度/行度分布，以及"每列只有 3 个非零"带来的理论上限
4. `TopK(q,32)` 选中的 token 与全体均值的距离，以及**被选中 token 的实际恢复增益**

第 2 项最关键：如果 64 个检查的相关系数普遍 > 0.9，那方向 C 就是根因；如果它们已经接近独立，那瓶颈就纯粹在 stage 4/5 的映射与 sigmoid。

## 项目红线（请不要违反）

1. **不要改训练策略参数**（lr、epoch、损失权重、阈值）——那些是用户自己的决定。你只提**架构**改动（组件划分、数据流、算子、接口）。
2. **不要直接改仓库代码**——提出方案 + 待执行步骤，由用户自己执行。仓库目录 `repo/` 是只读的上游克隆。
3. **不许在没有 assertion 和具体数字的情况下说"已验证"**（这是项目交接文档 §5 的硬纪律）。
4. 源码里已知两个**未解决**的坑，别当成新发现重复报：
   - `trackit/criteria/builder.py` **不转发 `gain_margin`** ⇒ `CodeTrackCriteria(gain_margin=0.8)` 永远用默认值，config 里也**没有** `gain_margin` 键 ⇒ mixin 也设不了。这挡住了第二轮"扫 gain_margin"的计划。
   - `d_input`/`d_before`/`d_after` 各自经过**不同的 LayerNorm**，而 `_relative_error` 是**余弦距离**（尺度不变）⇒ `d_before == d_input` 可能是纯归一化层尺度变化造成的假象。**norm-only baseline 从未实现**（HANDOFF §4.1/§4.2）。

## 项目已有文档（在服务器上，不在仓库里）

- `_handoff/HANDOFF.md` —— 完整交接，§4.2 是"改代码前先跑的判定探针"
- `训练过程问题分析报告-2026-10-04.md` —— 缺陷修复报告，P0/P1/P2 分级

---

## 一句话总结给外部 AI

**CodeTrack 的 `q`（逐 token 错误概率）排序正确（AUROC 0.66）但幅度死了（std=1e-4，正负样本均值差≈0），导致 TopK 路由递给下游的信号太弱、恢复分支形同虚设。方差在 `H_bar^T` 映射（~5×）和第二次 sigmoid（~6×）两级被压掉，两级都需要修。先测 `s` 的 64×64 相关矩阵，再决定是改映射、去 sigmoid，还是做去相关。**
