# 训练期数据流审计

用 `tools/dataflow_audit.py` 在**真实训练步**（batch=2，含 corruption，反向传播）上用 hook 抓取全部模块边界张量，再用 `tools/interaction_matrix.py` 归类。**不是读代码推断**——下面是实测输出。

复现：

```bash
export LD_LIBRARY_PATH=/root/autodl-tmp/lab/tools
python tools/dataflow_audit.py        # 抓取 + 输出 /tmp/dataflow_grads.json
python tools/interaction_matrix.py    # 归类成矩阵与梯度判定
```

---

## 1. 模块交互矩阵（实测）

| 生产者 | 消费者 | 张量 | 说明 |
|---|---|---|---|
| KalmanMotionPrior | TemporalMemory | `uncertainty (B,)` | `u_t` |
| KalmanMotionPrior | TemporalMemory | `mahalanobis (B,)` | `sqrt(νᵀS⁻¹ν)` |
| KalmanMotionPrior | H_RoutedSparseRefiner | `motion_map (B,1,16,16)` | **作为注意力对数偏置** |
| KalmanMotionPrior | NoiseModulatedDenoiser | `motion (B,2)` | `[mean(map), u_t]` |
| KalmanMotionPrior | TemplateProtectionGate | `uncertainty (B,)` | 门控输入之一 |
| CodeTrack._split | SyndromeDiagnosis | `X_t (B,256,768)` | TIR-position 搜索 token |
| CodeTrack._split | SyndromeDiagnosis | `X_aux (B,256,768)` | RGB-position 搜索 token |
| CodeTrack.template_pool | SyndromeDiagnosis | `template_context (B,768)` | 4 段模板池化 |
| ParityCheckMatrix | SyndromeDiagnosis | `H_bar (64,256)` | 行归一化、可学习边权 |
| SyndromeDiagnosis | TemporalMemory | `reliability = 1−q` | **detach** |
| SyndromeDiagnosis | H_RoutedSparseRefiner | `q (B,256)` | **既选 TopK、又作残差门控** |
| SyndromeDiagnosis | NoiseModulatedDenoiser | `syndrome (B,64)` | 帧级条件 |
| SyndromeDiagnosis | TemplateProtectionGate | `mean(q) (B,)` | 污染摘要 |
| TemporalMemory | H_RoutedSparseRefiner | `prior_tokens (B,8,128)` | TGS 输出 |
| TemporalMemory | NoiseModulatedDenoiser | `memory (B,128)` | prior tokens 按帧均值 |
| TemporalMemory | criterion | `trc_confidence (B,3)` | 受 `L_trc` 监督 |
| TemporalMemory | criterion | `frame_reliability (B,1)` | 受 `L_mem` 监督 |
| ParityCheckMatrix | H_RoutedSparseRefiner | `neighbour_index (256,8)` | `HᵀH` 共享校验几何 |
| CodeTrack.template_pool | H_RoutedSparseRefiner | `template_pool (B,768)` | 每 suspect 条件 |
| CodeTrack._split | H_RoutedSparseRefiner | `X_aux (B,256,768)` | 跨模态证据 |
| H_RoutedSparseRefiner | NoiseModulatedDenoiser | `X_rec (B,256,768)` | 精修 token 作去噪输入 |
| H_RoutedSparseRefiner | CodeTrack.condition_proj | `context (B,32,256)` | 散布到 suspect 行 |
| NoiseModulatedDenoiser | GOLA head | `X_final (B,256,768)` | **主路径** |
| NoiseModulatedDenoiser | TemplateProtectionGate | `recovery_confidence (B,)` | `1−cos(X_final,X_t)` |
| NoiseModulatedDenoiser | MeanVarCompletion | `X_denoised` | → `L_align` |
| GOLA head (corrupted) | criterion | `score_map/boxes` | `L_track^corr` 主监督 |
| GOLA head (clean) | criterion | `score_map/boxes` | `L_track^clean` |
| GOLA trunk (clean) | criterion | `X_clean` **DETACHED** | 诊断目标 |

**拓扑是分层的、无环的**：`运动 → {诊断, 记忆} → 路由恢复 → 迭代去噪 → 门控 → head`。没有跳过中间层直连 head 的旁路，也没有把 ground-truth 框直接注入 head 的路径（`motion_target` 只进 `L_motion`）。

---

## 2. 梯度实测：哪些边真的在训练？

| 接口张量 | \|grad\| | 判定 | 角色 |
|---|---:|---|---|
| `H_RoutedSparseRefiner.motion_map` | 2.263e+00 | training | 运动作注意力偏置（**最强**） |
| `TemplateProtectionGate.c_t` | 2.510e-01 | training | 模板门控 |
| `TemporalMemory.trc_confidence` | 9.835e-02 | training | TRC 可靠性（**修复后**） |
| `H_RoutedSparseRefiner.X_rec` | 5.217e-02 | training | 精修 token |
| `NoiseModulatedDenoiser.X_denoised` | 5.217e-02 | training | **主路径** |
| `MeanVarCompletion.pred_mean` | 1.650e-02 | training | `L_align` |
| `H_RoutedSparseRefiner.delta` | 4.849e-03 | training | 恢复残差 dX |
| `KalmanMotionPrior.uncertainty` | 1.264e-03 | training | `u_t` |
| `SyndromeDiagnosis.q` | 2.416e-05 | training | 误差严重度 |
| `TemporalMemory.prior_tokens` | 2.190e-06 | training | TGS 输出 |
| `SyndromeDiagnosis.s` | 4.269e-06 | training | syndrome |
| `SyndromeDiagnosis.H_bar` | 4.616e-05 | training | **H 边权确实在学习** |
| `KalmanMotionPrior.motion_map` | 8.682e-07 | weak | 先验图 |
| `SyndromeDiagnosis.C_obs` / `C_ref` | ~1.5e-07 | weak | check 节点 |
| `H_RoutedSparseRefiner.context` | 9.905e-07 | weak | 路由上下文 → 去噪条件 |
| `NoiseModulatedDenoiser.condition` | 9.747e-07 | weak | 去噪条件 |
| `H_RoutedSparseRefiner.suspect_score` | 6.108e-09 | weak | TopK 严重度 |

**DEAD 接口：无。** 每个模块至少有一条可达梯度路径。

---

## 3. 审计中发现并修复的两个真实缺陷

### 缺陷 1：推理时运动先验恒为 0（已修）

`_ensure_state` 把卡尔曼状态初始化为**全零**（框在图像原点、尺寸为 0），只在 `observe()` 被调用时才种子化——而**推理路径不传框**，所以状态永不种子化。实测：

```
修复前（eval，不传 gt_box）:
  _x = [0,0,0,0,0,0,0,0]
  motion_map max = 0.000e+00   ← 恒零
  -> attn_bias = log(1e-4)*0.5 ≡ -4.6  ← 常数，注意力对候选邻居毫无区分度
  -> u_t ≡ 2.85 常数 → TRC 门控与模板门控的该输入永不变化
修复后:
  _x = [0.5, 0.5, 0.3, 0.3, 0, 0, 0, 0]   ← 图像中心 + 中性尺度
  motion_map max = 4.946e-01
```

**修法**：新增 `_seed_unobserved()`，无观测时种子到图像中心（`cx=cy=0.5`）与默认尺度 `default_seed_side=0.3`，零速度，协方差保持 `P0`（所以 `u_t` 偏高，先验被当作弱证据——这是"还没看到目标"的诚实编码）。

> **仍未解决**：推理时状态**不更新**（因为没有观测），所以 `M_t` 逐帧不变。要真正用起来，需要把跟踪器自己的输出回灌为观测，而官方 eval pipeline 的 `post_process` 只接图像、不接模型。这需要改评测管线，属于下一步工作（见 §5）。

### 缺陷 2：TRC 门控是"有架构、没训练"（已修）

实测 `trc_gate` 的**全部 4 个张量梯度为 0**。根因：

- 门控输出只通过 `calibrated = mem · c` 影响 `prior_tokens`，而 `prior_tokens` 只有跟踪损失到达；
- 但 `tgs_modulator` **最后一层零初始化**，早期把该路径的梯度压到极小；
- 更根本的是**没有任何损失直接监督 `c_i` 该是多少**。

DTPTrack 不需要显式监督，是因为它把损失**在 4 个 per-frame aux head 上求和**，每个历史帧的表示都有直接任务梯度。我们只有一个 head，没有那条路径。

**修法**：新增 `L_trc`，用与 `L_mem` 相同的"该帧是否可信"信号直接监督 `c_i`（排除被锚定为 1.0 的 GT 槽位）。

```
修复前: TemporalMemory.trc_confidence  |grad| = 0.000e+00   ← DEAD
修复后: TemporalMemory.trc_confidence  |grad| = 9.835e-02   Loss/trc = 0.6757
```

---

## 4. "是否合理"——逐项判断

### ✅ 合理的部分

**1. 分层无环拓扑，符合架构图。** 运动/诊断 → 记忆 → 路由恢复 → 迭代去噪 → 门控 → head。没有旁路。

**2. `q` 的三重角色是一致的，不是耦合错误。**
`q` 同时用于 ①选 TopK ②作残差写回门控 ③作为 `L_diag` 目标。这三者共享同一个语义（"该 token 有多坏"），所以共享张量是正确的设计；而且它让 `L_diag` 与主跟踪损失通过同一条路径回传，避免诊断模块只被辅助损失孤立训练。

**3. 运动是"特征"而非"输出"，与 DTPTrack Table 6 的教训一致。**
实测 `H_RoutedSparseRefiner.motion_map` 梯度 2.26（**全表最强**）——运动确实在影响路由决策，而且是通过注意力偏置这种"软"方式。若改成"用 Kalman 预测框直接引导"，就会踩 DTPTrack momentum(73.8)/optical flow(73.2) vs 学习式 TGS(74.3) 的坑。

**4. 挡漏的三处 detach 都正确。**
`X_clean`、记忆库、`reliability=1−q` 均为 forward-only。特别是 `X_clean` detach 后，teacher 只作目标、不接收学生梯度——这是 clean-teacher residual 方案的正确实现。

**5. 跨模态证据路径存在且合理。**
`X_aux`（RGB-position）同时供给诊断（作 `C_ref` 的参考）与恢复（作每 suspect 的跨模态证据）。两条用途都符合"另一模态提供缺失证据"的定位。

### ⚠️ 需要你注意的三点

**1. 若干"设计上重要"的通路梯度很弱（1e-6 ~ 1e-7）。**

`context → denoiser.condition`（9.9e-7）、`syndrome`（4.3e-6）、`prior_tokens`（2.2e-6）。不是零，但比主路径弱 4–5 个数量级。原因是主路径的写入量被 `sigmoid(−8)=3.4e-4` 压低——这是为保初始化恒等性付的代价。这些模块主要靠各自的辅助损失（`L_diag`/`L_rec`/`L_align`/`L_trc`）训练，而非主损失。

**这不一定是问题**（辅助损失是我们的设计），但它意味着：**这些模块的最终效果高度依赖辅助损失的权重是否配得对**。消融时建议单独看每条的贡献。

**2. `Loss/rec` 恒为 0。**

`L_rec` 只在 suspect token 上算 `1−cos(X', X*)`，而 `X' = X + σ(−8)·σ(q)·dX ≈ X`，所以恢复量与清洁特征几乎重合 → 余弦差 ≈ 0。修复缺陷 1 时也顺带发现：**`L_rec` 实际并没有在推动恢复模块**——真正在推动它的是 `L_align`（3.62）和主跟踪损失。

这意味着当前实现里**"恢复质量"没有被有效监督**。两个选择：
- 把 `L_rec` 改成在**任务层面**度量（用 head 输出 vs teacher 输出），但 head 的输出路径也带 `σ(−8)`，梯度同样弱；
- 或先在 Stage 1（冻结 GOLA）单独把 `L_rec` 的梯度路径打通，再进入 Stage 2。

**建议先跑消融 `lambda_rec ∈ {0, 0.2, 0.5}`**，如果它对 SR 没有影响，说明这条通路确实无效，应该删掉而不是保留一个恒零的损失项（方案文档本身也要求：不为故事保留无效模块）。

**3. 运动先验在推理时逐帧不变（缺陷 1 的残留）。**

已修种子化，但状态不更新。当前它退化为一个**固定的、居中的弱先验**。在 Stage 4（时序扩展，改 causal clip 训练）之前，它不会成为真正的"运动"先验。这也是 `Loss/motion` 在单序列上恒为 0 的同一根源。

### ❌ 一个我不认为合理的点

**记忆库的时间衰减（`decay=0.6`）与 TRC 门控职责重叠。**
DTPTrack 只靠 TRC 门控决定历史帧权重，没有额外的时间衰减。我加了 `decay` 作为补充，但这意味着**同一件事被两个机制管**：门控决定"信不信"，衰减决定"久不久"。当两者冲突时（远期的可靠帧 vs 近期的不可靠帧），行为不明确。建议消融 `decay ∈ {1.0（去掉）, 0.6}`，若 `decay=1.0` 不差就删掉它。

---

## 5. 下一步建议（按优先级）

| 优先级 | 动作 | 依据 |
|---|---|---|
| **P0** | 消融 `lambda_rec ∈ {0, 0.2, 0.5}`，判定 `L_rec` 是否值得保留 | `Loss/rec` 实测恒为 0，可能是个无效损失项 |
| **P0** | 接上图像级 corruption（collator 插件） | 当前是死代码（见 `hyperparameters_reference.md` §13） |
| **P1** | 把跟踪器输出回灌为卡尔曼观测（改评测管线 `post_process` 或在那之前插 hook） | 否则运动先验在推理时永不更新 |
| **P1** | 连续序列（causal clip）训练 | `L_motion` 恒零；记忆的时序语义需要真序列 |
| **P2** | 消融 `decay ∈ {1.0, 0.6}` | 见上文"不合理的点" |
| **P2** | 给弱梯度通路（`context`/`syndrome`/`prior_tokens`）做单独消融 | 确认它们靠辅助损失能被训好 |

---

## 附：实测的接口形状（训练步，batch=2）

```
KalmanMotionPrior      in  box_xywh (2,4)              out motion_map (2,1,16,16), uncertainty (2,)
                                                          innovation (2,), mahalanobis (2,), state (2,8)
SyndromeDiagnosis      in  X_t (2,256,768), X_aux (2,256,768), H_bar (64,256), template_ctx (2,768)
                       out q (2,256), s (2,64), C_obs (2,64,128), C_ref (2,64,128), U/R (2,256,128)
TemporalMemory         in  tokens (2,256,768), reliability (2,256), target_mask (2,256),
                           uncertainty (2,), mahalanobis (2,)
                       out memory (2,3,128), prior_tokens (2,8,128), trc_confidence (2,3),
                           frame_reliability (2,1), readout (2,128)
H_RoutedSparseRefiner  in  X_t (2,256,768), X_aux (2,256,768), q (2,256), neighbour_index (256,8),
                           template_pool (2,768), memory_readout (2,8,128), motion_map (2,1,16,16)
                       out X_rec (2,256,768), delta (2,32,768), suspect_index (2,32),
                           attn (2,4,32,8), context (2,32,256)
NoiseModulatedDenoiser in  tokens (2,256,768), condition (2,256,768), syndrome (2,64),
                           motion (2,2), memory (2,128), alpha (2,256,1)
                       out X_denoised (2,256,768)
MeanVarCompletion      in  tokens (2,256,768)          out pred_mean/pred_logvar (2,768)
TemplateProtectionGate in  score (2,), q (2,), uncertainty (2,), recovery_conf (2,)
                       out c_t (2,)
```

**全部与架构图标注一致。**
