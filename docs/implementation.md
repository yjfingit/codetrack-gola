# CodeTrack 实现记录

实现日期：2026-10-04。基线：GOLA `339c737`（`third_party/GOLA`）。

对应架构图：**CodeTrack（改进版）：基于 GOLA 的可纠错时空表征恢复网络（带张量尺寸，DINOv2-B）**。

---

## 1. 模块结构与张量流

```
输入  z/x/d (B,6,112,112) / (B,6,224,224)
      │
      ▼
GOLA 主干 DINOv2 ViT-B/14（冻结）  patch_embed + 12 blocks + norm
      │  F_L (B, 768, 768)  = [z_v, x_v, z_i, x_i, d_v, d_i]
      ▼
切分（codetrack/codetrack.py::_split）
      Z_RGB (B,64,768)  X_RGB (B,256,768)  Z_TIR (B,64,768)
      X_TIR (B,256,768) Z_on (B,64,768)    D_TIR (B,64,768)
      │
      ├─(4)─► KalmanMotionPrior ──► M_t (B,1,16,16), u_t (B,), Mahalanobis (B,)
      │          TemporalMemory (TRC+TGS) ──► prior_tokens (B,8,128), c (B,F)
      │
      ├─(3)─► ParityCheckMatrix H (64,256) + SyndromeDiagnosis
      │          → s (B,64) → q (B,256)
      │
      ├─(5a)► H_RoutedSparseRefiner  TopK(q)=32 → X_rec (B,256,768)
      │
      ├─(5b)► NoiseModulatedDenoiser（2 步）→ X_final (B,256,768)
      │
      └─(6)─► TemplateProtectionGate → c_t (B,)
      ▼
原 GOLA Anchor-Free Head → score_map (B,16,16), boxes (B,16,16,4)
```

### 各节点实测形状（`tools/codetrack_verify.py` 第 4 项）

| 节点 | 实测 | 架构图 |
|---|---|---|
| `score_map` | (B,16,16) | 16×16 ✓ |
| `boxes` | (B,16,16,4) | 4 ✓ |
| `X_t` / recovered | (B,256,768) | 256×768 ✓ |
| `q` | (B,256) | [0,1]^256 ✓ |
| `s` | (B,64) | 64 ✓ |
| `C_obs` / `C_ref` | (B,64,128) | 64×128 ✓ |
| `suspect_index` | (B,32) | K≤32 ✓ |
| `motion_map` | (B,1,16,16) | 16×16 ✓ |
| `uncertainty u_t` | (B,) | ✓ |
| `c_t` | (B,) | [0,1] ✓ |
| **prior_tokens（TGS）** | **(B,8,128)** | 记忆 token ✓ |

### 与架构图的一处**有意偏离**

架构图在 CodeTrack 之前画了一层 **LayerNorm**。上游 `GOLA_DINOv2._fusion` **已经**对 `F_L` 施加了 `self.norm`，而 head 是在该归一化输出上训练的。若再插一层 LN，head 的输入尺度会改变，**step 0 的 `X_t' = X_t` 将不再成立**，加载 GOLA checkpoint 后会立刻掉点。

按既定优先级（**真实 GOLA 约束 > 架构图**），CodeTrack 直接消费已归一化的 `F_L`，不额外加 LN。已在 `codetrack/codetrack.py` 的模块 docstring 中记录。

---

## 2. 新增文件

| 文件 | 内容 |
|---|---|
| `codetrack/config.py` | `CodeTrackConfig`（全部超参 + 校验） |
| `codetrack/ecc.py` | `ParityCheckMatrix`（固定稀疏支撑 + 可学习边权）、`SyndromeDiagnosis`（H → s → q） |
| `codetrack/motion.py` | `KalmanMotionPrior`（常数速度 KF + 空间先验 + Mahalanobis）、`TemporalMemory`（DTPTrack TRC + TGS + 记忆准入） |
| `codetrack/recovery.py` | `H_RoutedSparseRefiner`（H 路由稀疏注意力）、`NoiseModulatedDenoiser`（2 步 noising-denoising）、`MeanVarCompletion` |
| `codetrack/template.py` | `TemplateProtectionGate`（`score>0.84 ∧ c_t>τ_c`） |
| `codetrack/codetrack.py` | 总装：切分 → 运动 → ECC → 恢复 → 去噪 → 门控 |
| `codetrack/corruption.py` | 图像级 + patch-token 级 corruption 模拟（含掩码） |
| `codetrack/criteria.py` | `CodeTrackCriteria`：跟踪损失（corr + clean）+ `L_diag/L_rec/L_align/L_pres/L_mem/L_gate/L_motion` |
| `tools/codetrack_verify.py` | 验证工具：parity / 恒等性 / checkpoint / 形状审计 / 梯度连通 |
| `config/GOLA/dinov2/codetrack_b.yaml` | CodeTrack 分支超参 |
| `config/GOLA/dinov2/codetrack.yaml` | CodeTrack-B 模型配置 |
| `config/GOLA/codetrack_smoke/config.yaml` | 单序列联合训练配置（自包含 `run`，含 codetrack criterion） |
| `config/GOLA/codetrack_eval/config.yaml` | 单序列评测配置（仅 eval task） |
| `scripts/codetrack_train_smoke.sh` | 单序列联合训练入口 |
| `scripts/codetrack_eval_single.sh` | 单序列推理入口 |
| `docs/motion_design.md` | 运动/时序模块的文献依据 |
| `docs/dtp_extraction.md` | DTPTrack 机制提取（含代码↔论文差异） |

---

## 3. 对上游 `trackit/` 的修改（仅 4 个文件，全部为必要改动）

```bash
diff -rq --exclude='__pycache__' --exclude='cache' third_party/GOLA/trackit trackit
```

| 文件 | 改动 | 为什么必要 |
|---|---|---|
| `models/methods/GOLA/gola.py` | ① 构造函数新增可选 `codetrack_config`；② 新增 `_forward_codetrack`（clean/corrupted 双分支 + token 级 corruption + 师生监督）；③ `state_dict` 对 buffer 健壮化 | 挂载 CodeTrack 的唯一接入点；`state_dict` 原实现会对非参数键 `get_parameter` 抛异常 |
| `models/methods/GOLA/builder.py` | 传入 `model_config.get('codetrack')` | 让配置能开关分支 |
| `criteria/builder.py` | 新增 `codetrack` criterion 类型 | 挂载辅助损失 |
| `runner/training/common/optimization/optimizer/per_parameter_options/apply.py` | **修正参数白名单**：补 `codetrack` 前缀，并给原条件加显式括号 | 原式为 `requires_grad and "lora.GA" in name or "lora.GB" in name or ...`，因 `and` 优先级高于 `or`，实际会选中**所有**含 `embed`/`head` 的参数（无视 `requires_grad`）；且 CodeTrack 参数**完全不在 optimizer 里**，训练等于空跑 |

> 除这 4 处外，`trackit/` 与上游逐字节一致；`main.py`、`profile_model.py`、`evaluation.py`、`requirements.txt` 未修改。

---

## 4. Checkpoint 加载

`weights/gola_b224.bin` 为 safetensors，1311 个键，仅含**可训练部分**（LoRA 的 `A/B/GA/GB`、`head`、`token_type_embed`、`lora_alpha`、`use_rslora`）。

```
unexpected keys                  : 0
missing (GOLA, 冻结主干)         : 221     ← 预期，来自 DINOv2 预训练缓存
missing (codetrack, 新增)        : 115     ← CodeTrack 参数，随机初始化
checkpoint 键命中                : 1311 / 1311
```

---

## 5. 参数量

| | 数值 |
|---|---|
| 总参数 | 106.37 M |
| 可训练参数 | 20.65 M |
| **CodeTrack 新增** | **10.03 M** |
| GOLA 侧（冻结主干） | 98.71 M 中约 85.7 M 冻结 |

> 架构图参考值给的是 "CodeTrack 新增 ~2–3M"。实测 10.03M 偏大，主要来自 `NoiseModulatedDenoiser` 的 768→256 交叉注意力（26 个张量）与 `H_RoutedSparseRefiner` 的稀疏注意力。这是**实现与图上估计的差异，如实报告**；若需压缩，首选减小 `diffusion_hidden` 或去掉 denoiser 的 FFN。

---

## 6. 梯度连通性（`tools/codetrack_verify.py` 第 5 项，真实 criterion 反传）

```
module                      grad-norm  params
H (parity check)           2.1248e-07  1
diagnosis (syndrome/q)     8.7023e-05  17
template_pool              2.1014e-05  4
refiner (H-routed)         1.0587e-03  19
denoiser (diffusion)       3.0414e-02  26
meanvar completion         9.2539e-01  4
motion prior (Kalman)      6.7377e-03  7
temporal memory            1.8706e-05  11
template gate              2.9158e-01  6
condition_proj             1.6128e-03  2
→ every CodeTrack module received a non-zero gradient
```

### 关键陷阱：零初始化输出层会饿死整个分支

恢复分支的输出层**不能零初始化**。若 `up.weight = 0` 则 `pred = 0`，残差以 `gate * pred` 注入，梯度 `∂/∂上游 ∝ up.weight = 0`，**运动先验、时序记忆、条件投影全部拿不到梯度**（名义连通、实际训不动）。

采用**小而非零**的残差门 `sigmoid(-8) ≈ 3.4e-4`：

| 残差门 | 输出扰动 | 到达时序记忆的梯度 |
|---|---:|---:|
| −12（3.1e-6） | ~1e-6 | 3.1e-07 |
| **−8（3.4e-4）** | **~2.4e-6** | **1.9e-05** |

---

## 7. 单序列联合训练

```bash
bash scripts/codetrack_train_smoke.sh     # 序列 10crosswhite，batch 2，6 次迭代
```

| 迭代 | loss | grad_norm | cls | box | rec | align | mem | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 22.84 | 34.14 | 4.96 | 0.99 | 0.00 | 3.31 | 0.38 | 1.29 |
| 1 | 20.21 | 19.85 | 3.84 | 0.74 | 0.23 | 3.07 | 0.76 | 0.81 |
| 2 | **19.95** | 26.00 | 4.35 | 0.74 | 0.00 | 3.01 | 0.38 | 1.29 |

- **无 NaN / Inf**；`grad_norm` 有限（19.8–34.1）；AMP loss scale 稳定在 65536。
- 训练后保存 1408 个张量（codetrack 97 个）；**689 个 GOLA adapter/head 张量亦被更新** → 确为**联合微调**而非只训新模块。

### 训练期修复的一个数值问题

两路前向（clean teacher + corrupted student）在 fp16 下会使 GOLA 侧激活溢出，导致 **LoRA/head 出现 inf 梯度**（`grad_norm: inf`、loss scale 连续减半）。修法：**冻结主干与 head 的前向在 fp32 下计算**（`torch.autocast(enabled=False)`）。修复后 `grad_norm` 由 inf 变为有限值，loss scale 不再下降。

---

## 8. 单序列推理（官方 eval pipeline）

```bash
bash scripts/codetrack_eval_single.sh     # 序列 10runone
```

| 指标 | 数值 |
|---|---:|
| success score | 0.7967 |
| precision | 1.0000 |
| norm precision | 0.9934 |
| success rate @ IoU 0.5 | 1.0000 |
| success rate @ IoU 0.75 | 0.8158 |
| **FPS** | **52.95** |

评测走官方 `OneStreamTracker_Evaluation_MainPipeline`（含模板更新、窗口惩罚 0.45），未自写推理循环。

---

## 9. 仍未完成 / 后续建议

1. **运动先验监督在单序列上退化为 0**（`Loss/motion = 0`）。原因：单序列的 6 次迭代里，Kalman 在观测到 GT 框后预测出的高斯与监督目标**按构造相同**，KL 恒为 0。需要在真正的连续序列训练（逐帧递推）中才能体现其价值，届时该损失才非退化。
2. **MoKA-HP 完全闭源**（Unpaywall/OpenAlex/S2 均无开放版本，无摘要，无代码）。本实现**不对其 KFMM 内部做任何假设**。若拿到原文，值得对照三点：Kalman 注入方式、协方差用法、historical prompts 的缓冲与更新规则。
3. **CodeTrack 参数量 10.03M** 高于架构图估计的 2–3M，主要来自 denoiser 的交叉注意力与 FFN。若要贴近图上的开销目标，先削减 `diffusion_hidden`。
4. **`h_locality_window`/`num_checks` 等仅做过形状与梯度验证**，未做性能消融；`M=64`、`K=32`、`K_n=8` 均为架构图取值。
5. 下一步：用完整 LasHeR 训练集跑 Stage 1/2（配置已就绪，把 `codetrack_smoke/config.yaml` 的数据源换回完整 train split 即可），再按 `docs/CodeTrack-实验文档.md` 的 ablation 树推进。
