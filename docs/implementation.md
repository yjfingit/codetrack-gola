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

---

## 10. 训练配方落地（第二轮：因果性修复）

外部评审指出 8 个会让模块"学不到东西"的问题。我逐条**实测核实**后修了其中 3 个（D1–D3），
另有 2 个（D4–D5）已定位待修。全部结论附实测数字。

### D1｜LoRA 从未训练（已修，致命）

**核实**：`gola.py` 的 `with torch.no_grad(), ...` 包住了整个 student trunk。5 步 AdamW 后：

```
LoRA (blocks)  相对位移=0.000e+00   移动 0/1296      ← 完全没进计算图
head           相对位移=2.629e-02   移动 12/12
CodeTrack      相对位移=8.165e-03   移动 97/97
```

不是"梯度弱"，是**根本没有梯度** —— LoRA 参数从未进入 autograd 图，S2 的"联合 PEFT"是假的。

**修法**：student trunk 改为建图，只靠 `requires_grad=False` 冻结 DINOv2 base；clean teacher 独立 `no_grad`。
原理：LoRA 增量是 `base(x) + lora_A(x) @ lora_B(x)` 的**加法**，保留图即可让增量拿到梯度而 base 恒定。

**修复后**（`tools/lora_grad_check.py`）：

```
LoRA (blocks)   相对位移=1.885e-02   移动 1296/1296   梯度为0: 0/1296
frozen params that changed value: 0                                  ← 主干严格冻结
PASS: LoRA trains, backbone stays frozen
```

### D2｜当前帧 GT 进入当前帧运动先验（已修，因果泄漏）

**核实**：`motion.py` 先 `observe(gt_t)` → 再预测 → 再建 `M_t`。所以 `M_t` 是当前帧 GT 的函数，
即"用它本该帮助预测的那一帧的答案"去构造先验。

**修法**：拆成 `predict → M_t → 消费 → observe`。
- `KalmanMotionPrior.forward()` 新增 `defer_observe`，为 True 时只预测不更新
- `CodeTrack.forward()` 在**所有消费方跑完之后**（head 之后）才调用 `observe()`，使状态为下一帧就绪

### D3｜运动先验头的学习幅度被归一化抵消（已修）

**核实**：`prior_map = exp(-0.5 sq) * sigmoid(logits)`，而消费方随后 `pm / pm.sum()` ——
**乘性幅度在归一化后完全消失**，该头拿不到有效空间梯度。

**修法**：prior head 从输出 1 维幅度改为输出 5 维几何量 `[dcx, dcy, dlog w, dlog h, log T]`，
在 Gaussian **之前**施加，因此改变的是先验的**形状**而非幅度。末层零初始化 → step 0 仍为解析高斯。

### 验证结果

`tools/causality_check.py`（新增）**6/6 通过**：

| 检查 | 结果 |
|---|---|
| C1 `M_t` 与当前帧 GT 无关 | PASS `max\|dM_t\| = 0.000e+00` |
| C2 `motion_box` 与当前帧 GT 无关 | PASS `max\|dbox\| = 0.000e+00` |
| C3 refiner 注意力偏置与当前帧 GT 无关 | PASS `max\|dbias\| = 0.000e+00` |
| C4 观测**确实**更新状态（保证 C1 非空洞） | PASS `max\|dx\| = 1.539e-02` |
| C5 历史会影响当前先验（时序信息被使用） | PASS `max\|dM_t\| = 4.982e-01` |
| C6 扰动 prior 头会改变**归一化后**的图 | PASS `max\|dM_t\| = 4.854e-01` |

**`L_motion` 从恒零变为非零**（逐帧：0.214 / 1.014 / 0.411 / 0.152 / 0.047）——
这正是 D2+D3 的直接目标。

`tools/codetrack_verify.py` 仍全绿：恒等性 `2.53e-4`、checkpoint 1311/1311、10/10 模块非零梯度。
训练冒烟无 NaN/Inf，LoRA 更新 **1152/1298**（修复前 0/1296）。

---

## 11. 已定位、尚未修完

**D4｜时序可靠性被 sigmoid 压在 0.5–0.73**（已核实，未修）
`motion.py:447` `cur_rel = sigmoid(reliability.mean(-1))`，而 `reliability = 1-q ∈ [0,1]`，
sigmoid 后值域仅 **0.50–0.73**，该模块**无法表达"不可信"**。
计划：改用软标签 `r* = clip(0.7·IoU + 0.3·(1−e_task), 0, 1)` 做 stop-grad 监督。

**D5｜历史帧丢失各自的标签 + 跨帧梯度被切断**（已核实，未修）
- `criteria.py` `tgt = trust.expand_as(c_dyn)` —— **当前帧 trust 广播给全部历史 slot**
- `motion.py` `memory_out = mem.detach()` —— 历史对未来 loss 无梯度
计划：记忆库随内容存**每帧自己的**标签；S3 起启用 TBPTT=3。

**D2 的 memory 部分不成立**（如实记录与评审意见的分歧）：
评审称"memory 用当前帧 GT mask 池化当前帧再参与当前帧 recovery"是因果泄漏。
我核查了执行顺序 —— `codetrack.py` 第 322 行读**旧**记忆，第 328 行才写**新**记忆，
**读写顺序本身是正确的**。`target_mask` 只作池化掩码，不进入 head。

**`L_rec` 恒为 0**（仍未解决）
`Loss/rec` 在冒烟与单序列上均为 `0.0000`。已按计划准备好强制单元实验（`corruption_prob=1.0`
下比较 `d_before = d(X_cor,X_clean)` 与 `d_after = d(X_final,X_clean)`）来区分
"wiring bug" 与 "loss 无效"，**尚未执行**。

**`L_motion` 在真实训练数据上可能仍未生效**
冒烟日志中**没有** `Loss/motion` 这一项（criterion 只在非零时记录）。独立构造输入时它确实非零，
说明真实训练路径下 `motion_target` 可能为 `None` —— 即训练数据没有向模型提供逐帧 `gt_box`。
**这一点尚未确认**，是下一步首先要查的。

---

## 12. 尚未落地的配方部分

四阶段训练（S1 预热 / S2 空间联合 / S3 causal clip / S4 混合精调）、分组学习率、
causal-clip 采样器、`L_gain`/`L_rel`/`L_mem_pred` 三项新损失，均**已设计完成并写入方案**
（见 `docs/hyperparameters_reference.md` 与提交说明），但**除 `L_gain` 的配置项外尚未实现**。

**已知不可实现项**：训练时 search crop 由 `SiamFCCropping` 基于数据集 GT 框离线裁剪，
模型无法反向影响裁剪中心，因此"让下一帧 crop 以预测框为中心"在当前架构下做不到，
只能靠 scheduled sampling 部分缓解 exposure bias。

---

## 13. 全量 LasHeR 训练的准入检查（2026-10-04）

`LasHeR` 训练集共 **979 个序列**。下列每一项都实测过。

### 硬阻塞项：`L_motion` 在真实训练下永远不计算

**根因**（已确认，非猜测）：

```python
# trackit/runner/training/default/model_wrapper.py
def forward(self, samples, targets):
    output = auto_unpack_and_call(samples, self.model)   # 只传 samples
    output = self.criterion(output, targets)             # targets 只给 criterion
```

`auto_unpack_and_call(samples, ...)` 把 `samples` 展开成 `**kwargs`。而 `gt_box` 并不在
`samples` 里 —— 它在 `targets['boxes']`（由 `box_with_score_map_label_collator` 写入，
归一化 `cxcywh`）：

```python
collated.target.update({'num_positive_samples': ..., 'boxes': collated_gt_bboxes})
```

所以模型 forward 收到 `gt_box=None` → `motion_target=None` → criterion 的
`if mp is not None and mt is not None` 不成立 → **`Loss/motion` 从不出现**。
这与冒烟日志里看不到该项完全一致。

**修法（已设计，未实现）**：把 `L_motion` 的目标构造从模型侧搬到 **criterion 侧**，
因为 criterion 能同时看到模型输出与 `targets`：

- `CodeTrack` 返回**未归一化**的 `motion_map`（问题：旧代码在模型内 `pm/pm.sum()` 并写回，
  若已在模型内归一化，注意别二次归一化）
- criterion 从 `targets['boxes']`（归一化 `cxcywh`）直接构造 GT 先验图，**无需 `image_size`**
  （因为先验图定义在归一化图像坐标上，与像素尺寸无关）
- 在 criterion 里算 KL

### 已就绪

| 项 | 状态 |
|---|---|
| LoRA 可训练 | **1296/1296 更新**，主干 0 漂移 |
| 运动先验因果性 | `tools/causality_check.py` **6/6** |
| `M_t` 摆脱恒零 | 独立输入下 `L_motion` = 0.214/1.014/0.411/0.152/0.047 |
| 显存 | micro_batch=8 峰值 **4.40 GiB / 23.5 GiB**；batch 128 需 `accumulation=16` |
| 数据集 | 979 序列可读 |

### 未就绪

| 项 | 为什么阻塞 |
|---|---|
| `L_motion` | 见上，硬阻塞（运动头拿不到监督） |
| 分组学习率 | 实测有明确收益：20 步后末次 loss **2.395（LoRA 2.5e-5）vs 4.817（LoRA 1e-4）**；两者都不发散，所以这是"跑得好"而非"跑得了" |
| 全量配置 | 现有配置 `samples_per_epoch=6, global_batch_size=2, num_epochs=1`，是为冒烟而设 |
| `L_rec` = 0 | 未解决，恢复质量仍无监督 |
| D4 / D5 | 时序可靠性被压在 0.50–0.73；历史 slot 共享当前帧标签 |
| causal-clip 采样器 | 未实现 → S3 无法开工 |

### 判断

**当前可以跑通全量训练，但不应该** —— 运行会得到"恢复/诊断分支被训练、运动分支完全没被监督"
的半成品，浪费算力且结论不可用。**先修 `L_motion` 进 criterion + 建全量配置 + 分组 LR**，
再去跑 979 序列。

---

## 14. 训练基础设施修复（第三轮：外部审查后）

外部审查判定 **NO-GO**，并指出 5 个会让"训练正常结束但学到的东西不对"的问题。逐条实测核实后**全部为真**，
已修复。另有一个是**我自己上一轮引入的回归**。

### P0-1｜scheduler 时间轴快 16 倍（已修）

`scheduler builder` 内部已做 `num_updates_per_epoch = num_iterations_per_epoch // grad_accumulation_steps`，
并据此推出 `warmup_t=2048`、`t_initial=10240`。但 runner 传的是 **micro-step 计数**：

```python
self._lr_scheduler_per_iteration.step_update(self._iteration)   # 修复前
```

后果：2 个 epoch 的 warmup 在 **128** 步就结束（应 2048），10 epoch 的 cosine 在 **640** 步
（≈0.625 epoch）就跑完，其余训练全部趴在 `lr_min`。已改为传 optimizer update 计数：

```python
optimizer_step = (self._iteration + 1) // self._grad_accumulation_steps
```

### P0-2｜梯度累积是"求和"而非"平均"（已修）

```python
loss_scale, grad_norm = self._parameter_updater.backward_and_unscale(
    criterion_output.loss + orth_loss, ...)     # 修复前：未除以累积步数
```

后果：每个梯度被放大 16 倍，等效学习率 16 倍，且 pre-clip 梯度范数虚高——这解释了此前
`grad_norm = 517` 对上 `max_grad_norm = 1.0` 的异常。已改为 `... / self._grad_accumulation_steps`。

### P0-3｜`L_diag` 实际上从未计算（已修，影响架构主张）

criterion 的诊断分支需要 `error_target` 与 `syndrome_target`，而**模型中从不生成这两个键**，
所以该分支永不进入。实测：训练日志里**从未出现 `Loss/diag`**。
这意味着**诊断头完全没有显式监督**，只能靠跟踪/恢复的间接梯度——对"ECC syndrome 能定位坏 token"
这一核心主张远远不够。

**修法**：目标由 **clean 与 corrupted teacher 特征的真实差异**导出，而非直接使用注入器的
`token_mask`（后者会让诊断头学会"读标签"，且无法迁移到 LasHeR 天然含有的退化）：

```
e_feat = 1 - cos(X_tir_corrupted, X_tir_clean)        # 表征侧偏差
e_task = 0.5 * (e_feat + e_aux)                       # 跨模态一致性
e*     = alpha*e_feat + (1-alpha)*e_task              # alpha=0.5
s*     = H_bar @ e*                                   # syndrome 目标
```

实测：**`Loss/diag = 0.55`**，`error_target (B,256)`、`syndrome_target (B,64)`，此前完全不存在。

### P0-4｜Denoiser 写回方向反了（已修）

```python
alpha = 1.0 - q                 # alpha = trust
eps   = eps * (1.0 - alpha)     # = q     -> 坏 token 加噪   ✓
x     = x + w * alpha * pred    # 按 trust 写回 -> 健康 token 修得多，坏 token 修得少  ✗
```

与"选择性恢复"**完全相反**。这是**我上一轮引入的**：原先写回无门控（均匀施加），我为它加上
`alpha` 门控时用了错误的语义。已改为按**错误概率**门控，并拆成 `token_error` / `token_trust`
两个名字以防再犯。实测：坏 token 改动是健康 token 的 **91.8×**。

### P0-5｜S1/S2 必须关闭 temporal（已修）

pair sampler 在同一 track 上**独立随机抽帧**，且训练 runner **不按批次 reset `reset_sequence()`**。
若 motion/memory 开启，递推状态会把序列 A 的历史带进序列 K 的当前帧——**假时序**。
已新建 `config/GOLA/dinov2/codetrack_spatial.yaml`（`motion/memory/template_protection` 全 false）
与 `config/GOLA/codetrack_s2/`，`codetrack_full` 保持全开供 S3/S4 使用。

### P1-6｜我自己引入的 weight-decay 回归（已修）

**审查在此纠正了我一个错误判断，我认为它是对的。** optimizer builder 里：

```python
optimizer_parameters = {'lr': lr, 'weight_decay': weight_decay}
optimizer = optimizer_cls(optimizer_param_groups, **optimizer_parameters)
```

构造函数**显式传了 `lr`**，所以没有 `lr` 的 param group 继承的是**基座 `1e-4`**，
而**不是** AdamW 签名默认的 `1e-3`。我上一轮"1e-3"的判断不成立。

更严重的是，我把分组规则提到 `zero_1d` 之前，导致 1 维参数（bias/norm）被前面的正则组吃掉，
**丢掉了 `weight_decay=0` 的语义**。

**修法**：`zero_1d_param_weight_decay` 的语义是"收集非衰减参数（ndim≤1 或 norm 层）并给 wd=0"，
且**自身不带 lr**。所以按 LR 分组各写一条、并加 `name_regex` 限定作用域，就能同时满足两者：

```yaml
- type: "zero_1d_param_weight_decay"
  name_regex: 'codetrack\.'
  lr: 1.e-4
- name_regex: 'codetrack\.'
  lr: 1.e-4
...
- type: "zero_1d_param_weight_decay"    # 兜底
```

实测 6 组、**1406/1406 张量全覆盖**、1 维参数 `wd=0` 且 lr 正确。

### P0-7｜`lora.A` / `lora.B` 从未被优化器更新（新发现，已修）

**这一条是我在修 6 时顺带发现的，比审查列出的问题更严重。**

优化器白名单原为 `("lora.GA", "lora.GB", "embed", "head", "codetrack")`。
而 `LoRALayer` 的实际参数名有两类：

| 名称 | 数量 | 前向是否使用 | 是否在白名单 |
|---|---:|---|---|
| `.lora.A` / `.lora.B` | **144** | **是**（`LoRALayer.forward` 用的就是它们） | **否** ❌ |
| `.lora.GA.{0..7}` / `.lora.GB.{0..7}` | 1152 | 是 | 是 |

实测确认**全部 1296 个 LoRA 张量都拿到梯度**，即 144 个真正参与前向的基座参数
**被静默排除在优化器之外**。已把 `lora.A`/`lora.B` 加入白名单，覆盖率从 **1262/1406 → 1406/1406**。

### P1-8｜图像级 corruption 污染了 clean teacher（部分修复）

数据 plugin 直接 `collated.input["x"] = corrupted_x`，**没有保留 clean 版本**。
所以对 image-level corruption 样本，模型里的所谓 teacher 拿到的仍是**被破坏的图像**——
`student = corrupted` 与 `teacher = corrupted` 相同，师生残差失去意义。

已修的部分：`image_corruption_mask` 现在并入 `was_corrupted`（此前图像级破坏被报成"可信帧"，
使记忆可靠性与模板门控都拿到错误标签）。
**未修**：保留 `x_clean` 需要改数据 plugin 的接口，留待 S1 前处理。

### 新增验收门

`tools/preflight_acceptance.py` —— **26/26 通过**，覆盖：优化器覆盖率、LR 分组、
1 维参数 `wd=0`、全部损失项确实计算、诊断目标存在、8 个模块梯度非零、
scheduler 时间轴、选择性恢复方向。

### 当前状态

| 验证 | 结果 |
|---|---|
| `tools/preflight_acceptance.py` | **26/26** |
| `tools/causality_check.py` | **6/6** |
| `tools/codetrack_verify.py` | 恒等性 `7.96e-5`、checkpoint 1311/1311、10/10 梯度 |
| `tools/lora_grad_check.py` | LoRA 1296/1296，主干 0 漂移 |

**S3 前仍需修**（审查列出，我同意）：causal clip 采样器、scheduled sampling 实际接线、
TBPTT、D4（可靠性值域 0.5–0.73）、D5（历史 slot 共享标签）、当前帧 GT 的 target-mask 泄漏。
