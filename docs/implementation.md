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
e_aux  = 1 - cos(X_rgb_corrupted, X_rgb_clean)        # 跨模态证据
e*     = alpha*e_feat + (1-alpha)*e_aux               # alpha=0.5（第四轮修正，见 §16.4）
s*     = H_bar @ e*                                   # 软 syndrome 目标
```

实测：**`Loss/diag = 0.55`**，`error_target (B,256)`、`syndrome_target (B,64)`，此前完全不存在。

**表述红线（第四轮审查提出，已采纳）**：`s* = H̄ e*` 只能称为
**"LDPC-inspired continuous syndrome / check error density"**，**不能**写成严格纠错码的 syndrome。
真正的二元 parity syndrome 不是简单的 `He`（它要作用在 GF(2) 上、且需要硬判决），
而我们这里 `H̄` 是行归一化后的正实数矩阵、`e*` 是连续余弦偏差。

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

**修法（第四轮修正，见 §15.1）**：`zero_1d_param_weight_decay` 的语义是"收集非衰减参数
（ndim≤1 或 norm 层）并给 `wd=0`"。第四轮发现该规则类型**在框架实现里完全忽略
`name_regex` / `ndim`**，所以"按 scope 写多条 `zero_1d`"这个方案本身是坏的。最终方案：
张量规则显式带 `ndim: 2`，`zero_1d` 规则显式带 `ndim: [0, 1]`，并修好框架实现使其真正尊重过滤器。

实测 8 组、**1406/1406 张量全覆盖**、1 维参数 `wd=0` 且 lr 归属正确。

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

### P1-8｜图像级 corruption 污染了 clean teacher（第四轮已修，见 §15.5）

数据 plugin 直接 `collated.input["x"] = corrupted_x`，**没有保留 clean 版本**。
所以对 image-level corruption 样本，模型里的所谓 teacher 拿到的仍是**被破坏的图像**——
`student = corrupted` 与 `teacher = corrupted` 相同，师生残差失去意义。

已修的部分：`image_corruption_mask` 现在并入 `was_corrupted`（此前图像级破坏被报成"可信帧"，
使记忆可靠性与模板门控都拿到错误标签）。
**`x_clean` 已在第四轮补齐**（§15.5）：plugin 在写回 `x` 之前保存 `collated.input["x_clean"]`，
`GOLA_DINOv2` 接受并把它交给 teacher 分支；eval 时该键不存在，回落为 `x`。

### 新增验收门

`tools/preflight_acceptance.py` —— 第四轮重写后 **55/55 通过**。它现在**直接读取
`config/GOLA/codetrack_s2/config.yaml`**（经 `custom_yaml_loader`，`!include` 正常解析）拿到
真实的 `per_parameter`，而不再复刻一份简化的规则表：上一版之所以"全绿"，正是因为它测的是自己
的 fixture，而真正会跑的配置把 head/LoRA 的 1 维参数塞进了 CodeTrack 组。

覆盖：配置自查（temporal 关闭、warmup=0、`ndim` 标注齐全）、优化器覆盖率与 lr/wd 路由、
**单组不得跨 scope**、1 维参数 `wd=0`、3 维以上参数不被 `zero_1d` 丢掉、全部损失项确实计算、
`Error/*` 指标存在、诊断 AUROC 可测、`d_before/d_after` 仅在 corrupt token 上统计、
写回计划（ramp 非零 / linear_noise 复现旧行为）、`x_clean` 通路、8 个模块梯度非零、
scheduler 用 update 计数且 `t_initial=10240`。

### 当前状态

| 验证 | 结果 |
|---|---|
| `tools/preflight_acceptance.py` | **104/104** (current; this row was 55/55 at the time of round 3) |
| `tools/causality_check.py` | **6/6** |
| `tools/codetrack_verify.py` | 恒等性 `1.15e-4`、checkpoint 1311/1311、10/10 梯度 |
| `tools/lora_grad_check.py` | LoRA 1296/1296，主干 0 漂移 |
| 300–500 update 短训 | 见 §15.8 |

**S3 前仍需修**（审查列出，我同意）：causal clip 采样器、scheduled sampling 实际接线、
TBPTT、D4（可靠性值域 0.5–0.73）、D5（历史 slot 共享标签）、当前帧 GT 的 target-mask 泄漏。

---

## 15. 第四轮：审查意见逐条核对与修复（2026-10-04）

审查基于 `a205ada`，而当前 HEAD 是 `e8ee251`——它列的 5 个 P0 里有 **3 个我已经修了**。
下面先给核对表，再写本轮真正要修的部分。

| 审查意见 | 实际状态 | 依据 |
|---|---|---|
| accumulation=16 时 scheduler 快 16 倍 | **已修**（`e8ee251`） | runner 传 `(iteration+1)//accum` |
| loss 没有 `/16` | **已修** | `backward_loss = (loss + orth)/accum` |
| `L_diag` 从未计算 | **已修** | `gola.py` 造 `error_target`/`syndrome_target` |
| Denoiser 写回方向反了 | **已修** | `recovery.py` 用 `token_error` 门控，实测 91.8× |
| S1/S2 必须关 temporal | **已修**（`codetrack_s2`），`codetrack_full` 保留全开供 S3 | `codetrack_spatial.yaml` |
| `zero_1d` 破坏 weight-decay 语义 | **未修，且比审查说的更严重** | §15.1 |
| image corruption 污染 clean teacher | **未修** | §15.5 |
| `L_gain` 写进 config 却没实现 | **未修** | §15.4 |
| 2-step denoiser 只有 1 次有效写回 | **未修** | §15.6 |
| S3 blockers（clip/scheduled sampling/TBPTT/D4/D5/GT mask） | 未做，维持 S3 前修 | 见 §14 末尾 |

### 15.1｜`zero_1d_param_weight_decay` 静默忽略 `name_regex` / `ndim`（P0，比审查判断更严重）

框架实现完全不看 rule 里的过滤器：

```python
# 修改前
for module_parameter_name in list(module_parameters.keys()):
    if module_parameter_name not in decay_parameter_names:      # 所有 ndim<=1 或 norm 层
        one_dim_params.append(module_parameters.pop(module_parameter_name))
```

`_Filter`（会正确应用 `name_regex`/`ndim`/`name_prefix`）**根本没被调用**。后果用实测数据说明：
开启 CodeTrack 后模型里有 **59 个** non-decay 参数（`refiner.norm.weight`、`denoiser.norm1/2.*`、
`head.*.bias`、全部 LoRA bias、`token_type_embed.bias` 等）。`codetrack_s2` 的第一条规则

```yaml
- type: "zero_1d_param_weight_decay"
  name_regex: 'codetrack\.'
  lr: 1.e-4
```

会把**全部 59 个**吸进同一个 group（含 head 与 LoRA 的 1 维参数），`name_regex` 形同不存在；
后面的 `^head\.` + `zero_1d` 规则一个参数都拿不到。**即上一轮"已修好"的自证是错的**，
而 `preflight_acceptance.py` 当时复刻了同一份规则的简化版，所以没抓到。

**修法（三处）**：

1. `zero_1d_param_weight_decay.py` **先在循环里判断过滤器再 pop**，尊重
   `name_regex` / `ndim` / `name_prefix` / `name`，保留"1 维 + norm 层不做 weight decay"
   的上游语义。**不能**简单换成普通 `ndim: 1` 规则：CodeTrack 里有 8 个 LayerNorm，
   `get_decay_parameter_names` 会把它们的权重也排除在衰减之外，普通 `ndim` 规则做不到这一点。
2. 规则表改为**显式维度划分**：张量规则 `ndim: 2`，`zero_1d` 规则 `ndim: [0, 1]`。
   配置文件里现在能一眼看出每条规则的适用范围。
3. **修掉两个我自己引入的 bug**（第五轮审查后重写为最终形态，见 §16.2 末尾）：
   * 曾走 `filter_out_params_by_rule_`（它**边匹配边 pop**），再把非衰减匹配放回 pool；
     而"空集早退"发生在放回之前 → `codetrack.memory.base_prior` 被静默丢弃
     （实测覆盖率 `1405/1406`，`requires_grad=True` 却永不更新）。
   * 更早一版还二次 `module_parameters.pop(name)` → `KeyError`。

   最终形态把过滤器判断**放在 pop 之前**，这两个失效模式都从结构上消失了；也不再需要
   `warnings` 提示。审查者指出原版路径应尽量少绕路，这一点已按它的建议落地。

上游 `config/GOLA/run.yaml` 的 `zero_1d` 无任何过滤器 → `_Filter.name_filter is None` → 全通过。
第五轮审查据此指出：应说 **"optimizer-semantically equivalent"**，而不是"逐位等价"——
新实现不再经历 pop→restore，也不再发 warning，程序行为并非字节级一致。已按此措辞修正。

**未预料到的连带缺陷**：CodeTrack 有唯一一个 3 维参数 `codetrack.memory.base_prior`。
它被兜底 `zero_1d` 规则匹配到、判定为"可衰减"、放回 pool，但**空集早退发生在放回之前**，
于是它被丢弃：`requires_grad=True` 却不属于任何 param group → **永远不更新**。
实测覆盖率 `1405/1406`。已把放回移到早退之前，并把这条写进验收门
（"3 维以上参数不被 `zero_1d` 丢掉"），覆盖率回到 **1406/1406**。

### 15.2｜配置一致性

* `codetrack_full/config.yaml` 的 `warmup_epochs` 仍是 `2`，与 `codetrack_s2` 的 `0` 矛盾。
  统一为 `0`，阶段 warmup 按 update 数在阶段驱动里给（S1 128 / S2 410 / S3 256 / S4 48）。
* `codetrack_smoke/config.yaml` 的注释仍在重复"`zero_1d` 不带 lr → AdamW 默认 1e-3"这个已被审查
  证伪的说法，且张量规则没有 `ndim`。注释改正、规则补 `ndim: 2`。
* `codetrack_s2` 里 `codetrack\.(motion|memory)\.` 这条规则在该阶段**匹配不到任何参数**
  （temporal 已关），会触发框架的 `assert len(named_params) > 0, "rule must be effective"`。
  已删除，并在原处写明原因——通用 `codetrack\.` 规则给的是同一个 `1e-4`，无损失。

### 15.3｜AMP 下的 `binary_cross_entropy` 崩溃（第一次真实训练就炸）

第一次跑真实训练时 `L_diag` 直接抛
`RuntimeError: torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast`。
原因是 `criteria.py` 对 `q`（sigmoid 输出）用了普通 BCE。已改为
`binary_cross_entropy_with_logits(q_logits, err)`；那条路径需要 `q_logits`，而 `CodeTrack.forward`
只返回了 `s_logits` —— 已补 `q_logits` 并接到 `extras`。同时把 `smooth_l1`/`err` 显式转 `float()`。
**这正是"preflight 短训"存在的意义：单模块脚本永远发现不了只在 AMP + 真实 criterion 下才炸的路径。**

### 15.4｜`L_gain` 与"失真度"指标（审查要求，已实现）

`Loss/rec = 1 - cos + 0.25*huber` 只说明"输出像不像 clean"，一个残差门很小的分支靠"什么都不改"
就能让它很小。新增：

```python
d_before = mean( (1 - cos(X_rec,   X_clean)) [corrupted tokens] )   # 精修后、去噪前
d_after  = mean( (1 - cos(X_final, X_clean)) [corrupted tokens] )   # 去噪后
Loss/gain = ReLU(d_after - 0.8 * d_before)                          # lambda_gain = 0.2
```

全部**只在 `corruption_mask` 为真的 token 上**统计（混进 ~86% 干净样本会把信号淹没）。
同时输出 `Error/{d_before,d_after,gain,corrupted_fraction,d_before_suspect,d_after_suspect,
gain_suspect,q_mean,q_auroc}`，训练日志按 `interval` 打印。

**一个在真实日志里才暴露的度量设计错误**：`d_before`/`d_after` 最初用
`‖x − clean‖ / ‖clean‖`，实测 `0.7402 / 0.7402` 四位小数完全一样，而同一时刻 `Loss/rec` 明明在动。
原因是**被破坏的 token 与 clean 的差异几乎全在方向上**（残差流 + LayerNorm 让 token 范数近似守恒），
欧氏距离读不出来。已统一改为 `1 - cos`，与 `Loss/rec` 口径一致，`gain` 与 `L_rec` 不会互相矛盾
（合成 batch 实测 `d_before=0.1144 / d_after=0.1144`，接近恒等，符合残差门设计）。

### 15.5｜图像级 corruption 保留 `x_clean`（已修）

* plugin 在写回 `collated.input["x"]` **之前**保存 `collated.input["x_clean"] = x`（clean 抽签时也写，
  保持形状稳定）。
* `GOLA_DINOv2.forward` / `_forward_codetrack` 新增 `x_clean: Optional[Tensor]`，teacher 分支用
  `x_teacher = x_clean if x_clean is not None else x`。eval 时 plugin 不跑、键不存在 → 回落 `x`，与原来一致。
* 效果：图像级破坏样本上 `student=corrupted`、`teacher=clean`，`e_feat`/`e_aux` 才是真实残差
  （修之前这类样本的 `e_feat` 恒为 0，诊断监督是**假的**）。
* 顺带修正 `e*` 的组合方式：原式 `alpha*e_feat + (1-alpha)*0.5*(e_feat+e_aux)` 在 `alpha=0.5` 时等于
  `0.75*e_feat + 0.25*e_aux`——`alpha` 并没有发挥配置里声称的作用。现改为
  `err = alpha*e_feat + (1-alpha)*e_aux`，`alpha` 是真正的混合系数，并额外输出
  `error_target_tir` / `error_target_rgb` 便于分辨两种模态。

### 15.6｜2-step denoiser 只有 1 次有效写回（已修，带开关）

实测 `alpha_bar = [1.0, 0.001]` → 旧写回 `w = 1 - ab = [0.0, 0.999]`：第 1 步算出 `pred`（以及梯度）
后乘 0 丢弃，模块宣称的"2-step refinement"实际只有 1 步能改动 token。

新增 `diffusion_write_schedule: "ramp" | "linear_noise"`（默认 `ramp`）：

| steps | `ramp`（新默认） | `linear_noise`（旧行为） |
|---:|---|---|
| 2 | `[0.5, 1.0]` | `[0.0, 0.999]` |
| 4 | `[0.25, 0.5, 0.75, 1.0]` | `[0.0, 0.134, 0.5, 0.999]` |

`linear_noise` 保留为可选项，因为消融需要"新旧写回计划对照"这一列；验收门同时断言
ramp 每步非零、linear_noise 仍复现旧行为。

**表述修正（第四轮审查提出）**：ramp 计划下**整体是近似恒等，但第 0 步不再是"严格零写回"**。
旧计划 `w_0 = 0` 使第 0 步的 `pred` 及其梯度被精确丢弃；ramp 让第 0 步也参与。
step-0 的精确恒等由 `sigmoid(residual_gate) ≈ 3.4e-4` 承担，而不是由写回权重承担——
实测恒等性 `1.15e-4 < 1e-2`，仍然满足"初始化不扰动预训练 head"这一硬约束。

### 15.7｜S1 的"恢复分支学得动吗"——实测，而不是推理

审查担心 gate `-8` 让 S1 白跑。用固定 batch 实测（**只训 `codetrack.*`，lr 1e-4，50 步**）：

| 模块 | max\|Δparam\| @50 步 |
|---|---|
| `codetrack.H.H`（ECC 边权） | **3.63e-3** |
| `refiner`（含 norm，不含 gate） | **5.79e-3** |
| `denoiser`（含 norm，不含 gate） | **6.25e-3** |
| `diagnosis` | **4.69e-3** |

同时 `Loss/diag` 从 0.581 降到 0.452、`Loss/gain` 从 0.032 降到 0.025。
**结论：分支确实在学。** 单步测得的 `|grad|` 很小（`H` 7e-7）是 Adam 一阶步的假象——AdamW 会按
梯度尺度归一化，所以多步位移才是判据。残差门本身开得很慢（50 步只从 −8 到 −7.994，因为
`sigmoid'(-8)=3.35e-4`），这是"step-0 必须是恒等"这一硬约束的代价，而不是 bug；真实的门行为要由
`Error/d_after < Error/d_before` 的趋势来判断（§15.8 的短训窗口）。

### 15.8｜300–500 update 准入短训（已执行）

`config/GOLA/codetrack_preflight/`（= `codetrack_s2` 的副本，`samples_per_epoch: 65536` →
每 epoch 512 个 optimizer update；`num_epochs` 保持 10 以便 cosine 视野与正式训练一致）。
脚本：`bash scripts/codetrack_wait_progress.sh outputs/preflight/run.log 4900 45`
（每 20 s 检查一次，到点打印里程碑表）。日志：`outputs/preflight/run.log`。

**实跑 310 个 optimizer update / 4967 micro-iteration，约 28 分钟。** 单 batch 读数：

| iter | upd | lr | grad_norm | loss | Loss/cls | Loss/diag | Loss/align | Loss/track_corr | Error/q_auroc |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 1.00e-4 | nan | 17.42 | 3.860 | 0.427 | 3.231 | 4.517 | — |
| 400 | 25 | 1.00e-4 | 2.57 | 7.33 | 1.108 | 0.224 | 2.334 | 1.480 | 0.706 |
| 800 | 50 | 1.00e-4 | 1.82 | 5.45 | 0.972 | 0.222 | 1.204 | 1.255 | 0.402 |
| 1600 | 100 | 1.00e-4 | 2.86 | 4.06 | 0.863 | 0.225 | 0.407 | 1.048 | 0.372 |
| 2400 | 150 | 1.00e-4 | 2.32 | 3.48 | 0.760 | 0.220 | 0.315 | 0.913 | 0.393 |
| 3200 | 200 | 1.00e-4 | 2.92 | 3.21 | 0.729 | 0.220 | 0.254 | 0.834 | 0.453 |
| 4000 | 250 | 9.91e-5 | 3.31 | 3.07 | 0.675 | 0.221 | 0.210 | 0.776 | 0.368 |
| 4800 | 300 | 9.89e-5 | 3.70 | 3.07 | 0.671 | 0.218 | 0.189 | 0.792 | 0.518 |

判定逐项：

| 审查验收项 | 读数 | 判定 |
|---|---|---|
| 各损失项确实出现并下降 | 全部出现；`loss 17.4→3.1`、`cls 3.86→0.67`、`align 3.23→0.19`、`track_corr 4.52→0.79` | PASS |
| `Loss/diag` 存在并下降 | `0.427 → 0.218` | PASS |
| `Loss/rec` / `Loss/gain` 有意义 | 均计算；`rec` 在 0.000–0.005、`gain` 在 0.002–0.007 | PASS（见下） |
| `q AUROC` | 0.37–0.71，在随机头应有的 0.5 附近抖动 | PASS（趋势需更长窗口） |
| `d_after < d_before` | 否：`0.0331/0.0331`、`0.0111/0.0111` | **符合预期，非失败** |
| 打印真实 LR、cosine 未提前跑完 | upd 0→300 时 `1.00e-4 → 9.89e-5` | PASS |
| 各模块梯度非零 | `grad_norm` 2.2–3.7（首 16 个 micro-step 为 `nan`，累积边界） | PASS |
| `Error/corrupted_fraction` | 稳定 `0.0498`，与 6%/6% 抽签率一致 | PASS |

关于 `d_after ≈ d_before`：残差门初始化是 `sigmoid(-8) = 3.35e-4`，**step-0 恒等是架构硬约束**，
所以"恢复改善了多少"在第 300 步必然还是噪声级。真正的判据是该差值随门打开而变负（
`Error/d_after < Error/d_before` 且 `Error/gain > 0`），这属于正式 S1 的观察目标。
反过来说，如果这里出现明显改善，反而说明恒等性被破坏了。

**结论：准入通过，可以启动 S1 → S2。** 唯一的观察项是 `q_auroc` 与 `gain` 的趋势，二者都需要
远长于 300 update 的窗口才有统计意义，不构成准入阻塞。

### 15.9｜尚未验证的部分（如实列出）

* `x_clean` 只做了**结构验证**（plugin 写入顺序、模型接收）与短训的隐式验证，没有单独构造
  "图像级破坏样本 + clean teacher" 的对照实验。
* `diffusion_write_schedule` 只验证了权重表与恒等性，**没有**做 ramp vs linear_noise 的收敛对照。
* `x_clean` 与写回计划这两项都进入了正式 S1 的变量集，S1 与 S2 之间不做进一步消融。
* 训练速度实测 `time ≈ 0.33–1.6 s/micro-step`，`000` 级 epoch 预算下达数天/阶段；本轮只验证
  "能不能正确训练"，未做吞吐优化。

---

## 16. 第五轮：审查否掉"直接启动 S1"后修的三件事（2026-10-04）

审查（针对 `d39552f`）给 **S1/S2 = NO-GO**，理由是三条具体的：denoiser 的噪声是死代码、
仓库里没有真正的 S1 freeze/stage driver、判据（AUROC / gain）不可用。**三条我逐条在代码里
核实，全部属实。** 另外审查纠正了我一个表述错误（"3200 步饱和"）。

### 16.1｜denoiser 的噪声完全没作用（P0，死代码）

```python
# recovery.py（旧）
if alpha is None:
    alpha = torch.ones(b, n, 1, ...)      # alpha 恒为 1
...
eps = eps * (1.0 - alpha)                 # 1 - 1 = 0  →  噪声恒为 0
```

而调用方 `codetrack.py` **只传 `token_error` / `token_trust`，从不传 `alpha`**。
所以：**写回门控是对的，但"noise-modulated"这一半完全失效**。这正是"loss 会降、
checkpoint 正常、但论文声称的模块没工作"的典型形态——光看训练曲线发现不了。

**修法**：删掉含义混乱的 `alpha`，噪声与写回各自用显式命名的门，并且调用方**显式**传：

```python
error_gate = token_error if token_error is not None else (1 - token_trust) ...
noise_gate = error_gate if noise_gate is None else noise_gate
write_gate = error_gate
...
eps = torch.randn_like(x) * sigma * noise_gate
```

无诊断输入时的兜底从"`alpha=1` → 不加噪"改成**显式 `ones`（全部可疑）**：
旧兜底的语义与注释正好相反，这是它能藏住的原因。

**实测**（`tools/preflight_acceptance.py` D 段新增门禁，8 次抽样取均值）：

| | 坏 token 处 `x_in` std | 健康 token 处 | 比值 |
|---|---:|---:|---:|
| 修复后 | 0.14142 | 0.00144 | **98.2×** |
| 修复前 | 0 | 0 | （噪声恒为 0） |

### 16.2｜S1 只是文档，没有实现（P0）

全仓库 grep 不到任何 S1 冻结逻辑；`codetrack_s2` 的 optimizer 仍然带
`lora 2.5e-5` / `head 1e-5`。也就是说 **README 里让跑 S1 和 S2 用的是同一条命令**，
实际执行的始终是 S2 式的 joint PEFT。文档里的"S1 warmup 128 updates"同样没有接线。

**修法（三处，全部配置驱动）**：

1. **`parameter_scope`**：optimizer 白名单从硬编码改为
   `optimizer_config.get("parameter_scope") or <7 项缺省>`。
   缺省逐字保留现状 → 上游 `config/GOLA/run.yaml` 零影响。
   - `config/GOLA/codetrack_s1/`：`parameter_scope: ["codetrack"]` → **真正的 S1**
   - `codetrack_s2` / `codetrack_full`：不写该键 → 框架缺省（joint PEFT）
2. **`stage.max_updates`**：阶段长度按 **optimizer update** 定义。
   runner 在 `optimizer_step >= max_updates` 时置停止标志，
   `GlobalContextManager.should_stop()` 让 application 的 epoch 循环提前结束。
   `num_epochs × samples_per_epoch` 降级为"上限"，不再是阶段长度的定义。
3. **update 级 warmup 与 cosine 视野**：
   `lr_scheduler.override.t_initial_updates` 与 `parameters.warmup_updates`。

| 配置 | scope | max_updates | warmup | t_initial |
|---|---|---:|---:|---:|
| `codetrack_s1` | `["codetrack"]` | 1500 | 128 | 1500 |
| `codetrack_s2` | 缺省 7 项 | 8000 | 410 | 8000 |
| `codetrack_full` | 缺省 7 项 | 6000 | 256 | 6000 |
| `codetrack_preflight` | `["codetrack"]` | 512 | 0 | 512 |

**实测（`outputs/s0/run.log`，真实 LasHeR）**：启动日志
`stage: S0-preflight: max_updates=512, warmup_updates=0, parameter_scope=['codetrack']`；
optimizer 只打印 `codetrack.*`（73 行），**没有任何 `blocks.*` / `head.*` / `lora`**
→ S1 冻结在真跑中生效。

**一个差点又藏住的日志 bug**：`parameter_scope` 属于 `optimization.optimizer`，
而启动打印最初从 `stage` 块里读，于是对一个 CodeTrack-only 阶段打印出
`parameter_scope=<default>`——正好是"冻结没发生"的假象。已修正读取位置，并在
`tools/preflight_acceptance.py` 里断言 S1 的 optimizer **只**含 `codetrack.*`（97/97、0 外来）。

### 16.3｜判据原本不可用（P0，指标设计问题）

**(a) `q_auroc` 的阈值卡在坏 token 均值之上。** 代码写死 `thr = 0.25`，而实测
健康 token ≈ 0.14、坏 token ≈ 0.22 → 正类只剩坏 token 的极端尾部，
所以 `0.37 → 0.71 → 0.40` 这种抖动**不能用来判断诊断头学没学会**。

**修法**：主指标改为 **`Error/q_auroc_mask` = AUROC(q, corruption_mask)**。
合成破坏阶段我们手里有**精确的逐 token 真值**，根本不需要阈值。
对软目标的版本降级为次要指标 `Error/q_auroc_target` 并在注释里写明不得单独作为依据；
另加 `Error/q_error_spearman`（q 与 `error_target` 的秩相关）。

**(b) `L_gain` 对照的基准不对。** 旧式 `Relu(d_after − 0.8·d_before)` 里
`d_before` 是 `X_rec` vs clean，所以它只证明**"denoiser 比 refiner 好"**，
而不是**"整个 CodeTrack 比它的输入好"**。

**修法**：暴露 `input_tokens`（student 的 `X_t`，代码里本来就有，零成本），
补上第三个点，并让 `L_gain` 与主判据对齐：

```
d_input = 1 - cos(X_t,     X_clean)      ← 输入（新增，detach）
d_before= 1 - cos(X_rec,   X_clean)
d_after = 1 - cos(X_final, X_clean)
L_gain  = Relu(d_after - 0.8 * d_input)          ← 基准改为 d_input
Error/gain_total      = d_input - d_after        ← 论文主判据
Error/gain_refiner    = d_input - d_before
Error/gain_denoiser   = d_before - d_after
```

`d_input` **必须 detach**（审查特别肯定了这一点）：否则网络可以靠"故意把输入搞坏"降 loss。

**(c) `tools/recovery_report.py` 名不副实。** 它用的是欧氏相对距离（与训练 criterion
的 `1-cos` 不一致），而且**只是 forward report**——不训练，因此两条臂在初始化时必然
完全一样，无法回答"哪条臂学得动"。已重写为**真正的短训对照**：固定 seed +
**固定 batch 池**（消除数据噪声）、只训 `codetrack.*`、逐 update 打印
`d_input / d_before / d_after / gain_total / q_auroc_mask / q_spearman / clean loss / gate`，
并给出机读判定。GO 条件加了 `min_gain = 5e-3` 的噪声地板：
初始化时 `gain_total` 是 fp32 舍入尘埃（~1e-7），不加地板会让"improving"在
什么都没发生时也亮。入口：

```bash
"$PYTHON" main.py GOLA codetrack_s1 --mixin_config codetrack_gate_neg8   # 或 _neg5
```

`config/GOLA/_mixin/codetrack_gate_neg{8,5}.yaml` 只改
`residual_gate_init`、`max_updates=600`、`t_initial_updates=600`、`warmup_updates=0`——
**唯一自变量是 gate 初值**。

### 16.4｜顺带修正的表述

* **"3200 步饱和"是我的表述错误。** 3200 是 **micro-iteration**，只有约 **200 optimizer
  update**；当时 200→300 update 仍在下降（loss 3.21→3.07、cls .729→.671、align .254→.189），
  **当时完全不足以认定饱和**。
* `s* = H̄e*` 改称 "LDPC-inspired continuous syndrome"（§15.3）。
* ramp 写回计划改称"整体近似恒等，第 0 步不再是零写回"（§15.6）。

### 16.5｜当前验收状态

| 门 | 结果 |
|---|---|
| `tools/preflight_acceptance.py` | **86/86**（新增 11 项：stage 预算/视野/warmup 一致性、S1 scope、噪声非死码、`d_input` 三点、mask AUROC） |
| `tools/causality_check.py` | **6/6** |
| `tools/lora_grad_check.py` | LoRA 1296/1296、主干 0 漂移 |
| `tools/codetrack_verify.py` | 恒等性 `1.154e-04`、checkpoint 1311/1311、10/10 模块梯度 |
| S0 真跑（512 update 预算） | 见 §16.6 |

### 16.6｜S0 真跑结论

见下一次提交的 `outputs/s0/run.log`。关键读数（早期）：

| 项 | 读数 |
|---|---|
| 启动日志 | `stage: S0-preflight: max_updates=512, warmup_updates=0, parameter_scope=['codetrack']` |
| optimizer 覆盖面 | 仅 `codetrack.*`（73 行），无 `blocks.*` / `head.*` / `lora` |
| `Loss/diag` | 0.43 → 0.26 |
| `Error/q_auroc_mask` | **0.68**（对 mask 的无阈值 AUROC，初始化即远离 0.5） |
| `Error/d_input` / `d_after` | 0.0428 / 0.0428（gate 近恒等，符合设计） |
| `Error/gain_total` | ≈ 0（fp32 噪声级） |

---

## 17. 第六轮：五轮修复之后的诚实状态（2026-10-04）

前五轮修的都是"基础设施"和"接线"问题。这一节记录**修完之后的真实状态**，包括**尚未解决的主线问题**。

### 17.1 已经确定修好、有实测断言的

| 项 | 断言 / 证据 |
|---|---|
| denoiser 死噪声 | 坏 token 处噪声 std / 健康 token 处 = **98.2×**（此前恒为 0） |
| 优化器覆盖与 LR 路由 | **1406/1406**，8 组，1 维参数 `wd=0` 且 LR 正确 |
| `base_prior` 不再被丢弃 | 1405/1406 → **1406/1406** |
| S1 真冻结 GOLA | 真跑 optimizer 只有 **73–97** 个 `codetrack.*`，**0** 个 `blocks.*`/`head.*`/`lora` |
| update 作为时间轴 | 真跑打印 `[stage] update budget reached; stopping after epoch 9` |
| cosine 完整走完 | S0 最终 `lr = 1.000e-06`（= `lr_min`） |
| 四阶段 optimizer scope 各不相同 | preflight 实测：S1=73、S2=1382、S3>1382（含 blocks.10/11 基础权重）、S4=1406（含 motion 24 个） |
| S4 重新冻结主干 | S4 的 optimizer 里 `blocks.*` 非 LoRA 参数 **0 个** |
| AMP 下不再崩 | BCE 改 `*_with_logits`；`samples_per_epoch`、`q_logits` 通路补齐 |
| `Loss/pres` 锚点口径 | **0.95 → 0.0001**（锚在"恢复未选中的 token"） |
| 判据可用 | 无阈值 `AUROC/AUPRC(q, mask)` + `q_std/q_pos_mean/q_neg_mean` + `d_input/d_before/d_after` |
| 恒等性 | `max|Δscore_map| = 1.15e-4` |
| 因果性 / LoRA / 互补性 | 6/6、1296/1296 且主干 0 漂移、17/17 |
| 综合门 | `tools/preflight_acceptance.py` **101 项** |

### 17.2 ★ 尚未解决的主线问题：恢复分支产不出正向效果

**定义**（这是"恢复有效"的唯一判据）：`d_final < d_input`，其中
`d_* = 1 - cos(·, X_clean)`，只在 `corruption_mask` 为真的 token 上统计。

三个独立实验，结论一致：

| 实验 | 规模 | `d_input` | `d_before` | `d_after` | `gain_total` |
|---|---|---:|---:|---:|---:|
| 10 序列快速验证 | 20 updates | 0.0462 | 0.0462 | 0.0462 | **0.0000** |
| A 臂消融（合成固定 batch） | 600 updates | 0.119 | 0.119 | 0.141 | **−0.022** |
| S0 真跑（全量 LasHeR） | 512 updates | 0.0179 | — | 0.2472 | **−0.2303** |

同时**诊断头确实在学**：`q_auroc_mask` 0.56 → 0.69（20 updates）、`Loss/diag` 0.43 → 0.23。
所以问题**只在恢复（refiner）与精修（denoiser）**，不在诊断。

**gate 不是主因**：A 臂 600 步里 `residual_gate` 只从 `−8.0001` 走到 `−7.9613`
（`sigmoid` 仍是 `3.5e-4`，等于没开），所以 0.19 的角距离增量**不可能来自 gate 放大**。

**三种互斥的可能，尚未区分**（接手后第一件事）：

1. **gate 没开** —— 写回被 `sigmoid(-8) ≈ 3.4e-4` 按住，改动被 fp32 的 7 位有效数字吞掉；
2. **分支真的断了** —— 某处 detach / 被覆盖，`X_final` 实际等于 `X_t`；
3. **度量精度** —— `1-cos` 在 1e-6 量级下四舍五入到同一位。

**判定探针**（不改任何文件）：扫 `residual_gate ∈ {−8,−5,−2,0,3}`，记录
`||X_final−X_t||max`、`||X_rec−X_t||max`、三个 `d_*`：

* 随 gate 单调放大 → 可能性 1 → 修法在**初始化或目标函数**；
* 完全不随 gate 变 → 可能性 2 → 修法在**接线**（用"配置字段 vs 实际读取"扫描器揪）；
* 只在 1e-6 量级变 → 可能性 3 → 修法在**度量口径**。

### 17.3 已知的"声明了但代码从不读取"字段

| 字段 | 位置 | 影响 |
|---|---|---|
| `scheduled_sampling_prob` | `codetrack/config.py` | S4 的 exposure bias 处理是空的 |
| `memory_tbptt_steps` | `codetrack/config.py` | 跨帧信用分配是空的（当前无条件 `mem.detach()`） |

这两个字段**在整仓库里只出现在 `config.py`**（全仓库 grep 命中数 = 2）。

### 17.4 框架缺口

`--resume` 被 argparse 接受、存进 `runtime_vars.resume`、传给 `DefaultApplication` 的
`application_state_file`，但**从不被读取**（全仓库 grep 只有那一处传参）。
所以框架层面的 resume 实际不生效；编排器的 resume 走的是"上一阶段的 checkpoint 作为
`--weight_path`"加"阶段边界与 epoch 边界对齐"，不依赖这条通路。
