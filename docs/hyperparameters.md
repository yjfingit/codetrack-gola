# CodeTrack 超参数清单与推荐值

按"需要你拍板的程度"分五层。**约束型**（改了会坏）不要动；**已定**的是方案/论文里已经决定或架构图给定的；**静默默认**是当前存在于代码、但 YAML 里没有暴露、你并不知道它们的值——这一层最需要你过一眼；**训练/损失**是真正需要实验调的部分。

`[需决策]` = 建议你确认或改；`[已定]` = 保持；`[约束]` = 不可改。

---

## A. 约束型（不可改，改了直接坏）

| 参数 | 值 | 约束来源 |
|---|---|---|
| `z_len` | **64** | `112/14=8` → 8×8 template tokens |
| `x_len` | **256** | `224/14=16` → 16×16 search tokens |
| `dim` | **768** | DINOv2 ViT-B/14 嵌入维 |
| `grid` | **16** | `sqrt(256)`，`validate()` 强制 `x_len == grid²` |
| `d_c`(=12) × `M`(=64) | **768** | 必须等于 `N × d_v` = `256×3`；`validate()` 强制 `M·d_c ≥ N·d_v` |
| GOLA 主干 | 冻结 | 上游 GOLA 的 PEFT 设计；全量微调在论文里更差 |
| 归一化统计 | `'mm'`（6 通道） | 必须与 GOLA 训练时一致 |

---

## B. 已定（架构图给定或方案已决定，保持）

| 参数 | 值 | 依据 |
|---|---|---|
| `mid_dim` | 128 | 架构图 3.1「768→128」 |
| `topk_tokens` | 32 | 架构图 `K ≤ 32 (≤12.5%)` |
| `num_neighbours` | 8 | 架构图 `K_n = 8` |
| `refiner_hidden` | 256 | 架构图「768→256→768」 |
| `diffusion_steps` | **2** | 方案红线：**不做 20–100 步 DDPM** |
| `memory_frames` | 3 | 架构图 `K = 3~4`；DTPTrack Table 7 显示帧数单调增益，4–5 略好但显存换收益 |
| `memory_tokens` / `memory_dim` | 8 / 128 | 架构图 `B×K×8×128` |
| `gola_update_threshold` | 0.84 | 上游官方值 |
| `diagnosis_alpha` | 0.5 | 方案文档 §5.5 第一版取值 |
| `detection_prior` | 0.2 | 方案文档 corruption 配比 |
| `w_track_clean` | 0.25 | 架构图 Loss 面板 |
| `w_track_corr` | **1.0** | 架构图；主角，不要动 |
| `M` (`num_checks`) | 64 | 架构图 `M = 64`；消融 `{32,64,96}` |
| `h_links_per_check` | 12 | `768/64 = 12`，与 `M` 绑定 |
| `h_min_col_degree` | 3 | 架构图 `d_v ≈ 3` |

---

## C. 原本是静默默认、现已暴露为可配置项 ★最需要你确认

这三项原先硬编码在代码里，**已经接进 `config.py` 与 `codetrack_b.yaml`**，可以直接调：

| 参数 | 当前值 | 作用（已实测） | 调整会怎样 |
|---|---|---|---|
| `h_free_edge_frac` | **0.25** | 每个 check 的 12 条边中有多少来自全局池。实测：0.00 / 0.25 / 1.00 三档的边集合几乎不重叠（重叠 35–43/768），check 监视跨度 **10.7 / 12.4 / 13.5** 格 | **调高 = 局部性变弱**。这是 `H` 结构消融（实验文档 H3）的核心旋钮 |
| `residual_gate_init` | **−8.0** | 一个标量同时决定 ①stage-0 与基线的接近度 ②整个修正分支的梯度可达性。实测 sigmoid：−12→6.1e-6、−8→3.4e-4、−4→1.8e-2；到达时序记忆的梯度 −12 时 3.1e-7、−8 时 1.9e-5 | **不要设成大负数**。−12 名义连通但训不动 |
| `motion_bias_scale` | **0.5** | `attn_bias = log(M_t) · scale`，运动先验影响证据路由的唯一旋钮。0 = refiner 完全忽略运动 | 调高 = 运动主导路由；调低 = 运动仅作弱提示 |

其余静默默认（保持即可）：

| 参数 | 当前值 | 位置 | 为什么 |
|---|---|---|---|
| `h_locality_window` | **5** | `codetrack_b.yaml` | `half=2` → 5×5=25 token 候选池，从中抽 9 条局部边 |
| `h_locality_wrap` | **True** | `config.py:43` | 网格边界环绕，避免边缘 check 候选不足 |
| `template_gate` 初值 bias | **−1.0** (c≈0.27) | `template.py:38` | 保守起步。+2 起步 c≈0.88 会饱和、梯度消失 |
| `template_gate_hidden` | 64 | `config.py:77` | 5 维输入的小 MLP，64 足够 |
| KL 目标高斯带宽 | 用 GT 框的 `w,h` | `codetrack.py:174` | 目标 = GT 框中心、尺度为 GT 框尺寸的单位质量高斯 |
| 帧级运动条件 | `[mean(M_t), u_t]` | `codetrack.py:339` | 2 维 |

### 本轮已修的两个实现不一致

1. ~~`lambda_mem` 默认值不一致~~ → **已修**：`config.py` 默认由 `0.0` 改为 `0.1`，与 `criteria.py` 一致。原先若 YAML 不显式给值，`L_mem` 会被静默关闭，而它是记忆可靠性头的**唯一**监督。
2. ~~`memory_pull_weight` 是死字段~~ → **已标注为 RESERVED/未使用**（默认 `None`）。它原本设想用于模板门控里 mean-TopK 与恢复置信度的加权混合，但当前实现把两者作为独立输入交给 MLP 自行加权，所以这个旋钮不起作用。保留名字是为了避免日后以不同语义重新引入。

---

## D. 损失权重（需要实验调，以下是启动值）

| 参数 | 启动值 | 含义 | 调整方向 |
|---|---:|---|---|
| `w_track_corr` | **1.0** | 恢复后特征的跟踪损失 | **不动**（主角） |
| `w_track_clean` | **0.25** | clean 支路跟踪损失 | 若 clean SR 掉 >0.3 则升到 0.5 |
| `lambda_diag` | **0.5** | `BCE(q,e*) + 0.5·SmoothL1(s,s*)` | 若 AUROC 不达标升到 1.0 |
| `lambda_rec` | **0.2** | 恢复质量 | **风险 #2 的旋钮**：若 `L_rec` 降但 SR 不涨 → 降到 0.1 或 0 |
| `lambda_align` | **0.2** | mean-var completion | 保持 |
| `lambda_pres` | **0.01** | 可靠 token 身份保持 | 代码层已严格 bypass，可降到 0 |
| `lambda_mem` | **0.1** | 记忆可靠性 BCE | 建议固定 0.1 |
| `lambda_gate` | **0.1** | 门控决策 BCE | 保持 |
| `lambda_motion` | **0.2** | 运动先验 KL | **注意**：单序列上恒为 0（退化）；连续序列训练时才有意义 |
| `L_GOLA-orth` | **1.4e-3** | GOLA 组正交（上游 runner 已含） | 不动 |

---

## E. Corruption（需要实验调）

| 参数 | 启动值 | 含义 | 备注 |
|---|---:|---|---|
| `corruption_enabled` | True | 总开关 | — |
| `corruption_image_prob` | **0.45** | 图像级 corruption 概率 | 方案配比 45% |
| patch 级触发概率 | **0.33**（硬编码 `(1-0.45)*0.6`） | token 级 corruption 概率 | **[需决策]** 实际配比偏离方案（方案要 20% token + 5% compound）。建议暴露为一个参数 |
| `corruption_token_ratio` | **0.4** | 被擦除的 patch 比例 | 方案文档 corruption 配比 |
| `corruption_severity` | **0.4** | 强度 | 配合 curriculum 可递增 |
| corruption 类型集合 | 5 图像 + 4 token | `corruption.py` | 图像：low_light / over_exposure / gaussian_noise / blur / occlusion；token：block_erase / burst_erase / feat_noise / modality_drop |

---

## F. 训练超参（启动值 = 上游 GOLA 配方）

| 参数 | 启动值 | 来源 | 备注 |
|---|---|---|---|
| optimizer | AdamW | 上游 | — |
| `betas` / `eps` | (0.9,0.999) / 默认 | 上游 | — |
| `weight_decay` | **0.1** | 上游 | bias/LayerNorm/embed 为 0 |
| `max_grad_norm` | **1.0** | 上游 | — |
| scheduler | cosine，**按 optimizer update 计数的分阶段 warmup** | 本仓库 | `lr_min=1e-6, warmup_lr=1e-7, warmup_epochs=0` |
| AMP dtype | float16 | 上游 | **主干与 head 强制 fp32**（见实现说明） |
| `num_epochs` | **10** | 上游 | 冒烟用 1；分阶段配方请改用 update 数定义阶段 |
| `global_batch_size` | **128** | 上游 | 冒烟用 2 |
| `samples_per_epoch` | **131072** | 上游 | 冒烟用 6；准入短训 65536 |
| `num_workers` | 4 | 本机 `consts.yaml` | 上游 6；6 会与另一实验抢 CPU |
| `max_gaps`（pair sampling） | 100 | 上游 | — |
| `torch_compile` | 关（冒烟）/ 可开（长训） | — | `disable_torch_compile` mixin |

### warmup：按 update 数，而不是按 epoch

`warmup_epochs: 2` 属于"单阶段 10 epoch"的配方。本仓库是分阶段训练，`samples_per_epoch=131072`
且 `8×16` 时一个 epoch 只有 1024 个 optimizer update，于是：

| 阶段 | 预算(updates) | 折合 epoch | 建议 warmup(updates) |
|---|---:|---:|---:|
| S0 准入短训 | 300–500 | 0.3–0.5 | 0 |
| S1 spatial recovery | 1500 | 1.46 | 128 |
| S2 joint PEFT | 8000 | 7.81 | 410 |
| S3 temporal | 6000 | 5.86 | 256 |
| S4 mixed | 2000 | 1.95 | 48 |

若沿用 `warmup_epochs: 2`，S1 整个阶段都会停在 warmup 里出不来。因此所有 stage 配置写
`warmup_epochs: 0`（`warmup_prefix: true` 保留，保证 cosine 从 step 0 起算），warmup 由阶段驱动按
**update 数**给。注意 scheduler builder 会用 `num_iterations_per_epoch // grad_accumulation_steps`
换算，所以它自己的 `t_initial` 是 10240（= 10 × 1024）而不是 163840，这也是训练循环必须传
"optimizer update 计数"而不是 micro-step 的原因。

### 参数组规则必须显式标注维度

```yaml
per_parameter:
  - name_regex: 'codetrack\.'
    ndim: 2              # 张量
    lr: 1.e-4
  - type: "zero_1d_param_weight_decay"
    ndim: [ 0, 1 ]       # bias / norm
    name_regex: 'codetrack\.'
    lr: 1.e-4
  - type: "zero_1d_param_weight_decay"   # 兜底
```

`zero_1d_param_weight_decay` 会**消耗**它匹配到的参数（从 pool 里移走），所以不标维度会让第一条
这样的规则把整个模型的 bias/norm 吸进同一个组。详见 `docs/implementation.md` §15.1。

### 分组学习率 —— **[需决策]，方案文档给的是策略不是数值**

我当前**没有**实现分组 LR（框架只有一个 `lr: 1e-4` 应用到所有可训练参数）。方案文档 §10.3 的建议是：

| 参数组 | 方案建议 LR | 理由 |
|---|---:|---|
| CodeTrack（随机初始化） | **1e-4** | 最大 |
| GOLA low-rank (`lora.GA/GB`) | **2e-5 ~ 3e-5** | 已是好 checkpoint |
| token-type embedding | **2e-5 ~ 3e-5** | 同上 |
| 原 GOLA head | **1e-5** | 决策面已好，防被合成 corruption 带偏 |
| Stage-3 DINO last-2 blocks | **1e-6 ~ 2e-6** | 最小 |

**当前是"所有组都用 1e-4"**，这与方案文档不一致。框架的 `per_parameter` 机制支持按 `name_regex` 设不同 `lr`，我可以照上表配出来。**这是我认为最值得先做的一项**，因为 head 用 1e-4 学习率在合成 corruption 上很容易漂移。

---

## G. 分阶段建议（对应实验文档 Stage 0–4）

| Stage | 训练范围 | LR | steps | batch |
|---|---|---|---|---|
| 0 复现 | 不训练 | — | — | — |
| 1 稳定化 | 仅 Diagnosis + Refiner，GOLA 冻结 | 1e-4 | 500–1000 | 32–64 |
| 2 主训练 | GOLA adapters + head + CodeTrack | 分组（见 F） | 6000–8000 | 64–128 |
| 3 可选 | + DINO last-2 blocks | DINO 1e-6~2e-6 | 1500–2000 | 64 |
| 4 时序 | + KF / memory / protection，改 causal clip | 同 Stage 2 | — | 32 |

**关于 batch**：4090 24GB 单卡跑 batch 128 需要 grad accumulation。实测双分支前向（主干 fp32 + CodeTrack）在 batch 2 时约 1.3GB 显存，线性外推 batch 128 约需 80GB+，所以必须 accumulation：

```
global_batch_size: 128 = micro_batch 8 × grad_accumulation 16
```

micro_batch 建议从 **8** 起（实测 batch 2 用 1.3GB，batch 8 约 5GB，留足余量），不够再加。

---

## H. 我认为最需要你现在拍板的 5 项

1. **分组学习率**（F 节末）——当前全用 1e-4，与方案文档冲突，且 head 有漂移风险。
2. **`h_free_edge_frac`（0.25）与 `motion_bias_scale`（0.5）**——已暴露为可配置项，直接决定 ECC 图结构与运动影响强度，且都是消融项（实验文档 A/B/C/E 与 H3）。
3. **token 级 corruption 实际配比（0.33）偏离方案（0.20）**——要不要按方案精确配比。
4. **`lambda_rec` 的启动值（0.2）**——它是风险 #2「重建变好但 SR 不涨」的唯一旋钮，方案给的区间是 0.2–0.3，我取了区间下沿。
5. **`memory_frames` 取 3 还是 4**——DTPTrack Table 7 显示 3→4 帧约 +0.5 AUC，代价是显存与算力；架构图给的是 3~4，我默认 3。

---

## 附：本次冒烟验证实际用的值（不是推荐值，别混淆）

| 项 | 冒烟值 | 正式建议 |
|---|---|---|
| `num_epochs` | 1 | 10 |
| `global_batch_size` | 2 | 128（accumulation） |
| `samples_per_epoch` | 6 | 131072 |
| `num_workers` | 2 | 4 |
| 训练序列数 | 1（`10crosswhite`） | 全量 979 |
| `torch_compile` | 关 | 可开 |

冒烟的目的是验证**梯度连通与端到端可跑**，不是性能。指标（success 0.7967）来自**单序列**，与论文的 LasHeR 全测试集（245 序列）不可比。
