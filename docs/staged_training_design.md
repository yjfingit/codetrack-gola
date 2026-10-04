# CodeTrack 四阶段一键训练系统 —— 设计与差距分析

> 状态：**设计稿**（未实现）。所有"现状"结论均来自对 `d39552f` 之后工作区的实际代码检查，
> 每条都附了文件与行级依据。第 9 节列出的差距是阻塞项，不是建议。

---

## 0. 一句话结论

四阶段**状态机**（freeze/unfreeze、optimizer 重建、update 级预算、分阶段 warmup、checkpoint）
在当前框架里已经具备 80% 的地基——因为第五轮已经把 `stage.max_updates`、
`parameter_scope`、update 级 `t_initial/warmup` 落进了 runner 与 scheduler。

**真正的阻塞项只有两个，且都不是编排层**：

1. **临时分支（S4）完全不存在**：没有 causal clip sampler、没有 `reset_sequence()` 接线、
   没有 TBPTT、D4/D5 未修、scheduled sampling 未接线。
2. **S1 现在会主动损害表征**（实测，见 §2）。这必须先用 -8/-5 对照消融定位，
   而不是先写编排器把错误的行为自动化。

---

## 1. 仓库现状 vs 规格：逐项对照

| 规格要求 | 现状 | 依据 |
|---|---|---|
| **update 作为时间轴** | ✅ 已有 | `stage.max_updates`；runner 在 `optimizer_step >= max_updates` 置停止标志；`GlobalContextManager.should_stop()` 让 epoch 循环提前结束 |
| **scheduler 用 update 计数** | ✅ 已有 | `lr_scheduler.override.t_initial_updates` / `parameters.warmup_updates`；`timm_scheduler/builder.py` |
| **S1 真冻结 GOLA** | ✅ 已有 | `optimizer.parameter_scope: ["codetrack"]` → 实测 optimizer 只有 97 个 CodeTrack 张量、0 个 `blocks.*`/`head.*`/`lora` |
| **S2 分组 LR** | ✅ 已有 | 8 组规则，1406/1406 覆盖，1 维参数 `wd=0` 且按 scope 路由 LR |
| **AMP fp16 + trunk/head fp32** | ✅ 已有，**规格也要求保留** | `autocast(enabled=False)` 包住 trunk/head |
| **denoiser 噪声真的作用** | ✅ 已修（第五轮） | `noise_gate` 显式传入；实测坏/健康 = 98.2× |
| **判据不用 `error_target > 0.25`** | ✅ 已修 | 主指标 `Error/q_auroc_mask`（无阈值，对 `corruption_mask`）；另加 `Error/q_error_spearman` |
| **`d_input` 三点统一 1-cos** | ✅ 已修 | `input_tokens` 暴露；`d_input/d_before/d_after` 全部 `1-cos`，只在 corrupt token 上统计 |
| **AUPRC** | ❌ **缺** | `codetrack/criteria.py` 只有 `_auroc_against_mask` / `_spearman` / `_rank_auroc`；全仓库无 `auprc/average_precision` |
| **PR / NPR / SR** | ⚠️ 有实现但未接入训练 | `trackit/data/components/result_collector/handler/one_pass_evaluation/ope_metrics.py`（`OPEMetrics`、`DatasetOPEMetricsList`）已有 `success_score/precision_score/normalized_precision_score`；`config/GOLA/codetrack_eval/` 已配 LasHeR `test` split |
| **属性维度 FM/PO/TO/LI/TC** | ⚠️ 有属性文件，未接入编排器 | `data/LasHeR/AttriSeqsTxt`、`Attributes_order.txt` |
| **per-stage best checkpoint** | ❌ **缺** | 框架只写 `checkpoint/<...>/latest`（`LatestCheckpointDumper`）与 epoch-trigger checkpoint；"best by metric" 需要 `_CheckpointDumper_with_metrics` 或编排器自管 |
| **resume auto** | ⚠️ 框架支持 `--resume`，但需要 `resumable: true` 且编排器要能找到"当前 stage 最近 checkpoint" | `config/.../checkpoint: resumable: false`（当前所有 stage 配置都是 false） |
| **stage report（json+txt）** | ❌ 缺 | 无 `reports/` 逻辑 |
| **stage 验收门 + GO/NO-GO** | ⚠️ 静态门有（`tools/preflight_acceptance.py` 86/86），**运行时门没有** | 无 `validate_stage*()` |
| **一键入口** | ❌ 缺 | `tools/` 下无编排器 |
| **S4 causal clip sampler** | ❌ **完全不存在** | 无任何 clip sampler |
| **`reset_sequence()` 接线** | ❌ 不存在 | `GOLA_DINOv2.reset_sequence()` 已实现，但训练 runner/数据管线从不调用 |
| **TBPTT** | ❌ 不存在 | `config.py: memory_tbptt_steps` 字段存在但**从未被读取** |
| **scheduled sampling** | ❌ 不存在 | `config.py: scheduled_sampling_prob` 字段存在但**从未被读取** |
| **D4 reliability 值域** | ❌ 未修 | `codetrack/motion.py` `cur_rel = sigmoid(mean(1-q))` → 被压在 `[0.50, 0.73]` |
| **D5 历史 slot 标签** | ❌ 未修 | 历史 slot 用当前帧 trust `expand` |
| **当前帧 GT 泄漏（未来帧）** | ❌ 未修 | `target_mask` 由当前 `gt_box_xywh` 生成后写进 memory，**read-before-write 保证不影响当前帧输出，但会影响未来帧** → train/test mismatch |
| **S3 解冻 DINO last-2** | ❌ 缺 | 无该 scope |
| **S3 自动回滚** | ❌ 缺 | 无 |

---

## 2. 必须先处理：S1 正在**损害**表征（实测证据）

这是写编排器之前必须定位的问题。`outputs/s0/run.log`（真实 LasHeR，`parameter_scope=["codetrack"]`，
512 update 预算，`residual_gate_init=-8`）在 **273 optimizer updates** 处的单 batch 读数：

| 指标 | 数值 | 含义 |
|---|---:|---|
| `Error/d_input` | 0.0550 | 输入 token 与 clean 的角距离 |
| `Error/d_before` | 0.0550 | refiner 之后 —— 未动 |
| `Error/d_after` | **0.2414** | denoiser 之后 —— **比输入差 4.4 倍** |
| `Error/gain_total` | **−0.1621** | 主判据为**负**：恢复在破坏 |
| `Error/d_input_suspect` | — | — |
| `Loss/pres` | 5.6e-5 → **0.9490**（EMA 0.31） | 健康 token 被推动，而它是被惩罚的 |
| `Loss/rec` | 1e-3 → **0.4364**（EMA 0.115） | 恢复输出离 clean 越来越远 |
| `Loss/gain` | EMA 0.0615 | 增益项在惩罚，但输给了别的项 |

**结论：S1 在 273 updates 内让恢复输出的误差从 0.055 涨到 0.241。** 这不是"还没学会"，
是"在学错的东西"。

**为什么值得警惕而不是等它自己好**：denoiser 的写回被
`write_weight · token_error · sigmoid(residual_gate)` 三重收缩，
初始 `sigmoid(-8) ≈ 3.4e-4`。要让角距离涨 0.19，`pred` 的量级必须非常大，
或者 gate 已经明显打开。两种可能：

* **(a) gate 被 Loss/rec 打开了**——`Loss/rec` 只要求"像 clean"，不要求"比输入好"，
  所以在一个随机初始化的分支上它可以把 gate 推大而不改善任何东西；
* **(b) `Loss/pres` 惩罚错误的对象**——它只惩罚 `~corruption_mask` 的 token，
  而 refiner 只改 `TopK(q)` 的 32 个 token；如果 `corruption_mask`（注入器的真值）
  与 `TopK(q)`（诊断头的判断）不重合，被改动的 token 大量落在"健康"集合里，
  于是 `Loss/pres` 上涨——两个读数同时出现正好符合这种情况。

**这必须用数据定，不能靠推理**。所需消融就是审查要求的那一个，但**理由变了**：
不再只是"看 gate -8 开不开"，而是"**哪条配置能让 `gain_total > 0`**"。

---

## 3. 四阶段设计

### 3.1 统一设置（与规格一致，且都是现状已支持的）

```text
LasHeR train 979 seq | template 112 | search 224 | DINOv2 ViT-B/14
AdamW | wd 0.1（bias/LayerNorm/embed 为 0） | cosine | lr_min 1e-6
AMP fp16，但 GOLA trunk/head 保持 fp32（现状，不动）
max_grad_norm 1.0 | micro_batch 8 | accum 16 | effective 128
samples_per_epoch 131072 | num_workers 4 | 1024 updates ≈ 1 epoch
```

### 3.2 阶段表

| | S1 Spatial Warm-up | S2 Joint PEFT | S3 DINO last-2 | S4 Temporal |
|---|---|---|---|---|
| updates | 1500 | 8000 | 1800 | 3000 |
| warmup updates | 128 | 410 | 256 | 48 |
| **trainable** | `codetrack.*` | +`lora.A/B/GA/GB`、`head`、`token_type_embed` | +`blocks.10.*`、`blocks.11.*` | +`codetrack.motion/memory/template_gate` |
| **frozen** | DINO、LoRA、head、embed、motion、memory、gate | DINO foundation | DINO blocks 0–9 | DINO 全部（含 S3 解冻过的两块，重新冻结） |
| temporal | 关 | 关 | 关 | **开** |
| sampler | pair | pair | pair | **causal clip (4 帧)** |
| micro_batch × accum | 8 × 16 | 8 × 16 | 8 × 16 | **2 × 16**（clip 显存更贵） |
| lr | codetrack 1e-4 | CT 1e-4 / LoRA 2.5e-5 / embed 2.5e-5 / head 1e-5 | CT 5e-5 / LoRA 1e-5 / embed 1e-5 / head 5e-6 / DINO-last2 1.5e-6 | motion 1e-4 / memory 1e-4 / gate 1e-4 / CT 5e-5 / LoRA 1e-5 / embed 1e-5 / head 5e-6 |
| loss | `w_track_corr 1.0, lambda_diag .5, rec .2, align .2, pres .01, gain .2` | +`w_track_clean 0.25` | 同 S2 | +`lambda_motion .2, lambda_mem .1, lambda_gate .1` |

**"禁止只改 `stage_id` 却训练相同参数"** 这条在设计上由 `parameter_scope` 保证：
每阶段的 optimizer 参数集合都不同（97 → 1406 → +DINO → +temporal），
并且验收门会断言这一点。

### 3.3 S3 的可回滚性

```text
S3 accepted  if  (SR_s3 - SR_s2) > 0  且  clean SR drop < 0.3 绝对点
else          reject → base_for_s4 = s2_best
```

S3 是"可选增强"，不是"必须完成的四个阶段之一"。这一点直接决定 S4 的初始化来源。

---

## 4. 编排器设计

### 4.1 为什么是"子进程 + 现有 harness"，而不是重写训练循环

现有 harness 已经打通：数据管线、AMP、梯度累积、optimizer/param groups、
scheduler、checkpoint、分布式、日志。S1/S2/S3 在同一 harness 内**只靠 `parameter_scope`
与 `max_updates` 区分**，因此编排器**不应该**复制一个训练循环——它只负责：

```text
prepare(stage) → spawn training subprocess → collect stage metrics
              → validate_stage() → 决定继续/回滚/停止
              → 写 checkpoint 指针 + report
```

这同时满足"每个 stage 真正不同"：不同点是**配置**（scope/budget/lr/sampler/loss），
而不是被编排器假装的 `stage_id`。

### 4.2 入口

```bash
python tools/train_codetrack_staged.py \
    --stages s1,s2,s3,s4 \
    --resume auto \
    --output-root outputs/staged \
    [--smoke]                 # 每阶段 10 updates，用于验证状态机
```

### 4.3 目录布局

```text
outputs/staged/
  checkpoints/s1/{last.pt,best.pt}
  checkpoints/s2/{last.pt,best.pt}
  checkpoints/s3/{last.pt,best.pt}
  checkpoints/s4/{last.pt,best.pt}
  reports/{stage1..4_report.json, final_report.json, stage*.txt}
  logs/{s1..s4}.log
  metrics/{s1..s4}.jsonl
  state.json          # 当前 stage、update、已完成阶段、S3 是否被接受
```

**checkpoint 内容**（编排器自己写，不依赖框架的 dump 格式）：
`model` / `optimizer` / `scheduler` / `scaler` / `stage` / `optimizer_update` /
`micro_step` / `rng`（torch/cuda/numpy/random）/ `best_metrics` / `config` / `git_commit`。

**resume auto**：读 `state.json` → 找"当前 stage 最近的 `last.pt`" →
`--weight_path` 指向它 → 恢复 optimizer/scheduler/scaler/rng。

> 注意：当前所有 stage 配置写的是 `checkpoint: resumable: false`。
> 编排器自管 checkpoint 就不需要这条，但**必须**在 smoke 测试里验证
> "kill 后 resume 到同一 update 号"。

### 4.4 分阶段数据/采样

* S1–S3：pair sampler（与现状一致，`siamese_training_pair_sampling`）。
* S4：**新的 causal clip sampler**（§6）。四种采样差异必须体现在配置里，
  验收门断言 S4 跑的是 clip 而不是 pair。

### 4.5 分阶段 checkpoint 与"最佳"

* `last.pt`：每个 update 边界覆盖写。
* `best.pt`：由**该阶段的验收指标**决定（S1/S2 用 `q_auroc_mask` 与 `gain_total`；
  S3/S4 用 SR 与属性维度）。
* 训练中途的周期性评估用**固定小验证集**（例如 LasHeR test 的 20–30 条序列），
  全量 245 条只在阶段结束跑一次。

---

## 5. 运行时验收门（`validate_stageN`）

### 5.1 公共检查

| 检查 | 判据 | 依据 |
|---|---|---|
| 参数真的动了 | `ΔH / Δdiagnosis / Δrefiner / Δdenoiser` 全部 > 0，打印数值 | 直接对比阶段首末 `state_dict` |
| 该冻结的没动 | DINO / LoRA / head 的 `max|Δ|` **严格为 0**，且 `grad is None` | 同上 |
| 无 NaN/Inf | loss、grad_norm、全部参数张量 | 现有 `tools/preflight_acceptance.py` 已有同类 |
| optimizer 覆盖与 LR 路由 | 该阶段 scope 内 100% 覆盖、0 个 scope 外参数、每个 scope 的 LR 等于配置值 | 复用 `parse_optimizer_per_params_config` |
| LR 轨迹 | 打印 step 0/10%/50%/100% 的真实 LR，断言 cosine 未提前跑完 | 日志解析 |

### 5.2 S1 专项

| 检查 | 判据 |
|---|---|
| Diagnosis 学到东西 | `AUROC(q, corruption_mask)` > 0.60（**sanity，不是论文结论**）；`AUPRC(q, corruption_mask)` 一并报告（**需新增**）；`Spearman(q, soft_error_target)` 报告 |
| q 未退化 | `q.std() > 1e-3`；`pos_mean - neg_mean > 0` |
| **Recovery 真的在修** | `d_final < d_input` 且 `gain_final > 0`（**硬门**） |
| residual gate | 打印 `raw` 与 `sigmoid(raw)`，断言 `sigmoid` 不再等于初值（即 gate 有梯度） |

> 依据 §2 的实测：**当前 S1 会 FAIL 这一条**。这正是它存在的意义。

### 5.3 S2 专项

* `CodeTrack / LoRA / head` 的 `grad > 0`；DINO `grad is None` 且 `Δ=0`。
* 固定诊断序列上的 `clean PR/NPR/SR` 与 `corrupted PR/NPR/SR`，
  与 **原始 GOLA** 和 **S1 checkpoint** 三方对比。
* 硬门：`clean SR drop < 0.3` 绝对点。
* 软门：robust SR 必须有改善，否则 **WARN** 并打印全部 recovery 指标（不停止）。

### 5.4 S3 专项

`S2 best` vs `S3 best` 的 `PR/NPR/SR`、`clean SR`、`robust SR`；
按 §3.3 决定 ACCEPT / REJECT+回滚。

### 5.5 S4 专项

| 类别 | 检查 |
|---|---|
| motion | `loss_motion > 0` 且下降；`M_t` 随历史变化；**`M_t` 与当前 GT 无关**（复用 `tools/causality_check.py` 的 6 项） |
| memory | `reliability` 的 mean/std/min/max、accepted/rejected 比例；断言值域**不**退化为 `[0.5, 0.73]` |
| 因果 | 改未来帧不影响过去输出；改 `t-1` 影响 `t` 的 prior |
| tracking | 总 `PR/NPR/SR` + 属性维度 `FM/PO/TO/LI/TC` + 长序列子集 |
| 决策 | 无收益 → **不允许**把 S4 作为 final model |

---

## 6. S4 需要新建的部分（最大工作量）

### 6.1 causal clip sampler

* 采样 `clip_length=4` 的**连续**帧 `{t-3, t-2, t-1, t}`，同一个 sequence、时间递增。
* 每个 clip 开始时调用 `GOLA_DINOv2.reset_sequence()`（方法已存在，**从未被训练路径调用**）。
* 序列边界绝不跨界：sequence A 的 memory 不得进入 B。
* 数据侧需要新的 `sampler` 类型（当前只有 `random`）与新的 collator 形态
  （batch 维度从"样本"变成"clip"，clip 内是帧序列）。

### 6.2 TBPTT

* `memory_tbptt_steps`（字段已存在，**从未被读取**）接进
  `TemporalMemory` 的写入路径：每 N 帧切断一次梯度，而不是无条件 `mem.detach()`。

### 6.3 D4 / D5 / GT 泄漏

* **D4**：`cur_rel = sigmoid(mean(1-q))` → 值域 `[0.50, 0.73]`。改为可直接覆盖 `[0,1]`
  的形式（如 `1 - mean(q)` 或 `1 - mean(TopK(q))`）。
* **D5**：每个 memory slot 存**它自己的** reliability label；禁止用当前帧 trust `expand`。
* **GT 泄漏（未来帧）**：`target_mask` 来自当前 `gt_box_xywh`。read-before-write 保证
  不影响当前输出，但它被写进 memory 后会影响**未来帧** → train/test mismatch。
  S4 前必须改为"上一帧预测/模板 mask"或"模型自预测 mask"。

### 6.4 scheduled sampling

只作用于 **Kalman observation** 与 **memory state update**，**不改** SiamFC crop：
`teacher forcing prob = 1.0 (0–20%) → 线性降到 0.3 (20–70%) → 0.3 (70–100%)`。
把 `scheduled_sampling_prob`（字段已存在，**从未被读取**）真正接进观测选择。

---

## 7. 需新增/修改的文件清单（预估）

| 文件 | 动作 | 说明 |
|---|---|---|
| `tools/train_codetrack_staged.py` | **新增** | 编排器：状态机、子进程、resume、报告 |
| `tools/stage_validate.py` | **新增** | `validate_stage1..4` 与公共检查 |
| `tools/stage_metrics.py` | **新增** | AUPRC、属性维度聚合、JSONL 写入 |
| `config/GOLA/codetrack_s3/config.yaml` | **新增** | DINO last-2 scope |
| `config/GOLA/codetrack_s4/config.yaml` | **新增** | clip sampler + motion/memory/gate scope + temporal loss |
| `config/GOLA/_mixin/*.yaml` | 扩展 | smoke 覆盖（10 updates）等 |
| `codetrack/criteria.py` | 改 | 新增 `_auprc_against_mask` |
| `codetrack/motion.py` | 改 | D4 值域、D5 每 slot 标签、TBPTT |
| `codetrack/codetrack.py` | 改 | 因果顺序与 GT mask 来源 |
| `trackit/.../data/sampler/` | **新增** | causal clip sampler |
| `trackit/runner/training/default/__init__.py` | 改 | 每 clip `reset_sequence()`、scheduled sampling 接线 |
| `config/GOLA/_dataset/` | 改 | clip 训练集入口 |

---

## 8. 完整启动命令

```bash
source scripts/00_env.sh

# 状态机 smoke（每阶段 10 updates，验证 freeze/unfreeze/optimizer 重建/
# scheduler/checkpoint/resume/temporal reset/validation 全部真跑）
"$PYTHON" tools/train_codetrack_staged.py --stages s1,s2,s3,s4 --smoke \
  --output-root outputs/staged_smoke

# 正式四阶段
"$PYTHON" tools/train_codetrack_staged.py --stages s1,s2,s3,s4 --resume auto \
  --output-root outputs/staged
```

---

## 9. 阻塞项（按优先级，未完成前不给 READY）

| # | 阻塞项 | 为什么阻塞 | 验收方式 |
|---|---|---|---|
| **B1** | **S1 的 `gain_total` 为负**（实测 −0.16，273 updates） | 编排器会把一个"损害表征"的 S1 自动化，S2 起点就是坏的 | -8/-5（必要时再加 `Loss/pres` 目标修正）对照短训达到 `gain_total > 0` |
| **B2** | 无 causal clip sampler / `reset_sequence` 未接线 | S4 没有可训练的时序输入，规格明确禁止用 pair 假装 temporal | clip sampler 单测 + S4 smoke 中 memory 跨序列不串 |
| **B3** | D4 / D5 / GT-mask 泄漏未修 | 时序模块会学到错误目标 | 值域断言 + 每 slot 标签断言 + 因果检查 |
| **B4** | TBPTT 与 scheduled sampling 未接线（字段存在但从未被读取） | S4 的跨帧信用分配与 exposure bias 处理是空的 | 断言梯度确实跨 N 帧 |
| **B5** | AUPRC 缺失 | 规格要求的诊断判据之一 | 单测 + S1 报告含 AUPRC |
| **B6** | PR/NPR/SR 未接入编排器 | S2/S3/S4 的核心决策依据 | 一次 20 序列子集评估产出 PR/NPR/SR |
| **B7** | best-checkpoint / resume auto / stage report 未实现 | 规格的第二十四节 | smoke 里 kill→resume 回到同一 update 号 |
| **B8** | S3 自动回滚未实现 | 规格要求 S3 可被拒绝 | smoke 中人为让 S3 变差，验证回滚到 S2 |

**B1 是唯一"必须先用实验定"的项，其余是实现工作量。**

---

## 10. 显式假设与非目标

* 不重写框架训练循环；编排器是**子进程**层。
* 不改 GOLA trunk/head 的 fp32 处理（规格明确要求保留）。
* 不改 SiamFC crop 生成逻辑（scheduled sampling 只作用于卡尔曼观测与 memory 状态）。
* 属性维度 `FM/PO/TO/LI/TC` 的具体聚合方式待定（LasHeR 属性文件已就位，
  但"哪些属性算 robust"需要一次讨论——这会直接影响 S4 的 ACCEPT/REJECT 判据）。
* 单卡 4090D 24GB / 80GB 内存；S4 的 `2×16` 是安全起点，实测 peak 后再谈 `4×8`。
