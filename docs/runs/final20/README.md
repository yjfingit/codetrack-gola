# Final20 实现与运行记录（2026-10-08，上海时间）

> 最新配方见 [FRAME_COVERAGE_RECIPE.md](FRAME_COVERAGE_RECIPE.md)：物理 GPU 3/4 两卡、连续阶段每卡 4 个 clip、每 4 轮验证、每轮遍历训练部分全部搜索帧。下文保留早期三卡随机采样实现的历史记录，其 batch、采样、启动命令和运行状态均不能作为当前配置依据。

## 实际状态

已完成一次 GPU 2/3/4 的三卡 DDP **pair-only** 20 轮训练。881 个训练序列与 98 个验证序列不重叠，`epoch_00`–`epoch_19` 的权重保留在：

`outputs/final20_retry5/GOLA-codetrack_final20-2026.10.07-16.35.19-766307/checkpoint/`

`PAIR_RUN_AUDIT.json` 记录文件大小、SHA256 和不符合计划的项目。该运行没有连续 clip curriculum、TBPTT、四帧分叉收益、新 quality head、OPE 验证选模或可恢复 optimizer 状态。不能将这些旧权重描述为修正版训练成果。

此前测试均已停止，不作为性能证据：原 baseline 配置选择了不匹配的 merged inference loader；旧 candidate 的动态 batch 状态未按序列身份隔离。LasHeR-test 已有部分测试暴露，不能再称为完全未接触的测试集。没有有效的“超过 GOLA-B”结论。

## 已修正的入口

`main.py GOLA codetrack_final20` 现在派发到 `tools/train_final20_causal.py`，不会再运行随机 pair application。旧配置留在 `config/GOLA/codetrack_final20_pair/config.yaml` 供追溯。独立配置在 `config/GOLA/codetrack_causal20/config.yaml`。

新实现包含：

- 同一目标的连续 8/16 帧真实图像 clip；每个 clip 重置状态，跨四帧保留 memory 计算图，在 TBPTT 边界 detach。
- epoch 1–4 空间 pair，epoch 5–12 8 帧，epoch 13–20 16 帧；历史预测使用概率分别从 .25 到 .75、从 .75 到 1；当前 GT 不进入预测。
- 四帧 utility 是从同一状态独立分叉的 no-op/candidate rollout 的 signed tracking-loss 差，两个分支各自推进 crop、template、memory 和 motion；目标生成恢复主轨迹状态及 RNG。
- quality head 输入当前特征、初始/在线模板上下文、q 和 motion uncertainty；输出即时/未来 utility、motion consistency、admission risk 和 no-op/provisional/commit logits。
- soft feature gate 参与训练；部署用硬状态门控。provisional 不提交 appearance bank、模板、crop 或 motion measurement；确认来自下一帧基线观测与上一帧全图 proposal 的几何一致性。
- 空间 Top-K route 在训练/推理中一致，abstention 由学习的三状态 head 实现，避免 q 固定阈值让所有四帧标签变成 no-op。
- baseline/candidate tracking、diagnosis、recovery、identity-preservation、utility/gate、全图 temporal displacement 和 GOLA orthogonal regularization；LoRA/SATR dropout=.1，AdamW，非 bias/norm 权重衰减=.1，梯度裁剪=1。
- 每 epoch 的 checkpoint 包含完整模型（含冻结主干/几何 buffer）、optimizer、scheduler、每 rank RNG、配置和 split manifest。恢复时检查配置、batch、split 与 world size。
- 每轮在 98 个验证序列做完整、按序列 reset 的官方 OPE；选择满足 PR/SR 不低于基线且 false-write rate ≤ .1 的权重，按 PR+SR、false-write、clean-no-write 排序。没有合格权重时禁止最终测试。
- 最终测试使用相同 training-class loader 与 crop/head/template 规则，按完整序列分三卡。只允许 20 个完整可恢复 epoch 且 validation-selected 的权重进入 test。

## Batch、采样与启动

GPU 映射为 rank 0/1/2 → 物理 GPU 2/3/4。使用实际 DDP，任意 rank 异常由 torchrun 停止全组；不自动重启训练。

推荐 `--local-clips 7`：空间阶段每卡 28 个 pair（global 84）；连续阶段每卡 7 个 clip（global 21），每次四帧反传包含 84 个图像观测。显存需以最新 smoke 为准；这是 clip batch，不能直接与旧 pair batch 相比。

每 epoch 每个训练序列随机访问 16 次，等长 DDP shard 用重复样本 padding，每轮所有 881 个序列都会被访问。这是**全序列集合采样**，不是逐帧无遗漏扫描整套 LasHeR 视频。padding、采样次数与更新数写入 protocol。若“全量”要求每轮每一张原始帧都被训练，需要另一个采样协议。

完整工作流（训练、验证选模、固定权重后两臂完整测试、比较报告）：

```bash
bash scripts/train_causal20.sh
```

保留用户原训练入口：

```bash
source scripts/00_env.sh
unset NCCL_P2P_DISABLE
CUDA_VISIBLE_DEVICES=2,3,4 "$PYTHON" -u main.py GOLA codetrack_final20 \
  --distributed_nproc_per_node 3 --distributed_do_spawn_workers \
  --disable_wandb --weight_path weights/gola_b224.bin \
  --output_dir outputs/causal20_full --local_clips 7
```

这个入口训练并逐轮验证；完整测试由 `scripts/train_causal20.sh` 编排。日志使用 tee 持久保存，文件描述符上限提高到 65535。这里保留显式启动命令，尚未执行第二次完整 20 轮训练，因为用户最初明确说“只有一次机会”。

## 验证证据与限制

- `outputs/causal20_checks/smoke_admission.log`：三卡真实 spatial/8-frame/16-frame optimizer 更新、各模块非零梯度、实际显存峰值；仅为 smoke。
- `outputs/causal20_checks/main_dispatch.log`：用户原入口与 worker spawn 的三卡集成检查。
- `outputs/causal20_checks/ddp_metrics.log`：CPU/GPU 指标在各 rank 可选 key 不同的情况下归约成功；缺失项不稀释有观测 rank 的平均值。
- `outputs/causal20_checks/witnesses_final/report.json`：真实验证前缀的 no-op 与 GOLA 逐点相等；分叉恢复 state/RNG、当前标签隔离、memory 跨帧梯度、冻结主干不变和 checkpoint 严格重载。

测试拒绝 partial/duplicate sequence coverage。最终成功要求 PR、SR 严格超过 baseline，同时 N-Precision 降幅不超过 1 pp、任一序列 SR 降幅不超过 20 pp；这些额外限值为预先固定的工程接受标准。false-write/contamination 由 offline GT 对比定义，不是统计置信保证。尚未执行修正版完整训练、98 序列全量逐轮验证或有效 LasHeR-test 对比；smoke 不能证明泛化提升。
