# GPU2 SATR 机制诊断

## 对象

- Checkpoint: `_probe/full_joint_satr/GOLA-codetrack_full-2026.10.05-13.24.00-843554/checkpoint/epoch_05/model.bin`
- 全量 LasHeR-test: GOLA `PR 77.19 / SR 61.79`; GPU2 SATR `PR 77.30 / SR 61.69`
- 逐帧诊断: `_probe/diag_gpu2_satr10.jsonl`，13,454 帧
- 典型序列诊断: `_probe/cf_base.jsonl` 和 `_probe/cf_base_prepost.jsonl`

## 内部行为

GPU2 的 q 绝对值很低，但当前路径先做帧内标准化，再固定取 q 最高的 32 个 token。于是即使整帧没有足够的错误证据，也会执行 32 个位置的 SATR 写回。10 条序列平均 `q_mean` 约为 `0.00174`，但 `selected_fraction` 始终为 `12.5%`。`Delta X` 的平均范数约为 `0.85`，所以低概率并没有成为“不修复”的条件。

## 属性和反事实结果

全量属性差异显示：TC、LI、HO、FL 平均受益；OV、DEF、MB、CM、TO 平均退化。6 条典型序列前 120 帧的反事实结果如下：

| 设置 | SR | PR |
| --- | ---: | ---: |
| GPU2 SATR | 67.40 | 87.92 |
| 关闭 SATR | 70.00 | 90.97 |
| 关闭 BP | 68.00 | 87.64 |
| 关闭历史 | 67.32 | 87.78 |
| 关闭 Kalman | 68.08 | 88.33 |
| 关闭 RGB 旁信息 | 70.46 | 91.39 |
| 绝对 q 门控，τ=0.01 | 70.37 | 90.97 |
| 绝对 q 门控，τ=0.02 | 69.97 | 90.97 |
| 绝对 q 门控，τ=0.03 | 70.00 | 90.97 |

在同一套 10 条代表序列上，τ=`0.01` 为 `PR 75.29 / SR 58.22`，低于原始 GPU2 SATR 的 `PR 76.48 / SR 58.68`，因此该阈值不能直接进入最终配置。6 条截断序列上的提升不足以代表完整序列，说明门控还需要结合目标出界状态和时间连续性。

典型序列：

- `carcominginlight`（LI/TC）：关闭 SATR 后 SR `+14.05`，关闭 RGB 后 SR `+17.78`。当前跨模态差异把照明变化当成错误，SATR 发生过修复。
- `blkhairgirltakingblkbag`（DEF）：关闭 BP 后 SR `+2.62`，关闭 SATR 后 SR `+1.78`。固定 Tanner 关系把形变产生的局部特征变化传播成错误位置。
- `blkboyhead`（FL/TC）：关闭 Kalman 后 SR `-6.86`、PR `-8.33`，运动先验对快速运动确实有贡献。
- `whitecarturn683`（OV）：全量结果 GOLA SR `64.95`，GPU2 SATR SR `29.00`。目标接近边界时 q 仍低且 TopK 仍强制修复，可靠历史/运动并没有触发“目标已失效”的拒绝状态。

## 可视化

每张图按顺序显示：syndrome、q、TopK、`Delta X` 范数、修复前响应、修复后响应、响应变化。

- [carcomingfromlight](mechanism_figures/carcomingfromlight.png)
- [hyalinepaperfrontface](mechanism_figures/hyalinepaperfrontface.png)
- [blkboyhead](mechanism_figures/blkboyhead.png)
- [carcominginlight](mechanism_figures/carcominginlight.png)
- [blkhairgirltakingblkbag](mechanism_figures/blkhairgirltakingblkbag.png)
- [whitecarturn683](mechanism_figures/whitecarturn683_frame260.png)

## 结论

SATR 当前有效的条件是：目标仍在视野内，运动先验集中，另一模态对目标区域提供稳定互补，且 syndrome 有较强绝对证据。它在快速运动、热交叉、重遮挡中可以把可靠空间和时序冗余传给受损 token。

它误修的条件是：模态差异本身是正常变化（低照度、照明变化、形变），或者目标已接近出视野。当前实现把“相对排名”误当成“绝对错误概率”，所以会在低 q 的正常帧上固定写回 32 个 token。

## 最值得改的两点

1. 把恢复路由改为绝对证据门控：`route = TopK(q) AND q >= tau`，并加入全帧 `no-error` 状态；τ 应在验证集校准，正常帧允许零 token 写回。
2. 给 OV 和 DEF 增加专门的拒绝/降权逻辑：Kalman 的边界不确定度或出界概率高时禁止历史驱动的写回；syndrome 在跨模态一致但形变较大的区域时降低 Tanner 传播权重，避免固定 H 把形变扩散成 corruption。

下一步应先在 10 条代表序列上比较 `tau`、边界拒绝和形变降权的组合，再决定是否用单阶段全量训练；当前不应直接用 GPU2 checkpoint 作为最终模型。

## 边界拒绝快速结果

在 10 条代表序列上，加入 Kalman 预测框边界拒绝后的结果为：

| 边界 margin | PR | SR |
| --- | ---: | ---: |
| 无拒绝 | 76.48 | 58.68 |
| 0.05 | 76.64 | 58.77 |
| 0.10 | 74.91 | 57.85 |
| 0.15 | 74.96 | 57.84 |

因此边界 margin 不应直接自由学习。推荐的最终形式是：

`edge_risk = geometric_out_of_view_constraint`

`r_t = sigmoid(g_phi(q statistics, response peak, motion uncertainty, history reliability))`

`route_i = TopK(q)_i * 1[edge_risk == 0] * 1[r_t > tau]`

其中边界约束保持硬规则，`r_t` 才由训练学习。这样可以学习当前帧是否值得译码，但不能学习出“目标已经出视野仍允许历史修复”的不安全策略。`tau` 先在验证集校准，之后再考虑端到端微调。
