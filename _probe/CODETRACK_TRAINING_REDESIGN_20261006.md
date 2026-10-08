# CodeTrack 训练范式重设计

## 结论

当前实现不能证明“发现会影响跟踪的坏 token，再把它修好”。它主要学习了“人工注入后哪些特征发生变化”，而不是“哪些 token 会让目标跟踪变差”。原因有四个：

1. `GOLA._forward_codetrack` 用 clean/corrupt 特征的余弦差作为主要 `error_target`，这个差异不等价于跟踪损失或框位置误差。
2. 随机 pair 训练每次都会清空 `CodeTrack` 状态，Kalman 和 memory 没有连续训练信号；训练中的 box 观测又来自 ground truth，推理时来自上一帧预测，存在条件错配。
3. `CodeTrack.forward` 对 q 做帧内标准化并固定选 Top-K，正常帧不能选择 K=0；因此 q 的绝对概率没有真正控制是否写回。
4. 当前 H 是随机稀疏、非负、行归一化的 token 图。它可以路由信息，但没有定义健康 RGB-T 表征为什么满足 `H x = 0`，所以 BP 的 parity 语义不完整。

之前的结果与此一致：q 在固定合成损坏上可以取得约 0.74 的 AUROC，但 oracle 路由仍可能让恢复变差；完整测试最好的组合只有 `PR 77.34 / SR 61.76`，GOLA 为 `77.19 / 61.79`。两轮外层“syndrome 下降 + 峰值不下降”在 6 条序列上降到 `PR 57.95 / SR 47.54`，不能保留。

## 坏 token 的定义

坏 token 不是“与 clean feature 距离大”的 token，而是“替换为可靠参考后，目标跟踪的可观测损失明显下降”的 token。对 search token `i` 定义一个因果影响近似：

```text
d_i = relu( grad_{X_i} L_track(X_in) dot (X_i - X_i^*) )
```

其中 `X_i^*` 是可靠参考，`L_track` 是冻结 GOLA head 对当前 ground-truth box 的分类和框回归损失。`d_i` 在构造标签时 stop-gradient。训练标签是

```text
y_i = sigmoid((d_i - median(d)) / (mad(d) + eps))
```

只有 `d_i` 高于验证集校准阈值、且目标可见时才监督 q；目标出视野、两路观测都不可靠或运动创新过大时标为 unknown，忽略 q 损失。这让 q 学习“跟踪影响”，而不是学习 injector 的纹理。

## 三路可靠参考和视觉校验

保留 RGB、TIR、融合三路，但不再把 RGB 直接当成 TIR 的 codeword。对每个 search token 建立三个观测：

```text
v_i = P_v(RGB_i)
i_i = P_i(TIR_i)
h_i = warp(P_i(TIR_{t-1}), Kalman prediction)
```

`h_i` 只使用上一帧状态和运动预测，不能使用当前 ground truth。模板特征提供目标身份权重，限制校验主要发生在目标候选区域。

H 改成固定、有符号、可解释的局部校验，而不是随机非负图：

1. RGB/TIR 观测差：`r_vi = v_i - i_i`；
2. TIR/历史观测差：`r_ih = i_i - h_i`；
3. 16x16 网格的局部 2x2 cycle check，使用 `(+1,-1,-1,+1)` 的有限差分系数；
4. 每类 check 的均值和协方差只用健康训练帧估计，syndrome 是白化后的 Mahalanobis 能量。

这样定义的是“健康目标区域的多路冗余残差”，不是声称任意 DINO token 天然满足一个随机线性码。一个 check 只有在 RGB、TIR、历史冗余同时不一致时才产生高置信异常；单独的低照度或形变不应自动等价于 corruption。

BP 只做两轮。H 的支持在训练中固定，前期冻结边权，防止 H 学成任意异常 MLP；只有 q 在独立验证集通过后，才允许很小学习率调整边权。

## SATR 写回规则

q 只代表 token 错误概率，路由单独决定是否译码：

```text
route_i = 1[q_i > tau_q]
          * 1[target_visible]
          * 1[at_least_two_references_agree]
```

`tau_q` 在 train-validation split 上校准，不能用帧内标准化替代。允许整帧 `K=0`，Top-K 只作为最大数量保护，不是强制选择。路由中的 q 使用 detach，避免 SATR 通过改变 q 来逃避恢复损失。

修正量只由另一模态、可靠 Tanner 邻居、可靠历史和运动先验生成：

```text
X_i' = X_i + route_i * g_i * q_i * Delta_i
```

正常 token 走严格 identity path。第二轮只有在同一 token 的白化 syndrome 下降、head 的分类/框损失不变差时才写回；如果没有可靠证据，直接保留第一轮或原始 GOLA 特征。删除 denoiser，不增加 diffusion 或额外注意力模块。

目标出视野时使用现有信息做硬拒绝：上一帧得分低、Kalman innovation 超阈值、预测框与搜索区域交叠不足任一成立，就 `route=0`，禁止历史写回和 template/memory 更新。这样历史只能帮助重新出现，不能凭空制造目标。

## 人工和天然退化的分工

人工 corruption 只提供密集 token 标签和可重复的单元测试，不代表真实数据分布。每个 batch 的来源固定为：

- 40% 原始 LasHeR 帧，不注入 corruption；
- 20% TIR-only：遮挡、模糊、噪声、局部丢失混合；
- 20% RGB-only；
- 20% 双模态或时序错位。

注入位置、形状、强度和类型随机化，训练和验证使用不重叠的 corruption 组合。人工 mask 的 q 监督权重最多占 q 损失的 0.25。

天然 LasHeR 退化负责真实跟踪目标和泛化：低照度、热交叉、遮挡、形变、模糊等帧只使用因果影响标签或两路一致性产生的高置信伪标签。RGB、TIR、历史三者不能形成多数一致时，不强迫模型判断，直接作为 abstain 样本。

## 两阶段训练

### 阶段一：可靠性诊断

- 冻结 DINOv2、GOLA LoRA、GOLA head、SATR、motion、memory；固定 H 支持和边权。
- 只训练 `P_v/P_i`、symbol encoder、2 轮 BP 的归一化/阻尼参数。
- 使用单帧 paired view，随机 pair 可以保留，因为本阶段不使用状态。
- 损失只有 `L_q`：高置信因果影响标签的 BCE，加一个小的 batch 内 pairwise ranking；unknown token 不计入。
- 阶段结束条件：合成未见 corruption AUROC >= 0.65，天然退化伪标签 AUROC 高于随机，干净帧 q 的假阳性率 < 5%，且 q 的均值不通过 bias 漂移。

### 阶段二：因果纠错和连续跟踪

- 载入阶段一 checkpoint。
- 冻结 DINOv2、GOLA LoRA 和 head；训练 SATR、template/memory 的小投影、motion prior 的空间形状头。BP 诊断默认 detach，后半程只以 0.1 倍学习率解冻 symbol encoder。
- 数据改为同一 LasHeR 序列的连续 clip，长度 4，首帧用真实初始化框，后续帧只使用上一帧预测框和置信度。训练中不再用当前帧 ground-truth 驱动 Kalman 或 memory。
- clip 中 75% 不注入 corruption，25% 使用阶段一的混合 corruption；这样时序模块主要学习天然退化，而不是只追踪人工 mask。
- 只有三项主损失：

```text
L = L_track(X_out)
  + 0.2 * L_q
  + 0.2 * L_rec_bad
  + 0.05 * L_id_healthy
  + 0.2 * L_gain
```

`L_rec_bad` 只在有可靠 clean/多路参考的坏 token 上使用 cosine/Huber；`L_id_healthy` 约束 q 低的 token `X_out-X_in`；`L_gain` 是 `relu(L_track(X_out)-L_track(X_in)+margin)`，同时作用于合成坏帧和天然高置信退化帧。这样特征靠近 clean 只是必要条件，真正的成功还必须转化为 tracking loss、分类峰值和框结果改善。

每个阶段都使用验证集早停：连续 3 次验证没有改善就停止，保存同时满足 tracking、q、identity 三个门的 checkpoint。不能只按 reconstruction loss 选模型。

## 必须完成的验证实验

1. **q 是否找对**：未见合成 corruption 的 token AUROC/AP、四个空间象限的 Top-k recall、原始干净帧的写回率；再按 LasHeR `TC/LI/HO/FL/DEF/MB/OV` 分组统计天然伪标签。
2. **SATR 是否修好**：四象限和 RGB-only/TIR-only 的 oracle q 与 learned q 对比；报告坏 token 的 `d_before -> d_after`、syndrome 能量、q 排序。oracle q 也不能改善时，淘汰 recovery，不再调学习率。
3. **healthy 是否保持**：干净帧和 q 低 token 的相对特征漂移、写回数量、head score 变化；目标是多数干净帧完全 `K=0`，而不是固定 Top-K。
4. **是否转化为跟踪**：同一 clip 同一初始化下比较 `K=0`、q shuffle、BP off、SATR off、motion off、memory off、RGB side-info off；记录每个序列和官方挑战属性的 PR/SR 差。接受条件是完整 LasHeR-test 的 PR 和 SR 都不低于 GOLA，且没有 OV/DEF 的显著回退。
5. **因果顺序检查**：训练模式和评测模式分别打印 memory admission、Kalman innovation、route fraction、q mean/std、healthy drift；确认没有 ground-truth box 进入当前帧修复路径。

## 实施顺序

先删掉旧的多项 recovery/syndrome 辅助损失和随机 H 边权学习，加入因果影响标签、固定 signed H 和真正的 K=0 路由；阶段一用 10 条训练序列做诊断门。阶段一通过后再实现连续 clip 阶段二。任何阶段一或 oracle recovery 门失败，都不启动全量训练。

## 2026-10-06 实验反馈

- 阶段一混合退化诊断（3 个空间区域训练，右下区域验证）通过：未见区域平均 AUROC 约 `0.90`；RGB 局部置零、TIR 噪声、TIR 模糊的 AUROC 分别约 `0.866 / 0.976 / 0.848`；干净帧 q 的 95 分位约 `0.303`。
- 阶段二固定特征恢复在按序列划分的验证上通过：oracle 路由特征增益 `0.00460`，learned q 增益 `0.00441`，healthy drift 约 `1.3e-5`。按空间留一块验证只有 `0.0007`，说明随机 H 的空间泛化仍弱，不能使用空间留出作为唯一门。
- 把阶段一二部分权重直接覆盖到完整 GPU2 模型后，10 条闭环评测只有 `PR 59.12 / SR 48.30`，与原始阶段部分版本相同，说明固定特征恢复并未转化为闭环跟踪收益。后续必须加入连续 clip 和 head/框级损失，且要用完整 CodeTrack checkpoint 合并，不能只加载部分模块。
- 新增的跨模态残差支路最后一层已经改为零初始化，以保证旧 SATR checkpoint 加载时严格 identity；只有阶段二训练后该支路才会产生非零修正。

## 连续 clip probe 反馈

- 已加入 `preserve_state=True`：训练 pair 仍逐次 reset，连续 clip 可以跨帧保留 Kalman/memory 状态；初始化检查通过。
- 独立 4 帧连续 probe 在 oracle q 下有效：`TopK=8` 时恢复增益约 `0.385`，但 healthy drift `0.28`；`TopK=4` 时增益约 `0.0055`，drift 降到 `0.134`。这说明固定大 TopK 是主要误修源，但简单减小 K 会同时损失恢复能力。
- 连续 clip 上重新训练诊断后，learned q 仍只有小幅收益，绝对 q 阈值 `0.25/0.4` 没有明显改变结果。当前 q 的概率校准和目标区域标签仍不匹配，不能进入完整训练。
- `route_q_override` 已加入 CodeTrack，仅用于严格区分 oracle 路由和 learned 路由的 probe，不参与生产 checkpoint。
- 连续 clip q 排序和预算约束后，learned route 的 q 均值约 `0.038`、激活比例约 `2.9%`，但 healthy drift 仍约 `0.135`；将运动图加入 route score 后 drift 反而升到约 `0.175`，因此不保留 motion route 加权作为默认策略。当前最值得改的是 q 的校准和目标影响标签，而不是继续增大 SATR。
- 进一步加入 q posterior mass 约束后，q 均值仍约 `0.036`，误修没有实质下降。说明问题不是 q 的全局均值，而是少量高 q 假阳性的排序；下一版应直接用“替换该 token 后 tracking loss 是否下降”的因果影响标签监督 q，并在 SATR 路由上加入逐 token 的 tracking-gain gate。
- 因果 token probe 已实现：逐 token 替换为 clean token，测 frozen GOLA head 输出误差下降，q 排序 AUROC `0.751`。但将该 q 直接用于 SATR，learned-route 验证增益接近 0，而 oracle route 仍有小幅正收益。这证明“发现坏 token”和“该 token 的修正方向是否可靠”是两个独立问题，下一步必须增加 token-level repair-gain gate，监督候选 `Delta X` 通过 head loss 后再写回。
- token-level repair-gain gate probe 已验证：候选 `Delta X` 的平均收益为负，但只接受逐 token frozen-head error 下降的候选后，head error 从 `0.773155` 降到 `0.772362`，净收益 `+0.000793`，正向候选约 `12.5%`。这支持“检查 → 修复 → 再检查 → 只写回正收益 token”的最终闭环。
- 将该 token gate 直接用于 6 条真实闭环评测后，结果降到 `PR 51.57 / SR 42.05`，说明当前帧 head 峰值检查不能作为逐 token 的真实目标定位判据；该开关保持关闭。下一版需要用 candidate box/response map 的目标区域损失，而不是全图峰值，训练一个真正的 repair-gain gate。
- 将 gate 改为比较原始预测 argmax 格子的响应和框几何后，6 条闭环仍为约 `PR 51.57 / SR 42.04`，与全图 gate 一样失败。即时推理反事实会破坏 tracker 状态，不能作为最终 gate；repair-gain 必须在训练阶段学习成稳定的 token gate，再冻结部署。
- 强制把原始 GOLA argmax 格加入 Tanner 路由后，6 条闭环为 `PR 57.76 / SR 47.01`，仍明显低于基线。目标位置硬接入也会把错误修复写到当前错误预测位置，不能作为默认方案；必须训练目标位置级 repair-gain gate。
- 目标格 repair-gain gate 训练 probe 的留出 AUC 约 `0.70`，但各阈值下被接受候选的平均收益接近 0，说明当前 SATR 产生的 `Delta X` 本身没有稳定的目标格修复方向。继续堆 gate 无意义，必须先改 SATR 的修复目标/条件，让候选修正具有目标格方向性。
- 进一步检查 causal q：clean head 目标格 q 几乎为 0，而高 q token 分布在其它位置，说明输入 corruption 经 GOLA transformer 传播后，影响 token 与最终目标响应格并不局部重合。SATR 当前能修一部分传播误差，却没有学会把它映射回目标响应；强制加入目标格也会退化。因此下一步应把修复目标从 token cosine 改成目标响应区域/框损失，并允许 Tanner 消息跨传播路径聚合，而不是只修原始局部 token。
