# CodeTrack 文献检索与方案映射

## 检索来源

本轮使用 arXiv 公共接口和论文原文摘要，重点检索 syndrome loss、RGB-T reliability、out-of-view tracking、selective tracking 和 uncertainty-aware tracking。

## 直接相关论文

### 1. Learning from the Syndrome

- Lugosch and Gross, arXiv:1810.10902, 2018。
- 论文提出 syndrome loss：不需要知道发送码字，也可以惩罚解码结果不满足码约束。
- 关键启示：训练不能只监督“q 是否像损坏标签”，还要监督纠错后的表示是否重新满足校验关系。
- 对 CodeTrack 的对应改法：计算修复前和修复后的跨模态 check residual，加入

  `L_post_syndrome = mean(check_residual_after)`

  和正常帧的保真约束。只有修复后 syndrome 下降，纠错才允许获得奖励。

### 2. Unsupervised Learning for Neural Network-based Polar Decoder via Syndrome Loss

- Teng and Wu, arXiv:1911.01710, 2019。
- 论文利用代码结构知识进行无标签 syndrome 训练，强调不同码结构需要不同的 syndrome 约束，不能只把 syndrome 当作普通异常分数。
- 对 CodeTrack 的启示：H 的意义必须进入训练目标；当前 q 的 BCE 监督还不足以保证修复后满足 H 关系。

### 3. FANet: Quality-Aware Feature Aggregation Network for Robust RGB-T Tracking

- Zhu et al., arXiv:1811.09855, 2018。
- 核心是显式估计 RGB/TIR 的质量，并根据可靠度自适应聚合，抑制低质量模态噪声。
- 对 CodeTrack 的启示：`U-R` 大不等于错误。低照度、形变、热交叉可能是正常模态差异。应分别估计 RGB 质量、TIR 质量和跨模态冲突，不能只用一个 q 驱动修复。

### 4. Self-Supervised RGB-T Tracking with Cross-Input Consistency

- Zhang and Demiris, arXiv:2301.11274, 2023。
- 论文用不同输入构造跨输入跟踪结果，并对低质量样本进行重加权，避免低质量样本主导训练。
- 对 CodeTrack 的启示：低质量 RGB/TIR 样本应降低损失和更新权重；当前训练中合成损坏和自然模态差异混在同一个 q 目标里，容易把自然变化学成错误。

### 5. Dynamic Attention guided Multi-Trajectory Analysis for Single Object Tracking

- Wang et al., arXiv:2103.16086, 2021。
- 针对遮挡和出视野，维护多个候选轨迹，最后根据整段历史选择轨迹，而不是每一帧强行相信单一路径。
- 对 CodeTrack 的启示：OV 时不能让单个 Kalman 预测强行驱动修复；至少要保留“继续跟踪”和“目标暂时不可见”两个假设，只有重新出现证据足够强时才恢复写回。

### 6. Flow Guided Short-term Trackers with Cascade Detection for Long-term Tracking

- arXiv:1909.00319, 2019。
- 用 tracker result judgement 判断短期跟踪是否可靠，再调用检测器重新捕获出视野或长时间丢失目标。
- 对 CodeTrack 的启示：应增加轻量目标存在性判断；出界风险高时 SATR 进入 abstain 状态，保留基线输出或等待重新检测，而不是继续写历史 token。

### 7. Selective Mask Propagation for Multi-Object Tracking

- Holmberg, arXiv:2606.13033, 2026。
- 只在不确定性触发时调用昂贵的传播模型；弱或矛盾证据时保留基础跟踪器输出。
- 对 CodeTrack 的启示：SATR 应是 selective refinement，而不是每帧 TopK 写回。当前实现的固定 Top-32 与该原则相反。

### 8. Uncertainty-Guided Inference-Time Depth Adaptation for Transformer-Based Visual Tracking

- arXiv:2602.16160, 2026。
- 根据不确定性动态改变推理深度，简单帧走短路径，困难帧才增加计算。
- 对 CodeTrack 的启示：可以让 K、SATR 轮数和是否译码由帧可靠度决定；默认 K=0，证据充分时才启用 K=8/16/32。

## 对当前 CodeTrack 的结论

当前 GPU2 SATR 的主要问题与上述文献高度一致：

1. q 的绝对证据很低，但帧内标准化后仍固定写回 32 个 token；
2. RGB/TIR 的自然差异被当成 corruption；
3. Kalman/历史只有单一假设，OV 时会把旧目标强行投射到当前帧；
4. 训练监督了错误位置，却没有明确要求“修复后 syndrome 下降”；
5. 当前模板门已经存在，但没有同时作为经过校准的 SATR dispatch 信号训练。

## 下一轮优先方案

### A. 后验 syndrome 约束

在 `X_final` 上重新计算 RGB/TIR check residual，加入：

`L_post = ReLU(s_after - rho * s_before)`

其中 `rho < 1`。正常帧使用零写回保真损失；修复只有在 syndrome 下降时才被奖励。

### B. 选择性译码

保留现有硬边界拒绝和帧级门，但将路由改为：

`K_t in {0, 8, 16, 32}`

由校准后的 frame reliability 决定。q 只负责 token 排序，frame gate 决定是否调用 SATR。

### C. 目标存在性分支

使用响应峰值、框边界风险、Kalman innovation、历史一致性形成二分类目标：

- visible / reliable：允许纠错和模板更新；
- absent / uncertain：SATR abstain，禁止历史写回，等待重新出现证据。

### D. 模态质量重加权

分别计算 RGB/TIR 的质量，跨模态冲突不直接等价为错误。形变和低照度场景降低冲突项权重，避免固定 Tanner 图传播误修。

下一次实验顺序：先做 A+B 的 10 条验证，再做 C；只有 10 条稳定超过当前门控候选，才进入一次完整训练。

## 实验反馈

实际训练 `lambda_post_syndrome=0.1` 的 400-update 候选在 10 条序列上只有 `PR 68.86 / SR 51.57`，明显失败。训练日志显示 syndrome 确实下降，但跟踪性能同步下降，证明当前可学习投影存在“投机式降低 syndrome”路径。因此后验 syndrome 不能单独作为强损失；下一版必须冻结或 stop-gradient 校验参考，并和目标模板匹配、跟踪损失及正常 token 保真联合约束。
