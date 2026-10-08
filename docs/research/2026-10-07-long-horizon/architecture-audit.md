# CodeTrack 长时序架构审计（2026-10-07）

审计对象：工作区 `/home/yangjuanfeng/lab/projects/gola-CodeTrack`，HEAD `3b6f6a56b52f4ce53171a766035074bdbaca17f0`。本审计只读代码、相关实验说明和必要的版本差异；没有运行训练或新推理，没有把代码中的潜在机制当成已经定位到某次失效的实验证据。未发现适用于此目录的 AGENTS.md。现有工作区含其他未提交改动，本审计未修改它们。

## 1. 最关键的结论

现有问题不只是单帧特征质量。系统把自己的定位结果同时变成下一帧观测窗口、运动测量和可更新身份记忆，存在多条相互强化的反馈路径。原生 word 路径已经开始保护状态，但最新保护策略只是暂缓采用修复结果，还没有“通过独立后续观测确认，然后安全提交”的闭环。因此，正确的新架构需要把 **当前观测证据、待确认假设和已提交状态** 分开，并增加目标不存在状态及全图重新搜索。

需要严格区分三个对象：

| 对象 | 修复输出是否更新 crop/KF | template/history | 当前证据地位 |
|---|---|---|---|
| E0028 初版 native rollout，`50f3670` | 是；用修复后 box/score | 修复帧不写入 | 实验报告中的短/长序列结果属于这一代机制 |
| 当前 native tracker，`3b6f6a5` | 修复帧仍用修复前 baseline box/score | 修复帧不写入 | 新增 abstain/gain 门控；不能直接沿用 E0028 的得分 |
| official evaluator + CodeTrack/SATR | 用修复输出，但内部 KF box 解码和外部 evaluator 可不同 | learned gate 控制 template；另有 rolling memory | 路径与 native tracker 不同，并有坐标/状态管理风险 |

不能把“E0028 反馈了错误修复”和“当前代码仍如此”混为一谈；也不能把当前防反馈分支直接描述成已解决长时序恢复。

## 2. 最新 native 路径的真实数据流

入口是 [tools/observed_tracker.py:57](/home/yangjuanfeng/lab/projects/gola-CodeTrack/tools/observed_tracker.py:57)。这里明确关闭模型内部 CodeTrack 分支，加载冻结 GOLA，再在 fused tokens 后挂 `NativeWordDecoder`，并不运行 legacy SATR。

初始化流程：

1. 用首帧 GT box 初始化 crop provider、112×112 immutable template `z`、在线 template `d` 和全图坐标 KF。
2. 保存初始原图和 box，并保留最多 3 个后续高置信历史帧。
3. 用首帧目标 foreground mask 对初始 RGB/TIR template tokens 做均值，得到固定 identity 向量。

依据：[tools/observed_tracker.py:98](/home/yangjuanfeng/lab/projects/gola-CodeTrack/tools/observed_tracker.py:98)。

每帧流程：

1. 围绕 crop provider 的上次 box 裁 224×224 局部 RGB/TIR；先运行原始 GOLA。
2. 保存 baseline 输出，得到 KF 的当前预测。
3. 对初始帧和至多 3 个历史帧，按 KF 的中心/尺度将历史图像 warp 到当前 crop；分别替换 RGB 或 TIR，再分别使用 online 或 initial template 重编码。每个历史帧产生 4 个 offered words，总计 4–16 个额外 GOLA 前向。
4. decoder 判断 word usefulness、abstention、gain 和 token conditional repair probability，从合格 word 中选一个，替换部分 TIR tokens，再运行 pointwise GOLA head。
5. 返回修复结果；若实际写 token，crop/KF 用 baseline，template/history 不更新；若不写，则按原基线规则更新。

依据：[tools/observed_tracker.py:124](/home/yangjuanfeng/lab/projects/gola-CodeTrack/tools/observed_tracker.py:124)、[tools/observed_tracker.py:189](/home/yangjuanfeng/lab/projects/gola-CodeTrack/tools/observed_tracker.py:189)。

GOLA 不是独立编码后简单投票：RGB、TIR、initial/online template tokens 共同进入同一 Transformer，最终只把 TIR search tokens 交给 head。融合见 [gola.py:650](/home/yangjuanfeng/lab/projects/gola-CodeTrack/trackit/models/methods/GOLA/gola.py:650)；pointwise MLP head 见 [mlp.py:70](/home/yangjuanfeng/lab/projects/gola-CodeTrack/trackit/models/methods/GOLA/modules/head/mlp.py:70)。因此 fused RGB/TIR 差异与 template 条件化输出并非统计独立的证据源。

## 3. 可确认的反馈通道与适用边界

| 通道 | 代码事实 | 对长序列的含义 | 证据边界 |
|---|---|---|---|
| box → next crop | `SiamFCCroppingParameterSimpleProvider.update` 接收 confidence 但完全不使用；只要 box 有效就覆盖 cached_bbox | 低置信定位也会移动窗口；目标离开局部 crop 后，本路径没有重新发现它的观测机会 | 确认的行为；某个具体失败帧需另做日志回放定位 |
| raw score → online template/history | 未修复且 score > .84 就可用预测 box 裁模板和记录历史 | 高置信错误身份可以进入后续证据；score 并非独立身份认证 | 确认更新规则；未声称所有高分更新都错误 |
| predicted box → KF | native 每帧都 observe；`valid` 默认全真；低 confidence 仅让 R 最多增加至 21 倍 | 连续错误测量仍可拉走运动状态，且错误但自洽的轨迹可变得低创新/低不确定 | 确认路径及条件性失效机制；不是 KF 数值发散证明 |
| KF → historical warp → repaired feature | 老目标像素被直接放到 KF 预测位置，再编码为候选 | 当前目标缺失时仍能制造视觉上像目标的候选；候选 head 高分不等于当前存在目标 | 结构性风险；不能从 warp 本身推出所有候选无用 |
| correction → future crop/KF | E0028 采用修复输出；当前采用 baseline | 旧版可把误修复放大；新版阻断直接写回，也阻断真正修复对后续轨迹的纠偏 | 版本差异已核对 |
| correction → template freeze | 最新版修复帧冻结 template/history | 即使 crop/KF 取 baseline，整条状态轨迹也不保证与独立 GOLA 相同，因为 template 更新已不同 | 必须避免称“完整 baseline trajectory 原封不动” |

具体引用：

- 无置信过滤的 crop 更新：[simple.py:39](/home/yangjuanfeng/lab/projects/gola-CodeTrack/trackit/runner/evaluation/common/siamfc_search_region_cropping_params_provider/simple.py:39)。
- 在线模板替换规则：[simple.py:46](/home/yangjuanfeng/lab/projects/gola-CodeTrack/trackit/runner/evaluation/distributed/tracker_evaluator/components/template_updater/simple.py:46)。
- 最新 provisional state 逻辑：[tools/observed_tracker.py:163](/home/yangjuanfeng/lab/projects/gola-CodeTrack/tools/observed_tracker.py:163)。
- KF 更新和 R inflation：[codetrack/motion.py:208](/home/yangjuanfeng/lab/projects/gola-CodeTrack/codetrack/motion.py:208)。Mahalanobis 被计算并输出，但这里没有据此拒绝观测。
- native 历史像素注入：[tools/observed_tracker.py:197](/home/yangjuanfeng/lab/projects/gola-CodeTrack/tools/observed_tracker.py:197)。

本次检查到的两条主要推理路径都没有完整的 `target absent → global reacquisition → verified recommit` 状态机。局部 feature repair 无法补回已经不在输入 crop 内的当前目标信息。

## 4. “已知信号”究竟是什么

真正已知的是 **首帧给定 box 内的一次目标观测**，以及从它抽取的 identity 信息。后续正确目标的外观、尺度、姿态、位置、遮挡、传感器退化参数都未知。历史高置信帧是带选择偏差的伪观测，不能等同于通信系统中始终无误的 pilot。

因此自然的检测模型是带 nuisance parameters 的 composite hypothesis test：当前真实观测中是否存在与首帧身份一致的目标。不是已知每帧 clean feature，只差恢复被翻转的 bits。

native decoder 的二值标签尤其需要准确解释：

- word 被训练标签认定 useful，要求 GT 有效、候选 IoU ≥ .5、比 baseline 增加至少 .02，且 task loss 更低。
- `bit_labels` 是 **给定该 useful reference，替换该 token 可降低 task loss** 的标记。
- XOR 标签是对这些 action-dependent utility bits 求模 2，不是对自然视觉特征中本来存在的校验码进行读取。

依据：[probe_native_reference_words.py:199](/home/yangjuanfeng/lab/projects/gola-CodeTrack/tools/probe_native_reference_words.py:199)、[probe_native_reference_words.py:269](/home/yangjuanfeng/lab/projects/gola-CodeTrack/tools/probe_native_reference_words.py:269)、[train_native_word_decoder.py:164](/home/yangjuanfeng/lab/projects/gola-CodeTrack/tools/train_native_word_decoder.py:164)。这些 GT 信息作为训练标签是合理的，本审计没有发现 native `track(image)` 接收当前 GT。

[syndrome_bp.py:3](/home/yangjuanfeng/lab/projects/gola-CodeTrack/codetrack/syndrome_bp.py:3) 自己也明确说明：错误变量不是量化 visual feature，且不声称任意 DINO feature 都是 codeword。其 check-to-variable 实现正确排除了接收者 incoming message，但有环图、重叠 learned visual evidence 和估计的 likelihood，使“exact sum-product update”不能被扩大为“真实 calibrated posterior”“物理 ECC 保证”或“误差一定能纠正”。

最新 `gain_lcb = mean − std` 是模型输出的一标准差下界型分数；在未独立校准前，不是具有指定覆盖率的统计置信下界。`q × quality × (1 − abstain)` 也不能因写成乘积就自动获得 joint posterior 语义：abstain 和 quality 的监督是互补标签，是否应重复相乘需要按实际选择风险验证。代码见 [word_decoder.py:78](/home/yangjuanfeng/lab/projects/gola-CodeTrack/codetrack/word_decoder.py:78)、[train_native_word_decoder.py:167](/home/yangjuanfeng/lab/projects/gola-CodeTrack/tools/train_native_word_decoder.py:167)。

## 5. E0028 与当前 HEAD 的差异

`git show 50f3670:tools/observed_tracker.py` 的第 148 行直接 `provider.update(confidence, pred)`，第 160–162 行把修复后 `pred` 交给 KF；只在 template/history 路径排除 repaired frames。word 选择是 `word_quality.argmax()`，没有当前独立 abstain/gain gate。

`3b6f6a5` 加入当前三重 word gate、token joint gate 和 baseline-state 暂缓提交。旧 E0026/E0027 checkpoint 缺少新 head 时使用 conservative 初始化：gain mean = 0，std > 0，因此 gain_lcb < 0，拒绝写入。这意味着 **用当前代码直接加载 E0027 checkpoint 不会复现其旧版有修复结果**，而会走保守不写入路径。依据：[tools/observed_tracker.py:66](/home/yangjuanfeng/lab/projects/gola-CodeTrack/tools/observed_tracker.py:66)、[word_decoder.py:49](/home/yangjuanfeng/lab/projects/gola-CodeTrack/codetrack/word_decoder.py:49)。

实验说明中可确认的历史结果仅用来定位版本：E0028 报告 200 帧 leftboy 与 928 帧 whitebetween 增益；E0029 记录同代 8,574 帧 rightbackcup 明显下降，并记录 shuffled parity 与 learned parity 在 leftboy 分数相当。见 [E0028:13](/home/yangjuanfeng/lab/projects/gola-CodeTrack/experiments/E0028-live-native-word-rollout.md:13)、[E0029:11](/home/yangjuanfeng/lab/projects/gola-CodeTrack/experiments/E0029-live-code-ablation.md:11)。本审计未独立重算这些数值；最终结果审计另行处理。它们支持“旧原生机制有长序列退化和机制归因不足”，不支持“当前 safety 版本已通过/未通过完整长序列”的判断。

## 6. official evaluator / legacy CodeTrack 的额外集成风险

以下问题属于另一条路径，不能套用到已经在整图物理坐标维护 KF、明确初始化重置的 `ObservedGOLATracker`。

**搜索坐标变换没有接入官方输入。** `one_stream.prepare_tracking` 保存 `x_cropping_params` 到 temporary_objects，却只把 z/x/d 与 search-region `image_size` 传给模型。CodeTrack 新增 `search_crop_params` 后才做 KF affine rebase。官方路径未传这个参数，所以在连续改变的局部坐标系中使用上一帧 box，可能形成运动错位。

依据：[one_stream/__init__.py:150](/home/yangjuanfeng/lab/projects/gola-CodeTrack/trackit/runner/evaluation/distributed/tracker_evaluator/default/pipelines/one_stream/__init__.py:150)、[codetrack.py:344](/home/yangjuanfeng/lab/projects/gola-CodeTrack/codetrack/codetrack.py:344)、[gola.py:467](/home/yangjuanfeng/lab/projects/gola-CodeTrack/trackit/models/methods/GOLA/gola.py:467)。

**KF 与外部 evaluator 可能使用不同 box。** GOLA 在内部取 raw classification map argmax 作为反馈 box；官方 postprocessor 若使用 Hann window penalty，则在加 penalty 后选位置，再读取该位置 raw score。这两个 argmax 不总一致，虽然 GOLA 注释声称 decode 完全一致。

依据：[gola.py:526](/home/yangjuanfeng/lab/projects/gola-CodeTrack/trackit/models/methods/GOLA/gola.py:526)、[box_with_score_map.py:37](/home/yangjuanfeng/lab/projects/gola-CodeTrack/trackit/runner/evaluation/distributed/tracker_evaluator/components/post_process/box_with_score_map.py:37)。启用 motion + window penalty 时该风险才相关；建议统一在 evaluator 已接受 box 后更新唯一状态。

**循环状态按 batch row 保存，没有按 task ID 的隔离。** CodeTrack 只在训练默认模式或 batch 数量变化时自动 reset；本次全仓库检索没有找到官方 evaluation harness 对 `reset_sequence()` 的调用。若调度在相同 batch 大小时换序列/换行位置，旧 KF、memory、score/box 可能留给其他任务。模型内部状态和 evaluator 的 per-task crop/template cache 管理并不一致。

依据：[codetrack.py:153](/home/yangjuanfeng/lab/projects/gola-CodeTrack/codetrack/codetrack.py:153)、[codetrack.py:307](/home/yangjuanfeng/lab/projects/gola-CodeTrack/codetrack/codetrack.py:307)、[one_stream/__init__.py:99](/home/yangjuanfeng/lab/projects/gola-CodeTrack/trackit/runner/evaluation/distributed/tracker_evaluator/default/pipelines/one_stream/__init__.py:99)。这是代码级风险，未在本轮动态复现跨序列污染。

**rolling memory admission 使用上一帧分数决定当前 token 是否进入。** forward 在 head 前计算当前 summary，但把 `_prev_score` 传给 memory；上一帧好、当前突发损坏时，当前 summary 仍可能进入；训练则默认全接纳。memory 只有 rolling slots，当前实现没有 immutable initial-template slot。有关注释仍部分引用旧 anchor 设计，不能当作实际代码行为。

依据：[codetrack.py:440](/home/yangjuanfeng/lab/projects/gola-CodeTrack/codetrack/codetrack.py:440)、[motion.py:532](/home/yangjuanfeng/lab/projects/gola-CodeTrack/codetrack/motion.py:532)、[motion.py:567](/home/yangjuanfeng/lab/projects/gola-CodeTrack/codetrack/motion.py:567)。

## 7. 对新架构和训练方案的约束

建议保留 GOLA 作为强局部候选生成器/特征主干，把主要创新放在 observation-to-state contract。数学工具和 long-term tracker 的已有做法应明确作为基础，而不是宣称 matched filter、Kalman、null hypothesis、global search 或 dual memory 本身新颖。

1. **immutable pilot 与 current-only verifier。** 首帧身份锚点保持不可写；候选身份/存在证据必须来自真实当前 RGB/TIR，而不是把历史目标像素写入当前 search image 后得到的分数。历史外观可作模板和 nuisance model，不能替代当前存在性证据。
2. **显式 null state 与多假设。** 允许“目标当前不可见/不在局部窗”；保留多个候选与 missing measurement，不强制每帧把最高分 box 当观测。用身份 likelihood 校正 motion prior，防止平滑而错误的轨迹自己证明自己。
3. **将 prediction、provisional 和 commit 分离。** 新候选先进入隔离队列/影子状态，只输出 provisional box；后续未被该候选污染的当前观测确认后，才原子提交 crop/KF/template。设计 rollback、timeout 和 replacement 规则，避免永久冻结 baseline 导致无法真正恢复。
4. **全图重新搜索是真实当前图像的候选发现。** 低置信/absence 时扩大搜索并做全图低分辨率候选扫描，再精细验证；校准必须包含全图 maxima 和多候选选择，不能复用单局部 crop threshold。
5. **风险目标按整段轨迹和状态写入定义。** 训练 hard negatives 应包含外观相似 distractor、目标离图、错误历史、运动模型失配和两模态共同退化。收集 model-rollout state，而不是只用 GT-centered 短 clip；监督 commit 的未来影响、错误提交概率、失败持续时间与恢复延迟。
6. **通信启发必须可证伪。** 若保留 syndrome，应表达多个真实观测/预测之间的 consistency residual，且以 matched unary、shuffled checks、independent-verifier-disabled 等对照证明增益。不要要求新方法为了名字而继续合成 feature words。

最有研究价值的待验证主张是：**在候选生成和算力相同的条件下，基于当前观测的序贯验证与可回滚状态提交，降低错误状态更新的长时传播，同时维持真正再检测后的恢复速度。** 该主张需要与强 long-term tracking / distractor association / adaptive memory baselines 对比，而不是仅与原 GOLA 或当前 safety gate 对比。

## 8. 建议优先做的最小判别实验

固定相同真实图像流与候选生成器，分别运行：原始反馈；仅冻结 template；仅阻断修复写 crop/KF；当前 HEAD；current-only verifier；verifier + delayed commit；再加 true global reacquisition。每个对照使用完整、独立序列，统一统计状态错误写入率、持续失败长度、恢复延迟、按运行时长分桶的成功率、PR/SR 和实际延迟。这样可区分 token 单帧有效、状态提交有效和全图重新搜索有效，避免将几种收益合在一起解释。

implementation preflight 必须先锁定：所有 box 的坐标协议；Hann 后接受 box 的唯一来源；每 task 独立状态与边界 reset；template/history provenance；repaired/verified/committed 三类状态日志；immutable identity 不写入测试。这些是让架构实验具有可解释性的前置正确性，不是新颖性贡献。
