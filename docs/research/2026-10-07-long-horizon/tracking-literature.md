# 长时跟踪漂移：核验后的相关论文与创新边界

核验日期：2026-10-07。任务范围：对 CodeTrack 的长序列闭环漂移提供最接近的视觉跟踪文献；数学检测理论由主报告另行覆盖。本文件不报告本项目的新实验结果。

读取了本地 `docs/paper/README.md` 以及 DTPTrack、RTracker、SCDT PDF 全文；其余论文通过 CVF / arXiv 原文读取。外部访问使用 HTTPS requests，并保留文字证据于同目录 `tracking-sources/`。没有修改模型代码。该机器不存在 README 所称的 `refs/repos/DTPTrack` 和 `refs/repos/RTracker`，因此没有声称核验这两个本地源码仓库。

## 最重要的结论

1. **“首帧锚点 + 可靠性门控 + 时序记忆”已经被 DTPTrack（CVPR 2026）非常直接地提出。** 若最终方案仅增加 sigmoid gate、固定初始模板或历史 token，创新性风险极高。
2. **“保持初始锚点，同时让动态分支适应，并对不同输出限制历史污染”也不是空白。** ToMP（CVPR 2022）明确发现历史预测对定位有利、对框回归有害，分类器使用历史，框回归只用初始标注。
3. **“发现丢失后全图重检测 + 正负样本记忆 + 恢复状态机”由 RTracker（CVPR 2024）直接覆盖。** 全局搜索解决目标不在局部 crop 的可观测性，但本身不能当作创新。
4. **真实失败轨迹回放、干扰物身份维护、记忆置信度已经出现于 KeepTrack（ICCV 2021）。** 长闭环训练可以成为贡献的一部分，但不能把“使用失败样本训练”本身称作首创。
5. 较有希望的区别应落在**候选产生与写入认证的独立信息路径、相对明确的背景假设、相关证据的校准、控制污染写入风险的序贯机制、带错误状态的闭环训练，以及有限时域内漂移与重捕获的系统评估**。这些目前是研究假设，不是已完成的查新证明或性能结论。
6. 没有一篇下面的工作证明“任意长时间绝不漂移”。原始视觉不可辨识、长时间完全遮挡、首帧外观过时、RGB/TIR 同时失效都需要进入 absent/uncertain，而不是强制输出高置信目标。

## 八篇核心论文

| 论文 | 作者 | 已核验出处 | 最关键的方法与训练事实 | 对方案的作用与重复风险 |
|---|---|---|---|---|
| **Drift-Resilient Temporal Priors for Visual Tracking (DTPTrack)** | Yuqing Huang, Liting Lin, Weijun Zhuang, Zhenyu He, Xin Li | **CVPR 2026**, pp. 6847–6856；CVF 正式页面与 arXiv:2604.02654 相互核验 | 初始模板+3历史帧做 bbox mask average pooling；MLP+sigmoid 为历史预测可靠性；固定首帧权重 1；加权历史经 MLP 调制 base prior tokens 并注入 host。冻结 DINOv2，训练 LoRA/模块/head；训练是 **5 帧采样片段**，BCE+GIoU | 最接近的漂移基线，必须实现/对比或解释不能复现；“门控记忆+锚点”不足以区别 |
| **RTracker: Recoverable Tracking via PN Tree Structured Memory** | Yuqing Huang, Xin Li, Zikun Zhou, Yaowei Wang, Zhenyu He, Ming-Hsuan Yang | **CVPR 2024**, pp. 19038–19047；CVF 正式页面、arXiv:2403.19242 | PN 树存储正/负样本，按相对特征距离判断 present/absent，连接局部 tracker 与全局 detector。相似度模型用 NIGHTS+LaSOT 三元组训练；LaSOT 负样本取同类别不同序列，采用 hinge 相对距离目标 | 闭环错误无法仅靠更强 local head 解决。正负记忆、失败检测、重检测恢复均有强先例 |
| **Learning Target Candidate Association To Keep Track of What Not To Track (KeepTrack)** | Christoph Mayer, Martin Danelljan, Danda Pani Paudel, Luc Van Gool | **ICCV 2021**, pp. 13444–13454；CVF 正式页面 | 对目标及干扰候选跨帧关联，图特征/匹配维护身份。使用部分监督+人工变换形成的自监督；离线运行 base tracker，在真实失败/多候选轨迹上挖掘难例；记忆按 age×confidence 管理 | 只对“是不是目标”建模不够，需显式知道“更像哪个干扰物”。困难轨迹回放与置信度记忆不能单独声称新颖 |
| **Transforming Model Prediction for Tracking (ToMP)** | Christoph Mayer, Martin Danelljan, Goutam Bhat, Matthieu Paul, Danda Pani Paudel, Fisher Yu, Luc Van Gool | **CVPR 2022**, pp. 8731–8740；CVF 正式页面 | Transformer 从训练帧+当前测试帧 transductively 预测分类/回归模型；训练2参考+1测试帧，DiMP分类损失+GIoU。推理分类使用初始+1高置信历史，回归仅使用初始帧 | 重要实验证据：共享历史对不同任务可有相反作用。应当分离 proposal/localization 与 verification/update 的通路，但通路分离概念已有 |
| **ODTrack: Online Dense Temporal Token Learning for Visual Tracking** | Yaozong Zheng, Bineng Zhong, Qihua Liang, Zhiyi Mo, Shengping Zhang, Xianxian Li | **AAAI 2024**；Crossref DOI **10.1609/aaai.v38i7.28591** 与 arXiv:2401.01686 核验 | 压缩目标的上下文/位置信息为时序 token，连续传播。训练输入 **3参考+2搜索**；focal+5L1+2GIoU，逐搜索帧计算并平均；MAE ViT-B，300 epochs | 代表隐状态递推方案；支持任意长度接口不等于训练覆盖了任意长的错误反馈。长闭环训练对照必须控制训练长度与状态来源 |
| **SeqTrack: Sequence to Sequence Learning for Visual Object Tracking** | Xin Chen, Houwen Peng, Dong Wang, Huchuan Lu, Han Hu | **CVPR 2023**, pp. 14572–14581；CVF 正式页面 | ViT encoder + causal decoder 将 bbox 的 x,y,w,h 离散为 token，自回归输出；训练纯 CE teacher forcing。动态模板用四个坐标 token 的平均 softmax 分数+间隔门限更新 | 必须区分**框内坐标 token 自回归**和**跨帧长历史递推**。坐标 likelihood 用作更新置信度已有，但不等于 calibrated identity likelihood |
| **ProContEXT: Exploring Progressive Context Transformer for Tracking** | Jin-Peng Lan, Zhi-Qi Cheng, Jun-Yan He, Chenyang Li, Bin Luo, Xu Bao, Wangmeng Xiang, Yifeng Geng, Xuansong Xie | **ICASSP 2023**；Crossref DOI **10.1109/ICASSP49357.2023.10094971** 与 arXiv:2210.15511 核验；**不是 CVPR 2023** | 多尺度静态+动态模板，通过 context-aware attention 学时空上下文；focal+IoU+L1；动态模板直接以最高响应是否超过阈值更新 | 重要负面对照是“直接用追踪器自己的最高响应批准自己写入”，会形成自我确认；固定/动态模板与多尺度都已有 |
| **Spatio-Temporal Conditional Denoising Transformer for Modality-Missing RGBT Tracking (SCDT)** | Andong Lu, Ziyi Zha, Jiandong Jin, Shihao Li, Chenglong Li, Jin Tang, Bin Luo | **CVPR 2026**, pp. 13584–13593；CVF 正式页面与本地全文核验 | ODTrack 初始化，空间可用模态+短历史cross-attention+长历史FiLM 作为条件做特征去噪。缺模态优化重建损失，完整模态优化统计对齐损失，均加 tracking loss | 与 CodeTrack 的“RGB-T缺失修复+历史条件去噪”直接竞争。若转向检测/风险控制，不应仍以“时间条件恢复模块”为主创新 |

## 逐篇原始来源与精读证据

### 1. DTPTrack

- 正式出版页：<https://openaccess.thecvf.com/content/CVPR2026/html/Huang_Drift-Resilient_Temporal_Priors_for_Visual_Tracking_CVPR_2026_paper.html>
- arXiv：<https://arxiv.org/abs/2604.02654>
- 作者代码入口：<https://github.com/NorahGreen/DTPTrack>（本轮未 clone/执行）
- §3.2.1：`[c1,c2,c3] = Sigmoid(MLP([s1,s2,s3]))`；首帧 `c0=1`。
- §4.1 原文短摘：“During training, we sample 5-frame sequences.”
- 附录 A.2：BCE 与 GIoU 系数均为 1；170 epochs，AdamW，batch 128。该公开描述未给出专门监督“该历史是否含身份错误”的标签损失，也未给出跨任意长时域的错误概率保证。
- 推荐区别：不是重新给历史打分，而是对**是否允许将预测升级为身份记忆**建立独立验证与时序证据协议；必须与 DTPTrack 相同 backbone、近似参数/训练预算对比。

### 2. RTracker

- 正式出版页：<https://openaccess.thecvf.com/content/CVPR2024/html/Huang_RTracker_Recoverable_Tracking_via_PN_Tree_Structured_Memory_CVPR_2024_paper.html>
- 全文含附录：<https://arxiv.org/abs/2403.19242>
- §3：tracker 负责帧间跟踪，detector 负责 global searching；PN tree walking 决定目标状态。
- 摘要短摘：“a relative distance-based criterion for a reliable assessment of target loss.”
- 附录：NIGHTS 20,000 triplets，LaSOT 4,000 triplets；目标第一帧作 template，相同对象后续 GT crop 作 positive，同类别其他序列作 negative；相似度不是 tracker 分类头自身 softmax。
- 推荐区别：相对正负样本也不新；应明确是否能估计背景条件分布并处理 RGB/TIR 依赖，是否控制错误写入/错误重捕获的 rate；避免仅把 PN tree 改成 memory bank 就宣称新方法。

### 3. KeepTrack

- 正式出版页：<https://openaccess.thecvf.com/content/ICCV2021/html/Mayer_Learning_Target_Candidate_Association_To_Keep_Track_of_What_Not_ICCV_2021_paper.html>
- §3.6 短摘：“mine the training dataset using the dumped predictions of the base tracker”。
- 候选由 base tracker 真实运行产生；在 LaSOT 训练集内划 train-train/train-val；冻结基础跟踪器，学习 association embedding；混合真实部分标签与 synthetic self-supervision。
- §3.8：memory loss `sum_k alpha_k beta_k Q(theta;x_k,y_k)`，记忆移除 age×confidence 最低者，association identity confidence 参与记忆可信度。
- 推荐区别：训练必须记录来自新模型自身分布的长闭环失败，不能把随机 bbox jitter 当作唯一 drift 模拟；但此训练策略仍应表述为具体协议与方法协同，而不是“首次用失败样本”。

### 4. ToMP

- 正式出版页：<https://openaccess.thecvf.com/content/CVPR2022/html/Mayer_Transforming_Model_Prediction_for_Tracking_CVPR_2022_paper.html>
- §3.5 短摘：“including predicted bounding box estimations degrades the bounding box regression performance”。
- 分类模型使用 initial+online predicted frame；框回归仅用 initial，使用 attention padding mask 在一个 batch 中并行两条 pass。
- 推荐区别：应先做消融检查是历史 appearance、geometry、crop 还是 update 在伤害长期性能；分任务切断污染有很强先验，但不能作为未经核验的新颖性主张。

### 5. ODTrack

- arXiv：<https://arxiv.org/abs/2401.01686>
- DOI：<https://doi.org/10.1609/aaai.v38i7.28591>
- Crossref 元数据：<https://api.crossref.org/works/10.1609/aaai.v38i7.28591>
- 全文训练段：3 reference frames，2 search frames，输入 search 与 temporal tokens 连续传递；不能从任意长度 API 推导出长时稳定性。
- 适用实验：同等骨干上比较不递推、直接 token 递推、带门控递推、独立验证后写入；闭环帧数必须至少覆盖真实问题出现的时域。

### 6. SeqTrack

- 正式出版页：<https://openaccess.thecvf.com/content/CVPR2023/html/Chen_SeqTrack_Sequence_to_Sequence_Learning_for_Visual_Object_Tracking_CVPR_2023_paper.html>
- §3.1/3.4 是单个 bbox `[x,y,w,h,end]` 的自回归；训练给先前 GT 坐标 token，推理给生成 token。
- §3.5：初始模板+dynamic template；四坐标 token 的平均 softmax 大于阈值且满足更新间隔才更新。
- 适用结论：likelihood 最大的框不是“目标存在”的概率；研究需要 H0/absent，并在难干扰/丢失片段上校准。

### 7. ProContEXT

- arXiv：<https://arxiv.org/abs/2210.15511>
- DOI：<https://doi.org/10.1109/ICASSP49357.2023.10094971>
- Crossref 元数据：<https://api.crossref.org/works/10.1109/ICASSP49357.2023.10094971>
- 算法1：`score > tau` 时 `D = crop(I_i,b_pred,K)`，否则动态模板维持不变；静态模板来自第一帧。
- 正确会议元数据为 ICASSP 2023，纠正检索任务中预设的 CVPR 2023。

### 8. SCDT

- 正式出版页：<https://openaccess.thecvf.com/content/CVPR2026/html/Lu_Spatio-Temporal_Conditional_Denoising_Transformer_for_Modality-Missing_RGBT_Tracking_CVPR_2026_paper.html>
- §3.3：short-term cue cross-attention，long-term cue FiLM；§3.4：`lambda1 L_recon + lambda2 L_align + L_track`；missing 用 recon，complete 用 align。
- §4.1：ODTrack RGB 预训练初始化，ViT-B，template 128、search 256，6×4090、batch24；LasHeR/RGBT234 相关设置 30 epochs，VTUAV 5 epochs（这些是原文训练配置，不意味着可直接与本项目成绩比较）。
- 推荐区别：保留模态补偿作为辅助和可选能力。用未经认证的 reconstructed feature 写入身份库存在自强化风险；认证观测应保留原始可观测证据通路。

## 近期额外检索，但不进入八篇主表

CVPR 2026 正式目录另检到并读全文：**Adaptive Capacity Autoregressive Visual Tracking (ARTrack-AC)**，Tong Lin, Yifan Bai, Shiyi Liang, Ruigang Niu, Xing Wei。来源：<https://openaccess.thecvf.com/content/CVPR2026/papers/Lin_Adaptive_Capacity_Autoregressive_Visual_Tracking_CVPR_2026_paper.pdf>。它用 diffusion trajectory estimator 估计后续难度并选择高/低计算容量，重点是动态效率，相关但不是本项目 drift 的最直接竞争者。不能把它说成已经证明抗长时漂移。其全文提取保留为 `tracking-sources/adaptivecapacityfull.txt`。

历史上 TLD 与 DaSiamRPN 长时版本也是 tracker/detector、distractor-aware 学习的重要背景；本轮未将它们增补为正式主表，避免重复增加较弱引用。主报告若需要历史背景，应另核验出版元数据与对应长时版本，不能仅凭方法名引用。

## 最值得进入最终方案的判别实验

- **训练分布**：仅 GT crop/jitter vs 真正 predicted crop closed-loop；短 2–5 帧 vs 长 rollout；GT history vs model history。所有历史来源都需可追溯。
- **污染路径**：不更新 vs 自分数更新 vs DTPTrack式 learned reliability vs 独立 verifier 认证；固定 proposal 能力，避免把更强 backbone 当防漂移收益。
- **认证意义**：和 RTracker式相对距离、温度缩放分类分数、简单 K 帧投票对比，再看序贯证据或背景归一化的必要性。
- **反馈来源**：observed feature vs reconstructed feature 写入；干净记忆 vs 注入 1/3/5 个错误 bbox 的记忆；报告错误写入后多久恢复。
- **长时评估**：全序列无 GT reset，按经过帧数/已发生失败事件报告 survival、ID-switch、错误写入率、absent false acceptance、recovery delay 与 time-to-first-failure。AUC 同时报，避免只用总体 AUC 掩盖后半程崩溃。
- **误报校准范围**：明确校准是 conditional、边际还是经验性的；RGB/TIR 相关且候选经 argmax 筛选，不能把 per-candidate 分数校准直接说成整帧/整段风险控制；不应宣称无限时域零漂移。

## 核验边界

本文件证据等级以 CVF 正式论文页 / Crossref DOI 元数据 / 原始 PDF 全文为主。DTPTrack 与 SCDT 的 CVPR 2026 已从正式 CVF 页面核验，并非只相信本地文件名。AAAI 站点本轮有代理连接失败，ODTrack 的正式刊会状态由 Crossref DOI 元数据和原文 AAAI 版权信息交叉核验。没有声称完整覆盖 2026-10-07 前全部跟踪文献；这里只选最接近的八篇用于方案判别，不构成穷尽查新。

## 补充查新：学习何时更新与未来影响监督（2026-10-07）

在最终方案出现“commit/defer 的未来影响监督”后，追加以下针对性查新。**结论需要收紧：未来影响驱动的更新策略，在视觉跟踪领域至少已见于 ICCV 2017，不能把它当作本项目新范式。** 先前八篇主表保持作为主要背景；以下是对新设计的必要补充，而不是穷尽性综述。

| 论文 / 正式来源 | 核验的方法及训练标签 | 与拟议设计的确切区别和风险 |
|---|---|---|
| **High-Performance Long-Term Tracking With Meta-Updater**；Kenan Dai, Yunhua Zhang, Dong Wang, Jianhua Li, Huchuan Lu, Xiaoyun Yang；**CVPR 2020**, pp.6298–6307。[CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Dai_High-Performance_Long-Term_Tracking_With_Meta-Updater_CVPR_2020_paper.html) | LTMU 把 bbox/response map/score/candidate-image 与 template 的历史送入 cascaded LSTM，输出二值 update。§3.2.3 标签：当前帧 IoU>0.5 为 1，当前 IoU=0 为 0，0<IoU<=0.5 丢弃。运行 tracker+updater → 重采状态 → 再训的过程迭代 K=3。系统有独立匹配网络、online verifier、local tracker 与 re-detector | 相同状态输入下，有限未来闭环损失标签可与其**当前正确性标签**区别；但独立 verifier、二值写入、历史证据、own-policy 数据迭代均已存在。必须做 same encoder / inputs / actions / budget 的 immediate-correctness baseline。不能只比低配单帧 gate |
| **Learning the Model Update for Siamese Trackers (UpdateNet)**；Lichao Zhang, Abel Gonzalez-Garcia, Joost van de Weijer, Martin Danelljan, Fahad Shahbaz Khan；**ICCV 2019**, pp.4010–4019。[CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Zhang_Learning_the_Model_Update_for_Siamese_Trackers_ICCV_2019_paper.html) | 网络输入初始 GT 模板、累计模板、当前预测 crop 模板。§3.4 优化 L2(updated template, next-frame GT template)。先真实运行线性更新 tracker 生成训练样本，再用上一训练阶段 UpdateNet 产生累计模板与位置，避免只用 GT 当前 crop | **下一帧外观预测**不同于 paired commit/defer 的多帧闭环 tracking cost，但“用未来信息学习更新”“anchor+history+current”“用自身预测分布迭代训练”已经明确存在 |
| **Tracking as Online Decision-Making: Learning a Policy From Streaming Videos With Reinforcement Learning**；James Steven Supancic III, Deva Ramanan；**ICCV 2017**, pp.322–331，DOI 10.1109/ICCV.2017.43。[CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Supancic_Tracking_as_Online_ICCV_2017_paper.html)，[arXiv:1707.04991](https://arxiv.org/abs/1707.04991) | POMDP action 包括 TRACK/REINIT 与 UPDATE/IGNORE。**§5 的 UPDATE 监督已看未来影响**：在 frame i 更新 appearance，统计未来 GT confidence 增加的帧数 Delta+、未来 track-error confidence 降低的帧数 Delta-；若 Delta+ + Delta- > 0.5N，则 oracle 选择 UPDATE。用启发监督初始化/正则 Q-learning，再从失败 sparse rewards 学长期回报。Figure2 直接画多帧 update/ignore 分支如何造成漂移 | 是比 LTMU 更危险的最近邻。新方案的 paired closed-loop future IoU 与其 fixed future-image confidence oracle 有可辨别差异，但不足以独占“未来风险更新”主张。需在同骨干、同状态/动作/训练预算上加入该 future-confidence objective 与有限 horizon rollout objective 的对照 |
| **Real-time visual tracking by deep reinforced decision making**；Janghoon Choi, Junseok Kwon, Kyoung Mu Lee；**Computer Vision and Image Understanding 2018**，DOI 10.1016/j.cviu.2018.05.009。[DOI](https://doi.org/10.1016/j.cviu.2018.05.009)，[arXiv:1702.06291](https://arxiv.org/abs/1702.06291) | 从模板池选择模板；REINFORCE policy gradient，episode末成败奖励，replay 经验；训练 50,000 episodes，长度30–300帧。用未来成功/失败回报训练决策而非当前 match score | 证明模板选择/维护的长时 return 目标早已有文献。此文是选择模板而非延迟认证写入，不覆盖具体 raw observation 确认协议，但否定“首次为跟踪更新优化长期回报”的表述 |

上述 4 篇均读取全文。CVF 页面保存为 `lmt.txt`、`updatenet.txt`、`supanciccvf.txt`，全文为对应 `*full.txt`；Choi 正式期刊元数据保存为 `choimeta.txt`，全文 `choifull.txt`。

**LTMU 关键原文**：§3.2.3：“label ... is determined based on whether the target is successfully located ... in the current ... frame”。其 matching network 单独训练、在 meta-updater 训练中固定，但系统 RTMDNet verifier 是 online 的。因此不能笼统写“以前所有 verifier 都和 tracker 共用权重”或“没有独立验证”。

**Supancic–Ramanan 关键原文**：§5：“UPDATE appearance with frame i whenever doing so improves the confidence of future ground-truth object locations”。本项目若要声称新颖，必须正面引用这一段。只有“把过去时序置信度换成预测未来收益”已经不够。

### Cost-to-go 文献名纠正

- **AggreVaTe** 对应 **Reinforcement and Imitation Learning via Interactive No-Regret Learning**，Stéphane Ross, J. Andrew Bagnell，2014；[arXiv:1406.5979](https://arxiv.org/abs/1406.5979)。本次核验到的 arXiv 页面写“Under review for NIPS 2014”，因此不要凭此宣称其正式 NIPS2014 发表。原文 Algorithm1：执行当前策略到 t，探索动作 a，再让 expert rollout，获得 action-conditioned cost-to-go，并聚合样本做 cost-sensitive learning。§2.3 明确讨论每个状态获得所有动作的 cost vector。后半 NRPI 还讨论 learner-policy cost-to-go。
- **Learning to Search Better Than Your Teacher** 是另一篇：Kai-Wei Chang, Akshay Krishnamurthy, Alekh Agarwal, Hal Daumé III, John Langford，**ICML 2015**，[arXiv:1502.02206](https://arxiv.org/abs/1502.02206)，算法 LOLS。不能把这个标题归给 Ross/Bagnell2014 或混同为 AggreVaTe。本次仅核验 arXiv 元数据，未精读其正文。
- 因而 same-state paired action rollouts 应如实表述为 cost-to-go / policy improvement 家族的针对性实例化。若 rollout 后续策略是 learner 而非 expert，不应直接搬用 AggreVaTe 的 expert-cost 保证。

### 对“冻结认证参考 + 后续原始图像确认”的剩余新颖性判断

这些先例没有在本轮读取范围内描述完全相同的组合：**每个待写候选在提出时冻结其认证参考/背景模型；候选在 quarantine 中不可影响用来批准自身的 verifier；后续真实原始 RGB/TIR 图像产生确认，且生成修复特征不能成为独立证据；确认后才一次性提交特定身份记忆；使用与该协议一致的污染干预闭环代价训练**。这可以作为更具体的研究切口，但不是已证明的文献空白。

必须避免把“冻结 verifier 权重”误当作信息独立：若 reference bank、crop、候选筛选、背景样本或 hidden state 被未认证候选污染，即使权重冻结仍自我确认。应写清每个 pending transaction 的 immutable reference snapshot、因果可访问字段、确认时刻和撤销规则。

后续3帧确认不是统计独立的3票。RGB/TIR共享目标运动和遮挡，多帧自相关，而且 proposal经过最大值筛选；风险校准须与**完整候选提出+认证+写入协议**一致。不存在标签泄漏：未来GT只供离线训练动作代价，推理只使用已经到达的原始帧，确认延迟必须计入性能/时延。

最小鉴别实验应同时包括：LTMU式 current-IoU 标签；Supancic式 future-confidence 标签；paired future-IoU rollout 标签；同预算K帧规则确认；上述输入与动作完全一致。只有这几项控制后，新收益才可能归因于特定认证语义或训练目标，而非额外时间、状态容量或观测权限。

### 最后一轮有界查新：SENTRY 与在线双分支试写

为检查“在线试写、等待后续真实帧、独立认证再提交”是否已有直接对应工作，本轮仅补做 3 个 arXiv 查询（tracking + shadow/counterfactual + template/update；tracking + probation/tentative + template；visual tracking + update + validation）。命中一篇必须纳入的近期近邻：

**SENTRY: SAM2-Enhanced Neighbor-Aware and Temporally Reasoned Memory for Visual Tracking**，Mohamad Alansari, Yonathan Michael, Hasan AlMarzouqi, Muzammal Naseer, Naoufel Werghi, Sajid Javed，2026；[arXiv:2606.24449v2](https://arxiv.org/abs/2606.24449)，[作者项目页](https://hamadya.github.io/SENTRY/page/)。arXiv 和作者页面均声称 ECCV 2026 接收；本轮未独立核验正式会议论文集，因此记录为**已核验原文、作者声称 ECCV2026 接收**，不冒充正式出版核验。

读取全文 §3.2 与 Algorithm1 后，可以具体区分：

- SENTRY 直接宣称 **refine-before-write / memory admission / temporal verification**。这些概念已经存在，不应成为本项目新颖性标题的唯一内容。
- SENTRY 在当前 t 为每个候选 mask，用相同 SAM2 promptable segmentation **回溯**过去 t−1…t−10 图像，形成 backward tracklets；与过去选出的目标/neighbor 轨迹比较平均 bbox IoU，并做邻居匹配；之后立即按原 baseline schedule 写入选中的 mask。完全不一致时可用 KF prior mask fallback。
- 原文没有描述 **等待将来到达 t+1,t+2 原始图像** 的 prospective probation；没有描述从同一个时刻状态复制 **trial-write 与 control-no-write** 两个分支；没有描述由试写前冻结且不接纳 pending artifact 的独立身份 verifier 同时评价二者。
- 因此，新方案最多可以提出以下**有限、待实验与进一步查新证实的机制差异**：部署时对同一个候选写入动作做短期前瞻对照，两个分支只在是否试写特定 raw ROI artifact 上不同，后续真实输入和随机性相同；用冻结的、候选不可反向污染的认证参考比较分支行为；最终只提交已经受测试的那个 raw artifact，而不是随意把后续高置信框替代为提交物。

这比“根据未来收益训练 gate”更具体，因为部署时实际观察 action-specific response，且与主状态隔离；也比 SENTRY 的历史回溯一致性不同。但**未找到完全相同算法不等于首创证明**，本轮有界查询可能漏掉 multi-expert tracking、test-time adaptation、online validation 或 constrained memory-tree inference 的其他名字；不能声称已经穷尽排除。

同预算最低限度对照还必须包括：直接迟写 K 帧、只保留一个 trial 分支、普通候选多分支跟踪、SENTRY式回溯 consistency、两个分支但用会被 trial 改写的 verifier，以及相同冻结 verifier 的 trial/control 比较。若仅双分支带来增益，很可能收益来自额外计算或多假设搜索；若冻结隔离与具体 artifact 测试缺一项也有效，则无需宣传复杂认证协议。控制支与试写支的 crop 可能随预测分叉，这是动作效果的一部分，但两支必须有同等搜索和观测权限；只给试写支更宽 crop 会混淆因果判断。

SENTRY 证据保存于 `tracking-sources/sentryabs.txt`、`sentryproject.txt`、`sentryfull.txt`。本附录结束此轮定向检索，不以不断增加模块来规避现有工作。
