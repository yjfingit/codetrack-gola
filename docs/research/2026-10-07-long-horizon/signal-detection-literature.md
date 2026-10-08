# 已知信号检测、同步失锁与长时状态更新：文献工具箱

检索日期：2026-10-07（Asia/Shanghai）。从本地 `docs/paper/math/`、`docs/paper/` 与既有文献笔记开始，发现原有数学材料偏 LDPC/Tanner/BP，缺少目标存在检测、背景检验和闭环状态分布。随后先访问 IEEE/ScienceDirect，再使用出版方 DOI 元数据、原始 arXiv 与 PMLR 正文补齐。IEEE 返回空白202，部分 ScienceDirect/Euclid 页面受访问限制；下表明确区分元数据核验和读到全文，未把访问失败写成已精读。

## 检索得到的主线

1. **匹配滤波/匹配子空间/GLRT**：初始目标描述提供已知方向，背景协方差规定哪些相似性才值得相信；允许未知幅度或外观系数。
2. **序贯检测与不完整观测滤波**：不能每帧强行选择一个目标；失去观测应保留不确定度、预测并重新获取。重复打分不等于新证据。
3. **闭环学习与长期错误控制**：训练要覆盖模型自身诱导的状态，评价更新的后续代价；每帧分数校准不自动给整段/无限时域保证。

## 核心论文与使用边界

| Paper | Venue | Year | Layer | Scenario | Method | Key result / tool | Limitation | Relevance | Source |
|---|---|---:|---|---|---|---|---|---|---|
| E. J. Kelly, **An Adaptive Detection Algorithm** | IEEE Transactions on Aerospace and Electronic Systems | 1986 | 统计检测 | 已知信号方向、未知干扰统计、辅助无目标样本 | 自适应检测 / GLRT | 已知方向检测与未知协方差联合处理的经典原点 | 参数/干扰与样本假设有条件；不是视觉分数天然CFAR | 提醒将背景建模纳入目标匹配 | ieee + DOI元数据；[DOI](https://doi.org/10.1109/TAES.1986.310745)，[IEEE](https://ieeexplore.ieee.org/document/4104190/)。本轮未取得原始全文 |
| L. L. Scharf, B. Friedlander, **Matched subspace detectors** | IEEE Transactions on Signal Processing | 1994 | 统计检测 | 目标位于已知子空间，系数未知，有噪声/干扰 | 子空间投影、GLRT | 用投影能量检验一族目标变化而非固定单个向量 | 子空间太大易吸收干扰；白化/协方差假设不可省略 | 目标外观变化的数学框架；rank-1为匹配滤波起点 | ieee + DOI元数据；[DOI](https://doi.org/10.1109/78.301849)。本轮未取得原始全文；引用关系由2024原文核对 |
| S. Kraut, L. L. Scharf, L. T. McWhorter, **Adaptive subspace detectors** | IEEE Transactions on Signal Processing, 49(1):1–16 | 2001 | 统计检测 | 未知噪声协方差，辅助样本 | 自适应子空间检测 | 系统区分不同未知量和适应性检验 | 不能把不同设置的AMF/ACE/Kelly统计量混称同一检验 | 给视觉匹配器选择明确概率假设 | ieee + [DOI](https://doi.org/10.1109/78.890324)元数据与2024原文参考文献；未取得2001全文 |
| O. Ledoit, M. Wolf, **A well-conditioned estimator for large-dimensional covariance matrices** | Journal of Multivariate Analysis, 88(2):365–411 | 2004 | 协方差估计 | 维数与样本量接近 | 协方差收缩 | 把样本协方差向结构化目标收缩，改善条件数 | 最优收缩结果依赖设定；视觉局部背景相关且可能被目标污染 | 96维投影下背景不足时使用收缩/对角协方差 | sciencedirect受限；[DOI](https://doi.org/10.1016/S0047-259X(03)00096-4)元数据已核验，未获全文 |
| A. Wald, **Sequential Tests of Statistical Hypotheses** | Annals of Mathematical Statistics, 16(2):117–186 | 1945 | 序贯统计 | 两个假设下逐次获得观测 | SPRT | 基于累积似然比和停止边界做序贯决策 | 有效似然/依赖结构必须满足；重编码同一图像不是独立样本 | 把确认看作新观测到达的过程；本文不直接套名义SPRT阈值 | DOI/原出版入口元数据；[DOI](https://doi.org/10.1214/aoms/1177731118)，原页访问受限 |
| E. S. Page, **Continuous Inspection Schemes** | Biometrika, 41(1–2):100–115 | 1954 | 变化检测 | 监控统计过程何时发生变化 | CUSUM | 累积弱异常、对失锁转变敏感 | 假警报/检测延迟依赖漂移和数据分布 | 可作低成本失锁基线；不必再增加可训练模块 | DOI元数据；[DOI](https://doi.org/10.1093/biomet/41.1-2.100)，原页受限 |
| B. Sinopoli, L. Schenato, M. Franceschetti, K. Poolla, M. I. Jordan, S. S. Sastry, **Kalman Filtering With Intermittent Observations** | IEEE Transactions on Automatic Control, 49(9):1453–1464 | 2004 | 状态估计 | 线性Gaussian系统、随机间歇测量 | 缺失观测下Riccati/Kalman分析 | 显示观测到达条件会决定期望误差协方差能否有界 | 不证明自生成错误bbox输入的视觉滤波稳定；缺测不等于错误测量 | 没有可靠当前观测时只predict；不要持续把argmax当测量 | ieee + [DOI](https://doi.org/10.1109/TAC.2004.834121)元数据；本轮未获原文，arXiv:0906.1637复核其研究设定 |
| S. Ross, G. J. Gordon, J. A. Bagnell, **A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning** | AISTATS / PMLR 15:627–635 | 2011 | 序贯学习 | 预测会影响以后遇到的观测状态 | DAgger | 训练分布切换到学习器自身诱导的状态；揭示纯expert-state监督的复合错误 | 保证依赖专家/代理损失等条件，不能直接声称本项目同样界成立 | 真实predicted-crop、错误memory状态聚合的依据 | web原会议；[PMLR全文](https://proceedings.mlr.press/v15/ross11a.html)，PDF已下载精读 |
| S. R. Howard, A. Ramdas, J. McAuliffe, J. Sekhon, **Time-uniform, nonparametric, nonasymptotic confidence sequences** | Annals of Statistics, 49(2):1055–1080 | 2021 | 序贯统计 | 反复观察和数据依赖停止 | time-uniform confidence sequences / martingale方法 | 同时覆盖多个时间点的置信序列，区别于单时刻interval | 仍需条件矩/鞅假设；“nonparametric”不等于无假设或任意分布漂移 | 明确why mean−std与逐帧gate不能变成无限时域认证 | [DOI](https://doi.org/10.1214/20-AOS1991)核实正式年份；[arXiv全文](https://arxiv.org/abs/1810.08240)已读（下载版2022修订，正式论文2021） |
| Aref Miri Rekavandi, **Towards Adaptive Subspace Detection in Heterogeneous Environment** | arXiv preprint；原文写submitted to IEEE TAES，未核实正式接收 | 2024 | 统计检测 | 测试cell与辅助训练cell的协方差结构不同 | 受约束的自适应GLRT | 正面处理经典同质/部分同质背景假设失配 | Gaussian与特定协方差邻域假设仍在；仿真结果不等于视觉收益 | 当前场景背景与首帧/历史背景不同，不能永久使用旧协方差 | web原始预印本；[arXiv全文](https://arxiv.org/abs/2401.12469)，已读全文 |

## 我们实际借用什么

### 目标匹配：从未经校准的cosine变成背景条件下的统计量

工作模型为 `H0: y=μ+n`，`H1: y=μ+Sa+n`，`n~N(0,Σ)`。在S、Σ固定且a自由的情形下，最大化a后，GLRT的单调等价统计量为：

\[
T=(y-\mu)^\top\Sigma^{-1}S(S^\top\Sigma^{-1}S)^{-1}S^\top\Sigma^{-1}(y-\mu).
\]

这是明确假设下的推导，**不是Kelly未知Σ联合GLRT的原式**。rank-1时就是白化匹配滤波。工程正则项ε会改变精确检验，但能改善数值稳定。若希望信号幅度为非负，rank-1分子可用 `max(0,sᵀΣ⁻¹(y−μ))²`，而不是接受反向特征。实际视觉训练保留有符号相关/背景竞争特征，避免单独平方投影丢失方向。

协方差应来自真实当前背景，并用shrinkage或对角近似；候选ROI由GOLA给出，验证feature不能已混入online-template或历史合成像素。白化分数只是有物理含义的输入统计量，其最终错误率需在真实RGB-T候选分布测量。

### 连续确认：新证据与重复自评分是两件事

SPRT/CUSUM启发我们等后续真实帧，但视频帧、RGB/TIR和重叠候选高度相关，不能直接对多个置信度相乘。初版使用固定3帧窗口的联合确认器，按完整序列校准。若用CUSUM作基线，可令 `g_t=max(0,g_{t−1}+κ−s_t)`，其中s_t是当前观测支持目标的标准化分数；κ和边界在验证序列选择，报告经验误报与延迟，不声称经典ARL公式有效。

### 稳定性：观测缺失、错误观测与策略分布失配要分开

Kalman的误差小并不意味着当前观测可信；如果持续把自己的错误输出当测量，可能得到低协方差的错误状态。应在没有可靠当前观测时仅predict、允许不确定度增大并扩大搜索。

DAgger为训练自己的状态分布提供直接依据，但本方案的commit/defer反事实多步监督更接近cost-to-go/policy improvement。它不是DAgger原算法原封不动应用，也不自动继承其理论界；相关模仿学习与learned updater的近邻在跟踪文献补检中单列。

### 不做的理论跳跃

- 不把视觉连续feature的差异叫真正channel bit flip。
- 不把参数化gain的 `μ−σ` 叫经过覆盖校准的置信下界。
- 不把保留首帧锚点叫任意时域稳定性证明。
- 不把24/44个相关样本上的无回退叫长期低风险保证。
- 不把一个局部crop内找不到目标归咎于token恢复容量：缺少观测首先要扩大搜索。

## 检索证据与未采用结果

`signal-sources/` 保存元数据、检索响应、实际取得的PDF与文字。一次凭记忆猜测的 DOI `10.1109/78.950787` 实际指向 *Matrix factorizations for reversible integer mapping*，核对后已弃用；Kraut正确DOI为 `10.1109/78.890324`。这条失败记录保留用于溯源，不进入引用表。

2026年的decision-directed tracking/CFAR检索也发现若干预印本，但与RGB-T场景距离较远、验证不足，未以“最新”替代更贴切的经典工具。近期实质性竞争方法以视觉文献中的DTPTrack与SCDT（CVPR2026）为主。

## Practical Takeaway

主流可靠路径是：已知身份条件匹配、背景/干扰竞争、状态估计以及可恢复的检测；高饱和方向是再加一组动态模板或confidence gate。较有研究价值的是把**提议来源、验证观测和状态提交后果**统一起来，证明为什么某次短期看似有利的修复不应成为未来的可信状态。方法是否有独立贡献，取决于与DTPTrack、RTracker、ToMP、Meta-Updater等直接对照，而不是工具命名。

## 本地factor graph原文核对

本轮还读取本地Kschischang, Frey, Loeliger (2001) *Factor Graphs and the Sum-Product Algorithm*。摘要明确说边缘量“either exactly or approximately”；引言将cycle-free图上的精确边缘计算与有环图的迭代算法分开。因此项目中sum-product算子公式正确，只能证明实现了该消息运算；不能推导有环图上有限三轮是精确后验，更不能推出learned visual likelihood已校准。这个限制进一步支持把真实观测与状态写入的可验证关系作为主线。
