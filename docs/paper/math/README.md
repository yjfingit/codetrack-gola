# CodeTrack 数学工具源头论文

这里收录的是 CodeTrack 所用**数学工具**的原始出处，而不是应用论文。
写 Method 时，公式可以直接沿着这些工作的数学框架改写，不必自造符号体系。

命名：`M<序号>_<短名>_<作者年份>_<来源>.pdf`

---

## 已入库（9 篇）

| 文件 | 论文 | CodeTrack 借什么 | 优先级 |
|---|---|---|---|
| `M01_Gallager1962_LowDensityParityCheckCodes.pdf` | Gallager, *Low-Density Parity-Check Codes*, 1962（MIT 博士论文） | **`H` 矩阵从哪里来**：sparse parity-check matrix、`s = Hyᵀ = Heᵀ`、syndrome 只反映错误 | **S** |
| `M02_Tanner1981_RecursiveApproachLowComplexityCodes.pdf` | Tanner, *A Recursive Approach to Low Complexity Codes*, IEEE T-IT 1981 | **`H` → 二部图**：variable node / check node、`N_H(i)`。直接决定 Method 里图和符号怎么画 | **S** |
| `M03_KschischangFreyLoeliger2001_FactorGraphsSumProduct.pdf` | Kschischang, Frey, Loeliger, *Factor Graphs and the Sum-Product Algorithm*, IEEE T-IT 2001 | **learned belief propagation**：`m_{i→j}` / `m_{j→i}` 消息传递、可靠度传播，把 `q = σ(Hᵀs+b)` 升级为 2 轮 BP | **S** |
| `M05_AttentionIsAllYouNeed_Vaswani2017_arXiv1706.03762.pdf` | Vaswani et al., *Attention Is All You Need*, 2017 | sparse token recovery 的基本算子：`softmax(QKᵀ/√d)V`；区别在于 `K,V = X_{N_H(i)∩R}`，即 **Tanner-constrained Sparse Attention** | **S** |
| `M06_GraphAttentionNetworks_Velickovic2018_arXiv1710.10903.pdf` | Veličković et al., *Graph Attention Networks*, ICLR 2018 | **fixed sparse support + learnable edge weights**：`H_ji = H_0,ji · α_ji`，`α_ji` 由 attention 学习 | **A** |
| `M07_DenoisingVisionTransformers_ECCV2024.pdf` | Yang et al., *Denoising Vision Transformers*, ECCV 2024 Oral | recovery 器最直接参考：`f_θ(X_raw) ≈ X_clean`，`L_rec = 1-cos(X̂,X_clean) + λ‖X̂-X_clean‖`。区别是我们**只恢复 syndrome 判坏的 token** | **S** |
| `M09_DDPM_Ho2020_arXiv2006.11239.pdf` | Ho, Jain, Abbeel, *Denoising Diffusion Probabilistic Models*, 2020 | forward/reverse corruption 的数学形式。**仅供理解，不作主方法**（tracker 跑不了 20–100 步） | **B** |
| `M11_FlowMatching_Lipman2023_arXiv2210.02747.pdf` | Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023 | 天然 paired data `X_corr ↔ X_clean`：路径插值 + 速度场回归 `‖v_θ - (X_clean-X_corr)‖²`，2–4 步 Euler | **A/B** |
| `M12_RectifiedFlow_Liu2022_arXiv2209.03003.pdf` | Liu, Gong, Liu, *Flow Straight and Fast (Rectified Flow)*, 2022/23 | 尽可能**直**的 transport path：`X_corr → X_clean` 是 feature-space transport，追求 1–2 步 ODE 恢复 | **A/B** |

`M07` 与 `../02_S_DenoisingVisionTransformers_ECCV2024.pdf` 为同一文件（应用论文清单与数学工具清单都引用它）。

---

## 定位：从"工程架构"到统一数学模型

把上述工具串起来，CodeTrack 的数学骨架可以写成：

$$
\boxed{
H
\rightarrow
\text{Tanner Graph}
\rightarrow
\text{Syndrome}
\rightarrow
\text{Message Passing}
\rightarrow
q_i
}
$$

$$
\boxed{
q_i
\rightarrow
\mathcal N_H(i)
\rightarrow
\text{Sparse Recovery}
}
$$

$$
\boxed{
\text{Historical state}
\rightarrow
\text{Motion prior} + \text{Uncertainty}
}
$$

$$
\boxed{
X^{corr}
\rightarrow
X^{clean}
}
$$

即下一版可整理为 **Neural Tanner Decoder for Visual Token Error Correction**：
把当前的 $q=\sigma(H^\top s+b)$ 升级为 **2 轮 learned belief propagation**。

### 表述红线

- 写成 **"LDPC-inspired structured redundancy checking"**，
  **不能**声称"我们真的构造了一个视觉 LDPC code"——视觉 feature 不满足 GF(2) 的 $HX=0$。
- 因此用 $C^{obs}=HX$ 与 $C^{ref}=HR$ 的 discrepancy 定义 syndrome，
  而不是直接令 $HX=0$。

---

## 未入库：下载不到，已放弃

| 论文 | 状态与尝试过的合法来源 |
|---|---|
| **Kalman, R. E., "A New Approach to Linear Filtering and Prediction Problems," 1960**（*J. Basic Engineering* 82(1):35–45） | **未获取**。Unpaywall `is_oa: false`；ASME Digital Collection 403 付费墙；MIT DSpace 检索无此篇；UNC / Harvard / UDelaware / Cornell / Stanford EE363 / Berkeley / MIT 2.160 / TU Delft 等公开镜像均 404。<br>**替代**：Kalman 理论在 tracking 中的标准形式可参考任意信号处理教材；CodeTrack 的 motion prior 用法（状态 `x=[c_x,c_y,w,h,v...]`、$x_t=Ax_{t-1}+w_t$、$P_{t\|t-1}=AP_{t-1\|t-1}A^\top+Q$、$u_t=\mathrm{tr}(P_t)$）属教科书内容，可直接按方案文档 §7 表述并引用该文出处。 |
| **Gallager 1963 专著**（MIT Press 版本） | 不需要：已获取 1962 年 MIT 博士论文原文（即 `M01`，LDPC 的真正源头）。 |

> 说明：Gallager 1962 与 Tanner 1981 均已通过**合法开放渠道**获得
> （MIT DSpace 机构库 / 大学课程镜像）。付费墙论文不做额外规避。

---

## 获取来源（便于复核）

| 文件 | 来源 |
|---|---|
| M01 Gallager | MIT DSpace 机构库（`hdl.handle.net/1721.1/11804`），经 DSpace REST API 取 bitstream |
| M02 Tanner | `csd.uoc.gr` 课程镜像（Crete 大学） |
| M03 Factor Graphs | INRIA 公开页面所附 PDF |
| M05 Attention / M06 GAT / M09 DDPM / M11 Flow Matching / M12 Rectified Flow | arXiv 官方 |
| M07 Denoising ViT | ECVA 官方开放获取 |

## 约定

- 本目录 PDF 仅作内部研究参考，不纳入版本库（见根目录 `.gitignore`）。
- 不复制论文文字与图；公式体系可沿用其数学框架，但须正确引用原作者。
