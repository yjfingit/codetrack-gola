# CodeTrack 参考论文库

本目录收录 CodeTrack 方案直接借鉴的论文。命名格式：

```
<序号>_<优先级>_<短名>_<一句话主题>_<发表处>.pdf
```

优先级沿用方案讨论中的划分：**S** = 直接决定方法骨架，**A** = 重要参考，**B** = 备选/待 MVP 通过后再看。

> **数学工具源头论文单独放在 `math/`**（LDPC / Tanner graph / factor graph / Kalman / Attention /
> GAT / DDPM / Flow Matching / Rectified Flow）。见 `math/README.md`——写 Method 时的公式体系以那边为准。

---

## 已入库（12 篇，全部经 PDF 首页标题核验）

| 文件 | 论文 | 在 CodeTrack 里借什么 |
|---|---|---|
| `01_S_GOLA_..._AAAI2026.pdf` | Group Orthogonal Low-Rank Adaptation for RGB-T Tracking (AAAI 2026) | **代码底座**：backbone、GOLA adapter、训练框架、tracking head。不重写 tracker |
| `02_S_DenoisingVisionTransformers_ECCV2024.pdf` | Denoising Vision Transformers (ECCV 2024 Oral) | **Sparse Token Refiner 的接口范式**：raw ViT feature → 轻量 Transformer → clean feature，含 DINOv2 feature denoising |
| `03_S_DTPTrack_..._CVPR2026.pdf` | Drift-Resilient Temporal Priors for Visual Tracking (CVPR 2026) | **Temporal Memory + Memory Protection**：TRC/TGS 设计，过滤错误历史。同为 TrackIt/LoRAT 体系，迁移成本低 |
| `03b_S_DTPTrack_arXiv2604.02654.pdf` | 同上（arXiv 版） | 同一工作的 arXiv 版本，便于检索与引用 |
| `04_S_TBSI_..._CVPR2023.pdf` | Bridging Search Region Interaction with Template for RGB-T Tracking (CVPR 2023) | **template 作为跨模态桥梁**：恢复条件的 `Z_0 / Z_online` 如何参与跨模态恢复 |
| `06_A_SCDT_..._CVPR2026.pdf` | Spatio-Temporal Conditional Denoising Transformer for Modality-Missing RGBT Tracking (CVPR 2026) | **必读竞争论文**：借其 condition 设计（spatial / short / long temporal），但**不得**变成 "GOLA + SCDT decoder" |
| `07_A_FlexTrack_..._ICCV2025.pdf` | What You Have is What You Track (ICCV 2025) | **corruption curriculum**：video-level modality masking / missing-modality augmentation |
| `09_A_LoRAT_..._ECCV2024.pdf` | Tracking Meets LoRA (ECCV 2024) | GOLA 的上游：PEFT 训练、MLP head、optimizer 与参数管理 |
| `10_B_BAT_..._AAAI2024.pdf` | Bi-directional Adapter for Multi-modal Tracking (AAAI 2024) | RGB→TIR / TIR→RGB 的轻量跨模态 feature transfer，对应"坏 token 向另一模态求助" |
| `12_A_MaskGIT_..._CVPR2022.pdf` | MaskGIT: Masked Generative Image Transformer (CVPR 2022) | 2-step token recovery 时"只修 mask/token，不重建整张图"的迭代思想 |
| `13_B_ConsistencyModels_arXiv2303.01469.pdf` | Consistency Models (ICML 2023) | one/few-step 生成式恢复。**仅在 MVP 成功后才考虑**，优先于 50-step DDPM |
| `14_B_RTracker_..._CVPR2024.pdf` | RTracker: Recoverable Tracking via PN Tree Structured Memory (CVPR 2024) | tracker 级 failure detection / 可靠 memory / 恢复协议，对应 Memory Protection |
| `14b_B_RTracker_full_arXiv2403.19242.pdf` | 同上（完整版，18 页 vs CVF 版 10 页） | 含更完整的方法与实现细节；来自作者仓库自带 PDF |

### 对应开源代码

| 论文 | 仓库 |
|---|---|
| GOLA | <https://github.com/MelanTech/GOLA> |
| MMLoRAT（GOLA 同作者，RGB-X 扩展） | <https://github.com/MelanTech/MMLoRAT> |
| Denoising ViT | <https://github.com/Jiawei-Yang/Denoising-ViT> |
| DTPTrack | <https://github.com/NorahGreen/DTPTrack> |
| LoRAT | <https://github.com/LitingLin/LoRAT> |
| BAT | <https://github.com/SparkTempest/BAT> |
| TBSI | <https://github.com/RyanHTR/TBSI> |
| FlexTrack / FlexTrack-V2 | <https://github.com/supertyd/FlexTrackV2> |
| RTracker | <https://github.com/NorahGreen/RTracker> |
| RGB-T tracking paper list | <https://github.com/DRGBT-Tracking/RGBT_Tracking> |

以上仓库已 clone 至 `refs/repos/`。

---

## 未入库（2 篇，付费墙，无合法开放版本）

这两篇是讨论清单里的 **S 级**，但**当前无法通过开放渠道获取**。

### 1. MoKA-HP: Motion-aware KAdaptation with historical prompts for efficient and robust RGB-T tracking

- 出处：*Neurocomputing*, Vol 665, 2026
- DOI：<https://doi.org/10.1016/j.neucom.2025.132163>
- 状态：Elsevier 订阅制。Unpaywall 查询结果 `is_oa: false`，无任何开放版本；ScienceDirect 直连返回 403。

### 2. RGB-T Tracking With Template-Bridged Search Interaction and Target-Preserved Template Updating

- 出处：*IEEE TPAMI*, 2025
- DOI：<https://doi.org/10.1109/TPAMI.2024.3475472>
- 状态：IEEE 订阅制。Unpaywall 查询结果 `is_oa: false`；IEEE 直连返回 202 挑战页。
- 备注：这是本目录 `04_S_TBSI` 的期刊扩展版（TBSI 作者组）。

### 获取建议

1. 用机构 VPN / 图书馆订阅下载后放入本目录；
2. 直接给通讯作者发邮件索要（学术界常规做法，作者通常乐意提供）；
3. 关注作者主页与 arXiv 是否有后续上传。

> 这两篇的内容在方案中的角色分别是 **motion prior（KFMM）** 与 **online-template protection**。
> 在拿到原文之前，可先用已入库的 `03_S_DTPTrack`（temporal reliability）与 `04_S_TBSI`
> （template bridging / updating 思路）作为代理参考；这两块在方案中也**不是核心创新点**
> （核心是 syndrome diagnosis + H-based routing），因此不构成阻塞。

---

## 收录与引用约定

- 代码复用按各仓库 license 执行；`third_party/` 与 `refs/repos/` 均为**只读参考**，不直接复制进 `trackit/`。
- **不复制论文文字与图**；方法、实验与论文表述须为本工作原创，并正确引用原作者。
- 本目录 PDF 仅作内部研究参考，不纳入版本库（见根目录 `.gitignore`）。
