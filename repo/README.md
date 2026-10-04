# 参考代码仓库

CodeTrack 论文库（`docs/paper/`）对应的官方/原作者实现。**只读参考**：不修改、不直接复制进 `trackit/`。

克隆方式：`git clone --depth 1 <url>`（浅克隆）。克隆时间 2026-10-04。

---

## 论文 ↔ 仓库对照表

| 目录 | 对应论文（`docs/paper/`） | 来源 | 说明 |
|---|---|---|---|
| `GOLA/` | `01_S_GOLA_..._AAAI2026.pdf` | <https://github.com/MelanTech/GOLA> | **主框架底座**。与 `third_party/GOLA` 同一 commit `339c737` |
| `MMLoRAT/` | 同上（GOLA 同作者的 RGB-X 扩展） | <https://github.com/MelanTech/MMLoRAT> | GOLA 作者后续工作，支持 LasHeR / online template / RGB-X 联合训练，**数据与评测问题可参考** |
| `Denoising-ViT/` | `02_S_DenoisingVisionTransformers_ECCV2024.pdf` | <https://github.com/Jiawei-Yang/Denoising-ViT> | 提供 DINOv2 feature denoising 实现，对应 Sparse Token Refiner 接口 |
| `DTPTrack/` | `03_S_DTPTrack_..._CVPR2026.pdf` | <https://github.com/NorahGreen/DTPTrack> | TRC / TGS（temporal reliability + guidance）。同为 TrackIt 体系，迁移成本低 |
| `TBSI/` | `04_S_TBSI_..._CVPR2023.pdf` | <https://github.com/RyanHTR/TBSI> | template-bridged search interaction |
| `LoRAT/` | `09_A_LoRAT_..._ECCV2024.pdf` | <https://github.com/LitingLin/LoRAT> | GOLA 的上游框架（`trackit` 即来自此处） |
| `BAT/` | `10_B_BAT_..._AAAI2024.pdf` | <https://github.com/SparkTempest/BAT> | 双向跨模态 adapter |
| `MaskGIT/` | `12_A_MaskGIT_..._CVPR2022.pdf` | <https://github.com/google-research/maskgit> | 官方 Jax 实现（仅算法参考） |
| `Consistency-Models/` | `13_B_ConsistencyModels_arXiv2303.01469.pdf` | <https://github.com/openai/consistency_models> | few-step 生成式恢复 |
| `FlexTrackV2/` | `07_A_FlexTrack_..._ICCV2025.pdf` | <https://github.com/supertyd/FlexTrackV2> | **FlexTrack v1 + v2 的实际代码都在这里**（v1 的 `supertyd/FlexTrack` 是占位仓库，已删）。提供 BMR（双向跨模态重建）+ CMA（完整性课程蒸馏）；对应 clean-teacher / 退化学生思路 |
| `RGBT-Tracking-List/` | —（论文汇总清单，非某篇论文代码） | <https://github.com/DRGBT-Tracking/RGBT_Tracking> | 查重与撞车检索用，已整理到 2026 年 |

---

## 已删除：无核心代码的仓库

以下仓库 clone 后核实**不含任何源代码**，已按"占位仓库即删除"的约定移除。仓库地址保留在此，供日后复查作者是否补放代码。

| 原仓库 | 对应论文 | 实际内容 | 处理 |
|---|---|---|---|
| `supertyd/FlexTrack` | `07_A_FlexTrack_..._ICCV2025.pdf` | 仅 `README.md` + `LICENSE`，README 明确指向 `FlexTrackV2` 为主代码仓库 | 已删。v1/v2 代码均已在 `FlexTrackV2/` |
| `NorahGreen/RTracker` | `14_B_RTracker_..._CVPR2024.pdf` | 仅 `README.md` + `RTracker_PAPER.pdf`，**无代码**（官方至今未放出实现） | 已删。仓库自带的 18 页论文版已保留为 `docs/paper/14b_B_RTracker_full_arXiv2403.19242.pdf` |

> 保留项：`RGBT-Tracking-List/` 源码文件数同样为 0，但它的内容**本身就是论文清单**（不是某篇论文的代码占位仓库），
> 故保留作为查重与撞车检索工具。

---

## 无可用仓库的论文

| 论文 | 状态 |
|---|---|
| `06_A_SCDT_..._CVPR2026.pdf` | **未开源**。已从 CVF 论文全文提取 URL，无任何 github 链接 |
| `MoKA-HP`（付费墙，未入库） | 未找到公开仓库 |
| `TPAMI` Template-Bridged（付费墙，未入库） | 未找到公开仓库 |
| `14_B_RTracker`（CVPR 2024） | 官方仓库无代码，见上表 |

---

## 许可与使用约定

- 代码复用须遵守各仓库自身 license；商用前逐仓库确认。
- `repo/` 与 `third_party/` 均为**只读参考**，不直接 import 进主工程。
- 不复制论文文字与图；方法、实验与论文表述须为本工作原创并正确引用原作者。
- 本目录与 `docs/paper/` 均不纳入版本库（见根目录 `.gitignore`）。
