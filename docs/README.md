# CodeTrack 文档索引

| 文档 | 内容 | 何时看 |
|---|---|---|
| `CodeTrack-方案文档.md` | 方法/架构/算法/训练配置/风险登记 | 写 Method、改架构前 |
| `CodeTrack-实验文档.md` | 执行计划、ablation 树、指标协议、go/no-go | 设计实验前 |
| **`implementation.md`** | **实现记录**：模块与张量形状、新增文件、对上游 4 处改动、checkpoint、参数量、梯度连通、训练/推理结果 | **接手代码第一份** |
| **`motion_design.md`** | 运动/时序模块的文献依据与实现决策（DTPTrack TRC/TGS、MoKA-HP 放弃说明） | 改运动/时序前 |
| `setup.md` | 环境修复与 GOLA 代码核实记录（turbojpeg、CuBLAS、safetensors、token 布局、掩码语义） | 环境出问题时 |
| `dtp_extraction.md` | DTPTrack 机制逐条提取（含代码↔论文差异、消融数值） | 引用 DTPTrack 时 |
| `paper/README.md` | 论文库清单（13 篇应用论文）+ 缺失说明 | 找文献 |
| `paper/math/README.md` | 数学工具源头论文（9 篇）+ 表述红线 | 写公式体系时 |
| `../repo/README.md` | 参考代码仓库 ↔ 论文对照 | 找参考实现 |

## 快速开始

```bash
source scripts/00_env.sh          # 必需：turbojpeg + CuBLAS 确定性
bash scripts/preflight.sh         # 环境/权重/数据集自检

# 验证（不训练）
"$PYTHON" tools/codetrack_verify.py    # 恒等性 + checkpoint + 形状审计 + 梯度连通

# 单序列联合训练
bash scripts/codetrack_train_smoke.sh

# 单序列推理（官方 eval pipeline）
bash scripts/codetrack_eval_single.sh
```

## 当前状态（2026-10-04）

- 架构已按图实现并逐节点核对形状；`X_t'=X_t` 在 step 0 成立（`max|Δscore_map| ≈ 9e-5`）。
- checkpoint 1311/1311 键命中，`unexpected = 0`。
- 10 个新模块**全部**非零梯度（真实 criterion 反传）。
- 单序列联合训练：loss 22.84 → 19.95，无 NaN/Inf，689 个 GOLA 张量同步更新。
- 单序列推理：success 0.7967、precision 1.0000、SR@0.5 1.0000。

未来计划（详见 `implementation.md` §9）：跑完整 LasHeR 训练、按实验文档的 ablation 树推进、在连续序列训练中体现运动先验损失的作用。
