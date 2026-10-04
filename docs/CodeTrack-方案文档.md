# CodeTrack 方案文档

> **版本**：v1.0（MVP-first 定稿）
> **状态**：主方法冻结，等待 Stage 0/Stage 1 前置验证
> **配套文档**：`docs/CodeTrack-实验文档.md`（执行计划、ablation、指标协议、go/no-go 判据）
> **一句话定位**：在 pretrained GOLA-B/DINOv2-B14 之上，用**结构化 syndrome 错误诊断 + H 路由的稀疏证据恢复 + 错误传播阻断**，把 GOLA 的表征变成 **Error-Correctable Tracking Representation**。

---

## 0. 阅读指引

本文档回答「**CodeTrack 到底是什么、为什么这么设计**」；实验文档回答「**怎么一步步证明它对**」。

全文的硬约束（不可协商，来自项目 owner）：

1. **最终主方法必须是 pretrained GOLA + CodeTrack 联合微调**；Frozen GOLA 只能作为消融，不能作为主模型。
2. **论文主线始终是 `Detect → Localize → Recover → Protect Memory`**，不是 "ECC + Motion + Diffusion" 的模块拼装。
3. **传统 20/50-step DDPM 不作主方法**；生成式恢复最多做到 2-step，且必须证明超过 1-step recolver 才有资格保留。
4. DINOv2 主干在主模型里**继续冻结**；GOLA 原有低秩参数**必须训练**。
5. 解冻 DINO 后 2 个 block 属于**增强实验**，不是默认主方案。

---

## 1. 结论先行

### 1.1 主方法定版

$$
\boxed{
\begin{aligned}
&\textbf{Pretrained GOLA-B / DINOv2-B14}\\
+&\textbf{Jointly Trainable GOLA Groups}\\
+&\textbf{Structured Syndrome Error Diagnosis}\\
+&\textbf{Fixed-Support / Learnable-Weight }H\\
+&\textbf{H-Routed Sparse 1/2-Step Token Refiner}\\
+&\textbf{Uncertainty-Aware Kalman Temporal Prior}\\
+&\textbf{Reliability-Gated Online Template Protection}
\end{aligned}
}
$$

### 1.2 三个核心贡献（不是 ECC / 运动 / 扩散）

$$
\boxed{\text{Structured Error Diagnosis}}
\qquad
\boxed{\text{Selective Evidence-Routed Recovery}}
\qquad
\boxed{\text{Error Propagation Prevention}}
$$

"联合训练" 的角色是把三者串成 $\boxed{\text{Error-Correctable Tracking Representation}}$。

### 1.3 优先级明确声明

$$
\boxed{
\text{1-step residual}
> \text{2-step iterative refiner}
> \text{2-step rectified flow}
> \text{consistency}
> \text{DDPM}
}
$$

**如果 two-step flow 最终 SR 没有显著超过 one-step refiner，就删掉 flow。** 不为故事保留 diffusion。

### 1.4 MVP（当前唯一应该动手的东西）

$$
\boxed{
\text{GOLA}
\rightarrow
\text{Syndrome}
\rightarrow
\text{Top-K Error Tokens}
\rightarrow
\text{H-Routed One-Step Recovery}
\rightarrow
\text{Original Head}
}
$$

MVP 阶段**完全删掉** diffusion、motion、temporal memory、template protection。

---

## 2. 文献调研：哪些工作真正对 CodeTrack 有启发

| 工作 | 和 CodeTrack 的关系 | 我们应该吸收什么 |
|---|---|---|
| **GOLA, AAAI 2026** | 直接 baseline | 不要破坏它的 PEFT 优势；继续训练 GOLA ranks |
| **LoRAT, ECCV 2024** | GOLA 的基础 tracking / PEFT 路线 | 说明 tracking 用 LoRA/PEFT 很合理 |
| **BAT, AAAI 2024** | RGB-T foundation model + adapter | 跨模态 complement 可以轻量实现 |
| **TBSI, CVPR 2023** | template 作为 RGB/TIR 信息桥梁 | template 很适合作为 error recovery reference |
| **PURA, CVPR 2025** | 防止在线适配漂移 | "恢复/保护已有知识" 思想可借鉴 |
| **SCDT, CVPR 2026** | 时空条件 denoising、模态缺失恢复 | **最容易撞车**，所以 diffusion 不能成为核心创新 |
| **MoKA-HP, 2026** | Kalman + historical prompts | 支持使用轻量 motion side information |
| **DTPTrack, CVPR 2026** | history reliability / 防 temporal drift | 强力支持"可靠历史才能进 memory" |
| **Denoising ViT, ECCV 2024** | feature-level denoising | 证明 feature recovery 本身是合理任务 |
| **Consistency / Flow Matching** | few-step generative recovery | 若做生成式恢复，比 DDPM 更合适 |
| **MaskGIT, CVPR 2022** | 并行预测 + 少量迭代 refinement | 少量迭代 refinement 比长自回归更高效 |

参考文献：

- GOLA (AAAI 2026): <https://ojs.aaai.org/index.php/AAAI/article/view/37843> ；PDF: <https://ojs.aaai.org/index.php/AAAI/article/download/37843/41805>
- LoRAT (ECCV 2024): <https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/113_ECCV_2024_paper.php>
- BAT (AAAI 2024): <https://ojs.aaai.org/index.php/AAAI/article/view/27852>
- SCDT (CVPR 2026): <https://openaccess.thecvf.com/content/CVPR2026/html/Lu_Spatio-Temporal_Conditional_Denoising_Transformer_for_Modality-Missing_RGBT_Tracking_CVPR_2026_paper.html>
- MoKA-HP: <https://www.sciencedirect.com/science/article/pii/S0925231225028358>
- DTPTrack (CVPR 2026): <https://openaccess.thecvf.com/content/CVPR2026/html/Huang_Drift-Resilient_Temporal_Priors_for_Visual_Tracking_CVPR_2026_paper.html>
- Denoising ViT (ECCV 2024): <https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/11504_ECCV_2024_paper.php>
- Consistency Models (ICML 2023): <https://proceedings.mlr.press/v202/song23a>
- MaskGIT (CVPR 2022): <https://openaccess.thecvf.com/content/CVPR2022/html/Chang_MaskGIT_Masked_Generative_Image_Transformer_CVPR_2022_paper.html>
- DINOv3: <https://ai.meta.com/research/publications/dinov3/>

### 2.1 论文定位的红线

GOLA 论文报告：GOLA-B 在 LasHeR 上 **77.5 PR / 73.9 NPR / 61.6 SR**，约 **99M 参数、85 GFLOPs、125 FPS**；且**全量微调反而低于 GOLA**。这直接说明不应该一上来全量解冻 DINO。

**SCDT 已在 CVPR 2026 做了 "spatio-temporal conditional denoising + missing/weak modality recovery"**，因此：

> ❌ 不能写："我们首次使用 diffusion 修复 RGBT tracking feature。"

> ✅ 真正应该抢占的是：**error diagnosis → structural error localization → selective token recovery → prevent propagation**

尤其这两个问题：

$$
\boxed{\text{Where is wrong?}}
\qquad\text{和}\qquad
\boxed{\text{From whom should this token be recovered?}}
$$

这才是 CodeTrack 与 SCDT 拉开距离的关键。

### 2.2 关于 DINOv3

**暂时不要把 DINOv2 换成 DINOv3。** DINOv3 的 dense feature 很强（Meta 也明确把 object tracking 列在其 dense-feature 能力中），但它会同时改变预训练权重、patch geometry 和 GOLA checkpoint，导致无法判断提升究竟来自 CodeTrack 还是更强 backbone。放到最后做 backbone-transfer experiment 更合适。

---

## 3. 官方 GOLA 代码核实结果（必须先改正图里的东西）

### 3.1 输入尺寸修正（🔴 必改）

> **GOLA-B 官方输入不是 template 128×128 / search 256×256，而是 template 112×112 / search 224×224。**

backbone 是 **DINOv2 ViT-B/14**，$P=14$，所以恰好：

$$
112/14=8,\qquad 224/14=16
$$

因而：

| 项 | 值 |
|---|---|
| template tokens | $8\times8=64$ |
| search tokens | $16\times16=256$ |
| channel | $C=768$ |

所以原图里的 **$N=256,\ C=768$ 是对的，但输入尺寸 128/256 必须改为 112/224。**

### 3.2 真实 token flow

官方 GOLA-B 的 token 流大致是：

$$
z_v,\ x_v,\ z_i,\ x_i,\ d_v,\ d_i
$$

其中：

$$
z_v,z_i,d_v,d_i\in\mathbb R^{B\times64\times768}
\qquad
x_v,x_i\in\mathbb R^{B\times256\times768}
$$

官方代码直接 concatenate：

$$
F_0=\big[\,z_v,\ x_v,\ z_i,\ x_i,\ d_v,\ d_i\,\big]
\;\Rightarrow\;
F_0\in\mathbb R^{B\times768\times768}
$$

> ⚠️ 注意：这里的第一个 768 是 **token 数**（$64+256+64+256+64+64=768$），第二个 768 是 **channel dimension**。

经过全部 DINOv2 Transformer blocks：

$$
F_L=\operatorname{Norm}\big(\operatorname{DINOv2}(F_0)\big)
$$

之后，官方当前 `_fuse_search()` 实现实际取的是 **TIR search 所在位置的 256 个 token**：

$$
X_t=X_t^{\text{TIR-pos}}\in\mathbb R^{B\times256\times768}
$$

再直接进入 **MLP Anchor-Free Head**。

> ⚠️ 这里虽然叫 TIR-position token，但它已经经过所有 RGB/TIR/template/online-template tokens 的 joint self-attention，所以**并不是"纯 TIR 特征"**。

### 3.3 CodeTrack 插入点

$$
\boxed{
\text{final DINO block}
\rightarrow
\text{LayerNorm}
\rightarrow
\textbf{CodeTrack}
\rightarrow
\text{原 GOLA Tracking Head}
}
$$

这是最干净的位置。**不要一开始改 DINO 内部 12 个 block。**

### 3.4 官方 template update 现状

$$
score_t>0.84 \;\Rightarrow\; \text{update}
$$

且 GOLA 自身 ablation 已显示 online template 对最终 tracking 很重要 —— 这是 CodeTrack 做 template protection 的合理性来源。

### 3.5 官方训练现状（影响 temporal 方案）

官方 GOLA 训练主要是 template / search / online-template 的 **pair sampling**，**不是真正连续的 causal clip**。

> 因此：**MVP 不要加 temporal memory。** 等核心 recovery 证明有效，再新写一个 4-frame causal sampler。

---

## 4. 张量流与形状表（改图依据）

### 4.1 双分支取出

最具决定性的一个改动：**同时把 final search-V 和 search-T 两个分支取出来。**

$$
\underbrace{X_t=X_t^{\text{TIR-pos}}\in\mathbb R^{B\times256\times768}}_{\text{baseline feature，与官方 GOLA 完全一致}}
$$

$$
\underbrace{X_t^{aux}=X_t^{\text{RGB-pos}}\in\mathbb R^{B\times256\times768}}_{\text{cross-modal recovery side information}}
$$

另外保留：

- $Z_0$：初始模板（initial template）
- $Z_{online}$：在线模板（online template）

于是主链是：

$$
X_t
\xrightarrow{\ \text{Diagnosis}\ }
q_t
\xrightarrow{\ \text{Sparse Recovery}\ }
X_t'
\xrightarrow{\ \text{Original GOLA Head}\ }
\hat b_t
$$

并且初始化时强制：

$$
\boxed{X_t'=X_t}
$$

这样加载 pretrained GOLA checkpoint 后，**第 0 个 iteration 不会把原模型性能炸掉**。这一点非常重要 —— 实现上通过 recovery head 的 **zero-init**（最后一层 $\Delta X$ 投影权重与 bias 清零）保证。

### 4.2 形状总表

| 符号 | 含义 | 形状 | 是否可训练 |
|---|---|---|---|
| $z_v,z_i,d_v,d_i$ | template / online-template tokens | $B\times64\times768$ | 输入 |
| $x_v,x_i$ | search tokens（RGB / TIR） | $B\times256\times768$ | 输入 |
| $F_0$ | concat 序列 | $B\times768\times768$ | 输入 |
| $F_L$ | final DINO block 输出（Norm 后） | $B\times768\times768$ | 冻结 |
| $X_t$ | TIR-pos search feature（baseline） | $B\times256\times768$ | 冻结产出 |
| $X_t^{aux}$ | RGB-pos search feature（aux） | $B\times256\times768$ | 冻结产出 |
| $U$ | 投影后 obs 特征 | $B\times256\times128$ | $W_x$ 可训练 |
| $R$ | 投影后 ref 特征 | $B\times256\times128$ | $W_r$ 可训练 |
| $H_0$ | Tanner graph sparse support | $\{0,1\}^{M\times256}$ | 固定 |
| $W$ | 边权 logits | $M\times256$ | 可训练 |
| $\bar H$ | 归一化边权矩阵 | $\mathbb R^{M\times256}$ | 由 $W,H_0$ 导出 |
| $C^{obs},C^{ref}$ | check-node 观测 / 参考 | $B\times M\times128$ | 导出 |
| $s$ | syndrome | $[0,1]^{B\times M}$ | 导出 |
| $q$ | token-level error severity | $[0,1]^{B\times256}$ | 导出 |
| $\mathcal E_t$ | Top-K 可疑 token 集 | $\vert\mathcal E_t\vert=K_{\max}=32$ | 导出 |
| $\Delta X_i$ | 残差修正量 | $B\times768$ | 可训练 |
| $X_t'$ | 恢复后 feature | $B\times256\times768$ | 导出 |
| $M_t$ | motion prior raster | $[0,1]^{B\times16\times16}$ | 导出（Stage 4） |
| $T_{mem}$ | temporal memory | $B\times4\times8\times128$ | 缓冲（Stage 4） |
| $\hat b_t$ | 输出 box | $B\times4$ | 导出 |

默认超参：$M=64$、$C_{mid}=128$、bottleneck $=256$、$K_{\max}=32$、邻居 top-8。

---

## 5. 模块一：Structured Syndrome Error Diagnosis（Detect + Localize）

### 5.1 设计原则：不假装是 GF(2) LDPC

**不做真正的 GF(2) LDPC 编解码。** 否则审稿人很容易问：

> 视觉连续 feature 为什么满足二元 parity equation？

正确定位是：

> **ECC-inspired structured redundancy graph**（受 ECC 启发的结构化冗余图）。

$H_0$ 只规定 Tanner graph 的 **sparse support**；真正的边权 $w_{ji}>0$ 是**可训练**的。

$$
\bar H_{ji}
=
\frac{H_{0,ji}\,\operatorname{softplus}(w_{ji})}
{\sum_k H_{0,jk}\,\operatorname{softplus}(w_{jk})}
$$

### 5.2 图规格

$$
\boxed{M=64}
$$

**不是图里的 16。** 消融做 $M\in\{32,64,96\}$。

度数设计目标：

| 量 | 目标 | 记号 |
|---|---|---|
| 每个 token 平均连接 check 数 | $\approx 3$ | $d_v$ |
| 每个 check 覆盖 token 数 | $\approx 12$ | $d_c$ |

一致性检查：$M\cdot d_c = 64\times12=768 = 256\times3=N\cdot d_v$ ✅

> 实现注记：用 **irregular** 构造（行/列度数采样后去重）严格满足"行和 768 / 列和约 3"很难同时精确；工程上按**每个 token 采 3 个 check、每个 check 采 12 个 token 后去重补足**，允许度分布轻微不均匀，并在论文中如实报告实测 $d_v,d_c$ 直方图与 overlap 比例。

**为什么 $M=64$ 而不是 16**：$M=16$ 时每个 check 平均覆盖 $768/16=48$ 个 token，check 过于"泛化"，syndrome 分辨率太粗，容易退化成全局 confidence；$M=64$ 给出 $B\times64$ 的 syndrome，粒度与 256 token 的定位需求匹配。

### 5.3 Syndrome 前向

**不要直接拿 768 维做 H 运算。** 先降维：

$$
U=W_x X_t\in\mathbb R^{B\times256\times128}
$$

$$
R=W_r X_t^{aux}+\text{TemplateContext}\in\mathbb R^{B\times256\times128}
$$

其中 $\text{TemplateContext}$ 由 $Z_0,Z_{online}$ 池化后广播得到。

然后：

$$
C^{obs}=\bar H U\in\mathbb R^{B\times64\times128}
$$

$$
C^{ref}=\bar H R\in\mathbb R^{B\times64\times128}
$$

> 这里 $\bar H$ 作用在 token 维（256）上，channel 维（128）逐通道独立。

每个 check 的 syndrome：

$$
s_j=
\operatorname{MLP}\Big[\,
C_j^{obs}-C_j^{ref},\;
\big|C_j^{obs}-C_j^{ref}\big|,\;
\cos\big(C_j^{obs},C_j^{ref}\big)
\,\Big]
$$

得到：

$$
s\in[0,1]^{B\times64}
$$

再反向投票（variable-node update）：

$$
q=\sigma\big(\bar H^\top s+b\big)
$$

得到：

$$
\boxed{q\in[0,1]^{B\times256}}
$$

即 **token-level error severity**。

### 5.4 ECC 逻辑的自洽性

$$
\text{variable node}
\rightarrow
\text{check node}
\rightarrow
\text{syndrome}
\rightarrow
\text{variable-node error probability}
$$

这条链路与 LDPC 的 belief propagation 一阶近似同构，但**每一跳都是可微、可训练的连续算子**，不依赖 GF(2) 假设。

### 5.5 诊断监督：不用 corruption mask 作为唯一目标（成败关键）

如果直接令 $q_i=y_i^{mask}$，模型很可能只学会：

> "识别你人工注入的 block erase。"

这正是要避免的。改用 **Clean Teacher Residual Target**。

同一训练 sample 走两条路径：

$$
I^{clean}\rightarrow\text{GOLA teacher}\rightarrow X_i^{*}
$$

$$
I^{corr}\rightarrow\text{CodeTrack student}\rightarrow X_i
$$

**teacher 不反传。**

特征退化度：

$$
e_i^{feat}=1-\cos\big(X_i,\ X_i^{*}\big)
$$

任务退化度（利用 GOLA head 的 tracking discrepancy）：

$$
e_i^{task}=\big|\sigma(p_i)-\sigma(p_i^{*})\big|+\beta\, d_{box}\big(b_i,\ b_i^{*}\big)
$$

最终误差目标：

$$
e_i^{*}=\alpha\, e_i^{feat}+(1-\alpha)\, e_i^{task},
\qquad \alpha=0.5\ \text{(第一版)}
$$

也就是说：

> **模型学习的不是"哪里被我遮掉了"，而是"哪里的 representation 真正偏离了 clean tracking representation"。**

这会比 binary corruption mask 强很多。

### 5.6 诊断伪代码

```python
def syndrome_diagnosis(X_t, X_t_aux, Z0, Z_on, H0, W_edges, mlp_s, b_vote):
    """
    X_t, X_t_aux : (B, 256, 768)
    H0           : (64, 256) bool/int sparse support
    W_edges      : (64, 256) learnable, masked by H0
    returns q    : (B, 256) error severity in [0, 1]
    """
    U = W_x(X_t)                       # (B, 256, 128)
    ctx = template_context(Z0, Z_on)   # (B, 128)  -> broadcast to (B, 256, 128)
    R = W_r(X_t_aux) + ctx             # (B, 256, 128)

    H_bar = normalize_edges(H0, W_edges)          # (64, 256), rows sum to 1
    C_obs = einsum('mt,btc->bmc', H_bar, U)       # (B, 64, 128)
    C_ref = einsum('mt,btc->bmc', H_bar, R)       # (B, 64, 128)

    diff  = C_obs - C_ref                          # (B, 64, 128)
    s = mlp_s(cat([diff, diff.abs(), cosine(C_obs, C_ref)], dim=-1))  # (B, 64, 1) -> (B, 64)
    s = s.sigmoid().squeeze(-1)

    q = torch.sigmoid(einsum('mt,bm->bt', H_bar, s) + b_vote)  # (B, 256)
    return q, s
```

---

## 6. 模块二：Selective Evidence-Routed Recovery（Recover）

### 6.1 第一版不做 diffusion

项目 owner 已明确要求：第一步先验证

$$
\text{GOLA} + \text{error diagnosis} + \text{selective recovery}
$$

是否有价值，再决定是否加入完整 diffusion / temporal memory。

**第一版就做 Syndrome-Guided Sparse Residual Refiner。**

### 6.2 Top-K 选择

$$
\mathcal E_t=\operatorname{TopK}\big(q_t\big),
\qquad
\boxed{K_{\max}=32}
$$

即最多修：

$$
32/256 = 12.5\%
$$

的 token。

**对健康 token 严格执行 identity bypass**（$X_i'=X_i$），不靠 loss 软约束。

### 6.3 H 路由的邻居选择

对于 corrupted token $i$，从

$$
A=\bar H^\top \bar H
$$

找它 Tanner graph 上最相关的可靠 token：

$$
\mathcal N_H(i)\cap\mathcal R_t
$$

取 top-8。

> $\mathcal R_t$ 是可靠 token 集（如 $\{i: q_i<\tau_{rel}\}$）。若邻居不足 8 个，用自身 + 空间最近邻补齐，避免空集。

### 6.4 Refiner 结构

condition 包括：

$$
\big\{\, X_{\mathcal N_H(i)},\ X^{aux}_i,\ Z_0,\ Z_{online} \,\big\}
$$

用一个非常轻的 bottleneck：

$$
768\rightarrow256
\rightarrow
\text{Sparse Cross Attention}
\rightarrow256
\rightarrow768
$$

预测残差：

$$
\Delta X_i
$$

然后：

$$
\boxed{X_i'=X_i+q_i\,\Delta X_i}
$$

健康 token：

$$
X_i'=X_i
$$

权重 $q_i$ 作为软门控有三重作用：(a) 让"诊断不确定"的 token 只被轻微修改；(b) 让 $q$ 直接进入主跟踪梯度路径（诊断模块不只被 $L_{diag}$ 监督）；(c) 天然满足 "正常 token 尽量不要动"。

**实现要求**：$\Delta X$ 输出层 zero-init，保证 step 0 时 $X_t'=X_t$。

### 6.5 恢复伪代码

```python
def sparse_recovery(X_t, X_t_aux, q, H_bar, Z0, Z_on, K_max=32, n_nb=8):
    """
    returns X_rec (B, 256, 768) with identity bypass on healthy tokens
    """
    X_rec = X_t.clone()                                  # identity baseline
    idx   = q.topk(K_max, dim=1).indices                 # (B, K_max)

    A = H_bar.transpose(0, 1) @ H_bar                    # (256, 256) token-token affinity

    b = torch.arange(X_t.shape[0], device=X_t.device)[:, None]
    nb = A[idx].topk(n_nb, dim=-1).indices                # (B, K_max, n_nb)
    nb_rel = X_t[b[:, :, None], nb]                       # (B, K_max, n_nb, 768)

    cond = torch.cat([
        nb_rel.flatten(2),                                # (B, K_max, 8*768)
        X_t_aux[b[:, :, None], idx],                      # (B, K_max, 768)
        Z0.mean(1)[:, None].expand(-1, K_max, -1),        # (B, K_max, 768)
        Z_on.mean(1)[:, None].expand(-1, K_max, -1),      # (B, K_max, 768)
    ], dim=-1)

    h = down_proj(cond)                                   # (B, K_max, 256)
    h = sparse_cross_attn(h, kv=nb_rel.flatten(2))        # (B, K_max, 256)
    dX = up_proj(h)                                       # (B, K_max, 768), zero-init

    gate = q[b, idx].unsqueeze(-1)                        # (B, K_max, 1)
    X_rec[b, idx] = X_rec[b, idx] + gate * dX             # healthy tokens untouched
    return X_rec
```

### 6.6 如果第一版成功，再做"生成式恢复"

优先级同上（1-step > 2-step iter > 2-step rectified flow > consistency > DDPM）。

若要真正带 generative formulation，**最推荐 conditional flow matching / rectified flow**，因为有天然 paired data：

$$
X^{corr}\leftrightarrow X^{clean}
$$

定义插值路径：

$$
X_\tau=(1-\tau)X^{clean}+\tau X^{corr}
$$

训练 velocity：

$$
v_\theta(X_\tau,\tau,C)\ \ \text{学习}\ \ X^{corr}\rightarrow X^{clean}
$$

测试时只 Euler 2 step。

**删除判据**：two-step flow 的 SR 未显著超过 one-step refiner $\Rightarrow$ 删掉 flow。

理论支撑：MaskGIT 证明"并行预测 + 少量迭代 refinement"比长自回归更高效；Consistency Models 专门解决 diffusion 多步采样慢的问题；Denoising ViT 表明轻量 Transformer 即可把有问题的 ViT feature 映射到更干净的 representation，不一定需要 heavyweight generative model。

---

## 7. 模块三：Motion Prior（Stage 4）

### 7.1 选型：Kalman，不选 Mamba / Transformer

不要贪大。MoKA-HP 已证明 RGB-T tracking 中 **Kalman motion modeling + historical feature prompts** 是实用、低开销的路线；DTPTrack 进一步指出 temporal history 最大的问题不是"历史不够多"，而是**错误历史会积累并导致 drift，所以必须估计历史可靠性**。

### 7.2 状态与预测

$$
s_t=\big[c_x,\ c_y,\ \log w,\ \log h,\ v_x,\ v_y,\ v_w,\ v_h\big]^\top
$$

标准 constant-velocity KF：

$$
s_{t|t-1}=A\,s_{t-1|t-1}
$$

获得：

$$
\hat b_t^{motion}
$$

再把 Gaussian rasterize 到：

$$
M_t\in[0,1]^{16\times16}
$$

### 7.3 关键不是位置，而是 covariance

$$
P_{t|t-1}
$$

定义 motion uncertainty：

$$
u_t=f\big(\operatorname{tr}P_t,\ \text{innovation}_t\big)
$$

因此：

- 匀速稳定运动 → motion prior 权重大；
- 快速变向 → uncertainty 增大；
- 相机快速运动 → uncertainty 增大；
- 遮挡恢复 → uncertainty 增大。

即：

$$
\boxed{\text{motion prior 是软证据，不是硬 gate}}
$$

这正适合"error-correction side information"的定位，也直接对应风险 #5（motion 在 fast motion 时反而害人）。

---

## 8. 模块四：Temporal Memory（Stage 4）

### 8.1 存储预算

先不要存 $256\times768\times\text{多帧}$ —— 太重。

建议：

$$
K=3\sim4
$$

个历史**可靠**帧。每个历史帧只存：

- box；
- motion state；
- reliability；
- 4–8 个 target-region pooled tokens；
- 每个 token 压到 128-d。

例如：

$$
T_{mem}\in\mathbb R^{B\times4\times8\times128}
$$

非常轻。

### 8.2 加权聚合

$$
w_k\propto r_k\exp(-\gamma\,\Delta t_k)
$$

即：**reliability 越高、时间越近，权重越大**。

### 8.3 前置依赖（硬约束）

官方 GOLA 训练是 pair sampling，不是 causal clip，所以：

> **MVP 不要加 temporal memory。** 等核心 recovery 证明有效，再新写一个 4-frame causal sampler。

---

## 9. 模块五：Online Template Protection（Protect Memory）

### 9.1 为什么值得保留

这个部分把论文从

> feature denoising

提升成

> **error propagation control**

### 9.2 设计

官方当前逻辑：

$$
score_t>0.84\Rightarrow \text{update}
$$

CodeTrack 改成：

$$
c_t=\sigma\Big(\operatorname{MLP}\big[\ score_t,\ \operatorname{meanTopK}(q),\ \max(q),\ u_t,\ r_t^{rec}\ \big]\Big)
$$

推理时：

$$
score_t>0.84\ \land\ c_t>\tau_c
$$

才更新。**第一版用 hard gate。**

### 9.3 不要一开始做 soft EMA

不要一开始做：

$$
Z_{online}^{t+1}=\alpha Z_t+(1-\alpha)Z_{online}^{t}
$$

因为官方 updater 保存的是新的 **image crop**，图像 EMA 很容易产生 ghosting。之后如果改成 feature memory，再考虑 soft EMA。

### 9.4 有效性的度量前提

`Protect Memory` 要成为论文贡献而非架构图上的一个框，必须有度量（见实验文档）：`WrongUpdateRate`、first failure frame、recovery after failure、template contamination duration 等。

---

## 10. 联合训练方案（硬约束落地）

### 10.1 必须是 Joint PEFT，不是 Full Fine-Tuning

GOLA 本身已给出实验证据：全量微调在这个任务里比 GOLA PEFT 差。

$$
\boxed{\text{Joint PEFT}}
\quad\text{而不是}\quad
\text{Full Fine-Tuning}
$$

### 10.2 参数划分

**冻结：**

$$
\text{DINOv2 original foundation weights}
$$

**训练：**

$$
\text{GOLA low-rank / group parameters}
\;+\;
\text{token type embeddings}
\;+\;
\text{GOLA head}
\;+\;
\text{CodeTrack}
$$

这已经完全满足"representation 与 CodeTrack co-adapt"要求，而不是 Frozen Adapter。

### 10.3 参数组学习率

| 参数 | LR |
|---|---:|
| CodeTrack diagnosis / recovery | **1e-4** |
| GOLA low-rank parameters | **2e-5 ~ 3e-5** |
| token-type embedding | **2e-5 ~ 3e-5** |
| original GOLA head | **1e-5** |
| optional LayerNorm affine | **1e-5** |
| Stage-3 DINO last 2 blocks | **1e-6 ~ 2e-6** |

理由：

- **CodeTrack 最大** —— 它是随机初始化的；
- **GOLA 小** —— 已经是一个很好的 tracking checkpoint；
- **head 更小** —— 原来的 decision surface 已经不错，不希望它为了 synthetic corruption 快速漂移。

若后面解冻 DINO：

$$
lr_{DINO}\ll lr_{GOLA}\ll lr_{CodeTrack}
$$

**不建议默认解冻 last 4/6 blocks。先 last 2 就够了。**

### 10.4 Optimizer / scheduler

延续 GOLA 训练习惯（AdamW、cosine schedule、warm-up、gradient clipping、AMP；10 epoch、batch 128、131,072 pairs/epoch）。

CodeTrack 推荐：

| 项 | 值 |
|---|---|
| optimizer | AdamW |
| $\beta$ | $(0.9,\ 0.999)$ |
| weight decay | $0.05\sim0.1$ |
| bias / LayerNorm / embedding | WD $=0$ |
| scheduler | 5% warmup + cosine decay |
| gradient clip | $1.0$ |
| AMP | 开 |
| EMA | **MVP 不开** |
| batch（单卡 24GB） | $8\sim16$ |
| gradient accumulation | 至 effective batch $=64$ |

### 10.5 最终 Loss

不要十几个 loss。最终：

$$
\boxed{
\mathcal L
=
\mathcal L_{track}^{corr}
+
0.25\,\mathcal L_{track}^{clean}
+
\lambda_{diag}\mathcal L_{diag}
+
\lambda_{rec}\mathcal L_{rec}
+
\lambda_{pres}\mathcal L_{pres}
+
1.4\times10^{-3}\,\mathcal L_{GOLA\text{-}orth}
+
\lambda_{mem}\mathcal L_{mem}
}
$$

**主角永远是 $\mathcal L_{track}$。**

GOLA 原 tracking loss 继续用：

$$
L_{cls}^{BCE}+L_{box}^{GIoU}
$$

**Diagnosis：**

$$
L_{diag}=\operatorname{BCE}(q,\ e^{*})+0.5\,\operatorname{SmoothL1}(s,\ s^{*})
$$

建议 $\lambda_{diag}=0.5$。

**Recovery**（只在 corrupted / suspect token 上）：

$$
L_{rec}=1-\cos\big(X_i',\ X_i^{*}\big)+0.25\,\operatorname{Huber}\big(X_i',\ X_i^{*}\big)
$$

建议 $\lambda_{rec}=0.2\sim0.3$。

**Preserve：**

$$
L_{pres}=\frac{1}{|\mathcal R|}\sum_{i\in\mathcal R}\big\|X_i'-X_i\big\|_1
$$

如果代码层已经严格 identity bypass，它甚至可以删掉。

> ⚠️ **梯度路径说明（实现时务必确认）**：$L_{rec}$ / $L_{diag}$ 的 teacher 支路 $X^{*},p^{*},b^{*}$ 必须 `detach`，且 teacher 前向不进入反向图（`torch.no_grad()`），否则会导致显存翻倍并污染 GOLA 冻结参数。但 $L_{track}^{corr}$ 必须**完整反传进 CodeTrack 与 GOLA adapters** —— 这才是 "learning error-correctable tracking representations" 的实现。

### 10.6 参数分组示意

```python
# 训练：GOLA low-rank / token-type emb / head / CodeTrack
# 冻结：DINOv2 foundation
param_groups = [
    {"params": codetrack.diagnosis_params(),  "lr": 1e-4,  "weight_decay": 0.05},
    {"params": codetrack.recovery_params(),   "lr": 1e-4,  "weight_decay": 0.05},
    {"params": gola.low_rank_params(),        "lr": 2.5e-5, "weight_decay": 0.05},
    {"params": gola.token_type_embeddings(),  "lr": 2.5e-5, "weight_decay": 0.0},
    {"params": gola.head_params(),            "lr": 1e-5,  "weight_decay": 0.05},
    {"params": norm_affine_params(),          "lr": 1e-5,  "weight_decay": 0.0},
]
# Stage 3 追加：
# {"params": dinov2.blocks[-2:].params(),   "lr": 1.5e-6, "weight_decay": 0.05}
```

---

## 11. Corruption Curriculum（训练数据构造）

### 11.1 三层验证协议（必须保留）

$$
\boxed{\text{Seen}\ \rightarrow\ \text{Held-out}\ \rightarrow\ \text{Natural}}
$$

### 11.2 训练 batch 配比

| 类型 | 占比 |
|---|---:|
| clean | 30% |
| image corruption | 45% |
| feature / token corruption | 20% |
| compound corruption | 5% |

注意：**30% clean 不是浪费** —— 它是抑制 GOLA 原始能力被破坏的关键正则（风险 #6）。

### 11.3 具体 corruption

**Image corruption — RGB：**

- gamma low-light；
- overexposure；
- Gaussian noise；
- blur；
- occlusion。

**Image corruption — TIR：**

- gain / offset；
- saturation；
- thermal crossover approximation；
- local contrast suppression。

**Feature corruption：**

- block erase；
- Gaussian feature noise；
- token replacement；
- cross-modal mismatch；
- partial modality drop。

### 11.4 最重要的原则

> **corruption mask 不作为唯一 error target。依然用 clean teacher residual。**

---

## 12. 训练阶段（不要一口气端到端）

### Stage 0 — Baseline reproduction

完全不训练。先复现：

$$
\text{GOLA-B}\approx 77.5 / 73.9 / 61.6
\quad(\text{LasHeR PR / NPR / SR})
$$

**如果这一步没对上，什么都别加。**

### Stage 1 — CodeTrack stabilization

约 $500\sim1000$ steps。GOLA 暂时 frozen，只训练：

$$
\text{Diagnosis} + \text{1-step Refiner}
$$

> 这里暂时冻结 GOLA 是 **warm-up**，不违反硬约束，因为它不是最终模型。

### Stage 2 — 主训练

$6000\sim8000$ optimizer steps，开启：

$$
\boxed{\text{GOLA adapters} + \text{head} + \text{CodeTrack}}
$$

联合训练。Tracking loss 必须直接：

$$
X'\rightarrow \text{Head}\rightarrow L_{track}
$$

反传进 CodeTrack 和 GOLA adapters。

### Stage 3 — 可选

只解冻：

$$
\text{DINOv2 last 2 blocks}
$$

约 $1500\sim2000$ steps，LR $1\sim2\times10^{-6}$。**如果没有提升，就不要放进最终模型。**

### Stage 4 — Temporal extension

只有前三阶段成功以后，再加入：

$$
\text{KF} + \text{temporal memory} + \text{memory protection}
$$

此时改 causal clip training。

---

## 13. 计算量目标

GOLA-B 本身：$99\text{M}$ Params、$85\text{G}$ FLOPs、$125$ FPS。

**CodeTrack diagnosis + refiner（256-d bottleneck）：**

$$
\sim 2\sim3\text{M}
$$

新增参数是合理目标。$H$ 本身只是 sparse edges，参数量几乎可以忽略。

如果每帧最多修复 32 tokens：

$$
K/N\le 12.5\%
$$

因此真实新增 FLOPs 有机会控制在：

$$
< 1\sim2\text{ G}
$$

级别。

### 13.1 论文最终目标（设计目标，非已测结果）

| 场景 | 目标 |
|---|---|
| bypass frame | GOLA 原速度的 **95%+** |
| corrected frame | 不低于原速度的 **75–85%** |
| average FPS | 最好 $\ge 100$ FPS |

> 这些是**设计目标 / 估计**，最终必须同卡 profile。

### 13.2 参数量核对清单

- [ ] 统计 `sum(p.numel() for p in model.parameters() if p.requires_grad)`，确认训练参数量落在 $2\sim3\text{M}$（不含 GOLA 已有可训练组）。
- [ ] 单独统计 CodeTrack 模块参数量与 GOLA low-rank 参数量，分别报告。
- [ ] $\bar H$ 只有 $64\times256$ 个标量，但存成 dense 会占显存；用 sparse 或 `masked_fill` 后仅保留 support。

---

## 14. 风险登记表

| # | 风险 | 预警信号 | 应对 | 降级/放弃判据 |
|---|---|---|---|---|
| 1 | **syndrome 退化成普通 confidence predictor** | 按 confidence 分桶后 syndrome AUROC 优势消失 | 不输入 head confidence；clean teacher residual supervision；做 confidence-stratified AUROC | 若 Syndrome AUROC $\le$ Confidence AUROC（匹配参数量下），核心假设不成立，需重设计 |
| 2 | **feature reconstruction 更好，但 SR 不涨** | $L_{rec}$ 下降而 SR 平坦 | $L_{track}$ 始终是主监督，降低 $L_{rec}$ | 若 $\lambda_{rec}\to0$ 后 SR 仍不涨，说明 recovery 无任务价值 |
| 3 | **H-routing 不比 spatial kNN 强** | 消融 D vs E 无差异 | 保留 syndrome diagnosis，把 H routing 降级 | 若 E $\approx$ D，不硬吹 Tanner routing，改写为"结构化先验" |
| 4 | **diffusion 根本没用** | 2-step flow $\le$ 1-step refiner | 直接删掉 diffusion | 已预设判据：SR 无显著提升即删 |
| 5 | **motion 在 fast motion 时反而害人** | FM 属性 SR 下降 | KF covariance + innovation 做 uncertainty gate | 若 gated 后仍伤 FM，删掉 motion prior |
| 6 | **joint tuning 把 GOLA 原本能力破坏** | clean SR 掉 > 0.3 | 小 LR、zero-init recovery、30% clean samples、保持 GOLA orthogonal loss | clean SR 持续下降则回退至更大 frozen 比例（但需保持硬约束：不能是纯 Frozen 主模型） |

---

## 15. 不做什么（负面清单）

明确排除，避免范围蔓延：

1. ❌ 真正的 GF(2) LDPC 编解码；
2. ❌ 20/50-step DDPM；
3. ❌ 修改 DINOv2 内部 12 个 block（主模型）；
4. ❌ MVP 阶段引入 temporal memory（依赖 causal clip，先不做）；
5. ❌ 图像级 soft EMA template update（ghosting）；
6. ❌ 更换 DINOv3 作为主 backbone；
7. ❌ 十几个 loss 的堆叠；
8. ❌ 把主贡献叙述成 "ECC + Motion + Diffusion"。

---

## 16. 术语表

| 术语 | 含义 |
|---|---|
| **Syndrome** | 由 check-node 观测-参考差异导出的结构化不一致度 |
| **Variable node** | 对应 256 个 search token 的图节点 |
| **Check node** | 对应 $M=64$ 个 parity 约束的图节点 |
| **$\bar H$** | 固定 support + 可学习边权归一化后的稀疏矩阵 |
| **$q$** | token-level error severity，$\in[0,1]^{256}$ |
| **Clean Teacher Residual** | 用 clean 前向结果作为误差监督目标，替代 corruption mask |
| **Error-Correctable Representation** | 经 joint training 后，表征本身适应 diagnosis/recovery |
| **Protect Memory** | 对 online template 更新加可靠性门控，阻断错误传播 |
| **H-routed** | 恢复时的证据来源由 Tanner graph 邻居决定，而非空间邻居 |

---

## 17. 附录：默认超参速查

```yaml
backbone:
  name: dinov2_vitb14
  patch: 14
  dim: 768
  frozen: true            # 主模型冻结；Stage 3 可选解冻最后 2 个 block

input:
  template: 112           # tokens 8x8 = 64
  search: 224             # tokens 16x16 = 256

codetrack:
  mid_dim: 128            # U, R, C 的通道
  bottleneck: 256         # refiner hidden
  num_checks: 64          # M，消融 {32, 64, 96}
  topk_tokens: 32         # K_max
  num_neighbors: 8        # H-route top-8
  recovery_init: zero     # dX 输出层 zero-init，保证 step0 identity

loss:
  w_track_corr: 1.0
  w_track_clean: 0.25
  lambda_diag: 0.5
  lambda_rec: 0.25        # 0.2 ~ 0.3
  lambda_pres: 0.01       # identity bypass 严格时可删
  w_gola_orth: 1.4e-3
  lambda_mem: 0.0         # Stage 4 再打开

optim:
  optimizer: adamw
  betas: [0.9, 0.999]
  weight_decay: 0.05
  warmup_ratio: 0.05
  scheduler: cosine
  grad_clip: 1.0
  amp: true
  ema: false              # MVP 不开

corruption_mix:
  clean: 0.30
  image: 0.45
  feature: 0.20
  compound: 0.05
```

---

*文档结束。实验计划、ablation 树、指标协议与 go/no-go 判据见 `docs/CodeTrack-实验文档.md`。*
