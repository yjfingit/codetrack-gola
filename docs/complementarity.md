# 互补性理论验证

问题：**这 6 个模块的数据通路，理论上能否互补？代码里有没有写错的地方？**

方法：把"互补"拆成 4 条可证伪的性质，每一条都写成**可执行的检查**（`tools/complementarity_check.py`），而不是散文论证。当前结果 **16/16 通过**。

```bash
export LD_LIBRARY_PATH=/home/yangjuanfeng/lab/tools/libjpeg-turbo/root/usr/lib/x86_64-linux-gnu
python tools/complementarity_check.py
```

---

## 互补性的定义（为什么是这 4 条）

两个模块互补 ⟺ **它们不是同一个变量的函数**。因此必须逐条排除"退化成同一件事"的所有途径：

| 性质 | 退化途径 | 如果违反会怎样 |
|---|---|---|
| **P1 输入通道独立** | 两个模块读同一个信息源 | 它们是一个变量的两个函数，加和/拼接无新增信息 |
| **P2 操作对象不同** | 两个模块对同一数学对象做同类运算 | 只是同一算子的两套参数（参数冗余，不是结构互补） |
| **P3 通道被切断** | 某模块能看到"答案"（corruption mask / clean 特征） | 它会绕过自己的推理去抄标签，与其他模块不再依赖彼此 |
| **P4 无旁路** | 辅助信号直接进 head | 主路径被绕过，模块不被任务梯度驱动，也就学不到互补的东西 |

---

## P1 输入通道独立 —— 每个模块的独有信息源

| 模块 | **独有的**输入 | 其他模块看不到它 |
|---|---|---|
| KalmanMotionPrior | `gt_box (B,4)` + `image_size (B,2)` | ✅ 唯一的**时间/位移**信息来源 |
| SyndromeDiagnosis | `X_t, X_aux (B,256,768)` + `H_bar` | ✅ 唯一的 **token 网格 + 跨模态一致性** |
| TemporalMemory | `reliability = 1−q` + `target_mask` | ✅ 唯一的**跨帧**信息来源 |
| H_RoutedSparseRefiner | `neighbour_index (256,8)` | ✅ 唯一的 **`HᵀH` 图关系** |
| NoiseModulatedDenoiser | `syndrome s (B,64)` + `alpha` | ✅ 唯一的**全格迭代**修正 |
| TemplateProtectionGate | `score` + `u_t` + `r_rec` | ✅ 唯一的**帧级决策**输出 |

**关键论证**：搜索 token 是**单帧**的，所以"时间"这一维信息**只存在于 Kalman 模块**。这就从信息论上保证了运动模块不可被其他模块替代 —— 与 DTPTrack Table 6 的实证一致（用手写运动先验替换学习式 TGS 会掉点，因为运动只是**一路**信息，不能替代时序聚合）。

同理：
- **跨模态一致性**只存在于 `C_obs` vs `C_ref` 的差异里（诊断模块独有）
- **跨帧**信息只存在于记忆库（记忆模块独有）
- **图关系**只存在于 `neighbour_index`（refiner 独有）

---

## P2 操作对象不同 —— 六个不同的数学对象

| 模块 | 对象 | 形状 |
|---|---|---|
| diagnosis | **token 集合** → check 节点 | `(B,256,768) → (B,64,128) → q (B,256)` |
| motion | **状态向量** → 空间网格 + 标量 | `(B,8) → M_t (B,1,16,16), u_t (B,)` |
| memory | **帧摘要库** → 跨帧 token | `(B,3,128) → prior_tokens (B,8,128)` |
| refiner | **稀疏残差**（仅 32 个 suspect） | `(B,32,768)` |
| denoiser | **稠密网格**（全部 256 token） | `(B,256,768)` |
| gate | **帧级标量** | `(B,)` |

对象类型互不相同：集合 / 网格+标量 / 跨帧 token / 稀疏残差 / 稠密网格 / 标量。

**这也解释了 refiner 与 denoiser 为什么不是冗余的**：
- refiner 在**稀疏、被 H 路由**的证据上做**一次性**修正，只碰 32 个 token；
- denoiser 在**稠密、被 syndrome+运动+记忆条件化**的场里做**两步迭代**，碰全部 token。
两者是"外科手术"与"全局精修"的分工，而非同一算子的两套参数。

---

## P3 被切断的通道 —— 审计发现的 5 处关键点

| 检查 | 结果 | 意义 |
|---|---|---|
| 诊断**不接收** corruption mask | PASS | 它必须**推断**错误，不能"读标签"。否则 `q` 退化成 mask 的拷贝 |
| refiner **不接收** clean/teacher 特征 | PASS | 否则恢复变平凡（直接抄答案） |
| denoiser **不接收** corruption mask | PASS | 噪声档位由调用方给，不由标签选 |
| refiner 残差**被 q 门控** | PASS | `delta = q_i · dX_i`，低严重度 suspect 几乎不动 |
| **denoiser 写回被 alpha 逐 token 门控** | **本轮修复** | 见下 |

### 本轮修复的缺陷：去噪器写回缺少逐 token 门控

修复前：

```python
eps = eps * (1.0 - alpha)        # 噪声：按 token 严重度加权 ✓
...
x = x + w * pred                  # 写回：全局标量 w，对所有 token 一视同仁 ✗
```

**不对称**：噪声只注入"被判坏"的 token，但修正却对全部 256 个 token 统一施行。这有两个后果：

1. 违背架构图自己的"**选择性恢复**" —— 健康 token 本应由 refiner 的严格 identity bypass 保护，却还被 denoiser 全局改了一次；
2. 削弱互补性 —— refiner 与 denoiser 的分工边界被抹掉（denoiser 在无差别地覆盖 refiner 的保守性）。

修复后 `x = x + w * alpha * pred`，实测：

```
健康 token (alpha≈0.01) 最大改动 : 0.00000989
损坏 token (alpha≈0.99) 最大改动 : 0.00086284
损坏/健康 改动比              : 87.2×   -> 选择性恢复生效
```

---

## P4 无旁路 —— 辅助信号不能绕过主路径

| 检查 | 结果 | 意义 |
|---|---|---|
| head 只吃 `X_final` | PASS | `motion_target` / `target_mask` 只进损失，**永不进 head** |
| clean 特征对诊断目标 detach | PASS | teacher 不会收到学生的诊断梯度 |
| clean 分支 head 保留计算图 | PASS | `L_track^clean` 是正则项，仍训练 adapter |

**为什么"无旁路"是互补性的前提**：如果任何一个模块（例如运动先验）能直接影响 head，主跟踪损失就会优先通过那条捷径回传，其他模块就拿不到有效的任务梯度 —— 它们会停留在"靠辅助损失训练"的状态，彼此之间也就不会形成"分工"。所以"所有信息必须经由 `X_final` 汇聚"是互补性的结构性保障。

---

## P5 实测推论（两条与风险直接相关）

### 1. H 路由**不是** spatial kNN 的伪装

这是风险 #3（"如果 H-routing 不比 spatial kNN 强，就不要硬吹 Tanner routing"）的**前置条件**：如果两者本来就一样，消融必然无差异。

实测 `HᵀH` 邻居（top-8）与空间切比雪夫 8 近邻的重叠：

| 指标 | 值 |
|---|---|
| 平均重叠 | **0.90 / 8** |
| 重叠 ≥6 的 token 占比 | **0.0%** |

→ **两者抓的是完全不同的关系**。`H` 邻居是"共享冗余校验的 token"，由稀疏校验矩阵的几何决定；空间邻居是"网格上挨着的 token"。这条通路确实带来了空间路由没有的信息。

### 2. `K_n = 8` 对每个 token 都可达

我原先担心"平均列度 3 → 共享校验邻居太少，取 8 会补零"。实测**这个估算错了**：

| 指标 | 值 |
|---|---|
| 每 token 共享校验邻居数 | min **27**, max 33, mean **31.5** |
| 邻居数 < 8 的 token | **0 / 256** |

原因：每个 check 横跨 12 个 token，所以 `HᵀH` 相当稠密（不是我以为的稀疏图）。**结论：`K_n=8` 是从 31 个候选中取 top-8，是真正的"选择"，不是补零。** 这也说明架构图的 `K_n=8` 是合理取值。

---

## 结论

**理论层面：数据通路支持互补。** 六个模块读不同的信息源（时间 / token 网格+跨模态 / 跨帧 / 图关系 / 全格条件 / 帧级决策），操作不同的数学对象，且没有一条通道能让某个模块抄到答案或绕过主路径。

**代码层面：本轮发现并修复 1 处真实缺陷**（denoiser 写回缺少逐 token 门控），另有 2 处此前已修（推理时运动先验恒零、TRC 门控零梯度）。

**尚未验证的部分（需要消融，但按你的要求先不做）**：理论互补性说明"它们*可以*互补"，不等于"训练后*确实*学到互补的东西"。后者只能由消融证明。当前能用静态证据支持的最强结论是 P5.1（H 路由与空间路由的重叠仅 0.90/8），它排除了"H 路由是空间 kNN 伪装"这一最可能的退化情形。

---

## 附：16 项检查清单

```
P1  运动是唯一的时间来源                    PASS
P1  诊断在 token 网格上工作（非帧向量）      PASS
P1  记忆由诊断驱动（q -> reliability）       PASS
P1  refiner 的 K/V 由 H 关联度 gather        PASS
P2  六个模块操作六个不同对象                PASS
P3  诊断不接收 corruption mask              PASS
P3  refiner 不接收 clean/teacher 特征        PASS
P3  denoiser 不接收 corruption mask         PASS
P3  refiner 残差被 q 门控                   PASS
P3  denoiser 写回被 alpha 逐 token 门控      PASS   <- 本轮修复
P4  head 只吃 X_final（无旁路）             PASS
P4  clean 特征对诊断目标 detach              PASS
P4  clean 分支 head 保留计算图               PASS
P5  TopK suspect 集合无重复                  PASS
P5  H 路由不是 spatial kNN（重叠 0.90/8）    PASS
P5  K_n=8 对每个 token 可达（min 27）        PASS
```
