# CodeTrack — 环境与 GOLA 代码核实记录

本文档记录**本机环境修复**、**与上游的差异清单**，以及**从官方代码中实际核实到的语义**。
凡是来自代码核实的结论都标注了文件与行号，便于复核。

上游基准：`https://github.com/MelanTech/GOLA`，commit **`339c737`**。

---

## 1. 环境修复（两个非显然的坑）

### 1.1 PyTurboJPEG 找不到 libturbojpeg

**症状**：训练/评测跑到数据加载阶段才崩，报

```
RuntimeError: Unable to locate turbojpeg library automatically.
```

**根因**：系统里已有 `/root/autodl-tmp/lab/tools/libturbojpeg.so.0`（→ `.so.0.2.0`），
但该目录不在 `LD_LIBRARY_PATH` 中，`turbojpeg.py` 的自动探测找不到它。

**修复**：`export LD_LIBRARY_PATH=/root/autodl-tmp/lab/tools`
（已固化在 `scripts/00_env.sh`）

> ⚠️ 注意路径：`.so` 文件在 `tools/` **根目录**，不在 `tools/libjpeg-turbo/`（后者是空目录）。

### 1.2 确定性算法要求 CUBLAS_WORKSPACE_CONFIG

**症状**：eval 阶段抛

```
RuntimeError: Deterministic behavior was enabled ... uses CuBLAS ...
set CUBLAS_WORKSPACE_CONFIG=:4096:8
```

**根因**：框架启用了 `torch.use_deterministic_algorithms(True)`，而 LoRA 前向包含 CuBLAS 矩阵乘。

**修复**：`export CUBLAS_WORKSPACE_CONFIG=:4096:8`（已固化在 `scripts/00_env.sh`）

### 1.3 数据集路径

`consts.yaml` 是**全局加载**的（`trackit/core/runtime/global_constant/__init__.py`），
不经配置树，因此 **mixin 无法覆盖它**，必须直接编辑 `consts.yaml`。

| key | 本机取值 |
|---|---|
| `LasHeR_PATH` | `/root/autodl-tmp/lab/dataset/LasHeR/` |
| `RGBT234_PATH` | `/root/autodl-tmp/lab/dataset/RGBT234/` |

`LasHeR_PATH` 指向的目录名不必叫 `LasHeR0428`：代码只做
`os.path.join(root_path, 'trainingset')` 与 `{root_path}trainingsetList.txt`
（`trackit/datasets/MMOT/datasets/LasHeR.py:28-34`），所以本地 `LasHeR/` 直接可用。

### 1.4 DINOv2 backbone 权重

DINOv2 ViT-B/14 主干**不需要单独下载**：`build_dino_v2_backbone` 会在首次使用时从
`https://dl.fbaipublicfiles.com/dinov2/...` 拉取并缓存到
`~/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth`（本机已缓存，346 MB）。

---

## 2. 与上游的差异清单

```bash
# 忽略运行产物后，上游代码应完全一致
diff -rq --exclude='__pycache__' --exclude='cache' third_party/GOLA/trackit trackit
diff -rq --exclude='smoke_train.yaml'             third_party/GOLA/config  config
```

实测结果：

| 路径 | 状态 |
|---|---|
| `trackit/` | **与上游逐字节一致** |
| `main.py` / `profile_model.py` / `evaluation.py` / `requirements.txt` | **与上游一致** |
| `config/` | 与上游一致，仅新增 `config/GOLA/_mixin/smoke_train.yaml` |
| `consts.yaml` | 上游模板 + 本地路径（机器相关，已 gitignore） |

| 路径 | 类型 | 说明 |
|---|---|---|
| `config/GOLA/_mixin/smoke_train.yaml` | **新增（我们）** | 小样本训练覆盖：batch 4 / 8 samples / 1 epoch |

> 结论：**没有修改任何上游代码**。mixin 是框架原生支持的扩展点
> （解析顺序见 `trackit/core/boot/funcs/mixin/__init__.py`），不构成代码侵入。

---

## 3. 从官方代码核实到的语义（重要，写代码前必读）

这些结论来自实际读码 + 真实前向验证，方案文档中相应条目已据此确认。

### 3.1 输入尺寸

`config/GOLA/dinov2/config.yaml`：

```yaml
common:
  template_size: [ 112, 112 ]
  search_region_size: [ 224, 224 ]
  template_feat_size: [ 8, 8 ]
  search_region_feat_size: [ 16, 16 ]
```

配合 patch size 14：`112/14=8`、`224/14=16`。**确认 template 112 / search 224。**

### 3.2 Token 布局与 `_fuse_search`

`trackit/models/methods/GOLA/gola.py:105-118`：

```python
fusion_feat = torch.cat((z_feat_v, x_feat_v, z_feat_i, x_feat_i, d_feat_v, d_feat_i), dim=1)
...
return feat[:, 2 * z_len + x_len : 2 * (z_len + x_len), :]   # search_i
```

实测前向（B=2）结果：

| 张量 | 形状 |
|---|---|
| `z_v, z_i, d_v, d_i` | `(B, 64, 768)` |
| `x_v, x_i` | `(B, 256, 768)` |
| `F0` | `(B, 768, 768)` |
| `X_t`（TIR-pos，送入 head） | `(B, 256, 768)` |
| `X_aux`（RGB-pos） | `(B, 256, 768)` |
| head 输出 | `score_map (B,16,16)`, `boxes (B,16,16,4)` |

- **确认** `_fuse_search` 取的是 **TIR-position** 的 256 个 token。
- **确认** RGB-position 分支可单独取出，索引 `feat[:, z_len : z_len+x_len, :]`
  —— 这是 CodeTrack 使用 `X_t^{aux}` 的前置能力，已验证可行。

### 3.3 token-type embedding 只有 5 个槽位

`gola.py:46,72,76,87,91,100,102`：

| 槽位 | 用途 |
|---|---|
| `[0:2]` | 初始模板 (visible / infrared) |
| `[2:4]` | 在线模板 (visible / infrared) |
| `[4]` | search（RGB 与 TIR **共用**） |

由 `z_feat_mask` / `d_feat_mask` 选择 `[0:2]` 或 `[2:4]`。

> ⚠️ **search 的模态区分完全靠内容，不靠类型 embedding。**
> CodeTrack 把 RGB search 当作 `X_t^{aux}` 时，必须注意两者类型 embedding 相同。

### 3.4 `z_feat_mask` / `d_feat_mask` 是前景掩码，不是全 1

**这是最容易踩的坑。**
`trackit/runner/.../pipelines/_common/template_foreground_indicating_mask_generation.py`
构造的是 **8×8 二值掩码**（1 = 模板中属于目标的 patch，0 = 背景），
由 `get_foreground_bounding_box` 把模板框折到特征图坐标得到
（`trackit/core/utils/bbox_mask_gen/__init__.py`）。

给全 1 掩码等于告诉模型"整个模板都是目标"——**分类分数看起来正常，但框回归会崩**。

### 3.5 box 是角点格式 (x1,y1,x2,y2)

`trackit/models/methods/GOLA/modules/head/mlp.py:80-84`：

```python
x1y1 = bbox_offset.unsqueeze(0) - lt
x2y2 = bbox_offset.unsqueeze(0) + rb
```

`bbox_offset` 来自 `get_anchor_free_reference_points(map_size, normalized=True)`，
是归一化到 (0,1) 的参考点；`PostProcessing_BoxWithScoreMap` 再乘以
`search_region_size`。所以模型/后处理输出的框是 **search-crop 像素下的角点坐标**。

`apply_siamfc_cropping_to_boxes` 逐元素做 `coord*scale + translation`，同样是**角点语义**。
反向映射必须用 `reverse_siamfc_cropping_params`（scale→1/s, translation→-t/s），
再 `bbox_clip_to_image_boundary_`，这与
`OneStreamTracker_Evaluation_MainPipeline.on_tracked` 完全一致。

> 教训：**不要手写 tracker 的裁剪/反变换/掩码**。官方 eval pipeline 已实现全部细节，
> 用 `scripts/single_seq_eval.sh` 走官方路径，不要另起一套复刻实现。

### 3.6 模板更新规则

`config/GOLA/run.yaml`：`template_update.update_threshold: 0.84`
—— 与方案文档记录的 `score_t > 0.84 ⇒ update` 一致。

### 3.7 训练超参（与方案文档 §10.4 对照）

`config/GOLA/run.yaml`：

| 项 | 官方值 |
|---|---|
| optimizer | AdamW, `lr 1e-4`, `weight_decay 0.1` |
| `embed` 参数 | `weight_decay 0` |
| `max_grad_norm` | 1.0 |
| scheduler | timm cosine, `lr_min 1e-6`, `warmup_epochs 2`, `warmup_lr 1e-7` |
| AMP | 开（float16） |
| torch_compile | 默认开（短跑用 `disable_torch_compile`） |
| epochs | 10 |
| batch | 128；`samples_per_epoch 131072` |

---

## 4. 权重文件

| 文件 | 内容 | 用途 |
|---|---|---|
| `weights/gola_b224.bin` | **GOLA-B**（论文 LasHeR 评测权重） | 主方法底座，后续微调起点 |
| `weights/gola_l224.bin` | GOLA-L（大模型变体） | 暂不需要 |

来源：README 的 Weights Google Drive 目录（`gdown`，需代理）。

**格式**：`weights/*.bin` 是 **safetensors**，不是 `torch.save` 的 pickle。
`torch.load` 会报 `UnpicklingError: unpickling stack underflow`；请用
`safetensors.torch.load_file`。

**内容**：GOLA-B 文件含 **1309 个 key / 12.99M 参数**，只覆盖**可训练部分**
（LoRA A/B/GA/GB、`token_type_embed`、head、`lora_alpha`、`use_rslora`），
**不含冻结的 DINOv2 主干**（那 221 个 key 来自 backbone 预训练缓存）。

实测加载（`strict=False`）：

```
missing    : 221   <- 全部是冻结的 DINOv2 主干参数，预期
unexpected : 0
模型总参数  : 98.71M   （论文报 ~99M）
可训练参数  : 12.99M
```

---

## 5. 验证入口

```bash
source scripts/00_env.sh
bash scripts/preflight.sh          # 环境/权重/数据集/参考快照自检
bash scripts/train_smoke.sh        # 训练循环（小样本）
bash scripts/single_seq_eval.sh    # 官方 eval pipeline，单序列
"$PYTHON" profile_model.py GOLA dinov2 --device cuda   # 前向 + FLOPs
```
