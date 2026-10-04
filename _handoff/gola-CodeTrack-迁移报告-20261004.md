# gola-CodeTrack 迁移交接报告（4090D → 175机）

日期：2026-10-04
执行：阿岩（WorkBuddy）
源：`RGBT-SOT-4090D:/root/autodl-tmp/migration/`
目标：`4090server175:/home/yangjuanfeng/lab/projects/`

---

## 1. 结果一句话

`gola-CodeTrack` 主仓库已完整落在 **175机 `/home/yangjuanfeng/lab/projects/gola-CodeTrack`**，
git HEAD = **`26c63e2`**、806 个实文件 + 15 个软链（已重指 175 数据集）、零字节丢失。
`CodeTrack.tar`（早期仓库，主人选择不展开）保留为压缩包备查。

---

## 2. 传输明细

| 文件 | 字节 | 方式 | 结果 |
|---|---:|---|---|
| `gola-CodeTrack.tar` | 706,140,160 | 175 直拉（单流 scp） | ✅ 61 s，≈11.5 MB/s |
| `CodeTrack.tar` | 383,682,560 | 175 直拉（单流 scp） | ✅ ≈10 min，≈0.66 MB/s（限速） |
| `HANDOFF.md` | 19,530 | scp | ✅ |
| `env_facts.txt` | 137 | scp | ✅ |
| `pip_freeze_gola_env.txt` | 2,193 | scp | ✅ |
| `CodeTrack_outputs_weights_inventory.txt` | 5,798 | scp | ✅ |

**路径选择**：最初走「4090D → 本机 PC → 175」中转，实测仅 2.6 MB/s；
改用 **175 上用 `~/.ssh/id_ed25519_autodl_push` 免密直拉**，首文件提速到 11.5 MB/s。
⇒ 以后 4090D→175 一律走直推/直拉，不经 PC。

---

## 3. ⚠️ 踩的坑（重要，以后必看）

**两个 tar 都没有顶层目录前缀**（内容以 `./` 开头），
所以直接 `tar -xf -C projects/` 会把两个仓库**平铺到同一层并互相覆盖**。

冲突顶层条目共 11 个：
`codetrack/` `docs/` `outputs/` `tools/` `.git/` `scripts/` `data/` `assets/` `README.md` `requirements.txt` `.gitignore`

**首次解压已发生污染**，已按主人指示**全部删除重做**：

```bash
# 正确做法：各自解到独立子目录
mkdir -p projects/gola-CodeTrack
tar -xf gola-CodeTrack.tar -C projects/gola-CodeTrack
```

---

## 4. 最终目录结构

```
/home/yangjuanfeng/lab/projects/
├── gola-CodeTrack/                    ← 主仓库（18 顶层条目 / 806 文件 / 679 MB）
│   ├── _handoff/HANDOFF.md            ← 交接文档
│   ├── codetrack/  config/  configs/  trackit/  tools/  scripts/
│   ├── docs/         （含 22 篇论文 PDF 于 docs/paper/）
│   ├── outputs/      （8 项，759 个 log/json/md/txt/csv 文本产物）
│   ├── weights/      （gola_b224.bin 52 MB / gola_l224.bin 130 MB）
│   ├── data/         （LasHeR_train10 / LasHeR_single / LasHeR_train_single，
│   │                  15 个软链已重指 175 数据集）
│   ├── consts.yaml.bak-20261004       ← 修改前备份
│   └── scripts/00_env.sh.bak-20261004 ← 修改前备份
└── _migration_incoming/                ← 暂存（tar 原件，校验后可删）
    ├── gola-CodeTrack.tar   (706 MB)
    └── CodeTrack.tar.kept   (384 MB)
```

---

## 5. 已做的适配修改（3 处，均留备份）

| 文件 | 改动 | 原因 |
|---|---|---|
| `consts.yaml` | `LasHeR_PATH` → `/home/yangjuanfeng/lab/dataset/LasHeR/`；`RGBT234_PATH` → `.../RGBT234/` | 原为 4090D 绝对路径 |
| `scripts/00_env.sh` | `PYTHON` 默认 → `/home/yangjuanfeng/lab/envs/gola/bin/python` | 原为 4090D venv |
| `scripts/00_env.sh` | `LD_LIBRARY_PATH` → `${TURBOJPEG_DIR:-/home/yangjuanfeng/lab/tools}` | 改为可用环境变量覆盖 |

`data/LasHeR_*` 下 15 个软链接全部重建并验证解析成功（原来指向 `/root/autodl-tmp/...`）。

---

## 6. 验收证据

| 检查 | 结果 |
|---|---|
| git HEAD | `26c63e2` ✅ 与 HANDOFF.md 声明一致 |
| git branch | `main` |
| git status | 仅 ` D repo/README.md` —— §11 已声明排除的第三方检出，**预期** |
| 文件数 | 实文件 806（tar 内 821，差 15 = 软链，`find -type f` 不计） |
| 字节数 | 701,806,705（不含 .git 与软链） |
| 论文 PDF | 22 篇 ✅ 与 HANDOFF §11 一致 |
| 权重 | `gola_b224.bin` + `gola_l224.bin` ✅ |
| 差异文件对账 | `comm` 比对：**"在 tar 不在盘" 仅 15 个软链；"在盘不在 tar" 0 个** |

---

## 7. 环境现况（175机）

| 项 | 状态 |
|---|---|
| `/home/yangjuanfeng/lab/envs/gola/bin/python` | ✅ 存在，**torch 2.5.1+cu124，CUDA 可用** —— 与 4090D 完全一致 |
| `/home/yangjuanfeng/lab/dataset/LasHeR` | ✅ 存在且结构匹配（annos/AttriSeqsTxt/trainingset/testingset） |
| **`libturbojpeg.so*`** | ❌ **175 上未找到** —— 唯一待补依赖 |

---

## 8. 下一步（按优先级）

1. **补 libturbojpeg**：从 4090D `/root/autodl-tmp/lab/tools/` 拷 `libturbojpeg.so*` 到 175，
   或 `apt install libturbojpeg0`；然后 `export TURBOJPEG_DIR=<目录>` 再 `source scripts/00_env.sh`。
2. **跑静态综合门**：`source scripts/00_env.sh && "$PYTHON" tools/preflight_acceptance.py`（期望 101 项全绿）。
3. **按 HANDOFF §4.2 跑判定探针（含 norm-only baseline）** —— 这是主人交代的「接手后第一件事，不要先改代码」。
4. 删暂存：`rm -rf /home/yangjuanfeng/lab/projects/_migration_incoming`
   （保留可回溯，建议先确认无误）。
