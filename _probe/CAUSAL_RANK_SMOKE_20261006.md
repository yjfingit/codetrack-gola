# Causal q ranking smoke, 2026-10-06

## Setup

- LasHeR-train curated 10 sequences, one seed, GPU 4.
- GOLA-DINOv2-B baseline checkpoint; 20 optimizer updates; two-round BP/SATR.
- No full benchmark evaluation because recovery showed no measurable feature gain.

## Results

1. Initial causal-rank run (`_probe/causal_s1_rank20`): causal AUROC stayed near 0.50, q standard deviation was about 0.003, and feature recovery gain was effectively zero.
2. Unknown-frame masking run (`_probe/causal_s1_validrank20`): causal AUROC peaked around 0.63 by epoch, q standard deviation stayed near 0.0024, and `d_input` / `d_after` remained equal to displayed precision. Syndrome did not decrease.
3. A synthetic criterion check confirmed that valid rows produce the expected ranking gradients and invalid rows receive exactly zero gradient.

## Interpretation

Masking unpaired frames removes false causal labels, but the short run still does not produce useful q separation or correction. The training route is not ready for 10-sequence tracking evaluation or full training. The next redesign should stop asking the current BP output to discover tracking utility from sparse token replacement labels alone; first verify the visual check residual and candidate correction direction against target-region tracking loss.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=4 \
LD_LIBRARY_PATH=/home/yangjuanfeng/lab/tools/libjpeg-turbo/root/usr/lib/x86_64-linux-gnu \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
TRACKIT_CONSTS_PATH=_probe/causal_train10/consts.yaml \
/home/yangjuanfeng/lab/envs/gola/bin/python main.py GOLA codetrack_s1 \
  --mixin_config causal_s1_unlocked20 \
  --distributed_nproc_per_node 1 --disable_wandb \
  --weight_path weights/gola_b224.bin \
  --output_dir=_probe/causal_s1_validrank20
```
