# Source this before running any GOLA/CodeTrack entry point on this box.
#
#   source scripts/00_env.sh
#
# Fixes encoded here:
#   * PyTurboJPEG cannot auto-locate libturbojpeg  -> LD_LIBRARY_PATH
#   * framework enables deterministic algorithms    -> CUBLAS_WORKSPACE_CONFIG
#   * torch.compile is not wanted for short runs    -> handled via mixin flag
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# PyTurboJPEG imports successfully even when the native library is missing, then fails only
# when the first data worker constructs TurboJPEG().  Point at the directory that actually
# contains libturbojpeg.so.0 on this host; TURBOJPEG_DIR remains an explicit override.
export LD_LIBRARY_PATH="${TURBOJPEG_DIR:-/home/yangjuanfeng/lab/tools/libjpeg-turbo/root/usr/lib/x86_64-linux-gnu}:${LD_LIBRARY_PATH:-}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export NCCL_P2P_DISABLE=1

PYTHON="${PYTHON:-/home/yangjuanfeng/lab/envs/gola/bin/python}"
GOLA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHT="${WEIGHT:-$GOLA_ROOT/weights/gola_b224.bin}"
export PYTHON GOLA_ROOT WEIGHT
