#!/usr/bin/env bash
# Preflight: verify the environment can actually run GOLA before launching a long job.
#
#   bash scripts/preflight.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck source=/dev/null
source scripts/00_env.sh

fail=0
ok()   { printf '  [ ok ] %s\n' "$1"; }
bad()  { printf '  [FAIL] %s\n' "$1"; fail=1; }
warn() { printf '  [warn] %s\n' "$1"; }

echo "== preflight"

# 1. python + torch
if "$PYTHON" -c 'import torch' 2>/dev/null; then
  ok "torch $("$PYTHON" -c 'import torch;print(torch.__version__)') cuda=$("$PYTHON" -c 'import torch;print(torch.cuda.is_available())')"
else
  bad "python/torch unavailable at $PYTHON"
fi

# Import every runtime dependency named by requirements.txt plus the CUDA companion packages
# used by the data/model path.  `pip check` catches incompatible installed versions.
if "$PYTHON" - <<'PY' >/dev/null 2>&1
import cv2, exifread, fvcore, k_means_constrained, matplotlib, numpy, PIL
import prettytable, psutil, rgbt, safetensors, scipy, tabulate, timm, torch
import torchvision, tqdm, turbojpeg, wandb, yaml, zmq
PY
then
  ok "Python runtime dependencies import successfully"
else
  bad "one or more Python runtime dependencies cannot be imported"
fi
if "$PYTHON" -m pip check >/dev/null 2>&1; then
  ok "pip dependency graph has no broken requirements"
else
  bad "pip check reports incompatible or missing requirements"
fi

# 2. PyTurboJPEG (silent blocker: dataset loading raises only mid-run)
if "$PYTHON" -c 'from turbojpeg import TurboJPEG; TurboJPEG()' 2>/dev/null; then
  ok "PyTurboJPEG can locate libturbojpeg"
else
  bad "PyTurboJPEG cannot load libturbojpeg (LD_LIBRARY_PATH=$LD_LIBRARY_PATH)"
fi

# 3. checkpoint
if [[ -f "$WEIGHT" ]]; then
  ok "checkpoint $WEIGHT ($(du -h "$WEIGHT" | cut -f1))"
else
  bad "checkpoint missing: $WEIGHT"
fi
if "$PYTHON" - "$WEIGHT" <<'PY' >/dev/null 2>&1
import sys, torch
from safetensors.torch import load_file
state = load_file(sys.argv[1])
assert state and all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point())
PY
then
  ok "checkpoint is readable safetensors and all floating tensors are finite"
else
  bad "checkpoint cannot be read as finite safetensors: $WEIGHT"
fi

# 4. DINOv2 backbone cache
BB="$HOME/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth"
if [[ -f "$BB" ]]; then
  ok "DINOv2 ViT-B/14 backbone cached"
else
  bad "DINOv2 backbone not cached: $BB (will attempt download)"
fi

# 5. datasets declared in consts.yaml
for key in LasHeR_PATH RGBT234_PATH; do
  p="$("$PYTHON" - "$key" <<'PY'
import sys, yaml
key = sys.argv[1]
print((yaml.safe_load(open('consts.yaml')) or {}).get(key, '') or '')
PY
)"
  if [[ -n "$p" && -d "$p" ]]; then ok "consts.$key -> $p"; else bad "consts.$key -> '${p:-<unset>}' (not a directory)"; fi
done

# 6. working tree vs read-only upstream
if [[ -d third_party/GOLA/trackit ]]; then
  ok "third_party/GOLA reference @ $(git -C third_party/GOLA rev-parse --short HEAD)"
else
  warn "third_party/GOLA reference checkout missing (optional; runtime uses the root working tree)"
fi

echo
if [[ $fail == 0 ]]; then echo "== preflight PASSED"; else echo "== preflight FAILED"; fi
exit $fail
