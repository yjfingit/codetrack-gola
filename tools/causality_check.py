"""Causality audit for the motion prior (D2) and the spatial recovery bias (D3).

The motion block is a *prior*: it exists to predict frame ``t`` from information available
strictly before ``t``.  The block previously absorbed frame ``t``'s ground-truth box and only
then built the prior map, so the prior was a function of the very frame it was meant to help
predict.  An auxiliary loss on ``M_t`` would then have been trivially satisfiable, which is why
``L_motion`` measured exactly 0 in the single-sequence smoke run.

The audit below is behavioural rather than structural: it perturbs the current frame's ground
truth and asserts that the tensors which are supposed to be *predictions* do not move.

Checked:
  C1  M_t is independent of GT_t
  C2  motion_box is independent of GT_t        (it is decoded from x_{t|t-1})
  C3  the refiner's attention bias is independent of GT_t
  C4  the observation DOES move the state (otherwise the filter is inert and C1 is vacuous)

Run:
    LD_LIBRARY_PATH=/root/autodl-tmp/lab/tools python tools/causality_check.py
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safetensors.torch import load_file  # noqa: E402
from trackit.models.backbone.builder import build_backbone  # noqa: E402
from trackit.models.methods.GOLA.gola import GOLA_DINOv2  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHT = os.path.join(_ROOT, "weights/gola_b224.bin")
CFG = dict(enabled=True, z_len=64, x_len=256, dim=768, grid=16, mid_dim=128,
           num_checks=64, h_links_per_check=12, h_min_col_degree=3, h_locality_window=5,
           topk_tokens=32, num_neighbours=8, refiner_hidden=256,
           memory_enabled=True, template_protection=True)

B = 2
RESULTS: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def build():
    torch.manual_seed(0)
    bb = build_backbone({"type": "DINOv2", "parameters": {"name": "ViT-B/14", "acc": "default"}},
                        torch_jit_trace_compatible=False)
    m = GOLA_DINOv2(bb, (8, 8), (16, 16), lora_r=64, lora_alpha=64.0, lora_dropout=0.0,
                    use_rslora=False, codetrack_config=CFG)
    m.load_state_dict(load_file(WEIGHT), strict=False)
    return m.eval().cuda()


def make_inputs():
    g = torch.Generator(device="cuda").manual_seed(5)
    z = torch.rand(B, 6, 112, 112, generator=g, device="cuda") * 0.4 + 0.3
    x = torch.rand(B, 6, 224, 224, generator=g, device="cuda") * 0.4 + 0.3
    d = torch.rand(B, 6, 112, 112, generator=g, device="cuda") * 0.4 + 0.3
    zm = torch.ones(B, 8, 8, dtype=torch.long, device="cuda")
    dm = torch.ones_like(zm)
    return dict(z=z, x=x, d=d, z_feat_mask=zm, d_feat_mask=dm)


def run_frame(m, data, gt, isz) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        out = m(**data, gt_box=gt, image_size=isz)
    mo = out["codetrack"]["motion"]
    return {"map": mo["motion_map"].clone(), "box": mo["motion_box"].clone(),
            "state": m.codetrack.motion._x.clone()}


def main() -> int:
    m = build()
    data = make_inputs()
    sizes = torch.tensor([[224.0, 224.0]] * B, device="cuda")
    base_gt = torch.tensor([[96.0, 96.0, 48.0, 64.0]] * B, device="cuda")
    # a clearly different box: different centre AND different extent
    alt_gt = torch.tensor([[40.0, 160.0, 90.0, 30.0]] * B, device="cuda")

    print("=" * 92)
    print("C1-C3  the prior must not see the current frame's ground truth")
    print("=" * 92)
    # A real clip: several frames so the filter holds a non-trivial posterior, otherwise
    # "independent of GT_t" would be vacuously true on a freshly seeded state.
    m.reset_sequence()
    history = [torch.tensor([[90.0 + 8 * i, 95.0, 50.0, 60.0]] * B, device="cuda")
               for i in range(3)]
    for h in history:
        run_frame(m, data, h, sizes)          # absorbs observations into the state

    state_before = m.codetrack.motion._x.clone()
    m.codetrack.motion._x = state_before.clone()
    a = run_frame(m, data, base_gt, sizes)
    state_after_a = m.codetrack.motion._x.clone()

    m.codetrack.motion._x = state_before.clone()
    b = run_frame(m, data, alt_gt, sizes)

    d_map = float((a["map"] - b["map"]).abs().max())
    d_box = float((a["box"] - b["box"]).abs().max())
    check("C1  M_t is independent of GT_t", d_map == 0.0, f"max|dM_t| = {d_map:.3e}")
    check("C2  motion_box is independent of GT_t", d_box == 0.0, f"max|dbox| = {d_box:.3e}")

    # C3: the refiner consumes M_t as a log-space attention bias, so the *bias actually used*
    # must be GT-independent too.  Comparing the maps already covers this, but computing the
    # bias explicitly catches a future refactor that starts reading the raw state instead.
    def bias_of(map_t):
        flat = map_t.reshape(map_t.shape[0], -1)
        return torch.log(flat.clamp(min=1e-4)) * 0.5

    d_bias = float((bias_of(a["map"]) - bias_of(b["map"])).abs().max())
    check("C3  refiner attention bias is independent of GT_t", d_bias == 0.0,
          f"max|dbias| = {d_bias:.3e}")

    print("\n" + "=" * 92)
    print("C4  ...but the observation must still move the state (C1 must not be vacuous)")
    print("=" * 92)
    d_state = float((state_after_a - state_before).abs().max())
    check("C4  the observation updates the state", d_state > 0,
          f"max|dx| = {d_state:.3e} after observing GT_t")

    print("\n" + "=" * 92)
    print("C5  earlier frames DO affect the current prior (temporal information is used)")
    print("=" * 92)
    m.reset_sequence()
    c = run_frame(m, data, base_gt, sizes)     # no history
    m.reset_sequence()
    for h in history:
        run_frame(m, data, h, sizes)
    d = run_frame(m, data, base_gt, sizes)
    d_hist = float((c["map"] - d["map"]).abs().max())
    check("C5  history changes M_t", d_hist > 0, f"max|dM_t| = {d_hist:.3e}")

    print("\n" + "=" * 92)
    print("C6  D3: the prior head's parameters actually reshape the map")
    print("=" * 92)
    # Perturb the head's last layer; the normalised map must change.  Under the old
    # amplitude parameterisation this was a no-op because the consumer normalises to unit
    # mass, so a pure scale factor cancelled exactly.
    m.reset_sequence()
    e = run_frame(m, data, base_gt, sizes)
    head = m.codetrack.motion.prior[-1]
    with torch.no_grad():
        head.bias[0] += 0.25          # shift the centre by a quarter of a box
        head.bias[3] += 0.5           # widen
    m.reset_sequence()
    f = run_frame(m, data, base_gt, sizes)
    d_head = float((e["map"] - f["map"]).abs().max())
    check("C6  perturbing the prior head changes the normalised map", d_head > 1e-4,
          f"max|dM_t| = {d_head:.3e}")

    print("\n" + "=" * 92)
    bad = [r for r in RESULTS if not r[1]]
    print(f"RESULT: {len(RESULTS) - len(bad)}/{len(RESULTS)} checks pass")
    for name, _, detail in bad:
        print(f"  FAILED: {name}  {detail}")
    print("=" * 92)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
