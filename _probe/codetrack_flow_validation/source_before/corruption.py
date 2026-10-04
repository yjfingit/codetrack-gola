"""Training-time corruption simulator.

The architecture figure's input panel takes **6-channel RGB-T crops**.  Corruption is
therefore injected at two points and both are required by the CodeTrack protocol:

* **image level** -- applied by the data collator to the already-normalised crop tensors,
  i.e. before the GOLA backbone.  Types: low-light (gamma), over-exposure, Gaussian noise,
  blur, occlusion.
* **patch-token level** -- applied inside the model to the patch embeddings, because the
  figure's ``X_t`` is a *token* grid and "block erase" is only meaningful there.  Types:
  random erase, burst erase, feature noise, partial modality drop.

Both paths return a **(B, N) boolean corruption mask**, which is what the diagnosis head
is supervised against (``L_diag``).  The mask is *never* the only target: the plan's
clean-teacher residual ``e* = alpha e_feat + (1-alpha) e_task`` is combined with it, so the
head cannot get away with merely re-detecting the injected pattern.

Because it lives here rather than in the dataset, the same corruption is used by training
and by short-sequence evaluation, and the images stay on disk untouched.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

# GOLA normalisation ('mm'): first 3 channels RGB (ImageNet), last 3 infrared.
RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)
TIR_MEAN = (0.449, 0.449, 0.449)
TIR_STD = (0.226, 0.226, 0.226)

IMAGE_KINDS = ("low_light", "over_exposure", "gaussian_noise", "blur", "occlusion")
TOKEN_KINDS = ("block_erase", "burst_erase", "feat_noise", "modality_drop")


def _chan_stats(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(list(RGB_MEAN) + list(TIR_MEAN), device=x.device, dtype=x.dtype)
    std = torch.tensor(list(RGB_STD) + list(TIR_STD), device=x.device, dtype=x.dtype)
    return mean.view(1, -1, 1, 1), std.view(1, -1, 1, 1)


def corrupt_image(x: torch.Tensor, kind: str, severity: float,
                  modality: str = "both", generator: Optional[torch.Generator] = None
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Corrupt a normalised 6-channel crop.  Returns ``(x_corrupted, channel_mask)``.

    ``channel_mask`` is (B, 6) with 1 where the modality was attacked, so "partial
    modality drop" can be told apart from "everything is noisy".
    """
    if severity <= 0:
        return x, torch.zeros(x.shape[0], 6, device=x.device, dtype=x.dtype)
    b, c, h, w = x.shape
    mean, std = _chan_stats(x)
    img = x * std + mean                       # back to ~[0,1]
    cmask = torch.zeros(b, c, device=x.device, dtype=x.dtype)
    ch_rgb = [0, 1, 2]
    ch_tir = [3, 4, 5]
    if modality in ("rgb", "both"):
        cmask[:, ch_rgb] = 1.0
    if modality in ("tir", "both"):
        cmask[:, ch_tir] = 1.0

    if kind == "low_light":
        gamma = 1.0 + 2.0 * float(severity)
        img = img.clamp(min=0) ** gamma
    elif kind == "over_exposure":
        gain = 1.0 + 2.0 * float(severity)
        img = img * gain
    elif kind == "gaussian_noise":
        # ``torch.randn_like`` does not accept a ``generator`` argument (only the
        # explicit-shape factories do), so build the noise with ``torch.randn``.  This bug
        # survived because image-level corruption was never actually invoked.
        noise = torch.randn(img.shape, generator=generator,
                            device=img.device, dtype=img.dtype)
        img = img + noise * (0.15 * float(severity))
    elif kind == "blur":
        k = 3 + 2 * int(round(3 * float(severity)))
        k = k if k % 2 == 1 else k + 1
        img = F.avg_pool2d(img, kernel_size=k, stride=1, padding=k // 2)
    elif kind == "occlusion":
        side_h = max(1, int(round(float(severity) * h)))
        side_w = max(1, int(round(float(severity) * w)))
        for i in range(b):
            top = int(torch.randint(0, max(1, h - side_h + 1), (1,), generator=generator))
            left = int(torch.randint(0, max(1, w - side_w + 1), (1,), generator=generator))
            img[i, :, top:top + side_h, left:left + side_w] = 0.0
    else:
        raise ValueError(f"unknown image corruption {kind!r}")

    if modality == "rgb":
        img[:, ch_tir] = (x * std + mean)[:, ch_tir]
    elif modality == "tir":
        img[:, ch_rgb] = (x * std + mean)[:, ch_rgb]

    return (img - mean) / std, cmask


def corrupt_tokens(tokens: torch.Tensor, kind: str, ratio: float, severity: float,
                   generator: Optional[torch.Generator] = None
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Corrupt patch embeddings ``(B, N, C)``.  Returns ``(tokens, mask)`` with mask (B, N)."""
    b, n, c = tokens.shape
    out = tokens.clone()
    mask = torch.zeros(b, n, device=tokens.device, dtype=torch.bool)
    k = max(1, int(round(ratio * n)))
    grid = int(round(n ** 0.5))

    for i in range(b):
        if kind == "block_erase":
            # a contiguous square block of erased patches (a spatial occlusion in token space)
            side = max(1, int(round(k ** 0.5)))
            if grid * grid == n:
                top = int(torch.randint(0, max(1, grid - side + 1), (1,), generator=generator))
                left = int(torch.randint(0, max(1, grid - side + 1), (1,), generator=generator))
                idx = [(top + dy) * grid + (left + dx)
                       for dy in range(side) for dx in range(side)
                       if top + dy < grid and left + dx < grid]
                idx = torch.tensor(idx[:k], device=tokens.device, dtype=torch.long)
            else:
                idx = torch.randperm(n, generator=generator)[:k].to(tokens.device)
            out[i, idx] = 0.0
            mask[i, idx] = True
        elif kind == "burst_erase":
            start = int(torch.randint(0, max(1, n - k), (1,), generator=generator))
            idx = torch.arange(start, min(start + k, n), device=tokens.device)
            out[i, idx] = 0.0
            mask[i, idx] = True
        elif kind == "feat_noise":
            idx = torch.randperm(n, generator=generator)[:k].to(tokens.device)
            noise = torch.randn(len(idx), c, device=tokens.device, dtype=tokens.dtype,
                                generator=generator)
            out[i, idx] = out[i, idx] + noise * float(severity) * out[i].std().clamp(min=1e-3)
            mask[i, idx] = True
        elif kind == "modality_drop":
            idx = torch.randperm(n, generator=generator)[: max(1, n // 2)].to(tokens.device)
            out[i, idx] = 0.0
            mask[i, idx] = True
        else:
            raise ValueError(f"unknown token corruption {kind!r}")
    return out, mask


class CorruptionSchedule:
    """Draws corruption kinds/severities for a batch, and keeps the per-sample choice."""

    def __init__(self, image_prob: float = 0.15, token_ratio: float = 0.4,
                 severity: float = 0.4, enabled: bool = True, seed: int = 0,
                 token_prob: Optional[float] = None):
        self.image_prob = float(image_prob)
        # Explicit token-level attack rate.  It used to be implied as
        # ``(1 - image_prob) * 0.6``, which coupled the two and made the clean fraction of a
        # batch impossible to set directly.  ``None`` keeps the old behaviour for callers that
        # pass nothing.
        self.token_prob = (float(token_prob) if token_prob is not None
                           else (1.0 - self.image_prob) * 0.6)
        self.token_ratio = float(token_ratio)
        self.severity = float(severity)
        self.enabled = bool(enabled)
        self.generator = torch.Generator().manual_seed(int(seed))

    def draw_image(self, batch: int, device: torch.device
                   ) -> Tuple[Optional[str], str, float, torch.Tensor]:
        """Returns (kind or None, modality, severity, apply_mask(B,) bool)."""
        if not self.enabled:
            return None, "both", 0.0, torch.zeros(batch, device=device, dtype=torch.bool)
        apply = torch.rand(batch, generator=self.generator) < self.image_prob
        kind = IMAGE_KINDS[int(torch.randint(len(IMAGE_KINDS), (1,), generator=self.generator))]
        sev = self.severity * float(0.5 + torch.rand(1, generator=self.generator))
        return kind, "both", sev, apply

    def draw_token(self, batch: int, device: torch.device
                   ) -> Tuple[Optional[str], float, torch.Tensor]:
        if not self.enabled:
            return None, 0.0, torch.zeros(batch, device=device, dtype=torch.bool)
        apply = torch.rand(batch, generator=self.generator) < self.token_prob
        kind = TOKEN_KINDS[int(torch.randint(len(TOKEN_KINDS), (1,), generator=self.generator))]
        return kind, self.severity, apply
