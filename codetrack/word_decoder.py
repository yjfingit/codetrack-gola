"""Native observation-word reliability and error-syndrome correction.

Receiver and reference symbols are ordinary frozen-backbone outputs. The decoder
learns whether the offered word is useful and which receiver symbols it can fix.
No learned free residual, denoiser, current GT or corruption mask is an input.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from .ecc import build_sparse_support
from .syndrome_bp import decode_error_syndrome


class NativeWordDecoder(nn.Module):
    def __init__(self, dim=768, mid=48, hidden=128, grid=16, rounds=3):
        super().__init__()
        self.grid, self.rounds = grid, rounds
        n = grid * grid
        support = build_sparse_support(n, n, 4, 4, grid, layout='binary_cycles').bool()
        self.register_buffer('support', support)
        self.register_buffer('check_indices', support.nonzero()[:, 1].reshape(n, 4))
        self.project = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, mid))
        self.identity_project = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, mid))
        self.symbol = nn.Sequential(nn.Linear(5 * mid + 2, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.parity = nn.Sequential(nn.Linear(4 * 2 * mid, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.word_quality = nn.Sequential(nn.Linear(4 * mid + 12 + mid + 5, hidden),
                                          nn.GELU(), nn.Linear(hidden, 1))
        self.channel_prior = nn.Parameter(torch.tensor([math.log(.05 / .95)]))
        nn.init.zeros_(self.symbol[-1].weight); nn.init.zeros_(self.symbol[-1].bias)
        nn.init.zeros_(self.parity[-1].weight); nn.init.zeros_(self.parity[-1].bias)
        nn.init.zeros_(self.word_quality[-1].weight)
        nn.init.constant_(self.word_quality[-1].bias, math.log(.05 / .95))

    def forward(self, current_ir, reference_ir, current_rgb, identity, motion, word_statistics):
        b, n, _ = current_ir.shape
        if n != self.grid * self.grid or word_statistics.shape != (b, 12):
            raise ValueError('native word inputs have incompatible dimensions')
        u, r, rgb = self.project(current_ir), self.project(reference_ir), self.project(current_rgb)
        ctx = self.identity_project(identity)
        diff = u - r
        yy, xx = torch.meshgrid(
            (torch.arange(self.grid, device=u.device, dtype=u.dtype) + .5) / self.grid,
            (torch.arange(self.grid, device=u.device, dtype=u.dtype) + .5) / self.grid, indexing='ij')
        coords = torch.stack((xx, yy), -1).reshape(1, n, 2)
        distance = ((coords - motion[:, None, :2]) / motion[:, None, 2:4].abs().clamp_min(.02)).square().sum(-1)
        log_prior = (-.5 * distance).clamp_min(-20.)
        unc = motion[:, 4:5].expand(-1, n)
        channel_logits = self.symbol(torch.cat([
            u, r, diff, (u - rgb).abs(), ctx[:, None].expand(-1, n, -1),
            log_prior[..., None], unc[..., None]], -1)).squeeze(-1) + self.channel_prior
        evidence = torch.cat([diff, diff.abs()], -1)
        syndrome_logits = self.parity(evidence[:, self.check_indices].flatten(2)).squeeze(-1)
        decoded = decode_error_syndrome(channel_logits, self.support, syndrome_logits, self.rounds)
        pooled = torch.cat([diff.mean(1), diff.abs().mean(1), diff.amax(1), diff.amin(1),
                            word_statistics, ctx, motion], -1)
        quality_logits = self.word_quality(pooled).squeeze(-1)
        quality = quality_logits.sigmoid()
        probability = decoded['q'] * quality[:, None]
        # Identical hard forward rule in training and deployment. The straight-
        # through backward path teaches tracking/preservation without an oracle route.
        # Chain rule: a useful reference and a harmful receiver symbol must both
        # hold. Two probabilities just above .5 are still jointly uncertain.
        accept = probability >= .5
        reconstructed = torch.where(accept[..., None], reference_ir, current_ir)
        if self.training:
            reconstructed = reconstructed + (probability - probability.detach())[..., None] * (reference_ir - current_ir)
        return {**decoded, 'channel_logits': channel_logits, 'syndrome_logits': syndrome_logits,
                'word_quality_logits': quality_logits, 'word_quality': quality,
                'accept': accept, 'reconstructed': reconstructed}
