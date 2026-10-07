"""Sum-product decoding of binary error variables with uncertain GF(2) syndromes.

e_i is a harmful-token indicator, not a quantised visual feature. An observed
syndrome bit s_j checks XOR_{i:H_ji=1} e_i. Visual redundancy must supply a
calibrated likelihood for s_j; no claim that arbitrary DINO features are codewords
is necessary. A completely uncertain check has syndrome_logit=0 and sends no
message. This file supplies the exact decoder, not the visual likelihood model.
"""
from __future__ import annotations

import torch


def decode_error_syndrome(error_logits: torch.Tensor, support: torch.Tensor,
                          syndrome_logits: torch.Tensor, iterations: int = 3,
                          damping: float = 0.) -> dict[str, torch.Tensor]:
    """Decode q(e_i=1) from channel and noisy parity evidence.

    Input logits use log P(1)/P(0). Internally messages use the conventional
    LLR log P(0)/P(1). Every check-to-variable message excludes its receiver;
    every variable-to-check message sums OTHER CHECKS at that variable.
    """
    if error_logits.ndim != 2 or support.ndim != 2:
        raise ValueError('expected channel [B,N] and parity support [M,N]')
    b, n = error_logits.shape
    m, variables = support.shape
    if variables != n or syndrome_logits.shape != (b, m):
        raise ValueError('incompatible channel, check and syndrome dimensions')
    if iterations < 1 or not 0. <= damping < 1.:
        raise ValueError('require iterations >= 1 and 0 <= damping < 1')
    edges = support.to(device=error_logits.device, dtype=torch.bool)
    if not bool(edges.any(-1).all()):
        raise ValueError('every parity check needs at least one variable')
    channel_llr = -error_logits.float()
    # E[(-1)^s] attenuates the check message by syndrome certainty. At P(s=1)=.5
    # the factor is uninformative, while positive/negative certainty enforces 0/1.
    syndrome_sign = -torch.tanh(syndrome_logits.float() * .5).unsqueeze(-1)
    mask = edges.unsqueeze(0)
    c2v = channel_llr.new_zeros(b, m, n)
    for _ in range(iterations):
        total = c2v.sum(dim=1, keepdim=True)
        v2c = channel_llr[:, None, :] + total - c2v
        symbols = torch.where(mask, torch.tanh(v2c * .5), torch.ones_like(v2c))
        # Prefix/suffix products exclude a receiver exactly, including at zero.
        # Absent edges contribute one and never affect signs or magnitudes.
        one = torch.ones_like(symbols[..., :1])
        prefix = torch.cat([one, symbols.cumprod(-1)[..., :-1]], dim=-1)
        suffix = torch.cat([symbols.flip(-1).cumprod(-1).flip(-1)[..., 1:], one], dim=-1)
        extrinsic = (syndrome_sign * prefix * suffix).clamp(-1. + 1e-6, 1. - 1e-6)
        proposal = torch.where(mask, 2. * torch.atanh(extrinsic), torch.zeros_like(extrinsic))
        c2v = damping * c2v + (1. - damping) * proposal
    posterior_logits = -(channel_llr + c2v.sum(dim=1))
    return {'q': posterior_logits.sigmoid(), 'q_logits': posterior_logits,
            'check_to_variable_llr': c2v, 'variable_to_check_llr': v2c}
