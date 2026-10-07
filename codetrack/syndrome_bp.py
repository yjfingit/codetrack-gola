"""Sum-product decoding of binary error variables with uncertain GF(2) syndromes.

e_i is a harmful-token indicator, not a quantised visual feature. An observed
syndrome bit s_j checks XOR_{i:H_ji=1} e_i. Visual redundancy must supply a
calibrated likelihood for s_j; no claim that arbitrary DINO features are codewords
is necessary. A completely uncertain check has syndrome_logit=0 and sends no
message. This file supplies the exact decoder, not the visual likelihood model.
"""
from __future__ import annotations

import torch
from torch import nn


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


class VisualSyndromeDiagnosis(nn.Module):
    """Learn distinct channel and parity likelihoods for harmful-token errors.

    The channel sees received TIR, target identity and causal motion. A local check
    sees ordered RGB/TIR discrepancies at its four variables. Supervision must
    include the XOR of causal error labels; q loss alone cannot identify a syndrome.
    The overlapping visual likelihoods form an approximate factor model, whose
    calibration and held-out gain over the unary channel must be measured.
    """
    uses_motion_evidence = True

    def __init__(self, dim=768, mid_dim=128, num_checks=256, num_variables=256,
                 syndrome_hidden=128, detection_prior=.05, bp_iterations=3,
                 bp_damping=.0, **_):
        super().__init__()
        self.bp_iterations, self.bp_damping = bp_iterations, bp_damping
        self.W_x = nn.Linear(dim, mid_dim)
        self.W_r = nn.Linear(dim, mid_dim)
        self.template_ctx = nn.Linear(dim, mid_dim)
        self.channel = nn.Sequential(nn.Linear(2 * mid_dim + 2, syndrome_hidden),
                                     nn.GELU(), nn.Linear(syndrome_hidden, 1))
        self.parity = nn.Sequential(nn.Linear(4 * 2 * mid_dim, syndrome_hidden),
                                    nn.GELU(), nn.Linear(syndrome_hidden, 1))
        self.vote_bias = nn.Parameter(torch.tensor([float(torch.logit(torch.tensor(detection_prior)))]))
        nn.init.zeros_(self.channel[-1].weight); nn.init.zeros_(self.channel[-1].bias)
        nn.init.zeros_(self.parity[-1].weight); nn.init.zeros_(self.parity[-1].bias)

    def forward(self, X_t, X_aux, H_bar, template_context=None,
                motion_map=None, uncertainty=None, return_checks=False):
        b, n, _ = X_t.shape
        support = H_bar > 0
        if not bool((support.sum(-1) == 4).all()):
            raise ValueError('visual syndrome likelihood requires four-variable checks')
        idx = support.nonzero(as_tuple=False)[:, 1].reshape(support.shape[0], 4)
        U, R = self.W_x(X_t), self.W_r(X_aux)
        template = self.template_ctx(template_context)[:, None, :] if template_context is not None else torch.zeros_like(U[:, :1])
        motion = U.new_zeros(b, n, 1)
        if motion_map is not None:
            logp = motion_map.reshape(b, n).clamp_min(1e-8).log()
            motion = (logp - logp.mean(-1, keepdim=True)).unsqueeze(-1)
        unc = U.new_zeros(b, n, 1) if uncertainty is None else uncertainty.reshape(b, 1, 1).expand(-1, n, -1)
        channel_logits = self.channel(torch.cat([U, U - template, motion, unc], -1)).squeeze(-1) + self.vote_bias
        residual = U - R
        evidence = torch.cat([residual, residual.abs()], -1)
        parity_logits = self.parity(evidence[:, idx].flatten(2)).squeeze(-1)
        decoded = decode_error_syndrome(channel_logits, support, parity_logits,
                                        self.bp_iterations, self.bp_damping)
        out = {**decoded, 'channel_logits': channel_logits, 'parity_logits': parity_logits,
               'symbol_logits': channel_logits - self.vote_bias,
               's': parity_logits.sigmoid(), 's_logits': parity_logits,
               'C_obs': torch.einsum('mn,bnd->bmd', H_bar, U),
               'C_ref': torch.einsum('mn,bnd->bmd', H_bar, R), 'U': U, 'R': R,
               'bp_messages': decoded['check_to_variable_llr']}
        if return_checks:
            out['H_bar'] = H_bar
        return out
