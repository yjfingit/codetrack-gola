"""Observable candidate utility and three-way state-admission policy.

Future utility is a training label, never an input. Confirmation compares the previous
proposal in full-image coordinates with the next independently observed baseline box.
"""
import torch
from torch import nn


class CandidateQualityHead(nn.Module):
    def __init__(self, dim=768, hidden=128):
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(dim * 3 + 4),
                                     nn.Linear(dim * 3 + 4, hidden), nn.GELU(),
                                     nn.Dropout(0.1), nn.Linear(hidden, 7))
        nn.init.normal_(self.network[-1].weight, std=0.002)
        nn.init.zeros_(self.network[-1].bias)
        with torch.no_grad():
            self.network[-1].bias[4:].copy_(torch.tensor([2., -1., -2.]))

    def forward(self, received, candidate, template, q, uncertainty):
        if uncertainty is None:
            uncertainty = q.new_zeros(q.shape[0])
        stats = torch.stack((q.mean(1), q.amax(1),
                             (candidate - received).square().mean((1, 2)).add(1e-8).sqrt(),
                             uncertainty.reshape(-1)), 1)
        result = self.network(torch.cat((received.mean(1), candidate.mean(1),
                                         template, stats), 1))
        return {"utility": result[:, :2], "motion_logit": result[:, 2],
                "admission_logit": result[:, 3], "state_logits": result[:, 4:],
                "write_probability": result[:, 4:].softmax(1)[:, 1:].sum(1)}


def box_iou(a, b):
    a, b = torch.as_tensor(a), torch.as_tensor(b)
    intersection = (torch.minimum(a[..., 2:], b[..., 2:]) -
                    torch.maximum(a[..., :2], b[..., :2])).clamp_min(0).prod(-1)
    union = (a[..., 2:] - a[..., :2]).clamp_min(0).prod(-1) + (
        b[..., 2:] - b[..., :2]).clamp_min(0).prod(-1) - intersection
    return intersection / union.clamp_min(1e-8)


def choose_state(quality, previous_proposal, baseline_box, candidate_box,
                 baseline_score, candidate_score):
    """0=no-op, 1=provisional, 2=commit; no annotations are accepted."""
    logits = quality['state_logits'].detach()
    proposed = logits.argmax(-1)
    useful = (quality['utility'][:, 0] > 0) & (quality['utility'][:, 1] > 0)
    safe = (quality['motion_logit'].sigmoid() >= .5) & (
        quality['admission_logit'].sigmoid() >= .5)
    safe &= candidate_score >= baseline_score
    state = torch.where(useful & safe & (proposed > 0),
                        torch.ones_like(proposed), torch.zeros_like(proposed))
    if previous_proposal is not None:
        confirmed = box_iou(previous_proposal, baseline_box) >= .5
        consistent = box_iou(baseline_box, candidate_box) >= .5
        state = torch.where((state == 1) & (proposed == 2) & confirmed & consistent,
                            torch.full_like(state, 2), state)
    return state
