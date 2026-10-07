"""Compare binary syndrome decoding against exhaustive posterior enumeration."""
import itertools
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codetrack.syndrome_bp import decode_error_syndrome
from codetrack.codetrack import CodeTrack
from codetrack.config import CodeTrackConfig


def exact(error_logits, support, syndrome_logits):
    bits = torch.tensor(list(itertools.product((0., 1.), repeat=error_logits.numel())), dtype=torch.float64)
    parity = (bits @ support.double().t()) % 2
    # log p(e) + log p(observed syndrome | H e); constants cancel.
    weights = (bits @ error_logits.double() + parity @ syndrome_logits.double()).softmax(0)
    return (weights[:, None] * bits).sum(0)


class SyndromeBP(unittest.TestCase):
    def test_single_noisy_check_equals_exact_posterior(self):
        h = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
        error = torch.tensor([[-2., .7, -1.3, 2.]])
        for s in (-8., -1., 0., 1., 8.):
            syn = torch.tensor([[s]])
            q = decode_error_syndrome(error, h, syn, iterations=1)['q'][0]
            torch.testing.assert_close(q.double(), exact(error[0], h, syn[0]), atol=2e-6, rtol=2e-6)

    def test_tree_messages_sum_over_checks_and_match_exact(self):
        h = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=torch.bool)
        error = torch.tensor([[-1.4, -.3, 1.2, -.8]])
        syn = torch.tensor([[2., -1., 3.]])
        q = decode_error_syndrome(error, h, syn, iterations=4)['q'][0]
        torch.testing.assert_close(q.double(), exact(error[0], h, syn[0]), atol=2e-6, rtol=2e-6)

    def test_nonedge_cannot_change_connected_posterior(self):
        h = torch.tensor([[1, 1, 0]], dtype=torch.bool)
        a = decode_error_syndrome(torch.tensor([[-1., .2, -20.]]), h, torch.tensor([[3.]]))
        b = decode_error_syndrome(torch.tensor([[-1., .2, 20.]]), h, torch.tensor([[3.]]))
        torch.testing.assert_close(a['q'][:, :2], b['q'][:, :2])
        self.assertEqual(float(a['check_to_variable_llr'][..., 2].abs().sum()), 0.)

    def test_uncertain_syndrome_preserves_channel(self):
        h = torch.tensor([[1, 1, 0], [0, 1, 1]], dtype=torch.bool)
        logits = torch.tensor([[.7, -1., 2.]])
        q = decode_error_syndrome(logits, h, torch.zeros(1, 2))['q']
        torch.testing.assert_close(q, logits.sigmoid())

    def test_zero_symbol_has_finite_and_correct_gradient(self):
        h = torch.tensor([[1, 1, 1]], dtype=torch.bool)
        logits = torch.tensor([[0., -.4, -1.]], requires_grad=True)
        q = decode_error_syndrome(logits, h, torch.tensor([[2.]]), iterations=1)['q']
        q[0, 1].backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertGreater(float(logits.grad[0, 0].abs()), 0.)

    def test_integrated_visual_decoder_starts_uncertain_and_preserves_tokens(self):
        cfg = CodeTrackConfig(enabled=True, dim=32, z_len=4, x_len=16, grid=4,
                              mid_dim=8, syndrome_hidden=16, num_checks=16,
                              h_layout='binary_cycles', h_links_per_check=4,
                              h_min_col_degree=4, decoder_type='syndrome_bp',
                              detection_prior=.05, topk_tokens=4, num_neighbours=4,
                              refiner_hidden=16, memory_dim=8, memory_enabled=False,
                              template_protection=False, abstain_enabled=True,
                              abstain_threshold=.5)
        ct = CodeTrack(cfg).eval()
        tokens = torch.randn(1, 48, 32)
        out = ct(F_L=tokens, image_size=torch.tensor([[224., 224.]]),
                 search_crop_params=torch.tensor([[[1., 1.], [0., 0.]]]))
        torch.testing.assert_close(out['q'], torch.full((1, 16), .05))
        torch.testing.assert_close(out['parity_logits'], torch.zeros(1, 16))
        torch.testing.assert_close(out['X_final'], ct._split(tokens)['X_TIR'], atol=0., rtol=0.)
        (out['q_logits'].square().mean() + out['parity_logits'].sum()).backward()
        self.assertGreater(float(ct.diagnosis.channel[-1].weight.grad.abs().sum()), 0.)
        self.assertGreater(float(ct.diagnosis.parity[-1].weight.grad.abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main()
