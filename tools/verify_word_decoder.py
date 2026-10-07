"""Native-word decoder erasure, exact preservation and train/inference witnesses."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codetrack.word_decoder import NativeWordDecoder


class WordDecoder(unittest.TestCase):
    def inputs(self):
        return dict(current_ir=torch.randn(2, 16, 32), reference_ir=torch.randn(2, 16, 32),
                    current_rgb=torch.randn(2, 16, 32), identity=torch.randn(2, 32),
                    motion=torch.tensor([[.5, .5, .3, .3, .1]]).repeat(2, 1),
                    word_statistics=torch.zeros(2, 12))

    def test_uncertain_word_is_an_exact_erasure(self):
        m = NativeWordDecoder(dim=32, mid=8, hidden=16, grid=4).eval()
        data = self.inputs(); out = m(**data)
        self.assertFalse(bool(out['accept'].any()))
        torch.testing.assert_close(out['reconstructed'], data['current_ir'], atol=0., rtol=0.)

    def test_both_word_and_symbol_evidence_are_required(self):
        m = NativeWordDecoder(dim=32, mid=8, hidden=16, grid=4).eval()
        data = self.inputs()
        with torch.no_grad():
            m.symbol[-1].bias.fill_(8.)
            m.gain[-1].bias.copy_(torch.tensor([1., -8.]))
            m.abstain[-1].bias.fill_(-8.)
        self.assertFalse(bool(m(**data)['accept'].any()))
        with torch.no_grad():
            m.word_quality[-1].bias.fill_(8.)
        out = m(**data)
        self.assertTrue(bool(out['accept'].all()))
        torch.testing.assert_close(out['reconstructed'], data['reference_ir'], atol=1e-6, rtol=1e-6)

    def test_training_forward_matches_deployment_and_has_tracking_gradients(self):
        m = NativeWordDecoder(dim=32, mid=8, hidden=16, grid=4)
        data = self.inputs()
        m.train(); train = m(**data)
        m.eval(); deployed = m(**data)
        torch.testing.assert_close(train['reconstructed'], deployed['reconstructed'], atol=0., rtol=0.)
        (train['reconstructed'] - data['reference_ir']).square().mean().backward()
        self.assertTrue(bool(torch.isfinite(m.word_quality[-1].bias.grad).all()))
        self.assertGreater(float(m.word_quality[-1].bias.grad.abs().sum()), 0.)
        with torch.no_grad():
            m.symbol[-1].bias.fill_(8.); m.word_quality[-1].bias.fill_(8.)
            m.gain[-1].bias.copy_(torch.tensor([1., -8.]))
            m.abstain[-1].bias.fill_(-8.)
        m.train(); accepted_training = m(**data)['reconstructed']
        m.eval(); accepted_deployment = m(**data)['reconstructed']
        torch.testing.assert_close(accepted_training, accepted_deployment, atol=0., rtol=0.)
        torch.testing.assert_close(accepted_deployment, data['reference_ir'], atol=0., rtol=0.)

    def test_two_weak_evidences_do_not_authorize_correction(self):
        m = NativeWordDecoder(dim=32, mid=8, hidden=16, grid=4).eval()
        with torch.no_grad():
            m.symbol[-1].bias.fill_(float(torch.logit(torch.tensor(.6))) - float(m.channel_prior))
            m.word_quality[-1].bias.fill_(float(torch.logit(torch.tensor(.6))))
        data = self.inputs(); out = m(**data)
        self.assertTrue(bool((out['q'] > .5).all() and (out['word_quality'] > .5).all()))
        self.assertFalse(bool(out['accept'].any()))
        torch.testing.assert_close(out['reconstructed'], data['current_ir'], atol=0., rtol=0.)

    def test_explicit_abstention_blocks_even_strong_token_evidence(self):
        m = NativeWordDecoder(dim=32, mid=8, hidden=16, grid=4).eval()
        with torch.no_grad():
            m.symbol[-1].bias.fill_(8.)
            m.word_quality[-1].bias.fill_(8.)
            m.gain[-1].bias.copy_(torch.tensor([1., -8.]))
            m.abstain[-1].bias.fill_(8.)
        data = self.inputs(); out = m(**data)
        self.assertTrue(bool((out['q'] > .5).all()))
        self.assertTrue(bool((out['word_quality'] > .5).all()))
        self.assertTrue(bool((out['abstain_probability'] > .5).all()))
        self.assertFalse(bool(out['word_accept'].any()))
        self.assertFalse(bool(out['accept'].any()))
        torch.testing.assert_close(out['reconstructed'], data['current_ir'], atol=0., rtol=0.)

    def test_uncertain_gain_blocks_positive_mean(self):
        m = NativeWordDecoder(dim=32, mid=8, hidden=16, grid=4).eval()
        with torch.no_grad():
            m.symbol[-1].bias.fill_(8.)
            m.word_quality[-1].bias.fill_(8.)
            m.abstain[-1].bias.fill_(-8.)
            # Positive mean but larger uncertainty: lower confidence bound remains negative.
            m.gain[-1].bias.copy_(torch.tensor([.1, 1.]))
        out = m(**self.inputs())
        self.assertTrue(bool((out['expected_gain'] > 0.).all()))
        self.assertTrue(bool((out['gain_lcb'] < 0.).all()))
        self.assertFalse(bool(out['word_accept'].any()))


if __name__ == '__main__':
    unittest.main()
