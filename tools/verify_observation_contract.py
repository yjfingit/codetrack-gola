"""Observation formation and train/inference recovery-state contract witnesses."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codetrack.observation_faults import TRAIN_FAULTS, HELDOUT_FAULTS, degrade_observation
from codetrack.codetrack import CodeTrack
from codetrack.config import CodeTrackConfig


class ObservationContract(unittest.TestCase):
    def test_impairments_preserve_other_modality_and_input(self):
        image = torch.rand(6, 96, 96, generator=torch.Generator().manual_seed(42)) * 255.
        original = image.clone()
        for family in TRAIN_FAULTS + HELDOUT_FAULTS:
            a = degrade_observation(image, family, .7, torch.Generator().manual_seed(9), [25., 25., 75., 75.])
            b = degrade_observation(image, family, .7, torch.Generator().manual_seed(9), [25., 25., 75., 75.])
            torch.testing.assert_close(a, b, atol=0., rtol=0.)
            self.assertEqual(a.shape, image.shape)
            self.assertTrue(bool(torch.isfinite(a).all() and (a >= 0).all() and (a <= 255).all()))
            self.assertGreater(float((a - image).abs().sum()), 0.)
            untouched = slice(3, 6) if family.startswith('rgb_') else slice(0, 3)
            torch.testing.assert_close(a[untouched], image[untouched], atol=0., rtol=0.)
            torch.testing.assert_close(image, original, atol=0., rtol=0.)

    def test_natural_observation_is_exact_identity(self):
        image = torch.rand(6, 40, 40) * 255.
        out = degrade_observation(image, 'natural', 1., torch.Generator().manual_seed(1))
        torch.testing.assert_close(out, image, atol=0., rtol=0.)

    def test_training_and_inference_use_the_same_route_and_causal_state(self):
        torch.manual_seed(42)
        cfg = CodeTrackConfig(enabled=True, dim=32, z_len=4, x_len=16, grid=4,
                              mid_dim=8, syndrome_hidden=16, num_checks=16,
                              h_layout='binary_cycles', h_links_per_check=4,
                              h_min_col_degree=4, decoder_type='syndrome_bp',
                              detection_prior=.05, topk_tokens=4, num_neighbours=4,
                              refiner_hidden=16, memory_dim=8, memory_enabled=False,
                              template_protection=False, abstain_enabled=True,
                              abstain_threshold=.5, match_inference_route_training=True,
                              soft_route_training=False)
        ct = CodeTrack(cfg)
        with torch.no_grad():
            ct.diagnosis.channel[-1].bias.fill_(4.)
        frames = [torch.randn(1, 48, 32) for _ in range(2)]
        crop_params = [torch.tensor([[[1., 1.], [0., 0.]]]),
                       torch.tensor([[[1.2, 1.2], [20., -10.]]])]
        results = []
        for training in (True, False):
            ct.train(training); ct.reset_sequence()
            outputs = []
            for t, frame in enumerate(frames):
                out = ct(F_L=frame, image_size=torch.tensor([[224., 224.]]),
                         search_crop_params=crop_params[t], preserve_state=(t > 0),
                         observe_motion=False, eval_observe=True)
                outputs.append((out['q'].detach().clone(), out['X_final'].detach().clone(),
                                out['motion_map_norm'].detach().clone()))
                ct.notify_tracking_score(torch.tensor([.9]), torch.tensor([[112., 110., 40., 40.]]))
            results.append(outputs)
        for train, inference in zip(*results):
            for a, b in zip(train, inference):
                torch.testing.assert_close(a, b, atol=0., rtol=0.)
        self.assertGreater(float((results[0][0][1] - ct._split(frames[0])['X_TIR']).norm()), 0.)


if __name__ == '__main__':
    unittest.main()
