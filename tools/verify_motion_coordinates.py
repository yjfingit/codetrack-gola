"""Physical-coordinate witnesses for the causal search-crop motion interface."""
import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codetrack.motion import KalmanMotionPrior


class MotionCoordinates(unittest.TestCase):
    def test_rebase_preserves_full_image_trajectory_and_covariance(self):
        m = KalmanMotionPrior()
        m.observe(torch.tensor([[100., 80., 30., 40.]]), torch.tensor([[224., 224.]]))
        m._x[:, 4:] = torch.tensor([[.02, -.03, .01, -.01]])
        old_scale = torch.tensor([[.5, .4]])
        old_shift = torch.tensor([[20., 30.]])
        new_scale = torch.tensor([[.8, .7]])
        new_shift = torch.tensor([[-40., 10.]])
        before, cov = m._x.clone(), m._P.clone()
        # Decode into physical image coordinates independently of the rebase.
        physical = before.clone()
        physical[:, :2] = (before[:, :2] * 224 - old_shift) / old_scale
        physical[:, 2:] = before[:, 2:] * 224 / old_scale.repeat(1, 3)
        ratio = new_scale / old_scale
        m.rebase(ratio, (new_shift - ratio * old_shift) / 224)
        after = m._x.clone()
        after[:, :2] = (after[:, :2] * 224 - new_shift) / new_scale
        after[:, 2:] = after[:, 2:] * 224 / new_scale.repeat(1, 3)
        torch.testing.assert_close(after, physical)
        old_j = 224 / old_scale.repeat(1, 4)
        new_j = 224 / new_scale.repeat(1, 4)
        torch.testing.assert_close(m._P * new_j[:, :, None] * new_j[:, None, :],
                                   cov * old_j[:, :, None] * old_j[:, None, :])
        torch.linalg.cholesky(m._P)

    def test_observation_consumes_exactly_one_frame_transition(self):
        m = KalmanMotionPrior()
        size = torch.tensor([[224., 224.]])
        m.observe(torch.tensor([[80., 90., 40., 30.]]), size)
        m._x[:, 4:6] = torch.tensor([[.03, .02]])
        expected_centre = m._x[:, :2].clone() + m._x[:, 4:6]
        result = m(image_size=size, defer_observe=True, advance=True)
        torch.testing.assert_close(result['state'][:, :2], expected_centre)
        current = result['state'][:, :4] * size.repeat(1, 2)
        m.observe(current, size, predict=False)
        torch.testing.assert_close(m.last_innovation, torch.zeros_like(current), atol=1e-6, rtol=0.)
        torch.testing.assert_close(m._x[:, :2], expected_centre)
        next_result = m(image_size=size, defer_observe=True, advance=True)
        torch.testing.assert_close(next_result['state'][:, :2], expected_centre + m._x[:, 4:6])

    def test_reset_clears_previous_sequence_innovation(self):
        m = KalmanMotionPrior()
        size = torch.tensor([[224., 224.]])
        m.observe(torch.tensor([[80., 90., 40., 30.]]), size)
        m.observe(torch.tensor([[100., 90., 40., 30.]]), size)
        self.assertGreater(float(m.last_innovation.abs().sum()), 0.)
        m.reset_state()
        result = m(image_size=size)
        self.assertEqual(float(result['innovation'].sum()), 0.)
        self.assertEqual(float(result['mahalanobis'].sum()), 0.)

    def test_empty_state_and_invalid_transform(self):
        m = KalmanMotionPrior()
        m.rebase(torch.ones(1, 2), torch.zeros(1, 2))
        m.observe(torch.tensor([[80., 90., 40., 30.]]), torch.tensor([[224., 224.]]))
        with self.assertRaises(ValueError):
            m.rebase(torch.zeros(1, 2), torch.zeros(1, 2))


if __name__ == '__main__':
    unittest.main()
