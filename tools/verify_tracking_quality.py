"""A worse box must not improve correction loss by lowering its own quality label."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codetrack.criteria import _tracking_loss, bbox_overlaps


class TrackingQuality(unittest.TestCase):
    def test_fixed_reference_removes_self_label_escape(self):
        truth = torch.tensor([[.4, .4, .6, .6]])
        good = {'score_map': torch.full((1, 1, 1), float(torch.logit(torch.tensor(.1)))),
                'boxes': truth.reshape(1, 1, 1, 4)}
        bad = {**good, 'boxes': torch.tensor([[[[.3, .3, .7, .7]]]])}
        target = {'num_positive_samples': torch.tensor([1.]), 'boxes': truth,
                  'positive_sample_batch_dim_indices': torch.tensor([0]),
                  'positive_sample_map_dim_indices': torch.tensor([0])}
        self.assertGreater(float(bbox_overlaps(truth, good['boxes'].reshape(1, 4), is_aligned=True)),
                           float(bbox_overlaps(truth, bad['boxes'].reshape(1, 4), is_aligned=True)))
        original_good = _tracking_loss(good, target)[0]
        original_bad = _tracking_loss(bad, target)[0]
        self.assertLess(float(original_bad), float(original_good))
        fixed = {**target, 'score_quality_map': torch.ones(1, 1, 1)}
        fixed_good = _tracking_loss(good, fixed)[0]
        fixed_bad = _tracking_loss(bad, fixed)[0]
        self.assertGreater(float(fixed_bad), float(fixed_good))
        print({'self_label_loss_good': float(original_good), 'self_label_loss_bad': float(original_bad),
               'fixed_label_loss_good': float(fixed_good), 'fixed_label_loss_bad': float(fixed_bad)})

    def test_per_sample_uses_the_same_reference_and_detaches_it(self):
        score = torch.full((2, 1, 1), -1., requires_grad=True)
        boxes = torch.tensor([[[[.4, .4, .6, .6]]], [[[.3, .3, .7, .7]]]], requires_grad=True)
        quality = torch.ones(2, 1, 1, requires_grad=True)
        target = {'num_positive_samples': torch.tensor([2.]),
                  'boxes': torch.tensor([[.4, .4, .6, .6]]).repeat(2, 1),
                  'positive_sample_batch_dim_indices': torch.tensor([0, 1]),
                  'positive_sample_map_dim_indices': torch.tensor([0, 0]),
                  'score_quality_map': quality}
        out = {'score_map': score, 'boxes': boxes}
        each = _tracking_loss(out, target, per_sample=True)[0]
        mean = _tracking_loss(out, target)[0]
        torch.testing.assert_close(each.mean(), mean.squeeze())
        mean.backward()
        self.assertIsNone(quality.grad)
        self.assertTrue(bool(torch.isfinite(score.grad).all() and torch.isfinite(boxes.grad).all()))


if __name__ == '__main__':
    unittest.main()
