"""Ordered frozen-GOLA image tracker used for native-frame research provenance.

Only initialize() accepts GT. track() receives a real image and past tracker state.
The normal backbone, repository postprocessor and template updater form the path.
"""
from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from safetensors.torch import load_file

from trackit.models import ModelImplSuggestions
from trackit.models.methods.GOLA.builder import build_GOLA_model
from trackit.core.transforms.dataset_norm_stats import get_dataset_norm_stats_transform
from trackit.core.utils.siamfc_cropping import (
    apply_siamfc_cropping, apply_siamfc_cropping_to_boxes,
    get_siamfc_cropping_params, reverse_siamfc_cropping_params)
from trackit.core.utils.bbox_mask_gen import get_foreground_bounding_box
from trackit.core.operator.numpy.bbox.utility.image import bbox_clip_to_image_boundary_
from trackit.runner.evaluation.common.siamfc_search_region_cropping_params_provider.simple import SiamFCCroppingParameterSimpleProvider
from trackit.runner.evaluation.distributed.tracker_evaluator.components.post_process.box_with_score_map import PostProcessing_BoxWithScoreMap
from trackit.runner.evaluation.distributed.tracker_evaluator.components.template_updater.simple import SimpleTemplateUpdater
from trackit.miscellanies.image.io import read_image_with_auto_retry
from tools.preflight_acceptance import load_stage_config
from codetrack.motion import KalmanMotionPrior


def decode_pair(paths, pin_memory=True):
    arrays = [read_image_with_auto_retry(str(p)) for p in paths]
    image = torch.from_numpy(np.concatenate(arrays, axis=-1)).permute(2, 0, 1).contiguous()
    return image.pin_memory() if pin_memory else image


def ordered_images(files, count, workers=4, prefetch=8):
    """CPU producers may finish out of order; the consumer always yields frame i."""
    if workers == 0:
        for frame in range(count):
            yield frame, decode_pair([fs[frame] for fs in files], pin_memory=False)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        next_frame = 0
        for _ in range(min(prefetch, count)):
            pending.append((next_frame, pool.submit(decode_pair, [fs[next_frame] for fs in files])))
            next_frame += 1
        while pending:
            frame, future = pending.popleft()
            image = future.result()
            if next_frame < count:
                pending.append((next_frame, pool.submit(decode_pair, [fs[next_frame] for fs in files])))
                next_frame += 1
            yield frame, image


class ObservedGOLATracker:
    def __init__(self, root: Path, amp=True):
        cfg = load_stage_config(str(root / 'config/GOLA/codetrack_s1/config.yaml'))
        cfg['model']['codetrack']['enabled'] = False
        self.model = build_GOLA_model(cfg, ModelImplSuggestions()).cuda().eval()
        self.model.load_state_dict(load_file(str(root / 'weights/gola_b224.bin')), strict=False)
        self.model.requires_grad_(False)
        self.amp = amp
        self.normalize = get_dataset_norm_stats_transform('mm', inplace=True)
        self.post = PostProcessing_BoxWithScoreMap(torch.device('cuda'), (16, 16), (224, 224), .45)
        self.post.start()
        self.updater = SimpleTemplateUpdater(.84, 2., (112, 112), 'mm', 'bilinear', False, torch.device('cuda'))
        # This auxiliary filter lives in whole-image coordinates, while the
        # trainable CodeTrack prior uses search-crop coordinates. Use physical
        # frame-scale noise and a 0.1% floor instead of the crop's 2% extent floor.
        self.motion = KalmanMotionPrior(process_noise=1e-4, measurement_noise=4e-4).cuda().eval()
        self.motion.MIN_SIDE = .001
        self.fused = None
        self.hook = self.model.norm.register_forward_hook(
            lambda _module, _inputs, output: setattr(self, 'fused', output.detach()))

    def _mask(self, box, params):
        bb = get_foreground_bounding_box(box, params, (14., 14.))
        bb[[0, 2]] = bb[[0, 2]].clip(0, 8)
        bb[[1, 3]] = bb[[1, 3]].clip(0, 8)
        mask = torch.zeros(1, 8, 8, device='cuda', dtype=torch.long)
        mask[:, bb[1]:bb[3], bb[0]:bb[2]] = 1
        return mask

    @torch.no_grad()
    def initialize(self, image, initial_box):
        self.motion.reset_state()
        self.provider = SiamFCCroppingParameterSimpleProvider(4., 10.)
        self.provider.initialize(initial_box)
        params = get_siamfc_cropping_params(initial_box, 2., np.array((112, 112)))
        z, self.mean, actual = apply_siamfc_cropping(image, np.array((112, 112)), params, 'bilinear', False)
        self.z = self.normalize(z.div(255.))[None]
        self.zm = self._mask(initial_box, actual)
        self.dm = self.zm.clone()
        self.updater.start(1, (6, 112, 112))
        self.updater.initialize(0, self.z[0])
        size = image.new_tensor([[image.shape[-1], image.shape[-2]]])
        bb = image.new_tensor(initial_box)[None]
        cxcywh = torch.cat([(bb[:, :2] + bb[:, 2:]) * .5, bb[:, 2:] - bb[:, :2]], -1)
        self.motion.observe(cxcywh, size)

    @torch.no_grad()
    def track(self, image, capture=False):
        params = self.provider.get(np.array((224, 224)))
        x, _, actual = apply_siamfc_cropping(image, np.array((224, 224)), params,
                                           'bilinear', False, self.mean)
        x = self.normalize(x.div(255.))[None]
        d, dm_before = self.updater.get(0)[None], self.dm.clone()
        size = image.new_tensor([[image.shape[-1], image.shape[-2]]])
        motion_x, motion_p = self.motion._x.detach().clone(), self.motion._P.detach().clone()
        motion = self.motion(image_size=size, defer_observe=True, advance=True)
        with torch.autocast('cuda', dtype=torch.float16, enabled=self.amp):
            out = self.model(z=self.z, x=x, d=d, z_feat_mask=self.zm, d_feat_mask=dm_before)
        if self.fused is None or self.fused.shape != (1, 768, 768):
            raise RuntimeError('normal-backbone fused-token capture failed')
        decoded = self.post(out)
        pred = decoded['box'][0].cpu().double().numpy()
        pred = apply_siamfc_cropping_to_boxes(pred, reverse_siamfc_cropping_params(actual))
        bbox_clip_to_image_boundary_(pred, np.array([image.shape[-1], image.shape[-2]]))
        confidence = float(decoded['confidence'][0])
        snapshot = None
        if capture:
            snapshot = {'F_L': self.fused.detach().cpu().clone(),
                        'z': self.z.detach().cpu().clone(), 'd': d.detach().cpu().clone(),
                        'z_mask': self.zm.detach().cpu().clone(), 'd_mask': dm_before.cpu(),
                        'motion_posterior_x': motion_x.cpu(), 'motion_posterior_P': motion_p.cpu()}
        self.provider.update(confidence, pred, np.array([image.shape[-1], image.shape[-2]]))
        self.updater.update(0, confidence, image, pred)
        if confidence > .84:
            self.dm = self._mask(pred, get_siamfc_cropping_params(pred, 2., np.array((112, 112))))
        bb = image.new_tensor(pred)[None]
        self.motion.observe(torch.cat([(bb[:, :2] + bb[:, 2:]) * .5, bb[:, 2:] - bb[:, :2]], -1),
                            size, confidence=image.new_tensor([confidence]), predict=False)
        return {'box': pred, 'confidence': confidence, 'crop_params': actual,
                'motion_prediction_xywh': motion['motion_box'][0].cpu().tolist(),
                'motion_uncertainty': float(motion['uncertainty'][0]), 'snapshot': snapshot}

    @torch.no_grad()
    def initialization_reference(self, image):
        """Encode the supplied initial observation without changing tracking state."""
        params = self.provider.get(np.array((224, 224)))
        x, _, actual = apply_siamfc_cropping(image, np.array((224, 224)), params,
                                           'bilinear', False, self.mean)
        x = self.normalize(x.div(255.))[None]
        d = self.updater.get(0)[None]
        with torch.autocast('cuda', dtype=torch.float16, enabled=self.amp):
            self.model(z=self.z, x=x, d=d, z_feat_mask=self.zm, d_feat_mask=self.dm)
        return actual, {'F_L': self.fused.detach().cpu().clone(),
                        'z': self.z.detach().cpu().clone(), 'd': d.detach().cpu().clone(),
                        'z_mask': self.zm.detach().cpu().clone(), 'd_mask': self.dm.cpu().clone(),
                        'motion_posterior_x': self.motion._x.cpu().clone(),
                        'motion_posterior_P': self.motion._P.cpu().clone()}

    def end_sequence(self):
        self.updater.stop()

    def close(self):
        self.hook.remove()
        self.post.stop()
