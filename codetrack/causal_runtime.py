"""Raw-image ordered rollout shared by causal training, validation and test.

One runtime owns one fixed batch of sequence identities. Only initialize() accepts an
annotation; step() accepts images and an optional training-only *previous* observation.
Current labels are consumed by loss code after prediction. State forks preserve crop,
templates, Kalman posterior, temporal bank and unconfirmed proposals.
"""
from dataclasses import dataclass
import copy

import numpy as np
import torch

from codetrack.safety import choose_state, box_iou
from trackit.core.utils.siamfc_cropping import (
    apply_siamfc_cropping, apply_siamfc_cropping_to_boxes, get_siamfc_cropping_params,
    reverse_siamfc_cropping_params)
from trackit.core.operator.scale_and_translate import scale_and_translate
from trackit.core.utils.bbox_mask_gen import get_foreground_bounding_box
from trackit.core.operator.numpy.bbox.utility.image import bbox_clip_to_image_boundary_
from trackit.core.transforms.dataset_norm_stats import get_dataset_norm_stats_transform
from trackit.runner.evaluation.distributed.tracker_evaluator.components.post_process.box_with_score_map import PostProcessing_BoxWithScoreMap


def tree_copy(value, detach=False):
    if torch.is_tensor(value):
        return (value.detach() if detach else value).clone()
    if isinstance(value, dict):
        return {k: tree_copy(v, detach) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(tree_copy(v, detach) for v in value)
    return copy.deepcopy(value)


CT_STATE = ('_state', '_prev_score', '_prev_box', '_prev_admitted', '_gate_inputs',
            '_last_decision', '_commit_streak', '_motion_crop_params',
            '_motion_crop_size', '_motion_observe_current')
KF_STATE = ('_x', '_P', '_initialised', 'last_innovation', 'last_mahalanobis')


def code_state(code, detach=False):
    return tree_copy({
        'code': {k: getattr(code, k) for k in CT_STATE},
        'motion': ({k: getattr(code.motion, k) for k in KF_STATE}
                   if code.motion is not None else None),
        'admitted_once': getattr(code.memory, '_admitted_once', False),
    }, detach)


def restore_code_state(code, state):
    for k, v in state['code'].items():
        setattr(code, k, tree_copy(v))
    if code.motion is not None:
        for k, v in state['motion'].items():
            setattr(code.motion, k, tree_copy(v))
    if code.memory is not None:
        code.memory._admitted_once = state['admitted_once']


def _mask(box, params, device):
    bb = get_foreground_bounding_box(box, params, (14., 14.))
    bb[[0, 2]] = bb[[0, 2]].clip(0, 8)
    bb[[1, 3]] = bb[[1, 3]].clip(0, 8)
    mask = torch.zeros(8, 8, dtype=torch.long, device=device)
    mask[bb[1]:bb[3], bb[0]:bb[2]] = 1
    return mask


def _batch_crop(images, params, output_size, device):
    """Pad a heterogeneous CPU batch once, then crop all images in one GPU op."""
    moved=[image.to(device, non_blocking=True).float() for image in images]
    h=max(int(image.shape[-2]) for image in moved); w=max(int(image.shape[-1]) for image in moved)
    channels=moved[0].shape[0]
    batch=moved[0].new_zeros((len(moved),channels,h,w))
    means=[]
    for i,image in enumerate(moved):
        batch[i,:,:image.shape[-2],:image.shape[-1]]=image
        means.append(image.mean(dim=(-2,-1)))
    means=torch.stack(means)
    out, scale, translation=scale_and_translate(
        batch, np.asarray(output_size), params[:,0], params[:,1], means,
        'bilinear', False, return_adjusted_params=True)
    return out, np.stack((scale,translation),axis=-2), means


@dataclass
class RolloutResult:
    outputs: dict
    baseline: dict
    candidate: dict
    quality: dict
    crop_params: np.ndarray
    boxes: np.ndarray
    baseline_boxes: np.ndarray
    candidate_boxes: np.ndarray
    scores: torch.Tensor
    states: torch.Tensor
    input_data: dict


class CausalRuntime:
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self.normalize = get_dataset_norm_stats_transform('mm', inplace=True)
        self.post = PostProcessing_BoxWithScoreMap(self.device, (16, 16), (224, 224), .45)
        self.post.start()
        self.previous_proposal = None

    def initialize(self, images, initial_boxes):
        self.model.reset_sequence()
        self.crop_boxes = np.asarray(initial_boxes, dtype=np.float64).copy()
        self.previous_proposal = None
        params=np.stack([get_siamfc_cropping_params(box,2.,np.array((112,112)))
                         for box in self.crop_boxes])
        template_batch,actual,means=_batch_crop(images,params,(112,112),self.device)
        self.normalize(template_batch.div_(255.))
        self.z=template_batch
        self.zm=torch.stack([_mask(box,p,self.device) for box,p in zip(self.crop_boxes,actual)])
        self.d = self.z.detach().clone()
        self.dm = self.zm.detach().clone()
        self.means = [means[i] for i in range(len(images))]

    def snapshot(self, detach=True):
        return tree_copy({'crop_boxes': self.crop_boxes, 'previous_proposal': self.previous_proposal,
                         'z': self.z, 'zm': self.zm, 'd': self.d, 'dm': self.dm,
                         'means': self.means, 'code': code_state(self.model.codetrack, detach)}, detach)

    def restore(self, state):
        for key in ('crop_boxes', 'previous_proposal', 'z', 'zm', 'd', 'dm', 'means'):
            setattr(self, key, tree_copy(state[key]))
        restore_code_state(self.model.codetrack, state['code'])

    def detach_state(self):
        self.restore(self.snapshot(detach=True))

    def step(self, images, force_action=None, previous_training_boxes=None):
        """Predict before consuming this frame's annotation.

        force_action is training-only intervention: 0=no-op, 2=commit candidate.
        previous_training_boxes are boxes from the PREVIOUS frame, for scheduled sampling.
        """
        if previous_training_boxes is not None:
            if not self.model.training:
                raise ValueError('teacher forcing is forbidden during validation/test')
            self.crop_boxes = np.asarray(previous_training_boxes).copy()
        code = self.model.codetrack
        state_before = code_state(code, detach=False)
        crop_params=np.stack([get_siamfc_cropping_params(
            np.maximum(box.copy(),np.r_[box[:2],box[:2]+1.]),4.,np.array((224,224)))
            for box in self.crop_boxes])
        crop_batch,crop_params,_=_batch_crop(images,crop_params,(224,224),self.device)
        self.normalize(crop_batch.div_(255.))
        inputs = dict(z=self.z, d=self.d, x=crop_batch, z_feat_mask=self.zm,
                      d_feat_mask=self.dm, preserve_state=True, observe_motion=False,
                      image_size=torch.full((len(images), 2), 224., device=self.device),
                      search_crop_params=torch.as_tensor(crop_params, device=self.device))
        # Model's legacy score-only observation is superseded below by the SAME Hann
        # decoded box used for crop/state updates. Keep the covariance prediction/rebase,
        # but undo the legacy posterior update before publishing the accepted result.
        original_notify = code.notify_tracking_score
        code.notify_tracking_score = lambda *args, **kwargs: None
        try:
            output = self.model(**inputs)
        finally:
            code.notify_tracking_score = original_notify
        extras = output['codetrack_extras']
        baseline = extras['baseline_head']
        candidate = {key: output[key] for key in ('score_map', 'boxes')}
        quality = extras['candidate_quality']
        bd, cd = self.post(baseline), self.post(candidate)
        def to_world(decoded):
            boxes = apply_siamfc_cropping_to_boxes(
                decoded['box'].cpu().double().numpy(), reverse_siamfc_cropping_params(crop_params))
            for box, image in zip(boxes, images):
                bbox_clip_to_image_boundary_(box, np.array((image.shape[-1], image.shape[-2])))
            return boxes
        base_boxes, candidate_boxes = to_world(bd), to_world(cd)
        state = choose_state(quality, self.previous_proposal,
                             torch.as_tensor(base_boxes, device=self.device),
                             torch.as_tensor(candidate_boxes, device=self.device),
                             bd['confidence'], cd['confidence'])
        if force_action is not None:
            state = torch.full_like(state, force_action)
        elif output['codetrack_extras'].get('c_t') is not None:
            # The separately trained template-protection head participates in admission,
            # rather than receiving an auxiliary gradient while being bypassed at runtime.
            state = torch.where((state == 2) & (extras['c_t'] < .5),
                                torch.ones_like(state), state)
        write = state > 0
        commit = state == 2
        if self.model.training and force_action is None:
            soft = quality['write_probability'].reshape(-1, 1, 1)
            # Supervise the actual interpolated feature through the original GOLA head.
            tokens = extras['input_tokens']
            output_head = self.model.head(tokens + soft * (extras['recovered'] - tokens))
        else:
            output_head = {k: torch.where(write.view(-1, *([1] * (candidate[k].ndim-1))),
                                          candidate[k], baseline[k]) for k in candidate}
        # Report hard deployed decisions even when fitting the differentiable soft output.
        boxes = np.where(write.cpu().numpy()[:, None], candidate_boxes, base_boxes)
        scores = torch.where(write, cd['confidence'], bd['confidence'])
        self.previous_proposal = torch.as_tensor(candidate_boxes, device=self.device).detach()
        self.previous_proposal = torch.where(write[:, None], self.previous_proposal,
                                              torch.full_like(self.previous_proposal, float('nan')))
        # Provisional candidates do not modify persistent appearance/Kalman/crop state.
        # No-op preserves the baseline's ordinary crop and template update behavior.
        self.crop_boxes = np.where((state != 1).cpu().numpy()[:, None], boxes, self.crop_boxes)
        proposed_mem = code._state
        old_mem = state_before['code']['_state']
        admission_probability = None
        if self.model.training and force_action is None:
            # Fit the admission path softly: a hard initial no-op would otherwise erase
            # every historical summary and make TBPTT nominally enabled but ineffective.
            # At evaluation only confirmed commits are admitted.
            admission_probability = (quality['admission_logit'].sigmoid() *
                                     quality['state_logits'].softmax(1)[:, 2])
            if extras.get('c_t') is not None:
                admission_probability = admission_probability * extras['c_t']
        for name in ('memory', 'memory_rel'):
            if proposed_mem.get(name) is not None:
                old = old_mem.get(name)
                if old is None:
                    old = torch.zeros_like(proposed_mem[name])
                if admission_probability is None:
                    proposed_mem[name] = torch.where(commit[:, None, None], proposed_mem[name], old)
                else:
                    weight=admission_probability[:,None,None]
                    proposed_mem[name] = old + weight * (proposed_mem[name]-old)
        if code.motion is not None:
            local_corners = torch.where(write[:, None], cd['box'], bd['box'])
            cxwh = torch.cat(((local_corners[:, :2] + local_corners[:, 2:]) * .5,
                              (local_corners[:, 2:] - local_corners[:, :2]).clamp_min(1.)), 1)
            code.motion.observe(cxwh, inputs['image_size'], confidence=scores,
                                valid=(state != 1), predict=False)
        code._prev_box = None  # posterior has already absorbed the accepted observation
        code._prev_score = scores.detach()
        update = (scores > .84) & (state != 1)
        self.d, self.dm = self.d.clone(), self.dm.clone()
        for i in range(len(images)):
            if bool(update[i]):
                params = get_siamfc_cropping_params(boxes[i], 2., np.array((112, 112)))
                template,actual,_=_batch_crop([images[i]],np.asarray([params]),(112,112),self.device)
                self.normalize(template.div_(255.))
                self.d[i] = template[0].detach()
                self.dm[i] = _mask(boxes[i], actual[0], self.device)
        output = {**output, **output_head}
        return RolloutResult(output, baseline, candidate, quality, crop_params, boxes,
                             base_boxes, candidate_boxes, scores, state, inputs)


def targets_for(boxes_xyxy, crop_params, device):
    """Ground truth -> loss labels AFTER prediction, fixed binary classification labels."""
    raw=np.asarray(boxes_xyxy)
    valid=np.isfinite(raw).all(1)&(raw[:,2:]>raw[:,:2]).all(1)
    boxes = apply_siamfc_cropping_to_boxes(np.where(valid[:,None],raw,0.), crop_params) / 224.
    n = len(boxes)
    normalized = torch.as_tensor(boxes, dtype=torch.float32, device=device)
    valid=torch.as_tensor(valid,device=device)
    lin = (torch.arange(16, device=device) + .5) / 16
    yy, xx = torch.meshgrid(lin, lin, indexing='ij')
    inside = ((xx[None] >= normalized[:, 0,None,None]) & (xx[None] <= normalized[:,2,None,None]) &
              (yy[None] >= normalized[:, 1,None,None]) & (yy[None] <= normalized[:,3,None,None]))
    inside=inside&valid[:,None,None]
    pb, pm = inside.flatten(1).nonzero(as_tuple=True)
    return {'boxes': normalized, 'valid':valid,'score_quality_map': inside.float(),
            'positive_sample_batch_dim_indices': pb,
            'positive_sample_map_dim_indices': pm,
            'num_positive_samples': inside.sum().float().clamp_min(1)}


def four_frame_utility(runtime, future_images, future_annotations, current_snapshot, gamma=.9,
                       future_available=None):
    """Independent intervention/no-op forks with their OWN future crop/state updates.

    Snapshot is from BEFORE the candidate frame. All four images and annotations are
    available only to this training target computation. Restore the caller's state and
    random generators afterwards so labels cannot mutate the live online policy.
    """
    from codetrack.criteria import _tracking_loss
    live = runtime.snapshot(detach=False)
    model = runtime.model
    was_training = model.training
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    try:
        model.eval()
        costs, boxes = [], []
        with torch.no_grad():
            for intervention in (0, 2):
                runtime.restore(current_snapshot)
                losses, trajectory = [], []
                for t, (images, annotation) in enumerate(zip(future_images, future_annotations)):
                    result = runtime.step(images, force_action=intervention if t==0 else None)
                    target = targets_for(annotation, result.crop_params, runtime.device)
                    losses.append(_tracking_loss(result.outputs, target, per_sample=True)[0])
                    trajectory.append(torch.as_tensor(result.boxes, device=runtime.device))
                costs.append(torch.stack(losses)); boxes.append(torch.stack(trajectory))
            difference = costs[0] - costs[1]
            if future_available is not None:
                difference=difference*future_available.to(difference)
            signed = sum(gamma**t * difference[t] for t in range(len(difference)))
            consistency = torch.stack([box_iou(b, c) for b,c in zip(boxes[0],boxes[1])]).mean(0)
            return difference[0].detach(), signed.detach(), consistency.detach()
    finally:
        model.train(was_training)
        runtime.restore(live)
        torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state_all(cuda_rng)
