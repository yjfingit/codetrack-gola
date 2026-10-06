# Modified by Zekai Shao
# Licensed under Apache-2.0: http://www.apache.org/licenses/LICENSE-2.0
# Add support for online template update

from typing import Dict, Tuple, Callable, Any, Optional, List
import json
import os
import numpy as np
import torch
from dataclasses import dataclass, field

from trackit.core.operator.numpy.bbox.utility.image import bbox_clip_to_image_boundary_
from trackit.core.utils.siamfc_cropping import apply_siamfc_cropping, apply_siamfc_cropping_to_boxes, \
    reverse_siamfc_cropping_params, apply_siamfc_cropping_subpixel, scale_siamfc_cropping_params
from trackit.core.transforms.dataset_norm_stats import get_dataset_norm_stats_transform
from trackit.runner.evaluation.common.siamfc_search_region_cropping_params_provider import CroppingParameterProvider
from ....components.post_process import TrackerOutputPostProcess
from ....components.segmentation import Segmentify_PostProcessor
from ....components.template_updater import TemplateUpdater
from ....components.tensor_cache import CacheService, TensorCache

from ... import TrackerEvaluationPipeline


@dataclass
class _LocalContext:
    reset_frame_indices: List[int] = field(default_factory=list)
    siamfc_cropping_params_provider: Optional[CroppingParameterProvider] = None


class OneStreamTracker_Evaluation_MainPipeline(TrackerEvaluationPipeline):
    def __init__(self, device: torch.device,
                 template_image_size: Tuple[int, int],
                 search_region_image_size: Tuple[int, int],  # W, H
                 search_curation_parameter_provider_factory: Callable[[], CroppingParameterProvider],
                 model_output_post_process: TrackerOutputPostProcess,
                 segmentify_post_process: Optional[Segmentify_PostProcessor],
                 template_updater: TemplateUpdater,
                 interpolation_mode: str, interpolation_align_corners: bool,
                 norm_stats_dataset_name: str,
                 visualization: bool):
        self.template_image_size = template_image_size
        self.search_region_image_size = search_region_image_size

        self.search_image_cropping_params_provider_factory = search_curation_parameter_provider_factory
        self.interpolation_mode = interpolation_mode
        self.interpolation_align_corners = interpolation_align_corners

        self.model_output_post_process = model_output_post_process
        self.segmentify_post_process = segmentify_post_process
        self.device = device

        self.image_normalization_transform_ = get_dataset_norm_stats_transform(norm_stats_dataset_name, inplace=True)
        self.visualization = visualization

        self.template_updater = template_updater

    def start(self, max_batch_size: int, global_shared_objects):
        if len(self.image_normalization_transform_.mean) == 6:
            template_shape = (6, self.template_image_size[1], self.template_image_size[0])
            search_region_shape = (6, self.search_region_image_size[1], self.search_region_image_size[0])
            self.all_tracking_template_image_mean_cache = CacheService(max_batch_size,
                                                                       TensorCache(max_batch_size, (6,), self.device))
        else:
            template_shape = (3, self.template_image_size[1], self.template_image_size[0])
            search_region_shape = (3, self.search_region_image_size[1], self.search_region_image_size[0])
            self.all_tracking_template_image_mean_cache = CacheService(max_batch_size,
                                                                       TensorCache(max_batch_size, (3,), self.device))

        self.all_tracking_task_local_contexts: Dict[Any, _LocalContext] = {}
        self.all_tracking_template_cache = CacheService(max_batch_size,
                                                        TensorCache(max_batch_size, template_shape, self.device))

        self.template_updater.start(max_batch_size, template_shape)

        global_shared_objects['template_cache'] = self.all_tracking_template_cache
        global_shared_objects['template_image_mean_cache'] = self.all_tracking_template_image_mean_cache

        self.cropping_parameter_cache = np.full((max_batch_size, 2, 2), float('nan'), dtype=np.float64)
        self.search_region_cache = torch.full((max_batch_size, *search_region_shape), float('nan'),
                                              dtype=torch.float, device=self.device)
        self.model_output_post_process.start()
        if self.segmentify_post_process is not None:
            self.segmentify_post_process.start(max_batch_size)

    def stop(self, global_shared_objects):
        if self.segmentify_post_process is not None:
            self.segmentify_post_process.stop()
        self.model_output_post_process.stop()
        assert len(self.all_tracking_task_local_contexts) == 0, "bug check: some tracking sequences are not finished"
        del self.cropping_parameter_cache
        del self.search_region_cache
        del self.all_tracking_template_cache
        del self.all_tracking_template_image_mean_cache
        del self.all_tracking_task_local_contexts

        self.template_updater.stop()

    def begin(self, context):
        for task in context.input_data.tasks:
            if task.task_creation_context is not None:
                assert task.id not in self.all_tracking_task_local_contexts
                self.all_tracking_task_local_contexts[task.id] = _LocalContext()

    def prepare_initialization(self, context, model_input_params):
        for task in context.input_data.tasks:
            if task.tracker_do_init_context is not None:
                init_context = task.tracker_do_init_context
                self.all_tracking_template_cache.put(task.id, init_context.input_data['curated_image'])
                self.all_tracking_template_image_mean_cache.put(task.id, init_context.input_data['image_mean'])
                cropping_params_provider = self.search_image_cropping_params_provider_factory()
                cropping_params_provider.initialize(init_context.gt_bbox)
                task_context = self.all_tracking_task_local_contexts[task.id]
                task_context.siamfc_cropping_params_provider = cropping_params_provider
                task_context.reset_frame_indices.append(init_context.frame_index)

                self.template_updater.initialize(task.id, init_context.input_data['curated_image'])

    def prepare_tracking(self, context, model_input_params):
        num_tracking_sequence = 0
        task_ids = []
        image_size_list = []
        frame_indices = []
        for task in context.input_data.tasks:
            if task.tracker_do_tracking_context is not None:
                track_context = task.tracker_do_tracking_context
                template_image_mean = self.all_tracking_template_image_mean_cache.get(task.id)
                cropping_params_provider = self.all_tracking_task_local_contexts[
                    task.id].siamfc_cropping_params_provider
                cropping_params = cropping_params_provider.get(np.array(self.search_region_image_size))
                x = track_context.input_data['image'].to(torch.float32)
                H, W = x.shape[-2:]
                image_size_list.append(np.array((W, H), dtype=np.int32))
                _, _, cropping_params = \
                    apply_siamfc_cropping(x, np.array(self.search_region_image_size), cropping_params,
                                          self.interpolation_mode, self.interpolation_align_corners,
                                          template_image_mean,
                                          out_image=self.search_region_cache[num_tracking_sequence, ...])
                self.cropping_parameter_cache[num_tracking_sequence, ...] = cropping_params
                num_tracking_sequence += 1
                task_ids.append(task.id)
                frame_indices.append(track_context.frame_index)

        if num_tracking_sequence == 0:
            return

        context.temporary_objects['task_ids'] = task_ids
        context.temporary_objects['x_frame_sizes'] = image_size_list
        context.temporary_objects['x_frame_indices'] = frame_indices
        context.temporary_objects['x_cropping_params'] = self.cropping_parameter_cache[: num_tracking_sequence, ...]

        z = self.all_tracking_template_cache.get_batch(task_ids)
        d = self.template_updater.get_batch(task_ids)
        x = self.search_region_cache[: num_tracking_sequence, ...]
        x = x / 255.
        self.image_normalization_transform_(x)

        # The motion prior normalises boxes against the frame, but the evaluation pipeline
        # never supplied ``image_size`` -- so the Kalman filter could not seed itself and its
        # output was a frozen constant for the whole sequence.  Publish the *search-region*
        # size: it is the frame the head's box is decoded into, so the prior, the head output
        # and the box we feed back as an observation all live in the same coordinates.
        model_input_params.update({'z': z, 'x': x, 'd': d})
        seq_n = len(task_ids)
        model_input_params['image_size'] = torch.tensor(
            [self.search_region_image_size] * seq_n, dtype=torch.float32,
            device=x.device)

    def on_tracked(self, model_outputs, context):
        if model_outputs is None:
            return
        task_ids = context.temporary_objects['task_ids']
        x_frame_sizes = context.temporary_objects['x_frame_sizes']
        x_frame_indices = context.temporary_objects['x_frame_indices']
        x_cropping_params = context.temporary_objects['x_cropping_params']

        # Optional per-frame CodeTrack telemetry.  It is deliberately outside the tracking
        # path and only enabled by CODETRACK_DIAG_PATH, so normal evaluation is unchanged.
        self._record_codetrack_diagnostics(model_outputs, context, task_ids, x_frame_indices)

        outputs = self.model_output_post_process(model_outputs)
        # shape: (num_tracking_sequence), dtype: torch.float
        all_predicted_score = outputs['confidence']
        # shape: (num_tracking_sequence, 4), dtype: torch.float
        all_predicted_bounding_box = outputs['box']
        # shape: (num_tracking_sequence, H, W), dtype: torch.bool, allow None
        all_predicted_mask = outputs.get('mask', None)

        assert all_predicted_score.ndim == 1
        assert all_predicted_bounding_box.ndim == 2
        assert all_predicted_bounding_box.shape[1] == 4
        assert len(task_ids) == len(all_predicted_score) == len(all_predicted_bounding_box)
        if all_predicted_mask is not None:
            assert all_predicted_mask.ndim == 3
            assert all_predicted_mask.shape[0] == len(task_ids)

        all_predicted_score = all_predicted_score.cpu()
        assert torch.all(torch.isfinite(all_predicted_score))
        all_predicted_bounding_box = all_predicted_bounding_box.cpu()
        assert torch.all(torch.isfinite(all_predicted_bounding_box))

        all_predicted_bounding_box = all_predicted_bounding_box.to(torch.float64)

        all_predicted_score = all_predicted_score.numpy()
        all_predicted_bounding_box = all_predicted_bounding_box.numpy()

        all_predicted_bounding_box_on_full_search_image = apply_siamfc_cropping_to_boxes(
            all_predicted_bounding_box, reverse_siamfc_cropping_params(x_cropping_params))
        for predicted_bounding_box_on_full_search_image, image_size in zip(
                all_predicted_bounding_box_on_full_search_image, x_frame_sizes):
            bbox_clip_to_image_boundary_(predicted_bounding_box_on_full_search_image, image_size)

        all_predicted_mask_on_full_search_image = None
        if all_predicted_mask is not None:
            all_predicted_mask_on_full_search_image = []
            for curr_mask, curr_image_size, curr_cropping_parameter in zip(
                    all_predicted_mask, x_frame_sizes, x_cropping_params):
                mask_h, mask_w = curr_mask.shape
                curr_cropping_parameter = scale_siamfc_cropping_params(curr_cropping_parameter,
                                                                       np.array(self.search_region_image_size),
                                                                       np.array((mask_w, mask_h)))
                predicted_mask_on_full_search_image = apply_siamfc_cropping_subpixel(
                    curr_mask.to(torch.float32).unsqueeze(0),
                    np.array(curr_image_size), reverse_siamfc_cropping_params(curr_cropping_parameter),
                    self.interpolation_mode, self.interpolation_align_corners)
                all_predicted_mask_on_full_search_image.append(
                    predicted_mask_on_full_search_image.squeeze(0).to(torch.bool).cpu().numpy())
        else:
            if self.segmentify_post_process is not None:
                full_search_region_images = []

                for task in context.input_data.tasks:
                    if task.tracker_do_tracking_context is not None:
                        full_search_region_images.append(task.tracker_do_tracking_context.input_data['image'])
                all_predicted_mask_on_full_search_image = (
                    self.segmentify_post_process(full_search_region_images,
                                                 all_predicted_bounding_box_on_full_search_image))

        for index, (task_id, image_size, frame_index) in enumerate(zip(task_ids, x_frame_sizes, x_frame_indices)):
            predicted_score = all_predicted_score[index].item()
            predicted_bounding_box_on_full_search_image = all_predicted_bounding_box_on_full_search_image[index]
            local_task_context = self.all_tracking_task_local_contexts[task_id]
            local_task_context.siamfc_cropping_params_provider.update(predicted_score,
                                                                      predicted_bounding_box_on_full_search_image,
                                                                      image_size)
            predicted_mask_on_full_search_image = all_predicted_mask_on_full_search_image[index] \
                if all_predicted_mask_on_full_search_image is not None else None
            context.result.submit(task_id,
                                  predicted_bounding_box_on_full_search_image,
                                  predicted_score,
                                  predicted_mask_on_full_search_image)
            if self.visualization:
                from .visualization import visualize_tracking_result
                sequence_info = context.all_tracks[task_id].sequence_info
                x = self.search_region_cache[index, ...]
                z = self.all_tracking_template_cache.get(task_id)
                d = self.template_updater.get(task_id)
                predicted_bounding_box = all_predicted_bounding_box[index]
                predicted_mask = all_predicted_mask[index] if all_predicted_mask is not None else None
                # if sequence_info.sequence_name == 'redetricycle':
                visualize_tracking_result(sequence_info.dataset_name, sequence_info.sequence_name, frame_index,
                                          z, x, d, predicted_bounding_box,
                                          predicted_mask, predicted_mask_on_full_search_image)

        assert context.result.is_all_submitted()

    @staticmethod
    def _record_codetrack_diagnostics(model_outputs, context, task_ids, frame_indices):
        path = os.environ.get('CODETRACK_DIAG_PATH')
        if not path:
            return
        detail_frames = {}
        for item in os.environ.get('CODETRACK_DIAG_DETAIL_FRAMES', '').split(','):
            if ':' in item:
                name, frame = item.rsplit(':', 1)
                try:
                    detail_frames[name] = int(frame)
                except ValueError:
                    pass
        code = model_outputs.get('codetrack') if isinstance(model_outputs, dict) else None
        extras = model_outputs.get('codetrack_extras') if isinstance(model_outputs, dict) else None
        if not isinstance(code, dict):
            return

        def cpu_tensor(value):
            if not torch.is_tensor(value):
                return None
            return value.detach().float().cpu()

        q = cpu_tensor(code.get('q'))
        s = cpu_tensor(code.get('s'))
        delta = cpu_tensor(code.get('delta'))
        alpha = cpu_tensor(code.get('alpha'))
        suspect = code.get('suspect_index')
        suspect = suspect.detach().long().cpu() if torch.is_tensor(suspect) else None
        motion_map = cpu_tensor(extras.get('motion_map_norm')) if isinstance(extras, dict) else None
        uncertainty = cpu_tensor(code.get('uncertainty'))
        c_t = cpu_tensor(code.get('c_t'))
        topk_q = cpu_tensor(code.get('mean_topk_q'))
        max_q = cpu_tensor(code.get('max_q'))
        syndrome_before = cpu_tensor(code.get('syndrome_energy_before'))
        syndrome_after = cpu_tensor(code.get('syndrome_energy_after'))
        if q is None or s is None:
            return
        selected = torch.zeros_like(q, dtype=torch.bool)
        if suspect is not None and suspect.ndim == 2:
            selected.scatter_(1, suspect, True)

        def scalar(x, i):
            if x is None:
                return None
            x = x.reshape(x.shape[0], -1)
            return float(x[i].mean()) if i < x.shape[0] else None

        rows = []
        for i, task_id in enumerate(task_ids):
            if i >= q.shape[0]:
                break
            seq = context.all_tracks[task_id].sequence_info
            qi, si = q[i], s[i]
            row = {
                'sequence': seq.sequence_name,
                'frame_index': int(frame_indices[i]),
                'q_mean': float(qi.mean()),
                'q_std': float(qi.std(unbiased=False)),
                'q_max': float(qi.max()),
                'syndrome_mean': float(si.mean()),
                'syndrome_std': float(si.std(unbiased=False)),
                'syndrome_max': float(si.max()),
                'selected_fraction': float(selected[i].float().mean()),
                'selected_q_mean': float(qi[selected[i]].mean()) if bool(selected[i].any()) else 0.0,
                'delta_norm_all': float(delta[i].norm(dim=-1).mean())
                    if delta is not None and delta.ndim == 3 else None,
                # SATR returns delta/alpha only for its K selected tokens (B, K, C)/(B, K),
                # while q and selected describe the full N-token grid.
                'delta_norm_selected': float(delta[i].norm(dim=-1).mean())
                    if delta is not None and delta.ndim == 3 else 0.0,
                'alpha_mean': scalar(alpha, i),
                'alpha_selected_mean': float(alpha[i].mean())
                    if alpha is not None and alpha.ndim == 2 else None,
                'uncertainty': scalar(uncertainty, i),
                'c_t': scalar(c_t, i),
                'mean_topk_q': scalar(topk_q, i),
                'max_q_output': scalar(max_q, i),
                'syndrome_energy_before': scalar(syndrome_before, i),
                'syndrome_energy_after': scalar(syndrome_after, i),
            }
            if motion_map is not None:
                mm = motion_map[i].reshape(-1).clamp_min(1e-8)
                row['motion_entropy'] = float(-(mm * mm.log()).sum())
                row['motion_max'] = float(mm.max())
            if isinstance(extras, dict):
                rel = cpu_tensor(extras.get('frame_reliability'))
                corrupt = cpu_tensor(extras.get('corruption_fraction'))
                row['frame_reliability'] = scalar(rel, i)
                row['corruption_fraction'] = scalar(corrupt, i)
            # Full tensors are written only for explicitly requested sequence/frame pairs.
            # This keeps routine telemetry compact while enabling publication-style heatmaps.
            if detail_frames.get(seq.sequence_name) == int(frame_indices[i]):
                score_map = model_outputs.get('score_map') if isinstance(model_outputs, dict) else None
                boxes = model_outputs.get('boxes') if isinstance(model_outputs, dict) else None
                pre = model_outputs.get('codetrack_pre') if isinstance(model_outputs, dict) else None
                row['q_values'] = qi.tolist()
                row['syndrome_values'] = si.tolist()
                row['selected_indices'] = suspect[i].tolist() if suspect is not None else []
                if delta is not None and delta.ndim == 3:
                    row['delta_norm_values'] = delta[i].norm(dim=-1).tolist()
                if torch.is_tensor(motion_map):
                    row['motion_map_values'] = motion_map[i].reshape(-1).tolist()
                if torch.is_tensor(score_map):
                    row['score_map_values'] = score_map.detach().float().cpu()[i].tolist()
                if torch.is_tensor(boxes):
                    row['box_values'] = boxes.detach().float().cpu()[i].tolist()
                if isinstance(pre, dict):
                    if torch.is_tensor(pre.get('score_map')):
                        row['pre_score_map_values'] = pre['score_map'].detach().float().cpu()[i].tolist()
                    if torch.is_tensor(pre.get('boxes')):
                        row['pre_box_values'] = pre['boxes'].detach().float().cpu()[i].tolist()
            rows.append(row)
        if rows:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, 'a', encoding='utf-8') as fh:
                for row in rows:
                    fh.write(json.dumps(row, ensure_ascii=True) + '\n')

    def do_custom_update(self, model, raw_model, context):
        # CodeTrack block 6 (template protection): the model exposes a per-frame ``confidence``
        # derived from the syndrome (`1 - mean severity of the most suspicious tokens`), so the
        # official 0.84 score rule is additionally gated by how trustworthy the frame's evidence
        # was.  ``raw_model`` is the unwrapped network; when CodeTrack is disabled the attribute
        # is absent and ``quality`` stays 1.0, leaving upstream behaviour untouched.
        quality = self._codetrack_quality(raw_model, context)
        for i, task in enumerate(context.input_data.tasks):
            self.template_updater.update(
                task.id, context.result.get(task.id).confidence,
                task.tracker_do_tracking_context.input_data['image'],
                context.result.get(task.id).box,
                quality=quality[i] if quality is not None else 1.0)

    @staticmethod
    def _codetrack_quality(raw_model, context):
        """Per-task template-protection confidence, or None when CodeTrack is off.

        The evaluation harness drives the model with one propagation step per call for a
        single sequence, so the model's ``_last_decision`` corresponds to the frame just
        scored.  It is read defensively: a missing or mis-shaped entry yields None (i.e. the
        upstream rule) rather than a crash mid-evaluation.
        """
        model = getattr(raw_model, 'module', raw_model)
        code = getattr(model, 'codetrack', None)
        if code is None:
            return None
        decision = getattr(code, '_last_decision', None)
        if not isinstance(decision, dict) or 'confidence' not in decision:
            return None
        conf = decision['confidence']
        try:
            values = conf.reshape(-1).tolist()
        except Exception:
            return None
        n = len(context.input_data.tasks)
        if len(values) < n:
            return None
        return values[:n]

    def end(self, context):
        for task in context.input_data.tasks:
            if task.do_task_finalization:
                self.all_tracking_template_cache.delete(task.id)
                self.all_tracking_template_image_mean_cache.delete(task.id)
                self.all_tracking_task_local_contexts.pop(task.id)

                self.template_updater.delete(task.id)
