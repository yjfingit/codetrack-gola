"""Frozen-GOLA causal trajectories for a bounded training-paradigm probe.

Only clip initialisation uses GT to form an input crop. Later annotations are
transformed through an already chosen crop for losses and paired-fault labels.
This is an off-policy feature cache, not a closed-loop candidate evaluation.
"""
from pathlib import Path
import json


class ReplayCache(list):
    """Feature rows plus supervision-only provenance, never student conditions."""
    def __init__(self):
        super().__init__()
        self.metadata = {}


def build_replay_cache(args):
    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from trackit.core.transforms.dataset_norm_stats import get_dataset_norm_stats_transform
    from trackit.core.utils.siamfc_cropping import (
        apply_siamfc_cropping, apply_siamfc_cropping_to_boxes,
        reverse_siamfc_cropping_params, get_siamfc_cropping_params)
    from trackit.core.utils.bbox_mask_gen import get_foreground_bounding_box
    from trackit.models import ModelImplSuggestions
    from trackit.models.methods.GOLA.builder import build_GOLA_model
    from trackit.runner.evaluation.common.siamfc_search_region_cropping_params_provider.simple import (
        SiamFCCroppingParameterSimpleProvider)
    from trackit.runner.evaluation.distributed.tracker_evaluator.components.post_process.box_with_score_map import (
        PostProcessing_BoxWithScoreMap)
    from tools.preflight_acceptance import load_stage_config
    from codetrack.observation_faults import TRAIN_FAULTS, HELDOUT_FAULTS, degrade_observation

    root = Path(__file__).resolve().parents[1]
    dataset = Path(args.dataset)
    names = Path(args.sequence_manifest).read_text().splitlines()
    cfg = load_stage_config(str(root / 'config/GOLA/codetrack_s1/config.yaml'))
    cfg['model']['codetrack'].update(corruption_enabled=False, motion_enabled=False,
                                   memory_enabled=False, template_protection=False)
    model = build_GOLA_model(cfg, ModelImplSuggestions()).cuda().float().eval()
    model.load_state_dict(load_file(str(root / 'weights/gola_b224.bin')), strict=False)
    normalize = get_dataset_norm_stats_transform('mm', inplace=True)
    post = PostProcessing_BoxWithScoreMap(torch.device('cuda'), (16, 16), (224, 224), .45)
    post.start()
    captured = []
    hook = model.codetrack.register_forward_pre_hook(
        lambda _m, _i, kw: captured.append(kw['F_L'].detach().clone()), with_kwargs=True)

    def read(files, frame):
        parts = []
        for fs in files:
            with Image.open(fs[frame]) as im:
                parts.append(np.array(im.convert('RGB'), copy=True))
        return torch.from_numpy(np.concatenate(parts, axis=-1)).permute(2, 0, 1).float().cuda()

    def crop(image, size, params, mean=None):
        out, _, actual = apply_siamfc_cropping(image, np.array(size), params,
                                             'bilinear', False, mean)
        return normalize(out.div(255.)), actual

    def template(image, box, mean):
        out, actual = crop(image, (112, 112),
                           get_siamfc_cropping_params(box, 2., np.array((112, 112))), mean)
        bb = get_foreground_bounding_box(box, actual, (14., 14.))
        bb[[0, 2]] = bb[[0, 2]].clip(0, 8)
        bb[[1, 3]] = bb[[1, 3]].clip(0, 8)
        mask = torch.zeros((1, 8, 8), dtype=torch.long, device='cuda')
        mask[:, bb[1]:bb[3], bb[0]:bb[2]] = 1
        return out[None], mask

    def features(z, x, d, zm, dm):
        captured.clear()
        model.reset_sequence()
        model(z=z, x=x[None], d=d, z_feat_mask=zm, d_feat_mask=dm)
        f = captured[-1]
        # Correction proposals never influence the cache policy. Use the frozen
        # baseline head applied to the received fused TIR tokens explicitly.
        out = model.head(model.codetrack._split(f)['X_TIR'])
        return f, out

    cache, telemetry = ReplayCache(), []
    physical = getattr(args, 'observation_recipe', 'legacy_blackout') == 'physical_mix'
    val_names = set(names[-args.val_count:])
    try:
        with torch.no_grad():
            for sequence_index, name in enumerate(names):
                seq = dataset / 'trainingset' / name
                gt = np.loadtxt(seq / 'init.txt', delimiter=',', ndmin=2)
                files = [sorted((seq / m).glob('*.jpg')) for m in ('visible', 'infrared')]
                count = min(len(gt), *(len(f) for f in files))
                if count < args.clip_length + 1:
                    raise ValueError(f'{name}: insufficient frames')
                # Middle-video windows expose transitions absent from first-4-frame probes.
                starts = np.linspace(.25, .75, args.clips_per_sequence)
                for window_index, frac in enumerate(starts):
                    start = min(int(frac * count), count - args.clip_length - 1)
                    init_box = gt[start].copy(); init_box[2:] += init_box[:2]
                    initial = read(files, start)
                    mean = initial.mean((-2, -1))
                    z, zm = template(initial, init_box, mean)
                    d, dm = z.clone(), zm.clone()
                    teacher_d, teacher_dm = d.clone(), dm.clone()
                    generator = torch.Generator(device='cuda').manual_seed(
                        args.seed + sequence_index * 1000 + window_index * 100)
                    families = TRAIN_FAULTS
                    if name in val_names and getattr(args, 'validation_degradation', 'seen') == 'heldout':
                        families = HELDOUT_FAULTS
                    family = families[(sequence_index * args.clips_per_sequence + window_index) % len(families)]
                    if name in val_names and getattr(args, 'validation_degradation', 'seen') == 'natural':
                        family = 'natural'
                    provider = SiamFCCroppingParameterSimpleProvider(4., 10.)
                    provider.initialize(init_box)
                    clean, corrupt, boxes, damage, params_rows, rows = [], [], [], [], [], []
                    for frame in range(start + 1, start + 1 + args.clip_length):
                        image = read(files, frame)
                        annotation = gt[frame].copy(); annotation[2:] += annotation[:2]
                        if args.crop_policy == 'gt_centered':
                            params = get_siamfc_cropping_params(annotation, 4., np.array((224, 224)))
                        else:
                            params = provider.get(np.array((224, 224)))
                        severity = (0., .5, .85, 0.)[(frame - start - 1) % 4]
                        if physical:
                            degraded_image = degrade_observation(
                                image, family, severity, generator, provider.cached_bbox.copy())
                            cor, actual = crop(degraded_image, (224, 224), params, mean)
                            x, teacher_params = crop(image, (224, 224), params, mean)
                            assert np.array_equal(actual, teacher_params)
                        else:
                            x, actual = crop(image, (224, 224), params, mean)
                            cor = x.clone()
                        target = apply_siamfc_cropping_to_boxes(annotation, actual)
                        cx, cy = (target[:2] + target[2:]) * .5
                        bw, bh = target[2:] - target[:2]
                        box = np.array([cx, cy, max(bw, .001), max(bh, .001)])
                        frac_damage = (0., .18, .32, 0.)[(frame - start - 1) % 4]
                        if not physical and frac_damage:
                            lo = np.floor([cx-frac_damage*bw, cy-frac_damage*bh]).clip(0, 224).astype(int)
                            hi = np.ceil([cx+frac_damage*bw, cy+frac_damage*bh]).clip(0, 224).astype(int)
                            cor[3:, lo[1]:hi[1], lo[0]:hi[0]] = 0.
                        if physical:
                            fc, student_head = features(z, cor, d, zm, dm)
                            if torch.equal(cor, x) and torch.equal(d, teacher_d):
                                f, head = fc, student_head
                            else:
                                f, head = features(z, x, teacher_d, zm, teacher_dm)
                            decoded = post(student_head)
                            frac_damage = severity if family != 'natural' else 0.
                        else:
                            f, head = features(z, x, d, zm, dm)
                            fc, student_head = features(z, cor, d, zm, dm) if frac_damage else (f, head)
                            decoded = post(head)
                        pred_crop = decoded['box'][0].cpu().double().numpy()
                        pred = apply_siamfc_cropping_to_boxes(pred_crop, reverse_siamfc_cropping_params(actual))
                        wh_image = np.array([image.shape[-1], image.shape[-2]])
                        pred[[0, 2]] = pred[[0, 2]].clip(0, wh_image[0])
                        pred[[1, 3]] = pred[[1, 3]].clip(0, wh_image[1])
                        conf = float(decoded['confidence'][0])
                        provider.update(conf, pred, wh_image)
                        if conf > .84 and bool((pred[2:] > pred[:2]).all()):
                            d, dm = template(degraded_image if physical else image, pred, mean)
                            teacher_d, teacher_dm = template(image, pred, mean)
                        clean.append(f); corrupt.append(fc); boxes.append(box)
                        damage.append(frac_damage); params_rows.append(actual)
                        def iou_against_target(head_out):
                            selected = post(head_out)['box'][0]
                            truth = selected.new_tensor(target)
                            intersection = (torch.minimum(selected[2:], truth[2:]) -
                                            torch.maximum(selected[:2], truth[:2])).clamp_min(0.).prod()
                            union = ((selected[2:] - selected[:2]).clamp_min(0.).prod() +
                                     (truth[2:] - truth[:2]).clamp_min(0.).prod() - intersection)
                            return float(intersection / union.clamp_min(1e-6))
                        teacher_iou = iou_against_target(head)
                        teacher_score = float(post(head)['confidence'][0])
                        feature_change = 1. - torch.nn.functional.cosine_similarity(
                            model.codetrack._split(fc)['X_TIR'], model.codetrack._split(f)['X_TIR'], dim=-1)
                        if physical and frac_damage > 0 and frame == start + 2 \
                                and getattr(args, 'save_observation_examples', False):
                            from PIL import ImageDraw
                            preview_dir = args.output.parent / 'observation_examples'
                            preview_dir.mkdir(parents=True, exist_ok=True)
                            canvas = Image.new('RGB', (448, 480), 'white')
                            draw = ImageDraw.Draw(canvas)
                            for row_index, source in enumerate((image, degraded_image)):
                                raw, _, _ = apply_siamfc_cropping(
                                    source, np.array((224, 224)), params, 'bilinear', False, mean)
                                for modality_index in range(2):
                                    pixels = raw[modality_index * 3:(modality_index + 1) * 3].clamp(0, 255)
                                    panel = Image.fromarray(pixels.byte().permute(1, 2, 0).cpu().numpy())
                                    canvas.paste(panel, (modality_index * 224, row_index * 240 + 16))
                            draw.text((4, 2), 'reference RGB / TIR (supervision only)', fill='black')
                            draw.text((4, 242), f'{family}: observed RGB / TIR', fill='black')
                            canvas.save(preview_dir / f'{sequence_index}_{window_index}_{family}.png')
                        rows.append({'frame': frame, 'crop_params': actual.tolist(),
                                     'gt_in_crop': box.tolist(), 'baseline_box': pred.tolist(),
                                     'baseline_confidence': conf,
                                     'degradation_family': family if physical else 'legacy_blackout',
                                     'severity': frac_damage,
                                     'teacher_selected_iou': teacher_iou,
                                     'student_selected_iou': iou_against_target(student_head),
                                     'teacher_reliable': teacher_iou >= .5 and teacher_score >= .5,
                                     'reference_equals_received': bool(torch.equal(f, fc)),
                                     'backbone_feature_deviation': float(feature_change.mean())})
                    label = f'{getattr(args, "label_prefix", "")}{name}:{start}'
                    cache.append((label, torch.cat(clean), torch.cat(corrupt),
                                  torch.tensor(np.asarray(boxes), device='cuda', dtype=torch.float32),
                                  torch.tensor(damage, device='cuda', dtype=torch.float32),
                                  torch.tensor(np.asarray(params_rows), device='cuda', dtype=torch.float32)))
                    cache.metadata[label] = rows
                    telemetry.append({'sequence': name, 'start': start, 'frames': rows})
                    print(json.dumps({'cached': label, 'crop_policy': args.crop_policy}), flush=True)
    finally:
        hook.remove(); post.stop()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.trajectory.json').write_text(json.dumps({
        'policy': args.crop_policy, 'gt_input_frames': 'clip initialisation only' if args.crop_policy == 'baseline' else 'every frame (oracle control)',
        'observation_recipe': getattr(args, 'observation_recipe', 'legacy_blackout'),
        'prediction_and_template_source': 'degraded student observations' if physical else 'clean baseline (legacy oracle-history control)',
        'teacher_use': 'supervision targets only' if physical else 'legacy control',
        'windows': telemetry}, indent=2) + '\n')
    return cache
