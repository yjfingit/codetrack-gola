"""Verify inference-available reference words on naturally failed GOLA observations.

Candidate encoding receives images/templates and past motion only. Current GT
is used afterwards to judge target reliability and label harmful token proposals.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', type=Path, nargs='+', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--mode', choices=('template', 'temporal'), default='template')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    import numpy as np
    import torch
    from safetensors.torch import load_file, save_file
    from tools.observed_tracker import ObservedGOLATracker, decode_pair
    from trackit.core.utils.siamfc_cropping import (
        apply_siamfc_cropping, get_siamfc_cropping_params,
        apply_siamfc_cropping_to_boxes, reverse_siamfc_cropping_params)
    from trackit.core.operator.numpy.bbox.utility.image import bbox_clip_to_image_boundary_
    from trackit.data.methods.siamese_tracker_train.transform.default.plugin.box_with_score_map_label_gen import positive_sample_assignment
    from trackit.data.components.result_collector.handler.one_pass_evaluation_compatible.ope_metrics import calc_iou_overlap
    from codetrack.criteria import _tracking_loss

    torch.manual_seed(42); torch.cuda.manual_seed_all(42); torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    engine = ObservedGOLATracker(ROOT)
    paths = sorted(p for folder in args.events for p in folder.glob('*.event.json'))
    if args.limit:
        paths = paths[:args.limit]
    results = []
    file_lists = {}

    def read(sequence, frame):
        if sequence not in file_lists:
            file_lists[sequence] = [sorted((Path('/home/yangjuanfeng/lab/dataset/LasHeR/trainingset') / sequence / m).glob('*.jpg'))
                                    for m in ('visible', 'infrared')]
        files = file_lists[sequence]
        return decode_pair([f[frame] for f in files]).to('cuda', non_blocking=True).float()

    def encode(raw_crop, z, d, zm, dm):
        # The reference uses only inference-available observations. No GT/mask/clean
        # annotation enters this call; initial z is the ordinary supplied template.
        x = engine.normalize(raw_crop.div(255.))[None]
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            head = engine.model(z=z, x=x, d=d, z_feat_mask=zm, d_feat_mask=dm)
        return engine.fused.detach().clone(), head

    for event_index, path in enumerate(paths):
        event = json.loads(path.read_text())
        row = event['clip'][-1]
        pack = load_file(event['tensors'])
        prefix = f'clip.{len(event["clip"]) - 1}.'
        z = pack[prefix + 'z'].cuda().float()
        d = pack[prefix + 'd'].cuda().float()
        zm = pack[prefix + 'z_mask'].cuda().long()
        dm = pack[prefix + 'd_mask'].cuda().long()
        received = pack[prefix + 'F_L'].cuda().float()
        params = np.asarray(row['crop_params'])
        image = read(event['sequence'], row['frame'])
        initial = read(event['sequence'], 0)
        mean = initial.mean((-2, -1))
        raw, _, actual = apply_siamfc_cropping(image, np.array((224, 224)), params, 'bilinear', False, mean)
        truth = np.asarray(row['gt_bbox'])
        gt_crop = apply_siamfc_cropping_to_boxes(truth, params)
        clipped = gt_crop.clip(0., 224.)
        pos = (positive_sample_assignment(clipped, np.array((16, 16)), np.array((224, 224)))
               if bool((clipped[2:] > clipped[:2]).all()) else np.empty(0, dtype=np.int64))
        target = {'num_positive_samples': torch.tensor([len(pos)], device='cuda', dtype=torch.float32),
                  'positive_sample_batch_dim_indices': torch.zeros(len(pos), device='cuda', dtype=torch.long),
                  'positive_sample_map_dim_indices': torch.tensor(pos, device='cuda', dtype=torch.long),
                  'boxes': torch.tensor(gt_crop[None] / 224., device='cuda', dtype=torch.float32),
                  'score_quality_map': torch.zeros(1, 16, 16, device='cuda')}
        target['score_quality_map'].flatten(1)[0, target['positive_sample_map_dim_indices']] = 1.

        def evaluate_word(fused, coordinate_params):
            tok = fused[:, 384:640].float()
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                out = engine.model.head(tok)
                loss = float(_tracking_loss(out, target)[0]) if np.array_equal(coordinate_params, params) else None
            decoded = engine.post(out)
            box = decoded['box'][0].cpu().double().numpy()
            box = apply_siamfc_cropping_to_boxes(box, reverse_siamfc_cropping_params(coordinate_params))
            bbox_clip_to_image_boundary_(box, np.array(row['image_size']))
            xywh = box.copy(); xywh[2:] -= xywh[:2]
            gt_xywh = truth.copy(); gt_xywh[2:] -= gt_xywh[:2]
            iou = float(calc_iou_overlap(xywh[None], gt_xywh[None])[0]) if row['gt_valid'] else -1.
            return {'iou': iou, 'loss': loss, 'confidence': float(decoded['confidence'][0]), 'bbox': box.tolist()}, tok, out

        baseline, base_tok, base_head = evaluate_word(received, params)
        reproduced, _ = encode(raw.clone(), z, d, zm, dm)
        difference = float((reproduced.float() - received).abs().max())
        candidates = []
        if args.mode == 'template':
            word, _ = encode(raw.clone(), z, z, zm, zm)
            candidates.append(('initial_template_word', word, params))
            mp = np.asarray(row['motion_prediction_xywh']); motion_box = mp.copy(); motion_box[2:] += motion_box[:2]
            motion_params = get_siamfc_cropping_params(motion_box, 4., np.array((224, 224)))
            motion_raw, _, motion_actual = apply_siamfc_cropping(
                image, np.array((224, 224)), motion_params, 'bilinear', False, mean)
            for name, online, mask in (('motion_observation_word', d, dm),
                                       ('motion_initial_template_word', z, zm)):
                word, _ = encode(motion_raw.clone(), z, online, zm, mask)
                candidates.append((name, word, motion_actual))
        else:
            mp = np.asarray(row['motion_prediction_xywh']); target_centre = mp[:2] + mp[2:] * .5
            for history_index, history in enumerate(event['history']):
                old = read(event['sequence'], history['frame'])
                bb = np.asarray(history['bbox']); size = np.maximum(bb[2:] - bb[:2], 1.)
                scale = np.maximum(mp[2:], 1.) / size
                shift = target_centre - scale * (bb[:2] + bb[2:]) * .5
                warp_params = np.stack([params[0] * scale, params[0] * shift + params[1]])
                warped, _, _ = apply_siamfc_cropping(old, np.array((224, 224)), warp_params, 'bilinear', False, old.mean((-2, -1)))
                for modality in ('RGB', 'TIR'):
                    hybrid = raw.clone()
                    channels = slice(0, 3) if modality == 'RGB' else slice(3, 6)
                    hybrid[channels] = warped[channels]
                    for suffix, online, mask in (('online', d, dm), ('initial', z, zm)):
                        word, _ = encode(hybrid.clone(), z, online, zm, mask)
                        candidates.append((f'history{history_index}_{modality}_{suffix}', word, params))
        comparisons, best_word, best_name, best_stats = [], None, None, None
        for name, word, coordinate_params in candidates:
            stats, _, _ = evaluate_word(word, coordinate_params)
            reliable = row['gt_valid'] and stats['iou'] >= .5 and stats['iou'] >= baseline['iou'] + .02
            stats.update({'name': name, 'reliable_tracking_target': bool(reliable),
                          'same_crop': bool(np.array_equal(coordinate_params, params))})
            comparisons.append(stats)
            if reliable and stats['same_crop'] and stats['loss'] < baseline['loss'] - 1e-4:
                if best_stats is None or stats['loss'] < best_stats['loss']:
                    best_word, best_name, best_stats = word, name, stats
        oracle = None
        if best_word is not None:
            reference, reference_head = best_word[:, 384:640].float(), engine.model.head(best_word[:, 384:640].float())
            labels = engine.model._causal_token_impact_target(
                base_tok, reference, base_head, reference_head, tracking_targets=target)
            bits = labels >= .5
            decoded_tokens = torch.where(bits[..., None], reference, base_tok)
            oracle_fused = received.clone(); oracle_fused[:, 384:640] = decoded_tokens
            oracle_stats, _, _ = evaluate_word(oracle_fused, params)
            oracle = {'reference': best_name, 'reference_stats': best_stats,
                      'strong_bad_fraction': float(bits.float().mean()),
                      'oracle_selected_iou': oracle_stats['iou'],
                      'oracle_tracking_loss_gain': baseline['loss'] - oracle_stats['loss']}
            save_file({'received': received.cpu(), 'reference': best_word.float().cpu(),
                       'causal_labels': labels.cpu()}, str(args.output / f'target{event_index:03d}.safetensors'))
        record = {'event': str(path), 'sequence': event['sequence'], 'frame': row['frame'],
                  'baseline': baseline, 'reencode_max_feature_difference': difference,
                  'candidates': comparisons, 'oracle_recovery': oracle}
        results.append(record)
        print(json.dumps({'sequence': event['sequence'], 'frame': row['frame'],
                          'baseline_iou': baseline['iou'], 'best_candidate_iou': max(c['iou'] for c in comparisons),
                          'qualified_same_crop_reference': best_name}), flush=True)
    engine.close()
    report = {'probe': 'native_inference_available_reference_words', 'mode': args.mode,
              'events': len(results), 'qualified_same_crop_targets': sum(r['oracle_recovery'] is not None for r in results),
              'results': results, 'no_current_gt_encoding_input': True, 'no_augmentation_for_bad_tokens': True}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'results'}), flush=True)


if __name__ == '__main__':
    main()
