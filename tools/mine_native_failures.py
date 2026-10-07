"""Mine real tracking failures and their causal observed histories, without augmentation."""
import argparse
from collections import deque
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=Path, default=Path('/home/yangjuanfeng/lab/dataset/LasHeR'))
    ap.add_argument('--sequences', type=Path, default=ROOT / 'experiments/train10.txt')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--prefetch', type=int, default=8)
    ap.add_argument('--verify-prefetch', type=int, default=128)
    ap.add_argument('--max-frames', type=int, default=0, help='bounded smoke only; zero means whole sequence')
    ap.add_argument('--events-per-sequence', type=int, default=8)
    ap.add_argument('--clip-length', type=int, default=8)
    args = ap.parse_args()
    if args.workers < 0 or args.prefetch < 1 or args.clip_length < 4:
        ap.error('invalid producer or clip settings')

    import numpy as np
    import torch
    from safetensors.torch import save_file
    from tools.observed_tracker import ObservedGOLATracker, ordered_images
    from trackit.data.components.result_collector.handler.one_pass_evaluation_compatible.ope_metrics import (
        compute_one_pass_evaluation_metrics, compute_OPE_metrics_mean, calc_iou_overlap)
    from trackit.data.components.result_collector.handler.utils.compatibility import ExternalToolkitCompatibilityHelper

    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    tracker = ObservedGOLATracker(ROOT)
    names = args.sequences.read_text().splitlines()
    records, sequence_metrics = [], []

    def load_sequence(name):
        path = args.dataset / 'trainingset' / name
        boxes = np.loadtxt(path / 'init.txt', delimiter=',', ndmin=2)
        files = [sorted((path / m).glob('*.jpg')) for m in ('visible', 'infrared')]
        count = min(len(boxes), *(len(f) for f in files))
        return boxes, files, count

    def xyxy(box):
        out = box.copy(); out[2:] += out[:2]; return out

    def prefix_predictions(name, workers, count):
        boxes, files, actual_count = load_sequence(name)
        preds = []
        for frame, cpu_image in ordered_images(files, min(actual_count, count), workers, args.prefetch):
            image = cpu_image.to('cuda', non_blocking=cpu_image.is_pinned()).float()
            if frame == 0:
                tracker.initialize(image, xyxy(boxes[0])); preds.append(xyxy(boxes[0]))
            else:
                preds.append(tracker.track(image)['box'])
        tracker.end_sequence()
        return np.asarray(preds)

    if args.verify_prefetch:
        serial = prefix_predictions(names[0], 0, args.verify_prefetch)
        producer = prefix_predictions(names[0], args.workers, args.verify_prefetch)
        equal = np.array_equal(serial, producer)
        annotations, _, _ = load_sequence(names[0])
        annotations = annotations[:len(serial)].copy(); annotations[:, 2:] += annotations[:, :2]
        serial_metrics, _ = compute_one_pass_evaluation_metrics(
            'LasHeR', serial, annotations, None, np.ones(len(serial)), ExternalToolkitCompatibilityHelper())
        producer_metrics, _ = compute_one_pass_evaluation_metrics(
            'LasHeR', producer, annotations, None, np.ones(len(serial)), ExternalToolkitCompatibilityHelper())
        verification = {'sequence': names[0], 'frames': len(serial),
                        'predictions_exactly_equal': bool(equal),
                        'max_box_difference_pixels': float(np.max(np.abs(serial - producer))),
                        'PR_serial': serial_metrics.precision_score, 'SR_serial': serial_metrics.success_score,
                        'PR_prefetch': producer_metrics.precision_score, 'SR_prefetch': producer_metrics.success_score}
        (args.output / 'prefetch_verification.json').write_text(json.dumps(verification, indent=2) + '\n')
        print(json.dumps({'prefetch_verification': verification}), flush=True)
        if not equal:
            raise RuntimeError('producer path changed ordered predictions')

    for sequence_index, name in enumerate(names):
        boxes, files, count = load_sequence(name)
        if args.max_frames:
            count = min(count, args.max_frames)
        preds, times, frame_rows = [], [], []
        recent, bank = deque(maxlen=args.clip_length), deque(maxlen=3)
        events, last_event = [], -args.clip_length
        started = time.monotonic()
        for frame, cpu_image in ordered_images(files, count, args.workers, args.prefetch):
            tick = time.monotonic()
            image = cpu_image.to('cuda', non_blocking=cpu_image.is_pinned()).float()
            annotation = xyxy(boxes[frame])
            if frame == 0:
                tracker.initialize(image, annotation)
                initial_params, initial_tensors = tracker.initialization_reference(image)
                initial_reference = {'row': {'frame': 0, 'bbox': annotation.tolist(),
                                             'gt_bbox': annotation.tolist(), 'gt_valid': True,
                                             'iou': 1., 'confidence': 1.,
                                             'source': 'provided tracker initialization',
                                             'image_size': [int(image.shape[-1]), int(image.shape[-2])],
                                             'crop_params': initial_params.tolist()},
                                     'tensors': initial_tensors}
                preds.append(annotation.copy()); times.append(time.monotonic() - tick)
                continue
            # Current annotation is evaluated only after this image-only tracking call.
            out = tracker.track(image, capture=True)
            preds.append(out['box']); times.append(time.monotonic() - tick)
            pred_xywh = out['box'].copy(); pred_xywh[2:] -= pred_xywh[:2]
            valid = bool((boxes[frame, 2:] > 0).all())
            iou = float(calc_iou_overlap(pred_xywh[None], boxes[frame:frame + 1])[0]) if valid else -1.
            row = {'frame': frame, 'bbox': out['box'].tolist(), 'confidence': out['confidence'],
                   'gt_bbox': annotation.tolist(), 'gt_valid': valid, 'iou': iou,
                   'image_size': [int(image.shape[-1]), int(image.shape[-2])],
                   'crop_params': out['crop_params'].tolist(),
                   'motion_prediction_xywh': out['motion_prediction_xywh'],
                   'motion_uncertainty': out['motion_uncertainty']}
            sample = {'row': row, 'tensors': out['snapshot']}
            recent.append(sample); frame_rows.append(row)
            hard = (not valid) or iou < .5
            if hard and len(recent) == args.clip_length and frame - last_event >= args.clip_length * 2 \
                    and len(events) < args.events_per_sequence:
                # Bank membership uses observed confidence only. GT reliability tags
                # remain supervision metadata; they never determine tracker updates.
                tensors = {}
                clip_rows = [s['row'] for s in recent]
                history = [initial_reference] + list(bank)
                for role, samples in (('clip', list(recent)), ('history', history)):
                    for sample_index, sample_record in enumerate(samples):
                        for key, value in sample_record['tensors'].items():
                            tensors[f'{role}.{sample_index}.{key}'] = value.contiguous().clone()
                event_path = args.output / f'seq{sequence_index:02d}_event{len(events):02d}.safetensors'
                save_file(tensors, str(event_path))
                event = {'sequence': name, 'frame': frame,
                         'kind': 'tracking_failure' if valid else 'unknown_visibility',
                         'clip': clip_rows, 'history': [s['row'] for s in history],
                         'qualified_past_reference_count': sum(s['row']['iou'] >= .5 for s in history),
                         'tensors': str(event_path)}
                event_json = event_path.with_suffix('.event.json')
                temporary = event_json.with_suffix('.tmp')
                temporary.write_text(json.dumps(event, indent=2) + '\n')
                temporary.replace(event_json)
                events.append(event)
                last_event = frame
            # Admission is causal and does not inspect this frame's annotation.
            if out['confidence'] > .84:
                bank.append(sample)
            if frame % 500 == 0:
                print(json.dumps({'sequence': name, 'frame': frame, 'count': count,
                                  'events': len(events), 'elapsed_seconds': time.monotonic() - started}), flush=True)
        tracker.end_sequence()
        pred_array = np.asarray(preds)
        anno_array = boxes[:count].copy(); anno_array[:, 2:] += anno_array[:, :2]
        metrics, _ = compute_one_pass_evaluation_metrics('LasHeR', pred_array, anno_array, None,
                                                        np.asarray(times), ExternalToolkitCompatibilityHelper())
        sequence_metrics.append(metrics)
        record = {'sequence': name, 'frames': count, 'complete_sequence': count == load_sequence(name)[2],
                  'PR': metrics.precision_score, 'SR': metrics.success_score,
                  'elapsed_seconds': time.monotonic() - started, 'events': events}
        records.append(record)
        (args.output / f'seq{sequence_index:02d}_trace.json').write_text(json.dumps(frame_rows) + '\n')
        (args.output / 'progress.json').write_text(json.dumps({'sequences': records}, indent=2) + '\n')
        print(json.dumps({k: v for k, v in record.items() if k != 'events'}), flush=True)
    tracker.close()
    mean = compute_OPE_metrics_mean(sequence_metrics)
    report = {'protocol': 'ordered native LasHeR-train closed-loop frozen GOLA mining',
              'no_augmentation': True, 'current_gt_model_input': False,
              'scope': 'train research / target discovery, not LasHeR-test',
              'PR': mean.precision_score, 'SR': mean.success_score, 'sequences': records,
              'configuration': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('sequences', 'configuration')}), flush=True)


if __name__ == '__main__':
    main()
