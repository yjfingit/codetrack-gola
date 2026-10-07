"""Single-stage native observation-word selection and syndrome-decoding pilot."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-report', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file, save_file
    from codetrack.word_decoder import NativeWordDecoder
    from codetrack.criteria import _tracking_loss
    from trackit.models.methods.GOLA.modules.head.mlp import MlpAnchorFreeHead
    from trackit.runner.evaluation.distributed.tracker_evaluator.components.post_process.box_with_score_map import PostProcessing_BoxWithScoreMap

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    manifest = json.loads(args.dataset_report.read_text())
    names = (ROOT / 'experiments/train10.txt').read_text().splitlines()
    val_names = set(names[-3:])
    frames, rows = [], []
    for record in manifest['results']:
        data = {k: v.cuda() for k, v in load_file(record['offered_words']).items()}
        frame_index = len(frames)
        frames.append((record, data))
        rows.extend((frame_index, word) for word in range(len(data['words'])))
    train_rows = [r for r in rows if frames[r[0]][0]['sequence'] not in val_names]
    val_indices = [i for i, f in enumerate(frames) if f[0]['sequence'] in val_names]
    if not train_rows or not val_indices:
        raise RuntimeError('native report must contain disjoint train and validation sequences')
    head = MlpAnchorFreeHead(768, (16, 16)).cuda().eval()
    weights = load_file(str(ROOT / 'weights/gola_b224.bin'))
    head.load_state_dict({k[5:]: v for k, v in weights.items() if k.startswith('head.')}, strict=True)
    del weights
    head.requires_grad_(False)
    model = NativeWordDecoder().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    post = PostProcessing_BoxWithScoreMap(torch.device('cuda'), (16, 16), (224, 224), .45)
    post.start()
    inputs = ('current_ir', 'current_rgb', 'identity', 'motion', 'word_statistics')

    def batch(entries):
        data = {k: [] for k in inputs}
        data.update(reference_ir=[], word_labels=[], bit_labels=[], boxes=[])
        pos_batch, pos_indices = [], []
        for i, (frame_index, word_index) in enumerate(entries):
            frame = frames[frame_index][1]
            for k in inputs:
                data[k].append(frame[k][word_index] if k == 'word_statistics' else frame[k])
            data['reference_ir'].append(frame['words'][word_index])
            data['word_labels'].append(frame['word_labels'][word_index])
            data['bit_labels'].append(frame['bit_labels'][word_index])
            data['boxes'].append(frame['gt_box'])
            indices = frame['gt_positive_indices']
            pos_indices.append(indices)
            pos_batch.append(torch.full_like(indices, i))
        data = {k: torch.stack(v).float() for k, v in data.items()}
        quality = torch.zeros(len(entries), 16, 16, device='cuda')
        pb, pi = torch.cat(pos_batch), torch.cat(pos_indices)
        quality.flatten(1)[pb, pi] = 1.
        target = {'num_positive_samples': torch.tensor([len(pi)], device='cuda', dtype=torch.float32),
                  'positive_sample_batch_dim_indices': pb, 'positive_sample_map_dim_indices': pi,
                  'boxes': data['boxes'], 'score_quality_map': quality}
        return data, target

    def forward(data):
        return model(**{k: data[k] for k in (*inputs, 'reference_ir')})

    def box_iou(tokens, gt):
        pred = post(head(tokens))['box'] / 224.
        inter = (torch.minimum(pred[:, 2:], gt[:, 2:]) - torch.maximum(pred[:, :2], gt[:, :2])).clamp_min(0).prod(-1)
        union = (pred[:, 2:] - pred[:, :2]).clamp_min(0).prod(-1) + (gt[:, 2:] - gt[:, :2]).clamp_min(0).prod(-1) - inter
        return inter / union.clamp_min(1e-8)

    @torch.no_grad()
    def evaluate():
        model.eval()
        comparisons = []
        quality_scores, quality_truth, bit_scores, bit_truth = [], [], [], []
        for index in val_indices:
            record, frame = frames[index]
            data, _ = batch([(index, w) for w in range(len(frame['words']))])
            out = forward(data)
            choice = int(out['word_quality'].argmax())
            original_iou = float(box_iou(data['current_ir'][choice:choice+1], data['boxes'][choice:choice+1])[0])
            final_iou = float(box_iou(out['reconstructed'][choice:choice+1], data['boxes'][choice:choice+1])[0])
            comparisons.append({'sequence': record['sequence'], 'frame': record['frame'],
                                'baseline_iou': original_iou, 'decoded_iou': final_iou,
                                'word_quality': float(out['word_quality'][choice]),
                                'written_tokens': int(out['accept'][choice].sum())})
            quality_scores.append(out['word_quality'])
            quality_truth.append(data['word_labels'])
            bit_scores.append(out['q'])
            bit_truth.append(data['bit_labels'])
        qs, qt = torch.cat(quality_scores), torch.cat(quality_truth)
        bs, bt = torch.cat(bit_scores).flatten(), torch.cat(bit_truth).flatten()
        return {'selected_box_iou_gain': sum(r['decoded_iou']-r['baseline_iou'] for r in comparisons)/len(comparisons),
                'word_brier': float((qs-qt).square().mean()), 'bit_brier': float((bs-bt).square().mean()),
                'frames': comparisons}

    best, history = None, []
    for step in range(args.steps):
        selected = torch.randint(len(train_rows), (args.batch_size,)).tolist()
        data, target = batch([train_rows[i] for i in selected])
        model.train()
        out = forward(data)
        bits = data['bit_labels']
        parity = torch.einsum('mn,bn->bm', model.support.float(), bits).remainder(2)
        detection = (F.binary_cross_entropy_with_logits(out['q_logits'], bits) +
                     F.binary_cross_entropy_with_logits(out['channel_logits'], bits) +
                     F.binary_cross_entropy_with_logits(out['syndrome_logits'], parity) +
                     F.binary_cross_entropy_with_logits(out['word_quality_logits'], data['word_labels']))
        tracking = _tracking_loss(head(out['reconstructed']), target, per_sample=True)[0]
        with torch.no_grad():
            original = _tracking_loss(head(data['current_ir']), target, per_sample=True)[0]
        positive = bits.bool()
        healthy = ~positive
        recovery = ((1-F.cosine_similarity(out['reconstructed'][positive], data['reference_ir'][positive], dim=-1)).mean()
                    if bool(positive.any()) else tracking.sum()*0.)
        preservation = (out['reconstructed'][healthy]-data['current_ir'][healthy]).square().mean()
        loss = detection + tracking.mean() + .5*F.relu(tracking-original).mean() + .2*recovery + preservation
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        if step % 25 == 0 or step == args.steps-1:
            validation = evaluate()
            row = {'step': step, 'loss': float(loss.detach()), 'validation': validation}
            history.append(row)
            print(json.dumps(row), flush=True)
            score = validation['selected_box_iou_gain'] - .01 * validation['word_brier']
            if best is None or score > best['score']:
                best = {'step': step, 'score': score,
                        'state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    args.output.mkdir(parents=True, exist_ok=True)
    save_file(best['state'], str(args.output/'decoder.safetensors'))
    model.load_state_dict(best['state'], strict=True)
    report = {'seed': args.seed, 'best_step': best['step'], 'train_sequences': names[:-3],
              'validation_sequences': names[-3:], 'training_rows': len(train_rows),
              'validation': evaluate(), 'history': history, 'checkpoint': str(args.output/'decoder.safetensors'),
              'scope': 'cached native representation pilot; no closed-loop PR/SR claim'}
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    post.stop()


if __name__ == '__main__':
    main()
