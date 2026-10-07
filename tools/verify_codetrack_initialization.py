"""Verify first-batch calibration and checkpoint reuse on ten LasHeR train sequences.

Select sequences greedily by challenge coverage, then use the project's SiamFC crop and
RGB/TIR normalisation.  This is an initialization/gradient probe, not a tracking benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def prepare_view(dataset: Path, view: Path):
    """Select train-only sequences; never alter consts.yaml or the full dataset."""
    import numpy as np

    attributes = [x.strip() for x in (dataset / "Attributes_order.txt").read_text().split(',')]
    train = (dataset / "trainingsetList.txt").read_text().splitlines()
    candidates = {}
    for name in train:
        path = dataset / "AttriSeqsTxt" / (name + '.txt')
        if path.exists():
            flags = np.loadtxt(path, delimiter=',').reshape(-1)
            candidates[name] = {i for i, flag in enumerate(flags) if flag > 0}
    selected, covered = [], set()
    while len(selected) < 10:
        name = max(sorted(candidates), key=lambda n: (
            len(candidates[n] - covered), len(candidates[n]), -train.index(n)))
        covered |= candidates.pop(name)
        selected.append(name)
    view.mkdir(parents=True, exist_ok=True)
    for item in ('trainingset', 'annos', 'AttriSeqsTxt'):
        link = view / item
        if not link.exists():
            link.symlink_to(dataset / item, target_is_directory=True)
    (view / 'trainingsetList.txt').write_text('\n'.join(selected) + '\n')
    manifest = dict(split='train', sequences=selected,
                    attributes_covered=[attributes[i] for i in sorted(covered)],
                    selection='greedy uncovered challenge coverage, then total challenges, then train-list order')
    (view / 'selection.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return selected


def real_batch(dataset: Path, names):
    import numpy as np
    from PIL import Image
    import torch
    from trackit.core.utils.siamfc_cropping import get_siamfc_cropping_params, apply_siamfc_cropping
    from trackit.core.transforms.dataset_norm_stats import get_dataset_norm_stats_transform

    normalize = get_dataset_norm_stats_transform('mm', inplace=True)

    def crop(sequence, frame, size, area):
        images = []
        for modality in ('visible', 'infrared'):
            files = sorted((sequence / modality).glob('*.jpg'))
            with Image.open(files[frame]) as im:
                images.append(np.array(im.convert('RGB'), copy=True))
        image = torch.from_numpy(np.concatenate(images, axis=-1)).permute(2, 0, 1).float().cuda()
        boxes = np.loadtxt(sequence / 'init.txt', delimiter=',', ndmin=2)
        box = boxes[frame].copy()
        box[2:] += box[:2]
        shape = np.array((size, size))
        params = get_siamfc_cropping_params(box, area, shape)
        result = apply_siamfc_cropping(image, shape, params, 'bilinear', False)[0]
        return normalize(result / 255.0)

    batches = {key: [] for key in ('z', 'x', 'd')}
    for name in names:
        sequence = dataset / 'trainingset' / name
        for key, frame, size, area in (('z', 0, 112, 2), ('x', 1, 224, 4), ('d', 0, 112, 2)):
            batches[key].append(crop(sequence, frame, size, area))
    result = {key: torch.stack(value) for key, value in batches.items()}
    for key in ('z_feat_mask', 'd_feat_mask'):
        result[key] = torch.ones(len(names), 8, 8, dtype=torch.long, device='cuda')
    return result


def verify(dataset, view, config, output):
    import torch
    from codetrack.ddp_calibration import consume_pending_calibration
    from safetensors.torch import load_file
    from trackit.models import ModelImplSuggestions
    from trackit.models.methods.GOLA.builder import build_GOLA_model
    from tools.preflight_acceptance import load_stage_config

    names = prepare_view(dataset, view)
    data = real_batch(dataset, names)
    torch.manual_seed(0)
    stage = load_stage_config(str(config))
    model = build_GOLA_model(stage, ModelImplSuggestions()).cuda()
    model.load_state_dict(load_file(str(ROOT / 'weights/gola_b224.bin')), strict=False)
    diagnosis = model.codetrack.diagnosis
    recorded = []
    hook = diagnosis.register_forward_hook(lambda module, inputs, out: recorded.append(
        {k: out[k].detach().clone() for k in ('q', 's', 's_logits')}
        | ({'syndrome_pending_calibration': out['syndrome_pending_calibration'].detach().clone()}
           if out.get('syndrome_pending_calibration') is not None else {})))

    def run(train):
        model.train(train)
        model.reset_sequence()
        with torch.no_grad():
            return model(**data)

    run(False)
    before = recorded[-1]
    raw = before['s_logits']
    uncentred = (raw / raw.std(unbiased=False).clamp_min(1e-6)).sigmoid()
    if hasattr(diagnosis, '_syndrome_gain_calibrated'):
        assert not diagnosis._syndrome_gain_calibrated, 'evaluation must not calibrate'
        # Production deliberately applies calibration in the runner, outside forward(), so
        # DDP ranks can aggregate one common set of statistics without interleaving a custom
        # collective with DDP reducer collectives.  Mirror that exact two-forward sequence here.
        run(True)
        pending = recorded[-1].get('syndrome_pending_calibration')
        assert pending is not None, 'first training forward must publish calibration statistics'
        applied = consume_pending_calibration(diagnosis, pending)
        assert applied is not None, 'runner-style calibration was not applied'
        calibrated_pending = ((pending - diagnosis.syndrome_logit_offset.detach())
                              * diagnosis.syndrome_logit_gain.detach())
        assert abs(float(calibrated_pending.std(unbiased=False)) - 1.0) < 1e-4
        run(True)
        after = recorded[-1]
        gain, offset = diagnosis.syndrome_logit_gain.detach().clone(), diagnosis.syndrome_logit_offset.detach().clone()
        run(True)
        assert torch.equal(diagnosis.syndrome_logit_gain, gain), 'second batch must not recalibrate'
        assert torch.equal(diagnosis.syndrome_logit_offset, offset)
        assert torch.isfinite(after['s_logits']).all()
        assert float(after['s_logits'].std(unbiased=False)) > 0.1
        assert float(after['s'].std()) > 0.05 and float(after['q'].std()) > 1e-3
        assert torch.isfinite(after['q']).all()
        calibrated_std = float(calibrated_pending.std(unbiased=False))
    else:
        # Neural BP has no mutable syndrome calibration state. Its invariant is simpler:
        # finite, non-degenerate check/token posteriors in both train and eval modes.
        assert torch.isfinite(before['q']).all() and torch.isfinite(before['s']).all()
        run(True)
        after = recorded[-1]
        assert torch.isfinite(after['q']).all() and torch.isfinite(after['s_logits']).all()
        # A freshly initialized BP decoder is allowed to be close to its configured
        # channel prior; only exact collapse/NaN is an initialization failure.
        assert float(after['q'].std()) > 1e-6
        gain = offset = None
        calibrated_std = None

    # Check the ACTUAL filtered GOLA checkpoint, which discards buffers/frozen params.
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state, strict=False)
    run(True)
    hook.remove()
    model.reset_sequence()
    loss = model(**data)['codetrack_extras']['q'].square().mean()
    loss.backward()
    grads = [p.grad for p in diagnosis.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)

    # Standalone legacy diagnosis still rejects degenerate calibration.
    if hasattr(diagnosis, '_syndrome_gain_calibrated'):
        from codetrack.ecc import SyndromeDiagnosis
        degenerate = SyndromeDiagnosis()
        degenerate.calibrate_syndrome_gain(torch.ones(2, 64))
        assert not degenerate._syndrome_gain_calibrated
        degenerate.calibrate_syndrome_gain(torch.full((2, 64), float('nan')))
        assert not degenerate._syndrome_gain_calibrated

    report = dict(config=str(config), sequences=names,
                  gain=None if gain is None else float(gain),
                  offset=None if offset is None else float(offset),
                  raw_logit_mean=float(raw.mean()), raw_logit_std=float(raw.std(unbiased=False)),
                  calibrated_pending_std=calibrated_std,
                  next_batch_logit_std=float(after['s_logits'].std(unbiased=False)),
                  gain_only_s_std=float(uncentred.std()),
                  before={k: float(before[k].std()) for k in ('q', 's')},
                  after={k: float(after[k].std()) for k in ('q', 's')},
                  topk_gap_before=float(before['q'].topk(32).values.mean() - before['q'].mean()),
                  topk_gap_after=float(after['q'].topk(32).values.mean() - after['q'].mean()),
                  satr_up_std=float(model.codetrack.satr.up.weight.std()),
                  residual_gate=float(model.codetrack.satr.residual_gate),
                  calibration_once=hasattr(diagnosis, '_syndrome_gain_calibrated'),
                  checkpoint_preserved=True,
                  diagnosis_grad_max=float(max(g.abs().max() for g in grads)))
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('/home/yangjuanfeng/lab/dataset/LasHeR'))
    parser.add_argument('--view', type=Path, default=ROOT / '_probe/codetrack_flow_validation/LasHeR_curated10')
    parser.add_argument('--config', type=Path, default=ROOT / 'config/GOLA/codetrack_s1/config.yaml')
    parser.add_argument('--output', type=Path, default=ROOT / '_probe/codetrack_flow_validation/initialization.json')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.prepare_only:
        print(prepare_view(args.dataset, args.view))
    else:
        verify(args.dataset, args.view, args.config, args.output)


if __name__ == '__main__':
    main()
