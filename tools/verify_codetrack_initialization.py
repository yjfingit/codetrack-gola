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
        {k: out[k].detach().clone() for k in ('q', 's', 's_logits')}))

    def run(train):
        model.train(train)
        model.reset_sequence()
        with torch.no_grad():
            return model(**data)

    run(False)
    before = recorded[-1]
    assert not diagnosis._syndrome_gain_calibrated, 'evaluation must not calibrate'
    raw = before['s_logits']
    uncentred = (raw / raw.std(unbiased=False)).sigmoid()
    run(True)
    after = recorded[-1]
    gain, offset = diagnosis.syndrome_logit_gain.detach().clone(), diagnosis.syndrome_logit_offset.detach().clone()
    run(True)
    assert torch.equal(diagnosis.syndrome_logit_gain, gain), 'second batch must not recalibrate'
    assert torch.equal(diagnosis.syndrome_logit_offset, offset)
    assert abs(float(after['s_logits'].std(unbiased=False)) - 1.0) < 1e-4
    assert float(after['s'].std()) > 0.05 and float(after['q'].std()) > 1e-3
    assert torch.isfinite(after['q']).all()

    # Check the ACTUAL filtered GOLA checkpoint, which discards buffers/frozen params.
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    assert 'codetrack.diagnosis.syndrome_logit_offset' in state
    model.load_state_dict(state, strict=False)
    assert diagnosis._syndrome_gain_calibrated
    run(True)
    assert torch.equal(diagnosis.syndrome_logit_gain, gain)
    assert torch.equal(diagnosis.syndrome_logit_offset, offset)
    hook.remove()
    model.reset_sequence()
    loss = model(**data)['codetrack_extras']['q'].square().mean()
    loss.backward()
    for name in ('syndrome_logit_gain', 'syndrome_logit_offset'):
        grad = getattr(diagnosis, name).grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().max() > 0

    # Standalone diagnosis rejects degenerate calibration without poisoning its parameters.
    from codetrack.ecc import SyndromeDiagnosis
    degenerate = SyndromeDiagnosis()
    degenerate.calibrate_syndrome_gain(torch.ones(2, 64))
    assert not degenerate._syndrome_gain_calibrated
    degenerate.calibrate_syndrome_gain(torch.full((2, 64), float('nan')))
    assert not degenerate._syndrome_gain_calibrated

    report = dict(config=str(config), sequences=names, gain=float(gain), offset=float(offset),
                  raw_logit_mean=float(raw.mean()), raw_logit_std=float(raw.std(unbiased=False)),
                  gain_only_s_std=float(uncentred.std()),
                  before={k: float(before[k].std()) for k in ('q', 's')},
                  after={k: float(after[k].std()) for k in ('q', 's')},
                  topk_gap_before=float(before['q'].topk(32).values.mean() - before['q'].mean()),
                  topk_gap_after=float(after['q'].topk(32).values.mean() - after['q'].mean()),
                  refiner_up_std=float(model.codetrack.refiner.up.weight.std()),
                  denoiser_up_std=float(model.codetrack.denoiser.up.weight.std()),
                  residual_gate=float(model.codetrack.refiner.residual_gate),
                  calibration_once=True, checkpoint_preserved=True,
                  gain_grad=float(diagnosis.syndrome_logit_gain.grad.abs().max()),
                  offset_grad=float(diagnosis.syndrome_logit_offset.grad.abs().max()))
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
