"""Controlled information-flow/learnability probe using the curated LasHeR train10 view.

Cache real backbone features once, freeze the backbone, and use a paired clean/corrupt
search crop.  Recovery-only fitting is a capacity test, not an SR/AUC evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--mode', choices=('learn', 'interfaces'), default='interfaces')
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--source-package', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.source_package:
        sys.path.insert(0, str(args.source_package))
        # Import before helpers insert ROOT into sys.path, so the baseline stays isolated.
        import codetrack

    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    from trackit.models import ModelImplSuggestions
    from trackit.models.methods.GOLA.builder import build_GOLA_model
    from tools.preflight_acceptance import load_stage_config
    from tools.verify_codetrack_initialization import real_batch

    view = ROOT / '_probe/codetrack_flow_validation/LasHeR_curated10'
    names = (view / 'trainingsetList.txt').read_text().splitlines()
    data = real_batch(view, names)
    torch.manual_seed(0)
    cfg = load_stage_config(str(ROOT / 'config/GOLA/codetrack_s4/config.yaml'))
    cfg['model']['codetrack']['corruption_enabled'] = False
    model = build_GOLA_model(cfg, ModelImplSuggestions()).cuda().eval()
    model.load_state_dict(load_file(str(ROOT / 'weights/gola_b224.bin')), strict=False)
    ct = model.codetrack
    captured = []
    handle = ct.register_forward_pre_hook(
        lambda module, inputs, kwargs: captured.append(kwargs['F_L'].detach().clone()), with_kwargs=True)
    with torch.no_grad():
        model.reset_sequence()
        model(**data)
        clean = captured[-1]
        damaged_data = dict(data)
        damaged_data['x'] = data['x'].clone()
        damaged_data['x'][:, 3:, 84:140, 84:140] = 0
        model.reset_sequence()
        model(**damaged_data)
        damaged = captured[-1]
    handle.remove()
    del data, damaged_data
    teacher = ct._split(clean)['X_TIR'].detach()
    xin = ct._split(damaged)['X_TIR'].detach()
    din = (1 - F.cosine_similarity(xin, teacher, dim=-1)).clamp(min=0)
    mask = din > (din.mean(dim=1, keepdim=True) + din.std(dim=1, keepdim=True))
    ct.train()
    # Remove sampling noise from this controlled capacity experiment, retaining all other
    # recovery operations; the harness smoke separately exercises training-time noise.
    ct.denoiser.eval()
    with torch.no_grad():
        ct.reset_sequence()
        ct(F_L=damaged)
    initial_state = {k: v.detach().clone() for k, v in ct.state_dict().items()}
    parameter_groups = ('H', 'diagnosis', 'template_pool', 'motion', 'memory', 'refiner',
                        'condition_proj', 'denoiser', 'meanvar', 'template_gate')

    def run():
        ct.reset_sequence()
        return ct(F_L=damaged)

    def error(output):
        return (1 - F.cosine_similarity(output['X_final'], teacher, dim=-1))[mask].mean()

    report = dict(mode=args.mode, sequences=names, steps=args.steps,
                  source_package=str(args.source_package), d_input=float(din[mask].mean()))
    import codetrack
    report['source_file'] = codetrack.__file__
    if args.mode == 'learn':
        params = [p for n, p in ct.named_parameters() if not n.startswith('head.') and p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0)
        history = []
        for i in range(args.steps):
            output = run()
            loss = error(output)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if i % 25 == 0 or i == args.steps - 1:
                with torch.no_grad():
                    evaluated = run()
                    row = dict(step=i, loss=float(loss), d_after=float(error(evaluated)),
                               q_std=float(evaluated['q'].std()))
                    history.append(row)
                    print(json.dumps(row), flush=True)
        report['history'] = history
        report['improvement'] = history[0]['d_after'] - history[-1]['d_after']
    else:
        # Sentinel layout checks use indices, independent of teacher/student agreement.
        sentinel = torch.arange(768, device='cuda').view(1, 768, 1).float()
        splits = ct._split(sentinel)
        report['slices'] = {k: [int(v[0, 0]), int(v[0, -1]) + 1] for k, v in splits.items()}
        report['matches_native_head'] = torch.equal(splits['X_TIR'], model._fuse_search(sentinel, 64, 256))
        output = run()
        loss = error(output)
        loss.backward()
        report['recovery_gradients'] = {}
        for group in parameter_groups:
            grads = [p.grad for n, p in ct.named_parameters() if n.startswith(group + '.') and p.grad is not None]
            report['recovery_gradients'][group] = float(sum(g.detach().float().abs().sum() for g in grads))
        with torch.no_grad():
            base = run()['X_final']
        report['ablations'] = {}
        # Paired reproduction of the pre-fix motion-bias formula on the same cached LasHeR
        # features.  This isolates centring/standardisation from all other branch changes.
        old_motion_normalise = ct.refiner.motion_bias_normalise
        ct.refiner.motion_bias_normalise = False
        with torch.no_grad():
            unnormalised_motion = run()['X_final']
        ct.refiner.motion_bias_normalise = old_motion_normalise
        report['motion_bias_normalisation_effect'] = float(
            (unnormalised_motion - base).norm() / base.norm())
        # Intervene at consumer boundaries; reset state and hold cached inputs fixed.
        for name, module, field in (
            ('aux_to_refiner', ct.refiner, 'X_aux'),
            ('q_to_refiner', ct.refiner, 'q'),
            ('motion_to_refiner', ct.refiner, 'motion_map'),
            ('memory_to_refiner', ct.refiner, 'memory_readout'),
            ('template_to_refiner', ct.refiner, 'template_pool'),
            ('s_to_denoiser', ct.denoiser, 'syndrome'),
            ('motion_to_denoiser', ct.denoiser, 'motion'),
            ('memory_to_denoiser', ct.denoiser, 'memory'),
        ):
            def intervene(mod, inputs, kwargs, field=field):
                kwargs = dict(kwargs)
                value = kwargs.get(field)
                if torch.is_tensor(value):
                    kwargs[field] = torch.zeros_like(value)
                return inputs, kwargs
            hook = module.register_forward_pre_hook(intervene, with_kwargs=True)
            with torch.no_grad():
                result = run()['X_final']
                report['ablations'][name] = float((result - base).norm() / base.norm())
            hook.remove()
        q = run()['q'].detach()
        gate = ct.template_gate(torch.ones(len(names), device='cuda'), q)
        report['gate_topk_gap'] = float((gate['mean_topk_q'] - q.mean(dim=-1)).mean())
        # Full-model outputs must publish the learned gate to the real updater interface.
        ct.eval()
        run()
        ct.notify_tracking_score(torch.full((len(names),), .99, device='cuda'))
        report['template_quality'] = ct._last_decision['confidence'].tolist()
        report['template_gate'] = ct._last_decision['c_t'].tolist()
        ct._prev_box = torch.ones(len(names), 4, device='cuda')
        ct.reset_sequence()
        report['reset_clears_box'] = ct._prev_box is None
        report['reset_clears_admission'] = not ct.memory._admitted_once
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
