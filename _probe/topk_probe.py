"""Causal TopK routing probe on a fixed real LasHeR batch.

This is an observation-only probe: it loads the shipped GOLA checkpoint, creates the
same controlled search corruption as the flow probe, and intervenes at the refiner's
q input.  It does not update parameters or alter project source/configuration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    from trackit.models import ModelImplSuggestions
    from trackit.models.methods.GOLA.builder import build_GOLA_model
    from tools.preflight_acceptance import load_stage_config
    from tools.verify_codetrack_initialization import real_batch

    torch.manual_seed(args.seed)
    device = torch.device('cuda')
    view = ROOT / '_probe/codetrack_flow_validation/LasHeR_curated10'
    names = (view / 'trainingsetList.txt').read_text().splitlines()
    data = real_batch(view, names)

    cfg = load_stage_config(str(ROOT / 'config/GOLA/codetrack_s1/config.yaml'))
    cfg['model']['codetrack']['corruption_enabled'] = False
    model = build_GOLA_model(cfg, ModelImplSuggestions()).to(device).eval()
    model.load_state_dict(load_file(str(ROOT / 'weights/gola_b224.bin')), strict=False)
    ct = model.codetrack

    captured: list[torch.Tensor] = []
    hook = ct.register_forward_pre_hook(
        lambda module, inputs, kwargs: captured.append(kwargs['F_L'].detach().clone()),
        with_kwargs=True)
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
    hook.remove()

    teacher = ct._split(clean)['X_TIR'].detach()
    xin = ct._split(damaged)['X_TIR'].detach()
    din = (1 - F.cosine_similarity(xin, teacher, dim=-1)).clamp(min=0)
    # A stable, non-injector target: tokens whose feature error exceeds the per-sample
    # mean by one standard deviation.  This is also the convention used by the existing
    # initialization/flow probes.
    mask = din > din.mean(dim=1, keepdim=True) + din.std(dim=1, keepdim=True)

    def run(**kwargs):
        ct.reset_sequence()
        with torch.no_grad():
            return model(**damaged_data, **kwargs)

    # First obtain the unmodified q/output.  The model is deterministic in eval mode.
    base = run()
    q_base = base['codetrack_extras']['q'].detach()
    b, n = q_base.shape
    k_default = int(ct.cfg.topk_tokens)

    def stats(out, q_used, label, k):
        ex = out['codetrack_extras']
        idx = ex['suspect_index']
        selected = torch.zeros_like(mask)
        selected.scatter_(1, idx, True)
        overlap = (selected & mask).float().sum(1) / mask.float().sum(1).clamp(min=1)
        # GOLA publishes routing diagnostics in codetrack_extras, while recovery
        # tensors are promoted to the model output by the wrapper.
        x_rec = out.get('X_rec', ex.get('recovered_pre_denoise'))
        x_final = out.get('X_final', ex.get('recovered'))
        tokens = out.get('tokens', ex.get('input_tokens', xin))
        if x_rec is None or x_final is None:
            raise RuntimeError('model output did not publish X_rec/X_final')
        d_before = (1 - F.cosine_similarity(x_rec, teacher, dim=-1)).clamp(min=0)
        d_after = (1 - F.cosine_similarity(x_final, teacher, dim=-1)).clamp(min=0)
        dm = mask
        delta_rec = (x_rec - tokens).norm(dim=-1)
        delta_final = (x_final - tokens).norm(dim=-1)
        return {
            'label': label, 'k': k,
            'q_std': float(q_used.std()),
            'q_logit_std': float(ex['q_logits'].std()),
            'topk_gap': float(q_used.topk(k, dim=-1).values.mean() - q_used.mean()),
            'selected_mask_overlap': float(overlap.mean()),
            'selected_count': float(selected.float().sum(1).mean()),
            'd_input': float(din[dm].mean()),
            'd_before': float(d_before[dm].mean()),
            'd_after': float(d_after[dm].mean()),
            'gain_before': float((din[dm] - d_before[dm]).mean()),
            'gain_after': float((din[dm] - d_after[dm]).mean()),
            'delta_rec_l2_selected': float(delta_rec[selected].mean()) if selected.any() else 0.0,
            'delta_final_l2_selected': float(delta_final[selected].mean()) if selected.any() else 0.0,
            'delta_final_l2_unselected': float(delta_final[~selected].mean()),
        }

    def intervention(label, q_new, k=k_default):
        def pre(mod, inputs, kwargs):
            kw = dict(kwargs)
            kw['q'] = q_new
            kw['topk'] = k
            return inputs, kw
        h = ct.refiner.register_forward_pre_hook(pre, with_kwargs=True)
        out = run()
        h.remove()
        return stats(out, q_new, label, k)

    rows = [stats(base, q_base, 'true', k_default)]
    perm = torch.stack([torch.randperm(n, device=device) for _ in range(b)])
    q_random = q_base.gather(1, perm)
    rows.append(intervention('random_permutation', q_random))
    q_fixed = q_base.mean(dim=1, keepdim=True).expand_as(q_base).clone()
    # Ties are intentionally deterministic in torch.topk; this is a fixed-index route.
    rows.append(intervention('uniform_fixed_tie', q_fixed))
    q_oracle = q_base.clone()
    # Preserve the observed q value distribution while assigning its highest scores to
    # the most damaged tokens.  This isolates routing quality from score amplitude.
    order = q_base.sort(dim=1, descending=True).values
    damage_order = din.argsort(dim=1, descending=True)
    q_oracle.scatter_(1, damage_order, order)
    rows.append(intervention('oracle_damage_rank', q_oracle))

    k_rows = []
    for k in (8, 16, 32, 64, 128, 256):
        k_rows.append(intervention(f'true_k{k}', q_base, k))

    # Gate scan is intentionally separate from q interventions.  It reveals whether a
    # selected set is being produced but then hidden by a near-zero write-back gate.
    gate_rows = []
    ref_gate_orig = ct.refiner.residual_gate.detach().clone()
    den_gate_orig = None if ct.denoiser is None else ct.denoiser.residual_gate.detach().clone()
    for which, module in [('refiner', ct.refiner), ('denoiser', ct.denoiser)]:
        if module is None:
            continue
        for value in (-8.0, -5.0, -2.0, 0.0, 3.0):
            with torch.no_grad():
                module.residual_gate.fill_(value)
            gate_rows.append(stats(run(), q_base, f'{which}_gate_{value:g}', k_default))
    with torch.no_grad():
        ct.refiner.residual_gate.copy_(ref_gate_orig)
        if ct.denoiser is not None and den_gate_orig is not None:
            ct.denoiser.residual_gate.copy_(den_gate_orig)

    # Norm-only comparison: normalize the input token vectors without any message
    # passing.  This is a diagnostic lower-level baseline, not a proposed model change.
    x_norm = F.layer_norm(xin, (xin.shape[-1],))
    d_norm_only = (1 - F.cosine_similarity(x_norm, teacher, dim=-1)).clamp(min=0)

    report = {
        'probe': 'topk_probe_v1', 'gpu': args.gpu, 'seed': args.seed,
        'sequences': names, 'batch': b, 'tokens': n, 'default_k': k_default,
        'corruption_mask_fraction': float(mask.float().mean()),
        'q_base_mean': float(q_base.mean()), 'q_base_std': float(q_base.std()),
        'q_base_min': float(q_base.min()), 'q_base_max': float(q_base.max()),
        'q_logit_std': float(base['codetrack_extras']['q_logits'].std()),
        'd_norm_only': float(d_norm_only[mask].mean()),
        'rows': rows, 'k_sweep': k_rows, 'gate_sweep': gate_rows,
        'source': str(__file__),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
