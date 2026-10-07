"""An identical-token intervention must never become a harmful-token label."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trackit.models.methods.GOLA.gola import GOLA_DINOv2
from trackit.models.methods.GOLA.modules.head.mlp import MlpAnchorFreeHead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--output', type=Path)
    ap.add_argument('--trials', type=int, default=12)
    args = ap.parse_args()
    torch.manual_seed(42)
    head = MlpAnchorFreeHead(768, (16, 16)).to(args.device).eval()
    state = load_file(str(ROOT / 'weights/gola_b224.bin'), device=args.device)
    head.load_state_dict({k[len('head.'):]: v for k, v in state.items() if k.startswith('head.')}, strict=True)
    del state
    target = {'num_positive_samples': torch.tensor([4.], device=args.device),
              'positive_sample_batch_dim_indices': torch.zeros(4, dtype=torch.long, device=args.device),
              'positive_sample_map_dim_indices': torch.tensor([119, 120, 135, 136], device=args.device),
              'boxes': torch.tensor([[.4, .4, .6, .6]], device=args.device)}
    trials = []
    with torch.no_grad():
        for _ in range(args.trials):
            tokens = torch.randn(1, 256, 768, device=args.device)
            base = head(tokens)
            labels = GOLA_DINOv2._causal_token_impact_target(
                SimpleNamespace(head=head), tokens, tokens.clone(), base, base,
                tracking_targets=target)
            repeated = head(tokens.expand(32, -1, -1).contiguous())
            discrepancy = max(float((repeated[k][0] - base[k][0]).abs().max()) for k in base)
            trials.append({'max_head_batch_discrepancy': discrepancy,
                           'false_positive_label_fraction': float((labels > 0).float().mean()),
                           'max_label': float(labels.max())})
    report = {'identical_input': True, 'trials': trials,
              'pass': all(row['max_label'] == 0. for row in trials)}
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
