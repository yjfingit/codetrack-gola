"""Merge the pretrained GOLA weights with causal diagnosis/template/H and SATR weights."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--satr", type=Path, required=True)
    ap.add_argument("--config", type=Path,
                    default=Path("config/GOLA/codetrack_eval/config.yaml"))
    ap.add_argument("--no-temporal", action="store_true")
    args = ap.parse_args()
    import torch
    from safetensors.torch import load_file, save_file
    from trackit.models import ModelImplSuggestions
    from trackit.models.methods.GOLA.builder import build_GOLA_model
    from tools.preflight_acceptance import load_stage_config

    cfg = load_stage_config(str(args.config))
    ct = cfg["model"]["codetrack"]
    ct.update({"h_layout": "grid", "h_free_edge_frac": 0.0,
               "topk_tokens": 8, "satr_rounds": 2,
               "abstain_enabled": True, "abstain_threshold": 0.6,
               "decode_gate_enabled": False, "accept_enabled": False})
    if args.no_temporal:
        ct.update({"motion_enabled": False, "memory_enabled": False})
    model = build_GOLA_model(cfg, ModelImplSuggestions()).cuda().float().eval()
    model.load_state_dict(load_file("weights/gola_b224.bin"), strict=False)
    bundle = load_file(str(args.bundle), device="cpu")
    diag = {k[len("diagnosis."):]: v for k, v in bundle.items() if k.startswith("diagnosis.")}
    pool = {k[len("template_pool."):]: v for k, v in bundle.items() if k.startswith("template_pool.")}
    model.codetrack.diagnosis.load_state_dict(diag, strict=True)
    model.codetrack.template_pool.load_state_dict(pool, strict=True)
    model.codetrack.H.H.data.copy_(bundle["H.H"].to(model.codetrack.H.H))
    model.codetrack.satr.load_state_dict(load_file(str(args.satr)), strict=True)
    # Save the full CodeTrack branch, including non-trainable Kalman/memory parameters and
    # buffers. GOLA's filtered state_dict intentionally omits frozen tensors, which is useful
    # for training checkpoints but unsafe when a candidate is evaluated without the exact
    # construction-time random state.
    state = {f"codetrack.{k}": v.detach().cpu()
             for k, v in model.codetrack.state_dict().items()}
    state.update({k: v.detach().cpu() for k, v in model.state_dict().items()
                  if not k.startswith("codetrack.")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v for k, v in state.items() if torch.is_tensor(v)}, str(args.output))
    print({"output": str(args.output), "keys": len(state),
           "satr_rounds": model.codetrack.cfg.satr_rounds,
           "topk": model.codetrack.cfg.topk_tokens,
           "h_layout": model.codetrack.cfg.h_layout})

if __name__ == "__main__":
    main()
