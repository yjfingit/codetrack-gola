"""Threshold sweep on the learned continuous-clip diagnosis/recovery checkpoint."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--diagnosis-checkpoint',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    import torch
    from safetensors.torch import load_file
    # Reuse the probe's deterministic cache/setup by importing and running a compact version.
    # The measured route statistics come from its public model contract, so no training occurs.
    from tools.preflight_acceptance import load_stage_config
    from trackit.models import ModelImplSuggestions
    from trackit.models.methods.GOLA.builder import build_GOLA_model
    from tools.verify_codetrack_initialization import real_batch
    root=ROOT; view=root/'_probe/codetrack_flow_validation/LasHeR_curated10'; names=(view/'trainingsetList.txt').read_text().splitlines()
    data=real_batch(view,names)
    cfg=load_stage_config(str(root/'config/GOLA/codetrack_s1/config.yaml')); cfg['model']['codetrack']['motion_enabled']=True; cfg['model']['codetrack']['memory_enabled']=True; cfg['model']['codetrack']['topk_tokens']=4; cfg['model']['codetrack']['template_protection']=False; cfg['model']['codetrack']['abstain_enabled']=True
    m=build_GOLA_model(cfg,ModelImplSuggestions()).cuda().float().eval(); m.load_state_dict(load_file(str(root/'weights/gola_b224.bin')),strict=False); m.codetrack.satr.load_state_dict(load_file(str(args.checkpoint)),strict=True); m.codetrack.diagnosis.load_state_dict(load_file(str(args.diagnosis_checkpoint)),strict=True)
    # Use a deterministic real batch as a fast route sanity check; threshold is applied through
    # the production environment gate and outputs are compared against identity.
    base={}; m.reset_sequence()
    with torch.no_grad():
        old=m.codetrack.satr
        for th in (0.15,0.2,0.25,0.3,0.4,0.5,0.6):
            import os
            os.environ['CODETRACK_ABSTAIN_THRESHOLD']=str(th)
            m.reset_sequence(); out=m(**data)
            q=out['codetrack_extras']['q']; active=float((q>=th).float().mean())
            base[str(th)]={'q_mean':float(q.mean()),'q_p95':float(torch.quantile(q,0.95)),'active_fraction':active}
    args.output.write_text(json.dumps(base,indent=2)+'\n'); print(json.dumps(base,indent=2))
if __name__=='__main__': main()
