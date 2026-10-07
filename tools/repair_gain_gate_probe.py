"""Evaluate a per-token repair-gain gate on a trained SATR checkpoint."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--satr',type=Path,required=True); ap.add_argument('--diagnosis',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    from trackit.models import ModelImplSuggestions
    from trackit.models.methods.GOLA.builder import build_GOLA_model
    from tools.preflight_acceptance import load_stage_config
    from tools.verify_codetrack_initialization import real_batch
    root=ROOT; view=root/'_probe/codetrack_flow_validation/LasHeR_curated10'; names=(view/'trainingsetList.txt').read_text().splitlines(); data=real_batch(view,names)
    cfg=load_stage_config(str(root/'config/GOLA/codetrack_s1/config.yaml')); cfg['model']['codetrack']['corruption_enabled']=False; cfg['model']['codetrack']['motion_enabled']=False; cfg['model']['codetrack']['memory_enabled']=False; cfg['model']['codetrack']['topk_tokens']=4
    m=build_GOLA_model(cfg,ModelImplSuggestions()).cuda().float().eval(); m.load_state_dict(load_file(str(root/'weights/gola_b224.bin')),strict=False); ct=m.codetrack; ct.diagnosis.load_state_dict(load_file(str(args.diagnosis)),strict=True); ct.satr.load_state_dict(load_file(str(args.satr)),strict=True)
    got=[]; h=ct.register_forward_pre_hook(lambda _m,_i,kwargs: got.append(kwargs['F_L'].detach().clone()),with_kwargs=True)
    with torch.no_grad():
        m.reset_sequence(); m(**data); clean=got[-1]; bad=dict(data); bad['x']=data['x'].clone(); bad['x'][:,3:,84:140,84:140]=0; m.reset_sequence(); m(**bad); corrupted=got[-1]
    h.remove(); clean_t=ct._split(clean)['X_TIR'].detach(); x=ct._split(corrupted)['X_TIR'].detach(); H=ct.H.matrix().detach(); parts=ct._split(corrupted); tpl=ct.template_pool(torch.cat([parts['Z_RGB'],parts['Z_TIR'],parts['Z_on'],parts['D_TIR']],1).mean(1)).detach();
    with torch.no_grad():
        d=ct.diagnosis(parts['X_TIR'],parts['X_RGB'],H,template_context=tpl); rec=ct.satr(parts['X_TIR'],parts['X_RGB'],d['q'],ct.neighbour_index,H_bar=H,bp_messages=d.get('bp_messages'),syndrome=d['s'],template_pool=tpl,topk=4)
        idx=rec['suspect_index']; delta=rec['delta']; bidx=torch.arange(x.shape[0],device='cuda')[:,None].expand_as(idx)
        clean_head=m.head(clean_t); clean_map=clean_head['score_map'].float().sigmoid().flatten(1); target_cell=clean_map.argmax(dim=-1)
        target_score=clean_map.gather(1,target_cell[:,None]).squeeze(1)
        target_box=clean_head['boxes'].float().reshape(x.shape[0],-1,4).gather(1,target_cell[:,None,None].expand(-1,1,4)).squeeze(1)
        base_head=m.head(x); base_map=base_head['score_map'].float().sigmoid().flatten(1); base_score=base_map.gather(1,target_cell[:,None]).squeeze(1)
        base_box=base_head['boxes'].float().reshape(x.shape[0],-1,4).gather(1,target_cell[:,None,None].expand(-1,1,4)).squeeze(1)
        base_err=(target_score-base_score).square()+0.25*(target_box-base_box).square().mean(1)
        gains=[]; accepted=[]; gated=[]
        for j in range(idx.shape[1]):
            cand=x.clone(); cand[bidx[:,j],idx[:,j]]=cand[bidx[:,j],idx[:,j]]+delta[:,j]
            pred=m.head(cand); pmap=pred['score_map'].float().sigmoid().flatten(1); pscore=pmap.gather(1,target_cell[:,None]).squeeze(1); pbox=pred['boxes'].float().reshape(x.shape[0],-1,4).gather(1,target_cell[:,None,None].expand(-1,1,4)).squeeze(1); err=(target_score-pscore).square()+0.25*(target_box-pbox).square().mean(1); gain=base_err-err; gains.append(gain); accepted.append(gain>0)
            gated=cand if j==0 else gated
        gain=torch.stack(gains,1); acc=torch.stack(accepted,1); route=torch.zeros_like(x)
        out=x.clone()
        for j in range(idx.shape[1]):
            take=acc[:,j]
            out[bidx[:,j],idx[:,j]]=torch.where(take[:,None],x[bidx[:,j],idx[:,j]]+delta[:,j],out[bidx[:,j],idx[:,j]])
        gated_head=m.head(out); gm=gated_head['score_map'].float().sigmoid().flatten(1); gs=gm.gather(1,target_cell[:,None]).squeeze(1); gb=gated_head['boxes'].float().reshape(x.shape[0],-1,4).gather(1,target_cell[:,None,None].expand(-1,1,4)).squeeze(1); gated_err=(target_score-gs).square()+0.25*(target_box-gb).square().mean(1)
    target_overlap=float((idx==target_cell[:,None]).float().mean())
    report={'probe':'repair_gain_gate_v1','mean_candidate_gain':float(gain.mean()),'positive_fraction':float(acc.float().mean()),'base_error':float(base_err.mean()),'gated_error':float(gated_err.mean()),'gated_gain':float((base_err-gated_err).mean()),'mean_selected_gain':float(gain.max(1).values.mean()),'target_cell_overlap':target_overlap}
    args.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
