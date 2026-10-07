"""Train a tiny causal repair-gain gate from candidate SATR edits.

The gate is trained on the frozen-head target-cell loss, then evaluated on held-out rows. It is
not wired into production inference until the held-out AUC and positive gain gates pass.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--steps',type=int,default=300); args=ap.parse_args()
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from safetensors.torch import load_file, save_file
    from trackit.models import ModelImplSuggestions
    from trackit.models.methods.GOLA.builder import build_GOLA_model
    from tools.preflight_acceptance import load_stage_config
    from tools.verify_codetrack_initialization import real_batch
    root=ROOT; view=root/'_probe/codetrack_flow_validation/LasHeR_curated10'; names=(view/'trainingsetList.txt').read_text().splitlines(); data=real_batch(view,names)
    cfg=load_stage_config(str(root/'config/GOLA/codetrack_s1/config.yaml')); cfg['model']['codetrack']['corruption_enabled']=False; cfg['model']['codetrack']['motion_enabled']=False; cfg['model']['codetrack']['memory_enabled']=False; cfg['model']['codetrack']['topk_tokens']=8
    m=build_GOLA_model(cfg,ModelImplSuggestions()).cuda().float().eval(); m.load_state_dict(load_file(str(root/'weights/gola_b224.bin')),strict=False); ct=m.codetrack; ct.diagnosis.load_state_dict(load_file(str(root/'_probe/causal_q_probe.safetensors')),strict=True); ct.satr.load_state_dict(load_file(str(root/'_probe/recovery_causal_q.safetensors')),strict=True)
    got=[]; h=ct.register_forward_pre_hook(lambda _m,_i,kwargs: got.append(kwargs['F_L'].detach().clone()),with_kwargs=True)
    with torch.no_grad():
        m.reset_sequence(); m(**data); clean=got[-1]; bad=dict(data); bad['x']=data['x'].clone(); bad['x'][:,3:,84:140,84:140]=0; m.reset_sequence(); m(**bad); corrupted=got[-1]
    h.remove(); H=ct.H.matrix().detach(); p=ct._split(corrupted); clean_t=ct._split(clean)['X_TIR'].detach(); tpl=ct.template_pool(torch.cat([p['Z_RGB'],p['Z_TIR'],p['Z_on'],p['D_TIR']],1).mean(1)).detach()
    with torch.no_grad():
        d=ct.diagnosis(p['X_TIR'],p['X_RGB'],H,template_context=tpl); rec=ct.satr(p['X_TIR'],p['X_RGB'],d['q'],ct.neighbour_index,H_bar=H,bp_messages=d.get('bp_messages'),syndrome=d['s'],template_pool=tpl,topk=8)
        idx,delta=rec['suspect_index'],rec['delta']; b=torch.arange(clean_t.shape[0],device='cuda')[:,None].expand_as(idx); clean_head=m.head(clean_t); cm=clean_head['score_map'].float().sigmoid().flatten(1); cell=cm.argmax(-1); cs=cm.gather(1,cell[:,None]).squeeze(1); cb=clean_head['boxes'].float().reshape(clean_t.shape[0],-1,4).gather(1,cell[:,None,None].expand(-1,1,4)).squeeze(1)
        base=m.head(p['X_TIR']); bm=base['score_map'].float().sigmoid().flatten(1); bs=bm.gather(1,cell[:,None]).squeeze(1); bb=base['boxes'].float().reshape(clean_t.shape[0],-1,4).gather(1,cell[:,None,None].expand(-1,1,4)).squeeze(1); base_err=(cs-bs).square()+0.25*(cb-bb).square().mean(-1)
        feats=[]; labels=[]
        for j in range(idx.shape[1]):
            cand=p['X_TIR'].clone(); cand[b[:,j],idx[:,j]]=cand[b[:,j],idx[:,j]]+delta[:,j]; pred=m.head(cand); pm=pred['score_map'].float().sigmoid().flatten(1); ps=pm.gather(1,cell[:,None]).squeeze(1); pb=pred['boxes'].float().reshape(clean_t.shape[0],-1,4).gather(1,cell[:,None,None].expand(-1,1,4)).squeeze(1); gain=base_err-((cs-ps).square()+0.25*(cb-pb).square().mean(-1)); aux=p['X_RGB'][b[:,j],idx[:,j]]; tir=p['X_TIR'][b[:,j],idx[:,j]]; feats.append(torch.cat([d['q'][b[:,j],idx[:,j],None],delta[:,j].norm(dim=-1,keepdim=True), (aux-tir).norm(dim=-1,keepdim=True),gain[:,None]],-1)); labels.append((gain>0).float()[:,None])
        feat=torch.cat(feats); y=torch.cat(labels).squeeze(-1); perm=torch.randperm(feat.shape[0],device='cuda'); split=int(feat.shape[0]*.7); tr,va=perm[:split],perm[split:]
    gate=nn.Sequential(nn.Linear(feat.shape[-1]-1,32),nn.GELU(),nn.Linear(32,1)).cuda(); opt=torch.optim.AdamW(gate.parameters(),lr=2e-3)
    x=feat[:,:-1];
    for _ in range(args.steps):
        logit=gate(x[tr]).squeeze(-1); loss=F.binary_cross_entropy_with_logits(logit,y[tr]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        prob=gate(x[va]).squeeze(-1).sigmoid(); order=torch.argsort(prob); yy=y[va].bool(); ranks=torch.empty(order.numel(),device='cuda',dtype=torch.float32); ranks[order]=torch.arange(1,order.numel()+1,device='cuda',dtype=torch.float32); n1=yy.sum().float(); n0=(~yy).sum().float(); auc=float(((ranks[yy].sum()-n1*(n1+1)/2)/(n1*n0)).detach());
        sweep=[]
        for th in (0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9):
            accept=prob>=th; gain=float(feat[va][accept,-1].mean()) if bool(accept.any()) else 0.; sweep.append({'threshold':th,'fraction':float(accept.float().mean()),'gain':gain})
        report={'probe':'repair_gate_train_v1','auc':auc,'positive_rate':float(y[va].mean()),'threshold_sweep':sweep,'features':int(x.shape[-1])}
    args.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
