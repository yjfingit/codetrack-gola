"""Causal token-impact diagnosis probe.

Labels are the decrease in frozen GOLA-head error obtained by replacing one corrupted token with
its clean counterpart. This tests the intended meaning of q directly and avoids supervising an
injector mask alone.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--save-checkpoint',type=Path,default=None); ap.add_argument('--steps',type=int,default=300); ap.add_argument('--lr',type=float,default=5e-4); ap.add_argument('--min-ratio',type=float,default=0.25); args=ap.parse_args()
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    from trackit.models import ModelImplSuggestions
    from trackit.models.methods.GOLA.builder import build_GOLA_model
    from tools.preflight_acceptance import load_stage_config
    from tools.verify_codetrack_initialization import real_batch
    root=ROOT; view=root/'_probe/codetrack_flow_validation/LasHeR_curated10'; names=(view/'trainingsetList.txt').read_text().splitlines()
    data=real_batch(view,names)
    cfg=load_stage_config(str(root/'config/GOLA/codetrack_s1/config.yaml')); cfg['model']['codetrack']['corruption_enabled']=False; cfg['model']['codetrack']['motion_enabled']=False; cfg['model']['codetrack']['memory_enabled']=False
    model=build_GOLA_model(cfg,ModelImplSuggestions()).cuda().float().eval(); model.load_state_dict(load_file(str(root/'weights/gola_b224.bin')),strict=False); ct=model.codetrack
    captured=[]; h=ct.register_forward_pre_hook(lambda _m,_i,kwargs: captured.append(kwargs['F_L'].detach().clone()),with_kwargs=True)
    with torch.no_grad():
        model.reset_sequence(); model(**data); clean_f=captured[-1]
        bad=dict(data); bad['x']=data['x'].clone(); bad['x'][:,3:,84:140,84:140]=0
        model.reset_sequence(); model(**bad); cor_f=captured[-1]
    h.remove(); clean=ct._split(clean_f)['X_TIR'].detach(); corrupted=ct._split(cor_f)['X_TIR'].detach()
    with torch.no_grad():
        target=model.head(clean); base=model.head(corrupted)
        target_vec=torch.cat([target['score_map'].flatten(1),target['boxes'].flatten(1)],dim=1)
        base_vec=torch.cat([base['score_map'].flatten(1),base['boxes'].flatten(1)],dim=1)
        base_err=(base_vec-target_vec).square().mean(dim=1)
        influence=[]
        for start in range(0,256,32):
            count=min(32,256-start); batch=corrupted.unsqueeze(1).expand(-1,count,-1,-1).reshape(-1,256,768).clone()
            idx=torch.arange(start,start+count,device='cuda').repeat(clean.shape[0]); rows=torch.arange(clean.shape[0],device='cuda').repeat_interleave(count)
            batch[torch.arange(batch.shape[0],device='cuda'),idx]=clean[rows,idx]
            pred=model.head(batch); pv=torch.cat([pred['score_map'].flatten(1),pred['boxes'].flatten(1)],dim=1)
            err=(pv-target_vec[rows]).square().mean(dim=1).reshape(clean.shape[0],count)
            influence.append(base_err[:,None]-err)
        impact=torch.cat(influence,dim=1).clamp_min(0)
    # Train only diagnosis on causal soft labels; H and SATR are frozen.
    for p in model.parameters(): p.requires_grad_(False)
    for p in ct.diagnosis.parameters(): p.requires_grad_(True)
    opt=torch.optim.AdamW(ct.diagnosis.parameters(),lr=args.lr,weight_decay=1e-5); H=ct.H.matrix().detach()
    # Match the production target: only a material positive counterfactual gain is a bad
    # token.  Tiny numerical improvements on every token must not turn into a dense repair
    # target.
    soft = model._causal_token_impact_target(corrupted, clean, base, target,
                                              max_tokens=256, temperature=2.0,
                                              min_ratio=float(args.min_ratio)).detach()
    clean_parts=ct._split(clean_f); ctx=ct.template_pool(torch.cat([clean_parts['Z_RGB'],clean_parts['Z_TIR'],clean_parts['Z_on'],clean_parts['D_TIR']],dim=1).mean(1)).detach()
    history=[]
    for step in range(args.steps):
        c=ct._split(cor_f); out=ct.diagnosis(c['X_TIR'],c['X_RGB'],H,template_context=ctx); qlog=out['q_logits'].float()
        pos=soft>0.6; neg=soft<0.4
        loss=F.binary_cross_entropy_with_logits(qlog,soft)+0.5*F.relu(0.5-qlog[pos].mean()+qlog[neg].mean())
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(ct.diagnosis.parameters(),1.0); opt.step()
        if step%50==0 or step==args.steps-1:
            with torch.no_grad():
                q=ct.diagnosis(c['X_TIR'],c['X_RGB'],H,template_context=ctx)['q'];
                order=torch.argsort(q.reshape(-1)); y=impact.reshape(-1); ranks=torch.empty(order.numel(),device='cuda',dtype=torch.float32); ranks[order]=torch.arange(1,order.numel()+1,device='cuda',dtype=torch.float32); yy=y>y.quantile(.8); auc=float(((ranks[yy].sum()-yy.sum()*(yy.sum()+1)/2)/(yy.sum()*(~yy).sum())).detach())
                yt=soft.reshape(-1)>0; ranks_t=torch.empty_like(ranks); ranks_t[order]=torch.arange(1,order.numel()+1,device='cuda',dtype=torch.float32); auc_t=float(((ranks_t[yt].sum()-yt.sum()*(yt.sum()+1)/2)/(yt.sum()*(~yt).sum())).detach()) if bool(yt.any()) and bool((~yt).any()) else None
                history.append({'step':step,'loss':float(loss),'q_mean':float(q.mean()),'q_p95':float(q.quantile(.95)),'target_mean':float(soft.mean()),'target_positive_fraction':float(yt.float().mean()),'impact_auc':auc,'target_auc':auc_t})
    if args.save_checkpoint is not None:
        from safetensors.torch import save_file
        args.save_checkpoint.parent.mkdir(parents=True,exist_ok=True)
        save_file({k:v.detach().cpu() for k,v in ct.diagnosis.state_dict().items()},str(args.save_checkpoint))
    args.output.write_text(json.dumps({'probe':'causal_q_v1','history':history,'impact_mean':float(impact.mean()),'impact_max':float(impact.max()),'sequences':names,'checkpoint':None if args.save_checkpoint is None else str(args.save_checkpoint)},indent=2)+'\n'); print(json.dumps(history[-1],indent=2))
if __name__=='__main__': main()
