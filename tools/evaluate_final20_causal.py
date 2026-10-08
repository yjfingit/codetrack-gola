#!/usr/bin/env python3
"""Sequence-isolated OPE and validation-only checkpoint selection for causal20.

All annotations after frame zero are used only by the offline metric collector.
Each sequence has a fresh CausalRuntime; a batch slot is never reused as an identity.
Only a checkpoint selected from completed validation records may enter test mode.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from safetensors.torch import load_file
from codetrack.causal_runtime import CausalRuntime
from codetrack.safety import box_iou
from tools.observed_tracker import ordered_images
from trackit.models import ModelImplSuggestions
from trackit.models.methods.GOLA.builder import build_GOLA_model
from trackit.data.components.result_collector.handler.one_pass_evaluation_compatible.ope_metrics import (
    compute_one_pass_evaluation_metrics, compute_OPE_metrics_mean)
from trackit.data.components.result_collector.handler.utils.compatibility import ExternalToolkitCompatibilityHelper


def encode(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    raise TypeError(type(value))


def make_baseline(config):
    model=build_GOLA_model(config,ModelImplSuggestions()).cuda().eval()
    state=load_file(str(ROOT/'weights/gola_b224.bin'))
    mismatch=model.load_state_dict(state,strict=False)
    params=dict(model.named_parameters())
    missing=[name for name in mismatch.missing_keys if name in params and params[name].requires_grad
             and not name.startswith('codetrack.')]
    if missing or mismatch.unexpected_keys:
        raise RuntimeError(f'baseline mapping failure: {missing}, {mismatch.unexpected_keys}')
    model.requires_grad_(False)
    return model


@torch.no_grad()
def evaluate_sequences(model,names,root,split,output,force_baseline=False,limit=0):
    model.eval(); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    records=[]
    for name in names:
        folder=Path(root)/split/name
        gt=np.loadtxt(folder/'init.txt',delimiter=',',ndmin=2)
        gt[:,2:]+=gt[:,:2]
        files=[sorted((folder/m).glob('*.jpg')) for m in ('visible','infrared')]
        if not len(gt)==len(files[0])==len(files[1]): raise RuntimeError(f'frame mismatch: {name}')
        count=min(len(gt),limit) if limit else len(gt)
        runtime=CausalRuntime(model)
        predictions=[]; times=[]; actions=[]; baseline_boxes=[]
        for frame,cpu_image in ordered_images(files,count,workers=4,prefetch=8):
            tick=time.perf_counter()
            image=cpu_image.to('cuda',non_blocking=True).float()
            if frame==0:
                runtime.initialize([image],gt[:1]); pred=gt[0].copy(); base=pred; action=0
            else:
                result=runtime.step([image],force_action=0 if force_baseline else None)
                pred=result.boxes[0]; base=result.baseline_boxes[0]; action=int(result.states[0])
            predictions.append(pred); baseline_boxes.append(base); actions.append(action)
            times.append(time.perf_counter()-tick)
        predictions=np.asarray(predictions)
        annotation=gt[:count]
        metrics,_=compute_one_pass_evaluation_metrics(
            'LasHeR',predictions,annotation,None,np.asarray(times),ExternalToolkitCompatibilityHelper())
        iou=box_iou(torch.tensor(predictions),torch.tensor(annotation)).numpy()
        base_iou=box_iou(torch.tensor(np.asarray(baseline_boxes)),torch.tensor(annotation)).numpy()
        actions=np.asarray(actions); healthy=(base_iou>=.5); valid=(annotation[:,2:]>annotation[:,:2]).all(1)
        written=(actions>0)&valid; committed=(actions==2)&valid
        row={'name':name,'frames':count,'PR':metrics.precision_score,'SR':metrics.success_score,
             'NPrecision':metrics.normalized_precision_score,'metrics':asdict(metrics),
             'written':int(written.sum()),'false_writes':int((written&(iou<base_iou-.01)).sum()),
             'committed':int(committed.sum()),'bad_admissions':int((committed&(iou<.5)).sum()),
             'healthy_frames':int(healthy.sum()),'healthy_noop':int((healthy&(actions==0)).sum()),
             'drift_sum':float(np.linalg.norm((predictions[:,:2]+predictions[:,2:])/2-
                              (annotation[:,:2]+annotation[:,2:])/2,axis=1)[valid].sum()),
             'valid_frames':int(valid.sum())}
        np.savez_compressed(output/f'{name}.npz',boxes_xyxy=predictions,
                            actions=actions,baseline_same_crop_boxes=np.asarray(baseline_boxes))
        (output/f'{name}.json').write_text(json.dumps(row,default=encode,indent=2)+'\n')
        print(json.dumps({k:row[k] for k in ('name','frames','PR','SR','NPrecision')}),flush=True)
        records.append(row)
        del runtime
    return records


def summarize(records):
    from trackit.data.components.result_collector.handler.one_pass_evaluation_compatible.ope_metrics import OPEMetrics
    metrics=[]
    for row in records:
        values=row['metrics'].copy()
        for k,v in values.items():
            if isinstance(v,list): values[k]=np.asarray(v)
        metrics.append(OPEMetrics(**values))
    mean=compute_OPE_metrics_mean(metrics)
    totals={key:sum(r[key] for r in records) for key in ('frames','written','false_writes',
        'committed','bad_admissions','healthy_frames','healthy_noop','drift_sum','valid_frames')}
    return {'sequences':len(records),'PR':mean.precision_score,'SR':mean.success_score,
            'NPrecision':mean.normalized_precision_score,**totals,
            'false_write_rate':totals['false_writes']/max(1,totals['written']),
            'admission_contamination_rate':totals['bad_admissions']/max(1,totals['committed']),
            'clean_no_write_rate':totals['healthy_noop']/max(1,totals['healthy_frames']),
            'acceptance_rate':totals['written']/max(1,totals['frames']-len(records)),
            'mean_center_drift_px':totals['drift_sum']/max(1,totals['valid_frames']),
            'risk_definition':'GT comparison to same-crop baseline; separate-arm OPE for tracking accuracy'}


def evaluate_validation_rank(model,config,manifest,output,epoch,rank):
    names=Path(manifest['val_list']).read_text().splitlines()
    import torch.distributed as dist
    world_size=dist.get_world_size()
    assigned=names[rank::world_size]
    directory=Path(output)/f'validation_{epoch:02d}'
    candidate=evaluate_sequences(model,assigned,manifest['dataset_root'],'trainingset',directory/f'rank_{rank}/candidate')
    base_dir=Path(output)/'baseline_validation'/f'rank_{rank}'
    if not (base_dir/'records.json').exists():
        baseline=make_baseline(config)
        records=evaluate_sequences(baseline,assigned,manifest['dataset_root'],'trainingset',base_dir,True)
        (base_dir/'records.json').write_text(json.dumps(records,default=encode)+'\n')
        del baseline
    (directory/f'rank_{rank}/records.json').write_text(json.dumps(candidate,default=encode)+'\n')
    import torch.distributed as dist
    dist.barrier()
    if rank==0:
        collect=lambda path:sum([json.loads((path/f'rank_{r}/records.json').read_text())
                                 for r in range(world_size)],[])
        candidates=collect(directory); bases=collect(Path(output)/'baseline_validation')
        if {r['name'] for r in candidates}!=set(names) or len(candidates)!=len(names):
            raise RuntimeError('incomplete or duplicate validation coverage')
        candidate_summary=summarize(candidates); baseline_summary=summarize(bases)
        summary={'epoch':epoch,'candidate':candidate_summary,'baseline':baseline_summary}
        (directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        select_checkpoint(output)


def select_checkpoint(output):
    output=Path(output); records=[]
    for file in output.glob('validation_*/summary.json'):
        record=json.loads(file.read_text()); c=record['candidate']; b=record['baseline']
        if c['PR']>=b['PR'] and c['SR']>=b['SR'] and c['false_write_rate']<=.1:
            records.append(record)
    if not records:
        (output/'selection.json').write_text(json.dumps({'status':'NO_ELIGIBLE_CHECKPOINT'})+'\n')
        return
    best=max(records,key=lambda r:(r['candidate']['SR']+r['candidate']['PR'],
                                   -r['candidate']['false_write_rate'],r['candidate']['clean_no_write_rate']))
    weight=output/f"epoch_{best['epoch']:02d}/model.safetensors"
    selection={'status':'VALIDATION_SELECTED','epoch':best['epoch'],'checkpoint':str(weight.resolve()),
               'sha256':hashlib.sha256(weight.read_bytes()).hexdigest(),'validation':best}
    (output/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--arm',choices=('baseline','candidate'),required=True)
    ap.add_argument('--shard',type=int,default=0); ap.add_argument('--shards',type=int,default=1)
    args=ap.parse_args()
    if args.shards<1 or not 0<=args.shard<args.shards:
        raise ValueError('invalid evaluation shard index/count')
    selection=json.loads((args.run/'selection.json').read_text())
    if selection['status']!='VALIDATION_SELECTED': raise RuntimeError('no validation-selected checkpoint')
    protocol=json.loads((args.run/'protocol.json').read_text())
    if protocol['smoke']:
        raise RuntimeError('a smoke artifact cannot be promoted to final test')
    if len(list(args.run.glob('epoch_*/state.pth')))!=20:
        raise RuntimeError('test requires 20 complete resumable epochs')
    cfg=json.loads((args.run/'config.json').read_text())
    manifest=json.loads((args.run/'manifest.json').read_text())
    weight=Path(selection['checkpoint'])
    if hashlib.sha256(weight.read_bytes()).hexdigest()!=selection['sha256']:
        raise RuntimeError('selected checkpoint changed')
    names=(Path(manifest['dataset_root'])/'testingsetList.txt').read_text().splitlines()
    model=make_baseline(cfg)
    if args.arm=='candidate':
        model.load_state_dict(load_file(str(weight)),strict=True)
    assigned=names[args.shard::args.shards]
    output=args.output/f'{args.arm}_shard_{args.shard}'
    if output.exists() and (output/'summary.json').exists():
        raise RuntimeError('test shard already evaluated; refusing an accidental repeat')
    records=evaluate_sequences(model,assigned,manifest['dataset_root'],'testingset',output,args.arm=='baseline')
    summary=summarize(records)
    (output/'records.json').write_text(json.dumps(records,default=encode)+'\n')
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


def compare_test_shards(run,output,shards=2):
    """Require complete matched coverage before emitting the success verdict."""
    run,output=Path(run),Path(output)
    manifest=json.loads((run/'manifest.json').read_text())
    names=(Path(manifest['dataset_root'])/'testingsetList.txt').read_text().splitlines()
    arms={}
    for arm in ('baseline','candidate'):
        records=sum([json.loads((output/f'{arm}_shard_{i}/records.json').read_text())
                     for i in range(shards)],[])
        if len(records)!=len(names) or {r['name'] for r in records}!=set(names):
            raise RuntimeError(f'incomplete/duplicate {arm} test coverage')
        arms[arm]={'summary':summarize(records),'sequences':{r['name']:r for r in records}}
    b,c=arms['baseline']['summary'],arms['candidate']['summary']
    deltas=[{'name':name,'PR_delta_pp':100*(arms['candidate']['sequences'][name]['PR']-
                                          arms['baseline']['sequences'][name]['PR']),
             'SR_delta_pp':100*(arms['candidate']['sequences'][name]['SR']-
                               arms['baseline']['sequences'][name]['SR'])} for name in names]
    passed=c['PR']>b['PR'] and c['SR']>b['SR']
    passed=passed and c['NPrecision']>=b['NPrecision']-.01 and min(d['SR_delta_pp'] for d in deltas)>=-20.
    report={'baseline':b,'candidate':c,'PR_delta_pp':100*(c['PR']-b['PR']),
            'SR_delta_pp':100*(c['SR']-b['SR']),'success':passed,
            'limits':{'NPrecision_max_drop_pp':1.,'per_sequence_SR_max_drop_pp':20.},
            'per_sequence':deltas}
    (output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__': main()
