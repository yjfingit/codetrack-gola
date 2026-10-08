#!/usr/bin/env python3
"""Behavioral witnesses for the corrected curriculum, using LasHeR-train only."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from codetrack.causal_runtime import CausalRuntime, four_frame_utility, targets_for
from codetrack.motion import TemporalMemory
from codetrack.safety import choose_state
from tools.evaluate_final20_causal import make_baseline, evaluate_sequences, summarize
from tools.observed_tracker import ObservedGOLATracker, ordered_images
from trackit.core.boot.funcs.utils.custom_yaml_loader import load_yaml
from trackit.models import ModelImplSuggestions
from trackit.models.methods.GOLA.builder import build_GOLA_model
from safetensors.torch import load_file


def tree_equal(a,b):
    if torch.is_tensor(a):
        return torch.equal(a,b) or (a.shape==b.shape and bool(torch.all((a==b)|(torch.isnan(a)&torch.isnan(b)))))
    if isinstance(a,np.ndarray): return np.array_equal(a,b,equal_nan=True)
    if isinstance(a,dict): return a.keys()==b.keys() and all(tree_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(tree_equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True)
    args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2); torch.manual_seed(42)
    cfg=load_yaml(str(ROOT/'config/GOLA/codetrack_causal20/config.yaml'))
    manifest=json.loads((ROOT/'outputs/final20/manifests/manifest.json').read_text())
    name=Path(manifest['val_list']).read_text().splitlines()[0]
    root=Path(manifest['dataset_root']); folder=root/'trainingset'/name
    gt=np.loadtxt(folder/'init.txt',delimiter=',',ndmin=2); gt[:,2:]+=gt[:,:2]
    files=[sorted((folder/m).glob('*.jpg')) for m in ('visible','infrared')]
    images=[image.cuda().float() for _,image in ordered_images(files,16,workers=2,prefetch=4)]
    baseline=make_baseline(cfg)
    original=ObservedGOLATracker(ROOT,amp=False)
    runtime=CausalRuntime(baseline)
    with torch.no_grad():
        runtime.initialize(images[:1],gt[:1]); original.initialize(images[0],gt[0])
        expected=[]; actual=[]
        for image in images[1:]:
            actual.append(runtime.step([image],force_action=0).boxes[0])
            expected.append(original.track(image)['box'])
        exact=np.array_equal(np.asarray(actual),np.asarray(expected))
        difference=float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
        if not exact: raise AssertionError(f'no-op deviated from training-class GOLA: {difference}')
        original.end_sequence()
        runtime.initialize(images[:1],gt[:1]); before=runtime.snapshot()
        runtime.step(images[1:2],force_action=0)
        live=runtime.snapshot(); cpu_rng=torch.get_rng_state().clone(); cuda_rng=torch.cuda.get_rng_state_all()
        labels=four_frame_utility(runtime,[[im] for im in images[1:5]],
                                  [gt[i:i+1] for i in range(1,5)],before)
        if not tree_equal(live,runtime.snapshot()): raise AssertionError('counterfactual changed live state')
        if not tree_equal(cpu_rng,torch.get_rng_state()) or not tree_equal(cuda_rng,torch.cuda.get_rng_state_all()):
            raise AssertionError('counterfactual consumed live RNG')
        # Invalid current GT affects loss labels but never the image-only prediction.
        runtime.restore(before); first=runtime.step(images[1:2],force_action=0).boxes.copy()
        targets_for(gt[1:2]+300.,runtime.snapshot()['code']['code']['_motion_crop_params'].cpu().numpy(),runtime.device)
        runtime.restore(before); second=runtime.step(images[1:2],force_action=0).boxes.copy()
        if not np.array_equal(first,second): raise AssertionError('current label leaked into prediction')
    mem=TemporalMemory(dim=8,frames=3,tokens=2,memory_dim=4)
    early=torch.randn(1,4,8,requires_grad=True); later=torch.randn(1,4,8,requires_grad=True)
    a=mem(early,torch.ones(1,4),detach_memory=False)
    b=mem(later,torch.ones(1,4),a['memory'],a['memory_reliability'],detach_memory=False)
    b['prior_tokens'].square().sum().backward()
    if early.grad is None or not bool(early.grad.abs().sum()>0): raise AssertionError('no cross-frame memory gradient')
    q={'state_logits':torch.tensor([[-2.,-1.,3.]]),'utility':torch.tensor([[1.,1.]]),
       'motion_logit':torch.tensor([2.]),'admission_logit':torch.tensor([2.])}
    box=torch.tensor([[0.,0.,10.,10.]]); score=torch.tensor([.9])
    if choose_state(q,None,box,box,score,score).item()!=1: raise AssertionError('unconfirmed commit')
    if choose_state(q,box,box,box,score,score).item()!=2: raise AssertionError('confirmation cannot commit')
    if choose_state(q,box+100.,box,box,score,score).item()!=1: raise AssertionError('inconsistent confirmation committed')
    checkpoint=load_file(str(args.checkpoint))
    loaded=build_GOLA_model(cfg,ModelImplSuggestions())
    loaded.load_state_dict(checkpoint,strict=True)
    initial=build_GOLA_model(cfg,ModelImplSuggestions())
    frozen=[n for n,p in initial.named_parameters() if not p.requires_grad]
    if not all(torch.equal(dict(initial.named_parameters())[n],dict(loaded.named_parameters())[n]) for n in frozen):
        raise AssertionError('DINOv2 frozen weights changed')
    # Real official metric collector on a bounded validation prefix, not LasHeR-test.
    rows=evaluate_sequences(baseline,[name],root,'trainingset',args.output/'val_prefix',True,limit=16)
    metrics=summarize(rows)
    report={'sequence':name,'frames':16,'no_op_exact_baseline':exact,'max_box_difference':difference,
            'counterfactual_restores_state_and_rng':True,'current_gt_only_used_after_prediction':True,
            'memory_cross_frame_gradient':float(early.grad.abs().sum()),'confirmation_policy':True,
            'frozen_backbone_exact':True,'checkpoint_strict_reload':True,
            'four_frame_utility_finite':all(bool(torch.isfinite(v).all()) for v in labels),
            'validation_prefix_OPE':metrics,'scope':'behavioral and integration witnesses, not final tracking performance'}
    if not report['four_frame_utility_finite']: raise AssertionError('nonfinite utility')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
