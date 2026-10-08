#!/usr/bin/env python3
"""Explicit two-GPU 20-epoch causal curriculum with auditable frame coverage.

This is a separate entry point from the completed pair-only run. --smoke executes
real optimizer updates on training sequences and saves them under a smoke directory.
Full launches refuse an existing nonempty run directory; resume is explicit.
"""
import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file, save_file

from codetrack.causal_runtime import CausalRuntime, targets_for, four_frame_utility
from codetrack.criteria import CodeTrackCriteria, _tracking_loss
from codetrack.safety import box_iou
from codetrack.frame_coverage import read_metadata, build_plan, shard_plan, indices_for
from trackit.core.boot.funcs.utils.custom_yaml_loader import load_yaml
from trackit.models import ModelImplSuggestions
from trackit.models.methods.GOLA.builder import build_GOLA_model
from trackit.models.methods.GOLA.modules.gola import compute_rand_pair_orth_loss
from tools.observed_tracker import decode_pair


def curriculum(epoch):
    if epoch < 4:
        return {'phase':'spatial', 'length':1, 'scheduled_sampling':1., 'utility_weight':.1}
    if epoch < 12:
        return {'phase':'causal8', 'length':8,
                'scheduled_sampling':.25 + .5*(epoch-4)/7, 'utility_weight':.2}
    return {'phase':'causal16', 'length':16,
            'scheduled_sampling':.75 + .25*(epoch-12)/7, 'utility_weight':.2}


def should_validate(epoch):
    """Epoch indices are zero-based; validate after completed rounds 4/8/12/16/20."""
    return (epoch + 1) % 4 == 0


class ClipDataset(Dataset):
    def __init__(self, root, names, epoch, rank, world_size, repeats, seed):
        self.root = Path(root); self.names=names; self.epoch=epoch; self.seed=seed
        self.length = curriculum(epoch)['length']
        # Each sequence is visited ``repeats`` times before sharding/padding. Padding
        # duplicates are documented and only equalize DDP batch counts.
        rng = np.random.default_rng(seed+epoch)
        visits = rng.permutation(np.tile(np.arange(len(names)), repeats)).tolist()
        size = math.ceil(len(visits)/world_size)*world_size
        self.padding = size-len(visits)
        visits += visits[:self.padding]
        self.visits=visits[rank::world_size]

    def __len__(self): return len(self.visits)

    def __getitem__(self, index):
        name=self.names[self.visits[index]]
        path=self.root/'trainingset'/name
        annotations=np.loadtxt(path/'init.txt', delimiter=',', ndmin=2)
        files=[sorted((path/m).glob('*.jpg')) for m in ('visible','infrared')]
        if len(annotations)!=len(files[0]) or len(files[0])!=len(files[1]):
            raise ValueError(f'frame/annotation count mismatch: {name}')
        rng=np.random.default_rng(self.seed + self.epoch*1000003 + index*53 + self.visits[index])
        valid=np.isfinite(annotations).all(1)&(annotations[:,2:]>0).all(1)
        if self.length==1:
            starts=np.flatnonzero(valid[:-1])
            if not len(starts): raise ValueError(f'no valid initialization: {name}')
            first=int(rng.choice(starts))
            last=min(len(annotations)-1,first+int(rng.integers(1,101)))
            indices=[first,last]
        else:
            required=1+self.length+3
            if len(annotations)<required:
                raise ValueError(f'sequence {name} too short for {required} ordered frames')
            starts=np.flatnonzero(valid[:len(annotations)-required+1])
            if not len(starts): raise ValueError(f'no valid clip initialization: {name}')
            first=int(rng.choice(starts))
            indices=list(range(first,first+required))
        boxes=annotations[indices].copy(); boxes[:,2:]+=boxes[:,:2]
        if not ((boxes[0,2:]-boxes[0,:2])>0).all():
            raise ValueError(f'invalid initialization box: {name}/{first}')
        images=[decode_pair([fs[i] for fs in files],pin_memory=False) for i in indices]
        return augment_clip(images,boxes,rng,{'name':name,'indices':indices})


def augment_clip(images,boxes,rng,metadata):
    # One geometric/color augmentation draw shared across the entire sequence.
    # Flip annotations only for training loss/initialization; no label drives a step.
    if rng.random()<.5:
        for j,image in enumerate(images):
            width=image.shape[-1]
            images[j]=image.flip(-1)
            boxes[j,[0,2]]=width-boxes[j,[2,0]]
    gain=float(rng.uniform(.7,1.3))
    images=[image.float().mul(gain).clamp(0,255) for image in images]
    return {**metadata,'images':images,'boxes':boxes}


class CoverageClipDataset(Dataset):
    """A finite epoch with explicit targets; every search frame is scheduled."""
    def __init__(self, metadata, entries, report, epoch, rank, world_size, local_batch, seed=42):
        self.metadata=metadata; self.entries=entries; self.report=report
        self.epoch=epoch; self.seed=seed
        self.ids,self.padding=shard_plan(entries,epoch,rank,world_size,local_batch,seed)

    def __len__(self): return len(self.ids)

    def __getitem__(self,index):
        plan_id=int(self.ids[index]); entry=self.entries[plan_id]
        sequence=self.metadata[int(entry[0])]
        indices,future_available=indices_for(entry,len(sequence.boxes_xywh))
        boxes=sequence.boxes_xywh[indices].copy();boxes[:,2:]+=boxes[:,:2]
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,plan_id]))
        images=[decode_pair([sequence.visible[i],sequence.infrared[i]],pin_memory=False) for i in indices]
        return augment_clip(images,boxes,rng,{'name':sequence.name,'indices':indices,
                            'plan_id':plan_id,'future_available':future_available})


def identity_collate(items): return items


class ClipObjective(nn.Module):
    def __init__(self, model):
        super().__init__(); self.model=model
        self.criterion=CodeTrackCriteria(lambda_rec=.2,lambda_pres=.1,lambda_motion=.1)

    def forward(self, clips, start, sampling, utility_weight):
        self.observation_counts={'search_presentations':0,'valid_search_presentations':0,
                                 'valid_without_positive_grid_cells':0,'full_future_utility_targets':0}
        runtime=CausalRuntime(self.model)
        if start==0:
            runtime.initialize([c['images'][0] for c in clips], [c['boxes'][0] for c in clips])
            previous_prediction=None
        else:
            runtime.restore(self.live_state)
            previous_prediction=self.previous_prediction
        length=curriculum(self.epoch)['length']
        total=next(self.model.parameters()).new_zeros(())
        facts=[]
        for t in range(start,min(start+4,length)):
            previous=None
            if t>0 and np.random.random()>sampling:
                previous=np.stack([c['boxes'][t] for c in clips])
                observed_valid=np.isfinite(previous).all(1)&(previous[:,2:]>previous[:,:2]).all(1)
                runtime.crop_boxes=np.where(observed_valid[:,None],previous,runtime.crop_boxes)
            # Both counterfactual arms start from the exact same training state,
            # including any observation of the previous frame chosen by the curriculum.
            before=runtime.snapshot(detach=True)
            images=[c['images'][t+1] for c in clips]
            result=runtime.step(images)
            annotation=np.stack([c['boxes'][t+1] for c in clips])
            target=targets_for(annotation,result.crop_params,runtime.device)
            self.observation_counts['search_presentations']+=len(clips)
            self.observation_counts['valid_search_presentations']+=int(target['valid'].sum())
            self.observation_counts['valid_without_positive_grid_cells']+=int((
                target['valid']&~target['score_quality_map'].flatten(1).bool().any(1)).sum())
            # The motion target expects centers/extents; tracking boxes are XYXY.
            gt=target['boxes']; lin=(torch.arange(16,device=runtime.device)+.5)/16
            yy,xx=torch.meshgrid(lin,lin,indexing='ij')
            centers=(gt[:,:2]+gt[:,2:])*.5
            wh=(gt[:,2:]-gt[:,:2]).clamp_min(.02)
            grid=torch.stack((xx,yy),-1)[None]
            motion_target=torch.exp(-.5*((grid-centers[:,None,None])/wh[:,None,None]).square().sum(-1)).flatten(1)
            motion_target=motion_target/motion_target.sum(1,keepdim=True).clamp_min(1e-6)
            motion_target=torch.where(target['valid'][:,None],motion_target,
                                      torch.full_like(motion_target,1./256))
            result.outputs['codetrack_extras']['motion_target']=motion_target
            criterion=self.criterion(result.outputs,target)
            baseline_loss=_tracking_loss(result.baseline,target)[0]
            candidate_loss=_tracking_loss(result.candidate,target)[0]
            if length==1:
                instant=(_tracking_loss(result.baseline,target,per_sample=True)[0]-
                         _tracking_loss(result.candidate,target,per_sample=True)[0]).detach()
                utility=instant; consistent=torch.ones_like(instant)
                full_future=torch.zeros_like(instant,dtype=torch.bool)
            else:
                future_images=[[c['images'][t+1+k] for c in clips] for k in range(4)]
                future_labels=[np.stack([c['boxes'][t+1+k] for c in clips]) for k in range(4)]
                future_mask=torch.as_tensor(np.stack([
                    c.get('future_available',np.ones(length+3,dtype=bool))[t:t+4]
                    for c in clips],axis=1),device=runtime.device)
                full_future=future_mask.all(0)
                instant,utility,consistent=four_frame_utility(
                    runtime,future_images,future_labels,before,future_available=future_mask)
            self.observation_counts['full_future_utility_targets']+=int(full_future.sum())
            pred_box=torch.as_tensor(result.candidate_boxes,device=runtime.device)
            gt_box=torch.as_tensor(annotation,device=runtime.device)
            safe=box_iou(pred_box,gt_box)>=.5
            useful=(instant>1e-4)&(utility>1e-4)&safe
            # Immediate useful edit is provisional; a positive future result and current
            # observable consistency permits a commit label. Spatial phase keeps no-op.
            label=torch.where(useful,torch.ones_like(instant,dtype=torch.long),
                              torch.zeros_like(instant,dtype=torch.long))
            if length>1:
                label=torch.where(useful & full_future & (consistent>=.5) & (t>0),torch.full_like(label,2),label)
            else:
                label.zero_()
            q=result.quality
            # Near a video's end a four-frame outcome does not exist. Fit immediate
            # utility, but mask the future head instead of inventing repeated-frame labels.
            utility_fit=nn.functional.smooth_l1_loss(
                q['utility'],torch.stack((instant,utility),1),reduction='none')
            utility_mask=torch.stack((torch.ones_like(full_future),full_future),1)
            quality_loss=(utility_fit*utility_mask).sum()/utility_mask.sum().clamp_min(1)
            quality_loss+=nn.functional.cross_entropy(q['state_logits'],label)
            motion_fit=nn.functional.binary_cross_entropy_with_logits(
                q['motion_logit'],(consistent>=.5).float(),reduction='none')
            motion_mask=full_future if length>1 else torch.ones_like(full_future)
            quality_loss+=(motion_fit*motion_mask).sum()/motion_mask.sum().clamp_min(1)
            quality_loss+=nn.functional.binary_cross_entropy_with_logits(q['admission_logit'],(useful&full_future).float())
            # Penalize clean/no-op rewrites, using received tokens as the identity anchor.
            extras=result.outputs['codetrack_extras']
            preservation=(extras['recovered']-extras['input_tokens']).square().mean()
            loss=criterion.loss + .5*baseline_loss + .5*candidate_loss
            loss+=utility_weight*quality_loss + .1*preservation
            # Fixed Hann-selected localization remains differentiable through gathered
            # corners. Compare frame-to-frame motion with GT displacement in full-image
            # coordinates; annotations remain a loss target, not a crop/provider input.
            sm=result.outputs['score_map'].detach().sigmoid().flatten(1)
            idx=(.55*sm+.45*runtime.post._window[None]).argmax(1)
            corners=result.outputs['boxes'].reshape(len(clips),-1,4)[
                torch.arange(len(clips),device=runtime.device),idx]*224.
            params=torch.as_tensor(result.crop_params,device=runtime.device,dtype=corners.dtype)
            world=((corners.reshape(-1,2,2)-params[:,1,None,:])/params[:,0,None,:]).flatten(1)
            if previous_prediction is not None:
                gt_previous=torch.as_tensor(np.stack([c['boxes'][t] for c in clips]),
                                            device=runtime.device,dtype=world.dtype)
                current_gt=torch.as_tensor(annotation,device=runtime.device,dtype=world.dtype)
                size=world.new_tensor([[im.shape[-1],im.shape[-2]] for im in images]).repeat(1,2)
                valid_temporal=target['valid']&torch.isfinite(gt_previous).all(1)&(
                    gt_previous[:,2:]>gt_previous[:,:2]).all(1)
                displacement=torch.nan_to_num(current_gt-gt_previous)
                temporal=nn.functional.smooth_l1_loss(
                    (world-previous_prediction)/size,displacement/size,reduction='none').mean(1)
                loss+=.1*(temporal*valid_temporal).sum()/valid_temporal.sum().clamp_min(1)
            previous_prediction=world
            total+=loss
            facts.append(float(loss.detach()))
        total/=len(facts)
        total+=compute_rand_pair_orth_loss(self.model,weight=1.4e-3,n_groups=8)
        # Preserve temporal graph within this four-frame chunk and detach exactly at
        # the chunk boundary; DDP sees the complete chunk as ONE forward/backward.
        self.live_state=runtime.snapshot(detach=True)
        self.previous_prediction=previous_prediction.detach()
        return total


def parameter_groups(model):
    groups={}; norm_ids={id(p) for m in model.modules() if isinstance(m,nn.LayerNorm)
                        for p in m.parameters(recurse=False)}
    for name,p in model.named_parameters():
        backbone=(name.startswith(('patch_embed.','pos_embed','norm.')) or
                  (name.startswith('blocks.') and '.lora.' not in name))
        if backbone and p.requires_grad:
            raise RuntimeError(f'DINOv2 unexpectedly trainable: {name}')
        if not backbone and not p.requires_grad:
            raise RuntimeError(f'non-backbone unexpectedly frozen: {name}')
        if not p.requires_grad: continue
        lr=7e-6 if '.lora.' in name or name=='token_type_embed' else (3e-6 if name.startswith('head.') else 2e-5)
        wd=0. if p.ndim<2 or id(p) in norm_ids or 'token_type_embed' in name else .1
        groups.setdefault((lr,wd),[]).append(p)
    return [{'params':p,'lr':lr,'weight_decay':wd} for (lr,wd),p in groups.items()]


def save_epoch(output,model,optimizer,scheduler,epoch,manifest,config,world_size):
    rank=dist.get_rank()
    rng={'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),
         'numpy':np.random.get_state(),'python':random.getstate()}
    states=[None]*world_size
    dist.all_gather_object(states,rng)
    if rank==0:
        folder=output/f'epoch_{epoch:02d}'; folder.mkdir()
        # Save ALL tensors, including frozen backbone and deterministic geometry buffers.
        # Clone aliases so the shared head is representable in safetensors.
        weights={k:v.detach().cpu().contiguous().clone()
                 for k,v in nn.Module.state_dict(model).items()}
        save_file(weights,str(folder/'model.safetensors'))
        torch.save({'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                    'epoch':epoch,'rng_by_rank':states,'manifest':manifest,'config':config,
                    'world_size':world_size},folder/'state.pth')
        (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps({'checkpoint':str(folder),'epoch':epoch}),flush=True)


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',type=Path,default=ROOT/'config/GOLA/codetrack_causal20/config.yaml')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--resume',type=Path)
    ap.add_argument('--local-clips',type=int,default=4)
    ap.add_argument('--repeats',type=int,default=16,help='legacy random sampler only')
    ap.add_argument('--sampling',choices=('random','frame_coverage'),default='frame_coverage')
    ap.add_argument('--plan-only',action='store_true',help='write coverage plans without CUDA or training')
    ap.add_argument('--smoke-tail',action='store_true',help='smoke only: exercise final video windows and missing future labels')
    args=ap.parse_args(argv)
    if args.smoke_tail and (not args.smoke or args.sampling!='frame_coverage'):
        ap.error('--smoke-tail requires --smoke and --sampling frame_coverage')
    if args.plan_only:
        if args.sampling!='frame_coverage': ap.error('--plan-only requires --sampling frame_coverage')
        manifest=json.loads((ROOT/'outputs/final20/manifests/manifest.json').read_text())
        names=Path(manifest['train_list']).read_text().splitlines()
        metadata=read_metadata(manifest['dataset_root'],names)
        if args.output.exists(): raise RuntimeError('plan output already exists')
        args.output.mkdir(parents=True)
        for length in (1,8,16):
            entries,report=build_plan(metadata,length)
            np.savez_compressed(args.output/f'plan_length_{length}.npz',entries=entries)
            (args.output/f'coverage_length_{length}.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({k:v for k,v in report.items() if k!='per_sequence'}),flush=True)
        return
    expected_devices=os.environ.get('CAUSAL_GPU_SET','3,4')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=expected_devices:
        raise RuntimeError(f'CUDA_VISIBLE_DEVICES must be exactly {expected_devices}')
    world_size=int(os.environ.get('WORLD_SIZE','1'))
    if world_size!=2:
        raise RuntimeError('exactly two DDP ranks required; launch with torchrun --nproc_per_node=2')
    rank=int(os.environ['RANK']); local=int(os.environ['LOCAL_RANK'])
    # Full-sequence validation shards can finish hours apart on long videos.
    # The default ten-minute NCCL timeout would abort healthy waiting ranks.
    torch.cuda.set_device(local)
    torch.set_float32_matmul_precision('high')
    dist.init_process_group('nccl', timeout=timedelta(hours=24),
                            device_id=torch.device('cuda', local))
    torch.set_num_threads(2)
    torch.manual_seed(42); np.random.seed(42+rank); random.seed(42+rank)
    args.output=args.output.resolve()
    if rank==0:
        if args.output.exists() and any(args.output.iterdir()) and not args.resume:
            raise RuntimeError('nonempty output directory; use an isolated directory or explicit --resume')
        args.output.mkdir(parents=True,exist_ok=True)
    dist.barrier()
    cfg=load_yaml(str(args.config))
    cfg['causal_training']={'local_clips':args.local_clips,'repeats':args.repeats,
                            'local_spatial_pairs':args.local_clips*4,
                            'world_size':world_size,'epochs':20,'tbptt':4,'seed':42}
    if args.sampling=='frame_coverage':
        cfg['causal_training'].update(sampling='frame_coverage_v1',repeats=None,
                                     tail_utility='masked_when_four_future_frames_unavailable')
    manifest_path=ROOT/'outputs/final20/manifests/manifest.json'
    manifest=json.loads(manifest_path.read_text())
    names=Path(manifest['train_list']).read_text().splitlines()
    val=Path(manifest['val_list']).read_text().splitlines()
    if set(names)&set(val): raise RuntimeError('train/val overlap')
    for key,seqs in (('train',names),('val',val)):
        actual=hashlib.sha256(('\n'.join(seqs)+'\n').encode()).hexdigest()
        if actual!=manifest[key+'_sha256']: raise RuntimeError('manifest hash mismatch')
    metadata=plans=None
    if args.sampling=='frame_coverage':
        metadata=read_metadata(manifest['dataset_root'],names)
        plans={length:build_plan(metadata,length) for length in (1,8,16)}
        cfg['causal_training']['plan_sha256']={str(length):value[1]['plan_sha256']
                                             for length,value in plans.items()}
        if rank==0:
            directory=args.output/'coverage';directory.mkdir(exist_ok=True)
            for length,(entries,report) in plans.items():
                np.savez_compressed(directory/f'plan_length_{length}.npz',entries=entries)
                (directory/f'planned_length_{length}.json').write_text(json.dumps(report,indent=2)+'\n')
                print(json.dumps({'coverage_plan':{k:v for k,v in report.items() if k!='per_sequence'}}),flush=True)
    model=build_GOLA_model(cfg,ModelImplSuggestions()).cuda()
    init=load_file(str(ROOT/'weights/gola_b224.bin'))
    mismatch=model.load_state_dict(init,strict=False)
    parameters=dict(model.named_parameters())
    missing=[k for k in mismatch.missing_keys if k in parameters and parameters[k].requires_grad
             and not k.startswith('codetrack.')]
    if missing or mismatch.unexpected_keys: raise RuntimeError(f'invalid baseline mapping: {missing}, {mismatch.unexpected_keys}')
    groups=parameter_groups(model)
    optimizer=torch.optim.AdamW(groups)
    updates_per_epoch=[]
    for epoch in range(20):
        phase_batch=args.local_clips*(4 if epoch<4 else 1)
        samples=(len(plans[curriculum(epoch)['length']][0]) if plans else len(names)*args.repeats)
        clips=math.ceil(samples/(world_size*phase_batch))
        updates_per_epoch.append(clips*math.ceil(curriculum(epoch)['length']/4))
    total=sum(updates_per_epoch); warmup=math.ceil(total*.05)
    def multiplier(step):
        if step<warmup: return max(.001,step/warmup)
        return .05+.95*.5*(1+math.cos(math.pi*min(1.,(step-warmup)/(total-warmup))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,multiplier)
    start=0
    if args.resume:
        state=torch.load(args.resume/'state.pth',map_location='cpu',weights_only=False)
        if state['manifest']!=manifest or state['world_size']!=world_size or state['config']!=cfg:
            raise RuntimeError('incompatible resume: config, batch, split or world size changed')
        model.load_state_dict(load_file(str(args.resume/'model.safetensors')),strict=True)
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        r=state['rng_by_rank'][rank]
        torch.set_rng_state(r['torch']); torch.cuda.set_rng_state_all(r['cuda'])
        np.random.set_state(r['numpy']); random.setstate(r['python']); start=state['epoch']+1
        print(json.dumps({'rank':rank,'resumed_from':str(args.resume),
                          'start_epoch':start,'scheduler_step':scheduler.last_epoch}),flush=True)
    objective=ClipObjective(model).cuda()
    # The smoke gradient witness showed every trainable scope participates; the
    # full run therefore uses a fixed DDP graph without per-step unused traversal.
    ddp=nn.parallel.DistributedDataParallel(objective,device_ids=[local],
                                            find_unused_parameters=False,static_graph=True)
    print(json.dumps({'rank':rank,'local_rank':local,'gpu_name':torch.cuda.get_device_name(local),
                      'local_clips':args.local_clips,'tbptt':4,'world_size':world_size,
                      'steps':total,'warmup':warmup}),flush=True)
    if rank==0:
        (args.output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        protocol={'seed':42,'epochs':20,'local_clips':args.local_clips,'world_size':world_size,
                  'repeats':args.repeats,'tbptt':4,'smoke':args.smoke,
                  'validation_interval_epochs':4,'validation_rounds':[4,8,12,16,20],
                  'updates_per_epoch':updates_per_epoch,'warmup_updates':warmup,
                  'entrypoint':str(Path(__file__).resolve())}
        protocol['sampling']=args.sampling
        if plans:
            protocol['coverage_scope']='Every search frame (index >=1) scheduled; initial frame is initialization only'
            protocol['repeats']=None
        (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    epochs=[0,4,12] if args.smoke else range(start,20)
    try:
        for epoch in epochs:
            model.train(); objective.epoch=epoch; stage=curriculum(epoch)
            phase_batch=args.local_clips*(4 if epoch<4 else 1)
            if plans:
                entries,report=plans[stage['length']]
                dataset=CoverageClipDataset(metadata,entries,report,epoch,rank,world_size,phase_batch)
                if args.smoke_tail:
                    tails=np.array([i for i,row in enumerate(entries)
                                    if int(row[1])+int(row[3])==len(metadata[int(row[0])].boxes_xywh)])
                    dataset.ids=np.resize(tails[rank::world_size],phase_batch)
            else:
                dataset=ClipDataset(manifest['dataset_root'],names,epoch,rank,world_size,args.repeats,42)
                # Legacy random sampling only; coverage mode pads its global plan once.
                padded=math.ceil(len(dataset)/phase_batch)*phase_batch
                dataset.visits+=(dataset.visits*(math.ceil((padded-len(dataset))/len(dataset))))[:padded-len(dataset)]
            loader=DataLoader(dataset,batch_size=phase_batch,num_workers=8,
                              collate_fn=identity_collate,drop_last=False,
                              pin_memory=True,persistent_workers=True,prefetch_factor=4)
            first_grad_check=True
            consumed_plan_ids=[]
            observation_counts={}
            for iteration,clips in enumerate(loader):
                for chunk in range(0,stage['length'],4):
                    optimizer.zero_grad(set_to_none=True)
                    loss=ddp(clips,chunk,stage['scheduled_sampling'],stage['utility_weight'])
                    finite=torch.isfinite(loss).to(torch.int32); dist.all_reduce(finite,op=dist.ReduceOp.MIN)
                    if not finite.item(): raise RuntimeError('nonfinite loss on one or more ranks')
                    loss.backward()
                    norm=nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    if first_grad_check:
                        prefixes=('blocks.','head.','token_type_embed','codetrack.H.',
                                  'codetrack.template_pool.','codetrack.diagnosis.','codetrack.satr.',
                                  'codetrack.motion.','codetrack.memory.','codetrack.template_gate.',
                                  'codetrack.quality_head.')
                        alive={prefix:sum(float(p.grad.detach().abs().sum()) for n,p in model.named_parameters()
                                        if n.startswith(prefix) and p.grad is not None) for prefix in prefixes}
                        if any(v<=0 or not math.isfinite(v) for v in alive.values()):
                            raise RuntimeError(f'module gradient check failed: {alive}')
                        print(json.dumps({'rank':rank,'gradients':alive}),flush=True)
                        first_grad_check=False
                    optimizer.step(); scheduler.step()
                    if plans:
                        for key,value in objective.observation_counts.items():
                            observation_counts[key]=observation_counts.get(key,0)+value
                    if rank==0 and iteration%16==0:
                        print(json.dumps({'epoch':epoch,'phase':stage,'iteration':iteration,'chunk':chunk,
                                          'local_batch':phase_batch,
                                          'loss':float(loss.detach()),'grad_norm':float(norm),
                                          'allocated_gib':torch.cuda.max_memory_allocated()/2**30}),flush=True)
                if plans:
                    consumed_plan_ids.extend(c['plan_id'] for c in clips)
                if args.smoke: break
            if plans:
                audit_directory=args.output/'coverage'/f'epoch_{epoch:02d}'
                audit_directory.mkdir(parents=True,exist_ok=True)
                np.save(audit_directory/f'rank_{rank}_consumed.npy',np.asarray(consumed_plan_ids,dtype=np.int64))
                (audit_directory/f'rank_{rank}_observations.json').write_text(json.dumps(observation_counts)+'\n')
                dist.barrier()
                if rank==0:
                    consumed=np.concatenate([np.load(audit_directory/f'rank_{r}_consumed.npy')
                                             for r in range(world_size)])
                    unique=np.unique(consumed)
                    complete=np.array_equal(unique,np.arange(len(entries)))
                    if not args.smoke and not complete: raise RuntimeError('incomplete epoch coverage; refusing completion claim')
                    audit={'epoch':epoch,'smoke':args.smoke,'plan_sha256':report['plan_sha256'],
                           'planned_windows':len(entries),'consumed_unique_windows':len(unique),
                           'consumed_presentations':len(consumed),'complete':bool(complete),
                           'coverage_fraction':1. if complete else None,
                           'planned_search_frames':report['search_frames'],
                           'invalid_search_annotations':report['invalid_search_annotations'],
                           'ddp_padding_presentations':dataset.padding}
                    observations=[json.loads((audit_directory/f'rank_{r}_observations.json').read_text())
                                  for r in range(world_size)]
                    audit['observations']={key:sum(part[key] for part in observations)
                                           for key in observations[0]}
                    (audit_directory/'completed.json').write_text(json.dumps(audit,indent=2)+'\n')
                dist.barrier()
            save_epoch(args.output,model,optimizer,scheduler,epoch,manifest,cfg,world_size)
            if not args.smoke and should_validate(epoch):
                # Ordered full-sequence OPE, never random pair validation. Arms run in
                # independent processes, distribute sequences across the DDP ranks, and
                # return before the next epoch begins.
                dist.barrier()
                from tools.evaluate_final20_causal import evaluate_validation_rank
                rng=(torch.get_rng_state(),torch.cuda.get_rng_state_all(),
                     np.random.get_state(),random.getstate())
                try:
                    evaluate_validation_rank(model,cfg,manifest,args.output,epoch,rank)
                finally:
                    torch.set_rng_state(rng[0]); torch.cuda.set_rng_state_all(rng[1])
                    np.random.set_state(rng[2]); random.setstate(rng[3])
                dist.barrier()
    finally:
        dist.destroy_process_group()


def run_from_runtime(runtime):
    if runtime.eval or runtime.device!='cuda':
        raise RuntimeError('causal20 requires CUDA training; use the validation-selected evaluator for test')
    if runtime.weight_path:
        expected=(ROOT/'weights/gola_b224.bin').resolve()
        if len(runtime.weight_path)!=1 or Path(runtime.weight_path[0]).resolve()!=expected:
            raise RuntimeError('causal20 starts from gola_b224.bin; use --resume for continuation')
    argv=['--config',str(ROOT/'config/GOLA/codetrack_causal20/config.yaml'),
          '--output',str(Path(runtime.output_dir)/runtime.run_id),
          '--local-clips',str(runtime.local_clips),
          '--sampling',runtime.sampling,
          '--repeats',str(runtime.repeats_per_sequence)]
    if runtime.causal_smoke: argv.append('--smoke')
    if runtime.resume: argv.extend(['--resume',runtime.resume])
    return main(argv)


if __name__=='__main__': main()
