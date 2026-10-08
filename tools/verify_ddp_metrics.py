"""Three-rank witness: conditional metric keys must neither crash nor dilute means."""
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from trackit.miscellanies.torch.distributed import init_torch_distributed
from trackit.miscellanies.torch.distributed.reduce_dict import reduce_dict, reduce_dict_async

init_torch_distributed('cuda',silent_non_local_master=False)
rank=int(os.environ['RANK'])
for tensor in (False,True):
    values={'shared':float(rank+1)}
    if rank==1: values['only_one']=9.
    if rank!=0: values['two_ranks']=float(2*rank)
    if tensor: values={k:torch.tensor(v,device='cuda') for k,v in values.items()}
    result=reduce_dict(values)
    assert float(result['shared'])==2.
    assert float(result['only_one'])==9.
    assert float(result['two_ranks'])==3.
    result=reduce_dict_async(values,average=False).get()
    assert float(result['shared'])==6.
    assert float(result['only_one'])==9.
    assert float(result['two_ranks'])==6.
print(f'rank {rank}: CPU/GPU optional metrics agree and preserve means',flush=True)
dist.destroy_process_group()
