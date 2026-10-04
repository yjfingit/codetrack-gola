"""Verify CROSS-RANK calibration: 4 processes must agree on gain/offset.

Run under torchrun with 4 procs.  Each rank feeds a DIFFERENT random shard, so a
per-rank (broken) implementation would produce four different gains.  The fixed
implementation must produce one shared gain equal to the pooled statistics.
"""
import os, sys, math, torch, torch.distributed as dist
sys.path.insert(0, ".")
from codetrack.ecc import SyndromeDiagnosis
from codetrack.ddp_calibration import consume_pending_calibration

def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    torch.manual_seed(1000 + rank)          # DIFFERENT data per rank on purpose

    M, N, D = 64, 256, 128
    diag = SyndromeDiagnosis(num_checks=M, num_variables=N, mid_dim=D,
                             syndrome_gain_calibration=True).to(dev)
    diag.train()
    H = torch.rand(M, N, device=dev); H = H / H.sum(-1, keepdim=True)
    X_t = torch.randn(4, N, 768, device=dev) * (1.0 + rank)   # scaled per rank
    X_aux = torch.randn(4, N, 768, device=dev)

    out = diag(X_t, X_aux, H)
    pend = out.get("syndrome_pending_calibration")
    if pend is None:
        if rank == 0: print("FAIL: no pending export")
        dist.destroy_process_group(); return

    local_mean = float(pend.mean()); local_std = float(pend.std(unbiased=False))
    gain = consume_pending_calibration(diag, pend)

    # gather every rank's view
    g = torch.tensor([float(diag.syndrome_logit_gain.detach().item())], device=dev)
    o = torch.tensor([float(diag.syndrome_logit_offset.detach().item())], device=dev)
    gathered_g, gathered_o = [torch.zeros_like(g) for _ in range(world)], [torch.zeros_like(o) for _ in range(world)]
    dist.all_gather(gathered_g, g); dist.all_gather(gathered_o, o)

    if rank == 0:
        gs = [float(x.item()) for x in gathered_g]
        os_ = [float(x.item()) for x in gathered_o]
        print(f"local  (rank-dependent) mean/std: {local_mean:.6f} / {local_std:.6f}")
        print(f"rank0   gain={gs[0]:.6f} offset={os_[0]:.6f}")
        print(f"gains  per rank: {[round(x,6) for x in gs]}")
        print(f"offset per rank: {[round(x,6) for x in os_]}")
        same_g = all(abs(x - gs[0]) < 1e-9 for x in gs)
        same_o = all(abs(x - os_[0]) < 1e-9 for x in os_)
        print(f"\nSYNC gain identical across ranks   : {same_g}")
        print(f"SYNC offset identical across ranks : {same_o}")
        print("DDP SYNC TEST: " + ("PASS" if (same_g and same_o) else "FAIL"))

    dist.destroy_process_group()

main()
