"""Focused persistent cross-output-tile TMA prefetch experiment.

Primary reference: Triton v3.8.0 python/tutorials/gluon/07-persistence.py.
Novel relative to current big_gemm: ring epochs and prefetch span output tile
boundaries, while the numerical reduction remains a plain full-K FP32 WGMMA.
CONTINUE=False is an identical persistent traversal with an explicit drain/reset.
No measured speedup claimed. Parent must allocate GPU budget before execution.
"""
import argparse
import statistics
import time
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma, warpgroup_mma, warpgroup_mma_wait
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


@gluon.jit
def pipeline(xd, wd, out, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
             GRID: gl.constexpr, DEPTH: gl.constexpr, CONTINUE: gl.constexpr):
    BT: gl.constexpr = xd.block_type.shape[0]
    BN: gl.constexpr = wd.block_type.shape[0]
    BK: gl.constexpr = xd.block_type.shape[1]
    KI: gl.constexpr = K // BK
    NM: gl.constexpr = triton.cdiv(M, BT)
    TILES: gl.constexpr = NM * triton.cdiv(N, BN)
    mma: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, BT, 16])
    xb = gl.allocate_shared_memory(xd.dtype, [DEPTH] + xd.block_type.shape, xd.layout)
    wb = gl.allocate_shared_memory(wd.dtype, [DEPTH] + wd.block_type.shape, wd.layout)
    ready = gl.allocate_shared_memory(gl.int64, [DEPTH, 1], mbarrier.MBarrierLayout())
    pid = gl.program_id(0)
    tiles = (TILES - pid + GRID - 1) // GRID
    BYTES: gl.constexpr = xd.block_type.nbytes + wd.block_type.nbytes
    rm = gl.arange(0, BT, layout=gl.SliceLayout(0, mma))
    rn = gl.arange(0, BN, layout=gl.SliceLayout(1, mma))
    if CONTINUE:
        for pre_s_1 in gl.static_range(DEPTH):
            mbarrier.init(ready.index(pre_s_1), count=1)
        steps = tiles * KI
        for pre_j_2 in gl.static_range(DEPTH - 1):
            pre_tile_2 = pid + (pre_j_2 // KI) * GRID
            pre_valid_2 = pre_j_2 < steps
            mbarrier.expect(ready.index(pre_j_2), BYTES, pred=pre_valid_2)
            tma.async_load(xd, [(pre_tile_2 % NM) * BT, (pre_j_2 % KI) * BK], ready.index(pre_j_2), xb.index(pre_j_2), pred=pre_valid_2)
            tma.async_load(wd, [(pre_tile_2 // NM) * BN, (pre_j_2 % KI) * BK], ready.index(pre_j_2), wb.index(pre_j_2), pred=pre_valid_2)
        acc = gl.zeros([BN, BT], gl.float32, mma)
        for i in range(steps):
            s = i % DEPTH
            mbarrier.wait(ready.index(s), (i // DEPTH) & 1)
            acc = warpgroup_mma(wb.index(s), xb.index(s).permute((1, 0)), acc, is_async=True)
            acc = warpgroup_mma_wait(1, deps=(acc,))
            j = i + DEPTH - 1
            sj = j % DEPTH
            tile_j = pid + (j // KI) * GRID
            valid = j < steps
            mbarrier.expect(ready.index(sj), BYTES, pred=valid)
            tma.async_load(xd, [(tile_j % NM) * BT, (j % KI) * BK], ready.index(sj), xb.index(sj), pred=valid)
            tma.async_load(wd, [(tile_j // NM) * BN, (j % KI) * BK], ready.index(sj), wb.index(sj), pred=valid)
            if (i + 1) % KI == 0:
                acc = warpgroup_mma_wait(0, deps=(acc,))
                tile = pid + (i // KI) * GRID
                mm = (tile % NM) * BT + rm
                nn = (tile // NM) * BN + rn
                gl.store(out + mm[None, :] * N + nn[:, None], acc.to(gl.bfloat16), (mm < M)[None, :] & (nn < N)[:, None])
                acc = gl.zeros([BN, BT], gl.float32, mma)
        for pre_s_3 in gl.static_range(DEPTH):
            mbarrier.invalidate(ready.index(pre_s_3))
    else:
        for ti in range(tiles):
            tile = pid + ti * GRID
            for pre_s_4 in gl.static_range(DEPTH):
                mbarrier.init(ready.index(pre_s_4), count=1)
            for pre_j_5 in gl.static_range(DEPTH - 1):
                pre_valid_5 = pre_j_5 < KI
                mbarrier.expect(ready.index(pre_j_5), BYTES, pred=pre_valid_5)
                tma.async_load(xd, [(tile % NM) * BT, pre_j_5 * BK], ready.index(pre_j_5), xb.index(pre_j_5), pred=pre_valid_5)
                tma.async_load(wd, [(tile // NM) * BN, pre_j_5 * BK], ready.index(pre_j_5), wb.index(pre_j_5), pred=pre_valid_5)
            acc = gl.zeros([BN, BT], gl.float32, mma)
            for i in range(KI):
                s = i % DEPTH
                mbarrier.wait(ready.index(s), (i // DEPTH) & 1)
                acc = warpgroup_mma(wb.index(s), xb.index(s).permute((1, 0)), acc, is_async=True)
                acc = warpgroup_mma_wait(1, deps=(acc,))
                j = i + DEPTH - 1
                sj = j % DEPTH
                valid = j < KI
                mbarrier.expect(ready.index(sj), BYTES, pred=valid)
                tma.async_load(xd, [(tile % NM) * BT, j * BK], ready.index(sj), xb.index(sj), pred=valid)
                tma.async_load(wd, [(tile // NM) * BN, j * BK], ready.index(sj), wb.index(sj), pred=valid)
            acc = warpgroup_mma_wait(0, deps=(acc,))
            mm = (tile % NM) * BT + rm
            nn = (tile // NM) * BN + rn
            gl.store(out + mm[None, :] * N + nn[:, None], acc.to(gl.bfloat16), (mm < M)[None, :] & (nn < N)[:, None])
            for pre_s_6 in gl.static_range(DEPTH):
                mbarrier.invalidate(ready.index(pre_s_6))
            gl.barrier()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layer', choices=['o', 'qkv_gate', 'pooler', 'unpooler'], default='o')
    ap.add_argument('--m', type=int, default=256)
    ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--full', action='store_true')
    args = ap.parse_args()
    n,k = dict(o=(8192,1024), qkv_gate=(2304,8192), pooler=(4096,8192), unpooler=(8192,4096))[args.layer]
    m=args.m
    torch.manual_seed(937)
    x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
    w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16) / k**0.5
    x[::8]=0
    x[::8,17]=2
    ref=x.double() @ w.double().T
    flush=torch.empty(64*1024*1024, device='cuda', dtype=torch.int32)
    started=time.monotonic()
    graphs=[]
    keep=[]
    configs = [(64,64,64,4),(64,64,128,4),(128,64,64,4)] if args.full else [(64,64,64,4)]
    for bt,bn,bk,depth in configs:
        xd=TensorDescriptor.from_tensor(x,[bt,bk],gl.NVMMASharedLayout.get_default_for([bt,bk],gl.bfloat16))
        wd=TensorDescriptor.from_tensor(w,[bn,bk],gl.NVMMASharedLayout.get_default_for([bn,bk],gl.bfloat16))
        nt=triton.cdiv(m,bt)*triton.cdiv(n,bn)
        for grid,cont in [(min(132,nt),False),(min(132,nt),True),(nt,False)]:
            if time.monotonic()-started > 60:
                print('COMPILE BUDGET reached, skip remaining variants',flush=True)
                break
            out=torch.empty(m,n,device='cuda',dtype=torch.bfloat16)
            def launch():
                return pipeline[(grid,)](xd,wd,out,m,n,k,grid,depth,cont,num_warps=4)
            tag=f'BT{bt} BN{bn} BK{bk} D{depth} grid{grid} continue{cont}'
            try:
                ker=launch()
                torch.cuda.synchronize()
                error=((out.double()-ref).norm()/ref.norm()).item()
                probe=((out[::8].double()-ref[::8]).norm()/ref[::8].norm()).item()
                assert error < .01 and probe < .001, (error,probe)
                # Replay changes x to verify neither cache nor stale ring behavior.
                x.mul_(.5)
                launch()
                torch.cuda.synchronize()
                assert ((out.double()-ref*.5).norm()/ref.norm()).item() < .005
                x.mul_(2)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph): launch()
                print(tag,'correct',error,'probe',probe,'regs',ker.n_regs,'smem',ker.metadata.shared,flush=True)
                graphs.append((tag,graph,[]))
                keep.extend([out,xd,wd])
            except Exception as exc:
                print(tag,'FAIL',repr(exc),flush=True)
    out_base=torch.empty(m,n,device='cuda',dtype=torch.bfloat16)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): torch.mm(x,w.T,out=out_base)
    graphs.append(('torch_cublas',graph,[]))
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    for r in range(args.reps):
        order=graphs if r%2==0 else list(reversed(graphs))
        for tag,g,ts in order:
            flush.fill_(r)
            a.record(); g.replay(); b.record(); b.synchronize()
            ts.append(a.elapsed_time(b)*1000)
    for tag,g,ts in graphs:
        print('RESULT',tag,'raw_us',round(statistics.median(ts),3),'minus4_us',round(statistics.median(ts)-4,3),flush=True)

if __name__ == '__main__': main()
