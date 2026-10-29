"""GPU-UNVALIDATED router MMA2 + cluster reduction, source/CPU prototype only.

BF16 hi/lo weights, BF16 x, separate FP32 accumulators, one FP32 final sum.
R32/T64/BK128, splitK cluster4/8, depth3 raw-byte cp.async ring (96KiB/CTA).
No global partial workspace, reduction tail kernel, output cache, or C++.
--syntax imports only stdlib; all performance and GPU accuracy UNPROVEN.
"""
from __future__ import annotations
import ast
from pathlib import Path
import sys
if '--syntax' in sys.argv:
    ast.parse(Path(__file__).read_text())
    print('syntax PASS; no GPU compile/correctness/timing proof')
    raise SystemExit(0)

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2,async_copy
from triton.experimental.gluon.language.nvidia.hopper import mbarrier,async_store


def layouts(c):
    """Construct nested CGA lists on the host, avoiding pinned3.8 JIT list equality bug."""
    if c not in (4,8):raise ValueError('cluster must be4 or8')
    src_cga=[[1<<i,0,0] for i in range(c.bit_length()-1)]
    dst_cga=[[0,0,1<<i] for i in range(c.bit_length()-1)]
    return dict(
        COPY=gl.BlockedLayout([1,1,8],[1,4,8],[1,2,2],[2,1,0],cga_layout=src_cga),
        MMA=gl.NVMMADistributedLayout([2,0],[1,2,2],[1,16,8],cga_layout=src_cga),
        A_SHARED=gl.NVMMASharedLayout.get_default_for([c,32,128],gl.bfloat16,cga_layout=src_cga),
        X_SHARED=gl.NVMMASharedLayout.get_default_for([c,64,128],gl.bfloat16,cga_layout=src_cga),
        SRC=gl.BlockedLayout([1,1,2],[1,8,4],[1,4,1],[2,1,0],cga_layout=src_cga),
        DST=gl.BlockedLayout([1,1,2],[1,8,4],[1,4,1],[2,1,0],cga_layout=dst_cga),
        MAIL=gl.SwizzledSharedLayout(1,1,1,[2,1,0],cga_layout=dst_cga))


@gluon.jit
def _copy_tile(ptr,stage,n0,step,K:gl.constexpr,C:gl.constexpr,ROWS:gl.constexpr,
               BK:gl.constexpr,COPY:gl.constexpr):
    batch=gl.arange(0,C,gl.SliceLayout(1,gl.SliceLayout(2,COPY)))
    row=gl.arange(0,ROWS,gl.SliceLayout(0,gl.SliceLayout(2,COPY)))
    kk=gl.arange(0,BK,gl.SliceLayout(0,gl.SliceLayout(0,COPY)))
    batch=batch[:,None][:,:,None]
    row=row[None,:][:,:,None]
    kk=kk[None,:][None,:,:]
    offsets=(n0+row)*K+batch*(K//C)+step*BK+kk
    async_copy.async_load(stage,ptr+offsets,mask=(step<K//(C*BK)) & (offsets>=0))


@gluon.jit
def _issue(hi,lo,x,hs,ls,xs,n0,t0,step,K:gl.constexpr,C:gl.constexpr,
           BK:gl.constexpr,COPY:gl.constexpr):
    _copy_tile(hi,hs,n0,step,K,C,32,BK,COPY)
    _copy_tile(lo,ls,n0,step,K,C,32,BK,COPY)
    _copy_tile(x,xs,t0,step,K,C,64,BK,COPY)
    async_copy.commit_group()


@gluon.jit
def router_cluster_bt64(hi,lo,x,out,M:gl.constexpr,N:gl.constexpr,K:gl.constexpr,
        C:gl.constexpr,DEPTH:gl.constexpr,ASYNC:gl.constexpr,
        COPY:gl.constexpr,MMA:gl.constexpr,A_SHARED:gl.constexpr,X_SHARED:gl.constexpr,
        SRC:gl.constexpr,DST:gl.constexpr,MAIL:gl.constexpr):
    BK:gl.constexpr=128
    gl.static_assert(C==4 or C==8)
    gl.static_assert(N%32==0 and M%64==0)
    gl.static_assert(K%(C*BK)==0)
    gl.static_assert(DEPTH>=2 and DEPTH<=4)
    KI:gl.constexpr=K//(C*BK)
    gl.static_assert(KI>=DEPTH)
    pid=gl.program_id(0)
    n0=(pid%(N//32))*32
    t0=(pid//(N//32))*64
    hb=gl.allocate_shared_memory(gl.bfloat16,[DEPTH,C,32,BK],A_SHARED)
    lb=gl.allocate_shared_memory(gl.bfloat16,[DEPTH,C,32,BK],A_SHARED)
    xb=gl.allocate_shared_memory(gl.bfloat16,[DEPTH,C,64,BK],X_SHARED)
    adot:gl.constexpr=gl.DotOperandLayout(0,MMA,2)
    bdot:gl.constexpr=gl.DotOperandLayout(1,MMA,2)
    for pre_s in gl.static_range(DEPTH):
        _issue(hi,lo,x,hb.index(pre_s),lb.index(pre_s),xb.index(pre_s),
               n0,t0,pre_s,K,C,BK,COPY)
    acc_hi=gl.zeros([C,32,64],gl.float32,MMA)
    acc_lo=gl.zeros([C,32,64],gl.float32,MMA)
    for it in range(KI):
        s=it%DEPTH
        async_copy.wait_group(DEPTH-1)
        gl.barrier()
        a_hi=hb.index(s).load(adot)
        a_lo=lb.index(s).load(adot)
        b=xb.index(s).permute((0,2,1)).load(bdot)
        gl.barrier()  # All shared operands have reached registers before refill.
        _issue(hi,lo,x,hb.index(s),lb.index(s),xb.index(s),
               n0,t0,it+DEPTH,K,C,BK,COPY)
        acc_hi=mma_v2(a_hi,b,acc_hi)
        acc_lo=mma_v2(a_lo,b,acc_lo)
    async_copy.wait_group(0)
    gl.barrier()  # Drain before the allocator can alias rings with mailboxes.
    values=gl.convert_layout(acc_hi+acc_lo,SRC)
    smem=gl.allocate_shared_memory(gl.float32,[C,32,64],MAIL)
    if ASYNC:
        bar=mbarrier.allocate_mbarrier()
        mbarrier.init(bar,count=1)
        mbarrier.fence_init_release_cluster()
        # Peers must finish their rings before any remote mailbox store can alias.
        gl.barrier(cluster=True)
        mbarrier.expect(bar,smem.nbytes_per_cta)
        async_store(smem,values,bar)
        mbarrier.wait(bar,phase=0,deps=[smem])
        local=smem.load(DST)
        gl.barrier(cluster=True)  # All senders/receivers stay alive through loads.
        mbarrier.invalidate(bar)
    else:
        # Required here as well: a peer can still be using an aliased ring.
        gl.barrier(cluster=True)
        smem.store(values)
        gl.barrier(cluster=True)
        local=smem.load(DST)
        gl.barrier(cluster=True)
    total=gl.convert_layout(gl.sum(local,axis=0),gl.SliceLayout(0,DST))
    epi:gl.constexpr=gl.SliceLayout(0,DST)
    row=gl.arange(0,32,gl.SliceLayout(1,epi))
    token=gl.arange(0,64,gl.SliceLayout(0,epi))
    gl.store(out+(t0+token[None,:])*N+n0+row[:,None],total)


def make(x,w,*,cluster=None,depth=3,async_epilogue=True):
    """Untimed standalone preparation; rejection leaves any caller's baseline intact."""
    m,k=x.shape
    n=w.shape[0]
    c=(8 if m==128 else 4) if cluster is None else cluster
    if (m not in (128,256) or n!=256 or w.shape[1]!=k or k!=8192 or
            x.dtype!=torch.bfloat16 or w.dtype!=torch.float32 or
            not x.is_contiguous() or not w.is_contiguous() or
            x.device!=w.device or c not in (4,8) or depth not in (2,3,4)):
        raise ValueError('requires contiguous BF16x/FP32W router shapes, C4/8, depth2..4')
    if not bool(torch.isfinite(w).all()):raise ValueError('nonfinite router weights')
    hi=w.to(torch.bfloat16)
    lo=(w-hi.float()).to(torch.bfloat16)
    if not bool(torch.isfinite(hi).all()&torch.isfinite(lo).all()):
        raise ValueError('router hi/lo overflow; caller must use fallback')
    cfg=layouts(c)
    def run():
        out=torch.empty((m,n),device=x.device,dtype=torch.float32)
        compiled=router_cluster_bt64[((n//32)*(m//64),)](
            hi,lo,x,out,m,n,k,c,depth,async_epilogue,
            **cfg,num_warps=4,num_ctas=c)
        return out
    # Keep prepared precision-preserving weight components alive in the closure.
    return run,dict(hi=hi,lo=lo,layouts=cfg,cluster=c,depth=depth)
