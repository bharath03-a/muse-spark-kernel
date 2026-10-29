"""CPU-only audit of existing Gluon cluster epilogue, no GPU allocations."""
import json,traceback
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import mbarrier,async_store
from triton.experimental.gluon._runtime import GluonASTSource
from triton.backends.compiler import GPUTarget
import inspect
print("MMA_LAYOUT_SIGNATURE",inspect.signature(gl.NVMMADistributedLayout),flush=True)
SRC2 = gl.BlockedLayout([1,1,2],[1,8,4],[1,4,1],[2,1,0],cga_layout=[[1, 0, 0]])
DST2 = gl.BlockedLayout([1,1,2],[1,8,4],[1,4,1],[2,1,0],cga_layout=[[0, 0, 1]])
SH2 = gl.SwizzledSharedLayout(1,1,1,order=[2,1,0],cga_layout=[[0, 0, 1]])
@gluon.jit
def reduce_partials2(P, O, R: gl.constexpr, T: gl.constexpr, ASYNC: gl.constexpr):
        # Each source CTA holds one K-split plane. Destination CTAs own tokens.
        src: gl.constexpr = SRC2
        dst: gl.constexpr = DST2
        shared: gl.constexpr = SH2
        sf = gl.arange(0, 2, gl.SliceLayout(1, gl.SliceLayout(2, src)))
        rr = gl.arange(0, R, gl.SliceLayout(0, gl.SliceLayout(2, src)))
        tt = gl.arange(0, T, gl.SliceLayout(0, gl.SliceLayout(0, src)))
        offs = sf[:, None][:, :, None] * (R*T) + rr[None, :][:, :, None] * T + tt[None, :][None, :, :]
        
        values = gl.load(P + gl.program_id(0)*(2*R*T) + offs)
        smem = gl.allocate_shared_memory(gl.float32, [2, R, T], shared)
        if ASYNC:
            bar = mbarrier.allocate_mbarrier()
            mbarrier.init(bar, count=1)
            mbarrier.fence_init_release_cluster()
            gl.barrier(cluster=True)
            mbarrier.expect(bar, smem.nbytes_per_cta)
            async_store(smem, values, bar)
            mbarrier.wait(bar, phase=0, deps=[smem])
            local = smem.load(dst)
            # Keep each peer's destination and barrier alive until all peers finish.
            gl.barrier(cluster=True)
            mbarrier.invalidate(bar)
        else:
            smem.store(values)
            gl.barrier(cluster=True)
            local = smem.load(dst)
            gl.barrier(cluster=True)
        total = gl.convert_layout(gl.sum(local, axis=0), gl.SliceLayout(0, dst))
        epi: gl.constexpr = gl.SliceLayout(0, dst)
        ro = gl.arange(0, R, gl.SliceLayout(1, epi))
        to = gl.arange(0, T, gl.SliceLayout(0, epi))
        outoffs = ro[:, None]*T + to[None, :]
        gl.store(O + gl.program_id(0)*(R*T) + outoffs, total)
SRC4 = gl.BlockedLayout([1,1,2],[1,8,4],[1,4,1],[2,1,0],cga_layout=[[1, 0, 0], [2, 0, 0]])
DST4 = gl.BlockedLayout([1,1,2],[1,8,4],[1,4,1],[2,1,0],cga_layout=[[0, 0, 1], [0, 0, 2]])
SH4 = gl.SwizzledSharedLayout(1,1,1,order=[2,1,0],cga_layout=[[0, 0, 1], [0, 0, 2]])
@gluon.jit
def reduce_partials4(P, O, R: gl.constexpr, T: gl.constexpr, ASYNC: gl.constexpr):
        # Each source CTA holds one K-split plane. Destination CTAs own tokens.
        src: gl.constexpr = SRC4
        dst: gl.constexpr = DST4
        shared: gl.constexpr = SH4
        sf = gl.arange(0, 4, gl.SliceLayout(1, gl.SliceLayout(2, src)))
        rr = gl.arange(0, R, gl.SliceLayout(0, gl.SliceLayout(2, src)))
        tt = gl.arange(0, T, gl.SliceLayout(0, gl.SliceLayout(0, src)))
        offs = sf[:, None][:, :, None] * (R*T) + rr[None, :][:, :, None] * T + tt[None, :][None, :, :]
        
        values = gl.load(P + gl.program_id(0)*(4*R*T) + offs)
        smem = gl.allocate_shared_memory(gl.float32, [4, R, T], shared)
        if ASYNC:
            bar = mbarrier.allocate_mbarrier()
            mbarrier.init(bar, count=1)
            mbarrier.fence_init_release_cluster()
            gl.barrier(cluster=True)
            mbarrier.expect(bar, smem.nbytes_per_cta)
            async_store(smem, values, bar)
            mbarrier.wait(bar, phase=0, deps=[smem])
            local = smem.load(dst)
            # Keep each peer's destination and barrier alive until all peers finish.
            gl.barrier(cluster=True)
            mbarrier.invalidate(bar)
        else:
            smem.store(values)
            gl.barrier(cluster=True)
            local = smem.load(dst)
            gl.barrier(cluster=True)
        total = gl.convert_layout(gl.sum(local, axis=0), gl.SliceLayout(0, dst))
        epi: gl.constexpr = gl.SliceLayout(0, dst)
        ro = gl.arange(0, R, gl.SliceLayout(1, epi))
        to = gl.arange(0, T, gl.SliceLayout(0, epi))
        outoffs = ro[:, None]*T + to[None, :]
        gl.store(O + gl.program_id(0)*(R*T) + outoffs, total)
SRC8 = gl.BlockedLayout([1,1,2],[1,8,4],[1,4,1],[2,1,0],cga_layout=[[1, 0, 0], [2, 0, 0], [4, 0, 0]])
DST8 = gl.BlockedLayout([1,1,2],[1,8,4],[1,4,1],[2,1,0],cga_layout=[[0, 0, 1], [0, 0, 2], [0, 0, 4]])
SH8 = gl.SwizzledSharedLayout(1,1,1,order=[2,1,0],cga_layout=[[0, 0, 1], [0, 0, 2], [0, 0, 4]])
@gluon.jit
def reduce_partials8(P, O, R: gl.constexpr, T: gl.constexpr, ASYNC: gl.constexpr):
        # Each source CTA holds one K-split plane. Destination CTAs own tokens.
        src: gl.constexpr = SRC8
        dst: gl.constexpr = DST8
        shared: gl.constexpr = SH8
        sf = gl.arange(0, 8, gl.SliceLayout(1, gl.SliceLayout(2, src)))
        rr = gl.arange(0, R, gl.SliceLayout(0, gl.SliceLayout(2, src)))
        tt = gl.arange(0, T, gl.SliceLayout(0, gl.SliceLayout(0, src)))
        offs = sf[:, None][:, :, None] * (R*T) + rr[None, :][:, :, None] * T + tt[None, :][None, :, :]
        
        values = gl.load(P + gl.program_id(0)*(8*R*T) + offs)
        smem = gl.allocate_shared_memory(gl.float32, [8, R, T], shared)
        if ASYNC:
            bar = mbarrier.allocate_mbarrier()
            mbarrier.init(bar, count=1)
            mbarrier.fence_init_release_cluster()
            gl.barrier(cluster=True)
            mbarrier.expect(bar, smem.nbytes_per_cta)
            async_store(smem, values, bar)
            mbarrier.wait(bar, phase=0, deps=[smem])
            local = smem.load(dst)
            # Keep each peer's destination and barrier alive until all peers finish.
            gl.barrier(cluster=True)
            mbarrier.invalidate(bar)
        else:
            smem.store(values)
            gl.barrier(cluster=True)
            local = smem.load(dst)
            gl.barrier(cluster=True)
        total = gl.convert_layout(gl.sum(local, axis=0), gl.SliceLayout(0, dst))
        epi: gl.constexpr = gl.SliceLayout(0, dst)
        ro = gl.arange(0, R, gl.SliceLayout(1, epi))
        to = gl.arange(0, T, gl.SliceLayout(0, epi))
        outoffs = ro[:, None]*T + to[None, :]
        gl.store(O + gl.program_id(0)*(R*T) + outoffs, total)

for ct,fn in [(2,reduce_partials2),(4,reduce_partials4),(8,reduce_partials8)]:
 for mode in (False,True):
  try:
   names=list(fn.arg_names); consts={'R':32,'T':64,'ASYNC':mode}
   sig={n:'*fp32' if n in ('P','O') else 'constexpr' for n in names}
   src=GluonASTSource(fn=fn,signature=sig,constexprs={(names.index(n),):v for n,v in consts.items()})
   k=triton.compile(src,target=GPUTarget('cuda',90,32),options={'num_warps':4,'num_ctas':ct}); ptx=k.asm['ptx']
   print(json.dumps({'ctas':ct,'async':mode,'pass':True,'shared':k.metadata.shared,'st_async':ptx.count('st.async'),'cluster_barriers':ptx.count('barrier.cluster'),'shuffle':ptx.count('shfl')}),flush=True)
   print('PTX_BEGIN',ct,mode,ptx,'PTX_END',flush=True)
  except Exception:
   print('CLUSTER_COMPILE_FAIL',ct,mode,traceback.format_exc(),flush=True)
