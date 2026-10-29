"""Cold-L2 HBM read bandwidth sweep: LDG.128 / cp.async.bulk 1D / 2D TMA, graph replay, L2 flushed.

    python bench/bw_sweep.py [--reps 50] [--dump out.cu]
"""
import argparse, os, statistics, sys, threading

_wd = threading.Timer(200, lambda: (print("WATCHDOG", flush=True), os._exit(0)))
_wd.daemon = True
_wd.start()

SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <vector>
__device__ __forceinline__ uint32_t su(const void* p){return (uint32_t)__cvta_generic_to_shared(p);}
__device__ __forceinline__ void waitp(uint64_t* b, int par){
  asm volatile("{.reg .pred P; W: mbarrier.try_wait.parity.shared::cta.b64 P,[%0],%1; @P bra D; bra W; D:}"::"r"(su(b)),"r"(par));}

template <int U, bool CONTIG>
__global__ void __launch_bounds__(256) ldg_k(const uint4* p, long n, int* sink){
  long tid = (long)blockIdx.x*blockDim.x + threadIdx.x, stride = (long)gridDim.x*blockDim.x;
  unsigned acc = 0;
  if (CONTIG){
    long per = n / gridDim.x, lo = per*blockIdx.x, hi = (blockIdx.x==gridDim.x-1)? n : lo+per;
    for (long i = lo+threadIdx.x; i < hi; i += (long)blockDim.x*U){
      uint4 v[U];
      #pragma unroll
      for (int u=0;u<U;++u){ long j=i+(long)u*blockDim.x; if (j<hi) asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(v[u].x),"=r"(v[u].y),"=r"(v[u].z),"=r"(v[u].w):"l"(p+j)); else v[u]=make_uint4(0,0,0,0);}
      #pragma unroll
      for (int u=0;u<U;++u) acc ^= v[u].x^v[u].y^v[u].z^v[u].w;
    }
  } else {
    for (long i = tid; i < n; i += stride*U){
      uint4 v[U];
      #pragma unroll
      for (int u=0;u<U;++u){ long j=i+(long)u*stride; if (j<n) asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(v[u].x),"=r"(v[u].y),"=r"(v[u].z),"=r"(v[u].w):"l"(p+j)); else v[u]=make_uint4(0,0,0,0);}
      #pragma unroll
      for (int u=0;u<U;++u) acc ^= v[u].x^v[u].y^v[u].z^v[u].w;
    }
  }
  if (acc == 0x9e3779b9u) sink[0] = acc;
}
template <int U, bool C> void ldg_run(torch::Tensor t, int64_t blocks, torch::Tensor s){
  ldg_k<U,C><<<blocks,256,0,at::cuda::getCurrentCUDAStream()>>>((const uint4*)t.data_ptr(), t.numel()/16, (int*)s.data_ptr<int>());}
void ldg(torch::Tensor t, int64_t blocks, int64_t u, int64_t contig, torch::Tensor s){
  if (u==4) { if (contig) ldg_run<4,true>(t,blocks,s); else ldg_run<4,false>(t,blocks,s);}
  else if (u==8) { if (contig) ldg_run<8,true>(t,blocks,s); else ldg_run<8,false>(t,blocks,s);}
  else if (u==16) { if (contig) ldg_run<16,true>(t,blocks,s); else ldg_run<16,false>(t,blocks,s);}
  else TORCH_CHECK(false,"u");
}

constexpr int MAXB = 64;
// each warp owns a ring of ST slots of CH bytes; warps*ctas workers split nch chunks
__global__ void bulk_k(const char* p, long nch, int CH, int ST, int inter){
  extern __shared__ __align__(128) unsigned char sm[];
  __shared__ __align__(8) uint64_t bar[MAXB];
  int w = threadIdx.x >> 5, W = blockDim.x >> 5;
  if (threadIdx.x == 0){ for (int s=0;s<W*ST;++s) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"::"r"(su(&bar[s]))); asm volatile("fence.mbarrier_init.release.cluster;"); }
  __syncthreads();
  if ((threadIdx.x & 31) != 0) return;
  long gw = (long)blockIdx.x*W + w, GW = (long)gridDim.x*W;
  long lo, step, n;
  if (inter){ lo = gw; step = GW; n = (nch > gw) ? (nch - gw + GW - 1)/GW : 0; }
  else { long a = nch*gw/GW, b = nch*(gw+1)/GW; lo = a; step = 1; n = b-a; }
  unsigned char* ring = sm + (long)w*ST*CH; uint64_t* bb = bar + w*ST;
  for (long i = 0; i < n + ST - 1; ++i){
    if (i >= ST){ long j=i-ST; int s=j%ST; waitp(&bb[s], (j/ST)&1); }
    if (i < n){
      int s = i%ST;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(su(&bb[s])),"r"(CH));
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"
        ::"r"(su(ring+(long)s*CH)),"l"(p+(lo+i*step)*(long)CH),"r"(CH),"r"(su(&bb[s])):"memory");
    }
  }
  for (long j = (n-ST+1>0? n-ST+1:0); j < n; ++j){ int s=j%ST; waitp(&bb[s], (j/ST)&1); }
}
void bulk(torch::Tensor t, int64_t ctas, int64_t ch, int64_t st, int64_t warps, int64_t inter){
  long smem = warps*st*ch;
  TORCH_CHECK(smem <= 227*1024 && warps*st <= MAXB && t.numel()%ch==0, "bulk cfg");
  cudaFuncSetAttribute(bulk_k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  bulk_k<<<ctas, 32*warps, smem, at::cuda::getCurrentCUDAStream()>>>((const char*)t.data_ptr(), t.numel()/ch, (int)ch, (int)st, (int)inter);
}

__global__ void tma_k(const __grid_constant__ CUtensorMap map, long tiles, int CB, int BI, int BR, int ST, int inter){
  extern __shared__ __align__(128) unsigned char sm[];
  __shared__ __align__(8) uint64_t bar[MAXB];
  if (threadIdx.x == 0){ for (int s=0;s<ST;++s) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"::"r"(su(&bar[s]))); asm volatile("fence.mbarrier_init.release.cluster;"); }
  __syncthreads();
  if (threadIdx.x != 0) return;
  long b = blockIdx.x, G = gridDim.x, lo, step, n;
  if (inter){ lo=b; step=G; n=(tiles>b)?(tiles-b+G-1)/G:0; } else { long a=tiles*b/G, e=tiles*(b+1)/G; lo=a; step=1; n=e-a; }
  int CH = BI*BR;
  for (long i = 0; i < n + ST - 1; ++i){
    if (i >= ST){ long j=i-ST; int s=j%ST; waitp(&bar[s], (j/ST)&1); }
    if (i < n){
      int s = i%ST; long t = lo+i*step; int c0=(int)(t%CB)*BI, c1=(int)(t/CB)*BR;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(su(&bar[s])),"r"(CH));
      asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
        ::"r"(su(sm+(long)s*CH)),"l"(&map),"r"(c0),"r"(c1),"r"(su(&bar[s])):"memory");
    }
  }
  for (long j = (n-ST+1>0? n-ST+1:0); j < n; ++j){ int s=j%ST; waitp(&bar[s], (j/ST)&1); }
}
constexpr int ROWB = 8192;
void tma(torch::Tensor t, int64_t ctas, int64_t bi, int64_t br, int64_t st, int64_t inter){
  long R = t.numel()/ROWB; TORCH_CHECK(R%br==0 && ROWB%bi==0, "tma cfg");
  CUtensorMap map{};
  cuuint64_t gdim[2] = {(cuuint64_t)ROWB, (cuuint64_t)R};
  cuuint64_t gstride[1] = {(cuuint64_t)ROWB};
  cuuint32_t bdim[2] = {(cuuint32_t)bi, (cuuint32_t)br}, es[2] = {1u,1u};
  CUresult res = cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, t.data_ptr(), gdim, gstride, bdim, es,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(res == CUDA_SUCCESS, "encode ", (int)res);
  long smem = st*bi*br; TORCH_CHECK(smem <= 227*1024 && st<=MAXB, "tma smem");
  cudaFuncSetAttribute(tma_k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  tma_k<<<ctas, 32, smem, at::cuda::getCurrentCUDAStream()>>>(map, (R/br)*(ROWB/bi), (int)(ROWB/bi), (int)bi, (int)br, (int)st, (int)inter);
}
"""
CPP = """void ldg(torch::Tensor t, int64_t blocks, int64_t u, int64_t contig, torch::Tensor s);
void bulk(torch::Tensor t, int64_t ctas, int64_t ch, int64_t st, int64_t warps, int64_t inter);
void tma(torch::Tensor t, int64_t ctas, int64_t bi, int64_t br, int64_t st, int64_t inter);"""

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=50)
ap.add_argument("--dump", default="")
args = ap.parse_args()
if args.dump:
    open(args.dump, "w").write(SRC); sys.exit(0)

import torch
from torch.utils.cpp_extension import load_inline

ext = load_inline(name="bw_sweep_v1", cpp_sources=CPP, cuda_sources=SRC, functions=["ldg", "bulk", "tma"],
                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], extra_ldflags=["-lcuda"])
SM = torch.cuda.get_device_properties(0).multi_processor_count
flush = torch.empty(256 * 2**20 // 4, device="cuda", dtype=torch.float32).normal_()
sink = torch.zeros(4, device="cuda", dtype=torch.int32)


def graph_of(fn):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def one(g):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    flush.sum(); s.record(); g.replay(); e.record(); e.synchronize()
    return s.elapsed_time(e) * 1e3


def med(g):
    return statistics.median(one(g) for _ in range(args.reps))


dummy = torch.zeros(1, device="cuda")
ov = med(graph_of(lambda: dummy.add_(1)))
print(f"SMs={SM} empty-graph replay {ov:.2f} us (subtracted)", flush=True)

MIB = 2**20
SIZES = [8, 16, 27, 36, 48, 64]  # MiB ~ 8,17,28,38,50,67 MB
bufs = {m: torch.randint(0, 255, (m * MIB,), device="cuda", dtype=torch.uint8) for m in SIZES}

cfgs = []
for cps in (1, 2, 4, 8):
    for u in (4, 8, 16):
        cfgs.append((f"ldg cps={cps} U={u} stride", lambda t, c=cps, u=u: ext.ldg(t, SM * c, u, 0, sink)))
        if u != 4:
            cfgs.append((f"ldg cps={cps} U={u} contig", lambda t, c=cps, u=u: ext.ldg(t, SM * c, u, 1, sink)))
for cps in (1, 2):
    for ch, st, wp in ((4096, 8, 1), (8192, 8, 1), (16384, 4, 1), (16384, 8, 1), (32768, 2, 1), (32768, 4, 1), (32768, 6, 1),
                       (4096, 4, 4), (8192, 4, 4), (8192, 2, 4), (16384, 2, 4), (4096, 2, 8), (8192, 2, 8)):
        if cps * wp * st * ch > 227 * 1024:
            continue
        for inter in (0, 1):
            cfgs.append((f"bulk cps={cps} ch={ch} st={st} warps={wp} {'inter' if inter else 'contig'}",
                         lambda t, c=cps, ch=ch, st=st, wp=wp, i=inter: ext.bulk(t, SM * c, ch, st, wp, i)))
for cps in (1, 2):
    for bi, br, st in ((256, 32, 8), (256, 64, 4), (256, 64, 8), (256, 128, 4), (128, 128, 4), (128, 64, 8)):
        if cps * st * bi * br > 227 * 1024:
            continue
        for inter in (0, 1):
            cfgs.append((f"tma cps={cps} box={bi}x{br} st={st} {'inter' if inter else 'contig'}",
                         lambda t, c=cps, bi=bi, br=br, st=st, i=inter: ext.tma(t, SM * c, bi, br, st, i)))

res = {}
for name, fn in cfgs:
    for m in SIZES:
        try:
            t = med(graph_of(lambda: fn(bufs[m]))) - ov
        except Exception as ex:
            print(f"SKIP {name} {m}MiB: {str(ex)[:80]}", flush=True)
            break
        res[(name, m)] = t
    else:
        print(f"{name:52s} " + " ".join(f"{res[(name, m)]:6.2f}" for m in SIZES), flush=True)

print("\nBEST per size (us, GB/s, cfg)")
fit = []
for m in SIZES:
    cand = [(res[(n, m)], n) for n, _ in cfgs if (n, m) in res]
    cand.sort()
    b = m * MIB
    print(f"{m:3d}MiB {b/1e6:6.1f}MB: " + " | ".join(f"{t:.2f}us {b/t/1e3:.0f}GB/s {n}" for t, n in cand[:3]), flush=True)
    for fam in ("ldg", "bulk", "tma"):
        c = [(t, n) for t, n in cand if n.startswith(fam)]
        if c:
            print(f"      best {fam}: {c[0][0]:.2f}us {b/c[0][0]/1e3:.0f}GB/s {c[0][1]}")
    fit.append((b / 1e6, cand[0][0]))
xs, ys = [f[0] for f in fit], [f[1] for f in fit]
mx, my = statistics.mean(xs), statistics.mean(ys)
slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
print(f"FIT best-of-all: t = {my - slope * mx:.2f} us + MB/{1e-3/slope:.0f} GB/s -> asymptotic {1e-3/slope:.0f} GB/s")
os._exit(0)
