"""W-stream floor: pure 1D bulk-copy read of N bytes cold, all SMs, no compute."""
import statistics, torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
__device__ __forceinline__ uint32_t su(const void* p){return (uint32_t)__cvta_generic_to_shared(p);}
template <int ST, int CH>
__global__ void __launch_bounds__(32,1) stream_k(const char* p, long per_cta, int* sink) {
  extern __shared__ __align__(128) unsigned char sm[];
  __shared__ __align__(8) uint64_t bar[ST];
  if (threadIdx.x == 0) { for (int s=0;s<ST;++s) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"::"r"(su(&bar[s]))); asm volatile("fence.mbarrier_init.release.cluster;"); }
  __syncwarp();
  const char* base = p + (long)blockIdx.x * per_cta;
  int n = per_cta / CH;
  if (threadIdx.x == 0) {
    for (int i = 0; i < n + ST - 1; ++i) {
      if (i >= ST) {  // wait slot i-ST complete before reuse (data landed)
        int j = i - ST; int s = j % ST;
        asm volatile("{.reg .pred P; W: mbarrier.try_wait.parity.shared::cta.b64 P,[%0],%1; @P bra D; bra W; D:}"::"r"(su(&bar[s])),"r"((j/ST)&1));
      }
      if (i < n) {
        int s = i % ST;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(su(&bar[s])),"r"(CH));
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"
                     ::"r"(su(sm + s*CH)),"l"(base + (long)i*CH),"r"(CH),"r"(su(&bar[s])):"memory");
      }
    }
    // drain: last ST-1 not waited above except those <n; wait all remaining
    for (int j = n - ST + 1 > 0 ? n - ST + 1 : 0; j < n; ++j) {
      if (j < 0) continue;
      int s = j % ST;
      asm volatile("{.reg .pred P; W2: mbarrier.try_wait.parity.shared::cta.b64 P,[%0],%1; @P bra D2; bra W2; D2:}"::"r"(su(&bar[s])),"r"((j/ST)&1));
    }
  }
}
template <int ST, int CH>
void run(torch::Tensor t, int64_t ctas) {
  long per = t.numel() / ctas; per -= per % CH;
  int smem = ST*CH;
  cudaFuncSetAttribute(stream_k<ST,CH>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  stream_k<ST,CH><<<ctas, 32, smem, at::cuda::getCurrentCUDAStream()>>>((const char*)t.data_ptr(), per, nullptr);
}
void stream(torch::Tensor t, int64_t ctas, int64_t st, int64_t ch) {
  if (st==8 && ch==16384) run<8,16384>(t,ctas);
  else if (st==12 && ch==16384) run<12,16384>(t,ctas);
  else if (st==6 && ch==32768) run<6,32768>(t,ctas);
  else if (st==4 && ch==32768) run<4,32768>(t,ctas);
  else if (st==16 && ch==8192) run<16,8192>(t,ctas);
  else if (st==24 && ch==8192) run<24,8192>(t,ctas);
  else TORCH_CHECK(false, "cfg");
}
"""
ext = load_inline(name="r6floor1", cpp_sources="void stream(torch::Tensor t, int64_t ctas, int64_t st, int64_t ch);",
                  cuda_sources=SRC, functions=["stream"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
flush = torch.empty(64 * 2**20, device="cuda", dtype=torch.float32).normal_()
def graph_of(fn):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    torch.cuda.synchronize(); return g
def t(g):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    flush.sum(); s.record(); g.replay(); e.record(); e.synchronize(); return s.elapsed_time(e) * 1e3
for mb, name in ((16.8, "o"), (37.7, "qkv"), (67.1, "pool/unpool")):
    buf = torch.empty(int(mb * 2**20 * 1.0), device="cuda", dtype=torch.uint8)
    for ctas in (66, 132):
        for st, ch in ((8, 16384), (12, 16384), (6, 32768), (16, 8192), (24, 8192)):
            g = graph_of(lambda: ext.stream(buf, ctas, st, ch))
            v = [t(g) for _ in range(60)]
            print(f"{name} {mb}MB ctas={ctas} st={st} ch={ch}: {statistics.median(v) - 4.0:.2f} us", flush=True)
