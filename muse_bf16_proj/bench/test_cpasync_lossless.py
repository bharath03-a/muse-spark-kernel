import time
import math
import torch
import reference
from common import weights
from torch.utils.cpp_extension import load_inline
from exp_lossless_m1 import pack_weights_lossless_12

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda.h>

using bf16 = __nv_bfloat16;
using bf162 = __nv_bfloat162;

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src) {
  const uint32_t smem = smem_u32(smem_dst);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(smem), "l"(gmem_src) : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

__device__ __forceinline__ void decode_group(uint32_t g0, uint32_t g1, uint32_t g2, uint32_t ebp,
                                            bf162 (&dst)[4]) {
  const uint32_t p0 = g0 & 0x0FFF0FFF;
  const uint32_t p1 = g1 & 0x0FFF0FFF;
  const uint32_t p2 = g2 & 0x0FFF0FFF;
  const uint32_t c6 = ((g0 >> 12) & 0xF) | (((g1 >> 12) & 0xF) << 4) | (((g2 >> 12) & 0xF) << 8);
  const uint32_t c7 = ((g0 >> 28) & 0xF) | (((g1 >> 28) & 0xF) << 4) | (((g2 >> 28) & 0xF) << 8);
  const uint32_t p3 = c6 | (c7 << 16);

  const uint32_t b0 = ((p0 & 0x08000800) << 4) | ((p0 & 0x07FF07FF) + ebp);
  const uint32_t b1 = ((p1 & 0x08000800) << 4) | ((p1 & 0x07FF07FF) + ebp);
  const uint32_t b2 = ((p2 & 0x08000800) << 4) | ((p2 & 0x07FF07FF) + ebp);
  const uint32_t b3 = ((p3 & 0x08000800) << 4) | ((p3 & 0x07FF07FF) + ebp);

  dst[0] = *reinterpret_cast<const bf162*>(&b0);
  dst[1] = *reinterpret_cast<const bf162*>(&b1);
  dst[2] = *reinterpret_cast<const bf162*>(&b2);
  dst[3] = *reinterpret_cast<const bf162*>(&b3);
}

// Asynchronously Pipelined 12-bit Lossless GEMV (m=1) using cp.async
// 8 warps (256 threads). Block computes 8 rows.
// 2-stage double buffering in shared memory.
template <int K_TOTAL>
__global__ void __launch_bounds__(256)
cpasync_lossless_gemv_m1(const uint4* __restrict__ w_packed,
                         const bf16* __restrict__ x,
                         bf16* __restrict__ out,
                         const int32_t* __restrict__ csr_row_ptr,
                         const int32_t* __restrict__ csr_cols,
                         const float* __restrict__ csr_diff,
                         int n, uint32_t ebp) {
  constexpr int WARPS = 8;
  constexpr int CHUNK_K = 256;      // 256 weights per pass
  constexpr int CHUNK_U4 = 24;     // 24 uint4 = 384 bytes
  constexpr int K_STEPS = K_TOTAL / CHUNK_K;

  __shared__ bf16 sx[K_TOTAL];
  __shared__ uint4 smem_w[2][WARPS][CHUNK_U4]; // 2 stages * 8 warps * 24 uint4 = 6 KB

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int row = blockIdx.x * WARPS + warp;

  // 1. Stage x in shared memory cooperatively
  for (int i = tid * 8; i < K_TOTAL; i += 256 * 8) {
    *reinterpret_cast<float4*>(&sx[i]) = *reinterpret_cast<const float4*>(&x[i]);
  }
  __syncthreads();

  const int row_stride_u4 = (K_TOTAL / 32) * 3;
  const uint4* row_ptr = w_packed + (size_t)row * row_stride_u4;

  // Pre-load stage 0 with cp.async
  if (row < n && lane < CHUNK_U4) {
    cp_async_16(&smem_w[0][warp][lane], &row_ptr[lane]);
  }
  cp_async_commit();

  float acc = 0.f;

  if (row < n) {
    for (int step = 0; step < K_STEPS; ++step) {
      const int s = step & 1;
      const int next_s = 1 - s;
      const int next_step = step + 1;

      // Issue prefetch for next stage
      if (next_step < K_STEPS && lane < CHUNK_U4) {
        cp_async_16(&smem_w[next_s][warp][lane], &row_ptr[next_step * CHUNK_U4 + lane]);
      }
      cp_async_commit();

      // Wait for current stage
      cp_async_wait<1>();
      __syncwarp();

      // Decode and compute current stage
      const uint32_t* u32_ptr = reinterpret_cast<const uint32_t*>(&smem_w[s][warp][0]);
      const uint32_t g0 = u32_ptr[lane * 3 + 0];
      const uint32_t g1 = u32_ptr[lane * 3 + 1];
      const uint32_t g2 = u32_ptr[lane * 3 + 2];

      bf162 wb[4];
      decode_group(g0, g1, g2, ebp, wb);

      const int k_base = step * CHUNK_K + lane * 8;
      const bf162* xb = reinterpret_cast<const bf162*>(&sx[k_base]);

      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 wf = __bfloat1622float2(wb[i]);
        const float2 xf = __bfloat1622float2(xb[i]);
        acc += wf.x * xf.x + wf.y * xf.y;
      }
    }
    cp_async_wait<0>();

    // Warp reduction
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
      acc += __shfl_down_sync(0xffffffff, acc, mask);
    }

    if (lane == 0) {
      // CSR Outlier Correction
      const int start = csr_row_ptr[row];
      const int end = csr_row_ptr[row + 1];
      for (int idx = start; idx < end; ++idx) {
        const int col = csr_cols[idx];
        const float diff = csr_diff[idx];
        acc += diff * __bfloat162float(sx[col]);
      }
      out[row] = __float2bfloat16(acc);
    }
  }
}

void launch_cpasync_lossless(torch::Tensor w_packed, torch::Tensor x, torch::Tensor out,
                            torch::Tensor csr_row_ptr, torch::Tensor csr_cols, torch::Tensor csr_diff,
                            int64_t ebp) {
  const int n = out.size(0);
  const int k = x.size(0);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const int blocks = (n + 7) / 8;

  if (k == 8192) {
    cpasync_lossless_gemv_m1<8192><<<blocks, 256, 0, stream>>>(
        reinterpret_cast<const uint4*>(w_packed.data_ptr()),
        static_cast<const bf16*>(x.data_ptr()),
        static_cast<bf16*>(out.data_ptr()),
        static_cast<const int32_t*>(csr_row_ptr.data_ptr()),
        static_cast<const int32_t*>(csr_cols.data_ptr()),
        static_cast<const float*>(csr_diff.data_ptr()),
        n, (uint32_t)ebp);
  } else if (k == 4096) {
    cpasync_lossless_gemv_m1<4096><<<blocks, 256, 0, stream>>>(
        reinterpret_cast<const uint4*>(w_packed.data_ptr()),
        static_cast<const bf16*>(x.data_ptr()),
        static_cast<bf16*>(out.data_ptr()),
        static_cast<const int32_t*>(csr_row_ptr.data_ptr()),
        static_cast<const int32_t*>(csr_cols.data_ptr()),
        static_cast<const float*>(csr_diff.data_ptr()),
        n, (uint32_t)ebp);
  } else if (k == 1024) {
    cpasync_lossless_gemv_m1<1024><<<blocks, 256, 0, stream>>>(
        reinterpret_cast<const uint4*>(w_packed.data_ptr()),
        static_cast<const bf16*>(x.data_ptr()),
        static_cast<bf16*>(out.data_ptr()),
        static_cast<const int32_t*>(csr_row_ptr.data_ptr()),
        static_cast<const int32_t*>(csr_cols.data_ptr()),
        static_cast<const float*>(csr_diff.data_ptr()),
        n, (uint32_t)ebp);
  }
}
"""

CPP_SRC = """
void launch_cpasync_lossless(torch::Tensor w_packed, torch::Tensor x, torch::Tensor out,
                            torch::Tensor csr_row_ptr, torch::Tensor csr_cols, torch::Tensor csr_diff,
                            int64_t ebp);
"""

def bench_fn(fn, *args, reps=100):
    for _ in range(5):
        fn(*args)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn(*args)
    torch.cuda.synchronize()
    flush = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    times = []
    for _ in range(reps):
        flush.fill_(1)
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        g.replay()
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0 - 4.0)
    times.sort()
    return times[len(times) // 2], out

def main():
    print("=== Testing cp.async Double-Buffered 12-Bit Lossless GEMV (m=1) ===")
    ext = load_inline(
        name="cpasync_lossless_m1_ext",
        cpp_sources=CPP_SRC,
        cuda_sources=CUDA_SRC,
        functions=["launch_cpasync_lossless"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
        extra_cflags=["-O3"],
    )

    w_dict = weights()
    layers = [
        ("pooler", 4096, 8192),
        ("unpooler", 1024, 4096),
        ("qkv_gate", 2304, 8192),
        ("o", 8192, 1024),
    ]

    for layer_name, n, k in layers:
        w = w_dict[layer_name]
        if layer_name == "unpooler":
            w = w[:1024].cuda().contiguous()
        else:
            w = w.cuda().contiguous()

        w_packed, csr_row_ptr, csr_cols, csr_diff, ebp, e_base = pack_weights_lossless_12(w)
        x = (torch.randn(k, dtype=torch.bfloat16, device="cuda") / math.sqrt(k)).contiguous()

        t_base, out_base = bench_fn(torch.matmul, x, w.T)

        out_lossless = torch.empty(n, dtype=torch.bfloat16, device="cuda")
        t_lossless, _ = bench_fn(ext.launch_cpasync_lossless, w_packed, x, out_lossless,
                                 csr_row_ptr, csr_cols, csr_diff, ebp)

        err = torch.norm((out_base - out_lossless).float()) / torch.norm(out_base.float())
        speedup = t_base / t_lossless
        bw_phys = (n * k * 1.5) / (t_lossless * 1e-6) / 1e12
        bw_eff = (n * k * 2.0) / (t_lossless * 1e-6) / 1e12

        print(f"\n--- {layer_name} (N={n}, K={k}) ---")
        print(f"  Baseline: {t_base:.2f} us")
        print(f"  cp.async Lossless: {t_lossless:.2f} us | Speedup: {speedup:.3f}x | Rel Diff: {err:.2e}")
        print(f"  Physical Bandwidth: {bw_phys:.2f} TB/s | Effective Bandwidth: {bw_eff:.2f} TB/s")

if __name__ == "__main__":
    main()
