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

__device__ __forceinline__ void mbar_init(uint64_t* b, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(b)), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
               ::"r"(smem_u32(b)), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred P1;\n LAB_WAIT:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      " @P1 bra DONE;\n bra LAB_WAIT;\n DONE:\n}\n"
      ::"r"(smem_u32(b)), "r"(parity) : "memory");
}

__device__ __forceinline__ void tma_load_2d_bytes(void* smem_dst, const CUtensorMap* map, int byte_col, int row_idx,
                                                  uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1, {%2, %3}], [%4];\n"
      ::"r"(smem_u32(smem_dst)), "l"(map), "r"(byte_col), "r"(row_idx), "r"(smem_u32(bar))
      : "memory");
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

// 2D TMA Asynchronously Pipelined 12-bit Lossless GEMV (m=1)
// 8 warps (256 threads). Each warp computes 1 row. Block computes 8 rows.
// TMA fetches packed chunks of 384 bytes (256 weights) per stage for all 8 rows.
// Stages = 4 ring buffer in smem.
template <int K_TOTAL>
__global__ void __launch_bounds__(256)
tma_lossless_gemv_m1(const __grid_constant__ CUtensorMap w_map,
                     const bf16* __restrict__ x,
                     bf16* __restrict__ out,
                     const int32_t* __restrict__ csr_row_ptr,
                     const int32_t* __restrict__ csr_cols,
                     const float* __restrict__ csr_diff,
                     int n, uint32_t ebp) {
  constexpr int WARPS = 8;
  constexpr int STAGES = 4;
  constexpr int CHUNK_K = 256;          // 256 weights per step
  constexpr int CHUNK_BYTES = 384;      // 256 * 1.5 = 384 bytes per row
  constexpr int STAGE_BYTES = WARPS * CHUNK_BYTES; // 8 * 384 = 3072 bytes per stage
  constexpr int K_STEPS = K_TOTAL / CHUNK_K;

  extern __shared__ __align__(1024) unsigned char smem[];
  bf16* sx = reinterpret_cast<bf16*>(smem);
  constexpr size_t X_BYTES = K_TOTAL * sizeof(bf16); // 16 KB for K=8192
  uint64_t* ready = reinterpret_cast<uint64_t*>(smem + X_BYTES);
  constexpr size_t MBAR_BYTES = ((STAGES * sizeof(uint64_t) + 127) & ~127);
  unsigned char* stage_base = smem + X_BYTES + MBAR_BYTES;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int row0 = blockIdx.x * WARPS;
  const int row = row0 + warp;

  // 1. Stage activation x in shared memory cooperatively
  for (int i = tid * 8; i < K_TOTAL; i += 256 * 8) {
    *reinterpret_cast<float4*>(&sx[i]) = *reinterpret_cast<const float4*>(&x[i]);
  }

  // 2. Init barriers (1 arrival: thread 0)
  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) mbar_init(&ready[s], 1);
  }
  __syncthreads();

  // Helper: issue TMA load for 1 stage
  auto issue_stage = [&](int s, int step) {
    mbar_expect_tx(&ready[s], STAGE_BYTES);
    const int byte_col = step * CHUNK_BYTES;
    unsigned char* dst = stage_base + s * STAGE_BYTES;
    tma_load_2d_bytes(dst, &w_map, byte_col, row0, &ready[s]);
  };

  // Pre-load stages
  if (tid == 0) {
    constexpr int PRE = STAGES - 1 < K_STEPS ? STAGES - 1 : K_STEPS;
    for (int s = 0; s < PRE; ++s) issue_stage(s, s);
  }

  float acc = 0.f;

  for (int step = 0; step < K_STEPS; ++step) {
    const int s = step % STAGES;
    const uint32_t parity = (step / STAGES) & 1;
    mbar_wait(&ready[s], parity);

    // Each warp processes its row from this stage in smem
    const unsigned char* warp_stage = stage_base + s * STAGE_BYTES + warp * CHUNK_BYTES;
    const uint32_t* u32_ptr = reinterpret_cast<const uint32_t*>(warp_stage);

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

    // Issue next TMA stage
    const int next_step = step + STAGES - 1;
    if (next_step < K_STEPS) {
      __syncthreads();
      if (tid == 0) {
        issue_stage(next_step % STAGES, next_step);
      }
    }
  }

  // Warp reduction
  #pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    acc += __shfl_down_sync(0xffffffff, acc, mask);
  }

  if (lane == 0 && row < n) {
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

inline CUtensorMap make_packed_tma_2d(const void* ptr, int rows, int bytes_per_row) {
  CUtensorMap map{};
  cuuint64_t gdim[2] = {(cuuint64_t)bytes_per_row, (cuuint64_t)rows};
  cuuint64_t gstride[1] = {(cuuint64_t)bytes_per_row};
  // Box dimension: 384 bytes x 8 rows
  cuuint32_t bdim[2] = {384u, 8u};
  cuuint32_t estride[2] = {1u, 1u};
  CUresult res = cuTensorMapEncodeTiled(
      &map,
      CU_TENSOR_MAP_DATA_TYPE_UINT8,
      2,
      const_cast<void*>(ptr),
      gdim,
      gstride,
      bdim,
      estride,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(res == CUDA_SUCCESS, "make_packed_tma_2d failed with error ", (int)res);
  return map;
}

void launch_tma_lossless(torch::Tensor w_packed, torch::Tensor x, torch::Tensor out,
                        torch::Tensor csr_row_ptr, torch::Tensor csr_cols, torch::Tensor csr_diff,
                        int64_t ebp) {
  const int n = out.size(0);
  const int k = x.size(0);
  const int bytes_per_row = (k / 8) * 12;
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const int blocks = (n + 7) / 8;

  CUtensorMap w_map = make_packed_tma_2d(w_packed.data_ptr(), n, bytes_per_row);
  constexpr size_t X_BYTES = 8192 * sizeof(bf16);
  constexpr size_t MBAR_BYTES = 128;
  constexpr size_t STAGE_BYTES = 8 * 384;
  constexpr size_t TOTAL_SMEM = X_BYTES + MBAR_BYTES + 4 * STAGE_BYTES;

  cudaFuncSetAttribute(reinterpret_cast<const void*>(tma_lossless_gemv_m1<8192>),
                      cudaFuncAttributeMaxDynamicSharedMemorySize, TOTAL_SMEM);

  if (k == 8192) {
    tma_lossless_gemv_m1<8192><<<blocks, 256, TOTAL_SMEM, stream>>>(
        w_map, static_cast<const bf16*>(x.data_ptr()), static_cast<bf16*>(out.data_ptr()),
        static_cast<const int32_t*>(csr_row_ptr.data_ptr()), static_cast<const int32_t*>(csr_cols.data_ptr()),
        static_cast<const float*>(csr_diff.data_ptr()), n, (uint32_t)ebp);
  } else if (k == 4096) {
    tma_lossless_gemv_m1<4096><<<blocks, 256, TOTAL_SMEM, stream>>>(
        w_map, static_cast<const bf16*>(x.data_ptr()), static_cast<bf16*>(out.data_ptr()),
        static_cast<const int32_t*>(csr_row_ptr.data_ptr()), static_cast<const int32_t*>(csr_cols.data_ptr()),
        static_cast<const float*>(csr_diff.data_ptr()), n, (uint32_t)ebp);
  } else if (k == 1024) {
    tma_lossless_gemv_m1<1024><<<blocks, 256, TOTAL_SMEM, stream>>>(
        w_map, static_cast<const bf16*>(x.data_ptr()), static_cast<bf16*>(out.data_ptr()),
        static_cast<const int32_t*>(csr_row_ptr.data_ptr()), static_cast<const int32_t*>(csr_cols.data_ptr()),
        static_cast<const float*>(csr_diff.data_ptr()), n, (uint32_t)ebp);
  }
}
"""

CPP_SRC = """
void launch_tma_lossless(torch::Tensor w_packed, torch::Tensor x, torch::Tensor out,
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
    print("=== Testing 2D TMA Asynchronous Pipelined 12-Bit Lossless GEMV (m=1) ===")
    ext = load_inline(
        name="tma_lossless_m1_ext",
        cpp_sources=CPP_SRC,
        cuda_sources=CUDA_SRC,
        functions=["launch_tma_lossless"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
        extra_cflags=["-O3"],
        extra_ldflags=["-lcuda"],
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
        t_lossless, _ = bench_fn(ext.launch_tma_lossless, w_packed, x, out_lossless,
                                 csr_row_ptr, csr_cols, csr_diff, ebp)

        err = torch.norm((out_base - out_lossless).float()) / torch.norm(out_base.float())
        speedup = t_base / t_lossless
        bw_phys = (n * k * 1.5) / (t_lossless * 1e-6) / 1e12
        bw_eff = (n * k * 2.0) / (t_lossless * 1e-6) / 1e12

        print(f"\n--- {layer_name} (N={n}, K={k}) ---")
        print(f"  Baseline: {t_base:.2f} us")
        print(f"  TMA Lossless: {t_lossless:.2f} us | Speedup: {speedup:.3f}x | Rel Diff: {err:.2e}")
        print(f"  Physical Bandwidth: {bw_phys:.2f} TB/s | Effective Bandwidth: {bw_eff:.2f} TB/s")

if __name__ == "__main__":
    main()
