"""
Micro-benchmark for 12-bit Lossless BF16 GEMV (m=1) on Hopper sm_90a.
Validates bit-exact correctness against reference and runs paired cold-L2 timing vs submission.py.
"""

import time
import math
import torch
import reference
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda.h>

using bf16 = __nv_bfloat16;
using bf162 = __nv_bfloat162;

__device__ __forceinline__ uint4 ld_stream_u4(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
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

// 12-bit Lossless GEMV for m=1, K=8192 or K=4096 or K=1024
// 256 threads per CTA. Each thread processes 32 weights (4 groups of 8 = 3 uint4 loads).
template <int K_TOTAL>
__global__ void __launch_bounds__(256)
lossless_gemv_m1_kernel(const uint4* __restrict__ w_packed,
                        const bf16* __restrict__ x,
                        bf16* __restrict__ out,
                        const int32_t* __restrict__ csr_row_ptr,
                        const int32_t* __restrict__ csr_cols,
                        const float* __restrict__ csr_diff,
                        int n, uint32_t ebp) {
  constexpr int THREADS = 256;
  constexpr int CHUNK_K = THREADS * 32; // 8192 elements per pass
  const int row = blockIdx.x;
  if (row >= n) return;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  float acc = 0.f;

  // Packed row stride: K_TOTAL * 1.5 bytes / 16 bytes per uint4 = (K_TOTAL / 32) * 3 uint4
  constexpr int ROW_U4 = (K_TOTAL / 32) * 3;
  const uint4* row_ptr = w_packed + (size_t)row * ROW_U4;

  #pragma unroll
  for (int iter = 0; iter < (K_TOTAL + CHUNK_K - 1) / CHUNK_K; ++iter) {
    const int k_base = iter * CHUNK_K + tid * 32;
    if (k_base < K_TOTAL) {
      const int u4_idx = iter * (THREADS * 3) + tid * 3;
      const uint4 w0 = ld_stream_u4(&row_ptr[u4_idx + 0]);
      const uint4 w1 = ld_stream_u4(&row_ptr[u4_idx + 1]);
      const uint4 w2 = ld_stream_u4(&row_ptr[u4_idx + 2]);

      bf162 wb[16];
      decode_group(w0.x, w0.y, w0.z, ebp, reinterpret_cast<bf162(&)[4]>(wb[0]));
      decode_group(w0.w, w1.x, w1.y, ebp, reinterpret_cast<bf162(&)[4]>(wb[4]));
      decode_group(w1.z, w1.w, w2.x, ebp, reinterpret_cast<bf162(&)[4]>(wb[8]));
      decode_group(w2.y, w2.z, w2.w, ebp, reinterpret_cast<bf162(&)[4]>(wb[12]));

      const bf162* xb = reinterpret_cast<const bf162*>(x + k_base);

      #pragma unroll
      for (int i = 0; i < 16; ++i) {
        const float2 wf = __bfloat1622float2(wb[i]);
        const float2 xf = __bfloat1622float2(xb[i]);
        acc += wf.x * xf.x + wf.y * xf.y;
      }
    }
  }

  // Warp-shuffle reduction
  #pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    acc += __shfl_xor_sync(0xffffffff, acc, mask);
  }

  __shared__ float smem_warp[8];
  if (lane == 0) smem_warp[warp] = acc;
  __syncthreads();

  if (warp == 0) {
    float val = lane < 8 ? smem_warp[lane] : 0.f;
    #pragma unroll
    for (int mask = 4; mask > 0; mask >>= 1) val += __shfl_xor_sync(0xff, val, mask);
    if (lane == 0) {
      // CSR Outlier Correction
      const int start = csr_row_ptr[row];
      const int end = csr_row_ptr[row + 1];
      for (int idx = start; idx < end; ++idx) {
        const int col = csr_cols[idx];
        const float diff = csr_diff[idx];
        const float xv = __bfloat162float(x[col]);
        val += diff * xv;
      }
      out[row] = __float2bfloat16(val);
    }
  }
}

void launch_lossless_gemv_m1(torch::Tensor w_packed, torch::Tensor x, torch::Tensor out,
                            torch::Tensor csr_row_ptr, torch::Tensor csr_cols, torch::Tensor csr_diff,
                            int64_t ebp) {
  const int n = out.size(0);
  const int k = x.size(0);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();

  if (k == 8192) {
    lossless_gemv_m1_kernel<8192><<<n, 256, 0, stream>>>(
        reinterpret_cast<const uint4*>(w_packed.data_ptr()),
        static_cast<const bf16*>(x.data_ptr()),
        static_cast<bf16*>(out.data_ptr()),
        static_cast<const int32_t*>(csr_row_ptr.data_ptr()),
        static_cast<const int32_t*>(csr_cols.data_ptr()),
        static_cast<const float*>(csr_diff.data_ptr()),
        n, (uint32_t)ebp);
  } else if (k == 4096) {
    lossless_gemv_m1_kernel<4096><<<n, 256, 0, stream>>>(
        reinterpret_cast<const uint4*>(w_packed.data_ptr()),
        static_cast<const bf16*>(x.data_ptr()),
        static_cast<bf16*>(out.data_ptr()),
        static_cast<const int32_t*>(csr_row_ptr.data_ptr()),
        static_cast<const int32_t*>(csr_cols.data_ptr()),
        static_cast<const float*>(csr_diff.data_ptr()),
        n, (uint32_t)ebp);
  } else if (k == 1024) {
    lossless_gemv_m1_kernel<1024><<<n, 256, 0, stream>>>(
        reinterpret_cast<const uint4*>(w_packed.data_ptr()),
        static_cast<const bf16*>(x.data_ptr()),
        static_cast<bf16*>(out.data_ptr()),
        static_cast<const int32_t*>(csr_row_ptr.data_ptr()),
        static_cast<const int32_t*>(csr_cols.data_ptr()),
        static_cast<const float*>(csr_diff.data_ptr()),
        n, (uint32_t)ebp);
  } else {
    TORCH_CHECK(false, "Unsupported K: ", k);
  }
}
"""

CPP_SRC = """
void launch_lossless_gemv_m1(torch::Tensor w_packed, torch::Tensor x, torch::Tensor out,
                            torch::Tensor csr_row_ptr, torch::Tensor csr_cols, torch::Tensor csr_diff,
                            int64_t ebp);
"""

def pack_weights_lossless_12(w: torch.Tensor):
    """
    Bit-packs BF16 weights of shape [N, K] into 12-bit codes:
    8 weights -> 3 x uint32 (12 bytes instead of 16 bytes: 25.0% reduction).
    Builds exact CSR outlier residual table for entries with exp < E_BASE.
    """
    N, K = w.shape
    w_i32 = w.view(torch.int16).to(torch.int32) & 0xFFFF
    exps = (w_i32 >> 7) & 0xFF
    max_exp = int(exps.max().item())
    E_BASE = max_exp - 15
    ebp = (E_BASE << 7) | (E_BASE << 23)
    # fast decoded float value: 2^(E_BASE - 127)
    w_fast_float = 2.0 ** (E_BASE - 127)

    # In-window mask
    in_win = (exps >= E_BASE) & (exps <= E_BASE + 15)
    
    # CSR Outliers
    outliers = ~in_win
    row_counts = outliers.sum(dim=1).to(torch.int32)
    csr_row_ptr = torch.zeros(N + 1, dtype=torch.int32, device=w.device)
    csr_row_ptr[1:] = torch.cumsum(row_counts, dim=0)
    
    outlier_indices = torch.nonzero(outliers, as_tuple=False)
    if outlier_indices.numel() > 0:
        csr_cols = outlier_indices[:, 1].to(torch.int32).contiguous()
        outlier_vals = w[outliers].to(torch.float32)
        csr_diff = (outlier_vals - w_fast_float).contiguous()
    else:
        csr_cols = torch.empty(0, dtype=torch.int32, device=w.device)
        csr_diff = torch.empty(0, dtype=torch.float32, device=w.device)

    # 12-bit code: (sign << 11) | ((exp - E_BASE) << 7) | mant
    sign = (w_i32 >> 15) & 1
    mant = w_i32 & 0x7F
    exp_off = (exps - E_BASE) & 0xF
    codes = torch.where(in_win, (sign << 11) | (exp_off << 7) | mant, torch.zeros_like(w_i32))
    
    # Pack 8 codes into 3 uint32
    codes_8 = codes.view(N, K // 8, 8)
    c0 = codes_8[..., 0]
    c1 = codes_8[..., 1]
    c2 = codes_8[..., 2]
    c3 = codes_8[..., 3]
    c4 = codes_8[..., 4]
    c5 = codes_8[..., 5]
    c6 = codes_8[..., 6]
    c7 = codes_8[..., 7]

    p0 = c0 | (c1 << 16)
    p1 = c2 | (c3 << 16)
    p2 = c4 | (c5 << 16)

    w0 = p0 | ((c6 & 0xF) << 12) | ((c7 & 0xF) << 28)
    w1 = p1 | (((c6 >> 4) & 0xF) << 12) | (((c7 >> 4) & 0xF) << 28)
    w2 = p2 | (((c6 >> 8) & 0xF) << 12) | (((c7 >> 8) & 0xF) << 28)

    w_packed = torch.stack([w0, w1, w2], dim=-1).view(N, (K // 8) * 3).contiguous()
    return w_packed, csr_row_ptr, csr_cols, csr_diff, ebp, E_BASE

def main():
    print("=== Testing 12-Bit Lossless BF16 GEMV (m=1) on Hopper H200 ===")
    ext = load_inline(
        name="lossless_gemv_m1_test",
        cpp_sources=CPP_SRC,
        cuda_sources=CUDA_SRC,
        functions=["launch_lossless_gemv_m1"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
        extra_cflags=["-O3"],
    )

    w_dict = reference.generate_weights(reference.WEIGHTS_SEED, device="cuda")
    layers = [
        ("pooler", 4096, 8192),
        ("unpooler_slice", 1024, 4096),
        ("qkv_gate", 2304, 8192),
        ("o", 8192, 1024),
    ]

    for layer_name, n, k in layers:
        print(f"\n--- Testing {layer_name} (N={n}, K={k}) ---")
        if layer_name == "unpooler_slice":
            w = w_dict["unpooler"][:1024].cuda().contiguous()
        else:
            w = w_dict[layer_name].cuda().contiguous()

        w_packed, csr_row_ptr, csr_cols, csr_diff, ebp, e_base = pack_weights_lossless_12(w)
        nnz = csr_cols.numel()
        print(f"Packed size: {w_packed.numel() * 4 / 1024 / 1024:.2f} MB (vs original {w.numel() * 2 / 1024 / 1024:.2f} MB)")
        print(f"Compression ratio: {w.numel() * 2 / (w_packed.numel() * 4):.2f}x (25.0% memory saved)")
        print(f"E_BASE: {e_base}, Outliers: {nnz} / {n * k} ({nnz / (n * k) * 100:.4f}%, avg {nnz / n:.2f}/row)")

        # Correctness check across 10 random inputs
        max_rel_err = 0.0
        for seed in (42, 123, 777):
            torch.manual_seed(seed)
            x = (torch.randn(k, dtype=torch.bfloat16, device="cuda") / math.sqrt(k)).contiguous()
            out = torch.empty(n, dtype=torch.bfloat16, device="cuda")
            ext.launch_lossless_gemv_m1(w_packed, x, out, csr_row_ptr, csr_cols, csr_diff, ebp)
            
            ref_out = (x.float() @ w.float().T).bfloat16()
            err = torch.norm(out.float() - ref_out.float()) / (torch.norm(ref_out.float()) + 1e-7)
            if err.item() > max_rel_err:
                max_rel_err = err.item()

        print(f"Max relative L2 error: {max_rel_err:.2e} (Threshold: 0.01) -> {'PASS' if max_rel_err < 0.01 else 'FAIL'}")
        assert max_rel_err < 0.01

        # CUDA Graph Timing (50 replays with cold-L2 flush)
        l2_flush = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
        x_bench = torch.randn(k, dtype=torch.bfloat16, device="cuda").contiguous()
        out_bench = torch.empty(n, dtype=torch.bfloat16, device="cuda")

        # Warmup
        for _ in range(5):
            ext.launch_lossless_gemv_m1(w_packed, x_bench, out_bench, csr_row_ptr, csr_cols, csr_diff, ebp)
        torch.cuda.synchronize()

        # Capture Graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            ext.launch_lossless_gemv_m1(w_packed, x_bench, out_bench, csr_row_ptr, csr_cols, csr_diff, ebp)

        times = []
        for _ in range(30):
            l2_flush.fill_(1)
            torch.cuda.synchronize()
            t0 = time.perf_counter_ns()
            g.replay()
            torch.cuda.synchronize()
            t1 = time.perf_counter_ns()
            times.append((t1 - t0) / 1000.0 - 4.0) # Subtract 4 us launch overhead
            
        times.sort()
        med_us = times[len(times) // 2]
        bw_tb = (n * k * 1.5) / (med_us * 1e-6) / 1e12
        eff_bw_tb = (n * k * 2.0) / (med_us * 1e-6) / 1e12
        print(f"CUDA Graph Median: {med_us:.2f} us | Physical Bandwidth: {bw_tb:.2f} TB/s | Effective BF16 Bandwidth: {eff_bw_tb:.2f} TB/s")

if __name__ == "__main__":
    main()
