import time
import torch
import reference
from common import weights
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda.h>

using bf16 = __nv_bfloat16;
using bf162 = __nv_bfloat162;

template <bool F32, int R, int MT, int PER>
__global__ void __launch_bounds__(256, 2)
row_gemv(const bf16* __restrict__ x, const void* __restrict__ w_void, void* __restrict__ out_void,
         int m, int n, int k, int stride_xm) {
  constexpr int THREADS_PER_CTA = 256;
  const int tid = threadIdx.x;
  const int row0 = blockIdx.x * R;
  if (row0 >= n) return;

  uint4 wv[R][PER];
  if constexpr (F32) {
    const float* w = static_cast<const float*>(w_void);
    for (int r = 0; r < R; ++r) {
      for (int p = 0; p < PER; ++p) {
        wv[r][p] = *reinterpret_cast<const uint4*>(w + (row0 + r) * k + p * THREADS_PER_CTA * 4 + tid * 4);
      }
    }
  } else {
    const bf16* w = static_cast<const bf16*>(w_void);
    for (int r = 0; r < R; ++r) {
      for (int p = 0; p < PER; ++p) {
        wv[r][p] = *reinterpret_cast<const uint4*>(w + (row0 + r) * k + p * THREADS_PER_CTA * 8 + tid * 8);
      }
    }
  }

  for (int t = 0; t < m; ++t) {
    const bf16* xt = x + t * stride_xm;
    float sum[R] = {0.0f};

    if constexpr (F32) {
      for (int p = 0; p < PER; ++p) {
        const int k_base = p * THREADS_PER_CTA * 4 + tid * 4;
        const uint2 xu = *reinterpret_cast<const uint2*>(xt + k_base);
        const bf16* xbp = reinterpret_cast<const bf16*>(&xu);
        for (int r = 0; r < R; ++r) {
          const float* wp = reinterpret_cast<const float*>(&wv[r][p]);
          for (int i = 0; i < 4; ++i) sum[r] += wp[i] * __bfloat162float(xbp[i]);
        }
      }
    } else {
      for (int p = 0; p < PER; ++p) {
        const int k_base = p * THREADS_PER_CTA * 8 + tid * 8;
        const uint4 xu = *reinterpret_cast<const uint4*>(xt + k_base);
        const uint32_t* xp32 = reinterpret_cast<const uint32_t*>(&xu);
        for (int r = 0; r < R; ++r) {
          const uint32_t* wp32 = reinterpret_cast<const uint32_t*>(&wv[r][p]);
          for (int i = 0; i < 4; ++i) {
            bf162 a = *reinterpret_cast<const bf162*>(&wp32[i]);
            bf162 b = *reinterpret_cast<const bf162*>(&xp32[i]);
            sum[r] += __bfloat162float(a.x) * __bfloat162float(b.x) + __bfloat162float(a.y) * __bfloat162float(b.y);
          }
        }
      }
    }

    for (int r = 0; r < R; ++r) {
      for (int mask = 16; mask > 0; mask >>= 1) sum[r] += __shfl_xor_sync(0xffffffff, sum[r], mask);
    }

    __shared__ float smem_red[R][8];
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (lane == 0) {
      for (int r = 0; r < R; ++r) smem_red[r][warp] = sum[r];
    }
    __syncthreads();

    if (tid < 8) {
      for (int r = 0; r < R; ++r) {
        float val = smem_red[r][tid];
        for (int mask = 4; mask > 0; mask >>= 1) val += __shfl_xor_sync(0xff, val, mask);
        if (tid == 0 && row0 + r < n) {
          if constexpr (F32) {
            static_cast<float*>(out_void)[t * n + row0 + r] = val;
          } else {
            static_cast<bf16*>(out_void)[t * n + row0 + r] = __float2bfloat16(val);
          }
        }
      }
    }
    __syncthreads();
  }
}

void launch_row_gemv_k8192(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t rows) {
  const int m = x.size(0), n = out.size(1), k = x.size(1);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const bf16* xp = static_cast<const bf16*>(x.data_ptr());
  
  if (rows == 1) {
    row_gemv<false, 1, 1, 4><<<n / 1, 256, 0, stream>>>(xp, w.data_ptr(), out.data_ptr(), m, n, k, (int)x.stride(0));
  } else if (rows == 2) {
    row_gemv<false, 2, 1, 4><<<n / 2, 256, 0, stream>>>(xp, w.data_ptr(), out.data_ptr(), m, n, k, (int)x.stride(0));
  } else if (rows == 4) {
    row_gemv<false, 4, 1, 4><<<n / 4, 256, 0, stream>>>(xp, w.data_ptr(), out.data_ptr(), m, n, k, (int)x.stride(0));
  }
}
"""

CPP_SRC = """
void launch_row_gemv_k8192(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t rows);
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
    print("=== Testing row_gemv for K=8192 (pooler, qkv_gate) on m=1 ===")
    ext = load_inline(
        name="test_row_gemv_k8192_ext",
        cpp_sources=CPP_SRC,
        cuda_sources=CUDA_SRC,
        functions=["launch_row_gemv_k8192"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
        extra_cflags=["-O3"],
    )

    w_dict = weights()
    for layer in ("pooler", "qkv_gate"):
        w = w_dict[layer].cuda().contiguous()
        n, k = w.shape
        x = (torch.randn(1, k, dtype=torch.bfloat16, device="cuda") / (k ** 0.5)).contiguous()
        
        t_base, out_base = bench_fn(torch.nn.functional.linear, x, w)
        print(f"\n--- {layer} (M=1, N={n}, K={k}) | Baseline: {t_base:.2f} us ---")
        
        for r in (1, 2, 4):
            out = torch.empty(1, n, dtype=torch.bfloat16, device="cuda")
            t, _ = bench_fn(ext.launch_row_gemv_k8192, x, w, out, r)
            err = torch.norm((out_base - out).float()) / torch.norm(out_base.float())
            speedup = t_base / t
            bw = (n * k * 2) / (t * 1e-6) / 1e12
            print(f"  row_gemv (rows={r}): {t:.2f} us | Speedup: {speedup:.3f}x | BW: {bw:.2f} TB/s | Diff: {err:.2e}")

if __name__ == "__main__":
    main()
