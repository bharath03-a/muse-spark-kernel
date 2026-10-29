SLICE_CPP = "void launch_slice(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t nb, int64_t nw);"
SLICE_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
constexpr int SK = 4096;
template <int NB, int NW>
__global__ void __launch_bounds__(NW * 32, 1)
slice_mma(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ out,
          int m, int N, int xs) {
  constexpr int CPW = SK / NW, IT = CPW / 32;
  __shared__ float red[NB][NW][8][8];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int row0 = blockIdx.x * NB * 8;
  const int c0 = warp * CPW + t * 8;
  uint4 xv[IT];
  #pragma unroll
  for (int i = 0; i < IT; ++i) {
    if (g < m) xv[i] = __ldg(reinterpret_cast<const uint4*>(x + (size_t)g * xs + c0 + i * 32));
    else xv[i] = make_uint4(0, 0, 0, 0);
  }
  float acc[NB][4];
  #pragma unroll
  for (int nb = 0; nb < NB; ++nb) {
    uint4 wv[IT];
    const __nv_bfloat16* wp = w + (size_t)(row0 + nb * 8 + g) * SK + c0;
    #pragma unroll
    for (int i = 0; i < IT; ++i) asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3},[%4];" : "=r"(wv[i].x), "=r"(wv[i].y), "=r"(wv[i].z), "=r"(wv[i].w) : "l"(wp + i * 32));
    #pragma unroll
    for (int j = 0; j < 4; ++j) acc[nb][j] = 0.f;
    #pragma unroll
    for (int i = 0; i < IT; ++i) {
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                   : "+f"(acc[nb][0]), "+f"(acc[nb][1]), "+f"(acc[nb][2]), "+f"(acc[nb][3])
                   : "r"(xv[i].x), "r"(0), "r"(xv[i].y), "r"(0), "r"(wv[i].x), "r"(wv[i].y));
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                   : "+f"(acc[nb][0]), "+f"(acc[nb][1]), "+f"(acc[nb][2]), "+f"(acc[nb][3])
                   : "r"(xv[i].z), "r"(0), "r"(xv[i].w), "r"(0), "r"(wv[i].z), "r"(wv[i].w));
    }
    red[nb][warp][g][2 * t] = acc[nb][0];
    red[nb][warp][g][2 * t + 1] = acc[nb][1];
  }
  __syncthreads();
  if (threadIdx.x < NB * 64) {
    const int nb = threadIdx.x >> 6, r = threadIdx.x & 63, tok = r >> 3, n = r & 7;
    float s = 0.f;
    #pragma unroll
    for (int wi = 0; wi < NW; ++wi) s += red[nb][wi][tok][n];
    if (tok < m) out[(size_t)tok * N + row0 + nb * 8 + n] = __float2bfloat16(s);
  }
}
template <int NB, int NW>
void run(torch::Tensor x, torch::Tensor w, torch::Tensor out) {
  const int N = out.size(1);
  slice_mma<NB, NW><<<N / (NB * 8), NW * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const __nv_bfloat16*)x.data_ptr(), (const __nv_bfloat16*)w.data_ptr(), (__nv_bfloat16*)out.data_ptr(), (int)x.size(0), N, (int)x.stride(0));
  C10_CUDA_CHECK(cudaGetLastError());
}
void launch_slice(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t nb, int64_t nw) {
  TORCH_CHECK(x.size(1) == SK && w.size(1) == SK && x.size(0) <= 8 && x.stride(0) % 8 == 0, "slice: shape");
  if (nb == 1 && nw == 8) run<1, 8>(x, w, out);
  else if (nb == 1 && nw == 16) run<1, 16>(x, w, out);
  else if (nb == 1 && nw == 4) run<1, 4>(x, w, out);
  else if (nb == 2 && nw == 8) run<2, 8>(x, w, out);
  else if (nb == 2 && nw == 16) run<2, 16>(x, w, out);
  else TORCH_CHECK(false, "slice: config");
}
"""
