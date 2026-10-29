"""r7 ZipGEMM 12-bit: register-direct streaming (no smem ring), x B-fragments straight from L2, mma.sync.

Format = exp_r6_zip_v2.pack (rb=4): per (row-group of 64 rows, 64-k unit, 16-row block) a 1536 B record in
mma A-fragment lane order. Each compute warp streams one contiguous range of records into a D-deep register
ring; one extra warp computes escape fixes (per-row CSR of true weights) concurrently. Split-K across warps (smem reduce) and
across a CS-CTA cluster (DSMEM reduce). MODE 0 = loads only, 1 = loads+decode, 2 = full GEMM.
"""

from __future__ import annotations

ZIP_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>

namespace cg = cooperative_groups;
using bf16 = __nv_bfloat16;

__device__ __forceinline__ void z7_cl_arrive_relaxed() { asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void z7_cl_arrive_release() { asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void z7_cl_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory"); }

__device__ __forceinline__ uint4 z7_ldw(const unsigned char* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

__device__ __forceinline__ uint4 z7_ldx(const bf16* p) {
  uint4 r;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

__device__ __forceinline__ uint32_t z7_prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
  return r;
}

__device__ __forceinline__ uint32_t z7_lop(uint32_t a) {
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(0x87878787u), "r"(0x38383838u));
  return r;
}

__device__ __forceinline__ void z7_mma(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                       uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <int NT>
__device__ __forceinline__ void z7_loadx(uint4 (&xv)[NT][2], const bf16* __restrict__ x, int m, int xstride,
                                         int u, int g, int t) {
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    const int tk = nt * 8 + g;
    if (tk < m) {
      const bf16* p = x + (size_t)tk * xstride + u * 64 + 8 * t;
      xv[nt][0] = z7_ldx(p);
      xv[nt][1] = z7_ldx(p + 32);
    } else {
      xv[nt][0] = make_uint4(0, 0, 0, 0);
      xv[nt][1] = make_uint4(0, 0, 0, 0);
    }
  }
}

template <int RB, int NT, int W, int CS, int D, int MINB, int MODE>
__global__ void __maxnreg__(((65536 / (MINB * (W + 1) * 32)) / 8) * 8 > 255 ? 255 : ((65536 / (MINB * (W + 1) * 32)) / 8) * 8)
z7_kernel(const unsigned char* __restrict__ wp, const bf16* __restrict__ x, bf16* __restrict__ out,
          const int* __restrict__ rowptr, const int* __restrict__ esck, const float* __restrict__ escd,
          int m, int N, int K, int xstride, float scale) {
  // escaped weights decode to 2^-15 (rebased); escd holds the true weight, so the fix is exact per product
  const float cesc = scale * 0.000030517578125f;
  static_assert(D % RB == 0, "D must be a multiple of RB");
  constexpr int BT = NT * 8, R = RB * 16, RP = R + 4, NTHR = (W + 1) * 32;
  constexpr int NV = (BT * R + NTHR - 1) / NTHR;
  extern __shared__ __align__(16) float zs[];
  float* red = zs;
  float* fixs = red + W * BT * RP;
  float* fixc = fixs + BT * R;
  float* tot = fixc + BT * R;

  cg::cluster_group cluster = cg::this_cluster();
  const int rank = CS > 1 ? (int)cluster.block_rank() : 0;
  if constexpr (CS > 1) z7_cl_arrive_relaxed();

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int rg = blockIdx.x / CS;
  const int U = K >> 6;

  if (warp < W) {
    const int ns = CS * W, sidx = rank * W + warp;
    const int u0 = sidx * U / ns, u1 = (sidx + 1) * U / ns;
    const int S = (u1 - u0) * RB;
    const unsigned char* base = wp + ((size_t)rg * U + u0) * (RB * 1536) + lane * 16;

    float acc[RB][NT][4];
#pragma unroll
    for (int rb = 0; rb < RB; ++rb)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[rb][nt][i] = 0.f;
    uint32_t xr = 0;

    uint4 buf[D][3];
#pragma unroll
    for (int d = 0; d < D; ++d) {
      if (d < S) {
        const unsigned char* p = base + d * 1536;
        buf[d][0] = z7_ldw(p);
        buf[d][1] = z7_ldw(p + 512);
        buf[d][2] = z7_ldw(p + 1024);
      }
    }
    uint4 xc[NT][2], xn[NT][2];
    if (MODE == 2) z7_loadx<NT>(xn, x, m, xstride, u0, g, t);

    for (int s0 = 0; s0 < S; s0 += D) {
#pragma unroll
      for (int d = 0; d < D; ++d) {
        const int s = s0 + d;
        if (s < S) {
          const int rb = d % RB;
          if (MODE == 2 && rb == 0) {
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) { xc[nt][0] = xn[nt][0]; xc[nt][1] = xn[nt][1]; }
            const int un = u0 + s / RB + 1;
            if (un < u1) z7_loadx<NT>(xn, x, m, xstride, un, g, t);
          }
          const uint4 w0 = buf[d][0], w1 = buf[d][1], wb = buf[d][2];
          if (MODE == 0) {
            xr ^= w0.x ^ w0.y ^ w0.z ^ w0.w ^ w1.x ^ w1.y ^ w1.z ^ w1.w ^ wb.x ^ wb.y ^ wb.z ^ wb.w;
          } else {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              const uint32_t a0w = reinterpret_cast<const uint32_t*>(&w0)[j];
              const uint32_t a1w = reinterpret_cast<const uint32_t*>(&w1)[j];
              const uint32_t bw = reinterpret_cast<const uint32_t*>(&wb)[j];
              const uint32_t h0 = z7_lop(bw);
              const uint32_t h1 = z7_lop(__funnelshift_l(bw, bw, 4));
              const uint32_t r0 = z7_prmt(a0w, h0, 0x5140);
              const uint32_t r1 = z7_prmt(a0w, h0, 0x7362);
              const uint32_t r2 = z7_prmt(a1w, h1, 0x5140);
              const uint32_t r3 = z7_prmt(a1w, h1, 0x7362);
              if (MODE == 1) {
                xr ^= r0 ^ r1 ^ r2 ^ r3;
              } else {
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                  const uint32_t* xw = reinterpret_cast<const uint32_t*>(&xc[nt][j >> 1]);
                  z7_mma(acc[rb][nt], r0, r1, r2, r3, xw[(j & 1) * 2], xw[(j & 1) * 2 + 1]);
                }
              }
            }
          }
          if (s + D < S) {
            const unsigned char* p = base + (size_t)(s + D) * 1536;
            buf[d][0] = z7_ldw(p);
            buf[d][1] = z7_ldw(p + 512);
            buf[d][2] = z7_ldw(p + 1024);
          }
        }
      }
    }
    if (MODE != 2) acc[0][0][0] = __uint_as_float(xr & 0x3f000000u);
    float* dst = red + warp * BT * RP;
#pragma unroll
    for (int rb = 0; rb < RB; ++rb)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int row = rb * 16 + g, tk = nt * 8 + 2 * t;
        dst[tk * RP + row] = acc[rb][nt][0];
        dst[(tk + 1) * RP + row] = acc[rb][nt][1];
        dst[tk * RP + row + 8] = acc[rb][nt][2];
        dst[(tk + 1) * RP + row + 8] = acc[rb][nt][3];
      }
  } else if (rank == 0) {
    for (int row = lane; row < R; row += 32) {
      float f[BT], fx[BT];
#pragma unroll
      for (int tk = 0; tk < BT; ++tk) { f[tk] = 0.f; fx[tk] = 0.f; }
      const int gr = rg * R + row;
      const int e1 = rowptr[gr + 1];
      for (int e = rowptr[gr]; e < e1; ++e) {
        const int kk = esck[e];
        const float dv = escd[e];
#pragma unroll
        for (int tk = 0; tk < BT; ++tk)
          if (tk < m) {
            const float xv = __bfloat162float(x[(size_t)tk * xstride + kk]);
            f[tk] += dv * xv;
            fx[tk] += xv;
          }
      }
#pragma unroll
      for (int tk = 0; tk < BT; ++tk) { fixs[tk * R + row] = f[tk]; fixc[tk * R + row] = fx[tk]; }
    }
  }
  __syncthreads();

  float sv[NV];
#pragma unroll
  for (int v = 0; v < NV; ++v) {
    const int idx = tid + v * NTHR;
    float s = 0.f;
    if (idx < BT * R) {
      const int tk = idx / R, row = idx % R;
#pragma unroll
      for (int w = 0; w < W; ++w) s += red[w * BT * RP + tk * RP + row];
    }
    sv[v] = s;
  }

  if constexpr (CS > 1) {
    z7_cl_wait();
    if (rank != 0) {
      float* dstc = cluster.map_shared_rank(tot, 0) + (rank - 1) * BT * R;
#pragma unroll
      for (int v = 0; v < NV; ++v) {
        const int idx = tid + v * NTHR;
        if (idx < BT * R) dstc[idx] = sv[v];
      }
    }
    z7_cl_arrive_release();
    z7_cl_wait();
    if (rank != 0) return;
#pragma unroll
    for (int v = 0; v < NV; ++v) {
      const int idx = tid + v * NTHR;
      if (idx < BT * R) {
#pragma unroll
        for (int c = 0; c < CS - 1; ++c) sv[v] += tot[c * BT * R + idx];
      }
    }
  }

#pragma unroll
  for (int v = 0; v < NV; ++v) {
    const int idx = tid + v * NTHR;
    if (idx < BT * R) {
      const int tk = idx / R, row = idx % R;
      if (tk < m) out[(size_t)tk * N + rg * R + row] = __float2bfloat16((sv[v] * scale - cesc * fixc[idx]) + fixs[idx]);
    }
  }
}

template <int RB, int NT, int W, int CS, int D, int MINB, int MODE>
void z7_launch(const torch::Tensor& x, const torch::Tensor& wp, torch::Tensor& out, const torch::Tensor& rowptr,
               const torch::Tensor& esck, const torch::Tensor& escd, double scale) {
  constexpr int BT = NT * 8, R = RB * 16, RP = R + 4;
  const int m = x.size(0), K = x.size(1), N = out.size(1);
  TORCH_CHECK(m <= BT && N % R == 0 && K % 64 == 0 && (K / 64) >= CS * W, "z7: shape not supported");
  const size_t smem = (size_t)(W * BT * RP + 2 * BT * R + (CS > 1 ? CS - 1 : 1) * BT * R) * sizeof(float);
  auto kern = z7_kernel<RB, NT, W, CS, D, MINB, MODE>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(kern),
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((N / R) * CS, 1, 1);
  cfg.blockDim = dim3((W + 1) * 32, 1, 1);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream().stream();
  cudaLaunchAttribute attr;
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = CS;
  attr.val.clusterDim.y = 1;
  attr.val.clusterDim.z = 1;
  cfg.attrs = &attr;
  cfg.numAttrs = CS > 1 ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, static_cast<const unsigned char*>(wp.data_ptr()),
                                    static_cast<const bf16*>(x.data_ptr()), static_cast<bf16*>(out.data_ptr()),
                                    static_cast<const int*>(rowptr.data_ptr()), static_cast<const int*>(esck.data_ptr()),
                                    static_cast<const float*>(escd.data_ptr()), m, N, K, (int)x.stride(0),
                                    (float)scale));
}

#define Z7_CASE(ID, RB_, NT_, W_, CS_, D_, MINB_, MODE_) \
  if (id == ID) { z7_launch<RB_, NT_, W_, CS_, D_, MINB_, MODE_>(x, wp, out, rowptr, esck, escd, scale); return; }

void launch_z7(torch::Tensor x, torch::Tensor wp, torch::Tensor out, torch::Tensor rowptr, torch::Tensor esck,
               torch::Tensor escd, double scale, int64_t id) {
  TORCH_CHECK(x.is_contiguous() || x.stride(1) == 1, "z7: x inner stride");
  Z7_CASES
  TORCH_CHECK(false, "z7: config id not compiled ", id);
}
"""

ZIP_CPP = ("void launch_z7(torch::Tensor x, torch::Tensor wp, torch::Tensor out, torch::Tensor rowptr,"
           " torch::Tensor esck, torch::Tensor escd, double scale, int64_t id);")

# id -> (RB, NT, W, CS, D, MINB, MODE)
CONFIGS: dict[int, tuple[int, ...]] = {
    0: (4, 1, 8, 2, 8, 1, 2),   # pooler m<=8, 128 CTAs
    1: (4, 2, 8, 2, 8, 1, 2),   # pooler m<=16
    2: (4, 1, 4, 4, 8, 2, 2),   # pooler m<=8, 256 CTAs (2/SM)
    3: (4, 2, 4, 4, 8, 2, 2),   # pooler m<=16, 256 CTAs
    4: (4, 1, 8, 1, 8, 1, 2),   # unpooler m<=8, 128 CTAs
    5: (4, 2, 8, 1, 8, 1, 2),   # unpooler m<=16
    6: (4, 2, 4, 2, 8, 2, 2),   # unpooler m<=16, 256 CTAs
    7: (4, 4, 8, 2, 4, 1, 2),   # pooler m<=32
    8: (4, 4, 8, 1, 4, 1, 2),   # unpooler m<=32
    9: (4, 4, 4, 4, 4, 2, 2),   # qkv m<=32, 144 CTAs
    10: (4, 1, 8, 2, 8, 1, 0),  # S1: loads only
    11: (4, 1, 8, 2, 8, 1, 1),  # S1: loads + decode
    12: (4, 1, 4, 4, 8, 2, 0),
    13: (4, 1, 4, 4, 8, 2, 1),
}


def source(configs: dict[int, tuple[int, ...]] = CONFIGS) -> str:
    cases = " ".join(f"Z7_CASE({i}, {', '.join(map(str, c))})" for i, c in configs.items())
    return ZIP_SRC.replace("Z7_CASES", cases)
