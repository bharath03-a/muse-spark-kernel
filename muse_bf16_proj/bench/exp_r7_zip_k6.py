"""r7 ZipGEMM 12-bit k6 (k5 + x L2 prefetch + early escape gather): per-warp private cp.async.bulk ring (no cross-warp sync), x B-fragments straight from L2, mma.sync.

Format = exp_r6_zip_v2.pack (rb=4): per (row-group of 64 rows, 64-k unit, 16-row block) a 1536 B record in
mma A-fragment lane order. Each compute warp streams one contiguous range of records into a D-deep register
ring; escapes are fixed in the epilogue from 8 per-row slots (prefetched to smem). Split-K across warps (smem reduce) and
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

__device__ __forceinline__ void z7_mbar_init(uint64_t* b, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(b))), "r"(count) : "memory");
}
__device__ __forceinline__ void z7_expect_tx(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
               ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(b))), "r"(bytes) : "memory");
}
__device__ __forceinline__ void z7_mbar_wait(uint64_t* b, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred P1;\n Z7_WAIT:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      " @P1 bra Z7_DONE;\n bra Z7_WAIT;\n Z7_DONE:\n}\n"
      ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(b))), "r"(parity) : "memory");
}
__device__ __forceinline__ void z7_bulk(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
      ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))), "l"(src), "r"(bytes),
        "r"(static_cast<uint32_t>(__cvta_generic_to_shared(bar))) : "memory");
}

constexpr int ZE = 8;  // escape slots per row: int2 {k or -1, true weight bits}

__device__ __forceinline__ void z7_cp16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))), "l"(src) : "memory");
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
__global__ void __launch_bounds__(W * 32, MINB)
z7_kernel(const unsigned char* __restrict__ wp, const bf16* __restrict__ x, bf16* __restrict__ out,
          const int2* __restrict__ esc, int m, int N, int K, int xstride, float scale, int ilv) {
  // escaped weights decode to 2^-15 (rebased); escd holds the true weight, so the fix is exact per product
  const float cesc = scale * 0.000030517578125f;
  constexpr int BT = NT * 8, R = RB * 16, RP = R + 4, NTHR = W * 32;
  constexpr int NV = (BT * R + NTHR - 1) / NTHR;
  extern __shared__ __align__(16) float zs[];
  static_assert(W * BT * RP * 4 <= W * D * RB * 1536, "red must fit in the ring");
  unsigned char* ringb = reinterpret_cast<unsigned char*>(zs);
  float* red = zs;
  int2* slots = reinterpret_cast<int2*>(ringb + (size_t)W * D * RB * 1536);
  float* tot = reinterpret_cast<float*>(slots + R * ZE);
  uint64_t* bars = reinterpret_cast<uint64_t*>(tot + (CS > 1 ? CS - 1 : 1) * BT * R);

  cg::cluster_group cluster = cg::this_cluster();
  const int rank = CS > 1 ? (int)cluster.block_rank() : 0;
  if constexpr (CS > 1) z7_cl_arrive_relaxed();

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int rg = blockIdx.x / CS;
  const int U = K >> 6;
  if (MODE == 2 && rank == 0) {
    const unsigned char* src = reinterpret_cast<const unsigned char*>(esc + (size_t)rg * R * ZE);
    for (int i = tid; i < R * ZE / 2; i += NTHR) z7_cp16(reinterpret_cast<unsigned char*>(slots) + i * 16, src + i * 16);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  }

  int2 sa[NV], sb[NV];
  float xa[NV], xb[NV];
  {
    const int ns = CS * W, sidx = rank * W + warp;
    // ilv: warps of a CTA take interleaved units so concurrent requests of a CTA are adjacent
    const int cu0 = rank * U / CS, cu1 = (rank + 1) * U / CS;
    const int u0 = ilv ? cu0 + warp : sidx * U / ns;
    const int ustep = ilv ? W : 1;
    const int NU = ilv ? (u0 < cu1 ? (cu1 - u0 + W - 1) / W : 0) : (sidx + 1) * U / ns - u0;

    float acc[RB][NT][4];
#pragma unroll
    for (int rb = 0; rb < RB; ++rb)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[rb][nt][i] = 0.f;
    uint32_t xr = 0;

    constexpr uint32_t SLOT = RB * 1536;
    const unsigned char* gsrc = wp + ((size_t)rg * U + u0) * SLOT;
    const size_t gstep = (size_t)ustep * SLOT;
    unsigned char* ring = ringb + (size_t)warp * D * SLOT;
    uint64_t* bar = bars + warp * D;
    if (lane == 0) {
#pragma unroll
      for (int d = 0; d < D; ++d) z7_mbar_init(&bar[d], 1);
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#pragma unroll
      for (int d = 0; d < D; ++d)
        if (d < NU) {
          z7_expect_tx(&bar[d], SLOT);
          z7_bulk(ring + d * SLOT, gsrc + d * gstep, SLOT, &bar[d]);
        }
    }
    if (MODE == 2 && lane == 0) {
      // pull this CTA's x k-range into L2 now; HBM latency under full load exceeds the 1-unit register lead
      for (int tk = warp; tk < m; tk += W)
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n"
                     ::"l"(x + (size_t)tk * xstride + cu0 * 64), "r"((uint32_t)(cu1 - cu0) * 128u) : "memory");
    }
    __syncwarp();
    uint4 xc[NT][2], xn[NT][2];
    if (MODE == 2) z7_loadx<NT>(xn, x, m, xstride, u0, g, t);

    for (int u = 0; u < NU; ++u) {
      const int d = u % D;
      if (MODE == 2) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) { xc[nt][0] = xn[nt][0]; xc[nt][1] = xn[nt][1]; }
        if (u + 1 < NU) z7_loadx<NT>(xn, x, m, xstride, u0 + (u + 1) * ustep, g, t);
      }
      z7_mbar_wait(&bar[d], (u / D) & 1);
      const uint4* sp = reinterpret_cast<const uint4*>(ring + d * SLOT) + lane;
#pragma unroll
      for (int rb = 0; rb < RB; ++rb) {
        const uint4 w0 = sp[rb * 96], w1 = sp[rb * 96 + 32], wb = sp[rb * 96 + 64];
        if (MODE == 0 || MODE == 3) {
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
      }
      __syncwarp();
      if (lane == 0 && u + D < NU) {
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        z7_expect_tx(&bar[d], SLOT);
        z7_bulk(ring + d * SLOT, gsrc + (u + D) * gstep, SLOT, &bar[d]);
      }
    }
    if (MODE == 3) {
      if (xr == 0x9e3779b9u) out[0] = __float2bfloat16(1.f);
      return;
    }
    if (MODE == 2 && rank == 0) asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();
    if (MODE == 2 && rank == 0) {
      // first two escape slots of each output row: issue the x gathers before the reduction
#pragma unroll
      for (int v = 0; v < NV; ++v) {
        const int idx = tid + v * NTHR;
        const int tk = idx / R, row = idx % R;
        const bool ok = idx < BT * R && tk < m;
        sa[v] = ok ? slots[row * ZE] : make_int2(-1, 0);
        sb[v] = ok ? slots[row * ZE + 1] : make_int2(-1, 0);
        xa[v] = sa[v].x >= 0 ? __bfloat162float(x[(size_t)tk * xstride + sa[v].x]) : 0.f;
        xb[v] = sb[v].x >= 0 ? __bfloat162float(x[(size_t)tk * xstride + sb[v].x]) : 0.f;
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
  float fw[NV], fx[NV];
  if (MODE == 2 && rank == 0) {
#pragma unroll
    for (int v = 0; v < NV; ++v) {
      const int idx = tid + v * NTHR;
      fw[v] = __int_as_float(sa[v].y) * xa[v];
      fx[v] = xa[v];
      fw[v] += __int_as_float(sb[v].y) * xb[v];
      fx[v] += xb[v];
      if (idx < BT * R && sb[v].x >= 0) {
        const int tk = idx / R, row = idx % R;
        if (tk < m) {
          for (int e = 2; e < ZE; ++e) {
            const int2 sl = slots[row * ZE + e];
            if (sl.x < 0) break;
            const float xv = __bfloat162float(x[(size_t)tk * xstride + sl.x]);
            fw[v] += __int_as_float(sl.y) * xv;
            fx[v] += xv;
          }
        }
      }
    }
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
      if (tk < m) out[(size_t)tk * N + rg * R + row] = __float2bfloat16((sv[v] * scale - cesc * fx[v]) + fw[v]);
    }
  }
}

template <int RB, int NT, int W, int CS, int D, int MINB, int MODE>
void z7_launch(const torch::Tensor& x, const torch::Tensor& wp, torch::Tensor& out, const torch::Tensor& esc,
               double scale, int64_t ilv) {
  constexpr int BT = NT * 8, R = RB * 16, RP = R + 4;
  const int m = x.size(0), K = x.size(1), N = out.size(1);
  TORCH_CHECK(m <= BT && N % R == 0 && K % 64 == 0 && (K / 64) >= CS * W, "z7: shape not supported");
  TORCH_CHECK(MODE != 2 || wp.numel() == (size_t)N * K * 3 / 2, "z7: packed size");
  const size_t smem = (size_t)W * D * RB * 1536 + (size_t)(2 * R * ZE + (CS > 1 ? CS - 1 : 1) * BT * R) * sizeof(float) +
                      (size_t)W * D * 8;
  TORCH_CHECK(smem <= 232448, "z7: smem ", smem);
  auto kern = z7_kernel<RB, NT, W, CS, D, MINB, MODE>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(kern),
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  C10_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(kern),
                                      cudaFuncAttributePreferredSharedMemoryCarveout, 100));
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((N / R) * CS, 1, 1);
  cfg.blockDim = dim3(W * 32, 1, 1);
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
                                    static_cast<const int2*>(esc.data_ptr()), m, N, K, (int)x.stride(0),
                                    (float)scale, (int)ilv));
}

#define Z7_CASE(ID, RB_, NT_, W_, CS_, D_, MINB_, MODE_) \
  if (id == ID) { z7_launch<RB_, NT_, W_, CS_, D_, MINB_, MODE_>(x, wp, out, esc, scale, ilv); return; }

void launch_z7(torch::Tensor x, torch::Tensor wp, torch::Tensor out, torch::Tensor esc, double scale, int64_t id,
               int64_t ilv) {
  TORCH_CHECK(x.is_contiguous() || x.stride(1) == 1, "z7: x inner stride");
  Z7_CASES
  TORCH_CHECK(false, "z7: config id not compiled ", id);
}
"""

ZIP_CPP = ("void launch_z7(torch::Tensor x, torch::Tensor wp, torch::Tensor out, torch::Tensor esc, double scale,"
           " int64_t id, int64_t ilv);")

# id -> (RB, NT, W, CS, D, MINB, MODE)
CONFIGS: dict[int, tuple[int, ...]] = {
    0: (4, 1, 8, 2, 3, 1, 2),   # pooler m<=8, R64 CS2
    1: (4, 2, 8, 2, 3, 1, 2),   # pooler m<=16
    2: (2, 1, 8, 1, 3, 1, 2),   # pooler m<=8, R32 CS1
    3: (2, 2, 8, 1, 3, 1, 2),   # pooler m<=16, R32 CS1
    5: (4, 2, 8, 1, 3, 1, 2),   # unpooler m<=16
    6: (4, 4, 8, 2, 3, 1, 2),   # pooler m<=32
    7: (4, 4, 8, 1, 3, 1, 2),   # unpooler m<=32
    8: (3, 1, 8, 2, 3, 1, 2),   # qkv m<=8, R48 CS2 (96 CTAs)
    9: (3, 2, 8, 2, 3, 1, 2),   # qkv m<=16
    10: (3, 4, 8, 2, 3, 1, 2),  # qkv m<=32
    11: (2, 4, 8, 1, 3, 1, 2),  # pooler m<=32, R32 CS1
    12: (4, 2, 8, 2, 2, 1, 2),
    13: (3, 1, 8, 2, 2, 1, 2),
}


def source(configs: dict[int, tuple[int, ...]] = CONFIGS) -> str:
    cases = " ".join(f"Z7_CASE({i}, {', '.join(map(str, c))})" for i, c in configs.items())
    return ZIP_SRC.replace("Z7_CASES", cases)
