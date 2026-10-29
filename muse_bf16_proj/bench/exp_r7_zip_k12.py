"""r7 ZipGEMM 12-bit k12 (k11 config sweep: ring depth 1-2, R16 for qkv): per-warp private cp.async.bulk ring carrying W (evict-first) + the unit's x rows,
mma.sync with A decoded in registers, a dedicated escape-fix warp (8 per-row slots of true weights), split-K
across warps (smem) and an optional CS-CTA cluster (DSMEM). Format = exp_r6_zip_v2.pack(rb=RB).
"""

from __future__ import annotations

ZIP_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda.h>

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
  asm(
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
__device__ __forceinline__ uint64_t z7_evict_first() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
  return pol;
}
// W is read exactly once: evict-first keeps x (re-read by every CTA) resident in L2
__device__ __forceinline__ void z7_bulk(void* dst, const void* src, uint32_t bytes, uint64_t* bar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
      ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))), "l"(src), "r"(bytes),
        "r"(static_cast<uint32_t>(__cvta_generic_to_shared(bar))), "l"(pol) : "memory");
}

__device__ __forceinline__ void z9_bulk_x(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
      ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))), "l"(src), "r"(bytes),
        "r"(static_cast<uint32_t>(__cvta_generic_to_shared(bar))) : "memory");
}

__device__ __forceinline__ void z10_tma_x(void* dst, const CUtensorMap* map, int k, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
      ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))), "l"(map), "r"(k), "r"(0),
        "r"(static_cast<uint32_t>(__cvta_generic_to_shared(bar))) : "memory");
}

inline CUtensorMap z10_xmap(const void* ptr, int rows, int cols, int row_stride, int box_rows) {
  CUtensorMap map{};
  cuuint64_t gdim[2] = {(cuuint64_t)cols, (cuuint64_t)rows};
  cuuint64_t gstride[1] = {(cuuint64_t)row_stride * sizeof(bf16)};
  cuuint32_t bdim[2] = {64u, (cuuint32_t)box_rows};
  cuuint32_t estride[2] = {1u, 1u};
  CUresult res = cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), gdim, gstride,
                                        bdim, estride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(res == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed with error ", (int)res);
  return map;
}

constexpr int ZE = 8;  // escape slots per row: int2 {k or -1, true weight bits}

__device__ __forceinline__ void z7_cp16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))), "l"(src) : "memory");
}

// One CTA = W compute warps + 1 escape-fix warp. Each compute warp owns a contiguous unit range and a private
// D-stage ring; a stage = one 64-k unit of W (RB records) + the m x-rows of that unit (128 B each, async bulk).
template <int RB, int NT, int W, int CS, int D, int MINB>
__global__ void __launch_bounds__((W + 1) * 32, MINB)
z9_kernel(const __grid_constant__ CUtensorMap xmap, const unsigned char* __restrict__ wp, const bf16* __restrict__ x,
          bf16* __restrict__ out,
          const int2* __restrict__ esc, int m, int N, int K, int xstride, float scale) {
  constexpr int BT = NT * 8, R = RB * 16, RP = R + 4, NTHR = (W + 1) * 32;
  constexpr int NV = (BT * R + NTHR - 1) / NTHR;
  constexpr uint32_t SLOT = RB * 1536, XB = BT * 128, XOFF = ((W * D * SLOT + 1023) / 1024) * 1024;
  constexpr uint32_t RING = XOFF + W * D * XB;
  static_assert(W * BT * RP * 4 <= RING, "red must fit in the ring");
  const float cesc = scale * 0.000030517578125f;  // escaped weights decode to 2^-15 (rebased)
  extern __shared__ __align__(1024) unsigned char zsm_raw[];
  // 128B-swizzled TMA destinations need 1024 B alignment
  unsigned char* zsm = zsm_raw + ((1024 - (static_cast<uint32_t>(__cvta_generic_to_shared(zsm_raw)) & 1023)) & 1023);
  unsigned char* ringb = zsm;
  float* red = reinterpret_cast<float*>(zsm);
  float* fixs = reinterpret_cast<float*>(zsm + RING);
  float* fixc = fixs + BT * R;
  float* tot = fixc + BT * R;
  uint64_t* bars = reinterpret_cast<uint64_t*>(tot + (CS > 1 ? CS - 1 : 1) * BT * R);

  cg::cluster_group cluster = cg::this_cluster();
  const int rank = CS > 1 ? (int)cluster.block_rank() : 0;
  if constexpr (CS > 1) z7_cl_arrive_relaxed();

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int rg = blockIdx.x / CS;
  const int U = K >> 6;

  float acc[RB][NT][4];
#pragma unroll
  for (int rb = 0; rb < RB; ++rb)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[rb][nt][i] = 0.f;

  if (warp < W) {
    const int ns = CS * W, sidx = rank * W + warp;
    const int u0 = sidx * U / ns, NU = (sidx + 1) * U / ns - u0;
    const unsigned char* gsrc = wp + ((size_t)rg * U + u0) * SLOT;
    unsigned char* ring = ringb + (size_t)warp * D * SLOT;
    unsigned char* xring = ringb + XOFF + (size_t)warp * D * XB;
    uint64_t* bar = bars + warp * D;
    const uint64_t pol = z7_evict_first();
    const uint32_t tx = SLOT + XB;  // rows >= m of the x box are zero-filled by TMA and still counted
    if (lane == 0) {
#pragma unroll
      for (int d = 0; d < D; ++d) z7_mbar_init(&bar[d], 1);
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#pragma unroll
      for (int d = 0; d < D; ++d)
        if (d < NU) {
          z7_expect_tx(&bar[d], tx);
          z7_bulk(ring + d * SLOT, gsrc + (size_t)d * SLOT, SLOT, &bar[d], pol);
          z10_tma_x(xring + d * XB, &xmap, (u0 + d) * 64, &bar[d]);
        }
    }
    __syncwarp();

    for (int u = 0; u < NU; ++u) {
      const int d = u % D;
      z7_mbar_wait(&bar[d], (u / D) & 1);
      const uint4* sp = reinterpret_cast<const uint4*>(ring + d * SLOT) + lane;
      const unsigned char* xst = xring + d * XB;
      uint4 w0[RB], w1[RB], wb[RB], xc[NT][2];
#pragma unroll
      for (int rb = 0; rb < RB; ++rb) { w0[rb] = sp[rb * 96]; w1[rb] = sp[rb * 96 + 32]; wb[rb] = sp[rb * 96 + 64]; }
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const unsigned char* xr = xst + (nt * 8 + g) * 128;
        xc[nt][0] = *reinterpret_cast<const uint4*>(xr + ((t ^ g) << 4));
        xc[nt][1] = *reinterpret_cast<const uint4*>(xr + (((t + 4) ^ g) << 4));
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int rb = 0; rb < RB; ++rb) {
          const uint32_t a0w = reinterpret_cast<const uint32_t*>(&w0[rb])[j];
          const uint32_t a1w = reinterpret_cast<const uint32_t*>(&w1[rb])[j];
          const uint32_t bw = reinterpret_cast<const uint32_t*>(&wb[rb])[j];
          const uint32_t h0 = z7_lop(bw);
          const uint32_t h1 = z7_lop(__funnelshift_l(bw, bw, 4));
          const uint32_t r0 = z7_prmt(a0w, h0, 0x5140);
          const uint32_t r1 = z7_prmt(a0w, h0, 0x7362);
          const uint32_t r2 = z7_prmt(a1w, h1, 0x5140);
          const uint32_t r3 = z7_prmt(a1w, h1, 0x7362);
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) {
            const uint32_t* xw = reinterpret_cast<const uint32_t*>(&xc[nt][j >> 1]);
            z7_mma(acc[rb][nt], r0, r1, r2, r3, xw[(j & 1) * 2], xw[(j & 1) * 2 + 1]);
          }
        }
      }
      __syncwarp();
      if (lane == 0 && u + D < NU) {
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        z7_expect_tx(&bar[d], tx);
        z7_bulk(ring + d * SLOT, gsrc + (size_t)(u + D) * SLOT, SLOT, &bar[d], pol);
        z10_tma_x(xring + d * XB, &xmap, (u0 + u + D) * 64, &bar[d]);
      }
    }
  } else if (rank == 0) {
    for (int row = lane; row < R; row += 32) {
      const int4* sp = reinterpret_cast<const int4*>(esc + (size_t)(rg * R + row) * ZE);
      int4 sl[ZE / 2];
#pragma unroll
      for (int i = 0; i < ZE / 2; ++i) sl[i] = sp[i];
      float f[BT], fx[BT];
#pragma unroll
      for (int tk = 0; tk < BT; ++tk) { f[tk] = 0.f; fx[tk] = 0.f; }
#pragma unroll
      for (int e = 0; e < ZE; ++e) {
        const int kk = (e & 1) ? sl[e >> 1].z : sl[e >> 1].x;
        const float wv = __int_as_float((e & 1) ? sl[e >> 1].w : sl[e >> 1].y);
        if (kk >= 0) {
#pragma unroll
          for (int tk = 0; tk < BT; ++tk)
            if (tk < m) {
              const float xv = __bfloat162float(x[(size_t)tk * xstride + kk]);
              f[tk] += wv * xv;
              fx[tk] += xv;
            }
        }
      }
#pragma unroll
      for (int tk = 0; tk < BT; ++tk) { fixs[tk * R + row] = f[tk]; fixc[tk * R + row] = fx[tk]; }
    }
  }
  __syncthreads();
  if (warp < W) {
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

template <int RB, int NT, int W, int CS, int D, int MINB>
void z9_launch(const torch::Tensor& x, const torch::Tensor& wp, torch::Tensor& out, const torch::Tensor& esc,
               double scale) {
  constexpr int BT = NT * 8, R = RB * 16;
  constexpr size_t XOFF = ((W * D * RB * 1536 + 1023) / 1024) * 1024, RING = XOFF + W * D * BT * 128;
  const int m = x.size(0), K = x.size(1), N = out.size(1);
  TORCH_CHECK(m >= 1 && m <= BT && N % R == 0 && K % 64 == 0 && (K / 64) >= CS * W, "z9: shape not supported");
  TORCH_CHECK(wp.numel() == (int64_t)N * K * 3 / 2 && esc.numel() == (int64_t)N * ZE * 2, "z9: packed size");
  TORCH_CHECK(x.stride(1) == 1 && x.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "z9: x layout");
  const size_t smem = 1024 + RING + (size_t)(2 * BT * R + (CS > 1 ? CS - 1 : 1) * BT * R) * sizeof(float) +
                      (size_t)W * D * 8;
  TORCH_CHECK(smem <= 232448, "z9: smem ", smem);
  auto kern = z9_kernel<RB, NT, W, CS, D, MINB>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(kern),
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  if (MINB > 1)
    C10_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(kern),
                                        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
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
  const CUtensorMap xmap = z10_xmap(x.data_ptr(), m, K, (int)x.stride(0), BT);
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, xmap, static_cast<const unsigned char*>(wp.data_ptr()),
                                    static_cast<const bf16*>(x.data_ptr()), static_cast<bf16*>(out.data_ptr()),
                                    static_cast<const int2*>(esc.data_ptr()), m, N, K, (int)x.stride(0),
                                    (float)scale));
}

#define Z9_CASE(ID, RB_, NT_, W_, CS_, D_, MINB_) \
  if (id == ID) { z9_launch<RB_, NT_, W_, CS_, D_, MINB_>(x, wp, out, esc, scale); return; }

void launch_z7(torch::Tensor x, torch::Tensor wp, torch::Tensor out, torch::Tensor esc, double scale, int64_t id,
               int64_t ilv) {
  Z7_CASES
  TORCH_CHECK(false, "z9: config id not compiled ", id);
}
"""

ZIP_CPP = ("void launch_z7(torch::Tensor x, torch::Tensor wp, torch::Tensor out, torch::Tensor esc, double scale,"
           " int64_t id, int64_t ilv);")

# id -> (RB, NT, W, CS, D, MINB)
CONFIGS: dict[int, tuple[int, ...]] = {
    25: (2, 1, 8, 1, 2, 2),   # pooler m<=8 (256 CTAs, 2/SM)
    26: (2, 1, 8, 1, 1, 2),
    27: (2, 2, 8, 1, 2, 2),   # pooler/unpooler m<=16, 2/SM
    28: (2, 2, 8, 1, 1, 2),
    29: (2, 4, 8, 1, 2, 1),   # pooler/unpooler m<=32
    30: (2, 4, 8, 1, 1, 2),
    31: (1, 1, 8, 1, 2, 2),   # qkv m<=8, R16, 144 CTAs
    32: (1, 2, 8, 1, 2, 2),   # qkv m<=16
    33: (2, 1, 8, 2, 1, 2),   # qkv m<=8, R32 CS2, ring depth 1
    34: (1, 4, 8, 1, 2, 2),   # qkv m<=32
}


def source(configs: dict[int, tuple[int, ...]] = CONFIGS) -> str:
    cases = " ".join(f"Z9_CASE({i}, {', '.join(map(str, c))})" for i, c in configs.items())
    return ZIP_SRC.replace("Z7_CASES", cases)
