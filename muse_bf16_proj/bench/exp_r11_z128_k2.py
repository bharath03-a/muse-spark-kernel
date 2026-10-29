"""r11 (k2: parallel escape fix from smem-staged entries, MODE diagnostics, phase stamps) m=128 compressed (12-bit zip, wgmma-order pack from r8) GEMM: register-A wgmma m64n128k16, W decoded in
registers, x [128 tok x 64 k] from swizzled smem. CTA = one 64-row W tile x two k-interleaved consumer warpgroups
(alternate stages, double-buffered A registers so decode of the next stage overlaps the in-flight wgmma) + a producer
warp (W half-record bulk copy + x TMA) + a fix warp that accumulates escape corrections for this CTA's token stripe
in local smem during the mainloop. Split-K S over a cluster, partials pushed to the stripe owner through DSMEM and
summed in fixed rank order. Spliced into the candidate by exp_r11_z128_build.py.
"""


def _z11_gen_wgmma() -> str:
    d = ", ".join(f"%{i}" for i in range(64))
    outs = ", ".join(f'"+f"(d[{j}][{r}])' for j in range(16) for r in range(4))
    return (
        "__device__ __forceinline__ void z11_wgmma(float (&d)[16][4], uint32_t a0, uint32_t a1, uint32_t a2,"
        " uint32_t a3, uint64_t db) {\n"
        '  asm volatile("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "\n'
        f'      "{{{d}}}, {{%64, %65, %66, %67}}, %68, 1, 1, 1, 0;\\n"\n'
        f"      : {outs}\n"
        '      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));\n'
        "}\n")


Z11_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda.h>

namespace cg = cooperative_groups;
using bf16 = __nv_bfloat16;

__device__ __forceinline__ uint32_t z11_su(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void z11_cl_arrive_release() { asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void z11_cl_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory"); }
__device__ __forceinline__ uint32_t z11_prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
  return r;
}
__device__ __forceinline__ uint32_t z11_lop(uint32_t a) {
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(0x87878787u), "r"(0x38383838u));
  return r;
}
__device__ __forceinline__ void z11_mbar_init(uint64_t* b, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(z11_su(b)), "r"(count) : "memory");
}
__device__ __forceinline__ void z11_expect_tx(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(z11_su(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void z11_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(z11_su(b)) : "memory");
}
__device__ __forceinline__ void z11_wait(uint64_t* b, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred P1;\n Z11_WAIT:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      " @P1 bra Z11_DONE;\n bra Z11_WAIT;\n Z11_DONE:\n}\n"
      ::"r"(z11_su(b)), "r"(parity) : "memory");
}
__device__ __forceinline__ uint64_t z11_evict_first() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
  return pol;
}
__device__ __forceinline__ void z11_bulk(void* dst, const void* src, uint32_t bytes, uint64_t* bar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
      ::"r"(z11_su(dst)), "l"(src), "r"(bytes), "r"(z11_su(bar)), "l"(pol) : "memory");
}
__device__ __forceinline__ void z11_tma_x(void* dst, const CUtensorMap* map, int k, int t0, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
      ::"r"(z11_su(dst)), "l"(map), "r"(k), "r"(t0), "r"(z11_su(bar)) : "memory");
}
__device__ __forceinline__ uint64_t z11_desc(uint32_t saddr) {
  return (uint64_t)((saddr >> 4) & 0x3FFF) | (1ull << 16) | (64ull << 32) | (1ull << 62);
}
__device__ __forceinline__ void z11_wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void z11_wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void z11_wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"); }
// keeps A fragments live (and unmoved) until the wgmma that reads them has retired
__device__ __forceinline__ void z11_keep(uint32_t (&a)[16]) {
#pragma unroll
  for (int i = 0; i < 16; ++i) asm volatile("" : "+r"(a[i])::"memory");
}
__device__ __forceinline__ void z11_keep_acc(float (&d)[16][4]) {
#pragma unroll
  for (int j = 0; j < 16; ++j)
#pragma unroll
    for (int r = 0; r < 4; ++r) asm volatile("" : "+f"(d[j][r])::"memory");
}
__device__ __forceinline__ uint64_t z11_gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
inline CUtensorMap z11_xmap(const void* ptr, int rows, int cols, int row_stride) {
  CUtensorMap map{};
  cuuint64_t gdim[2] = {(cuuint64_t)cols, (cuuint64_t)rows};
  cuuint64_t gstride[1] = {(cuuint64_t)row_stride * sizeof(bf16)};
  cuuint32_t bdim[2] = {64u, 128u};
  cuuint32_t estride[2] = {1u, 1u};
  CUresult res = cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), gdim, gstride,
                                        bdim, estride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(res == CUDA_SUCCESS, "z11: cuTensorMapEncodeTiled failed ", (int)res);
  return map;
}
""" + _z11_gen_wgmma() + r"""
constexpr uint32_t Z11_XST = 16384;  // 128 tokens x 64 k x 2 B (1024-aligned, first in each stage)
constexpr uint32_t Z11_WST = 6144;   // half of a 128-row record: 4 blocks x 3 planes x 32 lanes x 16 B
constexpr uint32_t Z11_STB = Z11_XST + Z11_WST;
constexpr int Z11_THREADS = 320;     // WG0, WG1 (k-interleaved consumers), producer warp 8, fix warp 9
constexpr int Z11_FP = 68;           // fix tile row pitch (floats): conflict-free fragment reads
constexpr int Z11_OP = 72;           // bf16 staging row pitch
constexpr int Z11_MAXESC = 1024;     // escape entries per 64-row tile staged in smem

template <int S>
struct Z11Geo {
  static constexpr int NT = 16;
  static constexpr int JPR = (NT + S - 1) / S;
  static constexpr uint32_t SLOT = JPR * 128 * 16;
  static constexpr uint32_t RECV = (S - 1) * SLOT;
  static constexpr uint32_t FIXB = 8 * JPR * Z11_FP * 4;
  static constexpr uint32_t OST = 8 * JPR * Z11_OP * 2;
  __device__ static constexpr int owner(int j) { return (j * S) / NT; }
  __device__ static constexpr int j0(int q) { return (q * NT + S - 1) / S; }
};

// one 64-k stage for this warp's 16x64 block: decode into A fragments, 4 async k16 wgmmas on the shared x stage
template <int D, int MODE>
__device__ __forceinline__ void z11_step(int i, bool rel, uint32_t (&a)[16], uint32_t (&ap)[16], float (&acc)[16][4],
                                         unsigned char* sm, uint64_t* full, uint64_t* empty, int blk, int lane) {
  const int s = i % D;
  z11_wait(&full[s], (i / D) & 1);
  const uint4* src = reinterpret_cast<const uint4*>(sm + s * Z11_STB + Z11_XST + blk * 1536) + lane;
  const uint4 w0 = src[0], w1 = src[32], wb = src[64];
  const uint32_t* p0 = reinterpret_cast<const uint32_t*>(&w0);
  const uint32_t* p1 = reinterpret_cast<const uint32_t*>(&w1);
  const uint32_t* pb = reinterpret_cast<const uint32_t*>(&wb);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint32_t h0 = z11_lop(pb[j]);
    const uint32_t h1 = z11_lop(__funnelshift_l(pb[j], pb[j], 4));
    a[j * 4 + 0] = z11_prmt(p0[j], h0, 0x5140);
    a[j * 4 + 1] = z11_prmt(p0[j], h0, 0x7362);
    a[j * 4 + 2] = z11_prmt(p1[j], h1, 0x5140);
    a[j * 4 + 3] = z11_prmt(p1[j], h1, 0x7362);
  }
  z11_keep(a);
  z11_wg_fence();
  const uint64_t db = z11_desc(z11_su(sm + s * Z11_STB));
#pragma unroll
  for (int j = 0; j < 4; ++j)
    if constexpr (MODE != 2) z11_wgmma(acc, a[j * 4], a[j * 4 + 1], a[j * 4 + 2], a[j * 4 + 3], db + (uint64_t)(2 * j));
  z11_wg_commit();
  z11_wg_wait<1>();
  z11_keep(ap);
  if (rel && lane == 0) z11_arrive(&empty[(i - 2) % D]);
}

template <int S, int D, int MODE>
__global__ void __launch_bounds__(Z11_THREADS, 1)
z11_kernel(const __grid_constant__ CUtensorMap xmap, const unsigned char* __restrict__ wp, const bf16* __restrict__ x,
           bf16* __restrict__ out, const int2* __restrict__ fl, const int* __restrict__ foff, int m, int N, int K,
           int xstride, float scale, unsigned long long* __restrict__ prof) {
  using G = Z11Geo<S>;
  static_assert(D > 2, "z11: ring must outlast one round of the two k-interleaved warpgroups");
  static_assert(G::RECV + G::OST <= D * Z11_STB && 16 * 128 * 16 <= D * Z11_STB, "z11: ring too small");
  constexpr uint32_t FOFF = D * Z11_STB, BOFF = FOFF + G::FIXB;
  extern __shared__ __align__(1024) unsigned char z11_raw[];
  unsigned char* sm = z11_raw + ((1024 - (z11_su(z11_raw) & 1023)) & 1023);
  float* fix = reinterpret_cast<float*>(sm + FOFF);
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + BOFF);
  uint64_t* empty = full + D;

  int rank = 0;
  if constexpr (S > 1) rank = (int)cg::this_cluster().block_rank();
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int tile = blockIdx.x / S;  // 64-row W tile
  const int n0 = tile * 64;
  const int U = K >> 6, u0 = rank * U / S, nk = (rank + 1) * U / S - u0;
  const int jb = G::j0(rank), ntok = 8 * (G::j0(rank + 1) - jb);

  if (tid == 0) {
    for (int d = 0; d < D; ++d) { z11_mbar_init(&full[d], 1); z11_mbar_init(&empty[d], 4); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  unsigned long long* pf = prof ? prof + (size_t)blockIdx.x * 6 : nullptr;
  if (pf && tid == 0) pf[0] = z11_gtime();

  float acc[16][4];
#pragma unroll
  for (int j = 0; j < 16; ++j)
#pragma unroll
    for (int r = 0; r < 4; ++r) acc[j][r] = 0.f;

  if (warp == 8) {
    if (lane == 0) {
      const uint64_t pol = z11_evict_first();
      const unsigned char* gw = wp + ((size_t)(tile >> 1) * U + u0) * 12288 + (tile & 1) * Z11_WST;
      for (int i = 0; i < nk; ++i) {
        const int s = i % D;
        if (i >= D) {
          z11_wait(&empty[s], ((i / D) - 1) & 1);
          asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        }
        unsigned char* st = sm + s * Z11_STB;
        if constexpr (MODE == 3) {
          if (i >= D) {  // diagnostic: no memory traffic after the first ring fill
            z11_arrive(&full[s]);
            continue;
          }
        }
        z11_expect_tx(&full[s], Z11_STB);  // x rows >= m are zero-filled and still counted
        z11_tma_x(st, &xmap, (u0 + i) * 64, 0, &full[s]);
        z11_bulk(st + Z11_XST, gw + (size_t)i * 12288, Z11_WST, &full[s], pol);
      }
    }
    __syncwarp();
  } else if (warp == 9) {
    // escape corrections for this CTA's token stripe: fix[t][r] += x[t][c] * (w - decoded), fixed order per (t, r).
    // entries staged in smem once; then every lane keeps H tokens x CB escapes of x loads in flight.
    constexpr int H = (8 * G::JPR + 31) / 32, CB = 32 / H;
    for (int e = lane; e < ntok * Z11_FP; e += 32) fix[e] = 0.f;
    const int e0 = foff[tile], ne = foff[tile + 1] - e0;
    int2* sfl = reinterpret_cast<int2*>(sm + BOFF + 2 * D * 8);
    for (int e = lane; e < ne; e += 32) sfl[e] = fl[e0 + e];
    __syncwarp();
    if constexpr (MODE != 1) {
      const bf16* xr[H];
      bool ok[H];
#pragma unroll
      for (int h = 0; h < H; ++h) {
        const int tq = lane + 32 * h, t = 8 * jb + tq;
        ok[h] = tq < ntok && t < m;
        xr[h] = x + (size_t)(ok[h] ? t : 0) * xstride;
      }
      for (int e = 0; e < ne; e += CB) {
        int2 en[CB];
        float xv[H][CB];
#pragma unroll
        for (int b = 0; b < CB; ++b) en[b] = e + b < ne ? sfl[e + b] : make_int2(0, 0);
#pragma unroll
        for (int h = 0; h < H; ++h)
#pragma unroll
          for (int b = 0; b < CB; ++b) xv[h][b] = __bfloat162float(xr[h][en[b].x >> 8]);
#pragma unroll
        for (int h = 0; h < H; ++h) {
          if (ok[h]) {
            float* fr = fix + (lane + 32 * h) * Z11_FP;
#pragma unroll
            for (int b = 0; b < CB; ++b)
              if (e + b < ne) fr[en[b].x & 63] += xv[h][b] * __int_as_float(en[b].y);
          }
        }
      }
    }
    __syncwarp();
    if (pf && lane == 0) pf[1] = z11_gtime();
    asm volatile("bar.arrive 3, 160;\n" ::: "memory");
  } else {
    const int wk = warp >> 2, blk = warp & 3;
    uint32_t a0[16], a1[16];
#pragma unroll
    for (int r = 0; r < 16; ++r) { a0[r] = 0u; a1[r] = 0u; }
    int i = wk;
    for (; i + 2 < nk; i += 4) {
      z11_step<D, MODE>(i, i >= 2, a0, a1, acc, sm, full, empty, blk, lane);
      z11_step<D, MODE>(i + 2, true, a1, a0, acc, sm, full, empty, blk, lane);
    }
    if (i < nk) z11_step<D, MODE>(i, i >= 2, a0, a1, acc, sm, full, empty, blk, lane);
    z11_wg_wait<0>();
    if (pf && tid == 0) pf[2] = z11_gtime();
    z11_keep_acc(acc);
    z11_keep(a0);
    z11_keep(a1);
    // fixed-order in-CTA reduce of the k-interleaved partials into WG0 (ring is free: all stages consumed)
    float4* scr = reinterpret_cast<float4*>(sm);
    const int lt = tid & 127;
    asm volatile("bar.sync 2, 256;\n" ::: "memory");
    if (wk == 1) {
#pragma unroll
      for (int j = 0; j < 16; ++j) scr[j * 128 + lt] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);
    }
    asm volatile("bar.sync 2, 256;\n" ::: "memory");
    if (wk == 0) {
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const float4 p = scr[j * 128 + lt];
        acc[j][0] += p.x;
        acc[j][1] += p.y;
        acc[j][2] += p.z;
        acc[j][3] += p.w;
      }
    }
  }
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  // every CTA of the cluster is past its mainloop (and its scratch reads): rings are free for reduce slots
  if constexpr (S > 1) {
    z11_cl_arrive_release();
    z11_cl_wait();
  } else {
    __syncthreads();
  }
  if (pf && tid == 0) pf[3] = z11_gtime();
  float4* recv = reinterpret_cast<float4*>(sm);
  bf16* ost = reinterpret_cast<bf16*>(sm + G::RECV);
  if constexpr (S > 1) {
    if (warp < 4) {
      cg::cluster_group cluster = cg::this_cluster();
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const int q = G::owner(j);
        if (q != rank) {
          const int slot = rank < q ? rank : rank - 1;
          float4* dst = cluster.map_shared_rank(recv + (slot * G::JPR + (j - G::j0(q))) * 128 + tid, q);
          *dst = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);
        }
      }
    }
    z11_cl_arrive_release();
    z11_cl_wait();
  }
  if (pf && tid == 0) pf[4] = z11_gtime();
  if (warp < 4) {
    const int g = lane >> 2, t4 = lane & 3, nl = warp * 16 + g;
    asm volatile("bar.sync 3, 160;\n" ::: "memory");  // fix tile complete
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      if (G::owner(j) == rank) {
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int src = 0; src < S; ++src) {
          float4 p;
          if (src == rank) {
            p = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);
          } else {
            const int slot = src < rank ? src : src - 1;
            p = recv[(slot * G::JPR + (j - jb)) * 128 + tid];
          }
          v.x += p.x;
          v.y += p.y;
          v.z += p.z;
          v.w += p.w;
        }
        const int tl = 8 * (j - jb) + 2 * t4;
        ost[tl * Z11_OP + nl] = __float2bfloat16(v.x * scale + fix[tl * Z11_FP + nl]);
        ost[(tl + 1) * Z11_OP + nl] = __float2bfloat16(v.y * scale + fix[(tl + 1) * Z11_FP + nl]);
        ost[tl * Z11_OP + nl + 8] = __float2bfloat16(v.z * scale + fix[tl * Z11_FP + nl + 8]);
        ost[(tl + 1) * Z11_OP + nl + 8] = __float2bfloat16(v.w * scale + fix[(tl + 1) * Z11_FP + nl + 8]);
      }
    }
    asm volatile("bar.sync 1, 128;\n" ::: "memory");
    for (int e = tid; e < ntok * 8; e += 128) {
      const int tl = e >> 3, cv = e & 7;
      const int t = 8 * jb + tl;
      if (t < m) {
        *reinterpret_cast<uint4*>(out + (size_t)t * N + n0 + cv * 8) =
            *reinterpret_cast<const uint4*>(ost + tl * Z11_OP + cv * 8);
      }
    }
    if (pf && tid == 0) pf[5] = z11_gtime();
  }
}

template <int S, int D, int MODE>
constexpr size_t z11_smem() {
  return 1024 + (size_t)D * Z11_STB + Z11Geo<S>::FIXB + (size_t)2 * D * 8 + (size_t)Z11_MAXESC * 8;
}

template <int S, int D, int MODE>
cudaLaunchConfig_t z11_cfg(int blocks, cudaLaunchAttribute* attr) {
  constexpr size_t smem = z11_smem<S, D, MODE>();
  static_assert(smem <= 232448, "z11: smem");
  auto kern = z11_kernel<S, D, MODE>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(kern),
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(blocks, 1, 1);
  cfg.blockDim = dim3(Z11_THREADS, 1, 1);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream().stream();
  attr->id = cudaLaunchAttributeClusterDimension;
  attr->val.clusterDim.x = S;
  attr->val.clusterDim.y = 1;
  attr->val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  return cfg;
}

template <int S, int D, int MODE>
void z11_launch(const torch::Tensor& x, const torch::Tensor& wp, torch::Tensor& out, const torch::Tensor& fl,
                const torch::Tensor& foff, double scale, const torch::Tensor& prof) {
  const int m = x.size(0), K = x.size(1), N = out.size(1);
  TORCH_CHECK(m >= 1 && m <= 128 && N % 128 == 0 && K % 64 == 0 && K / 64 >= S && out.size(0) == m &&
              out.is_contiguous(), "z11: shape");
  TORCH_CHECK(wp.numel() == (int64_t)N * K * 3 / 2 && foff.numel() == (int64_t)(N / 64) + 1 && fl.size(1) == 2,
              "z11: packed size");
  TORCH_CHECK(x.stride(1) == 1 && x.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "z11: x layout");
  const CUtensorMap xmap = z11_xmap(x.data_ptr(), m, K, (int)x.stride(0));
  cudaLaunchAttribute attr;
  const int blocks = (N / 64) * S;
  unsigned long long* pp = nullptr;
  if (prof.numel() > 0) {
    TORCH_CHECK(prof.is_cuda() && prof.scalar_type() == at::kLong && prof.numel() >= (int64_t)blocks * 6, "z11: prof");
    pp = reinterpret_cast<unsigned long long*>(prof.data_ptr());
  }
  cudaLaunchConfig_t cfg = z11_cfg<S, D, MODE>(blocks, &attr);
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, z11_kernel<S, D, MODE>, xmap, static_cast<const unsigned char*>(wp.data_ptr()),
                                    static_cast<const bf16*>(x.data_ptr()), static_cast<bf16*>(out.data_ptr()),
                                    static_cast<const int2*>(fl.data_ptr()), static_cast<const int*>(foff.data_ptr()),
                                    m, N, K, (int)x.stride(0), (float)scale, pp));
}

template <int S, int D, int MODE>
int64_t z11_clusters() {
  cudaLaunchAttribute attr;
  cudaLaunchConfig_t cfg = z11_cfg<S, D, MODE>(132 * S, &attr);
  int n = -1;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&n, reinterpret_cast<const void*>(z11_kernel<S, D, MODE>), &cfg));
  return n;
}

#define Z11_CASE(ID, S_, D_, M_)                                   \
  if (id == ID) {                                              \
    if (query) return z11_clusters<S_, D_, M_>();              \
    z11_launch<S_, D_, M_>(x, wp, out, fl, foff, scale, prof); \
    return 0;                                                  \
  }

int64_t launch_z11(torch::Tensor x, torch::Tensor wp, torch::Tensor out, torch::Tensor fl, torch::Tensor foff,
                   double scale, int64_t id, int64_t query, torch::Tensor prof) {
Z11_CASES
  TORCH_CHECK(false, "z11: config id not compiled ", id);
  return 0;
}
"""
# id -> (S, D): S-way cluster split-K over 64-row tiles, D-stage ring
# id -> (S, D, MODE); MODE 1 = no escape fix, 2 = no MMA, 3 = no memory after the first ring fill (log-only diagnostics)
Z11_CFGS = {1: (1, 8, 0), 2: (2, 8, 0), 3: (3, 8, 0), 11: (1, 8, 1), 12: (1, 8, 2), 13: (1, 8, 3), 21: (2, 8, 1)}
Z11_SRC = Z11_SRC.replace("Z11_CASES", "\n".join(f"  Z11_CASE({i}, {s}, {d}, {md})" for i, (s, d, md) in Z11_CFGS.items()))
Z11_CPP = ("int64_t launch_z11(torch::Tensor x, torch::Tensor wp, torch::Tensor out, torch::Tensor fl, "
           "torch::Tensor foff, double scale, int64_t id, int64_t query, torch::Tensor prof);")
# layer -> candidate config ids for m == 128, timed in prepare against the current path; fastest wins if it beats it
Z11_PLAN = {"pooler": (2,), "unpooler": (1,), "qkv_gate": (3,)}
Z11_DIAG = {"pooler": (2, 21), "unpooler": (1, 11, 12, 13)}
Z11_M = 128
Z11_DET_REPS = 10
Z11_BENCH_REPS = 30
Z11_MARGIN_US = 0.2
Z11_MAX_TILE_ESC = 1024


def _z11_build():
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((Z11_SRC + " ".join(CUDA_FLAGS)).encode()).hexdigest()[:10]
    return load_inline(name=f"muse_z11_{tag}", cpp_sources=Z11_CPP, cuda_sources=Z11_SRC, functions=["launch_z11"],
                       extra_cuda_cflags=CUDA_FLAGS, extra_cflags=["-O3"], extra_ldflags=CUDA_LDFLAGS)


def _z11_pack(w: torch.Tensor) -> dict:
    """cp7's 12-bit format, but natural k order inside each k16 step (wgmma reads x straight from smem) and
    tile-major records: [N/128, K/64, 8 blocks, 3 planes, 32 lanes, 16 B]."""
    n, k = w.shape
    rb = 8
    if w.dtype != torch.bfloat16 or n % 128 or k % 64:
        raise ValueError(f"z11: cannot pack {tuple(w.shape)} {w.dtype}")
    rg, u = n // 128, k // 64
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    hist = torch.bincount(((bits >> 8) & 0x7F).flatten(), minlength=128)
    eb = int(hist.unfold(0, 8, 1).sum(1).argmax())
    s, c, low = bits >> 15, (((bits >> 7) & 0xFF) >> 1) - eb, bits & 0xFF
    ok = (c >= 0) & (c < 8)
    s, c, low = (torch.where(ok, v, torch.zeros_like(v)) for v in (s, c, low))
    nib = (s << 3) | c

    def arr(v):  # [N, K] -> [RG, U, RB, kh, g, t, c, jl, h, eo]; k = 16 (2c + jl) + 8 kh + 2 t + eo
        return v.reshape(rg, rb, 2, 8, u, 2, 2, 2, 4, 2).permute(0, 4, 1, 7, 3, 8, 5, 6, 2, 9)

    a = arr(low).reshape(rg, u, rb, 2, 32, 16).to(torch.uint8)
    nb = arr(nib).reshape(rg, u, rb, 2, 32, 4, 4)
    nw, nx = nb[:, :, :, 0], nb[:, :, :, 1]
    byte = ((nw & 8) << 4) | (nw & 7) | (nx & 8) | (torch.roll(nx & 7, -1, dims=-1) << 4)
    buf = torch.cat([a, byte.reshape(rg, u, rb, 1, 32, 16).to(torch.uint8)], dim=3).contiguous()
    rows, cols = (~ok).nonzero(as_tuple=True)
    scale = 2.0 ** (2 * eb - ZIP_REBASE)
    return {"buf": buf.view(-1), "scale": scale, "rows": rows, "cols": cols,
            "wtrue": w[rows, cols].float()}


def _z11_unpack(buf: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """Emulates the kernel's decode -> rebased bf16 bit patterns [n, k] (natural-k layout)."""
    by = buf.view(-1, k // 64, 8, 3, 32, 4, 4).to(torch.int64)
    wd = by[..., 0] | (by[..., 1] << 8) | (by[..., 2] << 16) | (by[..., 3] << 24)
    a0w, a1w, bw = wd[:, :, :, 0], wd[:, :, :, 1], wd[:, :, :, 2]
    h0 = (bw & 0x87878787) | 0x38383838
    h1 = ((((bw << 4) | (bw >> 28)) & 0xFFFFFFFF) & 0x87878787) | 0x38383838

    def prmt(a, b, sel):
        src = [(a >> (8 * i)) & 0xFF for i in range(4)] + [(b >> (8 * i)) & 0xFF for i in range(4)]
        out = torch.zeros_like(a)
        for i in range(4):
            out |= src[(sel >> (4 * i)) & 7] << (8 * i)
        return out

    r = torch.stack([prmt(a0w, h0, 0x5140), prmt(a0w, h0, 0x7362), prmt(a1w, h1, 0x5140),
                     prmt(a1w, h1, 0x7362)], dim=-1)
    pair = torch.stack([r & 0xFFFF, (r >> 16) & 0xFFFF], dim=-1)
    val = pair.reshape(-1, k // 64, 8, 8, 4, 2, 2, 2, 2, 2)
    return val.permute(0, 2, 8, 3, 1, 5, 6, 7, 4, 9).reshape(n, k)


def _z11_verify(p: dict, w: torch.Tensor, rows_per_chunk: int = 512) -> None:
    n, k = w.shape
    bpr = k * 3 // 2
    esc = torch.zeros(n, k, dtype=torch.bool, device=w.device)
    esc[p["rows"], p["cols"]] = True
    for r0 in range(0, n, rows_per_chunk):
        r1 = min(n, r0 + rows_per_chunk)
        dec = _z11_unpack(p["buf"][r0 * bpr:r1 * bpr], r1 - r0, k)
        dec = dec.to(torch.int32).to(torch.int16).view(torch.bfloat16).float() * p["scale"]
        wf = w[r0:r1].float()
        hit = esc[r0:r1]
        if not torch.equal(torch.where(hit, wf, dec), wf):
            raise ValueError("z11: decode mismatch")
        if not bool((dec[hit] == 2.0 ** -15 * p["scale"]).all()):
            raise ValueError("z11: escape decode value")


def _z11_fixlist(p: dict, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Escapes as int2 {row_in_64_tile | k << 8, fp32 bits of (w - decoded escape)} grouped by 64-row tile,
    sorted by (row, k) inside a tile."""
    rows, cols = p["rows"].long(), p["cols"].long()
    tile, rl = rows // 64, rows % 64
    order = torch.argsort((tile * 64 + rl) * (1 << 24) + cols)
    cesc = torch.tensor(2.0 ** -15 * p["scale"], dtype=torch.float32, device=rows.device)
    delta = (p["wtrue"] - cesc)[order]
    fl = torch.stack([(rl | (cols << 8))[order].to(torch.int32), delta.view(torch.int32)], dim=1).contiguous()
    if fl.numel() == 0:
        fl = torch.zeros(1, 2, dtype=torch.int32, device=rows.device)
    cnt = torch.bincount(tile, minlength=n // 64)
    if int(cnt.max()) > Z11_MAX_TILE_ESC:
        raise ValueError(f"z11: {int(cnt.max())} escapes in one tile")
    foff = torch.zeros(n // 64 + 1, dtype=torch.int32, device=rows.device)
    foff[1:] = torch.cumsum(cnt, 0).to(torch.int32)
    return fl, foff


def _z11_run(p: dict, x: torch.Tensor, cid: int, prof=None) -> torch.Tensor:
    out = torch.empty(x.shape[0], p["n"], device=x.device, dtype=torch.bfloat16)
    p["ext"].launch_z11(x, p["buf"], out, p["fl"], p["foff"], p["scale"], cid, 0,
                        p["noprof"] if prof is None else prof)
    return out


def _z11_selftest(p: dict, w: torch.Tensor, cid: int, gen: torch.Generator) -> None:
    n, k = w.shape
    dev = w.device
    m = Z11_M
    pick = torch.randint(0, k, (m,), device=dev, generator=gen)
    esc_cols = p["cols"].long()
    if esc_cols.numel():
        ne = min(m // 2, esc_cols.numel())
        pick[:ne] = esc_cols[torch.randint(0, esc_cols.numel(), (ne,), device=dev, generator=gen)]
    xo = torch.zeros(m, k, device=dev, dtype=torch.bfloat16)
    xo[torch.arange(m, device=dev), pick] = 1.0
    if not torch.equal(_z11_run(p, xo, cid), w[:, pick].t()):
        raise ValueError("one-hot readback not bit-exact")
    x = torch.randn(m, k, device=dev, generator=gen).to(torch.bfloat16)
    x[:, ::97] *= 30.0
    ref = x.double() @ w.double().t()
    got = _z11_run(p, x, cid)
    gd = got.double()
    rel = float((gd - ref).norm() / ref.norm())
    mx = float((gd - ref).abs().max() / ref.abs().max())
    if not bool(torch.isfinite(gd).all()) or rel > 4e-3 or mx > 2e-2:
        raise ValueError(f"smoke rel={rel:.2e} max={mx:.2e}")
    for _ in range(Z11_DET_REPS):
        if not torch.equal(_z11_run(p, x, cid), got):
            raise ValueError("non-deterministic")


def _z11_time(graphs: dict, flush: torch.Tensor) -> dict:
    import statistics
    res = {nm: [] for nm in graphs}
    for r in range(Z11_BENCH_REPS):
        for nm in (list(graphs) if r % 2 == 0 else list(graphs)[::-1]):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            flush.sum()
            a.record()
            graphs[nm].replay()
            b.record()
            b.synchronize()
            res[nm].append(a.elapsed_time(b) * 1e3)
    return {nm: statistics.median(v) for nm, v in res.items()}


def _z11_graph(fn):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def _z11_diag(p: dict, layer: str, x: torch.Tensor, flush: torch.Tensor) -> None:
    ids = Z11_DIAG.get(layer, ())
    if not ids:
        return
    graphs = {f"id{c}m{Z11_CFGS[c][2]}": _z11_graph(lambda c=c: _z11_run(p, x, c)) for c in ids}
    med = _z11_time(graphs, flush)
    cid = ids[0]
    nb = (p["n"] // 64) * Z11_CFGS[cid][0]
    prof = torch.zeros(nb * 6, dtype=torch.int64, device=x.device)
    st = []
    for _ in range(5):
        flush.sum()
        _z11_run(p, x, cid, prof)
        torch.cuda.synchronize()
        q = prof.view(nb, 6).double() / 1e3
        t0 = q[:, 0].min()
        ph = torch.stack([q[:, 0] - t0, q[:, 1] - q[:, 0], q[:, 2] - q[:, 0], q[:, 3] - q[:, 2], q[:, 4] - q[:, 3],
                          q[:, 5] - q[:, 4]], 1)
        st.append(torch.cat([ph.median(0).values, ph.max(0).values, (q[:, 5].max() - t0).view(1)]))
    sm = torch.stack(st).median(0).values.tolist()
    names = ("skew", "fix", "main", "bar1", "red", "store")
    print(f"z11 diag {layer} (raw us): " + " ".join(f"{nm}={v:.2f}" for nm, v in med.items())
          + f" | id{cid} med " + " ".join(f"{a}={b:.2f}" for a, b in zip(names, sm[:6]))
          + " | max " + " ".join(f"{a}={b:.2f}" for a, b in zip(names, sm[6:12])) + f" | span={sm[12]:.2f}")


def _z11_prepare(state: dict) -> None:
    import time
    t_start = time.time()
    state["z11"] = {}
    if state.get("ext") is None:
        return
    try:
        ext = _z11_build()
        print(f"z11 build {time.time() - t_start:.1f}s")
    except Exception as exc:
        print(f"z11 path disabled: {type(exc).__name__}: {str(exc)[-3000:]}")
        return
    dev = state["pooler"].device
    e = torch.empty(0, device=dev)
    clusters = {}
    for cid in Z11_CFGS:
        try:
            clusters[cid] = int(ext.launch_z11(e, e, e, e, e, 1.0, cid, 1, torch.empty(0, dtype=torch.int64, device=dev)))
        except Exception as exc:
            print(f"z11 id={cid} occupancy query failed: {type(exc).__name__}: {exc}")
            clusters[cid] = 0
    print("z11 max clusters: " + " ".join(f"id{c}{Z11_CFGS[c]}={v}" for c, v in clusters.items()))
    gen = torch.Generator(device=dev).manual_seed(11)
    flush = torch.empty(64 * 2**20, device=dev, dtype=torch.float32)
    flush.normal_()
    picks = {}
    for layer, ids in Z11_PLAN.items():
        try:
            w = state[layer]
            n, k = w.shape
            p = _z11_pack(w)
            _z11_verify(p, w)
            p["fl"], p["foff"] = _z11_fixlist(p, n)
            p["ext"], p["n"], p["noprof"] = ext, n, torch.empty(0, dtype=torch.int64, device=dev)
            print(f"z11 pack {layer}: escapes={p['rows'].numel()} max/tile={int((p['foff'][1:] - p['foff'][:-1]).max())}")
        except Exception as exc:
            print(f"z11 pack failed for {layer}: {type(exc).__name__}: {exc}")
            continue
        good = []
        for cid in ids:
            s = Z11_CFGS[cid][0]
            if (n // 64) > clusters.get(cid, 0):
                print(f"z11 skip {layer} id={cid}: needs {n // 64} clusters > {clusters.get(cid, 0)}")
                continue
            try:
                _z11_selftest(p, w, cid, gen)
                good.append(cid)
            except Exception as exc:
                print(f"z11 selftest FAILED {layer} id={cid}: {type(exc).__name__}: {str(exc)[-1500:]}")
        if not good:
            continue
        try:
            x = torch.randn(Z11_M, k, device=dev, generator=gen).to(torch.bfloat16)
            data = {"x": x, "config": {"m": Z11_M, "n": n, "k": k, "layer": layer}}
            graphs = {"cur": _z11_graph(lambda: kernel_fn(data, state))}
            for cid in good:
                graphs[f"id{cid}"] = _z11_graph(lambda c=cid: _z11_run(p, x, c))
            med = _z11_time(graphs, flush)
            best = min(good, key=lambda c: med[f"id{c}"])
            win = med[f"id{best}"] < med["cur"] - Z11_MARGIN_US
            print(f"z11 bench {layer} m={Z11_M} (raw us, ~4 graph overhead): "
                  + " ".join(f"{nm}={v:.2f}" for nm, v in med.items()) + (f" -> id{best}" if win else " -> cur"))
            if win:
                picks[layer] = {k2: p[k2] for k2 in ("buf", "fl", "foff", "scale", "ext", "n", "noprof")} | {"cid": best}
        except Exception as exc:
            print(f"z11 bench error {layer}: {type(exc).__name__}: {exc}")
        try:
            _z11_diag(p, layer, x, flush)
        except Exception as exc:
            print(f"z11 diag error {layer}: {type(exc).__name__}: {exc}")
    state["z11"] = picks
    torch.cuda.synchronize()
    print(f"z11 prepare {time.time() - t_start:.1f}s picks={ {l: v['cid'] for l, v in picks.items()} }")


def _z11_try(x: torch.Tensor, layer: str, n: int, state: dict):
    p = state.get("z11", {}).get(layer)
    if p is None or x.shape[0] != Z11_M or n != p["n"] or x.shape[1] != state[layer].shape[1]:
        return None
    return _z11_run(p, x, p["cid"])
