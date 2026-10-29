"""r12 m=128 12-bit zip GEMM, mixed-input transform pattern: a producer warp TMA's x and bulk-copies compressed
planes; 4 decoder warps expand them into a 128B-swizzled K-major bf16 W tile in smem (exact escape values patched
in) and signal an mbarrier; consumer warpgroups run SS wgmma m64n128k16 exactly like the r9 kernel. Cluster split-K
with the fixed-order DSMEM reduce. Spliced into the candidate by exp_r12_zt_build.py.
"""


def _z12_gen_wgmma() -> str:
    d = ", ".join(f"%{i}" for i in range(64))
    outs = ", ".join(f'"+f"(d[{j}][{r}])' for j in range(16) for r in range(4))
    return (
        "__device__ __forceinline__ void z12_wgmma(float (&d)[16][4], uint64_t da, uint64_t db) {\n"
        '  asm volatile("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "\n'
        f'      "{{{d}}}, %64, %65, 1, 1, 1, 0, 0;\\n"\n'
        f"      : {outs}\n"
        '      : "l"(da), "l"(db));\n'
        "}\n")


Z12_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda.h>

namespace cg = cooperative_groups;
using bf16 = __nv_bfloat16;

__device__ __forceinline__ uint32_t z12_su(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void z12_cl_arrive_release() { asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void z12_cl_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory"); }
__device__ __forceinline__ uint32_t z12_prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
  return r;
}
__device__ __forceinline__ uint32_t z12_lop(uint32_t a) {
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(0x87878787u), "r"(0x38383838u));
  return r;
}
__device__ __forceinline__ void z12_mbar_init(uint64_t* b, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(z12_su(b)), "r"(count) : "memory");
}
__device__ __forceinline__ void z12_expect_tx(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(z12_su(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void z12_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(z12_su(b)) : "memory");
}
__device__ __forceinline__ void z12_wait(uint64_t* b, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred P1;\n Z12_WAIT:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      " @P1 bra Z12_DONE;\n bra Z12_WAIT;\n Z12_DONE:\n}\n"
      ::"r"(z12_su(b)), "r"(parity) : "memory");
}
__device__ __forceinline__ uint64_t z12_evict_first() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
  return pol;
}
__device__ __forceinline__ void z12_bulk(void* dst, const void* src, uint32_t bytes, uint64_t* bar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
      ::"r"(z12_su(dst)), "l"(src), "r"(bytes), "r"(z12_su(bar)), "l"(pol) : "memory");
}
__device__ __forceinline__ void z12_tma_x(void* dst, const CUtensorMap* map, int k, int t0, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
      ::"r"(z12_su(dst)), "l"(map), "r"(k), "r"(t0), "r"(z12_su(bar)) : "memory");
}
__device__ __forceinline__ uint64_t z12_desc(uint32_t saddr) {
  return (uint64_t)((saddr >> 4) & 0x3FFF) | (1ull << 16) | (64ull << 32) | (1ull << 62);
}
__device__ __forceinline__ void z12_wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void z12_wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void z12_wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"); }
// keeps A fragments live (and unmoved) until the wgmma that reads them has retired
template <int L>
__device__ __forceinline__ void z12_keep(uint32_t (&a)[L]) {
#pragma unroll
  for (int i = 0; i < L; ++i) asm volatile("" : "+r"(a[i])::"memory");
}
__device__ __forceinline__ void z12_keep_acc(float (&d)[16][4]) {
#pragma unroll
  for (int j = 0; j < 16; ++j)
#pragma unroll
    for (int r = 0; r < 4; ++r) asm volatile("" : "+f"(d[j][r])::"memory");
}
__device__ __forceinline__ uint64_t z12_gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
inline CUtensorMap z12_xmap(const void* ptr, int rows, int cols, int row_stride) {
  CUtensorMap map{};
  cuuint64_t gdim[2] = {(cuuint64_t)cols, (cuuint64_t)rows};
  cuuint64_t gstride[1] = {(cuuint64_t)row_stride * sizeof(bf16)};
  cuuint32_t bdim[2] = {64u, 128u};
  cuuint32_t estride[2] = {1u, 1u};
  CUresult res = cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), gdim, gstride,
                                        bdim, estride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(res == CUDA_SUCCESS, "z12: cuTensorMapEncodeTiled failed ", (int)res);
  return map;
}
""" + _z12_gen_wgmma() + r"""
constexpr uint32_t Z12_XST = 16384;  // 128 tokens x 64 k x 2 B
constexpr int Z12_MAXESC = 1024;     // escape entries per W tile staged in smem

template <int NWG, int NKW, int S>
struct Z12Geo {
  static constexpr int BN = 64 * NWG;
  static constexpr int NT = 16;
  static constexpr int NCT = NWG * 128;
  static constexpr int NCONS = NCT * NKW;
  static constexpr int DW0 = NCONS / 32, PW = DW0 + 4;   // 4 decoder warps, then the producer warp
  static constexpr int THREADS = NCONS + 160;
  static constexpr uint32_t WBF = NWG * 8192;            // decoded bf16 W tile: BN rows x 128 B, 128B swizzle
  static constexpr uint32_t WCP = NWG * 6144;            // compressed planes for BN rows x 64 k
  static constexpr uint32_t STB = Z12_XST + WBF + WCP;  // x | W bf16 | compressed (all 1024-aligned)
  static constexpr int JPR = (NT + S - 1) / S;
  static constexpr uint32_t SLOT = JPR * NCT * 16;
  static constexpr uint32_t RECV = (S - 1) * SLOT;
  static constexpr int OP = BN + 8;
  static constexpr uint32_t OST = 8 * JPR * OP * 2;
  __device__ static constexpr int owner(int j) { return (j * S) / NT; }
  __device__ static constexpr int j0(int q) { return (q * NT + S - 1) / S; }
};

// decode one 16-row x 64-k block of compressed planes into the swizzled K-major bf16 tile (SS-wgmma A layout).
// lane (g, t4): register 4j+q holds rows g + 8(q&1), k pair 16j + 2t4 + 8(q>>1)
__device__ __forceinline__ void z12_decode_blk(const unsigned char* cp, unsigned char* wt, int blk, int lane) {
  const uint4* src = reinterpret_cast<const uint4*>(cp + blk * 1536) + lane;
  const uint4 w0 = src[0], w1 = src[32], wb = src[64];
  const uint32_t* p0 = reinterpret_cast<const uint32_t*>(&w0);
  const uint32_t* p1 = reinterpret_cast<const uint32_t*>(&w1);
  const uint32_t* pb = reinterpret_cast<const uint32_t*>(&wb);
  const int g = lane >> 2, t4 = lane & 3;
  unsigned char* r0 = wt + (blk * 16 + g) * 128 + 4 * t4;
  unsigned char* r1 = r0 + 8 * 128;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint32_t h0 = z12_lop(pb[j]);
    const uint32_t h1 = z12_lop(__funnelshift_l(pb[j], pb[j], 4));
    const int c0 = ((2 * j) ^ g) << 4, c1 = ((2 * j + 1) ^ g) << 4;
    *reinterpret_cast<uint32_t*>(r0 + c0) = z12_prmt(p0[j], h0, 0x5140);
    *reinterpret_cast<uint32_t*>(r1 + c0) = z12_prmt(p0[j], h0, 0x7362);
    *reinterpret_cast<uint32_t*>(r0 + c1) = z12_prmt(p1[j], h1, 0x5140);
    *reinterpret_cast<uint32_t*>(r1 + c1) = z12_prmt(p1[j], h1, 0x7362);
  }
}

template <int NWG, int NKW, int S, int D>
__global__ void __launch_bounds__(Z12Geo<NWG, NKW, S>::THREADS, 1)
z12_kernel(const __grid_constant__ CUtensorMap xmap, const unsigned char* __restrict__ wp, bf16* __restrict__ out,
           const int2* __restrict__ fl, const int* __restrict__ foff, int m, int N, int K, float scale) {
  using G = Z12Geo<NWG, NKW, S>;
  static_assert(D > NKW, "z12: ring must outlast one round of the k-interleaved warpgroups");
  static_assert(G::RECV + G::OST <= D * G::STB && (NKW - 1) * 16 * G::NCT * 16 <= D * G::STB, "z12: ring too small");
  constexpr uint32_t BOFF = D * G::STB, EOFF = BOFF + 3 * D * 8;
  extern __shared__ __align__(1024) unsigned char z12_raw[];
  unsigned char* sm = z12_raw + ((1024 - (z12_su(z12_raw) & 1023)) & 1023);
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + BOFF);
  uint64_t* decd = full + D;
  uint64_t* empty = decd + D;
  int2* sfl = reinterpret_cast<int2*>(sm + EOFF);

  int rank = 0;
  if constexpr (S > 1) rank = (int)cg::this_cluster().block_rank();
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int tile = blockIdx.x / S;
  const int n0 = tile * G::BN;
  const int U = K >> 6, u0 = rank * U / S, nk = (rank + 1) * U / S - u0;
  const int jb = G::j0(rank), ntok = 8 * (G::j0(rank + 1) - jb);

  if (tid == 0) {
    for (int d = 0; d < D; ++d) {
      z12_mbar_init(&full[d], 1);
      z12_mbar_init(&decd[d], 4);
      z12_mbar_init(&empty[d], NWG * 4);
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();

  float acc[16][4];
#pragma unroll
  for (int j = 0; j < 16; ++j)
#pragma unroll
    for (int r = 0; r < 4; ++r) acc[j][r] = 0.f;

  if (warp == G::PW) {
    if (lane == 0) {
      const uint64_t pol = z12_evict_first();
      const unsigned char* gw = NWG == 2 ? wp + ((size_t)tile * U + u0) * 12288
                                         : wp + ((size_t)(tile >> 1) * U + u0) * 12288 + (tile & 1) * 6144;
      for (int i = 0; i < nk; ++i) {
        const int s = i % D;
        if (i >= D) {
          z12_wait(&empty[s], ((i / D) - 1) & 1);
          asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        }
        unsigned char* st = sm + s * G::STB;
        z12_expect_tx(&full[s], Z12_XST + G::WCP);  // x rows >= m are zero-filled and still counted
        z12_tma_x(st, &xmap, (u0 + i) * 64, 0, &full[s]);
        z12_bulk(st + Z12_XST + G::WBF, gw + (size_t)i * 12288, G::WCP, &full[s], pol);
      }
    }
    __syncwarp();
  } else if (warp >= G::DW0) {
    // decoder warps: compressed planes -> swizzled bf16 tile, then exact escape values patched in
    const int dw = warp - G::DW0, dt = tid - G::DW0 * 32;
    const int e0 = foff[tile], ne = foff[tile + 1] - e0;
    for (int e = dt; e < ne; e += 128) sfl[e] = fl[e0 + e];
    asm volatile("bar.sync 4, 128;\n" ::: "memory");
    int e = 0;
    while (e < ne && (sfl[e].x >> 8) < u0 * 64) ++e;
    for (int i = 0; i < nk; ++i) {
      const int s = i % D;
      z12_wait(&full[s], (i / D) & 1);
      unsigned char* st = sm + s * G::STB;
      unsigned char* wt = st + Z12_XST;
#pragma unroll
      for (int b = 0; b < NWG; ++b) z12_decode_blk(st + Z12_XST + G::WBF, wt, dw + 4 * b, lane);
      __syncwarp();
      const int kb = (u0 + i) * 64;
      int ee = e;
      while (ee < ne && (sfl[ee].x >> 8) < kb + 64) ++ee;
      for (int q = e + lane; q < ee; q += 32) {
        const int r = sfl[q].x & 255, c = (sfl[q].x >> 8) - kb;
        if (((r >> 4) & 3) == dw) {
          *reinterpret_cast<unsigned short*>(wt + r * 128 + ((((c >> 3) ^ (r & 7))) << 4) + (c & 7) * 2) =
              (unsigned short)sfl[q].y;
        }
      }
      e = ee;
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      __syncwarp();
      if (lane == 0) z12_arrive(&decd[s]);
    }
    __syncwarp();
  } else {
    const int cw = warp >> 2, rw = cw % NWG, wk = cw / NWG;
    int it = 0;
    for (int i = wk; i < nk; i += NKW, ++it) {
      const int s = i % D;
      z12_wait(&full[s], (i / D) & 1);
      z12_wait(&decd[s], (i / D) & 1);
      const unsigned char* st = sm + s * G::STB;
      const uint64_t da = z12_desc(z12_su(st + Z12_XST + rw * 8192));
      const uint64_t db = z12_desc(z12_su(st));
      z12_wg_fence();
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) z12_wgmma(acc, da + (uint64_t)(2 * ks), db + (uint64_t)(2 * ks));
      z12_wg_commit();
      z12_wg_wait<1>();
      if (it > 0 && lane == 0) z12_arrive(&empty[(i - NKW) % D]);
    }
    z12_wg_wait<0>();
    if constexpr (NKW > 1) {
      float4* scr = reinterpret_cast<float4*>(sm);
      const int lt = tid - wk * G::NCT;
      asm volatile("bar.sync 2, %0;\n" ::"n"(G::NCONS) : "memory");
      if (wk > 0) {
#pragma unroll
        for (int j = 0; j < 16; ++j)
          scr[((wk - 1) * 16 + j) * G::NCT + lt] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);
      }
      asm volatile("bar.sync 2, %0;\n" ::"n"(G::NCONS) : "memory");
      if (wk == 0) {
#pragma unroll
        for (int q = 1; q < NKW; ++q)
#pragma unroll
          for (int j = 0; j < 16; ++j) {
            const float4 p = scr[((q - 1) * 16 + j) * G::NCT + lt];
            acc[j][0] += p.x;
            acc[j][1] += p.y;
            acc[j][2] += p.z;
            acc[j][3] += p.w;
          }
      }
    } else if constexpr (NWG > 1) {
      asm volatile("bar.sync 2, %0;\n" ::"n"(G::NCONS) : "memory");
    }
  }
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  if constexpr (S > 1) {
    z12_cl_arrive_release();
    z12_cl_wait();
  } else {
    __syncthreads();
  }
  float4* recv = reinterpret_cast<float4*>(sm);
  bf16* ost = reinterpret_cast<bf16*>(sm + G::RECV);
  if constexpr (S > 1) {
    if (warp < NWG * 4) {
      cg::cluster_group cluster = cg::this_cluster();
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const int q = G::owner(j);
        if (q != rank) {
          const int slot = rank < q ? rank : rank - 1;
          float4* dst = cluster.map_shared_rank(recv + (slot * G::JPR + (j - G::j0(q))) * G::NCT + tid, q);
          *dst = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);
        }
      }
    }
    z12_cl_arrive_release();
    z12_cl_wait();
  }
  if (warp < NWG * 4) {
    const int g = lane >> 2, t4 = lane & 3, nl = (warp >> 2) * 64 + (warp & 3) * 16 + g;
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
            p = recv[(slot * G::JPR + (j - jb)) * G::NCT + tid];
          }
          v.x += p.x;
          v.y += p.y;
          v.z += p.z;
          v.w += p.w;
        }
        const int tl = 8 * (j - jb) + 2 * t4;
        ost[tl * G::OP + nl] = __float2bfloat16(v.x * scale);
        ost[(tl + 1) * G::OP + nl] = __float2bfloat16(v.y * scale);
        ost[tl * G::OP + nl + 8] = __float2bfloat16(v.z * scale);
        ost[(tl + 1) * G::OP + nl + 8] = __float2bfloat16(v.w * scale);
      }
    }
    asm volatile("bar.sync 1, %0;\n" ::"n"(G::NCT) : "memory");
    constexpr int V = G::BN / 8;
    for (int e = tid; e < ntok * V; e += G::NCT) {
      const int tl = e / V, cv = e % V;
      const int t = 8 * jb + tl;
      if (t < m) {
        *reinterpret_cast<uint4*>(out + (size_t)t * N + n0 + cv * 8) =
            *reinterpret_cast<const uint4*>(ost + tl * G::OP + cv * 8);
      }
    }
  }
}

template <int NWG, int NKW, int S, int D>
constexpr size_t z12_smem() {
  return 1024 + (size_t)D * Z12Geo<NWG, NKW, S>::STB + (size_t)3 * D * 8 + (size_t)Z12_MAXESC * 8;
}

template <int NWG, int NKW, int S, int D>
cudaLaunchConfig_t z12_cfg(int blocks, cudaLaunchAttribute* attr) {
  constexpr size_t smem = z12_smem<NWG, NKW, S, D>();
  static_assert(smem <= 232448, "z12: smem");
  auto kern = z12_kernel<NWG, NKW, S, D>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(kern),
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(blocks, 1, 1);
  cfg.blockDim = dim3(Z12Geo<NWG, NKW, S>::THREADS, 1, 1);
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

template <int NWG, int NKW, int S, int D>
void z12_launch(const torch::Tensor& x, const torch::Tensor& wp, torch::Tensor& out, const torch::Tensor& fl,
                const torch::Tensor& foff, double scale) {
  using G = Z12Geo<NWG, NKW, S>;
  const int m = x.size(0), K = x.size(1), N = out.size(1);
  TORCH_CHECK(m >= 1 && m <= 128 && N % 128 == 0 && K % 64 == 0 && K / 64 >= S && out.size(0) == m &&
              out.is_contiguous(), "z12: shape");
  TORCH_CHECK(wp.numel() == (int64_t)N * K * 3 / 2 && foff.numel() == (int64_t)(N / G::BN) + 1 && fl.size(1) == 2,
              "z12: packed size");
  TORCH_CHECK(x.stride(1) == 1 && x.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "z12: x layout");
  const CUtensorMap xmap = z12_xmap(x.data_ptr(), m, K, (int)x.stride(0));
  cudaLaunchAttribute attr;
  cudaLaunchConfig_t cfg = z12_cfg<NWG, NKW, S, D>((N / G::BN) * S, &attr);
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, z12_kernel<NWG, NKW, S, D>, xmap,
                                    static_cast<const unsigned char*>(wp.data_ptr()),
                                    static_cast<bf16*>(out.data_ptr()), static_cast<const int2*>(fl.data_ptr()),
                                    static_cast<const int*>(foff.data_ptr()), m, N, K, (float)scale));
}

template <int NWG, int NKW, int S, int D>
int64_t z12_clusters() {
  cudaLaunchAttribute attr;
  cudaLaunchConfig_t cfg = z12_cfg<NWG, NKW, S, D>(132 * S, &attr);
  int n = -1;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&n, reinterpret_cast<const void*>(z12_kernel<NWG, NKW, S, D>), &cfg));
  return n;
}

#define Z12_CASE(ID, NWG_, NKW_, S_, D_)                         \
  if (id == ID) {                                                \
    if (query) return z12_clusters<NWG_, NKW_, S_, D_>();        \
    z12_launch<NWG_, NKW_, S_, D_>(x, wp, out, fl, foff, scale); \
    return 0;                                                    \
  }

int64_t launch_z12(torch::Tensor x, torch::Tensor wp, torch::Tensor out, torch::Tensor fl, torch::Tensor foff,
                   double scale, int64_t id, int64_t query) {
Z12_CASES
  TORCH_CHECK(false, "z12: config id not compiled ", id);
  return 0;
}
"""
# id -> (NWG, NKW, S, D): BN = 64 NWG W rows per CTA, NKW k-interleaved consumer warpgroups per row block, S-way
# cluster split-K, D-stage ring (x | decoded bf16 W | compressed planes)
Z12_CFGS = {
    1: (2, 1, 2, 4),
    2: (1, 2, 1, 6),
    4: (2, 1, 3, 4),
    5: (1, 2, 2, 6),
    7: (1, 2, 3, 6),
    8: (2, 1, 5, 4),
}
Z12_SRC = Z12_SRC.replace("Z12_CASES", "\n".join(f"  Z12_CASE({i}, {', '.join(map(str, v))})" for i, v in Z12_CFGS.items()))
Z12_CPP = ("int64_t launch_z12(torch::Tensor x, torch::Tensor wp, torch::Tensor out, torch::Tensor fl, "
           "torch::Tensor foff, double scale, int64_t id, int64_t query);")
# layer -> candidate config ids for m == 128, timed in prepare against the current path; fastest wins if it beats it
Z12_PLAN = {"unpooler": (1, 2), "pooler": (4, 5), "qkv_gate": (7, 8)}
Z12_M = 128
Z12_DET_REPS = 10
Z12_BENCH_REPS = 30
Z12_MARGIN_US = 0.2


def _z12_build():
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((Z12_SRC + " ".join(CUDA_FLAGS)).encode()).hexdigest()[:10]
    return load_inline(name=f"muse_z12_{tag}", cpp_sources=Z12_CPP, cuda_sources=Z12_SRC, functions=["launch_z12"],
                       extra_cuda_cflags=CUDA_FLAGS, extra_cflags=["-O3"], extra_ldflags=CUDA_LDFLAGS)


def _z12_pack(w: torch.Tensor) -> dict:
    """cp7's 12-bit format, but natural k order inside each k16 step (wgmma reads x straight from smem) and
    tile-major records: [N/128, K/64, 8 blocks, 3 planes, 32 lanes, 16 B]."""
    n, k = w.shape
    rb = 8
    if w.dtype != torch.bfloat16 or n % 128 or k % 64:
        raise ValueError(f"z12: cannot pack {tuple(w.shape)} {w.dtype}")
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


def _z12_unpack(buf: torch.Tensor, n: int, k: int) -> torch.Tensor:
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


def _z12_verify(p: dict, w: torch.Tensor, rows_per_chunk: int = 512) -> None:
    n, k = w.shape
    bpr = k * 3 // 2
    esc = torch.zeros(n, k, dtype=torch.bool, device=w.device)
    esc[p["rows"], p["cols"]] = True
    for r0 in range(0, n, rows_per_chunk):
        r1 = min(n, r0 + rows_per_chunk)
        dec = _z12_unpack(p["buf"][r0 * bpr:r1 * bpr], r1 - r0, k)
        dec = dec.to(torch.int32).to(torch.int16).view(torch.bfloat16).float() * p["scale"]
        wf = w[r0:r1].float()
        hit = esc[r0:r1]
        if not torch.equal(torch.where(hit, wf, dec), wf):
            raise ValueError("z12: decode mismatch")
        if not bool((dec[hit] == 2.0 ** -15 * p["scale"]).all()):
            raise ValueError("z12: escape decode value")


def _z12_esclist(p: dict, n: int, bn: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Escapes as int2 {row_in_tile | k << 8, rebased bf16 bits of the true weight}, grouped by bn-row tile and
    sorted by (k, row); the decoder patches them into the smem tile, so the result stays exact."""
    rows, cols = p["rows"].long(), p["cols"].long()
    reb = (p["wtrue"] / p["scale"]).to(torch.bfloat16)
    if not torch.equal(reb.float() * p["scale"], p["wtrue"]):
        raise ValueError("z12: escape value not representable after rebasing")
    tile, rl = rows // bn, rows % bn
    order = torch.argsort((tile * (1 << 14) + cols) * 256 + rl)
    bits = reb.view(torch.int16).to(torch.int32) & 0xFFFF
    fl = torch.stack([(rl | (cols << 8))[order].to(torch.int32), bits[order]], dim=1).contiguous()
    if fl.numel() == 0:
        fl = torch.zeros(1, 2, dtype=torch.int32, device=rows.device)
    cnt = torch.bincount(tile, minlength=n // bn)
    if int(cnt.max()) > 1024:
        raise ValueError(f"z12: {int(cnt.max())} escapes in one {bn}-row tile")
    foff = torch.zeros(n // bn + 1, dtype=torch.int32, device=rows.device)
    foff[1:] = torch.cumsum(cnt, 0).to(torch.int32)
    return fl, foff


def _z12_run(p: dict, x: torch.Tensor, cid: int) -> torch.Tensor:
    out = torch.empty(x.shape[0], p["n"], device=x.device, dtype=torch.bfloat16)
    fl, foff = p["esc"][64 * Z12_CFGS[cid][0]]
    p["ext"].launch_z12(x, p["buf"], out, fl, foff, p["scale"], cid, 0)
    return out


def _z12_selftest(p: dict, w: torch.Tensor, cid: int, gen: torch.Generator) -> None:
    n, k = w.shape
    dev = w.device
    m = Z12_M
    pick = torch.randint(0, k, (m,), device=dev, generator=gen)
    esc_cols = p["cols"].long()
    if esc_cols.numel():
        ne = min(m // 2, esc_cols.numel())
        pick[:ne] = esc_cols[torch.randint(0, esc_cols.numel(), (ne,), device=dev, generator=gen)]
    xo = torch.zeros(m, k, device=dev, dtype=torch.bfloat16)
    xo[torch.arange(m, device=dev), pick] = 1.0
    if not torch.equal(_z12_run(p, xo, cid), w[:, pick].t()):
        raise ValueError("one-hot readback not bit-exact")
    x = torch.randn(m, k, device=dev, generator=gen).to(torch.bfloat16)
    x[:, ::97] *= 30.0
    ref = x.double() @ w.double().t()
    got = _z12_run(p, x, cid)
    gd = got.double()
    rel = float((gd - ref).norm() / ref.norm())
    mx = float((gd - ref).abs().max() / ref.abs().max())
    if not bool(torch.isfinite(gd).all()) or rel > 4e-3 or mx > 2e-2:
        raise ValueError(f"smoke rel={rel:.2e} max={mx:.2e}")
    for _ in range(Z12_DET_REPS):
        if not torch.equal(_z12_run(p, x, cid), got):
            raise ValueError("non-deterministic")


def _z12_time(graphs: dict, flush: torch.Tensor) -> dict:
    import statistics
    res = {nm: [] for nm in graphs}
    for r in range(Z12_BENCH_REPS):
        for nm in (list(graphs) if r % 2 == 0 else list(graphs)[::-1]):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            flush.sum()
            a.record()
            graphs[nm].replay()
            b.record()
            b.synchronize()
            res[nm].append(a.elapsed_time(b) * 1e3)
    return {nm: statistics.median(v) for nm, v in res.items()}


def _z12_graph(fn):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def _z12_prepare(state: dict) -> None:
    import time
    t_start = time.time()
    state["z12"] = {}
    if state.get("ext") is None:
        return
    try:
        ext = _z12_build()
        print(f"z12 build {time.time() - t_start:.1f}s")
    except Exception as exc:
        print(f"z12 path disabled: {type(exc).__name__}: {str(exc)[-3000:]}")
        return
    dev = state["pooler"].device
    e = torch.empty(0, device=dev)
    clusters = {}
    for cid in Z12_CFGS:
        try:
            clusters[cid] = int(ext.launch_z12(e, e, e, e, e, 1.0, cid, 1))
        except Exception as exc:
            print(f"z12 id={cid} occupancy query failed: {type(exc).__name__}: {exc}")
            clusters[cid] = 0
    print("z12 max clusters: " + " ".join(f"id{c}{Z12_CFGS[c]}={v}" for c, v in clusters.items()))
    gen = torch.Generator(device=dev).manual_seed(12)
    flush = torch.empty(64 * 2**20, device=dev, dtype=torch.float32)
    flush.normal_()
    picks = {}
    for layer, ids in Z12_PLAN.items():
        try:
            w = state[layer]
            n, k = w.shape
            p = _z12_pack(w)
            _z12_verify(p, w)
            p["esc"] = {bn: _z12_esclist(p, n, bn) for bn in (64, 128)}
            p["ext"], p["n"] = ext, n
            print(f"z12 pack {layer}: escapes={p['rows'].numel()}")
        except Exception as exc:
            print(f"z12 pack failed for {layer}: {type(exc).__name__}: {exc}")
            continue
        good = []
        for cid in ids:
            need = n // (64 * Z12_CFGS[cid][0])
            if need > clusters.get(cid, 0):
                print(f"z12 skip {layer} id={cid}: needs {need} clusters > {clusters.get(cid, 0)}")
                continue
            try:
                _z12_selftest(p, w, cid, gen)
                good.append(cid)
            except Exception as exc:
                print(f"z12 selftest FAILED {layer} id={cid}: {type(exc).__name__}: {str(exc)[-1500:]}")
        if not good:
            continue
        try:
            x = torch.randn(Z12_M, k, device=dev, generator=gen).to(torch.bfloat16)
            data = {"x": x, "config": {"m": Z12_M, "n": n, "k": k, "layer": layer}}
            graphs = {"cur": _z12_graph(lambda: kernel_fn(data, state))}
            for cid in good:
                graphs[f"id{cid}"] = _z12_graph(lambda c=cid: _z12_run(p, x, c))
            med = _z12_time(graphs, flush)
            best = min(good, key=lambda c: med[f"id{c}"])
            win = med[f"id{best}"] < med["cur"] - Z12_MARGIN_US
            print(f"z12 bench {layer} m={Z12_M} (raw us, ~4 graph overhead): "
                  + " ".join(f"{nm}={v:.2f}" for nm, v in med.items()) + (f" -> id{best}" if win else " -> cur"))
            if win:
                picks[layer] = {k2: p[k2] for k2 in ("buf", "esc", "scale", "ext", "n")} | {"cid": best}
        except Exception as exc:
            print(f"z12 bench error {layer}: {type(exc).__name__}: {exc}")
    state["z12"] = picks
    torch.cuda.synchronize()
    print(f"z12 prepare {time.time() - t_start:.1f}s picks={ {l: v['cid'] for l, v in picks.items()} }")


def _z12_try(x: torch.Tensor, layer: str, n: int, state: dict):
    p = state.get("z12", {}).get(layer)
    if p is None or x.shape[0] != Z12_M or n != p["n"] or x.shape[1] != state[layer].shape[1]:
        return None
    return _z12_run(p, x, p["cid"])
