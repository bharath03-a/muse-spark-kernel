"""r8 m=128 compressed GEMM (v2: register-accumulated escape fix, S in {2,3,4,5}): register-A wgmma m64n128k16 with in-register 12-bit decode.

CTA = 2 consumer warpgroups (64 W rows each, same 128-token x stage) + 1 producer warpgroup
(warp 8: W bulk copy + x 2D TMA; warps 9-11: escape fix-up). Split-K S over a cluster of S CTAs; partials
reduced through DSMEM into the rank that owns each token slice. This block is spliced into the candidate.
"""


def _z8_gen_wgmma() -> str:
    d = ", ".join(f"%{i}" for i in range(64))
    outs = ", ".join(f'"+f"(d[{j}][{r}])' for j in range(16) for r in range(4))
    return (
        "__device__ __forceinline__ void z8_wgmma(float (&d)[16][4], uint32_t a0, uint32_t a1, uint32_t a2,"
        " uint32_t a3, uint64_t db) {\n"
        '  asm volatile("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "\n'
        f'      "{{{d}}}, {{%64, %65, %66, %67}}, %68, 1, 1, 1, 0;\\n"\n'
        f"      : {outs}\n"
        '      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));\n'
        "}\n")


Z8_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda.h>

namespace cg = cooperative_groups;
using bf16 = __nv_bfloat16;

__device__ __forceinline__ uint32_t z8_su(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void z8_cl_arrive_relaxed() { asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void z8_cl_arrive_release() { asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void z8_cl_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory"); }
__device__ __forceinline__ uint32_t z8_prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
  return r;
}
__device__ __forceinline__ uint32_t z8_lop(uint32_t a) {
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(0x87878787u), "r"(0x38383838u));
  return r;
}
__device__ __forceinline__ void z8_mbar_init(uint64_t* b, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(z8_su(b)), "r"(count) : "memory");
}
__device__ __forceinline__ void z8_expect_tx(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(z8_su(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void z8_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(z8_su(b)) : "memory");
}
__device__ __forceinline__ void z8_wait(uint64_t* b, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred P1;\n Z8_WAIT:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      " @P1 bra Z8_DONE;\n bra Z8_WAIT;\n Z8_DONE:\n}\n"
      ::"r"(z8_su(b)), "r"(parity) : "memory");
}
__device__ __forceinline__ uint64_t z8_evict_first() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
  return pol;
}
__device__ __forceinline__ void z8_bulk(void* dst, const void* src, uint32_t bytes, uint64_t* bar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
      ::"r"(z8_su(dst)), "l"(src), "r"(bytes), "r"(z8_su(bar)), "l"(pol) : "memory");
}
__device__ __forceinline__ void z8_tma_x(void* dst, const CUtensorMap* map, int k, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
      ::"r"(z8_su(dst)), "l"(map), "r"(k), "r"(0), "r"(z8_su(bar)) : "memory");
}
// K-major operand, 128B swizzle, 8-row groups 1024 B apart
__device__ __forceinline__ uint64_t z8_desc(uint32_t saddr) {
  return (uint64_t)((saddr >> 4) & 0x3FFF) | (1ull << 16) | (64ull << 32) | (1ull << 62);
}
__device__ __forceinline__ void z8_wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void z8_wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void z8_wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"); }
// keeps A fragments live (and unmoved) until the wgmma that reads them has retired
__device__ __forceinline__ void z8_keep(uint32_t (&a)[16]) {
#pragma unroll
  for (int i = 0; i < 16; ++i) asm volatile("" : "+r"(a[i])::"memory");
}
__device__ __forceinline__ void z8_keep_acc(float (&d)[16][4]) {
#pragma unroll
  for (int j = 0; j < 16; ++j)
#pragma unroll
    for (int r = 0; r < 4; ++r) asm volatile("" : "+f"(d[j][r])::"memory");
}

inline CUtensorMap z8_xmap(const void* ptr, int rows, int cols, int row_stride) {
  CUtensorMap map{};
  cuuint64_t gdim[2] = {(cuuint64_t)cols, (cuuint64_t)rows};
  cuuint64_t gstride[1] = {(cuuint64_t)row_stride * sizeof(bf16)};
  cuuint32_t bdim[2] = {64u, 128u};
  cuuint32_t estride[2] = {1u, 1u};
  CUresult res = cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), gdim, gstride,
                                        bdim, estride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(res == CUDA_SUCCESS, "z8: cuTensorMapEncodeTiled failed ", (int)res);
  return map;
}
""" + _z8_gen_wgmma() + r"""
constexpr int Z8_RP = 132;           // padded row stride (floats) of a reduce slot: conflict-free
constexpr uint32_t Z8_WST = 12288;   // 128 rows x 64 k x 1.5 B
constexpr uint32_t Z8_XST = 16384;   // 128 tokens x 64 k x 2 B
constexpr int Z8_NFW = 3;            // escape fix-up warps

template <int S>
struct Z8Geo {
  static constexpr int JPR = (16 + S - 1) / S;   // max 8-token column blocks per rank
  static constexpr int TSM = 8 * JPR;            // tokens per reduce slot
  static constexpr int RSLOT = TSM * Z8_RP;      // floats per slot; S slots = (S-1) partials + fix
  __device__ static constexpr int owner(int j) { return (j * S) >> 4; }
  __device__ static constexpr int j0(int q) { return (16 * q + S - 1) / S; }
};

// one 64-k unit: decode this warp's 16x64 W block into A fragments, 4 async k16 wgmmas on the shared x stage
template <int D>
__device__ __forceinline__ void z8_step(int i, uint32_t (&a)[16], uint32_t (&ap)[16], float (&acc)[16][4],
                                        unsigned char* sm, uint64_t* full, uint64_t* empty, int blk, int lane) {
  const int s = i % D;
  z8_wait(&full[s], (i / D) & 1);
  const uint4* src = reinterpret_cast<const uint4*>(sm + D * Z8_XST + s * Z8_WST + blk * 1536) + lane;
  const uint4 w0 = src[0], w1 = src[32], wb = src[64];
  const uint32_t* p0 = reinterpret_cast<const uint32_t*>(&w0);
  const uint32_t* p1 = reinterpret_cast<const uint32_t*>(&w1);
  const uint32_t* pb = reinterpret_cast<const uint32_t*>(&wb);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint32_t h0 = z8_lop(pb[j]);
    const uint32_t h1 = z8_lop(__funnelshift_l(pb[j], pb[j], 4));
    a[j * 4 + 0] = z8_prmt(p0[j], h0, 0x5140);
    a[j * 4 + 1] = z8_prmt(p0[j], h0, 0x7362);
    a[j * 4 + 2] = z8_prmt(p1[j], h1, 0x5140);
    a[j * 4 + 3] = z8_prmt(p1[j], h1, 0x7362);
  }
  z8_keep(a);
  z8_wg_fence();
  const uint64_t db = z8_desc(z8_su(sm + s * Z8_XST));
#pragma unroll
  for (int j = 0; j < 4; ++j) z8_wgmma(acc, a[j * 4], a[j * 4 + 1], a[j * 4 + 2], a[j * 4 + 3], db + (uint64_t)(2 * j));
  z8_wg_commit();
  z8_wg_wait<1>();
  z8_keep(ap);
  if (i > 0 && lane == 0) z8_arrive(&empty[(i - 1) % D]);
}

template <int S, int D>
__global__ void __launch_bounds__(384, 1)
z8_kernel(const __grid_constant__ CUtensorMap xmap, const unsigned char* __restrict__ wp, const bf16* __restrict__ x,
          bf16* __restrict__ out, const int2* __restrict__ fl, const int* __restrict__ foff, int m, int N, int K,
          int xstride, float scale) {
  using G = Z8Geo<S>;
  constexpr uint32_t WOFF = D * Z8_XST, ROFF = WOFF + D * Z8_WST, BOFF = ROFF + S * G::RSLOT * 4;
  extern __shared__ __align__(1024) unsigned char z8_raw[];
  unsigned char* sm = z8_raw + ((1024 - (z8_su(z8_raw) & 1023)) & 1023);
  float* recv = reinterpret_cast<float*>(sm + ROFF);
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + BOFF);
  uint64_t* empty = full + D;

  cg::cluster_group cluster = cg::this_cluster();
  const int rank = S > 1 ? (int)cluster.block_rank() : 0;
  if constexpr (S > 1) z8_cl_arrive_relaxed();

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int tile = blockIdx.x / S;
  const int U = K >> 6, u0 = rank * U / S, nk = (rank + 1) * U / S - u0;

  if (tid == 0) {
    for (int d = 0; d < D; ++d) { z8_mbar_init(&full[d], 1); z8_mbar_init(&empty[d], 8); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();

  float acc[16][4];
  if (warp == 8) {
    if (lane == 0) {
      const uint64_t pol = z8_evict_first();
      const unsigned char* gw = wp + ((size_t)tile * U + u0) * Z8_WST;
      for (int i = 0; i < nk; ++i) {
        const int s = i % D;
        if (i >= D) {
          z8_wait(&empty[s], ((i / D) - 1) & 1);
          asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        }
        z8_expect_tx(&full[s], Z8_WST + Z8_XST);  // x rows >= m are zero-filled and still counted
        z8_bulk(sm + WOFF + s * Z8_WST, gw + (size_t)i * Z8_WST, Z8_WST, &full[s], pol);
        z8_tma_x(sm + s * Z8_XST, &xmap, (u0 + i) * 64, &full[s]);
      }
    }
    __syncwarp();
    if constexpr (S > 1) z8_cl_wait();
  } else if (warp > 8) {
    if constexpr (S > 1) z8_cl_wait();
    const int o = rank * Z8_NFW + (warp - 9), O = S * Z8_NFW;
    float* fp[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int tok = lane + 32 * i, q = G::owner(tok >> 3);
      float* base = recv + (S - 1) * G::RSLOT + (tok - 8 * G::j0(q)) * Z8_RP;
      if constexpr (S > 1) fp[i] = q == rank ? base : cluster.map_shared_rank(base, q);
      else fp[i] = base;
    }
    for (int row = o; row < 128; row += O) {
#pragma unroll
      for (int i = 0; i < 4; ++i) fp[i][row] = 0.f;
    }
    // run-length accumulate per row in registers; one store per (row, token) -- no remote read-modify-write
    const int e1 = foff[tile * O + o + 1];
    float cur[4] = {0.f, 0.f, 0.f, 0.f};
    int crow = -1;
    for (int e = foff[tile * O + o]; e < e1; e += 8) {
      int2 en[8];
      float xv[8][4];
#pragma unroll
      for (int b = 0; b < 8; ++b) en[b] = e + b < e1 ? fl[e + b] : make_int2(0, 0);
#pragma unroll
      for (int b = 0; b < 8; ++b) {
        const int kk = en[b].x >> 8;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int tok = lane + 32 * i;
          xv[b][i] = (e + b < e1 && tok < m) ? __bfloat162float(x[(size_t)tok * xstride + kk]) : 0.f;
        }
      }
#pragma unroll
      for (int b = 0; b < 8; ++b) {
        if (e + b < e1) {
          const int row = en[b].x & 255;
          if (row != crow) {
            if (crow >= 0) {
#pragma unroll
              for (int i = 0; i < 4; ++i) fp[i][crow] = cur[i];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) cur[i] = 0.f;
            crow = row;
          }
          const float dv = __int_as_float(en[b].y);
#pragma unroll
          for (int i = 0; i < 4; ++i) cur[i] += dv * xv[b][i];
        }
      }
    }
    if (crow >= 0) {
#pragma unroll
      for (int i = 0; i < 4; ++i) fp[i][crow] = cur[i];
    }
  } else {
    const int blk = warp;  // WG0 = blocks 0-3, WG1 = blocks 4-7 of the 128-row tile
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[j][r] = 0.f;
    uint32_t a0[16], a1[16];
#pragma unroll
    for (int r = 0; r < 16; ++r) { a0[r] = 0u; a1[r] = 0u; }
    int i = 0;
    for (; i + 2 <= nk; i += 2) {
      z8_step<D>(i, a0, a1, acc, sm, full, empty, blk, lane);
      z8_step<D>(i + 1, a1, a0, acc, sm, full, empty, blk, lane);
    }
    if (i < nk) z8_step<D>(i, a0, a1, acc, sm, full, empty, blk, lane);
    z8_wg_wait<0>();
    z8_keep_acc(acc);
    z8_keep(a0);
    z8_keep(a1);
    if constexpr (S > 1) {
      z8_cl_wait();
      const int g = lane >> 2, t = lane & 3, r0 = blk * 16 + g;
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const int q = G::owner(j);
        if (q != rank) {
          const int slot = rank < q ? rank : rank - 1;
          const int tl = 8 * (j - G::j0(q)) + 2 * t;
          float* dst = cluster.map_shared_rank(recv + slot * G::RSLOT, q);
          dst[tl * Z8_RP + r0] = acc[j][0];
          dst[(tl + 1) * Z8_RP + r0] = acc[j][1];
          dst[tl * Z8_RP + r0 + 8] = acc[j][2];
          dst[(tl + 1) * Z8_RP + r0 + 8] = acc[j][3];
        }
      }
    }
  }
  if constexpr (S > 1) {
    z8_cl_arrive_release();
    z8_cl_wait();
  } else {
    __syncthreads();
  }
  if (warp < 8) {
    const int g = lane >> 2, t = lane & 3, r0 = warp * 16 + g;
    const float* fix = recv + (S - 1) * G::RSLOT;
    bf16* ob = out + (size_t)tile * 128;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      if (G::owner(j) == rank) {
        const int tl = 8 * (j - G::j0(rank)) + 2 * t, tok = 8 * j + 2 * t;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const int row = r0 + (r >> 1) * 8, tk = tl + (r & 1);
          float v = acc[j][r];
#pragma unroll
          for (int sl = 0; sl < S - 1; ++sl) v += recv[sl * G::RSLOT + tk * Z8_RP + row];
          const float o = v * scale + fix[tk * Z8_RP + row];
          if (tok + (r & 1) < m) ob[(size_t)(tok + (r & 1)) * N + row] = __float2bfloat16(o);
        }
      }
    }
  }
}

template <int S, int D>
void z8_launch(const torch::Tensor& x, const torch::Tensor& wp, torch::Tensor& out, const torch::Tensor& fl,
               const torch::Tensor& foff, double scale) {
  using G = Z8Geo<S>;
  const int m = x.size(0), K = x.size(1), N = out.size(1);
  TORCH_CHECK(m >= 1 && m <= 128 && N % 128 == 0 && K % 64 == 0 && K / 64 >= S && out.size(0) == m, "z8: shape");
  TORCH_CHECK(wp.numel() == (int64_t)N * K * 3 / 2 && foff.numel() == (int64_t)(N / 128) * S * Z8_NFW + 1,
              "z8: packed size");
  TORCH_CHECK(x.stride(1) == 1 && x.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "z8: x layout");
  const size_t smem = 1024 + (size_t)D * (Z8_XST + Z8_WST) + (size_t)S * G::RSLOT * 4 + (size_t)2 * D * 8;
  TORCH_CHECK(smem <= 232448, "z8: smem ", smem);
  auto kern = z8_kernel<S, D>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(kern),
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((N / 128) * S, 1, 1);
  cfg.blockDim = dim3(384, 1, 1);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream().stream();
  cudaLaunchAttribute attr;
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = S;
  attr.val.clusterDim.y = 1;
  attr.val.clusterDim.z = 1;
  cfg.attrs = &attr;
  cfg.numAttrs = S > 1 ? 1 : 0;
  const CUtensorMap xmap = z8_xmap(x.data_ptr(), m, K, (int)x.stride(0));
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, xmap, static_cast<const unsigned char*>(wp.data_ptr()),
                                    static_cast<const bf16*>(x.data_ptr()), static_cast<bf16*>(out.data_ptr()),
                                    static_cast<const int2*>(fl.data_ptr()), static_cast<const int*>(foff.data_ptr()),
                                    m, N, K, (int)x.stride(0), (float)scale));
}

template <int S, int D>
int64_t z8_clusters() {
  using G = Z8Geo<S>;
  const size_t smem = 1024 + (size_t)D * (Z8_XST + Z8_WST) + (size_t)S * G::RSLOT * 4 + (size_t)2 * D * 8;
  auto kern = z8_kernel<S, D>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(kern),
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(132 * S, 1, 1);
  cfg.blockDim = dim3(384, 1, 1);
  cfg.dynamicSmemBytes = smem;
  cudaLaunchAttribute attr;
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = S;
  attr.val.clusterDim.y = 1;
  attr.val.clusterDim.z = 1;
  cfg.attrs = &attr;
  cfg.numAttrs = 1;
  int n = -1;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&n, reinterpret_cast<const void*>(kern), &cfg));
  return n;
}

#define Z8_CASE(ID, S_, D_)                                   \
  if (id == ID) {                                             \
    if (query) return z8_clusters<S_, D_>();                  \
    z8_launch<S_, D_>(x, wp, out, fl, foff, scale);           \
    return 0;                                                 \
  }

int64_t launch_z8(torch::Tensor x, torch::Tensor wp, torch::Tensor out, torch::Tensor fl, torch::Tensor foff,
                  double scale, int64_t id, int64_t query) {
  Z8_CASE(1, 2, 5) Z8_CASE(2, 4, 5) Z8_CASE(5, 3, 5) Z8_CASE(6, 5, 4)
  TORCH_CHECK(false, "z8: config id not compiled ", id);
  return 0;
}
"""
Z8_CPP = ("int64_t launch_z8(torch::Tensor x, torch::Tensor wp, torch::Tensor out, torch::Tensor fl, "
          "torch::Tensor foff, double scale, int64_t id, int64_t query);")
# config id -> split S (cluster size)
Z8_SPLIT = {1: 2, 2: 4, 5: 3, 6: 5}
# layer -> (config id, lo_m, hi_m) dispatched; Z8_BENCH: (layer, config id) timed in prepare (log only)
Z8_PLAN = {"pooler": (5, 97, 128), "unpooler": (1, 97, 128), "qkv_gate": (6, 97, 128)}
Z8_BENCH = (("pooler", 5), ("pooler", 2), ("unpooler", 1), ("qkv_gate", 6), ("qkv_gate", 5),
            ("qkv_gate", 2))
Z8_NFW_PY = 3


def _z8_build():
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((Z8_SRC + " ".join(CUDA_FLAGS)).encode()).hexdigest()[:10]
    return load_inline(name=f"muse_z8_{tag}", cpp_sources=Z8_CPP, cuda_sources=Z8_SRC, functions=["launch_z8"],
                       extra_cuda_cflags=CUDA_FLAGS, extra_cflags=["-O3"], extra_ldflags=CUDA_LDFLAGS)


def _z8_pack(w: torch.Tensor) -> dict:
    """cp7's 12-bit format, but natural k order inside each k16 step (wgmma reads x straight from smem) and
    tile-major records: [N/128, K/64, 8 blocks, 3 planes, 32 lanes, 16 B]."""
    n, k = w.shape
    rb = 8
    if w.dtype != torch.bfloat16 or n % 128 or k % 64:
        raise ValueError(f"z8: cannot pack {tuple(w.shape)} {w.dtype}")
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


def _z8_unpack(buf: torch.Tensor, n: int, k: int) -> torch.Tensor:
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


def _z8_verify(p: dict, w: torch.Tensor, rows_per_chunk: int = 512) -> None:
    n, k = w.shape
    bpr = k * 3 // 2
    esc = torch.zeros(n, k, dtype=torch.bool, device=w.device)
    esc[p["rows"], p["cols"]] = True
    for r0 in range(0, n, rows_per_chunk):
        r1 = min(n, r0 + rows_per_chunk)
        dec = _z8_unpack(p["buf"][r0 * bpr:r1 * bpr], r1 - r0, k)
        dec = dec.to(torch.int32).to(torch.int16).view(torch.bfloat16).float() * p["scale"]
        wf = w[r0:r1].float()
        hit = esc[r0:r1]
        if not torch.equal(torch.where(hit, wf, dec), wf):
            raise ValueError("z8: decode mismatch")
        if not bool((dec[hit] == 2.0 ** -15 * p["scale"]).all()):
            raise ValueError("z8: escape decode value")


def _z8_fixlist(p: dict, n: int, s: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Escapes as int2 {row_in_tile | k << 8, fp32 bits of (w - decoded)} grouped by (tile, fix owner)."""
    o_cnt = s * Z8_NFW_PY
    rows, cols = p["rows"].long(), p["cols"].long()
    tile, rl = rows // 128, rows % 128
    grp = tile * o_cnt + rl % o_cnt
    key = (grp * 128 + rl) * 16384 + cols
    order = torch.argsort(key)
    cesc = torch.tensor(2.0 ** -15 * p["scale"], dtype=torch.float32, device=rows.device)
    delta = (p["wtrue"] - cesc)[order]
    fl = torch.stack([(rl | (cols << 8))[order].to(torch.int32), delta.view(torch.int32)], dim=1).contiguous()
    if fl.numel() == 0:
        fl = torch.zeros(1, 2, dtype=torch.int32, device=rows.device)
    cnt = torch.bincount(grp, minlength=(n // 128) * o_cnt)
    foff = torch.zeros((n // 128) * o_cnt + 1, dtype=torch.int32, device=rows.device)
    foff[1:] = torch.cumsum(cnt, 0).to(torch.int32)
    return fl, foff


def _z8_run(p: dict, x: torch.Tensor, cid: int) -> torch.Tensor:
    out = torch.empty(x.shape[0], p["n"], device=x.device, dtype=torch.bfloat16)
    fl, foff = p["fix"][Z8_SPLIT[cid]]
    p["ext"].launch_z8(x, p["buf"], out, fl, foff, p["scale"], cid, 0)
    return out


def _z8_selftest(p: dict, w: torch.Tensor, cid: int, lo: int, hi: int) -> None:
    n, k = w.shape
    dev = w.device
    gen = torch.Generator(device=dev).manual_seed(5)
    # one-hot rows must read back W columns exactly (layout + escapes); include escaped columns
    esc_cols = p["cols"].long()
    for m in sorted({lo, hi}):
        pick = torch.randint(0, k, (m,), device=dev, generator=gen)
        if esc_cols.numel():
            ne = min(m // 2, esc_cols.numel())
            pick[:ne] = esc_cols[torch.randint(0, esc_cols.numel(), (ne,), device=dev, generator=gen)]
        xo = torch.zeros(m, k, device=dev, dtype=torch.bfloat16)
        xo[torch.arange(m, device=dev), pick] = 1.0
        if not torch.equal(_z8_run(p, xo, cid), w[:, pick].t()):
            raise ValueError(f"z8: one-hot mismatch at m={m}")
    for m in sorted({lo, (lo + hi) // 2, hi - 1, hi}):
        x = torch.randn(m, k, device=dev, generator=gen).to(torch.bfloat16)
        x[:, :: 97] *= 30.0
        ref = x.double() @ w.double().t()
        got = _z8_run(p, x, cid).double()
        err = (got - ref).abs().max() / ref.abs().max()
        rel = float((got - ref).norm() / ref.norm())
        if rel > 4e-3 or float(err) > 2e-2:
            raise ValueError(f"z8: smoke failed at m={m} rel={rel:.2e} max={float(err):.2e}")
    x = torch.randn(hi, k, device=dev, generator=gen).to(torch.bfloat16)
    first = _z8_run(p, x, cid)
    for _ in range(20):
        if not torch.equal(_z8_run(p, x, cid), first):
            raise ValueError("z8: nondeterministic")


def _z8_prepare(state: dict) -> None:
    state["z8"] = {}
    if state.get("ext") is None:
        return
    try:
        ext = _z8_build()
    except Exception as exc:
        print(f"z8 path disabled: {type(exc).__name__}: {str(exc)[-3000:]}")
        return
    want = {}
    for layer, cid in Z8_BENCH:
        want.setdefault(layer, set()).add(cid)
    for layer, (cid, _, _) in Z8_PLAN.items():
        want.setdefault(layer, set()).add(cid)
    packs = {}
    for layer, cids in want.items():
        try:
            w = state[layer]
            p = _z8_pack(w)
            _z8_verify(p, w)
            p["ext"], p["n"] = ext, w.shape[0]
            p["fix"] = {Z8_SPLIT[c]: _z8_fixlist(p, w.shape[0], Z8_SPLIT[c]) for c in cids}
            packs[layer] = p
            print(f"z8 pack {layer}: escapes={p['rows'].numel()}")
        except Exception as exc:
            print(f"z8 pack failed for {layer}: {type(exc).__name__}: {exc}")
    good = set()
    for layer, cids in want.items():
        if layer not in packs:
            continue
        for cid in sorted(cids):
            try:
                lo, hi = Z8_PLAN[layer][1:] if Z8_PLAN.get(layer, (None,))[0] == cid else (97, 128)
                _z8_selftest(packs[layer], state[layer], cid, lo, hi)
                good.add((layer, cid))
                print(f"z8 selftest ok: {layer} id={cid} S={Z8_SPLIT[cid]} "
                      f"max_clusters={ext.launch_z8(torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0), 1.0, cid, 1)}")
            except Exception as exc:
                print(f"z8 selftest FAILED {layer} id={cid}: {type(exc).__name__}: {str(exc)[-2000:]}")
    for layer, (cid, lo, hi) in Z8_PLAN.items():
        if (layer, cid) in good:
            p = dict(packs[layer])
            p["cid"], p["lo"], p["hi"] = cid, lo, hi
            state["z8"][layer] = p
            print(f"z8 path enabled: {layer} id={cid} m={lo}..{hi}")
    try:
        _z8_bench(state, packs, good)
    except Exception as exc:
        print(f"z8 bench error: {type(exc).__name__}: {exc}")


def _z8_try(x: torch.Tensor, layer: str, n: int, state: dict):
    p = state["z8"].get(layer)
    if p is None or n != p["n"] or x.shape[1] != state[layer].shape[1] or not p["lo"] <= x.shape[0] <= p["hi"]:
        return None
    return _z8_run(p, x, p["cid"])


def _z8_bench(state: dict, packs: dict, good: set) -> None:
    """Log-only cold-L2 graph timings: z8 variants vs the cp7 path at m=128."""
    import statistics
    flush = torch.empty(64 * 2**20, device="cuda", dtype=torch.float32)
    flush.normal_()

    def graph(fn):
        fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        return g

    def once(g):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        flush.sum()
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        return a.elapsed_time(b) * 1e3

    saved = state["z8"]
    state["z8"] = {}
    try:
        for layer in sorted({l for l, _ in Z8_BENCH}):
            w = state[layer]
            x = torch.randn(128, w.shape[1], device="cuda").to(torch.bfloat16)
            data = {"x": x, "config": {"m": 128, "n": w.shape[0], "k": w.shape[1], "layer": layer}}
            gs = {"cp7": graph(lambda: kernel_fn(data, state))}
            for l2, cid in Z8_BENCH:
                if l2 == layer and (layer, cid) in good:
                    pk = packs[layer]
                    gs[f"z8id{cid}"] = graph(lambda c=cid, q=pk: _z8_run(q, x, c))
                    fl, foff = pk["fix"][Z8_SPLIT[cid]]
                    pn = dict(pk, fix={Z8_SPLIT[cid]: (fl, torch.zeros_like(foff))})
                    gs[f"nofix{cid}"] = graph(lambda c=cid, q=pn: _z8_run(q, x, c))
            res = {nm: [] for nm in gs}
            for r in range(40):
                for nm in (list(gs) if r % 2 == 0 else list(gs)[::-1]):
                    res[nm].append(once(gs[nm]))
            print(f"z8 bench {layer} m=128 (raw us incl. ~4 graph overhead): "
                  + " ".join(f"{nm}={statistics.median(v):.2f}" for nm, v in res.items()))
    finally:
        state["z8"] = saved
