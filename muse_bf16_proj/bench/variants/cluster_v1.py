# okc: problem=muse_bf16_proj
"""Dense projections of Muse Spark Open, tuned for small token counts on H200.

Every shape is one kernel: a swap-AB Triton GEMM (out^T = W @ x^T, weight rows on the MMA's
M side, tokens padded to the N side) with an in-kernel split-K reduction done by the last CTA
to finish each tile. The router runs on a BF16 hi | lo copy of its FP32 weights, concatenated
along K, so one K loop accumulates both halves in FP32. The BF16 layers at m <= 32 use their own
tiles: a CUDA-core GEMV at m = 1 and a swap-AB tl.dot (one token tile, so W is read once) above.

Where the split-K tail dominated (router m <= 32, the unpooler's 1024-row slice), CUDA kernels built
with load_inline in prepare() take over (Triton stays as the fallback):
- row_gemv: no split-K, each CTA owns whole rows (router m = 1 on the FP32 weights, slice m <= 4);
- reg_gemm: the split-K CTAs of a tile form a thread-block cluster and push their partials into the
  owner rank's shared memory (DSMEM), one cluster barrier, fixed summation order, no workspace.
"""

import hashlib
from typing import NamedTuple, TypedDict

import torch
import triton
import triton.language as tl

try:
    from triton.tools.tensor_descriptor import TensorDescriptor
except ImportError:
    TensorDescriptor = None


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int      # output width; the unpooler's is 1024 up to 8 tokens (this GPU's slice), else 8192
    k: int
    layer: str  # which weight: "qkv_gate", "o", "pooler", "unpooler" or "router"


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


class Tile(NamedTuple):
    max_m: int
    block_n: int
    block_k: int
    block_t: int
    split: int
    warps: int
    stages: int
    kind: str = "dot"  # BF16 layers: "fma" (m = 1), "dot" or "tma" (W through a TMA descriptor)


# (layer -> tiles by increasing max_m), tuned on H200. Above the last bucket the layer uses cuBLAS.
TILES: dict[str, tuple[Tile, ...]] = {
    "router": (
        Tile(16, 32, 256, 16, 16, 4, 4),
        Tile(32, 32, 512, 32, 16, 4, 3),
        Tile(128, 64, 256, 64, 16, 8, 3),
        Tile(256, 64, 256, 64, 8, 4, 3),
    ),
}
# BF16 layers, tuned on H200 (bench/sweep_gemv.py). The unpooler's 1024-row slice (m <= 8) has its
# own key. No tile -> cuBLAS (m > 32, and qkv_gate above 16 where cuBLAS ties).
BF16_TILES: dict[str, tuple[Tile, ...]] = {
    "pooler": (Tile(1, 4, 1024, 1, 1, 4, 3, "fma"), Tile(8, 32, 256, 16, 1, 4, 5),
               Tile(16, 32, 256, 16, 1, 4, 5), Tile(32, 64, 128, 32, 2, 4, 5)),
    "qkv_gate": (Tile(1, 4, 1024, 1, 1, 4, 2, "fma"), Tile(8, 64, 128, 16, 4, 4, 4),
                 Tile(16, 64, 128, 16, 4, 4, 4, "tma")),
    "o": (Tile(1, 16, 256, 1, 1, 8, 3, "fma"), Tile(8, 64, 128, 16, 1, 4, 6),
          Tile(16, 64, 128, 16, 1, 4, 5), Tile(32, 64, 128, 32, 1, 4, 5)),
    "unpooler": (Tile(16, 64, 128, 16, 1, 4, 6), Tile(32, 64, 256, 32, 1, 4, 4, "tma")),
    "unpooler_slice": (Tile(1, 4, 1024, 1, 1, 8, 2, "fma"), Tile(8, 64, 128, 16, 8, 4, 4)),
}
UNPOOL_SLICE = 1024
BF16_WARMUP_M = (1, 2, 5, 7, 8, 9, 16, 17, 32)
# One token: CUDA-core GEMV on the FP32 router weights beats padding the token to an MMA tile.
GEMV_FP32 = Tile(1, 8, 512, 1, 16, 4, 3)
WS_FLOATS = 2**20  # split-K partials: max over configs of split * padded_m * n
COUNTERS = 4096
WARMUP_M = (1, 5, 7, 8, 9, 16, 17, 32, 33, 128, 130, 256)


CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>

namespace cg = cooperative_groups;
using bf16 = __nv_bfloat16;

constexpr int WARPS = 8;
constexpr int THREADS = WARPS * 32;
constexpr int MAXC = 16;
constexpr int SMEM_LIMIT = 232448 - 1024;  // static mbarriers + reserve come out of the 227 KB

template <int MT, int NT, int P>
struct Cfg {
  static constexpr int BN = 16 * MT, BT = 8 * NT;
  // 16-byte loads per k32 chunk per thread; two stages of U chunks stay in flight (reg kernel).
  static constexpr int U = (MT * 2 * P + NT) <= 8 ? 2 : 1;
};

template <int MT, int NT, int P>
struct Stage {
  uint4 w[Cfg<MT, NT, P>::U][P][MT][2];
  uint4 x[Cfg<MT, NT, P>::U][NT];
};

__device__ __forceinline__ uint4 ld_stream(const bf16* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.s32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

__device__ __forceinline__ void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void cluster_arrive_release() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void cluster_wait() {
  asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2,
                                         uint32_t a3, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <int MT, int NT, int P>
__device__ __forceinline__ void mma_chunk(float (&acc)[MT][NT][4], const uint4 (&a)[P][MT][2],
                                          const uint4 (&b)[NT]) {
#pragma unroll
  for (int p = 0; p < P; ++p)
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        mma16816(acc[i][j], a[p][i][0].x, a[p][i][1].x, a[p][i][0].y, a[p][i][1].y, b[j].x, b[j].y);
        mma16816(acc[i][j], a[p][i][0].z, a[p][i][1].z, a[p][i][0].w, a[p][i][1].w, b[j].z, b[j].w);
      }
}

struct Tile {
  int csize, rank, BN, E, n0, t0, wk, wr, wkk;
};

// Warp partials -> smem red[wkk][E] -> per-CTA sum pushed to the owner rank's recv[rank][.] -> one
// release/acquire cluster barrier -> each owner sums its slice locally and stores it. Nobody reads
// remote smem, so no exit barrier. Every thread of the CTA must call this (the barriers are aligned).
template <int MT, int NT, int P>
__device__ __forceinline__ void cluster_reduce(const float (&acc)[MT][NT][4], bool has_acc, float* red,
                                               const Tile& T, int m, int n, void* out) {
  constexpr int BNW = 16 * MT;
  cg::cluster_group cluster = cg::this_cluster();
  const int tid = threadIdx.x, lane = tid & 31, g = lane >> 2, q = lane & 3;
  const int BN = T.BN, E = T.E;
  float* recv = red + T.wk * E;
  if (has_acc) {
    float* mine = red + T.wkk * E + T.wr * BNW;
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        const int r = 16 * i + g, c = 8 * j + 2 * q;
        mine[c * BN + r] = acc[i][j][0];
        mine[(c + 1) * BN + r] = acc[i][j][1];
        mine[c * BN + r + 8] = acc[i][j][2];
        mine[(c + 1) * BN + r + 8] = acc[i][j][3];
      }
  }
  __syncthreads();
  const int slice = E / T.csize;
  cluster_wait();
  if (tid < THREADS) {
    for (int e = tid; e < E; e += THREADS) {
      float v = red[e];
      for (int ww = 1; ww < T.wk; ++ww) v += red[ww * E + e];
      const int owner = e / slice;
      cluster.map_shared_rank(recv, owner)[T.rank * slice + e - owner * slice] = v;
    }
  }
  cluster_arrive_release();
  cluster_wait();
  if (tid < THREADS) {
    for (int el = tid; el < slice; el += THREADS) {
      float v = 0.f;
      for (int c = 0; c < T.csize; ++c) v += recv[c * slice + el];
      const int e = T.rank * slice + el, tt = e / BN;
      if (T.t0 + tt < m) {
        const long o = (long)(T.t0 + tt) * n + T.n0 + e % BN;
        if constexpr (P == 2) static_cast<float*>(out)[o] = v;
        else static_cast<bf16*>(out)[o] = __float2bfloat16(v);
      }
    }
  }
}

__device__ __forceinline__ Tile make_tile(int bnw, int bt, int wk, int warp) {
  cg::cluster_group cluster = cg::this_cluster();
  Tile T;
  T.csize = cluster.num_blocks();
  T.rank = cluster.block_rank();
  T.BN = bnw * (WARPS / wk);
  T.E = T.BN * bt;
  T.n0 = (blockIdx.x / T.csize) * T.BN;
  T.t0 = blockIdx.y * bt;
  T.wk = wk;
  T.wr = warp / wk;
  T.wkk = warp % wk;
  return T;
}

// out[t, n] = sum_p sum_k x[t, k] * w[p, n, k]; P = 2 is the router's hi | lo pair (fp32 out).
// The CTA's warps form (WARPS / wk) row groups of 16 MT rows times wk slices of each k range.
// mode (timing diagnostics only): 1 skips the cluster reduce, 2 skips the weight stream.
template <int MT, int NT, int P>
__global__ void __launch_bounds__(THREADS, 1)
reg_gemm(const bf16* __restrict__ x, const bf16* __restrict__ w, void* __restrict__ out, int m, int n,
         int k, int ldx, long wplane, int wk, int mode) {
  using C_ = Cfg<MT, NT, P>;
  constexpr int BNW = C_::BN, BT = C_::BT, U = C_::U;
  extern __shared__ __align__(16) unsigned char smem[];
  float* red = reinterpret_cast<float*>(smem);
  // Peers' smem must be live before the first DSMEM store; this arrive is off the critical path.
  cluster_arrive_relaxed();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
  const Tile T = make_tile(BNW, BT, wk, warp);
  const int kw = k / (T.csize * wk);
  const int kbeg = (T.rank * wk + T.wkk) * kw + 8 * q;
  const int nst = mode == 2 ? 0 : kw / (32 * U);
  const long rs8 = 8L * k;
  const bf16* wt = w + (long)(T.n0 + T.wr * BNW + g) * k + kbeg;
  const bf16* xt[NT];
  bool xv[NT];
#pragma unroll
  for (int j = 0; j < NT; ++j) {
    const int t = T.t0 + 8 * j + g;
    xv[j] = t < m;
    xt[j] = x + (long)(xv[j] ? t : 0) * ldx + kbeg;
  }

  float acc[MT][NT][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.f;

  auto load = [&](Stage<MT, NT, P>& s, int koff) {
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int ko = koff + 32 * u;
#pragma unroll
      for (int p = 0; p < P; ++p)
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
          for (int h = 0; h < 2; ++h) s.w[u][p][i][h] = ld_stream(wt + p * wplane + (2 * i + h) * rs8 + ko);
#pragma unroll
      for (int j = 0; j < NT; ++j)
        s.x[u][j] = xv[j] ? __ldg(reinterpret_cast<const uint4*>(xt[j] + ko)) : make_uint4(0, 0, 0, 0);
    }
  };
  auto compute = [&](const Stage<MT, NT, P>& s) {
#pragma unroll
    for (int u = 0; u < U; ++u) mma_chunk<MT, NT, P>(acc, s.w[u], s.x[u]);
  };

  Stage<MT, NT, P> s0, s1;
  if (nst > 0) load(s0, 0);
#pragma unroll 1
  for (int st = 0; st < nst; st += 2) {
    if (st + 1 < nst) load(s1, (st + 1) * 32 * U);
    compute(s0);
    if (st + 2 < nst) load(s0, (st + 2) * 32 * U);
    if (st + 1 < nst) compute(s1);
  }
  if (mode == 1) {
    float v = 0.f;
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < NT; ++j) v += acc[i][j][0] + acc[i][j][1] + acc[i][j][2] + acc[i][j][3];
    if (v == 12345.f) static_cast<float*>(out)[0] = v;
    cluster_wait();
    return;
  }
  cluster_reduce<MT, NT, P>(acc, true, red, T, m, n, out);
}

inline cudaLaunchConfig_t cluster_cfg(dim3 grid, int threads, int smem, int csize, cudaLaunchAttribute* attr) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = dim3(threads, 1, 1);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream().stream();
  attr->id = cudaLaunchAttributeClusterDimension;
  attr->val.clusterDim.x = csize;
  attr->val.clusterDim.y = 1;
  attr->val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  return cfg;
}

template <typename K>
void set_attrs_once(K kern, int smem_max) {
  C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max));
  C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
}

template <int MT, int NT, int P>
void launch_cfg(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& out, int csize, int wk,
                int kind, int ks, int nstage, int mode) {
  using C_ = Cfg<MT, NT, P>;
  const int m = x.size(0), k = x.size(1), n = out.size(1);
  TORCH_CHECK(wk == 1 || wk == 2 || wk == 4 || wk == 8, "cluster_gemm: wk");
  TORCH_CHECK((csize & (csize - 1)) == 0, "cluster_gemm: cluster size must be a power of two");
  const int bn = C_::BN * (WARPS / wk);
  TORCH_CHECK(n % bn == 0, "cluster_gemm: n=", n, " is not a multiple of BN=", bn);
  const int red_bytes = (WARPS * C_::BN + bn) * C_::BT * (int)sizeof(float);
  const long wplane = P == 2 ? (long)w.stride(0) : 0L;
  const dim3 grid(n / bn * csize, (m + C_::BT - 1) / C_::BT, 1);
  const bf16* xp = static_cast<const bf16*>(x.data_ptr());
  const bf16* wp = static_cast<const bf16*>(w.data_ptr());
  cudaLaunchAttribute attr;
  if (kind == 0) {
    TORCH_CHECK(k % (csize * wk * 32 * C_::U) == 0, "cluster_gemm: k=", k, " does not split by ", csize);
    auto kern = reg_gemm<MT, NT, P>;
    static const bool init = (set_attrs_once(kern, SMEM_LIMIT), true);
    (void)init;
    cudaLaunchConfig_t cfg = cluster_cfg(grid, THREADS, red_bytes, csize, &attr);
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, xp, wp, out.data_ptr(), m, n, k, (int)x.stride(0), wplane,
                                      wk, mode));
  } else {
    TORCH_CHECK(false, "cluster_gemm: only the register kernel is compiled");
  }
}

__device__ __forceinline__ void to_float(const uint4& v, float (&f)[8]) {
  const uint32_t u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    f[2 * i] = __uint_as_float(u[i] << 16);
    f[2 * i + 1] = __uint_as_float(u[i] & 0xffff0000u);
  }
}
__device__ __forceinline__ void to_float(const uint2& v, float (&f)[4]) {
  f[0] = __uint_as_float(v.x << 16);
  f[1] = __uint_as_float(v.x & 0xffff0000u);
  f[2] = __uint_as_float(v.y << 16);
  f[3] = __uint_as_float(v.y & 0xffff0000u);
}

// No split-K: each CTA owns R whole rows, so the only reduction is in-CTA (fixed order). Every warp
// load is 512 contiguous bytes of one row and all of a CTA's loads are issued before the first FMA.
template <bool F32, int R, int MT, int PER>
__global__ void __launch_bounds__(THREADS)
row_gemv(const bf16* __restrict__ x, const void* __restrict__ w, void* __restrict__ out, int m, int n, int k,
         int ldx) {
  constexpr int VEC = F32 ? 4 : 8;
  __shared__ float part[WARPS][R * MT];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int r0 = blockIdx.x * R;
  uint4 wv[R][PER];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int j = 0; j < PER; ++j)
      wv[r][j] = ld_stream(reinterpret_cast<const bf16*>(static_cast<const char*>(w) +
                                                         ((long)(r0 + r) * k + (long)(tid + THREADS * j) * VEC) *
                                                             (F32 ? 4 : 2)));
  float acc[R][MT];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int t = 0; t < MT; ++t) acc[r][t] = 0.f;
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const int kk = (tid + THREADS * j) * VEC;
    float xf[MT][VEC];
#pragma unroll
    for (int t = 0; t < MT; ++t) {
      if constexpr (F32) {
        const uint2 v = t < m ? __ldg(reinterpret_cast<const uint2*>(x + (long)t * ldx + kk)) : make_uint2(0, 0);
        to_float(v, xf[t]);
      } else {
        const uint4 v = t < m ? __ldg(reinterpret_cast<const uint4*>(x + (long)t * ldx + kk)) : make_uint4(0, 0, 0, 0);
        to_float(v, xf[t]);
      }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      float wf[VEC];
      if constexpr (F32) {
        wf[0] = __uint_as_float(wv[r][j].x);
        wf[1] = __uint_as_float(wv[r][j].y);
        wf[2] = __uint_as_float(wv[r][j].z);
        wf[3] = __uint_as_float(wv[r][j].w);
      } else {
        to_float(wv[r][j], wf);
      }
#pragma unroll
      for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int v = 0; v < VEC; ++v) acc[r][t] = fmaf(wf[v], xf[t][v], acc[r][t]);
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int t = 0; t < MT; ++t) {
      float v = acc[r][t];
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
      if (lane == 0) part[warp][r * MT + t] = v;
    }
  __syncthreads();
  if (tid < R * MT) {
    const int r = tid / MT, t = tid % MT;
    float v = 0.f;
#pragma unroll
    for (int ww = 0; ww < WARPS; ++ww) v += part[ww][tid];
    if (t < m) {
      const long o = (long)t * n + r0 + r;
      if constexpr (F32) static_cast<float*>(out)[o] = v;
      else static_cast<bf16*>(out)[o] = __float2bfloat16(v);
    }
  }
}

void launch_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t rows, int64_t mt, int64_t per) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && out.is_cuda(), "row_gemv: CUDA tensors expected");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.stride(1) == 1 && x.stride(0) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "row_gemv: x");
  const bool f32 = w.scalar_type() == at::kFloat;
  TORCH_CHECK(f32 || w.scalar_type() == at::kBFloat16, "row_gemv: w dtype");
  TORCH_CHECK(w.dim() == 2 && w.stride(1) == 1 && w.stride(0) == w.size(1) && w.size(1) == x.size(1) &&
                  reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0, "row_gemv: w");
  const int m = x.size(0), k = x.size(1), n = out.size(1);
  TORCH_CHECK(out.dim() == 2 && out.is_contiguous() && out.size(0) == m && n <= w.size(0) &&
                  out.scalar_type() == (f32 ? at::kFloat : at::kBFloat16), "row_gemv: out");
  TORCH_CHECK(m <= mt && n % rows == 0 && k == per * THREADS * (f32 ? 4 : 8), "row_gemv: shape");
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const bf16* xp = static_cast<const bf16*>(x.data_ptr());
#define G(F_, R_, MT_, PER_)                                                                           \
  if (f32 == F_ && rows == R_ && mt == MT_ && per == PER_) {                                          \
    row_gemv<F_, R_, MT_, PER_><<<n / R_, THREADS, 0, stream>>>(xp, w.data_ptr(), out.data_ptr(), m, n, k, \
                                                                (int)x.stride(0));                    \
    C10_CUDA_CHECK(cudaGetLastError());                                                               \
    return;                                                                                           \
  }
  GEMV_CONFIGS
#undef G
  TORCH_CHECK(false, "row_gemv: config not compiled");
}

void launch(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t mt, int64_t nt, int64_t csize,
            int64_t wk, int64_t kind, int64_t ks, int64_t nstage, int64_t mode) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && out.is_cuda(), "cluster_gemm: CUDA tensors expected");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16, "cluster_gemm: bf16 x, w");
  TORCH_CHECK(x.dim() == 2 && x.stride(1) == 1 && x.stride(0) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "cluster_gemm: x rows must be 16B aligned");
  const int p = w.dim() == 3 ? 2 : 1;
  TORCH_CHECK(w.size(-1) == x.size(1) && w.stride(-1) == 1 && w.stride(-2) == w.size(-1) &&
                  reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0, "cluster_gemm: w must be [.., n, k] row-major");
  TORCH_CHECK(p == 1 || (w.size(0) == 2 && w.stride(0) == w.size(1) * w.size(2)), "cluster_gemm: w planes");
  TORCH_CHECK(out.dim() == 2 && out.is_contiguous() && out.size(0) == x.size(0) && out.size(1) <= w.size(-2),
              "cluster_gemm: out shape");
  TORCH_CHECK(out.scalar_type() == (p == 2 ? at::kFloat : at::kBFloat16), "cluster_gemm: out dtype");
  TORCH_CHECK(csize >= 1 && csize <= MAXC, "cluster_gemm: cluster size");
#define X(MT_, NT_, P_)                                                                              \
  if (mt == MT_ && nt == NT_ && p == P_) {                                                           \
    launch_cfg<MT_, NT_, P_>(x, w, out, (int)csize, (int)wk, (int)kind, (int)ks, (int)nstage, (int)mode); \
    return;                                                                                          \
  }
  CONFIGS
#undef X
  TORCH_CHECK(false, "cluster_gemm: config (", mt, ", ", nt, ", ", p, ") not compiled");
}
"""
CUDA_CPP = ("void launch(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t mt, int64_t nt,"
            " int64_t csize, int64_t wk, int64_t kind, int64_t ks, int64_t nstage, int64_t mode);\n"
            "void launch_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t rows, int64_t mt,"
            " int64_t per);")
CUDA_FLAGS = ["-O3", "-gencode=arch=compute_90a,code=sm_90a"]
# key -> (min m, max m, kind, params), tuned on H200 (bench/sweep_cluster.py). gemv params: rows per
# CTA, max tokens, 16-byte loads per thread per row. cluster params: (mt, nt, cluster size, warps on k).
CUDA_TILES: dict[str, tuple[tuple[int, int, str, tuple[int, ...]], ...]] = {
    "router": ((1, 1, "gemv", (1, 1, 8)), (2, 8, "cluster", (1, 1, 8, 8)), (17, 32, "cluster", (1, 4, 4, 8))),
    "unpooler_slice": ((1, 1, "gemv", (2, 1, 2)), (2, 2, "gemv", (2, 2, 2)), (3, 4, "gemv", (4, 4, 2)),
                       (5, 8, "cluster", (1, 1, 2, 8))),
}


def _build_cuda():
    from torch.utils.cpp_extension import load_inline

    clusters = sorted({(p[0], p[1], 2 if key == "router" else 1)
                       for key, tiles in CUDA_TILES.items() for _, _, kind, p in tiles if kind == "cluster"})
    gemvs = sorted({(key == "router", *p)
                    for key, tiles in CUDA_TILES.items() for _, _, kind, p in tiles if kind == "gemv"})
    src = ("#define CONFIGS " + " ".join(f"X({a},{b},{c})" for a, b, c in clusters) + "\n"
           + "#define GEMV_CONFIGS " + " ".join(f"G({str(f).lower()},{r},{t},{p})" for f, r, t, p in gemvs)
           + "\n" + CUDA_SRC)
    tag = hashlib.sha1((src + " ".join(CUDA_FLAGS)).encode()).hexdigest()[:10]
    return load_inline(name=f"muse_proj_cuda_{tag}", cpp_sources=CUDA_CPP, cuda_sources=src,
                       functions=["launch", "launch_gemv"], extra_cuda_cflags=CUDA_FLAGS, extra_cflags=["-O3"])


def _cuda_pick(state: dict, key: str, m: int):
    if state.get("ext") is None:
        return None
    for lo, hi, kind, params in CUDA_TILES.get(key, ()):
        if lo <= m <= hi:
            return kind, params
    return None


def _cuda_run(state: dict, x, w, n: int, dtype, kind: str, params: tuple[int, ...]) -> torch.Tensor:
    out = torch.empty(x.shape[0], n, device=x.device, dtype=dtype)
    if kind == "gemv":
        state["ext"].launch_gemv(x, w, out, *params)
    else:
        state["ext"].launch(x, w, out, *params, 0, 0, 0, 0)
    return out


@triton.jit
def _split_k_finish(acc, ws_ptrs, ws_stride, out_ptrs, mask, cnt_ptr, pid_s, SPLIT: tl.constexpr):
    """Store this split's partial; the last CTA of the tile sums all splits in order."""
    tl.store(ws_ptrs + pid_s * ws_stride, acc, mask=mask)
    # All warps' partials must be stored before thread 0's release; .cg skips the
    # non-coherent L1, which may hold this SM's lines from an earlier launch.
    tl.debug_barrier()
    done = tl.atomic_add(cnt_ptr, 1, sem="acq_rel", scope="gpu")
    if done == SPLIT - 1:
        total = tl.zeros_like(acc)
        # Unrolled so the partial loads overlap instead of one L2 round trip per split.
        for s in tl.static_range(SPLIT):
            total += tl.load(ws_ptrs + s * ws_stride, mask=mask, other=0.0, cache_modifier=".cg")
        tl.store(out_ptrs, total.to(out_ptrs.dtype.element_ty), mask=mask)
        tl.atomic_xchg(cnt_ptr, 0)


@triton.jit(do_not_specialize=["M"])
def _skinny_gemm(x_ptr, w_ptr, out_ptr, ws_ptr, cnt_ptr, M, stride_xm,
                 N: tl.constexpr, KX: tl.constexpr, KW: tl.constexpr, SPLIT: tl.constexpr,
                 BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_T: tl.constexpr):
    # W is [N, KW]; x columns are indexed k % KX, so KW = 2 * KX reads x twice for hi | lo.
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_t = tl.program_id(2)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = offs_t < M
    k_len: tl.constexpr = KW // SPLIT
    k_start = pid_s * k_len

    acc = tl.zeros((BLOCK_N, BLOCK_T), dtype=tl.float32)
    for kk in range(0, k_len, BLOCK_K):
        offs_k = k_start + kk + tl.arange(0, BLOCK_K)
        w = tl.load(w_ptr + offs_n[:, None] * KW + offs_k[None, :], eviction_policy="evict_first")
        # A [BLOCK_K, BLOCK_T] x^T load; measured faster than loading [BLOCK_T, BLOCK_K] + tl.trans.
        xt = tl.load(x_ptr + offs_t[None, :] * stride_xm + (offs_k % KX)[:, None],
                     mask=t_mask[None, :], other=0.0, eviction_policy="evict_last")
        acc = tl.dot(w, xt, acc)

    out_ptrs = out_ptr + offs_t[None, :] * N + offs_n[:, None]
    if SPLIT == 1:
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=t_mask[None, :])
    else:
        rows = tl.num_programs(2) * BLOCK_T
        ws_ptrs = ws_ptr + offs_t[None, :] * N + offs_n[:, None]
        _split_k_finish(acc, ws_ptrs, rows * N, out_ptrs, t_mask[None, :],
                        cnt_ptr + pid_t * tl.num_programs(0) + pid_n, pid_s, SPLIT)


@triton.jit
def _gemv_fp32(x_ptr, w_ptr, out_ptr, ws_ptr, cnt_ptr, N: tl.constexpr, K: tl.constexpr,
               SPLIT: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_len: tl.constexpr = K // SPLIT
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for kk in range(0, k_len, BLOCK_K):
        offs_k = pid_s * k_len + kk + tl.arange(0, BLOCK_K)
        w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :], eviction_policy="evict_first")
        xv = tl.load(x_ptr + offs_k).to(tl.float32)
        acc += tl.sum(w * xv[None, :], axis=1)
    _split_k_finish(acc, ws_ptr + offs_n, N, out_ptr + offs_n, offs_n < N, cnt_ptr + pid_n, pid_s,
                    SPLIT)


@triton.jit(do_not_specialize=["M"])
def _bf16_gemm(x_ptr, w_ptr, w_desc, out_ptr, ws_ptr, cnt_ptr, M, stride_xm,
               N: tl.constexpr, K: tl.constexpr, SPLIT: tl.constexpr, BLOCK_N: tl.constexpr,
               BLOCK_K: tl.constexpr, BLOCK_T: tl.constexpr, USE_TMA: tl.constexpr):
    # One token tile (BLOCK_T >= M): each W tile feeds every token, so W streams from HBM once.
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_t = tl.arange(0, BLOCK_T)
    t_mask = offs_t < M
    k_len: tl.constexpr = K // SPLIT
    k_start = pid_s * k_len
    acc = tl.zeros((BLOCK_N, BLOCK_T), dtype=tl.float32)
    for kk in range(0, k_len, BLOCK_K):
        offs_k = tl.max_contiguous(tl.multiple_of(k_start + kk + tl.arange(0, BLOCK_K), BLOCK_K), BLOCK_K)
        if USE_TMA:
            w = w_desc.load([pid_n * BLOCK_N, k_start + kk])
        else:
            w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :], eviction_policy="evict_first")
        xr = tl.load(x_ptr + offs_t[:, None] * stride_xm + offs_k[None, :],
                     mask=t_mask[:, None], other=0.0, eviction_policy="evict_last")
        acc = tl.dot(w, tl.trans(xr), acc)
    out_ptrs = out_ptr + offs_t[None, :] * N + offs_n[:, None]
    if SPLIT == 1:
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=t_mask[None, :])
    else:
        ws_ptrs = ws_ptr + offs_t[None, :] * N + offs_n[:, None]
        _split_k_finish(acc, ws_ptrs, BLOCK_T * N, out_ptrs, t_mask[None, :], cnt_ptr + pid_n, pid_s,
                        SPLIT)


@triton.jit
def _bf16_gemv(x_ptr, w_ptr, out_ptr, ws_ptr, cnt_ptr, N: tl.constexpr, K: tl.constexpr,
               SPLIT: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # m = 1: FMA into a [BLOCK_N, BLOCK_K] accumulator, one reduction at the end.
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_len: tl.constexpr = K // SPLIT
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for kk in range(0, k_len, BLOCK_K):
        offs_k = tl.max_contiguous(tl.multiple_of(pid_s * k_len + kk + tl.arange(0, BLOCK_K), BLOCK_K), BLOCK_K)
        w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :], eviction_policy="evict_first")
        xv = tl.load(x_ptr + offs_k, eviction_policy="evict_last")
        acc += w.to(tl.float32) * xv.to(tl.float32)[None, :]
    r = tl.sum(acc, axis=1)
    if SPLIT == 1:
        tl.store(out_ptr + offs_n, r.to(out_ptr.dtype.element_ty))
    else:
        _split_k_finish(r, ws_ptr + offs_n, N, out_ptr + offs_n, offs_n < N, cnt_ptr + pid_n, pid_s,
                        SPLIT)


def _pick(layer: str, m: int) -> Tile | None:
    for tile in TILES.get(layer, ()):
        if m <= tile.max_m:
            return tile
    return None


def _launch(x, w, out, state, tile: Tile, kx: int) -> None:
    m, n, kw = x.shape[0], w.shape[0], w.shape[1]
    grid = (n // tile.block_n, tile.split, triton.cdiv(m, tile.block_t))
    _skinny_gemm[grid](x, w, out, state["ws"], state["cnt"], m, x.stride(0),
                       N=n, KX=kx, KW=kw, SPLIT=tile.split, BLOCK_N=tile.block_n,
                       BLOCK_K=tile.block_k, BLOCK_T=tile.block_t,
                       num_warps=tile.warps, num_stages=tile.stages)


def _bf16_key(layer: str, n: int, w_rows: int) -> str:
    return "unpooler_slice" if layer == "unpooler" and n == UNPOOL_SLICE < w_rows else layer


def _bf16_pick(key: str, m: int) -> Tile | None:
    for tile in BF16_TILES.get(key, ()):
        if m <= tile.max_m:
            return tile if tile.kind != "fma" or m == 1 else None
    return None


def _bf16_launch(x, w, out, state, key: str, tile: Tile) -> None:
    m, (n, k) = x.shape[0], w.shape
    ws, cnt = state["ws"], state["cnt"]
    if tile.kind == "fma":
        _bf16_gemv[(n // tile.block_n, tile.split)](
            x, w, out, ws, cnt, N=n, K=k, SPLIT=tile.split, BLOCK_N=tile.block_n,
            BLOCK_K=tile.block_k, num_warps=tile.warps, num_stages=tile.stages)
        return
    desc = state["desc"][(key, tile)] if tile.kind == "tma" else w
    _bf16_gemm[(n // tile.block_n, tile.split)](
        x, w, desc, out, ws, cnt, m, x.stride(0), N=n, K=k, SPLIT=tile.split,
        BLOCK_N=tile.block_n, BLOCK_K=tile.block_k, BLOCK_T=tile.block_t,
        USE_TMA=tile.kind == "tma", num_warps=tile.warps, num_stages=tile.stages)


def _check_bf16_tiles(state: dict) -> None:
    for key, tiles in BF16_TILES.items():
        n, k = state[key].shape
        for t in tiles:
            if n % t.block_n or k % (t.split * t.block_k) or t.block_t < t.max_m:
                raise ValueError(f"{key} tile {t} does not fit n={n}, k={k}")
            if t.kind == "fma" and t.max_m != 1:
                raise ValueError(f"{key} tile {t}: fma is single-token")
            if t.split > 1 and (t.split * t.block_t * n > WS_FLOATS or n // t.block_n > COUNTERS):
                raise ValueError(f"{key} tile {t} needs more split-K workspace or counters")
            if t.kind == "tma":
                if TensorDescriptor is None:
                    raise RuntimeError("TMA tile without triton.tools.tensor_descriptor")
                state["desc"][(key, t)] = TensorDescriptor.from_tensor(state[key], [t.block_n, t.block_k])


def _check_tiles(n_by_layer: dict[str, tuple[int, int]]) -> None:
    for layer, tiles in TILES.items():
        n, kw = n_by_layer[layer]
        for t in tiles:
            rows = triton.cdiv(t.max_m, t.block_t) * t.block_t
            if n % t.block_n or kw % (t.split * t.block_k):
                raise ValueError(f"{layer} tile {t} does not divide n={n}, k={kw}")
            if t.split > 1 and t.split * rows * n > WS_FLOATS:
                raise ValueError(f"{layer} tile {t} needs more split-K workspace")
            if (n // t.block_n) * triton.cdiv(t.max_m, t.block_t) > COUNTERS:
                raise ValueError(f"{layer} tile {t} needs more tile counters")


def prepare(weights: Weights) -> dict:
    """Repack once: router as BF16 hi | lo [256, 16384]; allocate split-K workspace; warm JIT."""
    router = weights["router"].float()
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state = {name: w.contiguous() for name, w in weights.items()}
    try:
        state["ext"] = _build_cuda()
    except Exception as exc:  # Triton covers every shape on its own
        print(f"CUDA extension unavailable, using Triton only: {type(exc).__name__}: {exc}")
        state["ext"] = None
    state["router_hl"] = torch.stack([hi, lo]).contiguous()
    state["router_hilo"] = torch.cat([hi, lo], dim=1).contiguous()
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    device = router.device
    state["ws"] = torch.empty(WS_FLOATS, device=device, dtype=torch.float32)
    state["cnt"] = torch.zeros(COUNTERS, device=device, dtype=torch.int32)
    _check_tiles({"router": tuple(state["router_hilo"].shape)})
    n, k = router.shape
    g = GEMV_FP32
    if n % g.block_n or k % (g.split * g.block_k) or g.split * n > WS_FLOATS or n // g.block_n > COUNTERS:
        raise ValueError(f"router GEMV tile {g} does not fit n={n}, k={k}")
    for m in WARMUP_M:
        x = torch.zeros(m, router.shape[1], device=device, dtype=torch.bfloat16)
        kernel_fn({"x": x, "config": {"m": m, "n": 256, "k": router.shape[1], "layer": "router"}},
                  state)
    state["unpooler_slice"] = state["unpooler"][:UNPOOL_SLICE]
    state["desc"] = {}
    _check_bf16_tiles(state)
    for layer in ("qkv_gate", "o", "pooler", "unpooler"):
        w_rows, k = state[layer].shape
        for m in BF16_WARMUP_M:
            n = UNPOOL_SLICE if layer == "unpooler" and m <= 8 else w_rows
            x = torch.zeros(m, k, device=device, dtype=torch.bfloat16)
            kernel_fn({"x": x, "config": {"m": m, "n": n, "k": k, "layer": layer}}, state)
    torch.cuda.synchronize()
    return state


def _router(x: torch.Tensor, state: dict) -> torch.Tensor:
    m = x.shape[0]
    pick = _cuda_pick(state, "router", m)
    if pick is not None:
        kind, params = pick
        w = state["router"] if kind == "gemv" else state["router_hl"]
        return _cuda_run(state, x, w, state["router"].shape[0], torch.float32, kind, params)
    if m <= GEMV_FP32.max_m:
        w, g = state["router"], GEMV_FP32
        out = torch.empty(m, w.shape[0], device=x.device, dtype=torch.float32)
        _gemv_fp32[(w.shape[0] // g.block_n, g.split)](
            x, w, out, state["ws"], state["cnt"], N=w.shape[0], K=w.shape[1], SPLIT=g.split,
            BLOCK_N=g.block_n, BLOCK_K=g.block_k, num_warps=g.warps, num_stages=g.stages)
        return out
    tile = _pick("router", m)
    if tile is None:
        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
        return acc
    w = state["router_hilo"]
    out = torch.empty(m, w.shape[0], device=x.device, dtype=torch.float32)
    _launch(x, w, out, state, tile, kx=x.shape[1])
    return out


def kernel_fn(data: Inputs, state: dict) -> torch.Tensor:
    """Return x @ W[:n].T: bfloat16, or float32 for the router."""
    x, cfg = data["x"], data["config"]
    if x.stride(1) != 1:
        x = x.contiguous()
    if x.data_ptr() % 16 or x.stride(0) % 8:  # the CUDA kernels load 16-byte pieces of x rows
        x = x.clone(memory_format=torch.contiguous_format)
    if cfg["layer"] == "router":
        return _router(x, state)
    w_full = state[cfg["layer"]]
    key = _bf16_key(cfg["layer"], cfg["n"], w_full.shape[0])
    w = state[key] if cfg["n"] == state[key].shape[0] else w_full[: cfg["n"]]
    pick = _cuda_pick(state, key, x.shape[0]) if w is state[key] else None
    if pick is not None:
        return _cuda_run(state, x, w, w.shape[0], torch.bfloat16, *pick)
    tile = _bf16_pick(key, x.shape[0]) if w is state[key] else None
    if tile is None:
        return x @ w.T
    out = torch.empty(x.shape[0], w.shape[0], device=x.device, dtype=torch.bfloat16)
    _bf16_launch(x, w, out, state, key, tile)
    return out
