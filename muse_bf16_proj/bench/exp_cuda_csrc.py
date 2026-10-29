"""Custom CUDA C++ (sm_90a) kernel experiments for Muse Spark projections:
1. Close the router m=9..16 hole with reg_gemm<1, 2, 2> (BT=16) and tune router m=1, 8, 16, 32.
2. Extend row_gemv<F32, R, MT, PER, KWARPS, CHUNK> with j-ordered x+w prefetch and flexible KWARPS
   (1, 2, 4, 8 warps per row) to m=1 (decode 1) on qkv_gate, pooler, o, and router.
3. Extend reg_gemm<MT, NT, P, U_OVR> with zero-DSMEM csize=1 fast path to qkv_gate, o, and pooler
   at m=8, 16, 32.

Usage:
    python bench/exp_cuda_csrc.py --compile
    python bench/exp_cuda_csrc.py [--reps 40] [--only router] [--gemv-only] [--cluster-only]
"""

from __future__ import annotations

import argparse
import hashlib
import statistics
import sys
import time
from typing import NamedTuple

import torch

SRC = r"""
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
constexpr int SMEM_LIMIT = 232448 - 1024;

template <int MT, int NT, int P, int U_OVR = 0>
struct Cfg {
  static constexpr int BN = 16 * MT, BT = 8 * NT;
  // 16-byte loads per k32 chunk per thread; two stages of U chunks stay in flight (reg kernel).
  static constexpr int U = U_OVR > 0 ? U_OVR : ((MT * 2 * P + NT) <= 8 ? 2 : 1);
};

template <int MT, int NT, int P, int U_OVR = 0>
struct Stage {
  uint4 w[Cfg<MT, NT, P, U_OVR>::U][P][MT][2];
  uint4 x[Cfg<MT, NT, P, U_OVR>::U][NT];
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

// Warp partials -> smem red[wkk][E].
// If T.csize == 1: directly sum across T.wk in smem and store to out (no DSMEM or second cluster barrier).
// If T.csize > 1: push per-CTA sum to owner rank's recv[rank][.] -> one release/acquire cluster barrier -> store.
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
  cluster_wait();
  if (T.csize == 1) {
    if (tid < THREADS) {
      for (int e = tid; e < E; e += THREADS) {
        float v = red[e];
        for (int ww = 1; ww < T.wk; ++ww) v += red[ww * E + e];
        const int tt = e / BN;
        if (T.t0 + tt < m) {
          const long o = (long)(T.t0 + tt) * n + T.n0 + e % BN;
          if constexpr (P == 2) static_cast<float*>(out)[o] = v;
          else static_cast<bf16*>(out)[o] = __float2bfloat16(v);
        }
      }
    }
    return;
  }
  const int slice = E / T.csize;
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

template <int MT, int NT, int P, int U_OVR = 0>
__global__ void __launch_bounds__(THREADS, 1)
reg_gemm(const bf16* __restrict__ x, const bf16* __restrict__ w, void* __restrict__ out, int m, int n,
         int k, int ldx, long wplane, int wk, int mode) {
  using C_ = Cfg<MT, NT, P, U_OVR>;
  constexpr int BNW = C_::BN, BT = C_::BT, U = C_::U;
  extern __shared__ __align__(16) unsigned char smem[];
  float* red = reinterpret_cast<float*>(smem);
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

  auto load = [&](Stage<MT, NT, P, U_OVR>& s, int koff) {
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
  auto compute = [&](const Stage<MT, NT, P, U_OVR>& s) {
#pragma unroll
    for (int u = 0; u < U; ++u) mma_chunk<MT, NT, P>(acc, s.w[u], s.x[u]);
  };

  Stage<MT, NT, P, U_OVR> s0, s1;
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

template <int MT, int NT, int P, int U_OVR>
void launch_cfg(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& out, int csize, int wk,
                int kind, int ks, int nstage, int mode) {
  using C_ = Cfg<MT, NT, P, U_OVR>;
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
  TORCH_CHECK(kind == 0, "cluster_gemm: only the register kernel is compiled");
  TORCH_CHECK(k % (csize * wk * 32 * C_::U) == 0, "cluster_gemm: k=", k, " does not split by ", csize);
  auto kern = reg_gemm<MT, NT, P, U_OVR>;
  static const bool init = (set_attrs_once(kern, SMEM_LIMIT), true);
  (void)init;
  cudaLaunchConfig_t cfg = cluster_cfg(grid, THREADS, red_bytes, csize, &attr);
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, xp, wp, out.data_ptr(), m, n, k, (int)x.stride(0), wplane,
                                    wk, mode));
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

// Generalized row_gemv:
// - KWARPS warps cooperate along K on each row (NGROUPS = WARPS / KWARPS independent row-groups per CTA).
// - Each row-group owns R rows (CTA owns NGROUPS * R rows).
// - Outer loop steps in chunks of CHUNK (dividing PER), issuing both x (__ldg) and w (ld_stream)
//   in j-outer, r-inner order so j=0 FMAs start as soon as the first MT + R loads return.
// - When KWARPS == 1, reduction is pure warp-shuffle (no smem or __syncthreads).
template <bool F32, int R, int MT, int PER, int KWARPS = WARPS, int CHUNK = PER>
__global__ void __launch_bounds__(THREADS)
row_gemv(const bf16* __restrict__ x, const void* __restrict__ w, void* __restrict__ out, int m, int n, int k,
         int ldx) {
  static_assert(WARPS % KWARPS == 0, "KWARPS must divide WARPS");
  static_assert(PER % CHUNK == 0, "CHUNK must divide PER");
  constexpr int VEC = F32 ? 4 : 8;
  constexpr int NGROUPS = WARPS / KWARPS;
  constexpr int KTHREADS = KWARPS * 32;
  __shared__ float part[NGROUPS][KWARPS][R * MT];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int gid = warp / KWARPS, kwid = warp % KWARPS, ktid = kwid * 32 + lane;
  const int r0 = (blockIdx.x * NGROUPS + gid) * R;

  float acc[R][MT];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int t = 0; t < MT; ++t) acc[r][t] = 0.f;

  using XVec = std::conditional_t<F32, uint2, uint4>;
#pragma unroll
  for (int c0 = 0; c0 < PER; c0 += CHUNK) {
    XVec xv[CHUNK][MT];
    uint4 wv[CHUNK][R];
#pragma unroll
    for (int j = 0; j < CHUNK; ++j) {
      const int kk = (ktid + KTHREADS * (c0 + j)) * VEC;
#pragma unroll
      for (int t = 0; t < MT; ++t) {
        if constexpr (F32) {
          xv[j][t] = t < m ? __ldg(reinterpret_cast<const uint2*>(x + (long)t * ldx + kk)) : make_uint2(0, 0);
        } else {
          xv[j][t] = t < m ? __ldg(reinterpret_cast<const uint4*>(x + (long)t * ldx + kk)) : make_uint4(0, 0, 0, 0);
        }
      }
#pragma unroll
      for (int r = 0; r < R; ++r) {
        wv[j][r] = ld_stream(reinterpret_cast<const bf16*>(
            static_cast<const char*>(w) + ((long)(r0 + r) * k + (long)kk) * (F32 ? 4 : 2)));
      }
    }
#pragma unroll
    for (int j = 0; j < CHUNK; ++j) {
      float xf[MT][VEC];
#pragma unroll
      for (int t = 0; t < MT; ++t) to_float(xv[j][t], xf[t]);
#pragma unroll
      for (int r = 0; r < R; ++r) {
        float wf[VEC];
        if constexpr (F32) {
          wf[0] = __uint_as_float(wv[j][r].x);
          wf[1] = __uint_as_float(wv[j][r].y);
          wf[2] = __uint_as_float(wv[j][r].z);
          wf[3] = __uint_as_float(wv[j][r].w);
        } else {
          to_float(wv[j][r], wf);
        }
#pragma unroll
        for (int t = 0; t < MT; ++t)
#pragma unroll
          for (int v = 0; v < VEC; ++v) acc[r][t] = fmaf(wf[v], xf[t][v], acc[r][t]);
      }
    }
  }

#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int t = 0; t < MT; ++t) {
      float v = acc[r][t];
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
      if constexpr (KWARPS == 1) {
        if (lane == 0 && t < m) {
          const long o = (long)t * n + r0 + r;
          if constexpr (F32) static_cast<float*>(out)[o] = v;
          else static_cast<bf16*>(out)[o] = __float2bfloat16(v);
        }
      } else {
        if (lane == 0) part[gid][kwid][r * MT + t] = v;
      }
    }
  if constexpr (KWARPS > 1) {
    __syncthreads();
    if (ktid < R * MT) {
      const int r = ktid / MT, t = ktid % MT;
      float v = 0.f;
#pragma unroll
      for (int ww = 0; ww < KWARPS; ++ww) v += part[gid][ww][ktid];
      if (t < m) {
        const long o = (long)t * n + r0 + r;
        if constexpr (F32) static_cast<float*>(out)[o] = v;
        else static_cast<bf16*>(out)[o] = __float2bfloat16(v);
      }
    }
  }
}

void launch_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t rows, int64_t mt, int64_t per,
                 int64_t kwarps, int64_t chunk) {
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
  const int ngroups = WARPS / (int)kwarps;
  const int r_cta = ngroups * (int)rows;
  TORCH_CHECK(m <= mt && n % r_cta == 0 && k == per * (kwarps * 32) * (f32 ? 4 : 8), "row_gemv: shape");
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const bf16* xp = static_cast<const bf16*>(x.data_ptr());
#define G(F_, R_, MT_, PER_, KW_, CH_)                                                                        \
  if (f32 == F_ && rows == R_ && mt == MT_ && per == PER_ && kwarps == KW_ && chunk == CH_) {                 \
    row_gemv<F_, R_, MT_, PER_, KW_, CH_><<<n / r_cta, THREADS, 0, stream>>>(xp, w.data_ptr(), out.data_ptr(), \
                                                                             m, n, k, (int)x.stride(0));      \
    C10_CUDA_CHECK(cudaGetLastError());                                                                       \
    return;                                                                                                   \
  }
  GEMV_CONFIGS
#undef G
  TORCH_CHECK(false, "row_gemv: config not compiled");
}

void launch(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t mt, int64_t nt, int64_t csize,
            int64_t wk, int64_t kind, int64_t ks, int64_t nstage, int64_t mode, int64_t u_ovr) {
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
#define X(MT_, NT_, P_, U_)                                                                                  \
  if (mt == MT_ && nt == NT_ && p == P_ && u_ovr == U_) {                                                    \
    launch_cfg<MT_, NT_, P_, U_>(x, w, out, (int)csize, (int)wk, (int)kind, (int)ks, (int)nstage, (int)mode); \
    return;                                                                                                  \
  }
  CONFIGS
#undef X
  TORCH_CHECK(false, "cluster_gemm: config (", mt, ", ", nt, ", ", p, ", ", u_ovr, ") not compiled");
}
"""

CPP = ("void launch(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t mt, int64_t nt, int64_t csize,"
       " int64_t wk, int64_t kind, int64_t ks, int64_t nstage, int64_t mode, int64_t u_ovr);\n"
       "void launch_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t rows, int64_t mt,"
       " int64_t per, int64_t kwarps, int64_t chunk);")
CFLAGS = ["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v"]

# (MT, NT, P, U_OVR): BN = 16 * MT * (8 / wk), BT = 8 * NT, P = 2 for router hi|lo.
CONFIGS_ALL = (
    # Router (P = 2): m=8 (NT=1), m=16 (NT=2), m=32 (NT=4, default U=2 and U_OVR=1)
    (1, 1, 2, 0), (2, 1, 2, 0),
    (1, 2, 2, 0), (2, 2, 2, 0),
    (1, 4, 2, 0), (1, 4, 2, 1), (2, 4, 2, 0),
    # BF16 projections (P = 1): m=8 (NT=1), m=16 (NT=2), m=32 (NT=4)
    (1, 1, 1, 0), (2, 1, 1, 0), (4, 1, 1, 0),
    (1, 2, 1, 0), (2, 2, 1, 0), (4, 2, 1, 0),
    (1, 4, 1, 0), (2, 4, 1, 0), (4, 4, 1, 0),
)

# row_gemv (f32, rows_per_group, mt, per, kwarps, chunk)
GEMV_ALL = (
    # Router (f32, N=256, K=8192):
    (True, 1, 1, 8, 8, 8), (True, 1, 1, 8, 8, 4), (True, 2, 1, 8, 8, 4), (True, 1, 1, 16, 4, 4),
    # unpooler_slice (bf16, N=1024, K=4096):
    (False, 2, 1, 2, 8, 2), (False, 4, 1, 2, 8, 2), (False, 2, 2, 2, 8, 2), (False, 4, 4, 2, 8, 2),
    # qkv_gate (bf16, N=2304, K=8192) & pooler (bf16, N=4096, K=8192):
    (False, 1, 1, 4, 8, 4), (False, 2, 1, 4, 8, 4), (False, 3, 1, 4, 8, 4),
    (False, 4, 1, 4, 8, 4), (False, 4, 1, 4, 8, 2),
    (False, 6, 1, 4, 8, 2), (False, 8, 1, 4, 8, 2), (False, 9, 1, 4, 8, 2),
    (False, 1, 1, 8, 4, 4), (False, 2, 1, 8, 4, 4), (False, 3, 1, 8, 4, 4), (False, 4, 1, 8, 4, 2),
    # o (bf16, N=8192, K=1024):
    (False, 1, 1, 1, 4, 1), (False, 2, 1, 1, 4, 1), (False, 4, 1, 1, 4, 1), (False, 8, 1, 1, 4, 1),
    (False, 1, 1, 2, 2, 2), (False, 2, 1, 2, 2, 2), (False, 4, 1, 2, 2, 2),
    (False, 1, 1, 4, 1, 4), (False, 2, 1, 4, 1, 4), (False, 4, 1, 4, 1, 4),
)

WARPS = 8
SMEM_LIMIT = 232448 - 1024
OVERHEAD_US = 4.0

GEMV_CASES = [
    ("router", 1),
    ("qkv_gate", 1),
    ("pooler", 1),
    ("o", 1),
    ("unpooler", 1),
    ("unpooler", 2),
    ("unpooler", 4),
]
CLUSTER_CASES = [
    ("router", 8),
    ("router", 16),
    ("router", 32),
    ("qkv_gate", 8),
    ("qkv_gate", 16),
    ("qkv_gate", 32),
    ("o", 8),
    ("o", 16),
    ("o", 32),
    ("pooler", 8),
    ("pooler", 16),
    ("pooler", 32),
    ("unpooler", 8),
]
NEARBY = {
    1: (1,),
    2: (2,),
    4: (3, 4),
    8: (5, 7, 8),
    16: (9, 12, 16),
    32: (17, 24, 31, 32),
}
CSIZES = (1, 2, 4, 8, 16)
WKS = (8, 4, 2, 1)


class CTile(NamedTuple):
    mt: int
    nt: int
    csize: int
    wk: int = WARPS
    u_ovr: int = 0


class GTile(NamedTuple):
    rows: int
    mt: int
    per: int
    kwarps: int = WARPS
    chunk: int = 0


def stage_chunks(mt: int, nt: int, p: int, u_ovr: int = 0) -> int:
    if u_ovr > 0:
        return u_ovr
    return 2 if mt * 2 * p + nt <= 8 else 1


def block_n(t: CTile) -> int:
    return 16 * t.mt * (WARPS // t.wk)


def valid(t: CTile, p: int, n: int, k: int) -> bool:
    if n % block_n(t) != 0:
        return False
    u = stage_chunks(t.mt, t.nt, p, t.u_ovr)
    return k % (t.csize * t.wk * 32 * u) == 0


def source(configs=CONFIGS_ALL, gemv=GEMV_ALL) -> str:
    return (
        "#define CONFIGS " + " ".join(f"X({a},{b},{c},{u})" for a, b, c, u in configs) + "\n"
        + "#define GEMV_CONFIGS "
        + " ".join(f"G({str(f).lower()},{r},{t},{p},{kw},{ch})" for f, r, t, p, kw, ch in gemv)
        + "\n" + SRC
    )


def build(configs=CONFIGS_ALL, gemv=GEMV_ALL, verbose: bool = False):
    from torch.utils.cpp_extension import load_inline

    src = source(configs, gemv)
    tag = hashlib.sha1((src + " ".join(CFLAGS)).encode()).hexdigest()[:10]
    return load_inline(
        name=f"exp_cuda_csrc_{tag}",
        cpp_sources=CPP,
        cuda_sources=src,
        functions=["launch", "launch_gemv"],
        extra_cuda_cflags=CFLAGS,
        extra_cflags=["-O3"],
        verbose=verbose,
    )


def run_cluster(ext, x: torch.Tensor, w: torch.Tensor, n: int, t: CTile, mode: int = 0) -> torch.Tensor:
    dtype = torch.float32 if w.dim() == 3 else torch.bfloat16
    out = torch.empty(x.shape[0], n, device=x.device, dtype=dtype)
    ext.launch(x, w, out, t.mt, t.nt, t.csize, t.wk, 0, 0, 0, mode, t.u_ovr)
    return out


def run_gemv(ext, x: torch.Tensor, w: torch.Tensor, n: int, t: GTile) -> torch.Tensor:
    out = torch.empty(x.shape[0], n, device=x.device, dtype=w.dtype)
    ext.launch_gemv(x, w, out, t.rows, t.mt, t.per, t.kwarps, t.chunk or t.per)
    return out


def graph_of(fn):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    torch.cuda.synchronize()
    return g, out


def cluster_candidates(layer: str, m: int, n: int, k: int):
    p = 2 if layer == "router" else 1
    max_c = 16 if layer == "router" else (2 if layer == "o" else 4)
    min_ctas = 32 if layer == "router" else 64
    max_ctas = 256 if layer == "router" else 576
    for mt, nt, pp, u_ovr in CONFIGS_ALL:
        if pp != p:
            continue
        bt = 8 * nt
        if not (bt >= m and (bt < 2 * m or nt == 1)):
            continue
        for c in CSIZES:
            if c > max_c or (layer == "router" and c == 1):
                continue
            for wk in WKS:
                if layer == "router" and wk < 4:
                    continue
                t = CTile(mt, nt, c, wk, u_ovr)
                if not valid(t, p, n, k):
                    continue
                ctas = (n // block_n(t)) * c * -(-m // bt)
                if min_ctas <= ctas <= max_ctas:
                    yield t


def gemv_candidates(layer: str, m: int, n: int, k: int):
    f32 = layer == "router"
    vec = 4 if f32 else 8
    for f, r, mt, per, kw, ch in GEMV_ALL:
        r_cta = (WARPS // kw) * r
        if (
            f == f32
            and mt >= m
            and (mt < 2 * m or mt == 1)
            and n % r_cta == 0
            and k == per * (kw * 32) * vec
        ):
            ctas = n // r_cta
            if 32 <= ctas <= 2048:
                yield GTile(r, mt, per, kw, ch)


def _screen_and_time(sides: dict, extra_keys: tuple[str, ...], flush, reps: int, time_replay, top_k: int = 5) -> dict:
    cand_keys = [k for k in sides if k not in extra_keys]
    if len(cand_keys) > top_k:
        s_times = {k: [] for k in cand_keys}
        for rep in range(5):
            order = cand_keys if rep % 2 == 0 else cand_keys[::-1]
            for k in order:
                s_times[k].append(time_replay(sides[k], flush) - OVERHEAD_US)
        cand_keys = sorted(cand_keys, key=lambda k: statistics.median(s_times[k]))[:top_k]
    names = cand_keys + [k for k in extra_keys if k in sides]
    times = {nm: [] for nm in names}
    for rep in range(reps):
        order = names[rep % len(names):] + names[: rep % len(names)]
        if rep % 2:
            order = order[::-1]
        for nm in order:
            times[nm].append(time_replay(sides[nm], flush) - OVERHEAD_US)
    return {nm: statistics.median(v) for nm, v in times.items()}


def main() -> None:
    if "--compile" in sys.argv:
        t0 = time.time()
        build(verbose=True)
        print(
            f"built {len(CONFIGS_ALL)} reg_gemm + {len(GEMV_ALL)} row_gemv configs in {time.time() - t0:.1f}s",
            flush=True,
        )
        return

    from common import load_module, reference, weights
    from harness import FLUSH_BYTES, time_replay

    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--only", default="")
    ap.add_argument("--base", default="checkpoints/cp2_gluon_router.py")
    ap.add_argument("--gemv-only", action="store_true")
    ap.add_argument("--cluster-only", action="store_true")
    args = ap.parse_args()
    t_start = time.time()

    ext = build(verbose=False)
    print(f"ext ready in {time.time() - t_start:.1f}s", flush=True)
    w = weights()
    base = load_module(args.base, "base")
    base_state = base.prepare({kk: v.clone() for kk, v in w.items()})
    router = w["router"].float()
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    planes = {
        "router": torch.stack([hi, lo]).contiguous(),
        "qkv_gate": w["qkv_gate"].contiguous(),
        "o": w["o"].contiguous(),
        "pooler": w["pooler"].contiguous(),
        "unpooler": w["unpooler"].contiguous(),
    }
    flush = torch.empty(FLUSH_BYTES // 4, device="cuda", dtype=torch.float32).normal_()
    print(f"setup done at {time.time() - t_start:.1f}s (base={args.base})", flush=True)

    gemv_rows = []
    if not args.cluster_only:
        for layer, m in GEMV_CASES:
            if args.only and layer != args.only:
                continue
            data = reference.generate_input(m, layer, 1)
            n, k = data["config"]["n"], data["config"]["k"]
            wl = w[layer]
            x = data["x"].clone()
            sides = {}
            for t in gemv_candidates(layer, m, n, k):
                bad = []
                for seed, dist in ((3, "normal"), (4, "outliers")):
                    d = reference.generate_input(m, layer, seed, dist=dist)
                    out = run_gemv(ext, d["x"], wl, n, t)
                    torch.cuda.synchronize()
                    ok, msg = reference.check_implementation(d, out, w)
                    if not ok:
                        bad.append(f"{dist}: {msg}")
                if bad:
                    print(f"FAIL gemv {layer} m={m} {t}: {bad}", flush=True)
                    continue
                sides[t] = graph_of(lambda t=t: run_gemv(ext, x, wl, n, t))[0]
            sd = {"x": x, "config": data["config"]}
            sides["base"] = graph_of(lambda: base.kernel_fn(sd, base_state))[0]
            med = _screen_and_time(sides, ("base",), flush, args.reps, time_replay, top_k=5)
            nbytes = n * k * wl.element_size() + m * k * 2 + m * n * wl.element_size()
            sol = reference.speed_of_light_ns(
                m, layer, {"hbm_bytes_per_s": 4.8e12, "bf16_dense_flops_per_s": 989e12}
            ) / 1e3
            ranked = sorted((nm for nm in med if nm != "base"), key=med.get)
            print(f"\n== gemv {layer} m={m} n={n} k={k} sol {sol:.2f}us | base {med['base']:.2f}us", flush=True)
            for nm in ranked:
                print(f"   {nm}  {med[nm]:.2f}us  {nbytes / med[nm] / 1e6:.2f} TB/s", flush=True)
            if ranked:
                best = ranked[0]
                g, out = graph_of(lambda: run_gemv(ext, x, wl, n, best))
                g.replay()
                torch.cuda.synchronize()
                first = out.clone()
                diff = sum(int((g.replay(), not torch.equal(out, first))[1]) for _ in range(200))
                gemv_rows.append((layer, m, best, med[best], med["base"], nbytes, sol, diff))

    cluster_rows = []
    if not args.gemv_only:
        for layer, m in CLUSTER_CASES:
            if args.only and layer != args.only:
                continue
            data = reference.generate_input(m, layer, 1)
            n, k = data["config"]["n"], data["config"]["k"]
            wl = planes[layer]
            x = data["x"].clone()
            sides = {}
            for t in cluster_candidates(layer, m, n, k):
                try:
                    bad = []
                    for seed, dist in ((3, "normal"), (4, "outliers")):
                        d = reference.generate_input(m, layer, seed, dist=dist)
                        out = run_cluster(ext, d["x"], wl, n, t)
                        torch.cuda.synchronize()
                        ok, msg = reference.check_implementation(d, out, w)
                        if not ok:
                            bad.append(f"{dist}: {msg}")
                    if bad:
                        print(f"FAIL cluster {layer} m={m} {t}: {bad}", flush=True)
                        continue
                    sides[t] = graph_of(lambda t=t: run_cluster(ext, x, wl, n, t))[0]
                except Exception as exc:
                    print(f"ERR cluster {layer} m={m} {t}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
                    torch.cuda.synchronize()
            sd = {"x": x, "config": data["config"]}
            sides["base"] = graph_of(lambda: base.kernel_fn(sd, base_state))[0]
            if layer == "router":
                def cublas():
                    acc = torch.mm(x, base_state["router_hi_t"], out_dtype=torch.float32)
                    acc += torch.mm(x, base_state["router_lo_t"], out_dtype=torch.float32)
                    return acc
            else:
                wv = w[layer][:n]

                def cublas():
                    return x @ wv.T
            sides["cublas"] = graph_of(cublas)[0]

            med = _screen_and_time(sides, ("base", "cublas"), flush, args.reps, time_replay, top_k=5)
            wbytes = 4 if layer == "router" else 2
            nbytes = n * k * wbytes + m * k * 2 + m * n * wbytes
            sol = reference.speed_of_light_ns(
                m, layer, {"hbm_bytes_per_s": 4.8e12, "bf16_dense_flops_per_s": 989e12}
            ) / 1e3
            ranked = sorted((nm for nm in med if nm not in ("base", "cublas")), key=med.get)
            print(
                f"\n== cluster {layer} m={m} n={n} k={k} sol {sol:.2f}us | base {med['base']:.2f}us | "
                f"cublas {med['cublas']:.2f}us",
                flush=True,
            )
            for nm in ranked:
                print(f"   {nm}  {med[nm]:.2f}us  {nbytes / med[nm] / 1e6:.2f} TB/s", flush=True)
            if ranked:
                best = ranked[0]
                fails = []
                for mm in NEARBY[m]:
                    if 8 * best.nt < mm:
                        continue
                    for seed, dist in ((5, "normal"), (6, "outliers")):
                        d = reference.generate_input(mm, layer, seed, dist=dist)
                        nn = d["config"]["n"]
                        out = run_cluster(ext, d["x"], wl, nn, best)
                        torch.cuda.synchronize()
                        ok, msg = reference.check_implementation(d, out, w)
                        if not ok:
                            fails.append((mm, dist, msg))
                g, out = graph_of(lambda: run_cluster(ext, x, wl, n, best))
                g.replay()
                torch.cuda.synchronize()
                first = out.clone()
                diff = 0
                for _ in range(200):
                    g.replay()
                    diff += int(not torch.equal(out, first))
                torch.cuda.synchronize()
                print(f"   best {best}: nearby fails {fails or 'none'}; determinism {diff}/200 differ", flush=True)
                cluster_rows.append((layer, m, best, med[best], med["base"], med["cublas"], nbytes, sol, diff))

    if gemv_rows:
        print(f"\n{'kind':6} {'layer':9} {'m':>4} {'best':28} {'csrc':>8} {'base':>8} {'TB/s':>6} {'sol':>6} {'det':>5}")
        for layer, m, best, a, b, nbytes, sol, diff in gemv_rows:
            print(
                f"{'gemv':6} {layer:9} {m:4d} {str(tuple(best)):28} {a:8.2f} {b:8.2f} "
                f"{nbytes / a / 1e6:6.2f} {sol:6.2f} {diff:3d}/200"
            )
    if cluster_rows:
        print(f"\n{'kind':7} {'layer':9} {'m':>4} {'best':28} {'csrc':>8} {'base':>8} {'cublas':>8} {'TB/s':>6} {'sol':>6}")
        for layer, m, best, a, b, c, nbytes, sol, diff in cluster_rows:
            print(
                f"{'cluster':7} {layer:9} {m:4d} {str(tuple(best)):28} {a:8.2f} {b:8.2f} {c:8.2f} "
                f"{nbytes / a / 1e6:6.2f} {sol:6.2f}"
            )
    print(f"total {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
