"""Cluster split-K skinny GEMM (CUDA, load_inline): the split-K CTAs of one output tile form a
thread-block cluster and reduce their partials through distributed shared memory.

    python bench/cluster_dev.py --compile     # build (CPU container is fine) and print ptxas usage

Swap-AB: weight rows on the mma M side, tokens on N. Each thread's A and B fragments are 16-byte
pieces of its rows / token with the same k permutation inside each 32-wide k chunk.
kind 0 ("reg"): fragments straight from global into registers, two register stages in flight.
kind 1 ("bulk"): a producer warp streams each row's k slice with 1D cp.async.bulk into an mbarrier
ring in smem; 8 consumer warps read fragments from padded smem rows.
Both end in the same push-model cluster reduce (one release/acquire cluster barrier, no workspace).
"""

from __future__ import annotations

import hashlib
import sys
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
constexpr int MAXS = 8;
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

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint64_t* b, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(b)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(b)), "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred P1;\n LAB_WAIT:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      " @P1 bra DONE;\n bra LAB_WAIT;\n DONE:\n}\n" ::"r"(smem_u32(b)),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
          smem_u32(dst)),
      "l"(src), "r"(bytes), "r"(smem_u32(bar))
      : "memory");
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

// Stage layout: P * BN weight rows then BT token rows, each ks elements + 32 of padding, so the
// 8 rows one LDS.128 phase touches land in different bank halves.
template <int MT, int NT, int P>
__global__ void __launch_bounds__(THREADS + 32, 1)
bulk_gemm(const bf16* __restrict__ x, const bf16* __restrict__ w, void* __restrict__ out, int m, int n,
          int k, int ldx, long wplane, int wk, int ks, int nstage, int mode) {
  using C_ = Cfg<MT, NT, P>;
  constexpr int BNW = C_::BN, BT = C_::BT;
  extern __shared__ __align__(128) unsigned char smem[];
  __shared__ uint64_t full[MAXS], empty[MAXS];
  cluster_arrive_relaxed();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
  const Tile T = make_tile(BNW, BT, wk, warp < WARPS ? warp : 0);
  const int BN = T.BN;
  const int pitch = ks + 32;
  const int stage_elems = (P * BN + BT) * pitch;
  const bf16* stg = reinterpret_cast<const bf16*>(smem);
  float* red = reinterpret_cast<float*>(smem + (size_t)nstage * stage_elems * sizeof(bf16));
  const int kc = k / T.csize;
  const int kbase = T.rank * kc;
  const int nst = mode == 2 ? 0 : kc / ks;
  const int tv = min(BT, m - T.t0);
  if (threadIdx.x == 0) {
    for (int s = 0; s < nstage; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&empty[s], WARPS);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  float acc[MT][NT][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.f;

  if (warp == WARPS) {
    const int rows = P * BN + tv;
    const uint32_t bytes = (uint32_t)rows * ks * sizeof(bf16);
    for (int s = 0; s < nst; ++s) {
      const int slot = s % nstage;
      if (s >= nstage) mbar_wait(&empty[slot], ((s / nstage) & 1) ^ 1);
      if (lane == 0) mbar_expect_tx(&full[slot], bytes);
      __syncwarp();
      bf16* base = const_cast<bf16*>(stg) + slot * stage_elems;
      const int kk = kbase + s * ks;
      for (int r = lane; r < rows; r += 32) {
        const bf16* src;
        if (r < P * BN) {
          const int p = r / BN, rr = r - p * BN;
          src = w + p * wplane + (long)(T.n0 + rr) * k + kk;
        } else {
          src = x + (long)(T.t0 + r - P * BN) * ldx + kk;
        }
        bulk_g2s(base + r * pitch, src, ks * sizeof(bf16), &full[slot]);
      }
    }
  } else {
    const int cpw = ks / (32 * wk);
    const int rbase = T.wr * BNW + g;
    for (int s = 0; s < nst; ++s) {
      const int slot = s % nstage;
      mbar_wait(&full[slot], (s / nstage) & 1);
      const bf16* base = stg + slot * stage_elems;
#pragma unroll 2
      for (int c = 0; c < cpw; ++c) {
        const int ko = (T.wkk * cpw + c) * 32 + 8 * q;
        uint4 a[P][MT][2], b[NT];
#pragma unroll
        for (int p = 0; p < P; ++p)
#pragma unroll
          for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
              a[p][i][h] = *reinterpret_cast<const uint4*>(base + (p * BN + rbase + 16 * i + 8 * h) * pitch + ko);
#pragma unroll
        for (int j = 0; j < NT; ++j)
          b[j] = *reinterpret_cast<const uint4*>(base + (P * BN + 8 * j + g) * pitch + ko);
        mma_chunk<MT, NT, P>(acc, a, b);
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(&empty[slot]);
    }
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
  cluster_reduce<MT, NT, P>(acc, warp < WARPS, red, T, m, n, out);
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
    TORCH_CHECK(nstage >= 1 && nstage <= MAXS, "cluster_gemm: stages");
    TORCH_CHECK(ks % (32 * wk) == 0 && (k / csize) % ks == 0 && k % csize == 0, "cluster_gemm: ks=", ks);
    const int smem = nstage * (P * bn + C_::BT) * (ks + 32) * (int)sizeof(bf16) + red_bytes;
    TORCH_CHECK(smem <= SMEM_LIMIT, "cluster_gemm: smem ", smem, " too large");
    auto kern = bulk_gemm<MT, NT, P>;
    static const bool init = (set_attrs_once(kern, SMEM_LIMIT), true);
    (void)init;
    cudaLaunchConfig_t cfg = cluster_cfg(grid, THREADS + 32, smem, csize, &attr);
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, xp, wp, out.data_ptr(), m, n, k, (int)x.stride(0), wplane,
                                      wk, ks, nstage, mode));
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

CPP = ("void launch(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t mt, int64_t nt, int64_t csize,"
       " int64_t wk, int64_t kind, int64_t ks, int64_t nstage, int64_t mode);\n"
       "void launch_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t rows, int64_t mt,"
       " int64_t per);")
CFLAGS = ["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v"]

# (MT, NT, P): BN = 16 MT weight rows per row group, BT = 8 NT tokens, P = 2 for the router's hi | lo.
CONFIGS_ALL = (
    (1, 1, 2), (1, 2, 2), (1, 4, 2), (2, 1, 2), (2, 2, 2), (2, 4, 2), (1, 8, 2), (2, 8, 2),
    (1, 1, 1), (2, 1, 1), (4, 1, 1), (2, 2, 1), (4, 2, 1), (2, 4, 1), (4, 4, 1),
)
# row_gemv (f32 weights, rows per CTA, max tokens, 16-byte loads per thread per row).
GEMV_ALL = (
    (True, 1, 1, 8), (True, 1, 8, 8), (True, 2, 8, 8),
    (False, 2, 1, 2), (False, 4, 1, 2), (False, 2, 2, 2), (False, 4, 2, 2), (False, 2, 4, 2), (False, 4, 4, 2),
    (False, 2, 8, 2), (False, 4, 8, 2), (False, 8, 8, 2), (False, 2, 1, 4), (False, 4, 1, 4),
)
WARPS = 8
SMEM_LIMIT = 232448 - 1024


class CTile(NamedTuple):
    mt: int
    nt: int
    csize: int
    wk: int = WARPS  # warps along k; WARPS // wk row groups of 16 mt rows
    kind: int = 0    # 0 reg, 1 bulk
    ks: int = 0      # bulk: k elements per stage
    stages: int = 0  # bulk: ring depth


def stage_chunks(mt: int, nt: int, p: int) -> int:
    return 2 if mt * 2 * p + nt <= 8 else 1


def block_n(t: CTile) -> int:
    return 16 * t.mt * (WARPS // t.wk)


def smem_bytes(t: CTile, p: int) -> int:
    bn, bt = block_n(t), 8 * t.nt
    red = (WARPS * 16 * t.mt + bn) * bt * 4
    return red if t.kind == 0 else t.stages * (p * bn + bt) * (t.ks + 32) * 2 + red


def valid(t: CTile, p: int, n: int, k: int) -> bool:
    if n % block_n(t):
        return False
    if t.kind == 0:
        return k % (t.csize * t.wk * 32 * stage_chunks(t.mt, t.nt, p)) == 0
    return (t.ks % (32 * t.wk) == 0 and k % t.csize == 0 and (k // t.csize) % t.ks == 0
            and 1 <= t.stages <= 8 and smem_bytes(t, p) <= SMEM_LIMIT)


class GTile(NamedTuple):
    rows: int
    mt: int
    per: int


def source(configs, gemv) -> str:
    return ("#define CONFIGS " + " ".join(f"X({a},{b},{c})" for a, b, c in configs) + "\n"
            + "#define GEMV_CONFIGS " + " ".join(f"G({str(f).lower()},{r},{t},{p})" for f, r, t, p in gemv)
            + "\n" + SRC)


def build(configs=CONFIGS_ALL, gemv=GEMV_ALL, verbose: bool = False):
    from torch.utils.cpp_extension import load_inline

    src = source(configs, gemv)
    tag = hashlib.sha1((src + " ".join(CFLAGS)).encode()).hexdigest()[:10]
    return load_inline(name=f"cluster_gemm_dev_{tag}", cpp_sources=CPP, cuda_sources=src,
                       functions=["launch", "launch_gemv"], extra_cuda_cflags=CFLAGS, extra_cflags=["-O3"], verbose=verbose)


def run(ext, x: torch.Tensor, w: torch.Tensor, n: int, t: CTile, mode: int = 0) -> torch.Tensor:
    dtype = torch.float32 if w.dim() == 3 else torch.bfloat16
    out = torch.empty(x.shape[0], n, device=x.device, dtype=dtype)
    ext.launch(x, w, out, t.mt, t.nt, t.csize, t.wk, t.kind, t.ks, t.stages, mode)
    return out


def run_gemv(ext, x: torch.Tensor, w: torch.Tensor, n: int, t: GTile) -> torch.Tensor:
    out = torch.empty(x.shape[0], n, device=x.device, dtype=w.dtype)
    ext.launch_gemv(x, w, out, t.rows, t.mt, t.per)
    return out


if __name__ == "__main__":
    if "--compile" in sys.argv:
        import time

        t0 = time.time()
        if "--sub" in sys.argv:
            from common import load_module

            load_module("submission.py", "sub")._build_cuda()
            print(f"built submission extension in {time.time() - t0:.1f}s")
        else:
            build(verbose=True)
            print(f"built {len(CONFIGS_ALL)} configs in {time.time() - t0:.1f}s")
