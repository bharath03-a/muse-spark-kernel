"""Round 3 Custom CUDA C++ (sm_90a) Hopper TMA + wgmma + Cluster DSMEM & Streaming m=1 GEMV.

Fixes the 4 bottlenecks from Round 2 (`bench/exp_cuda_wgmma.py`):
1. Bank-conflict-free epilogue (`MPAD = 68` floats -> 1-way bank access across all 32 warp lanes + 16B aligned),
   compile-time `CSIZE` (zero runtime `/` or `%`), `float4` (`v4.f32`) DSMEM cluster reduction, and vectorized
   `uint2` (`__floats2bfloat162_rn`) / `float4` global stores.
2. Stacked `hi+lo` router kernel (`tma_wgmma_router_bn32<BT, BK, STAGES, CSIZE>`): packs `BN=32` rows of `W_hi`
   into `[0..31, 0..BK-1]` and `BN=32` rows of `W_lo` into `[32..63, 0..BK-1]`, executing a SINGLE
   `wgmma.m64n{BT}k16` per K-step, doubling N-tiles (`256/32 = 8`), halving W shared memory, and halving DSMEM traffic.
3. Expanded `(BT, BK, STAGES, CSIZE)` coverage for `qkv_gate`, `unpooler`, `pooler`, `o`, and `router` across
   `m = 8, 16, 32, 128, 256`.
4. Software-pipelined C++ `stream_gemv_m1<R, KWARPS, WARPS_CTA>` for `m=1` (`decode 1`) on `qkv_gate`, `pooler`, and `o`.

Usage:
    python bench/exp_cuda_wgmma_r3.py --compile
    python bench/exp_cuda_wgmma_r3.py [--reps 40] [--only router] [--gemv-only] [--wgmma-only]
"""

from __future__ import annotations

import argparse
import hashlib
import statistics
import sys
import time
from typing import NamedTuple

import torch

BASE_CP = "checkpoints/cp3_gluon_v2.py"

SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_bf16.h>

namespace cg = cooperative_groups;
using bf16 = __nv_bfloat16;
using bf162 = __nv_bfloat162;

constexpr int BN = 64;
constexpr int MPAD = 68;     // 68 floats (272B): 16B aligned for float4 AND 68 % 32 == 4 -> zero bank conflicts!
constexpr int THREADS = 128; // 1 warpgroup (4 warps)
constexpr int MAXC = 16;
constexpr int SMEM_LIMIT = 232448 - 1024;

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint64_t* b, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(b)), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
               ::"r"(smem_u32(b)), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred P1;\n LAB_WAIT:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      " @P1 bra DONE;\n bra LAB_WAIT;\n DONE:\n}\n"
      ::"r"(smem_u32(b)), "r"(parity) : "memory");
}

__device__ __forceinline__ void tma_load_2d(void* smem_dst, const CUtensorMap* map, int k_col, int row_idx,
                                            uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1, {%2, %3}], [%4];\n"
      ::"r"(smem_u32(smem_dst)), "l"(map), "r"(k_col), "r"(row_idx), "r"(smem_u32(bar))
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

__device__ __forceinline__ uint64_t make_smem_desc(const void* ptr) {
  const uint64_t addr = (smem_u32(ptr) >> 4) & 0x3FFFull;
  return addr | (1ull << 16) | (64ull << 32) | (1ull << 62);
}

__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
template <int N>
__device__ __forceinline__ void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

__device__ __forceinline__ void store_bf16x4(bf16* dst, const float4& v) {
  const bf162 b0 = __floats2bfloat162_rn(v.x, v.y);
  const bf162 b1 = __floats2bfloat162_rn(v.z, v.w);
  uint2 u;
  u.x = *reinterpret_cast<const uint32_t*>(&b0);
  u.y = *reinterpret_cast<const uint32_t*>(&b1);
  *reinterpret_cast<uint2*>(dst) = u;
}

template <int BT>
__device__ __forceinline__ void wgmma_m64n(float (&d)[BT / 8][4], uint64_t da, uint64_t db);

template <>
__device__ __forceinline__ void wgmma_m64n<8>(float (&d)[1][4], uint64_t da, uint64_t db) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3])
      : "l"(da), "l"(db));
}

template <>
__device__ __forceinline__ void wgmma_m64n<16>(float (&d)[2][4], uint64_t da, uint64_t db) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
        "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3])
      : "l"(da), "l"(db));
}

template <>
__device__ __forceinline__ void wgmma_m64n<32>(float (&d)[4][4], uint64_t da, uint64_t db) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1, 0, 0;\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
        "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
        "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3])
      : "l"(da), "l"(db));
}

template <>
__device__ __forceinline__ void wgmma_m64n<64>(float (&d)[8][4], uint64_t da, uint64_t db) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
      "%32, %33, 1, 1, 1, 0, 0;\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
        "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
        "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
        "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
        "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
        "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
        "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3])
      : "l"(da), "l"(db));
}

template <>
__device__ __forceinline__ void wgmma_m64n<128>(float (&d)[16][4], uint64_t da, uint64_t db) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
      "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
      "%64, %65, 1, 1, 1, 0, 0;\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
        "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
        "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
        "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
        "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
        "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
        "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
        "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
        "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]),
        "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]),
        "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]),
        "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]),
        "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3]),
        "+f"(d[14][0]), "+f"(d[14][1]), "+f"(d[14][2]), "+f"(d[14][3]),
        "+f"(d[15][0]), "+f"(d[15][1]), "+f"(d[15][2]), "+f"(d[15][3])
      : "l"(da), "l"(db));
}

// Upgraded BN=64 TMA + wgmma + Cluster DSMEM kernel with:
// - Compile-time CSIZE (zero runtime integer division)
// - Bank-conflict-free MPAD=68 shared memory staging
// - float4 (16B) DSMEM transfers & vectorized global stores
template <int BT, int BK, int STAGES, int P, int CSIZE>
__global__ void __launch_bounds__(THREADS, 1)
tma_wgmma_kernel(const __grid_constant__ CUtensorMap w0_map,
                 const __grid_constant__ CUtensorMap w1_map,
                 const __grid_constant__ CUtensorMap x_map,
                 void* __restrict__ out, int m, int n, int k) {
  constexpr int KBOX = BK / 64;
  constexpr int W_BOX_ELEMS = BN * 64;
  constexpr int X_BOX_ELEMS = BT * 64;
  constexpr int STAGE_ELEMS = (P * BN + BT) * BK;
  constexpr uint32_t STAGE_BYTES = STAGE_ELEMS * sizeof(bf16);
  constexpr int E = BN * BT;
  constexpr int E_V4 = E / 4;
  constexpr int SLICE_V4 = E_V4 / CSIZE;
  constexpr int NT = BT / 8;

  extern __shared__ __align__(1024) unsigned char smem[];
  bf16* stage_base = reinterpret_cast<bf16*>(smem);
  constexpr size_t TMA_Total_Bytes = (size_t)STAGES * STAGE_BYTES;
  uint64_t* ready = reinterpret_cast<uint64_t*>(smem + TMA_Total_Bytes);
  constexpr size_t RECV_OFFSET = TMA_Total_Bytes + ((STAGES * sizeof(uint64_t) + 15) & ~size_t(15));
  float4* recv4 = reinterpret_cast<float4*>(smem + RECV_OFFSET);

  cg::cluster_group cluster = cg::this_cluster();
  const int rank = CSIZE > 1 ? (int)cluster.block_rank() : 0;
  if constexpr (CSIZE > 1) cluster_arrive_relaxed();

  const int tid = threadIdx.x;
  const int n0 = (int)(blockIdx.x / CSIZE) * BN;
  const int t0 = (int)blockIdx.y * BT;
  const int k_steps = k / (CSIZE * BK);
  const int k_start = rank * k_steps;

  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) mbar_init(&ready[s], 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();

  auto w_box = [&](int s, int p, int kb) -> bf16* {
    return stage_base + (size_t)s * STAGE_ELEMS + (p * KBOX + kb) * W_BOX_ELEMS;
  };
  auto x_box = [&](int s, int kb) -> bf16* {
    return stage_base + (size_t)s * STAGE_ELEMS + (P * KBOX) * W_BOX_ELEMS + kb * X_BOX_ELEMS;
  };

  auto issue_tma = [&](int slot, int step) {
    mbar_expect_tx(&ready[slot], STAGE_BYTES);
    const int k_col = (k_start + step) * BK;
#pragma unroll
    for (int kb = 0; kb < KBOX; ++kb) {
      const int kc = k_col + kb * 64;
      tma_load_2d(w_box(slot, 0, kb), &w0_map, kc, n0, &ready[slot]);
      if constexpr (P == 2) {
        tma_load_2d(w_box(slot, 1, kb), &w1_map, kc, n0, &ready[slot]);
      }
      tma_load_2d(x_box(slot, kb), &x_map, kc, t0, &ready[slot]);
    }
  };

  if (tid == 0) {
    const int pre = STAGES - 1 < k_steps ? STAGES - 1 : k_steps;
    for (int s = 0; s < pre; ++s) issue_tma(s, s);
  }

  float acc[NT][4];
#pragma unroll
  for (int j = 0; j < NT; ++j)
#pragma unroll
    for (int r = 0; r < 4; ++r) acc[j][r] = 0.f;

  for (int i = 0; i < k_steps; ++i) {
    const int s = i % STAGES;
    mbar_wait(&ready[s], (i / STAGES) & 1);
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    wgmma_fence();
#pragma unroll
    for (int kb = 0; kb < KBOX; ++kb) {
      const uint64_t db_base = make_smem_desc(x_box(s, kb));
#pragma unroll
      for (int p = 0; p < P; ++p) {
        const uint64_t da_base = make_smem_desc(w_box(s, p, kb));
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
          wgmma_m64n<BT>(acc, da_base + (uint64_t)(ks * 2), db_base + (uint64_t)(ks * 2));
        }
      }
    }
    wgmma_commit();
    wgmma_wait<1>();
    const int j = i + STAGES - 1;
    if (j < k_steps && tid == 0) {
      issue_tma(j % STAGES, j);
    }
  }
  wgmma_wait<0>();
  __syncthreads();

  // Bank-conflict-free staging with row stride MPAD = 68 floats (16B aligned).
  float* mine = reinterpret_cast<float*>(smem);
  const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, q = lane & 3;
  const int r0 = warp * 16 + g;
#pragma unroll
  for (int j = 0; j < NT; ++j) {
    const int c = 8 * j + 2 * q;
    mine[c * MPAD + r0] = acc[j][0];
    mine[(c + 1) * MPAD + r0] = acc[j][1];
    mine[c * MPAD + r0 + 8] = acc[j][2];
    mine[(c + 1) * MPAD + r0 + 8] = acc[j][3];
  }
  __syncthreads();

  if constexpr (CSIZE == 1) {
    for (int tv = tid; tv < E_V4; tv += THREADS) {
      const int tt = tv >> 4;
      const int rr = (tv & 15) << 2;
      if (t0 + tt < m) {
        const float4 v = *reinterpret_cast<const float4*>(&mine[tt * MPAD + rr]);
        const long o = (long)(t0 + tt) * n + n0 + rr;
        if constexpr (P == 2) {
          *reinterpret_cast<float4*>(static_cast<float*>(out) + o) = v;
        } else {
          store_bf16x4(static_cast<bf16*>(out) + o, v);
        }
      }
    }
    return;
  }

  cluster_wait();
  for (int tv = tid; tv < E_V4; tv += THREADS) {
    const int tt = tv >> 4;
    const int rr = (tv & 15) << 2;
    const float4 v = *reinterpret_cast<const float4*>(&mine[tt * MPAD + rr]);
    const int owner = tv / SLICE_V4;
    const int off = tv % SLICE_V4;
    cluster.map_shared_rank(recv4, owner)[rank * SLICE_V4 + off] = v;
  }
  cluster_arrive_release();
  cluster_wait();
  for (int el = tid; el < SLICE_V4; el += THREADS) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int c = 0; c < CSIZE; ++c) {
      const float4 p = recv4[c * SLICE_V4 + el];
      v.x += p.x;
      v.y += p.y;
      v.z += p.z;
      v.w += p.w;
    }
    const int tv = rank * SLICE_V4 + el;
    const int tt = tv >> 4;
    const int rr = (tv & 15) << 2;
    if (t0 + tt < m) {
      const long o = (long)(t0 + tt) * n + n0 + rr;
      if constexpr (P == 2) {
        *reinterpret_cast<float4*>(static_cast<float*>(out) + o) = v;
      } else {
        store_bf16x4(static_cast<bf16*>(out) + o, v);
      }
    }
  }
}

// Stacked hi+lo Router Kernel (BN_OUT = 32 output rows per cluster):
// Rows [0..31] of W box hold W_hi[n0:n0+32], rows [32..63] hold W_lo[n0:n0+32].
// Executes ONE wgmma.m64n{BT}k16 per K-step, sums Warps 0..1 (hi) + Warps 2..3 (lo) in smem,
// and reduces [BT, 32] across CSIZE via float4 DSMEM!
template <int BT, int BK, int STAGES, int CSIZE>
__global__ void __launch_bounds__(THREADS, 1)
tma_wgmma_router_bn32(const __grid_constant__ CUtensorMap w0_map,
                      const __grid_constant__ CUtensorMap w1_map,
                      const __grid_constant__ CUtensorMap x_map,
                      float* __restrict__ out, int m, int n, int k) {
  constexpr int BN_OUT = 32;
  constexpr int KBOX = BK / 64;
  constexpr int HALF_W_ELEMS = BN_OUT * 64;  // 32 * 64 = 2048 bf16 (4096B, 1024B aligned!)
  constexpr int W_BOX_ELEMS = BN * 64;       // 64 * 64 = 4096 bf16 (8192B)
  constexpr int X_BOX_ELEMS = BT * 64;
  constexpr int STAGE_ELEMS = (BN + BT) * BK;
  constexpr uint32_t STAGE_BYTES = STAGE_ELEMS * sizeof(bf16);
  constexpr int E_OUT = BN_OUT * BT;
  constexpr int E_V4 = E_OUT / 4;
  constexpr int SLICE_V4 = E_V4 / CSIZE;
  constexpr int NT = BT / 8;

  extern __shared__ __align__(1024) unsigned char smem[];
  bf16* stage_base = reinterpret_cast<bf16*>(smem);
  constexpr size_t TMA_Total_Bytes = (size_t)STAGES * STAGE_BYTES;
  uint64_t* ready = reinterpret_cast<uint64_t*>(smem + TMA_Total_Bytes);
  constexpr size_t RECV_OFFSET = TMA_Total_Bytes + ((STAGES * sizeof(uint64_t) + 15) & ~size_t(15));
  float4* recv4 = reinterpret_cast<float4*>(smem + RECV_OFFSET);

  cg::cluster_group cluster = cg::this_cluster();
  const int rank = (int)cluster.block_rank();
  cluster_arrive_relaxed();

  const int tid = threadIdx.x;
  const int n0 = (int)(blockIdx.x / CSIZE) * BN_OUT;
  const int t0 = (int)blockIdx.y * BT;
  const int k_steps = k / (CSIZE * BK);
  const int k_start = rank * k_steps;

  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) mbar_init(&ready[s], 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();

  auto w_box = [&](int s, int kb) -> bf16* {
    return stage_base + (size_t)s * STAGE_ELEMS + kb * W_BOX_ELEMS;
  };
  auto x_box = [&](int s, int kb) -> bf16* {
    return stage_base + (size_t)s * STAGE_ELEMS + KBOX * W_BOX_ELEMS + kb * X_BOX_ELEMS;
  };

  auto issue_tma = [&](int slot, int step) {
    mbar_expect_tx(&ready[slot], STAGE_BYTES);
    const int k_col = (k_start + step) * BK;
#pragma unroll
    for (int kb = 0; kb < KBOX; ++kb) {
      const int kc = k_col + kb * 64;
      bf16* wb = w_box(slot, kb);
      tma_load_2d(wb, &w0_map, kc, n0, &ready[slot]);
      tma_load_2d(wb + HALF_W_ELEMS, &w1_map, kc, n0, &ready[slot]);
      tma_load_2d(x_box(slot, kb), &x_map, kc, t0, &ready[slot]);
    }
  };

  if (tid == 0) {
    const int pre = STAGES - 1 < k_steps ? STAGES - 1 : k_steps;
    for (int s = 0; s < pre; ++s) issue_tma(s, s);
  }

  float acc[NT][4];
#pragma unroll
  for (int j = 0; j < NT; ++j)
#pragma unroll
    for (int r = 0; r < 4; ++r) acc[j][r] = 0.f;

  for (int i = 0; i < k_steps; ++i) {
    const int s = i % STAGES;
    mbar_wait(&ready[s], (i / STAGES) & 1);
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    wgmma_fence();
#pragma unroll
    for (int kb = 0; kb < KBOX; ++kb) {
      const uint64_t da_base = make_smem_desc(w_box(s, kb));
      const uint64_t db_base = make_smem_desc(x_box(s, kb));
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        wgmma_m64n<BT>(acc, da_base + (uint64_t)(ks * 2), db_base + (uint64_t)(ks * 2));
      }
    }
    wgmma_commit();
    wgmma_wait<1>();
    const int j = i + STAGES - 1;
    if (j < k_steps && tid == 0) {
      issue_tma(j % STAGES, j);
    }
  }
  wgmma_wait<0>();
  __syncthreads();

  float* mine = reinterpret_cast<float*>(smem);
  const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, q = lane & 3;
  const int r0 = warp * 16 + g;
#pragma unroll
  for (int j = 0; j < NT; ++j) {
    const int c = 8 * j + 2 * q;
    mine[c * MPAD + r0] = acc[j][0];
    mine[(c + 1) * MPAD + r0] = acc[j][1];
    mine[c * MPAD + r0 + 8] = acc[j][2];
    mine[(c + 1) * MPAD + r0 + 8] = acc[j][3];
  }
  __syncthreads();

  cluster_wait();
  for (int tv = tid; tv < E_V4; tv += THREADS) {
    const int tt = tv >> 3;
    const int rr = (tv & 7) << 2;
    const float4 v_hi = *reinterpret_cast<const float4*>(&mine[tt * MPAD + rr]);
    const float4 v_lo = *reinterpret_cast<const float4*>(&mine[tt * MPAD + rr + 32]);
    const float4 v = make_float4(v_hi.x + v_lo.x, v_hi.y + v_lo.y, v_hi.z + v_lo.z, v_hi.w + v_lo.w);
    const int owner = tv / SLICE_V4;
    const int off = tv % SLICE_V4;
    cluster.map_shared_rank(recv4, owner)[rank * SLICE_V4 + off] = v;
  }
  cluster_arrive_release();
  cluster_wait();
  for (int el = tid; el < SLICE_V4; el += THREADS) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int c = 0; c < CSIZE; ++c) {
      const float4 p = recv4[c * SLICE_V4 + el];
      v.x += p.x;
      v.y += p.y;
      v.z += p.z;
      v.w += p.w;
    }
    const int tv = rank * SLICE_V4 + el;
    const int tt = tv >> 3;
    const int rr = (tv & 7) << 2;
    if (t0 + tt < m) {
      *reinterpret_cast<float4*>(out + (long)(t0 + tt) * n + n0 + rr) = v;
    }
  }
}

// Software-pipelined m=1 BF16 GEMV (`stream_gemv_m1<R, KWARPS, WARPS_CTA>`):
// Double-buffered `#pragma unroll 1` loop over K in steps of `KWARPS * 32 * 8` BF16 elements,
// streaming W with `ld.global.nc.L1::no_allocate.L2::256B.v4.s32` and reading x with `__ldg`.
__device__ __forceinline__ uint4 ld_stream_bf16(const bf16* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.s32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

__device__ __forceinline__ void acc_bf16x8(float& acc, const uint4& wv, const float (&xf)[8]) {
  const uint32_t u[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    acc = fmaf(__uint_as_float(u[i] << 16), xf[2 * i], acc);
    acc = fmaf(__uint_as_float(u[i] & 0xffff0000u), xf[2 * i + 1], acc);
  }
}

__device__ __forceinline__ void unpack_bf16x8(const uint4& v, float (&f)[8]) {
  const uint32_t u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    f[2 * i] = __uint_as_float(u[i] << 16);
    f[2 * i + 1] = __uint_as_float(u[i] & 0xffff0000u);
  }
}

template <int R, int KWARPS, int WARPS_CTA>
__global__ void __launch_bounds__(WARPS_CTA * 32)
stream_gemv_m1(const bf16* __restrict__ x, const bf16* __restrict__ w, bf16* __restrict__ out, int n, int k) {
  constexpr int NGROUPS = WARPS_CTA / KWARPS;
  constexpr int KTHREADS = KWARPS * 32;
  constexpr int STEP_K = KTHREADS * 8;
  __shared__ float part[NGROUPS][KWARPS][R];

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int gid = warp / KWARPS, kwid = warp % KWARPS, ktid = kwid * 32 + lane;
  const int r0 = (blockIdx.x * NGROUPS + gid) * R;
  const int nst = k / STEP_K;
  const bf16* wb = w + (long)r0 * k + ktid * 8;
  const bf16* xb = x + ktid * 8;

  float acc[R];
#pragma unroll
  for (int r = 0; r < R; ++r) acc[r] = 0.f;

  uint4 x0, x1;
  uint4 w0[R], w1[R];

  auto load_stage = [&](uint4& xs, uint4 (&ws)[R], int st) {
    const int ko = st * STEP_K;
    xs = __ldg(reinterpret_cast<const uint4*>(xb + ko));
#pragma unroll
    for (int r = 0; r < R; ++r) {
      ws[r] = ld_stream_bf16(wb + (long)r * k + ko);
    }
  };

  auto compute_stage = [&](const uint4& xs, const uint4 (&ws)[R]) {
    float xf[8];
    unpack_bf16x8(xs, xf);
#pragma unroll
    for (int r = 0; r < R; ++r) {
      acc_bf16x8(acc[r], ws[r], xf);
    }
  };

  if (nst > 0) load_stage(x0, w0, 0);
#pragma unroll 1
  for (int st = 0; st < nst; st += 2) {
    if (st + 1 < nst) load_stage(x1, w1, st + 1);
    compute_stage(x0, w0);
    if (st + 2 < nst) load_stage(x0, w0, st + 2);
    if (st + 1 < nst) compute_stage(x1, w1);
  }

#pragma unroll
  for (int r = 0; r < R; ++r) {
    float v = acc[r];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    if constexpr (KWARPS == 1) {
      if (lane == 0) out[r0 + r] = __float2bfloat16(v);
    } else {
      if (lane == 0) part[gid][kwid][r] = v;
    }
  }
  if constexpr (KWARPS > 1) {
    __syncthreads();
    if (ktid < R) {
      float v = 0.f;
#pragma unroll
      for (int ww = 0; ww < KWARPS; ++ww) v += part[gid][ww][ktid];
      out[r0 + ktid] = __float2bfloat16(v);
    }
  }
}

inline CUtensorMap make_tma_2d(const void* ptr, int rows, int cols, int row_stride, int box_rows) {
  CUtensorMap map{};
  cuuint64_t gdim[2] = {(cuuint64_t)cols, (cuuint64_t)rows};
  cuuint64_t gstride[1] = {(cuuint64_t)row_stride * sizeof(bf16)};
  cuuint32_t bdim[2] = {64u, (cuuint32_t)box_rows};
  cuuint32_t estride[2] = {1u, 1u};
  CUresult res = cuTensorMapEncodeTiled(
      &map,
      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      2,
      const_cast<void*>(ptr),
      gdim,
      gstride,
      bdim,
      estride,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(res == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed with error ", (int)res);
  return map;
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

template <int BT, int BK, int STAGES, int P, int CSIZE>
void launch_wgmma_64(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& out) {
  const int m = x.size(0), k = x.size(1), n = out.size(1);
  TORCH_CHECK(n % BN == 0 && k % (CSIZE * BK) == 0, "wgmma64: shape divisibility");
  const int tma_bytes = STAGES * (P * BN + BT) * BK * (int)sizeof(bf16);
  const int mbar_bytes = (STAGES * (int)sizeof(uint64_t) + 15) & ~15;
  const int recv_bytes = CSIZE > 1 ? (BN * BT * (int)sizeof(float)) : 0;
  const int smem = tma_bytes + mbar_bytes + recv_bytes;
  TORCH_CHECK(smem <= SMEM_LIMIT, "wgmma64: smem=", smem, " exceeds limit");

  const bf16* wp0 = static_cast<const bf16*>(w.data_ptr());
  const bf16* wp1 = P == 2 ? wp0 + w.stride(0) : wp0;
  const bf16* xp = static_cast<const bf16*>(x.data_ptr());

  CUtensorMap w0_map = make_tma_2d(wp0, n, k, k, BN);
  CUtensorMap w1_map = P == 2 ? make_tma_2d(wp1, n, k, k, BN) : w0_map;
  CUtensorMap x_map = make_tma_2d(xp, m, k, (int)x.stride(0), BT);

  const dim3 grid((n / BN) * CSIZE, (m + BT - 1) / BT, 1);
  auto kern = tma_wgmma_kernel<BT, BK, STAGES, P, CSIZE>;
  static const bool init = (set_attrs_once(kern, SMEM_LIMIT), true);
  (void)init;
  cudaLaunchAttribute attr;
  cudaLaunchConfig_t cfg = cluster_cfg(grid, THREADS, smem, CSIZE, &attr);
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, w0_map, w1_map, x_map, out.data_ptr(), m, n, k));
}

template <int BT, int BK, int STAGES, int CSIZE>
void launch_router_32(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& out) {
  constexpr int BN_OUT = 32;
  const int m = x.size(0), k = x.size(1), n = out.size(1);
  TORCH_CHECK(n % BN_OUT == 0 && k % (CSIZE * BK) == 0, "router32: shape divisibility");
  const int tma_bytes = STAGES * (BN + BT) * BK * (int)sizeof(bf16);
  const int mbar_bytes = (STAGES * (int)sizeof(uint64_t) + 15) & ~15;
  const int recv_bytes = BN_OUT * BT * (int)sizeof(float);
  const int smem = tma_bytes + mbar_bytes + recv_bytes;
  TORCH_CHECK(smem <= SMEM_LIMIT, "router32: smem=", smem, " exceeds limit");

  const bf16* wp0 = static_cast<const bf16*>(w.data_ptr());
  const bf16* wp1 = wp0 + w.stride(0);
  const bf16* xp = static_cast<const bf16*>(x.data_ptr());

  CUtensorMap w0_map = make_tma_2d(wp0, n, k, k, BN_OUT);
  CUtensorMap w1_map = make_tma_2d(wp1, n, k, k, BN_OUT);
  CUtensorMap x_map = make_tma_2d(xp, m, k, (int)x.stride(0), BT);

  const dim3 grid((n / BN_OUT) * CSIZE, (m + BT - 1) / BT, 1);
  auto kern = tma_wgmma_router_bn32<BT, BK, STAGES, CSIZE>;
  static const bool init = (set_attrs_once(kern, SMEM_LIMIT), true);
  (void)init;
  cudaLaunchAttribute attr;
  cudaLaunchConfig_t cfg = cluster_cfg(grid, THREADS, smem, CSIZE, &attr);
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, w0_map, w1_map, x_map, static_cast<float*>(out.data_ptr()), m, n, k));
}

void launch_wgmma(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t bn, int64_t bt, int64_t bk,
                  int64_t stages, int64_t csize) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && out.is_cuda(), "wgmma: CUDA tensors expected");
  const int p = w.dim() == 3 ? 2 : 1;
  if (bn == 32 && p == 2) {
#define R32(BT_, BK_, ST_, CS_)                                             \
  if (bt == BT_ && bk == BK_ && stages == ST_ && csize == CS_) {            \
    launch_router_32<BT_, BK_, ST_, CS_>(x, w, out);                        \
    return;                                                                 \
  }
    ROUTER32_CONFIGS
#undef R32
  }
  if (bn == 64) {
#define W64(BT_, BK_, ST_, P_, CS_)                                         \
  if (bt == BT_ && bk == BK_ && stages == ST_ && p == P_ && csize == CS_) { \
    launch_wgmma_64<BT_, BK_, ST_, P_, CS_>(x, w, out);                     \
    return;                                                                 \
  }
    WGMMA64_CONFIGS
#undef W64
  }
  TORCH_CHECK(false, "wgmma: config (bn=", bn, ", bt=", bt, ", bk=", bk, ", st=", stages, ", p=", p, ", c=", csize, ") not compiled");
}

void launch_gemv_m1(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t rows, int64_t kwarps,
                    int64_t warps_cta) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && out.is_cuda(), "gemv_m1: CUDA tensors expected");
  const int n = out.size(1), k = x.size(1);
  const int r_cta = (int)(warps_cta / kwarps) * (int)rows;
  TORCH_CHECK(x.size(0) == 1 && n % r_cta == 0 && k % (kwarps * 32 * 8) == 0, "gemv_m1: shape");
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const bf16* xp = static_cast<const bf16*>(x.data_ptr());
  const bf16* wp = static_cast<const bf16*>(w.data_ptr());
  bf16* op = static_cast<bf16*>(out.data_ptr());
#define GM1(R_, KW_, WC_)                                                               \
  if (rows == R_ && kwarps == KW_ && warps_cta == WC_) {                                \
    stream_gemv_m1<R_, KW_, WC_><<<n / r_cta, WC_ * 32, 0, stream>>>(xp, wp, op, n, k); \
    C10_CUDA_CHECK(cudaGetLastError());                                                 \
    return;                                                                             \
  }
  GEMV_M1_CONFIGS
#undef GM1
  TORCH_CHECK(false, "gemv_m1: config not compiled");
}
"""

CPP = ("void launch_wgmma(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t bn, int64_t bt,"
       " int64_t bk, int64_t stages, int64_t csize);\n"
       "void launch_gemv_m1(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t rows, int64_t kwarps,"
       " int64_t warps_cta);")
CFLAGS = ["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v"]
LDFLAGS = ["-lcuda"]

# (BT, BK, STAGES, P, CSIZE) for BN=64
WGMMA64_ALL = (
    # Small/Medium m (m = 8, 16, 32), P = 1:
    (8, 64, 8, 1, 1), (8, 64, 8, 1, 2), (8, 64, 8, 1, 4),
    (8, 128, 4, 1, 1), (8, 128, 4, 1, 2), (8, 128, 4, 1, 4),
    (8, 128, 8, 1, 1), (8, 128, 8, 1, 2), (8, 128, 8, 1, 4),
    (8, 256, 4, 1, 1), (8, 256, 4, 1, 2),
    (16, 64, 8, 1, 1), (16, 64, 8, 1, 2), (16, 64, 8, 1, 4),
    (16, 128, 4, 1, 1), (16, 128, 4, 1, 2), (16, 128, 4, 1, 4),
    (16, 128, 8, 1, 1), (16, 128, 8, 1, 2), (16, 128, 8, 1, 4),
    (16, 256, 4, 1, 1), (16, 256, 4, 1, 2),
    (32, 64, 8, 1, 1), (32, 64, 8, 1, 2), (32, 64, 8, 1, 4),
    (32, 128, 4, 1, 1), (32, 128, 4, 1, 2), (32, 128, 4, 1, 4),
    (32, 128, 8, 1, 1), (32, 128, 8, 1, 2), (32, 128, 8, 1, 4),
    (32, 256, 4, 1, 1), (32, 256, 4, 1, 2),
    # Large m (m = 128, 256), P = 1:
    (64, 64, 8, 1, 1), (64, 64, 8, 1, 2), (64, 64, 8, 1, 4),
    (64, 128, 4, 1, 1), (64, 128, 4, 1, 2), (64, 128, 4, 1, 4),
    (64, 128, 6, 1, 1), (64, 128, 6, 1, 2), (64, 128, 6, 1, 4),
    (128, 64, 6, 1, 1), (128, 64, 6, 1, 2),
    (128, 128, 4, 1, 1), (128, 128, 4, 1, 2),
    # Router BN=64 (P = 2):
    (32, 128, 4, 2, 8),
    (64, 128, 4, 2, 4), (64, 128, 4, 2, 8),
    (128, 64, 4, 2, 4), (128, 64, 4, 2, 8),
)

# (BT, BK, STAGES, CSIZE) for Stacked hi+lo Router BN_OUT=32:
ROUTER32_ALL = (
    (8, 64, 8, 8), (8, 128, 4, 8), (8, 128, 8, 8), (8, 64, 4, 16), (8, 128, 4, 16),
    (16, 64, 8, 8), (16, 128, 4, 8), (16, 128, 8, 8), (16, 64, 4, 16), (16, 128, 4, 16),
    (32, 64, 8, 8), (32, 128, 4, 4), (32, 128, 4, 8), (32, 128, 8, 8), (32, 64, 4, 16), (32, 128, 4, 16),
    (64, 64, 8, 4), (64, 64, 8, 8), (64, 128, 4, 4), (64, 128, 4, 8), (64, 128, 6, 4), (64, 128, 6, 8),
    (128, 64, 6, 4), (128, 64, 6, 8), (128, 128, 4, 4), (128, 128, 4, 8),
)

# (R, KWARPS, WARPS_CTA) for m=1 streaming GEMV:
GEMV_M1_ALL = (
    # 4 warps/CTA (128 threads, STEP_K = 1024 for KWARPS=4):
    (2, 4, 4), (4, 4, 4), (6, 4, 4), (8, 4, 4),
    # 8 warps/CTA (256 threads, 2 groups of 4 warps -> STEP_K = 1024):
    (2, 4, 8), (3, 4, 8), (4, 4, 8),
    # 8 warps/CTA (256 threads, STEP_K = 2048 for KWARPS=8):
    (4, 8, 8), (6, 8, 8), (8, 8, 8), (9, 8, 8),
    # Warp- specialized KWARPS=1 (STEP_K = 256) and KWARPS=2 (STEP_K = 512) for o (K=1024):
    (1, 1, 4), (2, 1, 4), (1, 1, 8), (2, 1, 8),
    (2, 2, 4), (4, 2, 4), (2, 2, 8), (4, 2, 8),
)

SMEM_LIMIT = 232448 - 1024
OVERHEAD_US = 4.0

GEMV_CASES = [
    ("qkv_gate", 1),
    ("pooler", 1),
    ("o", 1),
]

WGMMA_CASES = [
    ("router", 8),
    ("router", 16),
    ("router", 32),
    ("router", 128),
    ("router", 256),
    ("unpooler", 8),
    ("unpooler", 16),
    ("unpooler", 32),
    ("qkv_gate", 8),
    ("qkv_gate", 16),
    ("qkv_gate", 32),
    ("qkv_gate", 128),
    ("qkv_gate", 256),
    ("pooler", 8),
    ("pooler", 16),
    ("pooler", 32),
    ("pooler", 128),
    ("o", 8),
    ("o", 16),
    ("o", 32),
]

NEARBY = {
    1: (1,),
    8: (5, 7, 8),
    16: (9, 12, 16),
    32: (17, 24, 31, 32),
    128: (100, 127, 128),
    256: (200, 255, 256),
}


class WTile(NamedTuple):
    bn: int
    bt: int
    bk: int
    stages: int
    csize: int


class GM1Tile(NamedTuple):
    rows: int
    kwarps: int
    warps_cta: int


def smem_bytes(t: WTile, p: int) -> int:
    eff_p = 1 if (t.bn == 32 and p == 2) else p
    tma = t.stages * (eff_p * 64 + t.bt) * t.bk * 2
    mbar = (t.stages * 8 + 15) & ~15
    recv = (t.bn * t.bt * 4) if t.csize > 1 else 0
    return tma + mbar + recv


def valid(t: WTile, p: int, n: int, k: int) -> bool:
    if n % t.bn != 0 or k % (t.csize * t.bk) != 0:
        return False
    if (k // (t.csize * t.bk)) < t.stages:
        return False
    return smem_bytes(t, p) <= SMEM_LIMIT


def wgmma_candidates(layer: str, m: int, n: int, k: int):
    p = 2 if layer == "router" else 1
    if layer == "router":
        for bt, bk, st, c in ROUTER32_ALL:
            if (m <= 32 and bt == m) or (m > 32 and bt in (64, 128)):
                t = WTile(32, bt, bk, st, c)
                if valid(t, p, n, k):
                    yield t
    for bt, bk, st, pp, c in WGMMA64_ALL:
        if pp != p:
            continue
        if m <= 32 and bt != m:
            continue
        if m > 32 and bt not in (64, 128):
            continue
        if layer == "o" and c > 1:
            continue
        if layer == "pooler" and c > 2:
            continue
        t = WTile(64, bt, bk, st, c)
        if not valid(t, p, n, k):
            continue
        ctas = (n // 64) * c * -(-m // bt)
        if 32 <= ctas <= 576:
            yield t


def gemv_m1_candidates(layer: str, n: int, k: int):
    for r, kw, wc in GEMV_M1_ALL:
        r_cta = (wc // kw) * r
        step_k = kw * 32 * 8
        if n % r_cta == 0 and k % step_k == 0 and (k // step_k) >= 2:
            ctas = n // r_cta
            if 128 <= ctas <= 2048:
                yield GM1Tile(r, kw, wc)


def source() -> str:
    return (
        "#define WGMMA64_CONFIGS "
        + " ".join(f"W64({bt},{bk},{st},{p},{c})" for bt, bk, st, p, c in WGMMA64_ALL)
        + "\n#define ROUTER32_CONFIGS "
        + " ".join(f"R32({bt},{bk},{st},{c})" for bt, bk, st, c in ROUTER32_ALL)
        + "\n#define GEMV_M1_CONFIGS "
        + " ".join(f"GM1({r},{kw},{wc})" for r, kw, wc in GEMV_M1_ALL)
        + "\n"
        + SRC
    )


def build(verbose: bool = False):
    from torch.utils.cpp_extension import load_inline

    src = source()
    tag = hashlib.sha1((src + " ".join(CFLAGS + LDFLAGS)).encode()).hexdigest()[:10]
    return load_inline(
        name=f"exp_cuda_wgmma_r3_{tag}",
        cpp_sources=CPP,
        cuda_sources=src,
        functions=["launch_wgmma", "launch_gemv_m1"],
        extra_cuda_cflags=CFLAGS,
        extra_cflags=["-O3"],
        extra_ldflags=LDFLAGS,
        verbose=verbose,
    )


def run_wgmma(ext, x: torch.Tensor, w: torch.Tensor, n: int, t: WTile) -> torch.Tensor:
    dtype = torch.float32 if w.dim() == 3 else torch.bfloat16
    out = torch.empty(x.shape[0], n, device=x.device, dtype=dtype)
    ext.launch_wgmma(x, w, out, t.bn, t.bt, t.bk, t.stages, t.csize)
    return out


def run_gemv_m1(ext, x: torch.Tensor, w: torch.Tensor, n: int, t: GM1Tile) -> torch.Tensor:
    out = torch.empty(x.shape[0], n, device=x.device, dtype=torch.bfloat16)
    ext.launch_gemv_m1(x, w, out, t.rows, t.kwarps, t.warps_cta)
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


def _screen_and_time(sides: dict, extra_keys: tuple[str, ...], flush, reps: int, time_replay, top_k: int = 4) -> dict:
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
            f"built {len(WGMMA64_ALL)} wgmma64 + {len(ROUTER32_ALL)} router32 + {len(GEMV_M1_ALL)} gemv_m1 "
            f"configs in {time.time() - t0:.1f}s",
            flush=True,
        )
        return

    from common import load_module, reference, weights
    from harness import FLUSH_BYTES, time_replay

    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--only", default="")
    ap.add_argument("--base", default=BASE_CP)
    ap.add_argument("--gemv-only", action="store_true")
    ap.add_argument("--wgmma-only", action="store_true")
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
    if not args.wgmma_only:
        for layer, m in GEMV_CASES:
            if args.only and layer != args.only:
                continue
            data = reference.generate_input(m, layer, 1)
            n, k = data["config"]["n"], data["config"]["k"]
            wl = w[layer][:n].contiguous()
            x = data["x"].clone()
            sides = {}
            for t in gemv_m1_candidates(layer, n, k):
                bad = []
                for seed, dist in ((3, "normal"), (4, "outliers")):
                    d = reference.generate_input(m, layer, seed, dist=dist)
                    out = run_gemv_m1(ext, d["x"], wl, n, t)
                    torch.cuda.synchronize()
                    ok, msg = reference.check_implementation(d, out, w)
                    if not ok:
                        bad.append(f"{dist}: {msg}")
                if bad:
                    print(f"FAIL gemv_m1 {layer} m={m} {t}: {bad}", flush=True)
                    continue
                sides[t] = graph_of(lambda t=t: run_gemv_m1(ext, x, wl, n, t))[0]
            sd = {"x": x, "config": data["config"]}
            sides["base"] = graph_of(lambda: base.kernel_fn(sd, base_state))[0]
            med = _screen_and_time(sides, ("base",), flush, args.reps, time_replay, top_k=4)
            nbytes = n * k * 2 + m * k * 2 + m * n * 2
            sol = reference.speed_of_light_ns(
                m, layer, {"hbm_bytes_per_s": 4.8e12, "bf16_dense_flops_per_s": 989e12}
            ) / 1e3
            ranked = sorted((nm for nm in med if nm != "base"), key=med.get)
            print(f"\n== gemv_m1 {layer} m={m} n={n} k={k} sol {sol:.2f}us | base {med['base']:.2f}us", flush=True)
            for nm in ranked:
                print(f"   {nm}  {med[nm]:.2f}us  {nbytes / med[nm] / 1e6:.2f} TB/s", flush=True)
            if ranked:
                best = ranked[0]
                g, out = graph_of(lambda: run_gemv_m1(ext, x, wl, n, best))
                g.replay()
                torch.cuda.synchronize()
                first = out.clone()
                diff = sum(int((g.replay(), not torch.equal(out, first))[1]) for _ in range(200))
                gemv_rows.append((layer, m, best, med[best], med["base"], nbytes, sol, diff))

    wgmma_rows = []
    if not args.gemv_only:
        for layer, m in WGMMA_CASES:
            if args.only and layer != args.only:
                continue
            data = reference.generate_input(m, layer, 1)
            n, k = data["config"]["n"], data["config"]["k"]
            wl = planes[layer] if layer == "router" else w[layer][:n].contiguous()
            x = data["x"].clone()
            sides = {}
            for t in wgmma_candidates(layer, m, n, k):
                try:
                    bad = []
                    for seed, dist in ((3, "normal"), (4, "outliers")):
                        d = reference.generate_input(m, layer, seed, dist=dist)
                        out = run_wgmma(ext, d["x"], wl, n, t)
                        torch.cuda.synchronize()
                        ok, msg = reference.check_implementation(d, out, w)
                        if not ok:
                            bad.append(f"{dist}: {msg}")
                    if bad:
                        print(f"FAIL wgmma {layer} m={m} {t}: {bad}", flush=True)
                        continue
                    sides[t] = graph_of(lambda t=t: run_wgmma(ext, x, wl, n, t))[0]
                except Exception as exc:
                    print(f"ERR wgmma {layer} m={m} {t}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
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

            med = _screen_and_time(sides, ("base", "cublas"), flush, args.reps, time_replay, top_k=4)
            wbytes = 4 if layer == "router" else 2
            nbytes = n * k * wbytes + m * k * 2 + m * n * wbytes
            sol = reference.speed_of_light_ns(
                m, layer, {"hbm_bytes_per_s": 4.8e12, "bf16_dense_flops_per_s": 989e12}
            ) / 1e3
            ranked = sorted((nm for nm in med if nm not in ("base", "cublas")), key=med.get)
            print(
                f"\n== wgmma {layer} m={m} n={n} k={k} sol {sol:.2f}us | base {med['base']:.2f}us | "
                f"cublas {med['cublas']:.2f}us",
                flush=True,
            )
            for nm in ranked:
                print(f"   {nm}  {med[nm]:.2f}us  {nbytes / med[nm] / 1e6:.2f} TB/s", flush=True)
            if ranked:
                best = ranked[0]
                fails = []
                for mm in NEARBY.get(m, (m,)):
                    for seed, dist in ((5, "normal"), (6, "outliers")):
                        d = reference.generate_input(mm, layer, seed, dist=dist)
                        nn = d["config"]["n"]
                        w_mm = planes[layer] if layer == "router" else w[layer][:nn].contiguous()
                        out = run_wgmma(ext, d["x"], w_mm, nn, best)
                        torch.cuda.synchronize()
                        ok, msg = reference.check_implementation(d, out, w)
                        if not ok:
                            fails.append((mm, dist, msg))
                g, out = graph_of(lambda: run_wgmma(ext, x, wl, n, best))
                g.replay()
                torch.cuda.synchronize()
                first = out.clone()
                diff = 0
                for _ in range(200):
                    g.replay()
                    diff += int(not torch.equal(out, first))
                torch.cuda.synchronize()
                print(f"   best {best}: nearby fails {fails or 'none'}; determinism {diff}/200 differ", flush=True)
                wgmma_rows.append((layer, m, best, med[best], med["base"], med["cublas"], nbytes, sol, diff))

    if gemv_rows:
        print(f"\n{'kind':8} {'layer':9} {'m':>4} {'best (r,kw,wc)':22} {'csrc':>8} {'base':>8} {'TB/s':>6} {'sol':>6} {'det':>5}")
        for layer, m, best, a, b, nbytes, sol, diff in gemv_rows:
            print(
                f"{'gemv_m1':8} {layer:9} {m:4d} {str(tuple(best)):22} {a:8.2f} {b:8.2f} "
                f"{nbytes / a / 1e6:6.2f} {sol:6.2f} {diff:3d}/200"
            )
    if wgmma_rows:
        print(
            f"\n{'layer':9} {'m':>4} {'best (bn,bt,bk,st,csize)':28} {'wgmma':>8} {'base':>8} "
            f"{'cublas':>8} {'TB/s':>6} {'sol':>6} {'det':>5}"
        )
        for layer, m, best, a, b, c, nbytes, sol, diff in wgmma_rows:
            print(
                f"{layer:9} {m:4d} {str(tuple(best)):28} {a:8.2f} {b:8.2f} {c:8.2f} "
                f"{nbytes / a / 1e6:6.2f} {sol:6.2f} {diff:3d}/200"
            )
    print(f"total {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
