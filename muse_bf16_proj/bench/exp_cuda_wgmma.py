"""Custom CUDA C++ (sm_90a) Hopper TMA + wgmma.mma_async + Cluster DSMEM split-K kernel.

Combines:
1. 2D TMA (`cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes`) with 128B swizzle
2. Asynchronous warpgroup Tensor Core MMA (`wgmma.mma_async.sync.aligned.m64n{BT}k16.f32.bf16.bf16`)
3. Hopper Thread-Block Cluster Distributed Shared Memory (DSMEM) split-K reduction (`csize in {1, 2, 4, 8}`)
   with zero global workspace, zero global atomics, and deterministic rank-order accumulation.

Usage:
    python bench/exp_cuda_wgmma.py --compile
    python bench/exp_cuda_wgmma.py [--reps 35] [--only qkv_gate]
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

constexpr int BN = 64;
constexpr int THREADS = 128;  // 1 warpgroup (4 warps)
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

// SWIZZLE_128B K-major shared memory descriptor for 64-bf16 (128B) wide box aligned to 1024B:
// bits [0:13]  = start_address >> 4
// bits [16:29] = LBO (16B >> 4 = 1)
// bits [32:45] = SBO (1024B >> 4 = 64)
// bits [62:63] = layout_type (1 = SWIZZLE_128B)
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

template <int BT, int BK, int STAGES, int P>
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
  constexpr int NT = BT / 8;

  extern __shared__ __align__(1024) unsigned char smem[];
  bf16* stage_base = reinterpret_cast<bf16*>(smem);
  constexpr size_t TMA_Total_Bytes = (size_t)STAGES * STAGE_BYTES;
  uint64_t* ready = reinterpret_cast<uint64_t*>(smem + TMA_Total_Bytes);
  constexpr size_t RECV_OFFSET = TMA_Total_Bytes + ((STAGES * sizeof(uint64_t) + 15) & ~size_t(15));
  float* recv = reinterpret_cast<float*>(smem + RECV_OFFSET);

  cg::cluster_group cluster = cg::this_cluster();
  const int csize = cluster.num_blocks();
  const int rank = cluster.block_rank();
  if (csize > 1) cluster_arrive_relaxed();

  const int tid = threadIdx.x;
  const int n0 = (blockIdx.x / csize) * BN;
  const int t0 = blockIdx.y * BT;
  const int k_per_cta = k / csize;
  const int k_steps = k_per_cta / BK;
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

  // Reuse start of smem for local accumulator staging (disjoint from recv at RECV_OFFSET).
  float* mine = reinterpret_cast<float*>(smem);
  const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, q = lane & 3;
  const int r0 = warp * 16 + g;
#pragma unroll
  for (int j = 0; j < NT; ++j) {
    const int c = 8 * j + 2 * q;
    mine[c * BN + r0] = acc[j][0];
    mine[(c + 1) * BN + r0] = acc[j][1];
    mine[c * BN + r0 + 8] = acc[j][2];
    mine[(c + 1) * BN + r0 + 8] = acc[j][3];
  }
  __syncthreads();

  if (csize == 1) {
    for (int e = tid; e < E; e += THREADS) {
      const int tt = e / BN;
      if (t0 + tt < m) {
        const long o = (long)(t0 + tt) * n + n0 + (e % BN);
        if constexpr (P == 2) static_cast<float*>(out)[o] = mine[e];
        else static_cast<bf16*>(out)[o] = __float2bfloat16(mine[e]);
      }
    }
    return;
  }

  cluster_wait();
  const int slice = E / csize;
  for (int e = tid; e < E; e += THREADS) {
    const int owner = e / slice;
    cluster.map_shared_rank(recv, owner)[rank * slice + e - owner * slice] = mine[e];
  }
  cluster_arrive_release();
  cluster_wait();
  for (int el = tid; el < slice; el += THREADS) {
    float v = 0.f;
    for (int c = 0; c < csize; ++c) v += recv[c * slice + el];
    const int e = rank * slice + el, tt = e / BN;
    if (t0 + tt < m) {
      const long o = (long)(t0 + tt) * n + n0 + (e % BN);
      if constexpr (P == 2) static_cast<float*>(out)[o] = v;
      else static_cast<bf16*>(out)[o] = __float2bfloat16(v);
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

template <int BT, int BK, int STAGES, int P>
void launch_wgmma_cfg(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& out, int csize) {
  const int m = x.size(0), k = x.size(1), n = out.size(1);
  TORCH_CHECK((csize & (csize - 1)) == 0 && csize >= 1 && csize <= MAXC, "wgmma: csize");
  TORCH_CHECK(n % BN == 0 && k % (csize * BK) == 0, "wgmma: shape divisibility");
  const int tma_bytes = STAGES * (P * BN + BT) * BK * (int)sizeof(bf16);
  const int mbar_bytes = (STAGES * (int)sizeof(uint64_t) + 15) & ~15;
  const int recv_bytes = csize > 1 ? (BN * BT * (int)sizeof(float)) : 0;
  const int smem = tma_bytes + mbar_bytes + recv_bytes;
  TORCH_CHECK(smem <= SMEM_LIMIT, "wgmma: smem=", smem, " exceeds limit");

  const bf16* wp0 = static_cast<const bf16*>(w.data_ptr());
  const bf16* wp1 = P == 2 ? wp0 + w.stride(0) : wp0;
  const bf16* xp = static_cast<const bf16*>(x.data_ptr());

  CUtensorMap w0_map = make_tma_2d(wp0, n, k, k, BN);
  CUtensorMap w1_map = P == 2 ? make_tma_2d(wp1, n, k, k, BN) : w0_map;
  CUtensorMap x_map = make_tma_2d(xp, m, k, (int)x.stride(0), BT);

  const dim3 grid((n / BN) * csize, (m + BT - 1) / BT, 1);
  auto kern = tma_wgmma_kernel<BT, BK, STAGES, P>;
  static const bool init = (set_attrs_once(kern, SMEM_LIMIT), true);
  (void)init;
  cudaLaunchAttribute attr;
  cudaLaunchConfig_t cfg = cluster_cfg(grid, THREADS, smem, csize, &attr);
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, w0_map, w1_map, x_map, out.data_ptr(), m, n, k));
}

void launch_wgmma(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t bt, int64_t bk,
                  int64_t stages, int64_t csize) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && out.is_cuda(), "wgmma: CUDA tensors expected");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16, "wgmma: bf16");
  const int p = w.dim() == 3 ? 2 : 1;
#define W(BT_, BK_, ST_, P_)                                                \
  if (bt == BT_ && bk == BK_ && stages == ST_ && p == P_) {                 \
    launch_wgmma_cfg<BT_, BK_, ST_, P_>(x, w, out, (int)csize);             \
    return;                                                                 \
  }
  WGMMA_CONFIGS
#undef W
  TORCH_CHECK(false, "wgmma: config (", bt, ", ", bk, ", ", stages, ", ", p, ") not compiled");
}
"""

CPP = ("void launch_wgmma(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t bt, int64_t bk,"
       " int64_t stages, int64_t csize);")
CFLAGS = ["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v"]
LDFLAGS = ["-lcuda"]

# (BT, BK, STAGES, P): BN = 64 fixed, csize passed at runtime (1, 2, 4, 8).
WGMMA_ALL = (
    # BF16 layers (P = 1), small/medium m (m = 8, 16, 32):
    (8, 128, 4, 1), (8, 128, 8, 1), (8, 256, 4, 1),
    (16, 128, 4, 1), (16, 128, 8, 1), (16, 256, 4, 1),
    (32, 128, 4, 1), (32, 128, 8, 1), (32, 256, 4, 1),
    # BF16 layers (P = 1), large m (m = 128, 256):
    (64, 64, 8, 1), (64, 128, 4, 1), (64, 128, 6, 1),
    (128, 64, 4, 1), (128, 64, 6, 1), (128, 128, 4, 1),
    # Router (P = 2), m = 128, 256 (and m = 16, 32):
    (16, 128, 4, 2), (32, 128, 4, 2),
    (64, 64, 4, 2), (64, 128, 4, 2),
    (128, 64, 4, 2), (128, 128, 3, 2),
)

BN = 64
SMEM_LIMIT = 232448 - 1024
OVERHEAD_US = 4.0

CASES = [
    ("qkv_gate", 8),
    ("qkv_gate", 16),
    ("qkv_gate", 32),
    ("qkv_gate", 128),
    ("qkv_gate", 256),
    ("unpooler", 8),
    ("pooler", 8),
    ("pooler", 16),
    ("pooler", 32),
    ("pooler", 128),
    ("pooler", 256),
    ("o", 8),
    ("o", 16),
    ("o", 32),
    ("router", 16),
    ("router", 32),
    ("router", 128),
    ("router", 256),
]

NEARBY = {
    8: (5, 7, 8),
    16: (9, 12, 16),
    32: (17, 24, 31, 32),
    128: (100, 127, 128),
    256: (200, 255, 256),
}


class WTile(NamedTuple):
    bt: int
    bk: int
    stages: int
    csize: int


def smem_bytes(t: WTile, p: int) -> int:
    tma = t.stages * (p * BN + t.bt) * t.bk * 2
    mbar = (t.stages * 8 + 15) & ~15
    recv = (BN * t.bt * 4) if t.csize > 1 else 0
    return tma + mbar + recv


def valid(t: WTile, p: int, n: int, k: int) -> bool:
    if n % BN != 0 or k % (t.csize * t.bk) != 0:
        return False
    if (k // (t.csize * t.bk)) < t.stages:
        return False
    return smem_bytes(t, p) <= SMEM_LIMIT


def candidates(layer: str, m: int, n: int, k: int):
    p = 2 if layer == "router" else 1
    if layer == "router":
        csizes = (4, 8)
    elif layer == "unpooler":
        csizes = (2, 4, 8)
    elif layer == "qkv_gate":
        csizes = (1, 2, 4)
    elif layer == "pooler":
        csizes = (1, 2)
    else:
        csizes = (1,)

    for bt, bk, st, pp in WGMMA_ALL:
        if pp != p:
            continue
        if m <= 32 and bt != m:
            continue
        if m > 32 and bt not in (64, 128):
            continue
        for c in csizes:
            t = WTile(bt, bk, st, c)
            if not valid(t, p, n, k):
                continue
            ctas = (n // BN) * c * -(-m // bt)
            if 16 <= ctas <= 576:
                yield t


def source(configs=WGMMA_ALL) -> str:
    return (
        "#define WGMMA_CONFIGS "
        + " ".join(f"W({bt},{bk},{st},{p})" for bt, bk, st, p in configs)
        + "\n"
        + SRC
    )


def build(configs=WGMMA_ALL, verbose: bool = False):
    from torch.utils.cpp_extension import load_inline

    src = source(configs)
    tag = hashlib.sha1((src + " ".join(CFLAGS + LDFLAGS)).encode()).hexdigest()[:10]
    return load_inline(
        name=f"exp_cuda_wgmma_{tag}",
        cpp_sources=CPP,
        cuda_sources=src,
        functions=["launch_wgmma"],
        extra_cuda_cflags=CFLAGS,
        extra_cflags=["-O3"],
        extra_ldflags=LDFLAGS,
        verbose=verbose,
    )


def run_wgmma(ext, x: torch.Tensor, w: torch.Tensor, n: int, t: WTile) -> torch.Tensor:
    dtype = torch.float32 if w.dim() == 3 else torch.bfloat16
    out = torch.empty(x.shape[0], n, device=x.device, dtype=dtype)
    ext.launch_wgmma(x, w, out, t.bt, t.bk, t.stages, t.csize)
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
        print(f"built {len(WGMMA_ALL)} tma_wgmma configs in {time.time() - t0:.1f}s", flush=True)
        return

    from common import load_module, reference, weights
    from harness import FLUSH_BYTES, time_replay

    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=35)
    ap.add_argument("--only", default="")
    ap.add_argument("--base", default=BASE_CP)
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

    rows = []
    for layer, m in CASES:
        if args.only and layer != args.only:
            continue
        data = reference.generate_input(m, layer, 1)
        n, k = data["config"]["n"], data["config"]["k"]
        wl = planes[layer] if layer == "router" else w[layer][:n].contiguous()
        x = data["x"].clone()
        sides = {}
        for t in candidates(layer, m, n, k):
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
            rows.append((layer, m, best, med[best], med["base"], med["cublas"], nbytes, sol, diff))

    if rows:
        print(
            f"\n{'layer':9} {'m':>4} {'best (bt,bk,st,csize)':26} {'wgmma':>8} {'base':>8} "
            f"{'cublas':>8} {'TB/s':>6} {'sol':>6} {'det':>5}"
        )
        for layer, m, best, a, b, c, nbytes, sol, diff in rows:
            print(
                f"{layer:9} {m:4d} {str(tuple(best)):26} {a:8.2f} {b:8.2f} {c:8.2f} "
                f"{nbytes / a / 1e6:6.2f} {sol:6.2f} {diff:3d}/200"
            )
    print(f"total {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
