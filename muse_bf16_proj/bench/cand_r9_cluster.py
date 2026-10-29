# okc: problem=muse_bf16_proj
"""Dense projections of Muse Spark Open, tuned for small token counts on H200 (Checkpoint 4).

Checkpoint 4 integrates:
1. Custom CUDA C++ (sm_90a) 2D TMA + wgmma.mma_async + Cluster DSMEM:
   - Stacked hi+lo router (BN_OUT=32): 1 wgmma per K-step, doubles CTAs, float4 DSMEM reduction.
     Wins across m in {8, 16, 32, 128, 256}: saving -1.0 to -2.6 us per shape.
   - Bank-conflict-free MPAD=68 float4 DSMEM wgmma_kernel:
     Wins on unpooler_slice m=8 (-0.69 us), qkv_gate m=16 (-0.35 us), qkv_gate m=32 (-1.78 us),
     and pooler m=128 (-0.93 us).
   - Keeps row_gemv for m=1 on router and unpooler_slice.
2. Large-m Triton _big_gemm (WGMMA + 1-load commutative SPLIT=2 and canonical SPLIT=3):
   - qkv_gate m=128: BM128 BN64 BK64 SPLIT=3 (-0.80 us)
   - pooler m=128: BM64 BN128 BK64 SPLIT=2 SWAP=1 (-0.75 us)
   - pooler m=256: BM64 BN128 BK64 SPLIT=1 SWAP=0 (-1.09 us)
   - unpooler m=128: BM64 BN128 BK128 SPLIT=1 SWAP=0 (-0.16 us)
   - o m=256: BM128 BN128 BK64 SPLIT=1 SWAP=0 TMA_STORE=1 (-0.26 us)
3. Gluon TMA-ring + warpgroup_mma:
   - pooler m in {8, 16, 32}, unpooler m in {16, 32}, o m in {8, 16, 32}, qkv_gate m=8 (kfold).
4. Pipelined m=1 GEMV:
   - qkv_gate m=1: BN=2, BK=1024, st=2, cs_ca (-0.51 us)
   - o m=1: BN=16, BK=256, st=1, cg_ca (-0.08 us)
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

try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma, warpgroup_mma, warpgroup_mma_wait
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor as GluonDesc
except ImportError:
    gluon = None


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor
    config: Config


class Tile(NamedTuple):
    max_m: int
    block_n: int
    block_k: int
    block_t: int
    split: int
    warps: int
    stages: int
    kind: str = "dot"


TILES: dict[str, tuple[Tile, ...]] = {
    "router": (
        Tile(16, 32, 256, 16, 16, 4, 4),
        Tile(32, 32, 512, 32, 16, 4, 3),
        Tile(128, 64, 256, 64, 16, 8, 3),
        Tile(256, 64, 256, 64, 8, 4, 3),
    ),
}

BF16_TILES: dict[str, tuple[Tile, ...]] = {
    "pooler": (Tile(1, 4, 1024, 1, 1, 4, 2, "fma"), Tile(8, 32, 256, 16, 1, 4, 5),
               Tile(16, 32, 256, 16, 1, 4, 5), Tile(32, 64, 128, 32, 2, 4, 5)),
    "qkv_gate": (Tile(1, 4, 1024, 1, 1, 4, 3, "fma"), Tile(8, 64, 128, 16, 4, 4, 4),
                 Tile(16, 64, 128, 16, 4, 4, 4, "tma")),
    "o": (Tile(1, 2, 1024, 1, 1, 4, 3, "fma"), Tile(8, 64, 128, 16, 1, 4, 6),
          Tile(16, 64, 128, 16, 1, 4, 5), Tile(32, 64, 128, 32, 1, 4, 5)),
    "unpooler": (Tile(16, 64, 128, 16, 1, 4, 6), Tile(32, 64, 256, 32, 1, 4, 4, "tma")),
    "unpooler_slice": (Tile(1, 2, 2048, 1, 1, 8, 2, "fma"), Tile(8, 64, 128, 16, 8, 4, 4)),
}


class GTile(NamedTuple):
    min_m: int
    max_m: int
    block_n: int
    block_k: int
    block_t: int
    stages: int
    split: int = 1
    q: int = 1


GLUON_TILES: dict[str, tuple[GTile, ...]] = {
    "pooler": (GTile(2, 8, 64, 128, 8, 8), GTile(9, 16, 64, 128, 16, 8), GTile(17, 32, 64, 128, 32, 8)),
    "unpooler": (GTile(9, 16, 64, 256, 16, 4), GTile(17, 32, 64, 128, 32, 8)),
    "qkv_gate": (GTile(2, 8, 32, 256, 8, 4, q=2), GTile(9, 16, 32, 256, 16, 4, q=2),
                 GTile(17, 32, 64, 256, 32, 4, split=3)),
    "o": (GTile(2, 8, 64, 256, 8, 4), GTile(9, 16, 64, 128, 16, 4), GTile(17, 32, 64, 128, 32, 4)),
}
GLUON_SMEM = 220 * 1024

ROUTER_BIG: tuple[tuple[int, int, tuple[int, ...]], ...] = (
    (97, 128, (64, 32, 128, 8, 0, 4, 4)),
    (129, 256, (64, 32, 128, 4, 0, 4, 4)),
)
UNPOOL_SLICE = 1024
BF16_WARMUP_M = (1, 2, 5, 7, 8, 9, 16, 17, 32, 128, 256)
GEMV_FP32 = Tile(1, 8, 512, 1, 16, 4, 3)
WS_FLOATS = 4 * 2**20
COUNTERS = 8192
WARMUP_M = (1, 5, 7, 8, 9, 16, 17, 32, 33, 128, 130, 256)

# (layer, m) -> (BM, BN, BK, SPLIT, warps, stages, SWAP, TMA_STORE)
BIG_TILES = {
    ("qkv_gate", 128): (128, 64, 64, 3, 4, 8, 0, 0),
    ("pooler", 128): (64, 128, 64, 2, 4, 8, 1, 0),
    ("pooler", 256): (64, 128, 64, 1, 4, 8, 0, 0),
}


# ===========================================================================
# CUDA C++ (sm_90a): row_gemv + tma_wgmma_kernel + tma_wgmma_router_bn32
# ===========================================================================

CUDA_SRC = r"""
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
constexpr int MPAD = 68;     // 68 floats (272B): 16B aligned for float4 AND zero bank conflicts!
constexpr int THREADS = 128; // 1 warpgroup (4 warps)
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

template <int BT, int BK, int STAGES, int CSIZE>
__global__ void __launch_bounds__(THREADS, 1)
tma_wgmma_router_bn32(const __grid_constant__ CUtensorMap w0_map,
                      const __grid_constant__ CUtensorMap w1_map,
                      const __grid_constant__ CUtensorMap x_map,
                      float* __restrict__ out, int m, int n, int k) {
  constexpr int BN_OUT = 32;
  constexpr int KBOX = BK / 64;
  constexpr int HALF_W_ELEMS = BN_OUT * 64;
  constexpr int W_BOX_ELEMS = BN * 64;
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

// row_gemv for m=1 on router and unpooler_slice
template <bool F32, int R, int MT, int PER>
__global__ void __launch_bounds__(256, 2)
row_gemv(const bf16* __restrict__ x, const void* __restrict__ w_void, void* __restrict__ out_void,
         int m, int n, int k, int stride_xm) {
  constexpr int THREADS_PER_CTA = 256;
  const int tid = threadIdx.x;
  const int row0 = blockIdx.x * R;
  if (row0 >= n) return;

  uint4 wv[R][PER];
  if constexpr (F32) {
    const float* w = static_cast<const float*>(w_void);
    for (int r = 0; r < R; ++r) {
      for (int p = 0; p < PER; ++p) {
        wv[r][p] = *reinterpret_cast<const uint4*>(w + (row0 + r) * k + p * THREADS_PER_CTA * 4 + tid * 4);
      }
    }
  } else {
    const bf16* w = static_cast<const bf16*>(w_void);
    for (int r = 0; r < R; ++r) {
      for (int p = 0; p < PER; ++p) {
        wv[r][p] = *reinterpret_cast<const uint4*>(w + (row0 + r) * k + p * THREADS_PER_CTA * 8 + tid * 8);
      }
    }
  }

  for (int t = 0; t < m; ++t) {
    const bf16* xt = x + t * stride_xm;
    float sum[R] = {0.0f};

    if constexpr (F32) {
      for (int p = 0; p < PER; ++p) {
        const float* wp0 = reinterpret_cast<const float*>(&wv[0][p]);
        const int k_base = p * THREADS_PER_CTA * 4 + tid * 4;
        const uint2 xu = *reinterpret_cast<const uint2*>(xt + k_base);
        const bf16* xbp = reinterpret_cast<const bf16*>(&xu);
        for (int r = 0; r < R; ++r) {
          const float* wp = reinterpret_cast<const float*>(&wv[r][p]);
          for (int i = 0; i < 4; ++i) sum[r] += wp[i] * __bfloat162float(xbp[i]);
        }
      }
    } else {
      for (int p = 0; p < PER; ++p) {
        const int k_base = p * THREADS_PER_CTA * 8 + tid * 8;
        const uint4 xu = *reinterpret_cast<const uint4*>(xt + k_base);
        const uint32_t* xp32 = reinterpret_cast<const uint32_t*>(&xu);
        for (int r = 0; r < R; ++r) {
          const uint32_t* wp32 = reinterpret_cast<const uint32_t*>(&wv[r][p]);
          for (int i = 0; i < 4; ++i) {
            bf162 a = *reinterpret_cast<const bf162*>(&wp32[i]);
            bf162 b = *reinterpret_cast<const bf162*>(&xp32[i]);
            sum[r] += __bfloat162float(a.x) * __bfloat162float(b.x) + __bfloat162float(a.y) * __bfloat162float(b.y);
          }
        }
      }
    }

    for (int r = 0; r < R; ++r) {
      for (int mask = 16; mask > 0; mask >>= 1) sum[r] += __shfl_xor_sync(0xffffffff, sum[r], mask);
    }

    __shared__ float smem_red[R][8];
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (lane == 0) {
      for (int r = 0; r < R; ++r) smem_red[r][warp] = sum[r];
    }
    __syncthreads();

    if (tid < 8) {
      for (int r = 0; r < R; ++r) {
        float val = smem_red[r][tid];
        for (int mask = 4; mask > 0; mask >>= 1) val += __shfl_xor_sync(0xff, val, mask);
        if (tid == 0 && row0 + r < n) {
          if constexpr (F32) {
            static_cast<float*>(out_void)[t * n + row0 + r] = val;
          } else {
            static_cast<bf16*>(out_void)[t * n + row0 + r] = __float2bfloat16(val);
          }
        }
      }
    }
    __syncthreads();
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
void set_attrs(K kern, int smem_max) {
  C10_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(kern), cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max));
  C10_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(kern), cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
}

template <int BT, int BK, int STAGES, int P, int CSIZE>
void launch_wgmma_64(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& out) {
  const int m = x.size(0), k = x.size(1), n = out.size(1);
  const int tma_bytes = STAGES * (P * BN + BT) * BK * (int)sizeof(bf16);
  const int mbar_bytes = (STAGES * (int)sizeof(uint64_t) + 15) & ~15;
  const int recv_bytes = CSIZE > 1 ? (BN * BT * (int)sizeof(float)) : 0;
  const int smem = tma_bytes + mbar_bytes + recv_bytes;
  const bf16* wp0 = static_cast<const bf16*>(w.data_ptr());
  const bf16* wp1 = P == 2 ? wp0 + w.stride(0) : wp0;
  const bf16* xp = static_cast<const bf16*>(x.data_ptr());

  CUtensorMap w0_map = make_tma_2d(wp0, n, k, k, BN);
  CUtensorMap w1_map = P == 2 ? make_tma_2d(wp1, n, k, k, BN) : w0_map;
  CUtensorMap x_map = make_tma_2d(xp, m, k, (int)x.stride(0), BT);

  const dim3 grid((n / BN) * CSIZE, (m + BT - 1) / BT, 1);
  auto kern = tma_wgmma_kernel<BT, BK, STAGES, P, CSIZE>;
  set_attrs(kern, smem);
  cudaLaunchAttribute attr;
  cudaLaunchConfig_t cfg = cluster_cfg(grid, THREADS, smem, CSIZE, &attr);
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, w0_map, w1_map, x_map, out.data_ptr(), m, n, k));
}

template <int BT, int BK, int STAGES, int CSIZE>
void launch_router_32(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& out) {
  constexpr int BN_OUT = 32;
  const int m = x.size(0), k = x.size(1), n = out.size(1);
  const int tma_bytes = STAGES * (BN + BT) * BK * (int)sizeof(bf16);
  const int mbar_bytes = (STAGES * (int)sizeof(uint64_t) + 15) & ~15;
  const int recv_bytes = BN_OUT * BT * (int)sizeof(float);
  const int smem = tma_bytes + mbar_bytes + recv_bytes;

  const bf16* wp0 = static_cast<const bf16*>(w.data_ptr());
  const bf16* wp1 = wp0 + w.stride(0);
  const bf16* xp = static_cast<const bf16*>(x.data_ptr());

  CUtensorMap w0_map = make_tma_2d(wp0, n, k, k, BN_OUT);
  CUtensorMap w1_map = make_tma_2d(wp1, n, k, k, BN_OUT);
  CUtensorMap x_map = make_tma_2d(xp, m, k, (int)x.stride(0), BT);

  const dim3 grid((n / BN_OUT) * CSIZE, (m + BT - 1) / BT, 1);
  auto kern = tma_wgmma_router_bn32<BT, BK, STAGES, CSIZE>;
  set_attrs(kern, smem);
  cudaLaunchAttribute attr;
  cudaLaunchConfig_t cfg = cluster_cfg(grid, THREADS, smem, CSIZE, &attr);
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, w0_map, w1_map, x_map, static_cast<float*>(out.data_ptr()), m, n, k));
}

void launch_wgmma(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t bn, int64_t bt, int64_t bk,
                  int64_t stages, int64_t csize) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && out.is_cuda(), "wgmma: CUDA tensors expected");
  const int p = w.dim() == 3 ? 2 : 1;
  if (bn == 32 && p == 2) {
    if (bt == 8 && bk == 128 && stages == 8 && csize == 8) { launch_router_32<8, 128, 8, 8>(x, w, out); return; }
    if (bt == 16 && bk == 128 && stages == 8 && csize == 8) { launch_router_32<16, 128, 8, 8>(x, w, out); return; }
    if (bt == 32 && bk == 128 && stages == 8 && csize == 8) { launch_router_32<32, 128, 8, 8>(x, w, out); return; }
    if (bt == 128 && bk == 128 && stages == 4 && csize == 8) { launch_router_32<128, 128, 4, 8>(x, w, out); return; }
    if (bt == 128 && bk == 128 && stages == 4 && csize == 4) { launch_router_32<128, 128, 4, 4>(x, w, out); return; }
  }
  if (bn == 64) {
    if (bt == 8 && bk == 128 && stages == 8 && p == 1 && csize == 4) { launch_wgmma_64<8, 128, 8, 1, 4>(x, w, out); return; }
    if (bt == 16 && bk == 128 && stages == 4 && p == 1 && csize == 4) { launch_wgmma_64<16, 128, 4, 1, 4>(x, w, out); return; }
    if (bt == 32 && bk == 128 && stages == 4 && p == 1 && csize == 4) { launch_wgmma_64<32, 128, 4, 1, 4>(x, w, out); return; }
    if (bt == 128 && bk == 64 && stages == 6 && p == 1 && csize == 2) { launch_wgmma_64<128, 64, 6, 1, 2>(x, w, out); return; }
  }
  TORCH_CHECK(false, "wgmma: config (bn=", bn, ", bt=", bt, ", bk=", bk, ", st=", stages, ", p=", p, ", c=", csize, ") not compiled");
}

void launch_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t rows, int64_t mt, int64_t per) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && out.is_cuda(), "row_gemv: CUDA tensors expected");
  const bool f32 = out.scalar_type() == at::kFloat;
  const int m = x.size(0), n = out.size(1), k = x.size(1);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const bf16* xp = static_cast<const bf16*>(x.data_ptr());
  if (f32 && rows == 1 && mt == 1 && per == 8) {
    row_gemv<true, 1, 1, 8><<<n / 1, 256, 0, stream>>>(xp, w.data_ptr(), out.data_ptr(), m, n, k, (int)x.stride(0));
    C10_CUDA_CHECK(cudaGetLastError());
    return;
  }
  if (!f32 && rows == 2 && mt == 1 && per == 2) {
    row_gemv<false, 2, 1, 2><<<n / 2, 256, 0, stream>>>(xp, w.data_ptr(), out.data_ptr(), m, n, k, (int)x.stride(0));
    C10_CUDA_CHECK(cudaGetLastError());
    return;
  }
  if (!f32 && rows == 2 && mt == 2 && per == 2) {
    row_gemv<false, 2, 2, 2><<<n / 2, 256, 0, stream>>>(xp, w.data_ptr(), out.data_ptr(), m, n, k, (int)x.stride(0));
    C10_CUDA_CHECK(cudaGetLastError());
    return;
  }
  if (!f32 && rows == 4 && mt == 4 && per == 2) {
    row_gemv<false, 4, 4, 2><<<n / 4, 256, 0, stream>>>(xp, w.data_ptr(), out.data_ptr(), m, n, k, (int)x.stride(0));
    C10_CUDA_CHECK(cudaGetLastError());
    return;
  }
  TORCH_CHECK(false, "row_gemv: config not compiled");
}
"""

CUDA_CPP = ("void launch_wgmma(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t bn, int64_t bt,"
            " int64_t bk, int64_t stages, int64_t csize);\n"
            "void launch_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t rows, int64_t mt,"
            " int64_t per);")
CUDA_FLAGS = ["-O3", "-gencode=arch=compute_90a,code=sm_90a"]
CUDA_LDFLAGS = ["-lcuda"]

# key -> tuple of (lo_m, hi_m, kind, params)
CUDA_TILES: dict[str, tuple[tuple[int, int, str, tuple[int, ...]], ...]] = {
    "router": (
        (1, 1, "gemv", (1, 1, 8)),
        (8, 8, "wgmma", (32, 8, 128, 8, 8)),
        (16, 16, "wgmma", (32, 16, 128, 8, 8)),
        (32, 32, "wgmma", (32, 32, 128, 8, 8)),
        (128, 128, "wgmma", (32, 128, 128, 4, 8)),
        (256, 256, "wgmma", (32, 128, 128, 4, 4)),
    ),
    "unpooler_slice": (
        (2, 2, "gemv", (2, 2, 2)),
        (3, 4, "gemv", (4, 4, 2)),
        (8, 8, "wgmma", (64, 8, 128, 8, 4)),
    ),
    "qkv_gate": (
        (16, 16, "wgmma", (64, 16, 128, 4, 4)),
        (32, 32, "wgmma", (64, 32, 128, 4, 4)),
    ),
    "pooler": (
        (128, 128, "wgmma", (64, 128, 64, 6, 2)),
    ),
}


def _build_cuda():
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((CUDA_SRC + " ".join(CUDA_FLAGS + CUDA_LDFLAGS)).encode()).hexdigest()[:10]
    return load_inline(
        name=f"muse_proj_cuda_cp4_{tag}",
        cpp_sources=CUDA_CPP,
        cuda_sources=CUDA_SRC,
        functions=["launch_wgmma", "launch_gemv"],
        extra_cuda_cflags=CUDA_FLAGS,
        extra_cflags=["-O3"],
        extra_ldflags=CUDA_LDFLAGS,
    )


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
    elif kind == "wgmma":
        state["ext"].launch_wgmma(x, w, out, *params)
    return out


# ===========================================================================
# Triton & Gluon Kernels
# ===========================================================================

@triton.jit
def _split_k_finish(acc, ws_ptrs, ws_stride, out_ptrs, mask, cnt_ptr, pid_s, SPLIT: tl.constexpr):
    tl.store(ws_ptrs + pid_s * ws_stride, acc, mask=mask)
    tl.debug_barrier()
    done = tl.atomic_add(cnt_ptr, 1, sem="acq_rel", scope="gpu")
    if done == SPLIT - 1:
        total = tl.zeros_like(acc)
        for s in tl.static_range(SPLIT):
            total += tl.load(ws_ptrs + s * ws_stride, mask=mask, other=0.0, cache_modifier=".cg")
        tl.store(out_ptrs, total.to(out_ptrs.dtype.element_ty), mask=mask)
        tl.atomic_xchg(cnt_ptr, 0)


@triton.jit(do_not_specialize=["M"])
def _skinny_gemm(x_ptr, w_ptr, out_ptr, ws_ptr, cnt_ptr, M, stride_xm,
                 N: tl.constexpr, KX: tl.constexpr, KW: tl.constexpr, SPLIT: tl.constexpr,
                 BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_T: tl.constexpr):
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
def _bf16_gemm(x_ptr, w_ptr, desc, out_ptr, ws_ptr, cnt_ptr, M, stride_xm,
               N: tl.constexpr, K: tl.constexpr, SPLIT: tl.constexpr,
               BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_T: tl.constexpr,
               USE_TMA: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_t = tl.arange(0, BLOCK_T)
    t_mask = offs_t < M
    k_len: tl.constexpr = K // SPLIT
    k_start = pid_s * k_len

    acc = tl.zeros((BLOCK_N, BLOCK_T), dtype=tl.float32)
    for kk in range(0, k_len, BLOCK_K):
        if USE_TMA:
            w = desc.load([pid_n * BLOCK_N, k_start + kk])
        else:
            offs_k = k_start + kk + tl.arange(0, BLOCK_K)
            w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :], eviction_policy="evict_first")
        offs_k_scalar = k_start + kk + tl.arange(0, BLOCK_K)
        xt = tl.load(x_ptr + offs_t[None, :] * stride_xm + offs_k_scalar[:, None],
                     mask=t_mask[None, :], other=0.0, eviction_policy="evict_last")
        acc = tl.dot(w, xt, acc)

    out_ptrs = out_ptr + offs_t[None, :] * N + offs_n[:, None]
    if SPLIT == 1:
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=t_mask[None, :])
    else:
        ws_ptrs = ws_ptr + offs_t[None, :] * N + offs_n[:, None]
        _split_k_finish(acc, ws_ptrs, BLOCK_T * N, out_ptrs, t_mask[None, :],
                        cnt_ptr + pid_n, pid_s, SPLIT)


@triton.jit
def _bf16_gemv(x_ptr, w_ptr, out_ptr, ws_ptr, cnt_ptr, N: tl.constexpr, K: tl.constexpr,
               SPLIT: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
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


# Large-m GEMM with SWAP, TMA_STORE, 1-load commutative SPLIT=2 and canonical SPLIT=3
@triton.jit(do_not_specialize=["M"])
def _big_gemm(x_desc, w_desc, out_ref, ws_ptr, cnt_ptr, M,
              N: tl.constexpr, K: tl.constexpr, SPLIT: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
              SWAP: tl.constexpr, TMA_STORE: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    pid_m = pid % num_m
    pid_s = (pid // num_m) % SPLIT
    pid_n = pid // (num_m * SPLIT)
    KI: tl.constexpr = K // BK
    k0 = pid_s * KI // SPLIT
    k1 = (pid_s + 1) * KI // SPLIT

    if SWAP:
        acc = tl.zeros((BN, BM), dtype=tl.float32)
    else:
        acc = tl.zeros((BM, BN), dtype=tl.float32)

    for ki in range(k0, k1):
        a = x_desc.load([pid_m * BM, ki * BK])
        b = w_desc.load([pid_n * BN, ki * BK])
        if SWAP:
            acc = tl.dot(b, a.T, acc)
        else:
            acc = tl.dot(a, b.T, acc)

    if SPLIT == 1:
        if TMA_STORE:
            val = acc.to(tl.bfloat16)
            if SWAP:
                val = val.T
            out_ref.store([pid_m * BM, pid_n * BN], val)
        else:
            local_m = tl.arange(0, BM)
            local_n = tl.arange(0, BN)
            offs_m = pid_m * BM + local_m
            offs_n = pid_n * BN + local_n
            if SWAP:
                out_idx = offs_m[None, :] * N + offs_n[:, None]
                mask = (offs_m < M)[None, :]
            else:
                out_idx = offs_m[:, None] * N + offs_n[None, :]
                mask = (offs_m < M)[:, None]
            tl.store(out_ref + out_idx, acc.to(tl.bfloat16), mask=mask)
    else:
        TILE_ELEMS: tl.constexpr = BM * BN
        local_m = tl.arange(0, BM)
        local_n = tl.arange(0, BN)
        if SWAP:
            ws_offs = local_n[:, None] * BM + local_m[None, :]
        else:
            ws_offs = local_m[:, None] * BN + local_n[None, :]
        tile_id = pid_n * num_m + pid_m
        ws_tile = ws_ptr + tile_id * (SPLIT * TILE_ELEMS) + ws_offs
        tl.store(ws_tile + pid_s * TILE_ELEMS, acc)
        tl.debug_barrier()
        cnt = cnt_ptr + tile_id
        done = tl.atomic_add(cnt, 1, sem="acq_rel", scope="gpu")
        if done == SPLIT - 1:
            if SPLIT == 2:
                total = acc + tl.load(ws_tile + (1 - pid_s) * TILE_ELEMS, cache_modifier=".cg")
            else:
                total = tl.zeros_like(acc)
                for s in tl.static_range(SPLIT):
                    total += tl.load(ws_tile + s * TILE_ELEMS, cache_modifier=".cg")
            local_m = tl.arange(0, BM)
            local_n = tl.arange(0, BN)
            offs_m = pid_m * BM + local_m
            offs_n = pid_n * BN + local_n
            if SWAP:
                out_idx = offs_m[None, :] * N + offs_n[:, None]
                mask = (offs_m < M)[None, :]
            else:
                out_idx = offs_m[:, None] * N + offs_n[None, :]
                mask = (offs_m < M)[:, None]
            tl.store(out_ref + out_idx, total.to(tl.bfloat16), mask=mask)
            tl.atomic_xchg(cnt, 0)


def _big_launch(x, w, state, layer: str) -> torch.Tensor:
    m, (n, k) = x.shape[0], w.shape
    bm, bn, bk, split, warps, stages, swap, tma_store = BIG_TILES[(layer, m)]
    out = torch.empty(m, n, device=x.device, dtype=torch.bfloat16)
    out_target = state["big_out_desc"][(layer, m)] if tma_store else out
    _big_gemm[(triton.cdiv(m, bm) * (n // bn) * split,)](
        TensorDescriptor.from_tensor(x, [bm, bk]), state["big_desc"][(layer, m)], out_target, state["ws"],
        state["cnt"], m, N=n, K=k,
        SPLIT=split, BM=bm, BN=bn, BK=bk, SWAP=swap, TMA_STORE=tma_store,
        num_warps=warps, num_stages=stages)
    return out


# Gluon GEMV & Split-K
if gluon is not None:
    @gluon.jit
    def _gluon_gemv(w_desc, x_desc, out_ptr, ws_ptr, cnt_ptr, M, N: gl.constexpr, K: gl.constexpr,
                    STAGES: gl.constexpr, SPLIT: gl.constexpr, num_warps: gl.constexpr):
        BN: gl.constexpr = w_desc.block_type.shape[0]
        BK: gl.constexpr = w_desc.block_type.shape[1]
        BT: gl.constexpr = x_desc.block_type.shape[0]
        KI: gl.constexpr = K // BK
        mma: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[num_warps, 1],
                                                      instr_shape=[16, BT, 16])
        pid_n = gl.program_id(0)
        pid_s = gl.program_id(1)
        n0 = pid_n * BN
        if SPLIT == 1:
            k_start = 0
            k_steps = KI
        else:
            k_base: gl.constexpr = KI // SPLIT
            k_start = pid_s * k_base
            k_steps = KI - k_start if pid_s == SPLIT - 1 else k_base
        w_bufs = gl.allocate_shared_memory(w_desc.dtype, [STAGES] + w_desc.block_type.shape, w_desc.layout)
        x_bufs = gl.allocate_shared_memory(x_desc.dtype, [STAGES] + x_desc.block_type.shape, x_desc.layout)
        ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
        for i in gl.static_range(STAGES):
            mbarrier.init(ready.index(i), count=1)
        nbytes: gl.constexpr = w_desc.block_type.nbytes + x_desc.block_type.nbytes
        for i in gl.static_range(STAGES - 1):
            mbarrier.expect(ready.index(i), nbytes)
            tma.async_load(w_desc, [n0, (k_start + i) * BK], ready.index(i), w_bufs.index(i))
            tma.async_load(x_desc, [0, (k_start + i) * BK], ready.index(i), x_bufs.index(i))
        acc = gl.zeros([BN, BT], gl.float32, layout=mma)
        for i in range(k_steps):
            s = i % STAGES
            mbarrier.wait(ready.index(s), (i // STAGES) & 1)
            acc = warpgroup_mma(w_bufs.index(s), x_bufs.index(s).permute((1, 0)), acc, is_async=True)
            # wgmma i-1 has retired, so its slot can take tile i + STAGES - 1.
            acc = warpgroup_mma_wait(1, deps=(acc,))
            j = i + STAGES - 1
            sj = j % STAGES
            pred = j < k_steps
            mbarrier.expect(ready.index(sj), nbytes, pred=pred)
            tma.async_load(w_desc, [n0, (k_start + j) * BK], ready.index(sj), w_bufs.index(sj), pred=pred)
            tma.async_load(x_desc, [0, (k_start + j) * BK], ready.index(sj), x_bufs.index(sj), pred=pred)
        acc = warpgroup_mma_wait(0, deps=(acc,))
        offs_nl = gl.arange(0, BN, layout=gl.SliceLayout(1, mma))
        offs_t = gl.arange(0, BT, layout=gl.SliceLayout(0, mma))
        out_ptrs = out_ptr + offs_t[None, :] * N + (n0 + offs_nl)[:, None]
        mask = (offs_t < M)[None, :]
        if SPLIT == 1:
            gl.store(out_ptrs, acc.to(gl.bfloat16), mask=mask)
        else:
            ws_base = ws_ptr + pid_n * (SPLIT * BN * BT) + offs_nl[:, None] * BT + offs_t[None, :]
            gl.store(ws_base + pid_s * (BN * BT), acc)
            gl.barrier()
            done = gl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
            if done == SPLIT - 1:
                if SPLIT == 2:
                    total = acc + gl.load(ws_base + (1 - pid_s) * (BN * BT), cache_modifier=".cg")
                else:
                    total = gl.zeros([BN, BT], gl.float32, layout=mma)
                    for s in gl.static_range(SPLIT):
                        total += gl.load(ws_base + s * (BN * BT), cache_modifier=".cg")
                gl.store(out_ptrs, total.to(gl.bfloat16), mask=mask)
                gl.atomic_xchg(cnt_ptr + pid_n, 0)

    @gluon.jit
    def _kfold_issue(w_desc, x_desc, w_bufs, x_bufs, bar, s, i, n0, pred,
                     R: gl.constexpr, BT: gl.constexpr, BK: gl.constexpr, Q: gl.constexpr):
        NBYTES: gl.constexpr = Q * (w_desc.block_type.nbytes + x_desc.block_type.nbytes)
        mbarrier.expect(bar, NBYTES, pred=pred)
        for q in gl.static_range(Q):
            k_off = (Q * i + q) * BK
            tma.async_load(w_desc, [n0, k_off], bar, w_bufs.index(s).slice(q * R, R, dim=0), pred=pred)
            tma.async_load(x_desc, [0, k_off], bar, x_bufs.index(s).slice(q * BT, BT, dim=0), pred=pred)

    @gluon.jit
    def _gluon_gemv_kfold(w_desc, x_desc, out_ptr, M, N: gl.constexpr, K: gl.constexpr,
                          STAGES: gl.constexpr, Q: gl.constexpr, num_warps: gl.constexpr):
        R: gl.constexpr = w_desc.block_type.shape[0]
        BK: gl.constexpr = w_desc.block_type.shape[1]
        BT: gl.constexpr = x_desc.block_type.shape[0]
        BN: gl.constexpr = R * Q
        KI: gl.constexpr = K // (BK * Q)
        mma: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[num_warps, 1],
                                                      instr_shape=[16, Q * BT, 16])
        n0 = gl.program_id(0) * R
        w_bufs = gl.allocate_shared_memory(w_desc.dtype, [STAGES, BN, BK], w_desc.layout)
        x_bufs = gl.allocate_shared_memory(x_desc.dtype, [STAGES, Q * BT, BK], x_desc.layout)
        ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
        for i in gl.static_range(STAGES):
            mbarrier.init(ready.index(i), count=1)
        for i in gl.static_range(STAGES - 1):
            _kfold_issue(w_desc, x_desc, w_bufs, x_bufs, ready.index(i), i, i, n0, i < KI, R, BT, BK, Q)
        acc = gl.zeros([BN, Q * BT], gl.float32, layout=mma)
        for i in range(KI):
            s = i % STAGES
            mbarrier.wait(ready.index(s), (i // STAGES) & 1)
            acc = warpgroup_mma(w_bufs.index(s), x_bufs.index(s).permute((1, 0)), acc, is_async=True)
            acc = warpgroup_mma_wait(1, deps=(acc,))
            j = i + STAGES - 1
            sj = j % STAGES
            _kfold_issue(w_desc, x_desc, w_bufs, x_bufs, ready.index(sj), sj, j, n0, j < KI, R, BT, BK, Q)
        acc = warpgroup_mma_wait(0, deps=(acc,))
        TW: gl.constexpr = 32 if BT >= 32 else BT
        epi: gl.constexpr = gl.BlockedLayout([1, BT // TW], [32 // TW, TW], [num_warps, 1], [1, 0])
        red = gl.allocate_shared_memory(gl.float32, [BN, Q * BT],
                                        gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0]))
        red.store(acc)
        gl.barrier()
        tot = gl.zeros([R, BT], gl.float32, layout=epi)
        for q in gl.static_range(Q):
            tot += red.slice(q * R, R, dim=0).slice(q * BT, BT, dim=1).load(epi)
        offs_n = n0 + gl.arange(0, R, layout=gl.SliceLayout(1, epi))
        offs_t = gl.arange(0, BT, layout=gl.SliceLayout(0, epi))
        gl.store(out_ptr + offs_t[None, :] * N + offs_n[:, None], tot.to(gl.bfloat16),
                 mask=(offs_t < M)[None, :])


def _gluon_fits(t: GTile, n: int, k: int) -> bool:
    if t.q > 1:
        bn, qbt = t.block_n * t.q, t.block_t * t.q
        smem = t.stages * (bn + qbt) * t.block_k * 2 + bn * qbt * 4
        return (t.split == 1 and t.block_t >= t.max_m and n % t.block_n == 0
                and k % (t.block_k * t.q) == 0 and k // (t.block_k * t.q) >= t.stages
                and smem <= GLUON_SMEM)
    smem = t.stages * (t.block_n + t.block_t) * t.block_k * 2
    num_n = n // t.block_n
    ws_ok = (t.split == 1 or (t.split * num_n * t.block_n * t.block_t <= WS_FLOATS
                              and num_n <= COUNTERS))
    return (t.block_t >= t.max_m and n % t.block_n == 0 and k % t.block_k == 0
            and (k // t.block_k) // t.split >= t.stages and smem <= GLUON_SMEM and ws_ok)


def _gluon_wdesc(state: dict, key: str, t: GTile):
    dk = (key, t.block_n, t.block_k)
    if dk not in state["gl_desc"]:
        wl = gl.NVMMASharedLayout.get_default_for([t.block_n, t.block_k], gl.bfloat16)
        state["gl_desc"][dk] = GluonDesc.from_tensor(state[key], [t.block_n, t.block_k], wl)
    return state["gl_desc"][dk]


def _gluon_xlayout(t: GTile):
    return gl.NVMMASharedLayout.get_default_for([t.block_t, t.block_k], gl.bfloat16)


def _check_gluon_tiles(state: dict) -> None:
    state["gl_desc"] = {}
    state["gl_xl"] = {}
    if gluon is None:
        return
    for key, tiles in GLUON_TILES.items():
        n, k = state[key].shape
        for t in tiles:
            if not _gluon_fits(t, n, k):
                raise ValueError(f"{key} gluon tile {t} does not fit n={n}, k={k}")
            _gluon_wdesc(state, key, t)
            state["gl_xl"][(t.block_t, t.block_k)] = _gluon_xlayout(t)


def _gluon_pick(state: dict, key: str, m: int) -> GTile | None:
    if not state.get("gl_desc", None):
        return None
    for t in GLUON_TILES.get(key, ()):
        if t.min_m <= m <= t.max_m:
            return t
    return None


def _gluon_launch(x, out, state: dict, key: str, t: GTile) -> None:
    m = x.shape[0]
    n, k = state[key].shape
    xl = state["gl_xl"].get((t.block_t, t.block_k)) or _gluon_xlayout(t)
    xd = GluonDesc.from_tensor(x, [t.block_t, t.block_k], xl)
    wd = _gluon_wdesc(state, key, t)
    if t.q > 1:
        _gluon_gemv_kfold[(n // t.block_n,)](wd, xd, out, m, N=n, K=k, STAGES=t.stages, Q=t.q, num_warps=4)
    else:
        _gluon_gemv[(n // t.block_n, t.split)](wd, xd, out, state["ws"], state["cnt"], m, N=n, K=k,
                                               STAGES=t.stages, SPLIT=t.split, num_warps=4)


def _check_big_tiles(state: dict) -> None:
    state["big_desc"] = {}
    state["big_out_desc"] = {}
    if TensorDescriptor is None:
        return
    for (layer, m), (bm, bn, bk, split, warps, stages, swap, tma_store) in BIG_TILES.items():
        w = state[layer]
        n, k = w.shape
        state["big_desc"][(layer, m)] = TensorDescriptor.from_tensor(w, [bn, bk])
        if tma_store:
            # Epilogue TMA store requires out_desc with [BM, BN]
            dummy_out = torch.empty(m, n, device=w.device, dtype=torch.bfloat16)
            state["big_out_desc"][(layer, m)] = TensorDescriptor.from_tensor(dummy_out, [bm, bn])


def _router_descs(state: dict, bn: int, bk: int) -> None:
    k = (bn, bk)
    if k in state["router_desc"]:
        return
    state["router_desc"][k] = (
        TensorDescriptor.from_tensor(state["router_hi_t"], [bk, bn]),
        TensorDescriptor.from_tensor(state["router_lo_t"], [bk, bn]),
    )


def _router_big_fits(cfg: tuple[int, ...], hi_m: int, n: int, k: int) -> bool:
    bm, bn, bk, split, _, _, _ = cfg
    rows = triton.cdiv(hi_m, bm) * bm
    return not (n % bn or k % (split * bk) or split * rows * n > WS_FLOATS or (n // bn) * (rows // bm) > COUNTERS)


def _router_big_pick(state: dict, m: int) -> tuple[int, ...] | None:
    if not state.get("router_desc_ok", False):
        return None
    for lo, hi, cfg in ROUTER_BIG:
        if lo <= m <= hi:
            return cfg
    return None


@triton.jit(do_not_specialize=["M"])
def _router_big(x_desc, hi_desc, lo_desc, out_ptr, ws_ptr, cnt_ptr, M,
                N: tl.constexpr, K: tl.constexpr, SPLIT: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                SWAP: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    pid_m = pid % num_m
    pid_s = (pid // num_m) % SPLIT
    pid_n = pid // (num_m * SPLIT)
    KI: tl.constexpr = K // BK

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    m_mask = offs_m < M

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for ki in range(pid_s * KI // SPLIT, (pid_s + 1) * KI // SPLIT):
        xt = x_desc.load([pid_m * BM, ki * BK])
        hi = hi_desc.load([ki * BK, pid_n * BN])
        lo = lo_desc.load([ki * BK, pid_n * BN])
        acc += tl.dot(xt, hi, out_dtype=tl.float32)
        acc += tl.dot(xt, lo, out_dtype=tl.float32)

    idx = offs_m[:, None] * N + offs_n[None, :]
    mask = m_mask[:, None]
    if SPLIT == 1:
        tl.store(out_ptr + idx, acc, mask=mask)
    else:
        TILE_ELEMS: tl.constexpr = BM * BN
        tile_id = pid_n * num_m + pid_m
        ws_tile = ws_ptr + tile_id * (SPLIT * TILE_ELEMS) + tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]
        tl.store(ws_tile + pid_s * TILE_ELEMS, acc)
        tl.debug_barrier()
        cnt = cnt_ptr + tile_id
        done = tl.atomic_add(cnt, 1, sem="acq_rel", scope="gpu")
        if done == SPLIT - 1:
            if SPLIT == 2:
                total = acc + tl.load(ws_tile + (1 - pid_s) * TILE_ELEMS, cache_modifier=".cg")
            else:
                total = tl.zeros((BM, BN), dtype=tl.float32)
                for s in tl.static_range(SPLIT):
                    total += tl.load(ws_tile + s * TILE_ELEMS, cache_modifier=".cg")
            tl.store(out_ptr + idx, total, mask=mask)
            tl.atomic_xchg(cnt, 0)


def _router_big_launch(x: torch.Tensor, state: dict, cfg: tuple[int, ...]) -> torch.Tensor:
    m = x.shape[0]
    bm, bn, bk, split, swap, warps, stages = cfg
    hi_desc, lo_desc = state["router_desc"][(bn, bk)]
    out = torch.empty(m, 256, device=x.device, dtype=torch.float32)
    _router_big[(triton.cdiv(m, bm) * (256 // bn) * split,)](
        TensorDescriptor.from_tensor(x, [bm, bk]), hi_desc, lo_desc, out, state["ws"], state["cnt"],
        m, N=256, K=x.shape[1], SPLIT=split, BM=bm, BN=bn, BK=bk, SWAP=swap,
        num_warps=warps, num_stages=stages)
    return out


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



# ===========================================================================
# 12-bit lossless weights (pooler/unpooler/qkv_gate, m<=32): W packed into mma A-fragment records (sign+mantissa byte + 3-bit exponent
# code + implied exponent LSB), decoded in registers; escapes (exponent outside the 16-binade window) are fixed
# from per-row slots holding the true weights. Bit-exact round trip asserted in prepare; failure keeps bf16.
# ===========================================================================
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
      // esc = [rowptr (N+1) | pad | (k, weight bits) per escape, row-major]; any count per row
      const int* rowptr = reinterpret_cast<const int*>(esc);
      const int2* ent = esc + (N + 2) / 2;
      const int e0 = rowptr[rg * R + row], e1 = rowptr[rg * R + row + 1];
      float f[BT], fx[BT];
#pragma unroll
      for (int tk = 0; tk < BT; ++tk) { f[tk] = 0.f; fx[tk] = 0.f; }
      for (int eb = e0; eb < e1; eb += 4) {
        int2 sl[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) sl[i] = eb + i < e1 ? ent[eb + i] : make_int2(-1, 0);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int kk = sl[e].x;
          const float wv = __int_as_float(sl[e].y);
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
  TORCH_CHECK(wp.numel() == (int64_t)N * K * 3 / 2 && esc.numel() >= (int64_t)N + 2, "z9: packed size");
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
  Z9_CASE(25, 2, 1, 8, 1, 2, 2) Z9_CASE(27, 2, 2, 8, 1, 2, 2) Z9_CASE(28, 2, 2, 8, 1, 1, 2) Z9_CASE(31, 1, 1, 8, 1, 2, 2) 
  TORCH_CHECK(false, "zip: config id not compiled ", id);
}
"""
ZIP_CPP = "void launch_z7(torch::Tensor x, torch::Tensor wp, torch::Tensor out, torch::Tensor esc, double scale, int64_t id, int64_t ilv);"
ZIP_REBASE = 112
# (layer, 16-row blocks per packed record, lo_m, hi_m, kernel id); ids index ZIP_SRC's compiled configs
ZIP_PLAN = (
    ("pooler", 2, 1, 8, 25), ("pooler", 2, 9, 16, 27), ("unpooler", 2, 9, 16, 28), ("qkv_gate", 1, 1, 8, 31),
)
ZIP_DET_REPS = 20
ZIP_MAX_ESC_FRAC = 1e-2


def _build_zip():
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((ZIP_SRC + " ".join(CUDA_FLAGS)).encode()).hexdigest()[:10]
    return load_inline(name=f"muse_zip9_{tag}", cpp_sources=ZIP_CPP, cuda_sources=ZIP_SRC, functions=["launch_z7"],
                       extra_cuda_cflags=CUDA_FLAGS, extra_cflags=["-O3"], extra_ldflags=CUDA_LDFLAGS)


def _zip_pack(w: torch.Tensor, rb: int) -> dict:
    n, k = w.shape
    if w.dtype != torch.bfloat16 or n % (16 * rb) or k % 64:
        raise ValueError(f"zip: cannot pack {tuple(w.shape)} {w.dtype}")
    rg, u = n // (16 * rb), k // 64
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    hist = torch.bincount(((bits >> 8) & 0x7F).flatten(), minlength=128)
    eb = int(hist.unfold(0, 8, 1).sum(1).argmax())
    s, c, low = bits >> 15, (((bits >> 7) & 0xFF) >> 1) - eb, bits & 0xFF
    ok = (c >= 0) & (c < 8)
    s, c, low = (torch.where(ok, v, torch.zeros_like(v)) for v in (s, c, low))
    nib = (s << 3) | c

    def arr(v):  # [N, K] -> [RG, U, RB, p, g, t, c, jl, h, eo] (mma A-fragment lane order)
        return v.reshape(rg, rb, 2, 8, u, 2, 4, 2, 2, 2).permute(0, 4, 1, 8, 3, 6, 5, 7, 2, 9)

    a = arr(low).reshape(rg, u, rb, 2, 32, 16).to(torch.uint8)
    nb = arr(nib).reshape(rg, u, rb, 2, 32, 4, 4)
    nw, nx = nb[:, :, :, 0], nb[:, :, :, 1]
    byte = ((nw & 8) << 4) | (nw & 7) | (nx & 8) | (torch.roll(nx & 7, -1, dims=-1) << 4)
    buf = torch.cat([a, byte.reshape(rg, u, rb, 1, 32, 16).to(torch.uint8)], dim=3).contiguous()
    rows, cols = (~ok).nonzero(as_tuple=True)  # row-major order
    if rows.numel() > ZIP_MAX_ESC_FRAC * w.numel():
        raise ValueError(f"zip: {rows.numel()} escapes")
    off = (n + 2) // 2 * 2
    esc = torch.zeros(off + 2 * rows.numel(), dtype=torch.int32, device=w.device)
    esc[1:n + 1] = torch.cumsum(torch.bincount(rows, minlength=n), 0).to(torch.int32)
    ent = esc[off:].view(-1, 2)
    ent[:, 0] = cols.to(torch.int32)
    ent[:, 1] = w[rows, cols].float().view(torch.int32)
    return {"buf": buf.view(-1), "esc": esc, "scale": 2.0 ** (2 * eb - ZIP_REBASE), "rb": rb, "n": n}


def _zip_escapes(p: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(rows, cols, true fp32 weights) of every escape from the packed CSR."""
    n, esc = p["n"], p["esc"]
    rowptr = esc[: n + 1].long()
    ent = esc[(n + 2) // 2 * 2:].view(-1, 2)
    rows = torch.repeat_interleave(torch.arange(n, device=esc.device), rowptr[1:] - rowptr[:-1])
    return rows, ent[:, 0].long(), ent[:, 1].view(torch.float32)


def _zip_unpack(p: dict, n: int, k: int) -> torch.Tensor:
    """Emulates the kernel's lop3/funnel-shift/prmt decode -> rebased bf16 bit patterns [n, k]."""
    rb = p["rb"]
    by = p["buf"].view(-1, k // 64, rb, 3, 32, 4, 4).to(torch.int64)
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
    val = pair.reshape(-1, k // 64, rb, 8, 4, 2, 2, 2, 2, 2)
    return val.permute(0, 2, 8, 3, 1, 5, 4, 6, 7, 9).reshape(n, k)


def _zip_verify(p: dict, w: torch.Tensor, rows_per_chunk: int = 512) -> None:
    """Bit-exact: decoded*scale == w off the escapes, escapes decode to 2^-15*scale, CSR holds exact weights."""
    n, k = w.shape
    erows, ecols, ew = _zip_escapes(p)
    if erows.numel() != ecols.numel() or not torch.equal(ew, w[erows, ecols].float()):
        raise ValueError("zip: escape entry mismatch")
    rec = 16 * p["rb"]
    chunk = max(rec, rows_per_chunk // rec * rec)
    bytes_per_row = k * 3 // 2
    for r0 in range(0, n, chunk):
        r1 = min(n, r0 + chunk)
        sub = {"buf": p["buf"][r0 * bytes_per_row:r1 * bytes_per_row], "rb": p["rb"]}
        dec = _zip_unpack(sub, r1 - r0, k).to(torch.int32).to(torch.int16).view(torch.bfloat16).float() * p["scale"]
        sel = (erows >= r0) & (erows < r1)
        rows, cols = erows[sel] - r0, ecols[sel]
        hit = torch.zeros(r1 - r0, k, dtype=torch.bool, device=w.device)
        hit[rows, cols] = True
        wf = w[r0:r1].float()
        if not torch.equal(torch.where(hit, wf, dec), wf):
            raise ValueError("zip: decode mismatch")
        if not bool((dec[rows, cols] == 2.0 ** -15 * p["scale"]).all()):
            raise ValueError("zip: escape decode value")


def _zip_check(entry: tuple, w: torch.Tensor, gen: torch.Generator) -> None:
    """Self-test of one (range, config): fp64 accuracy, bit-exact one-hot column readback, run-to-run equality."""
    lo, hi, cid, p = entry
    k = w.shape[1]
    for m in sorted({lo, hi, (lo + hi) // 2}):
        x = torch.randn(m, k, device=w.device, generator=gen).to(torch.bfloat16)
        ref = x.double() @ w.double().t()
        got = _zip_launch(x, w.shape[0], cid, p).double()
        if not bool(torch.isfinite(got).all()) or float((got - ref).norm() / ref.norm()) > 1e-2:
            raise ValueError(f"accuracy at m={m}")
    cols = torch.randperm(k, device=w.device, generator=gen)[:hi]
    _, ecols, _ = _zip_escapes(p)
    if ecols.numel():
        cols[0] = ecols[0]
    xo = torch.zeros(hi, k, device=w.device, dtype=torch.bfloat16)
    xo[torch.arange(hi, device=w.device), cols] = 1.0
    if not torch.equal(_zip_launch(xo, w.shape[0], cid, p), w[:, cols].t()):
        raise ValueError("one-hot readback not bit-exact")
    x = torch.randn(hi, k, device=w.device, generator=gen).to(torch.bfloat16)
    first = _zip_launch(x, w.shape[0], cid, p)
    for _ in range(ZIP_DET_REPS):
        if not torch.equal(_zip_launch(x, w.shape[0], cid, p), first):
            raise ValueError("non-deterministic")


def _zip_prepare(state: dict) -> None:
    import time
    t0 = time.time()
    state["zip"] = {}
    if state.get("ext") is None:
        return
    try:
        ext = _build_zip()
        print(f"zip build {time.time() - t0:.1f}s")
    except Exception as exc:
        print(f"zip path disabled: {type(exc).__name__}: {exc}")
        return
    gen = torch.Generator(device=state["pooler"].device).manual_seed(0)
    packs = {}
    for layer, rb, lo, hi, cid in ZIP_PLAN:
        try:
            w = state[layer]
            if (layer, rb) not in packs:
                packs[layer, rb] = None
                p = _zip_pack(w, rb)
                _zip_verify(p, w)
                p["ext"] = ext
                packs[layer, rb] = p
            p = packs[layer, rb]
            if p is None:
                raise ValueError("packing failed earlier")
            entry = (lo, hi, cid, p)
            _zip_check(entry, w, gen)
            state["zip"].setdefault(layer, []).append(entry)
            print(f"zip path enabled: {layer} m {lo}-{hi} id {cid}")
        except Exception as exc:
            print(f"zip path disabled for {layer} m {lo}-{hi}: {type(exc).__name__}: {exc}")
    torch.cuda.synchronize()
    print(f"zip prepare {time.time() - t0:.1f}s")


def _zip_launch(x: torch.Tensor, n: int, cid: int, p: dict) -> torch.Tensor:
    out = torch.empty(x.shape[0], n, device=x.device, dtype=torch.bfloat16)
    p["ext"].launch_z7(x, p["buf"], out, p["esc"], p["scale"], cid, 0)
    return out


def _zip_try(x: torch.Tensor, layer: str, n: int, state: dict):
    entries = state["zip"].get(layer)
    if not entries or n != state[layer].shape[0] or x.shape[1] != state[layer].shape[1]:
        return None
    m = x.shape[0]
    for lo, hi, cid, p in entries:
        if lo <= m <= hi:
            return _zip_launch(x, n, cid, p)
    return None


# ===== r9: m=128/256 cluster split-K DSMEM-reduced wgmma GEMM =====
def _r9_gen_wgmma() -> str:
    parts = ["template <int BT>\n__device__ __forceinline__ void r9_mma(float (&d)[BT / 8][4], uint64_t da, uint64_t db);\n"]
    for n in (64, 128, 256):
        nt = n // 8
        regs = ", ".join(f"%{i}" for i in range(n // 2))
        outs = ", ".join(f'"+f"(d[{j}][{r}])' for j in range(nt) for r in range(4))
        parts.append(
            "template <>\n"
            f"__device__ __forceinline__ void r9_mma<{n}>(float (&d)[{nt}][4], uint64_t da, uint64_t db) {{\n"
            f'  asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 "\n'
            f'      "{{{regs}}}, %{n // 2}, %{n // 2 + 1}, 1, 1, 1, 0, 0;\\n"\n'
            f"      : {outs}\n"
            '      : "l"(da), "l"(db));\n'
            "}\n")
    return "".join(parts)


R9_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda.h>

namespace cg = cooperative_groups;
using bf16 = __nv_bfloat16;

__device__ __forceinline__ uint32_t r9_su(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void r9_cl_arrive_release() { asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void r9_cl_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void r9_mbar_init(uint64_t* b, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(r9_su(b)), "r"(count) : "memory");
}
__device__ __forceinline__ void r9_expect_tx(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(r9_su(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void r9_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(r9_su(b)) : "memory");
}
__device__ __forceinline__ void r9_wait(uint64_t* b, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred P1;\n R9_WAIT:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      " @P1 bra R9_DONE;\n bra R9_WAIT;\n R9_DONE:\n}\n"
      ::"r"(r9_su(b)), "r"(parity) : "memory");
}
__device__ __forceinline__ void r9_tma(void* dst, const CUtensorMap* map, int c0, int c1, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
      ::"r"(r9_su(dst)), "l"(map), "r"(c0), "r"(c1), "r"(r9_su(bar)) : "memory");
}
// K-major operand, 128B swizzle, 8-row groups 1024 B apart
__device__ __forceinline__ uint64_t r9_desc(const void* p) {
  return (uint64_t)((r9_su(p) >> 4) & 0x3FFF) | (1ull << 16) | (64ull << 32) | (1ull << 62);
}
__device__ __forceinline__ void r9_wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void r9_wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void r9_wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"); }
""" + _r9_gen_wgmma() + r"""
template <int NWG, int BT, int S>
struct R9Geo {
  static constexpr int BN = 64 * NWG;
  static constexpr int NT = BT / 8;
  static constexpr int NCT = NWG * 128;
  static constexpr int THREADS = NCT + 32;
  static constexpr int JPR = (NT + S - 1) / S;      // max 8-token column blocks owned per rank
  static constexpr uint32_t WST = BN * 128;         // W stage bytes (BN rows x 64 k)
  static constexpr uint32_t XST = BT * 128;         // x stage bytes (BT tokens x 64 k)
  static constexpr uint32_t STB = WST + XST;
  static constexpr uint32_t SLOT = JPR * NCT * 16;  // one peer's partials, fragment order (float4 per thread per j)
  static constexpr uint32_t RECV = (S - 1) * SLOT;
  static constexpr int OPITCH = BN + 8;             // bf16 staging row pitch: conflict-free, 16B rows
  static constexpr uint32_t OST = 8 * JPR * OPITCH * 2;
  __device__ static constexpr int owner(int j) { return (j * S) / NT; }
  __device__ static constexpr int j0(int q) { return (q * NT + S - 1) / S; }
};

template <int NWG, int BT, int S, int D>
__global__ void __launch_bounds__(NWG * 128 + 32, 1)
r9_kernel(const __grid_constant__ CUtensorMap wmap, const __grid_constant__ CUtensorMap xmap,
          bf16* __restrict__ out, int m, int N, int K, int TT) {
  using G = R9Geo<NWG, BT, S>;
  static_assert(G::RECV + G::OST <= D * G::STB, "r9: reduce buffers exceed the stage ring");
  static_assert(G::NT >= S, "r9: every rank must own a token stripe");
  constexpr uint32_t BOFF = D * G::STB;
  extern __shared__ __align__(1024) unsigned char r9_raw[];
  unsigned char* sm = r9_raw + ((1024 - (r9_su(r9_raw) & 1023)) & 1023);
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + BOFF);
  uint64_t* empty = full + D;

  int rank = 0;
  if constexpr (S > 1) rank = (int)cg::this_cluster().block_rank();
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int c = blockIdx.x / S;
  const int tt = c % TT, tile = c / TT;
  const int n0 = tile * G::BN, t0 = tt * BT;
  const int U = K >> 6, u0 = rank * U / S, nk = (rank + 1) * U / S - u0;

  if (tid == 0) {
    for (int d = 0; d < D; ++d) { r9_mbar_init(&full[d], 1); r9_mbar_init(&empty[d], NWG * 4); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();

  float acc[G::NT][4];
#pragma unroll
  for (int j = 0; j < G::NT; ++j)
#pragma unroll
    for (int r = 0; r < 4; ++r) acc[j][r] = 0.f;

  if (warp == NWG * 4) {
    if (lane == 0) {
      for (int i = 0; i < nk; ++i) {
        const int s = i % D;
        if (i >= D) {
          r9_wait(&empty[s], ((i / D) - 1) & 1);
          asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        }
        unsigned char* st = sm + s * G::STB;
        r9_expect_tx(&full[s], G::STB);  // x rows >= m are zero-filled and still counted
        r9_tma(st, &wmap, (u0 + i) * 64, n0, &full[s]);
        r9_tma(st + G::WST, &xmap, (u0 + i) * 64, t0, &full[s]);
      }
    }
    __syncwarp();
  } else {
    const int wg = warp >> 2;
    for (int i = 0; i < nk; ++i) {
      const int s = i % D;
      r9_wait(&full[s], (i / D) & 1);
      const unsigned char* st = sm + s * G::STB;
      const uint64_t da = r9_desc(st + wg * 8192);
      const uint64_t db = r9_desc(st + G::WST);
      r9_wg_fence();
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) r9_mma<BT>(acc, da + (uint64_t)(2 * ks), db + (uint64_t)(2 * ks));
      r9_wg_commit();
      r9_wg_wait<1>();
      if (i > 0 && lane == 0) r9_arrive(&empty[(i - 1) % D]);
    }
    r9_wg_wait<0>();
  }
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  // every CTA of the cluster is past its mainloop: the stage ring is free for reduce slots everywhere
  if constexpr (S > 1) {
    r9_cl_arrive_release();
    r9_cl_wait();
  } else {
    __syncthreads();
  }

  float4* recv = reinterpret_cast<float4*>(sm);
  bf16* ost = reinterpret_cast<bf16*>(sm + G::RECV);
  if constexpr (S > 1) {
    if (warp < NWG * 4) {
      cg::cluster_group cluster = cg::this_cluster();
#pragma unroll
      for (int j = 0; j < G::NT; ++j) {
        const int q = G::owner(j);
        if (q != rank) {
          const int slot = rank < q ? rank : rank - 1;
          float4* dst = cluster.map_shared_rank(recv + (slot * G::JPR + (j - G::j0(q))) * G::NCT + tid, q);
          *dst = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);
        }
      }
    }
    r9_cl_arrive_release();
    r9_cl_wait();
  }

  if (warp < NWG * 4) {
    const int wg = warp >> 2, wq = warp & 3, g = lane >> 2, t4 = lane & 3;
    const int nl = wg * 64 + wq * 16 + g;
    const int jb = G::j0(rank);
#pragma unroll
    for (int j = 0; j < G::NT; ++j) {
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
        ost[tl * G::OPITCH + nl] = __float2bfloat16(v.x);
        ost[(tl + 1) * G::OPITCH + nl] = __float2bfloat16(v.y);
        ost[tl * G::OPITCH + nl + 8] = __float2bfloat16(v.z);
        ost[(tl + 1) * G::OPITCH + nl + 8] = __float2bfloat16(v.w);
      }
    }
    asm volatile("bar.sync 1, %0;\n" ::"n"(G::NCT) : "memory");
    constexpr int V = G::BN / 8;
    const int ntok = 8 * (G::j0(rank + 1) - jb);
    for (int e = tid; e < ntok * V; e += G::NCT) {
      const int tl = e / V, cv = e % V;
      const int t = t0 + 8 * jb + tl;
      if (t < m) {
        *reinterpret_cast<uint4*>(out + (size_t)t * N + n0 + cv * 8) =
            *reinterpret_cast<const uint4*>(ost + tl * G::OPITCH + cv * 8);
      }
    }
  }
}

inline CUtensorMap r9_map(const void* ptr, int rows, int cols, int row_stride, int box_rows) {
  CUtensorMap map{};
  cuuint64_t gdim[2] = {(cuuint64_t)cols, (cuuint64_t)rows};
  cuuint64_t gstride[1] = {(cuuint64_t)row_stride * sizeof(bf16)};
  cuuint32_t bdim[2] = {64u, (cuuint32_t)box_rows};
  cuuint32_t estride[2] = {1u, 1u};
  CUresult res = cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), gdim, gstride,
                                        bdim, estride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(res == CUDA_SUCCESS, "r9: cuTensorMapEncodeTiled failed ", (int)res);
  return map;
}

template <int NWG, int BT, int S, int D>
constexpr size_t r9_smem() { return 1024 + (size_t)D * R9Geo<NWG, BT, S>::STB + (size_t)2 * D * 8; }

template <int NWG, int BT, int S, int D>
cudaLaunchConfig_t r9_cfg(int blocks, cudaLaunchAttribute* attr) {
  using G = R9Geo<NWG, BT, S>;
  constexpr size_t smem = r9_smem<NWG, BT, S, D>();
  static_assert(smem <= 232448, "r9: smem");
  auto kern = r9_kernel<NWG, BT, S, D>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(kern),
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  if (S > 8) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(kern),
                                        cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(blocks, 1, 1);
  cfg.blockDim = dim3(G::THREADS, 1, 1);
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

template <int NWG, int BT, int S, int D>
void r9_launch(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& out) {
  using G = R9Geo<NWG, BT, S>;
  const int m = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(m >= 1 && N % G::BN == 0 && K % 64 == 0 && K / 64 >= S && w.size(1) == K, "r9: shape");
  TORCH_CHECK(out.size(0) == m && out.size(1) == N && out.is_contiguous() && w.is_contiguous(), "r9: out/w layout");
  TORCH_CHECK(x.stride(1) == 1 && x.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "r9: x layout");
  const int TT = (m + BT - 1) / BT;
  const CUtensorMap wmap = r9_map(w.data_ptr(), N, K, K, G::BN);
  const CUtensorMap xmap = r9_map(x.data_ptr(), m, K, (int)x.stride(0), BT);
  cudaLaunchAttribute attr;
  cudaLaunchConfig_t cfg = r9_cfg<NWG, BT, S, D>((N / G::BN) * TT * S, &attr);
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, r9_kernel<NWG, BT, S, D>, wmap, xmap,
                                    static_cast<bf16*>(out.data_ptr()), m, N, K, TT));
}

template <int NWG, int BT, int S, int D>
int64_t r9_clusters() {
  cudaLaunchAttribute attr;
  cudaLaunchConfig_t cfg = r9_cfg<NWG, BT, S, D>(132 * S, &attr);
  int n = -1;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&n, reinterpret_cast<const void*>(r9_kernel<NWG, BT, S, D>), &cfg));
  return n;
}

#define R9_CASE(ID, NWG_, BT_, S_, D_)                    \
  if (id == ID) {                                         \
    if (query) return r9_clusters<NWG_, BT_, S_, D_>();   \
    r9_launch<NWG_, BT_, S_, D_>(x, w, out);              \
    return 0;                                             \
  }

int64_t launch_r9(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t id, int64_t query) {
R9_CASES
  TORCH_CHECK(false, "r9: config id not compiled ", id);
  return 0;
}
"""
# id -> (NWG, BT, S, D): BN = 64 NWG W rows per CTA, BT tokens, S-way cluster split-K, D-stage ring
R9_CFGS = {
    1: (2, 128, 5, 6),
    2: (3, 128, 8, 5),
    3: (1, 128, 3, 8),
    4: (2, 128, 3, 6),
    5: (2, 256, 5, 4),
    6: (2, 128, 2, 6),
    7: (2, 256, 2, 4),
    8: (2, 64, 2, 8),
    9: (2, 128, 4, 6),
    10: (2, 128, 1, 6),
}
R9_SRC = R9_SRC.replace("R9_CASES", "\n".join(f"  R9_CASE({i}, {a}, {b}, {c}, {d})" for i, (a, b, c, d) in R9_CFGS.items()))
R9_CPP = "int64_t launch_r9(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t id, int64_t query);"
# (layer, m) -> candidate config ids, timed in prepare against the cp8 path; fastest wins if it beats cp8
R9_PLAN = {
    ("qkv_gate", 128): (1, 2, 3, 9),
    ("qkv_gate", 256): (4, 5),
    ("pooler", 128): (4, 8),
    ("pooler", 256): (6,),
    ("unpooler", 128): (6,),
    ("unpooler", 256): (7, 10),
    ("o", 128): (6,),
    ("o", 256): (7, 10),
}
R9_DET_REPS = 10
R9_BENCH_REPS = 30
R9_MARGIN_US = 0.2


def _r9_build():
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((R9_SRC + " ".join(CUDA_FLAGS)).encode()).hexdigest()[:10]
    return load_inline(name=f"muse_r9_{tag}", cpp_sources=R9_CPP, cuda_sources=R9_SRC, functions=["launch_r9"],
                       extra_cuda_cflags=CUDA_FLAGS, extra_cflags=["-O3"], extra_ldflags=CUDA_LDFLAGS)


def _r9_run(ext, x: torch.Tensor, w: torch.Tensor, cid: int) -> torch.Tensor:
    out = torch.empty(x.shape[0], w.shape[0], device=x.device, dtype=torch.bfloat16)
    ext.launch_r9(x, w, out, cid, 0)
    return out


def _r9_fits(cid: int, n: int, k: int, m: int, max_clusters: int) -> str | None:
    nwg, bt, s, _ = R9_CFGS[cid]
    if n % (64 * nwg) or k % 64 or k // 64 < s:
        return "shape"
    need = (n // (64 * nwg)) * ((m + bt - 1) // bt)
    if need > max_clusters:
        return f"needs {need} clusters > {max_clusters} co-resident"
    return None


def _r9_selftest(ext, w: torch.Tensor, m: int, cid: int, gen: torch.Generator) -> None:
    k = w.shape[1]
    x = torch.randn(m, k, device=w.device, generator=gen).to(torch.bfloat16)
    x[:, ::97] *= 30.0
    ref = x.double() @ w.double().t()
    got = _r9_run(ext, x, w, cid)
    gd = got.double()
    rel = float((gd - ref).norm() / ref.norm())
    mx = float((gd - ref).abs().max() / ref.abs().max())
    if not bool(torch.isfinite(gd).all()) or rel > 4e-3 or mx > 2e-2:
        raise ValueError(f"smoke rel={rel:.2e} max={mx:.2e}")
    for _ in range(R9_DET_REPS):
        if not torch.equal(_r9_run(ext, x, w, cid), got):
            raise ValueError("non-deterministic")


def _r9_time(graphs: dict, flush: torch.Tensor) -> dict:
    import statistics
    res = {nm: [] for nm in graphs}
    for r in range(R9_BENCH_REPS):
        for nm in (list(graphs) if r % 2 == 0 else list(graphs)[::-1]):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            flush.sum()
            a.record()
            graphs[nm].replay()
            b.record()
            b.synchronize()
            res[nm].append(a.elapsed_time(b) * 1e3)
    return {nm: statistics.median(v) for nm, v in res.items()}


def _r9_graph(fn):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def _r9_prepare(state: dict) -> None:
    import time
    t_start = time.time()
    state["r9"] = {}
    if state.get("ext") is None:
        return
    try:
        ext = _r9_build()
        print(f"r9 build {time.time() - t_start:.1f}s")
    except Exception as exc:
        print(f"r9 path disabled: {type(exc).__name__}: {str(exc)[-3000:]}")
        return
    clusters = {}
    for cid in sorted({c for ids in R9_PLAN.values() for c in ids}):
        try:
            clusters[cid] = int(ext.launch_r9(torch.empty(0), torch.empty(0), torch.empty(0), cid, 1))
        except Exception as exc:
            print(f"r9 id={cid} occupancy query failed: {type(exc).__name__}: {exc}")
            clusters[cid] = 0
    print("r9 max clusters: " + " ".join(f"id{c}{R9_CFGS[c]}={v}" for c, v in clusters.items()))
    gen = torch.Generator(device=state["pooler"].device).manual_seed(9)
    flush = torch.empty(64 * 2**20, device="cuda", dtype=torch.float32)
    flush.normal_()
    picks = {}
    for (layer, m), ids in R9_PLAN.items():
        w = state[layer]
        n, k = w.shape
        good = []
        for cid in ids:
            why = _r9_fits(cid, n, k, m, clusters.get(cid, 0))
            if why:
                print(f"r9 skip {layer} m={m} id={cid}: {why}")
                continue
            try:
                _r9_selftest(ext, w, m, cid, gen)
                good.append(cid)
            except Exception as exc:
                print(f"r9 selftest FAILED {layer} m={m} id={cid}: {type(exc).__name__}: {str(exc)[-1500:]}")
        if not good:
            continue
        try:
            x = torch.randn(m, k, device=w.device, generator=gen).to(torch.bfloat16)
            data = {"x": x, "config": {"m": m, "n": n, "k": k, "layer": layer}}
            graphs = {"cp8": _r9_graph(lambda: kernel_fn(data, state))}
            for cid in good:
                graphs[f"id{cid}"] = _r9_graph(lambda c=cid: _r9_run(ext, x, w, c))
            med = _r9_time(graphs, flush)
            best = min(good, key=lambda c: med[f"id{c}"])
            win = med[f"id{best}"] < med["cp8"] - R9_MARGIN_US
            print(f"r9 bench {layer} m={m} (raw us, ~4 graph overhead): "
                  + " ".join(f"{nm}={v:.2f}" for nm, v in med.items()) + (f" -> id{best}" if win else " -> cp8"))
            if win:
                picks[(layer, m)] = best
        except Exception as exc:
            print(f"r9 bench error {layer} m={m}: {type(exc).__name__}: {exc}")
    state["r9_ext"] = ext
    state["r9"] = picks
    torch.cuda.synchronize()
    print(f"r9 prepare {time.time() - t_start:.1f}s picks={picks}")


def _r9_try(x: torch.Tensor, layer: str, n: int, state: dict):
    cid = state.get("r9", {}).get((layer, x.shape[0]))
    w = state[layer]
    if cid is None or n != w.shape[0] or x.shape[1] != w.shape[1]:
        return None
    return _r9_run(state["r9_ext"], x, w, cid)


def prepare(weights: Weights) -> dict:
    router = weights["router"].float()
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state = {name: w.contiguous() for name, w in weights.items()}
    try:
        state["ext"] = _build_cuda()
    except Exception as exc:
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
    state["router_desc"] = {}
    state["router_desc_ok"] = TensorDescriptor is not None
    if state["router_desc_ok"]:
        for _, hi_m, cfg in ROUTER_BIG:
            if not _router_big_fits(cfg, hi_m, *router.shape):
                raise ValueError(f"router big tile {cfg} does not fit up to m={hi_m}")
            _router_descs(state, cfg[1], cfg[2])
    n, k = router.shape
    g = GEMV_FP32
    if n % g.block_n or k % (g.split * g.block_k) or g.split * n > WS_FLOATS or n // g.block_n > COUNTERS:
        raise ValueError(f"router GEMV tile {g} does not fit n={n}, k={k}")
    for m in WARMUP_M:
        x = torch.zeros(m, router.shape[1], device=device, dtype=torch.bfloat16)
        kernel_fn({"x": x, "config": {"m": m, "n": 256, "k": router.shape[1], "layer": "router"}}, state)
    state["unpooler_slice"] = state["unpooler"][:UNPOOL_SLICE]
    state["desc"] = {}
    _check_bf16_tiles(state)
    _check_big_tiles(state)
    _check_gluon_tiles(state)
    for layer in ("qkv_gate", "o", "pooler", "unpooler"):
        w_rows, k = state[layer].shape
        for m in BF16_WARMUP_M:
            n = UNPOOL_SLICE if layer == "unpooler" and m <= 8 else w_rows
            x = torch.zeros(m, k, device=device, dtype=torch.bfloat16)
            kernel_fn({"x": x, "config": {"m": m, "n": n, "k": k, "layer": layer}}, state)
    _zip_prepare(state)
    _r9_prepare(state)
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
    cfg = _router_big_pick(state, m)
    if cfg is not None:
        return _router_big_launch(x, state, cfg)
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
    x, cfg = data["x"], data["config"]
    if x.stride(1) != 1:
        x = x.contiguous()
    if x.data_ptr() % 16 or x.stride(0) % 8:
        x = x.clone(memory_format=torch.contiguous_format)
    if cfg["layer"] == "router":
        return _router(x, state)
    out = _r9_try(x, cfg["layer"], cfg["n"], state)
    if out is not None:
        return out
    if cfg["layer"] in state.get("zip", {}):
        out = _zip_try(x, cfg["layer"], cfg["n"], state)
        if out is not None:
            return out
    w_full = state[cfg["layer"]]
    key = _bf16_key(cfg["layer"], cfg["n"], w_full.shape[0])
    w = state[key] if cfg["n"] == state[key].shape[0] else w_full[: cfg["n"]]
    pick = _cuda_pick(state, key, x.shape[0]) if w is state[key] else None
    if pick is not None:
        return _cuda_run(state, x, w, w.shape[0], torch.bfloat16, *pick)
    gtile = _gluon_pick(state, key, x.shape[0]) if w is state[key] else None
    if gtile is not None:
        out = torch.empty(x.shape[0], w.shape[0], device=x.device, dtype=torch.bfloat16)
        _gluon_launch(x, out, state, key, gtile)
        return out
    tile = _bf16_pick(key, x.shape[0]) if w is state[key] else None
    if tile is None and w is state[key] and (key, x.shape[0]) in state.get("big_desc", {}):
        return _big_launch(x, w, state, key)
    if tile is None:
        return torch.nn.functional.linear(x, w)
    out = torch.empty(x.shape[0], w.shape[0], device=x.device, dtype=torch.bfloat16)
    _bf16_launch(x, w, out, state, key, tile)
    return out
