# okc: problem=muse_decode_layer
"""Fully optimized decoder layer for Muse Spark Open on H200 (TP8/EP8):
- Ultra-fast sm_90a TMA + WGMMA + Cluster DSMEM projections from muse_bf16_proj
- Pipelined m=1 Triton BF16 and FP32 GEMV kernels
- Fused QK-norm + RoPE + FP8 quantize + KV-cache write kernel
- Fast single-pass Triton pre-expert RMSNorm kernel
- Tuned Marlin MoE with preallocated scratch and alignment buffers
- Direct fused post-expert norm weighted sum kernel
- Zero dynamic allocations inside kernel_fn (all buffers preallocated)
"""

from typing import NamedTuple, TypedDict
import hashlib
import types

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sgl_kernel.scalar_type import scalar_types
from torch import nn

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.moe.moe_wna16_marlin import (
    _jit_moe_wna16_marlin_module,
)
try:
    from sglang.kernels.ops.moe.moe_wna16_marlin import _MAX_THREAD_N
except ImportError:
    _MAX_THREAD_N = 128

from sglang.kernels.ops.moe import moe_align_block_size as sgl_moe_align_block_size
from sglang.kernels.ops.moe.moe_align_small_numel import (
    SMALL_NUMEL_LIMIT,
    moe_align_small_numel,
)
from sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing import FusedMuseSparkRouting
from sglang.kernels.ops.muse_spark_v12.fused_norm_gate import FusedRMSNormGate
from sglang.kernels.ops.muse_spark_v12.fused_norm_residual import FusedPostNormResidualPreNorm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    _post_expert_norm_weighted_sum_kernel,
)
from sglang.srt.layers.muse_spark_layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl

try:
    from triton.tools.tensor_descriptor import TensorDescriptor
except ImportError:
    TensorDescriptor = None

gluon = None


class Weights(TypedDict):
    qkv_gate: torch.Tensor
    o: torch.Tensor
    attn_residual_gate: torch.Tensor
    pre_ffn_norm: torch.Tensor
    router: torch.Tensor
    router_bias: torch.Tensor
    pooler: torch.Tensor
    pre_expert_norm: torch.Tensor
    w13: torch.Tensor
    w13_sf: torch.Tensor
    w13_gs: torch.Tensor
    w2: torch.Tensor
    w2_sf: torch.Tensor
    w2_gs: torch.Tensor
    post_expert_norm: torch.Tensor
    unpooler: torch.Tensor
    post_ffn_norm: torch.Tensor
    ffn_residual_gate: torch.Tensor
    next_pre_attn_norm: torch.Tensor


class Config(TypedDict):
    batch: int
    q_len: int
    tokens: int
    hidden: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    window: int
    rope_theta: float
    qk_scale_factor: float
    sm_scale: float
    kv_scale: float
    n_experts: int
    top_k: int
    local_experts: int


class Inputs(TypedDict):
    stream: torch.Tensor
    attn_input: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    positions: torch.Tensor
    cache_loc: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    scheduler_metadata: torch.Tensor | None
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    local_start: torch.Tensor
    config: Config


HIDDEN = 8192
RMS_EPS = 1e-5
POST_NORM_EPS = 1e-8
FP8 = torch.float8_e4m3fn
N_EXPERTS = 256
TOP_K = 8
LOCAL_EXPERTS = 32
D_EXPERT = 4096
ROUTER_FP32_MAX_TOKENS = 8
SLICED_UNPOOL_MAX_TOKENS = 8
UNPOOL_SLICE = HIDDEN // 8
ROPE_POSITIONS = 1 << 21
ROPE_THETA = 500_000.0
MAX_TOKENS = 512
WS_FLOATS = 4 * 2**20
COUNTERS = 8192
GLUON_SMEM = 220 * 1024

BLOCK_SIZE_M_MAP = {
    1: 8,
    4: 8,
    8: 8,
    16: 16,
    32: 8,
    64: 16,
    128: 32,
    256: 32,
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
constexpr int MPAD = 68;
constexpr int THREADS = 128;

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
  TORCH_CHECK(false, "wgmma: config not compiled");
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

CUDA_TILES = {
    "router": (
        (1, 1, "gemv", (1, 1, 8)),
        (8, 8, "wgmma", (32, 8, 128, 8, 8)),
        (16, 16, "wgmma", (32, 16, 128, 8, 8)),
        (32, 32, "wgmma", (32, 32, 128, 8, 8)),
        (128, 128, "wgmma", (32, 128, 128, 4, 8)),
        (256, 256, "wgmma", (32, 128, 128, 4, 4)),
    ),
    "unpooler_slice": (
        (1, 1, "gemv", (2, 1, 2)),
        (2, 2, "gemv", (2, 2, 2)),
        (3, 4, "gemv", (4, 4, 2)),
        (8, 8, "wgmma", (64, 8, 128, 8, 4)),
    ),
    "qkv_gate": (
        (8, 8, "wgmma", (64, 8, 128, 8, 4)),
        (16, 16, "wgmma", (64, 16, 128, 4, 4)),
        (32, 32, "wgmma", (64, 32, 128, 4, 4)),
    ),
    "o": (
        (8, 8, "wgmma", (64, 8, 128, 8, 4)),
        (16, 16, "wgmma", (64, 16, 128, 4, 4)),
        (32, 32, "wgmma", (64, 32, 128, 4, 4)),
    ),
    "pooler": (
        (8, 8, "wgmma", (64, 8, 128, 8, 4)),
        (16, 16, "wgmma", (64, 16, 128, 4, 4)),
        (32, 32, "wgmma", (64, 32, 128, 4, 4)),
        (128, 128, "wgmma", (64, 128, 64, 6, 2)),
    ),
    "unpooler": (
        (16, 16, "wgmma", (64, 16, 128, 4, 4)),
        (32, 32, "wgmma", (64, 32, 128, 4, 4)),
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


# ===========================================================================
# Triton Kernels & Dispatch
# ===========================================================================

class Tile(NamedTuple):
    max_m: int
    block_n: int
    block_k: int
    block_t: int
    split: int
    warps: int
    stages: int
    kind: str = "dot"


BF16_TILES = {
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


GLUON_TILES = {
    "pooler": (GTile(2, 8, 64, 128, 8, 8), GTile(9, 16, 64, 128, 16, 8), GTile(17, 32, 64, 128, 32, 8)),
    "unpooler": (GTile(9, 16, 64, 256, 16, 4), GTile(17, 32, 64, 128, 32, 8)),
    "qkv_gate": (GTile(2, 8, 32, 256, 8, 4, q=2), GTile(9, 16, 32, 256, 16, 4, q=2),
                 GTile(17, 32, 64, 256, 32, 4, split=3)),
    "o": (GTile(2, 8, 64, 256, 8, 4), GTile(9, 16, 64, 128, 16, 4), GTile(17, 32, 64, 128, 32, 4)),
}

BIG_TILES = {
    ("qkv_gate", 128): (128, 64, 64, 3, 4, 8, 0, 0),
    ("pooler", 128): (64, 128, 64, 2, 4, 8, 1, 0),
    ("pooler", 256): (64, 128, 64, 1, 4, 8, 0, 0),
}


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
                total = gl.zeros([BN, BT], gl.float32, layout=mma)
                for s in gl.static_range(SPLIT):
                    total += gl.load(ws_base + s * (BN * BT))
                gl.store(out_ptrs, total.to(gl.bfloat16), mask=mask)
                gl.atomic_xchg(cnt_ptr + pid_n, 0)

    @gluon.jit
    def _gluon_gemv_kfold(w_desc, x_desc, out_ptr, M, N: gl.constexpr, K: gl.constexpr,
                          STAGES: gl.constexpr, Q: gl.constexpr, num_warps: gl.constexpr):
        BN: gl.constexpr = w_desc.block_type.shape[0]
        BK: gl.constexpr = w_desc.block_type.shape[1]
        BT: gl.constexpr = x_desc.block_type.shape[0]
        KI: gl.constexpr = K // BK
        mma: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[num_warps, 1],
                                                      instr_shape=[16, BT, 16])
        pid_n = gl.program_id(0)
        n0 = pid_n * BN
        w_bufs = gl.allocate_shared_memory(w_desc.dtype, [STAGES] + w_desc.block_type.shape, w_desc.layout)
        x_bufs = gl.allocate_shared_memory(x_desc.dtype, [STAGES] + x_desc.block_type.shape, x_desc.layout)
        ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
        for i in gl.static_range(STAGES):
            mbarrier.init(ready.index(i), count=1)
        nbytes: gl.constexpr = w_desc.block_type.nbytes + x_desc.block_type.nbytes
        for i in gl.static_range(STAGES - 1):
            mbarrier.expect(ready.index(i), nbytes)
            tma.async_load(w_desc, [n0, i * BK], ready.index(i), w_bufs.index(i))
            tma.async_load(x_desc, [0, i * BK], ready.index(i), x_bufs.index(i))
        acc = gl.zeros([BN, BT], gl.float32, layout=mma)
        for i in range(KI):
            s = i % STAGES
            mbarrier.wait(ready.index(s), (i // STAGES) & 1)
            acc = warpgroup_mma(w_bufs.index(s), x_bufs.index(s).permute((1, 0)), acc, is_async=True)
            acc = warpgroup_mma_wait(1, deps=(acc,))
            j = i + STAGES - 1
            sj = j % STAGES
            pred = j < KI
            mbarrier.expect(ready.index(sj), nbytes, pred=pred)
            tma.async_load(w_desc, [n0, j * BK], ready.index(sj), w_bufs.index(sj), pred=pred)
            tma.async_load(x_desc, [0, j * BK], ready.index(sj), x_bufs.index(sj), pred=pred)
        acc = warpgroup_mma_wait(0, deps=(acc,))
        offs_nl = gl.arange(0, BN, layout=gl.SliceLayout(1, mma))
        offs_t = gl.arange(0, BT, layout=gl.SliceLayout(0, mma))
        out_ptrs = out_ptr + offs_t[None, :] * N + (n0 + offs_nl)[:, None]
        mask = (offs_t < M)[None, :]
        gl.store(out_ptrs, acc.to(gl.bfloat16), mask=mask)


def _cuda_pick(state: dict, key: str, m: int):
    if state.get("ext") is None:
        return None
    for lo, hi, kind, params in CUDA_TILES.get(key, ()):
        if lo <= m <= hi:
            return kind, params
    return None


def _cuda_run(state: dict, x, w, n: int, dtype, kind: str, params: tuple[int, ...], out_buf=None) -> torch.Tensor:
    out = out_buf if out_buf is not None else torch.empty(x.shape[0], n, device=x.device, dtype=dtype)
    if kind == "gemv":
        state["ext"].launch_gemv(x, w, out, *params)
    elif kind == "wgmma":
        state["ext"].launch_wgmma(x, w, out, *params)
    return out


def _gluon_wdesc(state: dict, key: str, t: GTile):
    dk = (key, t.block_n, t.block_k)
    if dk not in state["gl_desc"]:
        wl = gl.NVMMASharedLayout.get_default_for([t.block_n, t.block_k], gl.bfloat16)
        state["gl_desc"][dk] = GluonDesc.from_tensor(state["dense"][key], [t.block_n, t.block_k], wl)
    return state["gl_desc"][dk]


def _gluon_xlayout(t: GTile):
    return gl.NVMMASharedLayout.get_default_for([t.block_t, t.block_k], gl.bfloat16)


def _gluon_pick(state: dict, key: str, m: int) -> GTile | None:
    if not state.get("gl_desc", None):
        return None
    for t in GLUON_TILES.get(key, ()):
        if t.min_m <= m <= t.max_m:
            return t
    return None


def _gluon_launch(x, out, state: dict, key: str, t: GTile) -> None:
    m = x.shape[0]
    n, k = state["dense"][key].shape
    xl = state["gl_xl"].get((t.block_t, t.block_k)) or _gluon_xlayout(t)
    xd = GluonDesc.from_tensor(x, [t.block_t, t.block_k], xl)
    wd = _gluon_wdesc(state, key, t)
    if t.q > 1:
        _gluon_gemv_kfold[(n // t.block_n,)](wd, xd, out, m, N=n, K=k, STAGES=t.stages, Q=t.q, num_warps=4)
    else:
        _gluon_gemv[(n // t.block_n, t.split)](wd, xd, out, state["ws"], state["cnt"], m, N=n, K=k,
                                               STAGES=t.stages, SPLIT=t.split, num_warps=4)


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
    desc = state["desc"].get((key, tile))
    if desc is None:
        desc = w
    _skinny_gemm[(n // tile.block_n, tile.split, triton.cdiv(m, tile.block_t))](
        x, w, out, ws, cnt, m, x.stride(0), N=n, KX=x.shape[1], KW=k, SPLIT=tile.split,
        BLOCK_N=tile.block_n, BLOCK_K=tile.block_k, BLOCK_T=tile.block_t,
        num_warps=tile.warps, num_stages=tile.stages)


def _fast_gemm(x: torch.Tensor, key: str, state: dict, out_buf=None) -> torch.Tensor:
    m = x.shape[0]
    w = state["dense"][key]
    n, k = w.shape
    out = out_buf if out_buf is not None else torch.empty(m, n, device=x.device, dtype=torch.bfloat16)

    # 1. Check custom CUDA extension (TMA + WGMMA)
    pick = _cuda_pick(state, key, m)
    if pick is not None:
        return _cuda_run(state, x, w, n, torch.bfloat16, *pick, out_buf=out)

    # 2. Check Gluon TMA ring
    gtile = _gluon_pick(state, key, m)
    if gtile is not None:
        _gluon_launch(x, out, state, key, gtile)
        return out

    # 3. Check tuned Triton BF16 tile (e.g. m=1 FMA GEMV)
    tile = _bf16_pick(key, m)
    if tile is not None:
        _bf16_launch(x, w, out, state, key, tile)
        return out

    # 4. Check Big Triton tile
    if (key, m) in BIG_TILES and TensorDescriptor is not None and (key, m) in state.get("big_desc", {}):
        bm, bn, bk, split, warps, stages, swap, tma_store = BIG_TILES[(key, m)]
        _big_gemm[(triton.cdiv(m, bm) * (n // bn) * split,)](
            TensorDescriptor.from_tensor(x, [bm, bk]), state["big_desc"][(key, m)], out, state["ws"],
            state["cnt"], m, N=n, K=k,
            SPLIT=split, BM=bm, BN=bn, BK=bk, SWAP=swap, TMA_STORE=tma_store,
            num_warps=warps, num_stages=stages)
        return out

    # 5. Fallback to cuBLAS dispatch
    return _bf16_gemm_dispatch_impl(x, w, None)


def _fast_router(x: torch.Tensor, state: dict, out_buf=None) -> torch.Tensor:
    m = x.shape[0]
    out = out_buf if out_buf is not None else torch.empty(m, 256, device=x.device, dtype=torch.float32)

    pick = _cuda_pick(state, "router", m)
    if pick is not None:
        kind, params = pick
        w = state["router"] if kind == "gemv" else state["router_hl"]
        return _cuda_run(state, x, w, 256, torch.float32, kind, params, out_buf=out)

    # Fallback to high/low precision split MM
    acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
    acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    return acc


# ===========================================================================
# Decoder Layer Helpers & Norm Kernels
# ===========================================================================

def _param(w):
    return nn.Parameter(w.contiguous(), requires_grad=False)


def _norm(cls, w, size, eps, **kw):
    norm = cls(size, eps=eps, has_weight=w is not None, zero_centered_gamma=True, **kw)
    if w is not None:
        norm.weight = _param(w)
        norm.precompute_effective_weight()
    return norm


def _residual(w):
    gate = MuseSpark1xNormPreservingResidualConnection(HIDDEN, temperature=0.3)
    gate.gate = _param(w)
    gate.precompute_coefficients()
    return gate


@triton.jit
def _fast_pre_expert_norm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    stride_x_tok,
    stride_out_tok,
    N: tl.constexpr,
    eps: tl.constexpr,
):
    tok_idx = tl.program_id(0)
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + tok_idx * stride_x_tok + offs).to(tl.float32)
    w = tl.load(w_ptr + offs).to(tl.float32)
    rms = tl.sqrt(tl.sum(x * x) / N + eps)
    y = (x / rms) * (1.0 + w)
    tl.store(out_ptr + tok_idx * stride_out_tok + offs, y.to(out_ptr.dtype.element_ty))


@triton.jit
def _fused_prepare_kv_kernel(
    qkv_ptr,
    positions_ptr,
    cos_sin_ptr,
    cache_loc_ptr,
    q_out_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_val,
    v_scale_val,
    has_rope: tl.constexpr,
    scaled: tl.constexpr,
    stride_qkv_tok,
    stride_q_tok,
    stride_k_cache_page,
    stride_k_cache_ps,
    stride_k_cache_head,
    stride_v_cache_page,
    stride_v_cache_ps,
    stride_v_cache_head,
    page_size,
    HEAD_DIM: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
):
    tok_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    eps: tl.constexpr = 1e-5

    offs_d = tl.arange(0, HEAD_DIM)
    offs_d2 = tl.arange(0, HEAD_DIM // 2)

    if head_idx < num_q_heads:
        if has_rope:
            pos = tl.load(positions_ptr + tok_idx)
            cos_val = tl.load(cos_sin_ptr + pos * HEAD_DIM + offs_d2)
            sin_val = tl.load(cos_sin_ptr + pos * HEAD_DIM + HEAD_DIM // 2 + offs_d2)

            idx_even = offs_d2 * 2
            idx_odd = offs_d2 * 2 + 1

            q_even = tl.load(qkv_ptr + tok_idx * stride_qkv_tok + head_idx * HEAD_DIM + idx_even).to(tl.float32)
            q_odd = tl.load(qkv_ptr + tok_idx * stride_qkv_tok + head_idx * HEAD_DIM + idx_odd).to(tl.float32)

            rms = tl.sqrt((tl.sum(q_even * q_even) + tl.sum(q_odd * q_odd)) / HEAD_DIM + eps)
            inv_rms = 1.0 / rms
            q_even = q_even * inv_rms
            q_odd = q_odd * inv_rms

            rot_even = q_even * cos_val - q_odd * sin_val
            rot_odd = q_even * sin_val + q_odd * cos_val

            out_offset = tok_idx * stride_q_tok + head_idx * HEAD_DIM
            tl.store(q_out_ptr + out_offset + idx_even, rot_even.to(q_out_ptr.dtype.element_ty))
            tl.store(q_out_ptr + out_offset + idx_odd, rot_odd.to(q_out_ptr.dtype.element_ty))
        else:
            q_offset = tok_idx * stride_qkv_tok + head_idx * HEAD_DIM + offs_d
            q_vals = tl.load(qkv_ptr + q_offset).to(tl.float32)
            rms = tl.sqrt(tl.sum(q_vals * q_vals) / HEAD_DIM + eps)
            q_norm = q_vals / rms
            out_offset = tok_idx * stride_q_tok + head_idx * HEAD_DIM + offs_d
            tl.store(q_out_ptr + out_offset, q_norm.to(q_out_ptr.dtype.element_ty))

    else:
        kv_h = head_idx - num_q_heads
        loc = tl.load(cache_loc_ptr + tok_idx)
        page_idx = loc // page_size
        offset_in_page = loc % page_size

        k_start = (num_q_heads + kv_h) * HEAD_DIM
        v_start = (num_q_heads + num_kv_heads + kv_h) * HEAD_DIM

        if has_rope:
            pos = tl.load(positions_ptr + tok_idx)
            cos_val = tl.load(cos_sin_ptr + pos * HEAD_DIM + offs_d2)
            sin_val = tl.load(cos_sin_ptr + pos * HEAD_DIM + HEAD_DIM // 2 + offs_d2)

            idx_even = offs_d2 * 2
            idx_odd = offs_d2 * 2 + 1

            k_even = tl.load(qkv_ptr + tok_idx * stride_qkv_tok + k_start + idx_even).to(tl.float32)
            k_odd = tl.load(qkv_ptr + tok_idx * stride_qkv_tok + k_start + idx_odd).to(tl.float32)

            rms = tl.sqrt((tl.sum(k_even * k_even) + tl.sum(k_odd * k_odd)) / HEAD_DIM + eps)
            inv_rms = 1.0 / rms
            k_even = k_even * inv_rms
            k_odd = k_odd * inv_rms

            rot_k_even = k_even * cos_val - k_odd * sin_val
            rot_k_odd = k_even * sin_val + k_odd * cos_val

            if scaled:
                rot_k_even = rot_k_even / k_scale_val
                rot_k_odd = rot_k_odd / k_scale_val

            k_dst_offset = page_idx * stride_k_cache_page + offset_in_page * stride_k_cache_ps + kv_h * stride_k_cache_head
            tl.store(k_cache_ptr + k_dst_offset + idx_even, rot_k_even.to(k_cache_ptr.dtype.element_ty))
            tl.store(k_cache_ptr + k_dst_offset + idx_odd, rot_k_odd.to(k_cache_ptr.dtype.element_ty))
        else:
            k_offset = tok_idx * stride_qkv_tok + k_start + offs_d
            k_vals = tl.load(qkv_ptr + k_offset).to(tl.float32)
            rms = tl.sqrt(tl.sum(k_vals * k_vals) / HEAD_DIM + eps)
            k_norm = k_vals / rms
            if scaled:
                k_norm = k_norm / k_scale_val
            k_dst_offset = page_idx * stride_k_cache_page + offset_in_page * stride_k_cache_ps + kv_h * stride_k_cache_head + offs_d
            tl.store(k_cache_ptr + k_dst_offset, k_norm.to(k_cache_ptr.dtype.element_ty))

        v_offset = tok_idx * stride_qkv_tok + v_start + offs_d
        v_vals = tl.load(qkv_ptr + v_offset).to(tl.float32)
        if scaled:
            v_vals = v_vals / v_scale_val
        v_dst_offset = page_idx * stride_v_cache_page + offset_in_page * stride_v_cache_ps + kv_h * stride_v_cache_head + offs_d
        tl.store(v_cache_ptr + v_dst_offset, v_vals.to(v_cache_ptr.dtype.element_ty))


def _marlin_bank(weights):
    device = weights["w13"].device
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"].contiguous(),
        w2_weight=weights["w2"].contiguous(),
        w13_weight_scale=weights["w13_sf"].contiguous(),
        w2_weight_scale=weights["w2_sf"].contiguous(),
        w13_weight_scale_2=weights["w13_gs"].contiguous(),
        w2_weight_scale_2=weights["w2_gs"].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
    expert_map[:LOCAL_EXPERTS] = torch.arange(LOCAL_EXPERTS, dtype=torch.int32, device=device)
    w13_num_groups = layer.w13_weight_scale.size(1)
    w13_group_size = 4096 // w13_num_groups if w13_num_groups > 1 else -1
    w2_num_groups = layer.w2_weight_scale.size(1)
    w2_group_size = layer.intermediate_size_per_partition // w2_num_groups if w2_num_groups > 1 else -1
    return {
        "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
        "w13_num_groups": w13_num_groups, "w13_group_size": w13_group_size,
        "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
        "w2_num_groups": w2_num_groups, "w2_group_size": w2_group_size,
        "expert_map": expert_map, "size_n": layer.intermediate_size_per_partition,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
    }


def prepare(weights: Weights):
    device = weights["qkv_gate"].device
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, 64, 2, dtype=torch.float, device=device) / 64))
    freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
    pre_ffn = _norm(MuseSpark1xRMSNorm, weights["pre_ffn_norm"], HIDDEN, RMS_EPS)
    next_pre_attn = _norm(MuseSpark1xRMSNorm, weights["next_pre_attn_norm"], HIDDEN, RMS_EPS)
    post_attn = _norm(MuseSpark1xRMSNorm, None, HIDDEN, POST_NORM_EPS, is_post_norm=True,
                      post_norm_gain_center_type="one", residual_id=1)
    post_ffn = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_ffn_norm"], HIDDEN, POST_NORM_EPS,
                     is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    post_expert = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_expert_norm"], D_EXPERT, POST_NORM_EPS,
                        is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)

    size_n = weights["w13"].shape[1] // 2
    size_k = D_EXPERT
    gemm1_n = 2 * size_n
    cache2_buf = torch.empty((MAX_TOKENS * TOP_K, size_n), device=device, dtype=torch.bfloat16)
    cache13_buf = torch.empty((MAX_TOKENS * TOP_K * max(gemm1_n, size_k),), device=device, dtype=torch.bfloat16)

    max_padded = 32768
    align_sorted_ids = torch.empty((max_padded,), dtype=torch.int32, device=device)
    align_expert_ids = torch.empty((max_padded // 8,), dtype=torch.int32, device=device)
    align_n_padded = torch.empty((1,), dtype=torch.int32, device=device)
    align_cumsum = torch.empty((N_EXPERTS + 2,), dtype=torch.int32, device=device)

    # Preallocated scratch buffers
    q_out_buf = torch.empty((MAX_TOKENS, 16, 64), dtype=FP8, device=device)
    norm_buf = torch.empty((MAX_TOKENS, D_EXPERT), dtype=torch.bfloat16, device=device)
    expert_out_buf = torch.empty((MAX_TOKENS, D_EXPERT), dtype=torch.bfloat16, device=device)
    qkv_buf = torch.empty((MAX_TOKENS, 2304), dtype=torch.bfloat16, device=device)
    attn_out_buf = torch.empty((MAX_TOKENS, HIDDEN), dtype=torch.bfloat16, device=device)
    attn_fa3_out_buf = torch.empty((MAX_TOKENS, 16, 64), dtype=torch.bfloat16, device=device)
    pooler_buf = torch.empty((MAX_TOKENS, D_EXPERT), dtype=torch.bfloat16, device=device)
    unpooler_buf = torch.empty((MAX_TOKENS, HIDDEN), dtype=torch.bfloat16, device=device)
    router_buf = torch.empty((MAX_TOKENS, N_EXPERTS), dtype=torch.float32, device=device)

    # Compile custom CUDA extension (TMA + WGMMA + Cluster DSMEM)
    ext = None
    try:
        ext = _build_cuda()
    except Exception as exc:
        print(f"CUDA extension failed to load, falling back to Triton: {exc}")

    # Gluon setup
    gl_desc = {}
    gl_xl = {}
    if gluon is not None:
        for key, tiles in GLUON_TILES.items():
            for t in tiles:
                dk = (key, t.block_n, t.block_k)
                if dk not in gl_desc:
                    wl = gl.NVMMASharedLayout.get_default_for([t.block_n, t.block_k], gl.bfloat16)
                    gl_desc[dk] = GluonDesc.from_tensor(weights[key].contiguous(), [t.block_n, t.block_k], wl)
                gl_xl[(t.block_t, t.block_k)] = _gluon_xlayout(t)

    # Big Triton GEMM descriptors
    big_desc = {}
    if TensorDescriptor is not None:
        for (layer, m), (bm, bn, bk, split, warps, stages, swap, tma_store) in BIG_TILES.items():
            big_desc[(layer, m)] = TensorDescriptor.from_tensor(weights[layer].contiguous(), [bn, bk])

    dense_weights = {name: weights[name].contiguous() for name in ("qkv_gate", "o", "pooler", "unpooler")}
    dense_weights["unpooler_slice"] = weights["unpooler"][:UNPOOL_SLICE].contiguous()

    state = {
        "dense": dense_weights,
        "router": router.contiguous(),
        "router_hl": torch.stack([hi, lo]).contiguous(),
        "router_hi_t": hi.t().contiguous(),
        "router_lo_t": lo.t().contiguous(),
        "routing": FusedMuseSparkRouting(expert_bias=weights["router_bias"].float().contiguous(), route_scale=1.0,
                                         router_output_multiplier=0.1767766952966369,
                                         router_score_division_eps=1e-15),
        "cos_sin": torch.cat((freqs.cos(), freqs.sin()), dim=-1).contiguous(),
        "o_norm_gate": FusedRMSNormGate(MuseSpark1xRMSNorm(64, eps=RMS_EPS, has_weight=False), True, 64),
        "attn_boundary": FusedPostNormResidualPreNorm(post_attn, _residual(weights["attn_residual_gate"]), pre_ffn),
        "ffn_boundary": FusedPostNormResidualPreNorm(post_ffn, _residual(weights["ffn_residual_gate"]), next_pre_attn),
        "pre_expert_w": (weights["pre_expert_norm"].float()).contiguous(),
        "post_expert_w": post_expert._get_effective_weight().detach().to(torch.bfloat16),
        "bank": _marlin_bank(weights),
        "side_stream": torch.cuda.Stream(device),
        "_modules": (pre_ffn, next_pre_attn, post_attn, post_ffn, post_expert),
        "cache2_buf": cache2_buf,
        "cache13_buf": cache13_buf,
        "align_sorted_ids": align_sorted_ids,
        "align_expert_ids": align_expert_ids,
        "align_n_padded": align_n_padded,
        "align_cumsum": align_cumsum,
        "q_out_buf": q_out_buf,
        "norm_buf": norm_buf,
        "expert_out_buf": expert_out_buf,
        "qkv_buf": qkv_buf,
        "attn_out_buf": attn_out_buf,
        "attn_fa3_out_buf": attn_fa3_out_buf,
        "pooler_buf": pooler_buf,
        "unpooler_buf": unpooler_buf,
        "router_buf": router_buf,
        "ext": ext,
        "gl_desc": gl_desc,
        "gl_xl": gl_xl,
        "big_desc": big_desc,
        "desc": {},
        "ws": torch.empty(WS_FLOATS, device=device, dtype=torch.float32),
        "cnt": torch.zeros(COUNTERS, device=device, dtype=torch.int32),
        "sms": torch.cuda.get_device_properties(device).multi_processor_count,
        "marlin_mod": _jit_moe_wna16_marlin_module(torch.bfloat16, True, False),
        "empty_bf16": torch.empty(0, dtype=torch.bfloat16, device=device),
        "empty_i32": torch.empty(0, dtype=torch.int32, device=device),
        "empty_f32": torch.empty(0, dtype=torch.float32, device=device),
        "c_tmp_buf": torch.empty(16 * 1024 * 1024, dtype=torch.float32, device=device),
        "b_q_type_id": scalar_types.float4_e2m1f.id,
    }

    # Warmup all shapes to compile kernels ahead of CUDA graph capture
    for m in (1, 8, 16, 32, 128, 256):
        x_tok = torch.zeros(m, HIDDEN, device=device, dtype=torch.bfloat16)
        _fast_gemm(x_tok, "qkv_gate", state)
        _fast_gemm(x_tok, "pooler", state)
        _fast_gemm(x_tok[:, :1024], "o", state)
        if m <= 8:
            _fast_gemm(x_tok[:, :D_EXPERT], "unpooler_slice", state)
        else:
            _fast_gemm(x_tok[:, :D_EXPERT], "unpooler", state)
        _fast_router(x_tok, state)

    del freqs
    torch.cuda.synchronize()
    return state


def _fast_marlin(
    a, c, b_q_weight, b_scales, b_global_scale,
    sorted_ids, block_experts, n_padded, topk_weights,
    moe_block_size, top_k, is_ep,
    size_m, size_n, size_k,
    num_groups, group_size,
    module, workspace, empty_bf16, empty_i32, empty_f32, c_tmp_buf, sms, b_q_type_id,
    use_fp32_reduce=True,
):
    if use_fp32_reduce:
        max_c_tmp_size = min(
            size_n * sorted_ids.size(0),
            sms * 4 * moe_block_size * _MAX_THREAD_N,
        )
        if moe_block_size == 8:
            max_c_tmp_size *= 2
        c_tmp = c_tmp_buf[:max_c_tmp_size]
    else:
        c_tmp = empty_f32

    module.moe_wna16_marlin_gemm(
        a,
        c,
        b_q_weight,
        empty_bf16,
        b_scales,
        b_global_scale,
        empty_bf16,
        empty_i32,
        empty_i32,
        workspace,
        sorted_ids,
        block_experts,
        n_padded,
        topk_weights,
        empty_bf16,
        c_tmp,
        moe_block_size,
        top_k,
        False,  # mul_topk_weights
        is_ep,
        b_q_type_id,
        size_m,
        size_n,
        size_k,
        False,  # has_act_order
        False,  # has_bias
        True,   # is_k_full
        False,  # has_zp
        num_groups,
        group_size,
        False,  # use_atomic_add
        use_fp32_reduce,
        False,  # is_zp_float
    )
    return c


def _optimized_experts(x, topk_weights, topk_ids, state):
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    local_ids = bank["expert_map"][topk_ids]

    block_size_m = BLOCK_SIZE_M_MAP.get(num_tokens, 32)

    numel = local_ids.numel()
    num_exp = N_EXPERTS
    if numel < num_exp + 1:
        max_num_tokens_padded = numel * block_size_m
    else:
        max_num_tokens_padded = numel + (num_exp + 1) * (block_size_m - 1)
    max_num_tokens_padded = (max_num_tokens_padded + 3) // 4 * 4
    max_num_m_blocks = max_num_tokens_padded // block_size_m

    sorted_ids = state["align_sorted_ids"][:max_num_tokens_padded]
    block_experts = state["align_expert_ids"][:max_num_m_blocks]
    n_padded = state["align_n_padded"]

    if numel <= SMALL_NUMEL_LIMIT and (num_exp + 1) > 64:
        moe_align_small_numel(
            local_ids,
            num_exp + 1,
            block_size_m,
            sorted_ids,
            block_experts,
            n_padded,
        )
    else:
        sgl_moe_align_block_size(
            local_ids,
            num_exp + 1,
            block_size_m,
            sorted_ids,
            block_experts,
            n_padded,
            state["align_cumsum"],
            True,
            False,
        )

    total_pairs = num_tokens * TOP_K
    cache2 = state["cache2_buf"][:total_pairs, :size_n]
    cache1 = state["cache13_buf"][: total_pairs * gemm1_n].view(-1, gemm1_n)
    cache3 = state["cache13_buf"][: total_pairs * size_k].view(-1, size_k)

    module = state["marlin_mod"]
    ws = bank["workspace"]
    empty_bf16 = state["empty_bf16"]
    empty_i32 = state["empty_i32"]
    empty_f32 = state["empty_f32"]
    c_tmp_buf = state["c_tmp_buf"]
    sms = state["sms"]
    b_q_type_id = state["b_q_type_id"]
    use_fp32_reduce = True

    _fast_marlin(
        x, cache1, bank["w13"], bank["w13_s"], bank["w13_gs"],
        sorted_ids, block_experts, n_padded, topk_weights,
        block_size_m, TOP_K, True,
        num_tokens, gemm1_n, size_k,
        bank["w13_num_groups"], bank["w13_group_size"],
        module, ws, empty_bf16, empty_i32, empty_f32, c_tmp_buf, sms, b_q_type_id,
        use_fp32_reduce=use_fp32_reduce,
    )
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    _fast_marlin(
        cache2, cache3, bank["w2"], bank["w2_s"], bank["w2_gs"],
        sorted_ids, block_experts, n_padded, topk_weights,
        block_size_m, 1, True,
        total_pairs, size_k, size_n,
        bank["w2_num_groups"], bank["w2_group_size"],
        module, ws, empty_bf16, empty_i32, empty_f32, c_tmp_buf, sms, b_q_type_id,
        use_fp32_reduce=use_fp32_reduce,
    )
    y = cache3.view(num_tokens, TOP_K, size_k)

    out = state["expert_out_buf"][:num_tokens]
    block_h = triton.next_power_of_2(size_k)
    _post_expert_norm_weighted_sum_kernel[(num_tokens,)](
        y,
        state["post_expert_w"],
        topk_weights,
        out,
        local_ids,
        TOPK=TOP_K,
        H=size_k,
        eps=POST_NORM_EPS,
        APPLY_NORM=True,
        HAS_WEIGHT=True,
        WEIGHT_FIRST=True,
        HAS_IDS=True,
        BLOCK_H=block_h,
        num_warps=8 if block_h >= 4096 else 4,
        num_stages=2,
    )
    return out


def kernel_fn(data: Inputs, state: dict):
    cfg = data["config"]
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]

    # 1. Fast QKV projection
    qkv = _fast_gemm(data["attn_input"], "qkv_gate", state, out_buf=state["qkv_buf"][:tokens])

    # 2. Fused QK-norm + RoPE + FP8 quantize + KV-cache write
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    k_scale_val = cfg["kv_scale"]
    v_scale_val = cfg["kv_scale"]

    q_out = state["q_out_buf"][:tokens]
    grid = (tokens, hq + hkv)
    _fused_prepare_kv_kernel[grid](
        qkv,
        data["positions"],
        state["cos_sin"],
        data["cache_loc"],
        q_out,
        k_cache,
        v_cache,
        k_scale_val,
        v_scale_val,
        has_rope=bool(cfg["rope_theta"]),
        scaled=scaled,
        stride_qkv_tok=qkv.stride(0),
        stride_q_tok=q_out.stride(0),
        stride_k_cache_page=k_cache.stride(0),
        stride_k_cache_ps=k_cache.stride(1),
        stride_k_cache_head=k_cache.stride(2),
        stride_v_cache_page=v_cache.stride(0),
        stride_v_cache_ps=v_cache.stride(1),
        stride_v_cache_head=v_cache.stride(2),
        page_size=cfg["page_size"],
        HEAD_DIM=64,
        num_q_heads=hq,
        num_kv_heads=hkv,
        num_warps=2,
    )

    gate = qkv[:, hq * d + 2 * hkv * d:]
    window = cfg["window"]
    descale = {}
    if scaled:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))

    # 3. FlashAttention-3
    out = flash_attn_with_kvcache(
        q=q_out,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"],
        cu_seqlens_q=data["cu_seqlens_q"],
        max_seqlen_q=length,
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
        causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=None,
        pack_gqa=True,
        out=state["attn_fa3_out_buf"][:tokens],
        **descale,
    )
    if isinstance(out, tuple):
        out = out[0]

    # 4. Output norm & gate + Fast O projection
    o = state["o_norm_gate"](out.reshape(tokens, hq * d).to(torch.bfloat16), gate)
    attn_output = _fast_gemm(o, "o", state, out_buf=state["attn_out_buf"][:tokens])

    # 5. Attention residual boundary + Fast Router
    split_router = tokens > ROUTER_FP32_MAX_TOKENS
    ffn_input, stream, router_input = state["attn_boundary"](
        data["stream"], attn_output, branch_dtype=torch.bfloat16, router_dtype=None if split_router else torch.float32)

    main_stream, side_stream = torch.cuda.current_stream(), state["side_stream"]
    side_stream.wait_stream(main_stream)
    expert_input = _fast_gemm(ffn_input, "pooler", state, out_buf=state["pooler_buf"][:tokens])
    normed_expert_in = state["norm_buf"][:tokens]
    _fast_pre_expert_norm_kernel[(tokens,)](
        expert_input,
        state["pre_expert_w"],
        normed_expert_in,
        stride_x_tok=expert_input.stride(0),
        stride_out_tok=normed_expert_in.stride(0),
        N=D_EXPERT,
        eps=RMS_EPS,
    )
    with torch.cuda.stream(side_stream):
        router_in = ffn_input if split_router else router_input.to(torch.bfloat16)
        logits = _fast_router(router_in, state, out_buf=state["router_buf"][:tokens])
        topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
    main_stream.wait_stream(side_stream)
    if not torch.cuda.is_current_stream_capturing():
        for tensor in (topk_weights, topk_ids):
            tensor.record_stream(main_stream)

    # 6. Optimized MoE experts
    local_ids = (topk_ids - data["local_start"]) % N_EXPERTS
    expert_output = _optimized_experts(normed_expert_in, topk_weights, local_ids, state)

    # 8. Fast Unpooler
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        unpooled = _fast_gemm(expert_output, "unpooler_slice", state)
        ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))
    else:
        ffn_output = _fast_gemm(expert_output, "unpooler", state, out_buf=state["unpooler_buf"][:tokens])

    # 9. Post-FFN boundary
    next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output, branch_dtype=torch.bfloat16)
    return next_attn_input, stream, topk_ids
