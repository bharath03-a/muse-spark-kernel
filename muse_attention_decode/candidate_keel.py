# okc: problem=muse_attention_decode
# From github.com/keell0renz/muse-open-kernel-challenge (commit 8e30e1b), Copyright 2026 team keell0renz, Apache-2.0; see NOTICE_keell0renz.
# + keell0renz stack [bulk]: TMA bulk-reduction merge (cp.reduce.async.bulk) + 72-float workspace rows for the q_len-1 per-request kernels that are not one-tile builds (Madhavan113/ParadigmKernelChallenge c0f3460 dab77b3 012d92b, Apache-2.0)
# + ZhimaoL/sglang okc/x-vfgpdl okc_patch/_cuda_src_wst.py, _cuda_src_vfg.py (Apache-2.0)
# Based on github.com/vjeux/okc-2026 muse_attention_decode.py (Apache-2.0). Copied for the OKC starting point.
"""Gated attention decode for Muse Spark Open: one fused CUDA kernel per call.

Algorithm
---------
For each request, the kernel computes gated attention over the request's cached
keys/values plus the block's own new tokens:

    O = rmsnorm(softmax(Q K^T * sm_scale) V) * sigmoid(gate)

with Q/K first RMS-normalized (Q additionally scaled by qk_scale_factor), RoPE
applied on sliding-window layers, and K/V quantized to FP8 (E4M3) in the
cache. QK^T runs on FP8 tensor-core MMAs (queries rounded to E4M3, as FA3
does) and PV on FP16 MMAs (P in FP16, V exact in FP16) -- except q_len 8 on
sm_90, where PV is an FP8 wgmma (P in E4M3, as SGLang's FA3 FP8 path) -- with
an online softmax, so no full score matrix is ever materialized.

Design
------
* One CTA per (request, pair of KV heads, split-KV split). A "pair" is two KV
  heads, i.e. 16 query heads: the kernel's unit of work.
* Each CTA builds the block's new K/V (normalized, RoPE, FP8) and writes them
  to the cache, then streams its range of the paged FP8 cache through shared
  memory in 64-key tiles (cp.async, multi-stage pipeline).
* Splits are merged with FP32 atomics after rescaling to a fixed reference
  (QK-norm bounds the logits); the last CTA of each (request, KV pair)
  normalizes, applies rmsnorm * sigmoid(gate), writes the output, then zeroes
  the workspace for the next call.
* Launch shapes: per-request split-KV, per-request clusters (DSMEM
  reduction), a flat kernel (one CTA pass over all requests' keys, for full
  layers), and a cooperative finalize (every CTA finalizes a slice) for grids
  known resident. The host picks one in _plan(); the kernel itself is built
  once per launch mode (the MODE template parameter), so each build carries
  only its own epilogue.
* Fixed build: sm_90a only. There are no tuning switches left in the
  CUDA source; the taken path is the only path.
* CUDA-graph capturable: no host syncs in the timed path. The module-level
  caches hold only device properties, derived constants (RoPE frequencies),
  and scratch buffers (the workspace is zeroed by the kernel itself between
  calls), never call inputs or results.

Entry point: kernel_fn(data) -> output tensor. Shapes the CUDA path does not
cover fall back to _fallback(), a plain PyTorch implementation of the same
math.

CUDA layout: Cfg/Params/Smem (launch config, kernel
arguments, shared-memory layout), small device helpers (cp.async, mbarrier,
wgmma, conversions, cluster/DSMEM), the per-segment main loop (segment()),
the finalizes (sum_copies, fin_nc1, flat_finalize), attn_kernel, then the
host launchers (launch, run) at the end.
"""

from typing import TypedDict

import torch


# ---- shapes the kernel supports (see _supported) ----
HEAD_DIM = 64            # head_dim of the supported layout
Q_HEADS_PER_PAIR = 16    # query heads per KV-head pair (the kernel's unit of work)
SUPPORTED_Q_LENS = (1, 8)
MAX_BATCH = 512
MAX_REQUEST_PAIRS = 4096  # batch * (num_kv_heads // 2)

# ---- indexing ----
TILE_KEYS = 64            # keys per pipeline tile
INDEX_LIMIT = 1 << 31    # 32-bit indexing bound checked by _fits

# ---- scratch buffers ----
WORKSPACE_MIN_ROWS = 1 << 15  # _workspace grows to at least this many rows
WS_ROW = 72  # floats per workspace row (the q_len 1 per-request kernels pad their rows to 72; the others use 64)

LOG2E = 1.4426950408889634  # log2(e), to full double precision

# ---- packed kernel-mode bitfield (the `tma` argument to run) ----
# bit 1: cooperative finalize; bit 16: PDL.
COOP_BIT = 1   # bit 1: cooperative finalize
PDL_BIT = 16   # bit 16: programmatic dependent launch

# ---- per-(request, KV pair) launch grid: the per-request grids run slightly
# below the CTA target, leaving headroom for the finalize tail ----
PLAN_HEADROOM_NUM = 21
PLAN_HEADROOM_DEN = 22


class Config(TypedDict):
    batch: int
    q_len: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    window: int
    rope_theta: float
    qk_scale_factor: float
    sm_scale: float
    kv_scale: float


class Inputs(TypedDict):
    # scheduler_metadata, cache_seqlens and cu_seqlens_q are carried for the
    # harness interface; the CUDA path uses seq_lens / cache_loc / page_table
    # instead, and positions feeds only the fallback.
    qkv: torch.Tensor
    positions: torch.Tensor
    cache_loc: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    scheduler_metadata: torch.Tensor | None
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    config: Config


CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda.h>
#include <stdint.h>

namespace {

__device__ __noinline__ void spin_fail() { __trap(); }   // (out of line: keeps the trap off the polling loop's code)
#define SPIN_T0 unsigned _spin_n = 0   /* (an iteration count: a %globaltimer read per poll slowed the coop barriers ~1%) */
#define SPIN_CHECK do { if (++_spin_n > (1u << 24)) spin_fail(); } while (0)   /* (>= 2^24 global polls: seconds) */
template <int L, bool F>
struct Cfg {
  static constexpr int NT = (L == 1) ? 128 : 256;   // consumer threads per CTA
  // flat kernel on sm_90: one extra producer warp issues every K/V copy (warp specialization), so the
  // consumers never stall in cp.async issue; the stages are handed over with mbarriers
  static constexpr bool WS = F;
  static constexpr int NP = WS ? 128 : 0;   // producer threads
  static constexpr int NTX = NT + NP;                     // threads per CTA
  static constexpr int MINB = 2;   // CTAs per SM the registers must allow
  // setmaxnreg split of the CTA's registers (launch: 65536 / (MINB * NTX), rounded down to 8)
  static constexpr int BASEREGS = (65536 / (2 * NTX)) / 8 * 8 > 255 ? 248 : (65536 / (2 * NTX)) / 8 * 8;
  static constexpr int PREGS = (L == 1) ? 56 : 32;
  static constexpr int CREGS = ((BASEREGS * NTX - PREGS * NP) / (NT > 0 ? NT : 1)) / 8 * 8 > 248
                                   ? 248 : ((BASEREGS * NTX - PREGS * NP) / (NT > 0 ? NT : 1)) / 8 * 8;
  // cp.async pipeline depth (64-key tiles): deep for the long flat runs, shallow (smaller shared memory,
  // more CTAs per SM, all clusters resident) for the per-request grids of sliding windows / one request
  static constexpr int STAGES = F ? ((L == 1) ? 3 : 4) : 3;
  // producer passes: NP/8 rows each must tile the 64-row tile exactly (else rows go missing and the consumers wait
  // forever on the stage barrier)
  static_assert(!WS || (NP % 8 == 0 && NP >= 8 && 64 % (NP >= 8 ? NP / 8 : 1) == 0),
                "producer threads must cover a tile in whole passes");
};
constexpr int TILE = 64;
constexpr int STAGES = 4;
constexpr int ROWB = 128;                    // bytes per tile row: 2 KV heads x 64
constexpr int STAGE_BYTES = TILE * ROWB;     // 8 KB per stage for K (same for V)
constexpr int IDX_SLOTS = 2 * STAGES;
constexpr int MAXB = 512;               // flat partition: requests per call
constexpr int MAXSEG = 64;              // flat partition: segments per CTA

struct Params {
  const __nv_bfloat16* qkv;
  const int* seq_lens;
  const int64_t* cache_loc;
  uint8_t* k_cache;
  uint8_t* v_cache;
  const int* page_table;
  const float* k_scale;
  const float* v_scale;
  const float* inv_freq;
  __nv_bfloat16* out;
  float* ws_acc;
  float* ws_l;
  int* ws_cnt;
  int B, Hq, Hkv, P, max_pages, window, rope, n_splits, flat, nc;
  long long* dbg;      // optional per-CTA phase timestamps (globaltimer), null in normal runs
  int cs, ncl;          // cluster mode: cs CTAs per cluster (DSMEM reduction), ncl clusters per request
  int ld_qkv, row_bytes;
  float scale_log2;   // sm_scale * qk_scale_factor * log2(e)
  int coop;             // per-request grids that are all resident: every CTA finalizes a slice of the rows
  int lp;               // log2(page size)
  int lrb;              // log2(row bytes of a cache row: Hkv * 64)
  int off32;            // both caches below 4 GB (32-bit cache offsets); else the 64-bit path
  unsigned ns_magic;    // ceil(2^32 / n_splits) (0: n_splits == 1)
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async16_pred(uint32_t dst, const void* src, int src_bytes, bool pred) {
  // predicated copy (no branch): pred = 0 issues nothing, src_bytes = 0 zero-fills
  asm volatile("{\n .reg .pred q;\n setp.ne.b32 q, %3, 0;\n @q cp.async.cg.shared.global [%0], [%1], 16, %2;\n}\n"
               ::"r"(dst), "l"(src), "r"(src_bytes), "r"((int)pred));
}
__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// ---- sm_90 TMA row loads tracked by one mbarrier per pipeline stage ----
__device__ __forceinline__ void mbar_fence_init() {
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, int bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_init_n(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_u32(bar)) : "memory");
}
// unbounded wait for a phase (producer / consumer hand-off)
__device__ __forceinline__ void mbar_wait_full(uint64_t* bar, int parity) {
  asm volatile("{\n.reg .pred q;\nWAIT_%=:\nmbarrier.try_wait.parity.shared::cta.b64 q, [%0], %1;\n@!q bra WAIT_%=;\n}\n"
               ::"r"(smem_u32(bar)), "r"(parity) : "memory");
}
// the mbarrier receives one arrival when all of this thread's earlier cp.async copies have landed
__device__ __forceinline__ void cp_async_mbar_arrive(uint64_t* bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(smem_u32(bar)) : "memory");
}
// plain bulk copy global -> shared (16-byte aligned, multiple of 16 bytes), completion on an mbarrier
__device__ __forceinline__ void bulk_load(void* dst, const void* src, int bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
// TMA bulk reduction: global[dst .. dst + bytes) += shared[src ..] (fp32 add), tracked by the thread's bulk async-group
    __device__ __forceinline__ void bulk_red_add_f32(float* dst, const void* src, int bytes) {
      asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;\n"
                   ::"l"(dst), "r"(smem_u32(src)), "r"(bytes) : "memory");
    }
    __device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }
    __device__ __forceinline__ void bulk_wait_all() {   // every bulk op of this thread complete (writes performed)
      asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
    }
    __device__ __forceinline__ void fence_proxy_async_global() {   // async-proxy global writes ordered before generic ones
      asm volatile("fence.proxy.async.global;\n" ::: "memory");
    }
    // 16 zero bytes into shared memory through the same cp.async path (no global read)
    __device__ __forceinline__ void cp_async16_zero(uint32_t dst, const void* any_src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 0;\n" ::"r"(dst), "l"(any_src));
}
// barrier over the NT consumer threads only (named barrier 1); __syncthreads when there is no producer
// barrier over one warpgroup (named barriers 3 and 4)
__device__ __forceinline__ void bar_sync_n(int id, int n) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_arrive_n(int id, int n) { asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void wgsync(int wg) {
  asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
}
template <bool WS, int NT>
__device__ __forceinline__ void csync() {
  if constexpr (WS) asm volatile("bar.sync 1, %0;\n" ::"n"(NT) : "memory");
  else __syncthreads();
}

__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}
// two e4m3 (HIGH 16 bits of x) -> f16x2: ptxas folds the half select into F2FP's operand (.H1), no SHF
__device__ __forceinline__ uint32_t cvt2h(uint32_t x) {
  uint32_t d;
  asm("{\n\t.reg .b16 t;\n\tmov.b32 {_, t}, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, t;\n\t}" : "=r"(d) : "r"(x));
  return d;
}
// two e4m3 (low 16 bits of x) -> f16x2 (low byte -> low half)
__device__ __forceinline__ uint32_t cvt2(uint32_t x) {
  uint32_t d;
  asm("{\n\t.reg .b16 t;\n\tmov.b32 {t, _}, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, t;\n\t}" : "=r"(d) : "r"(x));
  return d;
}
__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}
// 2^lo, 2^hi as a packed f16 pair: the inputs rounded to f16 (|x| <= 8 here: error <= 2^-8), one MUFU op
__device__ __forceinline__ uint32_t ex2h2(float lo, float hi) {
  return pack_h2(ex2(lo), ex2(hi));
}
// conformance sensitivity probe only : round the attention output, which SGLang keeps in
// BF16 before rmsnorm * gate, to e4m3 -- the precision check must reject that build
__device__ __forceinline__ float pr8(float x) {
  return x;
}
__device__ __forceinline__ uint8_t to_e4m3(float x) {
  return (uint8_t)__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ void mma_f16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}


__device__ __forceinline__ uint4 lds128(const uint8_t* p) { return *reinterpret_cast<const uint4*>(p); }
__device__ __forceinline__ uint2 lds64(const uint8_t* p) { return *reinterpret_cast<const uint2*>(p); }

// V B-fragments for 16 keys starting at kb (rows kb+2tig, +1, +8, +9), head hk, dims 8g..8g+7.
__device__ __forceinline__ void load_v_frags(const uint8_t* V, int kb, int hk, int g, int tig, uint32_t* b0,
                                             uint32_t* b1) {
  const int ch = hk * 4 + (g >> 1), off = (g & 1) * 8;
  const int k0 = kb + 2 * tig;
  const uint2 v0 = lds64(V + k0 * ROWB + ((ch ^ (k0 & 7)) << 4) + off);
  const uint2 v1 = lds64(V + (k0 + 1) * ROWB + ((ch ^ ((k0 + 1) & 7)) << 4) + off);
  const uint2 v2 = lds64(V + (k0 + 8) * ROWB + ((ch ^ ((k0 + 8) & 7)) << 4) + off);
  const uint2 v3 = lds64(V + (k0 + 9) * ROWB + ((ch ^ ((k0 + 9) & 7)) << 4) + off);
  uint32_t x;
  x = prmt(v0.x, v1.x, 0x5140); b0[0] = cvt2(x); b0[1] = cvt2h(x);
  x = prmt(v0.x, v1.x, 0x7362); b0[2] = cvt2(x); b0[3] = cvt2h(x);
  x = prmt(v0.y, v1.y, 0x5140); b0[4] = cvt2(x); b0[5] = cvt2h(x);
  x = prmt(v0.y, v1.y, 0x7362); b0[6] = cvt2(x); b0[7] = cvt2h(x);
  x = prmt(v2.x, v3.x, 0x5140); b1[0] = cvt2(x); b1[1] = cvt2h(x);
  x = prmt(v2.x, v3.x, 0x7362); b1[2] = cvt2(x); b1[3] = cvt2h(x);
  x = prmt(v2.y, v3.y, 0x5140); b1[4] = cvt2(x); b1[5] = cvt2h(x);
  x = prmt(v2.y, v3.y, 0x7362); b1[6] = cvt2(x); b1[7] = cvt2h(x);
}

// the merge / barrier fences as fence.acq_rel.gpu (MEMBAR.ALL.GPU) instead of __threadfence's
// fence.sc.gpu (MEMBAR.SC.GPU + ERRBAR + CCTL.IVALL): the message-passing pattern needs release / acquire only
__device__ __forceinline__ void gpu_fence() {
  __threadfence();
}
__device__ __forceinline__ void red_f32(float* a, float x) {
  asm volatile("red.global.add.f32 [%0], %1;" ::"l"(a), "f"(x) : "memory");
}
__device__ __forceinline__ void red_v4(float* a, float x, float y, float z, float w) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(a), "f"(x), "f"(y), "f"(z), "f"(w) : "memory");
}


__device__ __forceinline__ void ldsm_x4_t(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
// four 8x8 b16 matrices from registers to shared memory (each lane gives one row address: lanes 8m..8m+7 the
// rows of matrix m; thread t holds row t/4, bytes 4(t%4)..4(t%4)+3 of each matrix)
__device__ __forceinline__ void stsm_x4(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};\n" ::"r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3)
               : "memory");
}


// ---- per-warp phase profile of the main loops (diagnostic builds only: -DOKC_PROF=1) ----
#define PROF_INIT
#define PROF_MARK(k)
#define PROF_TILE
#define PROF_DUMP

// ---- sm_90 warpgroup MMA helpers ----
__device__ __forceinline__ void fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
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
__device__ __forceinline__ void reg_fence(float& r) { asm volatile("" : "+f"(r)::"memory"); }
// K-major, 128-byte swizzled shared-memory matrix descriptor (8-row groups 1024 bytes apart)
__device__ __forceinline__ uint64_t desc_sw128(uint32_t addr) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
// D[64x64] (+)= A[64x32] (e4m3, registers) * B[64x32]^T (e4m3, shared memory descriptor)
__device__ __forceinline__ void wgmma_m64n64k32_e4m3(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                     uint64_t desc, int scale_d) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
      "{%32, %33, %34, %35}, %36, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(scale_d));
}
// D[64x64] (+)= A[64x32] (e4m3, shared memory descriptor) * B[64x32]^T (e4m3, shared memory descriptor)
__device__ __forceinline__ void wgmma_m64n64k32_e4m3_ss(float* d, uint64_t da, uint64_t db, int scale_d) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
      "%32, %33, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "l"(da), "l"(db), "r"(scale_d));
}
// the same with scale-d = 0 and write-only accumulators: the previous tile's (masked) scores do not feed this
// wgmma, so ptxas sees no non-wgmma definition of its accumulators inside the pipeline (warning C7515, which
// serializes every wgmma of the function)
// K-major operand with 64-byte rows, 64B swizzle (8-row atoms of 512 bytes)
__device__ __forceinline__ uint64_t desc_sw64(uint32_t addr) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(512 >> 4) << 32) | ((uint64_t)2 << 62);
}
// D[64x72] += A[64x32] (e4m3, registers) * B[72x32]^T (e4m3, K-major shared memory): d = 32 + 4 accumulators
__device__ __forceinline__ void wgmma_m64n72k32_e4m3(float* d, float* e, const uint32_t* a, uint64_t desc) {
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35}, "
      "{%36, %37, %38, %39}, %40, 1, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
        "+f"(e[0]), "+f"(e[1]), "+f"(e[2]), "+f"(e[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
// floor(x / d) for 0 <= x < 2^24 and 1 <= d < 2^24 through the float reciprocal (one correction step):
// ~8 instructions instead of the ~20 of an integer division, which matters in once-per-launch (cold) code
__device__ __forceinline__ int fdiv_i(int x, int d) {
  // approximate reciprocal (MUFU.RCP, no IEEE slow-path call: the correction below absorbs its error)
  const float rd = __fdividef(1.f, __int2float_rn(d));
  int q = __float2int_rz(__int2float_rn(x) * rd);
  const int r = x - q * d;
  q += (r >= d) - (r < 0);
  return q;
}
// sigmoid for the output gate: 0.5 + 0.5 tanh(x / 2) with the hardware tanh (3 instructions; abs error ~2^-12,
// below the BF16 output's rounding) or exp + fast reciprocal
__device__ __forceinline__ float sigm(float x) {
  return __fdividef(1.f, 1.f + __expf(-x));
}
// Programmatic dependent launch (sm_90): wait for the preceding grid (the one writing qkv) before the first qkv read;
// let the next grid start its own prologue once the main loop is done. Both are no-ops for a plain launch.
__device__ __forceinline__ void pdl_wait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}
__device__ __forceinline__ void pdl_launch_dependents() {
  // (no early trigger at the end of the tile loop -- the dependent kernel launches when this grid
  // exits; for decode_layer's chain A/B, no effect on standalone launches)
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
// 32-bit offsets for qkv, the workspace and the caches (the host checks the sizes): no 64-bit multiplies
// in the once-per-launch address math
typedef int ix_t;
__device__ __forceinline__ const uint8_t* cache_row(const uint8_t* base, const Params& p, int64_t row, int col_bytes) {
  if (p.off32) return base + (((uint32_t)row << p.lrb) + (uint32_t)col_bytes);
  return base + row * p.row_bytes + col_bytes;
}
// `for (e = tid; e < N; e += NTH)` with a compile-time trip count: fully unrolled, the tail predicated (the
// once-executed prologue loops otherwise keep 3-4 branches each; a taken branch costs 60-100 cycles cold)
template <int N, int NTH, class F>
__device__ __forceinline__ void tfor(int tid, F&& f) {
#pragma unroll
  for (int e0 = 0; e0 < N; e0 += NTH) {
    const int e = e0 + tid;
    if ((N % NTH) == 0 || e < N) f(e);
  }
}
// four floats -> four e4m3 bytes (x0 in the low byte), saturating
__device__ __forceinline__ uint32_t e4m3x4f(float x0, float x1, float x2, float x3) {
  uint16_t lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(lo) : "f"(x0), "f"(x1));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(hi) : "f"(x2), "f"(x3));
  return (uint32_t)lo | ((uint32_t)hi << 16);
}
// four f16 (two f16x2 words, low first) -> four e4m3 bytes (low byte first)
__device__ __forceinline__ uint32_t pack_e4m3x4(uint32_t h01, uint32_t h23) {
  __half2_raw a, b;
  a.x = (unsigned short)(h01 & 0xffffu); a.y = (unsigned short)(h01 >> 16);
  b.x = (unsigned short)(h23 & 0xffffu); b.y = (unsigned short)(h23 >> 16);
  const uint32_t lo = __nv_cvt_halfraw2_to_fp8x2(a, __NV_SATFINITE, __NV_E4M3);
  const uint32_t hi = __nv_cvt_halfraw2_to_fp8x2(b, __NV_SATFINITE, __NV_E4M3);
  return (lo & 0xffffu) | (hi << 16);
}


__device__ __forceinline__ float quad_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
  return v;
}
__device__ __forceinline__ float quad_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  return v;
}
__device__ __forceinline__ float oct_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  v += __shfl_xor_sync(0xffffffffu, v, 4);
  return v;
}

// ---- thread-block clusters (sm_90: DSMEM; elsewhere emulated through global memory for testing) ----
__device__ __forceinline__ int cl_rank(const Params& p) {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return (int)r;
}
__device__ __forceinline__ int cl_gid(const Params& p) {
  return (blockIdx.z * gridDim.y + blockIdx.y) * (gridDim.x / p.cs) + blockIdx.x / p.cs;
}
__device__ __forceinline__ void cl_sync(const Params& p, int phase) {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
// float4 at byte offset `off` of peer `rank`'s partial buffer (`local` = this CTA's copy)
__device__ __forceinline__ float4 cl_ld4(const Params& p, const float* local, int rank, int off_floats, int region_floats) {
  uint32_t a = smem_u32(local + off_floats), r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
  float4 v;
  asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(r));
  return v;
}
__device__ __forceinline__ float cl_ld1(const Params& p, const float* local, int rank, int off_floats, int region_floats) {
  uint32_t a = smem_u32(local + off_floats), r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
  float v;
  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(r));
  return v;
}

template <bool PW>
    struct WSR {   // workspace row layout: PW = padded rows (the q_len 1 per-request kernels that are not one-tile-per-split builds)
      static constexpr int STR = PW ? 72 : 64;   // row stride (floats); padded: the row sum in column 64 (else the separate ws_l vector)
    };
    constexpr int WSL = 64;                      // padded rows: the row sum's column
    // the workspace row sum of row `row` (padded rows: column 64 of the row; else the separate ws_l vector)
    #define WSL_AT(PW, row) ((PW) ? (p.ws_acc + (long long)(row) * 72 + WSL) : (p.ws_l + row))

    template <int L, bool F>
    struct Smem {
      uint8_t k[Cfg<L, F>::STAGES * STAGE_BYTES];
  uint8_t v[Cfg<L, F>::STAGES * STAGE_BYTES];
  uint64_t mbar[8];   // TMA mode: one per stage (64 bytes keeps the members after it 16-byte aligned)
  uint64_t pbar[4];   // producer mode: page-index slots landed (TMA)
  int pgc;            // page-index fills issued in this CTA's earlier segments
  int dbgn;           // debug builds: progress-mark count
  int vidx;           // flat weighted partition: this CTA's slot (2 * SM + arrival rank), -1 = uniform
  alignas(128) int pidx[4][80];   // producer mode: page indices of upcoming tiles (16-byte aligned window)
  uint64_t full[8];   // producer mode: stage landed (32 producer lanes arrive as their copies complete)
  uint64_t empty[8];  // producer mode: stage consumed (one arrival per consumer warp)
  uint64_t vtf[2];    // V^T slot written by the producers (one arrival per producer thread)
  uint64_t vte[2];    // V^T slot read by both heads' PV wgmmas (one arrival per consumer warp)
  uint64_t qdone;     // consumers have read the q staging (S.kvq) and staged S.kn / S.vn
  int claim[2];       // Cfg::F2: next unclaimed tile per KV head
  int nxt[4][2];      // Cfg::F2: each warpgroup's next claimed tile (double buffered)
  int idx[2 * Cfg<L, F>::STAGES * TILE];
  // raw q rows of the block (bf16, prologue only) | L==8: the current tile's K and V in f16 (2 x 2 x 64 x 64)
  // L == 8: q staging (producer mode) before the loop, then the per-tile f16 V copy -- or, FP8 PV, per tile
  // (double buffer) and KV head, V^T as 72 rows (64 dims + 8 rows of ones for the row sums) x 64 keys (permuted to
  // the P register order), K-major with 64B swizzle: 4 x 4608 bytes
  alignas(1024) uint8_t kvq[(L == 1) ? L * 16 * 64 * 2 : 18432];
  // L == 8 with FP8 PV and the block's e4m3 queries per KV head (64 rows x 64 dims, K-major, 64B swizzle)
  alignas(1024) uint8_t q8s[(L == 8) ? 8192 : 16];
  __device__ __forceinline__ __nv_bfloat16* qst() { return reinterpret_cast<__nv_bfloat16*>(kvq); }
  __nv_bfloat16 kvst[L * 4 * 64];   // raw k (2 heads) then v (2 heads) rows per token
  __nv_bfloat16 gst[L * 4 * 64];    // cluster mode: gate rows of this CTA's output slice (<= L*16/4 rows)
  uint8_t kn[L * ROWB];        // new keys of the block, [j][2 heads x 64] e4m3
  uint8_t vn[L * ROWB];
  float cs[L * 32];
  float sn[L * 32];
  float ifreq[32];          // RoPE inverse frequencies (staged with cp.async)
  float scl[4];             // k_scale, v_scale
  long long cloc[L];        // cache rows of the block's tokens
  int flag;
  int sense;          // the cooperative barrier's flag, read early (prologue) with the constants
  int cum[F ? MAXB + 1 : 1];
  int seg[F ? MAXSEG : 1][3];
  int nseg;
  int fcs, fce;             // this CTA's key range (warp 0 -> everyone)
  // flat dynamic chunks : the producer's segment list of its current chunk, and a queue that hands the
  // segments (request, first key, end key, copy; request -1 = done) to the consumers
  int pseg[F ? MAXSEG : 1][3];
  int pnseg, pchunk;
  int dq[F ? 16 : 1][4];
  uint64_t dqf[F ? 16 : 1];
  uint64_t dqe[F ? 16 : 1];
};

// Cluster mode, several clusters per request (Params::ncl > 1): add this CTA's reduced slice into the global
// workspace; the last cluster to arrive sums the copies back into RB/RL and finalizes. Out of line: no scored shape
// uses it, and keeping it out of the one-cluster epilogue keeps that code contiguous (cold instruction fetch).
template <int L, typename SM>
__device__ __noinline__ bool cluster_multi(const Params& p, SM& S, const int CS, const int rank, const int RPC,
                                           const int row0, const int bp, float* RB, float* RL) {
  constexpr int NT = Cfg<L, false>::NT;
  const int tid = threadIdx.x;
  bool fin = true;
    const int clu = fdiv_i(blockIdx.x, CS);
    const int copy = clu - fdiv_i(clu, p.nc) * p.nc;
    for (int e = tid; e < RPC * 16; e += NT) {
      const int row = row0 + (e >> 4);
      float* acc = p.ws_acc + ((ix_t)(bp * p.nc + copy) * L * 16 + row) * WSR<false>::STR + (e & 15) * 4;
      const float4 a = *reinterpret_cast<const float4*>(&RB[e * 4]);
      red_v4(acc, a.x, a.y, a.z, a.w);
    }
    for (int e = tid; e < RPC; e += NT) red_f32(WSL_AT(false, (bp * p.nc + copy) * L * 16 + row0 + e), RL[e]);
    __threadfence();
    __syncthreads();
    if (tid == 0) {
      // every cluster of the request arrives (empty ones with zero partials)
      const int old = atomicAdd(p.ws_cnt + bp * CS + rank, 1);
      S.flag = (old == p.ncl - 1);
    }
    __syncthreads();
    fin = S.flag;
    if (fin) {
      __threadfence();
      for (int e = tid; e < RPC * 16; e += NT) {
        const int row = row0 + (e >> 4);
        constexpr int NCM = 4;
        float4 v[NCM];
#pragma unroll
        for (int c = 0; c < NCM; ++c)
          if (c < p.nc) v[c] = __ldcg(reinterpret_cast<const float4*>(p.ws_acc + ((ix_t)(bp * p.nc + c) * L * 16 + row) * WSR<false>::STR + (e & 15) * 4));
        float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int c = 0; c < NCM; ++c) {
          if (c < p.nc) {
            a.x += v[c].x; a.y += v[c].y; a.z += v[c].z; a.w += v[c].w;
            *reinterpret_cast<float4*>(p.ws_acc + ((ix_t)(bp * p.nc + c) * L * 16 + row) * WSR<false>::STR + (e & 15) * 4) = make_float4(0.f, 0.f, 0.f, 0.f);
          }
        }
        *reinterpret_cast<float4*>(&RB[e * 4]) = a;
      }
      for (int e = tid; e < RPC; e += NT) {
        float a = 0.f;
        for (int c = 0; c < p.nc; ++c) {
          float* lw = WSL_AT(false, (bp * p.nc + c) * L * 16 + row0 + e);
          a += __ldcg(lw);
          *lw = 0.f;
        }
        RL[e] = a;
      }
      if (tid == 0) p.ws_cnt[bp * CS + rank] = 0;
      __syncthreads();
    }
  return fin;
}

// Cluster mode epilogue. Every CTA of the cluster has written its rescaled partial (PB: [L*16][64] o * 2^m,
// PL: [L*16] l * 2^m) into its own shared memory. CTA `rank` sums rows [rank*RPC, (rank+1)*RPC) over the
// cluster through DSMEM, then either finalizes them (one cluster per request) or accumulates them into the
// global workspace, where the last cluster to arrive for this slice finalizes.
template <int L, typename SM>
__device__ __forceinline__ void cluster_finish(const Params& p, SM& S, const int b, const int pair,
                                               const int n_keys, const int chunk) {
  constexpr int NT = Cfg<L, false>::NT;
  constexpr int STAGES = Cfg<L, false>::STAGES;
  constexpr int IDX_SLOTS = 2 * STAGES;
  const int tid = threadIdx.x;
  const int CS = p.cs, rank = cl_rank(p);
  const int RPC = fdiv_i(L * 16, CS);
  const int pairs = p.Hkv >> 1;
  const int bp = b * pairs + pair;
  // S.k and S.v are adjacent: PB [L*16*64] | PL [L*16, padded to 1024] | RB [RPC*64] | RL [RPC]
  float* PB = reinterpret_cast<float*>(S.k);
  float* PL = PB + L * 16 * 64;
  float* RB = PL + 1024;
  float* RL = RB + RPC * 64;
  constexpr int REGION = L * 16 * 64 + 1024;   // emulation: per-CTA scratch floats (PB, then PL at +L*16*64)
  cl_sync(p, 1);
  
  const int row0 = rank * RPC;
  if constexpr (L == 1) {   // compact reduction (this code runs once per launch from a cold instruction cache, so it is kept short):
      // (row, slot) items -- 16 float4 slots of o and one for l -- times the CS peers; CS consecutive lanes load
      // one item from every peer and sum it with shuffles. Two items per thread per pass keep loads in flight.
    const int TOT0 = RPC * 17 * CS;
    const int lcs = 31 - __clz(CS);
#pragma unroll 1
    int TOT = TOT0;
       // dry pass: same code, no work
    
    for (int k0 = 0; k0 < TOT; k0 += 2 * NT) {   // warp-uniform trip count
      float4 v[2];
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int k = k0 + u * NT + tid;
        const int item = k >> lcs, q = k & (CS - 1);
        const int rl = item / 17, slot = item - rl * 17;
        v[u] = make_float4(0.f, 0.f, 0.f, 0.f);
        if (k < TOT) {
          if (slot < 16) v[u] = cl_ld4(p, PB, q, (row0 + rl) * 64 + slot * 4, REGION);
          else v[u].x = cl_ld1(p, PL, q, row0 + rl, REGION);
        }
      }
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        for (int off = 1; off < CS; off <<= 1) {
          v[u].x += __shfl_xor_sync(0xffffffffu, v[u].x, off);
          v[u].y += __shfl_xor_sync(0xffffffffu, v[u].y, off);
          v[u].z += __shfl_xor_sync(0xffffffffu, v[u].z, off);
          v[u].w += __shfl_xor_sync(0xffffffffu, v[u].w, off);
        }
        const int k = k0 + u * NT + tid;
        const int item = k >> lcs, q = k & (CS - 1);
        const int rl = item / 17, slot = item - rl * 17;
        if (k < TOT && q == 0) {
          if (slot < 16) *reinterpret_cast<float4*>(&RB[(rl * 16 + slot) * 4]) = v[u];
          else RL[rl] = v[u].x;
        }
      }
    }
      } else
  {
  // compact: peers in groups of 4 (one loop body, warm after its first pass: this code runs once per launch)
  for (int e = tid; e < RPC * 16; e += NT) {
    const int off = (row0 + (e >> 4)) * 64 + (e & 15) * 4;
    const bool lrow = (e & 15) == 0;
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
    float la = 0.f;
#pragma unroll 1
    for (int q0 = 0; q0 < CS; q0 += 4) {
      float4 v[4];
      float lv[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        v[u] = cl_ld4(p, PB, q0 + u, off, REGION);
        lv[u] = lrow ? cl_ld1(p, PL, q0 + u, row0 + (e >> 4), REGION) : 0.f;
      }
#pragma unroll
      for (int u = 0; u < 4; ++u) { a.x += v[u].x; a.y += v[u].y; a.z += v[u].z; a.w += v[u].w; la += lv[u]; }
    }
    *reinterpret_cast<float4*>(&RB[e * 4]) = a;
    if (lrow) RL[e >> 4] = la;
  }
  
  }
  
  cl_sync(p, 2);   // peers are done reading this CTA's partial

  bool fin = true;
  if (p.ncl > 1) {
    fin = cluster_multi<L>(p, S, CS, rank, RPC, row0, bp, RB, RL);
  }
  if (fin) {
    // rmsnorm * sigmoid(gate) for the slice: 16 threads per row, 4 dims each
    
    const float vscale = S.scl[1];
    for (int eb = tid - (tid & 31); eb < RPC * 16; eb += NT) {   // warp-uniform trip count (shuffles below)
      const int e = eb + (tid & 31);
      const bool act = e < RPC * 16;
      const int rl = act ? (e >> 4) : 0, c4 = e & 15;
      const int row = row0 + rl;
      const int j = row >> 4, hh = row & 15;
      const float lsum = act ? RL[rl] : 0.f;
      const float inv = lsum > 0.f ? __fdividef(vscale, lsum) : 0.f;
      const float4 a = act ? *reinterpret_cast<const float4*>(&RB[e * 4]) : make_float4(0.f, 0.f, 0.f, 0.f);
      float y[4] = {pr8(a.x * inv), pr8(a.y * inv), pr8(a.z * inv), pr8(a.w * inv)};
      float ss = y[0] * y[0] + y[1] * y[1] + y[2] * y[2] + y[3] * y[3];
      ss += __shfl_xor_sync(0xffffffffu, ss, 1);
      ss += __shfl_xor_sync(0xffffffffu, ss, 2);
      ss += __shfl_xor_sync(0xffffffffu, ss, 4);
      ss += __shfl_xor_sync(0xffffffffu, ss, 8);
      const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
      const uint2 graw = *reinterpret_cast<const uint2*>(&S.gst[rl * 64 + c4 * 4]);
      const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw);
      uint2 ov;
      uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const float2 gt = __bfloat1622float2(g2[k]);
        const float s0 = sigm(gt.x), s1 = sigm(gt.y);
        __nv_bfloat162 r2 = __floats2bfloat162_rn(y[2 * k] * rn * s0, y[2 * k + 1] * rn * s1);
        ow[k] = *reinterpret_cast<uint32_t*>(&r2);
      }
      if (act) *reinterpret_cast<uint2*>(p.out + (ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + c4 * 4) = ov;
    }
    
  }
}

// Gate rows of this CTA's cluster slice -> S.gst (cp.async, in the caller's commit group)
template <int L, typename SM>
__device__ __forceinline__ void cluster_stage_gate(const Params& p, SM& S, const int b, const int pair) {
  constexpr int NT = Cfg<L, false>::NT;
  constexpr int STAGES = Cfg<L, false>::STAGES;
  constexpr int IDX_SLOTS = 2 * STAGES;
  const int RPC = fdiv_i(L * 16, p.cs), row0 = cl_rank(p) * RPC;
  for (int e = threadIdx.x; e < RPC * 8; e += NT) {
    const int rl = e >> 3, ch = e & 7;
    const int row = row0 + rl, j = row >> 4, hh = row & 15;
    cp_async16(smem_u32(&S.gst[rl * 64 + ch * 8]),
               p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + ch * 8, 16);
  }
}

// One (request b, KV pair) segment [start, end) of the request's key positions.
// Sum the nc workspace copies of one (row, 8-dim part) of request-pair bp, token j, head hh, and zero them. Copies in
// groups of 4 (loads of a group in flight together): one compact loop body, since this runs once per launch from a
// cold instruction cache (an 8-way unrolled version cost ~2x the instructions).
template <int L, bool PW>
__device__ __forceinline__ void sum_copies(const Params& p, const int bp, const int j, const int hh, const int part,
                                           const bool act, float (&y)[8], float& lsum) {
  if (p.nc == 1) {   // one copy (the common case): no copy loop, no per-copy branches (scout's census)
    if (act) {
      float* acc = p.ws_acc + ((ix_t)(bp * L + j) * 16 + hh) * WSR<PW>::STR + part * 8;
      const float4 xa = __ldcg(reinterpret_cast<const float4*>(acc));
      const float4 xb = __ldcg(reinterpret_cast<const float4*>(acc) + 1);
      const float lc = __ldcg(WSL_AT(PW, (bp * L + j) * 16 + hh));
      y[0] += xa.x; y[1] += xa.y; y[2] += xa.z; y[3] += xa.w;
      y[4] += xb.x; y[5] += xb.y; y[6] += xb.z; y[7] += xb.w;
      lsum += lc;
      *reinterpret_cast<float4*>(acc) = make_float4(0.f, 0.f, 0.f, 0.f);
      *(reinterpret_cast<float4*>(acc) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);
      if (part == 0) *WSL_AT(PW, (bp * L + j) * 16 + hh) = 0.f;
    }
    return;
  }
  if (p.nc == 8) {   // eight copies (the one-request q_len 8 plan): every copy's loads in one round trip
    if (act) {                   // (the rolled 4-copy loop below took two dependent L2 round trips)
      float4 xa[8], xb[8];
      float lc[8];
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        const float* acc = p.ws_acc + ((ix_t)((bp * 8 + c) * L + j) * 16 + hh) * WSR<PW>::STR + part * 8;
        xa[c] = __ldcg(reinterpret_cast<const float4*>(acc));
        xb[c] = __ldcg(reinterpret_cast<const float4*>(acc) + 1);
        lc[c] = __ldcg(WSL_AT(PW, ((bp * 8 + c) * L + j) * 16 + hh));
      }
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        y[0] += xa[c].x; y[1] += xa[c].y; y[2] += xa[c].z; y[3] += xa[c].w;
        y[4] += xb[c].x; y[5] += xb[c].y; y[6] += xb[c].z; y[7] += xb[c].w;
        lsum += lc[c];
        float* acc = p.ws_acc + ((ix_t)((bp * 8 + c) * L + j) * 16 + hh) * WSR<PW>::STR + part * 8;
        *reinterpret_cast<float4*>(acc) = make_float4(0.f, 0.f, 0.f, 0.f);
        *(reinterpret_cast<float4*>(acc) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);
        if (part == 0) *WSL_AT(PW, ((bp * 8 + c) * L + j) * 16 + hh) = 0.f;
      }
    }
    return;
  }
#pragma unroll 1
  for (int c0 = 0; c0 < p.nc; c0 += 4) {
    float4 xa[4], xb[4];
    float lc[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int c = c0 + u;
      xa[u] = make_float4(0.f, 0.f, 0.f, 0.f);
      xb[u] = make_float4(0.f, 0.f, 0.f, 0.f);
      lc[u] = 0.f;
      if (act && c < p.nc) {
        const float* acc = p.ws_acc + ((ix_t)((bp * p.nc + c) * L + j) * 16 + hh) * WSR<PW>::STR + part * 8;
        xa[u] = __ldcg(reinterpret_cast<const float4*>(acc));
        xb[u] = __ldcg(reinterpret_cast<const float4*>(acc) + 1);
        lc[u] = __ldcg(WSL_AT(PW, ((bp * p.nc + c) * L + j) * 16 + hh));
      }
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int c = c0 + u;
      if (act && c < p.nc) {
        y[0] += xa[u].x; y[1] += xa[u].y; y[2] += xa[u].z; y[3] += xa[u].w;
        y[4] += xb[u].x; y[5] += xb[u].y; y[6] += xb[u].z; y[7] += xb[u].w;
        lsum += lc[u];
        float* acc = p.ws_acc + ((ix_t)((bp * p.nc + c) * L + j) * 16 + hh) * WSR<PW>::STR + part * 8;
        *reinterpret_cast<float4*>(acc) = make_float4(0.f, 0.f, 0.f, 0.f);
        *(reinterpret_cast<float4*>(acc) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);
        if (part == 0) *WSL_AT(PW, ((bp * p.nc + c) * L + j) * 16 + hh) = 0.f;
      }
    }
  }
}

// Last-CTA finalize of a request with one workspace copy: sum, normalize, rmsnorm * sigmoid(gate), store,
// reset the workspace. Not inlined: this runs once per launch and is otherwise fetched cold.
template <int L, bool PW, int NT>
__device__ __forceinline__ void fin_nc1(const Params& p, const int b, const int pair, const int bp, const float vscale,
                                     const int dry) {
  const int tid = threadIdx.x;
  // loads of a group of passes first (one L2 round trip per group), then math and stores; groups of 2 passes in a
  // rolled loop (this runs once per launch from a cold instruction cache: q_len 8 has 4 passes)
  constexpr int PASSES = L * 16 / (NT / 8);
  constexpr int GP = PASSES >= 2 ? 2 : 1;
  const int part = tid & 7;
#pragma unroll 1
  for (int p0 = 0; p0 < PASSES; p0 += GP) {
    float4 xa[GP], xb[GP];
    float lc[GP];
    uint4 graw[GP];
#pragma unroll
    for (int u = 0; u < GP; ++u) {
      const int rr = (p0 + u) * (NT / 8) + (tid >> 3);
      const int j = rr >> 4, hh = rr & 15;
      const float* acc = p.ws_acc + ((ix_t)(bp * L + j) * 16 + hh) * WSR<PW>::STR + part * 8;
      xa[u] = __ldcg(reinterpret_cast<const float4*>(acc));
      xb[u] = __ldcg(reinterpret_cast<const float4*>(acc) + 1);
      lc[u] = __ldcg(WSL_AT(PW, (bp * L + j) * 16 + hh));
      graw[u] = *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + part * 8);
    }
#pragma unroll
    for (int u = 0; u < GP; ++u) {
      const int rr = (p0 + u) * (NT / 8) + (tid >> 3);
      const int j = rr >> 4, hh = rr & 15;
      float* acc = p.ws_acc + ((ix_t)(bp * L + j) * 16 + hh) * WSR<PW>::STR + part * 8;
      if (!dry) {
        *reinterpret_cast<float4*>(acc) = make_float4(0.f, 0.f, 0.f, 0.f);
        *(reinterpret_cast<float4*>(acc) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);
        if (part == 0) *WSL_AT(PW, (bp * L + j) * 16 + hh) = 0.f;
      }
      const float inv = lc[u] > 0.f ? __fdividef(vscale, lc[u]) : 0.f;
      float y[8] = {pr8(xa[u].x * inv), pr8(xa[u].y * inv), pr8(xa[u].z * inv), pr8(xa[u].w * inv),
                    pr8(xb[u].x * inv), pr8(xb[u].y * inv), pr8(xb[u].z * inv), pr8(xb[u].w * inv)};
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) ss += y[e] * y[e];
      ss = oct_sum(ss);
      const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
      const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw[u]);
      uint4 ov;
      uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 gt = __bfloat1622float2(g2[e]);
        const float s0 = sigm(gt.x), s1 = sigm(gt.y);
        __nv_bfloat162 r2 = __floats2bfloat162_rn(y[2 * e] * rn * s0, y[2 * e + 1] * rn * s1);
        ow[e] = *reinterpret_cast<uint32_t*>(&r2);
      }
      if (!dry) *reinterpret_cast<uint4*>(p.out + (ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + part * 8) = ov;
    }
  }
}

// the q_len 8 flat loop's rare new-row patch as a call, so ptxas cannot hoist its bounds math into the
// common path of every tile
template <int L, int NT, bool PTR, class SM>
__device__ __noinline__ bool ws_patch_ni(SM* Sp, int i, int st, int start, int seq, int end, int tid) {
  SM& S = *Sp;
  const int t0 = start + i * TILE;
  const int lo = max(t0, seq), hi = min(t0 + TILE, end);
  if (lo < hi) {
    for (int e = tid; e < (hi - lo) * 8; e += NT) {
      const int pos = lo + (e >> 3), c = e & 7, r = pos - t0;
      const int sw = ((c ^ (r & 7)) << 4);
      *reinterpret_cast<uint4*>(S.k + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + c * 16);
      *reinterpret_cast<uint4*>(S.v + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + c * 16);
    }
    if (L == 8) fence_proxy_async();
  }
  return lo < hi;
}

template <int L, bool FLAT, int ROLE, int MODE = 0>
__device__ __forceinline__ bool segment(const Params& p, Smem<L, FLAT>& S, const int b_, const int pair_, const bool first,
                                        const int start_, const int end_, const int seq_, const int n_keys_,
                                        const int copy_, const int chunk_, const int tb_ = 0, const int seq_real_ = 0,
                                        const bool skip_init_ = false) {
  // (returns false only in the builds, when the speculated length missed: nothing global was written yet and
  // the caller re-runs the segment with the real length)
  // per-segment values as lane-0 broadcasts : provably warp-uniform, so ptxas keeps the address math
  // derived from them in uniform registers (they come from shared memory or per-CTA math, uniform but unproven)
#define UNI(x) __shfl_sync(0xffffffffu, (x), 0)
  const int b = UNI(b_), pair = UNI(pair_), start = UNI(start_), end = UNI(end_), seq = UNI(seq_);
  const int n_keys = UNI(n_keys_), copy = UNI(copy_), chunk = UNI(chunk_), tb = UNI(tb_);
#undef UNI
  // tb: tiles this CTA's pipeline has already run (flat producer mode: the stage ring and its mbarrier phases
  // continue across segments, so the producer runs ahead into the next segment instead of draining)
  // MODE: the per-request (non-flat) kernel is built once per launch mode, so each build carries only its own
  // epilogue (once-per-launch code runs from a cold instruction cache): 1 plain split-KV, 2 clusters,
  // 3 cooperative; 0 = decided at run time (flat kernels)
  // MODE | 4 = a build for grids whose every split holds at most one tile (the host checks): the tile loop,
  // the in-loop copies and the first-tile tests fold at compile time (scout's once-path census, s1 / vs1)
  constexpr bool ONE = !FLAT && (MODE & 4) != 0;
  constexpr bool PADWS = L == 1 && !FLAT && !ONE;   // padded workspace rows + bulk-reduction merge
  const bool M_CL = !FLAT && ((MODE & 3) == 0 ? p.cs > 1 : (MODE & 3) == 2);
  const bool M_COOP = (MODE & 3) == 0 ? p.coop != 0 : (MODE & 3) == 3;
  constexpr int NT = Cfg<L, FLAT>::NT;
  constexpr int STAGES = Cfg<L, FLAT>::STAGES;
  // flat q_len 8 with FP8 PV: the producer warps (idle most of the time) build each tile's V^T for the PV wgmma
  constexpr int IDX_SLOTS = 2 * STAGES;
  constexpr bool WS = Cfg<L, FLAT>::WS;
  // (the warp index as a lane-0 broadcast, so ptxas can prove it warp-uniform and keep the addresses
  // derived from it (KV head, descriptors, buffers) in uniform registers; CUTLASS's canonical_warp_idx_sync idiom)
  const int tid = threadIdx.x, lane = tid & 31, warp = __shfl_sync(0xffffffffu, tid >> 5, 0);
  const int g = lane >> 2, tig = lane & 3;
  const int split = first ? 0 : 1;
  const int pairs = p.Hkv >> 1;
  const int bp = b * pairs + pair;
  const int ntiles = ONE ? 1 : (end - start + TILE - 1) / TILE;   // (ONE: a CTA without keys returned earlier)

  if constexpr (WS) {
    constexpr int NP = Cfg<L, FLAT>::NP;
    constexpr int PR = NP / 8;            // rows per producer pass
    constexpr int PK = TILE / PR;         // passes per tile
    if (!skip_init_) {   // (the first segment's barriers were set up at kernel start)
      if (ROLE == 2 && tid == 0) {   // fresh phases for this segment (the previous one ended with every stage consumed)
        for (int st = 0; st < STAGES; ++st) {
          mbar_init_n(&S.full[st], NP);
          mbar_init_n(&S.empty[st], NT);   // F2: one warpgroup per head takes each tile
        }
        
        mbar_fence_init();
      }
      __syncthreads();
    }
    if constexpr (ROLE == 1) {
      // ---- producer threads: 16-byte chunk (q & 7) of rows (q >> 3) + PR*k; the cache rows of every tile,
      // zeros past the range; the block's own new keys are patched in by the consumers
      const int q = tid - NT;
      const int ch = q & 7, r0 = q >> 3;
      const int* ptr = p.page_table + (ix_t)b * p.max_pages;
      const int cache_end = min(end, seq);
      int pg[PK];
      static_assert(PR % 8 == 0, "PFAST: the swizzle chunk is the same for all of a thread's rows");
      const bool pfast = p.off32 && p.lp == 0;
      const uint32_t pcol = (uint32_t)(pair * 128 + ch * 16);
      const uint32_t pso = r0 * ROWB + ((ch ^ (r0 & 7)) << 4);
      // page indices by TMA (q_len 1 flat only: f16 -2.4%, f32 -1.7%; the q_len 8 flat kernel lost ~0.7%)
      constexpr bool TMAPG_ON = (L == 1);
      // page indices by TMA: one 64-entry box per tile, PD tiles ahead, PS slots, producer barrier per tile
      constexpr int PS = 4, PD = 3;
      int pgb = 0;   // fills issued by earlier segments (slot / phase base)
      if constexpr (TMAPG_ON) {
          
          if (q == 0) {
            for (int st = 0; st < PS; ++st) mbar_init_n(&S.pbar[st], 1);
            mbar_fence_init();
          }
          asm volatile("bar.sync 2, %0;\n" ::"n"(NP) : "memory");
        
      }
      // the 16-byte aligned window holding the tile's page indices; reads stay inside the table's allocation
      // (CUDA allocations are 256-byte aligned and sized in larger multiples than 16 bytes)
      const long long pt_total = (long long)p.B * p.max_pages;
      const long long pt_lim = (pt_total + 3) & -4;
      auto fetch_pages = [&](int x) {
        if (q == 0 && x < ntiles) {
          const int slot = (pgb + x) % PS;
          const long long e0 = (long long)b * p.max_pages + ((start + x * TILE) >> p.lp);
          const long long w0 = e0 & -4;
          const long long w1 = min(pt_lim, (e0 + 64 + 3) & -4);
          const int bytes = (int)(w1 - w0) * 4;
          mbar_expect_tx(&S.pbar[slot], bytes);
          bulk_load(&S.pidx[slot][0], p.page_table + w0, bytes, &S.pbar[slot]);
        }
      };
      if constexpr (TMAPG_ON) {
        for (int x = 0; x < PD; ++x) fetch_pages(x);
      }
      auto load_pages_t = [&](int x, int* dst) {
        const int slot = (pgb + x) % PS;
        mbar_wait_full(&S.pbar[slot], ((pgb + x) / PS) & 1);
        const int fp = (start + x * TILE) >> p.lp;
        const int dl = (int)(((long long)b * p.max_pages + fp) & 3);   // window offset of the first page
#pragma unroll
        for (int k = 0; k < PK; ++k) {
          const int pos = start + x * TILE + r0 + PR * k;
          dst[k] = (pos < cache_end) ? S.pidx[slot][(pos >> p.lp) - fp + dl] : 0;
        }
      };
      auto load_pages = [&](int x, int* dst) {
#pragma unroll
        for (int k = 0; k < PK; ++k) {
          const int pos = start + x * TILE + r0 + PR * k;
          dst[k] = (x < ntiles && pos < cache_end) ? __ldg(ptr + (pos >> p.lp)) : 0;
        }
      };
      if constexpr (!TMAPG_ON) load_pages(0, pg);
      
      // tile y's V (both KV heads) -> V^T slot y & 1 (72 x 64 per head: 64 dims + 8 rows of e4m3 ones for the
      // row sums, keys permuted to the P register order, K-major, 64B swizzle), the layout the PV wgmma reads
      for (int x = 0; x < ntiles; ++x) {
        const int st = (tb + x) % STAGES;
        int pgn[PK];
        if constexpr (TMAPG_ON) {
          fetch_pages(x + PD);
          load_pages_t(x, pg);
        } else {
          load_pages(x + 1, pgn);   // in flight while this tile's copies issue
        }
        if (tb + x >= STAGES) mbar_wait_full(&S.empty[st], (((tb + x) / STAGES) - 1) & 1);
        const uint32_t kb = smem_u32(S.k + st * STAGE_BYTES), vb = smem_u32(S.v + st * STAGE_BYTES);
        // common tiles (every row cached, page size 1, caches below 4 GB): one shift-add per row, no per-row
        // branches (the producer's instructions take issue slots from the consumers on the same SMSPs)
        if (pfast && start + x * TILE + TILE <= cache_end) {
#pragma unroll
          for (int k = 0; k < PK; ++k) {
            const uint32_t off = ((uint32_t)pg[k] << p.lrb) + pcol;
            cp_async16(kb + pso + k * (PR * ROWB), p.k_cache + off, 16);
            cp_async16(vb + pso + k * (PR * ROWB), p.v_cache + off, 16);
          }
        } else
#pragma unroll
        for (int k = 0; k < PK; ++k) {
          const int r = r0 + PR * k;
          const int pos = start + x * TILE + r;
          const uint32_t so = r * ROWB + ((ch ^ (r & 7)) << 4);
          if (pos < cache_end) {
            if (p.off32) {   // caches below 4 GB: 32-bit offsets (uniform branch)
              const uint32_t off = ((((uint32_t)pg[k] << p.lp) + (uint32_t)(pos & (p.P - 1))) << p.lrb) + (uint32_t)(pair * 128 + ch * 16);
              cp_async16(kb + so, p.k_cache + off, 16);
              cp_async16(vb + so, p.v_cache + off, 16);
            } else
            {
              const int64_t off = (((int64_t)pg[k] << p.lp) + (pos & (p.P - 1))) * p.row_bytes + pair * 128 + ch * 16;
              cp_async16(kb + so, p.k_cache + off, 16);
              cp_async16(vb + so, p.v_cache + off, 16);
            }
          } else if (pos >= end) {
            cp_async16_zero(kb + so, p.k_cache);
            cp_async16_zero(vb + so, p.v_cache);
          }
        }
        cp_async_mbar_arrive(&S.full[st]);
        
        PROF_TILE;
        if constexpr (TMAPG_ON) {
          asm volatile("bar.sync 2, %0;\n" ::"n"(NP) : "memory");   // slot x % PS is free for tile x + PS
        } else {
#pragma unroll
          for (int k = 0; k < PK; ++k) pg[k] = pgn[k];
        }
      }
      
      
      PROF_DUMP;
      
        cp_wait<0>();
        __syncthreads();   // matches the consumers' end-of-segment barrier
      
      return true;
    }
  }
  if constexpr (ROLE != 1) {

  const int prow = tid >> 1, phalf = tid & 1;
  const int* pt_row = p.page_table + (ix_t)b * p.max_pages;
  const int part = tid & 7;

  // ---- producer helpers ----
  // K/V: thread -> 16-byte chunk (tid & 7) of rows (tid >> 3) + 16*k: a warp copies 4 whole 128-byte rows.
  // New-token rows come from S.kn / S.vn; rows past the range are zeroed.
  const int cchunk = tid & 7, crow0 = tid >> 3;
  constexpr int RPP = NT / 8;   // rows per pass
  auto issue_kv = [&](int x, bool with_new) {
    const int st = x % STAGES;
    const int* idxs = &S.idx[(x % IDX_SLOTS) * TILE];
    bool generic = false;   // st.shared into a K stage: the wgmma (async proxy) needs a proxy fence
    // every row of the tile cached (pos < min(end, seq)), page size 1, caches below 4 GB: one shift-add
    // per row, no per-row branches (the per-request issue runs from a cold instruction cache, mostly once)
    if (with_new && p.off32 && p.lp == 0 && start + x * TILE + TILE <= min(end, seq)) {   // (in-loop issues only: the
      // prologue's cold copy of the tile-0 issue stays the plain one, s1 measured +1.5% with it)
      const uint32_t kst = smem_u32(S.k + st * STAGE_BYTES), vst = smem_u32(S.v + st * STAGE_BYTES);
      const uint32_t col = (uint32_t)(pair * 128 + cchunk * 16);
#pragma unroll
      for (int k = 0; k < TILE / RPP; ++k) {
        const int r = crow0 + RPP * k;
        const uint32_t so = r * ROWB + ((cchunk ^ (r & 7)) << 4);
        const uint32_t off = ((uint32_t)idxs[r] << p.lrb) + col;
        cp_async16(kst + so, p.k_cache + off, 16);
        cp_async16(vst + so, p.v_cache + off, 16);
      }
      return;
    }
#pragma unroll
    for (int k = 0; k < TILE / RPP; ++k) {
      const int r = crow0 + RPP * k;
      const int pos = start + x * TILE + r;
      const int sw = ((cchunk ^ (r & 7)) << 4);
      uint8_t* dk = S.k + st * STAGE_BYTES + r * ROWB + sw;
      uint8_t* dv = S.v + st * STAGE_BYTES + r * ROWB + sw;
      if (pos < end && pos < seq) {
        if (p.off32) {   // caches below 4 GB: 32-bit offsets (uniform branch)
          const uint32_t off = ((((uint32_t)idxs[r] << p.lp) + (uint32_t)(pos & (p.P - 1))) << p.lrb) + (uint32_t)(pair * 128 + cchunk * 16);
          cp_async16(smem_u32(dk), p.k_cache + off, 16);
          cp_async16(smem_u32(dv), p.v_cache + off, 16);
        } else
        {
          const int64_t row = ((int64_t)idxs[r] << p.lp) + (pos & (p.P - 1));
          const int64_t off = row * p.row_bytes + pair * 128 + cchunk * 16;
          cp_async16(smem_u32(dk), p.k_cache + off, 16);
          cp_async16(smem_u32(dv), p.v_cache + off, 16);
        }
      } else if (pos < end) {
        if (with_new) {
          *reinterpret_cast<uint4*>(dk) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + cchunk * 16);
          *reinterpret_cast<uint4*>(dv) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + cchunk * 16);
          generic = true;
        }
      } else {   // past the range: masked out of S; V must be finite (P = 0 times V)
        *reinterpret_cast<uint4*>(dk) = make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(dv) = make_uint4(0, 0, 0, 0);
      }
    }
    if (L == 8 && generic) fence_proxy_async();
  };
  auto issue_idx = [&](int y) {
    if (phalf == 0 && prow < TILE && y < ntiles) {
      const int pos = start + y * TILE + prow;
      if (pos < end && pos < seq) cp_async4(smem_u32(&S.idx[(y % IDX_SLOTS) * TILE + prow]), pt_row + (pos >> p.lp));
    }
  };
  // producer mode: wait for stage `st` (tile i) and write the block's own new keys / values into it
  auto ws_patch = [&](int i, int st) {   // the block's own new rows (positions >= seq) into tile i's stage
    const int t0 = start + i * TILE;
    const int lo = max(t0, seq), hi = min(t0 + TILE, end);
    if (lo < hi) {   // same for every consumer thread
      for (int e = tid; e < (hi - lo) * 8; e += NT) {
        const int pos = lo + (e >> 3), c = e & 7, r = pos - t0;
        const int sw = ((c ^ (r & 7)) << 4);
        *reinterpret_cast<uint4*>(S.k + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + c * 16);
        *reinterpret_cast<uint4*>(S.v + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + c * 16);
      }
      if (L == 8) fence_proxy_async();
    }
    return lo < hi;
  };
  auto ws_acquire = [&](int i, int st) {
    mbar_wait_full(&S.full[st], ((tb + i) / STAGES) & 1);
    const int t0 = start + i * TILE;
    const int lo = max(t0, seq), hi = min(t0 + TILE, end);
    if (lo < hi) {   // same for every consumer thread
      for (int e = tid; e < (hi - lo) * 8; e += NT) {
        const int pos = lo + (e >> 3), c = e & 7, r = pos - t0;
        const int sw = ((c ^ (r & 7)) << 4);
        *reinterpret_cast<uint4*>(S.k + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + c * 16);
        *reinterpret_cast<uint4*>(S.v + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + c * 16);
      }
      if (L == 8) fence_proxy_async();
    }
    return lo < hi;
  };

  // ---- 1. page indices of the first STAGES-1 tiles; raw q / k / v rows into registers ----
  int tidL = tid, partL = part;   // (re-read after the page-index copies are issued)
  constexpr int QPASS = L * 16 / RPP;            // L*16 rows, RPP rows per pass
  const bool has_new = end > seq;
  constexpr int KPASS = (L * 4 + RPP - 1) / RPP; // L*4 rows (k, v of 2 heads), RPP rows per pass
  constexpr bool QDIR = L == 8 && !WS;
  constexpr bool IDXF = !WS;
  {
  if constexpr (!WS) {
#pragma unroll
    for (int y = 0; y < STAGES - 1; ++y) issue_idx(y);
  }
  cp_commit();
  if (!WS && (MODE & 8)) {   // (speculative builds only) a late copy of tid: the q / new-row / table math below cannot be hoisted above the page-index
    // copies (or out of the speculative retry loop to the kernel top: scout's v200 census, segment head 21 -> 62/97)
    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(tidL));
    partL = tidL & 7;
  }

  pdl_wait();   // qkv (q, new k / v, gate) is the preceding kernel's output; nothing above reads or writes it
  // raw q rows: L*16 rows x 128 B = 8 chunks of 16 B per row
  tfor<L * 16 * 8, NT>(tidL, [&](int e) {   // (straight-line: no loop branches on the once-path)
    const int rr = e >> 3, ch = e & 7;
    const int j = rr >> 4, hh = rr & 15;
    cp_async16(smem_u32(&S.qst()[rr * 64 + ch * 8]), p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (16 * pair + hh) * 64 + ch * 8, 16);
  });
  if (has_new) {
    tfor<L * 4 * 8, NT>(tidL, [&](int e) {   // (straight-line: no loop branches on the once-path)
      const int rr = e >> 3, ch = e & 7;
      const int kind = rr / (2 * L), jj = (rr % (2 * L)) >> 1, hh = rr & 1;
      const int col = (kind == 0 ? p.Hq : p.Hq + p.Hkv) * 64 + (2 * pair + hh) * 64 + ch * 8;
      cp_async16(smem_u32(&S.kvst[rr * 64 + ch * 8]), p.qkv + (ix_t)(b * L + jj) * p.ld_qkv + col, 16);
    });
  }
  if (M_CL) cluster_stage_gate<L>(p, S, b, pair);
  // small per-call constants: staged with the same group instead of cold loads in the middle of the prologue
  if (tidL < 8) cp_async16(smem_u32(&S.ifreq[tidL * 4]), p.inv_freq + tidL * 4, 16);
  else if (tidL == 8) cp_async4(smem_u32(&S.scl[0]), p.k_scale);
  else if (tidL == 9) cp_async4(smem_u32(&S.scl[1]), p.v_scale);
  else if (!FLAT && M_COOP && tidL == 10)   // the coop barrier's flag / epoch, early
    cp_async4(smem_u32(&S.sense), p.ws_cnt + 24576 + bp);
  else if (tidL >= 16 && tidL < 16 + L) {
    const int jj = tidL - 16;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(smem_u32(&S.cloc[jj])), "l"(p.cache_loc + b * L + jj));
  }
  cp_commit();
  if (split == 0 && (FLAT || !M_CL)) {  // warm L2 with the gate rows for whichever CTA finishes last
    tfor<L * 16, NT>(tidL, [&](int e) {   // (straight-line: no loop branches on the once-path)
      const __nv_bfloat16* gp = p.qkv + (ix_t)(b * L + (e >> 4)) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + (e & 15)) * 64;
      asm volatile("prefetch.global.L2 [%0];" ::"l"(gp));
    });
  }

  // ---- 2. K/V of the first STAGES-1 tiles in flight as soon as their page indices land ----
  // (per-request kernels): wait for the page-index group only (committed first, 256 B); the q / new
  // k,v / constants group (up to 16 KB) may still be in flight while the K/V copies go out
  cp_wait<IDXF ? 1 : 0>();
  csync<WS, NT>();
  if constexpr (!FLAT && (MODE & 8) != 0) {
    // this CTA's ranges and page indices came from the speculated length (see the kernel); the real
    // cache_seqlens value has landed by now (it was loaded alongside the indices). On a miss, nothing global has been
    // written: drain the copies and let the caller redo the segment with the real length (today's path)
    if (seq_real_ != seq) {
      cp_wait<0>();
      __syncthreads();
      return false;
    }
  }
  if constexpr (!WS) {
    constexpr int PRE = STAGES - 1;   // two streams: one cp.async group per tile, the loop issues pairs
    bool one = false;
    // a CTA with a single fully cached tile (page size 1, caches < 4 GB): its copies straight-line (one
    // shift-add per row), none of the loop / per-row tests / issue_idx past the last tile, then the commits the
    // pipeline's group count expects (scout's census: ~168 instr, 11 taken branches here on s1, cold)
    if (ONE && p.off32 && p.lp == 0) {
      one = true;
      const uint32_t kst = smem_u32(S.k), vst = smem_u32(S.v);
      const uint32_t col = (uint32_t)(pair * 128 + cchunk * 16);
      const int cend = min(end, seq);
#pragma unroll
      for (int k = 0; k < TILE / RPP; ++k) {
        const int r = crow0 + RPP * k;
        const uint32_t so = r * ROWB + ((cchunk ^ (r & 7)) << 4);
        
          // branch-free (scout's v200 census: the per-row branches cost 16 instr + 4 taken branches per warp, cold):
          // cached rows copy, rows past the range zero-fill (masked out of S; V must be finite), the block's new
          // rows are skipped (written by the patch pass after the new-k/v pass)
          const int pr = start + r;
          const bool cached = pr < cend;
          const bool doit = cached || pr >= end;
          const uint32_t off = ((uint32_t)S.idx[r] << p.lrb) + col;   // (zero-filled index past the cached rows)
          cp_async16_pred(kst + so, p.k_cache + off, cached ? 16 : 0, doit);
          cp_async16_pred(vst + so, p.v_cache + off, cached ? 16 : 0, doit);
            // (the block's new rows: written by the patch pass after the new-k/v pass, as with the general issue)
      }
#pragma unroll
      for (int k = 0; k < STAGES - 1; ++k) cp_commit();
    }
    if (!one) {
#pragma unroll
    for (int k = 0; k < STAGES - 1; ++k) {   // (unrolled: no loop branches; rolled: one copy of the issue code)
      if (k < PRE && k < ntiles) issue_kv(k, false);
      issue_idx(k + STAGES - 1);
      if (k < PRE - 1 || k == STAGES - 2) cp_commit();
    }
    }
  }
  }

  if constexpr (IDXF) {   // the q group (all but the STAGES-1 K/V groups just committed), visible to every thread
    cp_wait<STAGES - 1>();
    csync<WS, NT>();
  }
  // ---- 3. while they load: RoPE table, queries, the block's new keys / values ----
  if (p.rope) {
    tfor<L * 32, NT>(tidL, [&](int e) {   // (straight-line: no loop branches on the once-path)
      const int j = e >> 5, i = e & 31;
      const float ang = (float)(seq + j) * S.ifreq[i];   // fp32 angle, as the reference builds it
      float sv, cv;
      // Cody-Waite reduction mod 2 pi with a 3-part constant (6.25 has 5 significant bits: exact products for
      // |k| < 2^19, angles up to ~3e6), then the hardware sin / cos on [-pi, pi]: ~5e-7 absolute, against
      // ~30 cold instructions (and a Payne-Hanek call above |x| ~ 1e5) for the accurate sincosf
      const float kq = rintf(ang * 0.15915493667125702f);
      float rr = fmaf(-kq, 6.25f, ang);
      rr = fmaf(-kq, 0.03318530693650246f, rr);
      rr = fmaf(-kq, 2.430837753308879e-10f, rr);
      __sincosf(rr, &sv, &cv);
      S.cs[e] = cv;
      S.sn[e] = sv;
    });
  }
  csync<WS, NT>();

  const float kscale = S.scl[0], vscale = S.scl[1];
  const float sl2 = p.scale_log2 * kscale;

  if constexpr (QDIR) {
    // (q_len 8, QDIRECT): four threads per query row, 16 dims each (half the passes of the 8-thread
    // layout, two shuffle levels instead of three, one 16-byte e4m3 store); same math per element
    constexpr int RP4 = NT / 4;                    // rows per pass
    constexpr int QP4 = L * 16 / RP4;
    const int q4 = tidL & 3;
#pragma unroll
    for (int ps = 0; ps < QP4; ++ps) {
      const int rr = ps * RP4 + (tidL >> 2);
      const int j = rr >> 4, hh = rr & 15;
      const uint4 qa = *reinterpret_cast<const uint4*>(&S.qst()[rr * 64 + q4 * 16]);
      const uint4 qb = *reinterpret_cast<const uint4*>(&S.qst()[rr * 64 + q4 * 16 + 8]);
      const __nv_bfloat162* ha = reinterpret_cast<const __nv_bfloat162*>(&qa);
      const __nv_bfloat162* hb = reinterpret_cast<const __nv_bfloat162*>(&qb);
      float x[16];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 fa = __bfloat1622float2(ha[e]);
        const float2 fb = __bfloat1622float2(hb[e]);
        x[2 * e] = fa.x; x[2 * e + 1] = fa.y;
        x[8 + 2 * e] = fb.x; x[8 + 2 * e + 1] = fb.y;
      }
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 16; ++e) ss += x[e] * x[e];
      ss += __shfl_xor_sync(0xffffffffu, ss, 1);
      ss += __shfl_xor_sync(0xffffffffu, ss, 2);
      const float r = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
#pragma unroll
      for (int e = 0; e < 16; ++e) x[e] *= r;
      if (p.rope) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int i = q4 * 8 + e;
          const float c = S.cs[j * 32 + i], sn = S.sn[j * 32 + i];
          const float a0 = x[2 * e], a1 = x[2 * e + 1];
          x[2 * e] = a0 * c - a1 * sn;
          x[2 * e + 1] = a0 * sn + a1 * c;
        }
      }
      const int hk4 = hh >> 3, gg4 = hh & 7;
      const int r4 = j * 8 + gg4, kb = q4 * 16;
      *reinterpret_cast<uint4*>(S.q8s + hk4 * 4096 + r4 * 64 + (((kb >> 4) ^ ((r4 >> 1) & 3)) << 4)) =
          make_uint4(e4m3x4f(x[0], x[1], x[2], x[3]), e4m3x4f(x[4], x[5], x[6], x[7]),
                     e4m3x4f(x[8], x[9], x[10], x[11]), e4m3x4f(x[12], x[13], x[14], x[15]));
    }
  } else
#pragma unroll
  for (int ps = 0; ps < QPASS; ++ps) {
    const int rr = ps * RPP + (tidL >> 3);
    const int j = rr >> 4, hh = rr & 15;
    const uint4 qv = *reinterpret_cast<const uint4*>(&S.qst()[rr * 64 + partL * 8]);
    const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&qv);
    float x[8];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 f = __bfloat1622float2(h2[e]);
      x[2 * e] = f.x;
      x[2 * e + 1] = f.y;
    }
    float ss = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) ss += x[e] * x[e];
    ss = oct_sum(ss);
    const float r = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
#pragma unroll
    for (int e = 0; e < 8; ++e) x[e] *= r;
    if (p.rope) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int i = partL * 4 + e;
        const float c = S.cs[j * 32 + i], sn = S.sn[j * 32 + i];
        const float a0 = x[2 * e], a1 = x[2 * e + 1];
        x[2 * e] = a0 * c - a1 * sn;
        x[2 * e + 1] = a0 * sn + a1 * c;
      }
    }
    // f16 query pre-scaled to log2 units, staged in the (still free) last pipeline stage
    uint4 qw;
    uint32_t* qwp = reinterpret_cast<uint32_t*>(&qw);
    // (q_len 8 on sm_90 rounds the query to e4m3: keep it unit-RMS like SGLang does -- pre-scaling it by the
    // kv scale would push it into e4m3's subnormals -- and scale the scores instead)
    const float qsc = (L == 8) ? 1.f : sl2;
    const int hk = hh >> 3, gg = hh & 7;
    if constexpr (QDIR) {
      // straight to the e4m3 A tile of the SS QK^T wgmma (64-byte rows, 16-byte chunks swizzled by
      // (row >> 1) & 3), one fp32 -> e4m3 rounding, no f16 staging / re-read (scout's census: ~90 instr / warp)
      const int r = j * 8 + gg, kb = partL * 8;
      *reinterpret_cast<uint2*>(S.q8s + hk * 4096 + r * 64 + (((kb >> 4) ^ ((r >> 1) & 3)) << 4) + (kb & 15)) =
          make_uint2(e4m3x4f(x[0], x[1], x[2], x[3]), e4m3x4f(x[4], x[5], x[6], x[7]));
      continue;
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) qwp[e] = pack_h2(x[2 * e] * qsc, x[2 * e + 1] * qsc);
    if constexpr (L == 1) { const uint32_t t1 = qwp[1]; qwp[1] = qwp[2]; qwp[2] = t1; }   // (scout: quad order 0,2,1,3)
    if constexpr (WS) {   // in place over the raw row (same thread, same 16 bytes)
      *reinterpret_cast<uint4*>(S.kvq + rr * 128 + partL * 16) = qw;
    } else {
      uint8_t* qs = (hk ? S.v : S.k) + (STAGES - 1) * STAGE_BYTES;
      *reinterpret_cast<uint4*>(qs + (j * 8 + gg) * 128 + partL * 16) = qw;
    }
  }
  if constexpr (QDIR) fence_proxy_async();   // the e4m3 q tile (st.shared) is read by the wgmma after the loop's first barrier

  if (has_new) {
#pragma unroll
    for (int ps = 0; ps < KPASS; ++ps) {
      const int rr = ps * RPP + (tidL >> 3);
      const bool live = rr < L * 4;
      const int kind = rr / (2 * L);
      const int jj = (rr % (2 * L)) >> 1, hh = rr & 1;
      float x[8];
      const uint4 kvv = live ? *reinterpret_cast<const uint4*>(&S.kvst[rr * 64 + partL * 8]) : make_uint4(0, 0, 0, 0);
      const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&kvv);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 f = __bfloat1622float2(h2[e]);
        x[2 * e] = f.x;
        x[2 * e + 1] = f.y;
      }
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) ss += x[e] * x[e];
      ss = oct_sum(ss);   // every lane of the warp (rows of both kinds share warps)
      float sc;
      if (kind == 0) {
        const float r = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
#pragma unroll
        for (int e = 0; e < 8; ++e) x[e] *= r;
        if (p.rope && live) {
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int i = partL * 4 + e;
            const float c = S.cs[jj * 32 + i], sn = S.sn[jj * 32 + i];
            const float a0 = x[2 * e], a1 = x[2 * e + 1];
            x[2 * e] = a0 * c - a1 * sn;
            x[2 * e + 1] = a0 * sn + a1 * c;
          }
        }
        sc = __fdividef(1.f, kscale);   // (MUFU.RCP: no IEEE slow-path call and its branches on the once-path;
      } else {                          // exact for the power-of-two scales)
        sc = __fdividef(1.f, vscale);
      }
      if (live) {
        uint32_t w0 = 0, w1 = 0;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          w0 |= (uint32_t)to_e4m3(x[e] * sc) << (8 * e);
          w1 |= (uint32_t)to_e4m3(x[4 + e] * sc) << (8 * e);
        }
        uint8_t* dst = (kind == 0 ? S.kn : S.vn) + jj * ROWB + hh * 64 + partL * 8;
        *reinterpret_cast<uint2*>(dst) = make_uint2(w0, w1);
        const int pos = seq + jj;
        if (pos >= start && pos < end) {
          const int64_t row = S.cloc[jj];
          uint8_t* g8 = const_cast<uint8_t*>(cache_row(kind == 0 ? p.k_cache : p.v_cache, p, row, (2 * pair + hh) * 64 + partL * 8));
          *reinterpret_cast<uint2*>(g8) = make_uint2(w0, w1);
        }
      }
    }
  }
  csync<WS, NT>();
  if (!WS && has_new) {
    // new-token rows of the tiles already issued (their cached rows are in flight with cp.async)
    for (int x = 0; x < (STAGES - 1) && x < ntiles; ++x) {
      const int st = x % STAGES;
#pragma unroll
      for (int k = 0; k < TILE / RPP; ++k) {
        const int r = crow0 + RPP * k;
        const int pos = start + x * TILE + r;
        if (pos >= seq && pos < end) {
          const int sw = ((cchunk ^ (r & 7)) << 4);
          *reinterpret_cast<uint4*>(S.k + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + cchunk * 16);
          *reinterpret_cast<uint4*>(S.v + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + cchunk * 16);
        }
      }
    }
    if (L == 8) fence_proxy_async();   // new keys written with st.shared, read by the QK wgmma
  }

  
  // ---- consumer state ----
  constexpr int MT = 1;                    // m-tiles per warp
  constexpr int NROW = (L == 1) ? 1 : 2;   // rows per thread
  // (lane-0 broadcasts, so ptxas keeps them in uniform registers instead of re-deriving them every tile)
  const int hk = __shfl_sync(0xffffffffu, (L == 1) ? (warp & 1) : ((warp & 7) >> 2), 0);      // L==8: warpgroup = KV head
  const int wsub = __shfl_sync(0xffffffffu, (L == 1) ? (warp >> 1) : (warp & 3), 0);    // L==1: key half; L==8: m-tile (tokens 2*wsub, 2*wsub+1)
  const int wgi = hk;          // named barrier of this warpgroup
  float o[MT][8][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int t = 0; t < 8; ++t)
#pragma unroll
      for (int e = 0; e < 4; ++e) o[mt][t][e] = 0.f;
  float m_r[NROW], l_r[NROW];
#pragma unroll
  for (int r = 0; r < NROW; ++r) { m_r[r] = -INFINITY; l_r[r] = 0.f; }
  // tiles with no window / causal / range edge skip the per-element mask
  const int lo_all = (L == 1 || !p.window) ? start : max(start, seq + (L - 1) - p.window + 1);
  const int hi_all = __shfl_sync(0xffffffffu, min(end - 1, seq), 0);

  if constexpr (L == 1) {
  // Q A-fragments (f16, pre-scaled to log2 units) for 4 k-steps of 16 dims: thread holds dims
  // 16*tig + 4*ks + {0,1} (a0) and + {2,3} (a2); the K B-fragments use the same dims.
  uint32_t qa[4][2];
  {
    const uint8_t* qs = WS ? S.kvq + hk * 8 * 128 : (hk ? S.v : S.k) + (STAGES - 1) * STAGE_BYTES;
    const uint4 l0 = *reinterpret_cast<const uint4*>(qs + g * 128 + 32 * tig);
    const uint4 l1 = *reinterpret_cast<const uint4*>(qs + g * 128 + 32 * tig + 16);
    // (scout: chunk words stored (0, 2, 1, 3): l0 = {ks0 a0, ks1 a0, ks0 a2, ks1 a2}, l1 the same for ks 2, 3)
    qa[0][0] = l0.x; qa[0][1] = l0.y; qa[1][0] = l0.z; qa[1][1] = l0.w;
    qa[2][0] = l1.x; qa[2][1] = l1.y; qa[3][0] = l1.z; qa[3][1] = l1.w;
  }

  for (int i = 0; i < ntiles; ++i) {
    if constexpr (WS) {
      if (ws_acquire(i, (tb + i) % STAGES)) csync<WS, NT>();
    } else {
      cp_wait<STAGES - 2>();
      __syncthreads();
      if (i + STAGES - 1 < ntiles) issue_kv(i + STAGES - 1, true);
      issue_idx(i + 2 * STAGES - 2);
      cp_commit();
    }

    const uint8_t* K = S.k + ((tb + i) % STAGES) * STAGE_BYTES;
    const uint8_t* V = S.v + ((tb + i) % STAGES) * STAGE_BYTES;
    const int t0 = start + i * TILE;
    const bool interior = (t0 >= lo_all) && (t0 + TILE - 1 <= hi_all);
    // warp: kv head hk, keys [32*wsub, 32*wsub+32) of the tile
    float s[4][4];
    uint32_t kw[4][4];
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int key = 32 * wsub + 8 * nt + g;
      const uint4 kb = lds128(K + key * ROWB + (((hk * 4 + tig) ^ (key & 7)) << 4));
      kw[nt][0] = kb.x; kw[nt][1] = kb.y; kw[nt][2] = kb.z; kw[nt][3] = kb.w;
    }
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {   // (scout: unpadded quads; even k-steps -> rows g of se, odd -> rows g+8 of so)
      float se[4] = {0.f, 0.f, 0.f, 0.f}, so[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        mma_f16(se, qa[2 * h][0], qa[2 * h][1], qa[2 * h + 1][0], qa[2 * h + 1][1], cvt2(kw[nt][2 * h]), cvt2h(kw[nt][2 * h]));
        mma_f16(so, qa[2 * h][0], qa[2 * h][1], qa[2 * h + 1][0], qa[2 * h + 1][1], cvt2(kw[nt][2 * h + 1]), cvt2h(kw[nt][2 * h + 1]));
      }
      s[nt][0] = se[0] + so[2]; s[nt][1] = se[1] + so[3]; s[nt][2] = 0.f; s[nt][3] = 0.f;
    }
    if (!interior) {
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int pos = t0 + 32 * wsub + 8 * nt + 2 * tig + e;
          if (pos >= end) s[nt][e] = -INFINITY;
        }
    }
    float mx = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) mx = fmaxf(mx, fmaxf(s[nt][0], s[nt][1]));
    mx = quad_max(mx);
    const float m_new = fmaxf(m_r[0], mx);
    if (m_new != -INFINITY) {
      const float alpha = ex2(m_r[0] - m_new);
      m_r[0] = m_new;
      float ls = 0.f;
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float pv = ex2(s[nt][e] - m_new);
          s[nt][e] = pv;
          ls += pv;
        }
      if constexpr (ONE) {   // (scout: the only tile: o = 0 and l = 0, alpha would only multiply zeros)
        l_r[0] = ls;
      } else {
        l_r[0] = l_r[0] * alpha + ls;
#pragma unroll
        for (int t = 0; t < 8; ++t) { o[0][t][0] *= alpha; o[0][t][1] *= alpha; }
      }
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {   // (timing probe only, wrong results)
        uint32_t b0[8], b1[8];
        load_v_frags(V, 32 * wsub + 16 * ks, hk, g, tig, b0, b1);
        const uint32_t a0 = pack_h2(s[2 * ks][0], s[2 * ks][1]);
        const uint32_t a2 = pack_h2(s[2 * ks + 1][0], s[2 * ks + 1][1]);
#pragma unroll
        for (int t = 0; t < 8; ++t) mma_f16(o[0][t], a0, 0u, a2, 0u, b0[t], b1[t]);
      }
    }
    if constexpr (WS) {   // this warp is done with the stage
      __syncwarp();
      mbar_arrive(&S.empty[(tb + i) % STAGES]);
    }
    PROF_TILE;
  }
  PROF_DUMP;
  pdl_launch_dependents();
  } else {
  // ---- L == 8 on sm_90: QK^T with FP8 wgmma straight from the FP8 K tile in shared memory (warpgroup =
  // KV head, 64 rows = 8 tokens x 8 heads, K-major 128B-swizzled B operand), PV with f16 mma.sync from a
  // per-tile f16 copy of V. Softmax with a lazily updated per-row offset; row sums from a ones column.
  uint8_t* V16 = S.kvq;
  uint32_t qa8[2][4];   // e4m3 A fragments, k-steps of 32 dims
  if constexpr (!QDIR) {
    const uint8_t* qs = (hk ? S.v : S.k) + (STAGES - 1) * STAGE_BYTES;   // f16 staging, row = j*8 + g
    const int r0 = 16 * wsub + g, r1 = r0 + 8;
    // producer mode: in place in S.kvq, row j*16 + hk*8 + g
    const uint8_t* q0 = WS ? S.kvq + ((r0 >> 3) * 16 + hk * 8 + (r0 & 7)) * 128 : qs + r0 * 128;
    const uint8_t* q1 = WS ? S.kvq + ((r1 >> 3) * 16 + hk * 8 + (r1 & 7)) * 128 : qs + r1 * 128;
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {        // dims 32*ks + 16*hf + 4*tig .. +3
        const int dofs = (32 * ks + 16 * hf + 4 * tig) * 2;
        const uint2 lo = *reinterpret_cast<const uint2*>(q0 + dofs);
        const uint2 hi = *reinterpret_cast<const uint2*>(q1 + dofs);
        qa8[ks][2 * hf] = pack_e4m3x4(lo.x, lo.y);
        qa8[ks][2 * hf + 1] = pack_e4m3x4(hi.x, hi.y);
      }
    }
       // the A tile for the SS QK^T wgmma
#pragma unroll
      for (int ks = 0; ks < 2; ++ks)
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          const int kb = 32 * ks + 16 * hf + 4 * tig;
          *reinterpret_cast<uint32_t*>(S.q8s + hk * 4096 + r0 * 64 + (((kb >> 4) ^ ((r0 >> 1) & 3)) << 4) + (kb & 15)) =
              qa8[ks][2 * hf];
          *reinterpret_cast<uint32_t*>(S.q8s + hk * 4096 + r1 * 64 + (((kb >> 4) ^ ((r1 >> 1) & 3)) << 4) + (kb & 15)) =
              qa8[ks][2 * hf + 1];
        }
      fence_proxy_async();   // read by the wgmma after the loop's first barrier
    
    
  }
  float cr[2] = {0.f, 0.f};   // per-row offsets (log2 units), rows g and g+8
  const float ssc = sl2;   // raw score -> log2 units
  // the lazy-max test on raw scores against (offset + 8) / ssc (ssc > 0), updated with the offset
  float thr[2] = {0.f, 0.f};
  float lacc[4] = {0.f, 0.f, 0.f, 0.f};
  bool first = true;
     // the zero init stays before the loop: ptxas had sunk these moves into the first QK^T
    float* oz = &o[0][0][0];   // wgmma's pipeline stage (warning C7515: every wgmma of the kernel serialized)
#pragma unroll
    for (int e = 0; e < 32; ++e) reg_fence(oz[e]);
#pragma unroll
    for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
  
  const int ckey = tid >> 2, cq = tid & 3;   // V conversion: key, two 16-byte chunks
  // FP8 PV: the per-CTA kernels (the flat kernel's register budget serializes its wgmmas: measured slower)
  // transpose roles (warpgroup = head): warp wq takes keys 16wq..16wq+15, lane = (q, w) of the ldmatrix fragments
  const int vwq = warp & 3, vq = lane & 3, vw = lane >> 2;   // (warp: uniform)
  const uint32_t vsel0 = (vw >= 4) ? 0x7531u : 0x6420u, vsel1 = (vw >= 4) ? 0x6420u : 0x7531u;
  const int vr0 = (vw >= 4) ? 1 : 0;   // rows d + vr0 then d + 1 - vr0 (bank spread)
  // hoisted addressing (registers are tight in the flat kernel): V^T row n = 16*dgrp + 2*vw + r has swizzle chunk
  // vwq ^ (vw & 3) for every dgrp, so the 8 stores per tile are one per-thread base plus immediates; the two
  // ldmatrix rows are key = 16*vwq + 8*(mat & 1) + (lane & 7) with chunk (4*hk + 2u + (mat >> 1)) ^ (lane & 7)
  const uint32_t vt_off0 = 128 * vw + ((vwq ^ (vw & 3)) << 4) + 4 * vq + 64 * vr0;
  const uint32_t vt_off1 = 128 * vw + ((vwq ^ (vw & 3)) << 4) + 4 * vq + 64 * (1 - vr0);
  const int vkey_t = 16 * vwq + 8 * ((lane >> 3) & 1) + (lane & 7);
  const uint32_t vld_off0 = vkey_t * ROWB + (((4 * hk + 0 + ((lane >> 3) >> 1)) ^ (lane & 7)) << 4);
  // the V^T stores as two stmatrix.x4 per thread (instead of eight 4-byte stores). For ldmatrix group u,
  // matrix m = 2*g2 + ab holds, in row j, the 16 key bytes of V^T row 16(2u+g2) + 2j + (ab ^ (j >= 4)) (the same
  // rows and the same half swap for bank spread as the per-thread stores: thread (q, w) supplies row j = w)
  const int stj = lane & 7, stm = lane >> 3;
  const uint32_t vst_off = 1024 * (stm >> 1) + 128 * stj + 64 * ((stm & 1) ^ (stj >> 2)) + ((vwq ^ (stj & 3)) << 4);
  // (per-request kernels: stmatrix measured vs16 -2.5%, vs1/vs32 -1.5%, vf1 -1%; the flat kernel: +0..1.5%)
  constexpr bool STSM_ON = !FLAT;
  // P fragments of the in-flight PV wgmma: kept live (fenced) until the next wgmma wait, so the compiler cannot
  // hand their registers to anything the next tile writes before that
  uint32_t pa[2][4] = {{0u, 0u, 0u, 0u}, {0u, 0u, 0u, 0u}};
  // per-tile addressing as loop-carried values instead of re-derived from (tb + i) every tile: stage and
  // full-barrier phase counters, wgmma descriptors as base + stage offset (the 14-bit address field takes it without
  // a carry: shared memory < 256 KB), the V^T buffer and its descriptor toggled, new-row tiles found once
  // per-request (non-producer) q8 kernels with the split PV and the consumers' own V^T
  constexpr bool LSYNCN = !WS && !FLAT;
  constexpr bool LSYNC = WS;
  int st_c = tb % STAGES;
  uint32_t ph_c = (uint32_t)(tb / STAGES) & 1u;
  // descriptor low words broadcast from lane 0: a shuffle result is uniform and is not rematerialized, so the
  // bases stay in uniform registers (the high words are constants); +st*512 / +2 never carry out of the low word
  const uint32_t qlo0 = __shfl_sync(0xffffffffu, (uint32_t)((smem_u32(S.q8s) + hk * 4096) >> 4) | 0x10000u, 0);
  const uint32_t klo0 = __shfl_sync(0xffffffffu, (uint32_t)((smem_u32(S.k) + hk * 64) >> 4) | 0x10000u, 0);
  const uint64_t qdesc0 = ((uint64_t)((512 >> 4) | (2u << 30)) << 32) | qlo0;
  const uint32_t vsrc0 = smem_u32(S.v);
  uint8_t* vtb_c = S.kvq + hk * 4608;
  const uintptr_t vtb_x = uintptr_t(S.kvq + hk * 4608) ^ uintptr_t(S.kvq + (2 + hk) * 4608);
  uint64_t vdesc_c = desc_sw64(smem_u32(S.kvq + hk * 4608));
  const uint64_t vdesc_x = vdesc_c ^ desc_sw64(smem_u32(S.kvq + (2 + hk) * 4608));
  const int i_new = seq > start ? (seq - start) / TILE : 0;   // first tile that can hold the block's new rows
  // interior tiles (no mask) are i in [i_lo, i_lo + i_span], found once (shuffled: kept uniform)
  uint32_t i_lo = 0, i_span = 0;
  

  // (producer mode): the common tiles -- no mask, no new rows (every key before the block's first new
  // position seq) -- are i in [r_lo, r_lo + r_span], found once; the per-tile test is then one compare, and t0, the
  // exact mask test and the new-row patch run only on the rare tiles
  uint32_t r_lo = 0x7fffffffu, r_span = 0u;
  if constexpr (WS) {
    const int hi2 = min(end - 1, seq - 1);
    const int a = (lo_all - start + TILE - 1) / TILE;
    const int z = hi2 - (TILE - 1) - start;
    const int bz = z >= 0 ? z / TILE : -1;
    r_lo = __shfl_sync(0xffffffffu, bz >= a ? (uint32_t)a : 0x7fffffffu, 0);
    r_span = __shfl_sync(0xffffffffu, bz >= a ? (uint32_t)(bz - a) : 0u, 0);
  }
  // (flat producer mode): the two KV heads' warpgroups take turns in the exponential phase (named barriers
  // 5 / 6), so the 4 consumer warps of an SMSP do not all queue on the MUFU unit at once (scout: the exps segment
  // takes ~650-820 cycles for ~110 instructions; both warpgroups start each tile together on the same stage)
  
  int i_cur = 0, cnt2 = 0;   // F2: this warpgroup's claimed tile, and how many it has taken
  
  const int nit = ntiles;
  constexpr int UNR8 = WS ? 2 : 1;   // (pragma arguments are not macro-expanded;
  // plain split-KV with many tiles per CTA (b32 q8 sliding: 9) gains from the unroll, 1-tile coop grids lose)
#pragma unroll (UNR8)
  for (int jj = 0; jj < nit; ++jj) {
    const int i = jj;   // S2: tiles 2jj and 2jj + 1; F2: claimed
    
      if constexpr (WS) {
      // tile i landed (+ new rows); previous tile's PV of this warpgroup done (its V16 half is free). The two
      // warpgroups (one per KV head) only meet when both must: tile 0 (q staging) and tiles with new rows.
      bool patched;
      
        mbar_wait_full(&S.full[st_c], ph_c);
        patched = ((uint32_t)i - r_lo > r_span) && (i >= i_new) && ws_patch_ni<L, NT, false>(&S, i, st_c, start, seq, end, tid);
        
       
      if (patched || i == 0) csync<WS, NT>();
      // (FP8 PV: V^T is double-buffered and each warp's wgmma wait already covers the PV that read this tile's
      // buffer, so the tile needs no warpgroup barrier before its transpose)
    } else {
      cp_wait<STAGES - 2>();   // (timing probe only, wrong results)
      __syncthreads();       // tile i landed; previous tile's PV done (V16 free)
    }
      
    

    const int st = st_c;
    
    float s[32];
    wgmma_fence();
    
      const uint32_t kl = klo0 + (uint32_t)(st * (STAGE_BYTES >> 4));
      const uint64_t kd = ((uint64_t)((1024 >> 4) | (1u << 30)) << 32) | kl;
      const uint64_t kd2 = ((uint64_t)((1024 >> 4) | (1u << 30)) << 32) | (kl + 2u);
      const uint64_t qd2 = ((uint64_t)((512 >> 4) | (2u << 30)) << 32) | (qlo0 + 2u);
      
       wgmma_m64n64k32_e4m3_ss(s, qdesc0, kd, 0);
      wgmma_m64n64k32_e4m3_ss(s, qd2, kd2, 1);
     
    wgmma_commit();
    if constexpr (!WS) {   // the copies for tile i + STAGES - 1
      if (i + STAGES - 1 < ntiles) issue_kv(i + STAGES - 1, true);
      issue_idx(i + 2 * STAGES - 2);
      cp_commit();
    }
    if constexpr (FLAT) {
      // the previous tile's PV (committed before this QK^T) is done: its P registers are free for the transpose
      // below (the flat kernel's 104-register consumers spilled there) and o / lacc are settled
      wgmma_wait<1>();
#pragma unroll
      for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
      float* od0 = &o[0][0][0];
#pragma unroll
      for (int e = 0; e < 32; ++e) reg_fence(od0[e]);
#pragma unroll
      for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
    }
    uint8_t* V16t = V16 + 0;   // PV reads it asynchronously: double buffer
    // this head's V^T for this tile (PV reads it asynchronously: double buffer per stream)
    uint8_t* vtb = vtb_c;
    
      if (jj == 0) {   // rows 64..71 of this head's two V^T buffers: e4m3 ones (0x38), so the PV
                                     // wgmma also produces the row sums (after the first barrier: the q staging shares this space)
        const int wt = tid & 127;
        *reinterpret_cast<uint2*>(S.kvq + ((0 + (wt >> 6)) * 2 + hk) * 4608 + 4096 + (wt & 63) * 8) =
            make_uint2(0x38383838u, 0x38383838u);
      }
    
         // meanwhile: this head's V tile -> V^T (FP8, 72 x 64, keys in P-register order: k = 4q + 2h + b for
        // key = 8h + 2q + b within each 16-key block), two ldmatrix.trans (b16 pairs) + byte permutes per thread
      const uint32_t vsrc = vsrc0 + st * STAGE_BYTES;
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        uint32_t r0, r1, r2, r3;
        ldsm_x4_t(vsrc + (u ? (vld_off0 ^ 32u) : vld_off0), r0, r1, r2, r3);   // (scout: off1 == off0 ^ 32)
        // r0/r1: dims group 2u (h = 0 / 1), r2/r3: dims group 2u+1
        if constexpr (STSM_ON) {
          stsm_x4(smem_u32(vtb) + vst_off + 2048 * u, prmt(r0, r1, vsel0), prmt(r0, r1, vsel1), prmt(r2, r3, vsel0),
                  prmt(r2, r3, vsel1));
        } else {
#pragma unroll
        for (int g2 = 0; g2 < 2; ++g2) {
          const uint32_t R0 = g2 ? r2 : r0, R1 = g2 ? r3 : r1;
          const uint32_t w0 = prmt(R0, R1, vsel0), w1 = prmt(R0, R1, vsel1);
          *reinterpret_cast<uint32_t*>(vtb + vt_off0 + 1024 * (2 * u + g2)) = w0;   // row 16(2u+g2) + 2vw + vr0
          *reinterpret_cast<uint32_t*>(vtb + vt_off1 + 1024 * (2 * u + g2)) = w1;   // row 16(2u+g2) + 2vw + 1 - vr0
        }
        }
      }
      if (!LSYNC && !LSYNCN) fence_proxy_async();   // V^T written with st.shared, read by the PV wgmma
     
      if (LSYNC || LSYNCN) {   // (/ N: fence + barrier just before the PV)
    } else wgsync(wgi);   // this head's V16 / V^T ready
                              // V16 ready
    wgmma_wait<0>();
#pragma unroll
    for (int e = 0; e < 32; ++e) reg_fence(s[e]);
    
#pragma unroll
      for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
    
    if constexpr (WS) {   // K (wgmma) and V (conversion) of the stage are consumed
      mbar_arrive(&S.empty[st]);
    }
    

    const bool common_t = WS && ((uint32_t)i - r_lo <= r_span);
    const int t0 = start + i * TILE;
    const bool interior = common_t || ((t0 >= lo_all) && (t0 + TILE - 1 <= hi_all));
    if (!interior) {
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const int pj = seq + 2 * wsub + hf;
        const int vlo = p.window ? max(start, pj - p.window + 1) : start;
        const int vhi = min(end - 1, pj);
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int pos = t0 + 8 * nt + 2 * tig + e;
            if (pos < vlo || pos > vhi) s[4 * nt + 2 * hf + e] = -INFINITY;
          }
      }
    }
    // after a segment's first tile the offset test runs on the packed P instead of on 32 row-max steps: an
    // e4m3 byte above 0x78 (P > 256, i.e. a score past the offset + 8) sets bit 7 when 0x07 is added to every byte
    // (P >= 0: no carry between bytes); the (rare) hit moves the offset and repacks that half (see the PV loop)
    constexpr bool PCHK = WS;
    const bool chk = PCHK && !first;
    float mlo = -INFINITY, mhi = -INFINITY;
    if (!chk) {
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      mlo = fmaxf(mlo, fmaxf(s[4 * nt], s[4 * nt + 1]));
      mhi = fmaxf(mhi, fmaxf(s[4 * nt + 2], s[4 * nt + 3]));
    }
    }
    // the threshold check needs no row reduction: any element past the offset + 8 shows in its own thread's
    // partial max; only the (rare) offset move and the first tile reduce over the row's quad
    if (!chk) {
      mlo *= ssc;   // scores -> log2 units (ssc > 0; ssc = 1 when the query was pre-scaled)
      mhi *= ssc;
    }
    float dlo, dhi;
    if (first) {
       mlo = quad_max(mlo); mhi = quad_max(mhi); 
      
      dlo = (mlo == -INFINITY) ? 0.f : mlo;
      dhi = (mhi == -INFINITY) ? 0.f : mhi;
      cr[0] = dlo; cr[1] = dhi;
      
      first = false;
    } else {
      dlo = cr[0]; dhi = cr[1];
      if (!PCHK && __any_sync(0xffffffffu, (mlo > cr[0] + 8.f || mhi > cr[1] + 8.f))) {
        // rare: a score grew past 2^8 relative to the offset -> move the offset up, rescale the state
         mlo = quad_max(mlo); mhi = quad_max(mhi);    // (warp-uniform branch: full-mask shuffles)
        
        const float nlo = fmaxf(mlo, cr[0]), nhi = fmaxf(mhi, cr[1]);
        const float alo = ex2(cr[0] - nlo), ahi = ex2(cr[1] - nhi);
        // (the previous tile's PV wgmma completed at this tile's wgmma wait)
#pragma unroll
        for (int t = 0; t < 8; ++t) { o[0][t][0] *= alo; o[0][t][1] *= alo; o[0][t][2] *= ahi; o[0][t][3] *= ahi; }
        lacc[0] *= alo; lacc[1] *= alo; lacc[2] *= ahi; lacc[3] *= ahi;   // (FP8 PV: the row sums live here too)
        cr[0] = nlo; cr[1] = nhi;
        dlo = nlo; dhi = nhi;
        
      }
    }
       // my turn for the exponentials
    
       // P (e4m3, registers, k-steps of 32 permuted keys) x V^T (e4m3, K-major, 72 columns: 64 dims + 8 copies of
        // the row sum) as an asynchronous wgmma; the next tile's wgmma wait covers it before P, o or V^T are reused
      constexpr bool PK2 = PCHK;
      if constexpr (PK2) {
        auto pack2 = [&]() {   // both halves of P from the scores (32 independent chains)
#pragma unroll
          for (int kk = 0; kk < 2; ++kk) {
            const float* sv = s + 16 * kk;
            pa[kk][0] = e4m3x4f(ex2(fmaf(sv[0], ssc, -dlo)), ex2(fmaf(sv[1], ssc, -dlo)), ex2(fmaf(sv[4], ssc, -dlo)), ex2(fmaf(sv[5], ssc, -dlo)));
            pa[kk][1] = e4m3x4f(ex2(fmaf(sv[2], ssc, -dhi)), ex2(fmaf(sv[3], ssc, -dhi)), ex2(fmaf(sv[6], ssc, -dhi)), ex2(fmaf(sv[7], ssc, -dhi)));
            pa[kk][2] = e4m3x4f(ex2(fmaf(sv[8], ssc, -dlo)), ex2(fmaf(sv[9], ssc, -dlo)), ex2(fmaf(sv[12], ssc, -dlo)), ex2(fmaf(sv[13], ssc, -dlo)));
            pa[kk][3] = e4m3x4f(ex2(fmaf(sv[10], ssc, -dhi)), ex2(fmaf(sv[11], ssc, -dhi)), ex2(fmaf(sv[14], ssc, -dhi)), ex2(fmaf(sv[15], ssc, -dhi)));
          }
        };
        pack2();
        if (chk) {
          // an e4m3 byte above 0x78 (P > 256: a score past its row's offset + 8) sets bit 7 when 0x07 is added
          constexpr uint32_t PB = 0x07070707u;
          const uint32_t ov = ((pa[0][0] + PB) | (pa[0][1] + PB) | (pa[0][2] + PB) | (pa[0][3] + PB) |
                               (pa[1][0] + PB) | (pa[1][1] + PB) | (pa[1][2] + PB) | (pa[1][3] + PB)) & 0x80808080u;
          if (__any_sync(0xffffffffu, ov != 0u)) {
            // rare: offsets up to the tile's row maxima, state rescaled (no PV of this tile issued yet; the previous
            // tile's finished at its end), both halves repacked (P <= 1 afterwards: one pass)
            float hlo = -INFINITY, hhi = -INFINITY;
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
              hlo = fmaxf(hlo, fmaxf(s[4 * nt], s[4 * nt + 1]));
              hhi = fmaxf(hhi, fmaxf(s[4 * nt + 2], s[4 * nt + 3]));
            }
            hlo = quad_max(hlo) * ssc;
            hhi = quad_max(hhi) * ssc;
            const float nlo = fmaxf(hlo, cr[0]), nhi = fmaxf(hhi, cr[1]);
            const float alo = ex2(cr[0] - nlo), ahi = ex2(cr[1] - nhi);
#pragma unroll
            for (int t = 0; t < 8; ++t) { o[0][t][0] *= alo; o[0][t][1] *= alo; o[0][t][2] *= ahi; o[0][t][3] *= ahi; }
            lacc[0] *= alo; lacc[1] *= alo; lacc[2] *= ahi; lacc[3] *= ahi;
            cr[0] = nlo; cr[1] = nhi;
            dlo = nlo; dhi = nhi;
            pack2();
          }
        }
        float* odk = &o[0][0][0];
#pragma unroll
        for (int e = 0; e < 32; ++e) reg_fence(odk[e]);
#pragma unroll
        for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
#pragma unroll
        for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
        if constexpr (LSYNC) {   // every warp's V^T for this tile written (and visible to the async proxy)
          fence_proxy_async();
          wgsync(wgi);
        }
        wgmma_fence();
        
          wgmma_m64n72k32_e4m3(odk, lacc, pa[0], vdesc_c);
          wgmma_m64n72k32_e4m3(odk, lacc, pa[1], vdesc_c + 2);
         
        wgmma_commit();
      }
#pragma unroll
      for (int kk = 0; kk < (PK2 ? 0 : 2); ++kk) {
        const float* sv = s + 16 * kk;
#define EXQ(x, q) ex2(x)
        bool rchk = chk;   // one retry at most: after the offset move every finite P is <= 1 (NaN / garbage scores
                           // would otherwise retrigger the check forever: the SASS oracle's value-perturbing probe hung)
#pragma unroll 1
        for (;;) {   // (repeated once after a rare offset move; otherwise a single pass)
        pa[kk][0] = e4m3x4f(EXQ(fmaf(sv[0], ssc, -dlo), 0), EXQ(fmaf(sv[1], ssc, -dlo), 1), EXQ(fmaf(sv[4], ssc, -dlo), 2), EXQ(fmaf(sv[5], ssc, -dlo), 3));
        pa[kk][1] = e4m3x4f(EXQ(fmaf(sv[2], ssc, -dhi), 0), EXQ(fmaf(sv[3], ssc, -dhi), 1), EXQ(fmaf(sv[6], ssc, -dhi), 2), EXQ(fmaf(sv[7], ssc, -dhi), 3));
        pa[kk][2] = e4m3x4f(EXQ(fmaf(sv[8], ssc, -dlo), 0), EXQ(fmaf(sv[9], ssc, -dlo), 1), EXQ(fmaf(sv[12], ssc, -dlo), 2), EXQ(fmaf(sv[13], ssc, -dlo), 3));
        pa[kk][3] = e4m3x4f(EXQ(fmaf(sv[10], ssc, -dhi), 0), EXQ(fmaf(sv[11], ssc, -dhi), 1), EXQ(fmaf(sv[14], ssc, -dhi), 2), EXQ(fmaf(sv[15], ssc, -dhi), 3));
        if constexpr (!PCHK) break;
        else {
          if (!rchk) break;
          {
            // (test hook, a hit at P > 1 instead of P > 256: exercises the offset move on most tiles)
            constexpr uint32_t PB = 0x07070707u;
            const uint32_t ov = ((pa[kk][0] + PB) | (pa[kk][1] + PB) | (pa[kk][2] + PB) | (pa[kk][3] + PB)) & 0x80808080u;
            if (!__any_sync(0xffffffffu, ov != 0u)) break;
            {
              // rare: a score of this half passed its row's offset + 8: offsets up to the half's row maxima,
              // state rescaled (half 0's PV wgmma finished first), this half repacked
              float hlo = fmaxf(fmaxf(fmaxf(sv[0], sv[1]), fmaxf(sv[4], sv[5])), fmaxf(fmaxf(sv[8], sv[9]), fmaxf(sv[12], sv[13])));
              float hhi = fmaxf(fmaxf(fmaxf(sv[2], sv[3]), fmaxf(sv[6], sv[7])), fmaxf(fmaxf(sv[10], sv[11]), fmaxf(sv[14], sv[15])));
              hlo = quad_max(hlo) * ssc;
              hhi = quad_max(hhi) * ssc;
              const float nlo = fmaxf(hlo, cr[0]), nhi = fmaxf(hhi, cr[1]);
              const float alo = ex2(cr[0] - nlo), ahi = ex2(cr[1] - nhi);
              if ((kk == 1)) {   // (a PV wgmma may still accumulate into o)
                wgmma_wait<0>();
                float* odw = &o[0][0][0];
#pragma unroll
                for (int e = 0; e < 32; ++e) reg_fence(odw[e]);
#pragma unroll
                for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
              }
#pragma unroll
              for (int t = 0; t < 8; ++t) { o[0][t][0] *= alo; o[0][t][1] *= alo; o[0][t][2] *= ahi; o[0][t][3] *= ahi; }
              lacc[0] *= alo; lacc[1] *= alo; lacc[2] *= ahi; lacc[3] *= ahi;
              cr[0] = nlo; cr[1] = nhi;
              dlo = nlo; dhi = nhi;
              rchk = false;
            }
          }
        }
        }   // (pack / check loop)
#undef EXQ
           // each half of P goes to the tensor cores as soon as it is packed
          float* od = &o[0][0][0];
          if constexpr (LSYNCN) {   // every warp's V^T for this tile written (and visible to the async proxy)
            if (kk == 0) {
              fence_proxy_async();
              wgsync(wgi);
            }
          }
          if (kk == 0) {
#pragma unroll
            for (int e = 0; e < 32; ++e) reg_fence(od[e]);
#pragma unroll
            for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
          }
#pragma unroll
          for (int e = 0; e < 4; ++e) asm volatile("" : "+r"(pa[kk][e]));
          wgmma_fence();
          wgmma_m64n72k32_e4m3(od, lacc, pa[kk], vdesc_c + 2 * kk);
          
          wgmma_commit();
        
      }
      
      
      float* od = &o[0][0][0];
      
           // done before the next tile (no wgmma in flight across the loop edge)
          wgmma_wait<0>();
#pragma unroll
          for (int e = 0; e < 32; ++e) reg_fence(od[e]);
#pragma unroll
          for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
#pragma unroll
          for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
        
          // (!PVSPLIT)
     
    PROF_TILE;
    
    
      if (++st_c == STAGES) { st_c = 0; ph_c ^= 1u; }
      vtb_c = reinterpret_cast<uint8_t*>(reinterpret_cast<uintptr_t>(vtb_c) ^ vtb_x);
      vdesc_c ^= vdesc_x;
    
  }
  PROF_DUMP;
  pdl_launch_dependents();
  
  wgmma_wait<0>();
  
#pragma unroll
    for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
  
  {
    float* od = &o[0][0][0];
#pragma unroll
    for (int e = 0; e < 32; ++e) reg_fence(od[e]);
#pragma unroll
    for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
  }
  
  m_r[0] = first ? -INFINITY : cr[0];
  m_r[1] = first ? -INFINITY : cr[1];
  l_r[0] = 0.25f * lacc[0];   // every lane of the quad holds the full row sum; the epilogue quad-sums
  l_r[1] = 0.25f * lacc[2];
  
  }

  // ---- merge: rescale to the fixed reference 2^0 and accumulate with FP32 atomics ----
  bool do_red = true;
     // stream 1's state was merged into stream 0
  if (L == 1) {
    // warps 2,3 (second key half) hand their state to warps 0,1 (same heads) through shared memory
    // (outside cluster mode both halves add their own rescaled state to the workspace instead)
    cp_wait<0>();
    csync<WS, NT>();
    float* xs = reinterpret_cast<float*>(S.k) + ((warp & 1) * 32 + lane) * 18;
    if (wsub == 1) {
      xs[0] = m_r[0];
      xs[1] = l_r[0];
#pragma unroll
      for (int t = 0; t < 8; ++t) { xs[2 + 2 * t] = o[0][t][0]; xs[3 + 2 * t] = o[0][t][1]; }
    }
    csync<WS, NT>();
    if (wsub == 0) {
      const float m1 = xs[0], l1 = xs[1];
      const float mm = fmaxf(m_r[0], m1);
      const float f0 = (m_r[0] == -INFINITY) ? 0.f : ex2(m_r[0] - mm);
      const float f1 = (m1 == -INFINITY) ? 0.f : ex2(m1 - mm);
      m_r[0] = mm;
      l_r[0] = l_r[0] * f0 + l1 * f1;
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        o[0][t][0] = o[0][t][0] * f0 + xs[2 + 2 * t] * f1;
        o[0][t][1] = o[0][t][1] * f0 + xs[3 + 2 * t] * f1;
      }
    } else {
      do_red = false;
    }
  }
  if (M_CL) {
    cp_wait<0>();
    csync<WS, NT>();
    float* PB = reinterpret_cast<float*>(S.k);
    float* PL = PB + L * 16 * 64;
    // two tile streams (Cfg::S2): stream 0 writes its rows, then stream 1 (same rows) adds its own
#pragma unroll 1
    for (int ph = 0; ph < 1; ++ph) {
    if (ph == 1) csync<WS, NT>();
    const bool mine = true;
#pragma unroll
    for (int r = 0; r < NROW; ++r) {
      const int mt = r >> 1, hf = r & 1;
      const int j = (L == 1) ? 0 : 2 * (wsub + mt) + hf;
      const int row = j * 16 + hk * 8 + g;
      const float lsum = quad_sum(l_r[r]);
      if (do_red && mine) {
        const float f = (m_r[r] == -INFINITY) ? 0.f : ex2(fminf(m_r[r], 126.f));
        if constexpr (L == 1) {   // permuted dims: n-tile t, column c <-> dim 16*tig + 8*c + t
#pragma unroll
          for (int c = 0; c < 2; ++c)
#pragma unroll
            for (int t4 = 0; t4 < 8; t4 += 4)
              *reinterpret_cast<float4*>(&PB[row * 64 + 16 * tig + 8 * c + t4]) =
                  make_float4(o[mt][t4][2 * hf + c] * f, o[mt][t4 + 1][2 * hf + c] * f,
                              o[mt][t4 + 2][2 * hf + c] * f, o[mt][t4 + 3][2 * hf + c] * f);
        } else {                  // natural dims: n-tile t, column c <-> dim 8*t + 2*tig + c
#pragma unroll
          for (int t = 0; t < 8; ++t) {
            float2* d2 = reinterpret_cast<float2*>(&PB[row * 64 + 8 * t + 2 * tig]);
            const float2 pv = ph ? *d2 : make_float2(0.f, 0.f);
            *d2 = make_float2(pv.x + o[mt][t][2 * hf] * f, pv.y + o[mt][t][2 * hf + 1] * f);
          }
        }
        if (tig == 0) PL[row] = (ph ? PL[row] : 0.f) + lsum * f;
      }
    }
    }
    csync<WS, NT>();
    cluster_finish<L>(p, S, b, pair, n_keys, chunk);
    return true;
  }
  // BULKRED (q_len 1 per-request kernels, not the one-tile-per-split builds): the partial goes to shared memory (the V
      // stages; the K stages hold the exchange) in the workspace's own row layout ([16][72]: dims, the row sum in column 64,
      // pad) and one TMA bulk reduction adds it into the workspace below
      constexpr bool BULKRED = PADWS;
      constexpr int PSTR = WSR<PADWS>::STR;                        // smem partial row stride (floats)
      float* pbuf = reinterpret_cast<float*>(S.v);
      if (do_red) {   // (one branch, not one per row: the post-loop merge runs straight-line from a cold cache)
    #pragma unroll
      for (int r = 0; r < NROW; ++r) {
    const int mt = r >> 1, hf = r & 1;
    const int j = (L == 1) ? 0 : 2 * (wsub + mt) + hf;
    const int hh = hk * 8 + g;
    const float lsum = quad_sum(l_r[r]);
    const float f = (m_r[r] == -INFINITY) ? 0.f : ex2(fminf(m_r[r], 126.f));
    float* acc = p.ws_acc + ((ix_t)((bp * p.nc + copy) * L + j) * 16 + hh) * WSR<PADWS>::STR;
        if constexpr (BULKRED) {   // permuted dims (see below): float4 of n-tiles t4..t4+3 at dim 16*tig + 8*c + t4
    #pragma unroll
          for (int c = 0; c < 2; ++c)
    #pragma unroll
            for (int t4 = 0; t4 < 8; t4 += 4)
              *reinterpret_cast<float4*>(pbuf + hh * PSTR + 16 * tig + 8 * c + t4) =
                  make_float4(o[mt][t4][2 * hf + c] * f, o[mt][t4 + 1][2 * hf + c] * f,
                              o[mt][t4 + 2][2 * hf + c] * f, o[mt][t4 + 3][2 * hf + c] * f);
          if (tig == 0) {   // the row sum in column 64, the pad zeroed (the bulk op adds whole rows)
            *reinterpret_cast<float4*>(pbuf + hh * PSTR + WSL) = make_float4(lsum * f, 0.f, 0.f, 0.f);
            *reinterpret_cast<float4*>(pbuf + hh * PSTR + WSL + 4) = make_float4(0.f, 0.f, 0.f, 0.f);
          }
          continue;
        } else if constexpr (L == 1) {
    #pragma unroll
          for (int c = 0; c < 2; ++c)
#pragma unroll
        for (int t4 = 0; t4 < 8; t4 += 4)
          red_v4(acc + 16 * tig + 8 * c + t4, o[mt][t4][2 * hf + c] * f, o[mt][t4 + 1][2 * hf + c] * f,
                 o[mt][t4 + 2][2 * hf + c] * f, o[mt][t4 + 3][2 * hf + c] * f);
    } else {
      // 16-byte atomics (half the L2 atomic operations of 8-byte ones): lane pairs (tig, tig ^ 1) swap halves so
      // each lane holds 4 consecutive dims: even tig keeps t even, odd tig keeps t odd (dims 8t + 4(tig >> 1) + 0..3)
      const bool odd = tig & 1;
#pragma unroll
      for (int t = 0; t < 8; t += 2) {
        // mine: (t, 2tig..2tig+1) and (t+1, same); partner's: same t's at 2(tig^1)..
        const float a0 = o[mt][t][2 * hf] * f, a1 = o[mt][t][2 * hf + 1] * f;
        const float b0 = o[mt][t + 1][2 * hf] * f, b1 = o[mt][t + 1][2 * hf + 1] * f;
        // even lane sends its t+1 pair, odd lane sends its t pair
        const float s0 = odd ? a0 : b0, s1 = odd ? a1 : b1;
        const float r0 = __shfl_xor_sync(0xffffffffu, s0, 1), r1 = __shfl_xor_sync(0xffffffffu, s1, 1);
        // even lane: t (dims 2tig, 2tig+1, then the partner's); odd lane: t+1 (the partner's, then mine) -- selects,
        // one atomic, no branch
        red_v4(acc + 8 * (t + (odd ? 1 : 0)) + 4 * (tig >> 1), odd ? r0 : a0, odd ? r1 : a1, odd ? b0 : r0, odd ? b1 : r1);
      }
    }
    if (tig == 0) red_f32(WSL_AT(PADWS, ((bp * p.nc + copy) * L + j) * 16 + hh), lsum * f);
      }
      }
      if constexpr (BULKRED) {
        fence_proxy_async();   // the partial's st.shared, visible to the async proxy
        csync<WS, NT>();
        if (tid == 0) {
          bulk_red_add_f32(p.ws_acc + ((ix_t)((bp * p.nc + copy) * L) * 16) * WSR<PADWS>::STR, pbuf,
                           L * 16 * WSR<PADWS>::STR * 4);
          bulk_commit();
          bulk_wait_all();
          fence_proxy_async_global();   // ... performed, and ordered (through the CTA barrier) before the ticket's release
        }
      }
      // (last-CTA finalizes): no per-thread fences; one acq_rel atomic by thread 0 after the CTA barrier
  // releases every thread's reds (cumulativity through bar.sync, as CUTLASS's semaphore release) and acquires the
  // others' for the last CTA's reads
  const bool mode1 = !((FLAT && (L == 8)) && M_COOP) && !(!FLAT && M_COOP);
  // the same for the cooperative grids (the barrier's arrive is an acq_rel atomic after a CTA barrier)
  if (!(mode1 || !mode1)) gpu_fence();
  csync<WS, NT>();
  if ((FLAT && (L == 8)) && M_COOP) {
    // flat cooperative mode: every request is finalized after all CTAs have merged all their segments
  } else if (!FLAT && M_COOP) {
    // ---- all splits of this (request, kv pair) are resident: wait for every split's merge, then each
    // CTA finalizes its own slice of the rows (instead of one CTA doing all of them)
    // (the slice arithmetic first: it does not depend on the merge, so its divisions leave the barrier's chain)
    const int split_idx = (int)blockIdx.x;
    int nact = fdiv_i(n_keys + chunk - 1, chunk);    // splits with keys
    int R = fdiv_i(L * 16 + nact - 1, nact);          // rows per CTA
    // keep both divisions above the barrier wait (ptxas otherwise sinks them below the spin, where
    // they are first used: scout's v200 census found them after the poll); volatile asm keeps program order
    asm volatile("" : "+r"(nact), "+r"(R));
      if (tid == 0) {
      // two key counters alternating per launch (parity = this slot's epoch, read in the prologue): the last split
      // to arrive resets the OTHER counter (unused until the next launch) and bumps the epoch, so nobody needs a
      // departure counter (no atomic round trip at the very end of the kernel)
      const int ep = S.sense;
      int* c = p.ws_cnt + ((ep & 1) ? 12288 : 8192) + bp;
      const int mine = end - start;
      int old;
      asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n" : "=r"(old) : "l"(c), "r"(mine) : "memory");
      
      if (old + mine == n_keys) {
        p.ws_cnt[((ep & 1) ? 8192 : 12288) + bp] = 0;
        p.ws_cnt[24576 + bp] = ep + 1;
      } else {
        int v;
        SPIN_T0;
        for (;;) {
          asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(c) : "memory");
          if (v >= n_keys) break;
          SPIN_CHECK;
          __nanosleep(64);
        }
      }
    } else if (tid == 0) {
      atomicAdd(p.ws_cnt + bp, end - start);
      int v;
      SPIN_T0;
      for (int spin = 0;; ++spin) {
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(p.ws_cnt + bp) : "memory");
        if (v >= n_keys) break;
        SPIN_CHECK;
        
        __nanosleep(64);
      }
    }
    csync<WS, NT>();
    const int r_lo = split_idx * R, r_hi = min(L * 16, r_lo + R);
    const int part = tid & 7;
    for (int rbase = r_lo; rbase < r_hi; rbase += NT / 8) {   // warp-uniform trip count (shuffles below)
      const int rr = rbase + (tid >> 3);
      const bool act = rr < r_hi;
      const int j = act ? rr >> 4 : 0, hh = act ? rr & 15 : 0;
      const uint4 graw = act ? *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + part * 8)
                             : make_uint4(0, 0, 0, 0);
      float y[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      float lsum = 0.f;
      sum_copies<L, PADWS>(p, bp, j, hh, part, act, y, lsum);
      const float inv = lsum > 0.f ? __fdividef(vscale, lsum) : 0.f;
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) { y[e] = pr8(y[e] * inv); ss += y[e] * y[e]; }
      ss += __shfl_xor_sync(0xffffffffu, ss, 1);
      ss += __shfl_xor_sync(0xffffffffu, ss, 2);
      ss += __shfl_xor_sync(0xffffffffu, ss, 4);
      const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
      const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw);
      uint4 ov;
      uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 gt = __bfloat1622float2(g2[e]);
        const float s0 = sigm(gt.x), s1 = sigm(gt.y);
        __nv_bfloat162 r2 = __floats2bfloat162_rn(y[2 * e] * rn * s0, y[2 * e + 1] * rn * s1);
        ow[e] = *reinterpret_cast<uint32_t*>(&r2);
      }
      if (act) *reinterpret_cast<uint4*>(p.out + (ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + part * 8) = ov;
    }
    
  } else {
  if (tid == 0) {
    const int mine = end - start;
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n" : "=r"(old) : "l"(p.ws_cnt + bp), "r"(mine) : "memory");
    
    S.flag = (old + mine == n_keys);
  }
  csync<WS, NT>();
  if (S.flag) {
  

  // ---- last CTA: sum the copies, normalize, rmsnorm * sigmoid(gate), store; reset the workspace ----
  if (p.nc == 1) {
    fin_nc1<L, PADWS, NT>(p, b, pair, bp, vscale, 0);
  } else
  {
    const int part = tid & 7;
    for (int rr = tid >> 3; rr < L * 16; rr += NT / 8) {
      const int j = rr >> 4, hh = rr & 15;
      float y[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      float lsum = 0.f;
      // gate row in flight with the copies (one L2 round trip per row instead of two)
      const uint4 graw = *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + part * 8);
      sum_copies<L, PADWS>(p, bp, j, hh, part, true, y, lsum);
      const float inv = lsum > 0.f ? __fdividef(vscale, lsum) : 0.f;
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) { y[e] = pr8(y[e] * inv); ss += y[e] * y[e]; }
      ss = oct_sum(ss);
      const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
      const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw);
      uint4 ov;
      uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 gt = __bfloat1622float2(g2[e]);
        const float s0 = sigm(gt.x), s1 = sigm(gt.y);
        __nv_bfloat162 r2 = __floats2bfloat162_rn(y[2 * e] * rn * s0, y[2 * e + 1] * rn * s1);
        ow[e] = *reinterpret_cast<uint32_t*>(&r2);
      }
      *reinterpret_cast<uint4*>(p.out + (ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + part * 8) = ov;
    }
  }
  if (tid == 0) p.ws_cnt[bp] = 0;
  if (p.dbg && tid == 0) p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * 16 + 6] = 1;
  }
  }
     // consumers' shared memory is reused by the next segment
   __syncthreads();                   // shared memory is reused by the next segment
  }
  return true;
}

// Flat grids whose CTAs are all resident: once every CTA has merged all its segments (grid-wide count),
// each CTA finalizes its own slice of all requests' rows (sum the copies, normalize, rmsnorm * gate, store,
// zero the workspace). Runs on the NT consumer threads; the last CTA out resets the two counters.
template <int L, bool WS, int NT>
__device__ __forceinline__ void flat_finalize(const Params& p, const int* sense_s) {
  const int tid = threadIdx.x;
  pdl_wait();   // a CTA without segments has not waited yet (the gate rows are qkv)
  const int G = gridDim.x * gridDim.y * gridDim.z;
  const int c = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
  int* cnt = p.ws_cnt + 30000;
     // (thread 0's acq_rel arrive releases the CTA's reds)
  csync<WS, NT>();
  if (tid == 0) {
    // two grid counters alternating per launch (parity = the epoch thread 0 read at kernel start): the last CTA
    // to arrive resets the other counter and bumps the epoch; no departure counter at the end of the kernel
    cp_wait<0>();
    const int ep = *reinterpret_cast<const volatile int*>(sense_s);
    int* ac = p.ws_cnt + 30020 + (ep & 1);
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;\n" : "=r"(old) : "l"(ac) : "memory");
    
    if (old + 1 == G) {
      p.ws_cnt[30020 + ((ep & 1) ^ 1)] = 0;
      p.ws_cnt[30022] = ep + 1;
      
    } else {
      int v;
      SPIN_T0;
      for (;;) {
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(ac) : "memory");
        if (v >= G) break;
        SPIN_CHECK;
        __nanosleep(64);
      }
    }
  } else   if (tid == 0) {
    atomicAdd(cnt, 1);
    int v;
    SPIN_T0;
    for (;;) {
      asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(cnt) : "memory");
      if (v >= G) break;
      SPIN_CHECK;
      __nanosleep(64);
    }
  }
  csync<WS, NT>();
  const int pairs = p.Hkv >> 1;
  const int rows = p.B * pairs * L * 16;
  const int R = fdiv_i(rows + G - 1, G);
  const int r_lo = c * R, r_hi = min(rows, r_lo + R);
  const float vscale = __ldg(p.v_scale);
  const int part = tid & 7;
  for (int rbase = r_lo; rbase < r_hi; rbase += NT / 8) {   // warp-uniform trip count (shuffles below)
    const int r = rbase + (tid >> 3);
    const bool act = r < r_hi;
    const int bp = act ? r / (L * 16) : 0, rr = act ? r % (L * 16) : 0;
    const int j = rr >> 4, hh = rr & 15;
    const int b = (pairs == 1) ? bp : fdiv_i(bp, pairs), pair = bp - b * pairs;
    const uint4 graw = act ? *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + part * 8)
                           : make_uint4(0, 0, 0, 0);
    float y[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float lsum = 0.f;
    sum_copies<L, false>(p, bp, j, hh, part, act, y, lsum);
    const float inv = lsum > 0.f ? __fdividef(vscale, lsum) : 0.f;
    float ss = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) { y[e] = pr8(y[e] * inv); ss += y[e] * y[e]; }
    ss += __shfl_xor_sync(0xffffffffu, ss, 1);
    ss += __shfl_xor_sync(0xffffffffu, ss, 2);
    ss += __shfl_xor_sync(0xffffffffu, ss, 4);
    const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
    const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw);
    uint4 ov;
    uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 gt = __bfloat1622float2(g2[e]);
      const float s0 = sigm(gt.x), s1 = sigm(gt.y);
      __nv_bfloat162 r2 = __floats2bfloat162_rn(y[2 * e] * rn * s0, y[2 * e + 1] * rn * s1);
      ow[e] = *reinterpret_cast<uint32_t*>(&r2);
    }
    if (act) *reinterpret_cast<uint4*>(p.out + (ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + part * 8) = ov;
  }
  
}

template <int L, bool FLAT, int MODE = 0>
__global__ void __launch_bounds__(Cfg<L, FLAT>::NTX, Cfg<L, FLAT>::MINB) attn_kernel(const __grid_constant__ Params p) {
  constexpr int NT = Cfg<L, FLAT>::NT;
  constexpr int STAGES = Cfg<L, FLAT>::STAGES;
  constexpr int IDX_SLOTS = 2 * STAGES;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  // 1024-byte aligned base: the 128B-swizzled tiles must start on swizzle-pattern boundaries (wgmma)
  Smem<L, FLAT>& S = *reinterpret_cast<Smem<L, FLAT>*>(smem_raw + ((1024u - (smem_u32(smem_raw) & 1023u)) & 1023u));
  const int pair = blockIdx.z;
  if (!FLAT) {
    // grid (n_splits, B): request b, split s of its window / context
    const int split = blockIdx.x, b = blockIdx.y;
    // (MODE | 8 builds, chosen by the host for one request with page size 1): SGLang's FA3 backend builds
    // page_table = req_to_token[req_pool_indices, :max_seq_len_k] (sglang/srt/layers/attention/
    // flashattention_backend.py, init_forward_metadata: decode / target-verify paths), so with one request its row
    // is exactly seq_len + q_len entries wide. The page table's width is a shape: speculate seq = width - q_len and
    // issue the window's page indices in the same trip as cache_seqlens; the real length is checked (segment) before
    // the indices are used for any K/V address or any softmax work, and a miss re-runs everything with the real one
    const int seq_ld = __ldg(p.seq_lens + b);
    constexpr bool SPEC = (MODE & 8) != 0;
    int seq = SPEC ? p.max_pages - L : seq_ld;
    for (;;) {
    const int lo = p.window ? max(0, seq + 1 - p.window) : 0;
    const int n_keys = seq + L - lo;
    int chunk = fdiv_i(n_keys + p.n_splits - 1, p.n_splits);
    chunk = (chunk + 15) & -16;
    // plain split-KV: whole tiles per split (the per-CTA tile count is what costs); trailing splits may get no keys:
    // they skip the workspace merge (the merge counts keys, the cooperative finalize counts the splits with keys).
    // Not for clusters (an empty cluster CTA still runs the cluster epilogue: measured slower)
    if ((MODE & 3) == 0 ? p.cs <= 1 : (MODE & 3) != 2) {
      const int ntl = ((n_keys + 63) >> 6) + p.n_splits - 1;
      // ceil(2^32 / n_splits) from the host (exact for ntl < 2^20, n_splits <= 1024): one IMAD.HI instead
      // of the ~20-instruction fdiv_i on the once-path (scout's census)
      chunk = (p.ns_magic ? (int)__umulhi((unsigned)ntl, p.ns_magic) : ntl) << 6;
      
    }
    int start = lo + split * chunk;
    int end = min(seq + L, start + chunk);
    // splits beyond the SM count share an SM with an earlier split, which gets the warp schedulers'
    // priority: the first-wave splits take w0/1024 of a second-wave split's tiles (only when every split still gets
    // at least one tile, so all splits have keys and the cooperative finalize counts all of them)
    
    if (start >= end) {
      if (SPEC && seq != seq_ld) { seq = seq_ld; continue; }   // (an empty split only on the real length)
      if ((MODE & 3) == 0 ? p.cs > 1 : (MODE & 3) == 2) {   // still part of its cluster: contributes an empty partial, finalizes its slice
        pdl_wait();     // (the gate rows are qkv)
        cluster_stage_gate<L>(p, S, b, pair);
        if (threadIdx.x == 0) cp_async4(smem_u32(&S.scl[1]), p.v_scale);
        cp_commit();
        float* PB = reinterpret_cast<float*>(S.k);
        float* PL = PB + L * 16 * 64;
        for (int e = threadIdx.x; e < L * 16 * 64; e += NT) PB[e] = 0.f;
        for (int e = threadIdx.x; e < L * 16; e += NT) PL[e] = 0.f;
        cp_wait<0>();
        __syncthreads();
        cluster_finish<L>(p, S, b, pair, n_keys, chunk);
      }
      return;
    }
    if (segment<L, FLAT, 0, MODE>(p, S, b, pair, split == 0, start, end, seq, n_keys, split - fdiv_i(split, p.nc) * p.nc,
                                  chunk, 0, SPEC ? seq_ld : seq))
      return;
    seq = seq_ld;   // (miss: today's path with the real length)
    }
  }
  if constexpr (FLAT && (L == 8)) {
    if (p.coop && threadIdx.x == 0) {   // the grid barrier's epoch, long before it is needed
      cp_async4(smem_u32(&S.sense), p.ws_cnt + 30022);
      cp_commit();
    }
  }
  // the per-SM registration's atomics go out from warp 1 while warp 0 builds the prefix sums
  // the same uneven per-SM shares for the non-cooperative (q_len 1) flat grids. No grid barrier ends those,
  // so the registration state is generation-tagged: a monotonic launch counter picks one of two state copies, and
  // the launch's last registrant clears the other copy (the previous launch's) for the next launch
  int w0g_rank = 2, w0g_par = 0, w0g_gen = 0;
  unsigned w0_smid = 0;
  int w0_rank = 0;
  
  
  // grid (G): CTA c takes keys [c*T/G, (c+1)*T/G) of all requests' keys laid end to end (equal work)
  // warp 0 computes the key range and the segment list right after its prefix sums (one request per lane)
  constexpr bool fpart = L == 1;   // (compile time: no run-time branch in the main flow)
  if constexpr (L == 1 && Cfg<L, FLAT>::WS) {
    if (threadIdx.x == 32) {   // the first segment's stage barriers (segment skips its init for that segment)
      for (int st = 0; st < Cfg<L, FLAT>::STAGES; ++st) {
        mbar_init_n(&S.full[st], Cfg<L, FLAT>::NP);
        mbar_init_n(&S.empty[st], Cfg<L, FLAT>::NT);
      }
      mbar_fence_init();
    }
  }
  if (threadIdx.x < 32) {
    int run = 0;
    int incl_l = 0, n_l = 0, seq_l = 0;
    for (int b0 = 0; b0 < p.B; b0 += 32) {
      const int b = b0 + (int)threadIdx.x;
      int n = 0;
      if (b < p.B) {
        const int seq = __ldg(p.seq_lens + b);
        const int lo = p.window ? max(0, seq + 1 - p.window) : 0;
        n = seq + L - lo;
        seq_l = seq;
      }
      int incl = n;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int t = __shfl_up_sync(0xffffffffu, incl, o);
        if ((int)threadIdx.x >= o) incl += t;
      }
      if (b < p.B) S.cum[b + 1] = run + incl;
      run += __shfl_sync(0xffffffffu, incl, 31);
      incl_l = incl;
      n_l = n;
    }
    if (threadIdx.x == 0) S.cum[0] = 0;
    if (fpart && p.B > 32) {   // many requests: the generic partition by lane 0 (after the whole prefix sum)
      __syncwarp();
      if (threadIdx.x == 0) {
        constexpr int SEGKF = (L == 8) ? 600 : 400;
        const int64_t Tg = (int64_t)S.cum[p.B] + (SEGKF > 0 ? (int64_t)(p.B - 1) * SEGKF : 0);
        int csg = (int)((int64_t)blockIdx.x * Tg / gridDim.x), ceg = (int)((int64_t)(blockIdx.x + 1) * Tg / gridDim.x);
        if (SEGKF > 0) {
          auto vmapg = [&](int v) -> int {
            int lo_b = 0, hi_b = p.B - 1;
            while (lo_b < hi_b) {
              const int mid = (lo_b + hi_b + 1) >> 1;
              if (S.cum[mid] + mid * SEGKF <= v) lo_b = mid; else hi_b = mid - 1;
            }
            return min(v - lo_b * SEGKF, S.cum[lo_b + 1]);
          };
          csg = vmapg(csg);
          ceg = vmapg(ceg);
        }
        int b = 0, n = 0;
        while (S.cum[b + 1] <= csg) ++b;
        for (; b < p.B && S.cum[b] < ceg && n < MAXSEG; ++b) {
          S.seg[n][0] = b;
          S.seg[n][1] = max(csg, S.cum[b]) - S.cum[b];
          S.seg[n][2] = min(ceg, S.cum[b + 1]) - S.cum[b];
          ++n;
        }
        S.nseg = n;
        S.fcs = csg;
        S.fce = ceg;
      }
    } else if (fpart) {   // (one iteration above: lane b holds cum[b + 1] = incl_l, cum[b] = incl_l - n_l; lanes >= B: n = 0)
      const unsigned FULL = 0xffffffffu;
      const int lane_f = (int)threadIdx.x;
      constexpr int SEGKF = (L == 8) ? 600 : 400;
      const int excl = incl_l - n_l;
      const int64_t Tf = (int64_t)__shfl_sync(FULL, incl_l, 31) + (SEGKF > 0 ? (int64_t)(p.B - 1) * SEGKF : 0);
      const int cf = blockIdx.x, Gf = gridDim.x;
      int csf = (int)((int64_t)cf * Tf / Gf), cef = (int)((int64_t)(cf + 1) * Tf / Gf);
      if (SEGKF > 0) {   // virtual -> real: the last request whose virtual start (cum[b] + b * SEGK) is <= v
        const int vstart = excl + lane_f * SEGKF;
        const unsigned ms = __ballot_sync(FULL, lane_f < p.B && vstart <= csf);
        const unsigned me = __ballot_sync(FULL, lane_f < p.B && vstart <= cef);
        const int bs = 31 - __clz(ms), be = 31 - __clz(me);   // (lane 0 always qualifies: virtual start 0)
        const int is = __shfl_sync(FULL, incl_l, bs), ie = __shfl_sync(FULL, incl_l, be);
        csf = min(csf - bs * SEGKF, is);
        cef = min(cef - be * SEGKF, ie);
      }
      // segment list: the requests with cum[b + 1] > cs and cum[b] < ce (a contiguous run of lanes)
      const bool in = lane_f < p.B && incl_l > csf && excl < cef;
      const unsigned mi = __ballot_sync(FULL, in);
      const int rank = __popc(mi & ((1u << lane_f) - 1u));
      if (in && rank < MAXSEG) {
        S.seg[rank][0] = lane_f;
        S.seg[rank][1] = max(csf, excl) - excl;
        S.seg[rank][2] = min(cef, incl_l) - excl;
      }
      if (lane_f == 0) {
        S.nseg = min(__popc(mi), MAXSEG);
        S.fcs = csf;
        S.fce = cef;
      }
      if (mi != 0u && csf < cef) {   // the first segment's first STAGES tiles of page indices toward L2
        const int b0 = __ffs(mi) - 1;
        const int seqb = __shfl_sync(FULL, seq_l, b0), exb = __shfl_sync(FULL, excl, b0), inb = __shfl_sync(FULL, incl_l, b0);
        const int lob = p.window ? max(0, seqb + 1 - p.window) : 0;
        const int s0 = max(csf, exb) - exb + lob, s1 = min(cef, inb) - exb + lob;
        const int cend = min(min(s1, seqb), s0 + Cfg<L, FLAT>::STAGES * TILE);
        if (cend > s0) {
          const long long e0 = (long long)b0 * p.max_pages + (s0 >> p.lp);
          const long long e1 = (long long)b0 * p.max_pages + ((cend - 1) >> p.lp);
          const long long l0 = e0 >> 5, l1 = e1 >> 5;   // 128-byte lines of int32 page indices
          if (l0 + lane_f <= l1) asm volatile("prefetch.global.L2 [%0];" ::"l"(p.page_table + ((l0 + lane_f) << 5)));
        }
      }
    }
  }
  __syncthreads();
  const int64_t T0 = S.cum[p.B];
  // every request boundary costs a CTA that crosses it a whole extra segment (q staging, pipeline refill,
  // merge: ~7 us at q_len 8), so the shares are cut in a virtual key space with SEGK keys inserted before each
  // request after the first, then mapped back (a CTA crossing a boundary gets SEGK fewer real keys)
  constexpr int SEGK = (L == 8) ? 600 : 400;
  const int64_t T = T0 + (int64_t)(p.B - 1) * SEGK;
  const int c = blockIdx.x, G = gridDim.x;
  // (experiment: the first-dispatched half of the grid takes the second half of the keys)
  const int cpart = c;
  int cs, ce;
  if constexpr (fpart) {   // (computed by warp 0)
    cs = S.fcs;
    ce = S.fce;
  } else {
    cs = (int)((int64_t)cpart * T / G);
    ce = (int)((int64_t)(cpart + 1) * T / G);
  }   // (a float-estimate + correction
  // version without the 64-bit division call measured slower: q_len 1 b16 full +3 us, likely code layout)
  
  int vc = c;
  
  
  if (!fpart) {   // virtual -> real key positions
    auto vmap = [&](int v) -> int {
      int lo_b = 0, hi_b = p.B - 1;   // the last request whose virtual start is <= v
      while (lo_b < hi_b) {
        const int mid = (lo_b + hi_b + 1) >> 1;
        if (S.cum[mid] + mid * SEGK <= v) lo_b = mid; else hi_b = mid - 1;
      }
      return min(v - lo_b * SEGK, S.cum[lo_b + 1]);
    };
    cs = vmap(cs);
    ce = vmap(ce);
  }
  if (cs >= ce) {
    if constexpr (L == 8) {
      if (p.coop) {   // still part of the grid-wide finalize
        if constexpr (Cfg<L, FLAT>::WS) {
          if (threadIdx.x < Cfg<L, FLAT>::NT) flat_finalize<L, true, Cfg<L, FLAT>::NT>(p, &S.sense);
        } else {
          flat_finalize<L, false, Cfg<L, FLAT>::NTX>(p, &S.sense);
        }
      }
    }
    return;
  }
  
  
  // segment list in shared memory (keeps the per-segment loop state out of registers)
  if (threadIdx.x == 0 && !fpart) {
    int b = 0, n = 0;
    while (S.cum[b + 1] <= cs) ++b;
    for (; b < p.B && S.cum[b] < ce && n < MAXSEG; ++b) {
      S.seg[n][0] = b;
      S.seg[n][1] = max(cs, S.cum[b]) - S.cum[b];
      S.seg[n][2] = min(ce, S.cum[b + 1]) - S.cum[b];
      ++n;
    }
    S.nseg = n;
  }
  if (!fpart) __syncthreads();   // (warp 0 wrote the list before the barrier above)
  const int nseg = S.nseg;
  auto run_segments = [&](auto role) {
    constexpr int R = decltype(role)::value;
    int tb = 0;
#pragma unroll 1
    for (int k = 0; k < nseg; ++k) {
      const int b = S.seg[k][0];
      const int seq = __ldg(p.seq_lens + b);
      const int lo = p.window ? max(0, seq + 1 - p.window) : 0;
      const int n_keys = seq + L - lo;
      const int s0 = S.seg[k][1] + lo, s1 = S.seg[k][2] + lo;
      
      if (s0 < s1) {
        segment<L, FLAT, R>(p, S, b, pair, s0 == lo, s0, s1, seq, n_keys, vc - fdiv_i(vc, p.nc) * p.nc, 0,
                            0, 0,
                            L == 1 && Cfg<L, FLAT>::WS && k == 0);
        tb += (s1 - s0 + TILE - 1) / TILE;
      }
      
    }
       // the producer leaves with its copies landed
  };
  if constexpr (Cfg<L, FLAT>::WS) {
    // separate code for the two roles, so the producer warpgroup can hand its registers to the consumers
    if (threadIdx.x >= Cfg<L, FLAT>::NT) {
      asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(Cfg<L, FLAT>::PREGS));
      run_segments(std::integral_constant<int, 1>());
    } else {
      asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(Cfg<L, FLAT>::CREGS));
      run_segments(std::integral_constant<int, 2>());
      if constexpr (L == 8) {
        if (p.coop) flat_finalize<L, true, Cfg<L, FLAT>::NT>(p, &S.sense);
      }
    }
  } else {
    run_segments(std::integral_constant<int, 0>());
    if constexpr (FLAT && (L == 8)) {
      if (p.coop) flat_finalize<L, false, Cfg<L, FLAT>::NTX>(p, &S.sense);
    }
  }
}

}  // namespace

template <int L>
void launch(const Params& p0, dim3 grid, cudaStream_t stream, int cs, bool pdl) {
  Params p = p0;
  static bool init = false;
  const int sm_f = (int)sizeof(Smem<L, true>) + 1024, sm_n = (int)sizeof(Smem<L, false>) + 1024;
  const int sm = p.flat ? sm_f : sm_n;
  static int occ_c = 0, occ_f = 0, nsm = 0;
  static int occ_c1 = 0, occ_sp = 0, occ_sp1 = 0;   // the coop ONE / SPEC / SPEC-ONE instances
  typedef void (*KernFn)(const Params);
  // per-request kernels, one build per launch mode (1 plain split-KV, 2 clusters, 3 cooperative)
  static const KernFn kn[4] = {attn_kernel<L, false, 2>, attn_kernel<L, false, 1>,
                               attn_kernel<L, false, 2>, attn_kernel<L, false, 3>};
  static const KernFn kn1[4] = {0, attn_kernel<L, false, 5>, 0, attn_kernel<L, false, 7>};
  static const KernFn ksp[4] = {0, attn_kernel<L, false, 9>, 0, attn_kernel<L, false, 11>};
  static const KernFn ksp1[4] = {0, attn_kernel<L, false, 13>, 0, attn_kernel<L, false, 15>};
  if (!init) {
    for (int m = 1; m < 4; ++m)
      cudaFuncSetAttribute(kn[m], cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);
    cudaFuncSetAttribute(kn1[1], cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);
    cudaFuncSetAttribute(kn1[3], cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);
    for (int m = 1; m < 4; m += 2) {
      cudaFuncSetAttribute(ksp[m], cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);
      cudaFuncSetAttribute(ksp1[m], cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);
    }
    cudaFuncSetAttribute(attn_kernel<L, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_f);
    cudaFuncSetAttribute(kn[2], cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_c, kn[3], Cfg<L, false>::NTX, sm_n);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_c1, kn1[3], Cfg<L, false>::NTX, sm_n);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_sp, ksp[3], Cfg<L, false>::NTX, sm_n);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_sp1, ksp1[3], Cfg<L, false>::NTX, sm_n);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_f, attn_kernel<L, true>, Cfg<L, true>::NTX, sm_f);
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    init = true;
  }
  // the cooperative finalize waits for every split of a request: only when the whole grid is resident
  Params q = p;
  if (q.coop && (long long)grid.x * grid.y * grid.z > (long long)(p.flat ? occ_f : occ_c) * nsm) q.coop = 0;
  const bool cl = cs > 1;
  KernFn kern = p.flat ? (KernFn)attn_kernel<L, true> : kn[cl ? 2 : (q.coop ? 3 : 1)];
  if (!p.flat && !cl) {
    // every split holds at most one tile: whole tiles per split (TCHUNK) and at least as many splits as the longest
    // request's tiles (sliding windows: window + L - 1 keys)
    const long long kmax = p.window ? std::min<long long>((long long)p.window + L - 1, (long long)p.max_pages * p.P)
                                    : (long long)p.max_pages * p.P;
    if ((long long)p.n_splits * 64 >= kmax) kern = kn1[q.coop ? 3 : 1];
  }
  // one request, page size 1: the speculative-length builds (; the kernel verifies and falls back)
  if (!p.flat && !cl && p.B == 1 && p.P == 1 && p.max_pages >= L && (int)grid.y == 1) {
    const bool one = kern != kn[q.coop ? 3 : 1];
    kern = one ? ksp1[q.coop ? 3 : 1] : ksp[q.coop ? 3 : 1];
  }
  if (q.coop && !p.flat && !cl) {
    // the instance actually launched must hold the whole grid, else its grid barrier would wait forever
    int occ_k = occ_c;
    KernFn plain = kn[1];
    if (kern == kn1[3]) { occ_k = occ_c1; plain = kn1[1]; }
    if (kern == ksp[3]) { occ_k = occ_sp; plain = ksp[1]; }
    if (kern == ksp1[3]) { occ_k = occ_sp1; plain = ksp1[1]; }
    if ((long long)grid.x * grid.y * grid.z > (long long)occ_k * nsm) {
      q.coop = 0;
      kern = plain;
    }
  }
  if (cl || pdl) {
    // cluster dims and / or programmatic stream serialization (PDL: this grid may start while the preceding
    // kernel finishes; the kernel waits with griddepcontrol.wait before reading qkv)
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(p.flat ? Cfg<L, true>::NTX : Cfg<L, false>::NTX, 1, 1);
    cfg.dynamicSmemBytes = sm;   // (one cluster CTA per SM)
    cfg.stream = stream;
    cudaLaunchAttribute attr[2];
    int na = 0;
    if (cl) {
      attr[na].id = cudaLaunchAttributeClusterDimension;
      attr[na].val.clusterDim.x = cs;
      attr[na].val.clusterDim.y = 1;
      attr[na].val.clusterDim.z = 1;
      ++na;
    }
    if (pdl) {
      attr[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr[na].val.programmaticStreamSerializationAllowed = 1;
      ++na;
    }
    cfg.attrs = attr;
    cfg.numAttrs = na;
    cudaLaunchKernelEx(&cfg, kern, q);
  } else {
    kern<<<grid, p.flat ? Cfg<L, true>::NTX : Cfg<L, false>::NTX, sm, stream>>>(q);
  }
}

typedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                   const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                   CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
// [rows, row_bytes] byte view of a cache, boxes of one 128-byte row (one KV pair), 128-byte swizzle

void run(torch::Tensor qkv, torch::Tensor seq_lens, torch::Tensor cache_loc, torch::Tensor k_cache,
         torch::Tensor v_cache, torch::Tensor page_table, torch::Tensor k_scale, torch::Tensor v_scale,
         torch::Tensor inv_freq, torch::Tensor out, torch::Tensor ws_acc, torch::Tensor ws_l,
         torch::Tensor ws_cnt, int64_t B, int64_t L, int64_t Hq, int64_t Hkv, int64_t P, int64_t window,
         int64_t rope, int64_t n_splits, int64_t flat, int64_t nc, double scale_log2,
         std::optional<torch::Tensor> dbg, int64_t cs, int64_t ncl,
         int64_t tma) {
  Params p;
  p.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
  p.seq_lens = seq_lens.data_ptr<int>();
  p.cache_loc = cache_loc.data_ptr<int64_t>();
  p.k_cache = reinterpret_cast<uint8_t*>(k_cache.data_ptr());
  p.v_cache = reinterpret_cast<uint8_t*>(v_cache.data_ptr());
  p.page_table = page_table.data_ptr<int>();
  p.k_scale = k_scale.data_ptr<float>();
  p.v_scale = v_scale.data_ptr<float>();
  p.inv_freq = inv_freq.data_ptr<float>();
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.ws_acc = ws_acc.data_ptr<float>();
  p.ws_l = ws_l.data_ptr<float>();
  p.ws_cnt = ws_cnt.data_ptr<int>();
  p.B = (int)B; p.Hq = (int)Hq; p.Hkv = (int)Hkv; p.P = (int)P;
  TORCH_CHECK(P > 0 && (P & (P - 1)) == 0, "page size must be a power of two");
  TORCH_CHECK(k_cache.numel() / (Hkv * 64) < (1LL << 31), "cache rows must fit in 31 bits");
  p.lp = 0;
  while ((1 << p.lp) < P) ++p.lp;
  p.max_pages = (int)page_table.size(1);
  p.window = (int)window; p.rope = (int)rope; p.n_splits = (int)n_splits; p.flat = (int)flat; p.nc = (int)nc;
  p.ns_magic = (n_splits >= 2 && n_splits <= 1024) ? (unsigned)(((1ull << 32) + (uint64_t)n_splits - 1) / (uint64_t)n_splits) : 0u;
  TORCH_CHECK(n_splits <= 1024, "n_splits <= 1024");
  p.dbg = dbg.has_value() ? reinterpret_cast<long long*>(dbg->data_ptr<int64_t>()) : nullptr;
  p.cs = (int)cs; p.ncl = (int)ncl;
  p.ld_qkv = (int)qkv.size(-1);
  p.row_bytes = (int)(Hkv * 64);
  TORCH_CHECK((p.row_bytes & (p.row_bytes - 1)) == 0, "Hkv * 64 must be a power of two");
  p.lrb = 0;
  while ((1 << p.lrb) < p.row_bytes) ++p.lrb;
  p.off32 = (k_cache.numel() < (1LL << 32) && v_cache.numel() < (1LL << 32)) ? 1 : 0;   // larger caches: 64-bit offsets
  TORCH_CHECK(ws_acc.numel() < (1LL << 31), "workspace too large");
  TORCH_CHECK(B * (Hkv / 2) <= 4096, "at most 4096 (request, kv pair) slots in the counter workspace");
  TORCH_CHECK(qkv.numel() < (1LL << 31), "qkv too large");
  TORCH_CHECK((out.numel() < (1LL << 31) && page_table.numel() < (1LL << 31)), "out / page table too large");
  p.scale_log2 = (float)scale_log2;
  dim3 grid((unsigned)n_splits, flat ? 1u : (unsigned)B, (unsigned)(Hkv / 2));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  p.coop = (int)((tma >> 1) & 1);
  const bool pdl = ((tma >> 16) & 1);   // programmatic dependent launch (off by default)
  if (L == 1) launch<1>(p, grid, stream, (int)cs, pdl);
  else launch<8>(p, grid, stream, (int)cs, pdl);
}
"""


CPP_SRC = r"""
void run(torch::Tensor qkv, torch::Tensor seq_lens, torch::Tensor cache_loc, torch::Tensor k_cache,
         torch::Tensor v_cache, torch::Tensor page_table, torch::Tensor k_scale, torch::Tensor v_scale,
         torch::Tensor inv_freq, torch::Tensor out, torch::Tensor ws_acc, torch::Tensor ws_l,
         torch::Tensor ws_cnt, int64_t B, int64_t L, int64_t Hq, int64_t Hkv, int64_t P, int64_t window,
         int64_t rope, int64_t n_splits, int64_t flat, int64_t nc, double scale_log2,
         std::optional<torch::Tensor> dbg, int64_t cs, int64_t ncl,
         int64_t tma);
"""


_MOD = None
_INV_FREQ = {}
_WS = {}
_NUM_SMS = {}


def _cached(cache, key, make):
    """Memoized builder for device properties and small derived tensors."""
    value = cache.get(key)
    if value is None:
        value = make()
        cache[key] = value
    return value


# Per-shape launch overrides measured on the judge's H200:
# (batch, q_len, sliding, kv pairs) -> plan fields.
_TUNED = {
    (32, 8, True, 1): dict(n_splits=4, nc=1, coop=1),
    (1, 1, False, 1): dict(nc=8, coop=1, n_splits=252),
}



def _build(tag="", flags=()):
    # Fixed build: sm_90a only.
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name=f"okc_attn_v298bulk{tag}_90a",
        cpp_sources=CPP_SRC,
        cuda_sources=CUDA_SRC,
        functions=["run"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a",
                           "--expt-relaxed-constexpr", *flags],
        verbose=False,
    )


def _module():
    global _MOD
    if _MOD is None:
        _MOD = _build()
    return _MOD


def _inv_freq(theta, device, n_pairs=HEAD_DIM // 2):
    """RoPE inverse frequencies, 1 / theta^(2i / head_dim), one per pair."""

    def make():
        if theta:
            t = 1.0 / (theta ** (torch.arange(0, 2 * n_pairs, 2, device=device).float() / (2 * n_pairs)))
        else:
            t = torch.zeros(n_pairs, device=device)
        return t.float().contiguous()

    return _cached(_INV_FREQ, (float(theta), str(device), n_pairs), make)


def _workspace(n_rows, device):
    """Persistent FP32 accumulators (zero between calls: the last CTA resets them)."""
    key = str(device)
    ws = _WS.get(key)
    if ws is None or ws[0].numel() < n_rows * WS_ROW:
        n = max(n_rows, WORKSPACE_MIN_ROWS)
        ws = (torch.zeros(n * WS_ROW, dtype=torch.float32, device=device),
              torch.zeros(n, dtype=torch.float32, device=device),
              torch.zeros(n, dtype=torch.int32, device=device))
        _WS[key] = ws
    return ws


def _num_sms(device):
    return _cached(_NUM_SMS, str(device), lambda: torch.cuda.get_device_properties(device).multi_processor_count)


def _splits(tiles, quota):
    """Splits per (request, KV pair): at least 1, at most one per tile."""
    return max(1, min(tiles, quota))


def _copies(n, cap):
    """Workspace copies: at least 1, at most cap."""
    return max(1, min(n, cap))


def _headroom(target):
    """The per-request grids run slightly below the CTA target."""
    return (target * PLAN_HEADROOM_NUM) // PLAN_HEADROOM_DEN


def _plan(B, pairs, L, window, max_pages, P, device):
    """Launch plan for a shape: n_splits CTAs per (request, KV pair) or, flat, per KV pair over all
    requests; clusters of cs CTAs reduce through DSMEM (ncl clusters per request); nc workspace copies."""
    sms = _num_sms(device)
    target = 2 * sms
    if not window and 1 < B <= MAX_BATCH:
        # Full layer, several requests: equal shares of all requests' keys, one CTA pass per SM pair.
        return dict(n_splits=max(1, target // pairs), flat=1, cs=0, ncl=1, nc=1)
    keys_max = min(window + L - 1, max_pages * P) if window else max_pages * P
    tiles = max(1, (keys_max + TILE_KEYS - 1) // TILE_KEYS)
    reqs = B * pairs
    if not window:
        # One long request: plain split-KV over 2 CTAs per SM (clusters slower here).
        if L == 8:
            n = _splits(tiles, _headroom(target) // reqs)
            return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=_copies(n, 8))
        n = _splits(tiles, target // reqs)
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=_copies(n, 4), coop=0)
    if L == 1 and reqs < 24:
        # Few requests, sliding: plain split-KV with one tile per CTA and a single workspace copy.
        n = _splits(tiles, target // reqs)
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1)
    if L == 8 and reqs < 16:
        # q_len 8, few requests: plain split-KV with one tile per CTA and the cooperative finalize
        # (every CTA normalizes a slice of the rows) beats clusters.
        n = _splits(tiles, target // reqs)
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1, coop=1)
    if L == 8 and reqs >= 16:
        # q_len 8, many requests: plain split-KV with about one CTA per SM, plus the cooperative finalize.
        n = _splits(tiles, (target if reqs >= 24 else sms) // reqs)
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1, coop=1)
    if reqs >= 24:
        # Many requests: clusters no longer fit one CTA per SM, so plain split-KV spreads evenly.
        n = _splits(tiles, target // reqs)
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1)
    # Sliding window: one cluster per (request, KV pair); keep the grid small enough that all
    # clusters are resident at once.
    cs = 16
    limit = 2 * sms if L == 1 else sms
    if L == 8 and target == sms:
        limit = (sms * 15) // 16   # one CTA per SM: clusters of 8 fit 15 per 132 SMs, so stay below
    while cs > 4 and reqs * cs > limit:
        cs //= 2
    ncl = 1
    while cs * ncl > tiles and cs > 4:
        cs //= 2
    if cs * ncl > tiles or cs < 4:
        n = _splits(tiles, target // reqs)
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=_copies(n, 4))
    return dict(n_splits=cs * ncl, flat=0, cs=cs, ncl=ncl, nc=_copies(ncl, 4))


def _launch(data, dbg=None, _mod=None, pdl=None, **override):
    cfg = data["config"]
    B, L = cfg["batch"], cfg["q_len"]
    hq, hkv, d, P = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"], cfg["page_size"]
    qkv = data["qkv"]
    device = qkv.device
    pairs = hkv // 2
    window = cfg["window"]
    page_table = data["page_table"]
    out = torch.empty((B, L, hq * d), dtype=torch.bfloat16, device=device)
    plan = _plan(B, pairs, L, window, page_table.shape[1], P, device)
    tuned = _TUNED.get((B, L, bool(window), pairs))
    if tuned and not override:
        override = tuned
    if override:
        plan = dict(plan, **override)
        if "cs" in override and override["cs"] > 1:
            plan["flat"] = 0
            if "ncl" not in override:
                plan["ncl"] = max(1, plan["n_splits"] // override["cs"])
            plan["n_splits"] = plan["cs"] * plan["ncl"]
            if "nc" not in override:
                plan["nc"] = _copies(plan["ncl"], 4)
        elif "n_splits" in override and "nc" not in override:
            plan["nc"] = 1 if plan["flat"] else _copies(plan["n_splits"], 4)
    n_splits, flat, cs, ncl, nc = plan["n_splits"], plan["flat"], plan["cs"], plan["ncl"], plan["nc"]
    # Cooperative finalize (all CTAs resident): only for the long single-request grids, whose last-CTA
    # finalize reads and zeroes tens of KB from one SM; the short sliding grids finalize faster in one CTA.
    if flat:   # grid-wide finalize after all segments (the kernel re-checks residency)
        coop = int(plan.get("coop", int(L == 8 and
                                        n_splits * pairs <= 2 * _num_sms(device))))
    else:
        coop = int(plan.get("coop", 0 if (cs > 1 or n_splits < 64) else
                            int(n_splits * B * pairs <= 2 * _num_sms(device))))
    use_pdl = bool(pdl)
    # tma packs the kernel-mode bitfield: bit 1 = cooperative finalize, bit 16 = PDL.
    tma = int(plan.get("tma", 0)) | (coop << COOP_BIT) | (int(use_pdl) << PDL_BIT)
    if flat or cs <= 1:
        cs, ncl = 0, 1
    ws_acc, ws_l, ws_cnt = _workspace(B * pairs * nc * L * Q_HEADS_PER_PAIR, device)
    scale_log2 = cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E
    (_mod or _module()).run(qkv, data["seq_lens"], data["cache_loc"], data["k_cache"], data["v_cache"], page_table,
                  data["k_scale"], data["v_scale"], _inv_freq(cfg["rope_theta"], device), out, ws_acc, ws_l,
                  ws_cnt, B, L, hq, hkv, P, window, 1 if cfg["rope_theta"] else 0, n_splits, flat, nc, scale_log2,
                  dbg, cs, ncl, tma)
    return out


def _supported(cfg):
    P = cfg["page_size"]
    return (cfg["head_dim"] == HEAD_DIM and cfg["num_q_heads"] == 8 * cfg["num_kv_heads"] and cfg["num_kv_heads"] % 2 == 0
            and cfg["q_len"] in SUPPORTED_Q_LENS and cfg["batch"] <= MAX_BATCH and P > 0 and (P & (P - 1)) == 0
            and cfg["batch"] * (cfg["num_kv_heads"] // 2) <= MAX_REQUEST_PAIRS)


def _rms(x, eps=1e-5):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _rope(x, pos, theta):
    d = x.shape[-1]
    inv_freq = _inv_freq(theta, x.device, d // 2)
    ang = pos.float()[..., None] * inv_freq
    cos, sin = ang.cos().unsqueeze(-2), ang.sin().unsqueeze(-2)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack([even * cos - odd * sin, even * sin + odd * cos], -1).flatten(-2)


def _fallback(data):
    """Plain PyTorch path for shapes the CUDA kernel does not cover (other q_len / head layouts)."""
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d, page = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"], cfg["page_size"]
    q_in, k_in, v_in, gate = data["qkv"].split([hq * d, hkv * d, hkv * d, hq * d], -1)
    pos = data["positions"].view(batch, length)
    q = _rms(q_in.reshape(batch, length, hq, d)) * cfg["qk_scale_factor"]
    k = _rms(k_in.reshape(batch, length, hkv, d))
    if cfg["rope_theta"]:
        q, k = _rope(q, pos, cfg["rope_theta"]), _rope(k, pos, cfg["rope_theta"])
    q = q.to(torch.bfloat16)
    table = data["page_table"].long()
    slots = data["cache_loc"].view(batch, length)
    k_cache = data["k_cache"].view(-1, hkv, d)
    v_cache = data["v_cache"].view(-1, hkv, d)
    k_scale, v_scale = data["k_scale"], data["v_scale"]
    # The new K/V land in the cache, exactly as the CUDA path writes them.
    k_cache[slots] = (k / k_scale).to(torch.float8_e4m3fn)
    v_cache[slots] = (v_in.reshape(batch, length, hkv, d).float() / v_scale).to(torch.float8_e4m3fn)
    t = torch.arange(table.shape[1] * page, device=q.device)
    out = torch.empty((batch, length, hq, d), dtype=torch.bfloat16, device=q.device)
    for b in range(batch):
        rows = table[b].clamp(min=0)[t // page] * page + t % page
        keys = k_cache[rows].float() * k_scale
        vals = v_cache[rows].float() * v_scale
        visible = t[None, :] <= pos[b][:, None]
        if cfg["window"]:
            visible &= t[None, :] > pos[b][:, None] - cfg["window"]
        att = torch.einsum("lkgd,skd->lkgs", q[b].view(length, hkv, hq // hkv, d).float(), keys) * cfg["sm_scale"]
        att = att.masked_fill(~visible[:, None, None, :], float("-inf")).softmax(-1)
        out[b] = torch.einsum("lkgs,skd->lkgd", att, vals).reshape(length, hq, d)
    out = _rms(out) * torch.sigmoid(gate.float()).view(batch, length, hq, d)
    return out.reshape(batch, length, hq * d).to(torch.bfloat16)


def _fits(data):
    """Sizes the CUDA path's 32-bit indexing covers (caches of any size are fine: 64-bit offsets past 4 GB)."""
    cfg = data["config"]
    out_elems = cfg["batch"] * cfg["q_len"] * cfg["num_q_heads"] * cfg["head_dim"]
    return (data["qkv"].numel() < INDEX_LIMIT and data["page_table"].numel() < INDEX_LIMIT
            and out_elems < INDEX_LIMIT)


def _vj_kernel_fn(data: Inputs) -> torch.Tensor:
    if not _supported(data["config"]) or not _fits(data):
        return _fallback(data)
    return _launch(data)

# MEASUREMENT BUILD ONLY (his piecewise-linear exp): never submit this file.


# ====================================================================================================================
# ZhimaoL/sglang verify-attention kernels (Apache-2.0): wst_ct14b for full layers, vfg + merge for sliding layers.
# His CUDA with the edits marked "OKC:" (exact exp2 on packed f16 instead of his piecewise-linear exp, no TMA maps, trimmed
# to the routes used here). The vjeux kernels above stay the path for every other shape.
# ====================================================================================================================
import hashlib as _zh_hashlib
import threading as _zh_threading
from torch.utils.cpp_extension import load_inline as _zh_load_inline

ZH_WST_MIN_BATCH = 16      # full-layer verify (q_len 8, page 1, TP8, FP8): batches from here run wst_ct14b
ZH_WST_MAX_BATCH = 24
ZH_VFG_MIN_BATCH = 8      # sliding-layer verify: batches from here run vfg + merge
ZH_VFG_MAX_BATCH = 64
ZH_USE_VP2 = True
ZH_VFG_FULL1 = False
ZH_VFG_SLIDING = False
ZH_WST_CUDA = r"""// ZhimaoL/sglang okc_patch/_cuda_src_wst.py (Apache-2.0): wst_ct14b (full-layer verify, balanced over the batch). Trimmed to
// the cooperative balanced kernel (ct14 / q_len 1 / TMA-map parameters dropped). OKC edits are marked "OKC:".
// ZH_SCH=1 keeps his piecewise-linear (Schraudolph) exp for measurement builds only; the default is the exact exp.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#ifndef ZH_SCH
#define ZH_SCH 0
#endif

#define DINL __device__ __forceinline__

DINL void wgmma_fence()  { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
DINL void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> DINL void wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N)); }

DINL void mbar_init(unsigned bar, unsigned count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(bar), "r"(count));
}
DINL void mbar_arrive(unsigned bar) {
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"(bar) : "memory");
}
DINL void mbar_wait(unsigned bar, unsigned phase) {
    asm volatile(
        "{\n.reg .pred p;\nWAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
        "@!p bra WAIT_%=;\n}\n" :: "r"(bar), "r"(phase) : "memory");
}
DINL void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
DINL void fence_barrier_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
DINL void cpa_mbar_arrive_noinc(unsigned bar) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"(bar) : "memory");
}
DINL void fence_u(unsigned& x) { asm volatile("" : "+r"(x)); }
DINL void fence_f(float& x)    { asm volatile("" : "+f"(x)); }
template <int N> DINL void fence_ubank(unsigned (&b)[N]) {
#pragma unroll
    for (int i = 0; i < N; i++) fence_u(b[i]);
}
template <int N> DINL void fence_fbank(float (&b)[N]) {
#pragma unroll
    for (int i = 0; i < N; i++) fence_f(b[i]);
}

// no-swizzle (interleaved core matrices) descriptor - the V images
// 128B-swizzled K-major descriptor (TMA SWIZZLE_128B tiles, 128B rows, 8-row groups 1024B)
DINL unsigned long long make_desc_sw128(unsigned saddr) {
    return (unsigned long long)((saddr >> 4) & 0x3FFF)
         | (1ull << 16)
         | ((unsigned long long)(1024 >> 4) << 32)
         | (1ull << 62);
}

template <int ACC>
DINL void wgmma_qk(unsigned* d, unsigned* a, unsigned long long db) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3\n"
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
        "{%32,%33,%34,%35}, %36, p, 1, 1;\n}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]),
          "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]),
          "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]),
          "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31]),
          "+r"(a[0]), "+r"(a[1]), "+r"(a[2]), "+r"(a[3])
        : "l"(db), "n"(ACC));
}

DINL __half2 u32h2(unsigned v) { return *reinterpret_cast<__half2*>(&v); }
DINL unsigned h2u32(__half2 v) { return *reinterpret_cast<unsigned*>(&v); }
DINL unsigned prmt(unsigned a, unsigned b, unsigned sel) {
    unsigned r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
    return r;
}
// OKC: exact exp2 on a packed f16 pair (one MUFU per 2 scores) and the f16x2 -> e4m3x2 conversion of the P codes
DINL unsigned ex2_f16x2(unsigned x) {
    unsigned r;
    asm("ex2.approx.f16x2 %0, %1;" : "=r"(r) : "r"(x));
    return r;
}
DINL unsigned short cvt_e4m3x2(unsigned h2) {
    unsigned short r;
    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(r) : "r"(h2));
    return r;
}
DINL int pi_lane(int lam) {
    return (lam & 1) | (((lam >> 2) & 3) << 1) | (((lam >> 1) & 1) << 3) | (lam & 16);
}

constexpr int KIMG = 16384;             // K image: [128 keys][128B] SW128 (both kv heads)
constexpr int VIMGH = 8192;             // per-head V image (pi-permuted, interleaved)
constexpr int STG = KIMG + 2 * VIMGH;   // 32KB per stage
constexpr int RAWB = 16384;             // raw V slab: [128 keys][128B] SW128
constexpr unsigned NEG = 0xFC00;
#define M_INIT (-30000.0f)
#ifndef OKC_WST_ACC32
#define OKC_WST_ACC32 1
#endif
// serving: 4 dims of one split-KV O partial row as fp32 (fp32 partials, or the bank's bf16 ones)
DINL float4 ld_part4(const float* __restrict__ ws_acc, long row, int dq) {
#if OKC_WST_ACC32
    return __ldcg(reinterpret_cast<const float4*>(ws_acc + row * 64 + dq));
#else
    const uint2 x = __ldcg(reinterpret_cast<const uint2*>(reinterpret_cast<const __nv_bfloat16*>(ws_acc) + row * 64 + dq));
    const __nv_bfloat162 u0 = *reinterpret_cast<const __nv_bfloat162*>(&x.x), u1 = *reinterpret_cast<const __nv_bfloat162*>(&x.y);
    return make_float4(__bfloat162float(u0.x), __bfloat162float(u0.y), __bfloat162float(u1.x), __bfloat162float(u1.y));
#endif
}

DINL void mask_lohi(unsigned (&s)[32], int limA, int limB, int lowA, int lowB, int t) {
#pragma unroll
    for (int j = 0; j < 16; j++) {
        int k0 = 2 * t + 8 * j;
        if (k0 > limA || k0 < lowA)         s[2 * j]     = (s[2 * j]     & 0xFFFF0000u) | NEG;
        if (k0 + 1 > limA || k0 + 1 < lowA) s[2 * j]     = (s[2 * j]     & 0x0000FFFFu) | (NEG << 16);
        if (k0 > limB || k0 < lowB)         s[2 * j + 1] = (s[2 * j + 1] & 0xFFFF0000u) | NEG;
        if (k0 + 1 > limB || k0 + 1 < lowB) s[2 * j + 1] = (s[2 * j + 1] & 0x0000FFFFu) | (NEG << 16);
    }
}

// Schraudolph's 2^y in e4m3 (exponent = floor(y) + 7, mantissa = 8 * frac(y), linear in frac: +0..6%
// mid-octave, inside e4m3's own 6.25% rounding). t = fma(s, 8c, 1144 - 8m) lands in [1024, 2048) where
// the f16 ulp is 1, so the f16 bit pattern's low byte IS the code; max(t, 1024) clamps code >= 0
// (masked = -inf -> 0). The row sums come from the PV wgmma (ones column), so l matches the P used.
DINL void wgmma_pv72(float* d, unsigned* a, unsigned long long db) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %41, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3\n"
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35},"
        "{%36,%37,%38,%39}, %40, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
          "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]),
          "+r"(a[0]), "+r"(a[1]), "+r"(a[2]), "+r"(a[3])
        : "l"(db), "n"(1));
}

constexpr int VIMG9 = 9216;             // per-head V^T image with the ones row (72 rows x 128B, SW128)

constexpr int STG2 = KIMG + RAWB;
DINL void cpa16(unsigned dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
}
// ---------------------------------------------------------------- wst_ct10 helpers
// QK first k-step, write-only accumulators (a fresh S per key block)
DINL void wgmma_qkw(unsigned* d, unsigned* a, unsigned long long db) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, 0, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3\n"
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
        "{%32,%33,%34,%35}, %36, p, 1, 1;\n}\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]), "=r"(d[4]), "=r"(d[5]), "=r"(d[6]), "=r"(d[7]),
          "=r"(d[8]), "=r"(d[9]), "=r"(d[10]), "=r"(d[11]), "=r"(d[12]), "=r"(d[13]), "=r"(d[14]), "=r"(d[15]),
          "=r"(d[16]), "=r"(d[17]), "=r"(d[18]), "=r"(d[19]), "=r"(d[20]), "=r"(d[21]), "=r"(d[22]), "=r"(d[23]),
          "=r"(d[24]), "=r"(d[25]), "=r"(d[26]), "=r"(d[27]), "=r"(d[28]), "=r"(d[29]), "=r"(d[30]), "=r"(d[31])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db));
}
// softmax_sch without the accumulator rescale and the packing (both run after PV(kt-1) retires): mask, max, lazy
// reference (m, kc, al), e4m3 codes in place. Returns whether this warp's rows moved their reference.
DINL bool softmax_sch_a(unsigned (&s)[32], float (&m)[2], float (&al)[2], __half2 (&kc)[2], float qk2f,
                        __half2 qk8h2, int diag, int limA, int limB, int t) {
    if (diag) {
#pragma unroll
        for (int j = 0; j < 16; j++) {
            int k0 = 2 * t + 8 * j;
            if (k0 > limA)     s[2 * j]     = (s[2 * j]     & 0xFFFF0000u) | NEG;
            if (k0 + 1 > limA) s[2 * j]     = (s[2 * j]     & 0x0000FFFFu) | (NEG << 16);
            if (k0 > limB)     s[2 * j + 1] = (s[2 * j + 1] & 0xFFFF0000u) | NEG;
            if (k0 + 1 > limB) s[2 * j + 1] = (s[2 * j + 1] & 0x0000FFFFu) | (NEG << 16);
        }
    }
    __half2 t0v[8], t1v[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
        t0v[j] = __hmax2(u32h2(s[4 * j]), u32h2(s[4 * j + 2]));
        t1v[j] = __hmax2(u32h2(s[4 * j + 1]), u32h2(s[4 * j + 3]));
    }
#pragma unroll
    for (int st = 4; st > 0; st >>= 1)
#pragma unroll
        for (int j = 0; j < 8; j++) if (j < st) {
            t0v[j] = __hmax2(t0v[j], t0v[j + st]);
            t1v[j] = __hmax2(t1v[j], t1v[j + st]);
        }
    const float f0 = fmaxf(__half2float(__low2half(t0v[0])), __half2float(__high2half(t0v[0])));
    const float f1 = fmaxf(__half2float(__low2half(t1v[0])), __half2float(__high2half(t1v[0])));
    al[0] = 1.0f; al[1] = 1.0f;
    const bool ok = (f0 * qk2f <= m[0] + 0.75f) & (f1 * qk2f <= m[1] + 0.75f);
    const bool moved = !__all_sync(0xFFFFFFFFu, ok);
    if (moved) {
        float r0 = f0, r1 = f1;
        r0 = fmaxf(r0, __shfl_xor_sync(0xFFFFFFFFu, r0, 1));
        r0 = fmaxf(r0, __shfl_xor_sync(0xFFFFFFFFu, r0, 2));
        r1 = fmaxf(r1, __shfl_xor_sync(0xFFFFFFFFu, r1, 1));
        r1 = fmaxf(r1, __shfl_xor_sync(0xFFFFFFFFu, r1, 2));
        const float mn0 = ceilf(fmaxf(m[0], r0 * qk2f) * 8.0f) * 0.125f;   // 1/8-octave reference: kc exact in f16
        const float mn1 = ceilf(fmaxf(m[1], r1 * qk2f) * 8.0f) * 0.125f;
        al[0] = exp2f(m[0] - mn0);
        al[1] = exp2f(m[1] - mn1);
        m[0] = mn0; m[1] = mn1;
#if ZH_SCH
        kc[0] = __float2half2_rn(1144.0f - 8.0f * mn0);
        kc[1] = __float2half2_rn(1144.0f - 8.0f * mn1);
#else
        // OKC: kc = 8 - m (P = 2^(s*c - m + 8), the Schraudolph codes' scale); a row that has not seen a key keeps a finite -60000
        // (8 - m would overflow f16 to +inf, and -inf * c + inf is NaN, which would reach the e4m3 conversion)
        kc[0] = __float2half2_rn(mn0 < -20000.0f ? -60000.0f : 8.0f - mn0);
        kc[1] = __float2half2_rn(mn1 < -20000.0f ? -60000.0f : 8.0f - mn1);
#endif
    }
#if ZH_SCH
    const __half2 h1024 = __float2half2_rn(1024.0f);
#pragma unroll
    for (int j = 0; j < 32; j++)
        s[j] = h2u32(__hmax2(__hfma2(u32h2(s[j]), qk8h2, kc[j & 1]), h1024));
#else
    // OKC: exact exp. s[j] = f16x2 of 2^(s*c - m + 8), at most 2^8.75 = 430 (e4m3 max 448); masked scores (-inf) give 0
#pragma unroll
    for (int j = 0; j < 32; j++)
        s[j] = ex2_f16x2(h2u32(__hfma2(u32h2(s[j]), qk8h2, kc[j & 1])));
#endif
    return moved;
}

// softmax_sch_a for q_len 1: only row A (q rows g of warp 0) is live; row B registers (token 1, q = 0) hold exact
// zeros, which pack to e4m3 code 0 (P = 0), so they are left untouched.

// ---- attn-engine bal: wst_ct14b = wst_ct14 (FQ, PBF, in-kernel merge) with the work balanced over the batch (stream-K over requests) ----
template <int CORE = 0>
__device__ __forceinline__ void eng_bal_row(const float* __restrict__ ws_acc, const float* __restrict__ ws_m, const float* __restrict__ ws_l,
                                            long recb, int pst, int r, int dq, int np, float vsc, uint4 graw, float eps,
                                            __nv_bfloat16* __restrict__ op, unsigned gmask) {
    // one row, 8 threads x 8 dims: fold np partials (16 per batch, loads in flight together, online merge across batches)
    float M = -1e30f, Ls = 0.f, a[8];
#pragma unroll
    for (int i = 0; i < 8; i++) a[i] = 0.f;
#pragma unroll 1
    for (int p0 = 0; p0 < np; p0 += 6) {   // attn-next (bank 16)
        float mv[6], lv[6];
        float4 xv[6][2];
#pragma unroll
        for (int p = 0; p < 6; p++) {
            if (p0 + p < np) {
                const long row = recb + (long)(p0 + p) * pst + r;
                mv[p] = __ldcg(ws_m + row);
                lv[p] = __ldcg(ws_l + row);
                xv[p][0] = ld_part4(ws_acc, row, dq);
                xv[p][1] = ld_part4(ws_acc, row, dq + 4);
            } else {
                mv[p] = -1e30f; lv[p] = 0.f; xv[p][0] = xv[p][1] = make_float4(0.f, 0.f, 0.f, 0.f);
            }
        }
        float Mc = M;
#pragma unroll
        for (int p = 0; p < 6; p++) Mc = fmaxf(Mc, mv[p]);
        const float sc = exp2f(M - Mc);
        Ls *= sc;
#pragma unroll
        for (int i = 0; i < 8; i++) a[i] *= sc;
        M = Mc;
#pragma unroll
        for (int p = 0; p < 6; p++) {
            const float w = exp2f(mv[p] - M);
            Ls += lv[p] * w;
#pragma unroll
            for (int i = 0; i < 2; i++) {
                a[4 * i] += w * xv[p][i].x; a[4 * i + 1] += w * xv[p][i].y;
                a[4 * i + 2] += w * xv[p][i].z; a[4 * i + 3] += w * xv[p][i].w;
            }
        }
    }
    const float inv = vsc / Ls;
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) { a[i] *= inv; ss += a[i] * a[i]; }
    if constexpr (CORE) {   // serving: the riders' pre-norm attention output (no RMSNorm, no gate)
        uint4 c;
        unsigned* cu = reinterpret_cast<unsigned*>(&c);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const __nv_bfloat162 rr = __floats2bfloat162_rn(a[2 * i], a[2 * i + 1]);
            cu[i] = *reinterpret_cast<const unsigned*>(&rr);
        }
        *reinterpret_cast<uint4*>(op) = c;
        return;
    }
#pragma unroll
    for (int o2 = 1; o2 < 8; o2 <<= 1) ss += __shfl_xor_sync(gmask, ss, o2);   // the row's 8 lanes only
    const float rn = rsqrtf(ss * (1.0f / 64.f) + eps);
    const __nv_bfloat162* g = reinterpret_cast<const __nv_bfloat162*>(&graw);
    uint4 o;
    unsigned* ou = reinterpret_cast<unsigned*>(&o);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const float s0 = 1.f / (1.f + __expf(-__bfloat162float(g[i].x))), s1 = 1.f / (1.f + __expf(-__bfloat162float(g[i].y)));
        const __nv_bfloat162 rr = __floats2bfloat162_rn(a[2 * i] * rn * s0, a[2 * i + 1] * rn * s1);
        ou[i] = *reinterpret_cast<const unsigned*>(&rr);
    }
    *reinterpret_cast<uint4*>(op) = o;
}
// one WG: wait until every piece of request fb arrived (its counter is a multiple of 256), then fold rows j + k * nb (16 rows per pass)
template <int CORE = 0>
__device__ __forceinline__ void eng_bal_fold(const int* __restrict__ ctr, const float* __restrict__ ws_acc, const float* __restrict__ ws_m,
                                             const float* __restrict__ ws_l, const __nv_bfloat16* __restrict__ qkv, int qkv_stride,
                                             int gate_col0, __nv_bfloat16* __restrict__ out, float vsc, float eps,
                                             int fb, int wg, int j, int nb, int np, long recb, int L, int t128) {
    const int LG = L * 8;
    const int nk = j < LG ? (LG - j + nb - 1) / nb : 0;   // rows j + k nb of this folder (WG-uniform); np partials
    if (nk <= 0) return;
    const int kk = t128 >> 3, dq = (t128 & 7) * 8;       // 16 rows per pass, 8 threads per row
    if (t128 == 0) {
        unsigned v;
        do { asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(ctr) : "memory"); } while (v & 255u);
    }
    asm volatile("bar.sync %0, 128;" :: "r"(4 + wg) : "memory");
    const unsigned gm = 0xffu << (t128 & 24);
    for (int k = kk; k < nk; k += 16) {
        const int r = j + k * nb;
        const long tok_r = (long)fb * L + (r >> 3);
        const int qcol = (wg * 8 + (r & 7)) * 64 + dq;
        uint4 graw = make_uint4(0u, 0u, 0u, 0u);
        if constexpr (!CORE) graw = *reinterpret_cast<const uint4*>(qkv + tok_r * qkv_stride + gate_col0 + qcol);
        __nv_bfloat16* op = out + tok_r * (16 * 64) + qcol;
        eng_bal_row<CORE>(ws_acc, ws_m, ws_l, recb, 2 * LG, r, dq, np, vsc, graw, eps, op, gm);
    }
}
template <int NST, int PF, int NPW, int RR, int DEC, int CORE = 0>
__global__ void __launch_bounds__(256 + 128 * NPW, 1) wst_ct14b(
    const unsigned char* __restrict__ q8,
    const unsigned char* __restrict__ vc_,
    const int* __restrict__ pt,
    const int* __restrict__ lens,
    float* __restrict__ ws_acc, float* __restrict__ ws_m, float* __restrict__ ws_l,
    const float* __restrict__ k_scale,
    float smqk_log2e, int B, int L, int maxP, int spl, int S, int window,
    const __nv_bfloat16* __restrict__ qkv = nullptr, int qkv_stride = 0, int gate_col0 = 0,
    __nv_bfloat16* __restrict__ out = nullptr, int* __restrict__ tickets = nullptr,
    const float* __restrict__ v_scale = nullptr, const unsigned char* __restrict__ kc_ = nullptr,
    const long long* __restrict__ cache_loc = nullptr, float rcp_k = 1.f, float rcp_v = 1.f, float eps = 1e-5f)
{
    constexpr int PDL = 0, FUSE = 0, EX = 0, MODE = 0, PFL2 = 0, ABL = 0, FQ = CORE ? 0 : 1, PBF = 1;
    constexpr int NPT = 128 * NPW;               // producer threads
    constexpr int NWP = 4 * NPW;                 // producer warps
    constexpr int VP = (128 + 4 * NWP - 1) / (4 * NWP);   // cp.async passes per tile (4 rows per warp per pass; K + V)
    const int tid = threadIdx.x;
    const int c = blockIdx.x, G = gridDim.x;
    const int LG = L * 8;

    extern __shared__ __align__(1024) char smem[];
    const unsigned sbase = (unsigned)__cvta_generic_to_shared(smem);
    const unsigned vimg_base = sbase + NST * STG2;      // 2 bufs x 2 heads x 9KB (72-row V^T images)
    const unsigned mb = vimg_base + 2 * 2 * VIMG9;
    const unsigned mfull = mb, mempty = mb + NST * 8;
    int* sP = reinterpret_cast<int*>(smem + NST * STG2 + 4 * VIMG9 + 2 * NST * 8 + 64);   // tile prefix over the batch [B + 1]
    if (tid < NST) {
        mbar_init(mfull + tid * 8, NPT);        // every producer's cp.async noinc
        mbar_init(mempty + tid * 8, 256);       // both consumer WGs
    }
    if (tid < 256) {                             // ones row (64) + zero rows (65-71) of all 4 V^T images
        const int img = tid >> 6, o = (tid & 63) * 16;
        const unsigned v = o < 128 ? 0x38383838u : 0u;
        *reinterpret_cast<uint4*>(smem + NST * STG2 + img * VIMG9 + 8192 + o) = make_uint4(v, v, v, v);
        fence_proxy_async();
    }
    if (tid >= 256 && tid < 288) {               // one producer warp: tiles per request from the device-side lengths -> prefix
        const int ln = tid & 31;
        int run = 0;
#pragma unroll 1
        for (int cb = 0; cb < B; cb += 32) {
            const int bb = cb + ln;
            int tl = 0;
            if (bb < B) { const int sb = lens[bb]; tl = (sb + L - (window ? max(0, sb - window + 1) : 0) + 127) >> 7; }
            int x = tl;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) { const int y = __shfl_up_sync(0xffffffffu, x, o); if (ln >= o) x += y; }
            if (bb < B) sP[bb + 1] = run + x;
            run += __shfl_sync(0xffffffffu, x, 31);
        }
        if (ln == 0) sP[0] = 0;
    }
    fence_barrier_init();
    __syncthreads();
    const int T = sP[B];
    const int Ge = min(G, T);
    if (c >= Ge) return;                         // T < G: no tiles here
    // virtual map: request b = [V(b), V(b + 1)), V(b) = sP[b] + bw b: bw setup units (no tile; a piece start costs about bw tiles of
    // consumer time) then its tiles; CTA c owns virtual [c W / Ge, (c + 1) W / Ge), W = T + bw B
    const int bw = spl;
    const int W = T + bw * B;
    auto VV = [&](int bb) { return sP[bb] + bw * bb; };
    auto cof = [&](int v) { return (int)(((long long)(v + 1) * Ge - 1) / W); };   // the CTA holding virtual unit v
    const int v0 = (int)(((long long)c * W) / Ge), v1 = (int)(((long long)(c + 1) * W) / Ge);
    int b0, tbs = 0, ntl = 0;
    {
        int lo_ = 0, hi_ = B - 1;                // largest b with V(b) <= v0
        while (lo_ < hi_) { const int mid = (lo_ + hi_ + 1) >> 1; if (VV(mid) <= v0) lo_ = mid; else hi_ = mid - 1; }
        int bf = -1;
        for (int bb = lo_; bb < B && VV(bb) + bw < v1; bb++) {
            const int r0 = VV(bb) + bw, x0 = max(v0, r0), x1 = min(v1, VV(bb + 1));
            if (x1 > x0) { if (bf < 0) { bf = bb; tbs = x0 - r0; } ntl += x1 - x0; }
        }
        b0 = bf;
    }
    if (ntl == 0) return;                        // only setup units here

    if (tid >= 256) {                            // ---------- producer: K and V rows by cp.async, the CTA's tiles back to back
        if (NPW == 1) asm volatile("setmaxnreg.dec.sync.aligned.u32 56;" ::: "memory");
        if (NPW >= 2 && NPW <= 4) asm volatile("setmaxnreg.dec.sync.aligned.u32 40;" ::: "memory");
        if (NPW >= 5) asm volatile("setmaxnreg.dec.sync.aligned.u32 32;" ::: "memory");
        const int ptid = tid - 256;
        const int pw = ptid >> 5, pl = ptid & 31;
        const int vsub = pl >> 3, vch = pl & 7;     // 4 rows per instruction, 8 lanes per row
        if (!CORE && ptid < 32 * L) {   // q rows of the later pieces (their q build follows a drain) + gate rows of every piece (fold) -> L2
            const int isg = ptid >= 16 * L, tq = (isg ? ptid - 16 * L : ptid) >> 4, ln_ = ptid & 15;
#pragma unroll 1
            for (int pb = isg ? b0 : b0 + 1; pb < B && VV(pb) + bw < v1; pb++)
                asm volatile("prefetch.global.L2 [%0];" :: "l"(qkv + ((long)pb * L + tq) * qkv_stride + (isg ? gate_col0 : 0) + ln_ * 64));
        }
        int pvs[PF][VP];
        int sb = b0, st_ = tbs, snt = sP[b0 + 1] - sP[b0];                // stage cursor (request, tile of it)
        int sseq = lens[b0], sbs = window ? max(0, sseq - window + 1) : 0;
        int fb = sb, ft = st_, fnt = snt, fseq = sseq, fbase = sbs;        // fetch cursor (runs PF tiles ahead)
        const int* fprow = pt + (long)fb * maxP;
        auto fetch = [&](int (&pv)[VP]) {
            const int p0 = fbase + ft * 128;
            const int last = fseq + L - 1 - p0;       // >= 0: every tile holds a valid row
#pragma unroll
            for (int i = 0; i < VP; i++) {
                const int r = i * 4 * NWP + pw * 4 + vsub;
                if (r < 128) pv[i] = fprow[p0 + min(r, last)];
            }
            if (++ft == fnt) {
                ++fb; ft = 0;
                if (fb < B) { fnt = sP[fb + 1] - sP[fb]; fseq = lens[fb]; fbase = window ? max(0, fseq - window + 1) : 0; fprow = pt + (long)fb * maxP; }
            }
        };
        auto stage = [&](int kt, const int (&pv)[VP]) {
            const int sl = kt % NST;
            const int round = kt / NST;
            if (round > 0) mbar_wait(mempty + sl * 8, (round - 1) & 1);
            if (MODE == 2) {
                mbar_arrive(mfull + sl * 8);
            } else {
                const unsigned st = sbase + sl * STG2;
#pragma unroll
                for (int i = 0; i < VP; i++) {
                    const int r = i * 4 * NWP + pw * 4 + vsub;
                    if (r < 128) {
                        const long src = (long)pv[i] * 128 + vch * 16;
                        cpa16(st + r * 128 + ((vch ^ (r & 7)) << 4), kc_ + src);
                        cpa16(st + KIMG + r * 128 + ((vch ^ ((r >> 4) & 7)) << 4), vc_ + src);
                    }
                }
                cpa_mbar_arrive_noinc(mfull + sl * 8);
            }
        };
#pragma unroll
        for (int u = 0; u < PF; u++) {
#pragma unroll
            for (int i = 0; i < VP; i++) pvs[u][i] = 0;
            if (u < ntl) fetch(pvs[u]);
        }
        bool nrdone = false;
        auto newrows = [&](int nrb) {   // unit = (token, k|v, kv head): 8 lanes x 8 dims; k RMSNorm / scale -> e4m3 -> cache
            const int nu = ptid >> 3, nd0 = (ptid & 7) * 8;
            const int ntok = nu >> 2, nisv = (nu >> 1) & 1, nkh = nu & 1;
            const bool act = nu < 4 * L;
            float x[8];
            if (act) {
                const uint4 raw = *reinterpret_cast<const uint4*>(qkv + ((long)nrb * L + ntok) * qkv_stride
                                                                  + (16 + 2 * nisv + nkh) * 64 + nd0);
                const __nv_bfloat162* p2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
                for (int i = 0; i < 4; i++) { x[2 * i] = __bfloat162float(p2[i].x); x[2 * i + 1] = __bfloat162float(p2[i].y); }
            } else {
#pragma unroll
                for (int i = 0; i < 8; i++) x[i] = 0.f;
            }
            float ss = 0.f;
#pragma unroll
            for (int i = 0; i < 8; i++) ss += x[i] * x[i];
            ss += __shfl_xor_sync(0xffffffffu, ss, 1);
            ss += __shfl_xor_sync(0xffffffffu, ss, 2);
            ss += __shfl_xor_sync(0xffffffffu, ss, 4);
            const float sc = nisv ? rcp_v : rsqrtf(ss * (1.0f / 64) + eps) * rcp_k;
            if (act) {
                uint2 q8v;
                q8v.x = (unsigned)__nv_cvt_float2_to_fp8x2(make_float2(x[0] * sc, x[1] * sc), __NV_SATFINITE, __NV_E4M3)
                      | ((unsigned)__nv_cvt_float2_to_fp8x2(make_float2(x[2] * sc, x[3] * sc), __NV_SATFINITE, __NV_E4M3) << 16);
                q8v.y = (unsigned)__nv_cvt_float2_to_fp8x2(make_float2(x[4] * sc, x[5] * sc), __NV_SATFINITE, __NV_E4M3)
                      | ((unsigned)__nv_cvt_float2_to_fp8x2(make_float2(x[6] * sc, x[7] * sc), __NV_SATFINITE, __NV_E4M3) << 16);
                unsigned char* dst = const_cast<unsigned char*>(nisv ? vc_ : kc_) + cache_loc[(long)nrb * L + ntok] * 128 + nkh * 64 + nd0;
                *reinterpret_cast<uint2*>(dst) = q8v;
            }
            __threadfence();
            asm volatile("bar.sync 8, %0;" :: "r"(NPT) : "memory");   // every producer thread: rows in L2 before staging
        };
        for (int kt0 = 0; kt0 < ntl; kt0 += PF) {
#pragma unroll
            for (int u = 0; u < PF; u++) {
                const int kt = kt0 + u;
                if (kt < ntl) {
                    if (!CORE && !nrdone && sbs + st_ * 128 + 127 >= sseq) { newrows(sb); nrdone = true; }   // this piece holds new positions
                    stage(kt, pvs[u]);
                    if (kt + PF < ntl) fetch(pvs[u]);
                    if (++st_ == snt) {
                        ++sb; st_ = 0; nrdone = false;
                        if (sb < B) { snt = sP[sb + 1] - sP[sb]; sseq = lens[sb]; sbs = window ? max(0, sseq - window + 1) : 0; }
                    }
                }
            }
        }
        return;
    }

    // ---------- consumer warpgroups (WG w = kv head w)
    if (NPW == 1) asm volatile("setmaxnreg.inc.sync.aligned.u32 216;" ::: "memory");
    if (NPW == 2) asm volatile("setmaxnreg.inc.sync.aligned.u32 200;" ::: "memory");
    if (NPW == 3) asm volatile("setmaxnreg.inc.sync.aligned.u32 176;" ::: "memory");
    if (NPW == 4) asm volatile("setmaxnreg.inc.sync.aligned.u32 160;" ::: "memory");
    if (NPW == 5) asm volatile("setmaxnreg.inc.sync.aligned.u32 168;" ::: "memory");
    if (NPW == 6) asm volatile("setmaxnreg.inc.sync.aligned.u32 160;" ::: "memory");
    const int wg = tid >> 7;
    const int wiw = (tid >> 5) & 3;
    const int lane = tid & 31;
    const int t = lane & 3;
    const int h = lane >> 2;
    const int t128 = tid & 127;
    const int kcv = t128 & 7;                            // 16-key group: lanes 0-7 of each octet
    const int d0 = 16 * (t128 >> 5) + 4 * ((t128 >> 3) & 3);   // 4-dim window
    const int cbyte = 64 * wg + d0;

    const float qk2f = k_scale[0] * smqk_log2e;
    const __half2 qk2h2 = __float2half2_rn(qk2f);
    const int tok0 = 2 * wiw;
    int gb = 0;                                  // ring position (CTA-local tile index) of the piece's first tile
#pragma unroll 1
    for (int b = b0, tb0 = tbs; b < B && VV(b) + bw < v1; b++, tb0 = 0) {   // pieces: the requests with tiles in [v0, v1)
    const int tb1 = min(v1, VV(b + 1)) - (VV(b) + bw);
    if (tb1 <= tb0) continue;
    const int nst_total = tb1 - tb0;
    const int seqb = lens[b];
    const int tot = seqb + L;
    const int base = window ? max(0, seqb - window + 1) : 0;   // first position token 0 sees
    const int lo = base + tb0 * 128;
    const int hi = min(tot, base + tb1 * 128);
    unsigned qf[8];
    if constexpr (FQ) {   // dims 16 c4 + 4t .. +3 (c4 = 2 sl2 + hi) of q head 8 wg + h, tokens tok0, tok0 + 1
        float x[2][16];
#pragma unroll
        for (int rr = 0; rr < 2; rr++) {
            const int tok = tok0 + rr;
            const __nv_bfloat16* qp = qkv + ((long)b * L + tok) * qkv_stride + (wg * 8 + h) * 64 + 4 * t;
#pragma unroll
            for (int c4 = 0; c4 < 4; c4++) {
                uint2 raw = make_uint2(0u, 0u);
                if (tok < L) raw = *reinterpret_cast<const uint2*>(qp + 16 * c4);
                const __nv_bfloat162 u0 = *reinterpret_cast<const __nv_bfloat162*>(&raw.x);
                const __nv_bfloat162 u1 = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
                x[rr][4 * c4 + 0] = __bfloat162float(u0.x); x[rr][4 * c4 + 1] = __bfloat162float(u0.y);
                x[rr][4 * c4 + 2] = __bfloat162float(u1.x); x[rr][4 * c4 + 3] = __bfloat162float(u1.y);
            }
        }
#pragma unroll
        for (int rr = 0; rr < 2; rr++) {
            float ss = 0.f;
#pragma unroll
            for (int i = 0; i < 16; i++) ss += x[rr][i] * x[rr][i];
            ss += __shfl_xor_sync(0xffffffffu, ss, 1);
            ss += __shfl_xor_sync(0xffffffffu, ss, 2);
            const float rn = rsqrtf(ss * (1.0f / 64) + eps);
#pragma unroll
            for (int sl2 = 0; sl2 < 2; sl2++)
#pragma unroll
                for (int hi2 = 0; hi2 < 2; hi2++) {
                    const int c4 = 2 * sl2 + hi2;
                    qf[4 * sl2 + rr + 2 * hi2] =
                        (unsigned)__nv_cvt_float2_to_fp8x2(make_float2(x[rr][4 * c4] * rn, x[rr][4 * c4 + 1] * rn), __NV_SATFINITE, __NV_E4M3)
                        | ((unsigned)__nv_cvt_float2_to_fp8x2(make_float2(x[rr][4 * c4 + 2] * rn, x[rr][4 * c4 + 3] * rn), __NV_SATFINITE, __NV_E4M3) << 16);
                }
        }
    } else {
#pragma unroll
    for (int sl2 = 0; sl2 < 2; sl2++)
#pragma unroll
        for (int rr = 0; rr < 2; rr++) {
            const int tok = tok0 + rr;
            const unsigned char* qp = q8 + (((long)b * L + tok) * 16 + wg * 8 + h) * 64
                                    + sl2 * 32 + 4 * t;
            qf[4 * sl2 + rr]     = (tok < L) ? *reinterpret_cast<const unsigned*>(qp) : 0u;
            qf[4 * sl2 + rr + 2] = (tok < L) ? *reinterpret_cast<const unsigned*>(qp + 16) : 0u;
        }
    }

    float m[2] = {M_INIT, M_INIT}, al[2];
    __half2 kc[2] = {__float2half2_rn(-60000.0f), __float2half2_rn(-60000.0f)};
    const __half2 qk8h2 = ZH_SCH ? __float2half2_rn(8.0f * qk2f) : qk2h2;   // OKC: exact exp multiplies by c itself
    float acc[36];
#pragma unroll
    for (int i = 0; i < 36; i++) acc[i] = 0.f;
    float alc[2] = {1.f, 1.f};   // serving: lazy rescales since the last fold (rows A, B)
    bool fl = false;
    float* const fsm = reinterpret_cast<float*>(smem + (NST * STG2 + 4 * VIMG9 + 2 * NST * 8 + 64 + ((4 * (B + 1) + 15) & ~15))) + wg * 128 + t128;
    unsigned sv[32] = {}, pk[16] = {};

    const int vA = min(hi - 1, seqb + tok0);
    const int vB = min(hi - 1, seqb + tok0 + 1);
    const int wA = window ? (seqb + tok0 - window + 1) : -0x40000000;      // sliding lower bounds
    const int wB = window ? (seqb + tok0 + 2 - window) : -0x40000000;
    if (nst_total > 0) {   // FA3 order (prefill owner's fmha4): nothing pending across a back edge or branch merge
        unsigned voff[16], vsto[4];
#pragma unroll
        for (int ww = 0; ww < 4; ww++)
#pragma unroll
            for (int q = 0; q < 4; q++) {
                const int lamb = kcv * 16 + ww * 4;
                const int kk = ((lamb + q) & ~31) | pi_lane((lamb + q) & 31);
                voff[4 * ww + q] = KIMG + kk * 128 + ((((cbyte >> 4) ^ ((kk >> 4) & 7)) << 4) | (cbyte & 15));
            }
#pragma unroll
        for (int rr = 0; rr < 4; rr++) vsto[rr] = (d0 + rr) * 128 + ((kcv ^ ((d0 + rr) & 7)) << 4);
        fence_ubank(voff);
        fence_ubank(vsto);
        const unsigned long long dq0 = make_desc_sw128(sbase + 64 * wg);
        const unsigned long long dv0 = make_desc_sw128(vimg_base + wg * VIMG9);
        auto qk_issue = [&](unsigned (&s_)[32], int kt) {
            if (ABL & 64) return;
            const unsigned long long dq = dq0 + (unsigned long long)(((gb + kt) % NST) * (STG2 >> 4));
            wgmma_fence();
            wgmma_qkw(s_, qf, dq);
            wgmma_qk<1>(s_, qf + 4, dq + 2);
            wgmma_commit();
        };
        auto pv_issue = [&](int kt) {
            if (ABL & 4) return;
            const unsigned long long dv = dv0 + (unsigned long long)(((gb + kt) & 1) * (2 * VIMG9 >> 4));
            fence_fbank(acc);
            fence_ubank(pk);
            wgmma_fence();
#pragma unroll
            for (int js = 0; js < 4; js++) wgmma_pv72(acc, pk + 4 * js, dv + 2 * js);
            wgmma_commit();
        };
        auto vtrans = [&](int kt) {   // own V half of tile kt: raw (SW128) -> V^T image (gb + kt) & 1
            if (ABL & 1) {
                if (!(ABL & 4096)) fence_proxy_async();
                if (!(ABL & 8192)) asm volatile("bar.sync %0, 128;" :: "r"(4 + wg) : "memory");
                return;
            }
            const char* rb = smem + ((gb + kt) % NST) * STG2;
            unsigned w[4][4];
#pragma unroll
            for (int ww = 0; ww < 4; ww++) {
                unsigned lv[4];
#pragma unroll
                for (int q = 0; q < 4; q++) lv[q] = *reinterpret_cast<const unsigned*>(rb + voff[4 * ww + q]);
                const unsigned t0 = prmt(lv[0], lv[1], 0x5140);
                const unsigned t1 = prmt(lv[0], lv[1], 0x7362);
                const unsigned t2 = prmt(lv[2], lv[3], 0x5140);
                const unsigned t3 = prmt(lv[2], lv[3], 0x7362);
                w[ww][0] = prmt(t0, t2, 0x5410);
                w[ww][1] = prmt(t0, t2, 0x7632);
                w[ww][2] = prmt(t1, t3, 0x5410);
                w[ww][3] = prmt(t1, t3, 0x7632);
            }
            char* vimg = smem + NST * STG2 + (((gb + kt) & 1) * 2 + wg) * VIMG9;
#pragma unroll
            for (int rr = 0; rr < 4; rr++)
                *reinterpret_cast<uint4*>(vimg + vsto[rr]) = make_uint4(w[0][rr], w[1][rr], w[2][rr], w[3][rr]);
            fence_proxy_async();
            asm volatile("bar.sync %0, 128;" :: "r"(4 + wg) : "memory");
        };
        bool resc = false;
        auto sm_a = [&](unsigned (&s_)[32], int kt) {   // mask, max, lazy reference, e4m3 codes in place
            const int p0 = lo + kt * 128;
            const int limA = vA - p0, limB = vB - p0;
            const int lowA = wA - p0, lowB = wB - p0;
            const int diag = (limA < 127) | (limB < 127);
            const int wdiag = (lowA > 0) | (lowB > 0);
            if (wdiag) mask_lohi(s_, limA, limB, lowA, lowB, t);
            resc = softmax_sch_a(s_, m, al, kc, qk2f, qk8h2, diag & !wdiag, limA, limB, t);
        };
        auto sm_b = [&](unsigned (&s_)[32]) {   // after PV(kt-1): rescale acc, pack the codes as the next A operand
            if (resc) {
#pragma unroll
                for (int j = 0; j < 9; j++) {
                    acc[4 * j]     *= al[0]; acc[4 * j + 1] *= al[0];
                    acc[4 * j + 2] *= al[1]; acc[4 * j + 3] *= al[1];
                    if (j < 1) { alc[0] *= al[0]; alc[1] *= al[1]; }
                }
            }
#pragma unroll
            for (int js = 0; js < 4; js++) {
#if ZH_SCH
                pk[4 * js]     = prmt(s_[8 * js],     s_[8 * js + 2], 0x6420);
                pk[4 * js + 1] = prmt(s_[8 * js + 1], s_[8 * js + 3], 0x6420);
                pk[4 * js + 2] = prmt(s_[8 * js + 4], s_[8 * js + 6], 0x6420);
                pk[4 * js + 3] = prmt(s_[8 * js + 5], s_[8 * js + 7], 0x6420);
#else   // OKC: the f16 exps -> e4m3 pairs (cvt.rn.satfinite), two per register half
                pk[4 * js]     = (unsigned)cvt_e4m3x2(s_[8 * js])     | ((unsigned)cvt_e4m3x2(s_[8 * js + 2]) << 16);
                pk[4 * js + 1] = (unsigned)cvt_e4m3x2(s_[8 * js + 1]) | ((unsigned)cvt_e4m3x2(s_[8 * js + 3]) << 16);
                pk[4 * js + 2] = (unsigned)cvt_e4m3x2(s_[8 * js + 4]) | ((unsigned)cvt_e4m3x2(s_[8 * js + 6]) << 16);
                pk[4 * js + 3] = (unsigned)cvt_e4m3x2(s_[8 * js + 5]) | ((unsigned)cvt_e4m3x2(s_[8 * js + 7]) << 16);
#endif
            }
        };
        const bool live = !DEC || tok0 < L;      // DEC (q_len 1): a warp whose 16 q rows are all tokens >= L skips
        {   // kt = 0                               // the softmax (its P stays 0)
            unsigned s0[32];
            mbar_wait(mfull + (gb % NST) * 8, (gb / NST) & 1);
            fence_proxy_async();                    // K landed by cp.async (generic proxy) -> wgmma reads
            qk_issue(s0, 0);
            vtrans(0);
            wgmma_wait<0>();
            fence_ubank(s0);
            mbar_arrive(mempty + (gb % NST) * 8);
            if (live) {
                sm_a(s0, 0);
                sm_b(s0);
            }
        }
        if (RR && nst_total > 1 && wg == 1) asm volatile("bar.arrive 6, 256;" ::: "memory");   // WG0 issues first
        for (int kt = 1; kt < nst_total; kt++) {
            const int sl_ = (gb + kt) % NST;
            unsigned s1[32];
            mbar_wait(mfull + sl_ * 8, ((gb + kt) / NST) & 1);
            if (!(ABL & 8)) fence_proxy_async();
            if (RR) asm volatile("bar.sync %0, 256;" :: "r"(6 + wg) : "memory");
            qk_issue(s1, kt);
            if ((kt & 15) == 0) {   // serving: fp32 fold, the wgmma accumulator restarts at 0
#pragma unroll
                for (int i = 0; i < 36; i++) {
                    float* const p_ = fsm + i * 256;
                    *p_ = fl ? fmaf(*p_, alc[(i >> 1) & 1], acc[i]) : acc[i];
                    acc[i] = 0.f;
                }
                alc[0] = 1.f; alc[1] = 1.f; fl = true;
            }
            pv_issue(kt - 1);
            if (RR) asm volatile("bar.arrive %0, 256;" :: "r"(6 + (wg ^ 1)) : "memory");
            vtrans(kt);
            wgmma_wait<1>();
            fence_ubank(s1);
            mbar_arrive(mempty + sl_ * 8);          // K read, raw V transposed
            if (live && !(ABL & 2)) sm_a(s1, kt);
            fence_ubank(s1);                        // codes done before the PV(kt-1) wait (overlap the PV wgmma)
            wgmma_wait<0>();
            fence_fbank(acc);
            fence_ubank(pk);
            if (live && !(ABL & 2)) sm_b(s1);
        }
        if (RR && nst_total > 1 && wg == 0) asm volatile("bar.sync 6, 256;" ::: "memory");   // WG1's last arrive
        pv_issue(nst_total - 1);
        wgmma_wait<0>();
        if (fl) {
#pragma unroll
            for (int i = 0; i < 36; i++) acc[i] = fmaf(fsm[i * 256], alc[(i >> 1) & 1], acc[i]);
        }
    }
    fence_ubank(pk);
    fence_fbank(acc);
    {   // attn-engine bal: this piece's bf16 O partial -> slot c + b (staged in this WG's idle V^T image), then arrive on (b, kv head)
        const long pbase = ((long)(c + b) * 2 + wg) * LG;
#if OKC_WST_ACC32   // serving: fp32 O partial, one 8-B store per column pair from the accumulators
#pragma unroll
        for (int rr = 0; rr < 2; rr++) {
            const int tok = tok0 + rr;
            if (tok >= L) continue;
            const long row = pbase + tok * 8 + h;
            if (t == 0) {
                ws_m[row] = m[rr];
                ws_l[row] = acc[32 + 2 * rr];
            }
            float* op = ws_acc + row * 64 + 2 * t;
#pragma unroll
            for (int j = 0; j < 8; j++)
                *reinterpret_cast<float2*>(op + 8 * j) = make_float2(acc[4 * j + 2 * rr], acc[4 * j + 2 * rr + 1]);
        }
#else
        char* stg = smem + NST * STG2 + wg * VIMG9;
#pragma unroll
        for (int rr = 0; rr < 2; rr++) {
            const int tok = tok0 + rr;
            if (tok >= L) continue;
            const int r = tok * 8 + h;
#pragma unroll
            for (int j = 0; j < 8; j++)
                *reinterpret_cast<__nv_bfloat162*>(stg + r * 144 + (8 * j + 2 * t) * 2) =
                    __floats2bfloat162_rn(acc[4 * j + 2 * rr], acc[4 * j + 2 * rr + 1]);
            if (t == 0) {
                const long row = pbase + r;
                ws_m[row] = m[rr];
                ws_l[row] = acc[32 + 2 * rr];
            }
        }
        asm volatile("bar.sync %0, 128;" :: "r"(4 + wg) : "memory");
        __nv_bfloat16* gdst = reinterpret_cast<__nv_bfloat16*>(ws_acc) + pbase * 64;
        for (int i = t128; i < LG * 8; i += 128) {
            const int r = i >> 3, cc = i & 7;
            *reinterpret_cast<uint4*>(gdst + r * 64 + cc * 8) = *reinterpret_cast<const uint4*>(stg + r * 144 + cc * 16);
        }
        asm volatile("bar.sync %0, 128;" :: "r"(4 + wg) : "memory");   // staging reads done (the arrive waits for the CTA's last piece)
        // the staging overwrote rows 64-71 of this image (ones row + zero rows): restore them for the next piece's PV
        *reinterpret_cast<uint2*>(stg + 8192 + t128 * 8) = t128 < 16 ? make_uint2(0x38383838u, 0x38383838u) : make_uint2(0u, 0u);
        fence_proxy_async();
#endif
    }
    gb += nst_total;
    }   // pieces
    asm volatile("bar.sync %0, 128;" :: "r"(4 + wg) : "memory");   // every partial + m / l store of this WG issued
    if (t128 == 0) {   // one release for all pieces, then the per-request arrivals (+1, last piece + 256 - n_b + 1)
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
#pragma unroll 1
        for (int ab = b0; ab < B && VV(ab) + bw < v1; ab++) {
            if (min(v1, VV(ab + 1)) <= max(v0, VV(ab) + bw)) continue;   // no tiles of ab here
            const int cf = cof(VV(ab) + bw), cl = cof(VV(ab + 1) - 1);
            const unsigned inc = c == cl ? (unsigned)(256 - (cl - cf + 1) + 1) : 1u;
            asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" :: "l"(tickets + ab * 2 + wg), "r"(inc) : "memory");
        }
    }
    const float vsc = v_scale[0];
#pragma unroll 1
    for (int fb = b0; fb < B && VV(fb) + bw < v1; fb++) {   // fold: the interior pieces of each request (1-piece CTAs) fold its rows
        if (min(v1, VV(fb + 1)) <= max(v0, VV(fb) + bw)) continue;
        const int cf = cof(VV(fb) + bw), cl = cof(VV(fb + 1) - 1);
        const int nb = cl - cf + 1, j = c - cf;
        const int jlo = nb >= 3 ? 1 : 0, jhi = nb >= 3 ? nb - 2 : nb - 1;
        if (j < jlo || j > jhi) continue;
        eng_bal_fold<CORE>(tickets + fb * 2 + wg, ws_acc, ws_m, ws_l, qkv, qkv_stride, gate_col0, out, vsc, eps, fb, wg, j - jlo, jhi - jlo + 1, nb,
                     ((long)(cf + fb) * 2 + wg) * LG, L, t128);
    }
}

template <int NST, int PF = 2, int NPW = 4, int RR = 1, int CORE = 0>
static void launch_ct14b(int G, cudaStream_t stream,
                         const unsigned char* vcp, const unsigned char* kcp, const int* pt, const int* lens,
                         float* wa, float* wm, float* wl, const float* ks, float smqk, int B, int L, int maxP, int window, int bw,
                         const __nv_bfloat16* qkv, int qkv_stride, const long long* cache_loc, float rcp_k, float rcp_v, float eps,
                         __nv_bfloat16* out, int* tick, const float* vs, const unsigned char* q8 = nullptr) {
    auto f0 = wst_ct14b<NST, PF, NPW, RR, 0, CORE>;
    TORCH_CHECK(L > 1, "wst_attn_bal: q_len 1 is not built");
    const int smem = NST * STG2 + 4 * VIMG9 + 2 * NST * 8 + 64 + ((4 * (B + 1) + 15) & ~15) + 36864;   // + the step-7 fold copy
    static int set = 0;
    if (smem > set) {
        cudaFuncSetAttribute(f0, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        set = smem;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)G, 1, 1);
    cfg.blockDim = dim3(256 + 128 * NPW, 1, 1);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeCooperative;   // the in-kernel merge waits on other CTAs: co-residency guaranteed (or an error)
    attr[0].val.cooperative = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaError_t e = cudaLaunchKernelEx(&cfg, f0, q8, vcp, pt, lens, wa, wm, wl, ks, smqk,
                                       B, L, maxP, bw, G, window, qkv, qkv_stride, (16 + 2 * 2) * 64, out, tick, vs, kcp, cache_loc,
                                       rcp_k, rcp_v, eps);
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
// attn-engine bal: the fulls balanced over the batch (wst_ct14b, G co-resident CTAs, in-kernel merge)
void wst_attn_bal(torch::Tensor qkv, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor page_table,
                  torch::Tensor seq_lens, torch::Tensor cache_loc, torch::Tensor ws_acc, torch::Tensor ws_m,
                  torch::Tensor ws_l, torch::Tensor k_scale, double smqk_log2e, int64_t B, int64_t L, int64_t maxP,
                  int64_t G, int64_t window, int64_t bw, double rcp_k, double rcp_v, double eps,
                  torch::Tensor out, torch::Tensor tickets, torch::Tensor v_scale) {
    auto stream = at::cuda::getCurrentCUDAStream();
    launch_ct14b<4, 2, 4, 1>((int)G, stream, reinterpret_cast<const unsigned char*>(v_cache.data_ptr()),
                             reinterpret_cast<const unsigned char*>(k_cache.data_ptr()), page_table.data_ptr<int>(),
                             seq_lens.data_ptr<int>(), reinterpret_cast<float*>(ws_acc.data_ptr()), ws_m.data_ptr<float>(),
                             ws_l.data_ptr<float>(), k_scale.data_ptr<float>(), (float)smqk_log2e, (int)B, (int)L, (int)maxP,
                             (int)window, (int)bw, reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()), (int)qkv.stride(0),
                             reinterpret_cast<const long long*>(cache_loc.data_ptr()), (float)rcp_k, (float)rcp_v, (float)eps,
                             reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), tickets.data_ptr<int>(), v_scale.data_ptr<float>());
}
"""
ZH_WST_CPP = r"""
void wst_attn_bal(torch::Tensor qkv, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor page_table,
                  torch::Tensor seq_lens, torch::Tensor cache_loc, torch::Tensor ws_acc, torch::Tensor ws_m,
                  torch::Tensor ws_l, torch::Tensor k_scale, double smqk_log2e, int64_t B, int64_t L, int64_t maxP,
                  int64_t G, int64_t window, int64_t bw, double rcp_k, double rcp_v, double eps,
                  torch::Tensor out, torch::Tensor tickets, torch::Tensor v_scale);
"""
ZH_VFG_CUDA = r"""// ZhimaoL/sglang okc_patch/_cuda_src_vfg.py (Apache-2.0), verbatim except the edits marked "OKC:".
// ZH_SCH=1 keeps his piecewise-linear (Schraudolph) P codes; the default computes the VP2 tile's P with the exact exp.
#ifndef ZH_SCH
#define ZH_SCH 0
#endif
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cooperative_groups.h>
#include <stdint.h>
#include <stdexcept>
namespace cg = cooperative_groups;
#define V_ST 5
#define V_GRAN 128
#define V_TH 8.0f
#define V_PROF 0
#define V_ABL 0
#define V_CL2 1
#define V_SCL 0
#define V_NK2 1
#define V_CL3 1
#define V_NK3 1
#define V_IS 2
#define V_POG 72
#define V_PBF 0
#define V_MLS 1
#define V_FA3 5
#define V_FA3_PROF 0
#define V_FA3_VLATE 0
#define V_FA3_PROF2 0
#define V_FA3_NOPRO 0
#define V_FA3_PREDBG 0
#define V_FA3_EDUM 0
#define V_FA3_VSSS 3
#define V_FA3_SVIS 1
namespace vfgns {
#define V_ABL 0
#define V_CL2 1
#define V_CL3 1
#define V_CL4 0
#define V_DLA 2
#define V_DYN 0
#define V_FA3 5
#define V_FA3_EDUM 0
#define V_FA3_NOPRO 0
#define V_FA3_PREDBG 0
#define V_FA3_PROF 0
#define V_FA3_PROF2 0
#define V_FA3_SVIS 1
#define V_FA3_VLATE 0
#define V_FA3_VSSS 3
#define V_FM3 0
#define V_FMASK 0
#define V_FUSEDM 0
#define V_GPF 0
#define V_GRAN 128
#define V_IS 2
#define V_LS 0
#define V_MINB 1
#define V_MLS 1
#define V_MRG2 0
#define V_NK2 1
#define V_NK3 1
#define V_NKE 0
#define V_P2 0
#define V_PBF 0
#define V_PDL 0
#define V_POG 72
#define V_PROF 0
#define V_PROF2 0
#define V_PROF3 0
#define V_RR 0
#define V_SCL 0
#define V_ST 5
#define V_TH 8
#define V_TTAIL 0

namespace cg = cooperative_groups;

#define V_CLB 1216
#define V_NT (256 + (V_DYN ? 32 : 0))
#define V_L 8
#define V_HQ 16
#define V_HKV 2
#define V_D 64
#define V_QKVW ((2 * V_HQ + 2 * V_HKV) * V_D)
#define V_PO 68
#define V_PREC (V_PBF ? 40 : V_POG)
#define V_MOFF (V_PBF ? 32 : V_D)
#define V_MINIT (-1.0e30f)
#define V_TILE 16384

struct VSmem {
  uint8_t kt[V_ST][V_TILE];
  uint8_t vt[V_ST][V_TILE];
  uint8_t nk[4096];
  uint8_t nv[4096];
  uint8_t pt[2][2][8192];
  float al[2][64];
  float al2[2][2][64];
  unsigned long long mbm;
  unsigned long long dfull[V_ST], dempty[V_ST];
  int tix[V_ST];
};

__device__ __forceinline__ uint32_t v_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ unsigned long long v_now() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ void v_cpa16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(v_u32(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ void v_fence_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void v_wgbar(int h) { asm volatile("bar.sync %0, 128;" :: "r"(2 + h) : "memory"); }
__device__ __forceinline__ void v_allbar() { asm volatile("bar.sync 1, 256;" ::: "memory"); }
__device__ __forceinline__ bool v_wgbar_or(int h, bool v) {
  uint32_t r;
  asm volatile("{\n .reg .pred p, q;\n setp.ne.u32 q, %1, 0;\n bar.red.or.pred p, %2, 128, q;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(r) : "r"((uint32_t)v), "r"(2 + h) : "memory");
  return r != 0;
}
__device__ __forceinline__ float v_ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
// OKC: exact exp2 on a packed f16 pair and the f16x2 -> e4m3x2 conversion (P codes)
__device__ __forceinline__ uint32_t v_ex2h2(uint32_t x) {
  uint32_t r;
  asm("ex2.approx.f16x2 %0, %1;" : "=r"(r) : "r"(x));
  return r;
}
__device__ __forceinline__ uint32_t v_cvt_e4m3x2h(uint32_t h2) {
  unsigned short r;
  asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(r) : "r"(h2));
  return (uint32_t)r;
}
__device__ __forceinline__ uint32_t v_prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
  return r;
}

__device__ __forceinline__ uint32_t v_e4m3x2(float lo, float hi) {
  return (uint32_t)__nv_cvt_float2_to_fp8x2(make_float2(lo, hi), __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ float2 v_e4m3x2_dec(uint32_t two) {
  __half2_raw const h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(two & 0xFFFFu), __NV_E4M3);
  return __half22float2(*reinterpret_cast<const __half2*>(&h));
}
__device__ __forceinline__ void v_ldsm4t(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(addr));
}
__device__ __forceinline__ void v_sincos(float ang, float* sn, float* cs) {
  double const a = (double)ang;
  float const r = (float)(a - 6.283185307179586 * floor(a * 0.15915494309189535 + 0.5));
  __sincosf(r, sn, cs);
}
__device__ __forceinline__ unsigned long long v_desc(uint32_t saddr) {
  return (unsigned long long)((saddr >> 4) & 0x3FFF) | (1ull << 16) | ((unsigned long long)(1024 >> 4) << 32) | (1ull << 62);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }

template <int N> __device__ __forceinline__ void v_fence_f32(float (&x)[N]) {
#pragma unroll
  for (int i = 0; i < N; i++) asm volatile("" : "+f"(x[i]) :: "memory");
}
template <int N> __device__ __forceinline__ void v_fence_u32(uint32_t (&x)[N]) {
#pragma unroll
  for (int i = 0; i < N; i++) asm volatile("" : "+r"(x[i]) :: "memory");
}
__device__ __forceinline__ void v_fence_u16(uint32_t (&x)[4][4]) {
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int j = 0; j < 4; j++) asm volatile("" : "+r"(x[i][j]) :: "memory");
}

#define V_QK16_OUT "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}"

__device__ __forceinline__ void wg_qk16_0(uint32_t (&d)[32], const uint32_t (&a)[4], unsigned long long db) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3\n" V_QK16_OUT ", {%32,%33,%34,%35}, %36, p, 1, 1;\n}\n"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]), "=r"(d[4]), "=r"(d[5]), "=r"(d[6]), "=r"(d[7]),
        "=r"(d[8]), "=r"(d[9]), "=r"(d[10]), "=r"(d[11]), "=r"(d[12]), "=r"(d[13]), "=r"(d[14]), "=r"(d[15]),
        "=r"(d[16]), "=r"(d[17]), "=r"(d[18]), "=r"(d[19]), "=r"(d[20]), "=r"(d[21]), "=r"(d[22]), "=r"(d[23]),
        "=r"(d[24]), "=r"(d[25]), "=r"(d[26]), "=r"(d[27]), "=r"(d[28]), "=r"(d[29]), "=r"(d[30]), "=r"(d[31])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db));
}
__device__ __forceinline__ void wg_qk16_1(uint32_t (&d)[32], const uint32_t (&a)[4], unsigned long long db) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3\n" V_QK16_OUT ", {%32,%33,%34,%35}, %36, p, 1, 1;\n}\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]),
        "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]),
        "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]),
        "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db));
}

__device__ __forceinline__ void wg_qk4_0(uint32_t (&d)[8], const uint32_t (&a)[4], unsigned long long db) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k32.f16.e4m3.e4m3\n{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, %12, p, 1, 1;\n}\n"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]), "=r"(d[4]), "=r"(d[5]), "=r"(d[6]), "=r"(d[7])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db));
}
__device__ __forceinline__ void wg_qk4_1(uint32_t (&d)[8], const uint32_t (&a)[4], unsigned long long db) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k32.f16.e4m3.e4m3\n{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, %12, p, 1, 1;\n}\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db));
}
__device__ __forceinline__ __half2 v_h2(uint32_t v) { return *reinterpret_cast<__half2*>(&v); }
__device__ __forceinline__ uint32_t v_u(__half2 v) { return *reinterpret_cast<uint32_t*>(&v); }

__device__ __forceinline__ void wg_pv(float (&d)[32], const uint32_t (&a)[4], unsigned long long db) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3\n"
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
      "{%32,%33,%34,%35}, %36, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "n"(1));
}

template <bool SLIDE, bool CL, bool SCLT = false>
__global__ void __launch_bounds__(V_NT, V_MINB)
vfg_kernel(const __nv_bfloat16* __restrict__ qkv, const int* __restrict__ page_table, int pt_stride,
           const int* __restrict__ seq_lens, const long long* __restrict__ cache_loc,
           uint8_t* __restrict__ k_cache, uint8_t* __restrict__ v_cache, const float* __restrict__ inv_freq,
           int window, float c2, float k_rscale, float v_rscale, float eps, float* __restrict__ ws,
           unsigned long long* __restrict__ dbgp, __nv_bfloat16* __restrict__ out, float v_scale,
           unsigned int* __restrict__ tick) {
  extern __shared__ uint8_t smem_raw[];
  VSmem& sm = *reinterpret_cast<VSmem*>(smem_raw + ((1024u - (v_u32(smem_raw) & 1023u)) & 1023u));
  int const s = (int)blockIdx.x, NS = (int)gridDim.x, b = (int)blockIdx.y;
  int const tid = threadIdx.x, lane = tid & 31;
  int const h = tid >> 7;
  int const w = (tid >> 5) & 3;
  int const g = lane >> 2, t = lane & 3;
  bool const ntcta = (s == NS - 1);
#define V_STAMP(i) do { if (dbgp && tid == 0) dbgp[((long long)b * NS + s) * 8 + (i)] = v_now(); } while (0)
  unsigned long long pacc[8] = {0, 0, 0, 0, 0, 0, 0, 0}, pclk = 0;
#define V_PH(i) do { } while (0)
#define V_PH0() do { } while (0)
  V_STAMP(0);
  __nv_bfloat162 gpre = __float2bfloat162_rn(0.f);
  int const seq = seq_lens[b];
  int const lo = SLIDE ? max(0, seq - window + 1) : 0;
  int const n = seq - lo;
  int const chunk = (((n + NS - 1) / NS) + V_GRAN - 1) / V_GRAN * V_GRAN;
  int const p0 = min(seq, lo + s * chunk), p1 = min(seq, p0 + chunk);
  int const ntiles = (p1 - p0 + 127) >> 7;
  const int* ptb = page_table + (long long)b * pt_stride;

  uint2 qraw[2][4];
#pragma unroll
  for (int e = 0; e < 2; e++) {
    const __nv_bfloat16* qp = qkv + ((long long)b * V_L + 2 * w + e) * V_QKVW + (8 * h + g) * V_D;
#pragma unroll
    for (int c4 = 0; c4 < 4; c4++) qraw[e][c4] = *reinterpret_cast<const uint2*>(qp + 16 * c4 + 4 * t);
  }
  int const nu = tid >> 3, nd0 = (tid & 7) * 8;
  int const ntok = nu >> 2, nisv = (nu >> 1) & 1, nkh = nu & 1;
  uint4 nraw = make_uint4(0u, 0u, 0u, 0u);
  long long ncrow = 0;
  if (ntcta) {
    nraw = *reinterpret_cast<const uint4*>(qkv + ((long long)b * V_L + ntok) * V_QKVW + (V_HQ + nisv * V_HKV + nkh) * V_D + nd0);
    ncrow = cache_loc[(long long)b * V_L + ntok];
  }

  float2 qfr[4] = {make_float2(0.f, 0.f), make_float2(0.f, 0.f), make_float2(0.f, 0.f), make_float2(0.f, 0.f)};
  float4 nfr = make_float4(0.f, 0.f, 0.f, 0.f);
  if (SLIDE) {
#pragma unroll
    for (int c4 = 0; c4 < 4; c4++) qfr[c4] = *reinterpret_cast<const float2*>(inv_freq + 8 * c4 + 2 * t);
    if (ntcta) nfr = *reinterpret_cast<const float4*>(inv_freq + nd0 / 2);
  }

  long long nk_crow = 0;
  float nk_ifr = 0.f;
  __nv_bfloat162 nk_kraw = __float2bfloat162_rn(0.f), nk_vraw = nk_kraw, nk_qraw = nk_kraw;
  uint32_t qf[2][4];

  int const gch = tid & 7, grs = tid >> 3;
  constexpr int VIS = CL ? V_ST : (SLIDE ? V_FA3_SVIS : V_IS);
  int rws[V_ST][4];
  if constexpr (!(V_DYN && !SLIDE && !CL)) {
#pragma unroll
    for (int it = 0; it < V_ST; it++)
#pragma unroll
      for (int k = 0; k < 4; k++) {
        int const pos = p0 + it * 128 + grs + 32 * k;
        rws[it][k] = (it < ntiles && pos < p1) ? ptb[pos] : -1;
      }
    if (dbgp) {
      int sm_ = 0;
#pragma unroll
      for (int it = 0; it < V_ST; it++)
#pragma unroll
        for (int k = 0; k < 4; k++) sm_ += rws[it][k];
      if (tid == 0) dbgp[((long long)b * NS + s) * 8 + 5] = v_now() + (sm_ == 0x7fffffff ? 1 : 0);
    }
#pragma unroll
    for (int it = 0; it < VIS; it++) {
      if (it < ntiles) {
#pragma unroll
        for (int k = 0; k < 4; k++) {
          int const gi = grs + 32 * k;
          int const off = gi * 128 + ((gch ^ (gi & 7)) << 4);
          if (rws[it][k] >= 0) {
            v_cpa16(&sm.kt[it][0] + off, k_cache + (long long)rws[it][k] * 128 + gch * 16);
            v_cpa16(&sm.vt[it][0] + off, v_cache + (long long)rws[it][k] * 128 + gch * 16);
          } else {
            *reinterpret_cast<uint4*>(&sm.kt[it][0] + off) = make_uint4(0u, 0u, 0u, 0u);
            *reinterpret_cast<uint4*>(&sm.vt[it][0] + off) = make_uint4(0u, 0u, 0u, 0u);
          }
        }
      }
      asm volatile("cp.async.commit_group;" ::: "memory");
    }
  }
  if constexpr (!(V_DYN && !SLIDE && !CL)) V_STAMP(6);

  {
    float x[2][16];
#pragma unroll
    for (int e = 0; e < 2; e++) {
      int const tok = 2 * w + e;
      float ss = 0.f;
#pragma unroll
      for (int c4 = 0; c4 < 4; c4++) {
        uint2 const raw = qraw[e][c4];
        __nv_bfloat162 const u0 = *reinterpret_cast<const __nv_bfloat162*>(&raw.x);
        __nv_bfloat162 const u1 = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
        x[e][4 * c4 + 0] = __bfloat162float(u0.x); x[e][4 * c4 + 1] = __bfloat162float(u0.y);
        x[e][4 * c4 + 2] = __bfloat162float(u1.x); x[e][4 * c4 + 3] = __bfloat162float(u1.y);
#pragma unroll
        for (int i = 0; i < 4; i++) ss += x[e][4 * c4 + i] * x[e][4 * c4 + i];
      }
      ss += __shfl_xor_sync(0xffffffffu, ss, 1);
      ss += __shfl_xor_sync(0xffffffffu, ss, 2);
      float const rn = rsqrtf(ss * (1.0f / V_D) + eps);
      if (SLIDE) {
        float const pf = (float)(seq + tok);
#pragma unroll
        for (int c4 = 0; c4 < 4; c4++)
#pragma unroll
          for (int i2 = 0; i2 < 2; i2++) {
            float sn, cs;
            v_sincos(pf * (i2 ? qfr[c4].y : qfr[c4].x), &sn, &cs);
            float const ev = x[e][4 * c4 + 2 * i2] * rn, od = x[e][4 * c4 + 2 * i2 + 1] * rn;
            x[e][4 * c4 + 2 * i2] = ev * cs - od * sn;
            x[e][4 * c4 + 2 * i2 + 1] = ev * sn + od * cs;
          }
      } else {
#pragma unroll
        for (int i = 0; i < 16; i++) x[e][i] *= rn;
      }
    }

#pragma unroll
    for (int ks = 0; ks < 2; ks++)
#pragma unroll
      for (int e = 0; e < 2; e++)
#pragma unroll
        for (int hi = 0; hi < 2; hi++) {
          int const c4 = 2 * ks + hi;
          qf[ks][e + 2 * hi] = v_e4m3x2(x[e][4 * c4], x[e][4 * c4 + 1]) | (v_e4m3x2(x[e][4 * c4 + 2], x[e][4 * c4 + 3]) << 16);
        }
  }

  if (dbgp) {
    uint32_t qq = qf[0][0] ^ qf[1][3];
    if (tid == 0) dbgp[((long long)b * NS + s) * 8 + 7] = v_now() + (qq == 0x12345678u ? 1 : 0);
  }

  if (ntcta && !(CL && V_NK2 && V_CL2)) {
    for (int i = tid; i < 2 * 4096 / 16; i += 256) {
      int const row = (i & 255) >> 3;
      if (row >= V_L) reinterpret_cast<uint4*>(i < 256 ? sm.nk : sm.nv)[i & 255] = make_uint4(0u, 0u, 0u, 0u);
    }
    int const d0 = nd0, tok = ntok, isv = nisv, kh = nkh;
    uint4 const raw = nraw;
    float x[8];
    {
      __nv_bfloat162 const* p2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
      for (int i = 0; i < 4; i++) { x[2 * i] = __bfloat162float(p2[i].x); x[2 * i + 1] = __bfloat162float(p2[i].y); }
    }
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) ss += x[i] * x[i];
    ss += __shfl_xor_sync(0xffffffffu, ss, 1);
    ss += __shfl_xor_sync(0xffffffffu, ss, 2);
    ss += __shfl_xor_sync(0xffffffffu, ss, 4);
    if (!isv) {
      float const rn = rsqrtf(ss * (1.0f / V_D) + eps);
#pragma unroll
      for (int i = 0; i < 8; i++) x[i] *= rn;
      if (SLIDE) {
        float const pf = (float)(seq + tok);
#pragma unroll
        for (int i2 = 0; i2 < 4; i2++) {
          float sn, cs;
          v_sincos(pf * (i2 == 0 ? nfr.x : i2 == 1 ? nfr.y : i2 == 2 ? nfr.z : nfr.w), &sn, &cs);
          float const ev = x[2 * i2], od = x[2 * i2 + 1];
          x[2 * i2] = ev * cs - od * sn; x[2 * i2 + 1] = ev * sn + od * cs;
        }
      }
#pragma unroll
      for (int i = 0; i < 8; i++) x[i] *= k_rscale;
    } else {
#pragma unroll
      for (int i = 0; i < 8; i++) x[i] *= v_rscale;
    }
    uint2 q8;
    q8.x = v_e4m3x2(x[0], x[1]) | (v_e4m3x2(x[2], x[3]) << 16);
    q8.y = v_e4m3x2(x[4], x[5]) | (v_e4m3x2(x[6], x[7]) << 16);
    *reinterpret_cast<uint2*>((isv ? v_cache : k_cache) + ncrow * 128 + kh * 64 + d0) = q8;
    int const cb = kh * 64 + d0;
    *reinterpret_cast<uint2*>((isv ? sm.nv : sm.nk) + tok * 128 + ((((cb >> 4) ^ (tok & 7))) << 4) + (cb & 15)) = q8;
  }

  float o[32], la[32];
#pragma unroll
  for (int i = 0; i < 32; i++) { o[i] = 0.f; la[i] = 0.f; }
  float mref[2] = {V_MINIT, V_MINIT};
  float lsr[2] = {0.f, 0.f};
  int const tokA = 2 * w, tokB = 2 * w + 1;
  int const lowA = SLIDE ? seq + tokA - window + 1 : -0x40000000;
  int const lowB = SLIDE ? seq + tokB - window + 1 : -0x40000000;
  uint32_t const pt_base = v_u32(&sm.pt[h][0][0]);
  uint32_t const ones[4] = {0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u};
  __half2 const c8h = __float2half2_rn(8.f * c2);
  __half2 const lo2 = __float2half2_rn(1024.f);
  V_STAMP(1);

#define V_T3(i) do { } while (0)
  auto tile = [&](auto nj8c, uint32_t kaddr, uint32_t vaddr, int tbase, int tlim, bool isnew) {
    constexpr int NJ8 = decltype(nj8c)::value;
    uint32_t sc[2 * NJ8];
    uint32_t const kb = kaddr + 64 * h;
    wg_fence();
    if constexpr (NJ8 == 16) { wg_qk16_0(sc, qf[0], v_desc(kb)); wg_qk16_1(sc, qf[1], v_desc(kb + 32)); }
    else { wg_qk4_0(sc, qf[0], v_desc(kb)); wg_qk4_1(sc, qf[1], v_desc(kb + 32)); }
    wg_commit();
    wg_wait0();
    V_T3(0);
    bool const edge = !(V_ABL & 256) && (isnew || (tbase + 8 * NJ8 > tlim) || (SLIDE && (tbase < lowB)));
    if (edge) {
#pragma unroll
      for (int j8 = 0; j8 < NJ8; j8++)
#pragma unroll
        for (int e = 0; e < 2; e++) {
          int const pos = tbase + 8 * j8 + 2 * t + e;
          bool vA = pos < tlim && pos >= lowA, vB = pos < tlim && pos >= lowB;
          if (isnew) { vA = vA && (pos - seq) <= tokA; vB = vB && (pos - seq) <= tokB; }
          uint32_t const keep = e ? 0x0000FFFFu : 0xFFFF0000u, neg = e ? 0xFC000000u : 0x0000FC00u;
          if (!vA) sc[2 * j8] = (sc[2 * j8] & keep) | neg;
          if (!vB) sc[2 * j8 + 1] = (sc[2 * j8 + 1] & keep) | neg;
        }
    }
    __half2 mA2 = v_h2(sc[0]), mB2 = v_h2(sc[1]);
#pragma unroll
    for (int j8 = 1; j8 < NJ8; j8++) { mA2 = __hmax2(mA2, v_h2(sc[2 * j8])); mB2 = __hmax2(mB2, v_h2(sc[2 * j8 + 1])); }
    float mxA = fmaxf(__low2float(mA2), __high2float(mA2)), mxB = fmaxf(__low2float(mB2), __high2float(mB2));
    mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffffu, mxA, 1)); mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffffu, mxA, 2));
    mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffffu, mxB, 1)); mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffffu, mxB, 2));
    mxA *= c2; mxB *= c2;
    float alA = 1.f, alB = 1.f;
    bool resc = false;
    if (mxA > mref[0] + 8.5f) {
      float const nm = ceilf(mxA * 8.f) * 0.125f - 8.f;
      resc = mref[0] > -1.0e29f; alA = v_ex2(mref[0] - nm); mref[0] = nm; lsr[0] *= alA;
    }
    if (mxB > mref[1] + 8.5f) {
      float const nm = ceilf(mxB * 8.f) * 0.125f - 8.f;
      resc = resc || mref[1] > -1.0e29f; alB = v_ex2(mref[1] - nm); mref[1] = nm; lsr[1] *= alB;
    }
    V_T3(6);

    __half2 const bA = __float2half2_rn(mref[0] > -1.0e29f ? 1080.f - 8.f * mref[0] : -60000.f);
    __half2 const bB = __float2half2_rn(mref[1] > -1.0e29f ? 1080.f - 8.f * mref[1] : -60000.f);
#pragma unroll
    for (int j8 = 0; j8 < NJ8; j8++) {
      sc[2 * j8] = v_u(__hmax2(__hfma2(v_h2(sc[2 * j8]), c8h, bA), lo2));
      sc[2 * j8 + 1] = v_u(__hmax2(__hfma2(v_h2(sc[2 * j8 + 1]), c8h, bB), lo2));
    }
    V_T3(7);
    {
      int const rA = 16 * w + g, rB = rA + 8;
#pragma unroll
      for (int jj = 0; jj < NJ8 / 2; jj++) {
        uint32_t const wA = v_prmt(sc[4 * jj], sc[4 * jj + 2], 0x6420), wB = v_prmt(sc[4 * jj + 1], sc[4 * jj + 3], 0x6420);
        *reinterpret_cast<uint32_t*>(&sm.pt[h][0][0] + rA * 128 + ((jj ^ (rA & 7)) << 4) + 4 * t) = wA;
        *reinterpret_cast<uint32_t*>(&sm.pt[h][0][0] + rB * 128 + ((jj ^ (rB & 7)) << 4) + 4 * t) = wB;
      }
      if (t == 0) { sm.al[h][rA] = alA; sm.al[h][rB] = alB; }
    }
    V_T3(1);
    v_fence_async();
    bool const anyresc = v_wgbar_or(h, resc);
    V_T3(2);
    if (anyresc) {
#pragma unroll
      for (int jj = 0; jj < 8; jj++) {
        float2 const a2 = *reinterpret_cast<const float2*>(&sm.al[h][8 * jj + 2 * t]);
        o[4 * jj + 0] *= a2.x; o[4 * jj + 1] *= a2.y; o[4 * jj + 2] *= a2.x; o[4 * jj + 3] *= a2.y;
        la[4 * jj + 0] *= a2.x; la[4 * jj + 1] *= a2.y; la[4 * jj + 2] *= a2.x; la[4 * jj + 3] *= a2.y;
      }
    }

    uint32_t va[NJ8 / 4][4];
#pragma unroll
    for (int ks = 0; ks < NJ8 / 4; ks++) {
      int const key = 32 * ks + 8 * (lane >> 3) + (lane & 7);
      uint32_t r0, r1, r2, r3;
      v_ldsm4t(r0, r1, r2, r3, vaddr + key * 128 + ((((4 * h + w) ^ (key & 7))) << 4));
      va[ks][0] = v_prmt(r0, r1, 0x6420); va[ks][1] = v_prmt(r0, r1, 0x7531);
      va[ks][2] = v_prmt(r2, r3, 0x6420); va[ks][3] = v_prmt(r2, r3, 0x7531);
    }
    V_T3(3);
    wg_fence();
#pragma unroll
    for (int ks = 0; ks < NJ8 / 4; ks++) wg_pv(o, va[ks], v_desc(pt_base + 32 * ks));
#pragma unroll
    for (int ks = 0; ks < NJ8 / 4; ks++) wg_pv(la, ones, v_desc(pt_base + 32 * ks));
    wg_commit();
    wg_wait0();
    V_T3(4);
    v_wgbar(h);
    V_T3(5);
  };
  if (ntcta && !(V_ABL & 16) && !(CL && V_NK2 && V_CL2)) {
    v_fence_async();
    v_allbar();
    tile(std::integral_constant<int, 4>{}, v_u32(sm.nk), v_u32(sm.nv), seq, seq + V_L, true);
  }

  if (ntiles > 0) {
    constexpr int VISS = (VIS == V_ST) ? V_FA3_VSSS : VIS;
    static_assert(VISS >= 1 && VISS <= V_ST - 2 || VIS < V_ST, "ring lag");
    uint32_t sc[2][32];
    uint32_t va[2][4][4];
#define V_F3(i) do { } while (0)
#define V_F3A(i) do { } while (0)
#define V_F3B(i) V_F3(i)
    auto gissue = [&](int j) {
      {

        int const nt = j + VISS;
        if (nt >= VIS && nt < ntiles) {
          int const nst = nt % V_ST;
          uint32_t const sk = v_u32(&sm.kt[0][0]) + (uint32_t)nst * (uint32_t)V_TILE + (uint32_t)(grs * 128 + ((gch ^ (grs & 7)) << 4));
          uint32_t const dv = v_u32(&sm.vt[0][0]) - v_u32(&sm.kt[0][0]);
#pragma unroll
          for (int k = 0; k < 4; k++) {
            int row = -1;
            if (nt < V_ST) {
#pragma unroll
              for (int q = VIS; q < V_ST; q++) row = (q == nt) ? rws[q][k] : row;
            } else {
              int const pos = p0 + nt * 128 + grs + 32 * k;
              row = pos < p1 ? ptb[pos] : -1;
            }
            uint32_t const sz = row >= 0 ? 16u : 0u;
            long long const ro = (long long)(row >= 0 ? row : 0) * 128 + gch * 16;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(sk + 4096u * k), "l"(k_cache + ro), "r"(sz) : "memory");
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(sk + dv + 4096u * k), "l"(v_cache + ro), "r"(sz) : "memory");
          }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
      }
    };
    auto data_wait = [&](int j) {
      if (j < VIS) asm volatile("cp.async.wait_group %0;" :: "n"(VIS - 1) : "memory");
      else asm volatile("cp.async.wait_group %0;" :: "n"(VISS - 1) : "memory");
      v_fence_async();
      V_F3A(0);
      v_allbar();
      if (j == 0) V_STAMP(2);
      V_F3A(1);
      if constexpr (SLIDE) gissue(j);
      V_F3A(2);
    };
    auto issue_qk = [&](uint32_t (&s)[32], int j) {
      uint32_t const kb = v_u32(&sm.kt[j % V_ST][0]) + 64 * h;
      wg_fence();
      wg_qk16_0(s, qf[0], v_desc(kb));
      wg_qk16_1(s, qf[1], v_desc(kb + 32));
      wg_commit();
    };
    auto issue_pv = [&](uint32_t (&a)[4][4], int buf) {
      v_fence_f32(o); v_fence_f32(la); v_fence_u16(a);
      uint32_t const pd = v_u32(&sm.pt[h][buf][0]);
      wg_fence();
#pragma unroll
      for (int ks = 0; ks < 4; ks++) wg_pv(o, a[ks], v_desc(pd + 32 * ks));
#pragma unroll
      for (int ks = 0; ks < 4; ks++) wg_pv(la, ones, v_desc(pd + 32 * ks));
      wg_commit();
    };
    auto vfrags = [&](uint32_t (&a)[4][4], int j) {
      uint32_t const vb = v_u32(&sm.vt[j % V_ST][0]);
#pragma unroll
      for (int ks = 0; ks < 4; ks++) {
        int const key = 32 * ks + 8 * (lane >> 3) + (lane & 7);
        uint32_t r0, r1, r2, r3;
        v_ldsm4t(r0, r1, r2, r3, vb + key * 128 + ((((4 * h + w) ^ (key & 7))) << 4));
        a[ks][0] = v_prmt(r0, r1, 0x6420); a[ks][1] = v_prmt(r0, r1, 0x7531);
        a[ks][2] = v_prmt(r2, r3, 0x6420); a[ks][3] = v_prmt(r2, r3, 0x7531);
      }
      v_fence_u16(a);
    };
    auto softmax_store = [&](uint32_t (&s)[32], int j, int buf) -> bool {
      int const tbase = p0 + j * 128, tlim = p1;
      bool const edge = (tbase + 128 > tlim) || (SLIDE && (tbase < lowB));
      if (edge) {
#pragma unroll
        for (int j8 = 0; j8 < 16; j8++)
#pragma unroll
          for (int e = 0; e < 2; e++) {
            int const pos = tbase + 8 * j8 + 2 * t + e;
            bool const vA = pos < tlim && pos >= lowA, vB = pos < tlim && pos >= lowB;
            uint32_t const keep = e ? 0x0000FFFFu : 0xFFFF0000u, neg = e ? 0xFC000000u : 0x0000FC00u;
            if (!vA) s[2 * j8] = (s[2 * j8] & keep) | neg;
            if (!vB) s[2 * j8 + 1] = (s[2 * j8 + 1] & keep) | neg;
          }
      }
      __half2 mA2, mB2;
      if constexpr (SLIDE) {
      __half2 tA[8], tB[8];
#pragma unroll
      for (int i = 0; i < 8; i++) { tA[i] = __hmax2(v_h2(s[4 * i]), v_h2(s[4 * i + 2])); tB[i] = __hmax2(v_h2(s[4 * i + 1]), v_h2(s[4 * i + 3])); }
#pragma unroll
      for (int i = 0; i < 4; i++) { tA[i] = __hmax2(tA[i], tA[i + 4]); tB[i] = __hmax2(tB[i], tB[i + 4]); }
#pragma unroll
      for (int i = 0; i < 2; i++) { tA[i] = __hmax2(tA[i], tA[i + 2]); tB[i] = __hmax2(tB[i], tB[i + 2]); }
      mA2 = __hmax2(tA[0], tA[1]); mB2 = __hmax2(tB[0], tB[1]);
      } else {
        mA2 = v_h2(s[0]); mB2 = v_h2(s[1]);
#pragma unroll
        for (int j8 = 1; j8 < 16; j8++) { mA2 = __hmax2(mA2, v_h2(s[2 * j8])); mB2 = __hmax2(mB2, v_h2(s[2 * j8 + 1])); }
      }
      float mxA = fmaxf(__low2float(mA2), __high2float(mA2)), mxB = fmaxf(__low2float(mB2), __high2float(mB2));
      mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffffu, mxA, 1)); mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffffu, mxA, 2));
      mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffffu, mxB, 1)); mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffffu, mxB, 2));
      mxA *= c2; mxB *= c2;
      float alA = 1.f, alB = 1.f;
      bool resc = false;
      if (mxA > mref[0] + 8.5f) {
        float const nm = ceilf(mxA * 8.f) * 0.125f - 8.f;
        resc = mref[0] > -1.0e29f; alA = v_ex2(mref[0] - nm); mref[0] = nm;
      }
      if (mxB > mref[1] + 8.5f) {
        float const nm = ceilf(mxB * 8.f) * 0.125f - 8.f;
        resc = resc || mref[1] > -1.0e29f; alB = v_ex2(mref[1] - nm); mref[1] = nm;
      }
      __half2 const bA = __float2half2_rn(mref[0] > -1.0e29f ? 1080.f - 8.f * mref[0] : -60000.f);
      __half2 const bB = __float2half2_rn(mref[1] > -1.0e29f ? 1080.f - 8.f * mref[1] : -60000.f);
#pragma unroll
      for (int j8 = 0; j8 < 16; j8++) {
        s[2 * j8] = v_u(__hmax2(__hfma2(v_h2(s[2 * j8]), c8h, bA), lo2));
        s[2 * j8 + 1] = v_u(__hmax2(__hfma2(v_h2(s[2 * j8 + 1]), c8h, bB), lo2));
      }
      uint8_t* ptw = &sm.pt[h][buf][0];
      int const rA = 16 * w + g, rB = rA + 8;
#pragma unroll
      for (int jj = 0; jj < 8; jj++) {
        *reinterpret_cast<uint32_t*>(ptw + rA * 128 + ((jj ^ (rA & 7)) << 4) + 4 * t) = v_prmt(s[4 * jj], s[4 * jj + 2], 0x6420);
        *reinterpret_cast<uint32_t*>(ptw + rB * 128 + ((jj ^ (rB & 7)) << 4) + 4 * t) = v_prmt(s[4 * jj + 1], s[4 * jj + 3], 0x6420);
      }
      if (t == 0) { sm.al2[h][buf][rA] = alA; sm.al2[h][buf][rB] = alB; }
      v_fence_async();
      return v_wgbar_or(h, resc);
    };
    auto rescale = [&](bool anyresc, int buf) {
      if (anyresc) {
#pragma unroll
        for (int jj = 0; jj < 8; jj++) {
          float2 const a2 = *reinterpret_cast<const float2*>(&sm.al2[h][buf][8 * jj + 2 * t]);
          o[4 * jj + 0] *= a2.x; o[4 * jj + 1] *= a2.y; o[4 * jj + 2] *= a2.x; o[4 * jj + 3] *= a2.y;
          la[4 * jj + 0] *= a2.x; la[4 * jj + 1] *= a2.y; la[4 * jj + 2] *= a2.x; la[4 * jj + 3] *= a2.y;
        }
      }
    };

    auto mid = [&](auto pc, int j) {
      constexpr int P = decltype(pc)::value, Q = P ^ 1;
      issue_pv(va[Q], Q);
      V_F3(0);
      wg_wait<1>();
      v_fence_u32(sc[P]);
      V_F3(1);
      vfrags(va[P], j);
      V_F3(2);
      bool const anyresc = softmax_store(sc[P], j, P);
      V_F3(3);
      wg_wait<0>();
      v_fence_f32(o); v_fence_f32(la);
      V_F3(4);
      rescale(anyresc, P);
      data_wait(j + 1);
      issue_qk(sc[Q], j + 1);
      V_F3(5);
    };
    auto last = [&](auto pc, int j) {
      constexpr int P = decltype(pc)::value, Q = P ^ 1;
      issue_pv(va[Q], Q);
      V_F3(0);
      wg_wait<1>();
      v_fence_u32(sc[P]);
      V_F3(1);
      vfrags(va[P], j);
      V_F3(2);
      bool const anyresc = softmax_store(sc[P], j, P);
      V_F3(3);
      wg_wait<0>();
      v_fence_f32(o); v_fence_f32(la);
      V_F3(4);
      rescale(anyresc, P);
      V_F3(5);
      issue_pv(va[P], P);
      wg_wait<0>();
      v_fence_f32(o); v_fence_f32(la);
      V_F3(7);
    };

    data_wait(0);
    issue_qk(sc[0], 0);
    if constexpr (!SLIDE) gissue(0);
    wg_wait<0>();
    v_fence_u32(sc[0]);
    {
      bool const anyresc = softmax_store(sc[0], 0, 0);
      rescale(anyresc, 0);
    }
    vfrags(va[0], 0);
    V_F3(6);
    for (int j = 1; j < ntiles; j++) {
      data_wait(j);
      issue_qk(sc[0], j);
      issue_pv(va[0], (j + 1) & 1);
      if constexpr (!SLIDE) gissue(j);
      V_F3B(0);
      wg_wait<1>();
      v_fence_u32(sc[0]);
      V_F3B(1);
      V_F3A(4);
      bool const anyresc = softmax_store(sc[0], j, j & 1);
      V_F3(3);
      wg_wait<0>();
      v_fence_f32(o); v_fence_f32(la);
      V_F3B(4);
      rescale(anyresc, j & 1);
      V_F3B(5);
      vfrags(va[0], j);
      V_F3B(2);
      V_F3A(5);
    }
    issue_pv(va[0], (ntiles - 1) & 1);
    wg_wait<0>();
    v_fence_f32(o); v_fence_f32(la);
    V_F3(7);
#undef V_F3
  }
  V_STAMP(3);
  if constexpr (SLIDE && !CL && !SCLT) { if (tick == nullptr) asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
#define V_P2S(i) do { } while (0)
  asm volatile("cp.async.wait_all;" ::: "memory");
  V_P2S(1);

  float* wsb = ws + ((long long)b * NS + s) * (V_L * V_HQ) * V_PREC;
  float* mlh = reinterpret_cast<float*>(&sm.pt[h][0][0]);
#define V_MLW(R, j) mlh[(R) * 2 + (j)]
  if (t == 0) {
    int const rA = 16 * w + g, rB = rA + 8;
    int const RA = (rA >> 3) * V_HQ + 8 * h + (rA & 7), RB = (rB >> 3) * V_HQ + 8 * h + (rB & 7);
    V_MLW(RA, 0) = mref[0];
    V_MLW(RB, 0) = mref[1];
  }
#pragma unroll
  for (int jj = 0; jj < 8; jj++)
#pragma unroll
    for (int e = 0; e < 2; e++) {
      int const nq = 8 * jj + 2 * t + e;
      int const R = (nq >> 3) * V_HQ + 8 * h + (nq & 7);
      if (V_PBF)
        reinterpret_cast<__nv_bfloat162*>(wsb + R * V_PREC)[8 * w + g] = __floats2bfloat162_rn(o[4 * jj + e], o[4 * jj + 2 + e]);
      else if (!(V_ABL & 512))
        *reinterpret_cast<float2*>(wsb + R * V_PREC + 16 * w + 2 * g) = make_float2(o[4 * jj + e], o[4 * jj + 2 + e]);
      if (!V_LS && w == 0 && g == 0) V_MLW(R, 1) = la[4 * jj + e];
    }
  __syncthreads();
  {

    int const R = tid >> 1, hf = tid & 1;
    const float* src = reinterpret_cast<const float*>(&sm.pt[(R & 15) >> 3][0][0]) + 2 * R;
    float4 const v = hf == 0 ? make_float4(src[0], src[1], 0.f, 0.f) : make_float4(0.f, 0.f, 0.f, 0.f);
    *reinterpret_cast<float4*>(wsb + R * V_PREC + V_MOFF + 4 * hf) = v;
  }
#undef V_MLW
  V_P2S(2);
  V_STAMP(4);
}

#define V_MW 8
#define V_MP 3
__global__ void __launch_bounds__(V_MW * 32, 1)
vfgm_kernel(const float* __restrict__ ws, int NS, const __nv_bfloat16* __restrict__ qkv,
            __nv_bfloat16* __restrict__ out, float v_scale, float eps) {
  __shared__ float red[V_MW][32][4];
  int const R = (int)blockIdx.x, b = (int)blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int const tok = R / V_HQ, hq = R % V_HQ;
  __nv_bfloat162 const g2 = *reinterpret_cast<const __nv_bfloat162*>(
      qkv + ((long long)b * V_L + tok) * V_QKVW + (V_HQ + 2 * V_HKV) * V_D + hq * V_D + 2 * lane);
  float M = V_MINIT, Ls = 0.f, a0 = 0.f, a1 = 0.f;
  for (int base = warp; base < NS; base += V_MW * V_MP) {
    float2 ml[V_MP], ov[V_MP];
#pragma unroll
    for (int j = 0; j < V_MP; j++) {
      int const p = base + j * V_MW;
      if (p < NS) {
        const float* rp = ws + (((long long)b * NS + p) * (V_L * V_HQ) + R) * V_PREC;
        ml[j] = __ldcg(reinterpret_cast<const float2*>(rp + V_MOFF));
        if (V_PBF) {
          unsigned const u = __ldcg(reinterpret_cast<const unsigned*>(rp) + lane);
          ov[j] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u));
        } else {
          ov[j] = __ldcg(reinterpret_cast<const float2*>(rp + 2 * lane));
        }
      } else {
        ml[j] = make_float2(V_MINIT, 0.f); ov[j] = make_float2(0.f, 0.f);
      }
    }
    float Mn = M;
#pragma unroll
    for (int j = 0; j < V_MP; j++) Mn = fmaxf(Mn, ml[j].x);
    float const sc = v_ex2(M - Mn);
    Ls *= sc; a0 *= sc; a1 *= sc;
#pragma unroll
    for (int j = 0; j < V_MP; j++) {
      float const wt = v_ex2(ml[j].x - Mn);
      Ls += ml[j].y * wt; a0 += ov[j].x * wt; a1 += ov[j].y * wt;
    }
    M = Mn;
  }
  red[warp][lane][0] = M; red[warp][lane][1] = Ls; red[warp][lane][2] = a0; red[warp][lane][3] = a1;
  __syncthreads();
  if (warp == 0) {
    float M2 = V_MINIT;
#pragma unroll
    for (int i = 0; i < V_MW; i++) M2 = fmaxf(M2, red[i][lane][0]);
    float L2 = 0.f, b0 = 0.f, b1 = 0.f;
#pragma unroll
    for (int i = 0; i < V_MW; i++) {
      float const wt = v_ex2(red[i][lane][0] - M2);
      L2 += red[i][lane][1] * wt; b0 += red[i][lane][2] * wt; b1 += red[i][lane][3] * wt;
    }
    float const inv = v_scale / L2;
    b0 *= inv; b1 *= inv;
    float ss = b0 * b0 + b1 * b1;
#pragma unroll
    for (int o2 = 16; o2 > 0; o2 >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o2);
    float const rn = rsqrtf(ss * (1.0f / V_D) + eps);
    float const s0 = 1.f / (1.f + __expf(-__bfloat162float(g2.x)));
    float const s1 = 1.f / (1.f + __expf(-__bfloat162float(g2.y)));
    *reinterpret_cast<__nv_bfloat162*>(out + ((long long)b * V_L + tok) * (V_HQ * V_D) + hq * V_D + 2 * lane) =
        __floats2bfloat162_rn(b0 * rn * s0, b1 * rn * s1);
  }
}

__global__ void __launch_bounds__(256, 4)
vfgm2_kernel(const float* __restrict__ ws, int NS, const __nv_bfloat16* __restrict__ qkv,
             __nv_bfloat16* __restrict__ out, float v_scale, float eps) {
  int const tok = (int)blockIdx.x, b = (int)blockIdx.y, tid = threadIdx.x;
  int const hq = tid >> 4, dq = (tid & 15) * 4, R = tok * V_HQ + hq;
  uint2 const graw = *reinterpret_cast<const uint2*>(qkv + ((long long)b * V_L + tok) * V_QKVW + (V_HQ + 2 * V_HKV) * V_D + hq * V_D + dq);
  __nv_bfloat162 const g01 = *reinterpret_cast<const __nv_bfloat162*>(&graw.x);
  __nv_bfloat162 const g23 = *reinterpret_cast<const __nv_bfloat162*>(&graw.y);
  float const s0 = 1.f / (1.f + __expf(-__bfloat162float(g01.x))), s1 = 1.f / (1.f + __expf(-__bfloat162float(g01.y)));
  float const s2 = 1.f / (1.f + __expf(-__bfloat162float(g23.x))), s3 = 1.f / (1.f + __expf(-__bfloat162float(g23.y)));
  asm volatile("griddepcontrol.wait;" :: "f"(s0), "f"(s1), "f"(s2), "f"(s3) : "memory");
  float M = V_MINIT, Ls = 0.f, a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
  if (NS == 4) {
    float2 ml[4];
    float4 ov[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const float* rp = ws + (((long long)b * 4 + j) * (V_L * V_HQ) + R) * V_PREC;
      ml[j] = __ldcg(reinterpret_cast<const float2*>(rp + V_MOFF));
      if (V_PBF) {
        uint2 const u = __ldcg(reinterpret_cast<const uint2*>(rp) + (dq >> 2));
        float2 const f0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
        float2 const f1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
        ov[j] = make_float4(f0.x, f0.y, f1.x, f1.y);
      } else {
        ov[j] = __ldcg(reinterpret_cast<const float4*>(rp + dq));
      }
    }
    M = fmaxf(fmaxf(ml[0].x, ml[1].x), fmaxf(ml[2].x, ml[3].x));
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float const wt = v_ex2(ml[j].x - M);
      Ls += ml[j].y * wt; a0 += ov[j].x * wt; a1 += ov[j].y * wt; a2 += ov[j].z * wt; a3 += ov[j].w * wt;
    }
  } else
  for (int pb = 0; pb < NS; pb += 4) {
    float2 ml[4];
    float4 ov[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      int const p = pb + j;
      if (p < NS) {
        const float* rp = ws + (((long long)b * NS + p) * (V_L * V_HQ) + R) * V_PREC;
        ml[j] = __ldcg(reinterpret_cast<const float2*>(rp + V_MOFF));
        if (V_PBF) {
          uint2 const u = __ldcg(reinterpret_cast<const uint2*>(rp) + (dq >> 2));
          float2 const a0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
          float2 const a1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
          ov[j] = make_float4(a0.x, a0.y, a1.x, a1.y);
        } else {
          ov[j] = __ldcg(reinterpret_cast<const float4*>(rp + dq));
        }
      } else {
        ml[j] = make_float2(V_MINIT, 0.f); ov[j] = make_float4(0.f, 0.f, 0.f, 0.f);
      }
    }
    float Mn = M;
#pragma unroll
    for (int j = 0; j < 4; j++) Mn = fmaxf(Mn, ml[j].x);
    float const sc = v_ex2(M - Mn);
    Ls *= sc; a0 *= sc; a1 *= sc; a2 *= sc; a3 *= sc;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float const wt = v_ex2(ml[j].x - Mn);
      Ls += ml[j].y * wt; a0 += ov[j].x * wt; a1 += ov[j].y * wt; a2 += ov[j].z * wt; a3 += ov[j].w * wt;
    }
    M = Mn;
  }
  float const inv = v_scale / Ls;
  a0 *= inv; a1 *= inv; a2 *= inv; a3 *= inv;
  float ss = a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
#pragma unroll
  for (int o2 = 1; o2 < 16; o2 <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o2);
  float const rn = rsqrtf(ss * (1.0f / V_D) + eps);
  __nv_bfloat162 const r01 = __floats2bfloat162_rn(a0 * rn * s0, a1 * rn * s1);
  __nv_bfloat162 const r23 = __floats2bfloat162_rn(a2 * rn * s2, a3 * rn * s3);
  uint2 r;
  r.x = *reinterpret_cast<const uint32_t*>(&r01);
  r.y = *reinterpret_cast<const uint32_t*>(&r23);
  *reinterpret_cast<uint2*>(out + ((long long)b * V_L + tok) * (V_HQ * V_D) + hq * V_D + dq) = r;
}

void vfg_launch(int64_t qkv, int64_t page_table, int64_t pt_stride, int64_t seq_lens, int64_t cache_loc,
                int64_t k_cache, int64_t v_cache, int64_t inv_freq, int64_t out, int64_t B, int64_t NS, int64_t window,
                double c2, double k_rscale, double v_rscale, double v_scale, double eps, int64_t ws, int64_t dbgp,
                int64_t tick, int64_t stream) {
  static bool configured = false;
  if (!configured) {
    int const sm = (int)sizeof(VSmem) + 1024;
    cudaFuncSetAttribute(vfg_kernel<false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    cudaFuncSetAttribute(vfg_kernel<true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    configured = true;
  }
  cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
  const __nv_bfloat16* a_qkv = reinterpret_cast<const __nv_bfloat16*>(qkv);
  float* a_ws = reinterpret_cast<float*>(ws);
  __nv_bfloat16* a_out = reinterpret_cast<__nv_bfloat16*>(out);
  const int* a_pt = reinterpret_cast<const int*>(page_table);
  const int* a_sl = reinterpret_cast<const int*>(seq_lens);
  const long long* a_cl = reinterpret_cast<const long long*>(cache_loc);
  uint8_t* a_k = reinterpret_cast<uint8_t*>(k_cache);
  uint8_t* a_v = reinterpret_cast<uint8_t*>(v_cache);
  const float* a_f = reinterpret_cast<const float*>(inv_freq);
  unsigned long long* a_d = reinterpret_cast<unsigned long long*>(dbgp);
  dim3 grid((unsigned)NS, (unsigned)B);
  size_t const smem = sizeof(VSmem) + 1024;
  if (window > 0)
    vfg_kernel<true, false><<<grid, 256, smem, st>>>(a_qkv, a_pt, (int)pt_stride, a_sl, a_cl, a_k, a_v, a_f, (int)window,
        (float)c2, (float)k_rscale, (float)v_rscale, (float)eps, a_ws, a_d, a_out, (float)v_scale, (unsigned int*)nullptr);
  else
    vfg_kernel<false, false><<<grid, V_NT, smem, st>>>(a_qkv, a_pt, (int)pt_stride, a_sl, a_cl, a_k, a_v, a_f, 0,
        (float)c2, (float)k_rscale, (float)v_rscale, (float)eps, a_ws, a_d, a_out, (float)v_scale, (unsigned int*)nullptr);
  if (NS >= 32)
    vfgm_kernel<<<dim3(V_L * V_HQ, (unsigned)B), V_MW * 32, 0, st>>>(a_ws, (int)NS, a_qkv, a_out, (float)v_scale,
                                                                     (float)eps);
  else {
    cudaLaunchConfig_t mc = {};
    mc.gridDim = dim3(V_L, (unsigned)B);
    mc.blockDim = dim3(256, 1, 1);
    mc.dynamicSmemBytes = 0;
    mc.stream = st;
    cudaLaunchAttribute ma[1];
    ma[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    ma[0].val.programmaticStreamSerializationAllowed = 1;
    mc.attrs = ma;
    mc.numAttrs = 1;
    cudaError_t e = cudaLaunchKernelEx(&mc, vfgm2_kernel, (const float*)a_ws, (int)NS, (const __nv_bfloat16*)a_qkv, a_out,
                                       (float)v_scale, (float)eps);
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
  }
}

namespace b1vp2 {
#define VP2_PUSH 1
#define VP2_LATEQ 0
#define VP2_ST 0
#define VP2_QSM 1
#define VP2_LWG0 1
#define VP2_MLB 0
#define VP2_P64 0
#define VP2_NOTF 0
#define VP2_STRM 0
#define VP2_WG1A 1

#define Q_BLK 1120
#define Q_TX 1056
struct VP2Smem {
  uint8_t kt[128 * 128];
  uint8_t vt[128 * 128];
  uint8_t pt[64 * 128];

  uint8_t mrb[16 * Q_BLK];
  float2 tab[256];
  float kdec[8 * 65];
  float qdec[4 * 65];
  float vdec[8 * 64];
  float pp[4 * 8];
  float lrow[64];
  uint8_t qs[64 * 160];
  alignas(16) uint32_t nks[128 * 8];
  alignas(8) unsigned long long mbm;
  alignas(8) unsigned long long mbml;
  alignas(8) unsigned long long mbs[16];
};
#define Q_ST(i) do { } while (0)

__global__ void __launch_bounds__(256, 1)
vp2_kernel(const __nv_bfloat16* __restrict__ qkv, const int* __restrict__ page_table,
           const int* __restrict__ seq_lens, const long long* __restrict__ cache_loc,
           uint8_t* __restrict__ k_cache, uint8_t* __restrict__ v_cache, const float* __restrict__ inv_freq,
           int window, float c2, float k_rscale, float v_rscale, float eps,
           unsigned long long* __restrict__ dbgp, __nv_bfloat16* __restrict__ out, float v_scale,
           int pt_stride) {
  {
    int const b = (int)blockIdx.y;
    qkv += (long long)b * 8 * V_QKVW;
    out += (long long)b * 8 * (V_HQ * V_D);
    page_table += (long long)b * pt_stride;
    seq_lens += b;
    cache_loc += 8 * b;
  }
  extern __shared__ uint8_t smem_raw[];
  VP2Smem& sm = *reinterpret_cast<VP2Smem*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  int const tid = threadIdx.x, lane = tid & 31;
  int const wg = tid >> 7;
  int const w = (tid >> 5) & 3;
  int const g = lane >> 2, t = lane & 3;
  unsigned const r = cg::this_cluster().block_rank();
  int const hk = (int)blockIdx.x >> 4;
  int const otok = (int)r >> 1, oh0 = 8 * hk + 4 * ((int)r & 1);
  __nv_bfloat162 gpre = __float2bfloat162_rn(0.f);
  Q_ST(0 + 16 * wg);
  if (wg == 1) {

    uint32_t const d_ = v_u32(&sm.nks[(tid & 127) * 8]);
#pragma unroll
    for (int i = 0; i < 2; i++) {
      int const j = w + 4 * i;
      asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(d_ + 4 * i),
                   "l"(qkv + (long long)j * V_QKVW + (V_HQ + hk) * V_D + 2 * lane) : "memory");
      asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(d_ + 8 + 4 * i),
                   "l"(qkv + (long long)j * V_QKVW + (V_HQ + V_HKV + hk) * V_D + 2 * lane) : "memory");
    }
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(d_ + 16),
                 "l"(qkv + (long long)otok * V_QKVW + (oh0 + w) * V_D + 2 * lane) : "memory");
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" :: "r"(d_ + 24), "l"(cache_loc + otok) : "memory");
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
  if (tid == 0) {
    uint32_t const mb = v_u32(&sm.mbm);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mb) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(mb), "r"(VP2_MLB ? 16 * 512 : 16 * Q_TX) : "memory");
  }
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
  int const seq = seq_lens[0];
  int const lo = max(0, seq - window + 1);
  int const n = seq - lo;
  int const chunk = (((n + 15) / 16) + 127) / 128 * 128;
  int const p0 = min(seq, lo + (int)r * chunk), p1 = min(seq, p0 + chunk);
  int const ntiles = (p1 - p0 + 127) >> 7;
  float const nk_ifr = inv_freq[lane];

  __nv_bfloat162 nk_kraw[2], nk_vraw[2], nk_qraw = __float2bfloat162_rn(0.f);
  long long nk_crow = 0;
  auto small_loads = [&]() {
    if (wg == 1) {
    }
  };
  small_loads();

  int const gch = tid & 7, grs = tid >> 3;
  {
    int rws[4];
#pragma unroll
    for (int k = 0; k < 4; k++) {
      int const pos = p0 + grs + 32 * k;
      rws[k] = (ntiles > 0 && pos < p1) ? page_table[pos] : -1;
    }
    if (wg == 0) {
      int const qrow = tid >> 1, qh = tid & 1;
      const uint8_t* const qsrc = reinterpret_cast<const uint8_t*>(qkv)
          + 2 * ((long long)(qrow >> 3) * V_QKVW + (8 * hk + (qrow & 7)) * V_D) + 64 * qh;
#pragma unroll
      for (int c = 0; c < 4; c++) v_cpa16(sm.qs + qrow * 160 + 64 * qh + 16 * c, qsrc + 16 * c);
      asm volatile("cp.async.commit_group;" ::: "memory");
      gpre = *reinterpret_cast<const __nv_bfloat162*>(qkv + (long long)otok * V_QKVW + (V_HQ + 2 * V_HKV) * V_D
                                                       + (oh0 + w) * V_D + 2 * lane);
    }
#pragma unroll
    for (int k = 0; k < 4; k++) {
      int const gi = grs + 32 * k;
      int const off = gi * 128 + ((gch ^ (gi & 7)) << 4);
      if (rws[k] >= 0) {
        v_cpa16(sm.kt + off, k_cache + (long long)rws[k] * 128 + gch * 16);
        v_cpa16(sm.vt + off, v_cache + (long long)rws[k] * 128 + gch * 16);
      } else {
        *reinterpret_cast<uint4*>(sm.kt + off) = make_uint4(0u, 0u, 0u, 0u);
        *reinterpret_cast<uint4*>(sm.vt + off) = make_uint4(0u, 0u, 0u, 0u);
      }
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
  Q_ST(3 + 16 * wg);
  {
    float sn_, cs_;
    v_sincos((float)(seq + (tid >> 5)) * nk_ifr, &sn_, &cs_);
    sm.tab[tid] = make_float2(cs_, sn_);
  }
  asm volatile("cp.async.wait_group 1;" ::: "memory");
  __syncthreads();
  Q_ST(4 + 16 * wg);
  uint32_t qf[2][4];
  if (wg == 0) {
    float x[2][16];
#pragma unroll
    for (int e = 0; e < 2; e++) {
      int const tok = 2 * w + e;
      float ss = 0.f;
#pragma unroll
      for (int c4 = 0; c4 < 4; c4++) {
        uint2 const raw = *reinterpret_cast<const uint2*>(sm.qs + (8 * tok + g) * 160 + 32 * c4 + 8 * t);
        __nv_bfloat162 const u0 = *reinterpret_cast<const __nv_bfloat162*>(&raw.x);
        __nv_bfloat162 const u1 = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
        x[e][4 * c4 + 0] = __bfloat162float(u0.x); x[e][4 * c4 + 1] = __bfloat162float(u0.y);
        x[e][4 * c4 + 2] = __bfloat162float(u1.x); x[e][4 * c4 + 3] = __bfloat162float(u1.y);
#pragma unroll
        for (int i = 0; i < 4; i++) ss += x[e][4 * c4 + i] * x[e][4 * c4 + i];
      }
      ss += __shfl_xor_sync(0xffffffffu, ss, 1);
      ss += __shfl_xor_sync(0xffffffffu, ss, 2);
      float const rn = rsqrtf(ss * (1.0f / V_D) + eps);
#pragma unroll
      for (int c4 = 0; c4 < 4; c4++)
#pragma unroll
        for (int i2 = 0; i2 < 2; i2++) {
          float2 const t2_ = sm.tab[tok * 32 + 8 * c4 + 2 * t + i2];
          float const cs = t2_.x, sn = t2_.y;
          float const ev = x[e][4 * c4 + 2 * i2] * rn, od = x[e][4 * c4 + 2 * i2 + 1] * rn;
          x[e][4 * c4 + 2 * i2] = ev * cs - od * sn;
          x[e][4 * c4 + 2 * i2 + 1] = ev * sn + od * cs;
        }
    }
#pragma unroll
    for (int ks = 0; ks < 2; ks++)
#pragma unroll
      for (int e = 0; e < 2; e++)
#pragma unroll
        for (int hi = 0; hi < 2; hi++) {
          int const c4 = 2 * ks + hi;
          qf[ks][e + 2 * hi] = v_e4m3x2(x[e][4 * c4], x[e][4 * c4 + 1]) | (v_e4m3x2(x[e][4 * c4 + 2], x[e][4 * c4 + 3]) << 16);
        }
    Q_ST(5);
  } else {
    {
      uint32_t const* s_ = &sm.nks[(tid & 127) * 8];
      nk_kraw[0] = *reinterpret_cast<const __nv_bfloat162*>(s_); nk_kraw[1] = *reinterpret_cast<const __nv_bfloat162*>(s_ + 1);
      nk_vraw[0] = *reinterpret_cast<const __nv_bfloat162*>(s_ + 2); nk_vraw[1] = *reinterpret_cast<const __nv_bfloat162*>(s_ + 3);
      nk_qraw = *reinterpret_cast<const __nv_bfloat162*>(s_ + 4);
      nk_crow = *reinterpret_cast<const long long*>(s_ + 6);
    }
#pragma unroll
    for (int i = 0; i < 2; i++) {
      int const j = w + 4 * i;
      float2 kv = __bfloat1622float2(nk_kraw[i]);
      float2 const vv = __bfloat1622float2(nk_vraw[i]);
      float ks = kv.x * kv.x + kv.y * kv.y;
#pragma unroll
      for (int o2 = 16; o2 > 0; o2 >>= 1) ks += __shfl_xor_sync(0xffffffffu, ks, o2);
      float const krn = rsqrtf(ks * (1.0f / V_D) + eps);
      float2 const tk_ = sm.tab[j * 32 + lane];
      float const ck = tk_.x, sk = tk_.y;
      kv.x *= krn; kv.y *= krn;
      uint32_t const k8 = v_e4m3x2((kv.x * ck - kv.y * sk) * k_rscale, (kv.x * sk + kv.y * ck) * k_rscale);
      uint32_t const v8 = v_e4m3x2(vv.x * v_rscale, vv.y * v_rscale);
      if (j == otok && (r & 1) == 0) {
        *reinterpret_cast<uint16_t*>(k_cache + nk_crow * 128 + hk * 64 + 2 * lane) = (uint16_t)k8;
        *reinterpret_cast<uint16_t*>(v_cache + nk_crow * 128 + hk * 64 + 2 * lane) = (uint16_t)v8;
      }
      float2 const kd = v_e4m3x2_dec(k8), vd = v_e4m3x2_dec(v8);
      sm.kdec[j * 65 + 2 * lane] = kd.x; sm.kdec[j * 65 + 2 * lane + 1] = kd.y;
      *reinterpret_cast<float2*>(sm.vdec + j * 64 + 2 * lane) = vd;
    }
    {
      float2 qv = __bfloat1622float2(nk_qraw);
      float qs = qv.x * qv.x + qv.y * qv.y;
#pragma unroll
      for (int o2 = 16; o2 > 0; o2 >>= 1) qs += __shfl_xor_sync(0xffffffffu, qs, o2);
      float const qrn = rsqrtf(qs * (1.0f / V_D) + eps);
      float2 const tq_ = sm.tab[otok * 32 + lane];
      float const cq = tq_.x, sq = tq_.y;
      qv.x *= qrn; qv.y *= qrn;
      uint32_t const q8 = v_e4m3x2(qv.x * cq - qv.y * sq, qv.x * sq + qv.y * cq);
      float2 const qd = v_e4m3x2_dec(q8);
      sm.qdec[w * 65 + 2 * lane] = qd.x; sm.qdec[w * 65 + 2 * lane + 1] = qd.y;
    }
    Q_ST(16 + 5);
  }
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  v_fence_async();
  __syncthreads();
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
  Q_ST(6 + 16 * wg);
  uint8_t* const mrb = sm.mrb;
  uint32_t const mbpush = v_u32(&sm.mbm);
  if (wg == 0) {

    float mref[2] = {V_MINIT, V_MINIT};
    float lA = 0.f, lB = 0.f;
    int const tokA = 2 * w, tokB = 2 * w + 1;
    int const lowA = seq + tokA - window + 1, lowB = seq + tokB - window + 1;
    if (ntiles > 0) {
      uint32_t sc[32];
      uint32_t const kb = v_u32(sm.kt) + 64 * hk;
      wg_fence();
      wg_qk16_0(sc, qf[0], v_desc(kb));
      wg_qk16_1(sc, qf[1], v_desc(kb + 32));
      wg_commit();
      wg_wait0();
      Q_ST(7);
      int const tbase = p0, tlim = p1;
      if (tbase < lowB) {
#pragma unroll
        for (int e = 0; e < 2; e++) {
          int const pos = tbase + 2 * t + e;
          uint32_t const keep = e ? 0x0000FFFFu : 0xFFFF0000u, neg = e ? 0xFC000000u : 0x0000FC00u;
          if (pos < lowA) sc[0] = (sc[0] & keep) | neg;
          if (pos < lowB) sc[1] = (sc[1] & keep) | neg;
        }
      }
      if (tbase + 128 > tlim) {
#pragma unroll
        for (int j8 = 0; j8 < 16; j8++)
#pragma unroll
          for (int e = 0; e < 2; e++) {
            uint32_t const keep = e ? 0x0000FFFFu : 0xFFFF0000u, neg = e ? 0xFC000000u : 0x0000FC00u;
            if (tbase + 8 * j8 + 2 * t + e >= tlim) {
              sc[2 * j8] = (sc[2 * j8] & keep) | neg;
              sc[2 * j8 + 1] = (sc[2 * j8 + 1] & keep) | neg;
            }
          }
      }
      __half2 mA2 = v_h2(sc[0]), mB2 = v_h2(sc[1]);
#pragma unroll
      for (int j8 = 1; j8 < 16; j8++) { mA2 = __hmax2(mA2, v_h2(sc[2 * j8])); mB2 = __hmax2(mB2, v_h2(sc[2 * j8 + 1])); }
      float mxA = fmaxf(__low2float(mA2), __high2float(mA2)), mxB = fmaxf(__low2float(mB2), __high2float(mB2));
      mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffffu, mxA, 1)); mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffffu, mxA, 2));
      mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffffu, mxB, 1)); mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffffu, mxB, 2));
      mxA *= c2; mxB *= c2;
      if (mxA > mref[0] + 8.5f) mref[0] = ceilf(mxA * 8.f) * 0.125f - 8.f;
      if (mxB > mref[1] + 8.5f) mref[1] = ceilf(mxB * 8.f) * 0.125f - 8.f;
#if ZH_SCH
      __half2 const c8h = __float2half2_rn(8.f * c2);
      __half2 const lo2 = __float2half2_rn(1024.f);
      __half2 const bA = __float2half2_rn(mref[0] > -1.0e29f ? 1080.f - 8.f * mref[0] : -60000.f);
      __half2 const bB = __float2half2_rn(mref[1] > -1.0e29f ? 1080.f - 8.f * mref[1] : -60000.f);
#pragma unroll
      for (int j8 = 0; j8 < 16; j8++) {
        sc[2 * j8] = v_u(__hmax2(__hfma2(v_h2(sc[2 * j8]), c8h, bA), lo2));
        sc[2 * j8 + 1] = v_u(__hmax2(__hfma2(v_h2(sc[2 * j8 + 1]), c8h, bB), lo2));
      }
#else
      // OKC: exact exp. P = 2^(s*c - mref) (at most 2^8.5) = ex2(fma(s, c, -mref)); masked scores (-inf) give 0
      __half2 const c1h = __float2half2_rn(c2);
      __half2 const bA = __float2half2_rn(mref[0] > -1.0e29f ? -mref[0] : -60000.f);
      __half2 const bB = __float2half2_rn(mref[1] > -1.0e29f ? -mref[1] : -60000.f);
#pragma unroll
      for (int j8 = 0; j8 < 16; j8++) {
        sc[2 * j8] = v_ex2h2(v_u(__hfma2(v_h2(sc[2 * j8]), c1h, bA)));
        sc[2 * j8 + 1] = v_ex2h2(v_u(__hfma2(v_h2(sc[2 * j8 + 1]), c1h, bB)));
      }
#endif
      {
        int const rA = 16 * w + g, rB = rA + 8;
#pragma unroll
        for (int jj = 0; jj < 8; jj++) {
#if ZH_SCH
          uint32_t const wA = v_prmt(sc[4 * jj], sc[4 * jj + 2], 0x6420), wB = v_prmt(sc[4 * jj + 1], sc[4 * jj + 3], 0x6420);
#else
          uint32_t const wA = v_cvt_e4m3x2h(sc[4 * jj]) | (v_cvt_e4m3x2h(sc[4 * jj + 2]) << 16);
          uint32_t const wB = v_cvt_e4m3x2h(sc[4 * jj + 1]) | (v_cvt_e4m3x2h(sc[4 * jj + 3]) << 16);
#endif
          *reinterpret_cast<uint32_t*>(sm.pt + rA * 128 + ((jj ^ (rA & 7)) << 4) + 4 * t) = wA;
          *reinterpret_cast<uint32_t*>(sm.pt + rB * 128 + ((jj ^ (rB & 7)) << 4) + 4 * t) = wB;
          {
            float2 const a0 = v_e4m3x2_dec(wA), a1 = v_e4m3x2_dec(wA >> 16), c0 = v_e4m3x2_dec(wB), c1 = v_e4m3x2_dec(wB >> 16);
            lA += (a0.x + a0.y) + (a1.x + a1.y);
            lB += (c0.x + c0.y) + (c1.x + c1.y);
          }
        }
      }
    }
    lA += __shfl_xor_sync(0xffffffffu, lA, 1); lA += __shfl_xor_sync(0xffffffffu, lA, 2);
    lB += __shfl_xor_sync(0xffffffffu, lB, 1); lB += __shfl_xor_sync(0xffffffffu, lB, 2);
    v_fence_async();
    __threadfence_block();
    asm volatile("bar.arrive 1, 256;" ::: "memory");
    Q_ST(8);
    if (t == 0) {
      uint32_t d0, d1, b0, b1;
      int const owA = 4 * w + (g >> 2), owB = owA + 2, rib = g & 3;
      uint32_t const mbsrc = mbpush;
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d0) : "r"(v_u32(mrb + r * Q_BLK + 1088 + rib * 4)), "r"(owA));
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(b0) : "r"(mbsrc), "r"(owA));
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d1) : "r"(v_u32(mrb + r * Q_BLK + 1088 + rib * 4)), "r"(owB));
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(b1) : "r"(mbsrc), "r"(owB));
      asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];" :: "r"(d0), "f"(mref[0]), "r"(b0) : "memory");
      asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];" :: "r"(d1), "f"(mref[1]), "r"(b1) : "memory");
      asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];" :: "r"(d0 + 16), "f"(lA), "r"(b0) : "memory");
      asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];" :: "r"(d1 + 16), "f"(lB), "r"(b1) : "memory");
    }

    float nk_m = V_MINIT, nk_l = 0.f, nk_o0 = 0.f, nk_o1 = 0.f;
    {
      int const j = lane & 7, dq = 16 * (lane >> 3);
      float d = 0.f;
#pragma unroll
      for (int i = 0; i < 16; i++) d += sm.qdec[w * 65 + dq + i] * sm.kdec[j * 65 + dq + i];
      d += __shfl_xor_sync(0xffffffffu, d, 8);
      d += __shfl_xor_sync(0xffffffffu, d, 16);
      float const sj = j <= otok ? c2 * d : V_MINIT;
      float mx = sj;
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));
      float const pj = j <= otok ? v_ex2(sj - mx) : 0.f;
      float sl = pj;
      sl += __shfl_xor_sync(0xffffffffu, sl, 1);
      sl += __shfl_xor_sync(0xffffffffu, sl, 2);
      sl += __shfl_xor_sync(0xffffffffu, sl, 4);
      nk_m = mx; nk_l = sl;
      if (lane < 8) sm.pp[w * 8 + lane] = pj;
      __syncwarp();
#pragma unroll
      for (int jj = 0; jj < V_L; jj++) {
        float const p = sm.pp[w * 8 + jj];
        float2 const v = *reinterpret_cast<const float2*>(sm.vdec + jj * 64 + 2 * lane);
        nk_o0 += p * v.x; nk_o1 += p * v.y;
      }
    }
    Q_ST(9);
    asm volatile("{\n.reg .pred p;\nPHW_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n@!p bra PHW_%=;\n}\n"
                 :: "r"(v_u32(&sm.mbm)) : "memory");
    Q_ST(10);
    {
      int const sl = lane & 15;
      float const ms = *reinterpret_cast<const float*>(mrb + sl * Q_BLK + 1088 + w * 4);
      float const ls = *reinterpret_cast<const float*>(mrb + sl * Q_BLK + 1104 + w * 4);
      float M = fmaxf(ms, nk_m);
#pragma unroll
      for (int o2 = 1; o2 < 16; o2 <<= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o2));
      float const ws = v_ex2(ms - M);
      float L2 = ls * ws;
#pragma unroll
      for (int o2 = 1; o2 < 16; o2 <<= 1) L2 += __shfl_xor_sync(0xffffffffu, L2, o2);
      float const wn = v_ex2(nk_m - M);
      L2 += nk_l * wn;
      float b0 = nk_o0 * wn, b1 = nk_o1 * wn;
#pragma unroll
      for (int src = 0; src < 16; src++) {
        float const wt = __shfl_sync(0xffffffffu, ws, src);
        float2 const ov = *reinterpret_cast<const float2*>(mrb + src * Q_BLK + w * 272 + 8 * lane);
        b0 += ov.x * wt; b1 += ov.y * wt;
      }
      float const inv = __fdividef(v_scale, L2);
      b0 *= inv; b1 *= inv;
      float ss = b0 * b0 + b1 * b1;
#pragma unroll
      for (int o2 = 16; o2 > 0; o2 >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o2);
      float const rn = rsqrtf(ss * (1.0f / V_D) + eps);
      float const s0 = __fdividef(1.f, 1.f + __expf(-__bfloat162float(gpre.x)));
      float const s1 = __fdividef(1.f, 1.f + __expf(-__bfloat162float(gpre.y)));
      *reinterpret_cast<__nv_bfloat162*>(out + (long long)otok * (V_HQ * V_D) + (oh0 + w) * V_D + 2 * lane) =
          __floats2bfloat162_rn(b0 * rn * s0, b1 * rn * s1);
    }
    Q_ST(11);
  } else {

    float o[32], la[32];
#pragma unroll
    for (int i = 0; i < 32; i++) { o[i] = 0.f; la[i] = 0.f; }
    uint32_t va[4][4];
    if (ntiles > 0) {
      uint32_t const vaddr = v_u32(sm.vt);
#pragma unroll
      for (int ks = 0; ks < 4; ks++) {
        int const key = 32 * ks + 8 * (lane >> 3) + (lane & 7);
        uint32_t r0, r1, r2, r3;
        v_ldsm4t(r0, r1, r2, r3, vaddr + key * 128 + ((((4 * hk + w) ^ (key & 7))) << 4));
        va[ks][0] = v_prmt(r0, r1, 0x6420); va[ks][1] = v_prmt(r0, r1, 0x7531);
        va[ks][2] = v_prmt(r2, r3, 0x6420); va[ks][3] = v_prmt(r2, r3, 0x7531);
      }
    }
    Q_ST(16 + 7);
    asm volatile("bar.sync 1, 256;" ::: "memory");
    Q_ST(16 + 8);
    if (ntiles > 0) {
      uint32_t const ones[4] = {0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u};
      uint32_t const pb = v_u32(sm.pt);
      wg_fence();
#pragma unroll
      for (int ks = 0; ks < 4; ks++) wg_pv(o, va[ks], v_desc(pb + 32 * ks));
      wg_commit();
      wg_wait0();
    }
    Q_ST(16 + 9);

    Q_ST(16 + 10);
#pragma unroll
    for (int jj = 0; jj < 8; jj++) {
      int const ow = 2 * jj + (t >> 1);
      uint32_t mbr;
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(mbr) : "r"(mbpush), "r"(ow));
#pragma unroll
      for (int e = 0; e < 2; e++) {
        int const rib = 2 * (t & 1) + e;
        float const v0 = o[4 * jj + e], v1 = o[4 * jj + 2 + e];
        uint32_t dst;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(dst) : "r"(v_u32(mrb + r * Q_BLK + rib * 272 + (16 * w + 2 * g) * 4)), "r"(ow));
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1, %2}, [%3];"
                     :: "r"(dst), "f"(v0), "f"(v1), "r"(mbr) : "memory");
        if (!VP2_LWG0 && g == 0 && (jj >> 1) == w) {
          uint32_t dl;
          asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(dl) : "r"(v_u32(mrb + r * Q_BLK + 1104 + rib * 4)), "r"(ow));
          asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];" :: "r"(dl), "f"(la[4 * jj + e]), "r"(mbr) : "memory");
        }
      }
    }
    Q_ST(16 + 11);
  }
}

void vp2_launch(int64_t qkv, int64_t page_table, int64_t pt_stride, int64_t seq_lens, int64_t cache_loc,
                int64_t k_cache, int64_t v_cache, int64_t inv_freq, int64_t out, int64_t B, int64_t window,
                double c2, double k_rscale, double v_rscale,
                double v_scale, double eps, int64_t dbgp, int64_t stream) {
  static bool configured = false;
  int const smem = (int)sizeof(VP2Smem) + 1024;
  if (!configured) {
    cudaFuncSetAttribute(vp2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaFuncSetAttribute(vp2_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    configured = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(32, (unsigned)B, 1);
  cfg.blockDim = dim3(256, 1, 1);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = reinterpret_cast<cudaStream_t>(stream);
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = 16;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  cudaError_t e = cudaLaunchKernelEx(&cfg, vp2_kernel, reinterpret_cast<const __nv_bfloat16*>(qkv),
                                     reinterpret_cast<const int*>(page_table), reinterpret_cast<const int*>(seq_lens),
                                     reinterpret_cast<const long long*>(cache_loc), reinterpret_cast<uint8_t*>(k_cache),
                                     reinterpret_cast<uint8_t*>(v_cache), reinterpret_cast<const float*>(inv_freq),
                                     (int)window, (float)c2, (float)k_rscale, (float)v_rscale, (float)eps,
                                     reinterpret_cast<unsigned long long*>(dbgp), reinterpret_cast<__nv_bfloat16*>(out),
                                     (float)v_scale, (int)pt_stride);
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
#undef Q_BLK
#undef Q_TX
#undef Q_ST
#undef VP2_PUSH
#undef VP2_LATEQ
#undef VP2_ST
#undef VP2_QSM
#undef VP2_LWG0
#undef VP2_MLB
#undef VP2_P64
#undef VP2_NOTF
#undef VP2_STRM
}
}

void muse_vp2_launch(int64_t qkv, int64_t page_table, int64_t pt_stride, int64_t seq_lens, int64_t cache_loc,
    int64_t k_cache, int64_t v_cache, int64_t inv_freq, int64_t out, int64_t B, int64_t window, double c2,
    double k_rscale, double v_rscale, double v_scale, double eps, int64_t dbgp,
    int64_t stream) { vfgns::b1vp2::vp2_launch(qkv, page_table, pt_stride, seq_lens, cache_loc, k_cache, v_cache,
    inv_freq, out, B, window, c2, k_rscale, v_rscale, v_scale, eps, dbgp, stream); }
void muse_vfg_launch(int64_t qkv, int64_t page_table, int64_t pt_stride, int64_t seq_lens, int64_t cache_loc,
    int64_t k_cache, int64_t v_cache, int64_t inv_freq, int64_t out, int64_t B, int64_t NS, int64_t window, double c2,
    double k_rscale, double v_rscale, double v_scale, double eps, int64_t ws, int64_t dbgp,
    int64_t stream) { vfgns::vfg_launch(qkv, page_table, pt_stride, seq_lens, cache_loc, k_cache, v_cache, inv_freq,
    out, B, NS, window, c2, k_rscale, v_rscale, v_scale, eps, ws, dbgp, 0, stream); }
"""
ZH_VFG_CPP = r"""
void muse_vfg_launch(int64_t qkv, int64_t page_table, int64_t pt_stride, int64_t seq_lens, int64_t cache_loc,
                     int64_t k_cache, int64_t v_cache, int64_t inv_freq, int64_t out, int64_t B, int64_t NS,
                     int64_t window, double c2, double k_rscale, double v_rscale, double v_scale, double eps,
                     int64_t ws, int64_t dbgp, int64_t stream);
void muse_vp2_launch(int64_t qkv, int64_t page_table, int64_t pt_stride, int64_t seq_lens, int64_t cache_loc,
                     int64_t k_cache, int64_t v_cache, int64_t inv_freq, int64_t out, int64_t B, int64_t window,
                     double c2, double k_rscale, double v_rscale, double v_scale, double eps, int64_t dbgp,
                     int64_t stream);
"""
_ZH = {}
_ZH_WS = {}
_ZH_THREADS = []


def _zh_tag(*parts):
    return _zh_hashlib.sha1("\n".join(parts).encode()).hexdigest()[:10]


def _zh_build_wst():
    try:
        flags = ["-O3", "--use_fast_math", "-gencode=arch=compute_90a,code=sm_90a", "-DOKC_WST_ACC32=1", "-DZH_SCH=1"]
        _ZH["wst"] = _zh_load_inline(
            name="okc_zh_wst_" + _zh_tag(ZH_WST_CUDA, *flags), cpp_sources=[ZH_WST_CPP], cuda_sources=[ZH_WST_CUDA],
            functions=["wst_attn_bal"], extra_cuda_cflags=flags, verbose=False)
    except Exception as exc:  # noqa: BLE001  (a failed build leaves the vjeux path in charge)
        _ZH["wst_err"] = repr(exc)[-1500:]
        print("zh wst build failed:", _ZH["wst_err"])


def _zh_build_vfg():
    try:
        flags = ["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-U__CUDA_NO_HALF_OPERATORS__",
                 "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__", "-DZH_SCH=1"]
        _ZH["vfg"] = _zh_load_inline(
            name="okc_zh_vfg_" + _zh_tag(ZH_VFG_CUDA, *flags), cpp_sources=[ZH_VFG_CPP], cuda_sources=[ZH_VFG_CUDA],
            functions=["muse_vfg_launch", "muse_vp2_launch"], extra_cuda_cflags=flags, verbose=False)
    except Exception as exc:  # noqa: BLE001
        _ZH["vfg_err"] = repr(exc)[-1500:]
        print("zh vfg build failed:", _ZH["vfg_err"])


def _zh_start_builds():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        return
    jobs = [_module]
    jobs.append(_zh_build_wst)
    jobs.append(_zh_build_vfg)
    for fn in jobs:
        th = _zh_threading.Thread(target=fn, daemon=True)
        th.start()
        _ZH_THREADS.append(th)


def _zh_join():
    while _ZH_THREADS:
        _ZH_THREADS.pop().join()


def _zh_dev_ws(device):
    """Scratch for wst_ct14b: partial rows (G + B) x 2 kv heads x 64 (token, head) rows, sized once for B <= ZH_WST_MAX_BATCH
    (captured graphs keep the address) and the per-(B) fold tickets (+256 per launch, so they need no reset)."""
    key = str(device)
    ws = _ZH_WS.get(key)
    if ws is None:
        n = (_num_sms(device) + ZH_WST_MAX_BATCH) * 2 * 64
        ws = _ZH_WS[key] = (torch.empty((n, 64), dtype=torch.float32, device=device),
                            torch.empty((n,), dtype=torch.float32, device=device),
                            torch.empty((n,), dtype=torch.float32, device=device))
    return ws


def _zh_tickets(B, device):
    key = (B, str(device))
    t = _ZH_WS.get(key)
    if t is None:
        t = _ZH_WS[key] = torch.zeros((2 * B,), dtype=torch.int32, device=device)
    return t


def _zh_common_ok(data):
    cfg = data["config"]
    pt = data["page_table"]
    return (cfg["q_len"] == 8 and cfg["page_size"] == 1 and cfg["num_q_heads"] == 16 and cfg["num_kv_heads"] == 2
            and cfg["head_dim"] == 64 and pt.dtype == torch.int32 and pt.stride(1) == 1 and data["cache_loc"].dtype == torch.int64
            and data["seq_lens"].dtype == torch.int32 and data["k_cache"].dtype == torch.float8_e4m3fn
            and data["qkv"].dtype == torch.bfloat16 and data["qkv"].is_contiguous() and _fits(data))


def _zh_route(data):
    """'wst' / 'vfg' / 'vp2' for the scored verify shapes his kernels cover, else None (decided by config and shapes only)."""
    cfg = data["config"]
    if not _zh_common_ok(data):
        return None
    B = cfg["batch"]
    if not cfg["window"]:
        if ZH_WST_MIN_BATCH <= B <= ZH_WST_MAX_BATCH and "wst" in _ZH:
            return "wst"
        if ZH_VFG_FULL1 and B == 1 and "vfg" in _ZH:
            return "vfg"
    elif cfg["window"] == 2048:
        if ZH_VFG_SLIDING and ZH_VFG_MIN_BATCH <= B <= ZH_VFG_MAX_BATCH and "vfg" in _ZH:
            return "vfg"
        if ZH_USE_VP2 and B == 1 and "vfg" in _ZH:
            return "vp2"
    return None


def _zh_wst(data):
    cfg = data["config"]
    B, L = cfg["batch"], 8
    qkv = data["qkv"]
    device = qkv.device
    acc, m_, l_ = _zh_dev_ws(device)
    out = torch.empty((B, L, 16 * 64), dtype=torch.bfloat16, device=device)
    pt = data["page_table"]
    smqk = cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E
    rk = 1.0 / cfg["kv_scale"]
    _ZH["wst"].wst_attn_bal(qkv.view(B * L, -1), data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), pt,
                            data["seq_lens"], data["cache_loc"], acc, m_, l_, data["k_scale"], smqk, B, L, pt.stride(0),
                            _num_sms(device), 0, 3, rk, rk, 1e-5, out.view(B * L, -1), _zh_tickets(B, device), data["v_scale"])
    return out


def _zh_vfg(data, route):
    cfg = data["config"]
    B = cfg["batch"]
    qkv = data["qkv"]
    device = qkv.device
    window = cfg["window"]
    pt = data["page_table"]
    out = torch.empty((B, 8, 16 * 64), dtype=torch.bfloat16, device=device)
    freq = _inv_freq(cfg["rope_theta"], device).data_ptr() if window else 0
    c2 = cfg["qk_scale_factor"] * cfg["sm_scale"] * cfg["kv_scale"] * LOG2E
    rk = 1.0 / cfg["kv_scale"]
    stream = torch.cuda.current_stream().cuda_stream
    ext = _ZH["vfg"]
    if route == "vp2":
        ext.muse_vp2_launch(qkv.data_ptr(), pt.data_ptr(), pt.stride(0), data["seq_lens"].data_ptr(), data["cache_loc"].data_ptr(),
                            data["k_cache"].data_ptr(), data["v_cache"].data_ptr(), freq, out.data_ptr(), B, window, c2,
                            rk, rk, cfg["kv_scale"], 1e-5, 0, stream)
        return out
    G = _num_sms(device)
    ns = max(1, G // B)
    key = ("vfg_ws", str(device))
    ws = _ZH_WS.get(key)
    if ws is None:   # B * NS <= SMs partial records of 128 rows x 72 floats, shared by every B
        ws = _ZH_WS[key] = torch.empty((G * 128 * 72,), dtype=torch.float32, device=device)
    ext.muse_vfg_launch(qkv.data_ptr(), pt.data_ptr(), pt.stride(0), data["seq_lens"].data_ptr(), data["cache_loc"].data_ptr(),
                        data["k_cache"].data_ptr(), data["v_cache"].data_ptr(), freq, out.data_ptr(), B, ns, window, c2,
                        rk, rk, cfg["kv_scale"], 1e-5, ws.data_ptr(), 0, stream)
    return out


_zh_start_builds()


def kernel_fn(data: Inputs) -> torch.Tensor:
    if _ZH_THREADS:
        _zh_join()
    if data["config"]["q_len"] == 8:
        route = _zh_route(data)
        if route == "wst":
            return _zh_wst(data)
        if route is not None:
            return _zh_vfg(data, route)
    return _vj_kernel_fn(data)
