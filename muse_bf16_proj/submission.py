# okc: problem=muse_bf16_proj
# From github.com/keell0renz/muse-open-kernel-challenge (commit 8e30e1b), Copyright 2026 team keell0renz, Apache-2.0; see NOTICE_keell0renz.
# + per-thread primary CUDA context bind around kernel_fn (team keell0renz, 2026-10-31: a state first called from a fresh OS thread failed cuTensorMapEncodeTiled 201 at m 16/32; pattern from alygg77 / Matsvei)
# + v6 (team keell0renz, E330+): every m in 2..384 that is not one of the scored 1 / 8 / 16 / 32 / 128 / 256 gets a per-(layer, m) route built and validated in prepare() (pinned kernels called with the real m, rq plans with TMA zero-fill, cuBLAS on the K-major weight, the m=256 cuBLASLt pin); the scored m take exactly the v5 path
# + router m=128 rk K-split 16 (grid 128, was 12) from the S sweep (team keell0renz)
# + alygg77/okc-muse-spark-kernels muse_bf16_proj.py (Apache-2.0): pinned cuBLASLt m=256 plans + ctypes wrapper (pins and CUDA-context bind by Matsvei 'matvey'), routed per cell: qkv_gate:256, o:256, pooler:256, unpooler:256 (team keell0renz)
# Per-shape dispatcher over public kernels: chenwainuo/okc (Apache-2.0),  vjeux/okc-2026 (Apache-2.0), NathenFernandes/muse-bf16-proj (MIT); each file's code is kept verbatim inside a factory function, picked per (layer, m) from judge profiles (team keell0renz).
_PRUNE = True   # prepare only the cells the final _TABLE routes to each kernel (judge prepare limit 300 s)


def _make_chenwainuo():
    # okc: problem=muse_bf16_proj
    # Based on github.com/chenwainuo/okc muse_bf16_proj/submission.py @ 3d418e1 (Apache-2.0). Copied for the OKC starting point.
    """Stream-K skinny GEMM: out[m, n] = x @ W[:n].T for m <= 32 (and the router at any m).
W is repacked into mma.sync A-fragment order (2 KB contiguous per warp per 64-wide K chunk); every warp streams an
equal contiguous range of (tile, chunk) units; tiles split across warps meet through fp32 atomics and the last
arriver writes the output. The FP32 router runs as a BF16 hi + lo pair with FP32 accumulation.
v13/v14: up to 32 tokens, the BF16 projections and the router (12-bit hi + lo) run a register-direct GEMV on losslessly 12-bit-packed weights (see GVCU),
launch geometry picked per shape in prepare() by timing against the v12 path.
pw11 ("gvx"): r6-c12 unchanged (every r6 path is still baked and kept as the per-shape baseline "r6"), plus gv geometries
with 64-144 rows per CTA (x fetched once per 128 rows instead of once per 16) and a cross-CTA K split of 2/4/8, reduced
either through cluster DSMEM (path suffix a, r6's own S > 1 epilogue) or through an fp32 workspace + last-arriving CTA
(suffix b, V bit 2, no cluster); suffix -s = x in a shared-memory slab instead of through L1; the router gets the
12-bit codec on both hi and lo. Each new path is validated (one-hot x8 exact, random rows) and timed against the r6
choice in a second bake (BAKE / PICK lines); kernel_fn takes the pw11 pick when it exists, else r6's dispatch.
pw12: the same gv kernel with W streamed through a private per-warp cp.async.bulk smem ring (V bit 3, paths gb-*);
decode, mma, escapes, reduction and epilogue unchanged. PROBE gvbulk lines compare it with r6 on pooler m=1.
pw18 (= pw17 with codec E v3): r6 unchanged (its 12-bit paths, bake and dispatch are SASS-identical) plus codec E paths
(gv FMT 4, "ge-*"). The code length is fixed by a code's first 3 bits (LOP3 + PRMT on an 8-byte table), so the per-lane
bit-position chain has no memory access. The exponent pair value comes from a 256-entry smem LUT (single or lane-replicated)
off the chain. There are 1, 2 or 4 independent substreams per lane, and refill words come through a conflict-free
[substream][slot][lane] 4 B cp.async ring and are prefetched. 11.07 / 11.17 / 11.32 bits/weight (K 1 / 2 / 4) against 12
(proj_rsi/tools/CODEC_E_LAYOUT.md v3). Each pack is proven bit-exact by the reference unpacker before any E kernel runs. Each
path is validated like every second-bake path and timed against r6's pick (BAKE / PICK). PROBE ecod lines time the best E path
against r6 on pooler m=1 and m=16, alternated.
pw29: pw23 plus wgmma router paths at m = 128 / 256 (rwk, its own sm_90a extension), baked against pw23's rs / rp / rv
paths and r6's cuBLASLt path.
pw31 (scored merge): pw27h (P1: flag-zeroed split-K router m = 8/16/32 and unpooler m <= 8 on r6's gv, rr6 at router
m = 1) + pw29h (wgmma split-K router m = 128 / 256 with self-zeroing); the two regions are disjoint (3-way merge on pw23).
pw34 = pw33 + rs128 (audits/m128-plan.md): pw23 unchanged plus a swap-AB register-A split-K wgmma kernel for m exactly 128 / 256 on the
pooler, qkv_gate, unpooler and o layers: 128 W rows x all tokens per CTA, two math warpgroups decode the lossless 12-bit
pack (pw10 _pack_nat + _c12, re-blocked to 12 KB per 128-row k64 stage) into wgmma A registers, x is the smem B operand
(2D TMA, SW128), one full barrier per ring slot shared by a W bulk lane and an x TMA lane, K split over KS CTAs on a
one-wave grid, partials summed by cp.reduce.async.bulk.tensor bf16 add into out, which the kernel zeroes itself (each
CTA its slice, release counter, polled off the critical path; no memset node, no cluster). Built in a background thread
during r6's phase; BAKE / PICK lines time it against r6's own path per (layer, m); any failure leaves r6's choices."""
    from typing import TypedDict
    import math
    import statistics
    import time
    import torch
    _T0 = time.time()
    from torch.utils.cpp_extension import load_inline

    def _NC(layer, m):
        return (not _PRUNE) or _TABLE.get('%s:%d' % (layer, m), _DEFAULT) == 'chenwainuo'

    def _NCR(layer, ms):
        return any(_NC(layer, m_) for m_ in ms)


    class Weights(TypedDict):
        qkv_gate: torch.Tensor
        o: torch.Tensor
        pooler: torch.Tensor
        unpooler: torch.Tensor
        router: torch.Tensor


    class Config(TypedDict):
        m: int
        n: int
        k: int
        layer: str


    class Inputs(TypedDict):
        x: torch.Tensor
        config: Config


    CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>
#ifndef WPC
#define WPC 16
#endif
#ifndef V7
#define V7 0
#endif
#ifndef RSN
#define RSN 2
#endif
#ifndef CUN
#define CUN 2
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define TMA 1
#else
#define TMA 0
#endif

__device__ __forceinline__ void mma(float* c, const uint4& a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint4 lds(uint32_t a) { uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v; }
__device__ __forceinline__ void mbinit(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(b), "r"(n)); }
__device__ __forceinline__ void mbwait(uint32_t b, uint32_t ph) {
#if TMA
  asm volatile("{\n.reg .pred p;\nW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" :: "r"(b), "r"(ph) : "memory");
#else
  asm volatile("{\n.reg .pred p;\nW: mbarrier.test_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" :: "r"(b), "r"(ph) : "memory");
#endif
}
__device__ __forceinline__ void cp16(uint32_t s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(g)); }

struct Q {
  const uint8_t* A; const uint8_t* A2; const __nv_bfloat16* x; void* out; float* ws; int* cnt;
  int M, N, K, MT, KC, TB, f32;
};

__device__ __forceinline__ void put(const Q& q, int tok, int row, float v) {
  if (q.f32) ((float*)q.out)[(size_t)tok * q.N + row] = v; else ((__nv_bfloat16*)q.out)[(size_t)tok * q.N + row] = __float2bfloat16(v);
}

// Each warp streams an equal contiguous range of (tile, unit) work through its own RS-deep TMA ring; a unit is CU
// consecutive 64-wide K chunks (2 KB of A fragments each; the router's lo half rides along). Partial tiles are
// summed with fp32 red.add into ws; a PDL-launched finalize kernel converts ws to the output and clears it.
template <int NT, bool HL, int RS, int CU>
__global__ void __launch_bounds__(WPC * 32, 1) skinny(Q q) {
  constexpr int SB = (HL ? 2 : 1) * CU * 2048;
  constexpr int CH = NT == 1 ? 4 : (NT == 2 ? 2 : 1);
  extern __shared__ __align__(128) uint8_t smem[];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int W = gridDim.x * WPC, gw = blockIdx.x * WPC + wid;
  const uint32_t ring = (uint32_t)__cvta_generic_to_shared(smem) + wid * RS * SB;
  const uint32_t bars = (uint32_t)__cvta_generic_to_shared(smem) + WPC * RS * SB + wid * RS * 8;
#if TMA
  if (lane < RS) mbinit(bars + lane * 8, 1);
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");  // let the finalize kernel get scheduled early
#else
  if (lane < RS) mbinit(bars + lane * 8, 32);
#endif
  __syncwarp();
  const int KU = q.KC / CU;
  const long long U = (long long)q.TB * q.MT * KU;
  const long long L0 = U * gw / W, Le = U * (gw + 1) / W;
  if (L0 >= Le) return;
  long long pu = L0;
  auto issue = [&](int slot) {
    if (pu < Le) {
      const long long tile = pu / KU; const int cu = (int)(pu - tile * KU), mt = (int)(tile % q.MT);
      const size_t off = ((size_t)mt * q.KC + cu * CU) * 2048;
      const uint32_t d = ring + slot * SB, bar = bars + slot * 8;
#if TMA
      if (lane == 0) {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "n"(SB) : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(d), "l"(q.A + off), "n"(CU * 2048), "r"(bar) : "memory");
        if (HL) asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(d + CU * 2048), "l"(q.A2 + off), "n"(CU * 2048), "r"(bar) : "memory");
      }
#else
#pragma unroll
      for (int k = 0; k < 4 * CU; k++) cp16(d + k * 512 + lane * 16, q.A + off + k * 512 + lane * 16);
      if (HL) {
#pragma unroll
        for (int k = 0; k < 4 * CU; k++) cp16(d + CU * 2048 + k * 512 + lane * 16, q.A2 + off + k * 512 + lane * 16);
      }
      asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(bar) : "memory");
#endif
    } else {
#if !TMA
      asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(bars + slot * 8) : "memory");
#endif
    }
    pu++;
  };
#pragma unroll
  for (int i = 0; i < RS - 1; i++) issue(i);
  uint4 b0[NT][2], b1[NT][2];
  auto ldb = [&](long long cg, uint4 (&b)[NT][2]) {  // cg: global chunk index tile * KC + c
    const long long tile = cg / q.KC; const int c = (int)(cg - tile * q.KC), tb = (int)(tile / q.MT);
#pragma unroll
    for (int n = 0; n < NT; n++) {
      const int tok = min(tb * 32 + n * 8 + g, q.M - 1);
      const uint4* xr = (const uint4*)(q.x + (size_t)tok * q.K) + t * 2 + c * 8;
      b[n][0] = __ldg(xr); b[n][1] = __ldg(xr + 1);
    }
  };
  const long long CE = Le * CU;  // chunk range end
  ldb(L0 * CU, b0);
  float acc[NT][CH][4];
  int qn = 0;
  long long u = L0;
  while (u < Le) {
    const long long tile = u / KU; const int u0 = (int)(u - tile * KU);
    const int u1 = (int)min((long long)KU, u0 + (Le - u));
#pragma unroll
    for (int n = 0; n < NT; n++)
#pragma unroll
      for (int h = 0; h < CH; h++) acc[n][h][0] = acc[n][h][1] = acc[n][h][2] = acc[n][h][3] = 0.f;
    for (int uu = u0; uu < u1; uu++, u++, qn++) {
      const int slot = qn % RS;
      mbwait(bars + slot * 8, (qn / RS) & 1);
      __syncwarp();
      issue((qn + RS - 1) % RS);
#pragma unroll
      for (int i = 0; i < CU; i++) {
        const long long cg = u * CU + i;
        if (cg + 1 < CE) ldb(cg + 1, b1);
        const uint32_t sa = ring + slot * SB + i * 2048 + lane * 16;
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const uint4 a = lds(sa + j * 512);
          uint4 h; if (HL) h = lds(sa + CU * 2048 + j * 512);
#pragma unroll
          for (int n = 0; n < NT; n++) {
            const uint32_t* bb = (const uint32_t*)&b0[n][j >> 1];
            mma(acc[n][j % CH], a, bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);
            if (HL) mma(acc[n][(j + 1) % CH], h, bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);
          }
        }
#pragma unroll
        for (int n = 0; n < NT; n++) { b0[n][0] = b1[n][0]; b0[n][1] = b1[n][1]; }
      }
    }
    const int tb = (int)(tile / q.MT), mt = (int)(tile - (long long)tb * q.MT), row0 = mt * 16 + g, mrem = min(32, q.M - tb * 32);
    float* w = q.ws + (size_t)tb * 32 * q.N;
#pragma unroll
    for (int n = 0; n < NT; n++) {
      float f[4];
#pragma unroll
      for (int i2 = 0; i2 < 4; i2++) { f[i2] = 0.f;
#pragma unroll
        for (int h = 0; h < CH; h++) f[i2] += acc[n][h][i2]; }
      const int tk = n * 8 + 2 * t;
      if (tk < mrem) { atomicAdd(w + (size_t)tk * q.N + row0, f[0]); atomicAdd(w + (size_t)tk * q.N + row0 + 8, f[2]); }
      if (tk + 1 < mrem) { atomicAdd(w + (size_t)(tk + 1) * q.N + row0, f[1]); atomicAdd(w + (size_t)(tk + 1) * q.N + row0 + 8, f[3]); }
    }
#if V7
    {  // the last warp to add into this tile writes it out and clears ws (no finalize kernel)
      auto wof = [&](long long uu) { return (int)(((uu + 1) * W - 1) / U); };  // warp streaming unit uu
      const long long t0u = tile * KU, t1u = t0u + KU - 1;
      const int nc = wof(t1u) - wof(t0u) + 1;
      __syncwarp();
      int last = 0;
      if (lane == 0) { int v; asm volatile("atom.add.acq_rel.gpu.global.s32 %0, [%1], 1;\n" : "=r"(v) : "l"(q.cnt + tile) : "memory"); last = v == nc - 1; }
      last = __shfl_sync(0xffffffffu, last, 0);
      if (last) {
        for (int e = lane; e < 16 * mrem; e += 32) {
          const int tok = e >> 4, row = mt * 16 + (e & 15);
          if (row < q.N) { float* wp = w + (size_t)tok * q.N + row; put(q, tb * 32 + tok, row, __ldcg(wp)); *wp = 0.f; }
        }
        if (lane == 0) q.cnt[tile] = 0;
      }
    }
#endif
  }
}

// ws (fp32 sums, [M][N]) -> out; ws cleared for the next call. Launched as a programmatic dependent.
__global__ void __launch_bounds__(256) fin(Q q) {
#if TMA
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
  const int n4 = q.M * q.N / 4;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
    float4* p = (float4*)q.ws + i;
    const float4 v = __ldcg(p);
    *p = make_float4(0.f, 0.f, 0.f, 0.f);
    if (q.f32) ((float4*)q.out)[i] = v;
    else { __nv_bfloat162 o[2] = {__floats2bfloat162_rn(v.x, v.y), __floats2bfloat162_rn(v.z, v.w)}; ((uint2*)q.out)[i] = *(uint2*)o; }
  }
}

static int g_grid = 0;
template <int NT, bool HL> static void go(const Q& q, cudaStream_t st) {
  constexpr int RS = HL ? 3 : RSN, CU = HL ? 1 : CUN;
  const size_t sm = (size_t)WPC * RS * (HL ? 2 : 1) * CU * 2048 + WPC * RS * 8;
  static bool once = false;
  if (!once) { cudaFuncSetAttribute(skinny<NT, HL, RS, CU>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm); once = true; }
  skinny<NT, HL, RS, CU><<<g_grid, WPC * 32, sm, st>>>(q);
}
void run(torch::Tensor A, torch::Tensor A2, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t N) {
  if (!g_grid) { int dev; cudaGetDevice(&dev); cudaDeviceGetAttribute(&g_grid, cudaDevAttrMultiProcessorCount, dev); }
  Q q;
  q.A = (const uint8_t*)A.data_ptr(); q.A2 = A2.numel() ? (const uint8_t*)A2.data_ptr() : nullptr;
  q.x = (const __nv_bfloat16*)x.data_ptr(); q.out = out.data_ptr(); q.ws = ws.data_ptr<float>(); q.cnt = cnt.data_ptr<int>();
  q.M = x.size(0); q.K = x.size(1); q.N = (int)N; q.MT = (int)((N + 15) / 16); q.KC = q.K / 64; q.TB = (q.M + 31) / 32;
  q.f32 = out.scalar_type() == torch::kFloat32;
  auto st = at::cuda::getCurrentCUDAStream();
  const int nt = q.M >= 32 ? 4 : (q.M + 7) / 8;
  if (q.A2) { if (nt <= 1) go<1, true>(q, st); else if (nt <= 2) go<2, true>(q, st); else go<4, true>(q, st); }
  else { if (nt <= 1) go<1, false>(q, st); else if (nt <= 2) go<2, false>(q, st); else go<4, false>(q, st); }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(std::min(g_grid * 4, (q.M * q.N / 4 + 255) / 256)); cfg.blockDim = dim3(256); cfg.stream = st;
  cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1;
  static int major = 0; if (!major) { int dev; cudaGetDevice(&dev); cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev); }
  cfg.attrs = at; cfg.numAttrs = major >= 9 ? 1 : 0;
  if (!V7) cudaLaunchKernelEx(&cfg, fin, q);
}

// ---------------------------------------------------------------- v6: deep per-CTA TMA ring (stream over 16 KB stages)
// A stage is 8 consecutive 2 KB units (one 16-row tile, 8 K chunks); CTA c streams stages [ST c / G, ST (c + 1) / G)
// through an NS-deep ring filled by a producer warp, so most of a CTA's bytes are in flight at once. Consumer warp w
// takes unit w of every stage; at a tile's end the 8 partials meet in smem, and a tile shared by CTAs is summed in
// ws with fp32 atomics, the last arriver writing the output.
#ifndef NSG
#define NSG 12
#endif
__device__ __forceinline__ void mbarrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(b) : "memory"); }
template <int NT, bool HL>
__global__ void __launch_bounds__(288, 1) gemv(Q q) {
  constexpr int SB = (HL ? 2 : 1) * 16384, NS = HL ? NSG / 2 : NSG, CH = NT == 1 ? 4 : (NT == 2 ? 2 : 1);
  extern __shared__ __align__(128) uint8_t smem[];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const uint32_t base = (uint32_t)__cvta_generic_to_shared(smem);
  const uint32_t full = base + NS * SB, empty = full + NS * 8;
  float* red = (float*)(smem + NS * SB + NS * 16);
  if (threadIdx.x < NS) { mbinit(full + threadIdx.x * 8, TMA ? 1 : 32); mbinit(empty + threadIdx.x * 8, 8); }
#if TMA
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
  __syncthreads();
  const int SPT = q.KC / 8;  // stages per tile
  const int ST = q.MT * SPT, G = gridDim.x, c = blockIdx.x;
  const int s0 = (int)((long long)ST * c / G), s1 = (int)((long long)ST * (c + 1) / G);
  if (wid == 8) {  // producer
#if !TMA
    for (int i = s0; i < s1; i++) {
      const int k = i - s0, slot = k % NS;
      if (k >= NS) mbwait(empty + slot * 8, ((k / NS) - 1) & 1);
      for (int o = lane * 16; o < 16384; o += 512) {
        cp16(base + slot * SB + o, q.A + (size_t)i * 16384 + o);
        if (HL) cp16(base + slot * SB + 16384 + o, q.A2 + (size_t)i * 16384 + o);
      }
      asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(full + slot * 8) : "memory");
    }
    asm volatile("cp.async.wait_all;\n" ::: "memory");
#endif
    if (lane == 0) {
      for (int i = s0; i < s1 && TMA; i++) {
        const int k = i - s0, slot = k % NS;
        if (k >= NS) mbwait(empty + slot * 8, ((k / NS) - 1) & 1);
#if TMA
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(full + slot * 8), "n"(SB) : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(base + slot * SB), "l"(q.A + (size_t)i * 16384), "n"(16384), "r"(full + slot * 8) : "memory");
        if (HL) asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(base + slot * SB + 16384), "l"(q.A2 + (size_t)i * 16384), "n"(16384), "r"(full + slot * 8) : "memory");
#endif
      }
    }
    return;
  }
  const int tb = 0;
  float acc[NT][CH][4];
#pragma unroll
  for (int n = 0; n < NT; n++)
#pragma unroll
    for (int h = 0; h < CH; h++) acc[n][h][0] = acc[n][h][1] = acc[n][h][2] = acc[n][h][3] = 0.f;
  uint4 b0[NT][2];
  auto ldb = [&](int i, uint4 (&b)[NT][2]) {
    const int ch = (i % SPT) * 8 + wid;
#pragma unroll
    for (int n = 0; n < NT; n++) {
      const int tok = min(tb * 32 + n * 8 + g, q.M - 1);
      const uint4* xr = (const uint4*)(q.x + (size_t)tok * q.K) + t * 2 + ch * 8;
      b[n][0] = __ldg(xr); b[n][1] = __ldg(xr + 1);
    }
  };
  if (s0 < s1) ldb(s0, b0);
  for (int i = s0; i < s1; i++) {
    const int k = i - s0, slot = k % NS;
    uint4 b1[NT][2];
    if (i + 1 < s1) ldb(i + 1, b1);
    mbwait(full + slot * 8, (k / NS) & 1);
    const uint32_t sa = base + slot * SB + wid * 2048 + lane * 16;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const uint4 a = lds(sa + j * 512);
      uint4 h; if (HL) h = lds(sa + 16384 + j * 512);
#pragma unroll
      for (int n = 0; n < NT; n++) {
        const uint32_t* bb = (const uint32_t*)&b0[n][j >> 1];
        mma(acc[n][j % CH], a, bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);
        if (HL) mma(acc[n][(j + 1) % CH], h, bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);
      }
    }
    __syncwarp();
    if (lane == 0) mbarrive(empty + slot * 8);
#pragma unroll
    for (int n = 0; n < NT; n++) { b0[n][0] = b1[n][0]; b0[n][1] = b1[n][1]; }
    const int tile = i / SPT;
    if (i + 1 == s1 || (i + 1) % SPT == 0) {  // tile ends here (for this CTA): reduce the 8 warps' partials
#pragma unroll
      for (int n = 0; n < NT; n++) {
        float f[4];
#pragma unroll
        for (int i2 = 0; i2 < 4; i2++) { f[i2] = 0.f;
#pragma unroll
          for (int h = 0; h < CH; h++) f[i2] += acc[n][h][i2]; }
        ((float4*)red)[(n * 8 + wid) * 32 + lane] = make_float4(f[0], f[1], f[2], f[3]);
#pragma unroll
        for (int h = 0; h < CH; h++) acc[n][h][0] = acc[n][h][1] = acc[n][h][2] = acc[n][h][3] = 0.f;
      }
      asm volatile("bar.sync 1, 256;\n" ::: "memory");
      const int t0s = tile * SPT, t1s = t0s + SPT;
      const bool own = t0s >= s0 && t1s <= s1;
      const int row0 = tile * 16 + g;
      for (int n = wid; n < NT; n += 8) {
        float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int w2 = 0; w2 < 8; w2++) { const float4 v = ((float4*)red)[(n * 8 + w2) * 32 + lane]; sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w; }
        const int tk = n * 8 + 2 * t;
        const float fv[4] = {sum.x, sum.y, sum.z, sum.w};
#pragma unroll
        for (int e = 0; e < 4; e++) {
          const int tok = tk + (e & 1), row = row0 + (e >> 1) * 8;
          if (tok >= q.M || row >= q.N) continue;
          if (own) put(q, tok, row, fv[e]);
          else atomicAdd(q.ws + (size_t)tok * q.N + row, fv[e]);
        }
      }
      if (!own) {
        __threadfence();
        asm volatile("bar.sync 1, 256;\n" ::: "memory");
        __shared__ int lastf;
        if (threadIdx.x == 0) {
          auto cta = [&](int st2) { return (int)(((long long)(st2 + 1) * G - 1) / ST); };  // CTA streaming stage st2
          const int nc = cta(t1s - 1) - cta(t0s) + 1;
          lastf = atomicAdd(q.cnt + tile, 1) == nc - 1;
        }
        asm volatile("bar.sync 1, 256;\n" ::: "memory");
        if (lastf) {
          __threadfence();
          for (int e = threadIdx.x; e < 16 * q.M; e += 256) {
            const int tok = e >> 4, row = tile * 16 + (e & 15);
            if (row < q.N) { float* wp = q.ws + (size_t)tok * q.N + row; put(q, tok, row, __ldcg(wp)); *wp = 0.f; }
          }
          if (threadIdx.x == 0) q.cnt[tile] = 0;
        }
      }
      asm volatile("bar.sync 1, 256;\n" ::: "memory");
    }
  }
}
template <int NT, bool HL> static void go2(const Q& q, cudaStream_t st) {
  constexpr int NS = HL ? NSG / 2 : NSG;
  const size_t sm = (size_t)NS * (HL ? 32768 : 16384) + NS * 16 + 8 * 32 * 16 * NT;
  static bool once = false;
  if (!once) { cudaFuncSetAttribute(gemv<NT, HL>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm); once = true; }
  gemv<NT, HL><<<g_grid, 288, sm, st>>>(q);
}
void run2(torch::Tensor A, torch::Tensor A2, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t N) {
  if (!g_grid) { int dev; cudaGetDevice(&dev); cudaDeviceGetAttribute(&g_grid, cudaDevAttrMultiProcessorCount, dev); }
  Q q;
  q.A = (const uint8_t*)A.data_ptr(); q.A2 = A2.numel() ? (const uint8_t*)A2.data_ptr() : nullptr;
  q.x = (const __nv_bfloat16*)x.data_ptr(); q.out = out.data_ptr(); q.ws = ws.data_ptr<float>(); q.cnt = cnt.data_ptr<int>();
  q.M = x.size(0); q.K = x.size(1); q.N = (int)N; q.MT = (int)((N + 15) / 16); q.KC = q.K / 64; q.TB = 1;
  q.f32 = out.scalar_type() == torch::kFloat32;
  auto st = at::cuda::getCurrentCUDAStream();
  const int nt = (q.M + 7) / 8;
  if (q.A2) { if (nt <= 1) go2<1, true>(q, st); else if (nt <= 2) go2<2, true>(q, st); else go2<4, true>(q, st); }
  else { if (nt <= 1) go2<1, false>(q, st); else if (nt <= 2) go2<2, false>(q, st); else go2<4, false>(q, st); }
}
// ---------------------------------------------------------------- router (N 256, K 8192, m <= 32)
// CTA (ks, mt) owns row tile mt and 16 consecutive 64-wide K chunks, one per warp. Every warp loads its hi + lo A
// fragments and its x chunk straight into registers in one round trip (64 KB per CTA in flight); the 16 warp partials
// meet in smem and the last of the KS CTAs on a tile sums their partials (no atomics on the output, no second kernel).
#define RPART (1 << 20)
#define RCNT 2048
__device__ __forceinline__ uint4 ldnc(const void* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
template <int NT>
__global__ void __launch_bounds__(512, 1) rtr(Q q) {
  __shared__ float red[16 * 512];
  __shared__ int last;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, t = lane & 3;
  const int ks = blockIdx.x, mt = blockIdx.y, KS = gridDim.x, c = ks * 16 + w;
  const uint8_t* ah = q.A + ((size_t)mt * q.KC + c) * 2048 + lane * 16;
  const uint8_t* al = q.A2 + ((size_t)mt * q.KC + c) * 2048 + lane * 16;
  uint4 a[4], h[4], b[NT][2];
#pragma unroll
  for (int j = 0; j < 4; j++) { a[j] = ldnc(ah + j * 512); h[j] = ldnc(al + j * 512); }
#pragma unroll
  for (int n = 0; n < NT; n++) {
    const uint4* xr = (const uint4*)(q.x + (size_t)min(n * 8 + g, q.M - 1) * q.K + c * 64) + t * 2;
    b[n][0] = __ldg(xr); b[n][1] = __ldg(xr + 1);
  }
  float acc[NT][4];
#pragma unroll
  for (int n = 0; n < NT; n++) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
#pragma unroll
  for (int j = 0; j < 4; j++)
#pragma unroll
    for (int n = 0; n < NT; n++) {
      const uint32_t* bb = (const uint32_t*)&b[n][j >> 1];
      mma(acc[n], a[j], bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);
      mma(acc[n], h[j], bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);
    }
#pragma unroll
  for (int n = 0; n < NT; n++) {
    float* r = red + w * 512 + g * 32 + n * 8 + 2 * t;
    *(float2*)r = make_float2(acc[n][0], acc[n][1]);
    *(float2*)(r + 8 * 32) = make_float2(acc[n][2], acc[n][3]);
  }
  __syncthreads();
  const int row = tid >> 5, tok = tid & 31;  // this thread's output (row of the tile, token)
  float s = 0.f;
  if (tok < NT * 8) {
#pragma unroll
    for (int ww = 0; ww < 16; ww++) s += red[ww * 512 + tid];
  }
  float* P = q.ws + RPART + (size_t)mt * KS * 512;
  P[ks * 512 + tid] = s;
  __threadfence();
  __syncthreads();
  if (tid == 0) last = atomicAdd(q.cnt + RCNT + mt, 1) == KS - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  float tot = 0.f;
  for (int k2 = 0; k2 < KS; k2++) tot += __ldcg(P + k2 * 512 + tid);
  if (tok < q.M) ((float*)q.out)[(size_t)tok * q.N + mt * 16 + row] = tot;
  if (tid == 0) q.cnt[RCNT + mt] = 0;
}
void run3(torch::Tensor A, torch::Tensor A2, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t N) {
  Q q;
  q.A = (const uint8_t*)A.data_ptr(); q.A2 = (const uint8_t*)A2.data_ptr();
  q.x = (const __nv_bfloat16*)x.data_ptr(); q.out = out.data_ptr(); q.ws = ws.data_ptr<float>(); q.cnt = cnt.data_ptr<int>();
  q.M = x.size(0); q.K = x.size(1); q.N = (int)N; q.MT = (int)(N / 16); q.KC = q.K / 64; q.TB = 1; q.f32 = 1;
  const dim3 grid(q.KC / 16, q.MT);
  auto st = at::cuda::getCurrentCUDAStream();
  const int nt = (q.M + 7) / 8;
  if (nt <= 1) rtr<1><<<grid, 512, 0, st>>>(q); else if (nt <= 2) rtr<2><<<grid, 512, 0, st>>>(q); else rtr<4><<<grid, 512, 0, st>>>(q);
}
"""

    # ---------------------------------------------------------------- v13/v14: register-direct swap-AB GEMV on 12-bit-packed BF16
    # A unit is one 16-row tile x one 64-wide K chunk in mma.sync A-fragment order, per lane 3 x 16 B (c12) or 4 x 16 B (bf16):
    # c12 keeps each value's low byte (exponent LSB + mantissa) and a nibble (sign + 3-bit offset of the upper 7 exponent bits
    # from a per-matrix window); out-of-window values are fixed from a per-warp escape list (exact). FMT 2/3: the router's
    # BF16 hi + lo pair (bf16 / c12), fp32 out. Warp (wr, wk) of CTA (bx, by) streams tiles (bx * WR + wr) * RT + j, chunks
    # [(by * WK + wk) * KU, + KU) straight into registers; x comes from registers (XS 0) or a CTA-shared smem slab (XS 1);
    # partials meet in smem, then across the S CTAs of a cluster through DSMEM (sm_90) or a last-arriver workspace (older GPUs).
    GVCU = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

struct G { const uint4* W; const uint4* W2; const __nv_bfloat16* x; void* out; const uint32_t* esc; float4* ws; int* cnt; int m, N, K, KC; uint32_t T0, T1, U0, U1; };

__device__ __forceinline__ uint4 gnc(const void* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
template <int XP> __device__ __forceinline__ uint4 gxp(const void* p);  // pw16 x load policy (defined below)
__device__ __forceinline__ uint4 gx(const void* p) {
  uint4 v; asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
template <int XP> __device__ __forceinline__ uint4 gxp(const void* p) {
  uint4 v;
  if constexpr (XP == 1) asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  else if constexpr (XP == 2) asm volatile("ld.global.nc.L1::evict_last.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  else v = gx(p);
  return v;
}
__device__ __forceinline__ uint4 lds4(uint32_t a) {
  uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v;
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) { uint32_t d; asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }
__device__ __forceinline__ void hmma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// V & 1 == 0: nibble v = offset of value v (byte-permute table T0:T1 = base .. base + 7), signs at bits 3, 11, 19, 27 (values 0-3)
// and 7, 15, 23, 31 (values 4-7): 11 ops / 8 values
__device__ __forceinline__ void dec0(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {
  const uint32_t cm = C & 0x77777777u;
  const uint32_t hl = prmt(T0, T1, cm) | ((C << 4) & 0x80808080u), hh = prmt(T0, T1, cm >> 16) | (C & 0x80808080u);
  a[0] = prmt(L0, hl, 0x5140); a[1] = prmt(L0, hl, 0x7362); a[2] = prmt(L1, hh, 0x5140); a[3] = prmt(L1, hh, 0x7362);
}
// V & 1 == 1: C byte i: bits 0-2 offset of value i, bit 7 its sign, bits 4-6 offset of value 4 + i, bit 3 the sign of value 4 + (i + 3) % 4,
// so (C & 0x87878787) + base and the same on C rotated by 4 are the high bytes of values 0-3 and 4-7 (window base <= 120)
__device__ __forceinline__ void dec1(uint32_t L0, uint32_t L1, uint32_t C, uint32_t Tb, uint32_t, uint32_t* a) {
  const uint32_t hl = (C & 0x87878787u) + Tb, hh = (__funnelshift_r(C, C, 4) & 0x87878787u) + Tb;
  a[0] = prmt(L0, hl, 0x5140); a[1] = prmt(L0, hl, 0x7362); a[2] = prmt(L1, hh, 0x5140); a[3] = prmt(L1, hh, 0x7362);
}
__device__ __forceinline__ void sto(__nv_bfloat16* d, float v) { *d = __float2bfloat16(v); }
__device__ __forceinline__ void sto(float* d, float v) { *d = v; }
template <typename O>
__device__ __forceinline__ void emit(const G& p, int idx, int row0, float4 v, int MT) {
  const int l = idx & 31, r = idx >> 5, n = r % MT, row = row0 + (r / MT) * 16 + (l >> 2), tok = n * 8 + 2 * (l & 3);
  O* o = (O*)p.out + (size_t)tok * p.N + row;
  sto(o, v.x); sto(o + 8, v.z);
  if (tok + 1 < p.m) { sto(o + p.N, v.y); sto(o + p.N + 8, v.w); }
}
// pw27: split-K partial added into an output the caller zeroed (a memset node in the same graph); fp32 (router) or bf16
__device__ __forceinline__ void reda(float* d, float v) { atomicAdd(d, v); }
__device__ __forceinline__ void reda2(__nv_bfloat16* d, float a, float b) {  // native packed bf16x2 reduction (no return value)
  const __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
  asm volatile("red.global.add.noftz.bf16x2 [%0], %1;\n" :: "l"(d), "r"(*(const uint32_t*)&h) : "memory");
}
template <typename O>
__device__ __forceinline__ void redemit(const G& p, int idx, int row0, float4 v, int MT) {
  const int l = idx & 31, r = idx >> 5, n = r % MT, row = row0 + (r / MT) * 16 + (l >> 2), tok = n * 8 + 2 * (l & 3);
  O* o = (O*)p.out + (size_t)tok * p.N + row;
  if constexpr (std::is_same<O, float>::value) {
    reda(o, v.x); reda(o + 8, v.z);
    if (tok + 1 < p.m) { reda(o + p.N, v.y); reda(o + p.N + 8, v.w); }
  } else {  // bf16: rows (g, g+1) of lanes l, l ^ 4 (same tokens, same liveness) as one native packed bf16x2 red.add
    const unsigned am = __activemask();
    const float px = __shfl_xor_sync(am, v.x, 4), py = __shfl_xor_sync(am, v.y, 4), pz = __shfl_xor_sync(am, v.z, 4), pw = __shfl_xor_sync(am, v.w, 4);
    if (((l >> 2) & 1) == 0) {  // even row g: the pair starts at an even column (4-B aligned: row0 % 16 == 0, N even)
      reda2(o, v.x, px); reda2(o + 8, v.z, pz);
      if (tok + 1 < p.m) { reda2(o + p.N, v.y, py); reda2(o + p.N + 8, v.w, pw); }
    }
  }
}
__device__ __forceinline__ void add4(float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }
__device__ __forceinline__ bool live(int idx, int MT, int m) { return ((idx >> 5) % MT) * 8 + 2 * (idx & 3) < m; }

// ---- pw18: codec E v3, routine copied from proj_rsi/tools/codec_e.py CODEC_E_CUH ----
// ---- codec E v3 device decode (proj_rsi/tools/CODEC_E_LAYOUT.md v3) ------------------------------------------------
// K = 1, 2 or 4 independent substreams per lane; substream s owns steps j in [s*4/K, (s+1)*4/K) of every unit.
// Per lane and substream: lo, hi = 64-bit bit buffer (next code at bit 0 of lo); V = valid bits mod 128 (bits 8+ carry
// garbage from V -= len; only (V & 0x60) and (V & 31) are read); pb = bytes taken from the substream's lane stream;
// iss = words issued into the ring; nw = prefetched refill word.
// Code length: the first 3 bits of every code determine its length, so len = byte (lo & 7) of the 8-byte table (t0, t1):
// LOP3 + PRMT. The bit-position chain (LOP3 -> PRMT -> SHF) has no memory access. The value (both exponents) comes from
// lut[lo & 0xff] (LDS, off the chain); A = lop3(e, 0x807f807f, plane bytes, 0xb8) (bits 0-6 and 15 from the plane).
// Stream: per segment, K substream regions of P * 32 words; word q of lane l of substream s at soff + (s * P + q) * 32 + l
// (4 B pieces: coalesced 128 B per warp, padding to a word). Ring per warp and substream: ES = ED / K word slots x 32
// lanes ([s][slot][lane], 128 B per slot: one bank per lane, conflict-free); word q in slot q % ES.
// Per unit: ce_issue for each substream (NI = 4 / K predicated 4 B cp.async), one commit_group, wait_group(2) (the
// prologue issues 3 rounds); then ce_unit. A substream's first refill word of the unit is read after the wait, the
// others are prefetched one step ahead (same landed set, no issue in between).
struct CeS { uint32_t lo, hi, V, pb, iss, nw; };
__device__ __forceinline__ uint32_t ce_lds(uint32_t a) { uint32_t v; asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a)); return v; }
__device__ __forceinline__ uint32_t ce_prmt(uint32_t a, uint32_t b, uint32_t s) { uint32_t d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }
__device__ __forceinline__ uint32_t ce_sel(uint32_t e, uint32_t d) {  // (e & ~0x807f807f) | (d & 0x807f807f): immLut b ? c : a = 0xb8
  uint32_t r; asm("lop3.b32 %0, %1, %2, %3, 0xb8;" : "=r"(r) : "r"(e), "r"(0x807f807fu), "r"(d)); return r;
}
// smem address of the word at substream byte pb: rl = ring_s + lane * 4, ring_s aligned to ES * 128
template <int ES>
__device__ __forceinline__ uint32_t ce_addr(uint32_t rl, uint32_t pb) { return rl | ((pb << 5) & ((ES - 1) << 7)); }
// NI predicated 4 B cp.async; gsrc = stream + (soff[gw] + s * P * 32 + lane) * 4 (word 0 of this lane and substream)
template <int ES, int NI>
__device__ __forceinline__ void ce_issue(CeS& b, uint32_t P, const uint8_t* gsrc, uint32_t rl) {
#pragma unroll
  for (int k = 0; k < NI; k++) {
    const bool go = b.iss < P && b.iss < (b.pb >> 2) + ES;
    asm volatile("{ .reg .pred p; setp.ne.b32 p, %2, 0;\n @p cp.async.ca.shared.global [%0], [%1], 4; }\n"
                 :: "r"(rl | ((b.iss % ES) << 7)), "l"(gsrc + (size_t)b.iss * 128), "r"((uint32_t)go));
    b.iss += go;
  }
}
__device__ __forceinline__ void ce_refill(CeS& b, uint32_t w) {
  const bool need = (b.V & 0x60u) == 0u;
  w = need ? w : 0u;
  const uint32_t at = b.V & 31u;
  b.lo |= w << at; b.hi |= __funnelshift_l(w, 0u, at);
  b.V += need ? 32u : 0u; b.pb += need ? 4u : 0u;
}
// one unit: the K substreams' chains interleaved (independent), 16 A registers in r6 order A[4 j + i];
// rl = ring of substream 0 + lane * 4 (substream s at + s * ES * 128); LS = LUT index shift (2 single, 7 lane-replicated)
template <int ES, int LS, int K>
__device__ __forceinline__ void ce_unit(CeS* b, uint32_t lut, uint32_t rl, uint32_t t0, uint32_t t1, uint4 q0, uint4 q1, uint32_t* A) {
  constexpr int NS = 4 / K;
  const uint32_t L[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
  for (int t = 0; t < NS; t++) {
#pragma unroll
    for (int s = 0; s < K; s++) {
      const uint32_t rs = rl + s * ES * 128;
      ce_refill(b[s], t == 0 ? ce_lds(ce_addr<ES>(rs, b[s].pb)) : b[s].nw);
      if (t + 1 < NS) b[s].nw = ce_lds(ce_addr<ES>(rs, b[s].pb));
    }
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
      for (int s = 0; s < K; s++) {
        const int j = s * NS + t;
        const uint32_t len = ce_prmt(t0, t1, b[s].lo & 7u);
        const uint32_t e = ce_lds(lut + ((b[s].lo & 0xffu) << LS));
        b[s].lo = __funnelshift_r(b[s].lo, b[s].hi, len); b[s].hi = __funnelshift_r(b[s].hi, 0u, len); b[s].V -= len;
        A[4 * j + i] = ce_sel(e, ce_prmt(L[2 * j + (i >> 1)], 0u, (i & 1) ? 0x3322u : 0x1100u));
      }
  }
}
// escapes (r6 walk, unchanged except the XOR field): A[r] ^= (nx & 0xff) << (7 + 16 * ((nx >> 10) & 1))
// pw20: one step t (4 A registers) of ce_unit<ES, LS, 1>, identical operations in identical order (refill, prefetch for
// step t + 1, 4 pairs), so a caller can decode and consume a unit step by step with 4 A registers live instead of 16
template <int ES, int LS>
__device__ __forceinline__ void ce_step1(CeS& b, uint32_t lut, uint32_t rl, uint32_t t0, uint32_t t1, uint32_t Lw0, uint32_t Lw1, int t, uint32_t* A) {
  ce_refill(b, t == 0 ? ce_lds(ce_addr<ES>(rl, b.pb)) : b.nw);
  if (t < 3) b.nw = ce_lds(ce_addr<ES>(rl, b.pb));
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const uint32_t len = ce_prmt(t0, t1, b.lo & 7u);
    const uint32_t e = ce_lds(lut + ((b.lo & 0xffu) << LS));
    b.lo = __funnelshift_r(b.lo, b.hi, len); b.hi = __funnelshift_r(b.hi, 0u, len); b.V -= len;
    A[i] = ce_sel(e, ce_prmt((i >> 1) ? Lw1 : Lw0, 0u, (i & 1) ? 0x3322u : 0x1100u));
  }
}

// pw12: per-warp private bulk-copy ring for W (V bit 3). V bits 4-5: slots NS = 2 << code; bits 6-7: chunks per slot
// CU = 1 << code; bit 8: refill without fence.proxy.async (the slot's values are consumed by the decode, in program
// order, before the __syncwarp and the refill; CUTLASS-style release). Lane 0 of each warp fills its own slots with cp.async.bulk (one copy per row tile and matrix, CU
// contiguous 64-k chunks) on a per-slot mbarrier; the warp waits on it, reads its fragments with ld.shared.v4 at the
// offsets r6's ld.global.v4 used, and lane 0 refills the slot after the warp has decoded its last chunk.
template <int FMT, int MT, int WR, int WK, int KU, int S, int RT, int XS, int V>
struct GL {
  static constexpr int NL = FMT == 4 ? 2 : ((FMT & 1) ? 3 : 4), NA = (FMT == 2 || FMT == 3) ? 2 : 1, NW = WR * WK, NI = WR * RT * MT * 32;
  // pw18 codec E v3 (FMT 4): V bits 22-23 = log2 substreams CK (1/2/4); bit 19 = ring 32 words per lane (else 16), split
  // into ES = CED / CK slots per substream; bit 20 = LUT replicated per lane (32 KB, conflict-free); smem = per-warp ring
  // (2048-aligned) + LUT, after the slab / reduction buffers
  static constexpr int CK = 1 << ((V >> 22) & 3), CED = ((V >> 19) & 1) ? 32 : 16, CES = CED / CK;
  static constexpr int CLUT = FMT == 4 ? (((V >> 20) & 1) ? 32768 : 1024) : 0;
  static constexpr int CAL = CES * 128 > 2048 ? CES * 128 : 2048;  // ring alignment (ring_s | slot offset)
  static constexpr int CRING = FMT == 4 ? NW * 32 * CED * 4 + CAL : 0;  // + CAL: alignment slack
  static constexpr bool BK = (V & 8) != 0;
  static constexpr int CU = BK ? 1 << ((V >> 6) & 3) : 1, G = KU / CU, NS0 = BK ? 2 << ((V >> 4) & 3) : 1;
  static constexpr int NS = NS0 < G ? NS0 : G, SB = RT * NA * CU * NL * 512;
  static constexpr int SLAB = XS == 1 ? WK * KU * MT * 8 * 128 : 0, REDB = (WK > 1 || S > 1) ? (WK + 1) * NI * 16 : 0;
  static constexpr int BAR = BK ? ((NW * NS * 8 + 127) / 128) * 128 : 0, RING = BK ? NW * NS * SB : 0;
  static constexpr int SMEM = SLAB + REDB + BAR + RING + CLUT + CRING;
};
__device__ __forceinline__ void gxwait(uint32_t b, uint32_t ph) {
  asm volatile("{\n.reg .pred p;\nGXW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra GXW;\n}\n" :: "r"(b), "r"(ph) : "memory");
}

template <int FMT, int MT, int ST, int WR, int WK, int KU, int S, int RT, int XS, int V>
__global__ void __launch_bounds__(WR * WK * 32, ((V >> 12) & 7) ? ((V >> 12) & 7) : (XS == 2 && WR * WK * 32 < 512 ? 512 / (WR * WK * 32) : 1)) gv(G p) {
  // pw14 probes only (never registered as paths): V bits 9-11 = ladder level LAD (1: W loads only, 2: + x loads,
  // 3: + decode, 4: + mma; no reduction / store); bits 12-14 = __launch_bounds__ minBlocks override (register cap).
  constexpr int LAD = (V >> 9) & 7;
  constexpr bool LX = LAD != 1, LDEC = LAD == 0 || LAD >= 3, LMMA = LAD == 0 || LAD == 4;
  uint32_t sk = 0;
  constexpr bool C12 = FMT & 1, HL = FMT == 2 || FMT == 3, CE = FMT == 4;
  constexpr int NL = CE ? 2 : (C12 ? 3 : 4), NA = HL ? 2 : 1, NW = WR * WK, NT = NW * 32, NI = WR * RT * MT * 32, CH = WK * KU, TK = MT * 8;
  constexpr int NC = (MT * RT * NA >= 2 || (CE && ((V >> 24) & 1))) ? 1 : 2, XB = XS ? 1 : ST;  // pw20 lite: one accumulator set (register budget)  // XS 2: x straight from global at use (L1), no ring
  using O = typename std::conditional<HL, float, __nv_bfloat16>::type;
  extern __shared__ __align__(16) uint8_t smem[];
  float4* red = (float4*)(smem + (XS == 1 ? CH * TK * 128 : 0));
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, wr = w % WR, wk = w / WR, g = lane >> 2, t = lane & 3;
  constexpr bool XZ = ((V >> 29) & 1) != 0;  // P1 (V bit 29): grid.z plane 0 = zeroer CTAs (one per tile), the K splits follow
  const int bz = blockIdx.x, bx = blockIdx.y, by = (int)blockIdx.z - (XZ ? 1 : 0), GX = gridDim.y;
  if constexpr (XZ) {
    if (by < 0) {  // dispatched first (lowest z): zero this tile of the output, publish, exit; no W, no x
      O* o = (O*)p.out + (size_t)bz * TK * p.N + bx * WR * RT * 16;
      const int R = WR * RT * 16, mm = min(TK, p.m - bz * TK);
      for (int i = threadIdx.x; i < mm * R; i += WR * WK * 32) sto(o + (size_t)(i / R) * p.N + i % R, 0.f);
      __threadfence(); __syncthreads();
      if (threadIdx.x == 0) atomicAdd(p.cnt + 2 * (bz * GX + bx) + 1, 1);
      return;
    }
  }
  p.x += (size_t)bz * TK * p.K; p.out = (uint8_t*)p.out + (size_t)bz * TK * p.N * sizeof(O); p.m = min(TK, p.m - bz * TK);
  const int cb = by * CH, c0 = cb + wk * KU, row0 = bx * WR * RT * 16;
  const size_t wo = ((size_t)(bx * WR + wr) * RT * p.KC + c0) * (NL * 32) + lane, js = (size_t)p.KC * NL * 32;
  const __nv_bfloat16* xp[MT];
#pragma unroll
  for (int n = 0; n < MT; n++) xp[n] = p.x + (size_t)min(n * 8 + g, p.m - 1) * p.K + c0 * 64 + t * 16;
  uint4 wb[ST][RT][NA][NL], xb[XB][MT][2];
  using L = GL<FMT, MT, WR, WK, KU, S, RT, XS, V>;
  static_assert(KU % L::CU == 0, "chunks per slot must divide KU");
  auto ld = [&](int s, int u) {
    if constexpr (!L::BK) {
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int h = 0; h < NA; h++)
#pragma unroll
        for (int i = 0; i < NL; i++) wb[s][j][h][i] = gnc((h ? p.W2 : p.W) + wo + j * js + (u * NL + i) * 32);
    }
    if (!XS && LX) {
#pragma unroll
      for (int n = 0; n < MT; n++) { xb[s % XB][n][0] = gxp<(V >> 15) & 3>(xp[n] + u * 64); xb[s % XB][n][1] = gxp<(V >> 15) & 3>(xp[n] + u * 64 + 8); }
    }
  };
  uint32_t bbar = 0, bring = 0;
  auto bissue = [&](int slot, int grp) {  // lane 0 only: fill one slot with slot-group grp (CU chunks per row tile / matrix)
    const uint32_t bar = bbar + slot * 8;
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "n"(L::SB) : "memory");
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int h = 0; h < NA; h++)
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                     :: "r"(bring + slot * L::SB + (j * NA + h) * L::CU * NL * 512),
                        "l"((h ? p.W2 : p.W) + (wo - lane) + j * js + (size_t)grp * L::CU * NL * 32), "n"(L::CU * NL * 512), "r"(bar) : "memory");
  };
  if constexpr (L::BK) {
    const uint32_t s0 = (uint32_t)__cvta_generic_to_shared(smem) + L::SLAB + L::REDB;
    bbar = s0 + w * L::NS * 8;
    bring = s0 + L::BAR + w * L::NS * L::SB;
    if (lane == 0) {
#pragma unroll
      for (int k = 0; k < L::NS; k++) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(bbar + k * 8));
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#pragma unroll
      for (int k = 0; k < L::NS; k++) bissue(k, k);
    }
    __syncwarp();
  }
  int tk = 0;
  if constexpr (((V >> 27) & 1) != 0 || XZ) {  // P1 (V bit 27): no memset node; the split-0 CTA of the tile zeroes it, then raises a flag
    if (w == 0) {
      const int tile = bz * GX + bx, R = WR * RT * 16;
      if (lane == 0) tk = atomicAdd(p.cnt + 2 * tile, 1);
      if (by == 0 && !XZ) {
        O* o = (O*)p.out + row0;
        for (int i = lane; i < p.m * R; i += 32) sto(o + (size_t)(i / R) * p.N + i % R, 0.f);
        __threadfence(); __syncwarp();
        if (lane == 0) atomicAdd(p.cnt + 2 * tile + 1, 1);
      }
    }
  }
#pragma unroll
  for (int s = 0; s < ST; s++) if (s < KU) ld(s, s);
  CeS ceb[CE ? L::CK : 1];
  uint32_t cP = 0u, crl = 0u, clut = 0u, ct0 = 0u, ct1 = 0u;
  const uint8_t* cgs = nullptr;
  if constexpr (CE) {  // pw18: segment start, ring prologue (3 issue rounds, one commit group each), LUT + length table
#pragma unroll
    for (int s = 0; s < L::CK; s++) ceb[s] = CeS{0u, 0u, 0u, 0u, 0u, 0u};
    const uint32_t gw = (uint32_t)((by * GX + bx) * NW + w);
    const uint32_t* soff = (const uint32_t*)(((uint64_t)p.U1 << 32) | p.U0);
    const uint32_t s0 = __ldg(soff + gw);
    cP = (__ldg(soff + gw + 1) - s0) / (32u * L::CK);
    cgs = (const uint8_t*)p.W2 + ((size_t)s0 + lane) * 4;  // word 0 of this lane, substream 0 (substream s: + s * P * 128 B)
    const uint32_t cs00 = (uint32_t)__cvta_generic_to_shared(smem) + L::SLAB + L::REDB;
    const uint32_t cs0 = (cs00 + L::CAL - 1u) & ~(uint32_t)(L::CAL - 1);  // ring_s | slot offset needs ES * 128 alignment
    crl = cs0 + (w * 32 * L::CED + lane) * 4;
    clut = cs0 + NW * 32 * L::CED * 4 + (((V >> 20) & 1) ? lane * 4 : 0);
    const uint32_t* gl = (const uint32_t*)(((uint64_t)p.T1 << 32) | p.T0);
    ct0 = __ldg(gl + 256); ct1 = __ldg(gl + 257);
#pragma unroll
    for (int k = 0; k < 3; k++) {
#pragma unroll
      for (int s = 0; s < L::CK; s++) ce_issue<L::CES, 4 / L::CK>(ceb[s], cP, cgs + (size_t)s * cP * 128, crl + s * L::CES * 128);
      asm volatile("cp.async.commit_group;\n" ::: "memory");
    }
    uint32_t* sl = (uint32_t*)(smem + L::SLAB + L::REDB + (cs0 - cs00) + NW * 32 * L::CED * 4);
    constexpr int NE = (256 + NT - 1) / NT;  // LUT entries per thread: all loads in flight at once, then the stores
    uint32_t lv[NE];
#pragma unroll
    for (int k = 0; k < NE; k++) lv[k] = threadIdx.x + k * NT < 256 ? __ldg(gl + threadIdx.x + k * NT) : 0u;
#pragma unroll
    for (int k = 0; k < NE; k++) {
      const int x = threadIdx.x + k * NT;
      if (x < 256) {
        if constexpr (((V >> 20) & 1) != 0) {  // entry x for lane l at word 32 x + l: 8 x 16 B per entry
#pragma unroll
          for (int q = 0; q < 8; q++) ((uint4*)(sl + x * 32))[q] = make_uint4(lv[k], lv[k], lv[k], lv[k]);
        } else sl[x] = lv[k];
      }
    }
    __syncthreads();
  }
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem);
  if (XS == 1) {
    for (int e = threadIdx.x; e < CH * TK * 8; e += NT) {
      const int s8 = e & 7, tok = (e >> 3) % TK, ch = e / (TK * 8);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sb + ((ch * TK + tok) * 8 + (s8 ^ (tok & 7))) * 16),
                   "l"(p.x + (size_t)min(tok, p.m - 1) * p.K + (cb + ch) * 64 + s8 * 8));
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_all;\n" ::: "memory");
  }
  uint32_t e = ~0u, nx = ~0u;
  int ep = 0;
  if ((C12 || CE) && LDEC) { e = p.esc[((size_t)(by * GX + bx) * NW + w) * 32 + lane]; nx = __shfl_sync(~0u, e, 0); }
  if (XS == 1) __syncthreads();
  float acc[RT][MT][NC][4];
#pragma unroll
  for (int j = 0; j < RT; j++)
#pragma unroll
    for (int n = 0; n < MT; n++)
#pragma unroll
      for (int c = 0; c < NC; c++) acc[j][n][c][0] = acc[j][n][c][1] = acc[j][n][c][2] = acc[j][n][c][3] = 0.f;
  auto step = [&](const int u, const int s) {
    if constexpr (CE && ((V >> 24) & 1)) {  // pw20 lite (V bit 24): K = 1, decode and consume one k16 step at a time
      static_assert(RT == 1 && NA == 1 && XS != 1 && L::CK == 1, "pw20 lite path: RT 1, one substream, x not in a slab");
      ce_issue<L::CES, 4>(ceb[0], cP, cgs, crl);
      asm volatile("cp.async.commit_group;\n" ::: "memory");
      asm volatile("cp.async.wait_group 2;\n" ::: "memory");
      const uint4 q0 = wb[s][0][0][0], q1 = wb[s][0][0][1];
      uint32_t B[MT][8];
#pragma unroll
      for (int n = 0; n < MT; n++) {
        uint4 b0, b1;
        if (XS == 2) { b0 = gxp<(V >> 15) & 3>(xp[n] + u * 64); b1 = gxp<(V >> 15) & 3>(xp[n] + u * 64 + 8); }
        else { b0 = xb[s % XB][n][0]; b1 = xb[s % XB][n][1]; }
        B[n][0] = b0.x; B[n][1] = b0.y; B[n][2] = b0.z; B[n][3] = b0.w; B[n][4] = b1.x; B[n][5] = b1.y; B[n][6] = b1.z; B[n][7] = b1.w;
      }
      if (u + ST < KU) ld(s, u + ST);
      const uint32_t Lw[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
      for (int jj = 0; jj < 4; jj++) {
        uint32_t A4[4];
        ce_step1<L::CES, ((V >> 20) & 1) ? 7 : 2>(ceb[0], clut, crl, ct0, ct1, Lw[2 * jj], Lw[2 * jj + 1], jj, A4);
        uint32_t n3 = nx;  // this unit's escape run (warp-uniform), applied to step jj's registers
        int e3 = ep;
        while ((n3 >> 20) == (uint32_t)u) {
          if (lane == (int)((n3 >> 15) & 31) && (int)((n3 >> 13) & 3) == jj) {
            const uint32_t r = (n3 >> 11) & 3, xr = (n3 & 0xffu) << (7 + ((n3 >> 10) & 1) * 16);
#pragma unroll
            for (int i = 0; i < 4; i++) if (i == (int)r) A4[i] ^= xr;
          }
          n3 = ++e3 < 32 ? __shfl_sync(~0u, e, e3) : ~0u;
        }
        if (jj == 3) { nx = n3; ep = e3; }
#pragma unroll
        for (int n = 0; n < MT; n++) hmma(acc[0][n][jj % NC], A4, B[n][2 * jj], B[n][2 * jj + 1]);
      }
      return;
    }
    uint32_t A[RT][NA][16], B[MT][8];
    uint4 xd[XS == 2 ? MT : 1][2];
    if (XS == 2 && LX && !((V >> 25) & 1)) {  // pw21: the router token loop loads x at use
#pragma unroll
      for (int n = 0; n < MT; n++) { xd[n][0] = gxp<(V >> 15) & 3>(xp[n] + u * 64); xd[n][1] = gxp<(V >> 15) & 3>(xp[n] + u * 64 + 8); }
    }
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int h = 0; h < NA; h++) {
        uint4 qv[NL];
        if constexpr (L::BK) {
          const int grp = u / L::CU, ci = u % L::CU, slot = grp % L::NS;
          if (ci == 0 && j == 0 && h == 0) gxwait(bbar + slot * 8, (grp / L::NS) & 1);
#pragma unroll
          for (int i = 0; i < NL; i++) qv[i] = lds4(bring + slot * L::SB + ((j * NA + h) * L::CU + ci) * NL * 512 + i * 512 + lane * 16);
        } else {
#pragma unroll
          for (int i = 0; i < NL; i++) qv[i] = wb[s][j][h][i];
        }
        const uint4* q = qv;
        if constexpr (!LDEC) {  // ladder L0/L1: consume the W words with a cheap XOR, no decode
#pragma unroll
          for (int i = 0; i < NL; i++) sk ^= q[i].x ^ q[i].y ^ q[i].z ^ q[i].w;
        } else if constexpr (CE) {  // pw18: one issue round + commit per unit (u, j); the words this unit reads have landed
#pragma unroll
          for (int s = 0; s < L::CK; s++) ce_issue<L::CES, 4 / L::CK>(ceb[s], cP, cgs + (size_t)s * cP * 128, crl + s * L::CES * 128);
          asm volatile("cp.async.commit_group;\n" ::: "memory");
          asm volatile("cp.async.wait_group 2;\n" ::: "memory");
          ce_unit<L::CES, ((V >> 20) & 1) ? 7 : 2, L::CK>(ceb, clut, crl, ct0, ct1, q[0], q[1], A[j][h]);
        } else if (C12) {
          const uint32_t T0 = h ? p.U0 : p.T0, T1 = h ? p.U1 : p.T1;
          if (V & 1) {
            dec1(q[0].x, q[0].y, q[2].x, T0, T1, A[j][h]); dec1(q[0].z, q[0].w, q[2].y, T0, T1, A[j][h] + 4);
            dec1(q[1].x, q[1].y, q[2].z, T0, T1, A[j][h] + 8); dec1(q[1].z, q[1].w, q[2].w, T0, T1, A[j][h] + 12);
          } else {
            dec0(q[0].x, q[0].y, q[2].x, T0, T1, A[j][h]); dec0(q[0].z, q[0].w, q[2].y, T0, T1, A[j][h] + 4);
            dec0(q[1].x, q[1].y, q[2].z, T0, T1, A[j][h] + 8); dec0(q[1].z, q[1].w, q[2].w, T0, T1, A[j][h] + 12);
          }
        } else {
#pragma unroll
          for (int i = 0; i < 4; i++) { A[j][h][4 * i] = q[i].x; A[j][h][4 * i + 1] = q[i].y; A[j][h][4 * i + 2] = q[i].z; A[j][h][4 * i + 3] = q[i].w; }
        }
      }
    if constexpr (((V >> 25) & 1) != 0) {  // pw21 router m=128/256 (V bit 25): the decoded W chunk (hi + lo) in registers
      // multiplies every n8 token tile of the CTA in turn; tile n's x fragment is loaded at use (ld.global.nc, L1 shared
      // by the CTA's row-tile warps); one accumulator per (row tile, token tile)
      static_assert(XS == 2 && !L::BK, "token loop: x at use, register W ring");
      if (u + ST < KU) ld(s, u + ST);
      if ((C12 || CE) && LDEC) {
#pragma unroll
        for (int j = 0; j < RT; j++)
          while ((nx >> 20) == (uint32_t)(u * RT + j)) {
            if (lane == (int)((nx >> 15) & 31)) {
              const uint32_t r = (nx >> 11) & 15, hm = (nx >> 9) & 1, xr = (nx & 0x7fu) << (8 + ((nx >> 10) & 1) * 16);
#pragma unroll
              for (int h = 0; h < NA; h++)
#pragma unroll
                for (int i = 0; i < 16; i++) if (i == (int)r && h == (int)hm) A[j][h][i] ^= xr;
            }
            nx = ++ep < 32 ? __shfl_sync(~0u, e, ep) : ~0u;
          }
      }
#pragma unroll
      for (int n = 0; n < MT; n++) {
        const __nv_bfloat16* xq = p.x + (size_t)min(n * 8 + g, p.m - 1) * p.K + (c0 + u) * 64 + t * 16;
        const uint4 b0 = gx(xq), b1 = gx(xq + 8);
        const uint32_t Bq[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
        for (int q = 0; q < 4; q++)
#pragma unroll
          for (int j = 0; j < RT; j++)
#pragma unroll
            for (int h = 0; h < NA; h++) hmma(acc[j][n][0], A[j][h] + 4 * q, Bq[2 * q], Bq[2 * q + 1]);
      }
      return;
    }
    if constexpr (L::BK) {  // the slot's last chunk is decoded into registers: refill it with the group NS ahead
      const int grp = u / L::CU;
      if (u % L::CU == L::CU - 1 && grp + L::NS < L::G) {
        __syncwarp();
        if (lane == 0) {
          if constexpr (!(V & 256)) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");  // V bit 8: no proxy fence
          bissue(grp % L::NS, grp + L::NS);
        }
      }
    }
#pragma unroll
    for (int n = 0; n < MT; n++) {
      uint4 b0, b1;
      if (XS == 1) {
        const uint32_t a = sb + ((wk * KU + u) * TK + n * 8 + g) * 128;
        b0 = lds4(a + (((2 * t) ^ g) << 4)); b1 = lds4(a + (((2 * t + 1) ^ g) << 4));
      } else if (XS == 2) { b0 = xd[n % (XS == 2 ? MT : 1)][0]; b1 = xd[n % (XS == 2 ? MT : 1)][1]; }
      else { b0 = xb[s % XB][n][0]; b1 = xb[s % XB][n][1]; }
      B[n][0] = b0.x; B[n][1] = b0.y; B[n][2] = b0.z; B[n][3] = b0.w; B[n][4] = b1.x; B[n][5] = b1.y; B[n][6] = b1.z; B[n][7] = b1.w;
    }
    if constexpr (LAD == 2 || LAD == 3) {  // ladder L1/L2: consume x (and the decoded A) instead of the mma
#pragma unroll
      for (int n = 0; n < MT; n++)
#pragma unroll
        for (int i = 0; i < 8; i++) sk ^= B[n][i];
      if constexpr (LAD == 3) {
#pragma unroll
        for (int j = 0; j < RT; j++)
#pragma unroll
          for (int h = 0; h < NA; h++)
#pragma unroll
            for (int i = 0; i < 16; i++) sk ^= A[j][h][i];
      }
    }
    if (u + ST < KU) ld(s, u + ST);
    if ((C12 || CE) && LDEC) {
#pragma unroll
      for (int j = 0; j < RT; j++)
        while ((nx >> 20) == (uint32_t)(u * RT + j)) {
          if (lane == (int)((nx >> 15) & 31)) {
            const uint32_t r = (nx >> 11) & 15, hm = (nx >> 9) & 1, xr = CE ? (nx & 0xffu) << (7 + ((nx >> 10) & 1) * 16) : (nx & 0x7fu) << (8 + ((nx >> 10) & 1) * 16);
#pragma unroll
            for (int h = 0; h < NA; h++)
#pragma unroll
              for (int i = 0; i < 16; i++) if (i == (int)r && h == (int)hm) A[j][h][i] ^= xr;
          }
          nx = ++ep < 32 ? __shfl_sync(~0u, e, ep) : ~0u;
        }
    }
    if constexpr (LMMA) {
#pragma unroll
    for (int q = 0; q < 4; q++)
#pragma unroll
      for (int j = 0; j < RT; j++)
#pragma unroll
        for (int n = 0; n < MT; n++)
#pragma unroll
          for (int h = 0; h < NA; h++) hmma(acc[j][n][(q * NA + h) % NC], A[j][h] + 4 * q, B[n][2 * q], B[n][2 * q + 1]);
    }
  };
  if (V & 2) {  // rolled: small code (cold instruction fetch), no cross-group load hoisting
#pragma unroll 1
    for (int u0 = 0; u0 < KU; u0 += ST)
#pragma unroll
      for (int s = 0; s < ST; s++) if (u0 + s < KU) step(u0 + s, s);
  } else {
#pragma unroll
    for (int u = 0; u < KU; u++) step(u, u % ST);
  }
  if constexpr (LAD != 0) {  // ladder: keep the work alive with a store that never happens in practice, then exit
    if constexpr (LAD == 4) {
#pragma unroll
      for (int j = 0; j < RT; j++)
#pragma unroll
        for (int n = 0; n < MT; n++)
#pragma unroll
          for (int c = 0; c < NC; c++) sk ^= __float_as_uint(acc[j][n][c][0]) ^ __float_as_uint(acc[j][n][c][1]) ^ __float_as_uint(acc[j][n][c][2]) ^ __float_as_uint(acc[j][n][c][3]);
    }
    if (sk == (p.T0 ^ 0x9e3779b9u)) ((float*)p.ws)[threadIdx.x] = __uint_as_float(sk);
    return;
  }
  float4 f[RT][MT];
#pragma unroll
  for (int j = 0; j < RT; j++)
#pragma unroll
    for (int n = 0; n < MT; n++) {
      f[j][n] = make_float4(acc[j][n][0][0], acc[j][n][0][1], acc[j][n][0][2], acc[j][n][0][3]);
#pragma unroll
      for (int c = 1; c < NC; c++) add4(f[j][n], make_float4(acc[j][n][c][0], acc[j][n][c][1], acc[j][n][c][2], acc[j][n][c][3]));
    }
  if (WK == 1 && S == 1) {
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int n = 0; n < MT; n++) if (n * 8 + 2 * t < p.m) emit<O>(p, ((wr * RT + j) * MT + n) * 32 + lane, row0, f[j][n], MT);
    return;
  }
  float4* cs = red + WK * NI;
#pragma unroll
  for (int j = 0; j < RT; j++)
#pragma unroll
    for (int n = 0; n < MT; n++) red[wk * NI + ((wr * RT + j) * MT + n) * 32 + lane] = f[j][n];
  if constexpr (((V >> 27) & 1) != 0 || XZ) {  // acquire-poll the tile's zero flag (H100 probe: 0.35 us under volatile + fence)
    if (threadIdx.x == 0) {
      const int* fl = p.cnt + 2 * (bz * GX + bx) + 1, tgt = tk / S + 1;
      unsigned spins_ = 0;  // critic (pw27h): bounded wait; a stuck zeroer aborts visibly instead of hanging to the timeout
      while (true) { int v_; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(v_) : "l"(fl) : "memory"); if (v_ >= tgt) break; if (++spins_ > (1u << 24)) asm volatile("trap;"); }
    }
  }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (((V >> 28) & 1) != 0) asm volatile("griddepcontrol.wait;\n" ::: "memory");  // P1 (V bit 28): PDL zero kernel done
#endif
  __syncthreads();
  for (int idx = threadIdx.x; idx < NI; idx += NT) {
    if (!live(idx, MT, p.m)) continue;
    float4 v = red[idx];
#pragma unroll
    for (int q = 1; q < WK; q++) add4(v, red[q * NI + idx]);
    if (S == 1) emit<O>(p, idx, row0, v, MT);
    else if constexpr (((V >> 26) & 15) != 0) redemit<O>(p, idx, row0, v, MT);  // pw27: K split reduced by red.add, no cluster
    else {
      cs[idx] = v;
      if constexpr ((V & 4) != 0) __stcg(p.ws + ((size_t)(bz * GX + bx) * S + by) * NI + idx, v);  // pw11 (B): partial to its ws slot
    }
  }
  if (S == 1) return;
  if constexpr (((V >> 26) & 15) != 0) return;
  if constexpr ((V & 4) != 0) {  // pw11 (B): no cluster; the last-arriving CTA of the n-block sums the KS partials from L2
    asm volatile("fence.acq_rel.gpu;\n" ::: "memory");  // release this thread's partial stores (lighter than __threadfence's fence.sc)
    __syncthreads();
    __shared__ int gxlast;
    if (threadIdx.x == 0) gxlast = atomicAdd(p.cnt + bz * GX + bx, 1) == S - 1;
    __syncthreads();
    if (!gxlast) return;
    asm volatile("fence.acq_rel.gpu;\n" ::: "memory");  // acquire the other CTAs' partials
    const float4* wsp = p.ws + (size_t)(bz * GX + bx) * S * NI;
    for (int idx = threadIdx.x; idx < NI; idx += NT) {
      if (!live(idx, MT, p.m)) continue;
      float4 v = cs[idx];
#pragma unroll
      for (int q = 0; q < S; q++) if (q != by) add4(v, __ldcg(wsp + q * NI + idx));
      emit<O>(p, idx, row0, v, MT);
    }
    if (threadIdx.x == 0) p.cnt[bz * GX + bx] = 0;  // self-cleaning for graph replay
  } else {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
  uint32_t rk; asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rk));
  constexpr int PER = (NI + S - 1) / S;
  const uint32_t ca = (uint32_t)__cvta_generic_to_shared(cs);
  for (int idx = rk * PER + threadIdx.x; idx < min(NI, (int)(rk + 1) * PER); idx += NT) {
    if (!live(idx, MT, p.m)) continue;
    float4 v[S];
#pragma unroll
    for (int q = 0; q < S; q++) {
      uint32_t ra; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(ra) : "r"(ca + idx * 16), "r"(q));
      asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];\n" : "=f"(v[q].x), "=f"(v[q].y), "=f"(v[q].z), "=f"(v[q].w) : "r"(ra) : "memory");
    }
#pragma unroll
    for (int q = 1; q < S; q++) add4(v[0], v[q]);
    emit<O>(p, idx, row0, v[0], MT);
  }
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
#else
  float4* wsp = p.ws + (size_t)(bz * GX + bx) * S * NI;
  for (int idx = threadIdx.x; idx < NI; idx += NT) wsp[by * NI + idx] = cs[idx];
  __threadfence();
  __syncthreads();
  __shared__ int last;
  if (threadIdx.x == 0) last = atomicAdd(p.cnt + bz * GX + bx, 1) == S - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  for (int idx = threadIdx.x; idx < NI; idx += NT) {
    if (!live(idx, MT, p.m)) continue;
    float4 v = __ldcg(wsp + idx);
    for (int q = 1; q < S; q++) add4(v, __ldcg(wsp + q * NI + idx));
    emit<O>(p, idx, row0, v, MT);
  }
  if (threadIdx.x == 0) p.cnt[bz * GX + bx] = 0;
#endif
  }
}

// Predicated format1 repair; source inputs are copied before any output write.
__device__ __forceinline__ void gv_fix_pred(uint32_t (&a)[16],uint32_t r,uint32_t xr,int lane,uint32_t target,uint32_t hm){
  asm volatile("{ .reg .b32 ri, mask; .reg .pred pl, pa, pr; mov.b32 ri, %16; mov.b32 mask, %17; setp.eq.u32 pl, %18, %19; setp.eq.and.u32 pa, %20, 0, pl; setp.eq.and.u32 pr, ri, 0, pa; @pr xor.b32 %0, %0, mask; setp.eq.and.u32 pr, ri, 1, pa; @pr xor.b32 %1, %1, mask; setp.eq.and.u32 pr, ri, 2, pa; @pr xor.b32 %2, %2, mask; setp.eq.and.u32 pr, ri, 3, pa; @pr xor.b32 %3, %3, mask; setp.eq.and.u32 pr, ri, 4, pa; @pr xor.b32 %4, %4, mask; setp.eq.and.u32 pr, ri, 5, pa; @pr xor.b32 %5, %5, mask; setp.eq.and.u32 pr, ri, 6, pa; @pr xor.b32 %6, %6, mask; setp.eq.and.u32 pr, ri, 7, pa; @pr xor.b32 %7, %7, mask; setp.eq.and.u32 pr, ri, 8, pa; @pr xor.b32 %8, %8, mask; setp.eq.and.u32 pr, ri, 9, pa; @pr xor.b32 %9, %9, mask; setp.eq.and.u32 pr, ri, 10, pa; @pr xor.b32 %10, %10, mask; setp.eq.and.u32 pr, ri, 11, pa; @pr xor.b32 %11, %11, mask; setp.eq.and.u32 pr, ri, 12, pa; @pr xor.b32 %12, %12, mask; setp.eq.and.u32 pr, ri, 13, pa; @pr xor.b32 %13, %13, mask; setp.eq.and.u32 pr, ri, 14, pa; @pr xor.b32 %14, %14, mask; setp.eq.and.u32 pr, ri, 15, pa; @pr xor.b32 %15, %15, mask; }" : "+r"(a[0]), "+r"(a[1]), "+r"(a[2]), "+r"(a[3]), "+r"(a[4]), "+r"(a[5]), "+r"(a[6]), "+r"(a[7]), "+r"(a[8]), "+r"(a[9]), "+r"(a[10]), "+r"(a[11]), "+r"(a[12]), "+r"(a[13]), "+r"(a[14]), "+r"(a[15]) : "r"(r),"r"(xr),"r"(lane),"r"(target),"r"(hm));
}

template <int FMT, int MT, int ST, int WR, int WK, int KU, int S, int RT, int XS, int V>
__global__ void __launch_bounds__(WR * WK * 32, ((V >> 12) & 7) ? ((V >> 12) & 7) : (XS == 2 && WR * WK * 32 < 512 ? 512 / (WR * WK * 32) : 1)) gv_predxor(G p) {
  static_assert((FMT == 1 && MT == 1 && ST == 1 && WR == 2 && WK == 16 && KU == 8 && S == 1 && RT == 1 && XS == 0 && V == 3) ||
      (FMT == 1 && MT == 1 && ST == 2 && WR == 2 && WK == 8 && KU == 16 && S == 1 && RT == 1 && XS == 0 && V == 0) ||
      (FMT == 1 && MT == 1 && ST == 2 && WR == 2 && WK == 8 && KU == 16 && S == 1 && RT == 1 && XS == 0 && V == 1) ||
      (FMT == 1 && MT == 2 && ST == 2 && WR == 2 && WK == 8 && KU == 16 && S == 1 && RT == 1 && XS == 0 && V == 0) ||
      (FMT == 1 && MT == 2 && ST == 2 && WR == 2 && WK == 8 && KU == 16 && S == 1 && RT == 1 && XS == 0 && V == 1) ||
      (FMT == 1 && MT == 2 && ST == 2 && WR == 1 && WK == 4 && KU == 16 && S == 1 && RT == 1 && XS == 2 && V == 1) ||
      (FMT == 1 && MT == 1 && ST == 2 && WR == 1 && WK == 8 && KU == 16 && S == 1 && RT == 1 && XS == 0 && V == 1), "GVpredxor only seven reviewed format1 templates");
  // pw14 probes only (never registered as paths): V bits 9-11 = ladder level LAD (1: W loads only, 2: + x loads,
  // 3: + decode, 4: + mma; no reduction / store); bits 12-14 = __launch_bounds__ minBlocks override (register cap).
  constexpr int LAD = (V >> 9) & 7;
  constexpr bool LX = LAD != 1, LDEC = LAD == 0 || LAD >= 3, LMMA = LAD == 0 || LAD == 4;
  uint32_t sk = 0;
  constexpr bool C12 = FMT & 1, HL = FMT == 2 || FMT == 3, CE = FMT == 4;
  constexpr int NL = CE ? 2 : (C12 ? 3 : 4), NA = HL ? 2 : 1, NW = WR * WK, NT = NW * 32, NI = WR * RT * MT * 32, CH = WK * KU, TK = MT * 8;
  constexpr int NC = (MT * RT * NA >= 2 || (CE && ((V >> 24) & 1))) ? 1 : 2, XB = XS ? 1 : ST;  // pw20 lite: one accumulator set (register budget)  // XS 2: x straight from global at use (L1), no ring
  using O = typename std::conditional<HL, float, __nv_bfloat16>::type;
  extern __shared__ __align__(16) uint8_t smem[];
  float4* red = (float4*)(smem + (XS == 1 ? CH * TK * 128 : 0));
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, wr = w % WR, wk = w / WR, g = lane >> 2, t = lane & 3;
  constexpr bool XZ = ((V >> 29) & 1) != 0;  // P1 (V bit 29): grid.z plane 0 = zeroer CTAs (one per tile), the K splits follow
  const int bz = blockIdx.x, bx = blockIdx.y, by = (int)blockIdx.z - (XZ ? 1 : 0), GX = gridDim.y;
  if constexpr (XZ) {
    if (by < 0) {  // dispatched first (lowest z): zero this tile of the output, publish, exit; no W, no x
      O* o = (O*)p.out + (size_t)bz * TK * p.N + bx * WR * RT * 16;
      const int R = WR * RT * 16, mm = min(TK, p.m - bz * TK);
      for (int i = threadIdx.x; i < mm * R; i += WR * WK * 32) sto(o + (size_t)(i / R) * p.N + i % R, 0.f);
      __threadfence(); __syncthreads();
      if (threadIdx.x == 0) atomicAdd(p.cnt + 2 * (bz * GX + bx) + 1, 1);
      return;
    }
  }
  p.x += (size_t)bz * TK * p.K; p.out = (uint8_t*)p.out + (size_t)bz * TK * p.N * sizeof(O); p.m = min(TK, p.m - bz * TK);
  const int cb = by * CH, c0 = cb + wk * KU, row0 = bx * WR * RT * 16;
  const size_t wo = ((size_t)(bx * WR + wr) * RT * p.KC + c0) * (NL * 32) + lane, js = (size_t)p.KC * NL * 32;
  const __nv_bfloat16* xp[MT];
#pragma unroll
  for (int n = 0; n < MT; n++) xp[n] = p.x + (size_t)min(n * 8 + g, p.m - 1) * p.K + c0 * 64 + t * 16;
  uint4 wb[ST][RT][NA][NL], xb[XB][MT][2];
  using L = GL<FMT, MT, WR, WK, KU, S, RT, XS, V>;
  static_assert(KU % L::CU == 0, "chunks per slot must divide KU");
  auto ld = [&](int s, int u) {
    if constexpr (!L::BK) {
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int h = 0; h < NA; h++)
#pragma unroll
        for (int i = 0; i < NL; i++) wb[s][j][h][i] = gnc((h ? p.W2 : p.W) + wo + j * js + (u * NL + i) * 32);
    }
    if (!XS && LX) {
#pragma unroll
      for (int n = 0; n < MT; n++) { xb[s % XB][n][0] = gxp<(V >> 15) & 3>(xp[n] + u * 64); xb[s % XB][n][1] = gxp<(V >> 15) & 3>(xp[n] + u * 64 + 8); }
    }
  };
  uint32_t bbar = 0, bring = 0;
  auto bissue = [&](int slot, int grp) {  // lane 0 only: fill one slot with slot-group grp (CU chunks per row tile / matrix)
    const uint32_t bar = bbar + slot * 8;
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "n"(L::SB) : "memory");
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int h = 0; h < NA; h++)
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                     :: "r"(bring + slot * L::SB + (j * NA + h) * L::CU * NL * 512),
                        "l"((h ? p.W2 : p.W) + (wo - lane) + j * js + (size_t)grp * L::CU * NL * 32), "n"(L::CU * NL * 512), "r"(bar) : "memory");
  };
  if constexpr (L::BK) {
    const uint32_t s0 = (uint32_t)__cvta_generic_to_shared(smem) + L::SLAB + L::REDB;
    bbar = s0 + w * L::NS * 8;
    bring = s0 + L::BAR + w * L::NS * L::SB;
    if (lane == 0) {
#pragma unroll
      for (int k = 0; k < L::NS; k++) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(bbar + k * 8));
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#pragma unroll
      for (int k = 0; k < L::NS; k++) bissue(k, k);
    }
    __syncwarp();
  }
  int tk = 0;
  if constexpr (((V >> 27) & 1) != 0 || XZ) {  // P1 (V bit 27): no memset node; the split-0 CTA of the tile zeroes it, then raises a flag
    if (w == 0) {
      const int tile = bz * GX + bx, R = WR * RT * 16;
      if (lane == 0) tk = atomicAdd(p.cnt + 2 * tile, 1);
      if (by == 0 && !XZ) {
        O* o = (O*)p.out + row0;
        for (int i = lane; i < p.m * R; i += 32) sto(o + (size_t)(i / R) * p.N + i % R, 0.f);
        __threadfence(); __syncwarp();
        if (lane == 0) atomicAdd(p.cnt + 2 * tile + 1, 1);
      }
    }
  }
#pragma unroll
  for (int s = 0; s < ST; s++) if (s < KU) ld(s, s);
  CeS ceb[CE ? L::CK : 1];
  uint32_t cP = 0u, crl = 0u, clut = 0u, ct0 = 0u, ct1 = 0u;
  const uint8_t* cgs = nullptr;
  if constexpr (CE) {  // pw18: segment start, ring prologue (3 issue rounds, one commit group each), LUT + length table
#pragma unroll
    for (int s = 0; s < L::CK; s++) ceb[s] = CeS{0u, 0u, 0u, 0u, 0u, 0u};
    const uint32_t gw = (uint32_t)((by * GX + bx) * NW + w);
    const uint32_t* soff = (const uint32_t*)(((uint64_t)p.U1 << 32) | p.U0);
    const uint32_t s0 = __ldg(soff + gw);
    cP = (__ldg(soff + gw + 1) - s0) / (32u * L::CK);
    cgs = (const uint8_t*)p.W2 + ((size_t)s0 + lane) * 4;  // word 0 of this lane, substream 0 (substream s: + s * P * 128 B)
    const uint32_t cs00 = (uint32_t)__cvta_generic_to_shared(smem) + L::SLAB + L::REDB;
    const uint32_t cs0 = (cs00 + L::CAL - 1u) & ~(uint32_t)(L::CAL - 1);  // ring_s | slot offset needs ES * 128 alignment
    crl = cs0 + (w * 32 * L::CED + lane) * 4;
    clut = cs0 + NW * 32 * L::CED * 4 + (((V >> 20) & 1) ? lane * 4 : 0);
    const uint32_t* gl = (const uint32_t*)(((uint64_t)p.T1 << 32) | p.T0);
    ct0 = __ldg(gl + 256); ct1 = __ldg(gl + 257);
#pragma unroll
    for (int k = 0; k < 3; k++) {
#pragma unroll
      for (int s = 0; s < L::CK; s++) ce_issue<L::CES, 4 / L::CK>(ceb[s], cP, cgs + (size_t)s * cP * 128, crl + s * L::CES * 128);
      asm volatile("cp.async.commit_group;\n" ::: "memory");
    }
    uint32_t* sl = (uint32_t*)(smem + L::SLAB + L::REDB + (cs0 - cs00) + NW * 32 * L::CED * 4);
    constexpr int NE = (256 + NT - 1) / NT;  // LUT entries per thread: all loads in flight at once, then the stores
    uint32_t lv[NE];
#pragma unroll
    for (int k = 0; k < NE; k++) lv[k] = threadIdx.x + k * NT < 256 ? __ldg(gl + threadIdx.x + k * NT) : 0u;
#pragma unroll
    for (int k = 0; k < NE; k++) {
      const int x = threadIdx.x + k * NT;
      if (x < 256) {
        if constexpr (((V >> 20) & 1) != 0) {  // entry x for lane l at word 32 x + l: 8 x 16 B per entry
#pragma unroll
          for (int q = 0; q < 8; q++) ((uint4*)(sl + x * 32))[q] = make_uint4(lv[k], lv[k], lv[k], lv[k]);
        } else sl[x] = lv[k];
      }
    }
    __syncthreads();
  }
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem);
  if (XS == 1) {
    for (int e = threadIdx.x; e < CH * TK * 8; e += NT) {
      const int s8 = e & 7, tok = (e >> 3) % TK, ch = e / (TK * 8);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sb + ((ch * TK + tok) * 8 + (s8 ^ (tok & 7))) * 16),
                   "l"(p.x + (size_t)min(tok, p.m - 1) * p.K + (cb + ch) * 64 + s8 * 8));
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_all;\n" ::: "memory");
  }
  uint32_t e = ~0u, nx = ~0u;
  int ep = 0;
  if ((C12 || CE) && LDEC) { e = p.esc[((size_t)(by * GX + bx) * NW + w) * 32 + lane]; nx = __shfl_sync(~0u, e, 0); }
  if (XS == 1) __syncthreads();
  float acc[RT][MT][NC][4];
#pragma unroll
  for (int j = 0; j < RT; j++)
#pragma unroll
    for (int n = 0; n < MT; n++)
#pragma unroll
      for (int c = 0; c < NC; c++) acc[j][n][c][0] = acc[j][n][c][1] = acc[j][n][c][2] = acc[j][n][c][3] = 0.f;
  auto step = [&](const int u, const int s) {
    if constexpr (CE && ((V >> 24) & 1)) {  // pw20 lite (V bit 24): K = 1, decode and consume one k16 step at a time
      static_assert(RT == 1 && NA == 1 && XS != 1 && L::CK == 1, "pw20 lite path: RT 1, one substream, x not in a slab");
      ce_issue<L::CES, 4>(ceb[0], cP, cgs, crl);
      asm volatile("cp.async.commit_group;\n" ::: "memory");
      asm volatile("cp.async.wait_group 2;\n" ::: "memory");
      const uint4 q0 = wb[s][0][0][0], q1 = wb[s][0][0][1];
      uint32_t B[MT][8];
#pragma unroll
      for (int n = 0; n < MT; n++) {
        uint4 b0, b1;
        if (XS == 2) { b0 = gxp<(V >> 15) & 3>(xp[n] + u * 64); b1 = gxp<(V >> 15) & 3>(xp[n] + u * 64 + 8); }
        else { b0 = xb[s % XB][n][0]; b1 = xb[s % XB][n][1]; }
        B[n][0] = b0.x; B[n][1] = b0.y; B[n][2] = b0.z; B[n][3] = b0.w; B[n][4] = b1.x; B[n][5] = b1.y; B[n][6] = b1.z; B[n][7] = b1.w;
      }
      if (u + ST < KU) ld(s, u + ST);
      const uint32_t Lw[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
      for (int jj = 0; jj < 4; jj++) {
        uint32_t A4[4];
        ce_step1<L::CES, ((V >> 20) & 1) ? 7 : 2>(ceb[0], clut, crl, ct0, ct1, Lw[2 * jj], Lw[2 * jj + 1], jj, A4);
        uint32_t n3 = nx;  // this unit's escape run (warp-uniform), applied to step jj's registers
        int e3 = ep;
        while ((n3 >> 20) == (uint32_t)u) {
          if (lane == (int)((n3 >> 15) & 31) && (int)((n3 >> 13) & 3) == jj) {
            const uint32_t r = (n3 >> 11) & 3, xr = (n3 & 0xffu) << (7 + ((n3 >> 10) & 1) * 16);
#pragma unroll
            for (int i = 0; i < 4; i++) if (i == (int)r) A4[i] ^= xr;
          }
          n3 = ++e3 < 32 ? __shfl_sync(~0u, e, e3) : ~0u;
        }
        if (jj == 3) { nx = n3; ep = e3; }
#pragma unroll
        for (int n = 0; n < MT; n++) hmma(acc[0][n][jj % NC], A4, B[n][2 * jj], B[n][2 * jj + 1]);
      }
      return;
    }
    uint32_t A[RT][NA][16], B[MT][8];
    uint4 xd[XS == 2 ? MT : 1][2];
    if (XS == 2 && LX && !((V >> 25) & 1)) {  // pw21: the router token loop loads x at use
#pragma unroll
      for (int n = 0; n < MT; n++) { xd[n][0] = gxp<(V >> 15) & 3>(xp[n] + u * 64); xd[n][1] = gxp<(V >> 15) & 3>(xp[n] + u * 64 + 8); }
    }
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int h = 0; h < NA; h++) {
        uint4 qv[NL];
        if constexpr (L::BK) {
          const int grp = u / L::CU, ci = u % L::CU, slot = grp % L::NS;
          if (ci == 0 && j == 0 && h == 0) gxwait(bbar + slot * 8, (grp / L::NS) & 1);
#pragma unroll
          for (int i = 0; i < NL; i++) qv[i] = lds4(bring + slot * L::SB + ((j * NA + h) * L::CU + ci) * NL * 512 + i * 512 + lane * 16);
        } else {
#pragma unroll
          for (int i = 0; i < NL; i++) qv[i] = wb[s][j][h][i];
        }
        const uint4* q = qv;
        if constexpr (!LDEC) {  // ladder L0/L1: consume the W words with a cheap XOR, no decode
#pragma unroll
          for (int i = 0; i < NL; i++) sk ^= q[i].x ^ q[i].y ^ q[i].z ^ q[i].w;
        } else if constexpr (CE) {  // pw18: one issue round + commit per unit (u, j); the words this unit reads have landed
#pragma unroll
          for (int s = 0; s < L::CK; s++) ce_issue<L::CES, 4 / L::CK>(ceb[s], cP, cgs + (size_t)s * cP * 128, crl + s * L::CES * 128);
          asm volatile("cp.async.commit_group;\n" ::: "memory");
          asm volatile("cp.async.wait_group 2;\n" ::: "memory");
          ce_unit<L::CES, ((V >> 20) & 1) ? 7 : 2, L::CK>(ceb, clut, crl, ct0, ct1, q[0], q[1], A[j][h]);
        } else if (C12) {
          const uint32_t T0 = h ? p.U0 : p.T0, T1 = h ? p.U1 : p.T1;
          if (V & 1) {
            dec1(q[0].x, q[0].y, q[2].x, T0, T1, A[j][h]); dec1(q[0].z, q[0].w, q[2].y, T0, T1, A[j][h] + 4);
            dec1(q[1].x, q[1].y, q[2].z, T0, T1, A[j][h] + 8); dec1(q[1].z, q[1].w, q[2].w, T0, T1, A[j][h] + 12);
          } else {
            dec0(q[0].x, q[0].y, q[2].x, T0, T1, A[j][h]); dec0(q[0].z, q[0].w, q[2].y, T0, T1, A[j][h] + 4);
            dec0(q[1].x, q[1].y, q[2].z, T0, T1, A[j][h] + 8); dec0(q[1].z, q[1].w, q[2].w, T0, T1, A[j][h] + 12);
          }
        } else {
#pragma unroll
          for (int i = 0; i < 4; i++) { A[j][h][4 * i] = q[i].x; A[j][h][4 * i + 1] = q[i].y; A[j][h][4 * i + 2] = q[i].z; A[j][h][4 * i + 3] = q[i].w; }
        }
      }
    if constexpr (((V >> 25) & 1) != 0) {  // pw21 router m=128/256 (V bit 25): the decoded W chunk (hi + lo) in registers
      // multiplies every n8 token tile of the CTA in turn; tile n's x fragment is loaded at use (ld.global.nc, L1 shared
      // by the CTA's row-tile warps); one accumulator per (row tile, token tile)
      static_assert(XS == 2 && !L::BK, "token loop: x at use, register W ring");
      if (u + ST < KU) ld(s, u + ST);
      if ((C12 || CE) && LDEC) {
#pragma unroll
        for (int j = 0; j < RT; j++)
          while ((nx >> 20) == (uint32_t)(u * RT + j)) {
            if (lane == (int)((nx >> 15) & 31)) {
              const uint32_t r = (nx >> 11) & 15, hm = (nx >> 9) & 1, xr = (nx & 0x7fu) << (8 + ((nx >> 10) & 1) * 16);
#pragma unroll
              for (int h = 0; h < NA; h++)
#pragma unroll
                for (int i = 0; i < 16; i++) if (i == (int)r && h == (int)hm) A[j][h][i] ^= xr;
            }
            nx = ++ep < 32 ? __shfl_sync(~0u, e, ep) : ~0u;
          }
      }
#pragma unroll
      for (int n = 0; n < MT; n++) {
        const __nv_bfloat16* xq = p.x + (size_t)min(n * 8 + g, p.m - 1) * p.K + (c0 + u) * 64 + t * 16;
        const uint4 b0 = gx(xq), b1 = gx(xq + 8);
        const uint32_t Bq[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
        for (int q = 0; q < 4; q++)
#pragma unroll
          for (int j = 0; j < RT; j++)
#pragma unroll
            for (int h = 0; h < NA; h++) hmma(acc[j][n][0], A[j][h] + 4 * q, Bq[2 * q], Bq[2 * q + 1]);
      }
      return;
    }
    if constexpr (L::BK) {  // the slot's last chunk is decoded into registers: refill it with the group NS ahead
      const int grp = u / L::CU;
      if (u % L::CU == L::CU - 1 && grp + L::NS < L::G) {
        __syncwarp();
        if (lane == 0) {
          if constexpr (!(V & 256)) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");  // V bit 8: no proxy fence
          bissue(grp % L::NS, grp + L::NS);
        }
      }
    }
#pragma unroll
    for (int n = 0; n < MT; n++) {
      uint4 b0, b1;
      if (XS == 1) {
        const uint32_t a = sb + ((wk * KU + u) * TK + n * 8 + g) * 128;
        b0 = lds4(a + (((2 * t) ^ g) << 4)); b1 = lds4(a + (((2 * t + 1) ^ g) << 4));
      } else if (XS == 2) { b0 = xd[n % (XS == 2 ? MT : 1)][0]; b1 = xd[n % (XS == 2 ? MT : 1)][1]; }
      else { b0 = xb[s % XB][n][0]; b1 = xb[s % XB][n][1]; }
      B[n][0] = b0.x; B[n][1] = b0.y; B[n][2] = b0.z; B[n][3] = b0.w; B[n][4] = b1.x; B[n][5] = b1.y; B[n][6] = b1.z; B[n][7] = b1.w;
    }
    if constexpr (LAD == 2 || LAD == 3) {  // ladder L1/L2: consume x (and the decoded A) instead of the mma
#pragma unroll
      for (int n = 0; n < MT; n++)
#pragma unroll
        for (int i = 0; i < 8; i++) sk ^= B[n][i];
      if constexpr (LAD == 3) {
#pragma unroll
        for (int j = 0; j < RT; j++)
#pragma unroll
          for (int h = 0; h < NA; h++)
#pragma unroll
            for (int i = 0; i < 16; i++) sk ^= A[j][h][i];
      }
    }
    if (u + ST < KU) ld(s, u + ST);
    if ((C12 || CE) && LDEC) {
#pragma unroll
      for (int j = 0; j < RT; j++)
        while ((nx >> 20) == (uint32_t)(u * RT + j)) {
          const uint32_t r = (nx >> 11) & 15, hm = (nx >> 9) & 1, xr = (nx & 0x7fu) << (8 + ((nx >> 10) & 1) * 16);
          gv_fix_pred(A[j][0],r,xr,lane,(nx>>15)&31,hm);
          nx = ++ep < 32 ? __shfl_sync(~0u, e, ep) : ~0u;
        }
    }
    if constexpr (LMMA) {
#pragma unroll
    for (int q = 0; q < 4; q++)
#pragma unroll
      for (int j = 0; j < RT; j++)
#pragma unroll
        for (int n = 0; n < MT; n++)
#pragma unroll
          for (int h = 0; h < NA; h++) hmma(acc[j][n][(q * NA + h) % NC], A[j][h] + 4 * q, B[n][2 * q], B[n][2 * q + 1]);
    }
  };
  if (V & 2) {  // rolled: small code (cold instruction fetch), no cross-group load hoisting
#pragma unroll 1
    for (int u0 = 0; u0 < KU; u0 += ST)
#pragma unroll
      for (int s = 0; s < ST; s++) if (u0 + s < KU) step(u0 + s, s);
  } else {
#pragma unroll
    for (int u = 0; u < KU; u++) step(u, u % ST);
  }
  if constexpr (LAD != 0) {  // ladder: keep the work alive with a store that never happens in practice, then exit
    if constexpr (LAD == 4) {
#pragma unroll
      for (int j = 0; j < RT; j++)
#pragma unroll
        for (int n = 0; n < MT; n++)
#pragma unroll
          for (int c = 0; c < NC; c++) sk ^= __float_as_uint(acc[j][n][c][0]) ^ __float_as_uint(acc[j][n][c][1]) ^ __float_as_uint(acc[j][n][c][2]) ^ __float_as_uint(acc[j][n][c][3]);
    }
    if (sk == (p.T0 ^ 0x9e3779b9u)) ((float*)p.ws)[threadIdx.x] = __uint_as_float(sk);
    return;
  }
  float4 f[RT][MT];
#pragma unroll
  for (int j = 0; j < RT; j++)
#pragma unroll
    for (int n = 0; n < MT; n++) {
      f[j][n] = make_float4(acc[j][n][0][0], acc[j][n][0][1], acc[j][n][0][2], acc[j][n][0][3]);
#pragma unroll
      for (int c = 1; c < NC; c++) add4(f[j][n], make_float4(acc[j][n][c][0], acc[j][n][c][1], acc[j][n][c][2], acc[j][n][c][3]));
    }
  if (WK == 1 && S == 1) {
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int n = 0; n < MT; n++) if (n * 8 + 2 * t < p.m) emit<O>(p, ((wr * RT + j) * MT + n) * 32 + lane, row0, f[j][n], MT);
    return;
  }
  float4* cs = red + WK * NI;
#pragma unroll
  for (int j = 0; j < RT; j++)
#pragma unroll
    for (int n = 0; n < MT; n++) red[wk * NI + ((wr * RT + j) * MT + n) * 32 + lane] = f[j][n];
  if constexpr (((V >> 27) & 1) != 0 || XZ) {  // acquire-poll the tile's zero flag (H100 probe: 0.35 us under volatile + fence)
    if (threadIdx.x == 0) {
      const int* fl = p.cnt + 2 * (bz * GX + bx) + 1, tgt = tk / S + 1;
      unsigned spins_ = 0;  // critic (pw27h): bounded wait; a stuck zeroer aborts visibly instead of hanging to the timeout
      while (true) { int v_; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(v_) : "l"(fl) : "memory"); if (v_ >= tgt) break; if (++spins_ > (1u << 24)) asm volatile("trap;"); }
    }
  }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (((V >> 28) & 1) != 0) asm volatile("griddepcontrol.wait;\n" ::: "memory");  // P1 (V bit 28): PDL zero kernel done
#endif
  __syncthreads();
  for (int idx = threadIdx.x; idx < NI; idx += NT) {
    if (!live(idx, MT, p.m)) continue;
    float4 v = red[idx];
#pragma unroll
    for (int q = 1; q < WK; q++) add4(v, red[q * NI + idx]);
    if (S == 1) emit<O>(p, idx, row0, v, MT);
    else if constexpr (((V >> 26) & 15) != 0) redemit<O>(p, idx, row0, v, MT);  // pw27: K split reduced by red.add, no cluster
    else {
      cs[idx] = v;
      if constexpr ((V & 4) != 0) __stcg(p.ws + ((size_t)(bz * GX + bx) * S + by) * NI + idx, v);  // pw11 (B): partial to its ws slot
    }
  }
  if (S == 1) return;
  if constexpr (((V >> 26) & 15) != 0) return;
  if constexpr ((V & 4) != 0) {  // pw11 (B): no cluster; the last-arriving CTA of the n-block sums the KS partials from L2
    asm volatile("fence.acq_rel.gpu;\n" ::: "memory");  // release this thread's partial stores (lighter than __threadfence's fence.sc)
    __syncthreads();
    __shared__ int gxlast;
    if (threadIdx.x == 0) gxlast = atomicAdd(p.cnt + bz * GX + bx, 1) == S - 1;
    __syncthreads();
    if (!gxlast) return;
    asm volatile("fence.acq_rel.gpu;\n" ::: "memory");  // acquire the other CTAs' partials
    const float4* wsp = p.ws + (size_t)(bz * GX + bx) * S * NI;
    for (int idx = threadIdx.x; idx < NI; idx += NT) {
      if (!live(idx, MT, p.m)) continue;
      float4 v = cs[idx];
#pragma unroll
      for (int q = 0; q < S; q++) if (q != by) add4(v, __ldcg(wsp + q * NI + idx));
      emit<O>(p, idx, row0, v, MT);
    }
    if (threadIdx.x == 0) p.cnt[bz * GX + bx] = 0;  // self-cleaning for graph replay
  } else {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
  uint32_t rk; asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rk));
  constexpr int PER = (NI + S - 1) / S;
  const uint32_t ca = (uint32_t)__cvta_generic_to_shared(cs);
  for (int idx = rk * PER + threadIdx.x; idx < min(NI, (int)(rk + 1) * PER); idx += NT) {
    if (!live(idx, MT, p.m)) continue;
    float4 v[S];
#pragma unroll
    for (int q = 0; q < S; q++) {
      uint32_t ra; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(ra) : "r"(ca + idx * 16), "r"(q));
      asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];\n" : "=f"(v[q].x), "=f"(v[q].y), "=f"(v[q].z), "=f"(v[q].w) : "r"(ra) : "memory");
    }
#pragma unroll
    for (int q = 1; q < S; q++) add4(v[0], v[q]);
    emit<O>(p, idx, row0, v[0], MT);
  }
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
#else
  float4* wsp = p.ws + (size_t)(bz * GX + bx) * S * NI;
  for (int idx = threadIdx.x; idx < NI; idx += NT) wsp[by * NI + idx] = cs[idx];
  __threadfence();
  __syncthreads();
  __shared__ int last;
  if (threadIdx.x == 0) last = atomicAdd(p.cnt + bz * GX + bx, 1) == S - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  for (int idx = threadIdx.x; idx < NI; idx += NT) {
    if (!live(idx, MT, p.m)) continue;
    float4 v = __ldcg(wsp + idx);
    for (int q = 1; q < S; q++) add4(v, __ldcg(wsp + q * NI + idx));
    emit<O>(p, idx, row0, v, MT);
  }
  if (threadIdx.x == 0) p.cnt[bz * GX + bx] = 0;
#endif
  }
}


__global__ void p1zero(uint4* o, int n16) {  // P1 (V bit 28): the PDL primary, lets the gv grid launch at once
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;\n");
#endif
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += gridDim.x * blockDim.x) o[i] = make_uint4(0u, 0u, 0u, 0u);
}
template <int FMT, int MT, int ST, int WR, int WK, int KU, int S, int RT, int XS, int V>
static int64_t gvl(const G& p, cudaStream_t st, int occ) {
  constexpr int NI = WR * RT * MT * 32;
  auto k = gv<FMT, MT, ST, WR, WK, KU, S, RT, XS, V>;
  if constexpr ((FMT == 1 && MT == 1 && ST == 1 && WR == 2 && WK == 16 && KU == 8 && S == 1 && RT == 1 && XS == 0 && V == 3) ||
      (FMT == 1 && MT == 1 && ST == 2 && WR == 2 && WK == 8 && KU == 16 && S == 1 && RT == 1 && XS == 0 && V == 0) ||
      (FMT == 1 && MT == 1 && ST == 2 && WR == 2 && WK == 8 && KU == 16 && S == 1 && RT == 1 && XS == 0 && V == 1) ||
      (FMT == 1 && MT == 2 && ST == 2 && WR == 2 && WK == 8 && KU == 16 && S == 1 && RT == 1 && XS == 0 && V == 0) ||
      (FMT == 1 && MT == 2 && ST == 2 && WR == 2 && WK == 8 && KU == 16 && S == 1 && RT == 1 && XS == 0 && V == 1) ||
      (FMT == 1 && MT == 2 && ST == 2 && WR == 1 && WK == 4 && KU == 16 && S == 1 && RT == 1 && XS == 2 && V == 1) ||
      (FMT == 1 && MT == 1 && ST == 2 && WR == 1 && WK == 8 && KU == 16 && S == 1 && RT == 1 && XS == 0 && V == 1)) {
    const bool pooler = p.N == 4096 && p.K == 8192 && ((MT == 1 && (p.m == 1 || p.m == 8)) || (MT == 2 && p.m == 16));
    const bool unpooler = MT == 2 && p.N == 8192 && p.K == 4096 && p.m == 16;
    if (pooler || unpooler) k = gv_predxor<FMT, MT, ST, WR, WK, KU, S, RT, XS, V>;
  }
  if (p.N % (16 * WR * RT) || p.K != S * WK * KU * 64 || p.m < 1) return -1;
  const size_t sm = (XS == 1 ? (size_t)WK * KU * MT * 8 * 128 : 0) + ((WK > 1 || S > 1) ? (size_t)(WK + 1) * NI * 16 : 0)
                  + (size_t)(GL<FMT, MT, WR, WK, KU, S, RT, XS, V>::BAR + GL<FMT, MT, WR, WK, KU, S, RT, XS, V>::RING)
                  + (size_t)(GL<FMT, MT, WR, WK, KU, S, RT, XS, V>::CLUT + GL<FMT, MT, WR, WK, KU, S, RT, XS, V>::CRING);
  static bool once = false;
  if (!once) {
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    if ((V >> 17) & 1) { cudaFuncSetAttribute(k, cudaFuncAttributePreferredSharedMemoryCarveout, 0); cudaGetLastError(); }  // pw16
    if (S > 8) { cudaFuncSetAttribute(k, cudaFuncAttributeNonPortableClusterSizeAllowed, 1); cudaGetLastError(); }
    once = true;
  }
  static int cc = 0;
  if (!cc) { int dev; cudaGetDevice(&dev); cudaDeviceGetAttribute(&cc, cudaDevAttrComputeCapabilityMajor, dev); }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((p.m + 8 * MT - 1) / (8 * MT), p.N / (16 * WR * RT), S + (((V >> 29) & 1) ? 1 : 0)); cfg.blockDim = dim3(WR * WK * 32); cfg.dynamicSmemBytes = sm; cfg.stream = st;
  cudaLaunchAttribute at[1];
  const bool clu = S > 1 && cc >= 9 && !(V & 4) && !((V >> 26) & 15);  // pw11: V bit 2 = global last-arriver reduction, no cluster
  if (occ == 2) { cudaFuncAttributes fa; if (cudaFuncGetAttributes(&fa, k) != cudaSuccess) { cudaGetLastError(); return -1; } return fa.numRegs; }  // pw14
  if (occ) {  // pw11: how many CTAs of this launch can be co-resident (clusters: cudaOccupancyMaxActiveClusters)
    int n = 0;
    if (clu) {
      at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = 1; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = S; cfg.attrs = at; cfg.numAttrs = 1;
      if (cudaOccupancyMaxActiveClusters(&n, k, &cfg) != cudaSuccess) { cudaGetLastError(); return -1; }
      return (int64_t)n * S;
    }
    int dev, sms; cudaGetDevice(&dev); cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, k, WR * WK * 32, sm) != cudaSuccess) { cudaGetLastError(); return -1; }
    return (int64_t)n * sms;
  }
  if (clu) { at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = 1; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = S; cfg.attrs = at; cfg.numAttrs = 1; }
  if constexpr (((V >> 28) & 1) != 0) {
    const int n16 = (int)(((size_t)p.m * p.N * sizeof(typename std::conditional<FMT == 2 || FMT == 3, float, __nv_bfloat16>::type)) / 16);
    p1zero<<<(n16 + 255) / 256 < 8 ? (n16 + 255) / 256 : 8, 256, 0, st>>>((uint4*)p.out, n16);
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1; cfg.attrs = at; cfg.numAttrs = 1;
  }
  const cudaError_t r = cudaLaunchKernelEx(&cfg, k, p);
  if (r != cudaSuccess) cudaGetLastError();
  return (int64_t)r;
}

// FP32 router, small m: CTA = R full rows (no cross-CTA reduction), 512 threads split K, fp32 weights read once
template <int M, int R, int NT = 512>
__global__ void __launch_bounds__(NT) rr(const float4* W, const __nv_bfloat16* x, float* out, int m, int N, int K) {
  __shared__ float red[NT / 32][R * M];
  const int t = threadIdx.x, lane = t & 31, w = t >> 5, K4 = K >> 2, row0 = blockIdx.x * R;
  constexpr int NI = 2048 / NT;  // float4 per thread per row (K = 8192)
  float4 wv[R][NI];
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int i = 0; i < NI; i++) { const uint4 u = gnc(W + (size_t)(row0 + r) * K4 + t + i * NT); wv[r][i] = make_float4(__uint_as_float(u.x), __uint_as_float(u.y), __uint_as_float(u.z), __uint_as_float(u.w)); }
  float acc[R][M];
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int q = 0; q < M; q++) acc[r][q] = 0.f;
#pragma unroll
  for (int i = 0; i < NI; i++)
#pragma unroll
    for (int q = 0; q < M; q++) {
      if (q < m) {
        const uint2 xv = __ldg((const uint2*)(x + (size_t)q * K) + t + i * NT);
        const float2 a = __bfloat1622float2(*(const __nv_bfloat162*)&xv.x), b = __bfloat1622float2(*(const __nv_bfloat162*)&xv.y);
#pragma unroll
        for (int r = 0; r < R; r++) acc[r][q] = fmaf(wv[r][i].w, b.y, fmaf(wv[r][i].z, b.x, fmaf(wv[r][i].y, a.y, fmaf(wv[r][i].x, a.x, acc[r][q]))));
      }
    }
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int q = 0; q < M; q++) {
      float v = acc[r][q];
#pragma unroll
      for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
      if (lane == 0) red[w][r * M + q] = v;
    }
  __syncthreads();
  if (t < R * M && t % M < m) {
    float v = 0.f;
#pragma unroll
    for (int i = 0; i < NT / 32; i++) v += red[i][t];
    out[(size_t)(t % M) * N + row0 + t / M] = v;
  }
}

// ---------------------------------------------------------------- pw22: router m = 128 / 256 as a split-K GEMM
// out[tok][n] (fp32) = sum_k x[tok][k] (hi[n][k] + lo[n][k]), hi / lo = the BF16 split of the FP32 router weights, both
// products accumulated in fp32 by the same mma.sync (as SGLang above 8 tokens). CTA (n tile of BN, K split of KR, token
// block of 128): every x row slice (128 x KR) and hi / lo row slice (BN x KR) is requested up front with 16-B cp.async
// into XOR-swizzled smem (16-B chunk c of row r at r*KR*2 + ((c ^ (r & 7)) << 4): ldmatrix rows hit distinct banks),
// one wait, then KR/16 k16 steps of ldmatrix.x4 + mma.sync m16n8k16 (A = x rows, B = W rows: both K-major). 8 warps:
// warp w owns tokens 32 (w & 3) .. +31 and n 0.5 BN (w >> 2) .. +0.5 BN. Rows >= m are zero-filled (src-size 0).
// RED 0: red.global.add.v2.f32 into an output zeroed by the caller (a memset node in the same graph);
// RED 1: fp32 partial to a workspace slot, per-(token block, n tile) counter, the last arriving CTA sums the KS slots.
__device__ __forceinline__ void rs_cpa(uint32_t dst, const void* src, uint32_t sz) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(sz) : "memory");
}
__device__ __forceinline__ void rs_ldm4(uint32_t a, uint32_t* r) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void rs_red2(float* a, float x, float y) {
  asm volatile("red.global.add.v2.f32 [%0], {%1, %2};\n" :: "l"(a), "f"(x), "f"(y) : "memory");
}
template <int BN, int KR, int RED>
__global__ void __launch_bounds__(256, 1) rsk(const __nv_bfloat16* x, const __nv_bfloat16* hi, const __nv_bfloat16* lo, float* out,
                                            float4* ws, int* cnt, int m, int N, int K) {
  constexpr int TOK = 128, CH = KR / 8, WN = BN / 2, NT8 = WN / 8, RB = KR * 2;
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t sx = (uint32_t)__cvta_generic_to_shared(smem), sh = sx + TOK * RB, sl = sh + BN * RB;
  const int nt = blockIdx.x, ks = blockIdx.y, tb = blockIdx.z, NTL = gridDim.x, KS = gridDim.y;
  const int n0 = nt * BN, k0 = ks * KR, t0 = tb * TOK;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, t = lane & 3;
  for (int i = tid; i < TOK * CH; i += 256) {
    const int r = i / CH, c = i - r * CH, tok = t0 + r;
    rs_cpa(sx + r * RB + ((c ^ (r & 7)) << 4), x + (size_t)min(tok, m - 1) * K + k0 + c * 8, tok < m ? 16u : 0u);
  }
  for (int i = tid; i < BN * CH; i += 256) {
    const int r = i / CH, c = i - r * CH;
    const size_t o = (size_t)(n0 + r) * K + k0 + c * 8;
    const uint32_t d = r * RB + ((c ^ (r & 7)) << 4);
    rs_cpa(sh + d, hi + o, 16u); rs_cpa(sl + d, lo + o, 16u);
  }
  asm volatile("cp.async.commit_group;\ncp.async.wait_all;\n" ::: "memory");
  __syncthreads();
  const int wt = w & 3, wn = w >> 2;
  float acc[2][NT8][4];
#pragma unroll
  for (int i = 0; i < 2; i++)
#pragma unroll
    for (int j = 0; j < NT8; j++) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
  const int ar = lane & 15, ac = lane >> 4;                 // A: row (lane & 15) of the m16 tile, k chunk + (lane >> 4)
  const int bj = lane >> 3, br = (bj >> 1) * 8 + (lane & 7), bc = bj & 1;  // B: n row of the n16 pair, k chunk + (j & 1)
#pragma unroll 4
  for (int q = 0; q < KR / 16; q++) {
    uint32_t a[2][4];
#pragma unroll
    for (int mi = 0; mi < 2; mi++) {
      const int r = wt * 32 + mi * 16 + ar, c = 2 * q + ac;
      rs_ldm4(sx + r * RB + ((c ^ (r & 7)) << 4), a[mi]);
    }
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
      for (int nb = 0; nb < NT8 / 2; nb++) {
        const int r = wn * WN + nb * 16 + br, c = 2 * q + bc;
        uint32_t b[4];
        rs_ldm4((h ? sl : sh) + r * RB + ((c ^ (r & 7)) << 4), b);
#pragma unroll
        for (int mi = 0; mi < 2; mi++) {
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(acc[mi][2 * nb][0]), "+f"(acc[mi][2 * nb][1]), "+f"(acc[mi][2 * nb][2]), "+f"(acc[mi][2 * nb][3])
            : "r"(a[mi][0]), "r"(a[mi][1]), "r"(a[mi][2]), "r"(a[mi][3]), "r"(b[0]), "r"(b[1]));
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(acc[mi][2 * nb + 1][0]), "+f"(acc[mi][2 * nb + 1][1]), "+f"(acc[mi][2 * nb + 1][2]), "+f"(acc[mi][2 * nb + 1][3])
            : "r"(a[mi][0]), "r"(a[mi][1]), "r"(a[mi][2]), "r"(a[mi][3]), "r"(b[2]), "r"(b[3]));
        }
      }
  }
  if constexpr (RED == 0) {
#pragma unroll
    for (int mi = 0; mi < 2; mi++)
#pragma unroll
      for (int j = 0; j < NT8; j++) {
        const int tok = t0 + wt * 32 + mi * 16 + g, n = n0 + wn * WN + j * 8 + 2 * t;
        if (tok < m) rs_red2(out + (size_t)tok * N + n, acc[mi][j][0], acc[mi][j][1]);
        if (tok + 8 < m) rs_red2(out + (size_t)(tok + 8) * N + n, acc[mi][j][2], acc[mi][j][3]);
      }
  } else {
    constexpr int NF = 2 * NT8 * 32 * 8;  // float4 fragments per CTA partial (8 warps)
    float4* slot = ws + ((size_t)(tb * NTL + nt) * KS + ks) * NF;
#pragma unroll
    for (int mi = 0; mi < 2; mi++)
#pragma unroll
      for (int j = 0; j < NT8; j++) __stcg(slot + ((w * 2 + mi) * NT8 + j) * 32 + lane, make_float4(acc[mi][j][0], acc[mi][j][1], acc[mi][j][2], acc[mi][j][3]));
    __threadfence();
    __syncthreads();
    __shared__ int last;
    if (tid == 0) last = atomicAdd(cnt + tb * NTL + nt, 1) == KS - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    const float4* base = ws + (size_t)(tb * NTL + nt) * KS * NF;
    for (int i = tid; i < NF; i += 256) {
      float4 v = __ldcg(base + i);
      for (int k2 = 1; k2 < KS; k2++) { const float4 u = __ldcg(base + (size_t)k2 * NF + i); v.x += u.x; v.y += u.y; v.z += u.z; v.w += u.w; }
      const int l = i & 31, f = i >> 5, j = f % NT8, mi = (f / NT8) & 1, ww = f / NT8 / 2;
      const int tok = t0 + (ww & 3) * 32 + mi * 16 + (l >> 2), n = n0 + (ww >> 2) * WN + j * 8 + 2 * (l & 3);
      if (tok < m) { out[(size_t)tok * N + n] = v.x; out[(size_t)tok * N + n + 1] = v.y; }
      if (tok + 8 < m) { out[(size_t)(tok + 8) * N + n] = v.z; out[(size_t)(tok + 8) * N + n + 1] = v.w; }
    }
    if (tid == 0) cnt[tb * NTL + nt] = 0;  // self-cleaning for graph replay
  }
}
// ---------------------------------------------------------------- pw23: pipelined router split-K (rsk2)
// rsk with (1) the K slice loaded in NSTG commit groups, all issued up front; stage s is multiplied after
// cp.async.wait_group(NSTG - 1 - s) + a CTA barrier, so the mma of stage s overlaps the landing of s + 1 ..;
// (2) TOKS = 128 or 256 tokens per CTA: at m = 256 one CTA holds both 128-token blocks and every ldmatrix'd W fragment
// feeds MI = TOKS / 64 m16 tiles per warp (the W chain is not doubled); (3) RED 0 = red.add into an output zeroed by a
// memset node; RED 2 = red.add into a persistent fp32 workspace (zero at prepare), then per output tile a counter: the
// CTA that completes the tile copies it ws -> out and re-zeroes its ws tile and the counter (self-cleaning).
template <int BN, int KR, int NSTG, int TOKS, int RED, int MINB>
__global__ void __launch_bounds__(256, MINB) rsk2(const __nv_bfloat16* x, const __nv_bfloat16* hi, const __nv_bfloat16* lo, float* out,
                                                 float* ws, int* cnt, int m, int N, int K) {
  constexpr int CH = KR / 8, CS = CH / NSTG, WN = BN / 2, NT8 = WN / 8, RB = KR * 2, MI = TOKS / 64, QS = KR / 16 / NSTG;
  static_assert(CH % NSTG == 0 && NT8 % 2 == 0, "stages / n tiles");
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t sx = (uint32_t)__cvta_generic_to_shared(smem), sh = sx + TOKS * RB, sl = sh + BN * RB;
  const int nt = blockIdx.x, ks = blockIdx.y, tb = blockIdx.z, NTL = gridDim.x, KS = gridDim.y;
  const int n0 = nt * BN, k0 = ks * KR, t0 = tb * TOKS;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int s = 0; s < NSTG; s++) {  // stage s = 16-B chunks [s CS, (s + 1) CS) of every row
    for (int i = tid; i < TOKS * CS; i += 256) {
      const int r = i / CS, c = s * CS + i - (i / CS) * CS, tok = t0 + r;
      rs_cpa(sx + r * RB + ((c ^ (r & 7)) << 4), x + (size_t)min(tok, m - 1) * K + k0 + c * 8, tok < m ? 16u : 0u);
    }
    for (int i = tid; i < BN * CS; i += 256) {
      const int r = i / CS, c = s * CS + i - (i / CS) * CS;
      const size_t o = (size_t)(n0 + r) * K + k0 + c * 8;
      const uint32_t d = r * RB + ((c ^ (r & 7)) << 4);
      rs_cpa(sh + d, hi + o, 16u); rs_cpa(sl + d, lo + o, 16u);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  }
  const int wt = w & 3, wn = w >> 2;
  float acc[MI][NT8][4];
#pragma unroll
  for (int i = 0; i < MI; i++)
#pragma unroll
    for (int j = 0; j < NT8; j++) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
  const int ar = lane & 15, ac = lane >> 4;
  const int bj = lane >> 3, br = (bj >> 1) * 8 + (lane & 7), bc = bj & 1;
#pragma unroll
  for (int s = 0; s < NSTG; s++) {
    if (s == 0) asm volatile("cp.async.wait_group %0;\n" :: "n"(NSTG - 1) : "memory");
    else if (s == 1) asm volatile("cp.async.wait_group %0;\n" :: "n"(NSTG > 1 ? NSTG - 2 : 0) : "memory");
    else if (s == 2) asm volatile("cp.async.wait_group %0;\n" :: "n"(NSTG > 2 ? NSTG - 3 : 0) : "memory");
    else asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();
#pragma unroll 2
    for (int qq = 0; qq < QS; qq++) {
      const int q = s * QS + qq;
      uint32_t a[MI][4];
#pragma unroll
      for (int mi = 0; mi < MI; mi++) {  // warp wt owns tokens 32 wt .. of each 128-token block: mi = (block, m16)
        const int r = (mi >> 1) * 128 + wt * 32 + (mi & 1) * 16 + ar, c = 2 * q + ac;
        rs_ldm4(sx + r * RB + ((c ^ (r & 7)) << 4), a[mi]);
      }
#pragma unroll
      for (int h = 0; h < 2; h++)
#pragma unroll
        for (int nb = 0; nb < NT8 / 2; nb++) {
          const int r = wn * WN + nb * 16 + br, c = 2 * q + bc;
          uint32_t b[4];
          rs_ldm4((h ? sl : sh) + r * RB + ((c ^ (r & 7)) << 4), b);
#pragma unroll
          for (int mi = 0; mi < MI; mi++) {
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
              : "+f"(acc[mi][2 * nb][0]), "+f"(acc[mi][2 * nb][1]), "+f"(acc[mi][2 * nb][2]), "+f"(acc[mi][2 * nb][3])
              : "r"(a[mi][0]), "r"(a[mi][1]), "r"(a[mi][2]), "r"(a[mi][3]), "r"(b[0]), "r"(b[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
              : "+f"(acc[mi][2 * nb + 1][0]), "+f"(acc[mi][2 * nb + 1][1]), "+f"(acc[mi][2 * nb + 1][2]), "+f"(acc[mi][2 * nb + 1][3])
              : "r"(a[mi][0]), "r"(a[mi][1]), "r"(a[mi][2]), "r"(a[mi][3]), "r"(b[2]), "r"(b[3]));
          }
        }
    }
  }
  float* dst = RED == 0 ? out : ws;
#pragma unroll
  for (int mi = 0; mi < MI; mi++)
#pragma unroll
    for (int j = 0; j < NT8; j++) {
      const int tok = t0 + (mi >> 1) * 128 + wt * 32 + (mi & 1) * 16 + g, n = n0 + wn * WN + j * 8 + 2 * t;
      if (tok < m) rs_red2(dst + (size_t)tok * N + n, acc[mi][j][0], acc[mi][j][1]);
      if (tok + 8 < m) rs_red2(dst + (size_t)(tok + 8) * N + n, acc[mi][j][2], acc[mi][j][3]);
    }
  if constexpr (RED == 2) {  // the CTA completing this (token block, n tile) copies it to out and re-zeroes it
    __threadfence();
    __syncthreads();
    __shared__ int last;
    if (tid == 0) last = atomicAdd(cnt + tb * NTL + nt, 1) == KS - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    const int rows = min(TOKS, m - t0);
    for (int i = tid; i < rows * (BN / 4); i += 256) {
      const int r = i / (BN / 4), c4 = i - r * (BN / 4);
      float4* src = (float4*)(ws + (size_t)(t0 + r) * N + n0) + c4;
      const float4 v = __ldcg(src);
      __stcg(src, make_float4(0.f, 0.f, 0.f, 0.f));
      ((float4*)(out + (size_t)(t0 + r) * N + n0))[c4] = v;
    }
    if (tid == 0) cnt[tb * NTL + nt] = 0;
  }
}
template <int BN, int KR, int NSTG, int TOKS, int RED, int MINB>
static int rs2_go(const __nv_bfloat16* x, const __nv_bfloat16* hi, const __nv_bfloat16* lo, float* out, float* ws, long long wsn, int* cnt, int cntn,
                  int m, int N, int K, cudaStream_t st) {
  constexpr int SM = (TOKS + 2 * BN) * KR * 2;
  if (N % BN || K % KR || m < 1) return -4;
  const int TB = (m + TOKS - 1) / TOKS, KS = K / KR, NTL = N / BN;
  if (RED == 2 && ((long long)TB * TOKS * N > wsn || TB * NTL > cntn)) return -5;
  static bool once = false;
  if (!once) {
    if (cudaFuncSetAttribute(rsk2<BN, KR, NSTG, TOKS, RED, MINB>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM) != cudaSuccess) { cudaGetLastError(); return -6; }
    once = true;
  }
  rsk2<BN, KR, NSTG, TOKS, RED, MINB><<<dim3(NTL, KS, TB), 256, SM, st>>>(x, hi, lo, out, ws, cnt, m, N, K);
  return (int)cudaGetLastError();
}
// variant: 0 = pw22's rs64k32 (in-run reference); 5-11 = rsk2 (see RSP in the Python part)
template <int BN, int KR, int RED>
static int rs_go(const __nv_bfloat16* x, const __nv_bfloat16* hi, const __nv_bfloat16* lo, float* out, float4* ws, long long wsn, int* cnt, int cntn,
                 int m, int N, int K, cudaStream_t st) {
  constexpr int SM = (128 + 2 * BN) * KR * 2;
  if (N % BN || K % KR || m < 1) return -4;
  const int TB = (m + 127) / 128, KS = K / KR, NTL = N / BN;
  if (RED == 1 && ((long long)TB * NTL * KS * (BN / 16) * 2 * 32 * 8 > wsn || TB * NTL > cntn)) return -5;
  static bool once = false;
  if (!once) {
    if (cudaFuncSetAttribute(rsk<BN, KR, RED>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM) != cudaSuccess) { cudaGetLastError(); return -6; }
    once = true;
  }
  rsk<BN, KR, RED><<<dim3(NTL, KS, TB), 256, SM, st>>>(x, hi, lo, out, ws, cnt, m, N, K);
  return (int)cudaGetLastError();
}
int rs_launch(int v, const __nv_bfloat16* x, const __nv_bfloat16* hi, const __nv_bfloat16* lo, float* out, float4* ws, long long wsn, int* cnt, int cntn,
              int m, int N, int K, cudaStream_t st) {
  switch (v) {
    case 0: return rs_go<64, 256, 0>(x, hi, lo, out, ws, wsn, cnt, cntn, m, N, K, st);
    case 5: return rs2_go<64, 256, 4, 128, 0, 1>(x, hi, lo, out, (float*)ws, wsn * 4, cnt, cntn, m, N, K, st);
    case 6: return rs2_go<64, 256, 2, 128, 0, 1>(x, hi, lo, out, (float*)ws, wsn * 4, cnt, cntn, m, N, K, st);
    case 7: return rs2_go<64, 128, 2, 128, 0, 2>(x, hi, lo, out, (float*)ws, wsn * 4, cnt, cntn, m, N, K, st);
    case 8: return rs2_go<64, 256, 4, 128, 2, 1>(x, hi, lo, out, (float*)ws, wsn * 4, cnt, cntn, m, N, K, st);
    case 9: return rs2_go<64, 256, 4, 256, 0, 1>(x, hi, lo, out, (float*)ws, wsn * 4, cnt, cntn, m, N, K, st);
    case 10: return rs2_go<64, 128, 2, 256, 0, 2>(x, hi, lo, out, (float*)ws, wsn * 4, cnt, cntn, m, N, K, st);
    case 11: return rs2_go<64, 256, 4, 256, 2, 1>(x, hi, lo, out, (float*)ws, wsn * 4, cnt, cntn, m, N, K, st);
  }
  return -2;
}
// ---------------------------------------------------------------- pw27 PROBE ladder for the 6-8 MB regime (no paths)
// pr_ldg: G CTAs x 256 threads, every 16-B load of the CTA's share issued at launch (NL per thread, ld.global.nc
// .L1::no_allocate), then RED 0: nothing; 1: 64 fp32 atomicAdds per CTA into a zeroed 8 KB output; 2: cluster-8 DSMEM
// sum of 64 floats per CTA, rank 0 writes. pr_blk: G CTAs, one lane issues the CTA's share as CHK-byte cp.async.bulk
// copies on one mbarrier at launch, then waits.
template <int NL, int RED>
__global__ void __launch_bounds__(256) pr_ldg(const uint4* buf, long long n16, float* out, uint32_t* sink) {
  const long long T = (long long)gridDim.x * 256, t = (long long)blockIdx.x * 256 + threadIdx.x;
  uint4 v[NL];
#pragma unroll
  for (int i = 0; i < NL; i++) { const long long j = t + (long long)i * T; v[i] = j < n16 ? gnc(buf + j) : make_uint4(0u, 0u, 0u, 0u); }
  uint32_t a = 0;
#pragma unroll
  for (int i = 0; i < NL; i++) a ^= v[i].x ^ v[i].y ^ v[i].z ^ v[i].w;
  if constexpr (RED == 1) {
    __shared__ uint32_t sa[256];
    sa[threadIdx.x] = a;
    __syncthreads();
    if (threadIdx.x < 64) atomicAdd(out + (blockIdx.x * 64 + threadIdx.x) % 2048, (float)((sa[threadIdx.x] ^ sa[threadIdx.x + 64]) & 1u));
  } else if constexpr (RED == 2) {
    __shared__ float red[64];
    __shared__ uint32_t sa[256];
    sa[threadIdx.x] = a;
    __syncthreads();
    if (threadIdx.x < 64) red[threadIdx.x] = (float)((sa[threadIdx.x] ^ sa[threadIdx.x + 64]) & 1u);
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
    uint32_t rk; asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rk));
    if (rk == 0 && threadIdx.x < 64) {
      float s = 0.f;
      const uint32_t la = (uint32_t)__cvta_generic_to_shared(red + threadIdx.x);
#pragma unroll
      for (int q = 0; q < 8; q++) {
        uint32_t ra; float f;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(ra) : "r"(la), "r"(q));
        asm volatile("ld.shared::cluster.f32 %0, [%1];\n" : "=f"(f) : "r"(ra) : "memory");
        s += f;
      }
      out[(blockIdx.x / 8 * 64 + threadIdx.x) % 2048] = s;
    }
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
  }
  if (a == 0x9e3779b9u) sink[blockIdx.x] = a;
}
template <int CHK>
__global__ void __launch_bounds__(32) pr_blk(const uint8_t* buf, long long per, uint32_t* sink) {
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem), bar = sb + (uint32_t)per;
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(bar));
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "r"((uint32_t)per) : "memory");
    for (long long o = 0; o < per; o += CHK)
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                   :: "r"(sb + (uint32_t)o), "l"(buf + blockIdx.x * per + o), "r"(CHK), "r"(bar) : "memory");
  }
  __syncwarp();
  asm volatile("{\n.reg .pred p;\nPRB_W:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n@!p bra PRB_W;\n}\n" :: "r"(bar) : "memory");
  uint32_t a; asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(a) : "r"(sb + threadIdx.x * 4));
  if (a == 0x9e3779b9u) sink[blockIdx.x] = a;
}
// kind: 0-3 ldg NL 16 / 8 / 4 / 2 (no reduction), 4 ldg NL 8 + red, 5 ldg NL 8 + cluster-8 DSMEM, 6 bulk 8 KB, 7 bulk 16 KB
int pr_launch(int kind, int G, const void* buf, long long bytes, float* out, uint32_t* sink, cudaStream_t st) {
  const long long n16 = bytes / 16;
  switch (kind) {
    case 0: pr_ldg<16, 0><<<G, 256, 0, st>>>((const uint4*)buf, n16, out, sink); break;
    case 1: pr_ldg<8, 0><<<G, 256, 0, st>>>((const uint4*)buf, n16, out, sink); break;
    case 2: pr_ldg<4, 0><<<G, 256, 0, st>>>((const uint4*)buf, n16, out, sink); break;
    case 3: pr_ldg<2, 0><<<G, 256, 0, st>>>((const uint4*)buf, n16, out, sink); break;
    case 4: pr_ldg<8, 1><<<G, 256, 0, st>>>((const uint4*)buf, n16, out, sink); break;
    case 5: {
      cudaLaunchConfig_t cfg = {}; cfg.gridDim = dim3(G); cfg.blockDim = dim3(256); cfg.stream = st;
      cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeClusterDimension;
      at[0].val.clusterDim.x = 8; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1; cfg.attrs = at; cfg.numAttrs = 1;
      if (cudaLaunchKernelEx(&cfg, pr_ldg<8, 2>, (const uint4*)buf, n16, out, sink) != cudaSuccess) { cudaGetLastError(); return -3; }
      break;
    }
    case 6: case 7: {
      const long long per = bytes / G;
      const int chk = kind == 6 ? 8192 : 16384;
      if (per % chk || per > 200 * 1024) return -4;
      auto k = kind == 6 ? pr_blk<8192> : pr_blk<16384>;
      if (cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)per + 64) != cudaSuccess) { cudaGetLastError(); return -6; }
      k<<<G, 32, (int)per + 64, st>>>((const uint8_t*)buf, per, sink);
      break;
    }
    default: return -2;
  }
  return (int)cudaGetLastError();
}
int64_t rrrun(int64_t inst, torch::Tensor W, torch::Tensor x, torch::Tensor out) {
  const int m = x.size(0), K = x.size(1), N = out.size(1);
  auto st = at::cuda::getCurrentCUDAStream();
  const float4* w = (const float4*)W.data_ptr(); const __nv_bfloat16* xp = (const __nv_bfloat16*)x.data_ptr(); float* o = out.data_ptr<float>();
  if (K != 8192) return -1;
  switch (inst) {
    case 0: if (m > 1) return -1; rr<1, 1><<<N, 512, 0, st>>>(w, xp, o, m, N, K); break;
    case 1: if (m > 1) return -1; rr<1, 2><<<N / 2, 512, 0, st>>>(w, xp, o, m, N, K); break;
    case 2: if (m > 8) return -1; rr<8, 2><<<N / 2, 512, 0, st>>>(w, xp, o, m, N, K); break;
    case 3: if (m > 8) return -1; rr<8, 4><<<N / 4, 512, 0, st>>>(w, xp, o, m, N, K); break;
    case 4: if (m > 1) return -1; rr<1, 1, 256><<<N, 256, 0, st>>>(w, xp, o, m, N, K); break;
    case 5: if (m > 1) return -1; rr<1, 2, 256><<<N / 2, 256, 0, st>>>(w, xp, o, m, N, K); break;
    case 6: if (m > 1) return -1; rr<1, 1, 1024><<<N, 1024, 0, st>>>(w, xp, o, m, N, K); break;
    case 7: if (m > 1) return -1; rr<1, 2, 1024><<<N / 2, 1024, 0, st>>>(w, xp, o, m, N, K); break;
    default: return -2;
  }
  const cudaError_t r = cudaGetLastError();
  return (int64_t)r;
}

static int64_t gvgo(int64_t inst, torch::Tensor W, torch::Tensor W2, torch::Tensor x, torch::Tensor out, torch::Tensor esc, torch::Tensor ws, torch::Tensor cnt, int64_t T, int64_t U, int O) {
  G p;
  p.W = (const uint4*)W.data_ptr(); p.W2 = (const uint4*)W2.data_ptr(); p.x = (const __nv_bfloat16*)x.data_ptr(); p.out = out.data_ptr();
  p.esc = (const uint32_t*)esc.data_ptr(); p.ws = (float4*)ws.data_ptr(); p.cnt = cnt.data_ptr<int>();
  p.m = x.size(0); p.K = x.size(1); p.N = out.size(1); p.KC = p.K / 64;
  p.T0 = (uint32_t)(T & 0xffffffff); p.T1 = (uint32_t)((uint64_t)T >> 32); p.U0 = (uint32_t)(U & 0xffffffff); p.U1 = (uint32_t)((uint64_t)U >> 32);
  auto st = at::cuda::getCurrentCUDAStream();
  const int occ = O;
  switch (inst) {
@@CASES@@
  }
  return -2;
}
int64_t gvrun(int64_t inst, torch::Tensor W, torch::Tensor W2, torch::Tensor x, torch::Tensor out, torch::Tensor esc, torch::Tensor ws, torch::Tensor cnt, int64_t T, int64_t U) {
  return gvgo(inst, W, W2, x, out, esc, ws, cnt, T, U, 0);
}
int64_t gvocc(int64_t inst, torch::Tensor W, torch::Tensor W2, torch::Tensor x, torch::Tensor out, torch::Tensor esc, torch::Tensor ws, torch::Tensor cnt, int64_t T, int64_t U) {
  return gvgo(inst, W, W2, x, out, esc, ws, cnt, T, U, 1);
}
int64_t prrun(int64_t kind, int64_t G, torch::Tensor buf, torch::Tensor out, torch::Tensor sink) {
  return (int64_t)pr_launch((int)kind, (int)G, buf.data_ptr(), (long long)buf.numel() * buf.element_size(), out.data_ptr<float>(),
                            (uint32_t*)sink.data_ptr(), at::cuda::getCurrentCUDAStream());
}
int64_t rsrun(int64_t v, torch::Tensor x, torch::Tensor hi, torch::Tensor lo, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt) {
  if (x.size(1) != hi.size(1) || out.size(0) != x.size(0) || out.size(1) != hi.size(0)) return -10;
  return (int64_t)rs_launch((int)v, (const __nv_bfloat16*)x.data_ptr(), (const __nv_bfloat16*)hi.data_ptr(), (const __nv_bfloat16*)lo.data_ptr(),
                            out.data_ptr<float>(), (float4*)ws.data_ptr(), (long long)(ws.numel() / 4), cnt.data_ptr<int>(), (int)cnt.numel(),
                            (int)x.size(0), (int)out.size(1), (int)x.size(1), at::cuda::getCurrentCUDAStream());
}
int64_t gvregs(int64_t inst, torch::Tensor W, torch::Tensor W2, torch::Tensor x, torch::Tensor out, torch::Tensor esc, torch::Tensor ws, torch::Tensor cnt, int64_t T, int64_t U) {
  return gvgo(inst, W, W2, x, out, esc, ws, cnt, T, U, 2);
}
"""
    # candidate launch geometries (WR, WK, KU, S, RT, XS) per (layer key, MT); FMT is c12 (1/3) with a bf16 (0/2) twin of the
    # first geometry as the fallback when a 12-bit pack fails
    # candidates per (layer key, MT): (WR, WK, KU, S, RT, XS) -> kernel variants V (bit 0: 9-op add decode, else 11-op table
    # decode; bit 1: rolled K loop); H200 v15 judge: the v14 kernel (V 0) was faster at deep K, the rolled/9-op one (V 3) at m32
    _0, _1, _01 = (0,), (1,), (0, 1)
    GVC = {  # pw52 (wf_stack3): CONSERVATIVE bake. The proven pick (pw49s's judge run, job faf47019b8fc40fa, scored
        # 3155.7 = the best official draw) is listed FIRST in every group and is the baseline the pick rule compares
        # against (GVMARGIN 0.97 at m <= 32): a variant replaces it only if it is >= 3% faster in this bake. FACTS 14: the
        # judge's sub-1% in-bake flips measured 0.2-2.2 us WORSE officially, so the entries pw51s's judge bake (job
        # a5a8187115774027, scored 3222.0) preferred by < 3% are out of the lists (and out of the build).
        #   dropped vs pw51s: o m1/m8 ST 3 (1.5%/0.7%), o m1/m8 minBlocks 2 (1.5%/worse), o m16 minBlocks 2 (0.9%),
        #   qkv m1 V3 (worse) and XS 0 on the winner (worse), unpooler m16 WR 2 (0.2%) and minBlocks 2 (worse),
        #   unpooler_s ST 3 (2.6% -> ST < 4 at small m is the FACTS-14 loser).
        #   kept: unpooler_s XS 2 (judge 3.17/3.04 vs proven 3.42/3.26 = 7.3%/6.7% -> the one sweep entry that clears 3%).
        #   reordered: the proven entry of each group is first (v1 before v0 where the proven pick is the v1 instance).
        ("qkv_gate", 1): [((2, 16, 8, 1, 1, 2), _1), ((1, 8, 16, 1, 1, 0), _1), ((1, 16, 8, 1, 1, 2, 1), (1 | 2 << 12,)),
                          ((1, 4, 16, 2, 1, 0), _01), ((1, 4, 8, 4, 1, 0), _1)],
        ("qkv_gate", 2): [((2, 8, 16, 1, 1, 0, 2), _1), ((1, 4, 16, 2, 1, 0), _1), ((1, 4, 8, 4, 1, 2), (1, 3)),
                          ((1, 4, 16, 2, 1, 2), _01), ((1, 4, 16, 2, 1, 2, 1), _1), ((1, 4, 8, 4, 1, 2, 1), _1),
                          ((2, 16, 8, 1, 1, 2), (3,)), ((1, 16, 8, 1, 1, 2, 1), (1 | 2 << 12,))],
        ("qkv_gate", 4): [((1, 8, 16, 1, 1, 2), _01), ((1, 4, 16, 2, 1, 2), _01), ((2, 4, 16, 2, 1, 2), _1),
                          ((1, 4, 8, 4, 1, 2, 1), _01), ((1, 4, 8, 4, 1, 2), _1), ((1, 4, 16, 2, 1, 2, 1), _1)],
        ("o", 1): [((1, 8, 2, 1, 1, 0), _1), ((1, 8, 2, 1, 1, 0), _0), ((2, 4, 4, 1, 1, 0), _1), ((1, 4, 4, 1, 1, 0), _1)],
        ("o", 2): [((1, 8, 2, 1, 1, 0), _1), ((1, 8, 2, 1, 1, 0), _0), ((1, 8, 2, 1, 1, 2), _1), ((2, 4, 4, 1, 1, 0), _1),
                   ((2, 4, 4, 1, 1, 2), _1), ((1, 4, 4, 1, 1, 2, 1), _1)],
        ("o", 4): [((1, 4, 4, 1, 1, 2), _01), ((2, 4, 4, 1, 1, 2), _01), ((1, 8, 2, 1, 1, 0), _1),
                   ((1, 4, 4, 1, 1, 2, 1), _01), ((1, 8, 2, 1, 1, 2), _1), ((1, 8, 2, 1, 1, 2, 1), _1)],
        ("pooler", 1): [((2, 16, 8, 1, 1, 0), (3,)), ((2, 8, 16, 1, 1, 0), _1), ((2, 8, 16, 1, 1, 0), _0),
                        ((1, 8, 16, 1, 1, 0), _1)],
        ("pooler", 2): [((2, 8, 16, 1, 1, 0), _1), ((2, 8, 16, 1, 1, 0), _0), ((1, 8, 16, 1, 1, 2), _1),
                        ((1, 8, 16, 1, 1, 0), _1)],
        ("pooler", 4): [((1, 8, 16, 1, 1, 2), _0), ((1, 8, 16, 1, 1, 2), _1), ((1, 4, 16, 2, 1, 2), _1),
                        ((2, 4, 16, 2, 1, 2), _1)],
        # unpooler_s serves m=1 AND m=8: proven = v0 at m=1, v1 at m=8 (one list, so only v0 can be literally first;
        # the pick rule matches the proven entry by NAME, so both hold)
        ("unpooler_s", 1): [((1, 16, 4, 1, 1, 0), _0), ((1, 16, 4, 1, 1, 0), _1), ((1, 8, 2, 4, 1, 0), _1),
                            ((1, 8, 4, 2, 1, 0), _1), ((1, 16, 4, 1, 1, 2), _1)],
        ("unpooler", 2): [((1, 4, 16, 1, 1, 2), _1), ((1, 4, 16, 1, 1, 2), _0), ((1, 4, 16, 1, 1, 0), _1),
                          ((1, 8, 8, 1, 1, 0), _1)],
        ("unpooler", 4): [((1, 4, 16, 1, 1, 2), _1), ((1, 4, 16, 1, 1, 2), _0), ((2, 4, 16, 1, 1, 2), _1),
                          ((1, 8, 8, 1, 1, 2), _1)],
        # router: plain BF16 hi + lo fragments (FMT 2), every warp's units in flight at once, split-K over a 16-CTA cluster
        ("router", 1): [((1, 8, 2, 8, 1, 0), _0), ((1, 8, 1, 16, 1, 0), _0), ((1, 4, 2, 16, 1, 0), _0)],
        # pw57 (wf_jsweep5): + pw56's wf_jsweep4 entries (red.add = no 8-CTA cluster, minBlocks 2, WK8/KU4/S4,
        # WR2 512 thr) -- all within +-1.5% on the box, so only a fixed-latency effect can separate them.
        ("router", 2): [((1, 8, 2, 8, 1, 0), _0), ((1, 8, 1, 16, 1, 0), _0), ((1, 4, 2, 16, 1, 0), _0),
                        ((1, 8, 2, 8, 1, 0), (1 << 26,)), ((1, 8, 2, 8, 1, 0), (1 << 13,)),
                        ((1, 8, 4, 4, 1, 0), _0), ((2, 8, 2, 8, 1, 0), _0)],
        ("router", 4): [((1, 4, 2, 16, 1, 0), _0), ((1, 8, 2, 8, 1, 0), _0), ((1, 8, 1, 16, 1, 0), _0),
                        ((1, 4, 2, 16, 1, 0), (1 << 26,)), ((1, 4, 4, 8, 1, 0), _0)],
    }
    # pw52: the proven pick per bake group = the entry pw49s's judge run (faf47019b8fc40fa) picked. The pick rule keeps it
    # unless another entry is GVMARGIN faster in this bake; "v12" = the r6 path (no gv pick, unchanged).
    PROVEN_GV = {
        ("qkv_gate", 1): "1.1.1.2.16.8.1.1.2.1", ("qkv_gate", 8): "1.1.1.2.16.8.1.1.2.1",
        ("qkv_gate", 16): "1.2.2.2.8.16.1.1.0.1", ("qkv_gate", 32): "v12",
        ("o", 1): "1.1.2.1.8.2.1.1.0.1", ("o", 8): "1.1.2.1.8.2.1.1.0.1", ("o", 16): "1.2.2.1.8.2.1.1.0.1",
        ("o", 32): "v12",
        ("pooler", 1): "1.1.1.2.16.8.1.1.0.3", ("pooler", 8): "1.1.1.2.16.8.1.1.0.3",
        ("pooler", 16): "1.2.2.2.8.16.1.1.0.1", ("pooler", 32): "1.4.2.1.8.16.1.1.2.0",
        ("unpooler", 1): "1.1.2.1.16.4.1.1.0.0", ("unpooler", 8): "1.1.2.1.16.4.1.1.0.1",
        ("unpooler", 16): "1.2.2.1.4.16.1.1.2.1", ("unpooler", 32): "1.4.2.1.4.16.1.1.2.1",
        ("router", 1): "rr6", ("router", 8): "2.1.2.1.8.2.8.1.0.0", ("router", 16): "2.2.2.1.8.2.8.1.0.0",
        ("router", 32): "2.4.2.1.4.2.16.1.0.0",
    }
    # pw52: the proven pick per gx group (unpooler m1/m8 = uk-s4x); every other gx group's proven pick is r6 (no gx pick).
    FORCE_GX = {("unpooler", 8)}
    PROVEN_GX = {("unpooler", 1): "uk-s4x", ("unpooler", 8): "uk-s4x",
                 ("router", 8): "rk-s16w8x", ("router", 16): "rk-s16w8z", ("router", 32): "rk-s8z"}
    # pw52: the proven pick per rq32 group (pw49s's judge run); a group missing here was picked as r6 (no rq path).
    PROVEN_RQ32 = {("qkv_gate", 32): "t32s6hze-k7", ("qkv_gate", 16): "t16s4hze-k7", ("o", 32): "t32s6hze-k2",
                   ("pooler", 32): "t32s7hze-k4", ("unpooler", 32): "t32s6hze-k2"}
    GVRB = []  # [(128, 4), (256, 4)]
    GVB = {1: 1, 8: 1, 16: 2, 32: 4}  # bucket m -> MT (the kernel also runs m > 8 MT as token blocks in grid.x; slower than cuBLAS)


    def _gvst(geo):  # a 7th geometry field forces the register-ring depth
        if len(geo) == 7:
            return geo[6]
        wr, wk, ku, s, rt, xs = geo
        return 1 if wr * wk >= 32 else 3 if xs == 1 and rt == 1 else 2


    def _gvinst():
        out = []
        for (key, mt), geos in GVC.items():
            for k, (geo, vs) in enumerate(geos):
                if key == "router":
                    out.append((2, mt, _gvst(geo)) + geo[:6] + (0,))
                    continue
                out += [(1, mt, _gvst(geo)) + geo[:6] + (v,) for v in vs] + ([(0, mt, _gvst(geo)) + geo[:6] + (0,)] if k == 0 else [])
        return list(dict.fromkeys(out))


    # pw11 paths: (name, layer, buckets, (FMT, WR, WK, KU, S, RT, XS), V). V 1 = 9-op decode with r6's epilogue (S > 1:
    # cluster DSMEM), V 5 = same kernel with the global last-arriver reduction (no cluster). Rows per CTA = 16 WR RT.
    GX = True
    GXP = []
    GX11 = False  # pw17: pw11's wide-row gx paths never won (checks 824f32ce, f1e6a186, f03e2e77); not built
    for _ms, _xs in (((1, 8), 0), ((16, 32), 2)):
        for _wr in (9, 8):  # 144 rows: 16 x 8 = 128 CTAs; 128 rows: 2304 = 18 x 128 -> 144 CTAs
            for _ab, _v in (("a", 1), ("b", 5)):
                if _ab == "a":  # pw12 trim: q144a (cluster 8 declined on the judge, 120 < 128) and q128a dropped
                    continue
                _ms2 = tuple(m_ for m_ in _ms if not (_wr == 8 and m_ == 32))  # pw12 trim: q128b m=32 (L1-x, MT 4)
                if _ms2:
                    GXP.append(("gx-q%d%s" % (16 * _wr, _ab), "qkv_gate", _ms2, (1, _wr, 1, 16, 8, 1, _xs), _v))
    for _ab, _v in (("a", 1), ("b", 5)):
        if _ab == "b":  # pw12 trim: gx-q144a-s dropped (cluster 8 declined)
            GXP.append(("gx-q144%s-s" % _ab, "qkv_gate", (32,), (1, 9, 1, 16, 8, 1, 1), _v))
        GXP.append(("gx-p128%s" % _ab, "pooler", (16,), (1, 8, 2, 16, 4, 1, 2), _v))  # pw12 trim: m=32 L1-x (8 B spill) dropped
        GXP.append(("gx-p128%s-s" % _ab, "pooler", (32,), (1, 8, 2, 16, 4, 1, 1), _v))
        GXP.append(("gx-u128%s" % _ab, "unpooler", (16,), (1, 8, 2, 16, 2, 1, 2), _v))  # pw12 trim: m=32 L1-x dropped
        GXP.append(("gx-u128%s-s" % _ab, "unpooler", (32,), (1, 8, 2, 16, 2, 1, 1), _v))
    GXP.append(("gx-o64-s", "o", (16, 32), (1, 4, 4, 4, 1, 1, 1), 1))
    GXP.append(("gx-r12g8", "router", (1, 8, 16, 32), (3, 1, 8, 2, 8, 1, 0), 5))  # pw12 trim: r12c8 / r12c16 dropped
    if not GX11:
        GXP = []


    # pw12: the same kernels with W through a per-warp cp.async.bulk smem ring (V bit 3; slots 4/8; chunks per slot 1/2).
    # Geometries: r6's judge-picked one per (layer, bucket) (job 23cecdef1bb54c9e), r6's 16-warp pooler alternative, and
    # pw11's wide-row global-reduction geometries. An 8th geo field forces r6's register x-ring depth ST.
    def _bv(v, ns, cu=1, fence=False):
        return v | 8 | ({2: 0, 4: 1, 8: 2, 16: 3}[ns] << 4) | ({1: 0, 2: 1, 4: 2}[cu] << 6) | (0 if fence else 256)


    GB = True
    for _nm, _ly, _ms, _g, _v, _vars in (
            ("q", "qkv_gate", (1, 8), (1, 1, 8, 16, 1, 1, 0), 1, ((4, 1), (8, 1))),
            ("q144b", "qkv_gate", (1, 8), (1, 9, 1, 16, 8, 1, 0), 5, ((8, 1),)),
            ("q", "qkv_gate", (16,), (1, 1, 4, 8, 4, 1, 2, 1), 1, ((4, 1), (8, 1))),
            ("q144b", "qkv_gate", (16, 32), (1, 9, 1, 16, 8, 1, 2), 5, ((8, 1),)),
            ("q", "qkv_gate", (32,), (1, 1, 4, 16, 2, 1, 2, 1), 1, ((4, 1), (8, 1))),
            ("o", "o", (1, 8), (1, 1, 8, 2, 1, 1, 0), 0, ((4, 1),)),
            ("o", "o", (16,), (1, 1, 8, 2, 1, 1, 0), 1, ((4, 1),)),
            ("o64-s", "o", (16, 32), (1, 4, 4, 4, 1, 1, 1), 1, ((4, 1),)),
            ("p", "pooler", (1, 8), (1, 2, 16, 8, 1, 1, 0), 3, ((4, 1),)),
            ("p16w", "pooler", (1, 8, 16), (1, 2, 8, 16, 1, 1, 0), 1, ((4, 1), (8, 1), (4, 2))),
            ("p", "pooler", (32,), (1, 1, 8, 16, 1, 1, 2), 1, ((4, 1), (8, 1))),
            ("p128b", "pooler", (16, 32), (1, 8, 2, 16, 4, 1, 2), 5, ((4, 1),)),
            ("u", "unpooler", (1, 8), (1, 1, 16, 4, 1, 1, 0), 1, ((4, 1),)),
            ("u", "unpooler", (16,), (1, 1, 4, 16, 1, 1, 2), 1, ((4, 1), (8, 1))),
            ("u", "unpooler", (32,), (1, 1, 4, 16, 1, 1, 2), 0, ((4, 1), (8, 1))),
            ("u128b", "unpooler", (16, 32), (1, 8, 2, 16, 2, 1, 2), 5, ((4, 1),)),
            ("r", "router", (1, 8, 16), (2, 1, 8, 2, 8, 1, 0), 0, ((4, 1),)),
            ("r", "router", (32,), (2, 1, 4, 2, 16, 1, 0), 0, ((4, 1),)),
            ("r12g8", "router", (1, 8, 16, 32), (3, 1, 8, 2, 8, 1, 0), 5, ((4, 1),))):
        for _ns, _cu in _vars if GB else ():
            if True:  # pw21: no bulk-ring paths (gb-q-n4 never won)
                continue
            GXP.append(("gb-%s-n%d%s" % (_nm, _ns, "c%d" % _cu if _cu > 1 else ""), _ly, _ms, _g, _bv(_v, _ns, _cu)))


    # pw18: codec E v3 paths (FMT 4) on pw17's geometries (r6's judge-picked qkv 1.8.16, r6's 512-thread pooler 2.8.16,
    # unpooler 1.4.16; qkv m=16 on 1.8.16 since pw17's 1.4.8 S4 cluster was declined on occupancy). V: base | bits 22-23 =
    # log2 substreams (independent decode chains per lane) | bit 20 = LUT replicated per lane (32 KB, conflict-free) | bit 19
    # = 32-word ring (not baked). Per (layer, m): K1 single LUT, K1 / K2 / K4 replicated. m=32, o and the router: r6.
    GE = False  # pw21: codec E closed (pw17-pw20); r6 + router m=128/256 paths only
    _ER, _EK2, _EK4 = 1 << 20, 1 << 22, 2 << 22
    _GEG = {("qkv_gate", 1): (4, 1, 8, 16, 1, 1, 0), ("qkv_gate", 8): (4, 1, 8, 16, 1, 1, 0), ("qkv_gate", 16): (4, 1, 8, 16, 1, 1, 0),
            ("pooler", 1): (4, 2, 8, 16, 1, 1, 0), ("pooler", 8): (4, 2, 8, 16, 1, 1, 0), ("pooler", 16): (4, 2, 8, 16, 1, 1, 0),
            ("unpooler", 1): (4, 1, 4, 16, 1, 1, 0), ("unpooler", 8): (4, 1, 4, 16, 1, 1, 0), ("unpooler", 16): (4, 1, 4, 16, 1, 1, 2)}
    _GEV = (("", 0), ("r", _ER), ("r-k2", _ER | _EK2), ("r-k4", _ER | _EK4))
    _GEN = {"qkv_gate": "q", "pooler": "p512", "unpooler": "u"}
    # qkv (256 threads) carries the __launch_bounds__ minBlocks 2 override (V bits 12-14): r6's 144 CTAs need 2 per SM to stay in
    # one wave on 132 SMs, which caps registers at 128. qkv and pooler m=16 (MT 2) K2 / K4 spill at 128 and are not built.
    GEP = [("ge-%s%s" % (_GEN[ly] if not (ly == "unpooler" and m <= 8) else "u16", sfx), ly, (m,), g, 1 | b | (2 << 12 if ly == "qkv_gate" else 0))
           for (ly, m), g in _GEG.items() for sfx, b in _GEV if not (ly != "unpooler" and m == 16 and b & (3 << 22))] if GE else []
    # pw20 "lite" codec E paths at m = 1 / 8 (V bit 24): one substream, single LUT, plane ring ST 1 (8th geometry field), rolled
    # K loop, one accumulator set, each unit decoded and consumed one k16 step at a time (4 A registers live): 64 registers,
    # 0 spills (ptxas, sm_90a), so 1024 threads run per SM like r6's own m=1 geometry. __launch_bounds__ minBlocks (V bits
    # 12-14) = 1024 / CTA threads. Geometries: pooler 2.16.8 (r6's m=1 geometry, 128 CTAs x 1024) and 1.16.8 (256 x 512, 2/SM);
    # qkv 1.16.8 (144 x 512) and 1.8.16 (r6's, 144 x 256, up to 4/SM); unpooler (n = 1024) 1.8.8 (64 x 256).
    _EL = 1 << 24
    GEPL = [("ge-l%s" % nm, ly, (m,), g, 1 | 2 | ((1024 // nt) << 12) | _EL)
            for nm, ly, g, nt in (("p1k", "pooler", (4, 2, 16, 8, 1, 1, 0, 1), 1024), ("p16", "pooler", (4, 1, 16, 8, 1, 1, 0, 1), 512),
                                  ("q16", "qkv_gate", (4, 1, 16, 8, 1, 1, 0, 1), 512), ("q8", "qkv_gate", (4, 1, 8, 16, 1, 1, 0, 1), 256),
                                  ("u8", "unpooler", (4, 1, 8, 8, 1, 1, 0, 1), 256))
            for m in (1, 8)] if GE else []
    GEP += GEPL
    GXP += GEP

    # pw27 (P1, the 6-8 MB regime): split-K without a cluster, reduced by red.add into an output zeroed by a memset node in
    # the same graph (gv V bit 26), every W chunk of a warp requested at launch (ring depth ST = KU, 8th geometry field).
    # Router: plain BF16 hi + lo (FMT 2), as r6 at m <= 32 (r6: cluster 8 / 16 DSMEM, 4.35-6.29 judge-us). Unpooler (rows
    # 0..1023, K 4096) and o at m <= 8: r6's 12-bit pack (FMT 1, dec1).
    _RD = 1 << 26
    _ZF, _PZ, _XZ = 1 << 27, 1 << 28, 1 << 29
    GRD = []
    for _nm, _g, _ms, _vs in (("rk-s8", (2, 1, 8, 2, 8, 1, 0, 2), (8, 16, 32), "z"),           # 16 x 8 = 128 CTAs x 256 thr
                              ("rk-s16w8", (2, 1, 8, 1, 16, 1, 0, 1), (8, 16, 32), "z"),      # 256 CTAs x 256 thr, one chunk per warp
                              ("rk-s16w8", (2, 1, 8, 1, 16, 1, 0, 1), (8, 16), "x"),          # critic (pw27h): the x variant at m=32 is 272 CTAs > 264 resident slots
                              ("rk-s32", (2, 1, 4, 1, 32, 1, 0, 1), (8, 16), "x"),          # 512 CTAs x 128 thr
                              # wf_jsweep2: judge rk bake (job 1f6b5678b9994e38) has x > z by 0.19 at s16w8 and all four
                              # shipped rk shapes inside 0.55 us at m=8 -> 16 warps (one chunk per warp). rk-s8x was +5.4%
                              # on the box (8.06 vs 7.65) -> dropped.
                              ("rk-s16w16", (2, 1, 16, 1, 8, 1, 0, 1), (8,), "z")):
        for _c in _vs:
            GRD.append((_nm + _c, "router", _ms, _g, {"z": _ZF, "x": _XZ, "p": _PZ}[_c]))  # P1 self-zeroing variants
    for _nm, _ly, _g, _vs in (("uk-s4", "unpooler", (1, 1, 8, 2, 4, 1, 0, 2), "zx"),    # n=1024: 64 x 4 = 256 CTAs x 256 thr
                              ("uk-s4w4", "unpooler", (1, 1, 4, 4, 4, 1, 0, 4), "x")):    # 256 CTAs x 128 thr
        for _c in _vs:
            GRD.append((_nm + _c, _ly, (1, 8), _g, 1 | {"z": _ZF, "x": _XZ}[_c]))
    GXP += GRD

    # wf_qkv2: qkv_gate m=1/m=8 SM-coverage bake. The incumbent has no gv bake for these buckets (r6's own v12 bake picks
    # 72 CTAs x 1024 thr = 0.55 waves at both m=1 and m=8), so this registers the one feasible "all 132 SMs in one wave with
    # the same loads in flight per SM" configuration: 144 CTAs x 512 thr at 2 CTAs/SM = 32 warps/SM on 132 SMs (16 rows per
    # CTA, the same instance GVC[("qkv_gate", 1)] already builds -> no new kernel instance, no prepare-time cost).
    # Not registered (measured, see proj_rsi/candidates/wf_qkv2/NOTES.md): a balanced 132-CTA (60x18 + 72x17 rows) or
    # 128-CTA (18 rows) split is infeasible - the 12-bit pack interleaves 16 W rows into 3 uint4/lane (MMA m16n8k16 A
    # fragment), so the row granularity is fixed at 16 rows, 2304 = 144 tiles, and 132 does not divide 144; and
    # 144 CTAs x 1024 thr (WK 32 / KU 4, 1 CTA/SM, 1.09 waves) is +2.3 us on the box (wf_qkv1 b1: 14.10 vs 11.81).
    GXP += [("q144k16", "qkv_gate", (1, 8), (1, 1, 16, 8, 1, 1, 2), 1 | 2 << 12)]


    # pw14 PROBE ladder on r6's own gv (judge-picked geometries of job 23cecdef1bb54c9e), never registered as paths:
    # (layer, MT, geo (FMT, WR, WK, KU, S, RT, XS[, ST]), V, step). LAD (V bits 9-11) 1..4 = steps L0..L3; L4 = r6's real
    # pick, timed separately. capNN = the full kernel with the __launch_bounds__ minBlocks override (V bits 12-14).
    GL_P = (1, 2, 16, 8, 1, 1, 0)
    GL_P4 = (1, 1, 8, 16, 1, 1, 2)
    GL_Q = (1, 1, 8, 16, 1, 1, 0)
    GL_Q4 = (1, 1, 4, 16, 2, 1, 2, 1)
    GLAD = [("pooler", 1, GL_P, 3 | 1 << 9, "L0"), ("pooler", 1, GL_P, 3 | 4 << 9, "L3"),
            ("qkv_gate", 1, GL_Q, 1 | 1 << 9, "L0"), ("qkv_gate", 1, GL_Q, 1 | 4 << 9, "L3")]
    GLAD += [(ly, 4, g, 1 | k << 9, "L%d" % (k - 1)) for ly, g in (("pooler", GL_P4), ("qkv_gate", GL_Q4)) for k in (1, 2, 3, 4)]
    GLAD += [("pooler", 4, GL_P4, 1 | 4 << 12, "cap64"), ("pooler", 4, GL_P4, 1 | 1 << 12, "cap255"),
             ("qkv_gate", 4, GL_Q4, 1 | 1 << 12, "cap255"),
             ("pooler", 4, GL_P4, 1 | 8 | 2 << 4 | 256 | 1 << 9, "L0-bulk8")]
    GLAD = []  # pw16: the pw14 ladder is not built (its result is in audits/r6-lineage.md)

    # pw16: x cache-policy experiment on r6's own m=16/32 geometries (judge picks). V bits 15-16 = x load policy XP
    # (0 r6: ld.global.nc, L1-allocating; 1: ld.global.cg, L1 bypass = r6's L1 reuse removed; 2: ld.global.nc.L1::evict_last);
    # V bit 17 = PreferredSharedMemoryCarveout 0 (largest L1 the kernel's smem allows). W loads stay .nc.L1::no_allocate.
    GCX = (("-xcg", 1 << 15), ("-xel", 2 << 15), ("-L1", 1 << 17), ("-xelL1", 2 << 15 | 1 << 17))
    for _nm, _ly, _m, _g, _v in (("p", "pooler", 16, (1, 2, 8, 16, 1, 1, 0), 1), ("p", "pooler", 32, (1, 1, 8, 16, 1, 1, 2), 1),
                                 ("u", "unpooler", 16, (1, 1, 4, 16, 1, 1, 2), 1), ("u", "unpooler", 32, (1, 1, 4, 16, 1, 1, 2), 0),
                                 ("q", "qkv_gate", 16, (1, 1, 4, 8, 4, 1, 2, 1), 1), ("q", "qkv_gate", 32, (1, 1, 4, 16, 2, 1, 2, 1), 1)):
        for _sfx, _b in ():  # pw17: pw16's x-policy paths all lost (GCX); not built
            GXP.append(("gc-%s%s" % (_nm, _sfx), _ly, (_m,), _g, _v | _b))


    def _gxinst(geo7, v, bm):
        fmt, wr, wk, ku, s, rt, xs = geo7[:7]
        return (fmt, GVB.get(bm, 16), geo7[7] if len(geo7) > 7 else _gvst((wr, wk, ku, s, rt, xs)), wr, wk, ku, s, rt, xs, v)


    GVI = list(dict.fromkeys(_gvinst() + [_gxinst(g, v, bm) for _, _, ms, g, v in GXP for bm in ms]
                            + [_gxinst(g, v, {1: 1, 2: 16, 4: 32}[mt]) for _, mt, g, v, _ in GLAD]))
    _gvmod = None


    def _gvext():
        nonlocal _gvmod
        if _gvmod is None:
            cases = "\n".join("    case %d: return gvl<%s>(p, st, occ);" % (i, ",".join(map(str, c))) for i, c in enumerate(GVI))
            _gvmod = load_inline("gvp45_predxor_v2_%x" % (hash(tuple(GVI)) & 0xffffffff), cpp_sources="int64_t gvrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t); int64_t gvocc(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t); int64_t gvregs(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t); int64_t rrrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor); int64_t rsrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor); int64_t prrun(int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor);",
                                 cuda_sources=GVCU.replace("@@CASES@@", cases), functions=["gvrun", "gvocc", "gvregs", "rrrun", "rsrun", "prrun"], extra_cuda_cflags=["-O3", "-lineinfo"], verbose=False)
        return _gvmod


    # code-word bit of each value's 3-bit offset and of its sign, per decode variant (see dec0 / dec1)
    _OS = ((0, 4, 8, 12, 16, 20, 24, 28), (0, 8, 16, 24, 4, 12, 20, 28))
    _SS = ((3, 11, 19, 27, 7, 15, 23, 31), (7, 15, 23, 31, 11, 19, 27, 3))


    def _c12(fr, dv=1):
        """bf16 fragments [T, KC, 4, 32, 8] -> (uint8 [T, KC, 3, 32, 16], table, escapes [E, 6] (tile, ch, j, lane, v, xor)) or None"""
        T, KC = fr.shape[:2]
        dev = fr.device
        b = fr.reshape(T, KC, 4, 32, 8).view(torch.int16).int() & 0xffff
        h7, sg = (b >> 8) & 127, b >> 15
        win = torch.bincount(h7.flatten(), minlength=128).unfold(0, 8, 1).sum(1)
        B = int(win.argmax())
        d = h7 - B
        inw = (d >= 0) & (d < 8)
        d = torch.where(inw, d, 0).long()
        C = ((d << torch.tensor(_OS[dv], device=dev)) | (sg.long() << torch.tensor(_SS[dv], device=dev))).sum(-1)
        C = torch.where(C >= 2 ** 31, C - 2 ** 32, C).int()
        lo = (b & 255).to(torch.uint8)
        p0 = lo[:, :, 0:2].permute(0, 1, 3, 2, 4).reshape(T, KC, 32, 16)
        p1 = lo[:, :, 2:4].permute(0, 1, 3, 2, 4).reshape(T, KC, 32, 16)
        p2 = C.permute(0, 1, 3, 2).contiguous().view(torch.uint8).view(T, KC, 32, 16)
        data = torch.stack((p0, p1, p2), 2).contiguous()
        ei = (~inw).nonzero()
        esc = torch.cat((ei, (h7[tuple(ei.t())] ^ B)[:, None].long()), 1)
        tab = B * 0x01010101 if dv else sum((B + i) << (8 * i) for i in range(8))
        if not torch.equal(_unc12(data, tab, esc, dv), b):
            return None
        return data, tab - 2 ** 64 if tab >= 2 ** 63 else tab, esc


    def _unc12(data, tab, esc, dv):
        """independent decoder (mirrors the kernel's byte permutes) for the exactness proof"""
        T, KC = data.shape[:2]
        dev = data.device
        w = data.view(torch.int32).view(T, KC, 3, 32, 4).long() & 0xffffffff
        L = torch.stack((w[:, :, 0, :, 0], w[:, :, 0, :, 1], w[:, :, 0, :, 2], w[:, :, 0, :, 3], w[:, :, 1, :, 0], w[:, :, 1, :, 1], w[:, :, 1, :, 2], w[:, :, 1, :, 3]), 2).view(T, KC, 4, 2, 32)
        C = w[:, :, 2].permute(0, 1, 3, 2)
        off = (C[..., None] >> torch.tensor(_OS[dv], device=dev)) & 7
        sgn = (C[..., None] >> torch.tensor(_SS[dv], device=dev)) & 1
        tb = torch.tensor([(tab >> (8 * i)) & 255 for i in range(8)], device=dev)
        hi = ((tab & 255) + off if dv else tb[off]) | (sgn << 7)
        lob = (L.permute(0, 1, 2, 4, 3)[..., None] >> (8 * torch.arange(4, device=dev))) & 255
        val = (hi << 8) | lob.reshape(T, KC, 4, 32, 8)
        if len(esc):
            t, c, j, l, vv, x = esc.t()
            val[t, c, j, l, vv] ^= x << 8
        return val.int()


    def _gvesc(esc, T, geo):
        """per-warp escape lists for one launch geometry (32 entries per warp, sorted by (unit, tile)); None if a list overflows.
    esc: [E, 7] (tile, chunk, step, lane, value, xor, matrix)"""
        wr, wk, ku, s, rt, xs = geo
        gx, nw = T // (wr * rt), wr * wk
        out = torch.full((gx * s * nw * 32,), -1, dtype=torch.int64, device=esc.device)
        e = esc[esc[:, 0] < T]
        if len(e):
            t, c, j, l, v, x, h = e.t()
            gw = ((c // (wk * ku)) * gx + t // (wr * rt)) * nw + ((c // ku) % wk) * wr + (t // rt) % wr
            seq = (c % ku) * rt + t % rt
            o = torch.argsort(gw * 4096 + seq)
            gw, ent = gw[o], ((seq << 20) | (l << 15) | ((j * 4 + v // 2) << 11) | ((v % 2) << 10) | (h << 9) | x)[o]
            rank = torch.arange(len(gw), device=gw.device) - torch.searchsorted(gw, gw)
            if int(rank.max()) >= 32:
                return None
            out[gw * 32 + rank] = ent
        return torch.where(out >= 2 ** 31, out - 2 ** 32, out).int()


    def _bk(m):
        return 1 if m == 1 else 8 if m <= 8 else 16 if m <= 16 else 32 if m <= 32 else 128 if m == 128 else 256 if m == 256 else 0


    def _tm(fn, flush, it=15):
        """median replay time (ms) of a CUDA graph of fn, each replay after the validated cold-L2 READ flush (pw55)"""
        g = _capture(fn)
        ts = _replay({"a": g}, it)["a"]
        ts.sort(); return ts[len(ts) // 2]


    def _tmp(fn, base, flush, it=15):
        """pw55: fn timed ALTERNATING with `base` (order-swapped pairs, median paired ratio, fixed cost subtracted);
    returns fn's time in base units (ms), i.e. base_median / ratio -- the judge's own form."""
        ratio, bmed, cmed, ov = _paired(fn, base, max(2, it // 2))
        _LAST[0] = (ratio, bmed, cmed, ov)
        return bmed / ratio


    def _gvok(st, plan, exact, w, mr):
        """exact on every one-hot column, close on random rows"""
        ext, n, K = _gvext(), exact.shape[0], exact.shape[1]
        mm = (8 if plan[0] in (1002, 1003) else 1) if plan[0] >= 1000 else 8 * GVI[plan[0]][1]
        dt = exact.dtype
        eye = torch.eye(K, dtype=torch.bfloat16, device=w.device).view(K // mm, mm, K)
        got = torch.empty((K // mm, mm, n), dtype=dt, device=w.device)
        for i in range(K // mm):
            if _gvcall(st, plan, eye[i], got[i]):
                return False
        if not torch.equal(got.view(K, n), exact.t()):
            return False
        x = torch.randn((max(mm, mr), K), device=w.device).to(torch.bfloat16)
        out = torch.empty((max(mm, mr), n), dtype=dt, device=w.device)
        _gvcall(st, plan, x, out)
        ref = x.double() @ w.double().T
        return bool(torch.isfinite(out).all()) and float((out.double() - ref).norm() / ref.norm()) < (2e-5 if dt == torch.float32 else 4e-3)


    RRB = {1: (6, 7)}  # pw52: proven rr6 first (judge m1 2.56); rr7 kept (2.59, inside 3%); rr0 (2.94, +15%) and rr1 (3.01, +18%) dropped  # P1: rr2/rr3 (no cross-CTA reduction) re-measured at m=8 on the H100; fp32 router kernels per bucket (m8 rr: 7.7-14.9 us on the H200, v12 4.7)


    def _name(i):
        return "rr%d" % (i - 1000) if i >= 1000 else ".".join(map(str, GVI[i]))


    def _gvcall(st, p, x, out):
        if 0 <= p[0] < 1000 and (GVI[p[0]][9] >> 26) & 1:  # pw27 red.add split-K: the output accumulates from zero
            out.zero_()
        if 0 <= p[0] < 1000 and (GVI[p[0]][9] >> 27) & 5:  # P1: per-instance ticket / zero-flag counters (self-zeroing)
            c = st.setdefault("p1cnt", {}).get(p[0])
            if c is None:
                c = st["p1cnt"][p[0]] = torch.zeros((2048,), dtype=torch.int32, device=x.device)
            return _gvext().gvrun(p[0], p[1], p[5], x, out, p[2], st["gvws"], c, p[3], p[6])
        if p[0] == -7:  # pw22 router split-K
            return _rscall(st, p, x, out)
        if p[0] >= 1000:
            return _gvext().rrrun(p[0] - 1000, p[1], x, out)
        return _gvext().gvrun(p[0], p[1], p[5], x, out, p[2], st["gvws"], st["gvcnt"], p[3], p[6])


    def _gvprep(st, flush):
        dev = flush.device
        st["gvws"] = torch.zeros((1 << 24,), dtype=torch.float32, device=dev)
        st["gvcnt"] = torch.zeros((1 << 16,), dtype=torch.int32, device=dev)
        noesc = torch.full((1 << 16,), -1, dtype=torch.int32, device=dev)
        st["gvlog"] = []
        ok = {}
        z = torch.zeros(1, device=dev)
        off = _tm(lambda: z.zero_(), flush, 31)
        for layer in ("qkv_gate", "o", "pooler", "unpooler", "router"):
            hl = layer == "router"
            if not _NCR(layer, [bm_ for bm_, _mt in list(GVB.items()) + (GVRB if hl else [])]):
                continue
            frs = list(st[layer]) if hl else [st[layer][0]]
            css = [[_c12(f, dv) for f in frs] if GVC12 and not hl else [None] for dv in (0, 1)]
            c12 = all(c is not None for cs in css for c in cs)
            if c12:
                esc = torch.cat([torch.cat((c[2], torch.full_like(c[2][:, :1], h)), 1) for h, c in enumerate(css[1])])
                st.setdefault("gvc12", {})[layer] = (css, esc)  # pw11: reused by the second bake
            Tf, KC = frs[0].shape[:2]
            w = st[layer + "_raw"]
            for bm, mt in list(GVB.items()) + (GVRB if hl else []):  # router m128/256 as token blocks: 1.6x slower than v12 on the 3090
                if not _NC(layer, bm):
                    continue
                key = "unpooler_s" if layer == "unpooler" and bm <= 8 else layer
                n = 1024 if key == "unpooler_s" else Tf * 16
                T, wn = n // 16, w[:n]
                exact = (wn.to(torch.bfloat16).float() + (wn - wn.to(torch.bfloat16).float()).to(torch.bfloat16).float()) if hl else wn
                cands = []
                for k, (geo, vs) in enumerate(GVC.get((key, mt), [])):
                    if not c12 and k and not hl:
                        break
                    if geo[1] * geo[2] * geo[3] != KC or T % (geo[0] * geo[4]):
                        continue
                    e = _gvesc(esc, T, geo[:6]) if c12 else noesc
                    for v in vs if c12 else (0,):
                        fmt = 1 if c12 else 2 if hl else 0
                        i = GVI.index((fmt, mt, _gvst(geo)) + geo[:6] + (v,))
                        if e is None:
                            continue
                        cs = css[v & 1]
                        src = [c[0] for c in cs] if fmt & 1 else frs
                        plan = (i, src[0], e, cs[0][1] if fmt & 1 else 0, n, src[-1], cs[-1][1] if fmt & 1 else 0)
                        if (i, n, bm > 8 * mt) not in ok:
                            ok[(i, n, bm > 8 * mt)] = _gvok(st, plan, exact, wn, bm)
                        if ok[(i, n, bm > 8 * mt)]:
                            cands.append(plan)
                for i in RRB.get(bm, ()) if hl else ():
                    plan = (1000 + i, wn.contiguous(), noesc, 0, n, wn, 0)
                    if (plan[0], n, 0) not in ok:
                        ok[(plan[0], n, 0)] = _gvok(st, plan, wn, wn, bm)
                    if ok[(plan[0], n, 0)]:
                        cands.append(plan)
                x = torch.randn((bm, KC * 64), device=dev).to(torch.bfloat16)
                dt = torch.float32 if hl else torch.bfloat16
                out = torch.empty((bm, n), dtype=dt, device=dev)
                v12 = lambda: _v12(x, layer, n, st)
                t12 = _tm(v12, flush)  # absolute (ms); the floor `off` is subtracted for display and comparison
                res = [(layer, bm, "v12", t12 - off)]
                # pw59 (wf_bake3): one reference per group (see _pickrule): the proven pick when the box does not read it
                # slower than v12 (window = half GVMARGIN), else v12; every candidate is timed ALTERNATING with that same
                # reference (order-swapped pairs, median paired ratio, floor `off` subtracted) and the pick is the best
                # ratio if it beats GVMARGIN, else the reference.
                pv = PROVEN_GV.get((layer, bm))
                prov = next((p for p in cands if _name(p[0]) == pv), None)
                refn, refp, refmed, reffn = "v12", None, t12, v12
                cands2, done = [], set()
                if prov is not None:
                    done.add(id(prov))
                    tp = _tmp(lambda: _gvcall(st, prov, x, out), v12, flush)
                    res.append((layer, bm, _name(prov[0]), tp - off))
                    rn, rm, isp = _refsel(_name(prov[0]), tp, "v12", t12, GWIN)
                    if isp:
                        refn, refp, refmed, reffn = rn, prov, rm, (lambda: _gvcall(st, prov, x, out))
                    else:
                        cands2.append((_name(prov[0]), prov, t12 / tp))
                for p in cands:
                    if id(p) in done:
                        continue
                    us = _tmp(lambda: _gvcall(st, p, x, out), reffn, flush)
                    res.append((layer, bm, _name(p[0]), us - off))
                    cands2.append((_name(p[0]), p, refmed / us))
                bp = _pickrule((refn, refp, refmed), cands2, GVMARGIN)
                best = (bp[2], bp[1])
                st["gvlog"] += res
                print("gv %s m%d v12 %.2f best %s %.2f (ref %s %.2f)" % (layer, bm, (t12 - off) * 1e3, _name(best[1][0]) if best[1] is not None else "v12", (best[0] - off) * 1e3, refn, (refmed - off) * 1e3), flush=True)
                print("  " + " ".join("%s:%.2f" % (r[2], r[3] * 1e3) for r in res), flush=True)
                if best[1] is not None:
                    st["gv"][(layer, bm)] = best[1]
                if P1ALT and hl and bm == 1 and best[1] is not None and best[1][0] >= 1001:  # P1: the rr pick vs r6's rr0, alternated
                    p0 = next((p_ for p_ in cands if p_[0] == 1000), None)
                    if p0 is not None:
                        ta, tb = [], []
                        for _ in range(3):
                            ta.append(_tm(lambda: _gvcall(st, best[1], x, out), flush, 21)); tb.append(_tm(lambda: _gvcall(st, p0, x, out), flush, 21))
                        print("PROBE p1alt router m=1 %s %.2f rr0=%.2f delta=%+.2f" % (_name(best[1][0]), min(ta) * 1e3, min(tb) * 1e3, (min(ta) - min(tb)) * 1e3), flush=True)

    _mod = None
    # host-only (g++): the judge's nvcc crashes on these STL containers under -std=c++20
    LTCPP = r"""
// ---------------------------------------------------------------- cuBLASLt with an algorithm picked in prepare()
// layout bit 0: row-major descriptors with x as A (else column-major, W as A); bit 1: W pre-transposed [K, N] (packed)
// algorithm handle = layout * 100000 + index into that plan's list (heuristic results, then configs added by lt_add)
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <map>
#include <tuple>
#include <vector>
struct LtPlan { cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t a, b, c; std::vector<cublasLtMatmulHeuristicResult_t> algos; };
static cublasLtHandle_t g_lt = nullptr;
static void* g_ltws = nullptr;
static const size_t LTWS = 64ull << 20;
static std::map<std::tuple<int, int, int, int, int>, LtPlan> g_plans;
static cublasLtHandle_t lth() { if (!g_lt) { cublasLtCreate(&g_lt); cudaMalloc(&g_ltws, LTWS); } return g_lt; }
static LtPlan& ltplan(int m, int n, int k, int f32, int lay) {
  auto key = std::make_tuple(m, n, k, f32, lay);
  auto it = g_plans.find(key);
  if (it != g_plans.end()) return it->second;
  lth();
  const bool row = lay & 1, pk = lay & 2;
  const cudaDataType_t ct = f32 ? CUDA_R_32F : CUDA_R_16BF;
  LtPlan p;
  cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasOperation_t ta = !row && !pk ? CUBLAS_OP_T : CUBLAS_OP_N, tb = row && !pk ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
  cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
  if (row) {
    cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16BF, m, k, k);
    cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16BF, pk ? k : n, pk ? n : k, pk ? n : k);
    cublasLtMatrixLayoutCreate(&p.c, ct, m, n, n);
    cublasLtOrder_t o = CUBLASLT_ORDER_ROW;
    for (auto d : {p.a, p.b, p.c}) cublasLtMatrixLayoutSetAttribute(d, CUBLASLT_MATRIX_LAYOUT_ORDER, &o, sizeof(o));
  } else {
    cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16BF, pk ? n : k, pk ? k : n, pk ? n : k);
    cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16BF, k, m, k);
    cublasLtMatrixLayoutCreate(&p.c, ct, n, m, n);
  }
  cublasLtMatmulPreference_t pref;
  cublasLtMatmulPreferenceCreate(&pref);
  size_t ws = LTWS;
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
  p.algos.resize(32);
  int cnt = 0;
  if (cublasLtMatmulAlgoGetHeuristic(g_lt, p.op, p.a, p.b, p.c, p.c, pref, 32, p.algos.data(), &cnt) != CUBLAS_STATUS_SUCCESS) cnt = 0;
  p.algos.resize(cnt);
  cublasLtMatmulPreferenceDestroy(pref);
  return g_plans.emplace(key, p).first->second;
}
int64_t lt_algos(int64_t m, int64_t n, int64_t k, int64_t f32, int64_t lay) { return (int64_t)ltplan(m, n, k, f32, lay).algos.size(); }
std::vector<int64_t> lt_caps(int64_t f32, int64_t id) {
  cublasLtMatmulAlgo_t a;
  std::vector<int64_t> r;
  if (cublasLtMatmulAlgoInit(lth(), CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF,
                             f32 ? CUDA_R_32F : CUDA_R_16BF, f32 ? CUDA_R_32F : CUDA_R_16BF, (int)id, &a) != CUBLAS_STATUS_SUCCESS) return r;
  for (auto cap : {CUBLASLT_ALGO_CAP_TILE_IDS, CUBLASLT_ALGO_CAP_STAGES_IDS}) {
    size_t bytes = 0;
    cublasLtMatmulAlgoCapGetAttribute(&a, cap, nullptr, 0, &bytes);
    std::vector<uint32_t> v(bytes / 4);
    if (bytes) cublasLtMatmulAlgoCapGetAttribute(&a, cap, v.data(), bytes, &bytes);
    r.push_back((int64_t)v.size());
    for (auto x : v) r.push_back(x);
  }
  for (auto cap : {CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK}) {
    uint32_t v = 0; size_t w = 0;
    cublasLtMatmulAlgoCapGetAttribute(&a, cap, &v, sizeof(v), &w);
    r.push_back(v);
  }
  return r;
}
static bool ltset(cublasLtMatmulAlgo_t& a, int attr, int64_t v) {
  if (attr >= 7) { uint16_t x = (uint16_t)v; return cublasLtMatmulAlgoConfigSetAttribute(&a, (cublasLtMatmulAlgoConfigAttributes_t)attr, &x, sizeof(x)) == CUBLAS_STATUS_SUCCESS; }
  uint32_t x = (uint32_t)v; return cublasLtMatmulAlgoConfigSetAttribute(&a, (cublasLtMatmulAlgoConfigAttributes_t)attr, &x, sizeof(x)) == CUBLAS_STATUS_SUCCESS;
}
// cfg: CUBLASLT_ALGO_CONFIG_{ID, TILE_ID, SPLITK_NUM, REDUCTION_SCHEME, CTA_SWIZZLING, CUSTOM_OPTION, STAGES_ID, INNER_SHAPE_ID, CLUSTER_SHAPE_ID}
int64_t lt_add(int64_t m, int64_t n, int64_t k, int64_t f32, int64_t lay, std::vector<int64_t> cfg) {
  if (cfg.size() != 9) return -1;
  LtPlan& p = ltplan(m, n, k, f32, lay);
  cublasLtMatmulAlgo_t a;
  const cudaDataType_t ct = f32 ? CUDA_R_32F : CUDA_R_16BF;
  if (cublasLtMatmulAlgoInit(g_lt, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, ct, ct, (int)cfg[0], &a) != CUBLAS_STATUS_SUCCESS) return -1;
  for (int i = 1; i < 9; i++) if (!ltset(a, i, cfg[i]) && cfg[i]) return -1;
  if (cfg[2] > 1 && cfg[3] != CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE) return -1;
  uint64_t fl = 0; size_t w = 0;
  if (cublasLtMatmulAlgoCapGetAttribute(&a, CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS, &fl, sizeof(fl), &w) != CUBLAS_STATUS_SUCCESS) return -1;
  if ((fl & CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_TYPE_MASK) != CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F || (fl & CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_TF32)) return -1;
  cublasLtMatmulHeuristicResult_t r{};
  if (cublasLtMatmulAlgoCheck(g_lt, p.op, p.a, p.b, p.c, p.c, &a, &r) != CUBLAS_STATUS_SUCCESS || r.state != CUBLAS_STATUS_SUCCESS || r.workspaceSize > LTWS) return -1;
  r.algo = a;
  p.algos.push_back(r);
  return lay * 100000 + (int64_t)p.algos.size() - 1;
}
std::vector<int64_t> lt_cfg(int64_t m, int64_t n, int64_t k, int64_t f32, int64_t code) {
  LtPlan& p = ltplan(m, n, k, f32, code / 100000);
  std::vector<int64_t> r;
  const auto& a = p.algos.at(code % 100000).algo;
  for (int i = 0; i < 9; i++) {
    size_t w = 0;
    if (i >= 7) { uint16_t v = 0; cublasLtMatmulAlgoConfigGetAttribute(&a, (cublasLtMatmulAlgoConfigAttributes_t)i, &v, sizeof(v), &w); r.push_back(v); }
    else { uint32_t v = 0; cublasLtMatmulAlgoConfigGetAttribute(&a, (cublasLtMatmulAlgoConfigAttributes_t)i, &v, sizeof(v), &w); r.push_back(v); }
  }
  return r;
}
int64_t lt_mm(torch::Tensor x, torch::Tensor w, torch::Tensor wt, torch::Tensor out, int64_t n, int64_t code) {
  const int m = x.size(0), k = x.size(1), f32 = out.scalar_type() == torch::kFloat32, lay = (int)(code / 100000);
  LtPlan& p = ltplan(m, (int)n, k, f32, lay);
  const float alpha = 1.f, beta = 0.f;
  const void* wp = (lay & 2) ? wt.data_ptr() : w.data_ptr();
  const void* a = (lay & 1) ? x.data_ptr() : wp;
  const void* b = (lay & 1) ? wp : x.data_ptr();
  const auto& al = p.algos.at(code % 100000);
  return (int64_t)cublasLtMatmul(g_lt, p.op, &alpha, a, p.a, b, p.b, &beta, out.data_ptr(), p.c, out.data_ptr(), p.c,
                 &al.algo, g_ltws, LTWS, at::cuda::getCurrentCUDAStream());
}
"""
    _DBG = []
    RTR = True
    GV = True
    GVC12 = True
    GVMARGIN = 0.97  # pw52: a variant must beat the PROVEN pick by 3% at m <= 32 (FACTS 14)
    TUNE = True  # prepare(): time torch vs cuBLASLt algorithms per cuBLAS-served shape
    V6 = False  # deep per-CTA ring GEMV: slower than v5 on H200


    def _big():
        pr = torch.cuda.get_device_properties(0)
        sm = getattr(pr, "shared_memory_per_block_optin", None)
        return sm >= 180000 if sm else (pr.major, pr.minor) in ((9, 0), (10, 0))


    def _ext():
        nonlocal _mod
        if _mod is None:
            wpc = next((int(d[6:]) for d in _DBG if d.startswith("-DWPC=")), 16 if _big() else 8)
            _mod = load_inline("skinny_v20_%d" % wpc + "".join(_DBG).replace("-", "_").replace("=", ""),
                               cpp_sources=LTCPP + "void run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t); void run2(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t); void run3(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t);",
                               cuda_sources=CUDA, functions=["run", "run2", "run3", "lt_algos", "lt_mm", "lt_caps", "lt_add", "lt_cfg"], extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo", "-DWPC=%d" % wpc] + ([] if _big() else ["-DNSG=4"]) + [d for d in _DBG if not d.startswith("-DWPC=")], extra_ldflags=_ltflags(),
                               verbose=False)
        return _mod


    def _ltflags():
        """link the process's cuBLASLt (torch's nvidia-cublas wheel ships only the versioned soname)"""
        import glob, os
        try:
            import nvidia.cublas as nc
            d = os.path.join(list(nc.__path__)[0], "lib")
        except Exception:
            d = os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "lib64")
        so = sorted(glob.glob(os.path.join(d, "libcublasLt.so*")))
        return ["-L" + d, "-l:" + os.path.basename(so[0]) if so else "-lcublasLt", "-Wl,-rpath," + d]


    def _tune(st, flush):
        """pick, per cuBLAS-served shape, the fastest of torch's default and every cuBLASLt heuristic algorithm (timed here,
    on this GPU, as the judge times: CUDA graph replays from a cold L2); m 128/256 get the pinned-config search below"""
        ext = _ext()
        tm = lambda fn, it=12: _tm(fn, flush, it)
        dev = flush.device
        jobs = []
        for layer, (N, K) in (("qkv_gate", (2304, 8192)), ("o", (8192, 1024)), ("pooler", (4096, 8192)), ("unpooler", (8192, 4096))):
            for m in (1, 8, 16, 32, 128, 256):
                if _cublas(layer, m) and _NC(layer, m):
                    jobs.append((layer, m, 1024 if layer == "unpooler" and m <= 8 else N, K, st[layer + "_raw"], st.get(layer + "_t"), torch.bfloat16))
        for m in (128, 256):
            if _NC("router", m):
                jobs.append(("router", m, 512, 8192, st["router_hl"], st["router_hl_t"], torch.float32))
        t0 = time.time()
        for layer, m, n, K, W, Wt, dt in jobs:
            x = torch.randn((m, K), device=dev).to(torch.bfloat16)
            out = torch.empty((m, n), dtype=dt, device=dev)
            ref = torch.mm(x.float(), W[:n].float().T)
            f32 = int(dt == torch.float32)
            torch_fn = (lambda: torch.mm(x, W[:n].T, out_dtype=dt)) if f32 else (lambda: torch.mm(x, W[:n].T))
            ok = lambda c: ext.lt_mm(x, W, Wt if Wt is not None else W, out, n, c) == 0 and _close(out, ref)
            run = lambda c: ext.lt_mm(x, W, Wt if Wt is not None else W, out, n, c)
            if m < 128 or not LTSEARCH or Wt is None:
                best = (tm(torch_fn), -1)
                for i in range(ext.lt_algos(m, n, K, f32, 0)):
                    if ok(i):
                        best = min(best, (tm(lambda: run(i)), i))
                if best[1] >= 0:
                    st["lt"][(layer, m)] = best[1]
                continue
            st["lt"][(layer, m)] = _ltsearch(ext, layer, m, n, K, f32, ok, run, torch_fn, flush, t0)


    LTSEARCH = True
    LTBUDGET = 8.0 if _PRUNE else 22.0  # seconds for the whole m128/256 search (10 shapes)
    LTW = {"qkv_gate": 1.6, "o": 0.3, "pooler": 1.0, "unpooler": 0.7, "router": 1.6}  # v18 H200: o never beat torch
    # H200 picks reported for another submission's pinned search: (layout, epilogue, CUBLASLT_ALGO_CONFIG_* fields) -> seeds
    _LTSEED = [(2, [66, 17, 1, 0, 0, 0, 35, 0, 2]), (2, [66, 18, 1, 0, 0, 3, 35, 0, 5]), (3, [66, 20, 1, 0, 0, 4, 35, 0, 2]),
               (0, [66, 92, 1, 0, 0, 0, 35, 0, 5]), (0, [66, 124, 1, 0, 0, 0, 35, 0, 2]), (2, [66, 15, 1, 0, 0, 4, 35, 0, 2]),
               (3, [66, 18, 1, 0, 0, 0, 35, 0, 2]), (2, [66, 15, 1, 0, 0, 0, 35, 0, 2]), (2, [66, 15, 1, 0, 0, 1, 35, 0, 2]),
               (2, [66, 15, 1, 0, 0, 0, 35, 0, 4]), (2, [66, 18, 1, 0, 0, 4, 35, 0, 2]), (3, [66, 18, 1, 0, 0, 2, 35, 0, 5]),
               (3, [66, 18, 1, 0, 0, 5, 35, 0, 2]), (2, [66, 15, 1, 0, 0, 2, 35, 0, 5]), (2, [66, 18, 1, 0, 0, 0, 35, 0, 2]),
               (2, [66, 18, 1, 0, 0, 6, 35, 0, 5])]


    def _close(out, ref):
        torch.cuda.synchronize()
        return bool(torch.isfinite(out).all()) and float((out.float() - ref).norm() / ref.norm()) < 1e-2


    def _ltsearch(ext, layer, m, n, K, f32, ok, run, torch_fn, flush, t0):
        """heuristic algorithms of all four layouts + seeds + an algo-66 (sm90 nvjet) sweep over tile / stages / cluster, then
    split-K / custom option / cluster / stages refinement of the leaders; all checked numerically and timed (judge timer)"""
        if not _LTDONE and not _LT0:
            _LT0.append(time.time())
        t0 = _LT0[0]
        order = [(l, mm) for l in ("qkv_gate", "o", "pooler", "unpooler", "router") for mm in (128, 256) if _NC(l, mm)]
        wsum = sum(LTW[l] for l, _ in order)
        dl = t0 + LTBUDGET * sum(LTW[l] for l, mm in order[:order.index((layer, m)) + 1]) / wsum  # cumulative, weighted
        njob = wsum / LTW[layer]
        tried, res = set(), []
        def tryc(code):
            if code < 0 or code in tried:
                return
            tried.add(code)
            if ok(code):
                res.append((_tm(lambda: run(code), flush, 5), code))
        def addc(lay, cfg):
            c = ext.lt_add(m, n, K, f32, lay, cfg)
            if c >= 0:
                cfg2 = tuple(ext.lt_cfg(m, n, K, f32, c))
                if (lay, cfg2) in seen:
                    return
                seen.add((lay, cfg2))
                tryc(c)
        seen = set()
        t_torch = _tm(torch_fn, flush, 9)
        for lay in range(4):
            for i in range(min(ext.lt_algos(m, n, K, f32, lay), 8)):
                seen.add((lay, tuple(ext.lt_cfg(m, n, K, f32, lay * 100000 + i))))
                tryc(lay * 100000 + i)
        for lay, cfg in _LTSEED:
            addc(lay, cfg)
        caps = ext.lt_caps(f32, 66)
        if caps:
            nt = caps[0]; tiles = caps[1:1 + nt]; ns = caps[1 + nt]; stages = caps[2 + nt:2 + nt + ns]
            cmax, ksup, swz = caps[2 + nt + ns], caps[3 + nt + ns], caps[4 + nt + ns]
            stg = [35] if 35 in stages else stages[:1] or [0]
            for lay in (2, 0, 3, 1):
                for tile in tiles:
                    if time.time() > dl - 0.4 * LTBUDGET / njob:
                        break
                    for cl in (2, 5):
                        addc(lay, [66, tile, 1, 0, 0, 0, stg[0], 0, cl])
            res.sort()
            for _, code in res[:4]:
                if time.time() > dl:
                    break
                lay, cfg = code // 100000, list(ext.lt_cfg(m, n, K, f32, code))
                if cfg[0] != 66:
                    continue
                for cu in range(min(cmax, 7) + 1):
                    for cl in (0, 2, 3, 4, 5, 6):
                        if time.time() > dl:
                            break
                        addc(lay, cfg[:5] + [cu] + cfg[6:8] + [cl])
                if ksup:
                    for sk in (2, 3, 4):
                        addc(lay, cfg[:2] + [sk, 2] + cfg[4:])
                for sg in stages[::max(1, len(stages) // 6)]:
                    if time.time() > dl:
                        break
                    addc(lay, cfg[:6] + [sg] + cfg[7:])
        res.sort()
        best = (t_torch, -1)
        for _, code in res[:3]:  # re-time the leaders against torch's default
            a = _tm(lambda: run(code), flush, 15)
            t_torch = min(t_torch, _tm(torch_fn, flush, 15))
            if a < best[0]:
                best = (a, code)
        if best[1] >= 0 and best[0] > t_torch * 0.99:
            best = (t_torch, -1)
        _LTDONE.append((layer, m))
        cfg = "torch" if best[1] < 0 else "%d:%s" % (best[1] // 100000, ".".join(map(str, ext.lt_cfg(m, n, K, f32, best[1]))))
        print("lt %s m%d heur %.2f best %s %.2f (%d tried, %.1fs)" % (layer, m, t_torch * 1e3, cfg, best[0] * 1e3, len(tried), time.time() - t0), flush=True)
        return best[1] if best[1] >= 0 else None


    _LTDONE, _LT0 = [], []


    def _cublas(layer, m):
        return layer != "router" and (m > 1 or layer == "o") and not (layer == "unpooler" and m == 1)


    def _pack(w):
        """bf16 [N, K] -> A fragments [N/16, K/64, 32 lanes, 4 steps, 8] (K permuted so each lane's 16 k are contiguous)."""
        N, K = w.shape
        W = w.view(N // 16, 16, K // 64, 4, 4, 4)  # mt, i, ch, t, j, e
        lo, hi = W[:, :8], W[:, 8:]
        fr = torch.stack((lo[..., 0], lo[..., 1], hi[..., 0], hi[..., 1], lo[..., 2], lo[..., 3], hi[..., 2], hi[..., 3]), -1)
        return fr.permute(0, 2, 4, 1, 3, 5).contiguous()  # mt, ch, j, g, t, 8: per step j the 32 lanes are contiguous


    def prepare(weights):
        st = {}
        if weights["qkv_gate"].device.type != "cuda":  # pw11: CPU (proj_rsi cpu-check) runs the plain torch math
            return {"cpu": True, "raw": dict(weights)}
        rqt = _rqstart() if RQ else None  # rs128: compile overlapped with r6's phase (CPU only)
        rqt2 = _rqstart(RQPROD2, _rqmod2, _rqthr2, "rq32") if RQ and RQS else None  # rs32: m = 16 / 32 instances
        _thr = {}
        if _PRUNE:  # compile the gv and flush extensions beside _ext() and the rq builds (nvcc subprocesses; the judge's prepare is compile-bound)
            import threading
            for nm_, fn_ in (("gv", _gvext), ("flush", _flushext)):
                def _bg(fn_=fn_):
                    try:
                        fn_()
                    except Exception:  # the main thread retries the build and raises as before
                        pass
                _thr[nm_] = threading.Thread(target=_bg, daemon=True)
                _thr[nm_].start()
        for k, w in weights.items():
            if k == "router":
                hi = w.to(torch.bfloat16)
                lo = (w - hi.float()).to(torch.bfloat16)
                st[k] = (_pack(hi), _pack(lo))
                st["router_hl_t"] = torch.cat((hi, lo)).t().contiguous()  # [K, 2N]: one cuBLAS mm for m > 32
                st["router_hl"] = torch.cat((hi, lo)).contiguous()  # [2N, K] for cuBLASLt
            else:
                st[k] = (_pack(w), None)
                st[k + "_t"] = w.t().contiguous()  # [K, N] for the packed cuBLASLt layouts
            st[k + "_raw"] = w
        dev = weights["qkv_gate"].device
        st["ws"] = torch.zeros((256 * 8192,), dtype=torch.float32, device=dev)
        st["cnt"] = torch.zeros((8 * 512,), dtype=torch.int32, device=dev)
        st["none"] = torch.empty(0, dtype=torch.bfloat16, device=dev)
        _ext()
        # pw55: the judge-faithful bake measurement -- build + validate the cold-L2 read flush and measure the graph
        # fixed cost once, before any bake (FACTS 14: the old torch.sum flush left x and small weights L2-warm).
        if "flush" in _thr:
            _thr["flush"].join()
        _flushext()
        _flush_validate()
        _fixed_cost()
        print("PREP flush+fixed done at %.0fs" % (time.time() - _T0), flush=True)
        st["lt"] = {}
        if TUNE:
            _tune(st, torch.empty(64 << 20, dtype=torch.float32, device=dev))
        st["gv"] = {}
        if "gv" in _thr:
            _thr["gv"].join()
        if GV:
            _gvprep(st, torch.empty(64 << 20, dtype=torch.float32, device=dev))
        print("PREP r6 done in %.0fs" % (time.time() - _T0), flush=True)
        st["sel"] = {}
        st["rq"] = {}  # rs128 before the gx / router phases: its gain is the larger
        if RQ and rqt is not None:
            try:
                _rqbake(st, rqt)
            except Exception as error:  # the rs128 bake can only add m = 128 / 256 paths; r6's choices stand on any failure
                st["rq"] = {}
                print("PREP rq bake failed: %s" % str(error).splitlines()[-1][:200], flush=True)
            finally:
                torch.cuda.synchronize()
        print("PREP phase rq done at %.0fs" % (time.time() - _T0), flush=True)  # pw36: phase timestamps
        if GV and GX:
            try:
                _gxbake(st)
            except Exception as error:  # the second bake can only add paths; r6's choices stand on any failure
                st["sel"] = {}
                print("PREP gx bake failed: %s" % str(error).splitlines()[-1][:200], flush=True)
        print("PREP phase gx done at %.0fs" % (time.time() - _T0), flush=True)  # pw36: phase timestamps
        if RQ and RQS and rqt2 is not None:
            try:
                _rqsbake(st, rqt2)
            except Exception as error:  # can only add (layer, 16 / 32) rq paths; the incumbent stands on any failure
                for k_ in [k_ for k_ in st["rq"] if k_[1] in (16, 32)]:
                    del st["rq"][k_]
                print("PREP rq32 bake failed: %s" % str(error).splitlines()[-1][:200], flush=True)
            finally:
                torch.cuda.synchronize()
        print("PREP phase rq32 done at %.0fs" % (time.time() - _T0), flush=True)  # pw36: phase timestamps
        st.pop("rq_pk", None)
        if PR27 and GV and GX and time.time() - _T0 < PREPARE_BUDGET_S - 30:
            try:
                _pr27(st)
            except Exception as error:
                print("PROBE rd8 failed: %s" % str(error).splitlines()[-1][:200], flush=True)
        if GV and GX and _NCR("router", (128, 256)) and time.time() - _T0 < PREPARE_BUDGET_S - 20:
            try:
                _rsbake(st)
            except Exception as error:  # can only add router paths; r6's choices stand on any failure
                for b_ in (128, 256):
                    st["sel"].pop(("router", b_), None)
                print("PREP rs bake failed: %s" % str(error).splitlines()[-1][:200], flush=True)
            finally:  # critic (pw21): never leave a capture open or a side stream busy for the judge's own captures
                if torch.cuda.is_current_stream_capturing():
                    print("PROBE l2scratch capture_left_open", flush=True)
                torch.cuda.synchronize()
        print("PREP phase router done at %.0fs" % (time.time() - _T0), flush=True)  # pw36: phase timestamps
        st.pop("gvc12", None)
        st.pop("gepk", None)  # pw17: unpicked codec E packs (a picked plan keeps its own pack alive)
        print("PREP done in %.0fs" % (time.time() - _T0), flush=True)
        return st


    # ---------------------------------------------------------------- pw11 second bake (template conventions)
    PREPARE_BUDGET_S = 240.0
    MARGIN = 0.99
    MARGIN_RQ128 = 0.98  # pw52: a variant must beat the proven m=128 pick by 2% in-bake
    # pw60 (wf_stack4): m=128 layers whose pick is forced to the per-thread escape cursor variant (see the bake).
    RQPICK128 = {("pooler", 128): "d2s5hzwP", ("qkv_gate", 128): "d2s5hzwP", ("unpooler", 128): "d2s5hzwP"}
    # pw62 (wf_stack5): m=32 layers whose pick is forced to the NT=32 escape-cursor twin at the judge-proven K-split
    # (pw58/pw59 judge checks: pooler t32s6hze-k4, qkv_gate t32s6hze-k7, unpooler t32s6hze-k2). The judge's in-prepare
    # bake flips m=32 picks on noise (FACTS 14) and the cursor is a fixed-latency win the cold-L2 bake cannot resolve
    # (wf_escur32 W-resident: pooler -1.36/-1.38, qkv_gate -0.74/-0.70, unpooler -0.95/-1.17 us; streamed 0.00..+0.08).
    RQPICK32 = {("pooler", 32): "t32s6hzw-k4", ("qkv_gate", 32): "t32s6hzw-k7", ("unpooler", 32): "t32s6hzw-k2"}
    # pw63 (wf_escur16): the same forcing at NT=16, qkv_gate only (pooler/unpooler/o m=16 run on gv on the judge: rq NT=16
    # lost there by 1-2 us). K-split 7 = the judge's m=16 pick. Box lab16 (qkv_gate m=16, ks 7, 30 reps, W-resident = the
    # judge's regime): t16s5hzw 7.63 vs t16s5hze 8.29 (-0.66), t16s4hzw 8.32 vs t16s4hze 8.61 (-0.29), h16s5hzw2 7.62 vs
    # h16s5hze2 8.48 (-0.86); streamed -0.09/-0.13/-0.27. ST5 beats ST4 by 0.69 us W-resident on both walks, so the forced
    # twin is the ST5 one (the judge's own m=16 bake has flipped between t16s4hze and t16s5hze across runs).
    RQPICK16 = {("qkv_gate", 16): "t16s5hzw-k7"}
    # pw63 (coordinator, 20:05): o m=32 -> the HALF 2-CTA twin at K-split 2 (the judge's own pw58 bake picked it; box
    # resident screen 5.44 vs t32s6hze-k2 6.08 = -0.64 us, one-hot exact, regs 128). Not a cursor variant (FL 1075).
    RQPICK32B = {("o", 32): "h32s5hze2-k2"}
    MARGIN_RQ32 = 0.97  # pw52: ... and the proven m<=32 pick by 3%
    MARGIN_RS = 0.98  # pw52: ... and the proven router m=128/256 pick by 2%
    MARGIN_GX = 0.97  # pw52: ... and the proven gx (unpooler m1/m8) pick by 3%
    TOL = {"bf16": 4e-3, "f32": 2e-5}
    _EYE = {}


    # ---------------------------------------------------------------- pw55: judge-faithful bake measurement
    # FACTS 14: the bake's torch.sum flush left x and small weights partly L2-warm, so shallow rings / small-shape variants
    # measured 0.2-2.2 us BETTER in-bake and WORSE officially. pw55 ports judge_sim's method into the bake:
    #   (1) a vectorized READ flush over a >= 2x L2 buffer before every timed replay, validated at prepare start with
    #       ld.global probes that must run at <= HBM rate (judge_sim.validate_flush);
    #   (2) every candidate timed ALTERNATING with its group's baseline path (r6 or the proven pick), order swapped every
    #       pair, ranked by the median of paired ratios with the graph fixed cost subtracted (judge_sim.paired_ratio);
    #   (3) the same 2% (m=128) / 3% (m<=32) margins; fewer reps per path (PAIRS) to stay inside the prepare budget.
    FLUSH_MB = 128  # >= 2x L2 (50 MB on the H100/H200)
    FLUSH_BLOCKS, FLUSH_THREADS = 1056, 256
    HBM_RATE_TBPS = 3.35  # a cold probe must not beat this (judge_sim.HBM_RATE_TBPS)
    PROBE_BYTES = 6_300_032
    DISCRIM_BYTES = 16 << 20  # the probe the flush is validated on (6.3 MB is latency-bound and cannot discriminate)
    PAIRS = 8  # timed pairs per (candidate, baseline) comparison

    FLUSH_CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

// READ flush: every 16-byte line is loaded (default .ca policy, so it allocates in L2 and evicts whatever was there).
// The XOR accumulator is kept live by a comparison that is never true in practice, so no store is needed.
__global__ void flush_read_kernel(const uint4* __restrict__ p, long n4, float* sink) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const long stride = (long)gridDim.x * blockDim.x;
  unsigned a = 0u, b = 0u, c = 0u, d = 0u;
  for (; i < n4; i += stride) {
    uint4 v = p[i];
    a ^= v.x; b ^= v.y; c ^= v.z; d ^= v.w;
  }
  if ((a & b & c & d) == 0xFFFFFFFFu) sink[0] = 1.0f;
}

// probe: same access pattern, result written so the loads are live.
__global__ void probe_read_kernel(const uint4* __restrict__ p, long n4, float* out) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const long stride = (long)gridDim.x * blockDim.x;
  unsigned a = 0u, b = 0u, c = 0u, d = 0u;
  for (; i < n4; i += stride) {
    uint4 v = p[i];
    a ^= v.x; b ^= v.y; c ^= v.z; d ^= v.w;
  }
  if ((a & b & c & d) == 0xFFFFFFFFu) out[0] = 1.0f;
}

// empty kernel for the graph fixed-cost measurement (132 CTAs, like the judge's probe graph).
__global__ void nop_kernel(float* p) {
  if (threadIdx.x == 1024) p[0] = 1.0f;
}

void flush_read(torch::Tensor buf, torch::Tensor sink, int64_t blocks, int64_t threads) {
  long n4 = (long)(buf.numel() * buf.element_size() / 16);
  flush_read_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>((const uint4*)buf.data_ptr(), n4, sink.data_ptr<float>());
}
void probe_read(torch::Tensor buf, torch::Tensor out, int64_t blocks, int64_t threads) {
  long n4 = (long)(buf.numel() * buf.element_size() / 16);
  probe_read_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>((const uint4*)buf.data_ptr(), n4, out.data_ptr<float>());
}
void nop(torch::Tensor out, int64_t blocks, int64_t threads) {
  nop_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(out.data_ptr<float>());
}
"""

    FLUSH_CPP = r"""
#include <torch/extension.h>
void flush_read(torch::Tensor buf, torch::Tensor sink, int64_t blocks, int64_t threads);
void probe_read(torch::Tensor buf, torch::Tensor out, int64_t blocks, int64_t threads);
void nop(torch::Tensor out, int64_t blocks, int64_t threads);
"""

    _flext = [None]


    def _flushext():
        if _flext[0] is None:
            t0 = time.time()
            _flext[0] = load_inline(name="pw57_flush", cpp_sources=FLUSH_CPP, cuda_sources=FLUSH_CUDA,
                                    functions=["flush_read", "probe_read", "nop"],
                                    extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
            print("BUILD pw57_flush %.1fs" % (time.time() - t0), flush=True)
        return _flext[0]


    class _Flusher:
        """A validated cold-L2 flush: a READ of a buffer larger than L2, before every replay (judge_sim.Flusher)."""

        def __init__(self, mb=FLUSH_MB):
            self.ext = _flushext()
            self.buf = torch.empty(mb << 20, dtype=torch.uint8, device="cuda")
            self.sink = torch.zeros((), dtype=torch.float32, device="cuda")

        def __call__(self):
            self.ext.flush_read(self.buf, self.sink, FLUSH_BLOCKS, FLUSH_THREADS)


    _FL = [None]  # the flusher
    _OV = [0.0]   # graph fixed cost (us): the judge's 2*one_node - two_node
    _LAST = [None]  # (ratio, base_med, cand_med, ov) of the last paired comparison, for the BAKE lines


    def _flush():
        if _FL[0] is None:
            _FL[0] = _Flusher()
        return _FL[0]


    def _ev_median(fn, reps=20, fl=True):
        ts = []
        for _ in range(reps):
            if fl:
                _flush()()
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record(); fn(); b.record(); torch.cuda.synchronize()
            ts.append(a.elapsed_time(b) * 1e3)
        ts.sort()
        return ts[len(ts) // 2]


    def _flush_validate():
        """judge_sim.validate_flush: ld.global probes after the flush must run at <= HBM rate (the 6.3 MB probe is
    latency-bound and cannot discriminate; the pick is made on the 16 MB one)."""
        ext = _flushext()
        _flush()
        out = torch.zeros((), dtype=torch.float32, device="cuda")
        print("FLUSH_VALIDATE probes %.2f MB and %.2f MB; cold must be <= %.2f TB/s (pick on the larger probe)"
              % (PROBE_BYTES / 1e6, DISCRIM_BYTES / 1e6, HBM_RATE_TBPS), flush=True)
        ok_all = True
        for nb in (PROBE_BYTES, DISCRIM_BYTES):
            buf = torch.empty(nb, dtype=torch.uint8, device="cuda")
            buf.view(torch.float32).normal_()
            fn = lambda: ext.probe_read(buf, out, 528, 256)
            t = _ev_median(fn, fl=False)
            print("FLUSH_VALIDATE %-16s %6.2f MB %7.2f us %6.2f TB/s" % ("warm(no flush)", nb / 1e6, t,
                                                                         nb / 1e9 / (t * 1e-6) / 1e3), flush=True)
            t = _ev_median(fn)
            tb = nb / 1e9 / (t * 1e-6) / 1e3
            ok = tb <= HBM_RATE_TBPS
            if nb == DISCRIM_BYTES:
                ok_all = ok
            print("FLUSH_VALIDATE %-16s %6.2f MB %7.2f us %6.2f TB/s  %s"
                  % ("read/%dMB" % FLUSH_MB, nb / 1e6, t, tb, "COLD ok" if ok else "WARM reject"), flush=True)
        print("FLUSH_PICK read/%d %s" % (FLUSH_MB, "ok" if ok_all else "WARNING no cold flush"), flush=True)
        return ok_all


    def _capture(fn):
        """capture a CUDA graph of the zero-arg callable fn (two warm-up calls on a side stream, judge_sim.capture).
    pw55: capture_begin/capture_end directly instead of the `torch.cuda.graph` context manager, which calls
    torch.cuda.empty_cache() + _host_emptyCache() on entry -- measured 28 ms per capture on the box (vs 1.2 ms
    here), i.e. ~150 s over the bake's ~1000 captures. The tensors a graph touches are held by the bake's scope."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn(); fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(s):
            g.capture_begin()
            fn()
            g.capture_end()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        return g


    def _replay(graphs, pairs):
        """pairs replays per graph, order swapped every pair, flush before every replay (judge_sim.replay_ns).
    Returns {name: [us]}."""
        names = list(graphs)
        ev = {n: [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(pairs)] for n in names}
        for p in range(pairs):
            for n in (names if p % 2 == 0 else names[::-1]):
                _flush()()
                a, b = ev[n][p]
                a.record(); graphs[n].replay(); b.record()
        torch.cuda.synchronize()
        return {n: [a.elapsed_time(b) * 1e3 for a, b in ev[n]] for n in names}


    def _fixed_cost(pairs=PAIRS):
        """A graph replay's fixed cost, measured the way the judge does: 2*one_node - two_node."""
        ext = _flushext()
        out = torch.zeros((), dtype=torch.float32, device="cuda")
        one = _capture(lambda: ext.nop(out, 132, 128))
        two = _capture(lambda: (ext.nop(out, 132, 128), ext.nop(out, 132, 128)))
        t = _replay({"one": one, "two": two}, pairs)
        o, w = statistics.median(t["one"]), statistics.median(t["two"])
        _OV[0] = 2 * o - w
        print("FIXED_COST one_node %.2f us two_node %.2f us overhead %.2f us" % (o, w, _OV[0]), flush=True)
        return _OV[0]


    def _paired(fn, base, pairs=PAIRS):
        """fn and base captured as graphs and replayed alternately; returns (ratio, base_med, cand_med, ov) with
    ratio = median over pairs of sqrt(r_a*r_b), r = (base-fixed)/(cand-fixed) (judge_sim.paired_ratio)."""
        ga, gb = _capture(fn), _capture(base)
        t = _replay({"a": ga, "b": gb}, pairs)
        ta, tb = t["a"], t["b"]
        ov = _OV[0]
        lim = 0.5 * min(statistics.median(ta), statistics.median(tb))
        if ov > lim:  # never let the subtraction reach the times (a bad fixed-cost read must not blow up a ratio)
            print("PAIR ov %.2f clamped to %.2f (times %.2f/%.2f)" % (ov, lim, statistics.median(ta),
                                                                     statistics.median(tb)), flush=True)
            ov = max(0.0, lim)
        r = [(b - ov) / max(a - ov, 1e-3) for a, b in zip(ta, tb)]  # (base-ov)/(cand-ov), the judge's r
        pr = [math.sqrt(r[i] * r[i + 1]) for i in range(0, len(r) - 1, 2)]
        return statistics.median(pr), statistics.median(tb), statistics.median(ta), ov


    def _time_us(fn, x, out, flush, sink, it=15, base=None):
        """pw55: base=None -> absolute median replay time (us) after the validated cold-L2 read flush.
    base=fn2 -> the judge's paired form: fn timed ALTERNATING with fn2 (order swapped every pair, flush before every
    replay), ranked by the median of sqrt(r_a*r_b) with r = (base-fixed)/(cand-fixed); returns base_median/ratio (us),
    i.e. fn's time expressed in the baseline's units. `flush`/`sink` are kept for call-site compatibility."""
        if base is None:
            g = _capture(lambda: fn(x, out))
            ts = _replay({"a": g}, it)["a"]
            ts.sort()
            return ts[len(ts) // 2]
        ratio, bmed, cmed, ov = _paired(lambda: fn(x, out), lambda: base(x, out), max(2, it // 2))
        _LAST[0] = (ratio, bmed, cmed, ov)
        return bmed / ratio


    RWIN128 = 1.01   # pw59 (wf_bake3): a proven pick is a group's reference only if the box does not read it slower than
    RWIN32 = 1.015   # r6 by more than HALF the group's margin (2% -> 1%, 3% -> 1.5%); otherwise r6 is the reference and
    RWIN_RS = 1.01   # the proven pick is an ordinary candidate. pw58 used a 5% window, which let a 3.6%-slower proven pick
    RWIN_GX = 1.015  # become the reference -- then no candidate could ever lose to r6 (the qkv_gate m=256 failure).
    GWIN = 1.015     # the gv bake's window (half of GVMARGIN)


    def _refsel(prov_name, prov_med, base_name, base_med, window):
        """pw59: the group's reference. The proven pick when the box does not read it slower than the base (r6) by more
    than `window`, else the base. Returns (name, med, is_prov)."""
        if prov_name is not None and prov_med is not None and prov_med <= base_med * window:
            return prov_name, prov_med, True
        return base_name, base_med, False


    def _pickrule(ref, cands, margin):
        """pw59: the group's pick from ONE reference. ref = (name, plan, med); cands = [(name, plan, ratio)] with
    ratio = (ref-fixed)/(cand-fixed) > 1 meaning the candidate is faster than the reference -- every ratio in a group
    is measured against the same reference (pw58 chained them: candidate 2 against candidate 1, then compared the
    ratios as if they shared a reference). Returns (name, plan, med): the candidate with the best ratio if it beats
    the reference by `margin` (ratio > 1/margin), else the reference itself."""
        best = None
        for nm, plan, r in cands:
            if best is None or r > best[2]:
                best = (nm, plan, r)
        if best is not None and best[2] > 1.0 / margin:
            return best[0], best[1], ref[2] / best[2]
        return ref


    def _validate(fn, w, m, n, f32, gen, exact=None, m_odd=None):
        """template _validate: one-hot rows x8 bit-exact against the weight (for the router: against hi + lo, which is what
    the lossless 12-bit hi/lo pack carries), random rows within TOL of FP64 on the true weight; plus random rows at an
    odd m of the same bucket (kernel_fn serves the whole bucket with this plan)."""
        K = w.shape[1]
        dt = torch.float32 if f32 else torch.bfloat16
        out = torch.empty((m, n), dtype=dt, device=w.device)
        exact = (exact if exact is not None else w)[:n].t().to(dt)  # [K, n]
        rows = max(m, 1)
        if K not in _EYE:
            _EYE[K] = torch.eye(K, dtype=torch.bfloat16, device=w.device) * 8.0
        eye = _EYE[K]
        for start in range(0, K, rows):
            x = eye[start:start + rows]
            if x.shape[0] < rows:
                x = torch.cat((x, torch.zeros((rows - x.shape[0], K), dtype=x.dtype, device=x.device)))
            fn(x, out)
            got = out[: min(rows, K - start)]
            want = exact[start:start + got.shape[0]] * 8.0
            if not torch.isfinite(got).all() or not torch.equal(got.to(torch.float32), want.to(torch.float32)):
                return False, "one-hot mismatch at column %d" % start
        tol = TOL["f32" if f32 else "bf16"]
        for mm in (rows,) + ((m_odd,) if m_odd else ()):
            x = torch.randn((mm, K), generator=gen, device=w.device).to(torch.bfloat16)
            o = torch.empty((mm, n), dtype=dt, device=w.device)
            fn(x, o)
            ref = x.double() @ w[:n].double().t()
            err = float((o.double() - ref).norm() / ref.norm())
            row = float(((o.double() - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-30)).max())
            if not (err <= tol and row <= 5 * tol and torch.isfinite(o).all()):
                return False, "random m=%d rel L2 %.2e row %.2e > %.0e" % (mm, err, row, tol)
        return True, "ok"


    # ---------------------------------------------------------------- pw18: codec E v3 packer + reference unpacker
    # (copied verbatim from proj_rsi/tools/codec_e.py; spec proj_rsi/tools/CODEC_E_LAYOUT.md v3)
    ESC_MAX = 4e-4          # target escape rate (weights); keeps the 32-entry per-warp list far from overflow
    LMAX = 8                # code length limit: 4 pairs per step <= 32 bits (one refill per step suffices)
    PK = 3                  # v3: the first PK bits of a code (LSB-first) determine its length (8-entry byte table, one PRMT)


    def _package_merge(cnt, L):
        """optimal prefix-code lengths (max length L) for positive counts (python list)."""
        n = len(cnt)
        if n == 1:
            return [1]
        leaves = sorted((float(q), (i,)) for i, q in enumerate(cnt))
        cur = list(leaves)
        for _ in range(L - 1):
            pk = [(cur[2 * i][0] + cur[2 * i + 1][0], cur[2 * i][1] + cur[2 * i + 1][1]) for i in range(len(cur) // 2)]
            cur = sorted(leaves + pk, key=lambda t: t[0])
        ln = [0] * n
        for _, syms in cur[:2 * n - 2]:
            for s in syms:
                ln[s] += 1
        return ln


    def _canonical_lsb(lengths):
        """canonical codes, bit-reversed: the code of length l is the low l bits of the LSB-first peek window."""
        order = sorted(range(len(lengths)), key=lambda s: (lengths[s], s))
        code, prev, out = 0, None, [0] * len(lengths)
        for s in order:
            l = lengths[s]
            if prev is not None:
                code = (code + 1) << (l - prev)
            prev = l
            out[s] = int(format(code, "0%db" % l)[::-1], 2)
        return out


    def _geo(geo):
        wr, wk, ku, s, rt = geo[:5]
        return wr, wk, ku, s, rt


    def _to_segments(x, T, KC, geo):
        """[T, KC, 4, 32, 4, ...] (tile, chunk, step j, lane, reg-in-step i) -> [NSEG, 32, KU*RT*16, ...] in segment
    (gw) order and lane-stream order (u, jt, r = 4 j + i)."""
        wr, wk, ku, s, rt = _geo(geo)
        gx = T // (wr * rt)
        rest = x.shape[5:]
        x = x.reshape(gx, wr, rt, s, wk, ku, 4, 32, 4, *rest)
        nd = len(rest)
        x = x.permute(3, 0, 4, 1, 7, 5, 2, 6, 8, *range(9, 9 + nd))  # by, bx, wk, wr, lane, u, jt, j, i
        return x.reshape(s * gx * wk * wr, 32, ku * rt * 16, *rest)


    def _gvesc_e(t, c, j, l, v, x, T, geo, dev):
        """r6's _gvesc structure with an 8-bit exponent XOR (bits 0-7; the kernel applies x << (7 + 16 * (v & 1)))."""
        wr, wk, ku, s, rt = _geo(geo)
        gx, nw = T // (wr * rt), wr * wk
        out = torch.full((gx * s * nw * 32,), -1, dtype=torch.int64, device=dev)
        if len(t):
            gw = ((c // (wk * ku)) * gx + t // (wr * rt)) * nw + ((c // ku) % wk) * wr + (t // rt) % wr
            seq = (c % ku) * rt + t % rt
            o = torch.argsort(gw * 4096 + seq)
            gw, ent = gw[o], ((seq << 20) | (l << 15) | ((j * 4 + v // 2) << 11) | ((v % 2) << 10) | x)[o]
            rank = torch.arange(len(gw), device=dev) - torch.searchsorted(gw, gw)
            if int(rank.max()) >= 32:
                return None
            out[gw * 32 + rank] = ent
        return torch.where(out >= 2 ** 31, out - 2 ** 32, out).int()


    def _prefix_code(cnt, k=PK, L=LMAX):
        """optimal prefix code (max length L) in which the first k bits of every code determine its length: codes of length
    <= k, and buckets (one k-bit prefix each) of 2^(Lb - k) codes of length Lb. cnt: python list (any order).
    Returns (lengths, codes LSB-first, tab[2^k] = length per k-bit window)."""
        n, P = len(cnt), 2 ** k
        order = sorted(range(n), key=lambda s: (-cnt[s], s))
        p = [float(cnt[s]) for s in order]
        cum = [0.0]
        for x in p:
            cum.append(cum[-1] + x)
        INF = float("inf")
        D = [[INF] * (P + 1) for _ in range(n + 1)]
        B = [[None] * (P + 1) for _ in range(n + 1)]
        D[0][0] = 0.0
        for i in range(n):
            for u in range(P + 1):
                d = D[i][u]
                if d == INF:
                    continue
                for l in range(1, k + 1):
                    w = 2 ** (k - l)
                    if u + w <= P and d + p[i] * l < D[i + 1][u + w]:
                        D[i + 1][u + w], B[i + 1][u + w] = d + p[i] * l, (i, u, "s", l)
                if u + 1 <= P:
                    for lb in range(k + 1, L + 1):
                        j = min(n, i + 2 ** (lb - k))
                        v = d + (cum[j] - cum[i]) * lb
                        if v < D[j][u + 1]:
                            D[j][u + 1], B[j][u + 1] = v, (i, u, "b", lb)
        u = min(range(P + 1), key=lambda q: D[n][q])
        assert D[n][u] < INF, "no prefix code fits"
        items, i = [], n
        while i > 0:
            pi, pu, kind, l = B[i][u]
            items.append((kind, l, order[pi:i]))
            i, u = pi, pu
        items.reverse()
        # canonical assignment in the k-bit prefix space (virtual length: l for short codes, k for buckets), then bit-reversed
        items.sort(key=lambda it: it[1] if it[0] == "s" else k)
        lengths, codes, tab = [0] * n, [0] * n, [L] * P
        code, prev = 0, None
        for kind, l, syms in items:
            vl = l if kind == "s" else k
            if prev is not None:
                code = (code + 1) << (vl - prev)
            prev = vl
            pre = int(format(code, "0%db" % vl)[::-1], 2)
            if kind == "s":
                lengths[syms[0]], codes[syms[0]] = l, pre
                for w in range(P):
                    if w & (2 ** l - 1) == pre:
                        tab[w] = l
            else:
                tab[pre] = l
                for j, s in enumerate(syms):
                    lengths[s], codes[s] = l, pre | (j << k)
        return lengths, codes, tab


    def _wpp(ilp):
        """words per ring piece. v3: 4 B pieces at every ilp (padding to a word, not 16 B: ilp 1/2/4 pad 0.12/0.22/0.36
    bits/w against 0.16/0.55/0.36 with 16 B pieces); the code supports 16 B pieces with per-lane word rotation"""
        return 1


    def _substreams(x, ilp):
        """[NSEG, 32, KU*RT*16, ...] (unit, register) -> [NSEG, ilp, 32, KU*RT*16/ilp, ...]: substream s holds steps
    j in [s*4/ilp, (s+1)*4/ilp) of every unit, in unit order"""
        nseg, _, n = x.shape[:3]
        rest = x.shape[3:]
        x = x.reshape(nseg, 32, n // 16, ilp, 16 // ilp, *rest)
        x = x.permute(0, 3, 1, 2, 4, *range(5, 5 + len(rest)))
        return x.reshape(nseg, ilp, 32, n // ilp, *rest)


    def pack_e(fr, geo, layer="?", verbose=True, ilp=1):
        """v3. r6 fragments [T, KC, 4, 32, 8] bf16 + gv geometry (WR, WK, KU, S, RT, ...) -> dict or None (escape overflow).
    keys: plane uint8 [T, KC, 2, 32, 16]; lut int32 [258] (256 value entries + the 8-byte length table); stream int32;
    soff int32 [NSEG + 1] (in 4 B words; a segment = ilp substream regions of P * 32 pieces of 16 B, 4 B at ilp 4);
    esc int32 [NSEG * 32]; ilp; stats dict."""
        t0 = time.time()
        dev = fr.device
        T, KC = fr.shape[:2]
        wr, wk, ku, s, rt = _geo(geo)
        assert wk * ku * s == KC and T % (wr * rt) == 0 and ilp in (1, 2, 4), (geo, T, KC, ilp)
        b = fr.reshape(T, KC, 4, 32, 8).contiguous().view(torch.int16).int() & 0xffff
        e8 = (b >> 7) & 0xff
        plane = ((b >> 8) & 0x80) | (b & 0x7f)
        n = b.numel()

        cnt = torch.bincount(e8.flatten(), minlength=256)
        val = torch.argsort(cnt, descending=True, stable=True)
        rank = torch.empty(256, dtype=torch.long, device=dev)
        rank[val] = torch.arange(256, device=dev)
        rk = rank[e8].view(T, KC, 4, 32, 4, 2)
        pid = rk[..., 0] * 256 + rk[..., 1]
        pc = torch.bincount(pid.flatten(), minlength=65536)
        order = torch.argsort(pc, descending=True, stable=True)
        npos = int((pc > 0).sum())
        pcf = pc.view(256, 256)
        N = None
        for cand in list(range(16, 256, 8)) + [256]:
            cand = min(cand, npos)
            C = torch.zeros(65536, dtype=torch.bool, device=dev)
            C[order[:cand]] = True
            C = C.view(256, 256)
            one = C.any(1)[:, None] | C.any(0)[None, :]
            esc = int((pcf * (~C) * (2 - one.long())).sum())
            if esc <= ESC_MAX * n or cand == npos:
                N = cand
                break
        syms = order[:N]
        C = torch.zeros(65536, dtype=torch.bool, device=dev)
        C[syms] = True
        Cm = C.view(256, 256)
        a_i = torch.arange(256, device=dev)
        rowbest = torch.where(Cm, pcf, -1).argmax(1)
        colbest = torch.where(Cm, pcf, -1).argmax(0)
        A, B = torch.meshgrid(a_i, a_i, indexing="ij")
        ph = torch.where(Cm, A * 256 + B,
                         torch.where(Cm.any(1)[:, None], A * 256 + rowbest[:, None],
                                     torch.where(Cm.any(0)[None, :], colbest[None, :] * 256 + B, syms[0].expand(256, 256))))
        ph = ph.flatten()
        eff = torch.bincount(ph[pid.flatten()], minlength=65536)
        symlist = syms.tolist()
        ln, code, tab = _prefix_code([max(int(eff[x]), 1) for x in symlist])
        sym_len = torch.zeros(65536, dtype=torch.long, device=dev)
        sym_code = torch.zeros(65536, dtype=torch.long, device=dev)
        sym_len[syms] = torch.tensor(ln, device=dev)
        sym_code[syms] = torch.tensor(code, device=dev)
        x = torch.arange(256, device=dev)
        sv = torch.tensor(symlist, device=dev)
        ent = (val[sv // 256] << 7) | (val[sv % 256] << 23) | torch.tensor(ln, device=dev)
        hit = (x[None, :] & (2 ** torch.tensor(ln, device=dev)[:, None] - 1)) == torch.tensor(code, device=dev)[:, None]
        assert bool((hit.sum(0) <= 1).all())
        lut = (hit.long() * ent[:, None]).sum(0)
        tabv = sum(t << (8 * i) for i, t in enumerate(tab))
        lut = torch.cat((lut, torch.tensor([tabv & 0xffffffff, tabv >> 32], device=dev)))

        php = ph[pid]
        pr = torch.stack((php // 256, php % 256), -1)
        mis = pr != rk
        ei = mis.nonzero()
        t_, c_, j_, l_, i_, h_ = ei.unbind(1)
        xr = (val[rk[mis]] ^ val[pr[mis]]).long()
        escl = _gvesc_e(t_, c_, j_, l_, 2 * i_ + h_, xr, T, geo, dev)
        if escl is None:
            if verbose:
                print("STATS_E %s escape list overflow (esc=%.5f)" % (layer, len(ei) / n))
            return None

        L = _substreams(_to_segments(sym_len[php], T, KC, geo), ilp)                 # [NSEG, ilp, 32, NP]
        V = _substreams(_to_segments(sym_code[php], T, KC, geo), ilp)
        nseg = L.shape[0]
        wpp = _wpp(ilp)                                                              # words per piece: 16 B, or 4 B at ilp 4
        off = torch.cumsum(L, 3) - L
        tot = L.sum(3)                                                               # [NSEG, ilp, 32]
        P = (tot.flatten(1).max(1).values + 32 * wpp - 1) // (32 * wpp)              # pieces per lane per substream
        soff = torch.zeros(nseg + 1, dtype=torch.long, device=dev)                   # in words (4 B)
        soff[1:] = torch.cumsum(P * 32 * ilp * wpp, 0)
        nwords = int(soff[-1])
        wd, sh = off >> 5, off & 31
        lane = torch.arange(32, device=dev)[None, None, :, None]
        sub = torch.arange(ilp, device=dev)[None, :, None, None]
        rot = ((lane >> 3) & 3) if wpp == 4 else 0
        gword = lambda w: soff[:-1, None, None, None] + ((sub * P[:, None, None, None] + w // wpp) * 32 + lane) * wpp + ((w % wpp) ^ rot)
        words = torch.zeros(nwords + 4, dtype=torch.long, device=dev)
        words.scatter_add_(0, gword(wd).flatten(), ((V << sh) & 0xffffffff).flatten())
        spill = (sh + L) > 32
        words.scatter_add_(0, gword(wd + 1)[spill], (V >> (32 - sh))[spill])
        words = words[:nwords]
        stream = torch.where(words >= 2 ** 31, words - 2 ** 32, words).int()

        pl = plane.to(torch.uint8)
        p0 = pl[:, :, 0:2].permute(0, 1, 3, 2, 4).reshape(T, KC, 32, 16)
        p1 = pl[:, :, 2:4].permute(0, 1, 3, 2, 4).reshape(T, KC, 32, 16)
        planeb = torch.stack((p0, p1), 2).contiguous()
        lut32 = torch.where(lut >= 2 ** 31, lut - 2 ** 32, lut).int()

        stream_bits = nwords * 32
        pad_bits = stream_bits - int(tot.sum())
        total_bits = 8 * n + stream_bits + 32 * (nseg + 1) + 32 * 258 + 32 * escl.numel()
        st = dict(bits=total_bits / n, esc=len(ei) / n, maxpad=pad_bits / stream_bits, maxpad_bits=pad_bits / n,
                  N=N, code_bits=float(tot.sum()) / n, secs=time.time() - t0)
        if dev.type == "cuda":
            torch.cuda.synchronize()
            st["secs"] = time.time() - t0
        if verbose:
            print("STATS_E %s ilp%d bits=%.4f esc=%.5f maxpad=%.4f (code %.4f, pad %.4f bits/w, alphabet %d, len tab %s, %.2f s)"
                  % (layer, ilp, st["bits"], st["esc"], st["maxpad"], 8 + st["code_bits"], st["maxpad_bits"], N,
                     "".join(map(str, tab)), st["secs"]))
        return dict(plane=planeb, lut=lut32, stream=stream, soff=soff.int(), esc=escl, ilp=ilp, stats=st)


    def unpack_e(pk, T, KC, geo, ED=16):
        """v3 reference decoder: the device routine's logic per lane and substream (64-bit buffer, refill at step start when
    (V & 0x60) == 0, length = tab[lo & 7] (no memory on the chain), value = lut[lo & 0xff], V -= length), all lanes of
    all segments at once. Replays the per-substream smem ring protocol (ED words per lane in total, ED / K slots per
    substream; prologue: 3 rounds of NI = 4 / K predicated issues per substream + commit; per unit: one round + commit +
    wait_group(2)) and asserts every refill word (incl. the kernel's prefetched ones) is read from a landed, not yet
    overwritten piece. Returns bf16 bit patterns [T, KC, 4, 32, 8]."""
        dev = pk["stream"].device
        wr, wk, ku, s, rt = _geo(geo)
        K = pk["ilp"]
        NS = 4 // K
        M32 = 0xffffffff
        words = pk["stream"].long() & M32
        lut = pk["lut"].long() & M32
        tab = torch.tensor([((int(lut[256 + i // 4]) >> (8 * (i % 4))) & 255) for i in range(8)], device=dev)
        lut = lut[:256]
        soff = pk["soff"].long()
        nseg = soff.numel() - 1
        wpp = _wpp(K)
        P = (soff[1:] - soff[:-1]) // (32 * K * wpp)
        lane = torch.arange(32, device=dev)[None, None, :]
        sub = torch.arange(K, device=dev)[None, :, None]
        rot = ((lane >> 3) & 3) if wpp == 4 else 0
        z = torch.zeros(nseg, K, 32, dtype=torch.long, device=dev)
        lo, hi, Vr, ptr, iss = z.clone(), z.clone(), z.clone(), z.clone(), z.clone()
        NU = ku * rt
        E = torch.zeros(nseg, K, 32, NU, NS, 4, dtype=torch.long, device=dev)
        Pl = P[:, None, None].expand_as(z)
        groups = []
        ES, NI = ED // K, NS                                                         # slots per substream, issues per unit

        def issue():
            nonlocal iss
            for _ in range(NI):
                go = (iss < Pl) & (iss < ptr // wpp + ES)
                iss = iss + go.long()
            groups.append(iss.clone())

        for _ in range(3):
            issue()
        for un in range(NU):
            issue()
            landed = groups[len(groups) - 1 - 2]
            for t in range(NS):
                need = (Vr & 0x60) == 0
                pc = ptr // wpp                                                    # ptr counts words
                inb = pc < Pl
                rd = need & inb                                                    # words the decode uses (other reads are masked)
                assert bool((pc < landed)[rd].all()), "ring: piece not landed"
                assert bool((pc + ES > iss - 1)[rd].all()), "ring: slot overwritten"
                gi = soff[:-1, None, None] + ((sub * P[:, None, None] + pc) * 32 + lane) * wpp + ((ptr % wpp) ^ rot)
                w = torch.where(need & inb, words[gi.clamp(max=words.numel() - 1)], 0)
                at = Vr & 31
                lo = torch.where(need, lo | ((w << at) & M32), lo)
                hi = torch.where(need, hi | ((w >> (32 - at)) * (at > 0)), hi)
                Vr = torch.where(need, (Vr + 32) & M32, Vr)
                ptr = ptr + need.long()
                for q in range(4):
                    n_ = tab[lo & 7]
                    E[..., un, t, q] = lut[lo & 0xff]
                    lo, hi = ((lo >> n_) | (hi << (32 - n_))) & M32, hi >> n_
                    Vr = (Vr - n_) & M32
        # [nseg, K, 32, NU, NS, 4] -> [nseg, 32, NU, 4 j, 4 i] -> fragment order
        E = E.permute(0, 2, 3, 1, 4, 5).reshape(nseg, 32, NU, 4, 4)
        gx = T // (wr * rt)
        E = E.view(s, gx, wk, wr, 32, ku, rt, 4, 4).permute(1, 3, 6, 0, 2, 5, 7, 4, 8).reshape(T, KC, 4, 32, 4)
        pl = pk["plane"].long()
        pb = pl.view(T, KC, 2, 32, 2, 8).permute(0, 1, 2, 4, 3, 5).reshape(T, KC, 4, 32, 8)
        lowh, hih = pb[..., 0::2], pb[..., 1::2]
        dup0, dup1 = lowh | (lowh << 8), hih | (hih << 8)
        m = 0x807f
        v0 = (E & ~m & 0xffff) | (dup0 & m)
        v1 = ((E >> 16) & ~m & 0xffff) | (dup1 & m)
        vals = torch.stack((v0, v1), -1).reshape(T, KC, 4, 32, 8)
        esc = pk["esc"].long() & M32
        ent = esc[esc != M32]
        if len(ent):
            gwi = (esc != M32).nonzero().flatten() // 32
            seq, ln_, r, half, x = ent >> 20, (ent >> 15) & 31, (ent >> 11) & 15, (ent >> 10) & 1, ent & 0xff
            nw = wr * wk
            wwr, wwk, g2 = gwi % nw % wr, gwi % nw // wr, gwi // nw
            bx, by = g2 % gx, g2 // gx
            t = (bx * wr + wwr) * rt + seq % rt
            c = (by * wk + wwk) * ku + seq // rt
            v = (r % 4) * 2 + half
            vals[t, c, r // 4, ln_, v] ^= x << 7
        return vals.int()


    def _frag_bits(fr):
        return fr.reshape(*fr.shape[:2], 4, 32, 8).contiguous().view(torch.int16).int() & 0xffff


    def _gepack(st, layer, inst, n):
        """codec E v3 pack for one geometry and substream count, proven lossless by the reference unpacker (incl. the
    per-substream ring protocol at the instance's ring size) before any kernel uses it. pack_e runs once per (layer,
    rows, geometry, substreams). Returns the pack dict or None (escape overflow, >= 12 bits, or a failed proof)."""
        fmt, mt, stg, wr, wk, ku, s, rt, xs, v = inst
        ed = 32 if (v >> 19) & 1 else 16
        ilp = 1 << ((v >> 22) & 3)
        geo = (wr, wk, ku, s, rt)
        cache = st.setdefault("gepk", {})
        key = (layer, n, geo, ilp, ed)
        if key in cache:
            return cache[key]
        t0 = time.time()
        fr = st[layer][0][: n // 16]  # unpooler at m <= 8 serves rows 0..1023 only
        T, KC = fr.shape[:2]
        tag = "%s n=%d g%d.%d.%d.%d.%d" % (layer, n, wr, wk, ku, s, rt)
        base = (layer, n, geo, ilp)
        if base not in cache:
            cache[base] = pack_e(fr, geo, tag, ilp=ilp)
            if fr.is_cuda:
                torch.cuda.synchronize()
            print("PREP gepack %s ilp%d pack_e %.2fs" % (tag, ilp, time.time() - t0), flush=True)
        pk = cache[base]
        if pk is not None:
            if pk["stats"]["bits"] >= 12.0:
                print("GE %s ilp%d: %.3f bits >= 12, not served" % (tag, ilp, pk["stats"]["bits"]), flush=True)
                pk = None
            else:
                try:
                    good = torch.equal(unpack_e(pk, T, KC, geo, ed), _frag_bits(fr))
                except AssertionError as err:  # ring protocol violated for this data at this ring size
                    print("GE %s ilp%d ed%d: %s, not served" % (tag, ilp, ed, err), flush=True)
                    good = False
                if not good:
                    print("GE %s ilp%d ed%d: reference unpack failed, not served" % (tag, ilp, ed), flush=True)
                    pk = None
        cache[key] = pk
        if fr.is_cuda:
            torch.cuda.synchronize()
        print("PREP gepack %s ilp%d ed%d %s in %.2fs" % (tag, ilp, ed, "proven bit-exact" if pk is not None else "declined", time.time() - t0), flush=True)
        return pk


    def _gxplan(st, layer, inst, n, packs):
        fmt, mt, stg, wr, wk, ku, s, rt, xs, v = inst
        if fmt == 4:  # codec E v3: W = plane, W2 = stream, esc = 8-bit XOR list, T = LUT (+ length table) pointer, U = soff pointer
            pk = _gepack(st, layer, inst, n)
            if pk is None:
                return None
            return (GVI.index(inst), pk["plane"], pk["esc"], pk["lut"].data_ptr(), n, pk["stream"], pk["soff"].data_ptr(), pk)
        if not fmt & 1:  # plain bf16 fragments (r6's FMT 0 / router hi+lo FMT 2): no escapes
            frs = list(st[layer]) if fmt == 2 else [st[layer][0]]
            return (GVI.index(inst), frs[0], st["gvnoesc"], 0, n, frs[-1], 0)
        if layer not in packs:
            return None
        css, esc = packs[layer]
        e = _gvesc(esc, n // 16, (wr, wk, ku, s, rt, xs))
        if e is None:
            return None  # an escape list overflows (32 per warp): the path is not registered
        cs = css[v & 1]
        src = [c[0] for c in cs]
        return (GVI.index(inst), src[0], e, cs[0][1], n, src[-1], cs[-1][1])


    def _gxbake(st):
        ext = _gvext()
        dev = st["qkv_gate_raw"].device
        flush = torch.empty(64 << 20, dtype=torch.float32, device=dev)
        sink = torch.empty((), dtype=torch.float32, device=dev)
        gen = torch.Generator(device=dev).manual_seed(1234)
        packs = dict(st.get("gvc12", {}))
        frs = list(st["router"])  # 12-bit hi and lo: the lo half must fit the escape lists too (checked per geometry)
        rc = [[_c12(f, dv) for f in frs] for dv in (0, 1)] if _NCR("router", (1, 8, 16, 32, 128, 256)) else [[None]]
        if all(c is not None for cs in rc for c in cs):
            packs["router"] = (rc, torch.cat([torch.cat((c[2], torch.full_like(c[2][:, :1], h)), 1) for h, c in enumerate(rc[1])]))
            print("GX router c12 escapes hi %d lo %d" % (len(rc[1][0][2]), len(rc[1][1][2])), flush=True)
        st["gvnoesc"] = torch.full((1 << 16,), -1, dtype=torch.int32, device=dev)
        res_all = {}
        rw = st["router_raw"]
        rexact = rw.to(torch.bfloat16).float() + (rw - rw.to(torch.bfloat16).float()).to(torch.bfloat16).float()
        todo = []
        for layer in ("qkv_gate", "o", "pooler", "unpooler", "router"):
            for m in (1, 8, 16, 32, 128, 256):
                ps = [(nm, g, v) for nm, ly, ms, g, v in GXP if ly == layer and m in ms]
                if ps and _NC(layer, m):
                    todo.append((layer, m, ps))
        for layer, m, ps in todo:
            f32 = layer == "router"
            w = st[layer + "_raw"]
            K = w.shape[1]
            n = 1024 if layer == "unpooler" and m <= 8 else w.shape[0]
            x = torch.randn((m, K), generator=gen, device=dev).to(torch.bfloat16)
            out = torch.empty((m, n), dtype=torch.float32 if f32 else torch.bfloat16, device=dev)
            r6 = lambda xx, oo, layer=layer, n=n: _r6(xx, layer, n, st)
            best = ("r6", None, _time_us(r6, x, out, flush, sink))
            cands = []
            print("BAKE %s m=%d r6 %.2f (ref)" % (layer, m, best[2]), flush=True)
            for name, geo, v in ps:
                if time.time() - _T0 > PREPARE_BUDGET_S:
                    print("BAKE %s m=%d %s skipped-budget" % (layer, m, name), flush=True)
                    continue
                try:
                    inst = _gxinst(geo, v, m)
                    plan = _gxplan(st, layer, inst, n, packs)
                    if plan is None:
                        print("BAKE %s m=%d %s declined-%s" % (layer, m, name, "pack" if inst[0] == 4 else "escapes"), flush=True)
                        continue
                    a = (plan[1], plan[5], x, out, plan[2], st["gvws"], st["gvcnt"], plan[3], plan[6])
                    grid = ((m + 8 * inst[1] - 1) // (8 * inst[1])) * (n // (16 * inst[3] * inst[7])) * inst[6]
                    res = int(ext.gvocc(plan[0], *a))
                    if inst[6] > 1 and not (v & 4) and not ((v >> 26) & 15) and res < grid:  # cluster variant: the whole grid must be co-resident
                        print("BAKE %s m=%d %s declined-occ %d<%d" % (layer, m, name, res, grid), flush=True)
                        continue
                    if _gvcall(st, plan, x, out) != 0:
                        print("BAKE %s m=%d %s ERROR launch" % (layer, m, name), flush=True)
                        continue
                    fn = lambda xx, oo, plan=plan: _gvcall(st, plan, xx, oo)
                    ok, msg = _validate(fn, w, m, n, f32, gen, rexact if f32 else None, {8: 5, 16: 9, 32: 17, 128: 121, 256: 250}.get(m))
                    if not ok:
                        print("BAKE %s m=%d %s INVALID %s" % (layer, m, name, msg), flush=True)
                        continue
                    us = _time_us(fn, x, out, flush, sink)
                except Exception as error:
                    print("BAKE %s m=%d %s ERROR %s" % (layer, m, name, str(error).splitlines()[-1][:160]), flush=True)
                    continue
                print("BAKE %s m=%d %s %.2f (ref r6 %.2f)%s"
                      % (layer, m, name, us, best[2], "" if res >= grid else " (res %d<%d)" % (res, grid)), flush=True)
                res_all[(layer, m, name)] = (us, fn)
                cands.append((us, name, plan))
            # pw59 (wf_bake3): one reference per group, same rule as the rq bakes (the gx times are absolute, all against
            # the same r6 measurement): the proven pick (pw49s's judge run: uk-s4x at unpooler m=1/m=8, r6 elsewhere) is
            # the reference when the box does not read it slower than r6 (window = half the margin), else r6; the pick is
            # the best candidate if it beats the reference by MARGIN_GX, else the reference.
            prov = next((c for c in cands if c[1] == PROVEN_GX.get((layer, m))), None)
            refn, refp, refmed = "r6", None, best[2]
            if prov is not None:
                rn, rm, isp = _refsel(prov[1], prov[0], "r6", best[2], RWIN_GX)
                if isp:
                    refn, refp, refmed = rn, prov[2], rm
            cl = [(c[1], c[2], refmed / c[0]) for c in cands if not (refp is not None and c is prov)]
            best = _pickrule((refn, refp, refmed), cl, MARGIN_GX)
            if _PRUNE and (layer, m) in FORCE_GX and prov is not None:  # judge-proven in 17/17 runs; the in-prepare bake flipped it once (stack_v3)
                best = (prov[1], prov[2], prov[0])
            if best[1] is not None:
                st["sel"][(layer, m)] = best[1]
            print("PICK %s m=%d %s %.2f (ref %s %.2f)" % (layer, m, best[0], best[2], refn, refmed), flush=True)
            if P1ALT and best[1] is not None and layer in ("router", "unpooler") and m <= 32 and time.time() - _T0 < PREPARE_BUDGET_S - 30:
                fb = lambda xx, oo, plan=best[1]: _gvcall(st, plan, xx, oo)
                tb, tr = [], []
                for _ in range(3):  # P1: the pick vs r6, alternated (the bake's 15-replay medians drift ~0.3 us between runs)
                    tb.append(_time_us(fb, x, out, flush, sink, 21)); tr.append(_time_us(r6, x, out, flush, sink, 21))
                print("PROBE p1alt %s m=%d %s %.2f r6=%.2f delta=%+.2f" % (layer, m, best[0], min(tb), min(tr), min(tb) - min(tr)), flush=True)
            if layer == "pooler" and m in (1, 16):  # PROBE ecod: the best codec E path vs r6, alternated (pw17 kill test)
                cand = sorted((v_[0], k[2], v_[1]) for k, v_ in res_all.items() if k[:2] == (layer, m) and k[2].startswith("ge-"))
                if not cand:
                    print("PROBE ecod pooler m=%d none" % m, flush=True)
                elif time.time() - _T0 > PREPARE_BUDGET_S:
                    print("PROBE ecod pooler m=%d skipped-budget" % m, flush=True)
                else:
                    tb, tr = [], []
                    for _ in range(3):
                        tb.append(_time_us(cand[0][2], x, out, flush, sink, 21))
                        tr.append(_time_us(r6, x, out, flush, sink, 21))
                    print("PROBE ecod pooler m=%d %s %.2f r6=%.2f delta=%+.2f" % (m, cand[0][1], min(tb), min(tr), min(tb) - min(tr)), flush=True)


    # pw29: router m = 128 / 256 on wgmma (rwk). Own load_inline extension with sm_90a flags (pw24 lesson); r6's gv
    # extension and pw23's rsk / rsk2 are untouched. CTA = 64 router rows (the wgmma M, W as A) x TOKS tokens (the wgmma N:
    # m64n128k16 at 128 tokens, m64n256k16 or two n128 sets at 256) x a K slice of KR. One warpgroup (128 threads): every
    # 16-B cp.async of the slice issued up front in NSTG commit groups (or a ring of NBUF 64-k slots when NBUF < NSTG),
    # x / hi / lo tiles in 128B-swizzle K-major atoms (8 rows x 128 B, SBO 1024) read by both wgmma descriptors; per k16
    # step two wgmmas (hi, lo) into the same fp32 accumulators. Epilogue: accumulators -> smem [tok][68] -> 16-B
    # red.global.add.v4.f32 into the output zeroed by a memset node (half pw23's v2 op count). Emulated bit-exact by
    # proj_rsi/tools/rw_emu.py (layouts, descriptors, ring reuse, epilogue, partial second block).
    RWCU = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <stdint.h>
@@WG@@
__device__ __forceinline__ void rw_cpa(uint32_t dst, const void* src, uint32_t sz) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(sz) : "memory");
}
__device__ __forceinline__ void rw_cpwait(int n) {  // n is a constant after unrolling: the switch folds
  switch (n) {
    case 7: asm volatile("cp.async.wait_group 7;\n" ::: "memory"); break;
    case 6: asm volatile("cp.async.wait_group 6;\n" ::: "memory"); break;
    case 5: asm volatile("cp.async.wait_group 5;\n" ::: "memory"); break;
    case 4: asm volatile("cp.async.wait_group 4;\n" ::: "memory"); break;
    case 3: asm volatile("cp.async.wait_group 3;\n" ::: "memory"); break;
    case 2: asm volatile("cp.async.wait_group 2;\n" ::: "memory"); break;
    case 1: asm volatile("cp.async.wait_group 1;\n" ::: "memory"); break;
    default: asm volatile("cp.async.wait_group 0;\n" ::: "memory"); break;
  }
}
// K-major, 128B swizzle: start >> 4, LBO 1 (unused), SBO 1024 B (8 rows x 128 B), base offset 0, mode 1 << 62
__device__ __forceinline__ uint64_t rw_desc(uint32_t a) {
  return (uint64_t)((a & 0x3FFFF) >> 4) | (1ull << 16) | (64ull << 32) | (1ull << 62);
}
__device__ __forceinline__ void rw_red4(float* a, float4 v) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n" :: "l"(a), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}
template <int NA> __device__ __forceinline__ void rw_keep(float* d) {
#pragma unroll
  for (int i = 0; i < NA; i++) asm volatile("" : "+f"(d[i]) :: "memory");
}
template <int WN> __device__ __forceinline__ void rw_mma(float* d, uint64_t a, uint64_t b) {
  if constexpr (WN == 128) rw_mma128(d, a, b); else rw_mma256(d, a, b);
}
// slot = one 64-k chunk: x rows [0, TOKS), hi rows [TOKS, TOKS + 64), lo rows [TOKS + 64, TOKS + 128), 128 B each;
// 16-B column c of row r at slot + r * 128 + ((c ^ (r & 7)) << 4) (TOKS % 8 == 0: the hi / lo tiles keep the phase).
// MODE bits: 1 loads only (ladder), 2 no reductions (ladder), 4 self-zeroing (no memset node: the first CTA of each
// output tile to arrive in a call zeroes the tile and releases a per-tile flag; the others acquire it before their
// reds; per-tile arrival counters are monotonic across calls, so nothing is reset). CL > 1: cluster of CL K-splits
// pre-reduces its partials through DSMEM, each CTA then reds 1/CL of the token rows.
__device__ __forceinline__ int rw_ldacq(const int* p) {
  int v; asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ void rw_strel(int* p, int v) {
  asm volatile("st.release.gpu.global.s32 [%0], %1;\n" :: "l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned long long rw_gt() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(t) :: "memory"); return t;
}
__device__ __forceinline__ void rw_csync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
template <int KR, int NSTG, int NBUF, int TOKS, int WN, int MINB, int MODE, int CL>
__global__ void __launch_bounds__(128, MINB) rwk(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ hi,
                                                 const __nv_bfloat16* __restrict__ lo, float* out, int m, int N, int K,
                                                 int* cnt, int slot) {
#if defined(__CUDA_ARCH__) && !defined(__CUDA_ARCH_FEAT_SM90_ALL)
  __trap();  // an unexpected plain sm_90 pass compiles to a stub instead of failing the extension
#else
  constexpr bool RING = NBUF < NSTG;
  constexpr int QT = KR / 16, QS = QT / NSTG, CPS = 2 * QS, NSLOT = RING ? NBUF : KR / 64;
  constexpr int ROWS = TOKS + 128, SB = ROWS * 128, NA = TOKS / 2, NH = TOKS / WN, SP = 68, LI = ROWS * CPS / 128;
  static_assert(KR % 64 == 0 && QT % NSTG == 0 && (ROWS * CPS) % 128 == 0, "stages");
  static_assert(RING ? QS == 4 : NBUF == NSTG, "a ring stage is one 64-k chunk");
  static_assert(TOKS % WN == 0 && TOKS * SP * 4 <= NSLOT * SB, "epilogue staging");
  extern __shared__ uint8_t rw_smem[];
  const uint32_t raw = (uint32_t)__cvta_generic_to_shared(rw_smem), base = (raw + 1023u) & ~1023u;
  float* sd = (float*)(rw_smem + (base - raw));
  const int n0 = blockIdx.x * 64, k0 = blockIdx.y * KR, t0 = blockIdx.z * TOKS, tid = threadIdx.x;
  constexpr bool ST = (MODE & 16) != 0;  // MODE 16: globaltimer stamps by thread 0 into cnt[8192..] (8 x u64 per CTA)
  unsigned long long ts[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  if (ST && tid == 0) ts[0] = rw_gt();
  auto load = [&](int s) {
#pragma unroll
    for (int it = 0; it < LI; it++) {
      const int i = tid + it * 128, r = i / CPS, cc = s * CPS + i % CPS, c = cc & 7;
      const uint32_t dst = base + ((cc >> 3) % NSLOT) * SB + r * 128 + ((c ^ (r & 7)) << 4);
      if (r < TOKS) {
        const int tok = t0 + r;
        rw_cpa(dst, x + (size_t)min(tok, m - 1) * K + k0 + cc * 8, tok < m ? 16u : 0u);
      } else if (r < TOKS + 64) {
        rw_cpa(dst, hi + (size_t)(n0 + r - TOKS) * K + k0 + cc * 8, 16u);
      } else {
        rw_cpa(dst, lo + (size_t)(n0 + r - TOKS - 64) * K + k0 + cc * 8, 16u);
      }
    }
  };
  float d[NA];
#pragma unroll
  for (int i = 0; i < NA; i++) d[i] = 0.f;
#pragma unroll
  for (int s = 0; s < (RING ? NBUF : NSTG); s++) {
    load(s);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  }
  if (ST && tid == 0) ts[1] = rw_gt();
  const int rows = min(TOKS, m - t0);
  int* ctr = cnt + (slot * 64 + blockIdx.x + gridDim.x * blockIdx.z) * 2;  // per-tile arrival counter (monotonic)
  int arrived = 0;
  if constexpr ((MODE & 4) != 0) {  // self-zeroing: each of the G K-split CTAs zeroes 1/G of the tile, then arrives
    const int G = (int)gridDim.y, E = rows * 16, e0 = (int)blockIdx.y * E / G, e1 = ((int)blockIdx.y + 1) * E / G;
    for (int i = e0 + tid; i < e1; i += 128) {
      const int r = i >> 4, c4 = i & 15;
      *(float4*)(out + (size_t)(t0 + r) * N + n0 + 4 * c4) = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    __syncthreads();
    if (tid == 0) asm volatile("atom.add.release.gpu.global.s32 %0, [%1], 1;\n" : "=r"(arrived) : "l"(ctr) : "memory");
  }
  rw_keep<NA>(d);
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
  for (int s = 0; s < NSTG; s++) {
    rw_cpwait(RING ? (s == 0 ? NBUF - 1 : NBUF - 2) : NSTG - 1 - s);  // stage s landed (this thread's copies)
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");     // generic (cp.async) writes -> wgmma reads
    __syncthreads();
    if (ST && tid == 0 && s == 0) ts[2] = rw_gt();
    if (ST && tid == 0 && s == NSTG - 1) ts[3] = rw_gt();
#pragma unroll
    for (int qq = 0; qq < QS; qq++) {
      const int q = s * QS + qq;
      const uint32_t sb = base + ((q >> 2) % NSLOT) * SB + (q & 3) * 32;
      const uint64_t dh = rw_desc(sb + TOKS * 128), dl = rw_desc(sb + (TOKS + 64) * 128);
#pragma unroll
      for (int h = 0; h < NH; h++) {
        const uint64_t db = rw_desc(sb + h * WN * 128);
        if constexpr (!(MODE & 1)) {  // MODE 1 (ladder): loads only, no wgmma
          rw_mma<WN>(d + h * (WN / 2), dh, db);
          rw_mma<WN>(d + h * (WN / 2), dl, db);
        }
      }
    }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    if constexpr (RING) {
      if (s >= 1) {  // stage s - 1's wgmmas done in every warp -> refill its slot with stage s - 1 + NBUF
        asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
        __syncthreads();
        if (s - 1 + NBUF < NSTG) load(s - 1 + NBUF);
        asm volatile("cp.async.commit_group;\n" ::: "memory");  // possibly empty: keeps the group count uniform
        rw_keep<NA>(d);
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
      }
    }
  }
  if constexpr ((MODE & 4) != 0) {  // every slice of this call's tile zeroed (arrivals of a call are contiguous); polled
    if (tid == 0) {                 // while the last wgmmas run, ordered before the reds by the staging barriers
      const int G = (int)gridDim.y, target = (arrived / G + 1) * G;
      while (rw_ldacq(ctr) < target) { }
    }
    __syncwarp();
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  rw_keep<NA>(d);
  __syncthreads();  // every warp's wgmmas are done before the staging overwrites the tiles
  if (ST && tid == 0) ts[4] = rw_gt();
  const int lane = tid & 31, w = tid >> 5, g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int h = 0; h < NH; h++)
#pragma unroll
    for (int i = 0; i < WN / 8; i++) {  // D: d[4i + e] = (row 16w + g (+8 for e >= 2), token 8i + 2t (+1 for odd e))
      const int tok = h * WN + 8 * i + 2 * t, n = 16 * w + g;
      const float* e = d + h * (WN / 2) + 4 * i;
      sd[tok * SP + n] = e[0]; sd[(tok + 1) * SP + n] = e[1];
      sd[tok * SP + n + 8] = e[2]; sd[(tok + 1) * SP + n + 8] = e[3];
    }
  if constexpr (CL > 1) rw_csync(); else __syncthreads();
  if (ST && tid == 0) ts[5] = rw_gt();
  const int nr = (MODE & 2) ? 0 : rows;  // MODE 2 (ladder): no reductions
  if constexpr (CL == 1) {
    // MODE 8: rotate the row order by K split, so the G CTAs of a tile (finishing together) hit different lines
    const int rot = (MODE & 8) ? (int)blockIdx.y * (nr / (int)gridDim.y + 1) : 0;
    for (int i = tid; i < nr * 16; i += 128) {
      int r = (i >> 4) + rot;
      r = (MODE & 8) ? r % nr : r;
      const int c4 = i & 15;
      rw_red4(out + (size_t)(t0 + r) * N + n0 + 4 * c4, *(const float4*)(sd + r * SP + 4 * c4));
    }
  } else {
    uint32_t rk;
    asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rk));
    uint32_t rem[CL];
#pragma unroll
    for (int q = 0; q < CL; q++) asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(rem[q]) : "r"(base), "r"(q));
    const int mine = nr > (int)rk ? (nr - (int)rk + CL - 1) / CL : 0;  // token rows rk, rk + CL, ...
    for (int i = tid; i < mine * 16; i += 128) {
      const int r = (int)rk + CL * (i >> 4), c4 = i & 15;
      const uint32_t off = (uint32_t)(r * SP + 4 * c4) * 4u;
      float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
      for (int q = 0; q < CL; q++) {
        float4 v;
        asm volatile("ld.shared::cluster.v4.f32 {%0, %1, %2, %3}, [%4];\n" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(rem[q] + off) : "memory");
        a.x += v.x; a.y += v.y; a.z += v.z; a.w += v.w;
      }
      rw_red4(out + (size_t)(t0 + r) * N + n0 + 4 * c4, a);
    }
    rw_csync();  // no CTA leaves while a peer may still read its staging
  }
  if constexpr (ST) {
    __syncthreads();
    if (tid == 0) {
      ts[6] = rw_gt();
      unsigned long long* dbg = (unsigned long long*)(cnt + 16384) + 8 * (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z));
#pragma unroll
      for (int j = 0; j < 7; j++) dbg[j] = ts[j];
      uint32_t sm; asm volatile("mov.u32 %0, %%smid;\n" : "=r"(sm));
      dbg[7] = sm;
    }
  }
#endif
}
template <int KR, int NSTG, int NBUF, int TOKS, int WN, int MINB, int MODE, int CL>
static int rw_go(const __nv_bfloat16* x, const __nv_bfloat16* hi, const __nv_bfloat16* lo, float* out, int m, int N, int K,
                 int* cnt, int slot, cudaStream_t st) {
  constexpr int NSLOT = NBUF < NSTG ? NBUF : KR / 64, SM = NSLOT * (TOKS + 128) * 128 + 1024;
  if (N % 64 || K % KR || m < 1 || (K / KR) % CL || (N / 64) * ((m + TOKS - 1) / TOKS) > 64) return -4;
  auto kern = rwk<KR, NSTG, NBUF, TOKS, WN, MINB, MODE, CL>;
  static bool once = false;
  if (!once) {
    if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SM) != cudaSuccess) { cudaGetLastError(); return -6; }
    once = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(N / 64, K / KR, (m + TOKS - 1) / TOKS);
  cfg.blockDim = dim3(128, 1, 1);
  cfg.dynamicSmemBytes = SM;
  cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = 1; at[0].val.clusterDim.y = CL; at[0].val.clusterDim.z = 1;
  cfg.attrs = at;
  cfg.numAttrs = CL > 1 ? 1 : 0;
  if (cudaLaunchKernelEx(&cfg, kern, x, hi, lo, out, m, N, K, cnt, slot) != cudaSuccess) return (int)cudaGetLastError();
  return (int)cudaGetLastError();
}
template <int KR, int NSTG, int NBUF, int TOKS, int WN, int MINB, int MODE, int CL>
static int rw_occ(int m, int N, int K) {  // max co-resident clusters (CL > 1) or CTAs per SM (CL == 1)
  constexpr int NSLOT = NBUF < NSTG ? NBUF : KR / 64, SM = NSLOT * (TOKS + 128) * 128 + 1024;
  auto kern = rwk<KR, NSTG, NBUF, TOKS, WN, MINB, MODE, CL>;
  if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SM) != cudaSuccess) { cudaGetLastError(); return -6; }
  int n = -1;
  if (CL == 1) {
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kern, 128, SM) != cudaSuccess) { cudaGetLastError(); return -7; }
    return n;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(N / 64, K / KR, (m + TOKS - 1) / TOKS);
  cfg.blockDim = dim3(128, 1, 1);
  cfg.dynamicSmemBytes = SM;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = 1; at[0].val.clusterDim.y = CL; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  if (cudaOccupancyMaxActiveClusters(&n, kern, &cfg) != cudaSuccess) { cudaGetLastError(); return -8; }
  return n;
}
// ---- pw29m: rtk = warp-specialised TMA router. Warps 0-3: the consumer warpgroup (wgmma, self-zeroing, epilogue);
// warp 4 lane 0: producer issuing per 64-k stage three 2D TMA boxes (x {64, TOKS}, hi {64, 64}, lo {64, 64}, 128B
// swizzle, x rows >= m zero-filled) into an NBUF-slot ring guarded by full / empty mbarriers. Slot layout, descriptors,
// wgmma and epilogue are rwk's. MODE: 4 self-zeroing, 16 stamps (as rwk).
struct alignas(64) RtMap { unsigned long long v[16]; };
__device__ __forceinline__ void rt_mbinit(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(b), "r"(n) : "memory"); }
__device__ __forceinline__ void rt_mbwait(uint32_t b, uint32_t ph) {
  asm volatile("{\n.reg .pred p;\nRT_WAIT_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra RT_WAIT_%=;\n}\n" :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void rt_mbarrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(b) : "memory"); }
__device__ __forceinline__ void rt_mbexpect(uint32_t b, uint32_t tx) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(b), "r"(tx) : "memory"); }
__device__ __forceinline__ void rt_tile(uint32_t dst, const RtMap* tm, int k, int row, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
               :: "r"(dst), "l"((unsigned long long)tm), "r"(k), "r"(row), "r"(bar) : "memory");
}
__device__ __forceinline__ void rt_tile_mc(uint32_t dst, const RtMap* tm, int k, int row, uint32_t bar, uint16_t mask) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;\n"
               :: "r"(dst), "l"((unsigned long long)tm), "r"(k), "r"(row), "r"(bar), "h"(mask) : "memory");
}
__device__ __forceinline__ void rt_rarrive(uint32_t bar, uint32_t rank) {  // arrive on the same mbarrier in CTA `rank`
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(r) : "r"(bar), "r"(rank));
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];\n" :: "r"(r) : "memory");
}
__device__ __forceinline__ void rt_cbar() { asm volatile("bar.sync 1, 128;\n" ::: "memory"); }  // consumer warpgroup only
__device__ __forceinline__ void rt_kcsync() { asm volatile("barrier.cluster.arrive.relaxed;\nbarrier.cluster.wait.acquire;\n" ::: "memory"); }
template <int KR, int NBUF, int TOKS, int WN, int MINB, int MODE>
__global__ void __launch_bounds__((MODE & 4) ? 192 : 160, MINB) rtk(const __grid_constant__ RtMap tx, const __grid_constant__ RtMap th,
                                                 const __grid_constant__ RtMap tl, float* out, int m, int N, int K,
                                                 int* cnt, int slot) {
#if defined(__CUDA_ARCH__) && !defined(__CUDA_ARCH_FEAT_SM90_ALL)
  __trap();
#else
  constexpr int NSTG = KR / 64, ROWS = TOKS + 128, SB = ROWS * 128, NA = TOKS / 2, NH = TOKS / WN, SP = 68;
  constexpr int NSLOT = NBUF < NSTG ? NBUF : NSTG;
  static_assert(KR % 64 == 0 && TOKS % WN == 0 && TOKS * SP * 4 <= NSLOT * SB && SB % 1024 == 0, "rtk");
  constexpr bool ST = (MODE & 16) != 0;
  constexpr bool MC = (MODE & 32) != 0;  // cluster of CL n-tiles: each CTA multicasts 1/CL of the x box to all CL
  constexpr int CL = (MODE & 512) ? 2 : 4;  // MODE 512: cluster of 2 n-tiles (half the x traffic, 2x the clusters)
  constexpr bool KC = (MODE & 2048) != 0;   // MODE 2048: cluster of CLK K-splits, partials pre-reduced in DSMEM
  constexpr int CLK = (MODE & 4096) ? 4 : 2;  // MODE 4096: CLK 4 (else 2)
  static_assert(!MC || TOKS % 32 == 0, "mc");
  extern __shared__ uint8_t rw_smem[];
  const uint32_t raw = (uint32_t)__cvta_generic_to_shared(rw_smem), base = (raw + 1023u) & ~1023u;
  const uint32_t bars = base + NSLOT * SB;  // full[NSLOT], empty[NSLOT]
  float* sd = (float*)(rw_smem + (base - raw));
  const int n0 = blockIdx.x * 64, k0 = blockIdx.y * KR, t0 = blockIdx.z * TOKS, tid = threadIdx.x;
  unsigned long long ts[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  if (ST && tid == 0) ts[0] = rw_gt();
  if ((MODE & 64) && tid == 128) {  // MODE 64: fetch the three tensor maps before the barrier setup
    asm volatile("prefetch.tensormap [%0];\n" :: "l"((unsigned long long)&tx) : "memory");
    asm volatile("prefetch.tensormap [%0];\n" :: "l"((unsigned long long)&th) : "memory");
    asm volatile("prefetch.tensormap [%0];\n" :: "l"((unsigned long long)&tl) : "memory");
  }
  constexpr bool ET = (MODE & 128) != 0;  // MODE 128: the producer lane owns the full barriers and issues stage 0
  if (ET) {                               // before the CTA-wide sync, so the first TMA is in flight during it
    if (tid == 128) {
      for (int i = 0; i < NSLOT; i++) rt_mbinit(bars + 8 * i, 1);
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
      const uint32_t fb = bars;
      rt_mbexpect(fb, (uint32_t)SB);
      rt_tile(base, &tx, k0, t0, fb);
      rt_tile(base + TOKS * 128, &th, k0, n0, fb);
      rt_tile(base + (TOKS + 64) * 128, &tl, k0, n0, fb);
    }
    if (tid == 0) {
      for (int i = 0; i < NSLOT; i++) rt_mbinit(bars + 8 * (NSLOT + i), MC ? 4 * CL : 4);
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
  } else if (tid == 0) {
    for (int i = 0; i < NSLOT; i++) { rt_mbinit(bars + 8 * i, 1); rt_mbinit(bars + 8 * (NSLOT + i), MC ? 4 * CL : 4); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  if constexpr (MC) rw_csync(); else __syncthreads();  // MC: every CTA's barriers exist before any multicast / remote arrive
  const int rows = min(TOKS, m - t0);
  if (tid >= 160) {  // MODE 4: the zeroing warp (off the consumers' critical path): zero 1/G of the tile, arrive with
    int* ctr = cnt + (slot * 64 + blockIdx.x + gridDim.x * blockIdx.z) * 2;  // release, poll the call's G arrivals,
    const int G = (int)gridDim.y, E = rows * 16, e0 = (int)blockIdx.y * E / G, e1 = ((int)blockIdx.y + 1) * E / G;
    const int ln = tid - 160;                                                // then meet the consumers before the reds
    for (int i = e0 + ln; i < e1; i += 32) {
      const int r = i >> 4, c4 = i & 15;
      *(float4*)(out + (size_t)(t0 + r) * N + n0 + 4 * c4) = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    __syncwarp();
    if (ln == 0) {
      int arrived;
      asm volatile("atom.add.release.gpu.global.s32 %0, [%1], 1;\n" : "=r"(arrived) : "l"(ctr) : "memory");
      const int target = (arrived / G + 1) * G;
      while (rw_ldacq(ctr) < target) { }
    }
    __syncwarp();
    asm volatile("bar.sync 2, 160;\n" ::: "memory");
    if constexpr (MC) asm volatile("barrier.cluster.arrive.relaxed;\n" ::: "memory");
    if constexpr (KC) { rt_kcsync(); rt_kcsync(); }  // both cluster barriers: the peers read this CTA's staging
    return;
  }
  if (tid >= 128) {  // producer warp
    if (tid == 128) {
      for (int s = ET ? 1 : 0; s < NSTG; s++) {
        const int sl = s % NSLOT;
        if (s >= NSLOT) rt_mbwait(bars + 8 * (NSLOT + sl), (uint32_t)((s / NSLOT - 1) & 1));
        const uint32_t fb = bars + 8 * sl, dst = base + sl * SB;
        rt_mbexpect(fb, (uint32_t)SB);
        if constexpr (MC) {
          uint32_t rk;
          asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rk));
          rt_tile_mc(dst + rk * (TOKS / CL) * 128, &tx, k0 + 64 * s, t0 + (int)rk * (TOKS / CL), fb, (uint16_t)((1u << CL) - 1));
        } else {
          rt_tile(dst, &tx, k0 + 64 * s, t0, fb);
        }
        rt_tile(dst + TOKS * 128, &th, k0 + 64 * s, n0, fb);
        rt_tile(dst + (TOKS + 64) * 128, &tl, k0 + 64 * s, n0, fb);
      }
    }
    if constexpr (MC) asm volatile("barrier.cluster.arrive.relaxed;\n" ::: "memory");
    if constexpr (KC) { rt_kcsync(); rt_kcsync(); }
    return;
  }
  if (ST && tid == 0) ts[1] = rw_gt();
  float d[NA];
#pragma unroll
  for (int i = 0; i < NA; i++) d[i] = 0.f;
  rw_keep<NA>(d);
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll 1
  for (int s = 0; s < NSTG; s++) {
    const int sl = s % NSLOT;
    rt_mbwait(bars + 8 * sl, (uint32_t)((s / NSLOT) & 1));
    if (ST && tid == 0 && s == 0) ts[2] = rw_gt();
    if (ST && tid == 0 && s == NSTG - 1) ts[3] = rw_gt();
#pragma unroll
    for (int q = 0; q < 4; q++) {
      const uint32_t sb = base + sl * SB + q * 32;
      const uint64_t dh = rw_desc(sb + TOKS * 128), dl = rw_desc(sb + (TOKS + 64) * 128);
#pragma unroll
      for (int h = 0; h < NH; h++) {
        const uint64_t db = rw_desc(sb + h * WN * 128);
        rw_mma<WN>(d + h * (WN / 2), dh, db);
        rw_mma<WN>(d + h * (WN / 2), dl, db);
      }
    }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    if (s >= 1) {
      asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");  // stage s - 1 retired in this warp
      rw_keep<NA>(d);
      if ((tid & 31) == 0 && s - 1 + NSLOT < NSTG) {
        if constexpr (MC) {
#pragma unroll
          for (uint32_t q = 0; q < CL; q++) rt_rarrive(bars + 8 * (NSLOT + (s - 1) % NSLOT), q);
        } else {
          rt_mbarrive(bars + 8 * (NSLOT + (s - 1) % NSLOT));
        }
      }
      asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    }
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  rw_keep<NA>(d);
  rt_cbar();
  if (ST && tid == 0) ts[4] = rw_gt();
  const int lane = tid & 31, w = tid >> 5, g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int h = 0; h < NH; h++)
#pragma unroll
    for (int i = 0; i < WN / 8; i++) {
      const int tok = h * WN + 8 * i + 2 * t, n = 16 * w + g;
      const float* e = d + h * (WN / 2) + 4 * i;
      sd[tok * SP + n] = e[0]; sd[(tok + 1) * SP + n] = e[1];
      sd[tok * SP + n + 8] = e[2]; sd[(tok + 1) * SP + n + 8] = e[3];
    }
  if constexpr ((MODE & 4) != 0) asm volatile("bar.sync 2, 160;\n" ::: "memory");  // staged + this call's tile zeroed
  else rt_cbar();
  if constexpr (MC) asm volatile("barrier.cluster.arrive.relaxed;\nbarrier.cluster.wait.acquire;\n" ::: "memory");
  if constexpr (KC) rt_kcsync();
  if (ST && tid == 0) ts[5] = rw_gt();
  if constexpr (KC) {  // MODE 2048: sum the CLK partials of this cluster in DSMEM, then one red per CLK rows
    uint32_t rk;
    asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rk));
    uint32_t rem[CLK];
#pragma unroll
    for (int q = 0; q < CLK; q++) asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(rem[q]) : "r"(base), "r"(q));
    const int mine = rows > (int)rk ? (rows - (int)rk + CLK - 1) / CLK : 0;
    for (int i = tid; i < mine * 16; i += 128) {
      const int r = (int)rk + CLK * (i >> 4), c4 = i & 15;
      const uint32_t off = (uint32_t)(r * SP + 4 * c4) * 4u;
      float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
      for (int q = 0; q < CLK; q++) {
        float4 v;
        asm volatile("ld.shared::cluster.v4.f32 {%0, %1, %2, %3}, [%4];\n" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(rem[q] + off) : "memory");
        a.x += v.x; a.y += v.y; a.z += v.z; a.w += v.w;
      }
      rw_red4(out + (size_t)(t0 + r) * N + n0 + 4 * c4, a);
    }
    rt_kcsync();  // no CTA leaves while a peer may still read its staging
  } else {
  const int rot = (MODE & 1024) ? (int)blockIdx.y * (rows / (int)gridDim.y + 1) : 0;  // MODE 1024: rotate the row order
  for (int i = tid; i < rows * 16; i += 128) {
    int r = i >> 4;
    if (MODE & 1024) r = (r + rot) % rows;
    const int c4 = i & 15;
    rw_red4(out + (size_t)(t0 + r) * N + n0 + 4 * c4, *(const float4*)(sd + r * SP + 4 * c4));
  }
  }
  if constexpr (ST) {
    rt_cbar();
    if (tid == 0) {
      ts[6] = rw_gt();
      unsigned long long* dbg = (unsigned long long*)(cnt + 16384) + 8 * (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z));
#pragma unroll
      for (int j = 0; j < 7; j++) dbg[j] = ts[j];
      uint32_t smid; asm volatile("mov.u32 %0, %%smid;\n" : "=r"(smid));
      dbg[7] = smid;
    }
  }
#endif
}
typedef CUresult (*RtEncodeFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                               const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                               CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static_assert(sizeof(CUtensorMap) == sizeof(RtMap), "CUtensorMap size");
static int rt_map(RtMap& tm, const void* p, int rows, int K, int box_rows) {
  static RtEncodeFn enc = nullptr;
  if (!enc) {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult q;
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q) != cudaSuccess || !fn) { cudaGetLastError(); return -20; }
    enc = (RtEncodeFn)fn;
  }
  cuuint64_t dims[2] = {(cuuint64_t)K, (cuuint64_t)rows};
  cuuint64_t strides[1] = {(cuuint64_t)K * 2};
  cuuint32_t box[2] = {64, (cuuint32_t)box_rows};
  cuuint32_t estr[2] = {1, 1};
  return enc((CUtensorMap*)&tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)p, dims, strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
             CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS ? 0 : -22;
}
template <int KR, int NBUF, int TOKS, int WN, int MINB, int MODE>
static int rt_go(const __nv_bfloat16* x, const __nv_bfloat16* hi, const __nv_bfloat16* lo, float* out, int m, int N, int K,
                 int* cnt, int slot, cudaStream_t st) {
  constexpr int NSTG = KR / 64, NSLOT = NBUF < NSTG ? NBUF : NSTG, SM = NSLOT * (TOKS + 128) * 128 + 1024 + 16 * NSLOT;
  if (N % 64 || K % KR || m < 1 || (N / 64) * ((m + TOKS - 1) / TOKS) > 64) return -4;
  auto kern = rtk<KR, NBUF, TOKS, WN, MINB, MODE>;
  static bool once = false;
  static int slots = 0;  // co-resident CTAs (self-zeroing waits across the K splits of a tile: never launch a 2-wave grid)
  if (!once) {
    if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SM) != cudaSuccess) { cudaGetLastError(); return -6; }
    int per = 0, dev = 0, sms = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, kern, (MODE & 4) ? 192 : 160, SM) != cudaSuccess) { cudaGetLastError(); return -7; }
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    slots = per * sms;
    once = true;
  }
  constexpr int CL = (MODE & 512) ? 2 : 4;
  constexpr int CLK = (MODE & 4096) ? 4 : 2;
  if ((MODE & 2048) && (K / KR) % CLK) return -4;
  if ((MODE & 4) && ((MODE & 32) || (MODE & 2048))) {  // clustered + self-zeroing: every cluster must be co-resident
    cudaLaunchConfig_t oc = {};
    oc.gridDim = dim3(N / 64, K / KR, (m + TOKS - 1) / TOKS);
    oc.blockDim = dim3(192, 1, 1);
    oc.dynamicSmemBytes = SM;
    cudaLaunchAttribute oa[1];
    oa[0].id = cudaLaunchAttributeClusterDimension;
    oa[0].val.clusterDim.x = (MODE & 32) ? CL : 1; oa[0].val.clusterDim.y = (MODE & 2048) ? CLK : 1; oa[0].val.clusterDim.z = 1;
    oc.attrs = oa; oc.numAttrs = 1;
    int maxcl = -1;
    if (cudaOccupancyMaxActiveClusters(&maxcl, kern, &oc) != cudaSuccess) { cudaGetLastError(); return -31; }
    const int need = (MODE & 32) ? (N / 64 / CL) * (K / KR) * ((m + TOKS - 1) / TOKS) : (N / 64) * (K / KR / CLK) * ((m + TOKS - 1) / TOKS);
    if (maxcl < need) return -31;
  }
  if ((MODE & 4) && (N / 64) * (K / KR) * ((m + TOKS - 1) / TOKS) > slots) return -30;
  constexpr bool MC = (MODE & 32) != 0;
  RtMap tx, th, tl;
  int r = rt_map(tx, x, m, K, MC ? TOKS / CL : TOKS);
  if (!r) r = rt_map(th, hi, N, K, 64);
  if (!r) r = rt_map(tl, lo, N, K, 64);
  if (r) return r;
  if (MC && (N / 64) % CL) return -4;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(N / 64, K / KR, (m + TOKS - 1) / TOKS);
  cfg.blockDim = dim3((MODE & 4) ? 192 : 160, 1, 1);
  cfg.dynamicSmemBytes = SM;
  cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = MC ? CL : 1; at[0].val.clusterDim.y = (MODE & 2048) ? CLK : 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at;
  cfg.numAttrs = (MC || (MODE & 2048)) ? 1 : 0;
  if (cudaLaunchKernelEx(&cfg, kern, tx, th, tl, out, m, N, K, cnt, slot) != cudaSuccess) return (int)cudaGetLastError();
  return (int)cudaGetLastError();
}
template <int KR, int NBUF, int TOKS, int WN, int MINB, int MODE>
static int rt_occ(int m, int N, int K) {  // co-resident clusters of 4 (MODE 32) or CTAs per SM
  constexpr int NSTG = KR / 64, NSLOT = NBUF < NSTG ? NBUF : NSTG, SM = NSLOT * (TOKS + 128) * 128 + 1024 + 16 * NSLOT;
  auto kern = rtk<KR, NBUF, TOKS, WN, MINB, MODE>;
  if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SM) != cudaSuccess) { cudaGetLastError(); return -6; }
  const int nt = (MODE & 4) ? 192 : 160;
  int n = -1;
  if (!(MODE & 32) && !(MODE & 2048)) {
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kern, nt, SM) != cudaSuccess) { cudaGetLastError(); return -7; }
    return n;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(N / 64, K / KR, (m + TOKS - 1) / TOKS);
  cfg.blockDim = dim3(nt, 1, 1);
  cfg.dynamicSmemBytes = SM;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = (MODE & 32) ? ((MODE & 512) ? 2 : 4) : 1;
  at[0].val.clusterDim.y = (MODE & 2048) ? ((MODE & 4096) ? 4 : 2) : 1;
  at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  if (cudaOccupancyMaxActiveClusters(&n, kern, &cfg) != cudaSuccess) { cudaGetLastError(); return -8; }
  return n;
}
int64_t rwocc(int64_t v, int64_t m, int64_t N, int64_t K) {
  switch (v) {
@@RTOCC@@
@@RWOCC@@
  }
  return -2;
}
int64_t rwrun(int64_t v, torch::Tensor x, torch::Tensor hi, torch::Tensor lo, torch::Tensor out, torch::Tensor cnt) {
  if (x.size(1) != hi.size(1) || out.size(0) != x.size(0) || out.size(1) != hi.size(0) || !x.is_contiguous() || !out.is_contiguous()) return -10;
  if (((uintptr_t)out.data_ptr() & 15) || ((uintptr_t)x.data_ptr() & 15)) return -11;
  if (cnt.numel() < 128 * 64 * 2) return -12;
  if (v >= 20000 && cnt.numel() < 16384 + 2 * 8 * 1024) return -13;
  const __nv_bfloat16 *xp = (const __nv_bfloat16*)x.data_ptr(), *h = (const __nv_bfloat16*)hi.data_ptr(), *l = (const __nv_bfloat16*)lo.data_ptr();
  float* o = out.data_ptr<float>();
  int* c = cnt.data_ptr<int>();
  const int m = x.size(0), N = out.size(1), K = x.size(1);
  auto st = at::cuda::getCurrentCUDAStream();
  switch (v) {
@@RWCASES@@
@@RTCASES@@
  }
  return -2;
}
"""
    RWFLAGS = ["-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a", "-DNDEBUG"]
    # variant -> rwk<KR, NSTG, NBUF, TOKS, WN, MINB>
    RWV = {100: (256, 4, 4, 128, 128, 1), 101: (256, 8, 8, 128, 128, 1), 102: (128, 4, 4, 128, 128, 2), 103: (512, 8, 6, 128, 128, 1),
           110: (256, 4, 4, 256, 256, 1), 111: (256, 8, 8, 256, 256, 1), 112: (128, 4, 4, 256, 256, 2), 113: (256, 8, 8, 256, 128, 1),
           114: (512, 8, 4, 256, 256, 1)}
    RWV = {k: v + (0, 1) for k, v in RWV.items()}  # + (MODE, CL)
    RWV.update({104: (256, 4, 3, 128, 128, 2, 0, 1), 105: (256, 4, 2, 128, 128, 2, 0, 1), 106: (256, 4, 2, 128, 128, 2, 4, 1),
                120: RWV[100][:6] + (4, 1), 121: RWV[100][:6] + (0, 2), 123: RWV[100][:6] + (4, 2), 126: RWV[101][:6] + (4, 1),
                127: RWV[102][:6] + (4, 1), 107: (256, 4, 3, 128, 128, 2, 4, 1), 140: (256, 4, 3, 128, 128, 2, 4, 2),
                141: (256, 4, 3, 128, 128, 2, 0, 2), 142: RWV[101][:6] + (4, 2),
                150: RWV[101][:6] + (12, 1), 151: (256, 4, 3, 128, 128, 2, 12, 1), 152: (256, 4, 4, 128, 128, 1, 12, 1),
                130: RWV[113][:6] + (4, 1), 131: RWV[113][:6] + (0, 2), 136: RWV[112][:6] + (4, 1)})
    RWV.update({108: (512, 8, 3, 128, 128, 2, 0, 1)})
    RWST = (101, 104, 107, 103, 108, 113, 130)  # MODE 16 (globaltimer stamps) instances: variant + 20000
    RWV.update({v + 20000: RWV[v][:6] + (RWV[v][6] | 16, RWV[v][7]) for v in RWST})
    # pw29m rtk variants: (KR, NBUF, TOKS, WN, MINB, MODE)
    RTV = {200: (512, 6, 128, 128, 1, 4), 201: (256, 3, 128, 128, 2, 4), 202: (256, 4, 256, 128, 1, 4), 203: (512, 4, 256, 128, 1, 4),
           204: (1024, 6, 128, 128, 1, 4), 206: (512, 6, 128, 128, 1, 0), 210: (256, 4, 128, 128, 1, 4), 211: (256, 3, 128, 128, 2, 4),
           230: (512, 6, 128, 128, 1, 32), 231: (256, 3, 128, 128, 2, 32), 232: (256, 4, 128, 128, 1, 32),
           240: (512, 6, 128, 128, 1, 68), 241: (256, 3, 128, 128, 2, 68), 242: (256, 4, 128, 128, 1, 68),
           # wf_router: 250 = KR 128 (2x the CTAs at m=128), 251 = early TMA, 252 = KR 512 at 2 CTAs/SM, 253 = early TMA + map prefetch
           250: (128, 3, 128, 128, 2, 4), 251: (256, 3, 128, 128, 2, 132), 252: (512, 3, 128, 128, 2, 4),
           253: (256, 3, 128, 128, 2, 196), 254: (128, 3, 128, 128, 2, 132), 255: (512, 3, 128, 128, 2, 132),
           # wf_router batch 2: 256 = mc CL2 KR256 (m=128), 257 = mc CL2 KR512 (m=256), 258 = mc CL4 KR512,
           # 259 = TOKS 64 (2x the CTAs at m=128), 260 = mc CL2 KR256 + early TMA
           256: (256, 3, 128, 128, 1, 548), 257: (512, 3, 128, 128, 1, 548), 258: (512, 3, 128, 128, 1, 32),
           261: (256, 3, 128, 128, 2, 36), 262: (256, 3, 128, 128, 2, 548),
           263: (256, 4, 128, 128, 2, 36), 264: (512, 3, 128, 128, 2, 36),
           # wf_router batch 3: 1024 = rotate the red row order by K split; 265-267 = rotation (+ early TMA / prefetch),
           # 268-269 = NBUF 4 at KR 256 (1 CTA/SM, m=128 only) + early TMA (+ rotation)
           265: (256, 3, 128, 128, 2, 1028), 266: (256, 3, 128, 128, 2, 1156), 267: (256, 3, 128, 128, 2, 1220),
           268: (256, 4, 128, 128, 1, 132), 269: (256, 4, 128, 128, 1, 1156),
           # wf_router batch 4: 2048 = cluster of CLK K-splits pre-reduced in DSMEM (4096 -> CLK 4); 270/271 = + early TMA,
           # 272/273 = without early TMA, 274 = KR 512 (16 splits) + CLK 4 + early TMA
           270: (256, 3, 128, 128, 2, 2180), 271: (256, 3, 128, 128, 2, 6276), 272: (256, 3, 128, 128, 2, 2052),
           273: (256, 3, 128, 128, 2, 6148), 274: (512, 3, 128, 128, 2, 6276),
           # wf_router batch 5: KC CLK2 refinements (275 rotation, 276 map prefetch, 277 NBUF 4 at m=128)
           275: (256, 3, 128, 128, 2, 3076), 276: (256, 3, 128, 128, 2, 2116), 277: (256, 4, 128, 128, 1, 2052),
           # wf_router batch 6: TOKS 64 / WN 64 at m=128 (256 CTAs, 2 CTAs/SM, half the red bytes, 24 KB first box)
           278: (256, 3, 64, 64, 1, 4),  # TOKS 64 / WN 64: MINB 2 needs >168 regs (ptxas C7600), MINB 1 = 2 waves at m=128
           # wf_router batch 7: KR 512 NBUF 6 (128 CTAs at m=256: half the first-stage burst, 16 K splits) + DSMEM pre-reduce;
           # 285/286 = NBUF 2 at KR 256 (less request pressure on the first stage)
           282: (512, 6, 128, 128, 1, 2052), 283: (512, 6, 128, 128, 1, 3076), 284: (512, 6, 128, 128, 1, 6148),
           285: (256, 2, 128, 128, 2, 132), 286: (256, 2, 128, 128, 2, 4),
           # wf_router batch 8: KC CLK2 + early TMA + rotation / + prefetch + rotation; NBUF 4 + ET + rotation
           288: (256, 3, 128, 128, 2, 3204), 289: (256, 3, 128, 128, 2, 3140), 290: (256, 4, 128, 128, 1, 1156)}
    # 220-223 (multicast + self-zeroing) deadlock when not every cluster is co-resident (the zeroing wait spans clusters): not built
    RTV.update({v + 20000: c[:5] + (c[5] | 16,) for v, c in list(RTV.items())})
    RWLAD = True  # pw29h ladder instances: variant + 1000 * MODE (1 = loads only, 2 = no reds, 3 = loads only, no reds)
    if RWLAD:
        RWV.update({k + 1000 * md: RWV[k][:6] + (md, 1) for k in (101, 104) for md in (1, 2, 3)})


    def _rwwg(n):
        """the SS wgmma m64nNk16 (bf16 in, fp32 accumulate in place) as an inline-asm device function"""
        r = n // 2
        regs = ", ".join("%%%d" % i for i in range(r))
        outs = ", ".join('"+f"(d[%d])' % i for i in range(r))
        return ('__device__ __forceinline__ void rw_mma%d(float* d, uint64_t a, uint64_t b) {\n'
                '#if defined(__CUDA_ARCH_FEAT_SM90_ALL)\n'
                '  asm volatile("wgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 {%s}, %%%d, %%%d, 1, 1, 1, 0, 0;\\n"\n'
                '    : %s : "l"(a), "l"(b));\n'
                '#endif\n}\n') % (n, n, regs, r, r + 1, outs)


    def _rwsrc():
        use = {v: c for v, c in RWV.items() if RWALL or v in {p[1] for p in RSP}}
        ruse = {v: c for v, c in RTV.items() if RWALL or v in {p[1] for p in RSP}}
        rtc = "\n".join("    case %d: return rt_go<%s>(xp, h, l, o, m, N, K, c, %d, st);" % (v, ", ".join(map(str, c)), 64 + i) for i, (v, c) in enumerate(sorted(RTV.items())) if v in ruse)
        cases = "\n".join("    case %d: return rw_go<%s>(xp, h, l, o, m, N, K, c, %d, st);" % (v, ", ".join(map(str, c)), i) for i, (v, c) in enumerate(sorted(RWV.items())) if v in use)
        occ = "\n".join("    case %d: return rw_occ<%s>(m, N, K);" % (v, ", ".join(map(str, c))) for v, c in sorted(use.items()))
        return RWCU.replace("@@WG@@", _rwwg(128) + _rwwg(256)).replace("@@RWCASES@@", cases).replace("@@RWOCC@@", occ).replace("@@RTCASES@@", rtc).replace("@@RTOCC@@", "\n".join("    case %d: return rt_occ<%s>(m, N, K);" % (v, ", ".join(map(str, c))) for v, c in sorted(ruse.items())))


    _rwmod = []


    def _rwext():
        if not _rwmod:
            import zlib
            t = time.time()
            src = _rwsrc()
            import os
            arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
            os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"  # pw29h: torch otherwise adds compute_90 and ptxas rejects wgmma there
            try:
                _rwmod.append(load_inline("rw29_%x" % zlib.crc32(src.encode()), cpp_sources="int64_t rwrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor); int64_t rwocc(int64_t, int64_t, int64_t, int64_t);",
                                          cuda_sources=src, functions=["rwrun", "rwocc"], extra_cuda_cflags=RWFLAGS, verbose=False))
            finally:
                if arch is None:
                    os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
                else:
                    os.environ["TORCH_CUDA_ARCH_LIST"] = arch
            print("PREP rw compiled in %.0fs (at %.0fs)" % (time.time() - t, time.time() - _T0), flush=True)
        return _rwmod[0]


    # pw22/pw23: router m = 128 / 256 split-K kernels. (name, variant, zero-fill node, served m). rs64k32 = pw22's winner
    # (rsk BN 64 x KR 256, one wait), kept as the in-run reference. pw23 (rsk2): the K slice in NSTG commit groups issued up
    # front, stage s multiplied once it lands; rp64k32s4 / s2 = rs64k32's geometry with 4 / 2 stages; rp64k64x2 = BN 64 x
    # KR 128, 64 KB, 2 CTAs/SM (4 x 64 = 256 CTAs); rp64k32sc = s4 with the self-cleaning reduction (red.add into a
    # persistent zeroed workspace, the CTA completing a tile copies it out and re-zeroes it: no memset node). m = 256: one
    # CTA holds both 128-token blocks (TOKS 256: every W fragment feeds 4 m16 tiles): rv64k32 (192 KB, 128 CTAs), rv64k64x2
    # (96 KB, 256 CTAs at 2/SM), rv64k32sc (self-cleaning).
    RSPALL = [("rs64k32", 0, True, (128, 256)), ("rp64k32s4", 5, True, (128,)), ("rp64k32s2", 6, True, (128,)),
           ("rp64k64x2", 7, True, (128,)), ("rp64k32sc", 8, False, (128,)),
           ("rv64k32", 9, True, (256,)), ("rv64k64x2", 10, True, (256,)), ("rv64k32sc", 11, False, (256,)),
           # pw29 (rwk, wgmma; variant >= 100 = RWV): rw64k32s4 / s8 = 128 CTAs, 4 / 8 stages; rw64k64s4x2 = KR 128, 256
           # CTAs at 2/SM (also m=256 as two 128-token blocks, 512 CTAs); rw64k16r6 = KR 512 (64 CTAs), ring of 6 slots;
           # rwv* = both 128-token blocks in one CTA (n256, or h2 = two n128 sets), rwv64k16r4 = 64 CTAs, ring of 4
           ("rw64k32s4", 100, True, (128,)), ("rw64k32s8", 101, True, (128, 256)), ("rw64k64s4x2", 102, True, (128, 256)),
           ("rw64k16r6", 103, True, (128,)), ("rwv64k32s4", 110, True, (256,)), ("rwv64k32s8", 111, True, (256,)),
           ("rwv64k64s4x2", 112, True, (256,)), ("rwv64k32s8h2", 113, True, (256,)), ("rwv64k16r4", 114, True, (256,)),
           # pw29h: z = self-zeroing (no memset node), cN = cluster of N K-splits pre-reducing through DSMEM
           ("rw64k32s4z", 120, False, (128,)), ("rw64k32s4c2", 121, True, (128,)), ("rw64k32s4zc2", 123, False, (128,)),
           ("rw64k32s8z", 126, False, (128, 256)), ("rw64k64s4x2z", 127, False, (128, 256)),
           ("rw64k32r3x2", 104, True, (128, 256)), ("rw64k32r2x2", 105, True, (128, 256)), ("rw64k32r2x2z", 106, False, (128, 256)),
           ("rw64k32r3x2z", 107, False, (128, 256)), ("rw64k16r3x2", 108, True, (128, 256)), ("rw64k32r3x2zc2", 140, False, (128, 256)), ("rw64k32r3x2c2", 141, True, (128, 256)), ("rw64k32s8zc2", 142, False, (128,)),
           ("rw64k32s8zr", 150, False, (128, 256)), ("rw64k32r3x2zr", 151, False, (128, 256)), ("rw64k32s4zr", 152, False, (128,)),
           ("rt64k16b6z", 200, False, (128, 256)), ("rt64k32b3x2z", 201, False, (128, 256)), ("rtv64k32b4z", 202, False, (256,)),
           ("rtv64k16b4z", 203, False, (256,)), ("rt64k8b6z", 204, False, (128, 256)), ("rt64k16b6", 206, True, (128, 256)),
           ("rt64k32b4z", 210, False, (128, 256)), ("rt64k32b3x2bz", 211, False, (128,)),
           ("rt64k16b6m", 230, True, (128, 256)), ("rt64k32b3x2m", 231, True, (128, 256)), ("rt64k32b4m", 232, True, (128,)),
           ("rt64k16b6zp", 240, False, (128, 256)), ("rt64k32b3x2zp", 241, False, (128, 256)), ("rt64k32b4zp", 242, False, (128,)),
           ("rwv64k32s8h2z", 130, False, (256,)), ("rwv64k32s8h2c2", 131, True, (256,)), ("rwv64k64s4x2z", 136, False, (256,))]

    # pw29h shipped bake set (H100 rbench, judge-like read flush): pw23's rp64k32s4 as the in-run incumbent, pw29's
    # rw64k32s8 as the wgmma reference, the self-zeroing winners; every self-zeroing grid is one co-resident wave.
    RSP = [p for p in RSPALL if p[0] in ("rw64k32s8z", "rw64k32r3x2z", "rt64k32b4z", "rt64k32b3x2bz", "rt64k32b3x2z", "rt64k16b6z")]
    # pw35: m = 128: rw64k32s8z (the judge's current pick) vs the rtk paths; m = 256: rw64k32r3x2z vs rtk, all against r6 (F1).
    # One co-resident wave per self-zeroing grid (rt_go also declines otherwise).
    RSP = [(nm, v, zf, {"rw64k32s8z": (128,), "rt64k32b4z": (128,), "rt64k32b3x2bz": (128,), "rw64k32r3x2z": (256,),
                        "rt64k16b6z": (256,)}.get(nm, ms)) for nm, v, zf, ms in RSP]
    # wf_router (component A): pw47's bake list kept whole (the incumbent's own picks 201/211/210/200 + the wgmma
    # references) plus the two new fixed-term paths; <= 6 timed paths per m, r6 stays the F1 baseline.
    RSP = RSP + [
        ("rt64k32b4x2ze", 268, False, (128,)),         # NBUF 4 (whole tile in flight), 1 CTA/SM + early TMA
        ("rt64k32b3x2zk2pr", 289, False, (128, 256))]  # KC CLK 2 + prefetch + rotation
    # pw52: the proven pick (pw49s's judge run: rt64k32b3x2z at m=128 and m=256) is baked first and is the baseline.
    RSP = sorted(RSP, key=lambda p: 0 if p[0] == "rt64k32b3x2z" else 1)
    RWALL = False  # rbench sets True: build every RWV instance, not only the shipped ones


    P1ALT = True
    PR27 = False  # P1: the rd8 ladder was read on the H100 (audits/p1-h100.md)


    def _pr27(st):
        """PROBE rd8 (pw27, no paths): the floor of an 8 MB read with every load issued at launch, in the bake's timing
    (cold-L2 graph replay, in-prepare): ld.global 16 B x NL per thread at G = 128 / 132 / 264 / 528 CTAs x 256 threads;
    + 64 fp32 red.adds per CTA into a memset-zeroed 8 KB output (memset in the graph); + a cluster-8 DSMEM reduce;
    cp.async.bulk 8 / 16 KB chunks per CTA. 'PROBE rd8 <name> <us> <tb/s> tbs'."""
        dev = st["pooler_raw"].device
        ext = _gvext()
        flush = torch.empty(64 << 20, dtype=torch.float32, device=dev)
        sink32 = torch.empty((), dtype=torch.float32, device=dev)
        nb = 8 << 20
        buf = torch.randint(0, 255, (nb,), dtype=torch.uint8, device=dev)
        out = torch.zeros((2048,), dtype=torch.float32, device=dev)
        sink = torch.zeros((1024,), dtype=torch.int32, device=dev)
        x0 = torch.empty((1, 1), device=dev)
        zero = lambda: out.zero_()
        for name, kind, G, z in (("ldg-g128-nl16", 0, 128, False), ("ldg-g132-nl16", 0, 132, False), ("ldg-g264-nl8", 1, 264, False),
                                 ("ldg-g528-nl4", 2, 528, False), ("ldg-g1056-nl2", 3, 1056, False), ("ldg-g264-red", 4, 264, True),
                                 ("ldg-g264-c8", 5, 264, False), ("blk8k-g128", 6, 128, False), ("blk16k-g128", 7, 128, False),
                                 ("blk8k-g256", 6, 256, False)):
            try:
                def fn(xx, oo, kind=kind, G=G, z=z):
                    if z:
                        zero()
                    return ext.prrun(kind, G, buf, out, sink)
                code = fn(x0, out)
                if code:
                    print("PROBE rd8 %s launch-error %d" % (name, code), flush=True)
                    continue
                us = _time_us(fn, x0, out, flush, sink32)
                print("PROBE rd8 %s %.2f %.2f tbs" % (name, us, nb / us / 1e6), flush=True)
            except Exception as error:
                print("PROBE rd8 %s ERROR %s" % (name, str(error).splitlines()[-1][:120]), flush=True)
        print("PROBE rd8 empty-memset %.2f" % _time_us(lambda xx, oo: zero(), x0, out, flush, sink32), flush=True)


    def _rscall(st, p, x, out):
        if p[2]:
            out.zero_()  # a memset node in the captured graph (red.add accumulates into it)
        if p[1] >= 100:  # pw29 wgmma router (own sm_90a extension); 16-B cp.async sources: never let it decline at serve time
            return _rwext().rwrun(p[1], x if x.data_ptr() % 16 == 0 else x.clone(), st["rs_hi"], st["rs_lo"], out, st["rw_cnt"])
        return _gvext().rsrun(p[1], x, st["rs_hi"], st["rs_lo"], out, st["rs_ws"], st["rs_cnt"])  # ws / cnt: rsk2 RED 2 only


    def _rsbake(st):
        dev = st["router_raw"].device
        st["rs_hi"], st["rs_lo"] = st["router_hl"][:256].contiguous(), st["router_hl"][256:].contiguous()
        st["rs_ws"] = torch.zeros((256 * 256,), dtype=torch.float32, device=dev)  # RED 2: [<= 256 tokens][256] fp32, zero between calls
        st["rs_cnt"] = torch.zeros((64,), dtype=torch.int32, device=dev)
        st["rw_cnt"] = torch.zeros((16384 + 2 * 8 * 1024,), dtype=torch.int32, device=dev)  # pw29h self-zeroing: monotonic, never reset
        flush = torch.empty(64 << 20, dtype=torch.float32, device=dev)
        sink = torch.empty((), dtype=torch.float32, device=dev)
        gen = torch.Generator(device=dev).manual_seed(4242)
        w = st["router_raw"]
        rexact = st["rs_hi"].float() + st["rs_lo"].float()
        n, K = w.shape
        rw = False
        # F1 baseline rule (pw29m): r6 timed before the heavy work (the rw build) and again after the variants; the lower wins
        r6pre = {}
        for m in (128, 256):
            xx = torch.randn((m, K), generator=gen, device=dev).to(torch.bfloat16)
            oo = torch.empty((m, n), dtype=torch.float32, device=dev)
            r6pre[m] = _time_us(lambda a, b: _r6(a, "router", n, st), xx, oo, flush, sink)
        if time.time() - _T0 < PREPARE_BUDGET_S - 60:  # pw29: build the wgmma router extension; a failure drops only its paths
            try:
                _rwext()
                rw = True
            except Exception as error:
                print("PREP rw build failed: %s" % str(error).splitlines()[-1][:200], flush=True)
        else:
            print("PREP rw skipped-budget", flush=True)
        for m in (128, 256):
            x = torch.randn((m, K), generator=gen, device=dev).to(torch.bfloat16)
            out = torch.empty((m, n), dtype=torch.float32, device=dev)
            r6 = lambda xx, oo: _r6(xx, "router", n, st)
            r6med = _time_us(r6, x, out, flush, sink)
            print("BAKE router m=%d r6 %.2f (pre %.2f) at %.0fs" % (m, r6med, r6pre[m], time.time() - _T0), flush=True)

            def _try(name, v, zf):
                """validate one router path; returns (fn, plan) or None"""
                plan = (-7, v, zf, None, n, None, None)
                fn = lambda xx, oo, plan=plan: _rscall(st, plan, xx, oo)
                try:
                    if fn(x, out) != 0:
                        print("BAKE router m=%d %s ERROR launch" % (m, name), flush=True)
                        return None
                    ok, msg = _validate(fn, w, m, n, True, gen, rexact, {128: 121, 256: 250}[m])
                    if not ok:
                        print("BAKE router m=%d %s INVALID %s" % (m, name, msg), flush=True)
                        return None
                except Exception as error:
                    print("BAKE router m=%d %s ERROR %s" % (m, name, str(error).splitlines()[-1][:160]), flush=True)
                    return None
                return fn, plan

            # pw59 (wf_bake3): one reference per group, same rule as _rqbake: the proven pick (pw49s's judge run:
            # rt64k32b3x2z at m=128 and m=256) when the box does not read it slower than r6 (window = half the margin),
            # else r6; every candidate's paired ratio is taken against that same reference and the pick is the best ratio
            # if it beats the margin, else the reference.
            prov = next((p for p in RSP if p[0] == "rt64k32b3x2z" and m in p[3] and not (p[1] >= 100 and not rw)), None)
            refn, refp, refmed, reffn = "r6", None, r6med, r6
            cands, done = [], set()
            if prov is not None:
                done.add(prov[0])
                got = _try(prov[0], prov[1], prov[2])
                if got is not None:
                    fn, plan = got
                    us = _time_us(fn, x, out, flush, sink, base=r6)
                    print("BAKE router m=%d %s %.2f (ratio %.4f vs r6 %.2f) at %.0fs"
                          % (m, prov[0], us, _LAST[0][0], r6med, time.time() - _T0), flush=True)
                    rn, rm, isp = _refsel(prov[0], us, "r6", r6med, RWIN_RS)
                    if isp:
                        refn, refp, refmed, reffn = rn, plan, rm, fn
                    else:
                        cands.append((prov[0], plan, _LAST[0][0]))
            for name, v, zf, ms in RSP:
                if m not in ms or (v >= 100 and not rw) or name in done:
                    continue
                if time.time() - _T0 > PREPARE_BUDGET_S:
                    print("BAKE router m=%d %s skipped-budget" % (m, name), flush=True)
                    continue
                got = _try(name, v, zf)
                if got is None:
                    continue
                fn, plan = got
                us = _time_us(fn, x, out, flush, sink, base=reffn)
                print("BAKE router m=%d %s %.2f (ratio %.4f vs %s %.2f) at %.0fs"
                      % (m, name, us, _LAST[0][0], refn, refmed, time.time() - _T0), flush=True)
                cands.append((name, plan, _LAST[0][0]))
            best = _pickrule((refn, refp, refmed), cands, MARGIN_RS)
            if m == 128:  # PROBE rszero: the cost of the zeroing memset node alone (on the critical path of the red variants)
                print("PROBE rszero router m=128 %.2f" % _time_us(lambda xx, oo: oo.zero_(), x, out, flush, sink), flush=True)
            if best[1] is not None:
                st["sel"][("router", m)] = best[1]
            print("PICK router m=%d %s %.2f (ref %s %.2f)" % (m, best[0], best[2], refn, refmed), flush=True)
        # critic (pw23): the self-cleaning workspace/counters are shared by every RED-2 path; a declined or failed path
        # may have left them dirty, so restore the all-zero invariant before any picked path serves the judge.
        st["rs_ws"].zero_(); st["rs_cnt"].zero_(); torch.cuda.synchronize()


    def _gxladder(st):
        """PROBE gvlad: where r6's gv time goes as m grows (pw14). Each step is timed with the bake's _time_us."""
        ext = _gvext()
        dev = st["qkv_gate_raw"].device
        flush = torch.empty(64 << 20, dtype=torch.float32, device=dev)
        sink = torch.empty((), dtype=torch.float32, device=dev)
        gen = torch.Generator(device=dev).manual_seed(4321)
        packs = dict(st.get("gvc12", {}))
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        for layer in ("pooler", "qkv_gate"):
            for m in (1, 8, 16, 32):
                if time.time() - _T0 > PREPARE_BUDGET_S:
                    print("PROBE gvlad %s m=%d skipped-budget" % (layer, m), flush=True)
                    continue
                w = st[layer + "_raw"]
                n, K = w.shape
                x = torch.randn((m, K), generator=gen, device=dev).to(torch.bfloat16)
                out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
                for ly, mt, geo, v, step in GLAD:
                    if ly != layer or GVB[m] != mt:
                        continue
                    try:
                        inst = _gxinst(geo, v, m)
                        plan = _gxplan(st, layer, inst, n, packs)
                        if plan is None:
                            print("PROBE gvlad %s m=%d %s declined" % (layer, m, step), flush=True)
                            continue
                        a = (plan[1], plan[5], x, out, plan[2], st["gvws"], st["gvcnt"], plan[3], plan[6])
                        if _gvcall(st, plan, x, out) != 0:
                            print("PROBE gvlad %s m=%d %s launch-error" % (layer, m, step), flush=True)
                            continue
                        us = _time_us(lambda xx, oo, plan=plan: _gvcall(st, plan, xx, oo), x, out, flush, sink)
                        print("PROBE gvlad %s m=%d %s %.2f regs=%d ctas/sm=%.2f" % (layer, m, step, us, int(ext.gvregs(plan[0], *a)), int(ext.gvocc(plan[0], *a)) / sms), flush=True)
                    except Exception as error:
                        print("PROBE gvlad %s m=%d %s ERROR %s" % (layer, m, step, str(error).splitlines()[-1][:120]), flush=True)
                us = _time_us(lambda xx, oo: _r6(xx, layer, n, st), x, out, flush, sink)
                p = st["gv"].get((layer, _bk(m)))
                info = ""
                if p is not None and p[0] < 1000:
                    a = (p[1], p[5], x, out, p[2], st["gvws"], st["gvcnt"], p[3], p[6])
                    info = " %s regs=%d ctas/sm=%.2f" % (_name(p[0]), int(ext.gvregs(p[0], *a)), int(ext.gvocc(p[0], *a)) / sms)
                print("PROBE gvlad %s m=%d L4 %.2f%s" % (layer, m, us, info or " v12"), flush=True)


    def kernel_fn(data, state):
        cfg = data["config"]
        layer, n = cfg["layer"], cfg["n"]
        x = data["x"]
        if "cpu" in state:
            w = state["raw"][layer]
            return x.float() @ w[:n].float().t() if layer == "router" else x @ w[:n].t()
        rp = state.get("rq", {}).get((layer, x.shape[0]))  # rs128: m exactly 128 / 256, picked in the bake
        if rp is not None and rp["n"] == n:
            out = torch.empty((x.shape[0], n), dtype=torch.bfloat16, device=x.device)
            if _rqcall(state, rp, x.contiguous(), out) == 0:
                return out
        p = state["sel"].get((layer, _bk(x.shape[0])))
        if p is not None and p[4] == n:
            out = torch.empty((x.shape[0], n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
            _gvcall(state, p, x.contiguous(), out)
            return out
        return _r6(x, layer, n, state)


    def _r6(x, layer, n, state):
        """r6-c12's kernel_fn, unchanged"""
        p = state["gv"].get((layer, _bk(x.shape[0])))
        if p is not None and p[4] == n:
            out = torch.empty((x.shape[0], n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
            _gvcall(state, p, x.contiguous(), out)
            return out
        return _v12(x, layer, n, state)


    def _v12(x, layer, n, state):
        m = x.shape[0]
        if _cublas(layer, m):  # cuBLAS beats the skinny kernel here (judge v7: 0.45-0.9x); algorithm tuned in prepare()
            i = state["lt"].get((layer, m))
            if i is None:
                return x @ state[layer + "_raw"][:n].T
            out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
            _ext().lt_mm(x.contiguous(), state[layer + "_raw"], state[layer + "_t"], out, n, i)
            return out
        if layer == "router" and m > 32:
            i = state["lt"].get(("router", m))
            if i is None:
                a = torch.mm(x, state["router_hl_t"], out_dtype=torch.float32)
            else:
                a = torch.empty((m, 2 * n), dtype=torch.float32, device=x.device)
                _ext().lt_mm(x.contiguous(), state["router_hl"], state["router_hl_t"], a, 2 * n, i)
            return a[:, :n] + a[:, n:]
        A, A2 = state[layer]
        out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
        if layer == "router" and m <= 32 and RTR:
            _ext().run3(A, A2, x.contiguous(), out, state["ws"], state["cnt"], n)
            return out
        (_ext().run2 if V6 and m <= 32 else _ext().run)(A, A2 if A2 is not None else state["none"], x.contiguous(), out, state["ws"], state["cnt"], n)
        return out


    # ================================================================ RS128 (audits/m128-plan.md)
    # swap-AB register-A split-K kernel for m = 128 / 256: CTA = 128 W rows x NT tokens x K/KS; two math warpgroups (W rows
    # 0-63 / 64-127 of the block) decode the lossless 12-bit pack (pw10 _pack_nat + _c12 fragments, re-blocked to 128-row
    # blocks: one 12,288 B cp.async.bulk per k64 stage) into wgmma A registers; x is the smem B operand (2D TMA, K-major
    # SW128); producer warpgroup with a W lane (warp 8) and an x lane (warp 9) on split rings (W slot released at the LDS,
    # x slot after the wait_group that covers its wgmmas); setmaxnreg 40 / 232; partials reduced across the KS CTAs of a row
    # block by cp.reduce.async.bulk.tensor bf16 add into a zeroed out (memset node first); stmatrix.trans epilogue; no cluster.
    # PRB rungs (no epilogue unless 0): 1 W only | 2 W + x (waits / arrives only) | 3 + LDS + decode into a sink |
    # 4 + wgmma (full mainloop) | 5 bare W stream (producer self-paced, no consumer) | 6 bare W + x stream |
    # 7 epilogue only (no mainloop: D = constant; memset + reduce-add cost) | 8 decode + wgmma on resident slot 0, no TMA
    # and no barriers (SM-side rate) | 10 = 8 with constant A (no LDS / decode).
    RQV = [  # (name, NT, ST, GS, PRB, WM, INS, FL): ST k64 slots (W 12 KB + x NT*128 B each), GS k64 stages per consumer
        # round, WM 0 wait_group 1 + lagged x release / 1 decode-next under the wgmma + wait_group 0, FL bit 0 L2 hints,
        # bit 1 in-kernel zeroing (no memset node)
        ("f12", 128, 8, 1, 5, 0, 0, 0),
        ("f12x", 128, 8, 1, 6, 0, 0, 0),
        ("s0", 128, 8, 1, 1, 0, 0, 0),
        ("s1", 128, 8, 1, 2, 0, 0, 0),
        ("s2", 128, 8, 1, 3, 0, 0, 0),
        ("s3", 128, 8, 1, 4, 0, 0, 0),
        ("s3g2", 128, 8, 2, 4, 0, 0, 0),
        ("s3m1", 128, 8, 1, 4, 1, 0, 0),
        ("s3m2", 128, 8, 1, 4, 2, 0, 0),
        ("s3m2s7", 128, 7, 1, 4, 2, 0, 0),
        ("s3s6", 128, 6, 1, 4, 0, 0, 0),
        ("s3s5", 128, 5, 1, 4, 0, 0, 0),
        ("s3s4", 128, 4, 1, 4, 0, 0, 0),
        ("s3s6h", 128, 6, 1, 4, 0, 0, 1),
        ("s3s5h", 128, 5, 1, 4, 0, 0, 1),
        ("s3s6g2", 128, 6, 2, 4, 0, 0, 0),
        ("s1s6", 128, 6, 1, 2, 0, 0, 0),
        ("s1s5", 128, 5, 1, 2, 0, 0, 0),
        ("s4s6z", 128, 6, 1, 0, 0, 0, 2),
        ("s4s5z", 128, 5, 1, 0, 0, 0, 2),
        ("s4s6hz", 128, 6, 1, 0, 0, 0, 3),
        ("s4s5hz", 128, 5, 1, 0, 0, 0, 3),
        ("s4s5hzi", 128, 5, 1, 0, 0, 1, 3),
        ("s4s5zi", 128, 5, 1, 0, 0, 1, 2),
        ("s4s5zni", 128, 5, 1, 0, 0, 1, 2097154),
        ("s4s5zwi", 128, 5, 1, 0, 0, 1, 9437186),
        ("s4hz", 128, 8, 1, 0, 0, 0, 3),
        ("n256s5hz", 256, 5, 1, 0, 0, 0, 3),
        ("n256s4hz", 256, 4, 1, 0, 0, 0, 3),
        ("n256s5z", 256, 5, 1, 0, 0, 0, 2),
        # pw58 (wf_jsweep6): wf_n256anat's WM=1 / MRG / EH template flags at NT 256 (box: -0.70 qkv / -0.91 pooler
        # kernel-time vs n256s5hz). JUDGE-ONLY: the H100 ties n256 vs r6 at m=256 (box 29.38 vs 29.34) while the
        # judge separates them by 2.4% (qkv) / 9% (pooler), so only the judge's bake can rank these.
        ("n256s5m1hzmEH", 256, 5, 1, 0, 1, 0, 65603),
        ("n256s5m1hzm", 256, 5, 1, 0, 1, 0, 67),
        ("s3s7", 128, 7, 1, 4, 0, 0, 0),
        ("s3h", 128, 8, 1, 4, 0, 0, 1),
        ("s3m2h", 128, 8, 1, 4, 2, 0, 1),
        ("s4m2z", 128, 8, 1, 0, 2, 0, 3),
        ("r8m2", 128, 8, 1, 8, 2, 0, 0),
        ("s3m2i", 128, 8, 1, 4, 2, 1, 0),
        ("s3g2m1", 128, 8, 2, 4, 1, 0, 0),
        ("s4", 128, 8, 1, 0, 0, 0, 1),
        ("s4m1", 128, 8, 1, 0, 1, 0, 1),
        ("s4z", 128, 8, 1, 0, 0, 0, 3),
        ("s4m1z", 128, 8, 1, 0, 1, 0, 3),
        ("s4g2m1z", 128, 8, 2, 0, 1, 0, 3),
        ("ep", 128, 8, 1, 7, 0, 0, 0),
        ("epst", 128, 8, 1, 7, 0, 0, 4),
        ("ep0", 128, 8, 1, 7, 0, 0, 8),
        ("s4st", 128, 8, 1, 0, 0, 0, 5),
        ("s4zi", 128, 8, 1, 0, 0, 1, 3),
        ("r8", 128, 8, 1, 8, 0, 0, 0),
        ("r10", 128, 8, 1, 10, 0, 0, 0),  # wf_dilp: constant-A tensor floor probe
        ("r8m1", 128, 8, 1, 8, 1, 0, 0),
        ("s3i", 128, 8, 1, 4, 0, 1, 0),
        ("s3m1i", 128, 8, 1, 4, 1, 1, 0),
        ("s4m1zi", 128, 8, 1, 0, 1, 1, 3),
        # rs32: NT 32 / 16 (m exactly 32 / 16) and HALF (FL bit 4: 64 W rows, one math warpgroup; bit 2 = plain store, KS 1)
        ("t32s5z", 32, 5, 1, 0, 0, 0, 2),
        ("t32s5hz", 32, 5, 1, 0, 0, 0, 3),
        ("t32s8z", 32, 8, 1, 0, 0, 0, 2),
        ("t32s8hz", 32, 8, 1, 0, 0, 0, 3),
        ("t32s12z", 32, 12, 1, 0, 0, 0, 2),
        ("t32s12hz", 32, 12, 1, 0, 0, 0, 3),
        ("t32g2s8z", 32, 8, 2, 0, 0, 0, 2),
        ("t32m2s8z", 32, 8, 1, 0, 2, 0, 2),
        ("t32s8zi", 32, 8, 1, 0, 0, 1, 2),
        ("h32s8", 32, 8, 1, 0, 0, 0, 20),
        ("h32s8z", 32, 8, 1, 0, 0, 0, 18),
        ("h32s12", 32, 12, 1, 0, 0, 0, 20),
        ("h32s12z", 32, 12, 1, 0, 0, 0, 18),
        ("t16s5z", 16, 5, 1, 0, 0, 0, 2),
        ("t16s5hz", 16, 5, 1, 0, 0, 0, 3),
        ("t16s8z", 16, 8, 1, 0, 0, 0, 2),
        ("t16s8hz", 16, 8, 1, 0, 0, 0, 3),
        ("t16s12z", 16, 12, 1, 0, 0, 0, 2),
        ("t16s12hz", 16, 12, 1, 0, 0, 0, 3),
        ("t16g2s8z", 16, 8, 2, 0, 0, 0, 2),
        ("t16m2s8z", 16, 8, 1, 0, 2, 0, 2),
        ("t16s8zi", 16, 8, 1, 0, 0, 1, 2),
        ("h16s8", 16, 8, 1, 0, 0, 0, 20),
        ("h16s8z", 16, 8, 1, 0, 0, 0, 18),
        ("h16s12", 16, 12, 1, 0, 0, 0, 20),
        ("h16s12z", 16, 12, 1, 0, 0, 0, 18),
        ("t32s3z", 32, 3, 1, 0, 0, 0, 2),
        ("t32s3hz", 32, 3, 1, 0, 0, 0, 3),
        ("t32s4z", 32, 4, 1, 0, 0, 0, 2),
        ("t32s4hz", 32, 4, 1, 0, 0, 0, 3),
        ("t32s5hzi", 32, 5, 1, 0, 0, 1, 3),
        ("t16s3z", 16, 3, 1, 0, 0, 0, 2),
        ("t16s3hz", 16, 3, 1, 0, 0, 0, 3),
        ("t16s4z", 16, 4, 1, 0, 0, 0, 2),
        ("t16s4hz", 16, 4, 1, 0, 0, 0, 3),
        ("t16s5hzi", 16, 5, 1, 0, 0, 1, 3),
        ("t32s5hze", 32, 5, 1, 0, 0, 0, 35),
        ("t32s5ze", 32, 5, 1, 0, 0, 0, 34),
        ("t32s6hze", 32, 6, 1, 0, 0, 0, 35),
        ("t32s6hz", 32, 6, 1, 0, 0, 0, 3),
        ("t32s5hzei", 32, 5, 1, 0, 0, 1, 35),
        ("t32s6hzei", 32, 6, 1, 0, 0, 1, 35),
        ("t16s5hze", 16, 5, 1, 0, 0, 0, 35),
        ("t16s5ze", 16, 5, 1, 0, 0, 0, 34),
        ("t16s6hze", 16, 6, 1, 0, 0, 0, 35),
        ("t16s6hz", 16, 6, 1, 0, 0, 0, 3),
        ("t16s5hzei", 16, 5, 1, 0, 0, 1, 35),
        ("t16s6hzei", 16, 6, 1, 0, 0, 1, 35),
        ("h128s8e", 128, 8, 1, 0, 0, 0, 52),
        ("h128s5e", 128, 5, 1, 0, 0, 0, 52),  # wf_rq128b: HALF at ST 5 (72.7 KB smem: 2 CTAs/SM, launches)
        ("h32s4e", 32, 4, 1, 0, 0, 0, 52),
        ("h32s5e", 32, 5, 1, 0, 0, 0, 52),
        ("s4s5hze", 128, 5, 1, 0, 0, 0, 35),
        # wf_rq128b: FL bit 6 (64) merged empty barrier, bit 7 (128) relaxed arrive, bit 9 (512) no-escape probe,
        # bit 10 (1024) fixed-iteration escape loop
        ("s4s5hzem", 128, 5, 1, 0, 0, 0, 99),
        ("s4s5hzer", 128, 5, 1, 0, 0, 0, 163),
        ("s4s5hzemr", 128, 5, 1, 0, 0, 0, 227),
        ("s4s5hzen", 128, 5, 1, 0, 0, 0, 547),
        ("s4s5hzem2", 128, 5, 1, 0, 2, 0, 35),
        ("s4s5hzemf", 128, 5, 1, 0, 0, 0, 16483),
        ("s4s5hzemi", 128, 5, 1, 0, 0, 1, 99),
        ("s4s5hzeri", 128, 5, 1, 0, 0, 1, 163),
        ("s4s5hzemfi", 128, 5, 1, 0, 0, 1, 16483),
        # bit 11 (2048) patch-table walker
        ("s4s5hzew", 128, 5, 1, 0, 0, 0, 32803),
        ("s4s5hzewi", 128, 5, 1, 0, 0, 1, 32803),
        ("s4s5hzemw", 128, 5, 1, 0, 0, 0, 32867),
        ("s4s5hzemwi", 128, 5, 1, 0, 0, 1, 32867),
        ("s4s5hzeni", 128, 5, 1, 0, 0, 1, 547),
        ("h128s5", 128, 5, 1, 0, 0, 0, 20),
        ("h128s5z", 128, 5, 1, 0, 0, 0, 18),
        ("h128s8", 128, 8, 1, 0, 0, 0, 20),
        ("h128s8z", 128, 8, 1, 0, 0, 0, 18),
        ("s4s5zw", 128, 5, 1, 0, 0, 0, 9437186),
        ("s4s5hzw", 128, 5, 1, 0, 0, 0, 9437187),
        ("s4s5zn", 128, 5, 1, 0, 0, 0, 2097154),
        ("s4s5hzn", 128, 5, 1, 0, 0, 0, 2097155),
        ("s4s5zEH", 128, 5, 1, 0, 0, 0, 65538),
        ("s4s5hzEH", 128, 5, 1, 0, 0, 0, 65539),
        # pw62 (wf_stack5, from wf_escur32): NT=32 cursor twins (bit 20 + bit 23, EARLY bit 5 dropped as s4s5hzw relates
        # to s4s5hze) and the no-escape floor twins (bit 21, NOT exact: attribution only)
        ("t32s5hzw", 32, 5, 1, 0, 0, 0, 9437187),
        ("t32s6hzw", 32, 6, 1, 0, 0, 0, 9437187),
        ("t32s5zw", 32, 5, 1, 0, 0, 0, 9437186),
        ("h32s5hzw", 32, 5, 1, 0, 0, 0, 9437204),
        ("h32s4hzw", 32, 4, 1, 0, 0, 0, 9437204),
        ("t32s5hzwE", 32, 5, 1, 0, 0, 0, 9437219),
        ("t32s6hzwE", 32, 6, 1, 0, 0, 0, 9437219),
        ("t32s5hzn", 32, 5, 1, 0, 0, 0, 2097155),
        ("t32s6hzn", 32, 6, 1, 0, 0, 0, 2097155),
        ("t32s5zn", 32, 5, 1, 0, 0, 0, 2097154),
        ("h32s5hzn", 32, 5, 1, 0, 0, 0, 2097156),
        ("h32s4hzn", 32, 4, 1, 0, 0, 0, 2097156),
        ("t32s5hzwi", 32, 5, 1, 0, 0, 1, 9437187),
        ("t32s5hzni", 32, 5, 1, 0, 0, 1, 2097155),
        # pw63 (wf_escur16): NT=16 cursor twins, same derivation as the NT=32 ones (FL = e-row FL - bit 5 | bit 20 | bit 23).
        # h16s5hzw2 is the 2-CTA/SM HALF twin of h16s5hze2 (FL 1075 -> 9438227). The n/i entries are lab-only (floor and INS
        # stage clocks): they are in no RQSBAKE list, so prepare never builds them.
        ("t16s4hzw", 16, 4, 1, 0, 0, 0, 9437187),
        ("t16s5hzw", 16, 5, 1, 0, 0, 0, 9437187),
        ("h16s5hzw2", 16, 5, 1, 0, 0, 0, 9438227),
        ("t16s4hzn", 16, 4, 1, 0, 0, 0, 2097155),
        ("t16s5hzn", 16, 5, 1, 0, 0, 0, 2097155),
        ("t16s5hzwi", 16, 5, 1, 0, 0, 1, 9437187),
        ("t16s5hzni", 16, 5, 1, 0, 0, 1, 2097155),
        ("s4s6hze", 128, 6, 1, 0, 0, 0, 35),
        ("s4s7hze", 128, 7, 1, 0, 0, 0, 35),
        ("s4s8hze", 128, 8, 1, 0, 0, 0, 35),
        ("s4s8hzeEH", 128, 8, 1, 0, 0, 0, 65571),
        ("s4s5hzeEH", 128, 5, 1, 0, 0, 0, 65571),
        ("s4s6hzeEH", 128, 6, 1, 0, 0, 0, 65571),
        ("s4s5hzem1", 128, 5, 1, 0, 1, 0, 35),
        ("s4s5hzem2", 128, 5, 1, 0, 2, 0, 35),
        ("h128s6e", 128, 6, 1, 0, 0, 0, 52),
        ("h128s10e", 128, 10, 1, 0, 0, 0, 52),
        ("h128s8eg2", 128, 8, 2, 0, 0, 0, 52),
        ("h128s10eg2", 128, 10, 2, 0, 0, 0, 52),
        ("h128s8em1", 128, 8, 1, 0, 1, 0, 52),
        ("s4s5hze2", 128, 5, 2, 0, 0, 0, 35),
        ("s4s6hze2", 128, 6, 2, 0, 0, 0, 35),
        ("s4s5hzei", 128, 5, 1, 0, 0, 1, 35),
        ("h128s8ei", 128, 8, 1, 0, 0, 1, 52),
        # wf_rq32b: 2 CTAs per SM. H2 (FL bit 10) = 256-thread CTA, HALF 64 W rows, one math warpgroup, cap 128 regs
        # (math 216 / producer 40), grid.x = 2 NB, KS unchanged. F2 (FL bit 11) = 384 threads, full 128 rows, cap 84 regs
        # (math 104 / producer 40), KS doubled. FL 1075 = H2|EARLY|HALF|ZI|HINT, 2083 = F2|EARLY|ZI|HINT.
        ("h32s5hze2", 32, 5, 1, 0, 0, 0, 1075),
        ("h32s5hze2i", 32, 5, 1, 0, 0, 1, 1075),
        ("h32s6hze2", 32, 6, 1, 0, 0, 0, 1075),
        ("h32s8hze2", 32, 8, 1, 0, 0, 0, 1075),
        ("h32s5ze2", 32, 5, 1, 0, 0, 0, 1074),
        ("h32s4hze2", 32, 4, 1, 0, 0, 0, 1075),
        ("h32s7hze2", 32, 7, 1, 0, 0, 0, 1075),
        ("f32s5hze2", 32, 5, 1, 0, 0, 0, 2083),
        ("f32s5hze2i", 32, 5, 1, 0, 0, 1, 2083),
        ("h16s5hze2", 16, 5, 1, 0, 0, 0, 1075),
        ("h16s5hze2i", 16, 5, 1, 0, 0, 1, 1075),
        ("h16s8hze2", 16, 8, 1, 0, 0, 0, 1075),
        ("h16s4hze2", 16, 4, 1, 0, 0, 0, 1075),
        ("h16s6hze2", 16, 6, 1, 0, 0, 0, 1075),
        ("h16s7hze2", 16, 7, 1, 0, 0, 0, 1075),
        ("f16s5hze2", 16, 5, 1, 0, 0, 0, 2083),
        # wf_dilp: bit 12 (4096) 4-chain decode, bit 13 (8192) prefetched escape walk
        ("d1s5hze", 128, 5, 1, 0, 0, 0, 4131),
        ("d2s5hze", 128, 5, 1, 0, 0, 0, 12323),
        ("d2s5hzw", 128, 5, 1, 0, 0, 0, 9449507),  # pw60: d2s5hze + FL bit 20|23 (cursor)
        # pw64 (wf_stack6, from wf_escur3 H2b): the same cursor + a SHORTER apply block (FL bit 25: patch the 8 hi-byte
        # registers, then build A with the 16 PRMTs) so the taken branch over the apply is cheap. W-resident m=128:
        # s4s5hzwP 15.58/10.88/15.62 vs s4s5hzw 15.84/11.01/15.94 (pooler/qkv/unpooler), exact, regs 168 / local 0.
        ("d2s5hzwP", 128, 5, 1, 0, 0, 0, 42999939),  # d2s5hzw + FL bit 25 (9449507 | 33554432)
        ("s4s5hzwP", 128, 5, 1, 0, 0, 0, 42991619),  # s4s5hzw + FL bit 25 (9437187 | 33554432)
        ("d1s5hzei", 128, 5, 1, 0, 0, 1, 4131),
        ("d2s5hzei", 128, 5, 1, 0, 0, 1, 12323),
        ("d0s5hzen", 128, 5, 1, 0, 0, 0, 4643),
        ("d0s5hzeni", 128, 5, 1, 0, 0, 1, 4643),
        ("t32d1s5hze", 32, 5, 1, 0, 0, 0, 4131),
        ("t32d2s5hze", 32, 5, 1, 0, 0, 0, 12323),
        ("t32d1s5hzei", 32, 5, 1, 0, 0, 1, 4131),
        # wf_rq128: MS = k64 stages per ring slot (merged k128 slot: one full/empty barrier pair, one 24 KB W bulk, 2 x tiles
        # per slot; GS == MS). WM 1 = decode-next under the wgmma + wait_group 0.
        ("m2s3m1hz", 128, 3, 2, 0, 1, 0, 3),
        ("m2s4m1hz", 128, 4, 2, 0, 1, 0, 3),
        ("m2s3m1h", 128, 3, 2, 0, 1, 0, 1),
        ("m2s4m1h", 128, 4, 2, 0, 1, 0, 1),
        ("m2s3m1hze", 128, 3, 2, 0, 1, 0, 35),
        ("m2s4m1hze", 128, 4, 2, 0, 1, 0, 35),
        ("m2s3m2hz", 128, 3, 2, 0, 2, 0, 3),
        ("m2s4m2hz", 128, 4, 2, 0, 2, 0, 3),
        ("m2s3m1z", 128, 3, 2, 0, 1, 0, 2),
        ("m2s4m1z", 128, 4, 2, 0, 1, 0, 2),
        ("m2s3m1zEH", 128, 3, 2, 0, 1, 0, 65538),
        ("m2s4m1zEH", 128, 4, 2, 0, 1, 0, 65538),
        ("m2s3m1hzEH", 128, 3, 2, 0, 1, 0, 65539),
        ("m2s4m1hzEH", 128, 4, 2, 0, 1, 0, 65539),
        ("g2s5z", 128, 5, 2, 0, 0, 0, 2),
        ("g2s5hz", 128, 5, 2, 0, 1, 0, 3),
        ("g2s5hze", 128, 5, 2, 0, 1, 0, 35),
        ("g2s6hz", 128, 6, 2, 0, 1, 0, 3),
        ("g2s4hz", 128, 4, 2, 0, 1, 0, 3),
        ("s4s6hzew1", 128, 6, 1, 0, 1, 0, 35),
        ("s4s7hzew1", 128, 7, 1, 0, 1, 0, 35),
        ("s4s8hzew1", 128, 8, 1, 0, 1, 0, 35),
        ("m2s3m1hzi", 128, 3, 2, 0, 1, 1, 3),
        ("m2s4m1hzi", 128, 4, 2, 0, 1, 1, 3),
        # wf_jsweep: judge-side sweep of the axes the H100 box ties within noise (W L2-resident on the H200 = the judge
        # regime, FACTS 8). d2 = 4-chain decode + prefetched walk (pw48's m=128 pick); the new entries move one axis at a
        # time off it: no L2 hints (z), ring depth ST 4 / 6, WM 2 (one group in flight). GS 2 (k128 merged rounds) is NOT
        # here: it deadlocks on the box (see RESULTS.md).
        ("d2s5z", 128, 5, 1, 0, 0, 0, 12322),
        ("d2s4hze", 128, 4, 1, 0, 0, 0, 12323),
        ("d2s6hze", 128, 6, 1, 0, 0, 0, 12323),
        ("d2s5hzem2", 128, 5, 1, 0, 2, 0, 12323),
        # wf_jsweep2: the judge's m=128 bake is monotone in ring depth (qkv d2s4hze 16.19 < d2s5hze 16.64 = d2s6hze 16.64)
        # and hint-less beat hinted at ST 5 (d2s5z 16.54 < d2s5hze 16.64) -> ST 3, and ST 4 without the L2 hints.
        ("d2s3hze", 128, 3, 1, 0, 0, 0, 12323),
        ("d2s4z", 128, 4, 1, 0, 0, 0, 12322),
        # rs32 (NT = m): ring depth 4 / 7 off the shipped t32/t16s5-6hze shape
        ("t32s4hze", 32, 4, 1, 0, 0, 0, 35),
        ("t32s7hze", 32, 7, 1, 0, 0, 0, 35),
        ("t16s4hze", 16, 4, 1, 0, 0, 0, 35),
        ("t16s7hze", 16, 7, 1, 0, 0, 0, 35),
    ]
    RQMS = {nm: 2 for nm in ("m2s3m1hz", "m2s4m1hz", "m2s3m1h", "m2s4m1h", "m2s3m1hze", "m2s4m1hze", "m2s3m2hz", "m2s4m2hz",
                            "m2s3m1z", "m2s4m1z", "m2s3m1zEH", "m2s4m1zEH", "m2s3m1hzEH", "m2s4m1hzEH",
                            "m2s3m1hzi", "m2s4m1hzi")}
    RQI = {v[0]: i for i, v in enumerate(RQV)}
    RQTOL = {1: 4e-3, 2: 5e-3, 3: 6e-3, 4: 6e-3, 5: 7e-3, 6: 7e-3, 7: 8e-3, 8: 8e-3, 9: 8e-3}  # judge 1e-2 (rows 5x)

    RQKS = {"pooler": 4, "unpooler": 2, "o": 2, "qkv_gate": 7}  # one-wave KS grid: N/128 x KS <= 132
    # the extension the submission builds: the box's favoured instances plus the judge's pick (s4s5z: no L2 hints won on the
    # H200 at all three m=128 shapes, job 846e25d9df09433f); <= 6 timed paths per layer
    RQPROD = ["s4s5z", "s4s5hz", "s4s5hze", "d2s5hze", "n256s4hz", "n256s5hz", "n256s5z", "s4s5zEH", "s4s5hzEH"]
    RQI = {v[0]: i for i, v in enumerate(RQV)}
    RQCUR = {RQI[n] for n in ("s4s5zw", "s4s5hzw", "d2s5hzwP", "s4s5hzwP", "t32s5hzw", "t32s6hzw", "t32s5zw", "h32s5hzw", "h32s4hzw",
                              "t32s5hzwE", "t32s6hzwE", "t32s5hzwi",
                              "t16s4hzw", "t16s5hzw", "h16s5hzw2", "t16s5hzwi")}  # variants that need the FL bit 20 cursor array
    RQTOL = {1: 4e-3, 2: 5e-3, 3: 6e-3, 4: 6e-3, 5: 7e-3, 6: 7e-3, 7: 8e-3, 8: 8e-3, 9: 8e-3}  # judge 1e-2 (rows 5x)

    RQCU = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <type_traits>
struct alignas(64) RqTmap { unsigned long long v[16]; };
struct RqP {
  const uint8_t* W;                  // packed [NB][KC][8 tiles][1536]
  const uint32_t* esc;               // escape entries, [KS][NB][8][ecap] sentinel-terminated lists (+32 sentinels)
  const uint32_t* esc2;              // FL bit 20: per-(warp,lane) sub-lists [KS][NB][8][32][4], sentinel-terminated
  int ecap;                          // entries per (ks, nb, warp) list slot: fixed stride, so no offset load
  __nv_bfloat16* out;                // [m][N] (in-kernel zeroing only; the reduce-adds go through the out tensor map)
  int* cnt;                          // [2][NB] zero / poll counters (FL bit 1), all zero between launches
  unsigned long long* prof;          // INS stamps [KS][NB][2 WG][16]
  uint32_t* sink;
  int m, N, K, KC, NB, walias, ncnt;
  uint32_t Tb;
};
__device__ __forceinline__ void rq_wait(uint32_t b, uint32_t ph) {
  asm volatile("{\n.reg .pred p;\nRQ_W_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra RQ_W_%=;\n}\n" :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void rq_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(b) : "memory"); }
// wf_rq128b: relaxed (non-release) arrive. The x/W slots are only ever *read* by the consumer (LDS + wgmma, the latter
// retired by wait_group), so the release ordering of the default arrive -- which makes the arrive wait for the LDS just
// issued (P4: 95 clk) -- is not needed.
__device__ __forceinline__ void rq_arrive_r(uint32_t b) { asm volatile("mbarrier.arrive.relaxed.cta.shared::cta.b64 _, [%0];\n" :: "r"(b) : "memory"); }
__device__ __forceinline__ void rq_expect(uint32_t b, uint32_t tx) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(b), "r"(tx) : "memory"); }
__device__ __forceinline__ void rq_bulk(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void rq_bulkh(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n" :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ void rq_tile(uint32_t dst, const RqTmap* tm, int k0, int r0, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
               :: "r"(dst), "l"((unsigned long long)tm), "r"(k0), "r"(r0), "r"(bar) : "memory");
}
__device__ __forceinline__ void rq_tileh(uint32_t dst, const RqTmap* tm, int k0, int r0, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;\n"
               :: "r"(dst), "l"((unsigned long long)tm), "r"(k0), "r"(r0), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ uint4 rq_lds(uint32_t a) {
  uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory"); return v;
}
__device__ __forceinline__ uint32_t rq_prmt(uint32_t a, uint32_t b, uint32_t s) { uint32_t d; asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }
// r6 c12 decode (variant 1), pw10 pw_dec / pw_unit unchanged
__device__ __forceinline__ void rq_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t Tb, uint32_t* a) {
  const uint32_t hl = (C & 0x87878787u) + Tb, hh = (__funnelshift_r(C, C, 4) & 0x87878787u) + Tb;
  a[0] = rq_prmt(L0, hl, 0x5140); a[1] = rq_prmt(L0, hl, 0x7362); a[2] = rq_prmt(L1, hh, 0x5140); a[3] = rq_prmt(L1, hh, 0x7362);
}
__device__ __forceinline__ void rq_unit(const uint4& q0, const uint4& q1, const uint4& q2, uint32_t Tb, uint32_t* A) {
  rq_dec(q0.x, q0.y, q2.x, Tb, A); rq_dec(q0.z, q0.w, q2.y, Tb, A + 4);
  rq_dec(q1.x, q1.y, q2.z, Tb, A + 8); rq_dec(q1.z, q1.w, q2.w, Tb, A + 12);
}
// wf_dilp DILP (FL bit 12): the same 36 ALU ops, written as four independent 3-deep chains: every (hl, hh) pair is
// computed first (8 words, 20 ops, all independent), then the 16 PRMTs (all independent). rq_unit's per-dec form makes the
// compiler walk the four sub-blocks in order (each PRMT waits on its own LDS->LOP3->IADD chain); here the four chains'
// latencies overlap. Same values, same order of A[] writes.
__device__ __forceinline__ void rq_unit_ilp(const uint4& q0, const uint4& q1, const uint4& q2, uint32_t Tb, uint32_t* A) {
  const uint32_t hl0 = (q2.x & 0x87878787u) + Tb, hh0 = (__funnelshift_r(q2.x, q2.x, 4) & 0x87878787u) + Tb;
  const uint32_t hl1 = (q2.y & 0x87878787u) + Tb, hh1 = (__funnelshift_r(q2.y, q2.y, 4) & 0x87878787u) + Tb;
  const uint32_t hl2 = (q2.z & 0x87878787u) + Tb, hh2 = (__funnelshift_r(q2.z, q2.z, 4) & 0x87878787u) + Tb;
  const uint32_t hl3 = (q2.w & 0x87878787u) + Tb, hh3 = (__funnelshift_r(q2.w, q2.w, 4) & 0x87878787u) + Tb;
  A[0] = rq_prmt(q0.x, hl0, 0x5140); A[1] = rq_prmt(q0.x, hl0, 0x7362);
  A[2] = rq_prmt(q0.y, hh0, 0x5140); A[3] = rq_prmt(q0.y, hh0, 0x7362);
  A[4] = rq_prmt(q0.z, hl1, 0x5140); A[5] = rq_prmt(q0.z, hl1, 0x7362);
  A[6] = rq_prmt(q0.w, hh1, 0x5140); A[7] = rq_prmt(q0.w, hh1, 0x7362);
  A[8] = rq_prmt(q1.x, hl2, 0x5140); A[9] = rq_prmt(q1.x, hl2, 0x7362);
  A[10] = rq_prmt(q1.y, hh2, 0x5140); A[11] = rq_prmt(q1.y, hh2, 0x7362);
  A[12] = rq_prmt(q1.z, hl3, 0x5140); A[13] = rq_prmt(q1.z, hl3, 0x7362);
  A[14] = rq_prmt(q1.w, hh3, 0x5140); A[15] = rq_prmt(q1.w, hh3, 0x7362);
}
// wf_rq128b WK: apply one patch-table word (bit 15 valid, 11:8 reg, 7 half, 6:0 xor7) to a decoded A fragment. The
// predicate is per-lane but the *instruction count* is warp-uniform, so the wgmma.fence no longer waits for a walker.
__device__ __forceinline__ void rq_patch(uint32_t (&A)[16], uint32_t w) {
  if (w & 0x8000u) {
    const int r = (int)((w >> 8) & 15);
    const uint32_t xr = (w & 0x7fu) << (8 + ((w >> 7) & 1) * 16);
#pragma unroll
    for (int i = 0; i < 16; i++) if (i == r) A[i] ^= xr;
  }
}
// wf_rq128b FX2: the same walk with a fixed iteration count (warp-uniform instruction count, so the wgmma.fence of the
// warpgroup no longer waits for the warp that happens to walk the longest list). K must be >= the max hits per stage.
__device__ __forceinline__ void rq_fix2(uint32_t (&A)[16], int seq, uint32_t& nx, uint32_t& nxs, int& ei, uint32_t& ec, const uint32_t* el, int lane) {
#pragma unroll
  for (int it = 0; it < 2; it++) {
    if (nxs == (uint32_t)seq) {
      const int r = (int)((nx >> 8) & 15);
      const uint32_t xr = (nx & 0x7fu) << (8 + ((nx >> 7) & 1) * 16);
      const bool me = lane == (int)((nx >> 12) & 31);
#pragma unroll
      for (int i = 0; i < 16; i++) if (me && i == r) A[i] ^= xr;
      ++ei;
      if ((ei & 31) == 0) ec = __ldg(el + ei + lane);
      nx = __shfl_sync(0xffffffffu, ec, ei & 31);
      nxs = nx >> 17;
    }
  }
}
// escape fix (pw10 pw_fix3 entry format): entry = seq << 17 | lane << 12 | reg << 8 | half << 7 | xor7, seq = stage within
// the CTA's range. The warp's list is cached in registers, 32 entries per lane-spread window (entry i in lane i % 32), read
// by shfl: no dependent global load per hit (pw10 reloaded the next entry from L2 after every hit, ~600 clk per escape).
__device__ __forceinline__ void rq_fix(uint32_t (&A)[16], int seq, uint32_t& nx, uint32_t& nxs, int& ei, uint32_t& ec, const uint32_t* el, int lane) {
  while (nxs == (uint32_t)seq) {
    if (lane == (int)((nx >> 12) & 31)) {
      const int r = (int)((nx >> 8) & 15);
      const uint32_t xr = (nx & 0x7fu) << (8 + ((nx >> 7) & 1) * 16);
#pragma unroll
      for (int i = 0; i < 16; i++) if (i == r) A[i] ^= xr;
    }
    ++ei;
    if ((ei & 31) == 0) ec = __ldg(el + ei + lane);
    nx = __shfl_sync(0xffffffffu, ec, ei & 31);
    nxs = nx >> 17;
  }
}
// wf_dilp WPF (FL bit 13): the same walk with the next entry's shfl issued before the current entry's 16 predicated XORs,
// so the shfl latency (the only dependent chain in the walk) hides under the apply instead of stalling the wgmma.fence.
__device__ __forceinline__ void rq_fixp(uint32_t (&A)[16], int seq, uint32_t& nx, uint32_t& nxs, int& ei, uint32_t& ec, const uint32_t* el, int lane) {
  while (nxs == (uint32_t)seq) {
    uint32_t nxn = __shfl_sync(0xffffffffu, ec, (ei + 1) & 31);
    if (lane == (int)((nx >> 12) & 31)) {
      const int r = (int)((nx >> 8) & 15);
      const uint32_t xr = (nx & 0x7fu) << (8 + ((nx >> 7) & 1) * 16);
#pragma unroll
      for (int i = 0; i < 16; i++) if (i == r) A[i] ^= xr;
    }
    ++ei;
    if ((ei & 31) == 0) { ec = __ldg(el + ei + lane); nxn = __shfl_sync(0xffffffffu, ec, 0); }
    nx = nxn;
    nxs = nx >> 17;
  }
}
// FL bit 20: per-thread staged cursor.  E holds this thread's own escape entries sorted by stage (head = E.x),
// hs = E.x >> 17 kept pre-shifted so the common path is one compare + one branch; no shfl, no cross-lane dependency.
// The apply needs a runtime register index, so it is either 16 predicated XORs (if-converted -> 32 instrs EVERY stage,
// measured: loop body 2120 vs 1120 no-escape) or a real branch around them.  asm volatile cannot be if-converted, so
// rq_xor16 keeps the branch: common path = ISETP + BRA.  FL bit 22 = the __any_sync (warp-uniform branch) flavour.
__device__ __forceinline__ void rq_xor16(uint32_t (&A)[16], uint32_t nx) {
  const uint32_t xr = (nx & 0x7fu) << (8 + ((nx >> 7) & 1) * 16), r = (nx >> 8) & 15;
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.eq.u32 p, %16, 0;\n\t@p xor.b32 %0, %0, %17;\n\t"
      "setp.eq.u32 p, %16, 1;\n\t@p xor.b32 %1, %1, %17;\n\t"
      "setp.eq.u32 p, %16, 2;\n\t@p xor.b32 %2, %2, %17;\n\t"
      "setp.eq.u32 p, %16, 3;\n\t@p xor.b32 %3, %3, %17;\n\t"
      "setp.eq.u32 p, %16, 4;\n\t@p xor.b32 %4, %4, %17;\n\t"
      "setp.eq.u32 p, %16, 5;\n\t@p xor.b32 %5, %5, %17;\n\t"
      "setp.eq.u32 p, %16, 6;\n\t@p xor.b32 %6, %6, %17;\n\t"
      "setp.eq.u32 p, %16, 7;\n\t@p xor.b32 %7, %7, %17;\n\t"
      "setp.eq.u32 p, %16, 8;\n\t@p xor.b32 %8, %8, %17;\n\t"
      "setp.eq.u32 p, %16, 9;\n\t@p xor.b32 %9, %9, %17;\n\t"
      "setp.eq.u32 p, %16, 10;\n\t@p xor.b32 %10, %10, %17;\n\t"
      "setp.eq.u32 p, %16, 11;\n\t@p xor.b32 %11, %11, %17;\n\t"
      "setp.eq.u32 p, %16, 12;\n\t@p xor.b32 %12, %12, %17;\n\t"
      "setp.eq.u32 p, %16, 13;\n\t@p xor.b32 %13, %13, %17;\n\t"
      "setp.eq.u32 p, %16, 14;\n\t@p xor.b32 %14, %14, %17;\n\t"
      "setp.eq.u32 p, %16, 15;\n\t@p xor.b32 %15, %15, %17;\n\t}"
      : "+r"(A[0]), "+r"(A[1]), "+r"(A[2]), "+r"(A[3]), "+r"(A[4]), "+r"(A[5]), "+r"(A[6]), "+r"(A[7]),
        "+r"(A[8]), "+r"(A[9]), "+r"(A[10]), "+r"(A[11]), "+r"(A[12]), "+r"(A[13]), "+r"(A[14]), "+r"(A[15])
      : "r"(r), "r"(xr));
}
__device__ __forceinline__ void rq_adv(uint4& E, uint32_t& hs) { E.x = E.y; E.y = E.z; E.z = E.w; E.w = ~0u; hs = E.x >> 17; }
__device__ __forceinline__ void rq_xor16s(uint32_t (&A)[16], uint32_t nx) {  // if-convertible 16-way select
  const int r = (int)((nx >> 8) & 15);
  const uint32_t xr = (nx & 0x7fu) << (8 + ((nx >> 7) & 1) * 16);
#pragma unroll
  for (int i = 0; i < 16; i++) if (i == r) A[i] ^= xr;
}
__device__ __forceinline__ void rq_fixc(uint32_t (&A)[16], int seq, uint4& E, uint32_t& hs) {  // bit 20: real branch
  while (hs == (uint32_t)seq) { rq_xor16(A, E.x); rq_adv(E, hs); }
}
__device__ __forceinline__ void rq_fixv(uint32_t (&A)[16], int seq, uint4& E, uint32_t& hs) {  // bit 22: warp-uniform
  while (__any_sync(0xffffffffu, hs == (uint32_t)seq)) {
    if (hs == (uint32_t)seq) { rq_xor16s(A, E.x); rq_adv(E, hs); }
  }
}
// bit 23: warp-uniform branch (__any_sync) around the asm apply -> the apply is neither if-converted nor hoisted out of
// the loop (micro: loop body 107 vs 101 no-escape, 115 vote+select, 199 asm alone), so wgmma.fence has nothing to wait for
__device__ __forceinline__ void rq_fixw(uint32_t (&A)[16], int seq, uint4& E, uint32_t& hs) {
  while (__any_sync(0xffffffffu, hs == (uint32_t)seq)) {
    if (hs == (uint32_t)seq) { rq_xor16(A, E.x); rq_adv(E, hs); }
  }
}
// FL bit 25 (wf_escur3 H2b, ported): patch the 8 hi-byte registers instead of the 16 A registers.
// rq_unit_ilp builds A[4g+0]=PRMT(q0,hl_g,0x5140) A[4g+1]=PRMT(q0,hl_g,0x7362) A[4g+2]=PRMT(q0,hh_g,0x5140)
// A[4g+3]=PRMT(q0,hh_g,0x7362), so entry (reg, half) -> H[reg>>1], byte (reg&1)*2 + half => 8 predicated XORs
// instead of 16 (half the apply block, so the taken branch over it is cheap).
__device__ __forceinline__ void rq_hi8(const uint4& q0, const uint4& q1, const uint4& q2, uint32_t Tb, uint32_t* H) {
  H[0] = (q2.x & 0x87878787u) + Tb; H[1] = (__funnelshift_r(q2.x, q2.x, 4) & 0x87878787u) + Tb;
  H[2] = (q2.y & 0x87878787u) + Tb; H[3] = (__funnelshift_r(q2.y, q2.y, 4) & 0x87878787u) + Tb;
  H[4] = (q2.z & 0x87878787u) + Tb; H[5] = (__funnelshift_r(q2.z, q2.z, 4) & 0x87878787u) + Tb;
  H[6] = (q2.w & 0x87878787u) + Tb; H[7] = (__funnelshift_r(q2.w, q2.w, 4) & 0x87878787u) + Tb;
}
__device__ __forceinline__ void rq_prmtA(const uint4& q0, const uint4& q1, const uint32_t* H, uint32_t* A) {
  A[0] = rq_prmt(q0.x, H[0], 0x5140); A[1] = rq_prmt(q0.x, H[0], 0x7362);
  A[2] = rq_prmt(q0.y, H[1], 0x5140); A[3] = rq_prmt(q0.y, H[1], 0x7362);
  A[4] = rq_prmt(q0.z, H[2], 0x5140); A[5] = rq_prmt(q0.z, H[2], 0x7362);
  A[6] = rq_prmt(q0.w, H[3], 0x5140); A[7] = rq_prmt(q0.w, H[3], 0x7362);
  A[8] = rq_prmt(q1.x, H[4], 0x5140); A[9] = rq_prmt(q1.x, H[4], 0x7362);
  A[10] = rq_prmt(q1.y, H[5], 0x5140); A[11] = rq_prmt(q1.y, H[5], 0x7362);
  A[12] = rq_prmt(q1.z, H[6], 0x5140); A[13] = rq_prmt(q1.z, H[6], 0x7362);
  A[14] = rq_prmt(q1.w, H[7], 0x5140); A[15] = rq_prmt(q1.w, H[7], 0x7362);
}
__device__ __forceinline__ void rq_xor8(uint32_t* H, uint32_t nx) {
  const uint32_t r = (nx >> 8) & 15u;
  const uint32_t xr = (nx & 0x7fu) << (16 * (r & 1u) + 8 * ((nx >> 7) & 1u));
  const uint32_t ri = r >> 1;
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.eq.u32 p, %8, 0;\n\t@p xor.b32 %0, %0, %9;\n\t"
      "setp.eq.u32 p, %8, 1;\n\t@p xor.b32 %1, %1, %9;\n\t"
      "setp.eq.u32 p, %8, 2;\n\t@p xor.b32 %2, %2, %9;\n\t"
      "setp.eq.u32 p, %8, 3;\n\t@p xor.b32 %3, %3, %9;\n\t"
      "setp.eq.u32 p, %8, 4;\n\t@p xor.b32 %4, %4, %9;\n\t"
      "setp.eq.u32 p, %8, 5;\n\t@p xor.b32 %5, %5, %9;\n\t"
      "setp.eq.u32 p, %8, 6;\n\t@p xor.b32 %6, %6, %9;\n\t"
      "setp.eq.u32 p, %8, 7;\n\t@p xor.b32 %7, %7, %9;\n\t}"
      : "+r"(H[0]), "+r"(H[1]), "+r"(H[2]), "+r"(H[3]), "+r"(H[4]), "+r"(H[5]), "+r"(H[6]), "+r"(H[7])
      : "r"(ri), "r"(xr));
}
__device__ __forceinline__ void rq_fixp8(uint32_t* H, int seq, uint4& E, uint32_t& hs) {
  while (__any_sync(0xffffffffu, hs == (uint32_t)seq)) {
    if (hs == (uint32_t)seq) { rq_xor8(H, E.x); rq_adv(E, hs); }
  }
}
__device__ __forceinline__ unsigned long long rq_gt() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint64_t rq_desc(uint32_t saddr) {  // K-major SW128: LBO 1, SBO 1024 B, base offset 0
  return (uint64_t)((saddr & 0x3FFFFu) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
template <int ND> __device__ __forceinline__ void rq_facc(float (&D)[ND]) {
#pragma unroll
  for (int i = 0; i < ND; i++) asm volatile("" : "+f"(D[i]) :: "memory");
}
__device__ __forceinline__ void rq_fa(uint32_t (&A)[16]) {
#pragma unroll
  for (int i = 0; i < 16; i++) asm volatile("" : "+r"(A[i]) :: "memory");
}
__device__ __forceinline__ uint32_t rq_pk(float lo, float hi) {  // inline cvt: __float2bfloat16 serialises the RS pipeline (P4)
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}
template <int NT> __device__ __forceinline__ void rq_wgmma(float (&D)[NT / 2], const uint32_t* a, uint64_t desc, int scd);
template <int NT> __device__ __forceinline__ void rq_wgss(float (&D)[NT / 2], uint64_t da, uint64_t db, int scd);
@@WGMMA@@

template <int NT, int ST, int GS, int PRB, int WM, int INS, int FL, int MS>
struct RqC {
  static constexpr int XB = NT * 128, WB = (FL & 16) ? 6144 : 12288, ND = NT / 2;  // FL bit 4 (rs32): HALF, 64 W rows per CTA
  static constexpr int XBS = MS * XB, WBS = MS * WB;  // MS k64 stages per ring slot (MS = 2: merged k128 slot)
  static constexpr int PTW = 68;  // u32 per (slot, warp): hdr, u0 mask, u1 mask, 2 patch words per lane
  static constexpr int SMEM = 1024 + ST * (XBS + WBS) + 24 * ST + 16 + ((FL & 32768) ? ST * 8 * PTW * 4 : 0);  // WK patch table
};

// Ring: ST k64 slots, slot i = x tile (XB, 1024-aligned) in the x ring + 12 KB W chunk in the W ring. ONE full barrier
// per slot (count 2: the W lane and the x lane each arrive.expect_tx their own bytes; pw24h P4 lesson), a W empty barrier
// (released by every math warp right after its LDS: W is dead once in registers) and an x empty barrier (released after
// the wait_group that retires the slot's wgmmas). The W lane may refill a slot before the x lane: its next arrive on the
// slot's full barrier can only happen after the consumers saw the current phase complete (they LDS after the full wait),
// so phases never mix. Consumers run GS k64 stages per round: GS full waits, one wgmma.fence, 4*GS wgmma, one commit,
// one wait_group 1 (P4: k128 rounds amortise the per-round bookkeeping).
template <int NT, int ST, int GS, int PRB, int WM, int INS, int FL, int MS>
__global__ void __launch_bounds__((FL & 1024) ? 256 : 384, (FL & 3072) ? 2 : 1) rq(const RqP p, const __grid_constant__ RqTmap tx, const __grid_constant__ RqTmap to) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  using C = RqC<NT, ST, GS, PRB, WM, INS, FL, MS>;
  constexpr int XB = C::XB, WB = C::WB, ND = C::ND, XBS = C::XBS, WBS = C::WBS;
  constexpr bool HINT = FL & 1, ZI = (FL & 2) && PRB == 0;
  constexpr bool MRG = (FL & 64) != 0;    // one empty barrier per slot: the x release is folded into the W release
  constexpr bool RLAX = (FL & 128) != 0;  // relaxed (non-release) empty arrive
  constexpr bool NOE = (FL & 512) != 0;   // PROBE: no escape walk (wrong output, times the floor)
  constexpr bool FX2 = (FL & 16384) != 0;  // fixed-iteration uniform escape loop (bit 14: bit 10 is pw47's H2)
  constexpr bool WK = (FL & 32768) != 0;   // patch-table walker (warp PW+3) instead of the per-warp escape list walk (bit 15: bit 11 is pw47's F2)
  constexpr bool DILP = (FL & 4096) != 0; // wf_dilp: 4 independent decode chains (rq_unit_ilp)
  constexpr bool WPF = (FL & 8192) != 0;  // wf_dilp: escape walk with the next-entry shfl prefetched
  static_assert(!WK || (PRB == 0 && MS == 1), "WK: full kernel, MS == 1");  // ZI: in-kernel zeroing of out (no memset node)
  constexpr bool RES = PRB == 8 || PRB == 10;                           // resident: no TMA, no barriers
  constexpr bool XON = PRB != 1 && PRB != 5 && PRB != 7 && !RES;       // x streamed
  constexpr bool WON = PRB != 7 && !RES;                               // W streamed
  constexpr bool SELF = PRB == 5 || PRB == 6;                          // producers self-paced (no consumer)
  // rs32 HALF (FL bit 4): one math warpgroup, 64 W rows per CTA (= half h of the 128-row pack block: tiles 4h..4h+3, a
  // contiguous 6,144 B within each 12,288 B chunk); grid x = 2 NB; math WG1 exits at once; NW = math warps
  constexpr bool HALF = (FL & 16) != 0;
  static_assert(!(FL & 65536) || (NT == 128 && GS == MS && (ST >= 5 && ST <= 8) && PRB == 0 && WM <= 1 && INS == 0 && (FL == 65538 || FL == 65539 || FL == 65570 || FL == 65571))
                || (NT == 256 && GS == MS && ST == 5 && PRB == 0 && WM == 1 && INS == 0 && FL == 65603), "EH backport approved M128/ST5-8/selfzero full-output variants only");
  constexpr int NW = HALF ? 4 : 8, CW = HALF ? 64 : 128;
  // wf_rq32b: 2 CTAs per SM. FL bit 10 (H2): 256-thread CTA = 4 math warps + W lane + x lane + 2 zeroing warps, HALF
  // (64 W rows) only, launch cap 128 regs, math 216 / producer 40. FL bit 11 (F2): 384 threads, full 128 rows, launch
  // cap 84 regs, math 104 / producer 40 (KS must double so the grid still fills 2 x 132 SMs).
  constexpr bool H2 = (FL & 1024) != 0, F2 = (FL & 2048) != 0;
  static_assert(!H2 || HALF, "H2 needs HALF (64 W rows, one math warpgroup)");
  constexpr int PW = H2 ? 4 : 8;                       // first producer warp (W lane = PW, x lane = PW+1, zero = PW+2..)
  constexpr int MREG = H2 ? 128 : (F2 ? 96 : 232);     // math-warp setmaxnreg target (H2 pool = 256*128, F2 = 384*80)
  // rs32 EARLY (FL bit 5): the W lane thread itself inits the barriers and issues the first min(ST, KCS) W chunks and x
  // tiles before the CTA-wide __syncthreads (consumers touch the barriers only after it); the lanes resume at stage ST
  constexpr bool EARLY = (FL & 32) != 0 && PRB == 0;
  // MS (k64 stages per ring slot): one full/empty barrier pair, one MS*12 KB W bulk and MS x tiles per slot; the consumer
  // round must cover a whole slot (GS == MS) so a slot is released exactly once per round
  static_assert(MS == 1 || (GS == MS && !HALF), "MS > 1: GS == MS and no HALF");
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem), sbase = (sraw + 1023u) & ~1023u;
  const uint32_t xring = sbase, wring = sbase + ST * XBS, bars = wring + ST * WBS;
  const uint32_t full = bars, wempty = bars + 8 * ST, xempty = bars + 16 * ST, zflag = bars + 24 * ST;
  const uint32_t ptab = bars + 24 * ST + 16;  // WK patch table: [slot][warp][hdr + 2 words per lane]
  uint32_t* const ptabp = (uint32_t*)(smem + (sbase - sraw) + ptab);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int nb = HALF ? (int)(blockIdx.x >> 1) : (int)blockIdx.x, hh = HALF ? (int)(blockIdx.x & 1) : 0, ks = blockIdx.y, KS = gridDim.y;
  const int cb = nb * 128 + 64 * hh;  // first out column of this CTA
  const int s0 = (int)((long long)ks * p.KC / KS), KCS = (int)((long long)(ks + 1) * p.KC / KS) - s0;
  const unsigned long long gt0 = INS ? rq_gt() : 0;
  constexpr bool CUR = (FL & 1048576) != 0, NOESC = (FL & 2097152) != 0, VOTE = (FL & 4194304) != 0,
                 VASM = (FL & 8388608) != 0;
  constexpr bool CURP = (FL & 33554432) != 0;  // wf_escur3 H2b: cursor + 8-way hi-byte apply (rq_hi8/rq_fixp8/rq_prmtA)
  // escape-list window issued before the producers start the W flood (a load issued after it waits behind ~28 MB)
  const uint32_t* el = p.esc + (size_t)((ks * p.NB + nb) * 8 + (HALF ? 4 * hh + (warp & 3) : (warp & 7))) * p.ecap;
  // wf_rq32 FL bit 8 (ESPLIT): the entry x tiles are issued by the x lane (warp 9) in parallel with the W lane's bulks
  // (one bar.sync 6,64 after the init fence); bit 9 (EFENCE): the epilogue's fence.proxy.async.global is dropped (the
  // zeroing warp already fenced its own stores before publishing zflag).  Both measured within noise (batch 3):
  // ESPLIT -0.06..+0.11 us, EFENCE -0.22..+0.05 us on qkv m=32/16 -> no RQV entry uses them (dead code, kept for the
  // record; FL 35 is unchanged).
  constexpr bool ESPLIT = (FL & 256) != 0, EFENCE = (FL & 512) != 0;
  uint32_t ec = (warp < NW && !CUR) ? __ldg(el + lane) : 0u;
  uint4 E4 = make_uint4(~0u, ~0u, ~0u, ~0u);
  if (CUR && warp < NW)  // 16 B per thread, 512 B per warp, coalesced; latency hidden behind the first TMA wait
    E4 = *(const uint4*)(p.esc2 + (size_t)(((ks * p.NB + nb) * 8 + (HALF ? 4 * hh + (warp & 3) : (warp & 7))) * 32 + lane) * 4);
  uint32_t hs = E4.x >> 17;
  if (ESPLIT && EARLY) {
    if (warp == PW && lane == 0) {
      for (int i = 0; i < ST; i++) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(full + 8 * i), "r"(XON ? 2 : 1));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(wempty + 8 * i), "r"(NW));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(xempty + 8 * i), "r"(NW));
      }
      asm volatile("st.shared.u32 [%0], 0;\n" :: "r"(zflag) : "memory");
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    if (warp == PW || (warp == PW + 1 && XON)) asm volatile("bar.sync 6, 64;\n" ::: "memory");
    if (warp == PW && lane == 0) {
      const int nbw = p.walias ? nb % p.walias : nb;
      const uint8_t* src = p.W + ((size_t)nbw * p.KC + s0) * 12288 + hh * 6144;
      uint64_t pw = 0;
      if (HINT) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pw));
      for (int s = 0; s < ST && s < KCS; s++) {
        rq_expect(full + 8 * s, WB);
        if (HINT) rq_bulkh(wring + s * WB, src + (size_t)s * 12288, WB, full + 8 * s, pw);
        else rq_bulk(wring + s * WB, src + (size_t)s * 12288, WB, full + 8 * s);
      }
    }
    if (warp == PW + 1 && lane == 0 && XON) {
      uint64_t px = 0;
      if (HINT) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(px));
      for (int s = 0; s < ST && s < KCS; s++) {
        rq_expect(full + 8 * s, XB);
        if (HINT) rq_tileh(xring + s * XB, &tx, (s0 + s) * 64, 0, full + 8 * s, px);
        else rq_tile(xring + s * XB, &tx, (s0 + s) * 64, 0, full + 8 * s);
      }
    }
  } else if (tid == (EARLY ? PW * 32 : 0)) {
    for (int i = 0; i < ST; i++) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(full + 8 * i), "r"((XON ? 2 : 1) + (WK ? 1 : 0)));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(wempty + 8 * i), "r"(NW));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(xempty + 8 * i), "r"(NW));
    }
    asm volatile("st.shared.u32 [%0], 0;\n" :: "r"(zflag) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    if (EARLY) {
      const int nbw = p.walias ? nb % p.walias : nb;
      const uint8_t* src = p.W + ((size_t)nbw * p.KC + s0) * 12288 + hh * 6144;
      uint64_t pw = 0, px = 0;
      if (HINT) {
        asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pw));
        asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(px));
      }
      for (int s = 0; s < MS * ST && s < KCS; s += MS) {
        const int n = (KCS - s < MS) ? KCS - s : MS, slot = (s / MS) % ST;
        rq_expect(full + 8 * slot, n * WB);  // two arrivals (count 2), as the W and x lanes would make
        rq_expect(full + 8 * slot, n * XB);
        if (HINT) rq_bulkh(wring + slot * WBS, src + (size_t)s * 12288, n * WB, full + 8 * slot, pw);
        else rq_bulk(wring + slot * WBS, src + (size_t)s * 12288, n * WB, full + 8 * slot);
        for (int j = 0; j < n; j++) {
          if (HINT) rq_tileh(xring + slot * XBS + j * XB, &tx, (s0 + s + j) * 64, 0, full + 8 * slot, px);
          else rq_tile(xring + slot * XBS + j * XB, &tx, (s0 + s + j) * 64, 0, full + 8 * slot);
        }
      }
    }
  }
  if (tid == (PW + 2) * 32 && XON) asm volatile("prefetch.tensormap [%0];\n" :: "l"((unsigned long long)&tx) : "memory");
  __syncthreads();
  const unsigned long long cta = (unsigned long long)blockIdx.y * gridDim.x + blockIdx.x;
  uint32_t ec0 = 0xffffffffu;
  if (WK && warp == PW + 3 && lane < 8) ec0 = __ldg(p.esc + (size_t)((ks * p.NB + nb) * 8 + (lane & 7)) * p.ecap);
  auto walk = [&]() {  // WK patch-table walker (warp PW+3): lane l walks math warp l's escape list and publishes
    // (hdr, 2 words per lane) into the slot's table before arriving on the slot's full barrier
    const int lw = lane & 7;
    const uint32_t* myel = p.esc + (size_t)((ks * p.NB + nb) * 8 + lw) * p.ecap;
    uint32_t* const tb0 = ptabp + lw * C::PTW;
    int ei = 0, slot = 0, r = 0;
    uint32_t ec = ec0, u0 = 0, u1 = 0;
    for (int s = 0; s < KCS; s++) {
      if (s >= ST) rq_wait((MRG ? wempty : xempty) + 8 * slot, (r & 1) ^ 1);
      uint32_t* const tb = tb0 + slot * 8 * C::PTW;
      if (lane < 8) {  // clear the words this slot's previous fill set (the consumer is done with them: xempty)
        uint32_t q0 = tb[1], q1 = tb[2];
        while (q0) { const int l = __ffs((int)q0) - 1; tb[3 + l * 2] = 0u; q0 &= q0 - 1; }
        while (q1) { const int l = __ffs((int)q1) - 1; tb[4 + l * 2] = 0u; q1 &= q1 - 1; }
        tb[0] = 0u; tb[1] = 0u; tb[2] = 0u; u0 = u1 = 0u;
      }
      while (lane < 8 && (ec >> 17) == (uint32_t)s) {
        const int lf = (int)((ec >> 12) & 31), idx = (int)((u0 >> lf) & 1);
        tb[3 + lf * 2 + idx] = 0x8000u | (((ec >> 8) & 15u) << 8) | (((ec >> 7) & 1u) << 7) | (ec & 0x7fu);
        if (idx) u1 |= 1u << lf; else u0 |= 1u << lf;
        ++ei;
        ec = __ldg(myel + ei);
      }
      if (lane < 8) { tb[1] = u0; tb[2] = u1; tb[0] = (u0 | u1) ? 1u : 0u; }
      __syncwarp();
      if (lane == 0) rq_arrive(full + 8 * slot);
      if (++slot == ST) { slot = 0; r++; }
    }
  };
  if (warp >= PW) {  // ---------------- producer warpgroup: W lane (warp PW), x lane (warp PW+1)
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
    if (warp == PW && lane == 0 && WON) {
      const int nbw = p.walias ? nb % p.walias : nb;
      const uint8_t* src = p.W + ((size_t)nbw * p.KC + s0) * 12288 + hh * 6144;
      uint64_t pol = 0;
      if (HINT) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
      unsigned long long tw = 0;
      const int se = EARLY ? (KCS < MS * ST ? KCS : MS * ST) : 0;  // EARLY: stages 0 .. MS*ST-1 already issued
      int slot = 0, r = EARLY ? 1 : 0;
      for (int s = se; s < KCS; s += MS) {
        const int n = (KCS - s < MS) ? KCS - s : MS;
        if (s >= MS * ST) {
          const unsigned long long t0 = INS ? clock64() : 0;
          rq_wait((SELF ? full : wempty) + 8 * slot, (r & 1) ^ 1);
          if (INS) tw += clock64() - t0;
        }
        rq_expect(full + 8 * slot, n * WB);
        if (HINT) rq_bulkh(wring + slot * WBS, src + (size_t)s * 12288, n * WB, full + 8 * slot, pol);
        else rq_bulk(wring + slot * WBS, src + (size_t)s * 12288, n * WB, full + 8 * slot);
        if (++slot == ST) { slot = 0; r++; }
      }
      if (SELF && !XON) for (int s = (KCS > MS * ST ? KCS - MS * ST : 0); s < KCS; s += MS) rq_wait(full + 8 * ((s / MS) % ST), ((s / MS) / ST) & 1);
      if (INS) { p.prof[cta * 32 + 9] = tw; p.prof[cta * 32 + 15] = rq_gt(); }
    } else if (warp == PW + 1 && lane == 0 && XON) {
      uint64_t pol = 0;
      if (HINT) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));
      unsigned long long tw = 0;
      const int se = EARLY ? (KCS < MS * ST ? KCS : MS * ST) : 0;  // EARLY: stages 0 .. MS*ST-1 already issued
      int slot = 0, r = EARLY ? 1 : 0;
      for (int s = se; s < KCS; s += MS) {
        const int n = (KCS - s < MS) ? KCS - s : MS;
        if (s >= MS * ST) {
          const unsigned long long t0 = INS ? clock64() : 0;
          rq_wait((SELF ? full : (MRG ? wempty : xempty)) + 8 * slot, (r & 1) ^ 1);
          if (INS) tw += clock64() - t0;
        }
        rq_expect(full + 8 * slot, n * XB);
        for (int j = 0; j < n; j++) {
          if (HINT) rq_tileh(xring + slot * XBS + j * XB, &tx, (s0 + s + j) * 64, 0, full + 8 * slot, pol);
          else rq_tile(xring + slot * XBS + j * XB, &tx, (s0 + s + j) * 64, 0, full + 8 * slot);
        }
        if (++slot == ST) { slot = 0; r++; }
      }
      if (SELF) for (int s = (KCS > MS * ST ? KCS - MS * ST : 0); s < KCS; s += MS) rq_wait(full + 8 * ((s / MS) % ST), ((s / MS) / ST) & 1);
      if (INS) p.prof[cta * 32 + 10] = tw;
    } else if (ZI && warp >= PW + 2) {  // in-kernel zeroing: this CTA's 1/KS token slice of its row block's 128 out columns,
      // then a release increment of the block's counter; every reduce-add into the block first acquires counter == KS
      const int t = tid - (PW + 2) * 32, r0 = (int)((long long)ks * p.m / KS), r1 = (int)((long long)(ks + 1) * p.m / KS);
      for (int e = t; e < (r1 - r0) * (CW / 8); e += 64) {
        __nv_bfloat16* d = p.out + (size_t)(r0 + e / (CW / 8)) * p.N + cb + (e % (CW / 8)) * 8;
        asm volatile("st.global.v4.u32 [%0], {%1, %1, %1, %1};\n" :: "l"(d), "r"(0) : "memory");
      }
      asm volatile("fence.proxy.async.global;\n" ::: "memory");  // generic zero stores -> the TMA (async proxy) adds
      asm volatile("bar.sync 4, 64;\n" ::: "memory");
      if (t == 0) {
        asm volatile("red.release.gpu.global.add.u32 [%0], 1;\n" :: "l"(p.cnt + blockIdx.x) : "memory");
        // wait (off the critical path, while the mainloop runs) until every CTA of the block zeroed its slice; publish it
        // to the epilogue through a shared flag. Bounded: a non-co-resident grid traps instead of hanging.
        uint32_t c = 0;
        for (int it = 0; it < (1 << 22); it++) {
          asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(c) : "l"(p.cnt + blockIdx.x) : "memory");
          if ((int)c >= KS) break;
          __nanosleep(128);
        }
        if ((int)c < KS) asm volatile("trap;");  // not co-resident: fail loudly rather than reduce into unzeroed memory
        asm volatile("fence.proxy.async.global;\n" ::: "memory");
        asm volatile("st.release.cta.shared.u32 [%0], 1;\n" :: "r"(zflag) : "memory");
        // the last of the block's KS pollers resets both counters for the next launch (the kernel boundary orders it)
        if (atomicAdd(p.cnt + gridDim.x + blockIdx.x, 1) == KS - 1) { p.cnt[blockIdx.x] = 0; p.cnt[gridDim.x + blockIdx.x] = 0; }
      }
      if (WK && warp == PW + 3) walk();
    } else if (WK && warp == PW + 3) {
      walk();
    }
    return;
  }
  // ---------------- math warpgroups: warp w of WG g decodes W rows 64g + 16w .. +15 of the block (= tile 4g + w = warp)
  if constexpr (MREG == 232) asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
  else if constexpr (MREG == 128) asm volatile("setmaxnreg.inc.sync.aligned.u32 128;\n");
  else asm volatile("setmaxnreg.inc.sync.aligned.u32 96;\n");
  if (SELF || (HALF && warp >= 4)) return;
  const int g = warp >> 2, w = warp & 3;
  int ei = 0;
  uint32_t nx = __shfl_sync(0xffffffffu, ec, 0), nxs = nx >> 17;
  uint32_t Tb = p.Tb;
  asm volatile("" : "+r"(Tb));
  const uint32_t wl = wring + warp * 1536 + lane * 16;
  const uint64_t xd0 = rq_desc(xring);
  float D[ND];  // never zero-filled: scale-d 0 on the CTA's first k16 (a generic write would serialise the pipe, C7515)
  uint32_t A0[GS][16], A1[GS][16];
#pragma unroll
  for (int j = 0; j < GS; j++)
#pragma unroll
    for (int i = 0; i < 16; i++) A0[j][i] = A1[j][i] = 0u;
  uint4 Q[GS][3];
#pragma unroll
  for (int j = 0; j < GS; j++) Q[j][0] = Q[j][1] = Q[j][2] = make_uint4(0, 0, 0, 0);
  uint32_t sink = 0;
  unsigned long long pc[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, tq = 0, tq0 = 0;
#define RQ_T0() do { if (INS) tq = clock64(); } while (0)
#define RQ_T1(i) do { if (INS) { const unsigned long long t_ = clock64(); pc[i] += t_ - tq; tq = t_; } } while (0)
  if (INS) { tq0 = clock64(); tq = tq0; }
  const unsigned long long gtl0 = INS ? rq_gt() : 0;
  unsigned long long gtw0 = 0;
  const int NGF = KCS / GS, NGT = (KCS + GS - 1) / GS;  // full rounds, all rounds
  // round G = stages GS*G .. GS*G + GS - 1 (the last round may be partial)
  auto ldr = [&](int G) {  // full waits, LDS of this warp's 48 B per lane per stage, W slots released
#pragma unroll
    for (int j = 0; j < GS; j++) {
      const int s = GS * G + j;
      if (s < KCS) {
        const int slot = RES ? 0 : (s / MS) % ST;
        if (!RES && (j % MS) == 0) rq_wait(full + 8 * slot, ((s / MS) / ST) & 1);
        if (PRB != 1 && PRB != 2) {
          const uint32_t a = wl + slot * WBS + (s % MS) * WB;
          Q[j][0] = rq_lds(a); Q[j][1] = rq_lds(a + 512); Q[j][2] = rq_lds(a + 1024);
        }
      }
    }
    RQ_T1(1);
    if (!RES && !MRG) {
      __syncwarp();
      if (lane == 0) {
#pragma unroll
        for (int j = 0; j < GS; j += MS) if (GS * G + j < KCS) rq_arrive(wempty + 8 * (((GS * G + j) / MS) % ST));
      }
    }
    RQ_T1(2);
  };
  auto xrel = [&](int G) {
    if (!RES) {
      __syncwarp();
      if (lane == 0) {
#pragma unroll
        for (int j = 0; j < GS; j += MS) if (GS * G + j < KCS) {
          const uint32_t b = (MRG ? wempty : xempty) + 8 * (((GS * G + j) / MS) % ST);
          if (RLAX) rq_arrive_r(b); else rq_arrive(b);
        }
      }
    }
    RQ_T1(7);
  };
  auto dec = [&](int G, int ns, uint32_t (&A)[GS][16]) {
    if (INS) {  // force the LDS results in: time the data wait separately from the decode ALU work
      uint32_t t = 0;
#pragma unroll
      for (int j = 0; j < GS; j++) t ^= Q[j][0].x ^ Q[j][0].w ^ Q[j][1].x ^ Q[j][1].w ^ Q[j][2].x ^ Q[j][2].w;
      asm volatile("" : "+r"(t));
      sink += t & 1;
      RQ_T1(4);
    }
#pragma unroll
    for (int j = 0; j < GS; j++) {
      if (j < ns) {
        if (PRB == 10) {
#pragma unroll
          for (int i = 0; i < 16; i++) A[j][i] = Tb + i * lane;
        } else {
          if constexpr (CURP) {
            uint32_t H[8];
            rq_hi8(Q[j][0], Q[j][1], Q[j][2], Tb, H);
            rq_fixp8(H, GS * G + j, E4, hs);
            rq_prmtA(Q[j][0], Q[j][1], H, A[j]);
          } else {
          const uint32_t* const tb = ptabp + ((((GS * G + j) / MS) % ST) * 8 + (warp & 7)) * C::PTW;
          const uint32_t ph = WK ? tb[0] : 0u, pw0 = WK ? tb[3 + lane * 2] : 0u, pw1 = WK ? tb[4 + lane * 2] : 0u;
          if (DILP) rq_unit_ilp(Q[j][0], Q[j][1], Q[j][2], Tb, A[j]);
          else rq_unit(Q[j][0], Q[j][1], Q[j][2], Tb, A[j]);
          if (WK) {
            if (__builtin_expect(ph != 0u, 0)) { rq_patch(A[j], pw0); rq_patch(A[j], pw1); }
          } else if constexpr (CUR && VASM) rq_fixw(A[j], GS * G + j, E4, hs);
          else if constexpr (CUR && VOTE) rq_fixv(A[j], GS * G + j, E4, hs);
          else if constexpr (CUR) rq_fixc(A[j], GS * G + j, E4, hs);
          else if (NOE) { } else if (FX2) rq_fix2(A[j], GS * G + j, nx, nxs, ei, ec, el, lane);
          else if (WPF) rq_fixp(A[j], GS * G + j, nx, nxs, ei, ec, el, lane);
          else rq_fix(A[j], GS * G + j, nx, nxs, ei, ec, el, lane);
          }
        }
      }
    }
    RQ_T1(3);
  };
  auto mma = [&](int G, auto NSC, uint32_t (&A)[GS][16]) {  // NSC: compile-time stage count of this round
    constexpr int NS = decltype(NSC)::value;
    rq_facc<ND>(D);
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
    for (int j = 0; j < NS; j++) {
      const int s = GS * G + j;
      const uint64_t d = xd0 + (uint64_t)(((RES ? 0 : ((s / MS) % ST) * XBS + (s % MS) * XB)) >> 4);
#pragma unroll
      for (int q = 0; q < 4; q++) rq_wgmma<NT>(D, A[j] + 4 * q, d + 2 * q, (s | q) != 0);
    }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    rq_facc<ND>(D);
    if (INS && G == 0) gtw0 = rq_gt();
    RQ_T1(5);
  };
  auto fa = [&](uint32_t (&A)[GS][16]) {
#pragma unroll
    for (int j = 0; j < GS; j++) rq_fa(A[j]);
  };
  if (PRB == 1 || PRB == 2 || PRB == 3) {
#pragma unroll 1
    for (int G = 0; G < NGT; G++) {
      ldr(G);
      if (PRB == 3) {
        dec(G, KCS - GS * G, A0);
#pragma unroll
        for (int j = 0; j < GS; j++)
#pragma unroll
          for (int i = 0; i < 16; i++) sink ^= A0[j][i];
      }
      if (PRB != 1) xrel(G);
    }
  } else if (PRB == 0 || PRB == 4 || RES) {
    // WM 0: dec(G) -> issue(G) -> LDS(G+1) -> wait_group 1 -> release x(G-1)  (decode on the wait->issue chain)
    // WM 1: issue(G) -> LDS(G+1) + dec(G+1) under wgmma(G) -> wait_group 0 -> release x(G) at once
    // WM 2: issue(G) -> LDS(G+1) + dec(G+1) -> wait_group 1 -> release x(G-1): the decode of the next round runs under
    //       the two queued groups; A triple-buffered (G-1 still in flight, G queued, G+1 being written)
    uint32_t A2[GS][16];
#pragma unroll
    for (int j = 0; j < GS; j++)
#pragma unroll
      for (int i = 0; i < 16; i++) A2[j][i] = 0u;
    auto round = [&](int G, auto NSC, uint32_t (&A)[GS][16], uint32_t (&An)[GS][16], uint32_t (&Ap)[GS][16]) {
      if (WM == 0) {
        dec(G, decltype(NSC)::value, A);
        mma(G, NSC, A);
        if (G + 1 < NGT) ldr(G + 1);
        asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
        fa(Ap);
        RQ_T1(6);
        if (G > 0) xrel(G - 1);
      } else if (WM == 1) {
        mma(G, NSC, A);
        if (G + 1 < NGT) { ldr(G + 1); dec(G + 1, KCS - GS * (G + 1), An); }
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
        fa(A);
        RQ_T1(6);
        xrel(G);
      } else {
        mma(G, NSC, A);
        if (G + 1 < NGT) { ldr(G + 1); dec(G + 1, KCS - GS * (G + 1), An); }
        asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
        fa(Ap);
        RQ_T1(6);
        if (G > 0) xrel(G - 1);
      }
    };
    using FULLR = std::integral_constant<int, GS>;
    using ONER = std::integral_constant<int, 1>;
    ldr(0);
    if (WM != 0) dec(0, GS, A0);
    int G = 0;
    if (WM == 2) {  // buffer of round G = G % 3; host guarantees KCS >= GS
      round(0, FULLR{}, A0, A1, A2);
      G = 1;
#pragma unroll 1
      for (; G + 2 < NGF; G += 3) { round(G, FULLR{}, A1, A2, A0); round(G + 1, FULLR{}, A2, A0, A1); round(G + 2, FULLR{}, A0, A1, A2); }
      if (G < NGF) { round(G, FULLR{}, A1, A2, A0); G++; }
      if (G < NGF) { round(G, FULLR{}, A2, A0, A1); G++; }
      if (GS > 1 && NGT > NGF) {
        const int b = G % 3;
        if (b == 0) round(G, ONER{}, A0, A1, A2);
        else if (b == 1) round(G, ONER{}, A1, A2, A0);
        else round(G, ONER{}, A2, A0, A1);
        G++;
      }
    } else {
      round(0, FULLR{}, A0, A1, A1);
      G = 1;
#pragma unroll 1
      for (; G + 1 < NGF; G += 2) { round(G, FULLR{}, A1, A0, A0); round(G + 1, FULLR{}, A0, A1, A1); }
      if (G < NGF) { round(G, FULLR{}, A1, A0, A0); G++; }
      if (GS > 1 && NGT > NGF) {  // partial last round (KCS % GS stages; GS = 2: one stage), buffer parity by G
        if (G & 1) round(G, ONER{}, A1, A0, A0);
        else round(G, ONER{}, A0, A1, A1);
        G++;
      }
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    rq_facc<ND>(D);
    fa(A0); fa(A1); fa(A2);
    if (WM != 1) xrel(NGT - 1);
  }
  if (INS) pc[0] = clock64() - tq0;
  const unsigned long long gtl1 = INS ? rq_gt() : 0;
  RQ_T0();
  if (PRB == 0 || PRB == 7) {  // ---------------- epilogue: stmatrix.trans -> SW128 slab [NT tok][64 n] -> TMA reduce-add
    if (PRB == 7) {
#pragma unroll
      for (int i = 0; i < ND; i++) D[i] = (float)((i + tid) & 7) * 0.125f;
    }
    if constexpr (FL & 65536) {
    const unsigned long long ce0 = INS ? clock64() : 0;
    unsigned long long ce[5] = {0, 0, 0, 0, 0};
    asm volatile("bar.sync 1, %0;\n" :: "r"(NW * 32) : "memory");  // both WGs retired every wgmma: the x ring is free
    if (INS) ce[0] = clock64() - ce0;
    const uint32_t slab = xring + g * XB;
    const int j = lane >> 3, rr = lane & 7;
    // rs128b FL bit 16 (EH): two token halves per warpgroup slab, the first half's reduce-add issued before the second
    // half's stmatrix (the TMA read of half 0 overlaps the stmatrix of half 1); the out map's box is NT / 2 tokens.
    // FL bit 17 (EP): the issuing thread polls the zero flag before its stmatrix instead of after the slab barrier.
    constexpr bool EH = (FL & 65536) != 0, EP = false;  // EH only: historical EP is deliberately unported
    const bool issuer = (w == 0 && lane == 0);
    auto zpoll = [&]() {
      if (ZI) {  // the block's zeroing is complete (flag set by warp 10 long before; bounded spin, trap on timeout)
        uint32_t f = 0;
        for (int it = 0; it < (1 << 24) && !f; it++) asm volatile("ld.acquire.cta.shared.u32 %0, [%1];\n" : "=r"(f) : "r"(zflag) : "memory");
        if (!f) asm volatile("trap;");
        asm volatile("fence.proxy.async.global;\n" ::: "memory");
      }
    };
    if (EP && issuer) zpoll();
    auto stsm = [&](int pp) {
      const int i0 = 2 * pp, i = i0 + (j >> 1), ch = 2 * w + (j & 1), tok = 8 * i + rr;
      const uint32_t a = slab + tok * 128 + ((ch ^ rr) << 4);
      const uint32_t r0 = rq_pk(D[4 * i0], D[4 * i0 + 1]), r1 = rq_pk(D[4 * i0 + 2], D[4 * i0 + 3]);
      const uint32_t r2 = rq_pk(D[4 * i0 + 4], D[4 * i0 + 5]), r3 = rq_pk(D[4 * i0 + 6], D[4 * i0 + 7]);
      asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n" :: "r"(a), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
    };
    auto issue = [&](int tok0, uint32_t src) {
      if (FL & 4) asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n"
                               :: "l"((unsigned long long)&to), "r"(cb + 64 * g), "r"(tok0), "r"(src) : "memory");
      else if (!(FL & 8)) asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%0, {%1, %2}], [%3];\n"
                                       :: "l"((unsigned long long)&to), "r"(cb + 64 * g), "r"(tok0), "r"(src) : "memory");
    };
    if (EH) {
#pragma unroll
      for (int pp = 0; pp < NT / 32; pp++) stsm(pp);
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      asm volatile("bar.sync %0, 128;\n" :: "r"(2 + g) : "memory");
      if (issuer) {
        if (!EP) zpoll();
        issue(0, slab);
        asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
      }
#pragma unroll
      for (int pp = NT / 32; pp < NT / 16; pp++) stsm(pp);
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      asm volatile("bar.sync %0, 128;\n" :: "r"(2 + g) : "memory");
      if (INS) ce[1] = clock64() - ce0;
      if (issuer) {
        if (INS) ce[2] = clock64() - ce0;
        issue(NT / 2, slab + (NT / 2) * 128);
        asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
        if (INS) ce[3] = clock64() - ce0;
        asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
        if (INS) ce[4] = clock64() - ce0;
      }
    } else {
#pragma unroll
      for (int pp = 0; pp < NT / 16; pp++) stsm(pp);
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      asm volatile("bar.sync %0, 128;\n" :: "r"(2 + g) : "memory");
      if (INS) ce[1] = clock64() - ce0;
      if (issuer) {
        if (!EP) zpoll();
        if (INS) ce[2] = clock64() - ce0;
        issue(0, slab);
        asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
        if (INS) ce[3] = clock64() - ce0;
        asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
        if (INS) ce[4] = clock64() - ce0;
      }
    }
    RQ_T1(8);
    if (INS && g == 0 && w == 0 && lane == 0) {
#pragma unroll
      for (int i = 0; i < 5; i++) p.prof[cta * 32 + 26 + i] = ce[i];
    }
    } else {
    const unsigned long long ce0 = INS ? clock64() : 0;
    unsigned long long ce[5] = {0, 0, 0, 0, 0};
    asm volatile("bar.sync 1, %0;\n" :: "r"(NW * 32) : "memory");  // both WGs retired every wgmma: the x ring is free
    if (INS) ce[0] = clock64() - ce0;
    const uint32_t slab = xring + g * XB;
    const int j = lane >> 3, rr = lane & 7;
#pragma unroll
    for (int pp = 0; pp < NT / 16; pp++) {
      const int i0 = 2 * pp, i = i0 + (j >> 1), ch = 2 * w + (j & 1), tok = 8 * i + rr;
      const uint32_t a = slab + tok * 128 + ((ch ^ rr) << 4);
      const uint32_t r0 = rq_pk(D[4 * i0], D[4 * i0 + 1]), r1 = rq_pk(D[4 * i0 + 2], D[4 * i0 + 3]);
      const uint32_t r2 = rq_pk(D[4 * i0 + 4], D[4 * i0 + 5]), r3 = rq_pk(D[4 * i0 + 6], D[4 * i0 + 7]);
      asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n" :: "r"(a), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    asm volatile("bar.sync %0, 128;\n" :: "r"(2 + g) : "memory");
    if (INS) ce[1] = clock64() - ce0;
    if (w == 0 && lane == 0) {
      if (ZI) {  // the block's zeroing is complete (flag set by warp 10 long before; bounded spin, trap on timeout)
        uint32_t f = 0;
        for (int it = 0; it < (1 << 24) && !f; it++) asm volatile("ld.acquire.cta.shared.u32 %0, [%1];\n" : "=r"(f) : "r"(zflag) : "memory");
        if (!f) asm volatile("trap;");
        if (!EFENCE) asm volatile("fence.proxy.async.global;\n" ::: "memory");
      }
      if (INS) ce[2] = clock64() - ce0;
      if (FL & 4) asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n"
                               :: "l"((unsigned long long)&to), "r"(cb + 64 * g), "r"(0), "r"(slab) : "memory");
      else if (!(FL & 8)) asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%0, {%1, %2}], [%3];\n"
                                       :: "l"((unsigned long long)&to), "r"(cb + 64 * g), "r"(0), "r"(slab) : "memory");
      asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
      if (INS) ce[3] = clock64() - ce0;
      asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
      if (INS) ce[4] = clock64() - ce0;
    }
    RQ_T1(8);
    if (INS && g == 0 && w == 0 && lane == 0) {
#pragma unroll
      for (int i = 0; i < 5; i++) p.prof[cta * 32 + 26 + i] = ce[i];
    }
    }
  } else {
    if (sink == 0x9e3779b9u || D[0] == 1.2345f) p.sink[0] = sink;
  }
  if (INS && w == 0 && lane == 0) {
#pragma unroll
    for (int i = 0; i < 9; i++) p.prof[cta * 32 + 16 * g + i] = pc[i];
    if (g == 0) { p.prof[cta * 32 + 11] = gt0; p.prof[cta * 32 + 12] = gtl0; p.prof[cta * 32 + 13] = gtl1; p.prof[cta * 32 + 14] = rq_gt(); p.prof[cta * 32 + 25] = gtw0; }
  }
#undef RQ_T0
#undef RQ_T1
#endif
}

typedef CUresult (*RqEnc)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                          const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                          CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static_assert(sizeof(CUtensorMap) == sizeof(RqTmap), "CUtensorMap size");
// 2D bf16 map: dims {inner, outer}, box {64, box_outer}, 128B swizzle; out-of-range rows read as 0 and are not written
static int rq_map(RqTmap* tm, const void* ptr, int64_t inner, int64_t outer, int box_outer) {
  static RqEnc enc = nullptr;
  if (!enc) {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult q;
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q) != cudaSuccess || !fn) { cudaGetLastError(); return -20; }
    enc = (RqEnc)fn;
  }
  if (((uintptr_t)ptr) % 16 || (inner * 2) % 16 || box_outer < 1 || box_outer > 256) return -21;
  cuuint64_t dims[2] = {(cuuint64_t)inner, (cuuint64_t)outer};
  cuuint64_t strides[1] = {(cuuint64_t)inner * 2};
  cuuint32_t box[2] = {64, (cuuint32_t)box_outer};
  cuuint32_t estr[2] = {1, 1};
  return enc((CUtensorMap*)tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)ptr, dims, strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
             CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS ? 0 : -22;
}
template <int NT, int ST, int GS, int PRB, int WM, int INS, int FL, int MS>
static int rq_go(RqP& p, torch::Tensor x, torch::Tensor out, int KS, int info) {
  using C = RqC<NT, ST, GS, PRB, WM, INS, FL, MS>;
  auto k = rq<NT, ST, GS, PRB, WM, INS, FL, MS>;
  if (info) {
    cudaFuncAttributes a;
    if (cudaFuncGetAttributes(&a, k) != cudaSuccess) { cudaGetLastError(); return -1; }
    if (info == 4) {  // wf_rq32b: CTAs per SM at this launch geometry (regs + smem)
      int nb = 0;
      cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
      if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k, (FL & 1024) ? 256 : 384, C::SMEM) != cudaSuccess) { cudaGetLastError(); return -1; }
      return nb;
    }
    return info == 1 ? a.numRegs : info == 2 ? (int)a.localSizeBytes : C::SMEM;
  }
  if (p.m < 1 || p.m > NT || KS < 1 || p.KC / KS < GS || KS > 64) return -4;
  if constexpr (NT == 32 && (ST == 4 || ST == 5) && FL == 52) {
    if (p.m != 32 || p.N != 8192 || p.K != 1024 || p.KC != 16 || KS != 1 || p.walias != 0) return -45;
  }
  if (p.ncnt < 2 * p.NB * ((FL & 16) ? 2 : 1)) return -5;
  RqTmap tx, to;
  int r = rq_map(&tx, x.data_ptr(), p.K, p.m, NT);
  if (r) return r;
  r = rq_map(&to, out.data_ptr(), p.N, p.m, (FL & 65536) ? NT / 2 : NT);
  if (r) return r;
  static bool once = false;
  if (!once) {
    if (cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM) != cudaSuccess) { cudaGetLastError(); return -6; }
    once = true;
  }
  k<<<dim3(p.NB * ((FL & 16) ? 2 : 1), KS), dim3((FL & 1024) ? 256 : 384), C::SMEM, at::cuda::getCurrentCUDAStream()>>>(p, tx, to);
  const cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? 0 : (int)e;
}
int64_t rqrun(int64_t v, torch::Tensor W, torch::Tensor x, torch::Tensor out, torch::Tensor esc, torch::Tensor esc2, torch::Tensor cnt, int64_t KS,
              int64_t Tb, int64_t walias, torch::Tensor prof, torch::Tensor sink, int64_t info) {
  RqP p = {};
  if (!info) {
    if (W.dim() != 4 || W.size(2) != 8 || W.size(3) != 1536 || x.dim() != 2 || out.dim() != 2 || x.size(0) != out.size(0)) return -10;
    if (x.size(1) != W.size(1) * 64 || out.size(1) != W.size(0) * 128 || x.scalar_type() != torch::kBFloat16 || out.scalar_type() != torch::kBFloat16) return -11;
    const int64_t nl = KS * W.size(0) * 8;
    if (esc.numel() < nl * 32 + 32 || (esc.numel() - 32) % nl || sink.numel() < 1) return -12;
    if (cnt.numel() < 2 * W.size(0) || cnt.scalar_type() != torch::kInt32) return -13;
    p.ncnt = (int)cnt.numel();
    if (esc2.numel() && esc2.numel() < nl * 128) return -14;
    p.W = (const uint8_t*)W.data_ptr(); p.esc = (const uint32_t*)esc.data_ptr(); p.ecap = (int)((esc.numel() - 32) / nl);
    p.esc2 = esc2.numel() ? (const uint32_t*)esc2.data_ptr() : nullptr;
    p.out = (__nv_bfloat16*)out.data_ptr(); p.cnt = cnt.data_ptr<int>();
    p.prof = prof.numel() >= 2 * KS * W.size(0) * 32 ? (unsigned long long*)prof.data_ptr() : nullptr;
    p.sink = (uint32_t*)sink.data_ptr();
    p.m = (int)x.size(0); p.K = (int)x.size(1); p.N = (int)out.size(1); p.KC = (int)W.size(1); p.NB = (int)W.size(0);
    p.walias = (int)walias; p.Tb = (uint32_t)(Tb & 0xffffffff);
  }
  switch (v) {
@@CASES@@
  }
  return -2;
}
"""


    def _rqwg(n):
        """rq_wgmma<n>: wgmma.mma_async m64n<n>k16 f32 += bf16 (A: 4 registers) x bf16 (B: smem descriptor), scale-d from a
    register (0 on the CTA's first k16: no accumulator zero-fill), scale-a = scale-b = 1, trans-b = 0"""
        nd = n // 2
        d = ", ".join("%%%d" % i for i in range(nd))
        a = ", ".join("%%%d" % (nd + i) for i in range(4))
        outs = ", ".join('"+f"(D[%d])' % i for i in range(nd))
        return (("template <> __device__ __forceinline__ void rq_wgmma<%d>(float (&D)[%d], const uint32_t* a, uint64_t desc, int scd) {\n"
                 "  asm volatile(\"{\\n.reg .pred p;\\nsetp.ne.b32 p, %%%d, 0;\\n"
                 "wgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 {%s}, {%s}, %%%d, p, 1, 1, 0;\\n}\\n\"\n"
                 "    : %s\n    : \"r\"(a[0]), \"r\"(a[1]), \"r\"(a[2]), \"r\"(a[3]), \"l\"(desc), \"r\"(scd));\n}\n")
                % (n, nd, nd + 5, n, d, a, nd + 4, outs))


    def _rqss(n):
        nd = n // 2
        d = ", ".join("%%%d" % i for i in range(nd))
        outs = ", ".join('"+f"(D[%d])' % i for i in range(nd))
        return (("template <> __device__ __forceinline__ void rq_wgss<%d>(float (&D)[%d], uint64_t da, uint64_t db, int scd) {\n"
                 "  asm volatile(\"{\\n.reg .pred p;\\nsetp.ne.b32 p, %%%d, 0;\\n"
                 "wgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 {%s}, %%%d, %%%d, p, 1, 1, 0, 0;\\n}\\n\"\n"
                 "    : %s\n    : \"l\"(da), \"l\"(db), \"r\"(scd));\n}\n")
                % (n, nd, nd + 2, n, d, nd, nd + 1, outs))


    def _rqsrc(vs):
        cases = "\n".join("    case %d: return rq_go<%s>(p, x, out, (int)KS, (int)info);" % (i, ", ".join(str(a) for a in v[1:8] + (RQMS.get(v[0], 1),)))
                          for i, v in enumerate(vs))
        return RQCU.replace("@@WGMMA@@", _rqwg(16) + _rqwg(32) + _rqwg(128) + _rqwg(256) + _rqss(128) + _rqss(256)).replace("@@CASES@@", cases)


    RQFLAGS = ["-gencode=arch=compute_90a,code=sm_90a", "-O3", "-DNDEBUG", "--expt-relaxed-constexpr"]
    _rqmod = [None, None]
    _rqmod2 = [None, None]  # rs32: the m = 16 / 32 instances, a second extension built on its own thread


    def _rqext(vs=None, verbose=False, slot=None):
        vs = RQV if vs is None else vs
        src = _rqsrc(vs)
        slot = _rqmod if slot is None else slot
        if slot[0] is None or slot[1] != src:
            import os
            tag = ("rq128_%x" if slot is _rqmod else "rq32_%x") % (hash(src) & 0xffffffff)
            flags = RQFLAGS + (["-Xptxas=-v"] if verbose else [])
            slot[0] = load_inline(tag + ("v" if verbose else ""), cpp_sources="int64_t rqrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor, int64_t);",
                                    cuda_sources=src, functions=["rqrun"], extra_cuda_cflags=flags, verbose=verbose)
            slot[1] = src
        return slot[0]


    def _pack_nat(w):
        """pw10 _pack_nat: bf16 [N, K] -> A fragments [N/16, K/64, 4 steps, 32 lanes, 8] in natural k order (wgmma B = x
    in natural k): step q, lane g*4+t, value v = (kh, half, e) holds W[16T + 8*half + g][64c + 16q + 8*kh + 2t + e]"""
        N, K = w.shape
        W = w.view(N // 16, 2, 8, K // 64, 4, 2, 4, 2)  # T, half, g, c, q, kh, t, e
        return W.permute(0, 3, 4, 2, 6, 5, 1, 7).reshape(N // 16, K // 64, 4, 32, 8).contiguous()


    def _rqpack(w):
        """12-bit pack re-blocked to 128-row blocks: [N/128][K/64][8 tiles][1536 B] (one 12,288 B bulk copy per stage),
    the decode table word, and the escape rows (tile, chunk, step, lane, value, xor); None if the pack is not exact"""
        N, K = w.shape
        c = _c12(_pack_nat(w), 1)
        if c is None or N % 128:
            return None
        data, tab, esc = c
        T, KC = data.shape[:2]
        lay = data.reshape(N // 128, 8, KC, 1536).permute(0, 2, 1, 3).contiguous()
        return lay, tab, esc


    def _rqunpack(lay, tab, esc, N, K):
        """independent check of the re-blocking: unpack [NB][KC][8][1536] with _unc12 and undo _pack_nat -> bf16 [N, K]"""
        NB, KC = lay.shape[:2]
        data = lay.permute(0, 2, 1, 3).reshape(NB * 8, KC, 3, 32, 16)
        v = _unc12(data, tab, esc, 1)  # [T, KC, 4, 32, 8] int32 bf16 bits
        W = v.view(N // 16, K // 64, 4, 8, 4, 2, 2, 2)  # T, c, q, g, t, kh, half, e
        return W.permute(0, 6, 3, 1, 2, 5, 4, 7).reshape(N, K)  # int32 bf16 bits (compare with w.view(int16) & 0xffff)


    def _rqesc(esc6, NB, KC, KS, dev):
        """escape lists for grid (NB, KS): one fixed-stride slot of CAP entries per (ks, nb, warp) (CAP a multiple of 32,
    > the longest list, so every list ends in a sentinel), + 32 trailing sentinels; int32 [KS*NB*8*CAP + 32]"""
        G = KS * NB * 8
        if len(esc6):
            tile, ch, q, ln, v, xr = [t.long() for t in esc6.t()]
            nb, wt = tile // 8, tile % 8
            s0 = torch.arange(KS + 1, device=dev, dtype=torch.long) * KC // KS
            ks = torch.searchsorted(s0, ch.contiguous(), right=True) - 1
            u = ch - s0[ks]
            ent = (u << 17) | (ln << 12) | ((q * 4 + v // 2) << 8) | ((v % 2) << 7) | xr
            gw = (ks * NB + nb) * 8 + wt
            o = torch.argsort(gw * 65536 + u)
            gw, ent = gw[o], ent[o]
            rank = torch.arange(len(gw), device=dev) - torch.searchsorted(gw, gw)
            cap = (int(torch.bincount(gw, minlength=G).max()) + 1 + 31) // 32 * 32
        else:
            cap = 32
        arr = torch.full((G * cap + 32,), 0xffffffff, dtype=torch.long, device=dev)
        if len(esc6):
            arr[gw * cap + rank] = ent
        arr = torch.where(arr >= 2 ** 31, arr - 2 ** 32, arr).int()
        return arr.contiguous(), torch.zeros(1, dtype=torch.int32, device=dev)


    def _rqesc2(esc6, NB, KC, KS, dev, CAPT=4):
        """FL bit 20 lists: one fixed slot of CAPT entries per (ks, nb, warp, lane), sorted by stage, sentinel-terminated
    -> int32 [KS*NB*8*32*CAPT].  Returns (arr, max entries in any slot); arr is None if a slot overflows CAPT."""
        G = KS * NB * 8
        if not len(esc6):
            return torch.full((G * 32 * CAPT,), 0xffffffff, dtype=torch.int32, device=dev), 0
        tile, ch, q, ln, v, xr = [t.long() for t in esc6.t()]
        nb, wt = tile // 8, tile % 8
        s0 = torch.arange(KS + 1, device=dev, dtype=torch.long) * KC // KS
        ks = torch.searchsorted(s0, ch.contiguous(), right=True) - 1
        u = ch - s0[ks]
        ent = (u << 17) | (ln << 12) | ((q * 4 + v // 2) << 8) | ((v % 2) << 7) | xr
        slot = ((ks * NB + nb) * 8 + wt) * 32 + ln
        o = torch.argsort(slot * 65536 + u)
        slot, ent = slot[o], ent[o]
        rank = torch.arange(len(slot), device=dev) - torch.searchsorted(slot, slot)
        mx = int(rank.max()) + 1
        if mx > CAPT:
            return None, mx
        arr = torch.full((G * 32 * CAPT,), 0xffffffff, dtype=torch.long, device=dev)
        arr[slot * CAPT + rank] = ent
        return torch.where(arr >= 2 ** 31, arr - 2 ** 32, arr).int().contiguous(), mx


    RQKS = {"pooler": 4, "unpooler": 2, "o": 2, "qkv_gate": 7}  # one-wave KS grid: N/128 x KS <= 132
    # the extension the submission builds: the box's favoured instances plus the judge's pick (s4s5z: no L2 hints won on the
    # H200 at all three m=128 shapes, job 846e25d9df09433f); <= 6 timed paths per layer
    RQPROD = ["s4s5hze", "d2s5hze", "d2s6hze", "d2s5hzem2", "s4s5hzeEH", "n256s5m1hzmEH", "n256s5m1hzm",
              "d2s5hzwP", "s4s5hzw"]
    RQKSV = {"h128s8e": 1, "h128s6e": 1, "h128s10e": 1, "h128s8em1": 1, "h128s8eg2": 1, "h128s10eg2": 1}  # variant -> KS
    RQBAKE_L = {  # per (layer, m) probe list (falls back to RQBAKE[m]); <= 6 timed paths per (layer, m)
        # wf_jsweep: the current pick (d2s5hze at pooler/qkv, s4s5hze at unpooler) + one-axis moves off it. The box ties
        # these within noise; the judge's own bake (H200, W L2-resident) is the instrument that can rank them.
        # pw52: proven pick first (pw49s's judge run picked d2s5hze at all three m=128 shapes); a variant replaces it
        # only if MARGIN_RQ128 (2%) faster in-bake.
        # pw57 (wf_jsweep5): + pw56's ST5 EH entry (hint on) at all three m=128 shapes; pw56's ST6 EH entry is
        # NOT added (FACTS 14: ring depths frozen at the proven ST5).
        # pw60 (wf_stack4): + wf_escur's per-thread escape cursor (FL bit 20) on the proven pick's own template
        # (d2s5hze -> d2s5hzw) and on the ST5 no-decode shape (s4s5hzw); FACTS 16: -1.38/-0.83/-0.88 us resident at
        # pooler/qkv/unpooler m=128, exact, regs unchanged. 6 timed paths at unpooler, 5 at pooler/qkv.
        ("unpooler", 128): ["d2s5hze", "d2s5hzwP", "s4s5hzw", "s4s5hzeEH", "s4s5hze", "d2s6hze"],
        ("pooler", 128): ["d2s5hze", "d2s5hzwP", "s4s5hzw", "s4s5hzeEH", "d2s6hze"],
        ("qkv_gate", 128): ["d2s5hze", "d2s5hzwP", "s4s5hzw", "s4s5hzeEH", "d2s6hze"],
        # pw58 (wf_jsweep6): the n256 WM=1/MRG/EH entries at m=256 (r6 is the baseline; unpooler/o stay r6)
        ("qkv_gate", 256): ["n256s5m1hzmEH", "n256s5m1hzm"],
        ("pooler", 256): ["n256s5m1hzmEH", "n256s5m1hzm"],
    }
    RQBAKE = {128: [], 256: []}  # wf_jsweep: o m=128 (s4s5hz 13.06 vs r6 11.17) and all m=256 (n256* 30.8-31.9 vs r6
    # 27.7-28.1) lost by >1.9 us on the judge -> dropped, r6 serves them (unchanged picks, 3 fewer instances to compile)
    RQLAYERS = ("pooler", "qkv_gate", "unpooler", "o")
    RQ = True
    _rqthr = []


    def _rqstart(prod=None, slot=None, thr=None, tag="rq"):
        """compile the RS128 extension in a background thread while r6's phase runs (CPU-only work; never fatal)"""
        import threading
        prod = RQPROD if prod is None else prod
        thr = _rqthr if thr is None else thr

        def build():
            try:
                _rqext([RQV[RQI[nm]] for nm in prod], slot=slot)
                thr.append("ok")
                thr.append(time.time() - _T0)
            except Exception as error:
                thr.append("failed: %s" % (str(error).splitlines()[-1][:200] if str(error) else type(error).__name__))
        t = threading.Thread(target=build, daemon=True)
        t.start()
        print("PREP %s thread started at %.0fs" % (tag, time.time() - _T0), flush=True)
        return t


    def _rqcall(st, p, x, out):
        e2 = p.get("esc2")
        if e2 is None:
            e2 = st.get("rq_e2e")
            if e2 is None:
                e2 = st["rq_e2e"] = torch.empty(0, dtype=torch.int32, device=x.device)
        return p.get("mod", _rqmod[0]).rqrun(p["i"], p["lay"], x, out, p["esc"], e2, p["cnt"], p["KS"], p["tab"], 0, st["rq_e"], st["rq_sink"], 0)


    def _rqval(fn, w, m, n, gen, tol):
        """one-hot x8 rows bit-exact against W over every K column (the split-K bf16 reduce-add must keep them exact),
    random rows within the KS-dependent tolerance (judge: 1e-2, rows 5x)"""
        K = w.shape[1]
        out = torch.empty((m, n), dtype=torch.bfloat16, device=w.device)
        exact = w[:n].t()
        if K not in _EYE:
            _EYE[K] = torch.eye(K, dtype=torch.bfloat16, device=w.device) * 8.0
        eye = _EYE[K]
        for start in range(0, K, m):
            x = eye[start:start + m]
            if x.shape[0] < m:
                x = torch.cat((x, torch.zeros((m - x.shape[0], K), dtype=x.dtype, device=x.device)))
            if fn(x.contiguous(), out) != 0:
                return False, "launch error"
            got = out[: min(m, K - start)]
            want = exact[start:start + got.shape[0]] * 8.0
            if not torch.isfinite(got).all() or not torch.equal(got.float(), want.float()):
                return False, "one-hot mismatch at column %d" % start
        for dist in (1.0, 30.0):
            x = torch.randn((m, K), generator=gen, device=w.device)
            if dist != 1.0:
                x[:, torch.randperm(K, generator=gen, device=w.device)[: K // 100]] *= dist  # the task's 'outliers'
            x = x.to(torch.bfloat16)
            o = torch.empty((m, n), dtype=torch.bfloat16, device=w.device)
            fn(x, o)
            ref = x.double() @ w[:n].double().t()
            err = float((o.double() - ref).norm() / ref.norm())
            row = float(((o.double() - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-30)).max())
            if not (err <= tol and row <= 5 * tol and torch.isfinite(o).all()):
                return False, "random rel L2 %.2e row %.2e > %.0e" % (err, row, tol)
        return True, "ok %.1e" % err


    def _rqbake(st, thr):
        """RS128 bake: per (layer, m in 128 / 256) time r6's own path and each RS128 variant (graph replays from a cold L2,
    the judge's method), pick a variant only if it validates and beats r6 by MARGIN; kernel_fn serves picks at m exactly"""
        st["rq"] = {}
        if time.time() - _T0 > PREPARE_BUDGET_S - 45:
            print("BAKE rq skipped-budget %.0fs" % (time.time() - _T0), flush=True)
            return
        thr.join(timeout=max(1.0, PREPARE_BUDGET_S - 45 - (time.time() - _T0)))
        if not _rqthr or _rqthr[0] != "ok" or _rqmod[0] is None:
            print("BAKE rq build %s" % (_rqthr[0] if _rqthr else "not finished"), flush=True)
            return
        print("PREP rq thread ready at %.0fs (bake starts at %.0fs)" % (_rqthr[1], time.time() - _T0), flush=True)
        names = [v[0] for v in [RQV[RQI[nm]] for nm in RQPROD]]
        dev = st["pooler_raw"].device
        st["rq_e"] = torch.empty(0, dtype=torch.int64, device=dev)
        st["rq_sink"] = torch.zeros(4, dtype=torch.int32, device=dev)
        flush = torch.empty(64 << 20, dtype=torch.float32, device=dev)
        sink = torch.empty((), dtype=torch.float32, device=dev)
        gen = torch.Generator(device=dev).manual_seed(12828)
        for layer in RQLAYERS:
            if time.time() - _T0 > PREPARE_BUDGET_S - 30:
                print("BAKE rq %s skipped-budget" % layer, flush=True)
                continue
            ms_ = tuple(m_ for m_ in (128, 256) if _NC(layer, m_))
            if not ms_:
                continue
            w = st[layer + "_raw"]
            n, K = w.shape
            xs, r6a = {}, {}
            for m in ms_:  # r6 timed before the pack (critic F1: the first r6 timing after the pack read ~4 us high)
                xs[m] = torch.randn((m, K), generator=gen, device=dev).to(torch.bfloat16)
                o6 = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
                r6a[m] = _time_us(lambda xx, oo: _r6(xx, layer, n, st), xs[m], o6, flush, sink)
            pk = _rqpack(w)
            if pk is None or not torch.equal(_rqunpack(*pk, n, K), w.view(torch.int16).int() & 0xffff):
                print("BAKE rq %s pack not exact" % layer, flush=True)
                continue
            lay, tab, esc6 = pk
            st.setdefault("rq_pk", {})[layer] = pk  # rs32: the small-m phase (after gx) reuses the pack
            KS = RQKS[layer]
            _bl = RQBAKE_L.get((layer, 128), RQBAKE[128]) + RQBAKE_L.get((layer, 256), RQBAKE[256])
            ESC = {}
            for _kk in sorted(set([KS] + [RQKSV.get(nm, KS) for nm in _bl])):
                ESC[_kk] = _rqesc(esc6, lay.shape[0], lay.shape[1], _kk, dev)[0]
            esc = ESC[KS]
            esc2, mx2 = _rqesc2(esc6, lay.shape[0], lay.shape[1], KS, dev)
            print("BAKE rq %s esc2 slots max=%d %s" % (layer, mx2, "ok" if esc2 is not None else "OVERFLOW"), flush=True)
            cnt = torch.zeros(8 * lay.shape[0], dtype=torch.int32, device=dev)  # 2 per CTA, GX up to 4*NB (HALF x KS)
            # discarded warm-up: the first path timed after the pack reads ~3 us high (box: r6 31.3 vs 27.4, and an rq path
            # 27.8 vs its 24-25), which would bias whichever path is timed first
            _time_us(lambda xx, oo: _r6(xx, layer, n, st), xs[128], torch.empty((128, n), dtype=torch.bfloat16, device=dev), flush, sink, it=7)
            keep = False
            for m in ms_:
                x = xs[m]
                out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
                r6 = lambda xx, oo: _r6(xx, layer, n, st)
                r6med = _time_us(r6, x, out, flush, sink)
                print("BAKE %s m=%d r6 %.2f at %.0fs" % (layer, m, r6med, time.time() - _T0), flush=True)

                def _try(nm):
                    """validate one rq128 path; returns (fn, p) or None"""
                    ksv = RQKSV.get(nm, KS)
                    if esc2 is None and RQI[nm] in RQCUR:
                        return None
                    p = {"i": names.index(nm), "lay": lay, "esc": ESC[ksv], "esc2": esc2, "cnt": cnt, "KS": ksv, "tab": tab, "n": n}
                    fn = lambda xx, oo, p=p: _rqcall(st, p, xx, oo)
                    try:
                        r = fn(x, out)
                        torch.cuda.synchronize()
                        if r != 0:
                            print("BAKE %s m=%d rq:%s ERROR launch %d" % (layer, m, nm, r), flush=True)
                            return None
                        ok, msg = _rqval(fn, w, m, n, gen, RQTOL[KS])
                        torch.cuda.synchronize()
                        if int(cnt.abs().sum()):  # the zero handshake must leave its counters at 0 after every launch
                            cnt.zero_()
                            ok, msg = False, "counters not reset"
                        if not ok:
                            print("BAKE %s m=%d rq:%s INVALID %s" % (layer, m, nm, msg), flush=True)
                            return None
                    except Exception as error:
                        print("BAKE %s m=%d rq:%s ERROR %s" % (layer, m, nm, str(error).splitlines()[-1][:160]), flush=True)
                        return None
                    return fn, p

                # pw59 (wf_bake3): ONE reference per group. The proven pick (first entry of the list) is the reference
                # only if the box does not read it slower than r6 (window = half the group's margin); otherwise r6 is the
                # reference and the proven pick is an ordinary candidate. Every candidate's paired ratio is taken against
                # that same reference; the pick is the best ratio if it beats the margin, else the reference.
                names_m = RQBAKE_L.get((layer, m), RQBAKE[m])
                pv = names_m[0] if names_m else None
                refn, refp, refmed, reffn = "r6", None, r6med, r6
                cands, done = [], set()
                if pv is not None:
                    done.add(pv)
                    got = _try(pv)
                    if got is not None:
                        fn, p = got
                        us = _time_us(fn, x, out, flush, sink, base=r6)
                        print("BAKE %s m=%d rq:%s %.2f (ratio %.4f vs r6 %.2f) at %.0fs"
                              % (layer, m, pv, us, _LAST[0][0], r6med, time.time() - _T0), flush=True)
                        rn, rm, isp = _refsel("rq:" + pv, us, "r6", r6med, RWIN128)
                        if isp:
                            refn, refp, refmed, reffn = rn, p, rm, fn
                        else:
                            cands.append(("rq:" + pv, p, _LAST[0][0]))
                for nm in names_m:
                    if nm in done:
                        continue
                    if time.time() - _T0 > PREPARE_BUDGET_S - 30:
                        print("BAKE %s m=%d rq:%s skipped-budget %.0fs" % (layer, m, nm, time.time() - _T0), flush=True)
                        continue
                    got = _try(nm)
                    if got is None:
                        continue
                    fn, p = got
                    us = _time_us(fn, x, out, flush, sink, base=reffn)
                    print("BAKE %s m=%d rq:%s %.2f (ratio %.4f vs %s %.2f) at %.0fs"
                          % (layer, m, nm, us, _LAST[0][0], refn, refmed, time.time() - _T0), flush=True)
                    cands.append(("rq:" + nm, p, _LAST[0][0]))
                best = _pickrule((refn, refp, refmed), cands, MARGIN_RQ128)
                if best[1] is not None:
                    st["rq"][(layer, m)] = best[1]
                    keep = True
                print("PICK %s m=%d %s %.2f (ref %s %.2f)" % (layer, m, best[0], best[2], refn, refmed), flush=True)
                # pw60 (wf_stack4): the per-thread escape cursor is a fixed-latency win that the cold-L2 bake cannot
                # resolve (box bake |ratio-1| <= 1.5% at all three m=128 layers, i.e. no evidence against) but the
                # W-resident lab can: -1.25 / -0.68 / -0.86 us at pooler / qkv / unpooler m=128 (lab2 --plan d2cur,
                # FACTS 16), and judge_sim reads -0.74 us on the verify16 load sum. Override the m=128 pick when the
                # cursor variant validated in this bake.
                _fo = RQPICK128.get((layer, m))
                if _fo is not None:
                    for _t, _p, _r in cands:
                        if _t == "rq:" + _fo:
                            st["rq"][(layer, m)] = _p
                            print("PICK %s m=%d rq:%s (forced: cursor + short apply, resident -0.26/-0.13/-0.32 us, bake ratio %.4f)"
                                  % (layer, m, _fo, _r), flush=True)
                            break
            if not keep:
                del lay, esc, cnt
        torch.cuda.synchronize()


    # ---------------------------------------------------------------- rs32: rq at m exactly 32 / 16 (NT = m)
    # <= 4 timed paths per (layer, m): (variant, KS). Baked after gx against the incumbent kernel_fn would serve without rq
    # (st["sel"] if gx picked the shape, else r6's gv / cuBLAS choice); F1 baseline rule: one discarded warm-up timing, the
    # incumbent timed before and after the variants, the lower kept, both printed.
    RQS = True
    RQSBAKE = {  # box lab t1-t3: ST 5-6 best at both NT (3/4 and 8/12 slower); hints (h) and EARLY (e) -0.2..-0.3; qkv KS 6
        # (108 CTAs) -0.1..-0.3 vs KS 7; o: HALF (64 rows, KS 1, EARLY) only helps at m = 128 (h128s8e 8.54 vs cuBLAS 8.90)
        # wf_rq32b: the *2 names are the 2-CTA/SM HALF variants (256-thread CTA, 64 W rows, occ 2, grid.x = 2 NB, same KS);
        # they replace the never-winning no-hint / KS7 entries. Box lab c1-c3: -0.2..-0.45 us vs the incumbent at m=32/16.
        # wf_jsweep: the shipped pick + ring depth 4 / 7 and the 2-CTA HALF variant at K-split
        # +-1 (the box ties these; the judge's bake ranks them). o keeps its pw48 lists (o m=32's 0.13 us win is the only
        # o-side rq pick and is not part of this sweep).
        # wf_jsweep3: judge m32 bake t32s5hze-k7 14.14 / t32s6hze-k7 14.21 / t32s7hze-k7 14.62 / t32s4hze-k7 14.98 /
        # h32s5hze2-k6 14.82 / h32s5hze2-k8 18.43 -> drop the two losers, add t32s4hze at the pick's K-split +1 (k8) and
        # h32s5hze2 at K-split -1 (k5). No new build (both names already in RQPROD2).
        # box screen (pick h32s5hze2-k6 14.94): t32s5hze-k7 15.10 / t32s7hze-k7 15.39 / t32s6hze-k7 15.49 kept;
        # h32s5hze2-k5 15.46 (+3.5%) and t32s4hze-k8 19.94 (+33%) dropped.
        # pw52: proven pick first (pw49s's judge run: t32s6hze-k7 14.11); a variant replaces it only if MARGIN_RQ32
        # (3%) faster in-bake.
        # pw57 (wf_jsweep5): + the 2-CTA HALF shape at the proven K-split (7); no new build (name already in RQPROD2)
        # pw62 (wf_stack5): the NT=32 cursor twins first (forced pick t32s6hzw-k7; the rest are the fallback set if the
        # forced variant fails validation in this bake).
        ("qkv_gate", 32): [("t32s6hzw", 7), ("t32s5hzw", 7), ("t32s6hzwE", 7), ("t32s5hzwE", 7),
                           ("t32s6hze", 7), ("t32s5hze", 7)],
        # wf_jsweep2: h16s5hze2 at the pick's own K-split (7) - the judge's m16 pick is t16s5hze-k7 13.79, the HALF shape
        # was only timed at KS 6/8 (14.46/18.24); KS 7 reuses the compiled instance (no new build).
        # pw63 (wf_escur16): the NT=16 cursor twins appended as EXTRA entries at the proven K-split 7 (the forced pick is
        # t16s5hzw-k7; these are its fallback set and the bake's own record of the twins).
        ("qkv_gate", 16): [("t16s4hze", 7), ("t16s5hze", 7), ("t16s6hze", 7), ("h16s5hze2", 7), ("h16s5hze2", 6), ("h16s5hze2", 8),
                           ("t16s4hzw", 7), ("t16s5hzw", 7), ("h16s5hzw2", 7)],
        # wf_jsweep3: judge m32 bake t32s5hze-k2 10.50 / t32s6hze-k2 10.53 / t32s5ze-k2 10.62 / h32s5e-k1 11.04 /
        # h32s4e-k1 11.17 / r6 10.66 -> ST 4 and 7 on the pick, K-split +-1 on the pick, h32 2-CTA at K-split 1
        # (h32 2-CTA at K-split 3 was 29.18 on unpooler m32 -> not built). No new build.
        # box screen (pick r6 9.79): t32s5hze-k2 9.82 / t32s4hze-k2 9.92 / t32s7hze-k2 9.92 / h32s5hze2-k1 9.95 kept;
        # t32s5hze-k1 10.62 (+8.5%) and t32s5hze-k3 17.44 (+78%) dropped.
        # pw62 (wf_stack5): o keeps its bake (its HALF variants have escape cost 0) with the two cursor twins added at the
        # proven K-split 2; t32s4hze / t32s7hze (never won) dropped to hold the timed-path count at 6.
        ("o", 32): [("t32s6hzw", 2), ("t32s5hzw", 2), ("t32s6hze", 2), ("t32s5hze", 2), ("h32s5hze2", 1), ("h32s5hze2", 2)],
        ("o", 16): [("t16s5hze", 2), ("t16s6hze", 2), ("t16s5ze", 2), ("h16s5hze2", 2)],
        ("o", 128): [],  # wf_m128: HALF loses on the judge (latency-bound, box understates the gap) -> cuBLAS
        # pw62 (wf_stack5): cursor twins first (forced pick t32s6hzw-k4) + the fallback set.
        ("pooler", 32): [("t32s6hzw", 4), ("t32s5hzw", 4), ("t32s6hzwE", 4), ("t32s5hzwE", 4),
                         ("t32s6hze", 4), ("t32s5hze", 4)],
        ("pooler", 16): [("t16s5hze", 4), ("t16s4hze", 4), ("t16s6hze", 4), ("t16s7hze", 4), ("h16s5hze2", 3), ("h16s5hze2", 5)],
        # wf_jsweep3: judge m32 bake t32s6hze-k2 19.52 / t32s7hze-k2 19.87 / t32s4hze-k2 20.26 / r6 20.74 /
        # h32s5hze2-k1 22.27 / h32s5hze2-k3 29.18 -> K-split +-1 on the pick, ST 5, h32 2-CTA at K-split 1. No new build.
        # box screen: t32s5hze-k2 21.76 (NEW box pick, -1.6% vs the judge's t32s6hze-k2 22.11) / t32s7hze-k2 22.08 /
        # t32s6hze-k2 22.11 kept; h32s5hze2-k1 22.75 (+4.5%), t32s6hze-k1 23.20, t32s6hze-k3 30.72 dropped.
        # pw62 (wf_stack5): cursor twins first (forced pick t32s6hzw-k2) + the fallback set.
        ("unpooler", 32): [("t32s6hzw", 2), ("t32s5hzw", 2), ("t32s6hzwE", 2), ("t32s5hzwE", 2),
                           ("t32s6hze", 2), ("t32s5hze", 2)],
        ("unpooler", 16): [("t16s5hze", 2), ("t16s4hze", 2), ("t16s6hze", 2), ("t16s7hze", 2), ("h16s5hze2", 1), ("h16s5hze2", 3)],
    }
    RQCAP = {("o", 32): 6, ("pooler", 32): 6, ("qkv_gate", 32): 6, ("unpooler", 32): 6}  # pw62: timed candidates per (layer, m)
    RQPROD2 = list(dict.fromkeys(nm for ky, ps in RQSBAKE.items() if _NC(*ky) for nm, _ in ps))
    RQPROD = [nm for nm in RQPROD if not nm.startswith("n256") or _NCR("qkv_gate", (256,)) or _NCR("pooler", (256,)) or _NCR("unpooler", (256,))]
    _rqthr2 = []


    def _rqsbake(st, thr):
        if time.time() - _T0 > PREPARE_BUDGET_S - 30:
            print("BAKE rq32 skipped-budget %.0fs" % (time.time() - _T0), flush=True)
            return
        thr.join(timeout=max(1.0, PREPARE_BUDGET_S - 30 - (time.time() - _T0)))
        if not _rqthr2 or _rqthr2[0] != "ok" or _rqmod2[0] is None:
            print("BAKE rq32 build %s" % (_rqthr2[0] if _rqthr2 else "not finished"), flush=True)
            return
        print("PREP rq32 thread ready at %.0fs" % _rqthr2[1], flush=True)
        names = RQPROD2
        dev = st["pooler_raw"].device
        st.setdefault("rq_e", torch.empty(0, dtype=torch.int64, device=dev))
        st.setdefault("rq_sink", torch.zeros(4, dtype=torch.int32, device=dev))
        flush = torch.empty(64 << 20, dtype=torch.float32, device=dev)
        sink = torch.empty((), dtype=torch.float32, device=dev)
        gen = torch.Generator(device=dev).manual_seed(3216)
        print("PREP rq32 bake starts at %.0fs" % (time.time() - _T0), flush=True)
        for layer in RQLAYERS:
            if not _NCR(layer, (32, 16, 128)):
                continue
            if time.time() - _T0 > PREPARE_BUDGET_S - 30:
                print("BAKE rq32 %s skipped-budget" % layer, flush=True)
                continue
            w = st[layer + "_raw"]
            n, K = w.shape
            pk = st.get("rq_pk", {}).get(layer)
            if pk is None:
                pk = _rqpack(w)
                if pk is None or not torch.equal(_rqunpack(*pk, n, K), w.view(torch.int16).int() & 0xffff):
                    print("BAKE rq32 %s pack not exact" % layer, flush=True)
                    continue
            lay, tab, esc6 = pk
            escs = {}
            cnt = torch.zeros(4 * lay.shape[0], dtype=torch.int32, device=dev)
            for m in (32, 16, 128):
                ps = RQSBAKE.get((layer, m), [])
                if not ps or not _NC(layer, m):
                    continue
                x = torch.randn((m, K), generator=gen, device=dev).to(torch.bfloat16)
                out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
                sp = st["sel"].get((layer, _bk(m)))
                rp0 = st["rq"].get((layer, m))  # m = 128: an rq pick of the m=128 phase is the incumbent
                if rp0 is not None:
                    inc, iname = (lambda xx, oo, rp0=rp0: _rqcall(st, rp0, xx, oo)), "rq128"
                elif sp is not None and sp[4] == n:
                    inc, iname = (lambda xx, oo, sp=sp: _gvcall(st, sp, xx, oo)), "sel"
                else:
                    inc, iname = (lambda xx, oo: _r6(xx, layer, n, st)), "r6"
                _time_us(inc, x, out, flush, sink, it=5)  # discarded warm-up
                imed = _time_us(inc, x, out, flush, sink)
                print("BAKE %s m=%d %s %.2f at %.0fs" % (layer, m, iname, imed, time.time() - _T0), flush=True)

                def _try(nm, KS):
                    """validate one rq32 path; returns (fn, p, tag) or None"""
                    if KS not in escs:
                        escs[KS] = _rqesc(esc6, lay.shape[0], lay.shape[1], KS, dev)[0]
                    p = {"i": names.index(nm), "lay": lay, "esc": escs[KS], "cnt": cnt, "KS": KS, "tab": tab, "n": n, "mod": _rqmod2[0]}
                    if RQI[nm] in RQCUR:  # pw62 (wf_escur32): the FL bit 20 cursor array, keyed on the k-split
                        if ("e2", KS) not in escs:
                            _a2, _mx = _rqesc2(esc6, lay.shape[0], lay.shape[1], KS, dev)
                            escs[("e2", KS)] = _a2
                            print("BAKE rq32 %s esc2 ks=%d slots max=%d %s" % (layer, KS, _mx, "ok" if _a2 is not None else "OVERFLOW"), flush=True)
                        if escs[("e2", KS)] is None:
                            print("BAKE %s m=%d rq:%s esc2 OVERFLOW" % (layer, m, nm, KS), flush=True)
                            return None
                        p["esc2"] = escs[("e2", KS)]
                    fn = lambda xx, oo, p=p: _rqcall(st, p, xx, oo)
                    tag = "%s-k%d" % (nm, KS)
                    if nm.endswith("2"):
                        e0 = st["rq_e"]
                        if _rqmod2[0].rqrun(p["i"], e0, e0, e0, e0, e0, e0, 0, 0, 0, e0, e0, 4) < 2:
                            print("BAKE %s m=%d rq:%s SKIP occ<2" % (layer, m, tag), flush=True)
                            return None
                    try:
                        r = fn(x, out)
                        torch.cuda.synchronize()
                        if r != 0:
                            print("BAKE %s m=%d rq:%s ERROR launch %d" % (layer, m, tag, r), flush=True)
                            return None
                        ok, msg = _rqval(fn, w, m, n, gen, RQTOL[KS])
                        torch.cuda.synchronize()
                        if int(cnt.abs().sum()):
                            cnt.zero_()
                            ok, msg = False, "counters not reset"
                        if not ok:
                            print("BAKE %s m=%d rq:%s INVALID %s" % (layer, m, tag, msg), flush=True)
                            return None
                    except Exception as error:
                        print("BAKE %s m=%d rq:%s ERROR %s" % (layer, m, tag, str(error).splitlines()[-1][:160]), flush=True)
                        return None
                    return fn, p, tag

                # pw59 (wf_bake3): one reference per group, same rule as _rqbake: the proven pick (PROVEN_RQ32) when
                # the box does not read it slower than the incumbent (window = half the margin), else the incumbent; every
                # candidate's paired ratio is taken against that same reference and the pick is the best ratio if it beats
                # the margin, else the reference.
                # pw62 (wf_stack5): the forced m=32 cursor pick. The variant is still VALIDATED here (one-hot exact on
                # every K column + random rows) but not ranked: the cold-L2 bake cannot resolve the cursor's fixed-latency
                # win (wf_escur32: streamed delta 0.00..+0.08 us, W-resident -0.70..-1.38 us) and the judge's own bake
                # flips m=32 picks on noise. If the forced variant fails validation the normal sweep below runs instead.
                _fo = RQPICK32.get((layer, m))
                _fmsg = "NT=32 cursor, resident -0.70..-1.38 us"
                if _fo is None:  # pw63 (wf_escur16)
                    _fo = RQPICK16.get((layer, m))
                    _fmsg = "NT=16 cursor, resident -0.66 us"
                if _fo is None:  # pw63 (coordinator): o m=32 HALF twin
                    _fo = RQPICK32B.get((layer, m))
                    _fmsg = "resident -0.64 us, judge-proven pw58"
                if _fo is not None:
                    _fnm, _fks = _fo.rsplit("-k", 1)
                    got = _try(_fnm, int(_fks))
                    if got is not None:
                        fn, p, tag = got
                        us = _time_us(fn, x, out, flush, sink, base=inc)
                        st["rq"][(layer, m)] = p
                        print("PICK %s m=%d rq:%s (forced: %s, bake %.2f vs %s %.2f)"
                              % (layer, m, tag, _fmsg, us, iname, imed), flush=True)
                        continue
                    print("BAKE %s m=%d rq:%s forced variant INVALID -> sweep" % (layer, m, _fo), flush=True)
                pv = PROVEN_RQ32.get((layer, m))
                refn, refp, refmed, reffn = iname, None, imed, inc
                cands, done = [], set()
                if pv is not None:
                    for nm, KS in ps[:RQCAP.get((layer, m), 6)]:
                        if "%s-k%d" % (nm, KS) != pv:
                            continue
                        done.add(pv)
                        got = _try(nm, KS)
                        if got is not None:
                            fn, p, tag = got
                            us = _time_us(fn, x, out, flush, sink, base=inc)
                            print("BAKE %s m=%d rq:%s %.2f (ratio %.4f vs %s %.2f) at %.0fs"
                                  % (layer, m, tag, us, _LAST[0][0], iname, imed, time.time() - _T0), flush=True)
                            rn, rm, isp = _refsel("rq:" + tag, us, iname, imed, RWIN32)
                            if isp:
                                refn, refp, refmed, reffn = rn, p, rm, fn
                            else:
                                cands.append(("rq:" + tag, p, _LAST[0][0]))
                        break
                for nm, KS in ps[:RQCAP.get((layer, m), 6)]:  # wf_jsweep: 6 timed paths per (layer, m)
                    tag = "%s-k%d" % (nm, KS)
                    if tag in done:
                        continue
                    if time.time() - _T0 > PREPARE_BUDGET_S - 30:
                        print("BAKE %s m=%d rq:%s skipped-budget %.0fs" % (layer, m, tag, time.time() - _T0), flush=True)
                        continue
                    got = _try(nm, KS)
                    if got is None:
                        continue
                    fn, p, tag = got
                    us = _time_us(fn, x, out, flush, sink, base=reffn)
                    print("BAKE %s m=%d rq:%s %.2f (ratio %.4f vs %s %.2f) at %.0fs"
                          % (layer, m, tag, us, _LAST[0][0], refn, refmed, time.time() - _T0), flush=True)
                    cands.append(("rq:" + tag, p, _LAST[0][0]))
                best = _pickrule((refn, refp, refmed), cands, MARGIN_RQ32)
                if best[1] is not None:
                    st["rq"][(layer, m)] = best[1]
                print("PICK %s m=%d %s %.2f (ref %s %.2f)" % (layer, m, best[0], best[2], refn, refmed), flush=True)

    return {'prepare': prepare, 'kernel_fn': kernel_fn, 'rqcall': _rqcall}

def _make_vjeux():
    # okc: problem=muse_bf16_proj
    # Based on github.com/vjeux/okc-2026 muse_bf16_proj.py (Apache-2.0). Copied for the OKC starting point.
    """Dense projections for Muse Spark Open: skinny swap-AB tensor-core GEMV/GEMM with
losslessly compressed BF16 weights.

Math
----
``out[m, n] = x[m, k] @ W[n, k]^T`` for the five projection matrices of one
model (``qkv_gate``, ``o``, ``pooler``, ``unpooler``, ``router``). The router's
FP32 weights run as a BF16 hi + lo pair concatenated along K with FP32
accumulation, as SGLang does above 8 tokens.

Lossless 12-bit weights (BF16 projections)
-------------------------------------------
Each BF16 weight is stored as its low byte (exponent bit 0 + 7 mantissa bits,
raw) and a 4-bit code for its high byte (sign + exponent bits 7..1, as an
offset into an 8-value window chosen per matrix, the window covering the most
weights). The kernel rebuilds the exact BF16 bits in registers (a few integer
ops per weight). The rare weights whose exponent falls outside the window
(tiny magnitudes) are exact-corrected: the kernel adds ``x[:, k] * (w -
w_decoded)`` for each of them (both factors exact in FP32), so every output
equals the product with the original weights up to FP32 accumulation order.
``prepare()`` rejects a packing whose escape share exceeds
``1 / MAX_ESCAPE_DENOMINATOR`` or whose correction is not exact in FP32, and
falls back to uncompressed fragments.

Why the kernels look the way they do
------------------------------------
Weights dominate: for m <= 64 every weight byte is streamed once from HBM
straight into registers, in mma.m16n8k16 fragment order (prepared once), and
multiplied against x (staged in shared memory) on BF16 tensor cores with FP32
accumulation. Split-K across warps (shared-memory reduction) and across CTAs
(FP32 partials + a self-resetting semaphore; the last CTA of a row block
reduces) keeps ~128 CTAs streaming on every shape.

Routes (chosen per layer and token count m by ``_static_choice`` / ``_autotune``)
---------------------------------------------------------------------------------
* ``sk2`` -- m <= 32, swapped operands, no cross-CTA reduction (the production
  skinny path on sm_90). ``skinny`` (TMA-streamed tensor-core GEMV) is the
  fallback / pre-sk2 path, kept for non-sm_90 and as the autotuner's
  alternative.
* ``cgemv`` / ``sgemv`` / ``rgemv`` -- m == 1 scalar GEMV specialists.
* ``wide`` / ``gemm`` / ``gq`` / ``gqx`` -- m 33..256 TMA + wgmma GEMMs
  (``wide`` is the BF16 route; the router uses ``wide``/``gemm``/``rk``).
* ``rk`` -- router-only split-K GEMM (order-dependent FP32 adds, inside the
  router tolerance).
* ``cutlass`` -- CUTLASS pingpong GEMM, pooler m == 256 only, built in the
  background during ``prepare()``; a failed build just drops the route.
* ``torch`` / ``torchT`` -- cuBLAS fallbacks (``F.linear`` / ``mm`` on a
  K-major copy). Any custom path that fails ``_selftest`` is disabled in
  favour of cuBLAS.

Design of this file
--------------------
* One Python file; native CUDA only, compiled once by
  ``torch.utils.cpp_extension.load_inline`` (``_get_ext`` / ``_build_cutlass``).
* The CUDA sources are string constants, kept verbatim: the kernels, tile
  sizes, and launch configs in them are the tuned behaviour and are not
  touched by this refactor. Python only assembles dispatch tables into them
  (``@@PLACEHOLDER@@``) and prepares their inputs.
* ``prepare(weights)`` builds every weight layout once (compressed streams,
  escape tables, workspaces) into ``state``; ``kernel_fn(data, state)`` runs
  one projection. Fresh outputs are allocated on every call; nothing is
  cached across calls except the compiled extensions and the per-shape
  autotune choice in ``state``.
* CUDA-graph capturable: the timed paths perform no host synchronisation
  (``_selftest`` / ``_autotune`` synchronise only during ``prepare()``).
* Config tables live next to the kernel they configure. Constants that more
  than one helper needs live in the "Shared constants" section below.
"""

    import importlib.util
    import os
    import sys
    import threading
    from typing import TypedDict

    import torch
    import torch.nn.functional as F

    from torch.utils.cpp_extension import load_inline

    # Debug only, from the environment: one random escape warp per CTA stalls this
    # many ns at a random stage; ring waits trap after 2 s.
    _SK2_DELAY = int(os.environ.get("SK2_DELAY_ESC", "0"))


    # ---------------------------------------------------------------------------
    # Shared constants (tuning numbers used by more than one helper)
    # ---------------------------------------------------------------------------
    # Lossless 12-bit packing: an exponent window of EXPONENT_WINDOW consecutive
    # exponent values, picked per matrix to cover the most weights.
    EXPONENT_WINDOW = 8
    EXPONENT_VALUES = 128  # distinct values of BF16 exponent bits 7..1
    # Packing is rejected when more than this share of weights escapes the window.
    MAX_ESCAPE_DENOMINATOR = 200
    # base4 packs one window base byte into each byte of a uint32 (see decode8).
    BASE4_REPLICATE = 0x01010101
    # Stage-count clamps for the TMA skinny kernel's pipeline.
    TMA_MIN_STAGES = 2
    TMA_MAX_STAGES = 8
    # sm_90 / sm_80 shared-memory budgets (bytes) for that pipeline.
    SM90_SMEM_BUDGET_BYTES = 112 * 1024
    SM80_SMEM_BUDGET_BYTES = 40 * 1024
    # Stage-count clamps for the sk2 kernel's ring.
    SK2_MIN_STAGES = 2
    SK2_MAX_STAGES = 16
    # Failure seeds keep selftest / autotune reproducible across runs.
    SELFTEST_SEED = 1234
    AUTOTUNE_SEED = 4321
    AUTOTUNE_REPS = 24
    # Autotune replays from a cold L2: this read sweep is larger than L2.
    L2_FLUSH_ROWS = 65_536  # 65_536 x 1024 float32 = 256 MiB
    # sk2 ring sizing: a config name ending in _rNNN caps the ring at NNN KiB,
    # otherwise the default applies; the ring also never exceeds the smem budget
    # left after the rest of the kernel's shared-memory layout.
    SK2_DEFAULT_RING_CAP_KB = 200
    SK2_RING_SMEM_BUDGET_BYTES = 232_448



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
        x: torch.Tensor  # bfloat16 [m, k]
        config: Config


    # ---------------------------------------------------------------------------
    # CUDA sources. These string constants are the kernels, verbatim: tile sizes,
    # launch configs, and numerics in them are the tuned behaviour. Each kernel's
    # config table and dispatch builder live next to its source below; @@...@@
    # placeholders are filled by the builder for that route.
    # ---------------------------------------------------------------------------
    CUDA_SRC = r"""
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <stdint.h>


__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}

__device__ __forceinline__ uint32_t lop3_sel(uint32_t a, uint32_t b, uint32_t c) {
  // c ? b : a, bitwise
  uint32_t d;
  asm("lop3.b32 %0, %1, %2, %3, 0xD8;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

// Decode 8 weights: low bytes of weights 0..3 in lw0, 4..7 in lw1, codes in nw
// (low nibble of byte b -> weight b, high nibble -> weight 4+b; code = s<<3 | o).
// Output: 4 bf16x2 registers r[0..3] = (w0,w1) (w2,w3) (w4,w5) (w6,w7).
__device__ __forceinline__ void decode8(uint32_t lw0, uint32_t lw1, uint32_t nw, uint32_t base4, uint32_t* r) {
  uint32_t ha = lop3_sel(nw & 0x07070707u, nw << 4, 0x80808080u) + base4;
  uint32_t hb = lop3_sel((nw >> 4) & 0x07070707u, nw, 0x80808080u) + base4;
  r[0] = prmt(lw0, ha, 0x5140);
  r[1] = prmt(lw0, ha, 0x7362);
  r[2] = prmt(lw1, hb, 0x5140);
  r[3] = prmt(lw1, hb, 0x7362);
}

template <typename T> __device__ __forceinline__ T cvt_out(float v);
template <> __device__ __forceinline__ float cvt_out<float>(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 cvt_out<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

// Weights, uncompressed (COMP=0): [num_rg][S][32 lanes][16 B] A fragments, S = Kp / 16.
// Weights, compressed (COMP=1): [num_rg][S/4 units][3 parts][32 lanes][16 B]: parts 0,1 = low
//   bytes of the lane's 32 weights of 4 k-steps, part 2 = their 32 codes.
// x: [M][Kx] bf16.  Weight column k uses x column (k % Kx) (router hi|lo concat).
// CTA (rb, sp): row groups rb*RG .. +RG, k range [sp*Kc, (sp+1)*Kc).
// Warp w: row group rb*RG + w / WS, k sub-range (w % WS).
// D: prefetch depth in loads of 16 B per lane (k-steps if !COMP, unit parts if COMP).
@@MBAR@@
@@BODY@@
"""

    MBAR_HELPERS = r"""
// ------------------------------------------------------------------ mbarrier / bulk copy helpers
__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ bool mbar_test(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  while (!mbar_test(bar, parity)) {}
}
// escape correction from the stored original BF16 weight t: its 12-bit decode keeps the sign and low byte and clamps the
// exponent into [base, base + 7]; the exact difference (asserted in prepare) is formed here in FP32
__device__ __forceinline__ float esc_delta(uint32_t t, uint32_t bq) {
  const uint32_t hi = (t >> 8) & 0xFFu, e71 = hi & 0x7Fu;
  const uint32_t dhi = (hi & 0x80u) | (e71 < bq ? bq : bq + 7u);
  return __uint_as_float(t << 16) - __uint_as_float(((dhi << 8) | (t & 0xFFu)) << 16);
}
// wait with a suspend-time hint: the warp sleeps in hardware until the phase completes instead of re-polling
__device__ __forceinline__ void mbar_wait_sleep(uint32_t bar, uint32_t parity) {
  uint32_t ok = 0;
  do {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(bar), "r"(parity), "r"(0x989680u) : "memory");
  } while (!ok);
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared.b64 st, [%0];\n}\n" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void named_bar(int id, int n) {
  asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared.b64 st, [%0], %1;\n}\n" :: "r"(bar), "r"(bytes) : "memory");
}
// L2 policies (v132: used by the 2-D TMA boxes of the m >= 33 routes only - gq/gqx/gemm90/wide): weights evict_first,
// x evict_last
__device__ __forceinline__ uint64_t l2_pol_ef() {
  uint64_t p; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p)); return p;
}
__device__ __forceinline__ uint64_t l2_pol_el() {
  uint64_t p; asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p)); return p;
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
// wait on a local mbarrier whose phase another CTA of the cluster completes (acquire at cluster scope)
__device__ __forceinline__ void mbar_wait_cluster(uint32_t bar, uint32_t parity) {
  uint32_t ok = 0;
  do {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n"
                 " selp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  } while (!ok);
}

"""

    WG_HELPERS = r"""
// ---------------------------------------------------------------- wgmma helpers (sm_90a)
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }


// smem matrix descriptor: K-major, 128B swizzle, SBO = 1024 B (8 rows x 128 B)
__device__ __forceinline__ uint64_t make_desc_sw128(uint32_t saddr) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr & 0x3FFFF) >> 4);
  d |= (uint64_t)1 << 16;                 // LBO (ignored for swizzled K-major)
  d |= (uint64_t)(1024 >> 4) << 32;       // SBO
  d |= (uint64_t)1 << 62;                 // 128B swizzle
  return d;
}


"""


    def wgmma_rs_defs(bns):
        """Generate wgmma_bn (register-sourced A) overloads for m64nBNk16 bf16."""
        out = []
        for bn in bns:
            nacc = bn // 2
            regs = ", ".join(f"%{i}" for i in range(nacc))
            a0 = nacc
            cons = ", ".join(f'"+f"(d[{i}])' for i in range(nacc))
            pi = a0 + 5
            out.append(f"""
    template <int BN_> __device__ __forceinline__ typename std::enable_if<BN_ == {bn}>::type
    wgmma_bn(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {{
      asm volatile("{{\\n"
          ".reg .pred p;\\n"
          "setp.ne.b32 p, %{pi}, 0;\\n"
          "wgmma.mma_async.sync.aligned.m64n{bn}k16.f32.bf16.bf16 "
          "{{{regs}}}, {{%{a0}, %{a0+1}, %{a0+2}, %{a0+3}}}, %{a0+4}, p, 1, 1, 0;\\n"
          "}}\\n"
          : {cons}
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
    }}""")
        return "\n".join(out)


    TMA_SRC = r"""
// Weights: per-warp contiguous layout, uncompressed [nrg][S][32][16B], compressed [nrg][S/4][3][32][16B].
// One stage = KS k-steps (uncompressed) or KS/4 units (compressed) for every consumer warp.
// Warps 0..NW-1 consume, warp NW produces (one elected lane issues bulk copies on sm_90;
template <int MT, int RG, int WS, int KS, int NSTAGE, bool COMP, bool HL, bool WGM, typename OutT>
__global__ void __launch_bounds__((RG * WS + 1) * 32, 1)
skinny_tma_kernel(const uint4* __restrict__ Wp, const __nv_bfloat16* __restrict__ x,
                  OutT* __restrict__ out, float* __restrict__ ws, int* __restrict__ counters,
                  const int2* __restrict__ ptab, int pmax, uint32_t base4,
                  int M, int N, int Kp, int Kx, int splits, int nrg, int dist, int dg) {
  constexpr int NW = RG * WS;
  constexpr int NC = NW * 32;                       // consumer threads
  constexpr int TOK = MT * 8;
  constexpr int TOKP = TOK + 1;  // padded red row (bank-conflict-free epilogue reads)
  // bytes per warp per stage (HL: router hi and lo tiles of the same k-step back to back)
  constexpr int CHUNK = COMP ? (KS / 4) * 1536 : KS * 512 * (HL ? 2 : 1);
  constexpr int SLOT = NW * CHUNK;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  // layout: [bars: full NSTAGE, empty NSTAGE, xbar XP (pad 256)] [ring NSTAGE*SLOT] [x rows M*XS*2 + zero row] [red]
  constexpr int XP = 4;
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t full0 = sb, empty0 = sb + NSTAGE * 8, xbar0 = sb + 2 * NSTAGE * 8;
  const uint32_t ring = sb + 256;
  const int Kc = Kp / splits;
  const int XS = Kc + 8;
  // WGM: x as 128B-swizzled K-major tiles [Kc/64][TOK rows][128 B] (wgmma B operand), 1024-aligned
  const size_t x_off = WGM ? (size_t)((((sb + 256u + (uint32_t)(NSTAGE * SLOT)) + 1023u) & ~1023u) - sb)
                           : (size_t)256 + NSTAGE * SLOT;   // WGM: absolute shared address 1024-aligned
  const size_t x_bytes = WGM ? (size_t)Kc * TOK * 2 : (((size_t)M * XS * 2 + 16 + 15) & ~(size_t)15);
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem_raw + x_off);
  float* red = reinterpret_cast<float*>(smem_raw + x_off + x_bytes);
  int2* ptab_s = reinterpret_cast<int2*>(red + NW * 16 * TOKP + ((NW * 16 * TOKP) & 1));

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int rb = blockIdx.x;
  const int sp = blockIdx.y;
  const int k0 = sp * Kc;
  const int S = Kp >> 4;
  const int NS = Kc / (16 * WS);       // k-steps per consumer warp
  const int nst = NS / KS;              // stages
  // WS == 1: stream is [rb][stage over all K][warp][chunk] (shared with the wide kernel); else [cta][stage][warp][chunk]
  const char* wsrc = reinterpret_cast<const char*>(Wp) +
      (WS == 1 ? ((size_t)blockIdx.x * (S / KS) + (size_t)blockIdx.y * nst) : (size_t)(blockIdx.y * gridDim.x + blockIdx.x) * nst) * SLOT;

  if (threadIdx.x == 0) {
    for (int i = 0; i < NSTAGE; ++i) {
      mbar_init(full0 + 8 * i, 1);
      mbar_init(empty0 + 8 * i, NW);
    }
    for (int i = 0; i < XP; ++i) mbar_init(xbar0 + 8 * i, NC);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    // first stages go out right away (no wait for the CTA barrier)
    for (int st = 0; st < NSTAGE && st < nst; ++st) {
      mbar_expect_tx(full0 + 8 * st, SLOT);
      bulk_g2s(ring + st * SLOT, wsrc + (size_t)st * SLOT, SLOT, full0 + 8 * st);
    }
  }
  if (!WGM && threadIdx.x < 4) reinterpret_cast<uint32_t*>(xs + M * XS)[threadIdx.x] = 0u;  // zero row
  __syncthreads();

  if (warp == NW) {
    // ---------------- producer: this CTA's weight stream is contiguous
    const int st0 = NSTAGE < nst ? NSTAGE : nst;   // already issued by thread 0
    for (int st = st0; st < nst; ++st) {
      const int slot = st % NSTAGE;
      if (st >= NSTAGE) mbar_wait(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
      const uint32_t dst = ring + slot * SLOT;
      const char* src = wsrc + (size_t)st * SLOT;
      if (lane == 0) {
        mbar_expect_tx(full0 + 8 * slot, SLOT);
        bulk_g2s(dst, src, SLOT, full0 + 8 * slot);
      }
    }
    return;
  }

  // ---------------- consumers
  const int rg = rb * RG + warp % RG;   // warps of a warpgroup: 4 consecutive row groups, same k range
  const int wsub = warp / RG;
  if (threadIdx.x == 0 && splits > 1) asm volatile("prefetch.global.L2 [%0];" :: "l"(counters + rb));
  if (COMP && dg < 2) {
    // this CTA's exact-correction entries -> smem (tracked by x part 0's mbarrier below)
    const int cta = blockIdx.y * gridDim.x + blockIdx.x;
    const char* src = reinterpret_cast<const char*>(ptab + (size_t)cta * pmax);
    const uint32_t dst = (uint32_t)__cvta_generic_to_shared(ptab_s);
    for (int i = threadIdx.x; i < pmax / 2; i += NC)
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst + i * 16), "l"(src + i * 16) : "memory");
  }
  // x chunk in XP parts, each completion tracked by its own mbarrier (no CTA-wide wait)
  if (dg < 2) {
    const int kx0 = k0 % Kx;
    const int pch = Kc / 8 / XP;  // 16-byte chunks per token row per part
    const uint32_t xs_s = (uint32_t)__cvta_generic_to_shared(xs);
    for (int p = 0; p < XP; ++p) {
      if constexpr (WGM) {
        for (int idx = threadIdx.x; idx < TOK * pch; idx += NC) {
          int t = idx / pch, c = p * pch + (idx - t * pch);   // c: 16-byte chunk index along k
          const uint32_t dst = xs_s + (uint32_t)((((c >> 3) * TOK + t) << 7) + ((((c & 7) ^ (t & 7))) << 4));
          const __nv_bfloat16* src = x + (size_t)(t < M ? t : 0) * Kx + kx0 + c * 8;
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(t < M ? 16 : 0) : "memory");
        }
      } else {
        for (int idx = threadIdx.x; idx < M * pch; idx += NC) {
          int t = idx / pch, c = p * pch + (idx - t * pch);
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
                       "r"(xs_s + (uint32_t)((t * XS + c * 8) * 2)), "l"(x + (size_t)t * Kx + kx0 + c * 8) : "memory");
        }
      }
      asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"(xbar0 + 8 * p) : "memory");
    }
  }
  int xp_ready = -1;
  const int kpart = Kc / XP;  // k per part

  float acc[MT][4];
#pragma unroll
  for (int j = 0; j < MT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  float wacc[WGM ? TOK / 2 : 1];
#pragma unroll
  for (int q = 0; q < (WGM ? TOK / 2 : 1); ++q) wacc[q] = 0.f;
  uint32_t afrag[2][4];

  const uint32_t xs_base = (uint32_t)__cvta_generic_to_shared(xs);
  const int lrow = lane & 7, lmat = lane >> 3;
  const int kw0 = wsub * NS * 16;
  const uint32_t zero_row = xs_base + (uint32_t)(M * XS * 2);
  uint32_t xs_row[MT];
  uint32_t xs_adv[MT];
#pragma unroll
  for (int j = 0; j < MT; ++j) {
    const int t = j * 8 + lrow;
    xs_row[j] = t < M ? xs_base + (uint32_t)((t * XS + kw0 + lmat * 8) * 2) : zero_row;
    xs_adv[j] = t < M ? 32u : 0u;   // bytes per k-step pair... per k-step: 32 B
  }

  for (int st = 0; st < nst; ++st) {
    const int slot = st % NSTAGE;
    {
      const int need = (kw0 + (st + 1) * KS * 16 - 1) / kpart;  // x part holding this stage's last k
      if (need > xp_ready && dg < 2) {
        mbar_wait(xbar0 + 8 * need, 0);
        xp_ready = need;
        if constexpr (WGM) fence_async_smem();   // cp.async (generic proxy) writes -> wgmma (async proxy) reads
      }
    }
    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
    // pull this warp's stage data into registers, then hand the slot back to the producer at once
    constexpr int NL = CHUNK / 512;  // 16-byte loads per lane per stage
    uint4 wr[NL];
    {
      const uint32_t src = ring + slot * SLOT + warp * CHUNK + lane * 16;
#pragma unroll
      for (int q = 0; q < NL; ++q)
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(wr[q].x), "=r"(wr[q].y), "=r"(wr[q].z), "=r"(wr[q].w) : "r"(src + q * 512));
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    if constexpr (!COMP && HL) {
#pragma unroll
      for (int j2 = 0; j2 < KS; j2 += 2) {
        const uint4 h0 = wr[2 * j2], l0 = wr[2 * j2 + 1], h1 = wr[2 * j2 + 2], l1 = wr[2 * j2 + 3];
        const int s = st * KS + j2;
        uint32_t b[MT][4];
#pragma unroll
        for (int j = 0; j < MT; ++j)
          ldsm_x4(xs_row[j] + xs_adv[j] * (uint32_t)s, b[j][0], b[j][1], b[j][2], b[j][3]);
#pragma unroll
        for (int j = 0; j < MT; ++j) {
          mma_bf16(acc[j], h0.x, h0.y, h0.z, h0.w, b[j][0], b[j][1]);
          mma_bf16(acc[j], l0.x, l0.y, l0.z, l0.w, b[j][0], b[j][1]);
          mma_bf16(acc[j], h1.x, h1.y, h1.z, h1.w, b[j][2], b[j][3]);
          mma_bf16(acc[j], l1.x, l1.y, l1.z, l1.w, b[j][2], b[j][3]);
        }
      }
    } else if constexpr (!COMP) {
#pragma unroll
      for (int j2 = 0; j2 < KS; j2 += 2) {
        const uint4 a0 = wr[j2], a1 = wr[j2 + 1];
        const int s = st * KS + j2;
        uint32_t b[MT][4];
#pragma unroll
        for (int j = 0; j < MT; ++j)
          ldsm_x4(xs_row[j] + xs_adv[j] * (uint32_t)s, b[j][0], b[j][1], b[j][2], b[j][3]);
#pragma unroll
        for (int j = 0; j < MT; ++j) {
          mma_bf16(acc[j], a0.x, a0.y, a0.z, a0.w, b[j][0], b[j][1]);
          mma_bf16(acc[j], a1.x, a1.y, a1.z, a1.w, b[j][2], b[j][3]);
        }
      }
    } else if constexpr (WGM) {
#pragma unroll
      for (int u = 0; u < KS / 4; ++u) {
        const uint4 L0 = wr[3 * u], L1 = wr[3 * u + 1], C = wr[3 * u + 2];
        const int kl = kw0 + (st * KS + u * 4) * 16;   // local k of this unit's first k-step (multiple of 64)
        const uint32_t xt = xs_base + (uint32_t)((kl >> 6) * TOK * 128);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          uint32_t* af = afrag[j & 1];
          decode8(j < 2 ? (j == 0 ? L0.x : L0.z) : (j == 2 ? L1.x : L1.z),
                  j < 2 ? (j == 0 ? L0.y : L0.w) : (j == 2 ? L1.y : L1.w),
                  j == 0 ? C.x : j == 1 ? C.y : j == 2 ? C.z : C.w, base4, af);
#pragma unroll
          for (int q = 0; q < TOK / 2; ++q) asm volatile("" : "+f"(wacc[q]) :: "memory");
          wg_fence();
          wgmma_bn<TOK>(wacc, af[0], af[1], af[2], af[3], make_desc_sw128(xt + j * 32));
          wg_commit();
          wg_wait<1>();   // the wgmma that used afrag[(j+1)&1] is done -> it can be overwritten next step
#pragma unroll
          for (int q = 0; q < 4; ++q) asm volatile("" : "+r"(afrag[0][q]), "+r"(afrag[1][q]) :: "memory");
        }
      }
    } else {
#pragma unroll
      for (int u = 0; u < KS / 4; ++u) {
        const uint4 L0 = wr[3 * u], L1 = wr[3 * u + 1], C = wr[3 * u + 2];
        uint32_t a[16];
        decode8(L0.x, L0.y, C.x, base4, a + 0);
        decode8(L0.z, L0.w, C.y, base4, a + 4);
        decode8(L1.x, L1.y, C.z, base4, a + 8);
        decode8(L1.z, L1.w, C.w, base4, a + 12);
        const int s = st * KS + u * 4;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          uint32_t b[MT][4];
#pragma unroll
          for (int j = 0; j < MT; ++j)
            ldsm_x4(xs_row[j] + xs_adv[j] * (uint32_t)(s + 2 * h), b[j][0], b[j][1], b[j][2], b[j][3]);
#pragma unroll
          for (int j = 0; j < MT; ++j) {
            const uint32_t* a0 = a + 8 * h;
            mma_bf16(acc[j], a0[0], a0[1], a0[2], a0[3], b[j][0], b[j][1]);
            mma_bf16(acc[j], a0[4], a0[5], a0[6], a0[7], b[j][2], b[j][3]);
          }
        }
      }
    }
  }
  if constexpr (WGM) {
    wg_wait<0>();
#pragma unroll
    for (int q = 0; q < TOK / 2; ++q) asm volatile("" : "+f"(wacc[q]) :: "memory");
  }

  if constexpr (WGM) {
#pragma unroll
    for (int j = 0; j < MT; ++j) { acc[j][0] = wacc[4 * j]; acc[j][1] = wacc[4 * j + 1]; acc[j][2] = wacc[4 * j + 2]; acc[j][3] = wacc[4 * j + 3]; }
  }
  // per-warp partials -> shared memory: red[w][row16][TOK]
  {
    const int g = lane >> 2, t = lane & 3;
    float* r = red + warp * 16 * TOKP;
#pragma unroll
    for (int j = 0; j < MT; ++j) {
      r[g * TOKP + j * 8 + 2 * t] = acc[j][0];
      r[g * TOKP + j * 8 + 2 * t + 1] = acc[j][1];
      r[(g + 8) * TOKP + j * 8 + 2 * t] = acc[j][2];
      r[(g + 8) * TOKP + j * 8 + 2 * t + 1] = acc[j][3];
    }
  }
  if constexpr (COMP) {
    // all x parts and the patch table are in smem; apply exact corrections in parallel (smem atomics)
    mbar_wait(xbar0 + 8 * (XP - 1), 0);
    named_bar(1, NC);
    const int cnt = ptab_s[0].x;
    for (int p = 1 + warp; p <= cnt; p += NW) {
      const int2 e = ptab_s[p];
      const int rl = e.x >> 16, kl = e.x & 0xFFFF;
      const float d = __int_as_float(e.y);
      float* r = red + ((rl >> 4) * 16 + (rl & 15)) * TOKP;   // warp (wsub 0, row group rl/16)
      for (int t = lane; t < M; t += 32) {
        float xv;
        if constexpr (WGM) {   // swizzled K-major tiles [Kc/64][TOK][128 B]
          const int c = kl >> 3;
          xv = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
              reinterpret_cast<const unsigned char*>(xs) + (((c >> 3) * TOK + t) << 7) + (((c & 7) ^ (t & 7)) << 4) + (kl & 7) * 2));
        } else {
          xv = __bfloat162float(xs[t * XS + kl]);
        }
        atomicAdd(r + t, xv * d);
      }
    }
  }
  named_bar(1, NC);
  const int rows = RG * 16;
  const int row0 = rb * rows;
  if (splits == 1) {
    for (int idx = threadIdx.x; idx < rows * M; idx += NC) {
      int t = idx / rows, rr = idx - t * rows;
      int grp = rr >> 4, r16 = rr & 15;
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < WS; ++w) v += red[((w * RG + grp) * 16 + r16) * TOKP + t];
      out[(size_t)t * N + row0 + rr] = cvt_out<OutT>(v);
    }
    return;
  }
  for (int idx = threadIdx.x; idx < rows * M; idx += NC) {
    int t = idx / rows, rr = idx - t * rows;
    int grp = rr >> 4, r16 = rr & 15;
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < WS; ++w) v += red[((w * RG + grp) * 16 + r16) * TOKP + t];
    __stcg(ws + ((size_t)sp * M + t) * N + row0 + rr, v);
  }
  __threadfence();
  named_bar(1, NC);
  // dist: all `splits` CTAs of this row block are co-resident (grid fits the GPU at once): monotonic-counter
  // barrier, then each reduces its own slice of the row block.  Otherwise the last-arriving CTA reduces all.
  // Counters only ever increase (both modes), so the modes can share them.
  __shared__ int s_last;
  if (threadIdx.x == 0) {
    const int old = atomicAdd(counters + rb, 1);
    s_last = (old % splits) == splits - 1;
    if (dist) {
      const int target = (old / splits + 1) * splits;
      while (*((volatile int*)(counters + rb)) - target < 0) __nanosleep(20);
    }
  }
  named_bar(1, NC);
  if (dist || s_last) {
  __threadfence();
  {
    const int r4n = rows / 4;
    const int nch = M * r4n;
    const int c_lo = dist ? (int)(((long long)sp * nch) / splits) : 0;
    const int c_hi = dist ? (int)(((long long)(sp + 1) * nch) / splits) : nch;
    for (int c0 = c_lo + threadIdx.x; c0 < c_hi; c0 += 2 * NC) {
      float4 acc4[2];
      int tt[2], rr[2];
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int c = c0 + u * NC;
        tt[u] = c / r4n; rr[u] = (c - tt[u] * r4n) * 4;
        acc4[u] = make_float4(0.f, 0.f, 0.f, 0.f);
      }
      for (int q = 0; q < splits; q += 4) {
        float4 v[2][4];
#pragma unroll
        for (int u = 0; u < 2; ++u)
#pragma unroll
          for (int qq = 0; qq < 4; ++qq) {
            v[u][qq] = make_float4(0.f, 0.f, 0.f, 0.f);
            if (c0 + u * NC < c_hi && q + qq < splits)
              v[u][qq] = __ldcg(reinterpret_cast<const float4*>(ws + ((size_t)(q + qq) * M + tt[u]) * N + row0 + rr[u]));
          }
#pragma unroll
        for (int u = 0; u < 2; ++u)
#pragma unroll
          for (int qq = 0; qq < 4; ++qq) {
            acc4[u].x += v[u][qq].x; acc4[u].y += v[u][qq].y; acc4[u].z += v[u][qq].z; acc4[u].w += v[u][qq].w;
          }
      }
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        if (c0 + u * NC < c_hi) {
          OutT* o = out + (size_t)tt[u] * N + row0 + rr[u];
          o[0] = cvt_out<OutT>(acc4[u].x); o[1] = cvt_out<OutT>(acc4[u].y);
          o[2] = cvt_out<OutT>(acc4[u].z); o[3] = cvt_out<OutT>(acc4[u].w);
        }
      }
    }
  }
  }  // dist || s_last
}

template <int MT, int RG, int WS, int KS, int NSTAGE, bool COMP, bool HL, bool WGM, typename OutT>
static void launch_tma(const at::Tensor& Wp, const at::Tensor& x, at::Tensor& out,
                       at::Tensor& ws, at::Tensor& counters, const at::Tensor& ptab, int64_t pmax,
                       int64_t base4, int M, int N, int Kp, int Kx, int splits, int nrg, int dg) {
  constexpr int NW = RG * WS;
  constexpr int CHUNK = COMP ? (KS / 4) * 1536 : KS * 512 * (HL ? 2 : 1);
  int Kc = Kp / splits;
  size_t x_bytes = WGM ? (size_t)Kc * MT * 8 * 2 + 1024 : (((size_t)M * (Kc + 8) * 2 + 16 + 15) & ~(size_t)15);
  size_t smem = 256 + (size_t)NSTAGE * NW * CHUNK + x_bytes + (size_t)NW * 16 * (MT * 8 + 1) * 4 + 8 + (size_t)pmax * 8;
  auto kern = skinny_tma_kernel<MT, RG, WS, KS, NSTAGE, COMP, HL, WGM, OutT>;
  static bool attr_done = false;
  if (!attr_done) {
    int dev = 0, optin = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)fa.sharedSizeBytes);
    attr_done = true;
  }
  dim3 grid(N / (RG * 16), splits);
  int dev = 0, nsm = 0, per_sm = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, (NW + 1) * 32, smem);
  const int dist = (int)(grid.x * grid.y) <= per_sm * nsm ? 1 : 0;
  kern<<<grid, (NW + 1) * 32, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint4*>(Wp.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<OutT*>(out.data_ptr()), ws.data_ptr<float>(), counters.data_ptr<int>(),
      reinterpret_cast<const int2*>(ptab.data_ptr()), (int)pmax, (uint32_t)base4,
      M, N, Kp, Kx, splits, nrg, dist, dg);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void @@TMA_NAME@@(at::Tensor Wp, at::Tensor x, at::Tensor out, at::Tensor ws, at::Tensor counters,
                at::Tensor ptab, int64_t pmax, int64_t base4,
                int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t mt, int64_t rg, int64_t wsn,
                int64_t comp, int64_t nrg, int64_t ks, int64_t dg) {
  bool out_f32 = out.scalar_type() == at::kFloat;
  @@TMA_DISPATCH@@
  TORCH_CHECK(false, "skinny_tma: no config");
}
"""

    MAP_SRC = r"""
#include <cudaTypedefs.h>
#include <cstring>
__device__ __forceinline__ void tma_load_2d_pol(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1, uint64_t pol) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4}], [%2], %5;"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "l"(pol) : "memory");
}
__device__ __forceinline__ void tma_load_2d_ef(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1) {
  tma_load_2d_pol(dst, map, bar, c0, c1, l2_pol_ef());
}
__device__ __forceinline__ void tma_load_2d_el(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1) {
  tma_load_2d_pol(dst, map, bar, c0, c1, l2_pol_el());
}
__device__ __forceinline__ void tma_load_3d(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
static PFN_cuTensorMapEncodeTiled_v12000 get_encode() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult q;
    void* p = nullptr;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q);
    TORCH_CHECK(p != nullptr, "cuTensorMapEncodeTiled not found");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }
  return fn;
}

// x [M, K] bf16 as 3-D {64 (k in unit), M (token), K/64 (unit)}: one TMA brings `units` tiles of [rows x 128 B]
static CUtensorMap make_map_x3d(const void* ptr, uint64_t m, uint64_t k, uint32_t box_rows, uint32_t units) {
  CUtensorMap mp;
  cuuint64_t dims[3] = {64, m, k / 64};
  cuuint64_t strides[2] = {k * 2, 128};
  cuuint32_t box[3] = {64, box_rows, units};
  cuuint32_t estr[3] = {1, 1, 1};
  CUresult r = get_encode()(&mp, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(ptr), dims, strides, box, estr,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled (3d) failed: ", (int)r);
  return mp;
}

static CUtensorMap make_map_2d(const void* ptr, uint64_t rows, uint64_t cols, uint32_t box_rows) {
  CUtensorMap m;
  cuuint64_t dims[2] = {cols, rows};
  cuuint64_t strides[1] = {cols * 2};
  cuuint32_t box[2] = {64, box_rows};
  cuuint32_t estr[2] = {1, 1};
  CUresult r = get_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides, box, estr,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}

"""

    GEMM_SRC = r"""

@@WGMMA_SS_DEFS@@

template <int NWG, int BN, int STAGES, bool AS, typename OutT>
__global__ void __launch_bounds__(NWG * 128 + 128, 1)
gemm90_kernel(const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB,
              const uint8_t* __restrict__ Wst,
              OutT* __restrict__ out, float* __restrict__ ws, int* __restrict__ counters,
              int M, int N, int Kp, int Kx, int splits) {
  constexpr int BM = NWG * 64;
  constexpr int A_BYTES = BM * 128;       // BM rows x 64 bf16
  constexpr int B_BYTES = BN * 128;       // BN rows x 64 bf16
  constexpr int ST_BYTES = A_BYTES + B_BYTES;
  constexpr int NACC = BN / 2;
  constexpr int NT = NWG * 128;           // consumer threads
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  const uint32_t bars = sbase + STAGES * ST_BYTES;          // full[STAGES], empty[STAGES]
  const uint32_t full0 = bars, empty0 = bars + 8 * STAGES;

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int rb = blockIdx.x, sp = blockIdx.y;
  const int nkb_all = Kp >> 6;
  const int kb0 = (int)(((long long)sp * nkb_all) / splits);
  const int kb1 = (int)(((long long)(sp + 1) * nkb_all) / splits);
  const int nkb = kb1 - kb0;

  if (tid == 0) {
    for (int i = 0; i < STAGES; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NWG); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    if (splits > 1) asm volatile("prefetch.global.L2 [%0];" :: "l"(counters + rb));
  }
  __syncthreads();

  if (warp >= NWG * 4) {
    // ---------------- producer warp (one lane)
    if (warp == NWG * 4 && lane == 0) {
      asm volatile("prefetch.tensormap [%0];" :: "l"(&mapA) : "memory");
      asm volatile("prefetch.tensormap [%0];" :: "l"(&mapB) : "memory");
      for (int i = 0; i < nkb; ++i) {
        const int slot = i % STAGES;
        if (i >= STAGES) mbar_wait(empty0 + 8 * slot, ((i / STAGES) - 1) & 1);
        mbar_expect_tx(full0 + 8 * slot, ST_BYTES);
        const int kg = (kb0 + i) * 64;
          tma_load_2d_ef(sbase + slot * ST_BYTES, &mapA, full0 + 8 * slot, kg, rb * BM);
        tma_load_2d_el(sbase + slot * ST_BYTES + A_BYTES, &mapB, full0 + 8 * slot, kg % Kx, 0);
      }
    }
    return;
  }

  // ---------------- consumers
  const int wg = warp >> 2;
  float acc[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) acc[i] = 0.f;

  for (int i = 0; i < nkb; ++i) {
    const int slot = i % STAGES;
    mbar_wait(full0 + 8 * slot, (i / STAGES) & 1);
    const uint32_t a_st = sbase + slot * ST_BYTES + wg * (64 * 128);
    const uint32_t b_st = sbase + slot * ST_BYTES + A_BYTES;
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    wg_fence();
#pragma unroll
    for (int j = 0; j < 4; ++j)
      wgmma_ss<BN>(acc, make_desc_sw128(a_st + j * 32), make_desc_sw128(b_st + j * 32));
    wg_commit();
    // release the previous stage once its wgmmas are done (keep one group in flight)
    wg_wait<1>();
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    if (i > 0 && (tid & 127) == 0) mbar_arrive(empty0 + 8 * ((i - 1) % STAGES));
  }
  wg_wait<0>();
#pragma unroll
  for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");

  const int g = lane >> 2, t4 = lane & 3;
  const int row_w = rb * BM + (warp & 3) * 16 + wg * 64;
  if (splits == 1) {
#pragma unroll
    for (int j = 0; j < BN / 8; ++j) {
      const int t0 = 8 * j + 2 * t4;
      if (t0 < M) {
        out[(size_t)t0 * N + row_w + g] = cvt_out<OutT>(acc[4 * j]);
        out[(size_t)t0 * N + row_w + g + 8] = cvt_out<OutT>(acc[4 * j + 2]);
      }
      if (t0 + 1 < M) {
        out[(size_t)(t0 + 1) * N + row_w + g] = cvt_out<OutT>(acc[4 * j + 1]);
        out[(size_t)(t0 + 1) * N + row_w + g + 8] = cvt_out<OutT>(acc[4 * j + 3]);
      }
    }
    return;
  }
  {
    float4* wsp = reinterpret_cast<float4*>(ws) + (size_t)(rb * splits + sp) * (NACC / 4) * NT;
#pragma unroll
    for (int j = 0; j < NACC / 4; ++j)
      __stcg(wsp + j * NT + tid, make_float4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3]));
  }
  __threadfence();
  named_bar(1, NT);
  if (tid == 0) {
    int old = atomicAdd(counters + rb, 1);
    int target = (old / splits + 1) * splits;
    while (true) {
      int v = *((volatile int*)(counters + rb));
      if (v - target >= 0) break;
      __nanosleep(32);
    }
  }
  named_bar(1, NT);
  __threadfence();
  const int j0 = (int)(((long long)sp * (NACC / 4)) / splits);
  const int j1 = (int)(((long long)(sp + 1) * (NACC / 4)) / splits);
  const float4* base = reinterpret_cast<const float4*>(ws) + (size_t)rb * splits * (NACC / 4) * NT;
  for (int j = j0; j < j1; ++j) {
    float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
    int q = 0;
    for (; q + 4 <= splits; q += 4) {
      float4 v0 = __ldcg(base + ((size_t)(q + 0) * (NACC / 4) + j) * NT + tid);
      float4 v1 = __ldcg(base + ((size_t)(q + 1) * (NACC / 4) + j) * NT + tid);
      float4 v2 = __ldcg(base + ((size_t)(q + 2) * (NACC / 4) + j) * NT + tid);
      float4 v3 = __ldcg(base + ((size_t)(q + 3) * (NACC / 4) + j) * NT + tid);
      s.x += v0.x + v1.x + v2.x + v3.x; s.y += v0.y + v1.y + v2.y + v3.y;
      s.z += v0.z + v1.z + v2.z + v3.z; s.w += v0.w + v1.w + v2.w + v3.w;
    }
    for (; q < splits; ++q) {
      float4 v = __ldcg(base + ((size_t)q * (NACC / 4) + j) * NT + tid);
      s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
    }
    const int t0 = 8 * j + 2 * t4;
    if (t0 < M) {
      out[(size_t)t0 * N + row_w + g] = cvt_out<OutT>(s.x);
      out[(size_t)t0 * N + row_w + g + 8] = cvt_out<OutT>(s.z);
    }
    if (t0 + 1 < M) {
      out[(size_t)(t0 + 1) * N + row_w + g] = cvt_out<OutT>(s.y);
      out[(size_t)(t0 + 1) * N + row_w + g + 8] = cvt_out<OutT>(s.w);
    }
  }
}

template <int NWG, int BN, int STAGES, bool AS, typename OutT>
static void launch_gemm90(const at::Tensor& W, const at::Tensor& Wst, const at::Tensor& x, at::Tensor& out, at::Tensor& ws,
                          at::Tensor& counters, int M, int N, int Kp, int Kx, int splits) {
  constexpr int BM = NWG * 64;
  size_t smem = 1024 + (size_t)STAGES * (BM * 128 + BN * 128) + 16 * STAGES + 64;
  auto kern = gemm90_kernel<NWG, BN, STAGES, AS, OutT>;
  static bool attr_done = false;
  if (!attr_done) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attr_done = true;
  }
  CUtensorMap mA = make_map_2d(W.data_ptr(), (uint64_t)W.size(0), (uint64_t)W.size(1), BM);
  CUtensorMap mB = make_map_2d(x.data_ptr(), (uint64_t)M, (uint64_t)Kx, BN);
  dim3 grid(N / BM, splits);
  kern<<<grid, NWG * 128 + 128, smem, at::cuda::getCurrentCUDAStream()>>>(
      mA, mB, reinterpret_cast<const uint8_t*>(Wst.data_ptr()),
      reinterpret_cast<OutT*>(out.data_ptr()), ws.data_ptr<float>(), counters.data_ptr<int>(),
      M, N, Kp, Kx, splits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gemm90(at::Tensor W, at::Tensor Wst, at::Tensor x, at::Tensor out, at::Tensor ws, at::Tensor counters,
            int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t nwg, int64_t bn, int64_t as) {
  bool out_f32 = out.scalar_type() == at::kFloat;
  @@GEMM_DISPATCH@@
  TORCH_CHECK(false, "gemm90: no config");
}
"""


    SK2_SRC = r"""

#define SK2_WAIT(bar, par) mbar_wait_sleep((bar), (par))
#define SK2_DIAG 0
// ============================================================================ sk2 (m <= 32): swapped operands
// A = x tile (16 tokens x 16 k; ldmatrix from the TMA-loaded, 128B-swizzled tile), B = weights (8 rows x 16 k),
// decoded to registers from the 12-bit stream (BF16 hi + lo for the router).  Each CTA owns R rows over the whole
// K: no cross-CTA reduction.  Warps split the CTA's rows (NWR x NT n8 tiles) and, inside every stage, K (WSK).
// A stage = UPS 64-k units: x tiles [UPS][TOKR rows x 128 B] (TMA, rows >= M zero-filled) + this CTA's weights.
template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL, typename OutT,
          bool PK>
__global__ void __launch_bounds__((NWR * WSK + 1 + (COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1)) * 32, 1)
sk2_kernel(const __grid_constant__ CUtensorMap xmap, const uint8_t* __restrict__ Ws,
           const __nv_bfloat16* __restrict__ x, OutT* __restrict__ out,
           const int2* __restrict__ ptab, int pmax, uint32_t base4, int M, int N, int K, int dg,
           unsigned long long* __restrict__ dbg) {
  const int dgv = SK2_DIAG ? dg : 0;   // diagnostics (dg) compiled out of production builds
  constexpr int NW = NWR * WSK;
  constexpr int NC = NW * 32;
  constexpr int R = NWR * NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int TU = COMP ? 768 : (HL ? 2048 : 1024);
  constexpr int WBY = NW * UPW * NT * TU;
  constexpr int XT = TOKR * 128;
  constexpr int XBY = UPS * XT;
  constexpr int SST = ((XBY + WBY) + 1023) & ~1023;
  // partial sums [warp][token][RP] (rows fastest, padded): 64-bit stores in, 128-bit loads out
  constexpr int TOKS = 16 * MT;
  constexpr int RP = R + ((40 - (R % 32)) % 32);   // >= R, = 8 (mod 32): conflict-free 64-bit stores
  constexpr int CP = R + 1;                         // corr[token][CP]: odd stride, conflict-free per-token updates
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t bars = sbase + NSTAGE * SST;
  const uint32_t full0 = bars, empty0 = bars + 8 * NSTAGE, pbar = bars + 16 * NSTAGE, cbar = pbar + 8, qbar = pbar + 16;
  // corrections, one copy per escape warp: [NESC][TOKS tokens][CP]; escape warps take turns on the stages
  constexpr int NESC = COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1;
  static_assert(NESC <= NSTAGE, "sk2: escape warps need NESC <= NSTAGE");
  float* corr = reinterpret_cast<float*>(gbase + NSTAGE * SST + 16 * NSTAGE + 32);
  constexpr int CSZ = (TOKS * CP + 3) & ~3;
  float* peer = corr + NESC * CSZ;                       // [CL-1][WSK][TOKS][RP]: ranks > 0 per-warp sums
  int2* ptab_s = reinterpret_cast<int2*>(peer + (CL > 1 ? (CL - 1) * WSK * TOKS * RP : 0));   // [pmax]: escapes
  const uint32_t zrow = (uint32_t)__cvta_generic_to_shared(ptab_s + pmax);   // 128 B of zeros (x rows >= M)
  // m <= 8: the x box has exactly M rows per 64-k unit (packed, M*128 B per unit; no zero rows moved)
  // PK (M < TOKR, or TOKR == 8): packed x, exactly M rows per 64-k unit (no out-of-bounds TMA rows; rows >= M
  // read zrow); else the full TOKR-row tile with compile-time addressing
  const int Mb = PK ? M : TOKR;
  const uint32_t XU = (uint32_t)Mb * 128u;
  const uint32_t XBYr = UPS * XU;
  float* red = reinterpret_cast<float*>(gbase);   // reuses the ring once every stage is consumed

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  // CL = 2: a cluster of two CTAs shares a row block; each streams one half of K, rank 1 hands its sums to rank 0
  const int cta = blockIdx.x / CL, crank = CL > 1 ? (int)(blockIdx.x % CL) : 0;
  const int tst = K / (UPS * 64);                      // stages over all of K
  const int gs0 = (crank * tst) / CL;                  // this CTA's first stage (uneven split for CL = 3)
  const int nst = ((crank + 1) * tst) / CL - gs0;      // and its stage count
  // stage-major weight stream: [stage][row block][...] (at any moment all CTAs read one contiguous region)
  const uint8_t* wsrc = Ws + (size_t)cta * WBY;
  const size_t wstride = (size_t)(gridDim.x / CL) * WBY;

  if (tid == 0) {
    // stage 0 first of all (the first bytes' latency is the ramp of the whole kernel): the weight copy needs only
    // its barrier; the tensor map prefetch goes out before it so the x box follows right behind. In a real cluster
    // (CL > 1) a cluster-scope release issued AFTER a copy is in flight waits for it (~0.7 us): there every barrier
    // is initialized and fenced before the first copy.
    asm volatile("prefetch.tensormap [%0];" :: "l"(&xmap) : "memory");
    mbar_init(full0, 1);
    if constexpr (CL > 1) {
      mbar_init(empty0, NW + (COMP ? NESC : 0));
      for (int i = 1; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NW + (COMP ? NESC : 0)); }
      mbar_init(pbar, 1);
      mbar_init(cbar, 32 * NESC);
      mbar_init(qbar, 1);
      if (crank == 0) mbar_expect_tx(qbar, (CL - 1) * NC * MT * NT * 16);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    mbar_expect_tx(full0, ((dgv & 4) ? 0 : XBYr) + WBY);
    bulk_g2s(sbase + XBY, wsrc + (size_t)gs0 * wstride, WBY, full0);
    if (!(dgv & 4)) tma_load_3d(sbase, &xmap, full0, 0, 0, gs0 * UPS);
    if (COMP) {   // this CTA's escape table
      if constexpr (CL == 1) {
        mbar_init(pbar, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      }
      mbar_expect_tx(pbar, pmax * 8);
      bulk_g2s((uint32_t)__cvta_generic_to_shared(ptab_s), ptab + (size_t)cta * pmax, pmax * 8, pbar);
    }
    if constexpr (CL == 1) {
      mbar_init(empty0, NW + (COMP ? NESC : 0));
      for (int i = 1; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NW + (COMP ? NESC : 0)); }
      if (!COMP) mbar_init(pbar, 1);
      mbar_init(cbar, 32 * NESC);
      mbar_init(qbar, 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
  }
  if (tid >= 32 && tid < 64) reinterpret_cast<uint32_t*>(ptab_s + pmax)[tid - 32] = 0u;
  // local barriers are all this CTA needs to start streaming; the cluster barrier (rank 0's qbar initialized before
  // ranks > 0 send into it) is only arrived at here and waited on right before the remote stores
  __syncthreads();
  if constexpr (CL > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");   // inits fenced above

  if (warp == NW) {
    // producer warp: lane 0 refills the ring
    if (lane == 0) {
      for (int st = 1; st < nst; ++st) {
        const int slot = st % NSTAGE;
        if (st >= NSTAGE) SK2_WAIT(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
        const uint32_t s0 = sbase + slot * SST;
        mbar_expect_tx(full0 + 8 * slot, ((dgv & 4) ? 0 : XBYr) + WBY);
        bulk_g2s(s0 + XBY, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * slot);
        if (!(dgv & 4)) tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, (gs0 + st) * UPS);
      }
    }
    __syncwarp();
    if constexpr (CL > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }
  if (warp >= NW + 1) {
    // NESC escape warps take turns on the stages (each has NESC stage-times per stage): exact corrections for the
    // escapes, read from the x tiles of the ring as each stage lands (the table is in k order); lane t owns token t,
    // so corr[t][row] accumulates with no atomics; each warp has its own copy
    const int ew = warp - (NW + 1);
    if (COMP) {
      float* cr = corr + ew * CSZ;
      for (int i = lane; i < TOKS * CP; i += 32) cr[i] = 0.f;
      mbar_wait_sleep(pbar, 0);
      const int cnt = ptab_s[0].x;
      constexpr int KST = UPS * 64;
      const uint32_t bq = base4 & 0xFFu;
      int pe = 1;
      for (int st = 0; st < nst; ++st) {
        const int kbeg = (gs0 + st) * KST, kend = kbeg + KST;
        const int slot = st % NSTAGE;
        // Every escape warp observes every stage, in order, works only on its own (st % NESC == ew), and hands EVERY
        // stage back (the empty barriers count NW + NESC arrivals). The producer therefore never refills a slot
        // before every escape warp has seen the stage it held, so no warp can find its slot two phases ahead (a
        // parity wait that aliases and wedges the ring: astra review 2B, reproduced with a delayed escape warp).
        SK2_WAIT(full0 + 8 * slot, (st / NSTAGE) & 1);
        if ((st % NESC) != ew) {   // another warp's stage: skip its escapes, hand the slot back
          while (pe <= cnt && (ptab_s[pe].x & 0xFFFF) < kend) ++pe;
          __syncwarp();
          if (lane == 0) mbar_arrive(empty0 + 8 * slot);
          continue;
        }
        const unsigned char* xs = gbase + slot * SST;
        while (pe <= cnt) {
          const int2 e = ptab_s[pe];
          const int kk = e.x & 0xFFFF;
          if (kk >= kend) break;
          if (lane < M && kk >= kbeg) {
            const int kl = kk - kbeg, u = kl >> 6, kc = kl & 63;
            const __nv_bfloat16 v = *reinterpret_cast<const __nv_bfloat16*>(
                xs + u * XU + lane * 128 + (((kc >> 3) ^ ((u * Mb + lane) & 7)) << 4) + (kc & 7) * 2);
            cr[lane * CP + (e.x >> 16)] += esc_delta((uint32_t)e.y, bq) * __bfloat162float(v);
          }
          ++pe;
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      }
    }
    mbar_arrive(cbar);
    __syncwarp();
    if constexpr (CL > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }

  // ---------------- consumers
  const int wr = warp / WSK, wk = warp % WSK;
  const int g = lane >> 2, t4 = lane & 3;
  const int q = lane >> 3;
  const int chi = q >> 1;
  int trow[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) trow[mt] = TOKR >= 16 ? mt * 16 + (lane & 7) + 8 * (q & 1) : (lane & 7);
  float acc[NACC][MT][NT][4];
#pragma unroll
  for (int c = 0; c < NACC; ++c)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) acc[c][mt][nt][0] = acc[c][mt][nt][1] = acc[c][mt][nt][2] = acc[c][mt][nt][3] = 0.f;

  for (int st = 0; st < nst; ++st) {
    const int slot = st % NSTAGE;
    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
    const uint32_t s0 = sbase + slot * SST;
#pragma unroll
    for (int uu = 0; uu < UPW; ++uu) {
      const int ui = wk * UPW + uu;
      const uint32_t xt = s0 + ui * XU;
      const uint32_t wb = s0 + XBY + (warp * UPW + uu) * NT * TU;
      // 1) every shared-memory load of this unit first (weights of all tiles, x fragments of all 4 k-steps)
      uint4 lw[NT];
      uint2 cw[NT];
      uint32_t b[NT][8];
      uint32_t bl[HL ? NT : 1][8];
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        if constexpr (COMP) {
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(lw[nt].x), "=r"(lw[nt].y), "=r"(lw[nt].z), "=r"(lw[nt].w) : "r"(wb + nt * TU + lane * 16));
          asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(cw[nt].x), "=r"(cw[nt].y) : "r"(wb + nt * TU + 512 + lane * 8));
        } else {
          const uint32_t a0 = wb + nt * TU + lane * 32;
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(b[nt][0]), "=r"(b[nt][1]), "=r"(b[nt][2]), "=r"(b[nt][3]) : "r"(a0));
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(b[nt][4]), "=r"(b[nt][5]), "=r"(b[nt][6]), "=r"(b[nt][7]) : "r"(a0 + 16));
          if constexpr (HL) {
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bl[nt][0]), "=r"(bl[nt][1]), "=r"(bl[nt][2]), "=r"(bl[nt][3]) : "r"(a0 + 1024));
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bl[nt][4]), "=r"(bl[nt][5]), "=r"(bl[nt][6]), "=r"(bl[nt][7]) : "r"(a0 + 1040));
          }
        }
      }
      uint32_t a[4][MT][4];
      if (dgv & 16) {
      } else {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int mt = 0; mt < MT; ++mt)
            ldsm_x4((PK && trow[mt] >= M) ? zrow
                        : xt + trow[mt] * 128 + (((2 * j + chi) ^ ((ui * Mb + trow[mt]) & 7)) << 4),
                    a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);
      }
      // the stage's shared memory is in registers now: hand the slot back before decoding and multiplying
      if (UPW == 1 && uu == 0) {
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      }
      // 2) decode
      if constexpr (COMP) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          decode8(lw[nt].x, lw[nt].y, cw[nt].x, base4, &b[nt][0]);
          decode8(lw[nt].z, lw[nt].w, cw[nt].y, base4, &b[nt][4]);
        }
      }
      // 3) tensor-core work: independent accumulators per k-step
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) {
            constexpr int HM = HL ? 2 : 1;
            mma_bf16(acc[((uu * 4 + j) * HM) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], b[nt][2 * j], b[nt][2 * j + 1]);
            if constexpr (HL)
              mma_bf16(acc[((uu * 4 + j) * HM + 1) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], bl[nt][2 * j], bl[nt][2 * j + 1]);
          }
    }
    if (UPW > 1) {
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    }
  }

  // ---------------- epilogue: K-warps reduce through shared memory (the ring is free now)
  if constexpr (COMP) mbar_wait(cbar, 0);   // the escape warp reads the ring's x tiles until it is done
  if (CL > 1 && crank > 0) {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");   // rank 0's qbar is initialized
    // ranks > 0: every accumulator goes straight into rank 0's peer buffer (slot crank - 1) by st.async
    // (completes rank 0's qbar transaction count; no fence, no wait), escape corrections folded into k-warp 0's
    const uint32_t peer_l = (uint32_t)__cvta_generic_to_shared(peer + (crank - 1) * WSK * TOKS * RP);
    uint32_t peer_r, qbar_r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(peer_r) : "r"(peer_l));
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(qbar_r) : "r"(qbar));
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int row = (wr * NT + nt) * 8 + 2 * t4;
        const int tok = mt * 16 + g;
        float v[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          v[e] = acc[0][mt][nt][e];
#pragma unroll
          for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];
        }
        if (COMP && wk == 0) {
          v[0] += corr[tok * CP + row];
          v[1] += corr[tok * CP + row + 1];
          if (tok + 8 < 32) {
            v[2] += corr[(tok + 8) * CP + row];
            v[3] += corr[(tok + 8) * CP + row + 1];
          }
        }
        const uint32_t a0 = peer_r + (uint32_t)(((wk * TOKS + tok) * RP + row) * 4);
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];"
                     :: "r"(a0), "r"(__float_as_uint(v[0])), "r"(__float_as_uint(v[1])), "r"(qbar_r) : "memory");
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];"
                     :: "r"(a0 + 8 * RP * 4), "r"(__float_as_uint(v[2])), "r"(__float_as_uint(v[3])), "r"(qbar_r) : "memory");
      }
    return;
  }
  named_bar(1, NC);
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const int row = (wr * NT + nt) * 8 + 2 * t4;
      const int tok = mt * 16 + g;
      float* r0 = red + (wk * TOKS + tok) * RP + row;
      float v[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        v[e] = acc[0][mt][nt][e];
#pragma unroll
        for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];
      }
      *reinterpret_cast<float2*>(r0) = make_float2(v[0], v[1]);
      *reinterpret_cast<float2*>(r0 + 8 * RP) = make_float2(v[2], v[3]);
    }
  named_bar(1, NC);
  if constexpr (CL > 1) {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    mbar_wait_cluster(qbar, 0);
  }
  constexpr int R4 = R / 4;
  for (int idx = tid; idx < R4 * M; idx += NC) {
    const int tt = idx / R4, row = (idx - tt * R4) * 4;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if constexpr (COMP) {
#pragma unroll
      for (int c = 0; c < NESC; ++c) {
        v.x += corr[c * CSZ + tt * CP + row];
        v.y += corr[c * CSZ + tt * CP + row + 1];
        v.z += corr[c * CSZ + tt * CP + row + 2];
        v.w += corr[c * CSZ + tt * CP + row + 3];
      }
    }
#pragma unroll
    for (int w = 0; w < WSK; ++w) {
      const float4 p = *reinterpret_cast<const float4*>(red + (w * TOKS + tt) * RP + row);
      v.x += p.x; v.y += p.y; v.z += p.z; v.w += p.w;
#pragma unroll
      for (int c = 0; c < CL - 1; ++c) {
        const float4 q = *reinterpret_cast<const float4*>(peer + ((c * WSK + w) * TOKS + tt) * RP + row);
        v.x += q.x; v.y += q.y; v.z += q.z; v.w += q.w;
      }
    }
    OutT* o = out + (size_t)tt * N + cta * R + row;
    if (COMP && ptab_s[0].y != 0)   // escape-balanced row order (v135): 4-row groups moved whole, so they stay contiguous
      o = out + (size_t)tt * N + reinterpret_cast<const int*>(ptab_s + ptab_s[0].y)[row];
    if constexpr (sizeof(OutT) == 4) {
      *reinterpret_cast<float4*>(o) = v;
    } else {
      __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y), hi = __floats2bfloat162_rn(v.z, v.w);
      uint2 pk;
      pk.x = *reinterpret_cast<uint32_t*>(&lo);
      pk.y = *reinterpret_cast<uint32_t*>(&hi);
      *reinterpret_cast<uint2*>(o) = pk;
    }
  }
}

template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL, typename OutT>
static void launch_sk2(const at::Tensor& Ws, const at::Tensor& x, at::Tensor& out, const at::Tensor& ptab,
                       int64_t pmax, int64_t base4, int M, int N, int K, int64_t dg, unsigned long long* dbg) {
  constexpr int NW = NWR * WSK;
  constexpr int R = NWR * NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int TU = COMP ? 768 : (HL ? 2048 : 1024);
  constexpr int WBY = NW * UPW * NT * TU;
  constexpr int XBY = UPS * TOKR * 128;
  constexpr int SST = ((XBY + WBY) + 1023) & ~1023;
  constexpr int TOKS = 16 * MT;
  constexpr int RP = R + ((40 - (R % 32)) % 32);
  constexpr int CP = R + 1;
  constexpr int NESC = COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1;
  const size_t smem = 1024 + (size_t)NSTAGE * SST + 16 * NSTAGE + 32 + (size_t)NESC * ((TOKS * CP + 3) & ~3) * 4 +
                      (CL > 1 ? (size_t)(CL - 1) * WSK * TOKS * RP * 4 : 0) + (size_t)pmax * 8 + 128;
  auto go = [&](auto pkc) {
    constexpr bool PK = decltype(pkc)::value;
    auto kern = sk2_kernel<MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, COMP, HL, NACC, CL, OutT, PK>;
    static bool attr_done = false;
    if (!attr_done) {
      int dev = 0, optin = 0;
      cudaGetDevice(&dev);
      cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
      cudaFuncAttributes fa;
      cudaFuncGetAttributes(&fa, kern);
      cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)fa.sharedSizeBytes);
      attr_done = true;
    }
    CUtensorMap xm = make_map_x3d(x.data_ptr(), (uint64_t)M, (uint64_t)K, PK ? (uint32_t)M : (uint32_t)TOKR, UPS);
    const uint8_t* wsp = reinterpret_cast<const uint8_t*>(Ws.data_ptr());
    const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    OutT* op = reinterpret_cast<OutT*>(out.data_ptr());
    const int2* tp = reinterpret_cast<const int2*>(ptab.data_ptr());
    if constexpr (CL > 1) {
      cudaLaunchConfig_t lc = {};
      lc.gridDim = dim3(N / R * CL);
      lc.blockDim = dim3((NW + 1 + NESC) * 32);
      lc.dynamicSmemBytes = smem;
      lc.stream = at::cuda::getCurrentCUDAStream();
      cudaLaunchAttribute at_[1];
      at_[0].id = cudaLaunchAttributeClusterDimension;
      at_[0].val.clusterDim.x = CL;
      at_[0].val.clusterDim.y = 1;
      at_[0].val.clusterDim.z = 1;
      lc.attrs = at_;
      lc.numAttrs = 1;
      C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, xm, wsp, xp, op, tp, (int)pmax, (uint32_t)base4, M, N, K, (int)dg, dbg));
    } else {
      kern<<<N / R, (NW + 1 + NESC) * 32, smem, at::cuda::getCurrentCUDAStream()>>>(xm, wsp, xp, op, tp, (int)pmax,
                                                                               (uint32_t)base4, M, N, K, (int)dg, dbg);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  };
  if (TOKR == 8 || M < TOKR) go(std::true_type{}); else go(std::false_type{});
}

void sk2(at::Tensor Ws, at::Tensor x, at::Tensor out, at::Tensor ptab, int64_t pmax, int64_t base4,
         int64_t M, int64_t N, int64_t K, int64_t cfg, int64_t tokr, int64_t dg, at::Tensor dbgt) {
  unsigned long long* dbg = dbgt.numel() >= 8192 + 3 * 1024 ? reinterpret_cast<unsigned long long*>(dbgt.data_ptr()) : nullptr;
  @@SK2_DISPATCH@@
  TORCH_CHECK(false, "sk2: no config");
}
"""


    WIDE_SRC = r"""
// ============================================================================ wide kernel (m 33..128)
// Weights: the per-row-block compressed stream [rb][unit][warp][1536 B] (unit = 16 rows x 64 k per warp),
// one bulk copy per stage; x: one TMA tile [TOK x 64] (128B swizzle) per stage.  Consumers decode the
// weights into registers and run RS-wgmma m64nTOKk16 (sm_90) / mma.sync on the swizzled tile (sm_80 check
// path).  Split-K over units (uneven splits allowed); escapes corrected from a per-CTA table.
template <int NWG, int TOK, int STAGES, bool HLW, typename OutT>
__global__ void __launch_bounds__(NWG * 128, 1)
wide_kernel(const __grid_constant__ CUtensorMap mapB, const __grid_constant__ CUtensorMap mapA, const uint8_t* __restrict__ Wst,
            const __nv_bfloat16* __restrict__ x, OutT* __restrict__ out, float* __restrict__ ws,
            int* __restrict__ counters, const int2* __restrict__ ptab, int pmax, uint32_t base4,
            int M, int N, int K, int splits, int dist) {
  // No separate producer warp (17 warps would cap registers at 96 per thread: 5 warps on one SM sub-partition):
  // thread 0 issues the stage copies.
  constexpr int NWARP = NWG * 4;
  constexpr int NT = NWG * 128;
  // per warp per 64-k unit: compressed 1536 B, or (HLW, router) BF16 hi tile 2 KB + lo tile 2 KB in fragment order
  constexpr int WB = HLW ? 4096 : 1536;
  constexpr int A_BYTES = NWARP * WB;
  constexpr int B_BYTES = TOK * 128;
  constexpr int ST_BYTES = A_BYTES + B_BYTES;
  constexpr int NACC = TOK / 2;
  static_assert(ST_BYTES % 1024 == 0, "stage alignment");
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  __shared__ int s_last;
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t full0 = sbase + STAGES * ST_BYTES, empty0 = full0 + 8 * STAGES;
  int2* ptab_s = reinterpret_cast<int2*>(gbase + STAGES * ST_BYTES + 16 * STAGES);
  __nv_bfloat16* xpatch = reinterpret_cast<__nv_bfloat16*>(ptab_s + pmax);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int rb = blockIdx.x, sp = blockIdx.y, nrb = gridDim.x;
  const int zh = blockIdx.z;                    // token block: rows [zh*TOK, zh*TOK + Mz) of x / out
  const int t0z = zh * TOK;
  const int Mz = min(TOK, M - t0z);
  const int U = K >> 6;
  const int u0 = (int)(((long long)sp * U) / splits);
  const int u1 = (int)(((long long)(sp + 1) * U) / splits);
  const int nst = u1 - u0;
  const int cta = sp * nrb + rb;
  const uint8_t* wsrc = Wst + ((size_t)rb * U + u0) * A_BYTES;

  if (tid == 0) {
    for (int i = 0; i < STAGES; ++i) {
      mbar_init(full0 + 8 * i, 1);
      mbar_init(empty0 + 8 * i, NWG);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    if (splits > 1) asm volatile("prefetch.global.L2 [%0];" :: "l"(counters + zh * nrb + rb));
  }

  // stage issue (sm_90: thread 0, TMA + one bulk copy; sm_80 check path: warp 0, cp.async)
  auto issue = [&](int i) {
    const int slot = i % STAGES;
    const uint32_t st = sbase + slot * ST_BYTES;
    mbar_expect_tx(full0 + 8 * slot, ST_BYTES);
    tma_load_2d_el(st, &mapB, full0 + 8 * slot, (u0 + i) * 64, t0z);
    if constexpr (HLW) {   // router: BF16 hi and lo tiles [64 rows x 64 k] by TMA (128B swizzle) from [hi | lo]
      tma_load_2d_ef(st + B_BYTES, &mapA, full0 + 8 * slot, (u0 + i) * 64, rb * 64);
      tma_load_2d_ef(st + B_BYTES + 8192, &mapA, full0 + 8 * slot, K + (u0 + i) * 64, rb * 64);
    } else {
      bulk_g2s(st + B_BYTES, wsrc + (size_t)i * A_BYTES, A_BYTES, full0 + 8 * slot);
    }
  };
  const bool producer = tid == 0;
  if (producer) {   // thread 0: barriers are initialized (program order) -> issue before the CTA barrier
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapB) : "memory");
    if constexpr (HLW) asm volatile("prefetch.tensormap [%0];" :: "l"(&mapA) : "memory");
    for (int i = 0; i < STAGES && i < nst; ++i) issue(i);
  }
  __syncthreads();
  // escape table + the x values it needs -> shared memory (used after the main loop)
  {
    const int2* tab = ptab + (size_t)cta * pmax;
    const int cnt = pmax ? tab[0].x : 0;
    if (pmax) for (int p = tid; p <= cnt; p += NT) ptab_s[p] = tab[p];
    for (int idx = tid; idx < cnt * TOK; idx += NT) {
      const int p = idx / TOK + 1, t = idx - (p - 1) * TOK;
      const int kl = tab[p].x & 0xFFFF;
      xpatch[p * TOK + t] = t < Mz ? x[(size_t)(t0z + t) * K + u0 * 64 + kl] : __float2bfloat16(0.f);
    }
  }

  // ---------------- consumers
  const int g = lane >> 2, t4 = lane & 3;
  float acc[NACC];
#pragma unroll
  for (int q = 0; q < NACC; ++q) acc[q] = 0.f;
  uint32_t afrag[2][4];
  for (int i = 0; i < nst; ++i) {
    const int slot = i % STAGES;
    mbar_wait(full0 + 8 * slot, (i / STAGES) & 1);
    const uint32_t st = sbase + slot * ST_BYTES;
    const uint32_t wa = st + B_BYTES + warp * WB + lane * 16;
    if constexpr (HLW) {
      // router: SS-wgmma on the TMA'd hi and lo tiles (no register-sourced operands: the wgmma pipeline stays full)
#pragma unroll
      for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
      wg_fence();
#pragma unroll
      for (int j = 0; j < 4; ++j) wgmma_ss<TOK>(acc, make_desc_sw128(st + B_BYTES + j * 32), make_desc_sw128(st + j * 32));
#pragma unroll
      for (int j = 0; j < 4; ++j)
        wgmma_ss<TOK>(acc, make_desc_sw128(st + B_BYTES + 8192 + j * 32), make_desc_sw128(st + j * 32));
      wg_commit();
      wg_wait<1>();
#pragma unroll
      for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    } else {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      // k-step j: low bytes = 8 B at (j/2)*512 + (j%2)*8, codes = 4 B at 1024 + 4j (per-k-step loads: low pressure)
      uint32_t lw0, lw1, cw;
      asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(lw0), "=r"(lw1) : "r"(wa + (j >> 1) * 512 + (j & 1) * 8));
      asm volatile("ld.shared.u32 %0, [%1];" : "=r"(cw) : "r"(wa + 1024 + 4 * j));
      uint32_t* af = afrag[j & 1];
      decode8(lw0, lw1, cw, base4, af);
#pragma unroll
      for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
      wg_fence();
      wgmma_bn<TOK>(acc, af[0], af[1], af[2], af[3], make_desc_sw128(st + j * 32));
      wg_commit();
      wg_wait<1>();
#pragma unroll
      for (int q = 0; q < 4; ++q) asm volatile("" : "+r"(afrag[0][q]), "+r"(afrag[1][q]) :: "memory");
    }
    }  // !HLW
    // stage i-1's wgmmas are complete (one group in flight) -> its slot can be refilled
    if (i > 0) {
      if ((tid & 127) == 0) mbar_arrive(empty0 + 8 * ((i - 1) % STAGES));
      if (producer && i - 1 + STAGES < nst) {
        mbar_wait(empty0 + 8 * ((i - 1) % STAGES), ((i - 1) / STAGES) & 1);
        issue(i - 1 + STAGES);
      }
    }
  }
  wg_wait<0>();
#pragma unroll
  for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");

  // ---------------- exact corrections for escapes (table + x values staged at kernel start)
  named_bar(1, NT);
  {
    const int cnt = pmax ? ptab_s[0].x : 0;
    for (int p = 1; p <= cnt; ++p) {
      const int2 e = ptab_s[p];
      const int rl = e.x >> 16;
      if ((rl >> 4) != warp) continue;
      const int r16 = rl & 15;
      if (g != (r16 & 7)) continue;
      const float d = __int_as_float(e.y);
      const __nv_bfloat16* xv = xpatch + p * TOK;
      const bool hi = r16 >= 8;
#pragma unroll
      for (int jj = 0; jj < TOK / 8; ++jj) {
        const float v0 = d * __bfloat162float(xv[8 * jj + 2 * t4]);
        const float v1 = d * __bfloat162float(xv[8 * jj + 2 * t4 + 1]);
        if (hi) { acc[4 * jj + 2] += v0; acc[4 * jj + 3] += v1; }
        else { acc[4 * jj] += v0; acc[4 * jj + 1] += v1; }
      }
    }
  }

  const int row_w = rb * (NWARP * 16) + warp * 16;
  if (splits == 1) {
#pragma unroll
    for (int j = 0; j < TOK / 8; ++j) {
      const int t0 = 8 * j + 2 * t4;
      if (t0 < Mz) {
        out[(size_t)(t0z + t0) * N + row_w + g] = cvt_out<OutT>(acc[4 * j]);
        out[(size_t)(t0z + t0) * N + row_w + g + 8] = cvt_out<OutT>(acc[4 * j + 2]);
      }
      if (t0 + 1 < Mz) {
        out[(size_t)(t0z + t0 + 1) * N + row_w + g] = cvt_out<OutT>(acc[4 * j + 1]);
        out[(size_t)(t0z + t0 + 1) * N + row_w + g + 8] = cvt_out<OutT>(acc[4 * j + 3]);
      }
    }
    return;
  }
  {
    float4* wsp = reinterpret_cast<float4*>(ws) + ((size_t)(zh * nrb + rb) * splits + sp) * (NACC / 4) * NT;
#pragma unroll
    for (int j = 0; j < NACC / 4; ++j)
      __stcg(wsp + j * NT + tid, make_float4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3]));
  }
  named_bar(1, NT);
  if (tid == 0) {
    // release: every thread's partial (ordered before the CTA barrier) is visible before the count; acquire on the
    // spin makes the other CTAs' partials visible to this CTA after the barrier below
    int old;
    asm volatile("atom.add.release.gpu.global.s32 %0, [%1], 1;" : "=r"(old) : "l"(counters + zh * nrb + rb) : "memory");
    s_last = (old % splits) == splits - 1;
    if (dist) {
      const int target = (old / splits + 1) * splits;
      int v;
      do {
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(counters + zh * nrb + rb) : "memory");
      } while (v - target < 0);
    } else if (s_last) {
      asm volatile("fence.acq_rel.gpu;" ::: "memory");
    }
  }
  named_bar(1, NT);
  if (!dist && !s_last) return;
  const int j0 = dist ? (int)(((long long)sp * (NACC / 4)) / splits) : 0;
  const int j1 = dist ? (int)(((long long)(sp + 1) * (NACC / 4)) / splits) : NACC / 4;
  const float4* base = reinterpret_cast<const float4*>(ws) + (size_t)(zh * nrb + rb) * splits * (NACC / 4) * NT;
  for (int j = j0; j < j1; ++j) {
    float4 s4 = make_float4(0.f, 0.f, 0.f, 0.f);
    int q = 0;
    for (; q + 4 <= splits; q += 4) {
      const float4 v0 = __ldcg(base + ((size_t)(q + 0) * (NACC / 4) + j) * NT + tid);
      const float4 v1 = __ldcg(base + ((size_t)(q + 1) * (NACC / 4) + j) * NT + tid);
      const float4 v2 = __ldcg(base + ((size_t)(q + 2) * (NACC / 4) + j) * NT + tid);
      const float4 v3 = __ldcg(base + ((size_t)(q + 3) * (NACC / 4) + j) * NT + tid);
      s4.x += v0.x + v1.x + v2.x + v3.x; s4.y += v0.y + v1.y + v2.y + v3.y;
      s4.z += v0.z + v1.z + v2.z + v3.z; s4.w += v0.w + v1.w + v2.w + v3.w;
    }
    for (; q < splits; ++q) {
      const float4 v = __ldcg(base + ((size_t)q * (NACC / 4) + j) * NT + tid);
      s4.x += v.x; s4.y += v.y; s4.z += v.z; s4.w += v.w;
    }
    const int t0 = 8 * j + 2 * t4;
    if (t0 < Mz) {
      out[(size_t)(t0z + t0) * N + row_w + g] = cvt_out<OutT>(s4.x);
      out[(size_t)(t0z + t0) * N + row_w + g + 8] = cvt_out<OutT>(s4.z);
    }
    if (t0 + 1 < Mz) {
      out[(size_t)(t0z + t0 + 1) * N + row_w + g] = cvt_out<OutT>(s4.y);
      out[(size_t)(t0z + t0 + 1) * N + row_w + g + 8] = cvt_out<OutT>(s4.w);
    }
  }
}

template <int NWG, int TOK, int STAGES, bool HLW, typename OutT>
static void launch_wide(const at::Tensor& Wst, const at::Tensor& x, at::Tensor& out, at::Tensor& ws,
                        at::Tensor& counters, const at::Tensor& ptab, int64_t pmax, int64_t base4,
                        int M, int N, int K, int splits) {
  constexpr int ST_BYTES = NWG * 4 * (HLW ? 4096 : 1536) + TOK * 128;
  const int nz = (M + TOK - 1) / TOK;   // token blocks (same weight stream: concurrent CTAs share it in L2)
  const size_t smem = 1024 + (size_t)STAGES * ST_BYTES + 16 * STAGES + (size_t)pmax * 8 + (size_t)pmax * TOK * 2 + 64;
  auto kern = wide_kernel<NWG, TOK, STAGES, HLW, OutT>;
  int dev = 0;
  cudaGetDevice(&dev);
  static bool attr_done = false;
  if (!attr_done) {
    int optin = 0;
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)fa.sharedSizeBytes);
    attr_done = true;
  }
  int major = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  CUtensorMap mB, mA;
  memset(&mB, 0, sizeof(mB));
  if (major >= 9) mB = make_map_2d(x.data_ptr(), (uint64_t)M, (uint64_t)K, TOK);
  mA = mB;
  if (HLW && major >= 9) mA = make_map_2d(Wst.data_ptr(), (uint64_t)Wst.size(0), (uint64_t)Wst.size(1), 64);   // [hi | lo]
  dim3 grid(N / (NWG * 64), splits, nz);
  int nsm = 0, per_sm = 0;
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, NWG * 128, smem);
  const int dist = (int)(grid.x * grid.y * grid.z) <= per_sm * nsm ? 1 : 0;
  kern<<<grid, NWG * 128, smem, at::cuda::getCurrentCUDAStream()>>>(
      mB, mA, reinterpret_cast<const uint8_t*>(Wst.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<OutT*>(out.data_ptr()), ws.data_ptr<float>(), counters.data_ptr<int>(),
      reinterpret_cast<const int2*>(ptab.data_ptr()), (int)pmax, (uint32_t)base4, M, N, K, splits, dist);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void wide(at::Tensor Wst, at::Tensor x, at::Tensor out, at::Tensor ws, at::Tensor counters, at::Tensor ptab,
          int64_t pmax, int64_t base4, int64_t M, int64_t N, int64_t K, int64_t splits, int64_t nwg, int64_t tok) {
  const int hl = out.scalar_type() == at::kFloat ? 1 : 0;
  @@WIDE_DISPATCH@@
  TORCH_CHECK(false, "wide: no config");
}
"""


    def wgmma_ss_defs(bns):
        out = []
        for bn in bns:
            nacc = bn // 2
            regs = ", ".join(f"%{i}" for i in range(nacc))
            cons = ", ".join(f'"+f"(d[{i}])' for i in range(nacc))
            a, b, p = nacc, nacc + 1, nacc + 2
            out.append(f"""
    template <int BN_> __device__ __forceinline__ typename std::enable_if<BN_ == {bn}>::type
    wgmma_ss(float* d, uint64_t da, uint64_t db) {{
      asm volatile("{{\\n"
          ".reg .pred p;\\n"
          "setp.ne.b32 p, %{p}, 0;\\n"
          "wgmma.mma_async.sync.aligned.m64n{bn}k16.f32.bf16.bf16 "
          "{{{regs}}}, %{a}, %{b}, p, 1, 1, 0, 0;\\n"
          "}}\\n"
          : {cons}
          : "l"(da), "l"(db), "r"(1));
    }}""")
        return "\n".join(out)


    CPP_SRC = """
void skinny_tma_a(torch::Tensor Wp, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor counters,
                torch::Tensor ptab, int64_t pmax, int64_t base4,
                int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t mt, int64_t rg, int64_t wsn,
                int64_t comp, int64_t nrg, int64_t ks, int64_t dg);
void gemm90(torch::Tensor W, torch::Tensor Wst, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor counters,
            int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t nwg, int64_t bn, int64_t as);
void wide(torch::Tensor Wst, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor counters, torch::Tensor ptab,
          int64_t pmax, int64_t base4, int64_t M, int64_t N, int64_t K, int64_t splits, int64_t nwg, int64_t tok);
void gq(torch::Tensor W, torch::Tensor x, torch::Tensor out, int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nwg,
        int64_t cl, int64_t nstage);
void sk2(torch::Tensor Ws, torch::Tensor x, torch::Tensor out, torch::Tensor ptab, int64_t pmax, int64_t base4,
         int64_t M, int64_t N, int64_t K, int64_t cfg, int64_t tokr, int64_t dg, torch::Tensor dbgt);
void rgemv(torch::Tensor W, torch::Tensor x, torch::Tensor out, int64_t M);
void sgemv(torch::Tensor W, torch::Tensor x, torch::Tensor out, int64_t N);
void cgemv(torch::Tensor L, torch::Tensor C, torch::Tensor x, torch::Tensor etab, int64_t pmax, torch::Tensor out, int64_t base4, int64_t rows, int64_t nth);
void gqx(torch::Tensor W, torch::Tensor x, torch::Tensor out, int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nwg,
         int64_t cl, int64_t nstage);
void rk(torch::Tensor Wc, torch::Tensor x, torch::Tensor out, int64_t S, torch::Tensor zc);
"""

    # Weight rows per layer (the full matrices; some routes run on a prefix, e.g.
    # the unpooler slice at m <= 8).
    WEIGHT_ROWS = {"qkv_gate": 2304, "o": 8192, "pooler": 4096, "unpooler": 8192, "router": 256}
    COMPRESSED_LAYERS = {"qkv_gate", "o", "pooler", "unpooler"}
    # At small m the unpooler runs on its first rows only (the slice SGLang scores).
    UNPOOLER_SLICE_ROWS = 1024

    _ext = None


    # ---------------------------------------------------------------------------
    # Skinny (TMA) route config
    # ---------------------------------------------------------------------------
    TMA_CONFIGS = {  # (layer, mt) -> (splits, RG, WS, KS); x chunk M*Kc*2 small, ~128 CTAs; mt 0 = m == 1
        ("pooler", 0): (1, 2, 8, 4), ("pooler", 1): (2, 4, 4, 4), ("pooler", 2): (4, 8, 2, 4), ("pooler", 4): (8, 16, 1, 4),
        ("unpooler", 0): (2, 1, 8, 4), ("unpooler", 1): (2, 1, 8, 4), ("unpooler", 2): (2, 8, 2, 4), ("unpooler", 4): (4, 16, 1, 4),
        ("qkv_gate", 0): (8, 9, 2, 4), ("qkv_gate", 1): (8, 9, 2, 4), ("qkv_gate", 2): (8, 9, 2, 4), ("qkv_gate", 4): (8, 9, 1, 8),
        ("o", 0): (1, 4, 4, 4), ("o", 1): (1, 4, 4, 4), ("o", 2): (1, 4, 4, 4), ("o", 4): (1, 4, 4, 4),
        ("router", 0): (8, 1, 8, 4), ("router", 1): (8, 1, 8, 4), ("router", 2): (8, 1, 8, 4), ("router", 4): (8, 1, 8, 4),
    }
    USE_WGM = True   # m >= 9 compressed configs with RG % 4 == 0 use RS-wgmma (sm_90)
    WGM_SM80_TEST = False  # local check of the wgmma path's data layout (mma.sync emulation on sm_80)


    def _tma_nstage(nw, comp, sm90, ks, hl=False):
        """Pipeline stages for a skinny config: as many as fit the smem budget."""
        chunk = (ks // 4) * 1536 if comp else ks * 512 * (2 if hl else 1)
        budget = SM90_SMEM_BUDGET_BYTES if sm90 else SM80_SMEM_BUDGET_BYTES
        return max(TMA_MIN_STAGES, min(TMA_MAX_STAGES, budget // (nw * chunk)))


    # alternative skinny configs (more splits -> fewer x bytes per CTA); the prepare()-time autotuner times them
    # against the default config and cuBLAS on every scored shape
    TMA_ALT = {("pooler", 8): (8, 16, 1, 4), ("pooler", 16): (8, 16, 1, 4), ("unpooler", 1): (8, 4, 1, 4),
               ("unpooler", 8): (8, 4, 1, 4), ("unpooler", 16): (4, 16, 1, 4), ("o", 16): (4, 16, 1, 4),
               ("o", 32): (4, 16, 1, 4)}


    def _tma_cfg_items():
        items = [((layer, mtk), cfg) for (layer, mtk), cfg in sorted(TMA_CONFIGS.items())]
        items += [(("alt", layer, m), cfg) for (layer, m), cfg in sorted(TMA_ALT.items())]
        return items


    def _tma_srcs(sm90):
        lines = {}
        for key, (splits, rg, wsn, ks) in _tma_cfg_items():
            if key[0] == "alt":
                layer, mtk = key[1], _mt_key(key[2])
            else:
                layer, mtk = key
            mt = max(mtk, 1)
            f32 = layer == "router"
            ot = "float" if f32 else "__nv_bfloat16"
            comp = 0 if f32 else 1
            hl = f32
            ns = _tma_nstage(rg * wsn, comp, sm90, ks, hl)
            if f32 and mt >= 4:
                ns = min(ns, 3)  # x chunk (32 x 1024) + ring must fit in shared memory
            n_cfg = UNPOOLER_SLICE_ROWS if (layer == "unpooler" and mtk <= 1) else WEIGHT_ROWS[layer]
            wgm = bool((sm90 or WGM_SM80_TEST) and comp and USE_WGM and rg % 4 == 0 and mtk >= 2)
            line = (f"if (mt == {mt} && rg == {rg} && wsn == {wsn} && ks == {ks} && comp == {comp} && out_f32 == {str(f32).lower()}) "
                    f"{{ launch_tma<{mt}, {rg}, {wsn}, {ks}, {ns}, {str(bool(comp)).lower()}, {str(hl).lower()}, {str(wgm).lower()}, {ot}>(Wp, x, out, ws, "
                    f"counters, ptab, pmax, base4, M, N, Kp, Kx, splits, nrg, (int)dg); return; }}")
            lines.setdefault(line, []).append(key)
        return TMA_SRC.replace("@@TMA_DISPATCH@@", "\n  ".join(lines)).replace("@@TMA_NAME@@", "skinny_tma_a")


    GEMM_CONFIGS = {  # (layer, bn) -> (nwg, splits, stages); grid = (N / (64 nwg), splits) <= 132 (co-resident)
        ("pooler", 64): (2, 4, 8), ("unpooler", 64): (2, 2, 8), ("qkv_gate", 64): (2, 7, 8), ("o", 64): (2, 2, 8),
        ("router", 64): (2, 32, 8),
        ("pooler", 128): (2, 4, 6), ("unpooler", 128): (2, 2, 6), ("qkv_gate", 128): (2, 7, 6), ("o", 128): (2, 2, 6),
        ("router", 128): (2, 32, 6),
        ("pooler", 256): (2, 4, 4), ("unpooler", 256): (2, 2, 4), ("qkv_gate", 256): (2, 7, 4), ("o", 256): (2, 2, 4),
        ("router", 256): (2, 32, 4),
    }


    GEMM_DIAG = False   # diagnostic: also time gemm90 variants (TMA-2D A vs contiguous pre-swizzled A) on m 65..256




    def _gemm_src(sm90):
        lines = []
        for (layer, bn), (nwg, splits, stages) in sorted(GEMM_CONFIGS.items()):
            f32 = layer == "router"
            ot = "float" if f32 else "__nv_bfloat16"
            for a_s in ((0, 1) if (not f32 and GEMM_DIAG) else (0,)):
                lines.append(f"if (nwg == {nwg} && bn == {bn} && out_f32 == {str(f32).lower()} && as == {a_s}) "
                             f"{{ launch_gemm90<{nwg}, {bn}, {stages}, {str(bool(a_s)).lower()}, {ot}>(W, Wst, x, out, ws, counters, M, N, Kp, Kx, splits); return; }}")
        bns = sorted({bn for (_, bn) in GEMM_CONFIGS})
        return (GEMM_SRC.replace("@@WGMMA_SS_DEFS@@", wgmma_ss_defs(bns))
                .replace("@@GEMM_DISPATCH@@", "\n  ".join(dict.fromkeys(lines))))


    GQ_SRC = r"""
// ============================================================================ gq (m 33..256): BF16 weights, SS-wgmma
// One CTA = BM = NWG*64 weight rows x BN tokens: W tile [BM x 64 k] and x tile [BN x 64 k] arrive by TMA (128B
// swizzle) into a ring fed by a producer warp; warpgroup w multiplies rows [64w, 64w+64) of every stage (all share
// the x tile, so x is read from L2 once per BM rows).  CL > 1: a cluster of CL CTAs splits K; ranks > 0 hand their
// accumulators to rank 0 by st.async (completing rank 0's barrier transaction count); rank 0 adds and stores BF16.
template <int BN, int NWG, int NSTAGE, int CL>
__global__ void __launch_bounds__(NWG * 128 + 32, 1)
gq_kernel(const __grid_constant__ CUtensorMap mapW, const __grid_constant__ CUtensorMap mapX,
          __nv_bfloat16* __restrict__ out, int M, int N, int K) {
  constexpr int BM = NWG * 64;
  constexpr int A_BYTES = BM * 128;
  constexpr int B_BYTES = BN * 128;
  constexpr int ST = A_BYTES + B_BYTES;
  constexpr int NACC = BN / 2;
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t bars = sbase + NSTAGE * ST;
  const uint32_t full0 = bars, empty0 = bars + 8 * NSTAGE, qbar = bars + 16 * NSTAGE;
  float* peer = reinterpret_cast<float*>(gbase + NSTAGE * ST + 16 * NSTAGE + 16);   // [CL-1][NWG][NACC][128]
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int rb = blockIdx.x / CL, crank = CL > 1 ? (int)(blockIdx.x % CL) : 0;
  const int U = K >> 6;
  const int u0 = (crank * U) / CL, u1 = ((crank + 1) * U) / CL;
  const int nkb = u1 - u0;
  const int xrows = min(M, BN);   // x rows actually loaded (rows >= M of the tile only feed outputs that are dropped)

  if (tid == 0) {
    for (int i = 0; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NWG); }
    mbar_init(qbar, 1);
    if (CL == 2) mbar_expect_tx(qbar, NWG * 64 * NACC * 4);   // split epilogue: the peer's half of our rows
    else if (CL > 1 && crank == 0) mbar_expect_tx(qbar, (CL - 1) * NWG * NACC * 128 * 4);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    // the first ring of stages goes out right away (no cluster barrier in front of it: a cluster-scope release
    // issued after a copy is in flight waits for it, and the stages only need this CTA's barriers)
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapW) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapX) : "memory");
    for (int i = 0; i < NSTAGE && i < nkb; ++i) {
      mbar_expect_tx(full0 + 8 * i, A_BYTES + xrows * 128);
      const int kg = (u0 + i) * 64;
      tma_load_2d_ef(sbase + i * ST, &mapW, full0 + 8 * i, kg, rb * BM);
      tma_load_2d_el(sbase + i * ST + A_BYTES, &mapX, full0 + 8 * i, kg, 0);
    }
  }
  __syncthreads();
  // cluster barrier: arrived here (relaxed: the barrier inits are fenced above), waited on only before remote stores
  if constexpr (CL > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

  if (warp == NWG * 4) {
    if (lane == 0) {
      for (int i = NSTAGE; i < nkb; ++i) {
        const int slot = i % NSTAGE;
        if (i >= NSTAGE) mbar_wait(empty0 + 8 * slot, ((i / NSTAGE) - 1) & 1);   // busy wait
        mbar_expect_tx(full0 + 8 * slot, A_BYTES + xrows * 128);
        const int kg = (u0 + i) * 64;
        tma_load_2d_ef(sbase + slot * ST, &mapW, full0 + 8 * slot, kg, rb * BM);
        tma_load_2d_el(sbase + slot * ST + A_BYTES, &mapX, full0 + 8 * slot, kg, 0);
      }
    }
    __syncwarp();
    if constexpr (CL > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }

  const int wg = warp >> 2, t = tid & 127;
  float acc[NACC];
#pragma unroll
  for (int q = 0; q < NACC; ++q) acc[q] = 0.f;
  for (int i = 0; i < nkb; ++i) {
    const int slot = i % NSTAGE;
    mbar_wait_sleep(full0 + 8 * slot, (i / NSTAGE) & 1);
    const uint32_t a_st = sbase + slot * ST + wg * (64 * 128);
    const uint32_t b_st = sbase + slot * ST + A_BYTES;
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    wg_fence();
#pragma unroll
    for (int j = 0; j < 4; ++j) wgmma_ss<BN>(acc, make_desc_sw128(a_st + j * 32), make_desc_sw128(b_st + j * 32));
    wg_commit();
    wg_wait<1>();   // the previous stage's group is done: hand its slot back
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    if (i > 0 && t == 0) mbar_arrive(empty0 + 8 * ((i - 1) % NSTAGE));
  }
  wg_wait<0>();
#pragma unroll
  for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");

  if constexpr (CL <= 2) {
    // CL = 2: rank r finishes rows half r of every warpgroup (warps 2r, 2r+1); the other two warps hand their sums
    // to the peer by st.async (into [wg][warp][lane][NACC + 4]). The kept rows go out through a shared-memory
    // staging tile [wg][token][rows] so that each token's rows are written as contiguous 16-byte chunks.
    constexpr int RS = NACC + 4;
    constexpr int KR = CL == 2 ? 32 : 64;          // kept rows per warpgroup
    const int q = warp & 3;
    const int keep_lo = CL == 2 ? 2 * crank : 0;   // first kept warp of the warpgroup
    const bool keep = CL == 1 || (q >> 1) == crank;
    if (CL == 2 && !keep) {
      const int ql = q & 1;
      const uint32_t rl = (uint32_t)__cvta_generic_to_shared(peer + ((wg * 2 + ql) * 32 + lane) * RS);
      uint32_t rr, qb;
      asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");   // the peer's qbar is initialized
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rr) : "r"(rl), "r"(crank ^ 1));
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(qb) : "r"(qbar), "r"(crank ^ 1));
#pragma unroll
      for (int i = 0; i < NACC; i += 4)
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                     :: "r"(rr + i * 4), "r"(__float_as_uint(acc[i])), "r"(__float_as_uint(acc[i + 1])),
                        "r"(__float_as_uint(acc[i + 2])), "r"(__float_as_uint(acc[i + 3])), "r"(qb) : "memory");
      return;
    }
    if constexpr (CL == 2) {
      asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
      mbar_wait_cluster(qbar, 0);
      const float* rv = peer + ((wg * 2 + (q & 1)) * 32 + lane) * RS;
#pragma unroll
      for (int i = 0; i < NACC; i += 4) {
        const float4 p4 = *reinterpret_cast<const float4*>(rv + i);
        acc[i] += p4.x; acc[i + 1] += p4.y; acc[i + 2] += p4.z; acc[i + 3] += p4.w;
      }
    }
    // stage BF16 [wg][token][KR rows] in the idle ring (every warpgroup's ring reads are over: wg_wait<0> above;
    // the TMA writes all landed before the last full-barrier wait)
    if constexpr (NWG > 1) named_bar(1, NWG * (CL == 2 ? 64 : 128));
    __nv_bfloat16* stg = reinterpret_cast<__nv_bfloat16*>(gbase);
    {
      const int g = lane >> 2, t4 = lane & 3;
      const int rk = (q - keep_lo) * 16;   // this warp's first kept row
#pragma unroll
      for (int j = 0; j < BN / 8; ++j) {
        const int t0 = 8 * j + 2 * t4;
        __nv_bfloat16* s0 = stg + ((size_t)wg * BN + t0) * KR + rk + g;
        s0[0] = __float2bfloat16_rn(acc[4 * j]);
        s0[8] = __float2bfloat16_rn(acc[4 * j + 2]);
        s0[KR] = __float2bfloat16_rn(acc[4 * j + 1]);
        s0[KR + 8] = __float2bfloat16_rn(acc[4 * j + 3]);
      }
    }
    named_bar(2, NWG * (CL == 2 ? 64 : 128));
    {
      constexpr int CH = KR / 8;               // 16-byte chunks per token row
      const int nkt = NWG * (CL == 2 ? 64 : 128);
      const int kt = (q - keep_lo) * 32 + lane + wg * (CL == 2 ? 64 : 128);
      for (int idx = kt; idx < NWG * BN * CH; idx += nkt) {
        const int c = idx % CH, tt = (idx / CH) % BN, w2 = idx / (CH * BN);
        if (tt < M) {
          const uint4 v = *reinterpret_cast<const uint4*>(stg + ((size_t)w2 * BN + tt) * KR + c * 8);
          *reinterpret_cast<uint4*>(out + (size_t)tt * N + rb * BM + w2 * 64 + (CL == 2 ? crank * 32 : 0) + c * 8) = v;
        }
      }
    }
    return;
  }
}

template <int BN, int NWG, int NSTAGE, int CL>
static void launch_gq(const at::Tensor& W, const at::Tensor& x, at::Tensor& out, int M, int N, int K) {
  constexpr int BM = NWG * 64;
  constexpr int ST = BM * 128 + BN * 128;
  const size_t smem = 1024 + (size_t)NSTAGE * ST + 16 * NSTAGE + 16 +
                      (CL == 2 ? (size_t)NWG * 64 * (BN / 2 + 4) * 4 : (size_t)(CL - 1) * NWG * (BN / 2) * 128 * 4);
  auto kern = gq_kernel<BN, NWG, NSTAGE, CL>;
  static bool attr_done = false;
  if (!attr_done) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attr_done = true;
  }
  CUtensorMap mW = make_map_2d(W.data_ptr(), (uint64_t)W.size(0), (uint64_t)W.size(1), BM);
  CUtensorMap mX = make_map_2d(x.data_ptr(), (uint64_t)M, (uint64_t)K, (uint32_t)(M < BN ? M : BN));
  __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  if constexpr (CL > 1) {
    cudaLaunchConfig_t lc = {};
    lc.gridDim = dim3(N / BM * CL);
    lc.blockDim = dim3(NWG * 128 + 32);
    lc.dynamicSmemBytes = smem;
    lc.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at_[1];
    at_[0].id = cudaLaunchAttributeClusterDimension;
    at_[0].val.clusterDim.x = CL;
    at_[0].val.clusterDim.y = 1;
    at_[0].val.clusterDim.z = 1;
    lc.attrs = at_;
    lc.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, mW, mX, op, M, N, K));
  } else {
  }
}

void gq(at::Tensor W, at::Tensor x, at::Tensor out, int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nwg,
        int64_t cl, int64_t nstage) {
  @@GQ_DISPATCH@@
  TORCH_CHECK(false, "gq: no config");
}
"""

    GQ_CONFIGS = {  # (layer, bn) -> (nwg, cl, nstage): BM = 64*nwg rows per CTA, grid = N/BM * cl CTAs (cl-CTA clusters split K)
        # measured on H200 against cuBLAS (graph replay, cold L2): only the pooler shape wins (64 row blocks x 2 K halves)
        ("pooler", 64): (1, 2, 10), ("pooler", 128): (1, 2, 8),
    }
    USE_GQ = True

    GQX_SRC = r"""
// ============================================================================ gqx (m 33..256): BF16 weights, big tiles
// CTA = BM = 64 NWG weight rows x BN tokens over 1/CL of K; a cluster of CL CTAs covers all of K for one row block.
// Ring of TMA'd W [BM x 64] and x [BN x 64] tiles (128B swizzle) fed by a producer warp; warpgroup w runs SS-wgmma
// m64nBN on rows [64w, 64w + 64). Reduction: after every rank's main loop (cluster barrier: every ring is free), each
// rank owns BN/CL tokens; every thread st.async's the other ranks' token blocks of its accumulator fragment into the
// owner's ring space ([src rank][thread][4 JR floats]) and sums the CL-1 incoming copies of its own block.
template <int BN, int NWG, int NSTAGE, int CL>
__global__ void __launch_bounds__(NWG * 128 + 32, 1)
gqx_kernel(const __grid_constant__ CUtensorMap mapW, const __grid_constant__ CUtensorMap mapX,
           __nv_bfloat16* __restrict__ out, int M, int N, int K) {
  constexpr int BM = NWG * 64;
  constexpr int A_BYTES = BM * 128;
  constexpr int B_BYTES = BN * 128;
  constexpr int ST = A_BYTES + B_BYTES;
  constexpr int NACC = BN / 2;
  constexpr int NTH = NWG * 128;
  constexpr int NB = BN / 8;                 // 8-token blocks of the tile; rank r owns blocks [r NB / CL, (r+1) NB / CL)
  constexpr int JM = (NB + CL - 1) / CL;     // most blocks a rank owns
  static_assert(CL <= NB, "gqx: every rank owns at least one token block");
  static_assert(((NB + CL - 1) / CL) * 8 * BM * 2 <= NSTAGE * ST, "gqx: the BF16 staging tile must fit in the ring");
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t bars = sbase + NSTAGE * ST;
  const uint32_t full0 = bars, empty0 = bars + 8 * NSTAGE, rbar = bars + 16 * NSTAGE;
  // CL > 1: this rank's accumulator fragments [NB][NTH] float4, outside the ring (written right after the main loop)
  float4* frag = reinterpret_cast<float4*>(gbase + NSTAGE * ST + 16 * NSTAGE + 16);
  float* recv = reinterpret_cast<float*>(gbase);   // [CL][NTH][4 JM] (the ring, once every rank is done with it)
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  uint32_t crank = 0;
  if constexpr (CL > 1) asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(crank));
  const int rb = blockIdx.x / CL;
  const int U = K >> 6;
  const int u0 = ((int)crank * U) / CL, u1 = (((int)crank + 1) * U) / CL;
  const int nkb = u1 - u0;
  const int xrows = min(M, BN);

  if (tid == 0) {
    for (int i = 0; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NWG); }
    mbar_init(rbar, 1);

    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapW) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapX) : "memory");
    for (int i = 0; i < NSTAGE && i < nkb; ++i) {
      mbar_expect_tx(full0 + 8 * i, A_BYTES + xrows * 128);
      const int kg = (u0 + i) * 64;
      tma_load_2d_ef(sbase + i * ST, &mapW, full0 + 8 * i, kg, rb * BM);
      tma_load_2d_el(sbase + i * ST + A_BYTES, &mapX, full0 + 8 * i, kg, 0);
    }
  }
  __syncthreads();

  if (warp == NWG * 4) {   // producer: refills; then joins the cluster barrier
    if (lane == 0) {
      for (int i = NSTAGE; i < nkb; ++i) {
        const int slot = i % NSTAGE;
        mbar_wait(empty0 + 8 * slot, ((i / NSTAGE) - 1) & 1);   // busy wait: a sleeping producer refills late
        mbar_expect_tx(full0 + 8 * slot, A_BYTES + xrows * 128);
        const int kg = (u0 + i) * 64;
        tma_load_2d_ef(sbase + slot * ST, &mapW, full0 + 8 * slot, kg, rb * BM);
        tma_load_2d_el(sbase + slot * ST + A_BYTES, &mapX, full0 + 8 * slot, kg, 0);
      }
    }
    __syncwarp();
    if constexpr (CL > 1) {   // the consumers' two cluster barriers (fragments written, reads done)
      asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
      asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
    }
    return;
  }

  const int wg = warp >> 2, t = tid & 127;
  float acc[NACC];
#pragma unroll
  for (int q = 0; q < NACC; ++q) acc[q] = 0.f;
  for (int i = 0; i < nkb; ++i) {
    const int slot = i % NSTAGE;
    mbar_wait(full0 + 8 * slot, (i / NSTAGE) & 1);
    const uint32_t a_st = sbase + slot * ST + wg * (64 * 128);
    const uint32_t b_st = sbase + slot * ST + A_BYTES;
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    wg_fence();
#pragma unroll
    for (int j = 0; j < 4; ++j) wgmma_ss<BN>(acc, make_desc_sw128(a_st + j * 32), make_desc_sw128(b_st + j * 32));
    wg_commit();
    wg_wait<1>();
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    if (i > 0 && t == 0) mbar_arrive(empty0 + 8 * ((i - 1) % NSTAGE));
  }
  wg_wait<0>();
#pragma unroll
  for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");

  const int g = lane >> 2, t4 = lane & 3;
  if constexpr (CL > 1) {
    // 1) fragments out right away (own area: no need to wait for the other ranks' main loops), 2) one cluster
    // barrier, 3) the owner of each token block sums the other ranks' copies over DSMEM, stages BF16 [token][64 rows]
    // in its own (free) ring and 4) writes each token's 64 rows as 128 contiguous bytes; 5) exit barrier
#pragma unroll
    for (int j = 0; j < NB; ++j) frag[j * NTH + tid] = make_float4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3]);
    asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
    const uint32_t fl = (uint32_t)__cvta_generic_to_shared(frag);
    __nv_bfloat16* stg = reinterpret_cast<__nv_bfloat16*>(gbase);   // [owned tokens][BM rows]
    const int jlo = ((int)crank * NB) / CL, jhi = (((int)crank + 1) * NB) / CL;
    const int rl = wg * 64 + (warp & 3) * 16 + g;   // this thread's first row within the tile
#pragma unroll
    for (int r = 0; r < CL; ++r) {
      if (r != (int)crank) continue;
#pragma unroll
      for (int j = 0; j < NB; ++j) {
        if (j < (r * NB) / CL || j >= ((r + 1) * NB) / CL) continue;
        float4 pv[CL];
#pragma unroll
        for (int r2 = 0; r2 < CL; ++r2) {
          if (r2 == r) continue;
          uint32_t ra;
          asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(fl + (uint32_t)((j * NTH + tid) * 16)), "r"(r2));
          asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];"
                       : "=f"(pv[r2].x), "=f"(pv[r2].y), "=f"(pv[r2].z), "=f"(pv[r2].w) : "r"(ra) : "memory");
        }
        float v0 = acc[4 * j], v1 = acc[4 * j + 1], v2 = acc[4 * j + 2], v3 = acc[4 * j + 3];
#pragma unroll
        for (int r2 = 0; r2 < CL; ++r2) {
          if (r2 == r) continue;
          v0 += pv[r2].x; v1 += pv[r2].y; v2 += pv[r2].z; v3 += pv[r2].w;
        }
        const int tl = 8 * (j - (r * NB) / CL) + 2 * t4;   // token within the owned slice
        stg[tl * BM + rl] = __float2bfloat16_rn(v0);
        stg[tl * BM + rl + 8] = __float2bfloat16_rn(v2);
        stg[(tl + 1) * BM + rl] = __float2bfloat16_rn(v1);
        stg[(tl + 1) * BM + rl + 8] = __float2bfloat16_rn(v3);
      }
    }
    named_bar(1, NTH);
    {
      constexpr int CH = BM / 8;                 // 16-byte chunks per token row
      const int nt = 8 * (jhi - jlo);
      for (int idx = tid; idx < nt * CH; idx += NTH) {
        const int tt = idx / CH, c = idx - tt * CH;
        const int tok = 8 * jlo + tt;
        if (tok < M)
          *reinterpret_cast<uint4*>(out + (size_t)tok * N + rb * BM + c * 8) =
              *reinterpret_cast<const uint4*>(stg + tt * BM + c * 8);
      }
    }
    // nobody leaves while another rank may still read its fragments
    asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
  } else {
  }
}

template <int BN, int NWG, int NSTAGE, int CL>
static void launch_gqx(const at::Tensor& W, const at::Tensor& x, at::Tensor& out, int M, int N, int K) {
  constexpr int BM = NWG * 64;
  constexpr int ST = BM * 128 + BN * 128;
  const size_t smem = 1024 + (size_t)NSTAGE * ST + 16 * NSTAGE + 16 + (CL > 1 ? (size_t)(BN / 8) * NWG * 128 * 16 : 0);
  auto kern = gqx_kernel<BN, NWG, NSTAGE, CL>;
  static bool attr_done = false;
  if (!attr_done) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (CL > 8) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    attr_done = true;
  }
  CUtensorMap mW = make_map_2d(W.data_ptr(), (uint64_t)W.size(0), (uint64_t)W.size(1), BM);
  CUtensorMap mX = make_map_2d(x.data_ptr(), (uint64_t)M, (uint64_t)K, (uint32_t)(M < BN ? M : BN));
  __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  cudaLaunchConfig_t lc = {};
  lc.gridDim = dim3(N / BM * CL);
  lc.blockDim = dim3(NWG * 128 + 32);
  lc.dynamicSmemBytes = smem;
  lc.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at_[1];
  at_[0].id = cudaLaunchAttributeClusterDimension;
  at_[0].val.clusterDim.x = CL;
  at_[0].val.clusterDim.y = 1;
  at_[0].val.clusterDim.z = 1;
  lc.attrs = at_;
  lc.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, mW, mX, op, M, N, K));
}

void gqx(at::Tensor W, at::Tensor x, at::Tensor out, int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nwg,
         int64_t cl, int64_t nstage) {
  @@GQX_DISPATCH@@
  TORCH_CHECK(false, "gqx: no config");
}
"""

    GQX_CONFIGS = {  # (layer, bn) -> (nwg, cl, nstage): grid = N / (64 nwg) * cl CTAs
        ("qkv_gate", 128): (1, 3, 8),   # m 33..128: 36 row blocks x 3 K-thirds = 108 CTAs (cuBLAS 16.4 -> 14.0 us at m=128)
    }
    USE_GQX = True
    TORCHT_LAYERS = ("unpooler", "o")   # layers that also get a K-major BF16 copy for the "torchT" (mm(x, W^T)) route


    def _gqx_src(sm90):
        lines = []
        for (layer, bn), (nwg, cl, ns) in sorted(GQX_CONFIGS.items()):
            lines.append(f"if (bn == {bn} && nwg == {nwg} && cl == {cl} && nstage == {ns}) "
                         f"{{ launch_gqx<{bn}, {nwg}, {ns}, {cl}>(W, x, out, M, N, K); return; }}")
        return GQX_SRC.replace("@@GQX_DISPATCH@@", "\n  ".join(dict.fromkeys(lines)))


    def _gqx_bn(m):
        return 128 if m <= 128 else 256


    def _run_gq_family(x, layer, m, n, k, state, configs, bn_of_m, method):
        """Run a gq/gqx cluster GEMM: same call shape, different config table and
    entry point (gq = plain BF16 weights, gqx = big tiles)."""
        bn = bn_of_m(m)
        nwg, cl, ns = configs[(layer, bn)]
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        getattr(_get_ext(), method)(state["raw"][layer], x, out, m, n, k, bn, nwg, cl, ns)
        return out


    def _run_gqx(x, layer, m, n, k, state):
        return _run_gq_family(x, layer, m, n, k, state, GQX_CONFIGS, _gqx_bn, "gqx")



    RGEMV_SRC = r"""
// ============================================================================ router GEMV (m = 1): FP32 weights
// read directly (8.4 MB, no hi/lo split), ROWS rows per CTA, 256 threads; thread i owns k = 8i + 2048j (j < 4): every
// weight and x load is issued before the first FMA; warp butterfly then a fixed-order cross-warp sum (deterministic).
template <int ROWS, int MMAX>
__global__ void __launch_bounds__(256) rgemv_kernel(const float* __restrict__ W, const __nv_bfloat16* __restrict__ x,
                                                    float* __restrict__ out, int M, int N) {
  constexpr int K = 8192;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int r0 = blockIdx.x * ROWS;
  float4 w[ROWS][8];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float4* p = reinterpret_cast<const float4*>(W + (size_t)(r0 + r) * K + j * 2048 + tid * 8);
      w[r][2 * j] = __ldg(p); w[r][2 * j + 1] = __ldg(p + 1);
    }
  float acc[ROWS][MMAX];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int t = 0; t < MMAX; ++t) acc[r][t] = 0.f;
#pragma unroll
  for (int t = 0; t < MMAX; ++t) {
    if (t >= M) break;
    uint4 xv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) xv[j] = __ldg(reinterpret_cast<const uint4*>(x + (size_t)t * K + j * 2048 + tid * 8));
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&xv[j]);
      float xf[8];
#pragma unroll
      for (int e = 0; e < 4; ++e) { const float2 f = __bfloat1622float2(h[e]); xf[2 * e] = f.x; xf[2 * e + 1] = f.y; }
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        const float4 a = w[r][2 * j], b = w[r][2 * j + 1];
        float v = acc[r][t];
        v = fmaf(a.x, xf[0], v); v = fmaf(a.y, xf[1], v); v = fmaf(a.z, xf[2], v); v = fmaf(a.w, xf[3], v);
        v = fmaf(b.x, xf[4], v); v = fmaf(b.y, xf[5], v); v = fmaf(b.z, xf[6], v); v = fmaf(b.w, xf[7], v);
        acc[r][t] = v;
      }
    }
  }
  __shared__ float red[ROWS][MMAX][8];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int t = 0; t < MMAX; ++t) {
      float v = acc[r][t];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if (lane == 0) red[r][t][warp] = v;
    }
  __syncthreads();
  if (tid < ROWS * MMAX) {
    const int r = tid / MMAX, t = tid % MMAX;
    if (t < M) {
      float v = 0.f;
#pragma unroll
      for (int q = 0; q < 8; ++q) v += red[r][t][q];
      out[(size_t)t * N + r0 + r] = v;
    }
  }
}


// ============================================================================ cgemv (m = 1): 12-bit compressed GEMV
// Lossless planes, row-major: L [N][K] low bytes, C [N][K/2] 4-bit codes (sign | exponent offset from base), byte j of
// each 4-byte group g holding codes 8g + j and 8g + j + 4. Exponents outside the 8-value window decode wrong and are
// corrected exactly by a dedicated escape warp from a per-CTA table (entry 0 = count; ((row << 16) | k, delta bits)).
// Every weight and x load is issued before the first FMA; fixed-order reductions (deterministic).
__device__ __forceinline__ uint32_t cg_hib(uint32_t c, uint32_t base4) {
  return ((c & 0x07070707u) + base4) | ((c << 4) & 0x80808080u);
}
template <int ROWS, int NTH, int K>
__global__ void __launch_bounds__(NTH + 32) cgemv_kernel(const uint8_t* __restrict__ L, const uint8_t* __restrict__ C,
    const __nv_bfloat16* __restrict__ x, const int2* __restrict__ etab, int pmax,
    __nv_bfloat16* __restrict__ out, uint32_t base4) {
  constexpr int PER = K / (16 * NTH);
  constexpr int NWARP = NTH / 32;
  static_assert(PER >= 1, "cgemv: K >= 16 NTH");
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int r0 = blockIdx.x * ROWS;
  __shared__ float red[ROWS][NWARP + 1];
  if (warp == NWARP) {   // escape warp: lane i takes entries 1 + i, 33 + i, ... (entry 0 = count); x[k] * delta per row
    const int2* tb = etab + (size_t)blockIdx.x * pmax;
    int2 mine = make_int2(0, 0);
    if (lane < pmax) mine = __ldg(&tb[lane]);
    const int cnt = __shfl_sync(0xffffffffu, mine.x, 0);
    float ec[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) ec[r] = 0.f;
    for (int i0 = 0; i0 <= cnt; i0 += 32) {
      const int i = i0 + lane;
      int2 en = mine;
      if (i0 > 0) en = (i <= cnt) ? __ldg(&tb[i]) : make_int2(0, 0);
      if (i >= 1 && i <= cnt) {
        const float c = __bfloat162float(x[en.x & 0xFFFF]) * esc_delta((uint32_t)en.y, base4 & 0xFFu);
        const int rl = en.x >> 16;
#pragma unroll
        for (int r = 0; r < ROWS; ++r) ec[r] += (rl == r) ? c : 0.f;
      }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      float v = ec[r];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if (lane == 0) red[r][NWARP] = v;
    }
    __syncthreads();
    return;
  }
  uint4 lw[ROWS][PER]; uint2 cw[ROWS][PER]; uint4 xv[PER][2];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const size_t e0 = (size_t)(r0 + r) * K + (size_t)j * 16 * NTH + 16 * tid;
      lw[r][j] = __ldg(reinterpret_cast<const uint4*>(L + e0));
      cw[r][j] = __ldg(reinterpret_cast<const uint2*>(C + e0 / 2));
    }
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const __nv_bfloat16* xp = x + j * 16 * NTH + 16 * tid;
    xv[j][0] = __ldg(reinterpret_cast<const uint4*>(xp));
    xv[j][1] = __ldg(reinterpret_cast<const uint4*>(xp + 8));
  }
  float acc[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    float xf[16];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t* xw = reinterpret_cast<const uint32_t*>(&xv[j][h]);
#pragma unroll
      for (int e = 0; e < 4; ++e) { xf[8 * h + 2 * e] = __uint_as_float(xw[e] << 16); xf[8 * h + 2 * e + 1] = __uint_as_float(xw[e] & 0xFFFF0000u); }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const uint32_t lws[4] = {lw[r][j].x, lw[r][j].y, lw[r][j].z, lw[r][j].w};
      const uint32_t cws[2] = {cw[r][j].x, cw[r][j].y};
      float v = acc[r];
#pragma unroll
      for (int g = 0; g < 2; ++g) {           // 8-element groups: low bytes lws[2g], lws[2g+1]; codes cws[g]
        const uint32_t ha = cg_hib(cws[g] & 0x0F0F0F0Fu, base4);         // hi bytes of elements 0..3
        const uint32_t hb = cg_hib((cws[g] >> 4) & 0x0F0F0F0Fu, base4);  // hi bytes of elements 4..7
        const uint32_t la = lws[2 * g], lb = lws[2 * g + 1];
        const uint32_t p01 = __byte_perm(la, ha, 0x5140), p23 = __byte_perm(la, ha, 0x7362);
        const uint32_t p45 = __byte_perm(lb, hb, 0x5140), p67 = __byte_perm(lb, hb, 0x7362);
        const float* xg = xf + 8 * g;
        v = fmaf(__uint_as_float(p01 << 16), xg[0], v); v = fmaf(__uint_as_float(p01 & 0xFFFF0000u), xg[1], v);
        v = fmaf(__uint_as_float(p23 << 16), xg[2], v); v = fmaf(__uint_as_float(p23 & 0xFFFF0000u), xg[3], v);
        v = fmaf(__uint_as_float(p45 << 16), xg[4], v); v = fmaf(__uint_as_float(p45 & 0xFFFF0000u), xg[5], v);
        v = fmaf(__uint_as_float(p67 << 16), xg[6], v); v = fmaf(__uint_as_float(p67 & 0xFFFF0000u), xg[7], v);
      }
      acc[r] = v;
    }
  }
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    float v = acc[r];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) red[r][warp] = v;
  }
  __syncthreads();
  if (tid < ROWS) {
    float v = 0.f;
#pragma unroll
    for (int q = 0; q <= NWARP; ++q) v += red[tid][q];
    out[r0 + tid] = __float2bfloat16_rn(v);
  }
}

void cgemv(at::Tensor L, at::Tensor C, at::Tensor x, at::Tensor etab, int64_t pmax, at::Tensor out, int64_t base4,
           int64_t rows, int64_t nth) {
  const int N = out.size(1), K = L.size(1);
  TORCH_CHECK(x.size(0) == 1 && N % rows == 0 && N <= L.size(0), "cgemv: m = 1");
  auto st = at::cuda::getCurrentCUDAStream();
  auto a0 = L.data_ptr<uint8_t>(); auto a1 = C.data_ptr<uint8_t>();
  auto xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto tp = reinterpret_cast<const int2*>(etab.data_ptr<int>());
  auto op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  const uint32_t b4 = (uint32_t)base4;
  const int pm = (int)pmax;
  if (K == 8192 && rows == 2 && nth == 256) cgemv_kernel<2, 256, 8192><<<N / 2, 256 + 32, 0, st>>>(a0, a1, xp, tp, pm, op, b4);
  else if (K == 1024 && rows == 2 && nth == 64) cgemv_kernel<2, 64, 1024><<<N / 2, 64 + 32, 0, st>>>(a0, a1, xp, tp, pm, op, b4);
  else if (K == 4096 && rows == 2 && nth == 128) cgemv_kernel<2, 128, 4096><<<N / 2, 128 + 32, 0, st>>>(a0, a1, xp, tp, pm, op, b4);
  else if (K == 8192 && rows == 2 && nth == 512) cgemv_kernel<2, 512, 8192><<<N / 2, 512 + 32, 0, st>>>(a0, a1, xp, tp, pm, op, b4);
  else TORCH_CHECK(false, "cgemv: no instantiation for K=", K, " rows=", rows, " nth=", nth);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// BF16 GEMV (the unpooler slice at m = 1: W[:1024] bf16 [1024, 4096]): 4 rows per CTA, 512 threads, thread i owns
// k = 8i (one 16 B load per row and one of x), fixed-order reductions.
template <int ROWS, int NTH, int K>
__global__ void __launch_bounds__(NTH) sgemv_kernel(const __nv_bfloat16* __restrict__ W, const __nv_bfloat16* __restrict__ x,
                                                    __nv_bfloat16* __restrict__ out) {
  constexpr int PER = K / NTH / 8;
  constexpr int NWARP = NTH / 32;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int r0 = blockIdx.x * ROWS;
  uint4 w[ROWS][PER], xv[PER];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int j = 0; j < PER; ++j) w[r][j] = __ldg(reinterpret_cast<const uint4*>(W + (size_t)(r0 + r) * K + j * (NTH * 8) + tid * 8));
#pragma unroll
  for (int j = 0; j < PER; ++j) xv[j] = __ldg(reinterpret_cast<const uint4*>(x + j * (NTH * 8) + tid * 8));
  float acc[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const __nv_bfloat162* hx = reinterpret_cast<const __nv_bfloat162*>(&xv[j]);
    float xf[8];
#pragma unroll
    for (int e = 0; e < 4; ++e) { const float2 f = __bfloat1622float2(hx[e]); xf[2 * e] = f.x; xf[2 * e + 1] = f.y; }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const __nv_bfloat162* hw = reinterpret_cast<const __nv_bfloat162*>(&w[r][j]);
      float v = acc[r];
#pragma unroll
      for (int e = 0; e < 4; ++e) { const float2 a = __bfloat1622float2(hw[e]); v = fmaf(a.x, xf[2 * e], v); v = fmaf(a.y, xf[2 * e + 1], v); }
      acc[r] = v;
    }
  }
  __shared__ float red[ROWS][NWARP];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    float v = acc[r];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) red[r][warp] = v;
  }
  __syncthreads();
  if (tid < ROWS) {
    float v = 0.f;
#pragma unroll
    for (int q = 0; q < NWARP; ++q) v += red[tid][q];
    out[r0 + tid] = __float2bfloat16_rn(v);
  }
}

void sgemv(at::Tensor W, at::Tensor x, at::Tensor out, int64_t N) {
  TORCH_CHECK(W.size(1) == 4096 && x.size(0) == 1 && N % 4 == 0 && N <= W.size(0), "sgemv: m = 1, K = 4096");
  sgemv_kernel<4, 512, 4096><<<(int)(N / 4), 512, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rgemv(at::Tensor W, at::Tensor x, at::Tensor out, int64_t M) {
  TORCH_CHECK(W.size(1) == 8192 && M == 1, "rgemv: K = 8192, m = 1");
  rgemv_kernel<1, 1><<<(int)W.size(0), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      W.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), out.data_ptr<float>(), (int)M,
      (int)W.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


    def _gq_src(sm90):
        lines = []
        for (layer, bn), (nwg, cl, ns) in sorted(GQ_CONFIGS.items()):
            lines.append(f"if (bn == {bn} && nwg == {nwg} && cl == {cl} && nstage == {ns}) "
                         f"{{ launch_gq<{bn}, {nwg}, {ns}, {cl}>(W, x, out, M, N, K); return; }}")
        return GQ_SRC.replace("@@GQ_DISPATCH@@", "\n  ".join(dict.fromkeys(lines)))


    def _gq_bn(m):
        return 64 if m <= 64 else 128 if m <= 128 else 256


    def _run_gq(x, layer, m, n, k, state):
        return _run_gq_family(x, layer, m, n, k, state, GQ_CONFIGS, _gq_bn, "gq")



    WIDE_CONFIGS = {  # (layer, tok, nz) -> (nwg, splits, stages); grid = (N / (64 nwg), splits, nz) <= 132 (co-resident)
        ("pooler", 64, 1): (1, 2, 8), ("unpooler", 64, 1): (1, 1, 8), ("qkv_gate", 64, 1): (1, 3, 8), ("o", 64, 1): (1, 1, 8),
        ("pooler", 128, 1): (1, 2, 7), ("unpooler", 128, 1): (1, 1, 7), ("qkv_gate", 128, 1): (1, 3, 7), ("o", 128, 1): (1, 1, 7),
        ("pooler", 128, 2): (2, 1, 5), ("unpooler", 128, 2): (4, 1, 4), ("qkv_gate", 128, 2): (1, 1, 7), ("o", 128, 2): (4, 1, 4),
        # router: BF16 hi|lo tiles (plain), 64-row blocks
        ("router", 64, 1): (1, 32, 4), ("router", 128, 1): (1, 32, 4), ("router", 128, 2): (1, 16, 4),
    }


    def _wide_key(m):
        return (64, 1) if m <= 64 else (128, 1) if m <= 128 else (128, 2)


    def _wide_src(sm90):
        lines = []
        for (layer, tok, nz), (nwg, splits, stages) in sorted(WIDE_CONFIGS.items()):
            hl = layer == "router"
            ot = "float" if hl else "__nv_bfloat16"
            lines.append(f"if (nwg == {nwg} && tok == {tok} && hl == {int(hl)}) "
                         f"{{ launch_wide<{nwg}, {tok}, {stages}, {str(hl).lower()}, {ot}>(Wst, x, out, ws, counters, ptab, pmax, base4, "
                         f"M, N, K, splits); return; }}")
        return WIDE_SRC.replace("@@WIDE_DISPATCH@@", "\n  ".join(dict.fromkeys(lines)))


    RK_SRC = r"""
// ============================================================================ rk: router GEMM, m 33..256 (sm_90)
// out[M, 256] (FP32) = x[M, 8192] (BF16) @ (hi + lo)[256, 8192]^T on router_cat = [hi | lo] [256, 16384] (BF16).
// CTA (192 threads) = 1 consumer warpgroup (SS-wgmma m64n64k16 on the hi tile, then the lo tile, of each 64-k block)
// + 1 producer warp (TMA, RK_QS-stage ring of 24 KB: hi 8 KB | lo 8 KB | x 8 KB, 128B swizzle) + 1 helper warp.
// Tile = 64 W rows x 64 tokens; split-K over S CTAs per tile (grid 4 x ntt x S <= 128: one CTA per SM, all resident).
// Each CTA ADDS its partial tile into the FP32 output (smem-transposed, 16 B red.add.v4.f32): the reduction is
// ORDER-DEPENDENT (run-to-run differences ~1e-7 relative, inside the router tolerance; POD item 6). The output is
// cleared inside the kernel: every CTA's helper warp reads the grid counter of this grid size (generation =
// value / grid: earlier launches' counts are complete and this CTA has not counted yet), clears its slice of out,
// counts it (release), waits (acquire, 2 s limit -> trap) until every CTA has counted, and releases the consumers'
// adds through a CTA mbarrier. Monotonic counters (one per token-tile count = per grid size): no host reset.
#define RK_QS 6
__device__ __forceinline__ uint64_t rk_desc(uint32_t addr) {   // K-major tile, 128 B rows, 128B swizzle, SBO 1024
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)64 << 32) | (1ull << 62);
}
__device__ __forceinline__ void rk_mbar_init(uint32_t bar, uint32_t cnt) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(cnt)); }
__device__ __forceinline__ void rk_mbar_expect(uint32_t bar, uint32_t bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory"); }
__device__ __forceinline__ void rk_mbar_arrive(uint32_t bar) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar) : "memory"); }
__device__ __forceinline__ void rk_mbar_wait(uint32_t bar, uint32_t ph) {
  asm volatile("{\n.reg .pred P1;\nRKW_%=:\nmbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n@!P1 bra RKW_%=;\n}\n" :: "r"(bar), "r"(ph) : "memory");
}
__device__ __forceinline__ void rk_tma2d(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void rk_wgmma(float d[4][8], uint64_t da, uint64_t db) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n"
               : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7])
               : "l"(da), "l"(db));
}

__global__ void __launch_bounds__(192, 1) rk_kernel(const __grid_constant__ CUtensorMap mapW, const __grid_constant__ CUtensorMap mapX,
                                                     float* __restrict__ out, int M, int S, int ntt, int* __restrict__ zcnt) {
  constexpr int BM = 64, BN = 64, NJ = 4, QS = RK_QS;
  constexpr int W_BYTES = 2 * BM * 128, ST = W_BYTES + BN * 128;
  extern __shared__ __align__(1024) unsigned char rk_smem[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(rk_smem);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  const uint32_t fullb = sbase + QS * ST, emptyb = fullb + 8 * QS, zbar = emptyb + 8 * QS;
  const int rt = blockIdx.x & 3, tt = (blockIdx.x >> 2) % ntt, sp = (blockIdx.x >> 2) / ntt;
  const int kb0 = (sp * 128) / S, nk = ((sp + 1) * 128) / S - kb0;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapW) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapX) : "memory");
    for (int i = 0; i < QS; ++i) { rk_mbar_init(fullb + 8 * i, 1); rk_mbar_init(emptyb + 8 * i, 1); }
    rk_mbar_init(zbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp == 5) {   // helper: clear this CTA's slice of out, count it, wait for every slice, release the adds
    int target = 0;
    if (lane == 0) {
      int v;
      asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(zcnt) : "memory");
      target = (v / (int)gridDim.x + 1) * (int)gridDim.x;
    }
    const int n4 = M * 64;   // float4s in out
    const int per = (n4 + (int)gridDim.x - 1) / (int)gridDim.x;
    const int z0 = blockIdx.x * per, z1 = min(n4, z0 + per);
    for (int i = z0 + lane; i < z1; i += 32) reinterpret_cast<float4*>(out)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    __syncwarp();
    if (lane == 0) {
      asm volatile("red.release.gpu.global.add.s32 [%0], 1;" :: "l"(zcnt) : "memory");
      unsigned long long t0, t1;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
      for (;;) {
        int v;
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(zcnt) : "memory");
        if (v - target >= 0) break;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
        if (t1 - t0 > 2000000000ull) __trap();   // 2 s: a clear that never lands is a bug, never a slow node
        __nanosleep(128);
      }
      rk_mbar_arrive(zbar);
    }
    return;
  }
  if (warp == 4) {   // producer
    if (lane == 0) {
      for (int i = 0; i < nk; ++i) {
        const int s = i % QS;
        if (i >= QS) rk_mbar_wait(emptyb + 8 * s, ((i / QS) - 1) & 1);
        const uint32_t fb = fullb + 8 * s, st = sbase + s * ST;
        rk_mbar_expect(fb, ST);
        const int kb = kb0 + i;
        rk_tma2d(st, &mapW, fb, kb * 64, rt * BM);
        rk_tma2d(st + BM * 128, &mapW, fb, 8192 + kb * 64, rt * BM);
        rk_tma2d(st + W_BYTES, &mapX, fb, kb * 64, tt * BN);
      }
    }
    return;
  }
  // consumer warpgroup
  float d[NJ][8];
#pragma unroll
  for (int j = 0; j < NJ; ++j)
#pragma unroll
    for (int e = 0; e < 8; ++e) d[j][e] = 0.f;
  for (int i = 0; i < nk; ++i) {
    const int s = i % QS;
    rk_mbar_wait(fullb + 8 * s, (i / QS) & 1);
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    const uint32_t a = sbase + s * ST, b = a + W_BYTES;
#pragma unroll
    for (int k16 = 0; k16 < 4; ++k16) {
      rk_wgmma(d, rk_desc(a + 32 * k16), rk_desc(b + 32 * k16));
      rk_wgmma(d, rk_desc(a + BM * 128 + 32 * k16), rk_desc(b + 32 * k16));
    }
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
    if (i > 0 && tid == 0) rk_mbar_arrive(emptyb + 8 * ((i - 1) % QS));   // stage i-1's wgmmas are done
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
#pragma unroll
  for (int j = 0; j < NJ; ++j)
#pragma unroll
    for (int e = 0; e < 8; ++e) asm volatile("" : "+f"(d[j][e]) :: "memory");
  // partial tile -> shared memory [token][row] (the ring is idle: every stage was waited for), then 16 B adds
  constexpr int TP = 68;
  float* T = reinterpret_cast<float*>(rk_smem + (sbase - sraw));
  const int rl = (tid >> 5) * 16 + ((tid & 31) >> 2);
#pragma unroll
  for (int j = 0; j < NJ; ++j)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int cl = j * 16 + h * 8 + (tid & 3) * 2;
      T[cl * TP + rl] = d[j][4 * h]; T[cl * TP + rl + 8] = d[j][4 * h + 2];
      T[(cl + 1) * TP + rl] = d[j][4 * h + 1]; T[(cl + 1) * TP + rl + 8] = d[j][4 * h + 3];
    }
  asm volatile("bar.sync 1, 128;" ::: "memory");
  rk_mbar_wait(zbar, 0);   // every CTA's slice of out is clear (acquired by the helper warp)
  for (int idx = tid; idx < BN * 16; idx += 128) {
    const int tk = idx >> 4, qd = idx & 15;
    const int c = tt * BN + tk;
    if (c < M) {
      const float4 v = *reinterpret_cast<const float4*>(T + tk * TP + qd * 4);
      asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
                   :: "l"(out + (size_t)c * 256 + rt * BM + qd * 4), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
    }
  }
}
static PFN_cuTensorMapEncodeTiled_v12000 rk_encode() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult qr; void* p = nullptr;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &qr);
    TORCH_CHECK(p != nullptr, "rk: cuTensorMapEncodeTiled not found");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }
  return fn;
}
static CUtensorMap rk_map(const void* ptr, uint64_t rows, uint64_t cols) {
  CUtensorMap m;
  cuuint64_t dims[2] = {cols, rows}; cuuint64_t strides[1] = {cols * 2};
  cuuint32_t box[2] = {64, 64}; cuuint32_t estr[2] = {1, 1};
  CUresult r = rk_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides, box, estr,
                           CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                           CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "rk: cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}
void rk(at::Tensor Wc, at::Tensor x, at::Tensor out, int64_t S, at::Tensor zc) {
  const int M = x.size(0);
  TORCH_CHECK(Wc.size(0) == 256 && Wc.size(1) == 16384 && x.size(1) == 8192 && M >= 1 && M <= 256, "rk: shape");
  const int ntt = (M + 63) / 64;
  const int grid = 4 * ntt * (int)S;
  static int nsm = 0;
  if (!nsm) { int dev = 0; C10_CUDA_CHECK(cudaGetDevice(&dev)); C10_CUDA_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev)); }
  TORCH_CHECK(grid <= nsm, "rk: grid ", grid, " exceeds the SM count ", nsm, " (every CTA must be resident: one per SM)");
  const CUtensorMap mw = rk_map(Wc.data_ptr(), 256, 16384), mx = rk_map(x.data_ptr(), M, 8192);
  const int smem = 1024 + RK_QS * 24576 + 8 * (2 * RK_QS + 1) + 64;
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(rk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)); attr = true; }
  rk_kernel<<<grid, 192, smem, at::cuda::getCurrentCUDAStream()>>>(mw, mx, out.data_ptr<float>(), M, (int)S, ntt, zc.data_ptr<int>() + ntt);
  C10_CUDA_CHECK(cudaGetLastError());
}
"""


    def _rk_src(sm90):
        if sm90:
            return RK_SRC


    # ---------------------------------------------------------------------------
    # Extension build: everything above is assembled into one CUDA translation
    # ---------------------------------------------------------------------------
    def _get_ext():
        """Compile everything as one CUDA translation unit with load_inline.

    no_implicit_headers: the CUDA file includes only the ATen headers it needs
    (torch/types.h alone costs ~45 s of nvcc time).
    """
        nonlocal _ext
        if _ext is None:
            major, minor = torch.cuda.get_device_capability()
            sm90 = major >= 9
            arch = f"{major}{minor}a" if sm90 else f"{major}{minor}"
            skinny = CUDA_SRC.replace("@@MBAR@@", MBAR_HELPERS).replace("@@BODY@@", WG_HELPERS if sm90 else "")
            rs_defs = wgmma_rs_defs([8, 16, 32, 64, 128]) if sm90 else ""
            # Order matters only in that each route's helpers precede its kernel;
            # every route is self-contained apart from the shared helpers above.
            cuda_src = (skinny + rs_defs + _tma_srcs(sm90) + MAP_SRC + _gemm_src(sm90)
                        + _wide_src(sm90) + _sk2_src(sm90) + _gq_src(sm90) + _gqx_src(sm90)
                        + RGEMV_SRC + _rk_src(sm90))
            _ext = load_inline(
                name=f"okc_bf16proj_v141_{arch}" + (f"_dly{_SK2_DELAY}" if _SK2_DELAY else ""),
                cpp_sources="#include <torch/extension.h>\n" + CPP_SRC,
                cuda_sources=cuda_src,
                functions=["skinny_tma_a", "gemm90", "wide", "sk2", "gq", "gqx", "rgemv", "sgemv", "cgemv", "rk"],
                extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"] + ([f"-DSK2_DELAY_ESC={_SK2_DELAY}"] if _SK2_DELAY else []),
                no_implicit_headers=True,
                verbose=False,
            )
        return _ext


    # ---------------------------------------------------------------------------------------------- CUTLASS route (m 129..256)
    # CUTLASS 3.x sm90 TMA warp-specialized PINGPONG GEMM, persistent tile scheduler, swap-AB (D^T = W x^T: A = W [N, K]
    # row-major, B = x [M, K] as a K x M column-major matrix, D column-major [N, M] = out [M, N] row-major), BF16 in/out,
    # FP32 accumulate, tile 128 (W rows) x 64 (tokens) x 64, cluster 1x1. Headers: CUTLASS bundled with flashinfer (in the
    # SGLang image). Sweep (pod/cutsweep.py, 54 configs x 8 shapes): it beats cuBLAS only for pooler m=256 (31.3 vs 34.5 us
    # raw); a missing header or a failed build just leaves the route out.
    CUTLASS_LAYERS = set() if _PRUNE else {"pooler"}   # layers that get the CUTLASS candidate (scored m=256 only)
    CUTLASS_BUILD_TIMEOUT = 170   # seconds prepare waits for the CUTLASS build (main build ~25 s; prepare's limit is 300 s)
    CUTLASS_SRC = r"""
/***************************************************************************************************
 * The GEMM below instantiates NVIDIA CUTLASS 3.x/4.x templates (header-only, compiled from the CUTLASS headers that
 * ship in this image's FlashInfer package; nothing prebuilt is linked). Its type composition follows CUTLASS's Hopper
 * GEMM examples (examples/48_hopper_warp_specialized_gemm, 49_hopper_gemm_with_collective_builder), so it carries
 * CUTLASS's license notice:
 *
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************************************/
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>
using namespace cute;
using EA = cutlass::bfloat16_t; using LA = cutlass::layout::RowMajor;
using EB = cutlass::bfloat16_t; using LB = cutlass::layout::ColumnMajor;
using EC = cutlass::bfloat16_t; using LC = cutlass::layout::ColumnMajor;
using TileShape = Shape<_128, _64, _64>;
using ClusterShape = Shape<_1, _1, _1>;
using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, float, float, EC, LC, 8, EC, LC, 8,
    cutlass::epilogue::TmaWarpSpecialized>::CollectiveOp;
using Main = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, EA, LA, 8, EB, LB, 8, float, TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epi::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Main, Epi, cutlass::gemm::PersistentScheduler>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
int64_t cut_ws(at::Tensor W, at::Tensor x, at::Tensor out) {
  const int M = W.size(0), K = W.size(1), N = x.size(0);
  auto sA = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, cute::make_shape(M, K, 1));
  auto sB = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, cute::make_shape(N, K, 1));
  auto sC = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideC{}, cute::make_shape(M, N, 1));
  typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
      {reinterpret_cast<EA*>(W.data_ptr()), sA, reinterpret_cast<EB*>(x.data_ptr()), sB},
      {{1.f, 0.f}, reinterpret_cast<EC*>(out.data_ptr()), sC, reinterpret_cast<EC*>(out.data_ptr()), sC}};
  return (int64_t)Gemm::get_workspace_size(args);
}
void cut_gemm(at::Tensor W, at::Tensor x, at::Tensor out, at::Tensor ws) {
  const int M = W.size(0), K = W.size(1), N = x.size(0);
  auto sA = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, cute::make_shape(M, K, 1));
  auto sB = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, cute::make_shape(N, K, 1));
  auto sC = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideC{}, cute::make_shape(M, N, 1));
  typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
      {reinterpret_cast<EA*>(W.data_ptr()), sA, reinterpret_cast<EB*>(x.data_ptr()), sB},
      {{1.f, 0.f}, reinterpret_cast<EC*>(out.data_ptr()), sC, reinterpret_cast<EC*>(out.data_ptr()), sC}};
  Gemm gemm;
  TORCH_CHECK((size_t)ws.numel() >= Gemm::get_workspace_size(args), "cut_gemm: workspace");
  auto st = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess, "cut_gemm: cannot implement");
  TORCH_CHECK(gemm.initialize(args, ws.data_ptr(), st) == cutlass::Status::kSuccess, "cut_gemm: init");
  TORCH_CHECK(gemm.run(st) == cutlass::Status::kSuccess, "cut_gemm: run");
}
"""
    _cut_ext = None
    _cut_err = None


    def _build_cutlass():
        """Build the CUTLASS extension (sm_90 only); on any failure record it and leave the route out."""
        nonlocal _cut_ext, _cut_err
        try:
            major, minor = torch.cuda.get_device_capability()
            spec = importlib.util.find_spec("flashinfer")
            if spec is None or not spec.submodule_search_locations:
                raise RuntimeError("flashinfer (CUTLASS headers) not installed")
            root = os.path.join(list(spec.submodule_search_locations)[0], "data", "cutlass")
            inc = [os.path.join(root, "include"), os.path.join(root, "tools", "util", "include")]
            if not all(os.path.isdir(d) for d in inc):
                raise RuntimeError("CUTLASS headers missing under " + root)
            _cut_ext = load_inline(
                name=f"okc_bf16proj_v141_cutlass_{major}{minor}a",
                cpp_sources="#include <torch/extension.h>\nint64_t cut_ws(at::Tensor W, at::Tensor x, at::Tensor out);\n"
                            "void cut_gemm(at::Tensor W, at::Tensor x, at::Tensor out, at::Tensor ws);",
                cuda_sources=CUTLASS_SRC, functions=["cut_ws", "cut_gemm"], extra_include_paths=inc,
                extra_cuda_cflags=["-O3", "-std=c++20", "--expt-relaxed-constexpr", f"-gencode=arch=compute_{major}{minor}a,code=sm_{major}{minor}a", "-DNDEBUG"],
                verbose=False)
        except Exception as e:   # noqa: BLE001 - any failure only drops this optional route
            _cut_ext, _cut_err = None, f"{type(e).__name__}: {str(e)[:200]}"


    # ---------------------------------------------------------------------------
    # Weight preparation (host side, runs in prepare()): fragment orders and the
    # lossless 12-bit packing. The packing helpers share _exponent_window /
    # _escapes; the per-kernel stream/table builders stay separate because their
    # layouts and orderings differ (per-CTA streams vs k-ordered escape tables).
    # ---------------------------------------------------------------------------
    def _exponent_window(e71):
        """Best EXPONENT_WINDOW-wide window of BF16 exponent bits 7..1 for a matrix.

    Returns (base, offset): ``base`` is the window's lowest exponent, and
    ``offset`` is each weight's exponent clamped into [0, EXPONENT_WINDOW).
    The window is chosen to cover the most weights, so the fewest weights
    need exact correction as escapes.
    """
        hist = torch.bincount(e71.flatten(), minlength=EXPONENT_VALUES)
        win = torch.stack(
            [hist[b:b + EXPONENT_WINDOW].sum() for b in range(EXPONENT_VALUES - EXPONENT_WINDOW + 1)]
        )
        base = int(win.argmax())
        return base, (e71 - base).clamp(0, EXPONENT_WINDOW - 1)


    def _escapes(e71, base):
        """Weights whose exponent falls outside [base, base + EXPONENT_WINDOW)."""
        return (e71 < base) | (e71 > base + EXPONENT_WINDOW - 1)


    def pack_fragments(w: torch.Tensor) -> torch.Tensor:
        """W [N, K] bf16 -> [N/16, K/16, 32 lanes, 8] in mma.m16n8k16 A-fragment order."""
        n, k = w.shape
        t = w.view(n // 16, 2, 8, k // 16, 2, 4, 2)  # [rg, hr, g, s, hk, t, e]: row = hr*8+g, col = hk*8+2t+e
        t = t.permute(0, 3, 2, 5, 4, 1, 6)  # [rg, s, g, t, hk, hr, e]
        return t.contiguous().view(n // 16, k // 16, 32, 8)


    def pack_compressed(w: torch.Tensor):
        """Lossless 12-bit packing of BF16 W [N, K].  Returns None if the matrix doesn't suit it."""
        n, k = w.shape
        if n % 16 or k % 64:
            return None
        bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
        hi = bits >> 8                  # sign | exponent bits 7..1
        e71 = hi & 0x7F
        base, o = _exponent_window(e71)
        code = ((hi >> 7) << 3) | o
        dec_hi = ((hi >> 7) << 7) | (base + o)
        dec = ((dec_hi << 8) | (bits & 0xFF)).to(torch.int16).view(torch.bfloat16)
        esc = _escapes(e71, base)
        n_esc = int(esc.sum())
        if n_esc > n * k // MAX_ESCAPE_DENOMINATOR:
            return None
        rows, cols = esc.nonzero(as_tuple=True)
        wtrue = w[rows, cols].float()
        wdec = dec[rows, cols].float()
        delta = wtrue - wdec
        if not bool(((wdec + delta) == wtrue).all()):
            return None
        # fragment order for low bytes and codes: [rg, s, lane, 8]
        lo_f = pack_fragments((bits & 0xFF).to(torch.int16).view(torch.bfloat16)).view(torch.int16) & 0xFF
        cd_f = pack_fragments(code.to(torch.int16).view(torch.bfloat16)).view(torch.int16) & 0xF
        nrg, S = n // 16, k // 16
        lo_f = lo_f.view(nrg, S // 4, 4, 32, 8).to(torch.uint8)   # [rg, u, ks, lane, 8]
        cd_f = cd_f.view(nrg, S // 4, 4, 32, 8).to(torch.uint8)
        # parts 0,1: lane's low bytes for k-steps (0,1) and (2,3) -> [rg, u, 2, lane, 16]
        lo_p = lo_f.view(nrg, S // 4, 2, 2, 32, 8).permute(0, 1, 2, 4, 3, 5).reshape(nrg, S // 4, 2, 32, 16)
        # part 2: word i (k-step i): byte b = code[8i + b] | code[8i + 4 + b] << 4
        cw = (cd_f[..., 0:4] | (cd_f[..., 4:8] << 4))           # [rg, u, ks, lane, 4]
        cw = cw.permute(0, 1, 3, 2, 4).reshape(nrg, S // 4, 1, 32, 16)
        packed = torch.cat([lo_p, cw], dim=2).transpose(0, 1).contiguous()  # [u, rg, 3, lane, 16]
        order = torch.argsort(rows * k + cols)
        rows, cols, delta = rows[order], cols[order], delta[order]
        rg_idx = rows // 16
        poff = torch.zeros(nrg + 1, dtype=torch.int32, device=w.device)
        poff[1:] = torch.cumsum(torch.bincount(rg_idx, minlength=nrg), 0).to(torch.int32)
        return {
            "packed": packed,
            "base4": base * BASE4_REPLICATE,
            "poff": poff,
            "pk": cols.to(torch.int32).contiguous(),
            "pr": (rows % 16).to(torch.int32).contiguous(),
            "pd": delta.to(torch.float32).contiguous(),
            "n_esc": n_esc,
        }


    def build_patch_table(c, n_cfg, k, splits, rg):
        """Per-CTA exact-correction table for a (splits, RG) decomposition over the first n_cfg rows:
    int2 [num_ctas][pmax]; entry 0 = (count, 0); entries = ((row_local << 16) | k_local, delta bits)."""
        dev = c["pk"].device
        rows_cta = rg * 16
        nrb = n_cfg // rows_cta
        nunits = k // 64
        poff = c["poff"].long()
        nrg_all = poff.numel() - 1
        rgi = torch.repeat_interleave(torch.arange(nrg_all, device=dev), poff[1:] - poff[:-1])
        rows = rgi * 16 + c["pr"].long()
        cols = c["pk"].long()
        keep = rows < n_cfg
        rows, cols, delta = rows[keep], cols[keep], c["pd"][keep]
        unit = cols // 64
        spl = ((unit + 1) * splits + nunits - 1) // nunits - 1          # split owning this 64-k unit
        k_start = (spl * nunits // splits) * 64
        cta = spl * nrb + rows // rows_cta
        ncta = nrb * splits
        counts = torch.bincount(cta, minlength=ncta)
        pmax = int(counts.max().item()) + 1 if rows.numel() else 1
        pmax += pmax % 2
        order = torch.argsort(cta, stable=True)
        cta, rows, cols, delta, k_start = cta[order], rows[order], cols[order], delta[order], k_start[order]
        start = torch.zeros(ncta + 1, dtype=torch.long, device=dev)
        start[1:] = torch.cumsum(counts, 0)
        slot = torch.arange(cta.numel(), device=dev) - start[cta] + 1
        tab = torch.zeros(ncta, pmax, 2, dtype=torch.int32, device=dev)
        tab[:, 0, 0] = counts.to(torch.int32)
        tab[cta, slot, 0] = (((rows % rows_cta) << 16) | (cols - k_start)).to(torch.int32)
        tab[cta, slot, 1] = delta.contiguous().view(torch.int32)
        return tab.contiguous(), pmax


    def build_stream(src, comp, n_cfg, k, splits, rg, wsn, ks):
        """Per-CTA contiguous weight stream for a (splits, RG, WS, KS) decomposition of the first n_cfg rows.
    WS > 1: [cta = sp * nrb + rb][stage][warp][chunk].  WS == 1: [rb][stage over all K][warp][chunk] (independent of
    splits; a CTA reads its contiguous stage range).  src: compressed [nrg, K/64, 3, 32, 16] uint8 or uncompressed
    fragments [nrg, K/16, 32, 8] bf16."""
        dev = src.device
        nw = rg * wsn
        nrb = (n_cfg // 16) // rg
        ar = lambda n: torch.arange(n, device=dev)
        if wsn == 1:
            s_all = (k // 16) // ks
            rb = ar(nrb).view(-1, 1, 1, 1)
            st = ar(s_all).view(1, -1, 1, 1)
            w = ar(nw).view(1, 1, -1, 1)
            rg_idx = rb * rg + w
            if comp:
                u_idx = st * (ks // 4) + ar(ks // 4).view(1, 1, 1, -1)
                return src[rg_idx, u_idx].contiguous()
            s_idx = st * ks + ar(ks).view(1, 1, 1, -1)
            return src[rg_idx, s_idx].contiguous()
        kc = k // splits
        ns = kc // (16 * wsn)
        nst = ns // ks
        sp = ar(splits).view(-1, 1, 1, 1, 1)
        rb = ar(nrb).view(1, -1, 1, 1, 1)
        st = ar(nst).view(1, 1, -1, 1, 1)
        w = ar(nw).view(1, 1, 1, -1, 1)
        rg_idx = rb * rg + w % rg
        if comp:
            ku = ar(ks // 4).view(1, 1, 1, 1, -1)
            u_idx = sp * (kc // 64) + (w // rg) * (ns // 4) + st * (ks // 4) + ku
            out = src[rg_idx, u_idx]
        else:
            j = ar(ks).view(1, 1, 1, 1, -1)
            s_idx = sp * (kc // 16) + (w // rg) * ns + st * ks + j
            out = src[rg_idx, s_idx]
        return out.contiguous()


    # ---------------------------------------------------------------- sk2: m <= 32, swapped operands, no cross-CTA reduction
    SK2_CONFIGS = [  # (name, layer, n_rows, R rows/CTA, NT, NWR, WSK, UPW, NACC, CL); a name ending in _rNNN caps the ring at NNN KB
        ('o32h_r100', 'o', 8192, 32, 2, 2, 2, 2, 1, 1),
        ('o8h_r88', 'o', 8192, 32, 4, 1, 4, 1, 2, 1),
        ('pooler32_r144', 'pooler', 4096, 32, 2, 2, 2, 2, 1, 1),
        ('pooler8h_r72', 'pooler', 4096, 16, 1, 2, 2, 4, 1, 1),
        ('qkv32_r208', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 1, 1),
        ('router', 'router', 256, 8, 1, 1, 8, 1, 4, 3),
        ('slice8_r192', 'unpooler', UNPOOLER_SLICE_ROWS, 8, 1, 1, 8, 2, 2, 1),
        ('unpooler32_r144', 'unpooler', 8192, 64, 4, 2, 2, 2, 1, 1),
        # okc_tune (14173ac4) retimed winners, Oct 25 12:30 (search-harness ratios vs production in brackets); pair.py both
        # orders: pooler16 -0.43, unpooler16 -0.60, o16 -0.15, qkv8 -0.12, qkv16 -0.08 us (pooler32 r160 / o32 r88: no gain)
        ('pooler16t_r192', 'pooler', 4096, 32, 2, 2, 2, 4, 1, 1),      # tokr16 [0.973 vs 1.0]
        ('unpooler16t_r72', 'unpooler', 8192, 32, 2, 2, 2, 2, 1, 1),   # tokr16 [0.961 vs 0.969]
        ('qkv8t_r144', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 2, 1),         # tokr8 [0.969 vs 0.979]
        ('qkv16t_r200', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 2, 1),        # tokr16 [0.985 vs 0.994]
        ('o16t_r68', 'o', 8192, 32, 2, 2, 1, 4, 2, 1),                  # tokr16 [0.937 vs 0.957]
    ]
    SK2_PICK = {  # (layer, tokr) -> config name
        ('o', 8): 'o8h_r88',
        ('o', 16): 'o16t_r68',
        ('o', 32): 'o32h_r100',
        ('pooler', 8): 'pooler8h_r72',
        ('pooler', 16): 'pooler16t_r192',
        ('pooler', 32): 'pooler32_r144',
        ('qkv_gate', 8): 'qkv8t_r144',
        ('qkv_gate', 16): 'qkv16t_r200',
        ('qkv_gate', 32): 'qkv32_r208',
        ('router', 8): 'router',
        ('router', 16): 'router',
        ('router', 32): 'router',
        ('unpooler', 8): 'slice8_r192',
        ('unpooler', 16): 'unpooler16t_r72',
        ('unpooler', 32): 'unpooler32_r144',
    }
    SK2_K = {"pooler": 8192, "unpooler": 4096, "o": 1024, "qkv_gate": 8192, "router": 8192}
    USE_SK2 = True
    WIDE_BF16 = False  # the wide (RS-wgmma) kernel never beats cuBLAS for BF16 m 33..256


    def _sk2_name(layer, m):
        return SK2_PICK[(layer, _sk2_tokr(m))]


    def _sk2_tokr(m):
        return 8 if m <= 8 else 16 if m <= 16 else 32


    def _sk2_geom(cfg, tokr):
        name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
        hl = layer == "router"
        tu = 2048 if hl else 768
        ups = WSK * UPW
        wby = NWR * WSK * UPW * NT * tu
        xby = ups * tokr * 128
        sst = (xby + wby + 1023) // 1024 * 1024
        nst = -(-(SK2_K[layer] // (ups * 64)) // CL)   # (the largest rank's stages)
        mt = 2 if tokr == 32 else 1
        toks = 16 * mt
        cp = R + 1
        nesc = 4 if not hl else 1   # (3 when the ring has 3 stages; sized for 4 here)
        rp = R + ((40 - (R % 32)) % 32)
        # everything but the ring: alignment pad, barriers, corrections, peer sums, escape table (<= 64 entries), zero row
        extra = (1024 + 16 * 16 + 32 + nesc * ((toks * cp + 3) // 4 * 4) * 4 + ((CL - 1) * WSK * toks * rp * 4 if CL > 1 else 0)
                 + 64 * 8 + 128)
        cap_kb = SK2_DEFAULT_RING_CAP_KB
        tail = name.rsplit("_r", 1)
        if len(tail) == 2 and tail[1].isdigit():
            cap_kb = int(tail[1])
        ring = min(cap_kb * 1024, SK2_RING_SMEM_BUDGET_BYTES - extra)
        nstage = max(SK2_MIN_STAGES, min(SK2_MAX_STAGES, nst, ring // sst))
        return hl, nstage


    def _sk2_src(sm90):
        lines = []
        for ci, cfg in enumerate(SK2_CONFIGS):
            name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
            for tokr in (8, 16, 32):
                if SK2_PICK.get((layer, tokr)) != name:
                    continue
                hl, nstage = _sk2_geom(cfg, tokr)
                mt = 2 if tokr == 32 else 1
                ot = "float" if hl else "__nv_bfloat16"
                lines.append(f"if (cfg == {ci} && tokr == {tokr}) {{ launch_sk2<{mt}, {tokr}, {NT}, {NWR}, {WSK}, {UPW}, {nstage}, "
                             f"{str(not hl).lower()}, {str(hl).lower()}, {1 if (mt == 2 and not hl and nstage >= 4 and NT == 4) else NACC}, {CL}, {ot}>(Ws, x, out, ptab, pmax, base4, M, N, K, dg, dbg); return; }}")
        return SK2_SRC.replace("@@SK2_DISPATCH@@", "\n  ".join(lines))


    def sk2_tileunits(w):
        """W [N, K] -> [N/8, K/64, 32 lanes, 16] in mma.m16n8k16 B-fragment order: lane (g, t) holds row g and, for each
    k-step j of the 64-k unit, k = 16j + 2t, 2t+1, 2t+8, 2t+9."""
        n, k = w.shape
        t = w.view(n // 8, 8, k // 64, 4, 2, 4, 2)          # [tile, g, unit, j, h, t, p]: k = 64u + 16j + 8h + 2t + p
        return t.permute(0, 2, 1, 5, 3, 4, 6).reshape(n // 8, k // 64, 32, 16)


    def sk2_compress(w):
        """Lossless 12-bit packing of BF16 W [N, K] into B-fragment tile-units [N/8, K/64, 768] (512 B low bytes, lane-major,
    then 256 B of 4-bit codes) + escapes (row, k, delta) that the kernel corrects exactly.  Returns None if unsuitable."""
        n, k = w.shape
        tu = sk2_tileunits(w)
        bits = tu.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
        hi = bits >> 8
        low = bits & 0xFF
        e71 = hi & 0x7F
        sign = hi >> 7
        base, o = _exponent_window(e71)
        code = (sign << 3) | o
        esc = _escapes(e71, base)
        if int(esc.sum()) > n * k // MAX_ESCAPE_DENOMINATOR:
            return None
        T, U = n // 8, k // 64
        c8 = code.to(torch.uint8)
        w0 = c8[..., 0:4] | (c8[..., 4:8] << 4)
        w1 = c8[..., 8:12] | (c8[..., 12:16] << 4)
        codes = torch.cat([w0, w1], dim=-1)                  # [T, U, 32, 8]
        packed = torch.cat([low.to(torch.uint8).reshape(T, U, 512), codes.reshape(T, U, 256)], dim=-1).contiguous()
        idx = esc.nonzero()                                   # [E, 4]: tile, unit, lane, v
        dec_bits = ((((sign << 7) | (base + o)) << 8) | low)[esc]
        wdec = dec_bits.to(torch.int16).view(torch.bfloat16).float()
        wtrue = tu[esc].float()
        delta = wtrue - wdec
        if not bool(((wdec + delta) == wtrue).all()):
            return None
        lane, v = idx[:, 2], idx[:, 3]
        rows = idx[:, 0] * 8 + (lane >> 2)
        cols = idx[:, 1] * 64 + 16 * (v >> 2) + 8 * ((v >> 1) & 1) + 2 * (lane & 3) + (v & 1)
        wbits = (tu[esc].contiguous().view(torch.int16).to(torch.int32) & 0xFFFF).contiguous()   # the original BF16 weights
        return {"packed": packed, "base4": base * BASE4_REPLICATE, "rows": rows, "cols": cols, "wbits": wbits}


    CG_CONFIGS = {"qkv_gate": (2, 256), "o": (2, 64), "unpooler": (2, 128), "pooler": (2, 512)}   # layer -> (rows per CTA, threads); m = 1
    # (v137: pooler m=1 cgemv 2x512 measured -0.34 us vs sk2; 2x256 ties sk2, 4/8 rows or 1 row lose)


    def cg_compress(w, rows_per_cta):
        """Row-major 12-bit planes for cgemv (lossless; escapes corrected exactly from a per-CTA table)."""
        n, k = w.shape
        bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
        hi, low = bits >> 8, bits & 0xFF
        e71, sign = hi & 0x7F, hi >> 7
        base, o = _exponent_window(e71)
        code = ((sign << 3) | o).to(torch.uint8)
        esc = _escapes(e71, base)
        if int(esc.sum()) > n * k // MAX_ESCAPE_DENOMINATOR:
            return None
        g = code.view(n, k // 8, 8)
        C = (g[:, :, 0:4] | (g[:, :, 4:8] << 4)).reshape(n, k // 2).contiguous()
        L = low.to(torch.uint8).contiguous()
        rows, cols = esc.nonzero(as_tuple=True)          # row-major order
        dec = ((((sign[rows, cols] << 7) | (base + o[rows, cols])) << 8) | low[rows, cols]).to(torch.int16).view(torch.bfloat16).float()
        wtrue = w[rows, cols].float()
        delta = (wtrue - dec).contiguous()
        if not bool(((dec + delta) == wtrue).all()):
            return None
        ncta = n // rows_per_cta
        cta = rows // rows_per_cta
        counts = torch.bincount(cta, minlength=ncta)
        pmax = int(counts.max()) + 1 if rows.numel() else 1
        start = torch.cumsum(counts, 0) - counts
        pos = torch.arange(rows.numel(), device=w.device) - start[cta] + 1
        tab = torch.zeros(ncta, pmax, 2, dtype=torch.int32, device=w.device)
        tab[:, 0, 0] = counts.to(torch.int32)
        tab[cta, pos, 0] = (((rows - cta * rows_per_cta) << 16) | cols).to(torch.int32)
        tab[cta, pos, 1] = bits[rows, cols].to(torch.int32)   # the original BF16 weights; the kernel forms the correction
        return {"L": L, "C": C, "tab": tab.contiguous(), "pmax": pmax, "base4": base * BASE4_REPLICATE}


    def sk2_stream(tub, n_cfg, R, NT, NWR, WSK, UPW):
        """Stage-major stream [stage][cta][warp][unit][tile][bytes] for the first n_cfg rows."""
        dev = tub.device
        U = tub.shape[1]
        ups = WSK * UPW
        ar = lambda v: torch.arange(v, device=dev)
        s = ar(U // ups).view(-1, 1, 1, 1, 1)
        c = ar(n_cfg // R).view(1, -1, 1, 1, 1)
        w = ar(NWR * WSK).view(1, 1, -1, 1, 1)
        uu = ar(UPW).view(1, 1, 1, -1, 1)
        nt = ar(NT).view(1, 1, 1, 1, -1)
        tile = c * (R // 8) + (w // WSK) * NT + nt
        unit = s * ups + (w % WSK) * UPW + uu
        return tub[tile, unit].contiguous()


    SK2_BALANCE = True   # v135: deal 4-row groups to sk2 row blocks so every block carries ~the same number of escapes
    SK2_BALANCE_LAYERS = {"qkv_gate"}   # v136: only where it measured a gain (qkv m=8/16 -0.5 us; o/unpooler m=32 lost)


    def sk2_balance_perm(w, n_rows, R):
        """Row order for R-row sk2 blocks over the first n_rows rows of w: rows sorted by escape count and snake-dealt, so
    every block holds ~the same number of escapes (the escape warps' work sets the CTA tail). perm[p] = original row."""
        wl = w[:n_rows]
        bits = wl.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
        e71 = (bits >> 8) & 0x7F
        base, _ = _exponent_window(e71)
        cnt = _escapes(e71, base).sum(1)
        # 4-row groups are dealt (keeps 4-column output groups contiguous: vector stores in the epilogue)
        G4 = 4
        tcnt = cnt.view(n_rows // G4, G4).sum(1)
        order = torch.argsort(tcnt, descending=True, stable=True)
        TB = R // G4                                   # groups per block
        nb = n_rows // R
        i = torch.arange(n_rows // G4, device=w.device)
        rnd, pos = i // nb, i % nb
        blk = torch.where(rnd % 2 == 0, pos, nb - 1 - pos)
        tperm = torch.empty(n_rows // G4, dtype=torch.long, device=w.device)
        tperm[blk * TB + rnd] = order
        return (tperm.view(-1, 1) * G4 + torch.arange(G4, device=w.device).view(1, G4)).reshape(-1)


    def sk2_table_add_perm(tab, pmax, perm, R):
        """Append each CTA's row map (R int32 output columns) after its escapes; entry 0 .y = the map's int2 offset."""
        ncta = tab.shape[0]
        pm2 = pmax + R // 2
        pm2 += pm2 % 2
        out = torch.zeros(ncta, pm2, 2, dtype=torch.int32, device=tab.device)
        out[:, :pmax] = tab
        out[:, pmax:pmax + R // 2] = perm.to(torch.int32).view(ncta, R // 2, 2)
        out[:, 0, 1] = pmax
        return out.contiguous(), pm2


    def sk2_table(comp, n_cfg, R):
        """Per-CTA escape table int2 [ncta][pmax]: entry 0 = (count, 0); entries ((row_local << 16) | k, the escaped weight's
    original BF16 bits); the kernel forms the exact correction (w - decode(w)) itself."""
        dev = comp["rows"].device
        rows, cols, delta = comp["rows"], comp["cols"], comp["wbits"]
        keep = rows < n_cfg
        rows, cols, delta = rows[keep], cols[keep], delta[keep]
        ncta = n_cfg // R
        cta = rows // R
        counts = torch.bincount(cta, minlength=ncta)
        pmax = int(counts.max().item()) + 1 if rows.numel() else 1
        pmax += pmax % 2
        order = torch.argsort(cta.long() * 65536 + cols.long())   # per CTA in k order (the kernel walks K once)
        cta, rows, cols, delta = cta[order], rows[order], cols[order], delta[order]
        start = torch.zeros(ncta + 1, dtype=torch.long, device=dev)
        start[1:] = torch.cumsum(counts, 0)
        slot = torch.arange(cta.numel(), device=dev) - start[cta] + 1
        tab = torch.zeros(ncta, pmax, 2, dtype=torch.int32, device=dev)
        tab[:, 0, 0] = counts.to(torch.int32)
        tab[cta, slot, 0] = (((rows % R) << 16) | cols).to(torch.int32)
        tab[cta, slot, 1] = delta.to(torch.int32)
        return tab.contiguous(), pmax


    def sk2_prepare(weights, state):
        state["sk2"] = {}
        dev = weights["o"].device
        shared = {}
        for ci, cfg in enumerate(SK2_CONFIGS):
            name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
            w = weights[layer]
            if layer == "router":
                hi = w.to(torch.bfloat16)
                lo = (w - hi.float()).to(torch.bfloat16)
                T, U = w.shape[0] // 8, w.shape[1] // 64
                tub = torch.cat([sk2_tileunits(hi).contiguous().view(torch.uint8).reshape(T, U, 1024),
                                 sk2_tileunits(lo).contiguous().view(torch.uint8).reshape(T, U, 1024)], dim=-1)
                stream = sk2_stream(tub, n_rows, R, NT, NWR, WSK, UPW)
                tab, pmax, b4 = state["dummy"], 0, 0
            else:
                perm = None
                if SK2_BALANCE and CL == 1 and layer in SK2_BALANCE_LAYERS:
                    pkey = (layer, n_rows, R)
                    bal = state.setdefault("sk2_bal", {})
                    if pkey not in bal:
                        pr = sk2_balance_perm(w, n_rows, R)
                        bal[pkey] = (pr, sk2_compress(w[:n_rows][pr].contiguous()))
                    perm, comp = bal[pkey]
                    if comp is None:
                        perm = None
                if perm is None:
                    comp = state.setdefault("sk2_comp", {}).get(layer)
                    if comp is None:
                        comp = sk2_compress(w.contiguous())
                        state["sk2_comp"][layer] = comp
                if comp is None:
                    continue
                lkey = (layer, n_rows, R, NT, NWR, WSK, UPW, perm is not None)   # configs differing only in NACC / ring depth share a stream
                if lkey not in shared:
                    tab, pmax = sk2_table(comp, n_rows, R)
                    if perm is not None:
                        tab, pmax = sk2_table_add_perm(tab, pmax, perm, R)
                    shared[lkey] = (sk2_stream(comp["packed"], n_rows, R, NT, NWR, WSK, UPW), tab, pmax)
                stream, tab, pmax = shared[lkey]
                b4 = comp["base4"]
            state["sk2"][name] = (ci, stream, tab, pmax, b4, n_rows)
        state.pop("sk2_comp", None)
        state.pop("sk2_bal", None)


    def _run_sk2(x, layer, m, n, k, state, dg=0, dbg=None):
        name = _sk2_name(layer, m)
        ci, stream, tab, pmax, b4, n_rows = state["sk2"][name]
        out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
        _get_ext().sk2(stream, x, out, tab, pmax, b4, m, n, k, ci, _sk2_tokr(m), dg, state["dummy"] if dbg is None else dbg)
        return out


    def _mt_key(m):
        return 0 if m == 1 else 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 8


    # ---------------------------------------------------------------------------
    # prepare(): build every weight layout once into `state`. The CUTLASS
    # extension builds on a background thread while the weights are compressed.
    # ---------------------------------------------------------------------------
    def _ensure_skinny_stream(state, name, src, comp, c, kk, n, dev, splits, rgc, wsn, ks, n_cfg):
        """Ensure one skinny decomposition's prepared inputs exist in ``state``:
    the per-CTA weight stream, its split-K counter, and (compressed only) its
    per-CTA escape table. Idempotent: shared decompositions are built once."""
        key = (name, 0 if wsn == 1 else splits, rgc, wsn, ks, n_cfg)
        if key not in state["streams"]:
            state["streams"][key] = build_stream(src, comp, n_cfg, kk, splits, rgc, wsn, ks)
        ckey = (name, splits, rgc, wsn, ks, n_cfg)
        if ckey not in state["cnt"]:
            state["cnt"][ckey] = torch.zeros(n // 16, dtype=torch.int32, device=dev)  # per-config counters
        if comp and (name, splits, rgc, n_cfg) not in state["tabs"]:
            state["tabs"][(name, splits, rgc, n_cfg)] = build_patch_table(c, n_cfg, kk, splits, rgc)


    def prepare(weights):
        _get_ext()  # compile now, so a build failure surfaces in prepare(), not mid-run
        # the CUTLASS extension builds in the background (nvcc is a subprocess) while the weights are compressed; started
        # after the main build so the two load_inline calls never overlap
        cut_thread = None
        if CUTLASS_LAYERS and torch.cuda.get_device_capability()[0] >= 9 and _cut_ext is None and _cut_err is None:
            cut_thread = threading.Thread(target=_build_cutlass, daemon=True)
            cut_thread.start()
        dev = weights["qkv_gate"].device
        sm90_ = torch.cuda.get_device_capability()[0] >= 9
        # on sm_90 the sk2 kernels replace the older skinny kernels and the BF16 wide kernel (never faster than cuBLAS)
        old_paths = not (sm90_ and USE_SK2)
        state = {"raw": {}, "cnt": {}, "comp": {}, "mid_cnt": {}, "streams": {}, "tabs": {}, "base4": {}}
        state["dummy"] = torch.zeros(2, dtype=torch.int32, device=dev)
        # split-K partials: kernels run one at a time, so every layer shares one workspace
        ws_floats = 0
        for (lname, mtk), (splits, rgc, wsn, ks) in TMA_CONFIGS.items():
            mmax = {0: 1, 1: 8, 2: 16, 4: 32, 8: 64}[mtk]
            ws_floats = max(ws_floats, splits * mmax * WEIGHT_ROWS[lname])
        for (lname, mm), (splits, rgc, wsn, ks) in TMA_ALT.items():
            ws_floats = max(ws_floats, splits * mm * WEIGHT_ROWS[lname])
        state["ws_shared"] = torch.empty(ws_floats, dtype=torch.float32, device=dev)
        gemm_floats = 0
        for (lname, bn), (nwg, splits, stages) in GEMM_CONFIGS.items():
            gemm_floats = max(gemm_floats, (WEIGHT_ROWS[lname] // (64 * nwg)) * splits * (nwg * 128) * (bn // 2))
        for (lname, tok, nz), (nwg, splits, stages) in WIDE_CONFIGS.items():
            gemm_floats = max(gemm_floats, nz * (WEIGHT_ROWS[lname] // (64 * nwg)) * splits * (nwg * 64) * tok)
        state["mid_ws_shared"] = torch.empty(gemm_floats, dtype=torch.float32, device=dev)
        for name, w in weights.items():
            if name == "router":
                hi = w.to(torch.bfloat16)
                lo = (w - hi.float()).to(torch.bfloat16)
                state["router_hi_t"] = hi.t().contiguous()
                state["router_f32"] = w.float().contiguous()   # the router GEMV (m = 1) reads the FP32 weights as given
                state["router_lo_t"] = lo.t().contiguous()
                state["router_cat"] = torch.cat([hi, lo], dim=1).contiguous()   # gemm90 fallback (m 33..256), rk
                state["rk_zc"] = torch.zeros(8, dtype=torch.int32, device=w.device)   # rk: grid counters, one per token-tile count
                # hi and lo tiles of the same k-step back to back: [rg, s, 2, 32, 8]
                src = torch.stack([pack_fragments(hi), pack_fragments(lo)], dim=2).contiguous()
                comp, kk = False, w.shape[1]
                c = None
                # wide kernel stream: [rb][unit][warp][hi 4 k-steps x 512 B | lo 4 k-steps x 512 B], 64-row blocks
                nrg_r = w.shape[0] // 16
                u_all = w.shape[1] // 64
                t = src.view(nrg_r, u_all, 4, 2, 32, 8).permute(0, 1, 3, 2, 4, 5).contiguous()   # [rg, u, hl, 4, 32, 8]
                nwg_r = WIDE_CONFIGS[("router", 128, 1)][0]
                t = t.view(nrg_r // (4 * nwg_r), 4 * nwg_r, u_all, 2 * 4 * 32 * 8).permute(0, 2, 1, 3).contiguous()
                del t
                for (lname, tok, nz), (nwg, splits, stages) in WIDE_CONFIGS.items():
                    if lname == "router":
                        state["cnt"][("router", "wide", tok, nz)] = torch.zeros(w.shape[0] // 16, dtype=torch.int32, device=dev)
            else:
                w = w.contiguous()
                state["raw"][name] = w
                if name in CG_CONFIGS:
                    wc = w[:UNPOOLER_SLICE_ROWS] if name == "unpooler" else w   # the unpooler at m = 1 is the slice
                    cg = cg_compress(wc.contiguous(), CG_CONFIGS[name][0])
                    if cg is not None:
                        state.setdefault("cg", {})[name] = cg
                if name in TORCHT_LAYERS and sm90_:
                    state.setdefault("rawT", {})[name] = w.t().contiguous()
                c = pack_compressed(w) if name in COMPRESSED_LAYERS else None
                if c is not None:
                    src, comp = c["packed"].transpose(0, 1).contiguous(), True  # [rg, u, 3, 32, 16]
                    state["base4"][name] = c["base4"]
                    del c["packed"]
                else:
                    src, comp = pack_fragments(w), False
                kk = w.shape[1]
            n = w.shape[0]
            state["comp"][name] = comp
            for (lname, mtk), (splits, rgc, wsn, ks) in TMA_CONFIGS.items():
                if lname != name or not old_paths:
                    continue
                n_cfg = UNPOOLER_SLICE_ROWS if (name == "unpooler" and mtk <= 1) else n
                _ensure_skinny_stream(state, name, src, comp, c, kk, n, dev, splits, rgc, wsn, ks, n_cfg)
            if comp and old_paths:
                for (lname, mm), (splits, rgc, wsn, ks) in TMA_ALT.items():
                    if lname != name:
                        continue
                    n_cfg = UNPOOLER_SLICE_ROWS if (name == "unpooler" and mm <= 8) else n
                    _ensure_skinny_stream(state, name, src, comp, c, kk, n, dev, splits, rgc, wsn, ks, n_cfg)
            for (lname, tok, nz), (nwg, splits, stages) in WIDE_CONFIGS.items():
                if lname != name or not comp or not (old_paths or WIDE_BF16):   # unused on sm_90 (cuBLAS / gq / gqx)
                    continue
                rgc = 4 * nwg
                key = (name, 0, rgc, 1, 4, n)
                if key not in state["streams"]:
                    state["streams"][key] = build_stream(src, comp, n, kk, splits, rgc, 1, 4)
                if (name, splits, rgc, n) not in state["tabs"]:
                    state["tabs"][(name, splits, rgc, n)] = build_patch_table(c, n, kk, splits, rgc)
                state["cnt"][(name, "wide", tok, nz)] = torch.zeros(n // 16, dtype=torch.int32, device=dev)
            del src
            state["mid_cnt"][name] = torch.zeros(n // 16, dtype=torch.int32, device=dev)
        state["sm90"] = sm90_
        state["nsm"] = torch.cuda.get_device_properties(dev).multi_processor_count   # rk: one resident CTA per SM
        state["old_paths"] = old_paths
        sk2_prepare(weights, state)
        if cut_thread is not None:
            cut_thread.join(timeout=CUTLASS_BUILD_TIMEOUT)   # ~70-80 s on an H200 pod; never let it threaten prepare's limit
            if cut_thread.is_alive():
                print(f"[okc] CUTLASS route off: build still running after {CUTLASS_BUILD_TIMEOUT} s", file=sys.stderr)
        if _cut_ext is not None and (cut_thread is None or not cut_thread.is_alive()):
            # Persistent scheduler workspace: a few KB at most, sized with headroom.
            state["cut_ws"] = torch.zeros(1 << 20, dtype=torch.uint8, device=dev)
            state["cut_layers"] = set(CUTLASS_LAYERS) & set(state["raw"])
        elif _cut_err is not None:
            print(f"[okc] CUTLASS route off: {_cut_err}", file=sys.stderr)
        _selftest(weights, state)
        _autotune(weights, state)
        torch.cuda.empty_cache()
        return state


    # ---------------------------------------------------------------------------
    # Routing, selftest, autotune. The static route maps (layer, m) to a family;
    # _candidates adds the autotuner alternatives; _autotune times them as CUDA
    # graph replays from a cold L2 and keeps the fastest per (layer, m) in
    # state["choice"]. An untuned m in _nearby_choice takes its bucket's choice.
    # ---------------------------------------------------------------------------
    # Which implementation runs a (layer, m): "skinny" (TMA-streamed tensor-core GEMV, m <= 32),
    # "gemm" (TMA + wgmma split-K GEMM, sm_90, m <= 256) or "torch" (cuBLAS, as SGLang).
    ROUTE_BF16 = [(32, "skinny"), (256, "wide")]
    WIDE_SM80_TEST = False  # local check of the wide kernel's sm_80 path
    ROUTE_ROUTER = [(32, "skinny"), (256, "wide")]


    def _route(layer, m, sm90):
        for mmax, r in (ROUTE_ROUTER if layer == "router" else ROUTE_BF16):
            if m <= mmax:
                return r
        return "torch"


    USE_RK = True   # v141: router m 33..256 candidate 'rk' (order-dependent FP32 adds; stress compares it by tolerance)


    def _route_key(layer, m, route):
        if route == "sk2":
            return (layer, route, _sk2_name(layer, m), _sk2_tokr(m))
        if route == "skinny":
            return (layer, route, _mt_key(m))
        if route == "wide":
            return (layer, route) + _wide_key(m)
        if route == "rk":
            return (layer, route, (m + 63) // 64)
        return (layer, route, 64 if m <= 64 else 128 if m <= 128 else 256)


    SELFTEST_M = (1, 8, 16, 32, 33, 128, 200, 256)


    def _selftest(weights, state):
        """Run every custom path once on a random input against an FP64 reference; disable (-> cuBLAS) any path that
    does not match.  Guards against a kernel that misbehaves on a GPU this file was not tested on."""
        state["disabled"] = set()
        g = torch.Generator(device=weights["o"].device)
        g.manual_seed(SELFTEST_SEED)
        for layer, w in weights.items():
            n, k = w.shape
            for m in SELFTEST_M:
                if _PRUNE and _TABLE.get('%s:%d' % (layer, m), _DEFAULT) != 'vjeux':
                    continue
                nn = UNPOOLER_SLICE_ROWS if (layer == "unpooler" and m <= 8) else n
                x = torch.randn(m, k, device=w.device, generator=g).to(torch.bfloat16)
                ref = x.double() @ w[:nn].double().T
                if layer != "router":
                    ref = ref.to(torch.bfloat16).double()
                tol = 2e-5 if layer == "router" else 5e-3
                for ch in _candidates(layer, m, state):
                    if ch[0] == "torch":
                        continue
                    key = (("skinny_alt", layer, m) if (ch[0] == "skinny" and ch[2][0] == "alt")
                           else _route_key(layer, m, ch[0]))
                    try:
                        out = _run(ch, x, layer, m, nn, k, state).double()
                        torch.cuda.synchronize()
                        err = ((out - ref).norm() / ref.norm()).item()
                    except RuntimeError as e:  # launch failure -> fall back
                        err = float("inf")
                        print(f"[okc] selftest {layer} m={m} {ch[0]}: {e}", file=sys.stderr)
                    if not err < tol:
                        state["disabled"].add(key)
                        print(f"[okc] selftest {layer} m={m} {ch[0]}: rel err {err:.3e} -> disabled", file=sys.stderr)


    # ---------------------------------------------------------------------------
    # Running one projection: each _run_* allocates a fresh output and launches
    # one route. Nothing is cached across calls.
    # ---------------------------------------------------------------------------
    def _run_skinny(x, layer, m, n, k, state, cfg, tu_key, dg=0):
        splits, rg, wsn, ks = cfg
        mtk = _mt_key(m)
        out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
        stream = state["streams"][(layer, 0 if wsn == 1 else splits, rg, wsn, ks, n)]
        cnt = state["cnt"][(layer, splits, rg, wsn, ks, n)]
        fn = _get_ext().skinny_tma_a
        if state["comp"][layer]:
            tab, pmax = state["tabs"][(layer, splits, rg, n)]
            fn(stream, x, out, state["ws_shared"], cnt, tab, pmax, state["base4"][layer], m, n, k, k, splits,
               max(mtk, 1), rg, wsn, 1, WEIGHT_ROWS[layer] // 16, ks, dg)
        else:
            # Uncompressed: no escapes, so the patch table is the dummy and
            # pmax/base4 are 0 (the kernel skips exact correction).
            fn(stream, x, out, state["ws_shared"], cnt, state["dummy"], 2, 0, m, n, k, k, splits,
               max(mtk, 1), rg, wsn, 0, WEIGHT_ROWS[layer] // 16, ks, dg)
        return out


    def _run_wide(x, layer, m, n, k, state):
        tok, nz = _wide_key(m)
        nwg, splits, _ = WIDE_CONFIGS[(layer, tok, nz)]
        rgc = 4 * nwg
        if layer == "router":
            out = torch.empty((m, n), dtype=torch.float32, device=x.device)
            tab, pmax, b4 = state["dummy"], 0, 0
        else:
            out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
            (tab, pmax), b4 = state["tabs"][(layer, splits, rgc, n)], state["base4"][layer]
        wst = state["router_cat"] if layer == "router" else state["streams"][(layer, 0, rgc, 1, 4, n)]
        _get_ext().wide(wst, x, out, state["mid_ws_shared"],
                        state["cnt"][(layer, "wide", tok, nz)], tab, pmax, b4, m, n, k, splits, nwg, tok)
        return out


    def _run_gemm(x, layer, m, n, k, state):
        bn = 64 if m <= 64 else 128 if m <= 128 else 256
        nwg, splits, _ = GEMM_CONFIGS[(layer, bn)]
        out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
        w = state["router_cat"] if layer == "router" else state["raw"][layer]
        _get_ext().gemm90(w, state["dummy"], x, out, state["mid_ws_shared"], state["mid_cnt"][layer], m, n, w.shape[1], k,
                          splits, nwg, bn, 0)
        return out


    def _run_torch(x, layer, m, n, k, state):
        if layer == "router":
            acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
            acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
            return acc
        return F.linear(x, state["raw"][layer][:n])   # what SGLang's bf16 dispatch runs on H200


    def _run(choice, x, layer, m, n, k, state):
        kind = choice[0]
        if kind == "skinny":
            return _run_skinny(x, layer, m, n, k, state, choice[1], choice[2])
        if kind == "wide":
            return _run_wide(x, layer, m, n, k, state)
        if kind == "sk2":
            return _run_sk2(x, layer, m, n, k, state)
        if kind == "gemm":
            return _run_gemm(x, layer, m, n, k, state)
        if kind == "gq":
            return _run_gq(x, layer, m, n, k, state)
        if kind == "gqx":
            return _run_gqx(x, layer, m, n, k, state)
        if kind == "cgemv":   # m = 1: 12-bit compressed GEMV (qkv, o, unpooler slice, pooler)
            cg = state["cg"][layer]
            rows, nth = CG_CONFIGS[layer]
            out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
            _get_ext().cgemv(cg["L"], cg["C"], x, cg["tab"], cg["pmax"], out, cg["base4"], rows, nth)
            return out
        if kind == "sgemv":   # unpooler slice, m = 1: BF16 weights straight from memory (4 rows per CTA)
            out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
            _get_ext().sgemv(state["raw"][layer], x, out, n)
            return out
        if kind == "rgemv":   # router, m = 1: FP32 weights straight from memory (one row per CTA)
            out = torch.empty((m, n), dtype=torch.float32, device=x.device)
            _get_ext().rgemv(state["router_f32"], x, out, m)
            return out
        if kind == "cutlass":   # m 129..256: CUTLASS pingpong 128x64 (see CUTLASS_SRC)
            out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
            _cut_ext.cut_gemm(state["raw"][layer], x, out, state["cut_ws"])
            return out
        if kind == "rk":   # router m 33..256: dedicated split-K GEMM with order-dependent FP32 adds (see RK_SRC)
            out = torch.empty((m, n), dtype=torch.float32, device=x.device)
            _get_ext().rk(state["router_cat"], x, out, _rk_splits(m, state["nsm"]), state["rk_zc"])
            return out
        if kind == "torchT":   # cuBLAS on a K-major copy of W: a different (faster) kernel choice for unpooler m=128
            return torch.mm(x, state["rawT"][layer])
        return _run_torch(x, layer, m, n, k, state)


    RK_SPLITS = {1: 16, 2: 16, 3: 10, 4: 8}   # token tiles of 64 -> K splits (grid 4 x ntt x S <= 128; pod/rk.py sweep)


    def _rk_splits(m, nsm=132):
        """K splits for m tokens: the measured table, capped so the grid (4 x ntt x S, one CTA per SM, all resident: the
    in-kernel clear waits for every CTA) fits the GPU's SMs."""
        ntt = (m + 63) // 64
        return max(1, min(RK_SPLITS[ntt], nsm // (4 * ntt)))


    def _static_choice(layer, m, state):
        route = _route(layer, m, state["sm90"])
        if not state.get("old_paths", True):
            if route == "skinny":
                route = "sk2" if (_sk2_name(layer, m) in state.get("sk2", {})
                                  and _route_key(layer, m, "sk2") not in state.get("disabled", ())) else "torch"
            elif route == "wide" and layer != "router" and not WIDE_BF16:
                route = "torch"
        if route != "torch" and _route_key(layer, m, route) in state.get("disabled", ()):
            # fall back: router -> the wgmma GEMM (validated earlier) when possible, else cuBLAS
            route = "gemm" if (layer == "router" and route == "wide" and state["sm90"]
                               and _route_key(layer, m, "gemm") not in state["disabled"]) else "torch"
        if route == "skinny":
            mtk = _mt_key(m)
            return ("skinny", TMA_CONFIGS[(layer, mtk)], (layer, mtk))
        return (route,)


    def _candidates(layer, m, state):
        """Implementations worth timing for (layer, m): the static choice, alternative skinny configs, cuBLAS."""
        c = [_static_choice(layer, m, state)]
        if c[0][0] == "skinny" and (layer, m) in TMA_ALT and ("skinny_alt", layer, m) not in state["disabled"]:
            c.append(("skinny", TMA_ALT[(layer, m)], ("alt", layer, m)))
        if (c[0][0] == "wide" and layer == "router" and state["sm90"]
                and _route_key(layer, m, "gemm") not in state["disabled"]):
            c.append(("gemm",))
        if (m <= 32 and c[0][0] != "sk2" and _sk2_name(layer, m) in state.get("sk2", {})
                and _route_key(layer, m, "sk2") not in state["disabled"]):
            c.insert(0, ("sk2",))
        if (USE_GQ and state["sm90"] and 32 < m <= 256 and layer != "router" and (layer, _gq_bn(m)) in GQ_CONFIGS
                and _route_key(layer, m, "gq") not in state["disabled"]):
            c.insert(0, ("gq",))
        if (USE_GQX and state["sm90"] and 32 < m <= 256 and layer != "router" and (layer, _gqx_bn(m)) in GQX_CONFIGS
                and _route_key(layer, m, "gqx") not in state["disabled"]):
            c.insert(0, ("gqx",))
        if layer == "router" and m == 1 and "router_f32" in state:
            c.insert(0, ("rgemv",))
        if (USE_RK and layer == "router" and 32 < m <= 256 and state["sm90"] and "rk_zc" in state and state.get("nsm", 0) >= 16
                and _route_key(layer, m, "rk") not in state["disabled"]):
            c.insert(0, ("rk",))
        if layer == "unpooler" and m == 1:
            c.insert(0, ("sgemv",))
        if m == 1 and layer in state.get("cg", {}) and _route_key(layer, m, "cgemv") not in state["disabled"]:
            c.insert(0, ("cgemv",))
        if layer in TORCHT_LAYERS and 32 < m <= 256 and layer in state.get("rawT", {}):
            c.append(("torchT",))
        if (128 < m <= 256 and layer in state.get("cut_layers", ()) and _cut_ext is not None
                and _route_key(layer, m, "cutlass") not in state["disabled"]):
            c.append(("cutlass",))
        if c[-1][0] != "torch" and all(ch[0] != "torch" for ch in c):
            c.append(("torch",))
        return c


    AUTOTUNE = True
    TUNE_M = (1, 16, 32, 8, 128, 256)   # the scored token counts


    def _autotune(weights, state):
        """Time each candidate as a CUDA graph replay from a cold L2 (a read sweep larger than L2 before every replay,
    per-replay events, candidates interleaved), on synthetic activations (reads only the weights), and keep the
    fastest per (layer, m)."""
        state["choice"] = {}
        if not AUTOTUNE:
            return
        dev = weights["o"].device
        flush = torch.ones(L2_FLUSH_ROWS, 1024, dtype=torch.float32, device=dev)  # read-only sweep
        sink = torch.empty(flush.shape[1], dtype=torch.float32, device=dev)
        g = torch.Generator(device=dev)
        g.manual_seed(AUTOTUNE_SEED)
        reps = AUTOTUNE_REPS
        for m in TUNE_M:
            for layer, w in weights.items():
                if _PRUNE and _TABLE.get('%s:%d' % (layer, m), _DEFAULT) != 'vjeux':
                    continue
                n = UNPOOLER_SLICE_ROWS if (layer == "unpooler" and m <= 8) else w.shape[0]
                k = w.shape[1]
                cands = _candidates(layer, m, state)
                if len(cands) < 2:
                    continue
                x = torch.randn(m, k, device=dev, generator=g).to(torch.bfloat16)
                graphs = []
                for ch in cands:
                    _run(ch, x, layer, m, n, k, state)
                    torch.cuda.synchronize()
                    gr = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(gr):
                        _run(ch, x, layer, m, n, k, state)
                    graphs.append(gr)
                evs = []
                for r in range(reps):
                    order = range(len(cands)) if r % 2 == 0 else reversed(range(len(cands)))
                    for i in order:
                        torch.sum(flush, dim=0, out=sink)
                        e0 = torch.cuda.Event(enable_timing=True)
                        e1 = torch.cuda.Event(enable_timing=True)
                        e0.record()
                        graphs[i].replay()
                        e1.record()
                        evs.append((i, e0, e1))
                torch.cuda.synchronize()
                per = [[] for _ in cands]
                for i, e0, e1 in evs:
                    per[i].append(e0.elapsed_time(e1))
                med = [sorted(v)[len(v) // 2] for v in per]
                best = min(range(len(cands)), key=lambda i: med[i])
                state["choice"][(layer, m)] = cands[best]
                # CUTLASS is tuned for the scored m only (its partial token tiles are slow: m=129 +14 us, m=200 +4 us vs
                # cuBLAS): other m of the bucket take the best other candidate
                alt = [i for i in range(len(cands)) if cands[i][0] != "cutlass"]
                if alt:
                    state.setdefault("choice_alt", {})[(layer, m)] = cands[min(alt, key=lambda i: med[i])]
                state.setdefault("tune_log", []).append((layer, m, [(c[0], round(1000 * v, 2)) for c, v in zip(cands, med)]))
                del graphs
        del flush
        torch.cuda.synchronize()
        # one line per shape in the run's log: candidate times (us, raw graph replay incl. ~4 us replay cost) and the pick
        for layer, m, tl in state.get("tune_log", []):
            ch = state["choice"][(layer, m)]
            desc = " ".join(f"{name}{'/' + 'x'.join(map(str, c[1])) if c[0] == 'skinny' else ''}={t}"
                            for (name, t), c in zip(tl, _candidates(layer, m, state)))
            pick = ch[0] + ("/" + "x".join(map(str, ch[1])) if ch[0] == "skinny" else "")
            print(f"[okc] tune {layer} m={m}: {desc} -> {pick}", file=sys.stderr)
        print(f"[okc] disabled: {sorted(state['disabled'])}", file=sys.stderr)


    def _nearby_choice(layer, m, state):
        """An untuned m takes the choice tuned for the scored m of its bucket (1 | 2-8 | 9-16 | 17-32 | 33-128 |
    129-256): every implementation handles any m of its bucket."""
        tm = 1 if m <= 1 else 8 if m <= 8 else 16 if m <= 16 else 32 if m <= 32 else 128 if m <= 128 else 256 if m <= 256 else 0
        if not tm:
            return None
        if tm == 256 and m != 256 and layer != "router":
            # an untuned m in 129..255 runs SGLang's own call (F.linear): the m=256 picks (CUTLASS, gqx, torchT) were
            # only timed at m=256 and some lose to cuBLAS at partial tiles (pooler CUTLASS m=129 +14 us, unpooler
            # torchT m=240 +11%)
            return ("torch",)
        ch = state.get("choice", {}).get((layer, tm))
        if ch is None:
            return None
        if ch[0] == "cutlass" and m != tm:
            ch = state.get("choice_alt", {}).get((layer, tm))
            if ch is None:
                return None
        if ch[0] == "gq" and (layer, _gq_bn(m)) not in GQ_CONFIGS:
            return None
        return ch


    # ---------------------------------------------------------------------------
    # Entry point: the harness calls prepare(weights) once, then kernel_fn per call.
    # ---------------------------------------------------------------------------
    def kernel_fn(data, state):
        x = data["x"]
        cfg = data["config"]
        layer, m, n, k = cfg["layer"], cfg["m"], cfg["n"], cfg["k"]
        choice = state.get("choice", {}).get((layer, m)) or _nearby_choice(layer, m, state) or _static_choice(layer, m, state)
        out = _run(choice, x, layer, m, n, k, state)
        return out



    return {'prepare': prepare, 'kernel_fn': kernel_fn, 'warm': _get_ext}

_NB_C12 = False


def _make_nb():
    # Two kernels from github.com/NathenFernandes/muse-bf16-proj submission.py @ 7385e5d (MIT, Copyright (c) 2026
    # Nathen Fernandes), source unchanged, built as one extension; the wrappers and per-shape picks are ours.
    #   router, 1 <= m <= 32: rt__CUDA routertail_small (BF16 hi + lo mma.sync over a 16-CTA cluster, DSMEM reduce)
    #   o, m = 128 / 256: OLAYER_CUDA ol_projection (TMA + wgmma on swizzle-packed BF16 weights), instantiated for two
    #   configs as olayer_source() does
    import torch
    from torch.utils.cpp_extension import load_inline

    _RT_CUDA = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <algorithm>
#define RD __device__ __forceinline__
RD uint32_t routertail_smem(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
RD void routertail_init(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b),"r"(n):"memory"); }
RD void routertail_expect(uint32_t b, int n) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b),"r"(n):"memory"); }
RD void routertail_wait(uint32_t b, int ph) {
  asm volatile("{ .reg .pred p; wait: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1; @!p bra wait; }" ::"r"(b),"r"(ph):"memory");
}
RD void routertail_sync() { asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" :::"memory"); }
RD uint32_t routertail_map(uint32_t a, int r) { uint32_t b; asm("mapa.shared::cluster.u32 %0, %1, %2;":"=r"(b):"r"(a),"r"(r));return b; }
RD void routertail_push(uint32_t d, float4 v, uint32_t b) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1,%2,%3,%4}, [%5];" ::"r"(d),"f"(v.x),"f"(v.y),"f"(v.z),"f"(v.w),"r"(b):"memory");
}
RD float4 routertail_pull(uint32_t d) {
  float4 v; asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];":"=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w):"r"(d):"memory");return v;
}
RD void routertail_add(float4& a,float4 b) { a.x+=b.x;a.y+=b.y;a.z+=b.z;a.w+=b.w; }
RD uint64_t routertail_desc(uint32_t a) { return (uint64_t)((a&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62); }
RD void routertail_tma(uint32_t dst, const void* tm, int k,int r,uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2,%3}], [%4];" ::"r"(dst),"l"(tm),"r"(k),"r"(r),"r"(bar):"memory");
}

template<int MT,int S,int WK>
__global__ __launch_bounds__(WK*32) void routertail_small_kernel(const int4* wp,const __nv_bfloat16* x,float* out,int m) {
  constexpr int KU=128/S/WK, NI=MT*32, P=4, OWN=NI/P, PARTS=S*1;
  const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,rank=blockIdx.x%S,row=blockIdx.x/S*16;
  const int gw=(rank*16+row/16)*WK+warp;
  extern __shared__ __align__(16) unsigned char sm[];
  float4* recv=(float4*)sm;
  float4* part=recv+PARTS*OWN;
  const uint32_t rb=routertail_smem(sm+(PARTS*OWN+WK*NI)*16);
  int4 w[KU][8];uint4 xx[KU][MT][2];
#pragma unroll
  for(int k=0;k<KU;k++) {
#pragma unroll
    for(int j=0;j<8;j++) {
      const int4* p=wp+((k*(16*S*WK)+gw)*8+j)*32+lane;
      asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];":"=r"(w[k][j].x),"=r"(w[k][j].y),"=r"(w[k][j].z),"=r"(w[k][j].w):"l"(p));
    }
#pragma unroll
    for(int n=0;n<MT;n++) {
      const __nv_bfloat16* p=x+min(n*8+(lane>>2),m-1)*8192+((rank*WK+warp)*KU+k)*64+(lane&3)*16;
#pragma unroll
      for(int h=0;h<2;h++) asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];":"=r"(xx[k][n][h].x),"=r"(xx[k][n][h].y),"=r"(xx[k][n][h].z),"=r"(xx[k][n][h].w):"l"(p+h*8));
    }
  }
  if(rank<P && tid==0) {routertail_init(rb,1);asm volatile("fence.proxy.async.shared::cta;" :::"memory");}
  __syncthreads();
  if(rank<P && tid==0) {routertail_expect(rb,PARTS*OWN*16);asm volatile("fence.mbarrier_init.release.cluster;" :::"memory");}
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" :::"memory");
  float c[2][MT][4]={};
#pragma unroll
  for(int k=0;k<KU;k++) {
#pragma unroll
    for(int n=0;n<MT;n++) {
      uint32_t b[8]={xx[k][n][0].x,xx[k][n][0].y,xx[k][n][0].z,xx[k][n][0].w,xx[k][n][1].x,xx[k][n][1].y,xx[k][n][1].z,xx[k][n][1].w};
#pragma unroll
      for(int q=0;q<4;q++) {
#pragma unroll
        for(int h=0;h<2;h++) {
          const int4 a=w[k][q+h*4];
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                       :"+f"(c[h][n][0]),"+f"(c[h][n][1]),"+f"(c[h][n][2]),"+f"(c[h][n][3]):"r"(a.x),"r"(a.y),"r"(a.z),"r"(a.w),"r"(b[2*q]),"r"(b[2*q+1]));
        }
      }
    }
  }
#pragma unroll
  for(int n=0;n<MT;n++) part[warp*NI+n*32+lane]=make_float4(c[0][n][0]+c[1][n][0],c[0][n][1]+c[1][n][1],c[0][n][2]+c[1][n][2],c[0][n][3]+c[1][n][3]);
  __syncthreads();
  asm volatile("barrier.cluster.wait.acquire.aligned;" :::"memory");
  if(tid<NI) {
    float4 sum=make_float4(0,0,0,0);
#pragma unroll
    for(int k=0;k<WK;k++) routertail_add(sum,part[k*NI+tid]);
    const int dst=tid/OWN,local=tid%OWN;
    routertail_push(routertail_map(routertail_smem(recv+rank*OWN+local),dst),sum,routertail_map(rb,dst));
  }

  if(rank<P && tid<OWN) {
    routertail_wait(rb,0);
    float4 sum=make_float4(0,0,0,0);
#pragma unroll
    for(int k=0;k<PARTS;k++) routertail_add(sum,recv[k*OWN+tid]);
    const int v=rank*OWN+tid,ln=v%32,tok=(v/32)*8+(ln&3)*2,wr=row+(ln>>2);
    if(tok<m){out[tok*256+wr]=sum.x;out[tok*256+wr+8]=sum.z;}
    if(tok+1<m){out[(tok+1)*256+wr]=sum.y;out[(tok+1)*256+wr+8]=sum.w;}
  }

}

template<int MT,int S,int WK>
int routertail_small_t(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {
  auto fn=routertail_small_kernel<MT,S,WK>;
  constexpr int bytes=(S*1/4+WK)*MT*512+16;
  static bool configured=false;
  if(!configured){cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);cudaFuncSetAttribute(fn,cudaFuncAttributeNonPortableClusterSizeAllowed,1);configured=true;}
  if(query==2){cudaFuncAttributes fa;cudaFuncGetAttributes(&fa,fn);return fa.numRegs*1000+fa.localSizeBytes;}
  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(16*S);cfg.blockDim=dim3(WK*32);cfg.dynamicSmemBytes=bytes;cfg.stream=(cudaStream_t)stream;
  cudaLaunchAttribute at;at.id=cudaLaunchAttributeClusterDimension;at.val.clusterDim={S,1,1};cfg.attrs=&at;cfg.numAttrs=1;
  return (int)cudaLaunchKernelEx(&cfg,fn,(const int4*)w,(const __nv_bfloat16*)x,(float*)out,m);
}
int routertail_small(int64_t w,int64_t x,int64_t out,int m,int s,int wk,int64_t stream,int query) {
#define RSM(S,W) if(s==S && wk==W) {if(m<=8)return routertail_small_t<1,S,W>(w,x,out,m,stream,query);if(m<=16)return routertail_small_t<2,S,W>(w,x,out,m,stream,query);if(m<=32)return routertail_small_t<4,S,W>(w,x,out,m,stream,query);}
RSM(8,8) RSM(16,8) RSM(16,4)
return -99;
}

// ---- c12: the same kernel on lossless 12-bit planes (hi and lo BF16 of the FP32 router): per (16 rows x 64 k) warp tile and
// plane the lane reads 3 x 16 B (2 low-byte words groups + 1 nibble group) instead of 4 x 16 B; exact escape corrections in an
// extra warp. Plane layout [ku][S][16][WK][h][part 0..2][lane] int4; nibble = sign | (exp7 - base) (3 bits), 4 per byte pair.
RD void rt12_decode8(uint32_t l0, uint32_t l1, uint32_t nw, uint32_t b4, uint32_t* r) {
  const uint32_t ha = ((nw & 0x07070707u) + b4) | ((nw << 4) & 0x80808080u);
  const uint32_t hb = (((nw >> 4) & 0x07070707u) + b4) | (nw & 0x80808080u);
  r[0] = __byte_perm(l0, ha, 0x5140); r[1] = __byte_perm(l0, ha, 0x7362);
  r[2] = __byte_perm(l1, hb, 0x5140); r[3] = __byte_perm(l1, hb, 0x7362);
}
template<int MT,int S,int WK>
__global__ __launch_bounds__(WK*32) void routertail_c12_kernel(const int4* wp,const __nv_bfloat16* x,float* out,int m,const unsigned* etab,uint32_t bqh,uint32_t bql) {
  constexpr int KU=128/S/WK, NI=MT*32, P=4, OWN=NI/P, PARTS=S*1;
  const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,rank=blockIdx.x%S,row=blockIdx.x/S*16;
  const int gw=(rank*16+row/16)*WK+warp;
  extern __shared__ __align__(16) unsigned char sm[];
  float4* recv=(float4*)sm;
  float4* part=recv+PARTS*OWN;
  const uint32_t rb=routertail_smem(sm+(PARTS*OWN+WK*NI)*16);
  int4 w[KU][2][3];uint4 xx[KU][MT][2];
  const unsigned eword=lane<16?__ldg(etab+(size_t)(blockIdx.x*WK+warp)*16+lane):0u;   // [cta][warp][16]: word 0 = count, then entries
#pragma unroll
  for(int k=0;k<KU;k++) {
#pragma unroll
    for(int j=0;j<6;j++) {
      const int4* p=wp+((k*(16*S*WK)+gw)*6+j)*32+lane;
      asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];":"=r"(w[k][j/3][j%3].x),"=r"(w[k][j/3][j%3].y),"=r"(w[k][j/3][j%3].z),"=r"(w[k][j/3][j%3].w):"l"(p));
    }
#pragma unroll
    for(int n=0;n<MT;n++) {
      const __nv_bfloat16* p=x+min(n*8+(lane>>2),m-1)*8192+((rank*WK+warp)*KU+k)*64+(lane&3)*16;
#pragma unroll
      for(int h=0;h<2;h++) asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];":"=r"(xx[k][n][h].x),"=r"(xx[k][n][h].y),"=r"(xx[k][n][h].z),"=r"(xx[k][n][h].w):"l"(p+h*8));
    }
  }
  if(rank<P && tid==0) {routertail_init(rb,1);asm volatile("fence.proxy.async.shared::cta;" :::"memory");}
  __syncthreads();
  if(rank<P && tid==0) {routertail_expect(rb,PARTS*OWN*16);asm volatile("fence.mbarrier_init.release.cluster;" :::"memory");}
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" :::"memory");
  float c[2][MT][4]={};
  const int ecnt=__shfl_sync(0xffffffffu,(int)eword,0);
  const uint32_t b4h=bqh*0x01010101u,b4l=bql*0x01010101u;
#pragma unroll
  for(int k=0;k<KU;k++) {
    uint32_t a[2][4][4];
#pragma unroll
    for(int h=0;h<2;h++) {
      const uint32_t b4=h?b4l:b4h;
      rt12_decode8(w[k][h][0].x,w[k][h][0].y,w[k][h][2].x,b4,a[h][0]);
      rt12_decode8(w[k][h][0].z,w[k][h][0].w,w[k][h][2].y,b4,a[h][1]);
      rt12_decode8(w[k][h][1].x,w[k][h][1].y,w[k][h][2].z,b4,a[h][2]);
      rt12_decode8(w[k][h][1].z,w[k][h][1].w,w[k][h][2].w,b4,a[h][3]);
    }
    // exact escapes: overwrite the mis-decoded element with the true BF16 (entry: lane 0-4, q 5-6, reg 7-8, half 9, h 10, k 11, bits 16-31)
    for(int e=0;e<ecnt;e++) {
      const unsigned en=__shfl_sync(0xffffffffu,(int)eword,e+1);
      if(((en>>11)&1)!=(unsigned)k && KU>1) continue;
      if((int)(en&31)!=lane) continue;
      const int q=(en>>5)&3,rg=(en>>7)&3,hf=(en>>9)&1,hh=(en>>10)&1;
      const uint32_t sh=hf*16,msk=0xFFFFu<<sh,val=(en>>16)<<sh;
#pragma unroll
      for(int hq=0;hq<2;hq++)
#pragma unroll
        for(int qq=0;qq<4;qq++)
#pragma unroll
          for(int rr=0;rr<4;rr++)
            if(hq==hh && qq==q && rr==rg) a[hq][qq][rr]=(a[hq][qq][rr]&~msk)|val;
    }
#pragma unroll
    for(int n=0;n<MT;n++) {
      uint32_t b[8]={xx[k][n][0].x,xx[k][n][0].y,xx[k][n][0].z,xx[k][n][0].w,xx[k][n][1].x,xx[k][n][1].y,xx[k][n][1].z,xx[k][n][1].w};
#pragma unroll
      for(int q=0;q<4;q++) {
#pragma unroll
        for(int h=0;h<2;h++) {
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                       :"+f"(c[h][n][0]),"+f"(c[h][n][1]),"+f"(c[h][n][2]),"+f"(c[h][n][3]):"r"(a[h][q][0]),"r"(a[h][q][1]),"r"(a[h][q][2]),"r"(a[h][q][3]),"r"(b[2*q]),"r"(b[2*q+1]));
        }
      }
    }
  }
#pragma unroll
  for(int n=0;n<MT;n++) part[warp*NI+n*32+lane]=make_float4(c[0][n][0]+c[1][n][0],c[0][n][1]+c[1][n][1],c[0][n][2]+c[1][n][2],c[0][n][3]+c[1][n][3]);
  __syncthreads();
  asm volatile("barrier.cluster.wait.acquire.aligned;" :::"memory");
  if(tid<NI) {
    float4 sum=make_float4(0,0,0,0);
#pragma unroll
    for(int k=0;k<WK;k++) routertail_add(sum,part[k*NI+tid]);
    const int dst=tid/OWN,local=tid%OWN;
    routertail_push(routertail_map(routertail_smem(recv+rank*OWN+local),dst),sum,routertail_map(rb,dst));
  }

  if(rank<P && tid<OWN) {
    routertail_wait(rb,0);
    float4 sum=make_float4(0,0,0,0);
#pragma unroll
    for(int k=0;k<PARTS;k++) routertail_add(sum,recv[k*OWN+tid]);
    const int v=rank*OWN+tid,ln=v%32,tok=(v/32)*8+(ln&3)*2,wr=row+(ln>>2);
    if(tok<m){out[tok*256+wr]=sum.x;out[tok*256+wr+8]=sum.z;}
    if(tok+1<m){out[(tok+1)*256+wr]=sum.y;out[(tok+1)*256+wr+8]=sum.w;}
  }
}
template<int MT,int S,int WK>
int routertail_c12_t(int64_t w,int64_t x,int64_t out,int m,int64_t etab,int bqh,int bql,int64_t stream) {
  auto fn=routertail_c12_kernel<MT,S,WK>;
  constexpr int bytes=(S*1/4+WK)*MT*512+16;
  static bool configured=false;
  if(!configured){cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);cudaFuncSetAttribute(fn,cudaFuncAttributeNonPortableClusterSizeAllowed,1);configured=true;}
  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(16*S);cfg.blockDim=dim3(WK*32);cfg.dynamicSmemBytes=bytes;cfg.stream=(cudaStream_t)stream;
  cudaLaunchAttribute at;at.id=cudaLaunchAttributeClusterDimension;at.val.clusterDim={S,1,1};cfg.attrs=&at;cfg.numAttrs=1;
  return (int)cudaLaunchKernelEx(&cfg,fn,(const int4*)w,(const __nv_bfloat16*)x,(float*)out,m,(const unsigned*)etab,(uint32_t)bqh,(uint32_t)bql);
}
int routertail_c12(int64_t w,int64_t x,int64_t out,int m,int s,int wk,int64_t etab,int bqh,int bql,int64_t stream) {
#define RSC(S,W) if(s==S && wk==W) {if(m<=8)return routertail_c12_t<1,S,W>(w,x,out,m,etab,bqh,bql,stream);if(m<=16)return routertail_c12_t<2,S,W>(w,x,out,m,etab,bqh,bql,stream);if(m<=32)return routertail_c12_t<4,S,W>(w,x,out,m,etab,bqh,bql,stream);}
RSC(8,8) RSC(16,8) RSC(16,4)
return -99;
}
"""
    _OL_CUDA = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#define OD __device__ __forceinline__
OD uint32_t ol_addr(void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
OD void ol_wait(uint32_t b,int ph) {
  asm volatile("{.reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra W;}" :: "r"(b),"r"(ph) : "memory");
}
OD void ol_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];" :: "r"(b) : "memory"); }
OD uint64_t ol_desc(uint32_t a) { return uint64_t((a&0x3ffff)>>4) | (uint64_t(64)<<32) | (uint64_t(1)<<62); }
OD void ol_tma(uint32_t d,const void* map,int row,int kb,uint32_t b,uint64_t p) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{0,%2,%3}],[%4],%5;" :: "r"(d),"l"(map),"r"(row),"r"(kb),"r"(b),"l"(p) : "memory");
}
OD void ol_mma128(float* a,uint64_t w,uint64_t x) {
          asm volatile("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, 1, 1, 1, 0, 0;" : "+f"(a[0]),"+f"(a[1]),"+f"(a[2]),"+f"(a[3]),"+f"(a[4]),"+f"(a[5]),"+f"(a[6]),"+f"(a[7]),"+f"(a[8]),"+f"(a[9]),"+f"(a[10]),"+f"(a[11]),"+f"(a[12]),"+f"(a[13]),"+f"(a[14]),"+f"(a[15]),"+f"(a[16]),"+f"(a[17]),"+f"(a[18]),"+f"(a[19]),"+f"(a[20]),"+f"(a[21]),"+f"(a[22]),"+f"(a[23]),"+f"(a[24]),"+f"(a[25]),"+f"(a[26]),"+f"(a[27]),"+f"(a[28]),"+f"(a[29]),"+f"(a[30]),"+f"(a[31]),"+f"(a[32]),"+f"(a[33]),"+f"(a[34]),"+f"(a[35]),"+f"(a[36]),"+f"(a[37]),"+f"(a[38]),"+f"(a[39]),"+f"(a[40]),"+f"(a[41]),"+f"(a[42]),"+f"(a[43]),"+f"(a[44]),"+f"(a[45]),"+f"(a[46]),"+f"(a[47]),"+f"(a[48]),"+f"(a[49]),"+f"(a[50]),"+f"(a[51]),"+f"(a[52]),"+f"(a[53]),"+f"(a[54]),"+f"(a[55]),"+f"(a[56]),"+f"(a[57]),"+f"(a[58]),"+f"(a[59]),"+f"(a[60]),"+f"(a[61]),"+f"(a[62]),"+f"(a[63]) : "l"(w),"l"(x));
        }

template<int BN,int BK,int ST,int PIPE,int PACK,int HINT,int TW>
__global__ __launch_bounds__(128*TW+((PIPE&2)?128:0)) void ol_projection(
    const __grid_constant__ CUtensorMap tw,const __grid_constant__ CUtensorMap tx,
    const uint8_t* w,__nv_bfloat16* out,int m) {
  constexpr int T=BN*TW,R=PACK==2?32:64,CT=128*TW;
  constexpr int AB=64*BK*2,SB=(64+T)*BK*2,IT=1024/BK,TH=CT+((PIPE&2)?128:0);
  extern __shared__ __align__(1024) uint8_t sm[];
  uint32_t base=ol_addr(sm),full=base+ST*SB,empty=full+ST*8;
  int tid=threadIdx.x,nt=blockIdx.x,mt=blockIdx.y;
  uint64_t pw,px;
  if constexpr(HINT) {
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pw));
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(px));
  } else {
    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pw));px=pw;
  }
  auto load=[&](int i) {
    int s=i%ST;
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;" :: "r"(full+s*8),"r"(SB) : "memory");
    if constexpr(PACK==1) {
      const uint8_t* src=w+((long)nt*IT+i)*AB;
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1],%2,[%3],%4;" :: "r"(base+s*SB),"l"(src),"r"(AB),"r"(full+s*8),"l"(pw) : "memory");
    } else if constexpr(PACK==2) ol_tma(base+s*SB,&tw,0,nt*16+i*(BK/64),full+s*8,pw);
    else ol_tma(base+s*SB,&tw,nt*64,i*(BK/64),full+s*8,pw);
    ol_tma(base+s*SB+AB,&tx,mt*T,i*(BK/64),full+s*8,px);
  };
  if(tid==((PIPE&2)?CT:0)) {
    #pragma unroll
    for(int s=0;s<ST;s++) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(full+s*8) : "memory");
      if constexpr(PIPE&2) asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;" :: "r"(empty+s*8),"n"(TW) : "memory");
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    #pragma unroll
    for(int i=0;i<ST && i<IT;i++) load(i);
  }
  __syncthreads();
  if((PIPE&2) && tid>=CT) {
    if(tid==CT) {
      #pragma unroll 1
      for(int i=ST;i<IT;i++) {ol_wait(empty+(i%ST)*8,((i/ST)-1)&1);load(i);}
    }
  } else {
    float a[BN/2];
    #pragma unroll
    for(int j=0;j<BN/2;j++) asm volatile("mov.f32 %0,0f00000000;" : "=f"(a[j]) :: "memory");
    #pragma unroll 1
    for(int i=0;i<IT;i++) {
      int s=i%ST;
      ol_wait(full+s*8,(i/ST)&1);
      #pragma unroll
      for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");
      asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
      #pragma unroll
      for(int kk=0;kk<BK/16;kk++) {
        uint64_t aw=ol_desc(base+s*SB+(kk/4)*8192+(kk%4)*32);
        uint64_t ax=ol_desc(base+s*SB+AB+(kk/4)*T*128+(tid/128)*BN*128+(kk%4)*32);
        if constexpr(BN==128) ol_mma128(a,aw,ax);
      }
      asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
      if constexpr(PIPE&1) asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
      else asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
      #pragma unroll
      for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");
      int done=i-((PIPE&1)?1:0);
      if(done>=0 && done+ST<IT && tid%128==0) {
        if constexpr(PIPE&2) ol_arrive(empty+(done%ST)*8);
        else load(done+ST);
      }
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    #pragma unroll
    for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");
    asm volatile("bar.sync 1,%0;" :: "n"(CT) : "memory");
    #pragma unroll
    for(int j=0;j<BN/8;j++) {
      __nv_bfloat162 lo=__floats2bfloat162_rn(a[4*j],a[4*j+1]);
      __nv_bfloat162 hi=__floats2bfloat162_rn(a[4*j+2],a[4*j+3]);
      int rb=((tid%128)/32)*16+((tid%16)/8)*8,cb=(tid/128)*BN+8*j+(tid%8);
      uint32_t d=base+2*(cb*72+rb);
      asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0],{%1,%2};" :: "r"(d),"r"(*(uint32_t*)&lo),"r"(*(uint32_t*)&hi) : "memory");
    }
  }
  __syncthreads();
  for(int q=tid;q<T*(R/8);q+=TH) {
    int token=q/(R/8),row=(q%(R/8))*8;
    if(mt*T+token<m) *(uint4*)(out+(long)(mt*T+token)*8192+nt*R+row)=*(uint4*)((__nv_bfloat16*)sm+token*72+row);
  }
}

using OLEncode=decltype(&cuTensorMapEncodeTiled);
int ol_map(CUtensorMap* tm,const void* ptr,int rows,int box,int bk,bool tiled=false) {
  static OLEncode encode=nullptr;
  if(!encode) {
    cudaDriverEntryPointQueryResult q;
    auto rc=cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&encode,12000,cudaEnableDefault,&q);
    if(rc!=cudaSuccess || !encode) return -100;
  }
  uint64_t dims[]={64,uint64_t(tiled?32:rows),uint64_t(tiled?4096:16)},stride[]={uint64_t(tiled?128:2048),uint64_t(tiled?4096:128)};
  uint32_t b[]={64,uint32_t(box),uint32_t(bk/64)},es[]={1,1,1};
  return int(encode(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,(void*)ptr,dims,stride,b,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}
template<int BN,int BK,int ST,int PIPE,int PACK,int HINT,int TW>
int ol_launch_t(int64_t x,int64_t w,int64_t out,int m,int64_t stream,int query) {
  constexpr int T=BN*TW,R=PACK==2?32:64;
  constexpr int MAIN=ST*((64+T)*BK*2+16),EPI=T*72*2,SMEM=MAIN>EPI?MAIN:EPI;
  auto fn=ol_projection<BN,BK,ST,PIPE,PACK,HINT,TW>;
  static bool configured=false;
  if(!configured) {
    int rc=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM);if(rc)return rc;
    configured=true;
  }
  if(query) {cudaFuncAttributes a;int rc=cudaFuncGetAttributes(&a,fn);return rc?-rc:a.numRegs*10000+a.localSizeBytes;}
  CUtensorMap tw,tx;
  int rc=ol_map(&tw,(void*)w,8192,64,BK,PACK==2);if(rc)return rc;
  rc=ol_map(&tx,(void*)x,m,T,BK);if(rc)return rc;
  fn<<<dim3(8192/R,(m+T-1)/T),128*TW+((PIPE&2)?128:0),SMEM,(cudaStream_t)stream>>>(tw,tx,(const uint8_t*)w,(__nv_bfloat16*)out,m);
  return int(cudaGetLastError());
}
int ol_launch(int id,int64_t x,int64_t w,int64_t out,int m,int64_t stream,int query) {
  switch(id) {case 0: return ol_launch_t<128,128,4,3,1,1,1>(x,w,out,m,stream,query);
case 1: return ol_launch_t<128,64,5,3,1,1,2>(x,w,out,m,stream,query); default:return -99;}
}
"""
    _CPP = ("#include <stdint.h>\n#include <pybind11/pybind11.h>\n"
            "int routertail_small(int64_t,int64_t,int64_t,int,int,int,int64_t,int);\n"
            "int routertail_c12(int64_t,int64_t,int64_t,int,int,int,int64_t,int,int,int64_t);\n"
            "int ol_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);\n")
    RT_CFG = {8: (16, 8), 16: (16, 8), 32: (16, 4)}  # router m -> (S, WK) cluster size, warps per CTA
    OL_CFG = {128: 0}  # o m -> instantiated config: (128,128,4,3,1,1) (config 1, (128,64,5,3,1,1,2), measured no gain at m = 256)
    _ext = []

    _RT12_AT = _RT_CUDA.index("// ---- c12:")
    _RT_BASE, _RT12_SRC = _RT_CUDA[:_RT12_AT], _RT_CUDA[_RT12_AT:]
    _CPP_BASE = _CPP.replace("int routertail_c12(int64_t,int64_t,int64_t,int,int,int,int64_t,int,int,int64_t);\n", "")

    def ext():
        global _NB_C12
        if not _ext:
            try:
                if not RT12:
                    raise RuntimeError('RT12 disabled')
                _ext.append(load_inline("okc_nb_v2", _CPP, cuda_sources=[_RT_CUDA, _OL_CUDA],
                                        functions=["routertail_small", "ol_launch", "routertail_c12"],
                                        extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a"],
                                        no_implicit_headers=True, with_pytorch_error_handling=False))
                _NB_C12 = True
            except Exception as error:  # the c12 router kernel failed to build: the original extension, no c12 path
                print("RT12 off: %s" % str(error).splitlines()[-1][:200], flush=True)
                _NB_C12 = False
                _ext.append(load_inline("okc_nb_v1", _CPP_BASE, cuda_sources=[_RT_BASE, _OL_CUDA],
                                        functions=["routertail_small", "ol_launch"],
                                        extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a"],
                                        no_implicit_headers=True, with_pytorch_error_handling=False))
        return _ext[0]

    def _rt_pack(w, s, wk):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)

        def units(v):
            v = v.view(16, 2, 8, 128, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7).reshape(16, 128, 32, 4, 8)
            return v.permute(0, 1, 3, 2, 4)
        data = torch.cat([units(hi), units(lo)], dim=2).contiguous()
        ku = 128 // s // wk
        return data.view(16, s, wk, ku, 8, 32, 8).permute(3, 1, 0, 2, 4, 5, 6).contiguous()

    RT12 = False   # E240-E24x: the c12 router kernel is exact but 0.8-1.9 us slower than the raw routertail on Modal and on the judge's bake
    RT12_CAP = 16   # words per (CTA, warp) escape record (word 0 = count)

    def _rt12_pack(r, s, wk):
        """lossless 12-bit planes of the router's hi and lo BF16, in routertail's lane layout, + per-CTA escape tables.
        Returns (planes int32 tensor, etab int32 [16*S, CAP, 2], bqh, bql) or None."""
        dev = r.device
        hi = r.to(torch.bfloat16)
        lo = (r - hi.float()).to(torch.bfloat16)
        ku = 128 // s // wk

        def tr(v):  # [256, 8192] -> [rg, chunk, lane, q, i] (routertail's fragment order)
            return v.reshape(16, 2, 8, 128, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7).reshape(16, 128, 32, 4, 8)
        idr = tr(torch.arange(256, device=dev, dtype=torch.int32)[:, None].expand(256, 8192).contiguous())
        idk = tr(torch.arange(8192, device=dev, dtype=torch.int32)[None, :].expand(256, 8192).contiguous())
        planes, bases, ents = [], [], []
        lanei = torch.arange(32, device=dev).view(1, 1, 32, 1, 1).expand(16, 128, 32, 4, 8)
        qi = torch.arange(4, device=dev).view(1, 1, 1, 4, 1).expand(16, 128, 32, 4, 8)
        ii = torch.arange(8, device=dev).view(1, 1, 1, 1, 8).expand(16, 128, 32, 4, 8)
        for h, v in enumerate((hi, lo)):
            bits = tr(v.contiguous().view(torch.int16).reshape(256, 8192)).int() & 0xFFFF
            low, hib = bits & 0xFF, bits >> 8
            mag, sg = hib & 0x7F, hib >> 7
            cnt = torch.bincount(mag.flatten(), minlength=128).cpu()
            win = [int(cnt[b:b + 8].sum()) for b in range(121)]
            b0 = max(range(121), key=lambda b: win[b])
            esc = (mag < b0) | (mag > b0 + 7)
            nib = ((mag - b0).clamp(0, 7) | (sg << 3)).long()
            lw = low.reshape(16, 128, 32, 8, 4).long()
            lwords = lw[..., 0] | (lw[..., 1] << 8) | (lw[..., 2] << 16) | (lw[..., 3] << 24)          # [..., 8]
            nb_ = nib.reshape(16, 128, 32, 4, 2, 4)
            byt = nb_[..., 0, :] | (nb_[..., 1, :] << 4)                                                # [..., q, j]
            cwords = byt[..., 0] | (byt[..., 1] << 8) | (byt[..., 2] << 16) | (byt[..., 3] << 24)       # [..., 4]
            x12 = torch.cat([lwords, cwords], -1)                                                       # [rg, chunk, lane, 12]
            x12 = torch.where(x12 >= 2 ** 31, x12 - 2 ** 32, x12).int()
            planes.append(x12.reshape(16, 128, 32, 3, 4).permute(0, 1, 3, 2, 4))                       # [rg, chunk, part, lane, 4]
            bases.append(b0)
            sel = esc
            rgi = torch.arange(16, device=dev).view(16, 1, 1, 1, 1).expand(16, 128, 32, 4, 8)[sel].long()
            chk = torch.arange(128, device=dev).view(1, 128, 1, 1, 1).expand(16, 128, 32, 4, 8)[sel].long()
            ln, q_, i_ = lanei[sel].long(), qi[sel].long(), ii[sel].long()
            cta = rgi * s + chk // (wk * ku)
            warp = (chk // ku) % wk
            kk = chk % ku
            ent = ln | (q_ << 5) | ((i_ // 2) << 7) | ((i_ % 2) << 9) | (h << 10) | (kk << 11) | (bits[sel].long() << 16)
            ents.append((cta * wk + warp, ent))
        data = torch.stack(planes, 2)                                                                   # [rg, chunk, h, part, lane, 4]
        data = data.reshape(16, s, wk, ku, 2, 3, 32, 4).permute(3, 1, 0, 2, 4, 5, 6, 7).contiguous()
        key = torch.cat([e[0] for e in ents]); ent = torch.cat([e[1] for e in ents])
        o = torch.argsort(key, stable=True)
        key, ent = key[o], ent[o]
        rank = torch.arange(len(key), device=dev) - torch.searchsorted(key, key)
        if len(key) and int(rank.max()) + 1 >= RT12_CAP:
            return None
        et = torch.zeros((16 * s * wk, RT12_CAP), dtype=torch.long, device=dev)
        et[key, rank + 1] = ent
        et[:, 0] = torch.bincount(key, minlength=16 * s * wk)
        et = torch.where(et >= 2 ** 31, et - 2 ** 32, et).int().contiguous()
        return data, et, bases[0], bases[1], len(key)

    def _ol_pack(weight):
        columns = torch.arange(64, device=weight.device)[None, :] ^ ((torch.arange(64, device=weight.device)[:, None] & 7) * 8)
        tiled = weight.reshape(128, 64, 16, 64).permute(0, 2, 1, 3)
        return tiled.gather(-1, columns.expand(128, 16, 64, 64)).contiguous()

    def _rt(x, st, pin=None):
        m = x.shape[0]
        s, wk = RT_CFG[pin or m]   # v6: off-pin m runs the pin's instantiation (the kernel takes any m up to its MT * 8 rows)
        out = torch.empty((m, 256), device=x.device, dtype=torch.float32)
        rc = ext().routertail_small(st["rt", s, wk].data_ptr(), x.data_ptr(), out.data_ptr(), m, s, wk,
                                    torch.cuda.current_stream().cuda_stream, 0)
        if rc:
            raise RuntimeError(f"routertail launch failed: {rc}")
        return out

    def _ol(x, st):
        out = torch.empty((x.shape[0], 8192), device=x.device, dtype=torch.bfloat16)
        rc = ext().ol_launch(OL_CFG.get(x.shape[0], 0), x.data_ptr(), st["ol"].data_ptr(), out.data_ptr(), x.shape[0],
                             torch.cuda.current_stream().cuda_stream, 0)
        if rc:
            raise RuntimeError(f"olayer launch failed: {rc}")
        return out

    def _rt12(x, st, c):
        m = x.shape[0]
        s, wk = RT_CFG[m]
        data, et, bqh, bql = c
        out = torch.empty((m, 256), device=x.device, dtype=torch.float32)
        rc = ext().routertail_c12(data.data_ptr(), x.data_ptr(), out.data_ptr(), m, s, wk, et.data_ptr(), bqh, bql,
                                  torch.cuda.current_stream().cuda_stream)
        if rc:
            raise RuntimeError(f"routertail c12 launch failed: {rc}")
        return out

    def _rt12_bake(st, r):
        """per m: validate the c12 kernel against the raw one and time both from a cold L2 (graph replays); use c12 only if
        it is >= 2% faster"""
        dev = r.device
        flush = torch.empty(64 << 20, dtype=torch.float32, device=dev)
        gen = torch.Generator(device=dev).manual_seed(77)
        st["rt12_use"] = {}
        packs = {}
        for m in RT_CFG:
            s, wk = RT_CFG[m]
            if (s, wk) not in packs:
                pk = _rt12_pack(r, s, wk)
                if pk is None:
                    print("RT12 pack declined (escape table overflow) for S%d WK%d" % (s, wk), flush=True)
                packs[(s, wk)] = pk
            pk = packs[(s, wk)]
            if pk is None:
                continue
            c = pk[:4]
            x = torch.randn((m, 8192), generator=gen, device=dev).to(torch.bfloat16)
            x[:, ::97] *= 30.0
            ref = _rt(x, st)
            got = _rt12(x, st, c)
            torch.cuda.synchronize()
            err = float((got - ref).norm() / ref.norm())
            if not err < 1e-5:
                print("RT12 m=%d rejected: rel err %.2e" % (m, err), flush=True)
                continue
            gs = []
            for fn in (lambda: _rt(x, st), lambda: _rt12(x, st, c)):
                fn(); torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    fn()
                gs.append(g)
            ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(2)]
            ts = [[], []]
            for rep in range(41):
                for i in ((0, 1) if rep % 2 == 0 else (1, 0)):
                    flush.sum()
                    ev[i][0].record(); gs[i].replay(); ev[i][1].record()
                    torch.cuda.synchronize()
                    if rep:
                        ts[i].append(ev[i][0].elapsed_time(ev[i][1]) * 1e3)
            t0, t1 = sorted(ts[0])[len(ts[0]) // 2], sorted(ts[1])[len(ts[1]) // 2]
            use = t1 < 0.98 * t0
            print("RT12 m=%d raw %.2f c12 %.2f (err %.1e, %d escapes) -> %s" % (m, t0, t1, err, pk[4], "c12" if use else "raw"), flush=True)
            if use:
                st["rt12_use"][m] = c
                st["rt12_keep"] = st.get("rt12_keep", []) + [c]

    def prepare(weights):
        ext()
        r = weights["router"].float().contiguous()
        st = {("rt",) + c: _rt_pack(r, *c) for c in set(RT_CFG.values())}
        st["ol"] = _ol_pack(weights["o"].contiguous())
        for m in RT_CFG:  # first launches set the kernel attributes, outside capture
            _rt(torch.zeros((m, 8192), device=r.device, dtype=torch.bfloat16), st)
        for m in OL_CFG:
            _ol(torch.zeros((m, 1024), device=r.device, dtype=torch.bfloat16), st)
        torch.cuda.synchronize()
        st["rt12_use"] = {}
        if RT12 and _NB_C12:
            try:
                _rt12_bake(st, r)
            except Exception as error:  # the raw routertail stays on any failure
                st["rt12_use"] = {}
                print("RT12 bake failed: %s" % str(error).splitlines()[-1][:200], flush=True)
            finally:
                torch.cuda.synchronize()
        return st

    def kernel_fn(data, state):
        x = data["x"]
        layer, m = data["config"]["layer"], x.shape[0]
        assert x.is_contiguous()
        if layer == "router" and m in RT_CFG:
            c = state["rt12_use"].get(m)
            return _rt(x, state) if c is None else _rt12(x, state, c)
        if layer == "router" and 1 < m <= 32:   # v6: off-pin router m takes the pin's cluster config (8 / 16 / 32 rows)
            return _rt(x, state, 8 if m <= 8 else 16 if m <= 16 else 32)
        assert layer == "o"   # v6: ol_projection masks the rows past m (x tensor map has m rows, store guarded), any m
        return _ol(x, state)

    ext()  # build at import, outside prepare()'s budget
    return {"prepare": prepare, "kernel_fn": kernel_fn}


def _make_ag():
    # Pinned cuBLASLt plans for the large-m GEMMs, adapted from github.com/alygg77/okc-muse-spark-kernels
    # kernels/muse_bf16_proj.py (Apache-2.0): the _KLargeLt ctypes wrapper, the per-thread CUDA-context bind and the
    # pins (algo 66 tile / custom option / cluster per (layer, m)) come from there, which credits Matsvei (kernarc user
    # 'matvey') for the bind helper and the pins. Used here for the m = 256 cells (and m = 128 pooler when routed).
    import ctypes as C
    import threading

    import torch

    class _Ptr(C.c_void_p):
        def __deepcopy__(self, memo):
            return self

    P, I, SZ = _Ptr, C.c_int, C.c_size_t
    TLS = threading.local()
    CUDA = []

    def ctx_bind(x):
        if getattr(TLS, "device", None) == x.device.index:
            return
        if not CUDA:
            cu = C.CDLL("libcuda.so.1")
            for name, args in (("cuCtxGetCurrent", [C.POINTER(C.c_void_p)]),
                               ("cuDeviceGet", [C.POINTER(C.c_int), C.c_int]),
                               ("cuDevicePrimaryCtxRetain", [C.POINTER(C.c_void_p), C.c_int]),
                               ("cuCtxSetCurrent", [C.c_void_p])):
                fn = getattr(cu, name)
                fn.argtypes, fn.restype = args, C.c_int
            CUDA.append(cu)
        cu = CUDA[0]
        ctx = C.c_void_p()
        cu.cuCtxGetCurrent(C.byref(ctx))
        if not ctx.value:
            dev, primary = C.c_int(), C.c_void_p()
            if cu.cuDeviceGet(C.byref(dev), x.device.index or 0) == 0 and cu.cuDevicePrimaryCtxRetain(C.byref(primary), dev) == 0:
                cu.cuCtxSetCurrent(primary)
        TLS.device = x.device.index

    class Algo(C.Structure):
        _fields_ = [("data", C.c_uint64 * 8)]

    class Heur(C.Structure):
        _fields_ = [("algo", Algo), ("workspaceSize", C.c_size_t), ("state", C.c_int),
                    ("wavesCount", C.c_float), ("reserved", C.c_int * 4)]

    def load_lib():
        for name in ("libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so"):
            try:
                return C.CDLL(name)
            except OSError:
                pass
        raise RuntimeError("no libcublasLt")

    class Lt:
        def __init__(self):
            self.lib = load_lib()
            sigs = {
                "cublasLtCreate": [C.POINTER(P)],
                "cublasLtMatmulDescCreate": [C.POINTER(P), I, I],
                "cublasLtMatmulDescSetAttribute": [P, I, P, SZ],
                "cublasLtMatrixLayoutCreate": [C.POINTER(P), I, C.c_uint64, C.c_uint64, C.c_int64],
                "cublasLtMatrixLayoutSetAttribute": [P, I, P, SZ],
                "cublasLtMatmul": [P] * 12 + [C.POINTER(Algo), P, SZ, P],
                "cublasLtMatmulAlgoInit": [P, I, I, I, I, I, I, I, C.POINTER(Algo)],
                "cublasLtMatmulAlgoConfigSetAttribute": [C.POINTER(Algo), I, P, SZ],
                "cublasLtMatmulAlgoCheck": [P, P, P, P, P, P, C.POINTER(Algo), C.POINTER(Heur)],
            }
            for name, argtypes in sigs.items():
                fn = getattr(self.lib, name)
                fn.argtypes, fn.restype = argtypes, I
            self.handle = P()
            self.call("cublasLtCreate", C.byref(self.handle))
            self.desc = {}
            for mode, transpose in (("tn", 1), ("nn", 0)):
                desc = P()
                self.call("cublasLtMatmulDescCreate", C.byref(desc), 68, 0)
                op = I(transpose)
                self.call("cublasLtMatmulDescSetAttribute", desc, 3, C.byref(op), C.sizeof(op))
                self.desc[mode] = desc
            self.workspace = torch.empty(0, device="cuda", dtype=torch.uint8)
            self.required_workspace = 0
            self.alpha, self.beta = C.c_float(1), C.c_float(0)

        def __deepcopy__(self, memo):
            clone = self.__class__.__new__(self.__class__)
            clone.lib, clone.handle, clone.desc = self.lib, self.handle, dict(self.desc)
            clone.workspace = self.workspace.clone()
            clone.required_workspace = self.required_workspace
            clone.alpha, clone.beta = self.alpha, self.beta
            return clone

        def call(self, name, *args):
            status = getattr(self.lib, name)(*args)
            if status:
                raise RuntimeError(f"{name} status {status}")

        def layouts(self, mode, m, n, k):
            # (rows, cols, ld, batch stride); tn: W [n, k] row-major as a column-major k x n, x as k x m, out as n x m
            specs = ((k, n, k, k * n), (k, m, k, 0), (n, m, n, n))
            if mode == "nn":
                specs = ((n, k, n, n),) + specs[1:]  # W^T [k, n] row-major as a column-major n x k
            elif mode == "rownn":
                specs = ((m, k, k, 0), (k, n, n, 0), (m, n, n, 0))  # row-major x [m, k] @ W^T [k, n] -> [m, n]
            out = []
            for rows, cols, ld, stride in specs:
                desc = P()
                self.call("cublasLtMatrixLayoutCreate", C.byref(desc), 14, rows, cols, ld)
                if mode == "rownn":
                    order = I(1)
                    self.call("cublasLtMatrixLayoutSetAttribute", desc, 1, C.byref(order), C.sizeof(order))
                count, offset = I(1), C.c_int64(stride)
                self.call("cublasLtMatrixLayoutSetAttribute", desc, 5, C.byref(count), C.sizeof(count))
                self.call("cublasLtMatrixLayoutSetAttribute", desc, 6, C.byref(offset), C.sizeof(offset))
                out.append(desc)
            return out

        def plan(self, mode, m, n, k, cfg):
            layouts = self.layouts(mode, m, n, k)
            algo = Algo()
            self.call("cublasLtMatmulAlgoInit", self.handle, 68, 0, 14, 14, 14, 14, 66, C.byref(algo))
            for name, attr, ctype in (("tile", 1, C.c_uint32), ("splitk", 2, C.c_int32), ("red", 3, C.c_uint32),
                                      ("swz", 4, C.c_uint32), ("custom", 5, C.c_uint32), ("stages", 6, C.c_uint32),
                                      ("cluster", 8, C.c_uint16)):
                value = ctype(cfg[name])
                status = self.lib.cublasLtMatmulAlgoConfigSetAttribute(C.byref(algo), attr, C.byref(value), C.sizeof(value))
                if status:
                    raise RuntimeError(f"ag config {name} status {status}")
            result = Heur()
            status = self.lib.cublasLtMatmulAlgoCheck(self.handle, self.desc["nn" if mode == "rownn" else mode], *layouts,
                                                      layouts[2], C.byref(algo), C.byref(result))
            if status:
                raise RuntimeError(f"ag AlgoCheck status {status}")
            self.required_workspace = max(self.required_workspace, int(result.workspaceSize))
            return mode, layouts, algo

        def run(self, x, weight, n, plan):
            ctx_bind(x)
            mode, (a, b, c), algo = plan
            out = torch.empty((x.shape[0], n), device=x.device, dtype=torch.bfloat16)
            ap = P(x.data_ptr() if mode == "rownn" else weight.data_ptr())
            bp = P(weight.data_ptr() if mode == "rownn" else x.data_ptr())
            self.call("cublasLtMatmul", self.handle, self.desc["nn" if mode == "rownn" else mode], C.byref(self.alpha), ap, a,
                      bp, b, C.byref(self.beta), P(out.data_ptr()), c, P(out.data_ptr()), c,
                      C.byref(algo), P(self.workspace.data_ptr()), self.workspace.numel(),
                      P(torch.cuda.current_stream().cuda_stream))
            return out

    def cfg_of(tile, custom, cluster):
        return {"tile": tile, "stages": 35, "splitk": 1, "red": 0, "swz": 0, "custom": custom, "cluster": cluster}

    # (layer, m) -> (layout mode, tile, custom option, cluster)
    PINS = {("qkv_gate", 256): ("tn", 124, 0, 2), ("pooler", 256): ("rownn", 18, 0, 2), ("o", 256): ("tn", 20, 6, 2),
            ("unpooler", 256): ("rownn", 20, 4, 2), ("pooler", 128): ("nn", 15, 4, 2)}
    TRANSPOSED = {("unpooler", 256), ("pooler", 256), ("pooler", 128)}

    def prepare(weights):
        lt = Lt()
        st = {"lt": lt, "plans": {}, "w": {}}
        wt = {}
        for key, (mode, tile, custom, cluster) in PINS.items():
            layer, m = key
            w = weights[layer]
            n, k = w.shape
            try:
                plan = lt.plan(mode, m, n, k, cfg_of(tile, custom, cluster))
                if key in TRANSPOSED:
                    if layer not in wt:
                        wt[layer] = w.t().contiguous()
                    weight = wt[layer]
                else:
                    weight = w
                lt.workspace = torch.empty(max(lt.required_workspace, lt.workspace.numel()), device="cuda", dtype=torch.uint8)
                g = torch.Generator(device=w.device).manual_seed(7 + m)
                x = torch.randn((m, k), generator=g, device=w.device).to(torch.bfloat16)
                got = lt.run(x, weight, n, plan)
                torch.cuda.synchronize()
                ref = x.float() @ w.float().T
                if bool(torch.isfinite(got).all()) and float((got.float() - ref).norm() / ref.norm()) < 1e-2:
                    st["plans"][key] = plan
                    st["w"][key] = weight
                else:
                    print(f"AG pin {key} failed the self-test", flush=True)
            except Exception as error:  # a pin the library rejects just routes that cell back to the default
                print(f"AG pin {key} unavailable: {str(error)[:120]}", flush=True)
        return st

    def kernel_fn(data, state):
        x = data["x"]
        key = (data["config"]["layer"], x.shape[0])
        return state["lt"].run(x, state["w"][key], data["config"]["n"], state["plans"][key])

    DIMS = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096)}

    def extra_plan(st, layer, m):
        """v6: the m = 256 pin's algorithm on the layouts of another m (cublasLtMatmulAlgoCheck decides if it is valid there);
        True when st['plans'][(layer, m)] exists afterwards. Only called from prepare (the workspace may grow)."""
        key = (layer, m)
        if key in st["plans"]:
            return True
        if (layer, 256) not in st["plans"] or (layer, 256) not in PINS or layer not in DIMS:
            return False
        mode, tile, custom, cluster = PINS[(layer, 256)]
        n, k = DIMS[layer]
        try:
            lt = st["lt"]
            plan = lt.plan(mode, m, n, k, cfg_of(tile, custom, cluster))
            if lt.required_workspace > lt.workspace.numel():
                lt.workspace = torch.empty(lt.required_workspace, device="cuda", dtype=torch.uint8)
        except Exception:
            return False
        st["plans"][key] = plan
        st["w"][key] = st["w"][(layer, 256)]
        return True

    return {"prepare": prepare, "kernel_fn": kernel_fn, "has": lambda state, key: key in state["plans"], "extra_plan": extra_plan}


_TABLE = {'o:1': 'vjeux', 'o:128': 'nb', 'o:16': 'chenwainuo', 'o:256': 'ag', 'o:32': 'vjeux', 'o:8': 'chenwainuo', 'pooler:1': 'vjeux', 'pooler:128': 'chenwainuo', 'pooler:16': 'chenwainuo', 'pooler:256': 'ag', 'pooler:32': 'vjeux', 'pooler:8': 'chenwainuo', 'qkv_gate:1': 'vjeux', 'qkv_gate:128': 'chenwainuo', 'qkv_gate:16': 'vjeux', 'qkv_gate:256': 'ag', 'qkv_gate:32': 'chenwainuo', 'qkv_gate:8': 'chenwainuo', 'router:1': 'vjeux', 'router:128': 'vjeux', 'router:16': 'nb', 'router:256': 'vjeux', 'router:32': 'nb', 'router:8': 'nb', 'unpooler:1': 'vjeux', 'unpooler:128': 'chenwainuo', 'unpooler:16': 'vjeux', 'unpooler:256': 'ag', 'unpooler:32': 'vjeux', 'unpooler:8': 'chenwainuo'}
_DEFAULT = 'chenwainuo'
_K = {'chenwainuo': _make_chenwainuo(), 'vjeux': _make_vjeux(), 'nb': _make_nb(), 'ag': _make_ag()}


import math as _v6_math
import time as _v6_time

import torch as _v6_torch

# ======================================================================================================================
# v6: nearby m. The 6 scored token counts (1, 8, 16, 32, 128, 256) take exactly the v5 path (_TABLE). Every other m in
# 2..384 used to run `_DEFAULT` = chenwainuo -> SGLang's own cuBLAS call (speedup 1.00; audit_nearby.md). Now prepare() builds a
# route table for those m: for each (layer, range of m) the candidates are the pinned kernels of the range's bucket (vjeux sk2 /
# rk, nb routertail / ol_projection take any m up to their pin), chenwainuo's rq plans of the m=32 / 128 pins (the kernel builds
# its tensor maps from the real m: TMA zero-fills the x rows past m and does not write the out rows past m) and the m=256 cuBLASLt
# pin on the layouts of the real m. Each candidate is validated (rel L2 vs an fp32 / fp64 reference, the probe rows, normal and
# outlier inputs) at EVERY m it serves and timed (cold-L2 graph replays) at a few m of its range against the default; a
# candidate that does not beat the default by V6_GAIN stays out, so a bad pick can only cost the default's speed.
V6_ON = True
V6_BUDGET_S = 60.0   # timing + validation budget of the route build (v5's prepare is ~72 s; the limit is 300 s)
V6_GAIN = 0.97       # a candidate must run in under 97% of the default's time at that m (cold-L2 graph replays) to earn the route
V6_REPS = 14
_V6_LAYERS = ("qkv_gate", "o", "pooler", "unpooler", "router")
_V6_N = {"qkv_gate": 2304, "o": 8192, "pooler": 4096, "unpooler": 8192, "router": 256}
_V6_K = {"qkv_gate": 8192, "o": 1024, "pooler": 8192, "unpooler": 4096, "router": 8192}
_V6_RANGES = ((9, 15), (17, 31), (33, 127), (129, 255), (257, 384), (2, 7))   # order = priority if the budget runs out
_V6_DETAIL = (20, 28, 33, 40, 48, 64, 80, 100, 127, 129, 160, 200, 255, 257, 300, 384)   # m whose candidate times are logged
_V6_LOG = []


def _v6_say(text):
    _V6_LOG.append(text)


def _v6_n(layer, m):
    return 1024 if (layer == "unpooler" and m <= 8) else _V6_N[layer]


def _v6_up(m):
    return 1 if m <= 1 else 8 if m <= 8 else 16 if m <= 16 else 32 if m <= 32 else 128 if m <= 128 else 256 if m <= 256 else 0


def _v6_cands(layer, m, state):
    """[(tag, route)] for an off-pin (layer, m): the default first, then the other ways to run it."""
    out = [("def", ("def",))]
    up, n = _v6_up(m), _v6_n(layer, m)
    if up and up != m:
        nm = _TABLE.get("%s:%d" % (layer, up))
        if nm in ("vjeux", "nb"):  # the backend pinned at the next bucket up takes any m below its pin
            out.append((nm[0] + str(up), ("pin", nm)))
    if layer == "o" and m > 16 and all(r != ("pin", "nb") for _, r in out):
        out.append(("nbol", ("pin", "nb")))
    if m > 8:
        for key, p in sorted(state["chenwainuo"].get("rq", {}).items()):
            if isinstance(key, tuple) and key[0] == layer and key[1] > m and p["n"] == n:
                out.append(("rq%d" % key[1], ("rq", key[1])))
    if layer != "router" and m > 16 and _K["ag"]["extra_plan"](state["ag"], layer, m):
        out.append(("ag", ("ag",)))
    if layer != "router" and m > 8 and (layer + "_t") in state["chenwainuo"]:
        out.append(("tT", ("tT",)))   # cuBLAS on the K-major copy of W: another heuristic pick
    return out


def _v6_call(route, data, state):
    kind = route[0]
    if kind == "def":
        return _K[_DEFAULT]["kernel_fn"](data, state[_DEFAULT])
    if kind == "pin":
        return _K[route[1]]["kernel_fn"](data, state[route[1]])
    if kind == "ag":
        return _K["ag"]["kernel_fn"](data, state["ag"])
    cw, x = state["chenwainuo"], data["x"]
    if kind == "tT":
        return _v6_torch.mm(x, cw[data["config"]["layer"] + "_t"])
    p = cw["rq"][(data["config"]["layer"], route[1])]
    out = _v6_torch.empty((x.shape[0], p["n"]), dtype=_v6_torch.bfloat16, device=x.device)
    r = _K["chenwainuo"]["rqcall"](cw, p, x.contiguous(), out)
    if r != 0:
        raise RuntimeError("rq launch %d" % r)
    return out


def _v6_x(m, K, dev, gen, outliers):
    """the task's input: standard normal bf16 (outliers: 1% of the channels x30) with a one-hot probe row at the first and last row"""
    x = _v6_torch.randn((m, K), generator=gen, device=dev)
    if outliers:
        x[:, _v6_torch.randperm(K, generator=gen, device=dev)[: max(1, K // 100)]] *= 30
    for j, r in enumerate(sorted({0, m - 1})):
        x[r] = 0
        x[r, (r * 977 + 31 * j + 5) % K] = 64.0 if j == 0 else -32.0
    return x.to(_v6_torch.bfloat16)


def _v6_ok(route, layer, m, state, ref_w, gen, dev):
    """the route's output at (layer, m) on normal and outlier inputs: shape, dtype, rel L2 and the probe rows against the reference"""
    K, n = _V6_K[layer], _v6_n(layer, m)
    router = layer == "router"
    tol, ptol = (3e-5, 3e-4) if router else (6e-3, 1e-3)
    rows = sorted({0, m - 1})
    try:
        for outliers in (False, True):
            x = _v6_x(m, K, dev, gen, outliers)
            out = _v6_call(route, {"x": x, "config": {"m": m, "n": n, "k": K, "layer": layer}}, state)
            if tuple(out.shape) != (m, n) or out.dtype != (_v6_torch.float32 if router else _v6_torch.bfloat16):
                return False
            w = ref_w[layer][:n]
            ref = (x.double() @ w.T) if router else (x.float() @ w.T)
            o = out.double() if router else out.float()
            err = float((o - ref).norm() / ref.norm())
            pr = float(((o[rows] - ref[rows]).norm(dim=1) / ref[rows].norm(dim=1)).max())
            if not (err < tol and pr < ptol):
                _v6_say("V6|bad|%s m%d %s err %.2e probe %.2e" % (layer, m, route[0], err, pr))
                return False
        return True
    except Exception as error:  # noqa: BLE001  (a launch failure just rules the candidate out)
        _v6_say("V6|exc|%s m%d %s %s" % (layer, m, route[0], str(error).replace("\n", " ")[:100]))
        return False


def _v6_bench(routes, data, state, flush, sink, reps):
    """median cold-L2 graph-replay time (us) of every route (None when it fails), replays interleaved in alternating order"""
    gs = []
    for r in routes:
        try:
            _v6_call(r, data, state)
            _v6_torch.cuda.synchronize()
            g = _v6_torch.cuda.CUDAGraph()
            with _v6_torch.cuda.graph(g):
                _v6_call(r, data, state)
            gs.append(g)
        except Exception as error:  # noqa: BLE001
            _v6_say("V6|exc|bench %s %s" % (r[0], str(error).replace("\n", " ")[:100]))
            gs.append(None)
    live = [i for i, g in enumerate(gs) if g is not None]
    ev = [(_v6_torch.cuda.Event(enable_timing=True), _v6_torch.cuda.Event(enable_timing=True)) for _ in gs]
    ts = [[] for _ in gs]
    for rep in range(reps + 3):
        for i in (live if rep % 2 == 0 else live[::-1]):
            _v6_torch.sum(flush, dim=0, out=sink)
            ev[i][0].record()
            gs[i].replay()
            ev[i][1].record()
            _v6_torch.cuda.synchronize()
            if rep >= 3:
                ts[i].append(ev[i][0].elapsed_time(ev[i][1]) * 1e3)
    del gs
    return [sorted(v)[len(v) // 2] if v else None for v in ts]


def _v6_routes(state, weights):
    t0 = _v6_time.perf_counter()
    dev = weights["o"].device
    gen = _v6_torch.Generator(device=dev).manual_seed(6061)
    ref_w = {l: (weights[l].double() if l == "router" else weights[l].float()) for l in _V6_LAYERS}
    flush = _v6_torch.zeros(65536, 1024, dtype=_v6_torch.float32, device=dev)   # 256 MB read sweep (5x the L2) before every replay
    sink = _v6_torch.empty(1024, dtype=_v6_torch.float32, device=dev)
    routes, chosen, cells, skipped = {}, {l: {} for l in _V6_LAYERS}, 0, 0
    for lo, hi in _V6_RANGES:
        for layer in _V6_LAYERS:
            for m in range(lo, hi + 1):
                if "%s:%d" % (layer, m) in _TABLE:
                    continue
                if _v6_time.perf_counter() - t0 > V6_BUDGET_S:
                    skipped += 1
                    continue
                try:
                    cands = _v6_cands(layer, m, state)
                    if len(cands) < 2:
                        continue
                    K, n = _V6_K[layer], _v6_n(layer, m)
                    x = _v6_x(m, K, dev, gen, False)
                    data = {"x": x, "config": {"m": m, "n": n, "k": K, "layer": layer}}
                    ts = _v6_bench([r for _, r in cands], data, state, flush, sink, V6_REPS)
                    cells += 1
                    d = ts[0]
                    if m in _V6_DETAIL:
                        _v6_say("V6|D|%s|%d|def %.1f %s" % (layer, m, d or -1, " ".join(
                            "%s %s" % (c[0], "%.2f" % (t / d) if (t and d) else "x") for c, t in zip(cands[1:], ts[1:]))))
                    if not d:
                        continue
                    for t, tag, route in sorted((t, c[0], c[1]) for c, t in zip(cands[1:], ts[1:]) if t and t < V6_GAIN * d):
                        if not _v6_ok(route, layer, m, state, ref_w, gen, dev):
                            continue
                        if t > 0.9 * d:   # a small win: confirm it with a longer paired run before taking it
                            t2 = _v6_bench([("def",), route], data, state, flush, sink, 3 * V6_REPS)
                            if not (t2[0] and t2[1] and t2[1] < 0.985 * t2[0]):
                                continue
                        routes[(layer, m)] = route
                        chosen[layer][m] = (tag, t / d)
                        break
                except Exception as error:  # noqa: BLE001
                    _v6_say("V6|failed|%s m%d %s" % (layer, m, str(error).replace("\n", " ")[:120]))
    for layer in _V6_LAYERS:
        runs, last = [], None
        for m in sorted(chosen[layer]):
            tag, ratio = chosen[layer][m]
            if last and last[2] == tag and last[1] == m - 1:
                last[1] = m
                last[3].append(ratio)
            else:
                last = [m, m, tag, [ratio]]
                runs.append(last)
        _v6_say("V6|SUM|%s|%s" % (layer, " ".join("%d-%d:%s@%.2f" % (r[0], r[1], r[2], sum(r[3]) / len(r[3])) for r in runs)
                                  or "default everywhere"))
    _v6_say("V6|DONE|%d routes, %d cells timed, %d skipped, %.1f s" % (len(routes), cells, skipped, _v6_time.perf_counter() - t0))
    del ref_w, flush
    _v6_torch.cuda.empty_cache()
    return routes


def prepare(weights):
    import threading
    warm = []
    if _PRUNE:  # vjeux's extension compiles while chenwainuo prepares (nvcc subprocess)
        for k in _K.values():
            if k.get("warm"):
                def _bg(f=k["warm"]):
                    try:
                        f()
                    except Exception:
                        pass
                t = threading.Thread(target=_bg, daemon=True)
                t.start()
                warm.append(t)
    out = {}
    for name, k in _K.items():
        if name == "vjeux":
            for t in warm:
                t.join()
        out[name] = k["prepare"](weights)
    out["routes"] = {}
    if V6_ON:
        try:
            out["routes"] = _v6_routes(out, weights)
        except BaseException as error:  # noqa: BLE001  (any failure leaves the v5 behaviour)
            _v6_say("V6|FAILED|%s: %s" % (type(error).__name__, str(error).replace("\n", " ")[:160]))
            out["routes"] = {}
        print("\n".join(_V6_LOG), flush=True)
    return out


def kernel_fn(data, state):
    cfg = data["config"]
    name = _TABLE.get(f"{cfg['layer']}:{cfg['m']}")
    if name is None:  # off the scored token counts: the v6 route when prepare() found one, else the v5 default
        route = state.get("routes", {}).get((cfg["layer"], cfg["m"]))
        if route is not None:
            return _v6_call(route, data, state)
        name = _DEFAULT
    if name == "ag" and not _K["ag"]["has"](state["ag"], (cfg["layer"], cfg["m"])):
        name = _DEFAULT
    return _K[name]["kernel_fn"](data, state[name])

import ctypes as _CtxC
import threading as _CtxThreading

_CTX_TLS = _CtxThreading.local()
_CTX_API = None


def _ctx_api():
    global _CTX_API
    if _CTX_API is None:
        api = _CtxC.CDLL("libcuda.so.1")
        for name, args in (
            ("cuCtxGetCurrent", [_CtxC.POINTER(_CtxC.c_void_p)]),
            ("cuCtxGetDevice", [_CtxC.POINTER(_CtxC.c_int)]),
            ("cuDeviceGet", [_CtxC.POINTER(_CtxC.c_int), _CtxC.c_int]),
            ("cuDevicePrimaryCtxRetain", [_CtxC.POINTER(_CtxC.c_void_p), _CtxC.c_int]),
            ("cuCtxSetCurrent", [_CtxC.c_void_p]),
        ):
            f = getattr(api, name)
            f.argtypes, f.restype = args, _CtxC.c_int
        _CTX_API = api
    return _CTX_API


def _ctx_check(status, op):
    if status:
        raise RuntimeError(f"{op} failed with CUDA driver status {status}")


def _ctx_bind(x):
    index = x.device.index if x.device.index is not None else 0
    if getattr(_CTX_TLS, "device", None) == index:
        return
    api = _ctx_api()
    ctx = _CtxC.c_void_p()
    _ctx_check(api.cuCtxGetCurrent(_CtxC.byref(ctx)), "cuCtxGetCurrent")
    cur = _CtxC.c_int(-1)
    if ctx.value:
        _ctx_check(api.cuCtxGetDevice(_CtxC.byref(cur)), "cuCtxGetDevice")
    if not ctx.value or cur.value != index:
        dev, primary = _CtxC.c_int(), _CtxC.c_void_p()
        _ctx_check(api.cuDeviceGet(_CtxC.byref(dev), index), "cuDeviceGet")
        _ctx_check(api.cuDevicePrimaryCtxRetain(_CtxC.byref(primary), dev), "cuDevicePrimaryCtxRetain")
        _ctx_check(api.cuCtxSetCurrent(primary), "cuCtxSetCurrent")
    _CTX_TLS.device = index


_ctx_parent_kernel_fn = kernel_fn


def kernel_fn(data, state):
    _ctx_bind(data["x"])
    return _ctx_parent_kernel_fn(data, state)
