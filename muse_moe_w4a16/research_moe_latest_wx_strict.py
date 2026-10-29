# okc: problem=muse_moe_w4a16
# Adapted from lee101/openkernels commit d2057db, Apache-2.0.
# Copyright 2026 Lee Penkman. See OPENKERNELS_LICENSE.txt.
"""BF16 gate/up boundary variant of pinned v48; no judged results yet. v48: v47 + gk small-M variants (KS2 1: each CTA's W2 row tiles are its output rows, y stays in smem, 2 grid barriers like g1; PPS 2: one 4608 B box / barrier per two contiguous pieces), absent token slots read a zero row (no tensor-core toggling on padding columns), lean gk tails (y partial sums and per-route norms with all loads in flight, per-token route table, CTA row weights in smem, batched SwiGLU pass), host-side smem checks instead of device traps, SM clock in every diag line. v47: v46 + gk to M <= 32 (KS1 = 2: W13 K split over two CTA groups with x half-rows in smem, fp32 gate/up partials and a grid-wide SwiGLU pass; n-tiles up to 4 reduced 2 at a time), gk/g1 tails without serial L2 round trips (norm factors prefetched, partial sums loaded all at once), g1 load-path variants (P1: one 2304 B box per ring slot, 4 slots per warp; RT16: one reduction round at M1), fs candidates retired. v46: v45 + "gk" path for 3 <= M <= 16 (CUDA_K): g1's structure with all M activation rows in smem (B fragments by LDS, n-tiles of 8 tokens per dequant), 16 warps = tile groups of SW warps that split K of each tile (SW-way reduction behind a group-local named barrier), W2 K split over KS2 CTA groups with inter staged per group (chunked by whole experts when it does not fit) and fp32 y partials in global, then every CTA owns ~32 output rows: partial sums, sum-of-squares exchange, norm + combine (3 grid barriers, no atomics). g1 FENCE1 variant (one __threadfence before the grid barrier instead of one per reduction round); DQHI dropped. v45: v44 + "g1" decode path for M <= 2 (CUDA_G): every warp owns one 256-K slice of every weight row of its CTA (activation slice in registers, one 16-way smem reduction per tile), W13 over a contiguous tile range of all local experts -> inter in global -> grid barrier (the ring keeps streaming W2) -> W2 by output row tiles over all experts with full K (no fp32 atomics) -> per-route sum-of-squares exchange -> norm + combine of the CTA's rows. Tune on 3 judge-generator routings per bucket, f1s3 restored. v44: v43 + wx with N in steps of 8 (N8STEP: less padded tensor work), wx offered from M8, wait-fraction counters (WSTAT) in the diag. v43: wx with one ring of 4-chunk group slots (activation tiles loaded 4 chunks at a time, like the weights) and a 16-consumer-warp variant (4 warpgroups x 1 m64 block, 256-row units). v42: v41 with split weight / activation rings (4-chunk weight TMA slots, deeper weight stream) and an RPW 2 candidate (256-row units). v41: v37 + one-pass-per-expert wgmma RS expert GEMM (CUDA_X "wx": 128-row units over all of an expert's tokens, N <= 256, 64-K chunk slots of TMA weights + gathered pre-swizzled activations, CUTLASS-order RS pipeline) as a tuned candidate for M >= 32. v37: v36 + MT2 only for CTAs whose segments all have <= 8 tokens, cost weights 30/45/76 from H200 per-granule data, phase-A issue cursor without per-stage divisions, branch-free B prefetch. v36: v35 + MT2 (two W13 tiles per warp share activation loads) + wait/class diagnostics. v35: v34 + last-arriver combine option (LASTA: no grid barrier). v34: v30 + fused split-K path (fs) for M <= 32: each CTA owns ~1 32-column inter granule per active expert,
streams its W13 rows and the matching W2 column slice back to back (no W13->W2 grid dependency, no tile queue), fp32 red into
a self-cleaning accumulator, one grid barrier, per-token norm + combine. v30: per-M choice (autotuned in prepare on the judge GPU) between v28 and the ring path below with K split 1-8 and
lookahead-1 queue fetch. v29: M <= MBIG runs v28 (v17 per-warp rings, rotation-free dequant) on the v22 weight layout; M > MBIG runs CUDA_L (v22
CTA-cooperative ring, activations shared through smem, all W13 tiles queued before W2). One persistent kernel: routing -> W13 (SwiGLU fused) -> W2 (+post-norm stats) -> norm + weighted sum.
Weights repacked per 16-row tile into mma.sync A-fragment order with a bit layout that dequantizes
NVFP4 -> BF16 exactly in 11 integer ops per 8 values (v28: no rotations, so ptxas moves 4 of 5 shifts to IMAD on the FMA pipe); streamed by TMA bulk copies (cp.async fallback)."""
from typing import TypedDict
import torch
from torch.utils.cpp_extension import load_inline
import time
_T0 = 0.0


class Weights(TypedDict):
    w13: torch.Tensor
    w13_sf: torch.Tensor
    w13_gs: torch.Tensor
    w2: torch.Tensor
    w2_sf: torch.Tensor
    w2_gs: torch.Tensor
    post_norm_w: torch.Tensor


class Config(TypedDict):
    bs: int
    d_hidden: int
    d_expert: int
    n_experts: int
    local_start: int
    local_count: int
    top_k: int
    sharding: str


class Inputs(TypedDict):
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    config: Config


CUDA = '\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cuda_bf16.h>\n// Match baseline silu_and_mul: BF16 gate/up inputs, FP32 activation/product,\n// BF16 product store at each caller. Do not add a nonexistent SiLU tensor boundary.\n__device__ __forceinline__ float muse_swiglu_bf16(float gate, float up) {\n  gate = __bfloat162float(__float2bfloat16_rn(gate));\n  up = __bfloat162float(__float2bfloat16_rn(up));\n  return gate / (1.f + __expf(-gate)) * up;\n}\n\n#include <stdint.h>\n\n#define HD 4096\n#define NLOC 32\n#define STG 4608\n#define SPT 8\n#define T1E 512\n#define T2E 256\n#ifndef NTM\n#define NTM 4\n#endif\n#ifndef SCR\n#define SCR 1024\n#endif\n#define KSM 4\n#ifndef KSF\n#define KSF 0\n#endif\n#ifndef EARLYREL\n#define EARLYREL 1\n#endif\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900\n#define TMA 1\n#else\n#define TMA 0\n#endif\n\nstruct P {\n  const uint8_t* w1; const uint8_t* w2; const float* g1; const float* g2;\n  const __nv_bfloat16* x; const int* ids; const float* tw; const float* nw;\n  __nv_bfloat16* inter; __nv_bfloat16* y; float* ss; __nv_bfloat16* out; int* ctr; int* tcnt; float4* ws;\n  int M, lo; unsigned long long* prof;\n};\n#ifdef PROF\n__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }\n#define PSET(k, v) do { if (lane == 0) p.prof[gw * 16 + (k)] = (v); } while (0)\n#define PADD(k, v) do { if (lane == 0) p.prof[gw * 16 + (k)] += (v); } while (0)\n#else\n#define PSET(k, v)\n#define PADD(k, v)\n#endif\n\n__device__ __forceinline__ void cp16(uint32_t s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\\n" :: "r"(s), "l"(g)); }\n__device__ __forceinline__ void cpc() { asm volatile("cp.async.commit_group;\\n"); }\ntemplate <int N> __device__ __forceinline__ void cpw() { asm volatile("cp.async.wait_group %0;\\n" :: "n"(N)); }\n__device__ __forceinline__ uint4 lds(uint32_t a) { uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v; }\n__device__ __forceinline__ uint2 lds64(uint32_t a) { uint2 v; asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\\n" : "=r"(v.x), "=r"(v.y) : "r"(a)); return v; }\n__device__ __forceinline__ uint32_t bmul(uint32_t a, uint32_t b) { uint32_t d; asm("fma.rn.bf16x2 %0, %1, %2, %3;\\n" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u)); return d; }\n__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\\n"\n    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n__device__ __forceinline__ int ldacq(const int* p) { int v; asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\\n" : "=r"(v) : "l"(p) : "memory"); return v; }\n__device__ __forceinline__ void fence_ar() { asm volatile("fence.acq_rel.gpu;\\n" ::: "memory"); }\n__device__ __forceinline__ void gbar(int id, int n) { asm volatile("bar.sync %0, %1;\\n" :: "r"(id), "r"(n) : "memory"); }\n__device__ __forceinline__ uint32_t rotl(uint32_t q, int d) { return __funnelshift_l(q, q, d); }\n__device__ __forceinline__ uint32_t sc2(uint32_t w, uint32_t sel) { return __byte_perm(w, 0, sel) * 16u + 0x77007700u; }\n#if TMA\n__device__ __forceinline__ void mb_init(uint32_t b) { asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\\n" :: "r"(b)); }\n__device__ __forceinline__ void tma_ld(uint32_t dst, const void* src, uint32_t bar) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" :: "r"(bar), "n"(STG) : "memory");\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\\n" :: "r"(dst), "l"(src), "n"(STG), "r"(bar) : "memory");\n}\n__device__ __forceinline__ void mb_wait(uint32_t bar, uint32_t ph) {\n  asm volatile("{\\n.reg .pred p;\\nW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\\n@!p bra W;\\n}\\n" :: "r"(bar), "r"(ph) : "memory");\n}\n#endif\n\n#ifndef HEAVY1\n#define HEAVY1 1\n#endif\n#define QI (NLOC + 2)  // ctr[QI]: dynamic tile queue\n#define RED (2 * 2 * 32 * 16)  // per-warp split-K partial slots: 2 tile parities x NT <= 2 float4 per lane\nstruct Sm { int cnt[NLOC], off[NLOC], fill[NLOC], act[NLOC], na, ks, maxc; int ids[16][4]; };\n\n// Work = tiles from one global queue (all W13 tiles, then all W2 tiles, expert-major). A group of ks warps of one CTA\n// takes a tile together, each streaming SPT/ks K-stages; partials meet in smem; the finisher rotates per tile.\ntemplate <int S>\nstruct Ctx {\n  const P* p; Sm* sm; const uint16_t* list; uint32_t ring, bars; int lane, N1, N, gi, gj, ks;\n  int q; unsigned long long wcyc;\n  int pi, pp, ps;  // producer: group tile index, pass, stage\n  __device__ __forceinline__ int tid_of(int i) const { return ((volatile int*)sm->ids[gi])[i & 3]; }\n  __device__ __forceinline__ void item(int u, int& g2, int& e, int& mt) const {\n    if (u < N1) { g2 = 0; e = sm->act[u >> 9]; mt = u & 511; }\n    else { int v = u - N1; g2 = 1; e = sm->act[v >> 8]; mt = v & 255; }\n  }\n  __device__ __forceinline__ void issue(int slot) {\n    const int u = tid_of(pi);\n    if (u < N) {\n      int g2, e, mt; item(u, g2, e, mt);\n      const int span = SPT / ks, s = gj * span + ps;\n      const uint8_t* src = g2 ? p->w2 + ((size_t)((p->lo + e) * T2E + mt) * SPT + s) * STG\n                              : p->w1 + ((size_t)((p->lo + e) * T1E + mt) * SPT + s) * STG;\n#if TMA\n      if (lane == 0) tma_ld(ring + slot * STG, src, bars + slot * 8);\n#else\n      uint32_t d = ring + slot * STG + lane * 16; src += lane * 16;\n#pragma unroll\n      for (int k = 0; k < 9; k++) cp16(d + k * 512, src + k * 512);\n#endif\n      if (++ps == span) { ps = 0; if (++pp == (sm->cnt[e] + 8 * NTM - 1) / (8 * NTM)) { pp = 0; pi++; } }\n    }\n#if !TMA\n    cpc();\n#endif\n  }\n  __device__ __forceinline__ uint32_t acquire() {  // wait for stage q, refill the slot freed last iteration\n    const int slot = q % S;\n#ifdef PROF\n    long long c0 = clock64();\n#endif\n#if TMA\n    mb_wait(bars + slot * 8, (q / S) & 1);\n    __syncwarp();\n#else\n    cpw<S - 2>();\n#endif\n#ifdef PROF\n    wcyc += clock64() - c0;\n#endif\n#if !EARLYREL\n    issue((q + S - 1) % S);\n#endif\n    q++;\n    return ring + slot * STG + lane * 16;\n  }\n  __device__ __forceinline__ void release() {  // stage q - 1 consumed: refill its slot now, before any epilogue\n#if EARLYREL\n    __syncwarp();\n    issue((q - 1) % S);\n#endif\n  }\n};\n\ntemplate <int NT, int G2, int S>\n__device__ __forceinline__ bool run(Ctx<S>& C, int e, int mt, int pass, int ti, int s0, int s1) {\n  const P& p = *C.p;\n  const int lane = C.lane, g = lane >> 2, t = lane & 3;\n  const int cnt = C.sm->cnt[e], off = C.sm->off[e];\n  constexpr int CH = NT == 1 ? 4 : (NT == 2 ? 2 : 1);\n#ifndef BD1\n#define BD1 7\n#endif\n  constexpr int D = NT == 1 ? BD1 : (NT == 2 ? 3 : 1);  // B prefetch distance in chunks; (D + 1) divides 8 so slots stay static\n  const uint4* rp[NT];\n#pragma unroll\n  for (int n = 0; n < NT; n++) {\n    int i = pass * 8 * NTM + n * 8 + g; int r = C.list[off + (i < cnt ? i : 0)];\n    rp[n] = (const uint4*)((G2 ? p.inter + (size_t)r * HD : p.x + (size_t)(r >> 3) * HD) + t * 16);\n  }\n  uint4 B[D + 1][NT][2];\n  auto ldb = [&](uint4 (&b)[NT][2], int s, int c) {\n#pragma unroll\n    for (int n = 0; n < NT; n++) {\n      const uint4* q = rp[n] + s * 64 + c * 8;\n      if (G2) { b[n][0] = __ldca(q); b[n][1] = __ldca(q + 1); } else { b[n][0] = __ldg(q); b[n][1] = __ldg(q + 1); }\n    }\n  };\n  float acc[NT][CH][4];\n#pragma unroll\n  for (int n = 0; n < NT; n++)\n#pragma unroll\n    for (int h = 0; h < CH; h++) acc[n][h][0] = acc[n][h][1] = acc[n][h][2] = acc[n][h][3] = 0.f;\n#pragma unroll\n  for (int c = 0; c < D; c++) ldb(B[c], s0, c);\n  for (int s = s0; s < s1; s++) {\n    const uint32_t slot = C.acquire();\n    const uint2 sh0 = lds64(slot - lane * 8 + 2048), sh1 = lds64(slot - lane * 8 + 4352);  // slot = stage + lane * 16\n    const uint4 sc = make_uint4(sh0.x, sh0.y, sh1.x, sh1.y);\n#pragma unroll\n    for (int c = 0; c < 8; c++) {\n      const uint4 w = lds(slot + c * 512 + (c >= 4 ? 256 : 0));\n      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w));\n      const uint32_t sA = sc2(sw, (c & 1) ? 0x4242 : 0x4040), sB = sc2(sw, (c & 1) ? 0x4343 : 0x4141);\n      if (c + D < 8) ldb(B[(c + D) % (D + 1)], s, c + D);\n      else if (s + 1 < s1) ldb(B[(c + D) % (D + 1)], s + 1, c + D - 8);\n      const uint4 (&bc)[NT][2] = B[c % (D + 1)];\n      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};\n#pragma unroll\n      for (int j = 0; j < 4; j++) {\n        const uint32_t q = ww[j];\n#ifdef FAKEDQ\n        const uint32_t a0 = q ^ sA, a1 = q ^ sB, a2 = q + sA, a3 = q + sB;\n#else\n        const uint32_t a0 = bmul(q & 0x81C081C0u, sA);\n        const uint32_t a1 = bmul(((q << 3) & 0x01C001C0u) | ((q << 1) & 0x80008000u), sB);\n        const uint32_t a2 = bmul((q << 6) & 0x81C081C0u, sA);\n        const uint32_t a3 = bmul(((q >> 5) & 0x01C001C0u) | ((q << 5) & 0x80008000u), sB);\n#endif\n#pragma unroll\n        for (int n = 0; n < NT; n++) {\n          const uint32_t* bb = (const uint32_t*)&bc[n][j >> 1];\n#ifdef NOMMA\n          acc[n][j % CH][0] += __uint_as_float(a0 ^ a1 ^ a2 ^ a3 ^ bb[(j & 1) * 2] ^ bb[(j & 1) * 2 + 1]);\n#else\n          mma(acc[n][j % CH], a0, a1, a2, a3, bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);\n#endif\n        }\n      }\n    }\n    C.release();\n  }\n#ifdef PROF\n  const long long ce0 = clock64();\n  struct EpiT { long long c; int lane; unsigned long long* pf; int gw; __device__ ~EpiT() { if (lane == 0 && pf) pf[gw * 16 + 15] += clock64() - c; } } epit{ce0, lane, p.prof, (int)(blockIdx.x * WPC + (threadIdx.x >> 5))};\n#endif\n  extern __shared__ __align__(16) uint8_t smem[];\n  float f[NT][4];\n#pragma unroll\n  for (int n = 0; n < NT; n++)\n#pragma unroll\n    for (int i = 0; i < 4; i++) { float v = 0.f;\n#pragma unroll\n      for (int h = 0; h < CH; h++) v += acc[n][h][i];\n      f[n][i] = v; }\n  if (NT <= 2 && C.ks > 1) {  // group split-K: partials meet in smem (double-buffered by tile parity)\n    const int wid = threadIdx.x >> 5, ks = C.ks, w0 = wid - C.gj;\n    float4* red = (float4*)(smem + (size_t)WPC * S * STG + WPC * SCR + WPC * S * 8) + (ti & 1) * WPC * 64;\n#pragma unroll\n    for (int n = 0; n < NT; n++) red[(wid * 2 + n) * 32 + lane] = make_float4(f[n][0], f[n][1], f[n][2], f[n][3]);\n    gbar(1 + C.gi, ks * 32);\n    if (C.gj != (ti & (ks - 1))) return false;\n    for (int o = w0; o < w0 + ks; o++) {\n      if (o == wid) continue;\n#pragma unroll\n      for (int n = 0; n < NT; n++) { float4 v = red[(o * 2 + n) * 32 + lane]; f[n][0] += v.x; f[n][1] += v.y; f[n][2] += v.z; f[n][3] += v.w; }\n    }\n  }\n  __nv_bfloat16* scr = (__nv_bfloat16*)(smem + (size_t)(blockDim.x >> 5) * S * STG + (threadIdx.x >> 5) * SCR);\n  if (!G2) {\n    const float gs = p.g1[p.lo + e];\n#pragma unroll\n    for (int n = 0; n < NT; n++)\n#pragma unroll\n      for (int h = 0; h < 2; h++) {\n        float gt = f[n][h] * gs, up = f[n][2 + h] * gs;\n        scr[(n * 8 + 2 * t + h) * 8 + g] = __float2bfloat16(muse_swiglu_bf16(gt, up));\n      }\n    __syncwarp();\n    for (int i = lane; i < NT * 8; i += 32) {\n      int ii = pass * 8 * NTM + i;\n      if (ii < cnt) { int r = C.list[off + ii]; *(uint4*)(p.inter + (size_t)r * HD + mt * 8) = *(uint4*)(scr + i * 8); }\n    }\n  } else {\n    const float gs = p.g2[p.lo + e];\n    const float n0 = p.nw[mt * 16 + g], n1 = p.nw[mt * 16 + g + 8];\n#pragma unroll\n    for (int n = 0; n < NT; n++) {\n      float q0 = __bfloat162float(__float2bfloat16(f[n][0] * gs)), q1 = __bfloat162float(__float2bfloat16(f[n][1] * gs)), q2 = __bfloat162float(__float2bfloat16(f[n][2] * gs)), q3 = __bfloat162float(__float2bfloat16(f[n][3] * gs));\n      float v0=q0*n0, v1=q1*n0, v2=q2*n1, v3=q3*n1;\n      scr[(n * 8 + 2 * t) * 16 + g] = __float2bfloat16(q0);\n      scr[(n * 8 + 2 * t + 1) * 16 + g] = __float2bfloat16(q1);\n      scr[(n * 8 + 2 * t) * 16 + g + 8] = __float2bfloat16(q2);\n      scr[(n * 8 + 2 * t + 1) * 16 + g + 8] = __float2bfloat16(q3);\n      float sa = v0 * v0 + v2 * v2, sb = v1 * v1 + v3 * v3;\n#pragma unroll\n      for (int o = 4; o < 32; o <<= 1) { sa += __shfl_xor_sync(0xffffffffu, sa, o); sb += __shfl_xor_sync(0xffffffffu, sb, o); }\n      if (g == 0) {\n        int i0 = pass * 8 * NTM + n * 8 + 2 * t;\n        if (i0 < cnt) atomicAdd(p.ss + C.list[off + i0], sa);\n        if (i0 + 1 < cnt) atomicAdd(p.ss + C.list[off + i0 + 1], sb);\n      }\n    }\n    __syncwarp();\n    for (int i = lane; i < NT * 16; i += 32) {\n      int ii = pass * 8 * NTM + (i >> 1);\n      if (ii < cnt) { int r = C.list[off + ii]; *(uint4*)(p.y + (size_t)r * HD + mt * 16 + (i & 1) * 8) = *(uint4*)(scr + i * 8); }\n    }\n  }\n  __syncwarp();\n  return true;\n}\n\ntemplate <int G2, int S>\n__device__ __forceinline__ bool dispatch(Ctx<S>& C, int nt, int e, int mt, int pass, int ti, int s0, int s1) {\n  if (nt <= 1) return run<1, G2, S>(C, e, mt, pass, ti, s0, s1);\n  if (nt <= 2) return run<2, G2, S>(C, e, mt, pass, ti, s0, s1);\n#if NTM >= 8\n  if (nt > 4) return run<8, G2, S>(C, e, mt, pass, ti, s0, s1);\n#endif\n  return run<4, G2, S>(C, e, mt, pass, ti, s0, s1);\n}\n\n__global__ void __launch_bounds__(WPC * 32, 1) moe_kernel(const __grid_constant__ P p) {\n  constexpr int S = NSTG;\n  extern __shared__ __align__(16) uint8_t smem[];\n  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, R = p.M * 8;\n  const bool small = p.M <= 64;  // split-K groups (and their smem) only for small batches\n  Sm* sm = (Sm*)(smem + WPC * S * STG + WPC * SCR + WPC * S * 8 + (small ? WPC * RED : 0));\n  uint16_t* list = (uint16_t*)(sm + 1);\n  uint8_t* lid = (uint8_t*)(list + R);\n  const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem);\n  const int gw = blockIdx.x * WPC + wid;\n#ifdef PROF\n  if (lane == 0) { for (int k = 0; k < 16; k++) p.prof[gw * 16 + k] = 0; unsigned sid; asm("mov.u32 %0, %smid;" : "=r"(sid)); p.prof[gw * 16 + 10] = sid; }\n  PSET(0, gtime()); PSET(13, clock64());\n#endif\n#if TMA\n  if (lane < S) mb_init(sbase + WPC * S * STG + WPC * SCR + (wid * S + lane) * 8);\n  asm volatile("fence.mbarrier_init.release.cluster;\\n" ::: "memory");\n#endif\n  if (tid < NLOC) sm->cnt[tid] = 0;\n  __syncthreads();\n  for (int i = tid; i < R; i += blockDim.x) { unsigned e = p.ids[i] - p.lo; lid[i] = e < NLOC ? e : 0xff; if (e < NLOC) atomicAdd(&sm->cnt[e], 1); }\n  __syncthreads();\n  if (wid == 0) {\n    int c = sm->cnt[lane], v = c;\n#pragma unroll\n    for (int o = 1; o < 32; o <<= 1) { int n = __shfl_up_sync(0xffffffffu, v, o); if (lane >= o) v += n; }\n    sm->off[lane] = v - c; sm->fill[lane] = v - c;\n    unsigned m = __ballot_sync(0xffffffffu, c > 0);\n#if HEAVY1\n    {  // active experts heaviest first: their multi-pass tiles start early instead of forming the tail\n      int rk = 0;\n#pragma unroll\n      for (int j = 0; j < 32; j++) { const int cj = __shfl_sync(0xffffffffu, c, j); rk += (cj > c) || (cj == c && j < lane); }\n      if (c > 0) sm->act[rk] = lane;\n    }\n#else\n    if (c > 0) sm->act[__popc(m & ((1u << lane) - 1))] = lane;\n#endif\n    int mx = c;\n#pragma unroll\n    for (int o = 16; o; o >>= 1) mx = max(mx, __shfl_xor_sync(0xffffffffu, mx, o));\n    if (lane == 0) { sm->na = __popc(m); sm->maxc = mx; const int tw = __popc(m) * (T1E + T2E) / (int)(gridDim.x * WPC / 16);  // tiles per 16 warps\n      sm->ks = (small && mx <= 16) ? (KSF ? KSF : (tw < 32 ? 4 : (tw < 96 ? 2 : 1))) : 1; }\n  }\n  __syncthreads();\n  for (int i = tid; i < R; i += blockDim.x) { unsigned e = lid[i]; if (e < NLOC) list[atomicAdd(&sm->fill[e], 1)] = i; }\n  const int ks = sm->ks, na = sm->na;\n  Ctx<S> C;\n  C.p = &p; C.sm = sm; C.list = list; C.lane = lane; C.ks = ks; C.gi = wid / ks; C.gj = wid % ks;\n  C.N1 = na * T1E; C.N = C.N1 + na * T2E;\n  C.ring = sbase + wid * S * STG; C.bars = sbase + WPC * S * STG + WPC * SCR + wid * S * 8; C.q = 0; C.wcyc = 0;\n  C.pi = 0; C.pp = 0; C.ps = 0;\n  const bool lead = C.gj == 0 && lane == 0;\n  if (lead) { sm->ids[C.gi][0] = atomicAdd(p.ctr + QI, 1); sm->ids[C.gi][1] = atomicAdd(p.ctr + QI, 1); }\n  __syncthreads();\n#ifdef PROF\n  PSET(1, gtime());\n#endif\n#pragma unroll\n  for (int i = 0; i < S - 1 + EARLYREL; i++) C.issue(i);\n  for (int i = 0;; i++) {\n    const int u = C.tid_of(i);\n    if (u >= C.N) break;\n    int nxt = 0;\n    if (lead) nxt = atomicAdd(p.ctr + QI, 1);  // tile for i + 2; latency hidden behind this tile\n    int g2, e, mt; C.item(u, g2, e, mt);\n    const int cnt = sm->cnt[e], np = (cnt + 8 * NTM - 1) / (8 * NTM);\n#ifdef PROF\n    long long c0 = clock64();\n#endif\n    if (g2) {\n      if (lane == 0) while (ldacq(p.ctr + e) < T1E) __nanosleep(32);\n      __syncwarp();\n#ifdef PROF\n      PADD(4, clock64() - c0); c0 = clock64();\n#endif\n    } else if (mt == 0 && C.gj == 0) {\n      for (int k = lane; k < cnt; k += 32) p.ss[list[sm->off[e] + k]] = 0.f;\n    }\n    if (lead) sm->ids[C.gi][(i + 2) & 3] = nxt;  // read 2 tiles later, after the group barrier (ks > 1) / syncwarp\n    __syncwarp();\n    const int span = SPT / ks, s0 = C.gj * span;\n    bool fin = false;\n    for (int ps = 0; ps < np; ps++) {\n      int nt = min(NTM, (cnt - ps * 8 * NTM + 7) >> 3);\n      fin = g2 ? dispatch<1, S>(C, nt, e, mt, ps, i, s0, s0 + span) : dispatch<0, S>(C, nt, e, mt, ps, i, s0, s0 + span);\n    }\n    if (fin) {  // release this tile\'s stores to the dependent phase (no full fence: the release add orders them)\n      __syncwarp();\n      if (lane == 0) asm volatile("red.release.gpu.global.add.s32 [%0], 1;\\n" :: "l"(p.ctr + (g2 ? NLOC : e)) : "memory");\n    }\n#ifdef PROF\n    PADD(g2 ? 3 : 2, clock64() - c0); PADD(8, 1);\n#endif\n  }\n#if !TMA\n  cpw<0>();\n#endif\n  const int T2 = sm->na * T2E;\n#ifdef PROF\n  PSET(5, gtime()); PSET(9, C.wcyc); PSET(11, sm->na); PSET(12, sm->ks);\n#endif\n  if (lane == 0) while (ldacq(p.ctr + NLOC) < T2) __nanosleep(32);\n  __syncwarp();\n#ifdef PROF\n  PSET(6, gtime());\n#endif\n  for (int it = gw; it < p.M * 16; it += gridDim.x * WPC) {\n    const int t = it >> 4, col = (it & 15) * 256 + lane * 8;\n    float a[8] = {0, 0, 0, 0, 0, 0, 0, 0};\n#pragma unroll\n    for (int k = 0; k < 8; k++) {\n      const int r = t * 8 + k;\n      unsigned e = p.ids[r] - p.lo;\n      if (e < NLOC) {\n        float cf = p.tw[r] * rsqrtf(__ldcg(p.ss + r) * (1.f / HD) + 1e-8f);\n        uint4 v = __ldcg((const uint4*)(p.y + (size_t)r * HD + col));\n        const __nv_bfloat162* h = (const __nv_bfloat162*)&v;\n#pragma unroll\n        for (int j = 0; j < 4; j++) { float2 f = __bfloat1622float2(h[j]); a[2 * j] += cf * f.x; a[2 * j + 1] += cf * f.y; }\n      }\n    }\n    uint4 o; __nv_bfloat162* ob = (__nv_bfloat162*)&o;\n#pragma unroll\n    for (int j = 0; j < 4; j++) ob[j] = __floats2bfloat162_rn(a[2*j]*p.nw[col+2*j], a[2*j+1]*p.nw[col+2*j+1]);\n    *(uint4*)(p.out + (size_t)t * HD + col) = o;\n  }\n#ifdef PROF\n  PSET(7, gtime()); PSET(14, clock64());\n#endif\n  __syncthreads();\n  if (tid == 0) {\n    __threadfence();\n    if (atomicAdd(p.ctr + NLOC + 1, 1) == (int)gridDim.x - 1) {\n      for (int i = 0; i < NLOC + 3; i++) p.ctr[i] = 0;\n      __threadfence();\n    }\n  }\n}\n\nstatic int g_grid = 0;\nstatic size_t smem_bytes(int M) { return (size_t)WPC * NSTG * STG + WPC * SCR + WPC * NSTG * 8 + (M <= 64 ? WPC * RED : 0) + sizeof(Sm) + (size_t)M * 8 * 3; }\n\nvoid moe(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor w1, torch::Tensor w2, torch::Tensor g1,\n         torch::Tensor g2, torch::Tensor nw, torch::Tensor inter, torch::Tensor y, torch::Tensor ss, torch::Tensor ctr,\n         torch::Tensor tcnt, torch::Tensor ws, torch::Tensor out, int64_t lo, torch::Tensor prof) {\n  int M = x.size(0);\n  if (!g_grid) {\n    int dev; cudaGetDevice(&dev); int mx; cudaDeviceGetAttribute(&mx, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);\n    cudaDeviceGetAttribute(&g_grid, cudaDevAttrMultiProcessorCount, dev);\n    cudaFuncSetAttribute(moe_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mx);\n  }\n  P p{w1.data_ptr<uint8_t>(), w2.data_ptr<uint8_t>(), g1.data_ptr<float>(), g2.data_ptr<float>(),\n      (const __nv_bfloat16*)x.data_ptr(), ids.data_ptr<int>(), tw.data_ptr<float>(), nw.data_ptr<float>(),\n      (__nv_bfloat16*)inter.data_ptr(), (__nv_bfloat16*)y.data_ptr(), ss.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(),\n      ctr.data_ptr<int>(), tcnt.data_ptr<int>(), (float4*)ws.data_ptr(), M, (int)lo,\n      prof.numel() ? (unsigned long long*)prof.data_ptr() : nullptr};\n  moe_kernel<<<g_grid, WPC * 32, smem_bytes(M), at::cuda::getCurrentCUDAStream()>>>(p);\n}\n'

CUDA_L = '\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cuda_bf16.h>\n// Match baseline silu_and_mul: BF16 gate/up inputs, FP32 activation/product,\n// BF16 product store at each caller. Do not add a nonexistent SiLU tensor boundary.\n__device__ __forceinline__ float muse_swiglu_bf16(float gate, float up) {\n  gate = __bfloat162float(__float2bfloat16_rn(gate));\n  up = __bfloat162float(__float2bfloat16_rn(up));\n  return gate / (1.f + __expf(-gate)) * up;\n}\n\n#include <stdint.h>\n\n#define HD 4096\n#define NLOC 32\n#define HSB 2304\n#define NHS 16\n#ifndef TP\n#define TP 32\n#endif\n#define XROW 528\n#ifndef NCW\n#define NCW 8\n#endif\n#define NTH ((NCW + 1) * 32)\n#define TPR (NCW * RPW)\n#define WB (TPR * HSB)\n#define SLOTB (WB + TP * XROW)\n#define NRB1 (8192 / (16 * TPR))\n#define NRB2 (4096 / (16 * TPR))\n#define C_W2 32\n#define C_Q 33\n#define C_DONE 34\n#define C_UC 64\n#ifndef LAG\n#define LAG 64\n#endif\n#ifndef CPB\n#define CPB 1\n#endif\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900\n#define TMA 1\n#define FULLCNT 33\n#else\n#define TMA 0\n#define FULLCNT 32\n#endif\n\nstruct P {\n  const uint8_t* w1; const uint8_t* w2; const float* g1; const float* g2;\n  const __nv_bfloat16* x; const int* ids; const float* tw; const float* nw;\n  __nv_bfloat16* inter; __nv_bfloat16* y; float* ss; __nv_bfloat16* out; int* ctr; float* ws;\n  int M, lo, ks; unsigned long long* prof;\n};\nstruct Hdr { int g, e, rb, ps, s, ntok, flags, pad; };\nstruct Sm {\n  Hdr hdr[NSLOT];\n  int cnt[NLOC], off[NLOC], act[NLOC], np[NLOC], base[NLOC];\n  int segst[2 * NLOC + 1], segk[2 * NLOC];\n  int nseg, na, ntot, nw2, flag;\n  int wc[NTH / 32][NLOC];\n};\n\n#ifdef PROF\n__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }\n#define PSET(k, v) do { if (lane == 0 && p.prof) p.prof[(blockIdx.x * (NTH / 32) + wid) * 16 + (k)] = (v); } while (0)\n#define PADD(k, v) do { if (lane == 0 && p.prof) p.prof[(blockIdx.x * (NTH / 32) + wid) * 16 + (k)] += (v); } while (0)\n#else\n#define PSET(k, v)\n#define PADD(k, v)\n#endif\n\n__device__ __forceinline__ void cp16(uint32_t s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\\n" :: "r"(s), "l"(g) : "memory"); }\n__device__ __forceinline__ uint32_t bmul(uint32_t a, uint32_t b) { uint32_t d; asm("fma.rn.bf16x2 %0, %1, %2, %3;\\n" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u)); return d; }\n__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {\n  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\\n"\n    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n__device__ __forceinline__ int ldacq(const int* p) { int v; asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\\n" : "=r"(v) : "l"(p) : "memory"); return v; }\n__device__ __forceinline__ int atom_ar(int* p, int v) { int o; asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\\n" : "=r"(o) : "l"(p), "r"(v) : "memory"); return o; }\n__device__ __forceinline__ void red_rel(int* p, int v) { asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" :: "l"(p), "r"(v) : "memory"); }\n__device__ __forceinline__ void redf(float* p, float v) { asm volatile("red.relaxed.gpu.global.add.f32 [%0], %1;\\n" :: "l"(p), "f"(v) : "memory"); }\n__device__ __forceinline__ void cbar() { asm volatile("bar.sync 1, %0;\\n" :: "n"(NCW * 32) : "memory"); }\n__device__ __forceinline__ uint32_t rotl(uint32_t q, int d) { return __funnelshift_l(q, q, d); }\n__device__ __forceinline__ uint32_t sc2(uint32_t w, uint32_t sel) { return __byte_perm(w, 0, sel) * 16u + 0x77007700u; }\n__device__ __forceinline__ void mbinit(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\\n" :: "r"(b), "r"(n)); }\n__device__ __forceinline__ void mbarrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\\n" :: "r"(b) : "memory"); }\n__device__ __forceinline__ void cpa_arrive(uint32_t b) { asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\\n" :: "r"(b) : "memory"); }\n__device__ __forceinline__ bool mbtest(uint32_t b, uint32_t ph) {\n  uint32_t r;\n  asm volatile("{\\n.reg .pred p;\\nmbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\\nselp.u32 %0, 1, 0, p;\\n}\\n" : "=r"(r) : "r"(b), "r"(ph) : "memory");\n  return r;\n}\n__device__ __forceinline__ void mbwait(uint32_t b, uint32_t ph) {\n#if TMA\n  asm volatile("{\\n.reg .pred p;\\nW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\\n@!p bra W;\\n}\\n" :: "r"(b), "r"(ph) : "memory");\n#else\n  asm volatile("{\\n.reg .pred p;\\nW: mbarrier.test_wait.parity.shared::cta.b64 p, [%0], %1;\\n@!p bra W;\\n}\\n" :: "r"(b), "r"(ph) : "memory");\n#endif\n}\n#if TMA\n__device__ __forceinline__ void expect_tx(uint32_t bar, uint32_t n) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" :: "r"(bar), "r"(n) : "memory"); }\n__device__ __forceinline__ void bulk(uint32_t dst, const void* src, uint32_t n, uint32_t bar) {\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\\n" :: "r"(dst), "l"(src), "r"(n), "r"(bar) : "memory");\n}\n#endif\n\ntemplate <int NT>\n__device__ __forceinline__ void stage(float (&acc)[RPW][4][4], const uint8_t* wsm, const uint8_t* xs, int lane) {\n  const int g = lane >> 2, t = lane & 3;\n  uint2 sc[RPW];\n#pragma unroll\n  for (int r = 0; r < RPW; r++) sc[r] = *(const uint2*)(wsm + r * HSB + 2048 + lane * 8);\n#pragma unroll\n  for (int c = 0; c < 4; c++) {\n    uint4 w[RPW];\n#pragma unroll\n    for (int r = 0; r < RPW; r++) w[r] = *(const uint4*)(wsm + r * HSB + c * 512 + lane * 16);\n    uint4 b[NT][2];\n#pragma unroll\n    for (int n = 0; n < NT; n++)\n#pragma unroll\n      for (int h = 0; h < 2; h++) b[n][h] = *(const uint4*)(xs + (n * 8 + g) * XROW + c * 128 + t * 32 + h * 16);\n    uint32_t sA[RPW], sB[RPW];\n#pragma unroll\n    for (int r = 0; r < RPW; r++) {\n      const uint32_t sw = c < 2 ? sc[r].x : sc[r].y;\n      sA[r] = sc2(sw, (c & 1) ? 0x4242 : 0x4040); sB[r] = sc2(sw, (c & 1) ? 0x4343 : 0x4141);\n    }\n#pragma unroll\n    for (int j = 0; j < 4; j++) {\n#pragma unroll\n      for (int r = 0; r < RPW; r++) {\n        const uint32_t q = j == 0 ? w[r].x : j == 1 ? w[r].y : j == 2 ? w[r].z : w[r].w;\n#ifdef FAKEDQ\n        const uint32_t a0 = q ^ sA[r], a1 = q ^ sB[r], a2 = q + sA[r], a3 = q + sB[r];\n#else\n        const uint32_t a0 = bmul(q & 0x81C081C0u, sA[r]);\n        const uint32_t a1 = bmul(((q << 3) & 0x01C001C0u) | ((q << 1) & 0x80008000u), sB[r]);\n        const uint32_t a2 = bmul((q << 6) & 0x81C081C0u, sA[r]);\n        const uint32_t a3 = bmul(((q >> 5) & 0x01C001C0u) | ((q << 5) & 0x80008000u), sB[r]);\n#endif\n        constexpr int CH = NT == 1 ? 4 : (NT == 2 ? 2 : 1);\n#pragma unroll\n        for (int n = 0; n < NT; n++) {\n          const uint32_t* bb = (const uint32_t*)&b[n][j >> 1];\n#ifdef NOMMA\n          acc[r][n * CH + j % CH][0] += __uint_as_float((a0 ^ a1 ^ a2 ^ a3 ^ bb[(j & 1) * 2] ^ bb[(j & 1) * 2 + 1]) & 0x3fffffffu);\n#else\n          mma(acc[r][n * CH + j % CH], a0, a1, a2, a3, bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);\n#endif\n        }\n      }\n    }\n  }\n}\n\n// unit finished: acc holds this CTA\'s partial (ks > 1) or full sums for rows of tiles rb*TPR + wid*RPW + r, tokens of pass ps\ntemplate <int NT>\n__device__ __forceinline__ void epilogue(const P& p, Sm* sm, const uint16_t* list, float (&acc)[RPW][4][4], const Hdr& h, int wid, int lane, int& pend) {\n  const int g = lane >> 2, t = lane & 3, e = h.e, ntok = h.ntok;\n  const uint16_t* lst = list + sm->off[e] + h.ps * TP;\n  const int rows = h.g ? 4096 : 8192;\n  constexpr int CH = NT == 1 ? 4 : (NT == 2 ? 2 : 1);\n  if (CH > 1) {\n#pragma unroll\n    for (int r = 0; r < RPW; r++)\n#pragma unroll\n      for (int n = 0; n < NT; n++)\n#pragma unroll\n        for (int i = 0; i < 4; i++) {\n          float v = acc[r][n * CH][i];\n#pragma unroll\n          for (int k = 1; k < CH; k++) v += acc[r][n * CH + k][i];\n          acc[r][n][i] = v;\n        }\n  }\n  if (p.ks > 1) {  // partials meet in L2; this warp\'s arrival is checked one stage later (latency hidden)\n    float* wsb = p.ws + (h.g ? (size_t)p.M * 8 * 8192 : 0);\n#pragma unroll\n    for (int r = 0; r < RPW; r++) {\n      const int mt = h.rb * TPR + wid * RPW + r;\n#pragma unroll\n      for (int n = 0; n < NT; n++)\n#pragma unroll\n        for (int i = 0; i < 4; i++) {\n          const int tok = n * 8 + 2 * t + (i & 1);\n          if (tok < ntok) redf(wsb + (size_t)lst[tok] * rows + mt * 16 + g + 8 * (i >> 1), acc[r][n][i]);\n        }\n    }\n    __syncwarp();\n    if (lane == 0) pend = atom_ar(p.ctr + C_UC + ((h.g * NLOC + e) * 64 + h.rb) * 8 + h.ps, 1);\n    return;\n  }\n  if (!h.g) {\n    const float gs = p.g1[p.lo + e];\n#pragma unroll\n    for (int r = 0; r < RPW; r++) {\n      const int mt = h.rb * TPR + wid * RPW + r;\n#pragma unroll\n      for (int n = 0; n < NT; n++)\n#pragma unroll\n        for (int hh = 0; hh < 2; hh++) {\n          const int tok = n * 8 + 2 * t + hh;\n          const float gt = acc[r][n][hh] * gs, up = acc[r][n][2 + hh] * gs;\n          if (tok < ntok) p.inter[(size_t)lst[tok] * HD + mt * 8 + g] = __float2bfloat16(muse_swiglu_bf16(gt, up));\n        }\n    }\n    __syncwarp();\n    if (lane == 0) red_rel(p.ctr + e, 1);\n  } else {\n    const float gs = p.g2[p.lo + e];\n    float sa[NT], sb[NT];\n#pragma unroll\n    for (int n = 0; n < NT; n++) sa[n] = sb[n] = 0.f;\n#pragma unroll\n    for (int r = 0; r < RPW; r++) {\n      const int mt = h.rb * TPR + wid * RPW + r;\n      const float n0 = p.nw[mt * 16 + g], n1 = p.nw[mt * 16 + g + 8];\n#pragma unroll\n      for (int n = 0; n < NT; n++) {\n        const float q0=__bfloat162float(__float2bfloat16(acc[r][n][0]*gs)), q1=__bfloat162float(__float2bfloat16(acc[r][n][1]*gs)), q2=__bfloat162float(__float2bfloat16(acc[r][n][2]*gs)), q3=__bfloat162float(__float2bfloat16(acc[r][n][3]*gs));\n        const float v0=q0*n0, v1=q1*n0, v2=q2*n1, v3=q3*n1;\n        const int t0 = n * 8 + 2 * t;\n        if (t0 < ntok) { __nv_bfloat16* yr = p.y + (size_t)lst[t0] * HD + mt * 16 + g; yr[0] = __float2bfloat16(q0); yr[8] = __float2bfloat16(q2); }\n        if (t0 + 1 < ntok) { __nv_bfloat16* yr = p.y + (size_t)lst[t0 + 1] * HD + mt * 16 + g; yr[0] = __float2bfloat16(q1); yr[8] = __float2bfloat16(q3); }\n        sa[n] += v0 * v0 + v2 * v2; sb[n] += v1 * v1 + v3 * v3;\n      }\n    }\n#pragma unroll\n    for (int n = 0; n < NT; n++) {\n#pragma unroll\n      for (int o = 4; o < 32; o <<= 1) { sa[n] += __shfl_xor_sync(0xffffffffu, sa[n], o); sb[n] += __shfl_xor_sync(0xffffffffu, sb[n], o); }\n      const int t0 = n * 8 + 2 * t;\n      if (g == 0) {\n        if (t0 < ntok) atomicAdd(p.ss + lst[t0], sa[n]);\n        if (t0 + 1 < ntok) atomicAdd(p.ss + lst[t0 + 1], sb[n]);\n      }\n    }\n    __syncwarp();\n    if (lane == 0) red_rel(p.ctr + C_W2, 1);\n  }\n}\n\n// last arriving warp of a K-split unit: full sums from the workspace (zeroed behind), epilogue for all of the unit\'s rows\n__device__ __noinline__ void finish(const P& p, const Sm* sm, const uint16_t* list, const Hdr h, int lane) {\n  const int e = h.e, ntok = h.ntok;\n  const uint16_t* lst = list + sm->off[e] + h.ps * TP;\n  if (!h.g) {\n    const float gs = p.g1[p.lo + e];\n    for (int tok = 0; tok < ntok; tok++) {\n      const int r = lst[tok];\n      for (int c = lane; c < TPR * 8; c += 32) {\n        const int mt = h.rb * TPR + (c >> 3);\n        float* a = p.ws + (size_t)r * 8192 + mt * 16 + (c & 7);\n        const float gt = __ldcg(a) * gs, up = __ldcg(a + 8) * gs;\n        __stcg(a, 0.f); __stcg(a + 8, 0.f);\n        p.inter[(size_t)r * HD + mt * 8 + (c & 7)] = __float2bfloat16(muse_swiglu_bf16(gt, up));\n      }\n    }\n  } else {\n    const float gs = p.g2[p.lo + e];\n    float* wsb = p.ws + (size_t)p.M * 8 * 8192;\n    for (int tok = 0; tok < ntok; tok++) {\n      const int r = lst[tok];\n      float s2 = 0.f;\n      for (int i = lane; i < TPR * 16; i += 32) {\n        const int row = h.rb * TPR * 16 + i;\n        float* a = wsb + (size_t)r * 4096 + row;\n        const float q = __bfloat162float(__float2bfloat16(__ldcg(a) * gs));\n        const float v = q * p.nw[row];\n        __stcg(a, 0.f);\n        p.y[(size_t)r * HD + row] = __float2bfloat16(q);\n        s2 += v * v;\n      }\n#pragma unroll\n      for (int o = 1; o < 32; o <<= 1) s2 += __shfl_xor_sync(0xffffffffu, s2, o);\n      if (lane == 0) atomicAdd(p.ss + r, s2);\n    }\n  }\n  __syncwarp();\n  if (lane == 0) {\n    p.ctr[C_UC + ((h.g * NLOC + e) * 64 + h.rb) * 8 + h.ps] = 0;\n    red_rel(p.ctr + (h.g ? C_W2 : e), NCW);\n  }\n}\n\n__device__ __forceinline__ void decode(const Sm* sm, int ks, int u, int& g, int& e, int& rb, int& kp, int& ps) {\n  int k = 0;\n  while (sm->segst[k + 1] <= u) k++;\n  const int code = sm->segk[k];\n  g = code >> 6; e = sm->act[code & 63];\n  const int np = sm->np[e];\n  int v = u - sm->segst[k];\n  kp = v % ks; v /= ks; ps = v % np; rb = v / np;\n}\n\n__global__ void __launch_bounds__(NTH, CPB) moe_kernel(const __grid_constant__ P p) {\n  extern __shared__ __align__(128) uint8_t smem[];\n  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, R = p.M * 8, ks = p.ks;\n  Sm* sm = (Sm*)(smem + NSLOT * SLOTB + NSLOT * 16);\n  uint16_t* list = (uint16_t*)(sm + 1);\n  uint8_t* lid = smem;                       // prologue scratch in the ring\n  uint16_t* posi = (uint16_t*)(smem + 4096);\n  const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem);\n  const uint32_t fullb = sbase + NSLOT * SLOTB, emptyb = fullb + NSLOT * 8;\n#ifdef PROF\n  PSET(0, gtime()); PSET(8, clock64());\n#endif\n  if (tid < NSLOT) { mbinit(fullb + tid * 8, FULLCNT); mbinit(emptyb + tid * 8, NCW); }\n  if (tid < NLOC) sm->base[tid] = 0;\n#if TMA\n  asm volatile("fence.mbarrier_init.release.cluster;\\n" ::: "memory");\n#endif\n  int u = 0, nx = 0;\n#ifdef NX1\n  if (tid == NCW * 32) u = atomicAdd(p.ctr + C_Q, 1);\n#else\n  if (tid == NCW * 32) { u = atomicAdd(p.ctr + C_Q, 1); nx = atomicAdd(p.ctr + C_Q, 1); }  // queue heads early: latency hides behind routing\n#endif\n  for (int i = tid; i < R; i += NTH) { unsigned e = p.ids[i] - p.lo; lid[i] = e < NLOC ? e : 0xff; }\n  __syncthreads();\n  const unsigned ltm = (1u << lane) - 1;\n  for (int c0 = 0; c0 < R; c0 += NTH) {  // stable per-expert lists (route order), identical in every CTA\n    const int i = c0 + tid;\n    const int e = i < R ? lid[i] : 0xff;\n    const unsigned m = __match_any_sync(0xffffffffu, e);\n    sm->wc[wid][lane] = 0;\n    __syncwarp();\n    if (e < NLOC && (m & ltm) == 0) sm->wc[wid][e] = __popc(m);\n    __syncthreads();\n    if (tid < NLOC) { int s = sm->base[tid];\n      for (int w = 0; w < NTH / 32; w++) { const int v = sm->wc[w][tid]; sm->wc[w][tid] = s; s += v; }\n      sm->base[tid] = s; }\n    __syncthreads();\n    if (e < NLOC) posi[i] = sm->wc[wid][e] + __popc(m & ltm);\n    __syncthreads();\n  }\n  if (wid == 0) {\n    const int c = sm->base[lane];\n    int v = c;\n#pragma unroll\n    for (int o = 1; o < 32; o <<= 1) { int n = __shfl_up_sync(0xffffffffu, v, o); if (lane >= o) v += n; }\n    sm->cnt[lane] = c; sm->off[lane] = v - c; sm->np[lane] = (c + TP - 1) / TP;\n    int rk = 0;\n#pragma unroll\n    for (int j = 0; j < 32; j++) { const int cj = __shfl_sync(0xffffffffu, c, j); rk += (cj > c) || (cj == c && j < lane); }\n    if (c > 0) sm->act[rk] = lane;\n    const unsigned am = __ballot_sync(0xffffffffu, c > 0);\n    int w2 = c > 0 ? NRB2 * ((c + TP - 1) / TP) * NCW : 0;\n#pragma unroll\n    for (int o = 16; o; o >>= 1) w2 += __shfl_xor_sync(0xffffffffu, w2, o);\n    __syncwarp();\n    if (lane == 0) {\n      const int na = __popc(am);\n      int ns = 0, u = 0;\n      auto push = [&](int g, int a) { sm->segst[ns] = u; sm->segk[ns] = g * 64 + a; u += (g ? NRB2 : NRB1) * ks * sm->np[sm->act[a]]; ns++; };\n      for (int a = 0; a < na; a++) { push(0, a); if (a >= LAG) push(1, a - LAG); }\n      for (int a = na - LAG < 0 ? 0 : na - LAG; a < na; a++) push(1, a);\n      sm->segst[ns] = u; sm->nseg = ns; sm->ntot = u; sm->na = na; sm->nw2 = w2;\n    }\n  }\n  __syncthreads();\n  for (int i = tid; i < R; i += NTH) { const int e = lid[i]; if (e < NLOC) list[sm->off[e] + posi[i]] = i; }\n  __syncthreads();\n#ifdef PROF\n  PSET(1, gtime());\n#endif\n  const int ntot = sm->ntot;\n  if (wid == NCW) {  // producer\n    u = __shfl_sync(0xffffffffu, u, 0); nx = __shfl_sync(0xffffffffu, nx, 0);\n    const int nst = NHS / ks;\n    unsigned depok = 0;\n    int seq = 0;\n#ifdef PROF\n    long long wemp = 0, wdep = 0;\n#endif\n    for (; u < ntot;) {\n      int g, e, rb, kp, ps; decode(sm, ks, u, g, e, rb, kp, ps);\n      const int ntok = min(TP, sm->cnt[e] - ps * TP);\n      const uint16_t* lst = list + sm->off[e] + ps * TP;\n      const uint8_t* wsrc = (g ? p.w2 + (size_t)((p.lo + e) * 256 + rb * TPR) * NHS * HSB\n                               : p.w1 + (size_t)((p.lo + e) * 512 + rb * TPR) * NHS * HSB);\n      const uint8_t* xsrc = g ? (const uint8_t*)p.inter : (const uint8_t*)p.x;\n      int myrow = -1;\n      if (lane < ntok) { const int r = lst[lane]; myrow = g ? r : r >> 3; }\n      for (int i = 0; i < nst; i++, seq++) {\n        const int slot = seq % NSLOT, s = kp * nst + i;\n#ifdef NX1\n        if (i == nst - 1 && lane == 0) nx = atomicAdd(p.ctr + C_Q, 1);\n#endif\n#ifdef PROF\n        long long c0 = clock64();\n#endif\n        if (seq >= NSLOT) mbwait(emptyb + slot * 8, ((seq / NSLOT) - 1) & 1);\n#ifdef PROF\n        wemp += clock64() - c0;\n#endif\n        if (lane == 0) { sm->hdr[slot] = Hdr{g, e, rb, ps, s, ntok, (i == 0 ? 1 : 0) | (i == nst - 1 ? 2 : 0) | ((!g && rb == 0 && kp == 0) ? 4 : 0), 0}; __threadfence_block(); }\n        const uint32_t dst = sbase + slot * SLOTB;\n        const uint32_t fb = fullb + slot * 8;\n#if TMA\n        if (lane == 0) expect_tx(fb, WB);\n        __syncwarp();\n        if (lane < TPR) bulk(dst + lane * HSB, wsrc + ((size_t)lane * NHS + s) * HSB, HSB, fb);\n#else\n        for (int q = lane; q < WB / 16; q += 32) {\n          const int tile = q / (HSB / 16), o = q % (HSB / 16);\n          cp16(dst + tile * HSB + o * 16, wsrc + ((size_t)tile * NHS + s) * HSB + o * 16);\n        }\n#endif\n        if (g && i == 0 && !(depok >> e & 1)) {\n#ifdef PROF\n          long long c1 = clock64();\n#endif\n#if defined(DEPPF) && TMA\n          {  // dependency not met yet: pull this unit\'s remaining weight stages into L2 meanwhile\n            const int tgt = NRB1 * sm->np[e] * NCW;\n            int ok = 0;\n            if (lane == 0) ok = ldacq(p.ctr + e) >= tgt;\n            if (!__shfl_sync(0xffffffffu, ok, 0)) {\n              if (lane < TPR) for (int k = i + 1; k < nst; k++)\n                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\\n" :: "l"(wsrc + ((size_t)lane * NHS + kp * nst + k) * HSB), "n"(HSB) : "memory");\n              if (lane == 0) while (ldacq(p.ctr + e) < tgt) __nanosleep(64);\n            }\n          }\n#else\n          if (lane == 0) { const int tgt = NRB1 * sm->np[e] * NCW; while (ldacq(p.ctr + e) < tgt) __nanosleep(64); }\n#endif\n          __syncwarp();\n          depok |= 1u << e;\n#ifdef PROF\n          wdep += clock64() - c1;\n#endif\n        }\n        for (int rw = 0; rw < ntok; rw++) {\n          const int row = __shfl_sync(0xffffffffu, myrow, rw);\n          cp16(dst + WB + rw * XROW + lane * 16, xsrc + (size_t)row * (HD * 2) + s * 512 + lane * 16);\n        }\n        cpa_arrive(fb);\n      }\n#ifdef NX1\n      u = __shfl_sync(0xffffffffu, nx, 0);\n#else\n      u = nx;\n      if (lane == 0 && nx < ntot) nx = atomicAdd(p.ctr + C_Q, 1);\n      nx = __shfl_sync(0xffffffffu, nx, 0);\n#endif\n    }\n    const int slot = seq % NSLOT;\n    if (seq >= NSLOT) mbwait(emptyb + slot * 8, ((seq / NSLOT) - 1) & 1);\n    if (lane == 0) sm->hdr[slot].g = -1;\n#if TMA\n    if (lane == 0) mbarrive(fullb + slot * 8);\n#endif\n    cpa_arrive(fullb + slot * 8);\n#ifdef PROF\n    PSET(2, gtime()); PSET(3, wemp); PSET(4, wdep);\n#endif\n  } else {  // consumers\n    float acc[RPW][4][4];\n#ifdef PROF\n    long long wful = 0; int nun = 0;\n#endif\n    int pend = 0, pv = 0;\n    Hdr ph;\n    for (int q = 0;; q++) {\n      const int slot = q % NSLOT;\n#ifdef PROF\n      long long c0 = clock64();\n#endif\n      if (pv && !mbtest(fullb + slot * 8, (q / NSLOT) & 1)) {  // never sit on a finished K-split unit (its W2 may be what we wait for)\n        pv = 0; if (__shfl_sync(0xffffffffu, pend, 0) == ks * NCW - 1) finish(p, sm, list, ph, lane);\n      }\n      mbwait(fullb + slot * 8, (q / NSLOT) & 1);\n#ifdef PROF\n      wful += clock64() - c0;\n#endif\n      const Hdr h = sm->hdr[slot];\n      if (h.g < 0) {\n        if (pv && __shfl_sync(0xffffffffu, pend, 0) == ks * NCW - 1) finish(p, sm, list, ph, lane);\n        break;\n      }\n      if (h.flags & 1) {\n#pragma unroll\n        for (int r = 0; r < RPW; r++)\n#pragma unroll\n          for (int n = 0; n < 4; n++) acc[r][n][0] = acc[r][n][1] = acc[r][n][2] = acc[r][n][3] = 0.f;\n        if ((h.flags & 4) && wid == 0 && lane < h.ntok) p.ss[list[sm->off[h.e] + h.ps * TP + lane]] = 0.f;\n      }\n      const uint8_t* sl = smem + slot * SLOTB;\n      const uint8_t* wsm = sl + wid * RPW * HSB;\n      const uint8_t* xs = sl + WB;\n#ifdef ONLYNT\n      const int nt = ONLYNT;\n#else\n      const int nt = (h.ntok + 7) >> 3;\n#endif\n#ifndef NOCOMP\n      if (nt == 1) stage<1>(acc, wsm, xs, lane);\n      else if (nt == 2) stage<2>(acc, wsm, xs, lane);\n      else if (nt == 3) stage<3>(acc, wsm, xs, lane);\n      else stage<4>(acc, wsm, xs, lane);\n#endif\n      __syncwarp();\n      if (lane == 0) mbarrive(emptyb + slot * 8);\n      if (pv) { pv = 0; if (__shfl_sync(0xffffffffu, pend, 0) == ks * NCW - 1) finish(p, sm, list, ph, lane); }\n      if (h.flags & 2) {\n        if (ks > 1) { pv = 1; ph = h; }\n#ifdef PROF\n        nun++;\n#endif\n        if (nt == 1) epilogue<1>(p, sm, list, acc, h, wid, lane, pend);\n        else if (nt == 2) epilogue<2>(p, sm, list, acc, h, wid, lane, pend);\n        else if (nt == 3) epilogue<3>(p, sm, list, acc, h, wid, lane, pend);\n        else epilogue<4>(p, sm, list, acc, h, wid, lane, pend);\n      }\n    }\n#ifdef PROF\n    PSET(2, gtime()); PSET(3, wful); PSET(5, nun);\n#endif\n  }\n  const int nw2 = sm->nw2;\n  if (lane == 0) while (ldacq(p.ctr + C_W2) < nw2) __nanosleep(32);\n  __syncwarp();\n#ifdef PROF\n  PSET(6, gtime());\n#endif\n  for (int it = blockIdx.x * (NTH / 32) + wid; it < p.M * 16; it += gridDim.x * (NTH / 32)) {\n    const int t = it >> 4, col = (it & 15) * 256 + lane * 8;\n    float a[8] = {0, 0, 0, 0, 0, 0, 0, 0};\n#pragma unroll\n    for (int k = 0; k < 8; k++) {\n      const int r = t * 8 + k;\n      unsigned e = p.ids[r] - p.lo;\n      if (e < NLOC) {\n        float cf = p.tw[r] * rsqrtf(__ldcg(p.ss + r) * (1.f / HD) + 1e-8f);\n        uint4 v = __ldcg((const uint4*)(p.y + (size_t)r * HD + col));\n        const __nv_bfloat162* hh = (const __nv_bfloat162*)&v;\n#pragma unroll\n        for (int j = 0; j < 4; j++) { float2 f = __bfloat1622float2(hh[j]); a[2 * j] += cf * f.x; a[2 * j + 1] += cf * f.y; }\n      }\n    }\n    uint4 o; __nv_bfloat162* ob = (__nv_bfloat162*)&o;\n#pragma unroll\n    for (int j = 0; j < 4; j++) ob[j] = __floats2bfloat162_rn(a[2*j]*p.nw[col+2*j], a[2*j+1]*p.nw[col+2*j+1]);\n    *(uint4*)(p.out + (size_t)t * HD + col) = o;\n  }\n#ifdef PROF\n  PSET(7, gtime()); PSET(9, clock64());\n#endif\n  __syncthreads();\n  if (tid == 0) {\n    __threadfence();\n    if (atomicAdd(p.ctr + C_DONE, 1) == (int)gridDim.x - 1) {\n      for (int i = 0; i <= C_DONE; i++) p.ctr[i] = 0;\n      __threadfence();\n    }\n  }\n}\n\nstatic int g_grid = 0;\nstatic size_t smem_bytes(int M) { return (size_t)NSLOT * SLOTB + NSLOT * 16 + sizeof(Sm) + (size_t)M * 8 * 2; }\n\nvoid moe(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor w1, torch::Tensor w2, torch::Tensor g1,\n         torch::Tensor g2, torch::Tensor nw, torch::Tensor inter, torch::Tensor y, torch::Tensor ss, torch::Tensor ctr,\n         torch::Tensor ws, torch::Tensor out, int64_t lo, int64_t ks, torch::Tensor prof) {\n  int M = x.size(0);\n  if (!g_grid) {\n    int dev; cudaGetDevice(&dev); int mx; cudaDeviceGetAttribute(&mx, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);\n    cudaDeviceGetAttribute(&g_grid, cudaDevAttrMultiProcessorCount, dev);\n    cudaFuncSetAttribute(moe_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mx);\n    g_grid *= CPB;\n  }\n  if (CPB > 1) { int nb = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, moe_kernel, NTH, smem_bytes(M)); TORCH_CHECK(nb >= CPB, "CPB not resident"); }\n  TORCH_CHECK(M * 8 <= 4096 && M * 8 * 3 <= NSLOT * SLOTB, "M too large");\n  P p{w1.data_ptr<uint8_t>(), w2.data_ptr<uint8_t>(), g1.data_ptr<float>(), g2.data_ptr<float>(),\n      (const __nv_bfloat16*)x.data_ptr(), ids.data_ptr<int>(), tw.data_ptr<float>(), nw.data_ptr<float>(),\n      (__nv_bfloat16*)inter.data_ptr(), (__nv_bfloat16*)y.data_ptr(), ss.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(),\n      ctr.data_ptr<int>(), ws.data_ptr<float>(), M, (int)lo, (int)ks,\n      prof.numel() ? (unsigned long long*)prof.data_ptr() : nullptr};\n  moe_kernel<<<g_grid, NTH, smem_bytes(M), at::cuda::getCurrentCUDAStream()>>>(p);\n}\n'

_mod = None
_DBG = []
# pair p of each 32-bit word: (eem rotation, sign rotation); bf16 targets m->6, e0->7, e1->8, s->15 (+16 for the high value)
_ROT = ((0, 0), (3, 1), (6, 6), (-5, 5))


def _big():
    pr = torch.cuda.get_device_properties(0)
    sm = getattr(pr, "shared_memory_per_block_optin", None)
    return sm >= 180000 if sm else (pr.major, pr.minor) in ((9, 0), (10, 0))


_mod8 = None
LARGE = True  # M > 64: 8 warps per SM with 4-stage rings and up to 64 tokens per pass (one dequant per stage)


def _ext(large=False):
    global _mod, _mod8
    if (_mod8 if large else _mod) is None:
        big = _big()
        wpc, stg = (16, 2) if big else (8, 2)
        extra = []
        if large:
            wpc, stg, extra = 8, (4 if big else 2), ["-DNTM=8", "-DSCR=2048"]
        for d in _DBG:
            if d.startswith("-DCFG=") and not large: wpc, stg = map(int, d[6:].split("x"))
        m = load_inline("moe_gridconstant_v30_%d_%d%s%s" % (wpc, stg, "_L" if large else "", "".join(_DBG).replace("-", "_").replace("=", "")),
                        cpp_sources="void moe(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,torch::Tensor);",
                        cuda_sources=CUDA, functions=["moe"],
                        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo", "-DWPC=%d" % wpc, "-DNSTG=%d" % stg] + extra + [d for d in _DBG if not d.startswith("-DCFG=")], verbose=False)
        if large: _mod8 = m
        else: _mod = m
    return _mod8 if large else _mod


def _place(word, code, pair, hi):
    de, ds = _ROT[pair]
    for bit, tgt, d in ((0, 6, de), (1, 7, de), (2, 8, de), (3, 15, ds)):
        word |= ((code >> bit) & 1) << ((tgt + 16 * hi - d) % 32)
    return word


def _repack(w, sf, perm):
    """w uint8 [R, K/2], sf e4m3 [R, K/16] -> uint8 [R/16, K/256, 2304]: per 16-row tile and 256-K half stage, 4 x 512 B of
    mma A fragments (lane-major uint4) then 32 lanes x 8 B of block scales."""
    R, K = w.shape[0], w.shape[1] * 2
    MT, NS = R // 16, K // 512
    codes = torch.stack((w & 15, w >> 4), -1).view(R, K)[perm].to(torch.int64)
    s8 = sf.view(torch.uint8)[perm].to(torch.int64)
    neg = (s8 >> 7).repeat_interleave(16, 1)
    codes = codes ^ (neg << 3)
    zero = ((s8 & 0x7F) == 0).repeat_interleave(16, 1)
    codes = torch.where(zero, torch.zeros_like(codes), codes)
    s8 = torch.where((s8 & 0x7F) == 0, torch.full_like(s8, 0x08), s8 & 0x7F)
    if ((s8 >> 3) == 0).any():
        raise ValueError("subnormal E4M3 block scales are not supported by this fast path")
    C = codes.view(MT, 16, NS, 8, 4, 4, 4)
    a, b = C[:, :8], C[:, 8:]
    word = torch.zeros(a.shape[:-1], dtype=torch.int64, device=w.device)
    for pair, (src, e0) in enumerate(((a, 0), (b, 0), (a, 2), (b, 2))):
        word = _place(word, src[..., e0], pair, 0)
        word = _place(word, src[..., e0 + 1], pair, 1)
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
    word = word.permute(0, 2, 3, 1, 4, 5).contiguous().view(torch.uint8).view(MT, NS, 2, 4, 512)
    S = s8.to(torch.uint8).view(MT, 16, NS, 8, 4)
    sc = torch.stack((S[:, :8], S[:, 8:]), -1).permute(0, 2, 1, 4, 3, 5).contiguous().view(MT, NS, 32, 2, 8)
    sc = sc.permute(0, 1, 3, 2, 4).contiguous()
    return torch.cat((word.view(MT, NS, 2, 2048), sc.view(MT, NS, 2, 256)), -1).contiguous().view(MT, NS * 2, 2304)


CUDA_F = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
// Match baseline silu_and_mul: BF16 gate/up inputs, FP32 activation/product,
// BF16 product store at each caller. Do not add a nonexistent SiLU tensor boundary.
__device__ __forceinline__ float muse_swiglu_bf16(float gate, float up) {
  gate = __bfloat162float(__float2bfloat16_rn(gate));
  up = __bfloat162float(__float2bfloat16_rn(up));
  return gate / (1.f + __expf(-gate)) * up;
}

#include <stdint.h>

#define HD 4096
#define NLOC 32
#define STG 4608
#define GR 128
#define RB 64
#define REC 1152
#ifndef FW1
#define FW1 30
#endif
#ifndef FW2
#define FW2 45
#endif
#ifndef FW4
#define FW4 76
#endif
#ifndef WPC
#define WPC 16
#endif
#ifndef SMAX
#define SMAX 3
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define TMA 1
#else
#define TMA 0
#endif

struct P {
  const uint8_t* w1; const uint8_t* w2; const float* g1; const float* g2;
  const __nv_bfloat16* x; const int* ids; const float* tw; const float* nw;
  float* yacc; __nv_bfloat16* out; int* ctr; int M, lo, smem; unsigned long long* prof;
};
struct Seg { int e, gr0, ng, T, NT, ioff; };
struct Sm { int cnt[NLOC], off[NLOC], fill[NLOC], act[NLOC]; int na, g, ks, S, ns, rounds, target, ioff, red, roff, ntm; Seg seg[3]; float ssr[8][WPC]; int fin[3], tq[32], ntq, mt; float g2s[NLOC], tws[256]; };

__device__ __forceinline__ void cp16(uint32_t s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(g) : "memory"); }
__device__ __forceinline__ uint4 lds(uint32_t a) { uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v; }
__device__ __forceinline__ uint2 lds64(uint32_t a) { uint2 v; asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\n" : "=r"(v.x), "=r"(v.y) : "r"(a)); return v; }
__device__ __forceinline__ uint32_t lds16(uint32_t a) { unsigned short v; asm volatile("ld.shared.u16 %0, [%1];\n" : "=h"(v) : "r"(a)); return v; }
__device__ __forceinline__ uint32_t bmul(uint32_t a, uint32_t b) { uint32_t d; asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u)); return d; }
__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ int ldacq(const int* p) { int v; asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void gbar(int id, int n) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ uint32_t sc2(uint32_t w, uint32_t sel) { return __byte_perm(w, 0, sel) * 16u + 0x77007700u; }
__device__ __forceinline__ void mb_init(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(b), "r"(n)); }
__device__ __forceinline__ void mb_wait(uint32_t bar, uint32_t ph) {
#if TMA
  asm volatile("{\n.reg .pred p;\nW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" :: "r"(bar), "r"(ph) : "memory");
#else
  asm volatile("{\n.reg .pred p;\nW: mbarrier.test_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" :: "r"(bar), "r"(ph) : "memory");
#endif
}
#ifdef FAKEDQ
#define BMUL(a, b) ((a) ^ (b))
#else
#define BMUL(a, b) bmul(a, b)
#endif
#ifdef NOMMA
#define MMA(c, a0, a1, a2, a3, b0, b1) (c)[0] += __uint_as_float((a0) ^ (a1) ^ (a2) ^ (a3) ^ (b0) ^ (b1))
#else
#define MMA mma
#endif
#ifdef PROF
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
#define PST(k) do { if (threadIdx.x == 0 && p.prof) p.prof[blockIdx.x * 16 + (k)] = gtime(); } while (0)
#else
#define PST(k)
#endif

// per-warp TMA ring over a fixed byte sequence: W13 stages (lt, s) of this warp, then W2 records (rowblock, segment, 4 granules)
struct Ctx {
  const P* p; Sm* sm; uint32_t ring, bars; int lane, wid, S, q; unsigned long long wns;
  int r, s, j, sg, gc; bool ph2;
  __device__ __forceinline__ void load(int slot, const uint8_t* src, int bytes) {
    const uint32_t dst = ring + slot * STG, bar = bars + slot * 8;
#if TMA
    if (lane == 0) {
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "r"(bytes) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
    }
#else
    for (int k = lane; k < bytes / 16; k += 32) cp16(dst + k * 16, src + k * 16);
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" :: "r"(bar) : "memory");
#endif
  }
  __device__ __forceinline__ void load2(int slot, const uint8_t* sa, const uint8_t* sb) {  // two 2304 B half-stages
    const uint32_t dst = ring + slot * STG, bar = bars + slot * 8;
#if TMA
    if (lane == 0) {
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "n"(STG) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 2304, [%2];\n" :: "r"(dst), "l"(sa), "r"(bar) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 2304, [%2];\n" :: "r"(dst + 2304), "l"(sb), "r"(bar) : "memory");
    }
#else
    for (int k = lane; k < 144; k += 32) { cp16(dst + k * 16, sa + k * 16); cp16(dst + 2304 + k * 16, sb + k * 16); }
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" :: "r"(bar) : "memory");
#endif
  }
  // phase-A cursor: the per-unit source pointer is computed once per unit (no divisions per stage)
  int a_step, a_w, a_h0, a_span, a_units, a_mt; const uint8_t* ab;
  __device__ __forceinline__ void init_a() {
    const int ks = sm->ks;
    a_mt = sm->mt; a_step = WPC / ks; a_w = wid / ks; a_span = 8 * a_mt / ks; a_h0 = (wid % ks) * a_span; a_units = 4 * sm->g / a_mt;
  }
  __device__ __forceinline__ void issue(int slot) {
    if (!ph2) {
      if (s == 0) {
        const int u = r * a_step + a_w;
        if (u >= a_units) ph2 = true;
        else {
          const int lt = u * a_mt, gq = lt >> 2, si = gq < sm->seg[0].ng ? 0 : (gq < sm->seg[0].ng + sm->seg[1].ng ? 1 : 2);
          const Seg& sq = sm->seg[si];
          const int gl = gq - (si ? sm->seg[0].ng + (si > 1 ? sm->seg[1].ng : 0) : 0);
          ab = p->w1 + ((size_t)(p->lo + sq.e) * 512 + (sq.gr0 + gl) * 4 + (lt & 3)) * 8 * STG;
        }
      }
      if (!ph2) {
        const int h = a_h0 + s;
        if (a_mt == 2) { const uint8_t* src = ab + (h >> 1) * STG + (h & 1) * 2304; load2(slot, src, src + 8 * STG); }
        else load(slot, ab + h * STG, STG);
        if (++s == a_span) { s = 0; r++; }
        return;
      }
    }
    while (j < RB / WPC) {
      if (sg < sm->ns) {
        const Seg& sq = sm->seg[sg];
        const int n = min(4, sq.ng - gc), rb = wid + j * WPC;
        load(slot, p->w2 + (((size_t)sq.e * RB + rb) * GR + sq.gr0 + gc) * REC, n * REC);
        gc += 4; if (gc >= sq.ng) { gc = 0; sg++; }
        return;
      }
      sg = 0; j++;
    }
  }
  __device__ __forceinline__ uint32_t acquire() {
    const int slot = q % S;
#ifdef PROF
    const unsigned long long w0 = gtime();
#endif
    mb_wait(bars + slot * 8, (q / S) & 1);
    __syncwarp();
#ifdef PROF
    wns += gtime() - w0;
#endif
    q++;
    return ring + slot * STG;
  }
  __device__ __forceinline__ void release() { __syncwarp(); issue((q - 1) % S); }
};

template <int NT>
__device__ __forceinline__ void run13(Ctx& C, const uint16_t* list, int off, int cnt, int s0, int s1, float (&f)[NT][4]) {
  const P& p = *C.p;
  const int lane = C.lane, g = lane >> 2, t = lane & 3;
  constexpr int CH = NT == 1 ? 4 : (NT == 2 ? 2 : 1);
#ifndef FD1
#define FD1 7
#endif
  constexpr int D = NT == 1 ? FD1 : (NT == 2 ? 3 : 1);
  const uint4* rp[NT];
#pragma unroll
  for (int n = 0; n < NT; n++) { int i = n * 8 + g; int r = list[off + (i < cnt ? i : 0)]; rp[n] = (const uint4*)(p.x + (size_t)(r >> 3) * HD + t * 16); }
  uint4 B[D + 1][NT][2];
  auto ldb = [&](uint4 (&b)[NT][2], int s, int c) {
#pragma unroll
    for (int n = 0; n < NT; n++) {
#ifdef NOXB
      b[n][0] = make_uint4(s, c, n, 1); b[n][1] = make_uint4(c, s, n, 2);
#else
      const uint4* q = rp[n] + s * 64 + c * 8; b[n][0] = __ldg(q); b[n][1] = __ldg(q + 1);
#endif
    }
  };
  float acc[NT][CH][4];
#pragma unroll
  for (int n = 0; n < NT; n++)
#pragma unroll
    for (int h = 0; h < CH; h++) acc[n][h][0] = acc[n][h][1] = acc[n][h][2] = acc[n][h][3] = 0.f;
#pragma unroll
  for (int c = 0; c < D; c++) ldb(B[c], s0, c);
  for (int s = s0; s < s1; s++) {
    const uint32_t slot = C.acquire() + lane * 16;
#ifdef NOCOMP
    { const uint4 w = lds(slot); acc[0][0][0] += __uint_as_float(w.x ^ w.y); C.release(); continue; }
#endif
    const uint2 sh0 = lds64(slot - lane * 8 + 2048), sh1 = lds64(slot - lane * 8 + 4352);
    const uint4 sc = make_uint4(sh0.x, sh0.y, sh1.x, sh1.y);
#pragma unroll
    for (int c = 0; c < 8; c++) {
      const uint4 w = lds(slot + c * 512 + (c >= 4 ? 256 : 0));
      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w));
      const uint32_t sA = sc2(sw, (c & 1) ? 0x4242 : 0x4040), sB = sc2(sw, (c & 1) ? 0x4343 : 0x4141);
      if (c + D < 8) ldb(B[(c + D) % (D + 1)], s, c + D);
      else ldb(B[(c + D) % (D + 1)], s + 1 < s1 ? s + 1 : s, c + D - 8);  // clamped: no branch
      const uint4 (&bc)[NT][2] = B[c % (D + 1)];
      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const uint32_t q = ww[j];
        const uint32_t a0 = BMUL(q & 0x81C081C0u, sA);
        const uint32_t a1 = BMUL(((q << 3) & 0x01C001C0u) | ((q << 1) & 0x80008000u), sB);
        const uint32_t a2 = BMUL((q << 6) & 0x81C081C0u, sA);
        const uint32_t a3 = BMUL(((q >> 5) & 0x01C001C0u) | ((q << 5) & 0x80008000u), sB);
#pragma unroll
        for (int n = 0; n < NT; n++) {
          const uint32_t* bb = (const uint32_t*)&bc[n][j >> 1];
          MMA(acc[n][j % CH], a0, a1, a2, a3, bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);
        }
      }
    }
    C.release();
  }
#pragma unroll
  for (int n = 0; n < NT; n++)
#pragma unroll
    for (int i = 0; i < 4; i++) { float v = 0.f;
#pragma unroll
      for (int h = 0; h < CH; h++) v += acc[n][h][i];
      f[n][i] = v; }
}

// one W13 tile part: (optional group split-K reduction) then SwiGLU into the CTA's inter buffer
template <int NT>
__device__ __forceinline__ void tile13(Ctx& C, const uint16_t* list, uint8_t* smem, int r, int lt) {
  Sm* sm = C.sm; const P& p = *C.p;
  const int lane = C.lane, wid = C.wid, g = lane >> 2, t = lane & 3, ks = sm->ks;
  const int gq = lt >> 2, si = gq < sm->seg[0].ng ? 0 : (gq < sm->seg[0].ng + sm->seg[1].ng ? 1 : 2);
  const Seg& sq = sm->seg[si];
  const int gl = gq - (si ? sm->seg[0].ng + (si > 1 ? sm->seg[1].ng : 0) : 0);
  const int e = sq.e, cnt = sq.T, off = sm->off[e];
  const int s0 = (wid % ks) * (8 / ks);
  float f[NT][4];
  run13<NT>(C, list, off, cnt, s0, s0 + 8 / ks, f);
  if (ks > 1) {
    float4* red = (float4*)(smem + sm->red);
    const int ntm = sm->ntm;
#pragma unroll
    for (int n = 0; n < NT; n++) red[((wid * 2 + (r & 1)) * ntm + n) * 32 + lane] = make_float4(f[n][0], f[n][1], f[n][2], f[n][3]);
    gbar(1 + wid / ks, ks * 32);
    if (wid % ks != (r & (ks - 1))) return;
    const int w0 = wid - wid % ks;
    for (int o = w0; o < w0 + ks; o++) {
      if (o == wid) continue;
#pragma unroll
      for (int n = 0; n < NT; n++) { float4 v = red[((o * 2 + (r & 1)) * ntm + n) * 32 + lane]; f[n][0] += v.x; f[n][1] += v.y; f[n][2] += v.z; f[n][3] += v.w; }
    }
  }
  const float gs = p.g1[p.lo + e];
  __nv_bfloat16* is = (__nv_bfloat16*)(smem + sm->ioff + sq.ioff) + (size_t)gl * NT * 8 * 32 + (lt & 3) * 8 + g;
#pragma unroll
  for (int n = 0; n < NT; n++)
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const float gt = f[n][h] * gs, up = f[n][2 + h] * gs;
      is[(n * 8 + 2 * t + h) * 32] = __float2bfloat16(muse_swiglu_bf16(gt, up));
    }
}

#ifdef MT2
// two adjacent W13 tiles per warp: each ring slot holds the same 256-K half-stage of both, so every activation B fragment
// is loaded once for two tiles (half the activation traffic and B registers of run13)
template <int NT>
__device__ __forceinline__ void run13p(Ctx& C, const uint16_t* list, int off, int cnt, int h0, int h1, float (&f)[2][NT][4]) {
  const P& p = *C.p;
  const int lane = C.lane, g = lane >> 2, t = lane & 3;
  constexpr int CH = NT == 1 ? 2 : 1;
#ifndef PD1
#define PD1 3
#endif
  constexpr int D = NT == 1 ? PD1 : 1;  // (D + 1) divides 4
  const uint4* rp[NT];
#pragma unroll
  for (int n = 0; n < NT; n++) { int i = n * 8 + g; int r = list[off + (i < cnt ? i : 0)]; rp[n] = (const uint4*)(p.x + (size_t)(r >> 3) * HD + t * 16); }
  uint4 B[D + 1][NT][2];
  auto ldb = [&](uint4 (&b)[NT][2], int h, int c) {
#pragma unroll
    for (int n = 0; n < NT; n++) {
#ifdef NOXB
      b[n][0] = make_uint4(h, c, n, 1); b[n][1] = make_uint4(c, h, n, 2);
#else
      const uint4* q = rp[n] + (h >> 1) * 64 + ((h & 1) * 4 + c) * 8; b[n][0] = __ldg(q); b[n][1] = __ldg(q + 1);
#endif
    }
  };
  float acc[2][NT][CH][4];
#pragma unroll
  for (int k = 0; k < 2; k++)
#pragma unroll
    for (int n = 0; n < NT; n++)
#pragma unroll
      for (int hh = 0; hh < CH; hh++) acc[k][n][hh][0] = acc[k][n][hh][1] = acc[k][n][hh][2] = acc[k][n][hh][3] = 0.f;
#pragma unroll
  for (int c = 0; c < D; c++) ldb(B[c], h0, c);
  for (int h = h0; h < h1; h++) {
    const uint32_t slot = C.acquire() + lane * 16;
    uint2 sh[2];
#pragma unroll
    for (int k = 0; k < 2; k++) sh[k] = lds64(slot - lane * 8 + k * 2304 + 2048);
#pragma unroll
    for (int c = 0; c < 4; c++) {
      if (c + D < 4) ldb(B[(c + D) % (D + 1)], h, c + D);
      else ldb(B[(c + D) % (D + 1)], h + 1 < h1 ? h + 1 : h, c + D - 4);
      const uint4 (&bc)[NT][2] = B[c % (D + 1)];
#pragma unroll
      for (int k = 0; k < 2; k++) {
        const uint4 w = lds(slot + k * 2304 + c * 512);
        const uint32_t sw = c < 2 ? sh[k].x : sh[k].y;
        const uint32_t sA = sc2(sw, (c & 1) ? 0x4242 : 0x4040), sB = sc2(sw, (c & 1) ? 0x4343 : 0x4141);
        const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const uint32_t q = ww[j];
          const uint32_t a0 = BMUL(q & 0x81C081C0u, sA);
          const uint32_t a1 = BMUL(((q << 3) & 0x01C001C0u) | ((q << 1) & 0x80008000u), sB);
          const uint32_t a2 = BMUL((q << 6) & 0x81C081C0u, sA);
          const uint32_t a3 = BMUL(((q >> 5) & 0x01C001C0u) | ((q << 5) & 0x80008000u), sB);
#pragma unroll
          for (int n = 0; n < NT; n++) {
            const uint32_t* bb = (const uint32_t*)&bc[n][j >> 1];
            MMA(acc[k][n][j % CH], a0, a1, a2, a3, bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);
          }
        }
      }
    }
    C.release();
  }
#pragma unroll
  for (int k = 0; k < 2; k++)
#pragma unroll
    for (int n = 0; n < NT; n++)
#pragma unroll
      for (int i = 0; i < 4; i++) { float v = 0.f;
#pragma unroll
        for (int hh = 0; hh < CH; hh++) v += acc[k][n][hh][i];
        f[k][n][i] = v; }
}

template <int NT>
__device__ __forceinline__ void tile13p(Ctx& C, const uint16_t* list, uint8_t* smem, int r, int u) {
  Sm* sm = C.sm; const P& p = *C.p;
  const int lane = C.lane, wid = C.wid, g = lane >> 2, t = lane & 3, ks = sm->ks;
  const int lt = 2 * u, gq = lt >> 2, si = gq < sm->seg[0].ng ? 0 : (gq < sm->seg[0].ng + sm->seg[1].ng ? 1 : 2);
  const Seg& sq = sm->seg[si];
  const int gl = gq - (si ? sm->seg[0].ng + (si > 1 ? sm->seg[1].ng : 0) : 0);
  const int e = sq.e, cnt = sq.T, off = sm->off[e];
  const int h0 = (wid % ks) * (16 / ks);
  float f[2][NT][4];
  run13p<NT>(C, list, off, cnt, h0, h0 + 16 / ks, f);
  if (ks > 1) {
    float4* red = (float4*)(smem + sm->red);
    const int ntm = sm->ntm;
#pragma unroll
    for (int k = 0; k < 2; k++)
#pragma unroll
      for (int n = 0; n < NT; n++) red[(((wid * 2 + (r & 1)) * 2 + k) * ntm + n) * 32 + lane] = make_float4(f[k][n][0], f[k][n][1], f[k][n][2], f[k][n][3]);
    gbar(1 + wid / ks, ks * 32);
    if (wid % ks != (r & (ks - 1))) return;
    const int w0 = wid - wid % ks;
    for (int o = w0; o < w0 + ks; o++) {
      if (o == wid) continue;
#pragma unroll
      for (int k = 0; k < 2; k++)
#pragma unroll
        for (int n = 0; n < NT; n++) { float4 v = red[(((o * 2 + (r & 1)) * 2 + k) * ntm + n) * 32 + lane]; f[k][n][0] += v.x; f[k][n][1] += v.y; f[k][n][2] += v.z; f[k][n][3] += v.w; }
    }
  }
  const float gs = p.g1[p.lo + e];
#pragma unroll
  for (int k = 0; k < 2; k++) {
    __nv_bfloat16* is = (__nv_bfloat16*)(smem + sm->ioff + sq.ioff) + (size_t)gl * NT * 8 * 32 + ((lt + k) & 3) * 8 + g;
#pragma unroll
    for (int n = 0; n < NT; n++)
#pragma unroll
      for (int hh = 0; hh < 2; hh++) {
        const float gt = f[k][n][hh] * gs, up = f[k][n][2 + hh] * gs;
        is[(n * 8 + 2 * t + hh) * 32] = __float2bfloat16(muse_swiglu_bf16(gt, up));
      }
  }
}
#endif

// W2 column slice of one segment for one 64-row block: partial y (fp32) reduced into yacc
template <int NT>
__device__ __forceinline__ void seg2(Ctx& C, const uint16_t* list, uint8_t* smem, int sgi, int rb) {
  Sm* sm = C.sm; const P& p = *C.p;
  const int lane = C.lane, g = lane >> 2, t = lane & 3;
  const Seg& sq = sm->seg[sgi];
  const uint32_t ib = (uint32_t)__cvta_generic_to_shared(smem + sm->ioff + sq.ioff) + (g * 32 + t * 8) * 2;
  float acc[4][NT][4];
#pragma unroll
  for (int rt = 0; rt < 4; rt++)
#pragma unroll
    for (int n = 0; n < NT; n++) acc[rt][n][0] = acc[rt][n][1] = acc[rt][n][2] = acc[rt][n][3] = 0.f;
  for (int gc = 0; gc < sq.ng; gc += 4) {
    const uint32_t slot = C.acquire();
    const int n4 = min(4, sq.ng - gc);
#ifdef NOCOMP
    { const uint2 w = lds64(slot + lane * 8); acc[0][0][0] += __uint_as_float(w.x ^ w.y); C.release(); continue; }
#endif
#pragma unroll
    for (int gi = 0; gi < 4; gi++) {
      if (gi < n4) {
        uint4 bv[NT];
#pragma unroll
        for (int n = 0; n < NT; n++) bv[n] = lds(ib + ((gc + gi) * NT * 8 + n * 8) * 64);
#pragma unroll
        for (int rt = 0; rt < 4; rt++) {
          const uint32_t base = slot + gi * REC + rt * 288;
          const uint2 w = lds64(base + lane * 8);
          const uint32_t s16 = lds16(base + 256 + g * 4 + (t >> 1) * 2);
          const uint32_t sA = sc2(s16, 0x4040), sB = sc2(s16, 0x4141);
#pragma unroll
          for (int h = 0; h < 2; h++) {
            const uint32_t q = h ? w.y : w.x;
            const uint32_t a0 = BMUL(q & 0x81C081C0u, sA);
            const uint32_t a1 = BMUL(((q << 3) & 0x01C001C0u) | ((q << 1) & 0x80008000u), sB);
            const uint32_t a2 = BMUL((q << 6) & 0x81C081C0u, sA);
            const uint32_t a3 = BMUL(((q >> 5) & 0x01C001C0u) | ((q << 5) & 0x80008000u), sB);
#pragma unroll
            for (int n = 0; n < NT; n++) MMA(acc[rt][n], a0, a1, a2, a3, h ? bv[n].z : bv[n].x, h ? bv[n].w : bv[n].y);
          }
        }
      }
    }
    C.release();
  }
  const int off = sm->off[sq.e], T = sq.T;
  const bool odd = (g & 1) != 0; (void)odd;
#pragma unroll
  for (int rt = 0; rt < 4; rt++)
#pragma unroll
    for (int n = 0; n < NT; n++) {
      const float c0 = acc[rt][n][0], c1 = acc[rt][n][1], c2 = acc[rt][n][2], c3 = acc[rt][n][3];
      const int row = rb * 64 + rt * 16 + 2 * g;
#if TMA
      const float r0 = __shfl_xor_sync(0xffffffffu, odd ? c0 : c1, 4), r1 = __shfl_xor_sync(0xffffffffu, odd ? c2 : c3, 4);
      const int tok = n * 8 + 2 * t + (odd ? 1 : 0);
      if (tok < T) {
        float* a = p.yacc + (size_t)list[off + tok] * HD + (odd ? row - 2 : row);
        const float4 v = odd ? make_float4(r0, r1, c1, c3) : make_float4(c0, c2, r0, r1);
#ifndef NORED
        asm volatile("red.global.add.v4.f32 [%0], {%1,%2,%3,%4};\n" :: "l"(a), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
#else
        if (v.x == 123.f) *a = v.y;
#endif
      }
#else
      const int ta = n * 8 + 2 * t;
      if (ta < T) { float* a = p.yacc + (size_t)list[off + ta] * HD + row; atomicAdd(a, c0); atomicAdd(a + 1, c2); }
      if (ta + 1 < T) { float* a = p.yacc + (size_t)list[off + ta + 1] * HD + row; atomicAdd(a, c1); atomicAdd(a + 1, c3); }
#endif
    }
}

__global__ void __launch_bounds__(WPC * 32, 1) moe_f(P p) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, R = p.M * 8;
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem);
  Sm* sm = (Sm*)(smem + WPC * SMAX * 8);
  uint16_t* list = (uint16_t*)(sm + 1);
  uint8_t* lid = (uint8_t*)(list + R);
  const int fixed = (WPC * SMAX * 8 + (int)sizeof(Sm) + R * 3 + 127) & ~127;
  PST(0);
  if (lane < SMAX) mb_init(sb + (wid * SMAX + lane) * 8, TMA ? 1 : 32);
#if TMA
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
  if (tid < NLOC) sm->cnt[tid] = 0;
  __syncthreads();
  for (int i = tid; i < R; i += blockDim.x) { unsigned e = p.ids[i] - p.lo; lid[i] = e < NLOC ? e : 0xff; if (e < NLOC) atomicAdd(&sm->cnt[e], 1); if (i < 256) sm->tws[i] = p.tw[i]; }
  if (tid < NLOC) sm->g2s[tid] = p.g2[p.lo + tid];
  __syncthreads();
  if (wid == 0) {
    const int c = sm->cnt[lane]; int v = c;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { int n = __shfl_up_sync(0xffffffffu, v, o); if (lane >= o) v += n; }
    sm->off[lane] = v - c; sm->fill[lane] = v - c;
    const unsigned m = __ballot_sync(0xffffffffu, c > 0);
    if (c > 0) sm->act[__popc(m & ((1u << lane) - 1))] = lane;
    __syncwarp();
    if (lane == 0) {
      const int na = __popc(m);
      // cost-weighted granule ranges: a granule of an expert with more tokens costs more (x re-reads, mma per n-block)
      long long Ct = 0;
      for (int a = 0; a < na; a++) { const int T = sm->cnt[sm->act[a]]; Ct += GR * (T <= 8 ? FW1 : (T <= 16 ? FW2 : FW4)); }
      const long long lo_c = (long long)blockIdx.x * Ct / gridDim.x, hi_c = (long long)(blockIdx.x + 1) * Ct / gridDim.x;
      int ns = 0, io = 0, ntm = 1, G1 = 0, G0 = 0;
      for (int k = 0; k < 3; k++) sm->seg[k] = Seg{0, 0, 0, 0, 1, 0};
      long long base = 0;
      for (int a = 0; a < na; a++) {
        const int e = sm->act[a], T = sm->cnt[e], w = T <= 8 ? FW1 : (T <= 16 ? FW2 : FW4);
        const int NT = T <= 8 ? 1 : (T <= 16 ? 2 : 4);
        int k0 = lo_c > base ? (int)((lo_c - base + w - 1) / w) : 0, k1 = hi_c > base ? (int)((hi_c - base + w - 1) / w) : 0;
        k0 = min(k0, GR); k1 = min(k1, GR);
        if (k1 > k0 && ns < 3) { sm->seg[ns++] = Seg{e, k0, k1 - k0, T, NT, io}; io += (k1 - k0) * NT * 8 * 64; ntm = max(ntm, NT); G1 += k1 - k0; }
        base += (long long)GR * w;
      }
      const int g = G1 - G0, tiles = 4 * g;
#ifdef MT2
      const int MTL = ntm == 1 ? 2 : 1;  // two-tile warps only when every segment has <= 8 tokens
#else
      const int MTL = 1;
#endif
      sm->mt = MTL;
      int bk = 1; float bc = 1e30f;
      for (int ks = 1; ks <= 8; ks <<= 1) {
        if (ks > 1 && ntm > 2) break;
        const int rounds = (tiles / MTL * ks + WPC - 1) / WPC;
        const float cost = rounds * (8.f * MTL / ks + (ks > 1 ? 0.35f : 0.f));
        if (cost < bc - 1e-3f) { bc = cost; bk = ks; }
      }
#ifdef KSF
      if (KSF == 1 || ntm <= 2) bk = KSF;
#endif
      const int red = bk > 1 ? WPC * 2 * ntm * 512 * MTL : 0, roff = (fixed + io + red + 127) & ~127;
      int S = (p.smem - roff) / (WPC * STG);
      sm->roff = roff; sm->ntm = ntm;
      sm->na = na; sm->g = g; sm->ks = bk; sm->ns = ns; sm->rounds = (tiles / MTL * bk + WPC - 1) / WPC;
      sm->ioff = fixed; sm->red = fixed + io; sm->S = S < SMAX ? (S < 1 ? 1 : S) : SMAX;
      if (S < 1) { sm->g = 0; asm volatile("trap;"); }
    }
  }
  __syncthreads();
  Ctx C;
  C.p = &p; C.sm = sm; C.lane = lane; C.wid = wid; C.S = sm->S; C.q = 0; C.wns = 0;
  C.ring = sb + sm->roff + wid * C.S * STG;
  C.bars = sb + wid * SMAX * 8;
  C.r = 0; C.s = 0; C.j = 0; C.sg = 0; C.gc = 0; C.ph2 = false;
  C.init_a();
  const bool any = sm->g > 0;
  if (any)
    for (int i = 0; i < C.S; i++) C.issue(i);
  for (int i = tid; i < R; i += blockDim.x) { unsigned e = lid[i]; if (e < NLOC) list[atomicAdd(&sm->fill[e], 1)] = i; }
  __syncthreads();
  PST(1);
  if (any) {
    const int ks = sm->ks, g = sm->g;
    for (int r = 0; r < sm->rounds; r++) {
#ifdef MT2
     if (sm->mt == 2) {
      const int u = r * (WPC / ks) + wid / ks;
      if (u >= 2 * g) break;
      const int gq2 = u >> 1, si2 = gq2 < sm->seg[0].ng ? 0 : (gq2 < sm->seg[0].ng + sm->seg[1].ng ? 1 : 2);
      const int nt2 = sm->seg[si2].NT;
      if (nt2 == 1) tile13p<1>(C, list, smem, r, u);
      else if (nt2 == 2) tile13p<2>(C, list, smem, r, u);
      else tile13p<4>(C, list, smem, r, u);
      continue;
     }
#endif
      const int lt = r * (WPC / ks) + wid / ks;
      if (lt >= 4 * g) break;
      const int gq = lt >> 2, si = gq < sm->seg[0].ng ? 0 : (gq < sm->seg[0].ng + sm->seg[1].ng ? 1 : 2);
      const int nt = sm->seg[si].NT;
      if (nt == 1) tile13<1>(C, list, smem, r, lt);
      else if (nt == 2) tile13<2>(C, list, smem, r, lt);
      else tile13<4>(C, list, smem, r, lt);
    }
  }
#ifdef PROF
  if (lane == 0 && p.prof) atomicAdd(p.prof + blockIdx.x * 16 + 8, C.wns);
  C.wns = 0;
  if (tid == 0 && p.prof) {
    unsigned long long cmp = 0;
    for (int k = 0; k < sm->ns; k++) cmp += (unsigned long long)sm->seg[k].ng * (sm->seg[k].NT == 1 ? 1ull : (sm->seg[k].NT == 2 ? 1000ull : 1000000ull));
    p.prof[blockIdx.x * 16 + 11] = cmp; p.prof[blockIdx.x * 16 + 12] = sm->ks * 10 + sm->S;
  }
#endif
  __syncthreads();
  PST(2);
  if (any) {
    for (int j = 0; j < RB / WPC; j++)
      for (int k = 0; k < sm->ns; k++) {
        const int nt = sm->seg[k].NT, rb = wid + j * WPC;
        if (nt == 1) seg2<1>(C, list, smem, k, rb);
        else if (nt == 2) seg2<2>(C, list, smem, k, rb);
        else seg2<4>(C, list, smem, k, rb);
      }
  }
#ifdef PROF
  if (lane == 0 && p.prof) atomicAdd(p.prof + blockIdx.x * 16 + 9, C.wns);
#endif
  __syncthreads();
  PST(3);
#ifdef LASTA
  // no grid barrier: each CTA adds its granule count of every segment to the counters of that segment's tokens; the CTA
  // that brings a token to 128 x (its local routes) normalizes and combines it. Counters are reset by their completers.
  {
    int* tokd = p.ctr + 9280;
    if (tid == 0) sm->ntq = 0;
    __syncthreads();
    for (int k = 0; k < sm->ns; k++) {
      const int e = sm->seg[k].e, T = sm->seg[k].T, off = sm->off[e], ng = sm->seg[k].ng;
      if (tid < T) {
        const int t = list[off + tid] >> 3;
        int nl = 0;
#pragma unroll
        for (int j = 0; j < 8; j++) nl += lid[t * 8 + j] < NLOC;
        int old;
        asm volatile("atom.add.acq_rel.gpu.global.s32 %0, [%1], %2;\n" : "=r"(old) : "l"(tokd + t), "r"(ng) : "memory");
        if (old + ng == GR * nl) { tokd[t] = 0; sm->tq[atomicAdd(&sm->ntq, 1)] = t; }
      }
    }
    __syncthreads();
    PST(4);
    const int ntq = sm->ntq;
    float* stg = (float*)(smem + sm->roff);
    const bool stage = 8 * HD * 4 <= sm->S * WPC * STG;
    constexpr int CPT = HD / (WPC * 32);
    for (int qi = 0; qi < ntq; qi++) {
      const int t = sm->tq[qi];
      float ssk[8];
#pragma unroll
      for (int k = 0; k < 8; k++) ssk[k] = 0.f;
#pragma unroll
      for (int c = 0; c < CPT; c += 8) {
        const int col = (c / 8 * WPC * 32 + tid) * 8;
        const float4 n0 = *(const float4*)(p.nw + col), n1 = *(const float4*)(p.nw + col + 4);
        float4 u[8][2];
#pragma unroll
        for (int k = 0; k < 8; k++) {
          if (lid[t * 8 + k] < NLOC) { const float* yr = p.yacc + (size_t)(t * 8 + k) * HD + col; u[k][0] = __ldcg((const float4*)yr); u[k][1] = __ldcg((const float4*)(yr + 4)); }
          else u[k][0] = u[k][1] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
#pragma unroll
        for (int k = 0; k < 8; k++) {
          if (stage && lid[t * 8 + k] < NLOC) { *(float4*)(stg + k * HD + col) = u[k][0]; *(float4*)(stg + k * HD + col + 4) = u[k][1]; }
          const float a0 = u[k][0].x * n0.x, a1 = u[k][0].y * n0.y, a2 = u[k][0].z * n0.z, a3 = u[k][0].w * n0.w;
          const float a4 = u[k][1].x * n1.x, a5 = u[k][1].y * n1.y, a6 = u[k][1].z * n1.z, a7 = u[k][1].w * n1.w;
          ssk[k] += a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3 + a4 * a4 + a5 * a5 + a6 * a6 + a7 * a7;
        }
      }
#pragma unroll
      for (int k = 0; k < 8; k++) {
        float ss = ssk[k];
#pragma unroll
        for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        if (lane == 0) sm->ssr[k][wid] = ss;
      }
      __syncthreads();
      PST(6);
      float coef[8];
#pragma unroll
      for (int k = 0; k < 8; k++) {
        const unsigned e = lid[t * 8 + k];
        float s = 0.f;
#pragma unroll
        for (int w = 0; w < WPC; w++) s += sm->ssr[k][w];
        const float gs = e < NLOC ? sm->g2s[e] : 0.f;
        coef[k] = e < NLOC ? sm->tws[t * 8 + k] * gs * rsqrtf(s * gs * gs * (1.f / HD) + 1e-8f) : 0.f;
      }
      PST(7);
#pragma unroll
      for (int c = 0; c < CPT; c += 8) {
        const int col = (c / 8 * WPC * 32 + tid) * 8;
        const float4 n0 = *(const float4*)(p.nw + col), n1 = *(const float4*)(p.nw + col + 4);
        float o[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int k = 0; k < 8; k++) {
          if (lid[t * 8 + k] >= NLOC) continue;
          float4 u0, u1;
          if (stage) { u0 = *(const float4*)(stg + k * HD + col); u1 = *(const float4*)(stg + k * HD + col + 4); }
          else { const float* yr = p.yacc + (size_t)(t * 8 + k) * HD + col; u0 = __ldcg((const float4*)yr); u1 = __ldcg((const float4*)(yr + 4)); }
          const float cf = coef[k];
          o[0] += cf * u0.x; o[1] += cf * u0.y; o[2] += cf * u0.z; o[3] += cf * u0.w;
          o[4] += cf * u1.x; o[5] += cf * u1.y; o[6] += cf * u1.z; o[7] += cf * u1.w;
        }
        uint4 ov; __nv_bfloat162* ob = (__nv_bfloat162*)&ov;
        ob[0] = __floats2bfloat162_rn(o[0] * n0.x, o[1] * n0.y); ob[1] = __floats2bfloat162_rn(o[2] * n0.z, o[3] * n0.w);
        ob[2] = __floats2bfloat162_rn(o[4] * n1.x, o[5] * n1.y); ob[3] = __floats2bfloat162_rn(o[6] * n1.z, o[7] * n1.w);
        *(uint4*)(p.out + (size_t)t * HD + col) = ov;
#pragma unroll
        for (int k = 0; k < 8; k++)
          if (lid[t * 8 + k] < NLOC) { float* yr = p.yacc + (size_t)(t * 8 + k) * HD + col; __stcg((float4*)yr, make_float4(0.f, 0.f, 0.f, 0.f)); __stcg((float4*)(yr + 4), make_float4(0.f, 0.f, 0.f, 0.f)); }
      }
      __syncthreads();
    }
    if ((int)blockIdx.x < p.M) {  // tokens without a local route
      int nl = 0;
#pragma unroll
      for (int j = 0; j < 8; j++) nl += lid[blockIdx.x * 8 + j] < NLOC;
      if (nl == 0)
        for (int c = tid * 8; c < HD; c += WPC * 256) *(uint4*)(p.out + (size_t)blockIdx.x * HD + c) = make_uint4(0, 0, 0, 0);
    }
    PST(5);
    return;
  }
#endif
  if (tid == 0) {
    int old;
    asm volatile("atom.add.release.gpu.global.s32 %0, [%1], 1;\n" : "=r"(old) : "l"(p.ctr) : "memory");
    sm->target = (old / (int)gridDim.x + 1) * (int)gridDim.x;
  }
  __syncthreads();
  // combine: token b's 4096 columns split over P CTAs (P = 32 at M <= 4); per-slice partial sums of squares go through a
  // flag-synchronized exchange (flags carry the call epoch: no reset), then each CTA normalizes and sums its column slice.
  int P = 32;
  while (P > 1 && P * p.M > (int)gridDim.x) P >>= 1;
  if ((int)blockIdx.x >= p.M * P) return;
  const int b = blockIdx.x / P, js = blockIdx.x % P, nthr = HD / 8 / P;
  const bool act = tid < nthr;
  const int col = (js * nthr + (act ? tid : 0)) * 8;
  int* flg = p.ctr + 64 + b * 32;
  float* ssp = (float*)(p.ctr + 64 + 1024) + b * 8 * 32;
  float gsk[8], twk[8];
#pragma unroll
  for (int k = 0; k < 8; k++) { const unsigned e = lid[b * 8 + k]; gsk[k] = e < NLOC ? p.g2[p.lo + e] : 0.f; twk[k] = p.tw[b * 8 + k]; }
  const float4 n0 = *(const float4*)(p.nw + col), n1 = *(const float4*)(p.nw + col + 4);
  if (tid == 0) { while (ldacq(p.ctr) < sm->target) __nanosleep(32); }
  __syncthreads();
  PST(4);
  const int epoch = sm->target / (int)gridDim.x;
  {
    float4 u[8][2];
#pragma unroll
    for (int k = 0; k < 8; k++) {
      if (act && gsk[k] != 0.f) { const float* yr = p.yacc + (size_t)(b * 8 + k) * HD + col; u[k][0] = __ldcg((const float4*)yr); u[k][1] = __ldcg((const float4*)(yr + 4)); }
      else u[k][0] = u[k][1] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
#pragma unroll
    for (int k = 0; k < 8; k++) {
      const float a0 = u[k][0].x * n0.x, a1 = u[k][0].y * n0.y, a2 = u[k][0].z * n0.z, a3 = u[k][0].w * n0.w;
      const float a4 = u[k][1].x * n1.x, a5 = u[k][1].y * n1.y, a6 = u[k][1].z * n1.z, a7 = u[k][1].w * n1.w;
      float ss = a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3 + a4 * a4 + a5 * a5 + a6 * a6 + a7 * a7;
#pragma unroll
      for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
      if (lane == 0) sm->ssr[k][wid] = ss;
    }
  }
  __syncthreads();
  const int nw_ = (nthr + 31) / 32;
  if (tid == 0) {  // one thread writes all partials, so its release store alone orders them
#pragma unroll
    for (int k = 0; k < 8; k++) { float s = 0.f; for (int w = 0; w < nw_; w++) s += sm->ssr[k][w]; ssp[k * 32 + js] = s; }
    asm volatile("st.release.gpu.global.b32 [%0], %1;\n" :: "l"(flg + js), "r"(epoch) : "memory");
  }
  if (wid == 0) {
    if (lane < P) { while (ldacq(flg + lane) != epoch) __nanosleep(20); }
    __syncwarp();
    if (lane < 8) {
      float s = 0.f;
      for (int j = 0; j < P; j++) s += __ldcg(ssp + lane * 32 + j);
      sm->ssr[lane][0] = s;
    }
  }
  __syncthreads();
  PST(6);
  float coef[8];
#pragma unroll
  for (int k = 0; k < 8; k++) { const float gs = gsk[k]; coef[k] = gs != 0.f ? twk[k] * gs * rsqrtf(sm->ssr[k][0] * gs * gs * (1.f / HD) + 1e-8f) : 0.f; }
  PST(7);
  if (act) {
    float4 u[8][2];
#pragma unroll
    for (int k = 0; k < 8; k++) {
      if (gsk[k] != 0.f) { const float* yr = p.yacc + (size_t)(b * 8 + k) * HD + col; u[k][0] = __ldcg((const float4*)yr); u[k][1] = __ldcg((const float4*)(yr + 4)); }
      else u[k][0] = u[k][1] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    float o[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int k = 0; k < 8; k++) {
      const float cf = coef[k];
      o[0] += cf * u[k][0].x; o[1] += cf * u[k][0].y; o[2] += cf * u[k][0].z; o[3] += cf * u[k][0].w;
      o[4] += cf * u[k][1].x; o[5] += cf * u[k][1].y; o[6] += cf * u[k][1].z; o[7] += cf * u[k][1].w;
    }
    uint4 ov; __nv_bfloat162* ob = (__nv_bfloat162*)&ov;
    ob[0] = __floats2bfloat162_rn(o[0] * n0.x, o[1] * n0.y); ob[1] = __floats2bfloat162_rn(o[2] * n0.z, o[3] * n0.w);
    ob[2] = __floats2bfloat162_rn(o[4] * n1.x, o[5] * n1.y); ob[3] = __floats2bfloat162_rn(o[6] * n1.z, o[7] * n1.w);
    *(uint4*)(p.out + (size_t)b * HD + col) = ov;
#pragma unroll
    for (int k = 0; k < 8; k++)
      if (gsk[k] != 0.f) { float* yr = p.yacc + (size_t)(b * 8 + k) * HD + col; __stcg((float4*)yr, make_float4(0.f, 0.f, 0.f, 0.f)); __stcg((float4*)(yr + 4), make_float4(0.f, 0.f, 0.f, 0.f)); }
  }
  PST(5);
}

static int g_gridF = 0, g_smemF = 0;
void moe(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor w1, torch::Tensor w2, torch::Tensor g1,
         torch::Tensor g2, torch::Tensor nw, torch::Tensor yacc, torch::Tensor ctr, torch::Tensor out, int64_t lo, torch::Tensor prof) {
  if (!g_gridF) {
    int dev; cudaGetDevice(&dev); int mx; cudaDeviceGetAttribute(&mx, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaFuncSetAttribute(moe_f, cudaFuncAttributeMaxDynamicSharedMemorySize, mx);
#ifdef SMREQ
    if (SMREQ * 1024 < mx) {  // leave the rest of the unified 256 KB to L1 (activation rows)
      mx = SMREQ * 1024;
      cudaFuncSetAttribute(moe_f, cudaFuncAttributePreferredSharedMemoryCarveout, (mx * 100 + 228 * 1024 - 1) / (228 * 1024));
    }
#endif
    g_smemF = mx; g_gridF = sms;
  }
  P p{w1.data_ptr<uint8_t>(), w2.data_ptr<uint8_t>(), g1.data_ptr<float>(), g2.data_ptr<float>(),
      (const __nv_bfloat16*)x.data_ptr(), ids.data_ptr<int>(), tw.data_ptr<float>(), nw.data_ptr<float>(),
      yacc.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(), ctr.data_ptr<int>(), (int)x.size(0), (int)lo, g_smemF,
      prof.numel() ? (unsigned long long*)prof.data_ptr() : nullptr};
  moe_f<<<g_gridF, WPC * 32, g_smemF, at::cuda::getCurrentCUDAStream()>>>(p);
}
"""


_modF = {}
FS_MAX = 32  # fused split-K path for M <= FS_MAX (tokens per expert <= 32: one pass, NT <= 4)
FS_CFG = ()  # retired in v47 (g1 / gk win every M <= 32 bucket): was (0, 3, "-DMT2 -DLASTA -DSMREQ=196"), (0, 3, "-DMT2 -DSMREQ=196"), (1, 3, "")  # (forced W13 K split or 0 = auto, ring depth cap, extra defines)
DIAGF = False  # prepare(): print the fused path per-CTA phase timeline (PROF build) for a few M


def _extF(ksf=0, smax=3, extra=""):
    k = (ksf, smax, extra)
    if k not in _modF:
        wpc = 16 if _big() else 8
        fl = ([] if "-DWPC=" in extra else ["-DWPC=%d" % wpc]) + ["-DSMAX=%d" % smax] + (["-DKSF=%d" % ksf] if ksf else []) + extra.replace("+", " ").split() + [d for d in _DBG if not d.startswith("-DCFG=")]
        _modF[k] = load_inline("moe_w4a16_v48F" + "".join(fl).replace("-", "_").replace("=", ""),
                               cpp_sources="void moe(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,torch::Tensor);",
                               cuda_sources=CUDA_F, functions=["moe"], extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"] + fl, verbose=False)
    return _modF[k]


def _repack2(w, sf):
    """W2 for the fused path: uint8 [64 rowblocks, 128 granules (32 K), 4 rowtiles, 288]: per 16-row tile and 32-K granule,
    32 lanes x 2 words (mma A fragments of two k16 steps; lane t holds K 8t..8t+7) then 32 B of block scales [g][blk][row g / g+8].
    Rows permuted inside each tile (mma row i < 8 -> 2i, i >= 8 -> 2(i-8)+1) so a lane's C fragment holds adjacent output rows."""
    R, K = w.shape[0], w.shape[1] * 2
    RT, NG = R // 16, K // 32
    dev = w.device
    pi = torch.tensor([2 * i for i in range(8)] + [2 * i + 1 for i in range(8)], device=dev)
    perm = (torch.arange(RT, device=dev)[:, None] * 16 + pi[None, :]).reshape(-1)
    codes = torch.stack((w & 15, w >> 4), -1).view(R, K)[perm].to(torch.int64)
    s8 = sf.view(torch.uint8)[perm].to(torch.int64)
    neg = (s8 >> 7).repeat_interleave(16, 1)
    codes = codes ^ (neg << 3)
    zero = ((s8 & 0x7F) == 0).repeat_interleave(16, 1)
    codes = torch.where(zero, torch.zeros_like(codes), codes)
    s8 = torch.where((s8 & 0x7F) == 0, torch.full_like(s8, 0x08), s8 & 0x7F)
    if ((s8 >> 3) == 0).any():
        raise ValueError("subnormal E4M3 block scales are not supported by this fast path")
    C = codes.view(RT, 16, NG, 4, 2, 4)
    a, b = C[:, :8], C[:, 8:]
    word = torch.zeros(a.shape[:-1], dtype=torch.int64, device=dev)
    for pair, (src, e0) in enumerate(((a, 0), (b, 0), (a, 2), (b, 2))):
        word = _place(word, src[..., e0], pair, 0)
        word = _place(word, src[..., e0 + 1], pair, 1)
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
    word = word.permute(0, 2, 1, 3, 4).contiguous().view(torch.uint8).view(RT, NG, 256)
    S = s8.to(torch.uint8).view(RT, 2, 8, NG, 2).permute(0, 3, 2, 4, 1).contiguous().view(RT, NG, 32)
    rec = torch.cat((word, S), -1).view(RT // 4, 4, NG, 288).permute(0, 2, 1, 3).contiguous()
    return rec


def _fs(data, state, cfg):
    x = data["hidden_states"].contiguous()
    ids = data["topk_ids"].to(torch.int32).contiguous()
    tw = data["topk_weights"].float().contiguous()
    out = torch.empty_like(x)
    _extF(*cfg[1:]).moe(x, ids, tw, state["w1"], state["w2f"], state["g1"], state["g2"], state["nw"], state["yacc"],
                              state["ctrF"], out, data["config"]["local_start"], state["profF"])
    return out


CUDA_G = '#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cuda_bf16.h>\n// Match baseline silu_and_mul: BF16 gate/up inputs, FP32 activation/product,\n// BF16 product store at each caller. Do not add a nonexistent SiLU tensor boundary.\n__device__ __forceinline__ float muse_swiglu_bf16(float gate, float up) {\n  gate = __bfloat162float(__float2bfloat16_rn(gate));\n  up = __bfloat162float(__float2bfloat16_rn(up));\n  return gate / (1.f + __expf(-gate)) * up;\n}\n\n#include <stdint.h>\n\n// v45 "g1": decode GEMV for M <= 2 (<= 2 tokens per expert). Every warp owns one 256-K slice of every weight row the\n// CTA touches, so the activation slice lives in registers and nothing is split per round across warps except a final\n// 16-way smem reduction. W13: the CTA\'s contiguous range of 16-row tiles over all local experts -> SwiGLU -> inter\n// (global). Grid barrier (the ring keeps streaming W2 meanwhile). W2: the CTA\'s contiguous range of 16-row tiles of the\n// output, for every local expert, full K (W2 in the same 16-row tile / 256-K half-stage layout as W13) -> y rows in smem, per-route sum of squares -> grid barrier -> norm + combine\n// of the CTA\'s rows. No atomics on y, no accumulator to clean.\n#define HD 4096\n#define NLOC 32\n#define STG 4608\n#define PC 2304\n#define NW 16\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900\n#define SMALLSM 1  // sm_80/86 emulation: 99 KB of smem per block\n#endif\n#ifdef P1\n#define PPS 1  // one 2304 B piece (one TMA box, one barrier) per ring slot\n#else\n#define PPS 2  // two pieces (two tiles\' slices) per ring slot\n#endif\n#define SLOT (PPS * PC)\n#ifndef SG\n#ifdef SMALLSM\n#define SG (2 / PPS)\n#else\n#define SG (4 / PPS)   // ring slots per warp (16 x 4 x 2304 B = 147 KB in flight per SM)\n#endif\n#endif\n#ifdef SMALLSM\n#undef RT\n#define RT 2   // 99 KB of smem: small rounds whatever the candidate asks for\n#endif\n#ifndef RT\n#define RT 8   // pieces per reduction round\n#endif\n#define MAXR 16\n#define MAXRT 4\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900\n#define TMA 1\n#else\n#define TMA 0\n#endif\n\nstruct P {\n  const uint8_t* w1; const uint8_t* w2; const float* g1; const float* g2; const __nv_bfloat16* x; const int* ids;\n  const float* tw; const float* nw; __nv_bfloat16* inter; float* ssp; __nv_bfloat16* out; int* ctr; int M, lo, smem;\n  unsigned long long* prof;\n};\nstruct Sm {\n  int act[NLOC], aoff[NLOC + 1];\n  int na, nr, t0, n13, r0, nrt, n2, ns13, ns2, target, stage;\n  int rexp[MAXR], rid[MAXR]; float coef[MAXR], cf[MAXR], g2r[MAXR];\n};\n\n__device__ __forceinline__ uint4 lds(uint32_t a) { uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v; }\n__device__ __forceinline__ uint2 lds64(uint32_t a) { uint2 v; asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\\n" : "=r"(v.x), "=r"(v.y) : "r"(a)); return v; }\n__device__ __forceinline__ uint32_t lds16(uint32_t a) { unsigned short v; asm volatile("ld.shared.u16 %0, [%1];\\n" : "=h"(v) : "r"(a)); return v; }\n__device__ __forceinline__ void cp16(uint32_t s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\\n" :: "r"(s), "l"(g) : "memory"); }\n__device__ __forceinline__ uint32_t bmul(uint32_t a, uint32_t b) { uint32_t d; asm("fma.rn.bf16x2 %0, %1, %2, %3;\\n" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u)); return d; }\n__device__ __forceinline__ uint32_t sc2(uint32_t w, uint32_t sel) { return __byte_perm(w, 0, sel) * 16u + 0x77007700u; }\n__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {\n#ifdef NOMMA\n  c[0] += __uint_as_float(a0 ^ a1 ^ a2 ^ a3 ^ b0 ^ b1);\n#else\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\\n"\n    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n#endif\n}\n__device__ __forceinline__ int ldacq(const int* p) { int v; asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\\n" : "=r"(v) : "l"(p) : "memory"); return v; }\n__device__ __forceinline__ void mb_init(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\\n" :: "r"(b), "r"(n)); }\n__device__ __forceinline__ void mb_wait(uint32_t bar, uint32_t ph) {\n#if TMA\n  asm volatile("{\\n.reg .pred p;\\nW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\\n@!p bra W;\\n}\\n" :: "r"(bar), "r"(ph) : "memory");\n#else\n  asm volatile("{\\n.reg .pred p;\\nW: mbarrier.test_wait.parity.shared::cta.b64 p, [%0], %1;\\n@!p bra W;\\n}\\n" :: "r"(bar), "r"(ph) : "memory");\n#endif\n}\n#ifdef PROF\n__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }\n#define PST(k) do { if (threadIdx.x == 0 && p.prof) p.prof[blockIdx.x * 16 + (k)] = gtime(); } while (0)\n#else\n#define PST(k)\n#endif\n#ifdef DQHI\n#define SHR5(q) __umulhi((q), 1u << 27)\n#else\n#define SHR5(q) ((q) >> 5)\n#endif\n// dequant one word of 8 e2m1 codes into an m16n8k16 A fragment (scale pair sA rows g, sB rows g + 8)\n#define DQ(q, sA, sB, a0, a1, a2, a3) \\\n  const uint32_t a0 = bmul((q) & 0x81C081C0u, sA); \\\n  const uint32_t a1 = bmul((((q) << 3) & 0x01C001C0u) | (((q) << 1) & 0x80008000u), sB); \\\n  const uint32_t a2 = bmul(((q) << 6) & 0x81C081C0u, sA); \\\n  const uint32_t a3 = bmul(((SHR5(q)) & 0x01C001C0u) | (((q) << 5) & 0x80008000u), sB);\n\n// one 2304 B piece: 16 rows x this warp\'s 256-K slice (4 x 64-K chunks; lane t holds 16 contiguous K per chunk = one\n// NVFP4 block, so one scale byte pair per chunk). B: the activation slice in mma B-fragment order.\n__device__ __forceinline__ void piece(uint32_t pb, int lane, const uint4 (&B)[4][2], float (&acc)[2][4]) {\n  const uint2 sh = lds64(pb + 2048 + lane * 8);\n#pragma unroll\n  for (int c = 0; c < 4; c++) {\n    const uint4 w = lds(pb + c * 512 + lane * 16);\n    const uint32_t sw = c < 2 ? sh.x : sh.y;\n    const uint32_t sA = sc2(sw, (c & 1) ? 0x4242 : 0x4040), sB = sc2(sw, (c & 1) ? 0x4343 : 0x4141);\n    const uint32_t ww[4] = {w.x, w.y, w.z, w.w};\n#pragma unroll\n    for (int j = 0; j < 4; j++) {\n      DQ(ww[j], sA, sB, a0, a1, a2, a3)\n      const uint32_t* bb = (const uint32_t*)&B[c][j >> 1];\n      mma(acc[j & 1], a0, a1, a2, a3, bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);\n    }\n  }\n}\n\nstruct Ring {\n  const P* p; Sm* sm; uint32_t ring, bars; int lane, wid; unsigned long long wns;\n  // piece sequence of this warp: W13 slots (2 tiles\' 256-K slices each), then W2 slots (2 (expert, row tile) slices)\n  __device__ __forceinline__ void copy(uint32_t dst, uint32_t bar, const uint8_t* src) {\n#if TMA\n    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\\n" :: "r"(dst), "l"(src), "n"(PC), "r"(bar) : "memory");\n#else\n    for (int k = lane; k < PC / 16; k += 32) cp16(dst + k * 16, src + k * 16);\n#endif\n  }\n  __device__ __forceinline__ void issue(int q) {\n    const int ns13 = sm->ns13;\n    if (q >= ns13 + sm->ns2) return;\n    const uint32_t slot = q % SG, dst = ring + slot * SLOT, bar = bars + slot * 8;\n    const uint8_t* s0; const uint8_t* s1 = nullptr;\n    if (q < ns13) {\n      const int tt = sm->t0 + PPS * q, k = wid * PC;\n      s0 = p->w1 + ((size_t)(p->lo + sm->act[tt >> 9]) * 512 + (tt & 511)) * 8 * STG + k;\n      if (PPS == 2 && 2 * q + 1 < sm->n13) s1 = p->w1 + ((size_t)(p->lo + sm->act[(tt + 1) >> 9]) * 512 + ((tt + 1) & 511)) * 8 * STG + k;\n    } else {\n      const int j = PPS * (q - ns13), nrt = sm->nrt;\n      int a = j / nrt, rt = sm->r0 + j - a * nrt;\n      s0 = p->w2 + (((size_t)(p->lo + sm->act[a]) * 256 + rt) * 16 + wid) * PC;\n      if (PPS == 2 && j + 1 < sm->n2) { if (++rt == sm->r0 + nrt) { rt = sm->r0; a++; } s1 = p->w2 + (((size_t)(p->lo + sm->act[a]) * 256 + rt) * 16 + wid) * PC; }\n    }\n#ifdef NOLOAD  // diag: consumers run on stale ring contents (compute-only rate)\n#if TMA\n    if (lane == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\\n" :: "r"(bar) : "memory");\n#else\n    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\\n" :: "r"(bar) : "memory");\n#endif\n    return;\n#endif\n#if TMA\n    if (lane == 0) {\n      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" :: "r"(bar), "r"(s1 ? 2 * PC : PC) : "memory");\n      copy(dst, bar, s0);\n      if (s1) copy(dst + PC, bar, s1);\n    }\n#else\n    copy(dst, bar, s0);\n    if (s1) copy(dst + PC, bar, s1);\n    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\\n" :: "r"(bar) : "memory");\n#endif\n  }\n  __device__ __forceinline__ uint32_t acquire(int q) {\n    const int slot = q % SG;\n#ifdef PROF\n    const unsigned long long w0 = gtime();\n#endif\n    mb_wait(bars + slot * 8, (q / SG) & 1);\n    __syncwarp();\n#ifdef PROF\n    wns += gtime() - w0;\n#endif\n    return ring + slot * SLOT;\n  }\n  __device__ __forceinline__ void release(int q) { __syncwarp(); issue(q + SG); }\n};\n\n__global__ void __launch_bounds__(NW * 32, 1) moe_g(P p) {\n  extern __shared__ __align__(128) uint8_t smem[];\n  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, g = lane >> 2, t = lane & 3, R = p.M * 8, G = gridDim.x;\n  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem);\n  Sm* sm = (Sm*)(smem + NW * SG * 8);\n  const int xo = (NW * SG * 8 + (int)sizeof(Sm) + 127) & ~127;\n  __nv_bfloat16* xs = (__nv_bfloat16*)(smem + xo);\n  float4* red = (float4*)(smem + xo + p.M * HD * 2);\n  float* yb = (float*)(smem + xo + p.M * HD * 2 + RT * NW * 8 * 16);\n  const int ro = (xo + p.M * HD * 2 + RT * NW * 8 * 16 + MAXR * MAXRT * 16 * 4 + 127) & ~127;\n  const int io = ro + NW * SG * SLOT;\n  PST(0);\n  if (lane < SG) mb_init(sb + (wid * SG + lane) * 8, TMA ? 1 : 32);\n#if TMA\n  asm volatile("fence.mbarrier_init.release.cluster;\\n" ::: "memory");\n#endif\n  if (wid == 0) {  // routing in one warp (R <= 16 routes): experts ascending, routes of an expert in route order\n    int e = -1;\n    if (lane < R) { const unsigned u = p.ids[lane] - p.lo; e = u < NLOC ? (int)u : -1; }\n    const unsigned am = __reduce_or_sync(0xffffffffu, e >= 0 ? 1u << e : 0u);\n    const int na = __popc(am), nr = __popc(__ballot_sync(0xffffffffu, e >= 0));\n    int k = 0;\n    for (int j = 0; j < R; j++) { const int ej = __shfl_sync(0xffffffffu, e, j); k += ej >= 0 && ej < e; }\n    const unsigned peers = __match_any_sync(0xffffffffu, e);\n    const int rank = __popc(peers & ((1u << lane) - 1));\n    if (e >= 0) {\n      const int a = __popc(am & ((1u << e) - 1)), r = k + rank;\n      sm->rid[r] = lane; sm->rexp[r] = a;\n      if (rank == 0) { sm->act[a] = e; sm->aoff[a] = k; }\n    }\n    if (lane == 0) {\n    sm->aoff[na] = nr; sm->na = na; sm->nr = nr;\n    const int T13 = na * 512, b = blockIdx.x;\n    sm->t0 = (int)((long long)b * T13 / G); sm->n13 = (int)((long long)(b + 1) * T13 / G) - sm->t0;\n    sm->r0 = b * 256 / G; sm->nrt = (b + 1) * 256 / G - sm->r0;\n    sm->n2 = na * sm->nrt; sm->ns13 = (sm->n13 + PPS - 1) / PPS; sm->ns2 = (sm->n2 + PPS - 1) / PPS;\n    sm->stage = io + nr * HD * 2 <= p.smem;\n    if (sm->nrt > MAXRT || io > p.smem || G > 256) asm volatile("trap;");\n    }\n  }\n  __syncthreads();\n  Ring C; C.p = &p; C.sm = sm; C.lane = lane; C.wid = wid; C.wns = 0;\n  C.ring = sb + ro + wid * SG * SLOT; C.bars = sb + wid * SG * 8;\n#pragma unroll\n  for (int i = 0; i < SG; i++) C.issue(i);\n  PST(1);\n  const int na = sm->na, nr = sm->nr, n13 = sm->n13, n2 = sm->n2, ns13 = sm->ns13;\n  for (int i = tid; i < p.M * HD / 8; i += NW * 32) ((uint4*)xs)[i] = ((const uint4*)p.x)[i];\n  if (tid < nr) { const float g2 = p.g2[p.lo + sm->act[sm->rexp[tid]]]; sm->g2r[tid] = g2; sm->cf[tid] = p.tw[sm->rid[tid]]; }  // norm factors, off the tail\n  __syncthreads();\n  int q = 0;\n  unsigned long long wsy = 0;\n#ifdef PROF\n#define SYNCW() do { const unsigned long long t_ = gtime(); __syncthreads(); wsy += gtime() - t_; } while (0)\n  if (tid == 0 && p.prof) p.prof[blockIdx.x * 16 + 11] = clock64();\n#else\n#define SYNCW() __syncthreads()\n#endif\n  if (na) {\n    // ---- W13: warp slice K [256 wid, +256); B (activation) fragments in registers, reloaded per expert\n    uint4 B[4][2];\n    int curA = -1;\n    for (int rd = 0; rd * RT < n13; rd++) {\n      const int pend = min(n13, (rd + 1) * RT);\n      for (; PPS * q < pend; q++) {\n        const uint32_t sbase = C.acquire(q);\n#pragma unroll\n        for (int pp = 0; pp < PPS; pp++) {\n          const int pc = PPS * q + pp;\n          if (pc < n13) {\n            const int a = (sm->t0 + pc) >> 9;\n            if (a != curA) {\n              curA = a;\n              const int T = sm->aoff[a + 1] - sm->aoff[a];\n              const int row = g < T ? (sm->rid[sm->aoff[a] + g] >> 3) : -1;\n#pragma unroll\n              for (int c = 0; c < 4; c++)\n#pragma unroll\n                for (int h = 0; h < 2; h++)\n                  B[c][h] = row >= 0 ? ((const uint4*)(xs + row * HD + wid * 256 + c * 64 + t * 16))[h] : make_uint4(0, 0, 0, 0);\n            }\n            float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};\n            piece(sbase + pp * PC, lane, B, acc);\n            if (t == 0) red[((pc - rd * RT) * NW + wid) * 8 + g] = make_float4(acc[0][0] + acc[1][0], acc[0][1] + acc[1][1], acc[0][2] + acc[1][2], acc[0][3] + acc[1][3]);\n          }\n        }\n        C.release(q);\n      }\n      SYNCW();\n      if (tid < (pend - rd * RT) * 8) {\n        const int rp = tid >> 3, gg = tid & 7;\n        float4 s = make_float4(0.f, 0.f, 0.f, 0.f);\n#pragma unroll\n        for (int w = 0; w < NW; w++) { const float4 v = red[(rp * NW + w) * 8 + gg]; s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w; }\n        const int tt = sm->t0 + rd * RT + rp, a = tt >> 9, col = (tt & 511) * 8 + gg, T = sm->aoff[a + 1] - sm->aoff[a];\n        const float gs = p.g1[p.lo + sm->act[a]];\n#pragma unroll\n        for (int n = 0; n < 2; n++)\n          if (n < T) {\n            const float gt = (n ? s.y : s.x) * gs, up = (n ? s.w : s.z) * gs;\n            p.inter[(size_t)(sm->aoff[a] + n) * HD + col] = __float2bfloat16(muse_swiglu_bf16(gt, up));\n          }\n#ifndef FENCE1\n        __threadfence();\n#endif\n      }\n      SYNCW();\n    }\n    PST(2);\n#ifdef PROF\n    if (tid == 0 && p.prof) p.prof[blockIdx.x * 16 + 12] = clock64();\n    if (lane == 0 && p.prof) atomicAdd(p.prof + blockIdx.x * 16 + 10, wsy);\n    if (lane == 0 && p.prof) atomicAdd(p.prof + blockIdx.x * 16 + 8, C.wns);\n    C.wns = 0;\n#endif\n    // ---- grid barrier 1: every tile of inter is written (the ring is already streaming W2)\n    if (tid == 0) {\n      int old;\n#ifdef FENCE1\n      __threadfence();  // after the __syncthreads: orders the CTA\'s inter stores before the release\n#endif\n      asm volatile("atom.add.release.gpu.global.s32 %0, [%1], 1;\\n" : "=r"(old) : "l"(p.ctr) : "memory");\n      const int target = (old / G + 1) * G;\n      while (ldacq(p.ctr) < target) __nanosleep(20);\n    }\n    __syncthreads();\n    PST(3);\n    const __nv_bfloat16* ib = p.inter;\n    if (sm->stage) {\n      __nv_bfloat16* is = (__nv_bfloat16*)(smem + io);\n      for (int i = tid; i < nr * HD / 8; i += NW * 32) ((uint4*)is)[i] = __ldcg((const uint4*)p.inter + i);\n      __syncthreads();\n      ib = is;\n    }\n    // ---- W2: warp slice K [256 wid, +256) of (expert, 16-row tile) units\n    uint4 B2[4][2];\n    int curB = -1;\n    for (int rd = 0; rd * RT < n2; rd++) {\n      const int pend = min(n2, (rd + 1) * RT);\n      for (; PPS * (q - ns13) < pend; q++) {\n        const uint32_t sbase = C.acquire(q);\n#pragma unroll\n        for (int pp = 0; pp < PPS; pp++) {\n          const int j = PPS * (q - ns13) + pp;\n          if (j < n2) {\n            const int a = j / sm->nrt;\n            if (a != curB) {\n              curB = a;\n              const int T = sm->aoff[a + 1] - sm->aoff[a];\n              const uint4* src = (const uint4*)(ib + (size_t)(sm->aoff[a] + (g < T ? g : 0)) * HD + wid * 256 + t * 16);\n#pragma unroll\n              for (int c = 0; c < 4; c++)\n#pragma unroll\n                for (int h = 0; h < 2; h++) B2[c][h] = g < T ? (sm->stage ? src[c * 8 + h] : __ldcg(src + c * 8 + h)) : make_uint4(0, 0, 0, 0);\n            }\n            float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};\n            piece(sbase + pp * PC, lane, B2, acc);\n            if (t == 0) red[((j - rd * RT) * NW + wid) * 8 + g] = make_float4(acc[0][0] + acc[1][0], acc[0][1] + acc[1][1], acc[0][2] + acc[1][2], acc[0][3] + acc[1][3]);\n          }\n        }\n        C.release(q);\n      }\n      __syncthreads();\n      if (tid < (pend - rd * RT) * 8) {\n        const int rp = tid >> 3, gg = tid & 7, j = rd * RT + rp, a = j / sm->nrt, rtl = j - a * sm->nrt, T = sm->aoff[a + 1] - sm->aoff[a];\n        float4 s = make_float4(0.f, 0.f, 0.f, 0.f);\n#pragma unroll\n        for (int w = 0; w < NW; w++) { const float4 v = red[(rp * NW + w) * 8 + gg]; s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w; }\n        float* y0 = yb + (sm->aoff[a]) * (MAXRT * 16) + rtl * 16 + gg;\n        const float down_scale = sm->g2r[sm->aoff[a]];\n        y0[0] = __bfloat162float(__float2bfloat16(s.x * down_scale)); y0[8] = __bfloat162float(__float2bfloat16(s.z * down_scale));\n        if (T > 1) { y0[MAXRT * 16] = __bfloat162float(__float2bfloat16(s.y * down_scale)); y0[MAXRT * 16 + 8] = __bfloat162float(__float2bfloat16(s.w * down_scale)); }\n      }\n      __syncthreads();\n    }\n    PST(4);\n#ifdef PROF\n    if (lane == 0 && p.prof) atomicAdd(p.prof + blockIdx.x * 16 + 9, C.wns);\n#endif\n    // ---- per-route partial sum of squares over this CTA\'s rows\n    const int nrows = sm->nrt * 16, row0 = sm->r0 * 16;\n    if (wid < nr) {\n      float ss = 0.f;\n      for (int r = lane; r < nrows; r += 32) { const float v = yb[wid * MAXRT * 16 + r] * p.nw[row0 + r]; ss += v * v; }\n#pragma unroll\n      for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);\n      if (lane == 0) p.ssp[blockIdx.x * MAXR + wid] = ss;\n      __threadfence();\n    }\n    __syncthreads();\n    if (tid == 0) {\n      int old;\n      asm volatile("atom.add.release.gpu.global.s32 %0, [%1], 1;\\n" : "=r"(old) : "l"(p.ctr) : "memory");\n      const int target = (old / G + 1) * G;\n      while (ldacq(p.ctr) < target) __nanosleep(20);\n    }\n    __syncthreads();\n    PST(5);\n    if (wid < nr) {\n      float s = 0.f;\n      float v[8];\n#pragma unroll\n      for (int k = 0; k < 8; k++) { const int c = lane + 32 * k; v[k] = c < G ? __ldcg(p.ssp + c * MAXR + wid) : 0.f; }  // all loads in flight at once\n      s = ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7]));\n#pragma unroll\n      for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);\n      const float gs = sm->g2r[wid];\n      if (lane == 0) sm->coef[wid] = sm->cf[wid] * rsqrtf(s * (1.f / HD) + 1e-8f);\n    }\n    __syncthreads();\n  }\n  // ---- combine this CTA\'s output rows\n  {\n    const int nrows = sm->nrt * 16, row0 = sm->r0 * 16;\n    for (int i = tid; i < p.M * nrows; i += NW * 32) {\n      const int tk = i / nrows, r = i - tk * nrows;\n      float o = 0.f;\n      for (int k = 0; k < nr; k++) if ((sm->rid[k] >> 3) == tk) o += sm->coef[k] * yb[k * MAXRT * 16 + r];\n      p.out[(size_t)tk * HD + row0 + r] = __float2bfloat16(o * p.nw[row0 + r]);\n    }\n  }\n  PST(6);\n}\n\nstatic int g_gridG = 0, g_smemG = 0;\nvoid moe(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor w1, torch::Tensor w2, torch::Tensor g1, torch::Tensor g2,\n         torch::Tensor nw, torch::Tensor inter, torch::Tensor ssp, torch::Tensor ctr, torch::Tensor out, int64_t lo, torch::Tensor prof) {\n  if (!g_gridG) {\n    int dev; cudaGetDevice(&dev); int mx; cudaDeviceGetAttribute(&mx, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);\n    int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);\n    cudaFuncSetAttribute(moe_g, cudaFuncAttributeMaxDynamicSharedMemorySize, mx);\n    g_smemG = mx; g_gridG = sms;\n  }\n  P p{w1.data_ptr<uint8_t>(), w2.data_ptr<uint8_t>(), g1.data_ptr<float>(), g2.data_ptr<float>(), (const __nv_bfloat16*)x.data_ptr(),\n      ids.data_ptr<int>(), tw.data_ptr<float>(), nw.data_ptr<float>(), (__nv_bfloat16*)inter.data_ptr(), ssp.data_ptr<float>(),\n      (__nv_bfloat16*)out.data_ptr(), ctr.data_ptr<int>(), (int)x.size(0), (int)lo, g_smemG,\n      prof.numel() ? (unsigned long long*)prof.data_ptr() : nullptr};\n  moe_g<<<g_gridG, NW * 32, g_smemG, at::cuda::getCurrentCUDAStream()>>>(p);\n}\n'
_modG = {}
GS_MAX = 2       # g1 for M <= GS_MAX (<= 2 tokens per expert)
G_CFG = ("-DFENCE1 -DRT=16", "-DFENCE1")  # g1 candidates (FENCE1: one fence before the grid barrier; P1: one 2304 B box per slot; RT=16: one reduction round at M1)
DIAGG = False    # prepare(): g1 per-CTA timeline (PROF build) + bytes-in-flight estimate + per-load timing at M1


def _extG(extra=""):
    if extra not in _modG:
        fl = extra.replace("+", " ").split() + [d for d in _DBG if not d.startswith("-DCFG=")]
        _modG[extra] = load_inline("moe_w4a16_v48G_strict_down" + "".join(fl).replace("-", "_").replace("=", ""),
                                   cpp_sources="void moe(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,torch::Tensor);",
                                   cuda_sources=CUDA_G, functions=["moe"], extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"] + fl, verbose=False)
    return _modG[extra]


def _g1(data, state, cfg, prof=None):
    x = data["hidden_states"].contiguous()
    ids = data["topk_ids"].to(torch.int32).contiguous()
    tw = data["topk_weights"].float().contiguous()
    out = torch.empty_like(x)
    ex = cfg[3] if len(cfg) > 3 else ""
    _extG(ex).moe(x, ids, tw, state["w1"], state["w2"], state["g1"], state["g2"], state["nw"], state["interG"], state["sspG"],
                  state["ctrG"], out, data["config"]["local_start"], state["prof"][:0] if prof is None else prof)
    return out


def _diagG(state):
    """g1 load-path variants at M1 (and M2) on judge-generator routings: per-CTA timeline (us from the first CTA start), the
    consumers' wait fraction on full slots (waitA/B) and in the per-round __syncthreads (syncA), the SM clock over W13,
    achieved stream rates and the ring's bytes in flight against what the HBM latency needs"""
    dev = state["w1"].device
    flush = torch.empty(2 ** 27, device=dev); sink = torch.empty((), device=dev)
    G = torch.cuda.get_device_properties(dev).multi_processor_count
    prof = torch.zeros(G * 16, dtype=torch.int64, device=dev)
    runs = [(1, sd, x) for x in G_CFG for sd in range(3)] + [(2, 0, x) for x in G_CFG]
    for M, sd, ab in runs:
        data = _synth(M, 900 + 7 * sd + M, dev)
        ids = data["topk_ids"]; na = int(torch.unique(ids[ids < 32]).numel())
        ev = []
        for _ in range(3):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            prof.zero_(); sink.copy_(flush.sum()); e0.record(); _g1(data, state, ("g1", 0, 0, ("-DPROF " + ab).strip()), prof); e1.record()
            torch.cuda.synchronize(); ev.append(e0.elapsed_time(e1) * 1e3)
        P = prof.view(-1, 16).cpu().double(); t0 = P[:, 0].min()
        q = lambda k, f: float((P[:, k] - t0).quantile(f)) / 1e3
        d13 = (P[:, 2] - P[:, 1]).clamp(min=1); d2 = (P[:, 4] - P[:, 3]).clamp(min=1)
        wA = float((P[:, 8] / (16 * d13)).median()); wB = float((P[:, 9] / (16 * d2)).median()); sy = float((P[:, 10] / (16 * d13)).median())
        mhz = float(((P[:, 12] - P[:, 11]) / d13).median()) * 1e3
        b13 = na * 512 * 36864 / G; b2 = na * 256 * 36864 / G
        r13 = b13 / float(d13.median()); r2 = b2 / float(d2.median())  # bytes per ns per SM = GB/s per SM
        cap = 16 * 4 * 2304 / 1024
        print(f"GD M{M} {_nm(('g1', 0, 0, ab))} na {na} evt {ev[-1]:.1f}: pro {q(1, .5):.1f}/{q(1, 1):.1f} W13 {q(2, .5):.1f}/{q(2, 1):.1f} bar1 {q(3, 1):.1f} "
              f"W2 {q(4, .5):.1f}/{q(4, 1):.1f} bar2 {q(5, 1):.1f} end {q(6, 1):.1f} | waitA {wA:.2f} syncA {sy:.2f} waitB {wB:.2f} | SM {mhz:.0f} MHz | "
              f"W13 {r13 * G / 1e3:.2f} TB/s W2 {r2 * G / 1e3:.2f} TB/s | ring {cap:.0f} KB/SM, Little latency at W13 rate {cap * 1024 / r13 / 1e3:.1f} us", flush=True)


CUDA_K = '#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cuda_bf16.h>\n// Match baseline silu_and_mul: BF16 gate/up inputs, FP32 activation/product,\n// BF16 product store at each caller. Do not add a nonexistent SiLU tensor boundary.\n__device__ __forceinline__ float muse_swiglu_bf16(float gate, float up) {\n  gate = __bfloat162float(__float2bfloat16_rn(gate));\n  up = __bfloat162float(__float2bfloat16_rn(up));\n  return gate / (1.f + __expf(-gate)) * up;\n}\n\n#include <stdint.h>\n\n// v47 "gk": g1\'s structure for M <= MAXM. Activation rows of the CTA\'s W13 K range sit in smem (rows padded by 16 B), so\n// B fragments come from LDS for any token count (n-tiles of 8 tokens, one dequant feeds every n-tile). 16 warps = TG tile\n// groups of SW warps; a group\'s warps split the K range of each of its tiles (2304 B pieces through a per-warp TMA ring)\n// and reduce SW-way behind a group-local named barrier. W13: KS1 CTA groups each own one K range (KS1 = 1: SwiGLU in the\n// reduction -> inter; KS1 > 1: fp32 gate/up partials -> grid barrier -> every CTA finishes a slice of inter). Grid\n// barrier. W2: KS2 CTA groups each own one K range of inter (staged in smem, chunked by whole experts when it does not\n// fit) and a contiguous range of 16-row output tiles of every expert -> fp32 y partials. Grid barrier. Each CTA owns ~32\n// output rows: sums the KS2 partials, per-route sum of squares -> grid barrier -> norm + combine of its rows. No atomics.\n#define HD 4096\n#define NLOC 32\n#define STG 4608\n#define PC 2304\n#define NW 16\n#ifndef SW\n#define SW 4                 // warps per tile\n#endif\n#define TG (NW / SW)         // tile groups\n#ifndef KS1\n#define KS1 1                // W13 K split over CTA groups\n#endif\n#ifndef KS2\n#define KS2 2                // W2 K split over CTA groups\n#endif\n#define PPT (16 / KS1 / SW)  // 2304 B pieces per W13 tile slice\n#define W2S (16 / KS2 / SW)  // pieces per W2 unit slice\n// KS2 = 1: a CTA\'s W2 row tiles are its output rows (y stays in smem, no y partials, one grid barrier less)\n#ifndef PPS\n#define PPS 1                // contiguous 2304 B pieces per ring slot (one TMA box, one barrier)\n#endif\n#ifndef SG\n#define SG 2                 // ring slots per warp\n#endif\n#define SLOTB (PPS * PC)\n#define ZB (2048 + 64)       // zero row for absent token slots (B of a missing token is 0: no tensor-core toggling)\nstatic_assert(PPT % PPS == 0 && W2S % PPS == 0, "slice pieces must fill whole slots");\n#ifndef MAXM\n#define MAXM 16\n#endif\n#define MAXR (MAXM * 8)\n#define MAXNT ((MAXM + 7) / 8)\n#define RW ((MAXR + 31) / 32)  // routing warps\n#define XS (HD / KS1 + 8)    // padded smem row (bf16 elements) of x\n#define IS (HD / KS2 + 8)    // padded smem row of staged inter\n#if KS2 == 1\n#define YBR (MAXR * 64 * 4)  // y rows kept at the end of the union region (live while inter is staged)\n#else\n#define YBR 0\n#endif\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900\n#define TMA 1\n#else\n#define TMA 0\n#if defined(__CUDA_ARCH__)\n#define EMUX 1  // sm_80/86 emulation (99 KB smem): the x / inter / y region lives in a global scratch, B via generic loads\n#endif\n#endif\n#define SCR (160 * 1024)  // per-CTA scratch bytes for EMUX\n#ifdef EMUX\ntypedef const uint8_t* BA;\n#define LDB(a) (*(const uint4*)(a))\n#else\ntypedef uint32_t BA;\n#define LDB(a) lds(a)\n#endif\n\nstruct P {\n  const uint8_t* w1; const uint8_t* w2; const float* g1; const float* g2; const __nv_bfloat16* x; const int* ids;\n  const float* tw; const float* nw; __nv_bfloat16* inter; float* gp; float* yp; float* ssp; __nv_bfloat16* out; int* ctr; uint8_t* scr;\n  int M, lo, smem; unsigned long long* prof;\n};\nstruct Sm {\n  int act[NLOC], aoff[NLOC + 1], hist[RW][NLOC];\n  int na, nr, k1, t0, n13, ksp, r0, nrt, o0, onr, nch, cha[NLOC + 1];\n  float gs1[NLOC], gs2[NLOC], cf[MAXR];          // W13 global scale per expert; tw x W2 global scale per route\n  short tr[MAXM][8];                             // route slot of (token, top-k position), -1 if not local\n  float nws[64];                                 // post-norm weight of this CTA\'s output rows\n  float coef[MAXR]; short rid[MAXR]; uint8_t rexp[MAXR];  // 16 / 8-bit: the MAXM 32 layout needs every byte\n};\n\n__device__ __forceinline__ uint4 lds(uint32_t a) { uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v; }\n__device__ __forceinline__ uint2 lds64(uint32_t a) { uint2 v; asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\\n" : "=r"(v.x), "=r"(v.y) : "r"(a)); return v; }\n__device__ __forceinline__ void cp16(uint32_t s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\\n" :: "r"(s), "l"(g) : "memory"); }\n__device__ __forceinline__ uint32_t bmul(uint32_t a, uint32_t b) { uint32_t d; asm("fma.rn.bf16x2 %0, %1, %2, %3;\\n" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u)); return d; }\n__device__ __forceinline__ uint32_t sc2(uint32_t w, uint32_t sel) { return __byte_perm(w, 0, sel) * 16u + 0x77007700u; }\n__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {\n#ifdef NOMMA\n  c[0] += __uint_as_float(a0 ^ a1 ^ a2 ^ a3 ^ b0 ^ b1);\n#else\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\\n"\n    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n#endif\n}\n__device__ __forceinline__ int ldacq(const int* p) { int v; asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\\n" : "=r"(v) : "l"(p) : "memory"); return v; }\n__device__ __forceinline__ void mb_init(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\\n" :: "r"(b), "r"(n)); }\n__device__ __forceinline__ void mb_wait(uint32_t bar, uint32_t ph) {\n#if TMA\n  asm volatile("{\\n.reg .pred p;\\nW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\\n@!p bra W;\\n}\\n" :: "r"(bar), "r"(ph) : "memory");\n#else\n  asm volatile("{\\n.reg .pred p;\\nW: mbarrier.test_wait.parity.shared::cta.b64 p, [%0], %1;\\n@!p bra W;\\n}\\n" :: "r"(bar), "r"(ph) : "memory");\n#endif\n}\n__device__ __forceinline__ void gbar(int id, int n) { asm volatile("bar.sync %0, %1;\\n" :: "r"(id), "r"(n) : "memory"); }\n#ifdef PROF\n__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }\n#define PST(k) do { if (threadIdx.x == 0 && p.prof) p.prof[blockIdx.x * 16 + (k)] = gtime(); } while (0)\n#else\n#define PST(k)\n#endif\n#define DQ(q, sA, sB, a0, a1, a2, a3) \\\n  const uint32_t a0 = bmul((q) & 0x81C081C0u, sA); \\\n  const uint32_t a1 = bmul((((q) << 3) & 0x01C001C0u) | (((q) << 1) & 0x80008000u), sB); \\\n  const uint32_t a2 = bmul(((q) << 6) & 0x81C081C0u, sA); \\\n  const uint32_t a3 = bmul((((q) >> 5) & 0x01C001C0u) | (((q) << 5) & 0x80008000u), sB);\n\n#ifndef CW2\n#define CW2 11\n#endif\n#ifndef CW3\n#define CW3 15\n#endif\n#ifndef CW4\n#define CW4 19\n#endif\n__device__ __forceinline__ int cwt(int nt) { return nt <= 1 ? 8 : (nt == 2 ? CW2 : (nt == 3 ? CW3 : CW4)); }\n\n// grid barrier on a monotone counter (every CTA adds 1; the target is the next multiple of the grid)\n__device__ __forceinline__ void grid_sync(int* ctr, int G) {\n  __syncthreads();\n  if (threadIdx.x == 0) {\n    int old;\n    __threadfence();\n    asm volatile("atom.add.release.gpu.global.s32 %0, [%1], 1;\\n" : "=r"(old) : "l"(ctr) : "memory");\n    const int target = (old / G + 1) * G;\n    while (ldacq(ctr) < target) __nanosleep(20);\n  }\n  __syncthreads();\n}\n\n// one 2304 B piece (16 rows x 256 K: 4 x 64-K chunks, lane t owns 16 contiguous K = one NVFP4 block per chunk) against NT\n// n-tiles of activations read from smem: bx[n] = this lane\'s B row address (token slot 8n + g; an absent slot reads slot\n// 0\'s row: its products only reach that slot\'s own, discarded, output column) at the piece\'s K offset\ntemplate <int NT>\n__device__ __forceinline__ void piece(uint32_t pb, int lane, const BA (&bx)[NT], float (&acc)[NT][2][4]) {\n  const uint2 sh = lds64(pb + 2048 + lane * 8);\n#pragma unroll\n  for (int c = 0; c < 4; c++) {\n    const uint4 w = lds(pb + c * 512 + lane * 16);\n    uint4 B[NT][2];\n#pragma unroll\n    for (int n = 0; n < NT; n++) { B[n][0] = LDB(bx[n] + c * 128); B[n][1] = LDB(bx[n] + c * 128 + 16); }\n    const uint32_t sw = c < 2 ? sh.x : sh.y;\n    const uint32_t sA = sc2(sw, (c & 1) ? 0x4242 : 0x4040), sB = sc2(sw, (c & 1) ? 0x4343 : 0x4141);\n    const uint32_t ww[4] = {w.x, w.y, w.z, w.w};\n#pragma unroll\n    for (int j = 0; j < 4; j++) {\n      DQ(ww[j], sA, sB, a0, a1, a2, a3)\n#pragma unroll\n      for (int n = 0; n < NT; n++) {\n        const uint32_t* bb = (const uint32_t*)&B[n][j >> 1];\n        mma(acc[n][j & 1], a0, a1, a2, a3, bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);\n      }\n    }\n  }\n}\n\nstruct Ring {\n  const P* p; Sm* sm; uint32_t ring, bars; int lane, grp, s; unsigned long long wns;\n  int n13w, n2w;  // this warp\'s W13 pieces (tiles x PPT) and W2 pieces (units x W2S)\n  // issue cursor (pieces are issued strictly in sequence): W13 tile tt / piece j, then W2 expert ca / row tile crt / piece j\n  int nq, tt, j, ca, crt;\n  __device__ __forceinline__ void init_cursor() {\n    nq = 0; tt = sm->t0 + grp; j = 0;\n    ca = grp / sm->nrt; crt = grp - ca * sm->nrt;\n  }\n  __device__ __forceinline__ void issue_next() {  // one slot: PPS consecutive pieces of one slice (contiguous half-stages)\n    if (nq >= n13w + n2w) return;\n    const uint32_t slot = (nq / PPS) % SG, dst = ring + slot * SLOTB, bar = bars + slot * 8;\n    const uint8_t* src;\n    if (nq < n13w) {\n      src = p->w1 + ((size_t)(p->lo + sm->act[tt >> 9]) * 512 + (tt & 511)) * 8 * STG + (size_t)(sm->k1 * (16 / KS1) + s * PPT + j) * PC;\n      j += PPS; if (j == PPT) { j = 0; tt += TG; }\n    } else {\n      src = p->w2 + ((size_t)(p->lo + sm->act[ca]) * 256 + sm->r0 + crt) * 16 * PC + (size_t)(sm->ksp * (16 / KS2) + s * W2S + j) * PC;\n      j += PPS; if (j == W2S) { j = 0; crt += TG; while (crt >= sm->nrt) { crt -= sm->nrt; ca++; } }\n    }\n    nq += PPS;\n#if TMA\n    if (lane == 0) {\n      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" :: "r"(bar), "n"(SLOTB) : "memory");\n      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\\n" :: "r"(dst), "l"(src), "n"(SLOTB), "r"(bar) : "memory");\n    }\n#else\n    for (int k = lane; k < SLOTB / 16; k += 32) cp16(dst + k * 16, src + k * 16);\n    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\\n" :: "r"(bar) : "memory");\n#endif\n  }\n  __device__ __forceinline__ uint32_t acquire(int q) {\n#ifdef PROF\n    const unsigned long long w0 = gtime();\n#endif\n    const int qs = q / PPS;\n    mb_wait(bars + (qs % SG) * 8, (qs / SG) & 1);\n    __syncwarp();\n#ifdef PROF\n    wns += gtime() - w0;\n#endif\n    return ring + (qs % SG) * SLOTB;\n  }\n  __device__ __forceinline__ void release() { __syncwarp(); issue_next(); }\n};\n\n// one tile slice / unit slice: npc pieces accumulated over the warp\'s K slice, then the group\'s SW-way reduction in rounds\n// of 2 n-tiles; out(n, f, v) consumes fragment f (row f >> 2 / + 8, token slots 8n + 2 (f & 3) + {0, 1}) of n-tile n\ntemplate <int NT, class F>\n__device__ __forceinline__ void slice(Ring& C, int& q, int npc, const BA (&bx)[NT], float4* red, int wid, int grp, int s, int lane, F out) {\n  float acc[NT][2][4];\n#pragma unroll\n  for (int n = 0; n < NT; n++)\n#pragma unroll\n    for (int h = 0; h < 2; h++) acc[n][h][0] = acc[n][h][1] = acc[n][h][2] = acc[n][h][3] = 0.f;\n  for (int j = 0; j < npc; j += PPS, q += PPS) {\n    const uint32_t sbase = C.acquire(q);\n#pragma unroll\n    for (int jj = 0; jj < PPS; jj++) {\n      BA b2[NT];\n#pragma unroll\n      for (int n = 0; n < NT; n++) b2[n] = bx[n] + (j + jj) * 512;\n      piece<NT>(sbase + jj * PC, lane, b2, acc);\n    }\n    C.release();\n  }\n#pragma unroll\n  for (int nb = 0; nb < NT; nb += 2) {\n#pragma unroll\n    for (int n = nb; n < nb + 2 && n < NT; n++)\n      red[(wid * 2 + n - nb) * 32 + lane] = make_float4(acc[n][0][0] + acc[n][1][0], acc[n][0][1] + acc[n][1][1], acc[n][0][2] + acc[n][1][2], acc[n][0][3] + acc[n][1][3]);\n    gbar(1 + grp, SW * 32);\n    if (lane < 2 * (32 / SW)) {\n      const int n = lane / (32 / SW), f = s * (32 / SW) + lane % (32 / SW);\n      if (nb + n < NT) {\n        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);\n#pragma unroll\n        for (int w = 0; w < SW; w++) { const float4 u = red[((grp * SW + w) * 2 + n) * 32 + f]; v.x += u.x; v.y += u.y; v.z += u.z; v.w += u.w; }\n        out(nb + n, f, v);\n      }\n    }\n    gbar(1 + grp, SW * 32);  // partials consumed before they are overwritten\n  }\n}\n\n// sum over the grid\'s per-CTA partials of route r (all loads issued before the first add)\n__device__ __forceinline__ float sum_ssp(const float* ssp, int r, int G, int lane) {\n  float v[8];\n#pragma unroll\n  for (int k = 0; k < 8; k++) { const int c = lane + 32 * k; v[k] = c < G ? __ldcg(ssp + (size_t)c * MAXR + r) : 0.f; }\n  float s = ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7]));\n#pragma unroll\n  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);\n  return s;\n}\n\n__global__ void __launch_bounds__(NW * 32, 1) moe_k(P p) {\n  extern __shared__ __align__(128) uint8_t smem[];\n  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, R = p.M * 8, G = gridDim.x;\n  const int grp = wid / SW, s = wid % SW;\n  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem);\n  Sm* sm = (Sm*)(smem + NW * SG * 8);\n  const int ro = (NW * SG * 8 + (int)sizeof(Sm) + 127) & ~127;              // ring\n  const int rdo = ro + NW * SG * SLOTB;                                       // partials: [NW][2 n-tiles][32] float4\n  const int zo = rdo + NW * 2 * 32 * 16;                                      // zero row\n  const int xo = zo + ZB;                                                     // x rows / staged inter / y rows (union)\n  float4* red = (float4*)(smem + rdo);\n#ifdef EMUX\n  uint8_t* rg = p.scr + (size_t)blockIdx.x * SCR;\n  const int rsz = SCR;\n  const BA rb = rg;\n#else\n  uint8_t* rg = smem + xo;\n  const int rsz = p.smem - xo;\n  const BA rb = sb + xo;\n#endif\n  __nv_bfloat16* xs = (__nv_bfloat16*)rg;\n#ifdef EMUX\n  const BA zb = smem + zo + (lane & 3) * 32;\n#else\n  const BA zb = sb + zo + (lane & 3) * 32;\n#endif\n  PST(0);\n  if (lane < SG) mb_init(sb + (wid * SG + lane) * 8, TMA ? 1 : 32);\n#if TMA\n  asm volatile("fence.mbarrier_init.release.cluster;\\n" ::: "memory");\n#endif\n  // routing: deterministic slots (experts ascending, routes of an expert in route order)\n  int e = -1, rank = 0;\n  if (wid < RW) {\n    if (tid < R) { const unsigned u = p.ids[tid] - p.lo; e = u < NLOC ? (int)u : -1; }\n    const unsigned peers = __match_any_sync(0xffffffffu, e);\n    rank = __popc(peers & ((1u << lane) - 1));\n    sm->hist[wid][lane] = 0;\n    __syncwarp();\n    if (e >= 0 && rank == 0) sm->hist[wid][e] = __popc(peers);\n  }\n  __syncthreads();\n  if (wid == 0) {\n    int c = 0;\n#pragma unroll\n    for (int w = 0; w < RW; w++) { const int h = sm->hist[w][lane]; sm->hist[w][lane] = c; c += h; }\n    const unsigned am = __ballot_sync(0xffffffffu, c > 0);\n    int v = c;\n#pragma unroll\n    for (int o = 1; o < 32; o <<= 1) { const int n = __shfl_up_sync(0xffffffffu, v, o); if (lane >= o) v += n; }\n    const int na = __popc(am), nr = __shfl_sync(0xffffffffu, v, 31);\n    if (c > 0) { const int a = __popc(am & ((1u << lane) - 1)); sm->act[a] = lane; sm->aoff[a] = v - c; }\n    // cost-weighted contiguous W13 tile ranges (lane = expert): a tile of an expert with NT n-tiles costs CW[NT - 1] / 8\n    const int b = blockIdx.x;\n    int k = 0;\n    while ((k + 1) * G / KS1 <= b) k++;\n    int g0 = k * G / KS1, gn = (k + 1) * G / KS1 - g0;\n    const int w = c > 0 ? cwt((c + 7) >> 3) : 0;\n    int cs = 512 * w;\n#pragma unroll\n    for (int o = 1; o < 32; o <<= 1) { const int n = __shfl_up_sync(0xffffffffu, cs, o); if (lane >= o) cs += n; }\n    const int Ct = __shfl_sync(0xffffffffu, cs, 31), base = cs - 512 * w;\n    const int lo = (b - g0) * Ct / gn, hi = (b - g0 + 1) * Ct / gn;\n    const int tl = __reduce_add_sync(0xffffffffu, w && lo > base ? min(512, (lo - base + w - 1) / w) : 0);\n    const int th = __reduce_add_sync(0xffffffffu, w && hi > base ? min(512, (hi - base + w - 1) / w) : 0);\n    __syncwarp();  // act / aoff of every lane visible to lane 0\n    if (lane == 0) {\n      sm->aoff[na] = nr; sm->na = na; sm->nr = nr;\n      sm->k1 = k; sm->t0 = tl; sm->n13 = th - tl;\n      k = 0;\n      while ((k + 1) * G / KS2 <= b) k++;\n      g0 = k * G / KS2; gn = (k + 1) * G / KS2 - g0;\n      sm->ksp = k; sm->r0 = (b - g0) * 256 / gn; sm->nrt = (b - g0 + 1) * 256 / gn - sm->r0;\n#if KS2 == 1\n      sm->o0 = sm->r0 * 16; sm->onr = sm->nrt * 16;\n#else\n      sm->o0 = b * HD / G; sm->onr = (b + 1) * HD / G - sm->o0;\n#endif\n      // W2 staging chunks: runs of whole experts whose routes fit the union region (normally one chunk)\n#ifdef FITCAP\n      const int fit = min((rsz - YBR) / (IS * 2), FITCAP);  // test: force chunked staging\n#else\n      const int fit = (rsz - YBR) / (IS * 2);\n#endif\n      int nch = 0, c0 = 0;\n      sm->cha[0] = 0;\n      for (int a = 0; a < na; a++)\n        if (sm->aoff[a + 1] - sm->aoff[c0] > fit) { sm->cha[++nch] = a; c0 = a; }\n      sm->cha[++nch] = na; sm->nch = nch;\n      if (p.M > MAXM || p.M * XS * 2 > rsz || MAXR * 64 * 4 > rsz || sm->onr > 64 || fit < MAXM || xo > p.smem || G > 256 || (KS2 == 1 && p.M * XS * 2 > rsz - YBR)) asm volatile("trap;");\n    }\n  }\n  __syncthreads();\n  const int na = sm->na, nr = sm->nr, n13 = sm->n13, nrt = sm->nrt;\n  Ring C; C.p = &p; C.sm = sm; C.lane = lane; C.grp = grp; C.s = s; C.wns = 0;\n  C.ring = sb + ro + wid * SG * SLOTB; C.bars = sb + wid * SG * 8;\n  const int nt13 = n13 > grp ? (n13 - grp + TG - 1) / TG : 0, nu = na * nrt, nu2 = nu > grp ? (nu - grp + TG - 1) / TG : 0;\n  C.n13w = nt13 * PPT; C.n2w = nu2 * W2S;\n  C.init_cursor();\n#pragma unroll\n  for (int i = 0; i < SG; i++) C.issue_next();\n  PST(1);\n  if (tid < p.M * 8) sm->tr[tid >> 3][tid & 7] = -1;\n  if (tid < ZB / 16) *(uint4*)(smem + zo + tid * 16) = make_uint4(0, 0, 0, 0);\n  if (tid < 64) sm->nws[tid] = tid < sm->onr ? p.nw[sm->o0 + tid] : 0.f;\n  __syncthreads();\n  if (tid < R && e >= 0) {\n    int a = 0;\n    while (sm->act[a] != e) a++;\n    const int r = sm->aoff[a] + sm->hist[wid][e] + rank;\n    sm->rid[r] = tid; sm->rexp[r] = a; sm->tr[tid >> 3][tid & 7] = r;\n    sm->cf[r] = p.tw[tid];\n    if (rank == 0 && sm->hist[wid][e] == 0) { sm->gs1[a] = p.g1[p.lo + e]; sm->gs2[a] = p.g2[p.lo + e]; }\n  }\n  {\n    const int kx = sm->k1 * (HD / KS1);\n    for (int i = tid; i < p.M * (HD / KS1 / 8); i += NW * 32) {\n      const int row = i / (HD / KS1 / 8), c8 = i - row * (HD / KS1 / 8);\n      *(uint4*)(xs + row * XS + c8 * 8) = *((const uint4*)(p.x + (size_t)row * HD + kx) + c8);\n    }\n  }\n  __syncthreads();\n  int q = 0;\n#ifdef PROF\n  if (tid == 0 && p.prof) p.prof[blockIdx.x * 16 + 11] = clock64();\n#endif\n  if (na) {\n    // ---- W13: group grp takes tiles grp, grp + TG, ... of the CTA\'s range; warp s the K slice s of the CTA\'s K range\n    for (int i = 0; i < nt13; i++) {\n      const int tt = sm->t0 + grp + i * TG, a = tt >> 9, T = sm->aoff[a + 1] - sm->aoff[a], ao = sm->aoff[a];\n      const BA xb = rb + (s * (HD / KS1 / SW) + (lane & 3) * 16) * 2;\n      const int g = lane >> 2;\n      const int col0 = (tt & 511) * 8;\n      const float gs = sm->gs1[a];\n      auto out = [&](int n, int f, float4 v) {\n        const int gg = f >> 2, tt2 = f & 3, col = col0 + gg;\n#pragma unroll\n        for (int h = 0; h < 2; h++) {\n          const int slot = n * 8 + 2 * tt2 + h;\n          if (slot < T) {\n#if KS1 == 1\n            const float gt = (h ? v.y : v.x) * gs, up = (h ? v.w : v.z) * gs;\n            p.inter[(size_t)(ao + slot) * HD + col] = __float2bfloat16(muse_swiglu_bf16(gt, up));\n#else\n            *(float2*)(p.gp + (((size_t)sm->k1 * MAXR + ao + slot) * HD + col) * 2) = make_float2(h ? v.y : v.x, h ? v.w : v.z);\n#endif\n          }\n        }\n      };\n#define BXR(n) ((n) * 8 + g < T ? xb + (sm->rid[ao + (n) * 8 + g] >> 3) * XS * 2 : zb)\n      if (T <= 8) { BA bx[1] = {BXR(0)}; slice<1>(C, q, PPT, bx, red, wid, grp, s, lane, out); }\n#if MAXNT >= 2\n      else if (T <= 16) { BA bx[2] = {BXR(0), BXR(1)}; slice<2>(C, q, PPT, bx, red, wid, grp, s, lane, out); }\n#endif\n#if MAXNT >= 3\n      else if (T <= 24) { BA bx[3] = {BXR(0), BXR(1), BXR(2)}; slice<3>(C, q, PPT, bx, red, wid, grp, s, lane, out); }\n      else { BA bx[4] = {BXR(0), BXR(1), BXR(2), BXR(3)}; slice<4>(C, q, PPT, bx, red, wid, grp, s, lane, out); }\n#endif\n#undef BXR\n    }\n    PST(2);\n#ifdef PROF\n    if (tid == 0 && p.prof) p.prof[blockIdx.x * 16 + 12] = clock64();\n    if (lane == 0 && p.prof) atomicAdd(p.prof + blockIdx.x * 16 + 8, C.wns);\n    C.wns = 0;\n#endif\n    grid_sync(p.ctr, G);\n#if KS1 > 1\n    // ---- inter from the KS1 gate/up partials: this CTA\'s slice of all (route, column) pairs\n    {\n      const int N = nr * HD, i0 = (int)((long long)blockIdx.x * N / G), i1 = (int)((long long)(blockIdx.x + 1) * N / G);\n      for (int ib = i0; ib < i1; ib += 4 * NW * 32) {  // 4 elements per thread with all partial loads in flight together\n        float2 u[4][KS1];\n#pragma unroll\n        for (int j = 0; j < 4; j++) {\n          const int i = ib + j * NW * 32 + tid, ic = i < i1 ? i : i0, r = ic / HD, col = ic - r * HD;\n#pragma unroll\n          for (int k = 0; k < KS1; k++) u[j][k] = __ldcg((const float2*)(p.gp + (((size_t)k * MAXR + r) * HD + col) * 2));\n        }\n#pragma unroll\n        for (int j = 0; j < 4; j++) {\n          const int i = ib + j * NW * 32 + tid, r = i / HD, col = i - r * HD;\n          if (i < i1) {\n            float gx = 0.f, ux = 0.f;\n#pragma unroll\n            for (int k = 0; k < KS1; k++) { gx += u[j][k].x; ux += u[j][k].y; }\n            const float gs = sm->gs1[sm->rexp[r]], gt = gx * gs, up = ux * gs;\n            p.inter[(size_t)r * HD + col] = __float2bfloat16(muse_swiglu_bf16(gt, up));\n          }\n        }\n      }\n    }\n    grid_sync(p.ctr, G);\n#endif\n    PST(3);\n    // ---- W2: unit u = (expert u / nrt, row tile r0 + u % nrt), group grp takes units grp, grp + TG, ...; warp s the K slice.\n    // inter[route][K range of this CTA group] is staged in smem per chunk of whole experts (normally one chunk)\n    const int kb = sm->ksp * (HD / KS2);\n    int i = 0;\n    float* yp = p.yp + (size_t)sm->ksp * MAXR * HD;\n    float* ybl = (float*)(rg + rsz - YBR);  // KS2 = 1 only\n    for (int ch = 0; ch < sm->nch; ch++) {\n      const int ca0 = sm->cha[ch], ca1 = sm->cha[ch + 1], rb0 = sm->aoff[ca0], nrc = sm->aoff[ca1] - rb0;\n      for (int k = tid; k < nrc * (HD / KS2 / 8); k += NW * 32) {\n        const int r = k / (HD / KS2 / 8), c8 = k - r * (HD / KS2 / 8);\n        *(uint4*)(xs + r * IS + c8 * 8) = __ldcg((const uint4*)(p.inter + (size_t)(rb0 + r) * HD + kb) + c8);\n      }\n      __syncthreads();\n      for (; i < nu2 && grp + i * TG < ca1 * nrt; i++) {\n        const int u = grp + i * TG, a = u / nrt, rt = sm->r0 + u - a * nrt, T = sm->aoff[a + 1] - sm->aoff[a], ao = sm->aoff[a];\n        const int g = lane >> 2, ab = ao - rb0;\n        const BA xb = rb + (s * (HD / KS2 / SW) + (lane & 3) * 16) * 2;\n        auto out = [&](int n, int f, float4 v) {\n          const int gg = f >> 2, tt2 = f & 3, row = rt * 16 + gg;\n#pragma unroll\n          for (int h = 0; h < 2; h++) {\n            const int slot = n * 8 + 2 * tt2 + h;\n#if KS2 == 1\n            if (slot < T) { float* y = ybl + (ao + slot) * 64 + row - sm->o0; y[0] = h ? v.y : v.x; y[8] = h ? v.w : v.z; }\n#else\n            if (slot < T) { float* y = yp + (size_t)(ao + slot) * HD + row; y[0] = h ? v.y : v.x; y[8] = h ? v.w : v.z; }\n#endif\n          }\n        };\n#define BXR(n) ((n) * 8 + g < T ? xb + (ab + (n) * 8 + g) * IS * 2 : zb)\n        if (T <= 8) { BA bx[1] = {BXR(0)}; slice<1>(C, q, W2S, bx, red, wid, grp, s, lane, out); }\n#if MAXNT >= 2\n        else if (T <= 16) { BA bx[2] = {BXR(0), BXR(1)}; slice<2>(C, q, W2S, bx, red, wid, grp, s, lane, out); }\n#endif\n#if MAXNT >= 3\n        else if (T <= 24) { BA bx[3] = {BXR(0), BXR(1), BXR(2)}; slice<3>(C, q, W2S, bx, red, wid, grp, s, lane, out); }\n        else { BA bx[4] = {BXR(0), BXR(1), BXR(2), BXR(3)}; slice<4>(C, q, W2S, bx, red, wid, grp, s, lane, out); }\n#endif\n#undef BXR\n      }\n      __syncthreads();\n    }\n    PST(4);\n#ifdef PROF\n    if (lane == 0 && p.prof) atomicAdd(p.prof + blockIdx.x * 16 + 9, C.wns);\n#endif\n#if KS2 == 1\n    float* yb = ybl;\n    PST(5);\n#else\n    grid_sync(p.ctr, G);\n    PST(5);\n    // ---- this CTA\'s output rows: y = sum of the KS2 partials (8 elements per thread in flight together)\n    float* yb = (float*)rg;\n    for (int ib = 0; ib < nr * 64; ib += 8 * NW * 32) {\n      float v[8][KS2];\n#pragma unroll\n      for (int j = 0; j < 8; j++) {\n        const int i2 = ib + j * NW * 32 + tid, r = i2 >> 6, rr = i2 & 63;\n        const bool ok = i2 < nr * 64 && rr < sm->onr;\n#pragma unroll\n        for (int k = 0; k < KS2; k++) v[j][k] = ok ? __ldcg(p.yp + ((size_t)k * MAXR + r) * HD + sm->o0 + rr) : 0.f;\n      }\n#pragma unroll\n      for (int j = 0; j < 8; j++) {\n        const int i2 = ib + j * NW * 32 + tid;\n        float t = 0.f;\n#pragma unroll\n        for (int k = 0; k < KS2; k++) t += v[j][k];\n        if (i2 < nr * 64) yb[i2] = t;\n      }\n    }\n#endif\n    __syncthreads();\n    // Complete the W2 reduction and BF16 materialization before norm weighting.\n    for (int i = tid; i < nr * 64; i += NW * 32) {\n      const int r = i >> 6, rr = i & 63;\n      if (rr < sm->onr) yb[i] = __bfloat162float(__float2bfloat16(yb[i] * sm->gs2[sm->rexp[r]]));\n    }\n    __syncthreads();\n    const int onr = sm->onr;\n    for (int r = wid; r < nr; r += NW) {\n      float ss = 0.f;\n      for (int rr = lane; rr < onr; rr += 32) { const float v = yb[r * 64 + rr] * sm->nws[rr]; ss += v * v; }\n#pragma unroll\n      for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);\n      if (lane == 0) p.ssp[(size_t)blockIdx.x * MAXR + r] = ss;\n    }\n    grid_sync(p.ctr, G);\n    PST(6);\n    // per-route norm: 16 threads per route, all of a thread\'s (<= 16) partial loads in flight together\n    for (int rb2 = 0; rb2 < nr; rb2 += NW * 2) {\n      const int r = rb2 + (tid >> 4), l = tid & 15;\n      float v[16];\n#pragma unroll\n      for (int j = 0; j < 16; j++) { const int c = l + 16 * j; v[j] = r < nr && c < G ? __ldcg(p.ssp + (size_t)c * MAXR + r) : 0.f; }\n      float ssum = (((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7]))) + (((v[8] + v[9]) + (v[10] + v[11])) + ((v[12] + v[13]) + (v[14] + v[15])));\n#pragma unroll\n      for (int o = 8; o; o >>= 1) ssum += __shfl_xor_sync(0xffffffffu, ssum, o);\n      if (l == 0 && r < nr) { const float g2 = sm->gs2[sm->rexp[r]]; sm->coef[r] = sm->cf[r] * rsqrtf(ssum * (1.f / HD) + 1e-8f); }\n    }\n    __syncthreads();\n  }\n  {\n    const int o0 = sm->o0, onr = sm->onr;\n#if KS2 == 1\n    const float* yb = (const float*)(rg + rsz - YBR);\n#else\n    const float* yb = (const float*)rg;\n#endif\n    for (int i2 = tid; i2 < p.M * onr; i2 += NW * 32) {\n      const int tk = i2 / onr, rr = i2 - tk * onr;\n      float o = 0.f;\n#pragma unroll\n      for (int k = 0; k < 8; k++) { const int r = sm->tr[tk][k]; if (r >= 0) o += sm->coef[r] * yb[r * 64 + rr]; }\n      p.out[(size_t)tk * HD + o0 + rr] = __float2bfloat16(o * sm->nws[rr]);\n    }\n  }\n  PST(7);\n}\n\nstatic int g_gridK = 0, g_smemK = 0;\nvoid moe(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor w1, torch::Tensor w2, torch::Tensor g1, torch::Tensor g2,\n         torch::Tensor nw, torch::Tensor inter, torch::Tensor gp, torch::Tensor yp, torch::Tensor ssp, torch::Tensor ctr, torch::Tensor scr, torch::Tensor out, int64_t lo, torch::Tensor prof) {\n  if (!g_gridK) {\n    int dev; cudaGetDevice(&dev); int mx; cudaDeviceGetAttribute(&mx, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);\n    int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);\n    cudaFuncSetAttribute(moe_k, cudaFuncAttributeMaxDynamicSharedMemorySize, mx);\n    g_smemK = mx; g_gridK = sms;\n  }\n  {\n    int dev; cudaGetDevice(&dev); int ma; cudaDeviceGetAttribute(&ma, cudaDevAttrComputeCapabilityMajor, dev);\n    const int M = (int)x.size(0), ro = (NW * SG * 8 + (int)sizeof(Sm) + 127) & ~127, xo = ro + NW * SG * SLOTB + NW * 2 * 32 * 16 + ZB;\n    const int rsz = ma >= 9 ? g_smemK - xo : SCR;\n    TORCH_CHECK(M <= MAXM && xo <= g_smemK && g_gridK <= 256 && g_gridK >= 64, "gk: unsupported shape");\n    TORCH_CHECK((long long)M * XS * 2 <= rsz - YBR && MAXR * 64 * 4 <= rsz && (rsz - YBR) / (IS * 2) >= MAXM, "gk: smem layout does not fit");\n  }\n  P p{w1.data_ptr<uint8_t>(), w2.data_ptr<uint8_t>(), g1.data_ptr<float>(), g2.data_ptr<float>(), (const __nv_bfloat16*)x.data_ptr(),\n      ids.data_ptr<int>(), tw.data_ptr<float>(), nw.data_ptr<float>(), (__nv_bfloat16*)inter.data_ptr(), gp.data_ptr<float>(), yp.data_ptr<float>(),\n      ssp.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(), ctr.data_ptr<int>(), scr.data_ptr<uint8_t>(), (int)x.size(0), (int)lo, g_smemK,\n      prof.numel() ? (unsigned long long*)prof.data_ptr() : nullptr};\n  moe_k<<<g_gridK, NW * 32, g_smemK, at::cuda::getCurrentCUDAStream()>>>(p);\n}\n'
_modK = {}
GK_MAX = 32      # gk for M <= GK_MAX (activations of all tokens, or of half of K, in smem)
GK_MIN = 1
K_CFG = ("-DSW=4 -DKS2=2", "-DSW=4 -DKS2=4", "-DSW=4 -DKS1=2 -DKS2=4 -DMAXM=32", "-DSW=4 -DKS2=1 -DMAXM=2 -DPPS=2", "-DSW=4 -DKS2=1 -DMAXM=8")  # tile-group width x W13 / W2 K splits (MAXM 16 unless given)
K32_MIN = 12     # smallest M offered the MAXM 32 variant


def _gkmax(x):
    return int(x.split("-DMAXM=")[1].split()[0]) if "-DMAXM=" in x else 16
DIAGK = False    # prepare(): gk per-CTA timeline (PROF build) at M8 / M16 on judge routings


def _extK(extra=""):
    if extra not in _modK:
        fl = extra.replace("+", " ").split() + [d for d in _DBG if not d.startswith("-DCFG=")]
        _modK[extra] = load_inline("moe_fullbf16_v48K" + "".join(fl).replace("-", "_").replace("=", ""),
                                   cpp_sources="void moe(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,torch::Tensor);",
                                   cuda_sources=CUDA_K, functions=["moe"], extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"] + fl, verbose=False)
    return _modK[extra]


def _gk(data, state, cfg, prof=None):
    x = data["hidden_states"].contiguous()
    ids = data["topk_ids"].to(torch.int32).contiguous()
    tw = data["topk_weights"].float().contiguous()
    out = torch.empty_like(x)
    _extK(cfg[3] if len(cfg) > 3 else "").moe(x, ids, tw, state["w1"], state["w2"], state["g1"], state["g2"], state["nw"], state["interK"],
                                              state["gpK"], state["ypK"], state["sspK"], state["ctrK"], state["scrK"], out, data["config"]["local_start"],
                                              state["prof"][:0] if prof is None else prof)
    return out


def _diagK(state):
    """gk on judge-generator routings: per-CTA timeline (us from the first CTA start), stream rates and the SM clock over
    W13, for each gk shape at the M it targets"""
    dev = state["w1"].device
    flush = torch.empty(2 ** 27, device=dev); sink = torch.empty((), device=dev)
    G = torch.cuda.get_device_properties(dev).multi_processor_count
    prof = torch.zeros(G * 16, dtype=torch.int64, device=dev)
    runs = [(1, K_CFG[3]), (1, K_CFG[4]), (1, K_CFG[0]), (2, K_CFG[3]), (8, K_CFG[4]), (8, K_CFG[0]), (16, K_CFG[0]), (24, K_CFG[2]), (32, K_CFG[2])]
    for M, x in runs:
        for sd in range(2):
            data = _synth(M, 900 + 7 * sd + M, dev)
            ids = data["topk_ids"]; na = int(torch.unique(ids[ids < 32]).numel()); nr = int((ids < 32).sum())
            mt = int(torch.bincount(ids[ids < 32].long(), minlength=32).max())
            ev = []
            try:
                for _ in range(3):
                    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                    prof.zero_(); sink.copy_(flush.sum()); e0.record(); _gk(data, state, ("gk", 0, 0, (x + " -DPROF").strip()), prof); e1.record()
                    torch.cuda.synchronize(); ev.append(e0.elapsed_time(e1) * 1e3)
            except Exception as ex:
                print(f"KD M{M} {_nm(('gk', 0, 0, x))} failed {str(ex)[:80]}", flush=True); continue
            P = prof.view(-1, 16).cpu().double(); t0 = P[:, 0].min()
            q = lambda k, f: float((P[:, k] - t0).quantile(f)) / 1e3
            d13 = (P[:, 2] - P[:, 1]).clamp(min=1); d2 = (P[:, 4] - P[:, 3]).clamp(min=1)
            wA = float((P[:, 8] / (16 * d13)).median()); wB = float((P[:, 9] / (16 * d2)).median())
            mhz = float(((P[:, 12] - P[:, 11]) / d13).median()) * 1e3
            b13 = na * 512 * 36864 / G; b2 = na * 256 * 36864 / G
            print(f"KD M{M} {_nm(('gk', 0, 0, x))} na {na} nr {nr} maxT {mt} evt {ev[-1]:.1f}: pro {q(1, .5):.1f}/{q(1, 1):.1f} W13 {q(2, .5):.1f}/{q(2, 1):.1f} "
                  f"bar1 {q(3, 1):.1f} W2 {q(4, .5):.1f}/{q(4, 1):.1f} bar2 {q(5, 1):.1f} bar3 {q(6, 1):.1f} end {q(7, 1):.1f} | waitA {wA:.2f} waitB {wB:.2f} | SM {mhz:.0f} MHz | "
                  f"W13 {b13 / float(d13.median()) * G / 1e3:.2f} TB/s W2 {b2 / float(d2.median()) * G / 1e3:.2f} TB/s", flush=True)


CUDA_X = '#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cuda_bf16.h>\n// Strict SGLang boundaries: BF16 gate/up -> FP32 SiLU/product -> BF16 inter.\n__device__ __forceinline__ float strict_swiglu(float gate, float up) {\n  gate=__bfloat162float(__float2bfloat16_rn(gate));\n  up=__bfloat162float(__float2bfloat16_rn(up));\n  return gate/(1.f+__expf(-gate))*up;\n}\n\n#include <stdint.h>\n\n// v43 "wx": one ring of 4-chunk group slots; NCW consumer warps (8 or 16), RPW m64 blocks per warpgroup: one pass per (expert, 128-row block) over ALL the expert\'s tokens (N <= 256): every weight is dequantized\n// once per expert. Slots are 64-K chunks: [activation tile N x 128 B, SW128-swizzled, from a per-call gathered buffer]\n// [8 tiles x 576 B of weights (512 B mma A-fragment words + 64 B block scales)], both by TMA. Two consumer warpgroups\n// (m64 each) run wgmma RS: A = dequantized fragment in registers, B = the activation tile; per k16 step\n// wait<3> / dequant / fence / wgmma / commit (CUTLASS mixed-input RS order: four A register sets in rotation).\n#define HD 4096\n#define NLOC 32\n#define TPU 8\n#define REC 576\n#define WREC (TPU * REC)\n#define NCH 64\n#ifndef NCW\n#define NCW 8\n#endif\n#ifndef RPW\n#define RPW 1  // m64 blocks per consumer warpgroup: unit = 128 * RPW rows sharing each activation tile\n#endif\n#ifndef TPX\n#define TPX ((RPW == 1 && NCW == 8) ? 128 : 64)  // tokens per pass (accumulators RPW * TPX / 2 per thread)\n#endif\n#define XB (TPX * 128)\n#define WCH 4                      // 64-K chunks per slot\n#define UBL ((NCW / 8) * RPW)      // 128-row blocks per unit\n#define WSB (WCH * WREC * UBL)\n#define ASB (WCH * XB)\n#define GSB (((ASB + WSB) + 1023) & ~1023)\n#ifndef NSL\n#define NSL (GSB <= 73728 ? 3 : 2)\n#endif\n#define RING (NSL * GSB)\n#define NRB1 (64 / UBL)\n#define NRB2 (32 / UBL)\n#define NTH ((NCW + 1) * 32)\n#define C_W2 32\n#define C_Q 33\n#define C_DONE 34\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900\n#define TMA 1\n#else\n#define TMA 0\n#endif\n\nstruct P {\n  const uint8_t* w1; const uint8_t* w2; const float* g1; const float* g2;\n  const uint8_t* xg; uint8_t* ig; const int* ids; const float* tw; const float* nw;\n  __nv_bfloat16* y; float* ss; __nv_bfloat16* out; int* ctr;\n  int M, lo; unsigned long long* prof;\n};\nstruct Hdr { int g, e, rb, ps, c, ntok, flags, pad; };\nstruct Sm {\n  Hdr hdr[NSL];\n  int cnt[NLOC], off[NLOC], act[NLOC], np[NLOC], base[NLOC], npad[NLOC];\n  long long xoff[NLOC];\n  int segst[2 * NLOC + 1], segk[2 * NLOC];\n  int nseg, na, ntot, nw2;\n  int wc[NTH / 32][NLOC];\n};\n\n#ifdef PROF\n__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }\n#define PSET(k, v) do { if (lane == 0 && p.prof) p.prof[(blockIdx.x * (NTH / 32) + wid) * 16 + (k)] = (v); } while (0)\n#else\n#define PSET(k, v)\n#endif\n\n__device__ __forceinline__ int kperm(int col) {  // wgmma K order inside each 64-K chunk\n  const int a = col & 63;\n  return (col & ~63) | (((a >> 2) & 3) << 4) | (((a & 3) >> 1) << 3) | ((a >> 4) << 1) | (a & 1);\n}\n__device__ __forceinline__ uint32_t xaddr_of(int pcol, int pos, int npad) {  // byte offset in an expert\'s gathered block\n  const int b = (pcol & 63) * 2;\n  return ((uint32_t)(pcol >> 6) * npad + pos) * 128u + ((((b >> 4) ^ (pos & 7))) << 4) + (b & 15);\n}\n__device__ __forceinline__ void cp16(uint32_t s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\\n" :: "r"(s), "l"(g) : "memory"); }\n__device__ __forceinline__ uint32_t bmul(uint32_t a, uint32_t b) { uint32_t d; asm("fma.rn.bf16x2 %0, %1, %2, %3;\\n" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u)); return d; }\n__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {\n  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\\n"\n    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n__device__ __forceinline__ int ldacq(const int* p) { int v; asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\\n" : "=r"(v) : "l"(p) : "memory"); return v; }\n__device__ __forceinline__ void red_rel(int* p, int v) { asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" :: "l"(p), "r"(v) : "memory"); }\n__device__ __forceinline__ uint32_t sc2(uint32_t w, uint32_t sel) { return __byte_perm(w, 0, sel) * 16u + 0x77007700u; }\n__device__ __forceinline__ void mbinit(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\\n" :: "r"(b), "r"(n)); }\n__device__ __forceinline__ void mbarrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\\n" :: "r"(b) : "memory"); }\n__device__ __forceinline__ void cpa_arrive(uint32_t b) { asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\\n" :: "r"(b) : "memory"); }\n__device__ __forceinline__ void mbwait(uint32_t b, uint32_t ph) {\n#if TMA\n  asm volatile("{\\n.reg .pred p;\\nW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\\n@!p bra W;\\n}\\n" :: "r"(b), "r"(ph) : "memory");\n#else\n  asm volatile("{\\n.reg .pred p;\\nW: mbarrier.test_wait.parity.shared::cta.b64 p, [%0], %1;\\n@!p bra W;\\n}\\n" :: "r"(b), "r"(ph) : "memory");\n#endif\n}\n__device__ __forceinline__ uint32_t lds32(uint32_t a) { uint32_t v; asm volatile("ld.shared.u32 %0, [%1];\\n" : "=r"(v) : "r"(a)); return v; }\n__device__ __forceinline__ uint32_t lds16(uint32_t a) { unsigned short v; asm volatile("ld.shared.u16 %0, [%1];\\n" : "=h"(v) : "r"(a)); return v; }\n__device__ __forceinline__ uint4 lds128(uint32_t a) { uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v; }\n\n#if TMA\n__device__ __forceinline__ void expect_tx(uint32_t bar, uint32_t n) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" :: "r"(bar), "r"(n) : "memory"); }\n__device__ __forceinline__ void bulk(uint32_t dst, const void* src, uint32_t n, uint32_t bar) {\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\\n" :: "r"(dst), "l"(src), "r"(n), "r"(bar) : "memory");\n}\n__device__ __forceinline__ uint64_t gdesc(uint32_t saddr) {  // K-major SW128: LBO 16, SBO 1024\n  return (uint64_t)((saddr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)64 << 32) | (1ull << 62);\n}\n__device__ __forceinline__ uint64_t gdesc_add(uint64_t d, uint32_t off) {\n  const uint32_t lo = (uint32_t)d + (off >> 4), hi = (uint32_t)(d >> 32);\n  return ((uint64_t)hi << 32) | lo;\n}\n__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\\n" ::: "memory"); }\n__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\\n" ::: "memory"); }\ntemplate <int K> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\\n" :: "n"(K) : "memory"); }\ntemplate <int N> __device__ __forceinline__ void wg_rs(float* d, const uint32_t* a, uint64_t db, int sd);\ntemplate <int N> __device__ __forceinline__ void fence_acc(float* d);\ntemplate <> __device__ __forceinline__ void wg_rs<8>(float* d, const uint32_t* a, uint64_t db, int sd) {\n  asm volatile("{\\n.reg .pred p;\\nsetp.ne.b32 p, %9, 0;\\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, p, 1, 1, 0;\\n}\\n"\n    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])\n    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(sd));\n}\ntemplate <> __device__ __forceinline__ void fence_acc<8>(float* d) {\n#pragma unroll\n  for (int i = 0; i < 4; i++) asm volatile("" : "+f"(d[i]) :: "memory");\n}\ntemplate <> __device__ __forceinline__ void wg_rs<16>(float* d, const uint32_t* a, uint64_t db, int sd) {\n  asm volatile("{\\n.reg .pred p;\\nsetp.ne.b32 p, %13, 0;\\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, %12, p, 1, 1, 0;\\n}\\n"\n    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])\n    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(sd));\n}\ntemplate <> __device__ __forceinline__ void fence_acc<16>(float* d) {\n#pragma unroll\n  for (int i = 0; i < 8; i++) asm volatile("" : "+f"(d[i]) :: "memory");\n}\ntemplate <> __device__ __forceinline__ void wg_rs<32>(float* d, const uint32_t* a, uint64_t db, int sd) {\n  asm volatile("{\\n.reg .pred p;\\nsetp.ne.b32 p, %21, 0;\\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, p, 1, 1, 0;\\n}\\n"\n    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])\n    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(sd));\n}\ntemplate <> __device__ __forceinline__ void fence_acc<32>(float* d) {\n#pragma unroll\n  for (int i = 0; i < 16; i++) asm volatile("" : "+f"(d[i]) :: "memory");\n}\ntemplate <> __device__ __forceinline__ void wg_rs<64>(float* d, const uint32_t* a, uint64_t db, int sd) {\n  asm volatile("{\\n.reg .pred p;\\nsetp.ne.b32 p, %37, 0;\\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1, 0;\\n}\\n"\n    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])\n    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(sd));\n}\ntemplate <> __device__ __forceinline__ void fence_acc<64>(float* d) {\n#pragma unroll\n  for (int i = 0; i < 32; i++) asm volatile("" : "+f"(d[i]) :: "memory");\n}\ntemplate <> __device__ __forceinline__ void wg_rs<128>(float* d, const uint32_t* a, uint64_t db, int sd) {\n  asm volatile("{\\n.reg .pred p;\\nsetp.ne.b32 p, %69, 0;\\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, {%64,%65,%66,%67}, %68, p, 1, 1, 0;\\n}\\n"\n    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])\n    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(sd));\n}\ntemplate <> __device__ __forceinline__ void fence_acc<128>(float* d) {\n#pragma unroll\n  for (int i = 0; i < 64; i++) asm volatile("" : "+f"(d[i]) :: "memory");\n}\ntemplate <> __device__ __forceinline__ void wg_rs<256>(float* d, const uint32_t* a, uint64_t db, int sd) {\n  asm volatile("{\\n.reg .pred p;\\nsetp.ne.b32 p, %133, 0;\\nwgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127}, {%128,%129,%130,%131}, %132, p, 1, 1, 0;\\n}\\n"\n    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])\n    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(sd));\n}\ntemplate <> __device__ __forceinline__ void fence_acc<256>(float* d) {\n#pragma unroll\n  for (int i = 0; i < 128; i++) asm volatile("" : "+f"(d[i]) :: "memory");\n}\n\ntemplate <> __device__ __forceinline__ void wg_rs<24>(float* d, const uint32_t* a, uint64_t db, int sd) {\n  asm volatile("{\\n.reg .pred p;\\nsetp.ne.b32 p, %17, 0;\\nwgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, p, 1, 1, 0;\\n}\\n"\n    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11])\n    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(sd));\n}\ntemplate <> __device__ __forceinline__ void fence_acc<24>(float* d) {\n#pragma unroll\n  for (int i = 0; i < 12; i++) asm volatile("" : "+f"(d[i]) :: "memory");\n}\ntemplate <> __device__ __forceinline__ void wg_rs<48>(float* d, const uint32_t* a, uint64_t db, int sd) {\n  asm volatile("{\\n.reg .pred p;\\nsetp.ne.b32 p, %29, 0;\\nwgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23}, {%24,%25,%26,%27}, %28, p, 1, 1, 0;\\n}\\n"\n    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23])\n    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(sd));\n}\ntemplate <> __device__ __forceinline__ void fence_acc<48>(float* d) {\n#pragma unroll\n  for (int i = 0; i < 24; i++) asm volatile("" : "+f"(d[i]) :: "memory");\n}\n#else\n// sm_80/86 emulation: each warp runs mma.sync for its 16 rows on the same swizzled B tile\n__device__ __forceinline__ uint64_t gdesc(uint32_t saddr) { return saddr; }\n__device__ __forceinline__ uint64_t gdesc_add(uint64_t d, uint32_t off) { return d + off; }\n__device__ __forceinline__ void wg_fence() {}\n__device__ __forceinline__ void wg_commit() {}\ntemplate <int K> __device__ __forceinline__ void wg_wait() {}\ntemplate <int N> __device__ __forceinline__ void fence_acc(float* d) {}\ntemplate <int N> __device__ __forceinline__ void wg_rs(float* d, const uint32_t* a, uint64_t db, int sd) {\n  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;\n  const uint32_t baddr = (uint32_t)db, atom = baddr & ~127u, kc = (baddr & 127u) >> 4;\n  if (!sd) {\n#pragma unroll\n    for (int i = 0; i < N / 2; i++) d[i] = 0.f;\n  }\n#pragma unroll\n  for (int i = 0; i < N / 8; i++) {\n    const int row = i * 8 + g;\n    const uint32_t rb = atom + row * 128;\n    mma(d + i * 4, a[0], a[1], a[2], a[3], lds32(rb + ((kc ^ (row & 7)) << 4) + t * 4), lds32(rb + (((kc + 1) ^ (row & 7)) << 4) + t * 4));\n  }\n}\n#endif\n\n// gather: route i of a local expert e -> row pos(i) of e\'s block, K in the wgmma order, 16-B chunks swizzled\n__global__ void __launch_bounds__(256) xgather(const __nv_bfloat16* x, const int* ids, int M, int lo, uint8_t* xg) {\n  __shared__ int cnt[NLOC];\n  __shared__ long long xo[NLOC];\n  __shared__ int npad[NLOC];\n  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, R = M * 8;\n  if (tid < NLOC) cnt[tid] = 0;\n  __syncthreads();\n  for (int i = tid; i < R; i += 256) { const unsigned e = ids[i] - lo; if (e < NLOC) atomicAdd(&cnt[e], 1); }\n  __syncthreads();\n  if (tid == 0) { long long o = 0; for (int e = 0; e < NLOC; e++) { npad[e] = (cnt[e] + 7) & ~7; xo[e] = o; o += (long long)npad[e] * NCH * 128; } }\n  __syncthreads();\n  const int i = blockIdx.x * 8 + wid;\n  if (i >= R) return;\n  const unsigned e = ids[i] - lo;\n  if (e >= NLOC) return;\n  int pos = 0;\n  for (int j0 = 0; j0 < i; j0 += 32) {\n    const int j = j0 + lane;\n    const bool m = j < i && (unsigned)(ids[j] - lo) == e;\n    pos += __popc(__ballot_sync(0xffffffffu, m));\n  }\n  const uint32_t* src = (const uint32_t*)(x + (size_t)(i >> 3) * HD);\n  uint8_t* dst = xg + xo[e];\n  for (int q = lane; q < HD / 2; q += 32) {\n    const uint32_t v = __ldg(src + q);\n    *(uint32_t*)(dst + xaddr_of(kperm(q * 2), pos, npad[e])) = v;\n  }\n}\n\ntemplate <int N>\n__device__ __forceinline__ void epilogue(const P& p, const Sm* sm, const uint16_t* list, float* acc, const Hdr& h, int mt, int lane) {\n  const int g = lane >> 2, t = lane & 3, e = h.e, ntok = h.ntok;\n  const uint16_t* lst = list + sm->off[e] + h.ps * TPX;\n  if (!h.g) {\n    const float gs = p.g1[p.lo + e];\n    uint8_t* ib = p.ig + sm->xoff[e];\n    const int pc = kperm(mt * 8 + g), npad = sm->npad[e];\n#pragma unroll\n    for (int i = 0; i < N / 8; i++)\n#pragma unroll\n      for (int hh = 0; hh < 2; hh++) {\n        const int tok = i * 8 + 2 * t + hh;\n        const float gt = acc[i * 4 + hh] * gs, up = acc[i * 4 + 2 + hh] * gs;\n        if (tok < ntok) *(__nv_bfloat16*)(ib + xaddr_of(pc, h.ps * TPX + tok, npad)) = __float2bfloat16(strict_swiglu(gt, up));\n      }\n  } else {\n    const float gs = p.g2[p.lo + e];\n    const float n0 = p.nw[mt * 16 + g], n1 = p.nw[mt * 16 + g + 8];\n#pragma unroll\n    for (int i = 0; i < N / 8; i++) {\n      // Complete W2, apply its own expert\'s global scale, then round BEFORE norm.\n      const float q0=__bfloat162float(__float2bfloat16(acc[i*4]*gs)), q1=__bfloat162float(__float2bfloat16(acc[i*4+1]*gs)), q2=__bfloat162float(__float2bfloat16(acc[i*4+2]*gs)), q3=__bfloat162float(__float2bfloat16(acc[i*4+3]*gs));\n      const float v0=q0*n0, v1=q1*n0, v2=q2*n1, v3=q3*n1;\n      float sa = v0 * v0 + v2 * v2, sb = v1 * v1 + v3 * v3;\n      const int t0 = i * 8 + 2 * t;\n      if (t0 < ntok) { __nv_bfloat16* yr = p.y + (size_t)lst[t0] * HD + mt * 16 + g; yr[0] = __float2bfloat16(q0); yr[8] = __float2bfloat16(q2); }\n      if (t0 + 1 < ntok) { __nv_bfloat16* yr = p.y + (size_t)lst[t0 + 1] * HD + mt * 16 + g; yr[0] = __float2bfloat16(q1); yr[8] = __float2bfloat16(q3); }\n#pragma unroll\n      for (int o = 4; o < 32; o <<= 1) { sa += __shfl_xor_sync(0xffffffffu, sa, o); sb += __shfl_xor_sync(0xffffffffu, sb, o); }\n      if (g == 0) {\n        if (t0 < ntok) atomicAdd(p.ss + lst[t0], sa);\n        if (t0 + 1 < ntok) atomicAdd(p.ss + lst[t0 + 1], sb);\n      }\n    }\n  }\n}\n\n// one group slot (4 x 64-K chunks) of a unit on the consumer side. Per k16 step (CUTLASS mixed-input RS order):\n// wait<3> retires the group that last used this step\'s A / descriptor registers, dequant the RPW blocks, fence, RPW\n// wgmmas, commit. The previous slot is released at step 3 of this slot\'s first chunk, once all its groups retired.\n// FG: the unit\'s first slot (scale-d 0 on its first step, nothing to release).\ntemplate <int N, bool FG>\n__device__ __forceinline__ void group(float (&acc)[RPW][N / 2], uint32_t (&A)[RPW][4][4], uint64_t (&db)[4], int& qs, int& ps_,\n                                      uint32_t sbase, uint32_t full, uint32_t empty, int wid, int lane, long long& wcy) {\n  const int s = qs % NSL;\n#ifdef WSTAT\n  const long long w0 = clock64();\n#endif\n  mbwait(full + s * 8, (qs / NSL) & 1);\n#ifdef WSTAT\n  wcy += clock64() - w0;\n#endif\n  qs++;\n  const uint32_t sb = sbase + s * GSB;\n#pragma unroll\n  for (int cc = 0; cc < WCH; cc++) {\n    const uint32_t xa = sb + cc * XB;\n    uint32_t ww[RPW][4], sA[RPW], sB[RPW];\n#pragma unroll\n    for (int r = 0; r < RPW; r++) {\n      const uint32_t wb = sb + ASB + (((wid >> 3) * RPW + r) * WCH + cc) * WREC + (wid & 7) * REC;\n      const uint4 w = lds128(wb + lane * 16);\n      const uint32_t s16 = lds16(wb + 512 + lane * 2);\n      sA[r] = sc2(s16, 0x4040); sB[r] = sc2(s16, 0x4141);\n      ww[r][0] = w.x; ww[r][1] = w.y; ww[r][2] = w.z; ww[r][3] = w.w;\n    }\n#pragma unroll\n    for (int j = 0; j < 4; j++) {\n      wg_wait<3>();\n      if (!FG && cc == 0 && j == 3) mbarrive(empty + ps_ * 8);  // every consumer lane arrives (count NCW * 32)\n#pragma unroll\n      for (int r = 0; r < RPW; r++) {\n        const uint32_t qq = ww[r][j];\n#if defined(NOCOMP)  // NOCOMP: no dequant, no mma (stream only); NOMMA: real dequant, no mma\n        A[r][j][0] = qq ^ sA[r]; A[r][j][1] = qq ^ sB[r]; A[r][j][2] = qq + sA[r]; A[r][j][3] = qq + sB[r];\n#else\n        A[r][j][0] = bmul(qq & 0x81C081C0u, sA[r]);\n        A[r][j][1] = bmul(((qq << 3) & 0x01C001C0u) | ((qq << 1) & 0x80008000u), sB[r]);\n        A[r][j][2] = bmul((qq << 6) & 0x81C081C0u, sA[r]);\n        A[r][j][3] = bmul(((qq >> 5) & 0x01C001C0u) | ((qq << 5) & 0x80008000u), sB[r]);\n#endif\n      }\n      db[j] = gdesc(xa + j * 32);\n#if defined(NOMMA) || defined(NOCOMP)\n#pragma unroll\n      for (int r = 0; r < RPW; r++) acc[r][j] += __uint_as_float((A[r][j][0] ^ A[r][j][1] ^ A[r][j][2] ^ A[r][j][3] ^ (uint32_t)db[j]) & 0x3fffffffu);\n#else\n#pragma unroll\n      for (int r = 0; r < RPW; r++) fence_acc<N>(acc[r]);\n      wg_fence();\n#pragma unroll\n      for (int r = 0; r < RPW; r++) wg_rs<N>(acc[r], A[r][j], db[j], (FG && cc == 0 && j == 0) ? 0 : 1);\n      wg_commit();\n#endif\n    }\n  }\n  ps_ = s;\n}\n\n#ifdef WXG2\n// WXG2: one wgmma group per 64-K chunk instead of per k16 step. The chunk\'s 4 words are dequantized into one of two A register\n// sets (chunk parity), then fence, 4 x RPW wgmmas, commit, wait_group 1 (retires the previous chunk: its A set is free for the\n// next chunk, which is dequantized while this chunk\'s wgmmas run). One descriptor per slot (+ constant chunk / step offsets),\n// one warpgroup arrive / depbar per chunk: ~6 fewer instructions per word than the per-k16-step order.\ntemplate <int N, bool FG>\n__device__ __forceinline__ void group2(float (&acc)[RPW][N / 2], uint32_t (&A)[2][RPW][4][4], int& qs, int& ps_,\n                                       uint32_t sbase, uint32_t full, uint32_t empty, int wid, int lane, long long& wcy) {\n  const int s = qs % NSL;\n#ifdef WSTAT\n  const long long w0 = clock64();\n#endif\n  mbwait(full + s * 8, (qs / NSL) & 1);\n#ifdef WSTAT\n  wcy += clock64() - w0;\n#endif\n  qs++;\n  const uint32_t sb = sbase + s * GSB;\n  const uint64_t d0 = gdesc(sb);\n#pragma unroll\n  for (int cc = 0; cc < WCH; cc++) {\n    uint32_t (&As)[RPW][4][4] = A[cc & 1];\n#pragma unroll\n    for (int r = 0; r < RPW; r++) {\n      const uint32_t wb = sb + ASB + (((wid >> 3) * RPW + r) * WCH + cc) * WREC + (wid & 7) * REC;\n      const uint4 w = lds128(wb + lane * 16);\n      const uint32_t s16 = lds16(wb + 512 + lane * 2);\n      const uint32_t sA = sc2(s16, 0x4040), sB = sc2(s16, 0x4141);\n      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};\n#pragma unroll\n      for (int j = 0; j < 4; j++) {\n        const uint32_t qq = ww[j];\n#if defined(NOCOMP)\n        As[r][j][0] = qq ^ sA; As[r][j][1] = qq ^ sB; As[r][j][2] = qq + sA; As[r][j][3] = qq + sB;\n#else\n        As[r][j][0] = bmul(qq & 0x81C081C0u, sA);\n        As[r][j][1] = bmul(((qq << 3) & 0x01C001C0u) | ((qq << 1) & 0x80008000u), sB);\n        As[r][j][2] = bmul((qq << 6) & 0x81C081C0u, sA);\n        As[r][j][3] = bmul(((qq >> 5) & 0x01C001C0u) | ((qq << 5) & 0x80008000u), sB);\n#endif\n      }\n    }\n#if defined(NOMMA) || defined(NOCOMP)\n#pragma unroll\n    for (int r = 0; r < RPW; r++)\n#pragma unroll\n      for (int j = 0; j < 4; j++) acc[r][j] += __uint_as_float((As[r][j][0] ^ As[r][j][1] ^ As[r][j][2] ^ As[r][j][3] ^ (uint32_t)gdesc_add(d0, cc * XB + j * 32)) & 0x3fffffffu);\n#else\n#pragma unroll\n    for (int r = 0; r < RPW; r++) fence_acc<N>(acc[r]);\n    wg_fence();\n#pragma unroll\n    for (int j = 0; j < 4; j++)\n#pragma unroll\n      for (int r = 0; r < RPW; r++) wg_rs<N>(acc[r], As[r][j], gdesc_add(d0, cc * XB + j * 32), (FG && cc == 0 && j == 0) ? 0 : 1);\n    wg_commit();\n    wg_wait<1>();\n#endif\n    if (!FG && cc == 0) mbarrive(empty + ps_ * 8);  // the previous slot\'s last chunk has retired (every consumer lane arrives)\n  }\n  ps_ = s;\n}\n#endif\n\ntemplate <int N>\n__device__ __forceinline__ void unit(const P& p, const Sm* sm, const uint16_t* list, uint32_t sbase, uint32_t full, uint32_t empty,\n                                     int& qs, int wid, int lane, const Hdr& h0, long long& wcy) {\n  float acc[RPW][N / 2];\n  uint32_t A[RPW][4][4];\n  uint64_t db[4] = {0, 0, 0, 0};\n  int ps_ = 0;\n#if defined(NOMMA) || defined(NOCOMP)\n#pragma unroll\n  for (int r = 0; r < RPW; r++)\n#pragma unroll\n    for (int i = 0; i < N / 2; i++) acc[r][i] = 0.f;\n#endif\n#ifdef WXG2\n  uint32_t A2[2][RPW][4][4];\n  group2<N, true>(acc, A2, qs, ps_, sbase, full, empty, wid, lane, wcy);\n#pragma unroll 1\n  for (int c4 = 1; c4 < NCH / WCH; c4++) group2<N, false>(acc, A2, qs, ps_, sbase, full, empty, wid, lane, wcy);\n#else\n  group<N, true>(acc, A, db, qs, ps_, sbase, full, empty, wid, lane, wcy);\n#pragma unroll 1\n  for (int c4 = 1; c4 < NCH / WCH; c4++) group<N, false>(acc, A, db, qs, ps_, sbase, full, empty, wid, lane, wcy);\n#endif\n  wg_wait<0>();\n#pragma unroll\n  for (int r = 0; r < RPW; r++) fence_acc<N>(acc[r]);\n  mbarrive(empty + ps_ * 8);\n#ifndef NOCOMP\n#pragma unroll\n  for (int r = 0; r < RPW; r++) epilogue<N>(p, sm, list, acc[r], h0, ((h0.rb * UBL) + (wid >> 3) * RPW + r) * TPU + (wid & 7), lane);\n#else\n  if (acc[0][0] == 1234.5f) p.y[0] = __float2bfloat16(acc[0][1]);\n#endif\n  __syncwarp();\n#if TMA\n  if (!h0.g) asm volatile("fence.proxy.async.global;\\n" ::: "memory");  // the W2 producer reads inter with TMA\n#endif\n  __syncwarp();\n  if (lane == 0) red_rel(p.ctr + (h0.g ? C_W2 : h0.e), 1);\n}\n\n__device__ __forceinline__ void decode(const Sm* sm, int u, int& g, int& e, int& rb, int& ps) {\n  int k = 0;\n  while (sm->segst[k + 1] <= u) k++;\n  const int code = sm->segk[k];\n  g = code >> 6; e = sm->act[code & 63];\n  const int np = sm->np[e];\n  const int v = u - sm->segst[k];\n  ps = v % np; rb = v / np;\n}\n\n__global__ void __launch_bounds__(NTH, 1) moe_x(P p) {\n  extern __shared__ __align__(1024) uint8_t smem[];\n  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, R = p.M * 8;\n  Sm* sm = (Sm*)(smem + RING + 2 * NSL * 8);\n  uint16_t* list = (uint16_t*)(sm + 1);\n  uint8_t* lid = smem;\n  uint16_t* posi = (uint16_t*)(smem + 4096);\n  const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem);\n  const uint32_t full = sbase + RING, empty = full + NSL * 8;\n#ifdef PROF\n  PSET(0, gtime()); PSET(8, clock64());\n#endif\n  if (tid < NSL) { mbinit(full + tid * 8, TMA ? 1 : 32); mbinit(empty + tid * 8, NCW * 32); }\n  if (tid < NLOC) sm->base[tid] = 0;\n#if TMA\n  asm volatile("fence.mbarrier_init.release.cluster;\\n" ::: "memory");\n#endif\n  int u = 0, nx = 0;\n  if (tid == NCW * 32) u = atomicAdd(p.ctr + C_Q, 1);\n  for (int i = tid; i < R; i += NTH) { unsigned e = p.ids[i] - p.lo; lid[i] = e < NLOC ? e : 0xff; }\n  __syncthreads();\n  const unsigned ltm = (1u << lane) - 1;\n  for (int c0 = 0; c0 < R; c0 += NTH) {  // stable per-expert lists (route order), identical in every CTA and in xgather\n    const int i = c0 + tid;\n    const int e = i < R ? lid[i] : 0xff;\n    const unsigned m = __match_any_sync(0xffffffffu, e);\n    sm->wc[wid][lane] = 0;\n    __syncwarp();\n    if (e < NLOC && (m & ltm) == 0) sm->wc[wid][e] = __popc(m);\n    __syncthreads();\n    if (tid < NLOC) { int s = sm->base[tid];\n      for (int w = 0; w < NTH / 32; w++) { const int v = sm->wc[w][tid]; sm->wc[w][tid] = s; s += v; }\n      sm->base[tid] = s; }\n    __syncthreads();\n    if (e < NLOC) posi[i] = sm->wc[wid][e] + __popc(m & ltm);\n    __syncthreads();\n  }\n  if (wid == 0) {\n    const int c = sm->base[lane];\n    int v = c;\n#pragma unroll\n    for (int o = 1; o < 32; o <<= 1) { int n = __shfl_up_sync(0xffffffffu, v, o); if (lane >= o) v += n; }\n    sm->cnt[lane] = c; sm->off[lane] = v - c; sm->np[lane] = (c + TPX - 1) / TPX;\n    const int pd = (c + 7) & ~7;\n    int vp = pd;\n#pragma unroll\n    for (int o = 1; o < 32; o <<= 1) { int n = __shfl_up_sync(0xffffffffu, vp, o); if (lane >= o) vp += n; }\n    sm->npad[lane] = pd; sm->xoff[lane] = (long long)(vp - pd) * NCH * 128;\n    int rk = 0;\n#pragma unroll\n    for (int j = 0; j < 32; j++) { const int cj = __shfl_sync(0xffffffffu, c, j); rk += (cj > c) || (cj == c && j < lane); }\n    if (c > 0) sm->act[rk] = lane;\n    const unsigned am = __ballot_sync(0xffffffffu, c > 0);\n    int w2 = c > 0 ? NRB2 * ((c + TPX - 1) / TPX) * NCW : 0;\n#pragma unroll\n    for (int o = 16; o; o >>= 1) w2 += __shfl_xor_sync(0xffffffffu, w2, o);\n    __syncwarp();\n    if (lane == 0) {\n      const int na = __popc(am);\n      int ns = 0, uu = 0;\n      for (int g = 0; g < 2; g++)  // all W13 units (heaviest expert first), then all W2 units\n        for (int a = 0; a < na; a++) { sm->segst[ns] = uu; sm->segk[ns] = g * 64 + a; uu += (g ? NRB2 : NRB1) * sm->np[sm->act[a]]; ns++; }\n      sm->segst[ns] = uu; sm->nseg = ns; sm->ntot = uu; sm->na = na; sm->nw2 = w2;\n    }\n  }\n  __syncthreads();\n  for (int i = tid; i < R; i += NTH) { const int e = lid[i]; if (e < NLOC) list[sm->off[e] + posi[i]] = i; }\n  __syncthreads();\n#ifdef PROF\n  PSET(1, gtime());\n#endif\n  const int ntot = sm->ntot;\n  long long wcy = 0;  // consumer slot-wait cycles (WSTAT)\n#ifdef WSTAT\n  const long long ct0 = clock64();\n#endif\n  if (wid == NCW) {  // producer warp\n    u = __shfl_sync(0xffffffffu, u, 0);\n    unsigned depok = 0;\n    int sq = 0;\n#ifdef WSTAT\n    long long pw = 0;\n    const long long pt0 = clock64();\n#endif\n    for (; u < ntot;) {\n      int g, e, rb, ps; decode(sm, u, g, e, rb, ps);\n      const int ntok = min(TPX, sm->cnt[e] - ps * TPX), rows = (ntok + 7) & ~7;\n      const uint8_t* wsrc = (g ? p.w2 + ((size_t)e * 32 + rb * UBL) * NCH * WREC : p.w1 + ((size_t)e * 64 + rb * UBL) * NCH * WREC);\n      const uint8_t* xsrc = (g ? p.ig : p.xg) + sm->xoff[e] + (size_t)ps * TPX * 128;\n      const int npad = sm->npad[e];\n      if (g && !(depok >> e & 1)) {  // W2 needs every W13 unit of e (all its inter columns)\n        if (lane == 0) { const int tgt = NRB1 * sm->np[e] * NCW; while (ldacq(p.ctr + e) < tgt) __nanosleep(64); }\n        __syncwarp();\n#if TMA\n        asm volatile("fence.proxy.async.global;\\n" ::: "memory");\n#endif\n        depok |= 1u << e;\n      }\n      if (lane == 0) nx = atomicAdd(p.ctr + C_Q, 1);\n      for (int c = 0; c < NCH; c += WCH, sq++) {\n        const int s = sq % NSL;\n#ifdef WSTAT\n        const long long w0 = clock64();\n#endif\n        if (sq >= NSL) mbwait(empty + s * 8, ((sq / NSL) - 1) & 1);\n#ifdef WSTAT\n        pw += clock64() - w0;\n#endif\n        if (lane == 0) { sm->hdr[s] = Hdr{g, e, rb, ps, c, ntok, (c == 0 ? 1 : 0) | ((!g && rb == 0) ? 4 : 0), 0}; __threadfence_block(); }\n        const uint32_t dst = sbase + s * GSB, fb = full + s * 8;\n#if TMA\n        if (lane == 0) {\n          uint32_t bytes = 0;\n#ifndef NOACT\n          bytes += WCH * rows * 128;\n#endif\n#ifndef NOWGT\n          bytes += WSB;\n#endif\n          expect_tx(fb, bytes);\n#ifndef NOACT\n#pragma unroll\n          for (int cc = 0; cc < WCH; cc++) bulk(dst + cc * XB, xsrc + (size_t)(c + cc) * npad * 128, rows * 128, fb);\n#endif\n#ifndef NOWGT\n#pragma unroll\n          for (int b2 = 0; b2 < UBL; b2++) bulk(dst + ASB + b2 * WCH * WREC, wsrc + ((size_t)b2 * NCH + c) * WREC, WCH * WREC, fb);\n#endif\n        }\n#else\n        for (int cc = 0; cc < WCH; cc++)\n          for (int k = lane; k < rows * 8; k += 32) cp16(dst + cc * XB + k * 16, xsrc + (size_t)(c + cc) * npad * 128 + k * 16);\n        for (int b2 = 0; b2 < UBL; b2++)\n          for (int k = lane; k < WCH * WREC / 16; k += 32) cp16(dst + ASB + b2 * WCH * WREC + k * 16, wsrc + ((size_t)b2 * NCH + c) * WREC + k * 16);\n        cpa_arrive(fb);\n#endif\n      }\n      u = __shfl_sync(0xffffffffu, nx, 0);\n    }\n#ifdef WSTAT\n    if (lane == 0 && p.prof) { atomicAdd(p.prof + 2, (unsigned long long)pw); atomicAdd(p.prof + 3, (unsigned long long)(clock64() - pt0)); }\n#endif\n    const int s = sq % NSL;\n    if (sq >= NSL) mbwait(empty + s * 8, ((sq / NSL) - 1) & 1);\n    if (lane == 0) sm->hdr[s].g = -1;\n#if TMA\n    if (lane == 0) mbarrive(full + s * 8);\n#else\n    cpa_arrive(full + s * 8);\n#endif\n#ifdef PROF\n    PSET(2, gtime());\n#endif\n  } else {  // consumers: two warpgroups, warp wid owns tile wid of every unit\n    int qs = 0;\n    for (;;) {\n      const int s = qs % NSL;\n      mbwait(full + s * 8, (qs / NSL) & 1);\n      const Hdr h = sm->hdr[s];\n      if (h.g < 0) break;\n      if ((h.flags & 4) && wid == 0) for (int i = lane; i < h.ntok; i += 32) p.ss[list[sm->off[h.e] + h.ps * TPX + i]] = 0.f;\n      const int n8 = (h.ntok + 7) >> 3;\n#define U_(n) unit<n>(p, sm, list, sbase, full, empty, qs, wid, lane, h, wcy)\n      if (n8 <= 1) U_(8);\n      else if (n8 <= 2) U_(16);\n#ifdef N8STEP\n      else if (n8 <= 3) U_(24);\n      else if (n8 <= 4) U_(32);\n      else if (n8 <= 6) { if constexpr (TPX >= 64) U_(48); }\n#else\n      else if (n8 <= 4) U_(32);\n#endif\n      else if (n8 <= 8) { if constexpr (TPX >= 64) U_(64); }\n      else if (n8 <= 16) { if constexpr (TPX >= 128) U_(128); }\n      else { if constexpr (TPX >= 256) U_(256); }\n    }\n#ifdef PROF\n    PSET(2, gtime());\n#endif\n  }\n#ifdef WSTAT\n  if (wid < NCW && lane == 0 && p.prof) { atomicAdd(p.prof + 0, (unsigned long long)wcy); atomicAdd(p.prof + 1, (unsigned long long)(clock64() - ct0)); }\n#endif\n  const int nw2 = sm->nw2;\n  if (lane == 0) while (ldacq(p.ctr + C_W2) < nw2) __nanosleep(32);\n  __syncwarp();\n#ifdef PROF\n  PSET(6, gtime());\n#endif\n  for (int it = blockIdx.x * (NTH / 32) + wid; it < p.M * 16; it += gridDim.x * (NTH / 32)) {\n    const int t = it >> 4, col = (it & 15) * 256 + lane * 8;\n    float a[8] = {0, 0, 0, 0, 0, 0, 0, 0};\n#pragma unroll\n    for (int k = 0; k < 8; k++) {\n      const int r = t * 8 + k;\n      unsigned e = p.ids[r] - p.lo;\n      if (e < NLOC) {\n        float cf = p.tw[r] * rsqrtf(__ldcg(p.ss + r) * (1.f / HD) + 1e-8f);\n        uint4 v = __ldcg((const uint4*)(p.y + (size_t)r * HD + col));\n        const __nv_bfloat162* hh = (const __nv_bfloat162*)&v;\n#pragma unroll\n        for (int j = 0; j < 4; j++) { float2 f = __bfloat1622float2(hh[j]); a[2 * j] += cf * f.x; a[2 * j + 1] += cf * f.y; }\n      }\n    }\n    uint4 o; __nv_bfloat162* ob = (__nv_bfloat162*)&o;\n#pragma unroll\n    for (int j = 0; j < 4; j++) ob[j] = __floats2bfloat162_rn(a[2*j]*p.nw[col+2*j], a[2*j+1]*p.nw[col+2*j+1]);\n    *(uint4*)(p.out + (size_t)t * HD + col) = o;\n  }\n#ifdef PROF\n  PSET(7, gtime()); PSET(9, clock64());\n#endif\n  __syncthreads();\n  if (tid == 0) {\n    __threadfence();\n    if (atomicAdd(p.ctr + C_DONE, 1) == (int)gridDim.x - 1) {\n      for (int i = 0; i <= C_DONE; i++) p.ctr[i] = 0;\n      __threadfence();\n    }\n  }\n}\n\nstatic int g_gridX = 0;\nstatic size_t smem_x(int M) { return (size_t)RING + 2 * NSL * 8 + sizeof(Sm) + (size_t)M * 8 * 2 + 1024; }\n\nvoid moe(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor w1, torch::Tensor w2, torch::Tensor g1,\n         torch::Tensor g2, torch::Tensor nw, torch::Tensor xg, torch::Tensor ig, torch::Tensor y, torch::Tensor ss,\n         torch::Tensor ctr, torch::Tensor out, int64_t lo, torch::Tensor prof) {\n  const int M = x.size(0);\n  if (!g_gridX) {\n    int dev; cudaGetDevice(&dev); int mx; cudaDeviceGetAttribute(&mx, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);\n    cudaDeviceGetAttribute(&g_gridX, cudaDevAttrMultiProcessorCount, dev);\n    cudaFuncSetAttribute(moe_x, cudaFuncAttributeMaxDynamicSharedMemorySize, mx);\n  }\n  TORCH_CHECK(M * 8 <= 4096 && M * 8 * 3 + 4096 <= RING, "M too large");\n  auto st = at::cuda::getCurrentCUDAStream();\n  xgather<<<(M * 8 + 7) / 8, 256, 0, st>>>((const __nv_bfloat16*)x.data_ptr(), ids.data_ptr<int>(), M, (int)lo, xg.data_ptr<uint8_t>());\n  P p{w1.data_ptr<uint8_t>(), w2.data_ptr<uint8_t>(), g1.data_ptr<float>(), g2.data_ptr<float>(),\n      xg.data_ptr<uint8_t>(), ig.data_ptr<uint8_t>(), ids.data_ptr<int>(), tw.data_ptr<float>(), nw.data_ptr<float>(),\n      (__nv_bfloat16*)y.data_ptr(), ss.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(), ctr.data_ptr<int>(), M, (int)lo,\n      prof.numel() ? (unsigned long long*)prof.data_ptr() : nullptr};\n  moe_x<<<g_gridX, NTH, smem_x(M), st>>>(p);\n}\n'


# ---- v41: one-pass-per-expert wgmma RS expert GEMM (CUDA_X) --------------------------------------------------------
_modX = {}
WX_CFG = (("-DRPW=2",), ("-DNCW=16",), ("-DNCW=16", "-DN8STEP"))  # 256-row units, N <= 64: 2 warpgroups x 2 m64 blocks, or 4 warpgroups x 1 (4 warps / scheduler)
WX_ABL = ("-DNOCOMP", "-DNOMMA", "-DWSTAT")  # diag (NCW 16): stream only / dequant without mma / wait-cycle counters
WX_MIN = 8      # smallest M bucket that tries it (hot experts need >= 16 tokens for the one-pass dequant to pay)
DIAGW = False   # prepare(): per-load timing of the wx path against the best other candidate + M256 ablations


def _sm90a():
    return ["-gencode=arch=compute_90a,code=sm_90a"] if torch.cuda.get_device_capability() == (9, 0) else []


def _extX(extra=()):
    key = tuple(extra)
    if key not in _modX:
        dfl = [] if _big() else ["-DTPX=16", "-DNSL=2"]
        fl = dfl + list(_DBG) + list(extra)
        _modX[key] = load_inline("moe_strict_wxg2_v52X" + "".join(fl).replace("-", "_").replace("=", ""),
                                 cpp_sources="void moe(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,torch::Tensor);",
                                 cuda_sources=CUDA_X, functions=["moe"], extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"] + _sm90a() + fl, verbose=False)
    return _modX[key]


def _repackX(w):
    """[E, tiles, 16 half-stages, 2304] -> [E, tiles / 8, 64 chunks, 8 tiles, 576]: per 16-row tile and 64-K chunk the
    512 B of mma A-fragment words (lane-major uint4) then the 32 lanes x 2 B block scales (row g, row g + 8)"""
    E, T = w.shape[0], w.shape[1]
    words = w[..., :2048].reshape(E, T, 64, 512)
    sc = w[..., 2048:].reshape(E, T, 16, 32, 4, 2).permute(0, 1, 2, 4, 3, 5).reshape(E, T, 64, 64)
    return torch.cat((words, sc), -1).view(E, T // 8, 8, 64, 576).permute(0, 1, 3, 2, 4).contiguous()


def _wx(data, state, extra=()):
    x = data["hidden_states"].contiguous()
    M = x.shape[0]
    ids = data["topk_ids"].to(torch.int32).contiguous()
    tw = data["topk_weights"].float().contiguous()
    rows = M * 8 + 32 * 7
    xg = torch.empty((rows * 8192,), dtype=torch.uint8, device=x.device)
    ig = torch.empty((rows * 8192,), dtype=torch.uint8, device=x.device)
    y = torch.empty((M * 8, 4096), dtype=torch.bfloat16, device=x.device)
    ss = torch.empty((M * 8,), dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    _extX(extra).moe(x, ids, tw, state["w1x"], state["w2x"], state["g1"], state["g2"], state["nw"], xg, ig, y, ss, state["ctrX"],
                     out, data["config"]["local_start"], state.get("profW", state["prof"]))
    return out


def _diagW(state):
    """per judge load: best other candidate vs wx on fresh judge-style routings (graph replays, cold L2); M256 ablations"""
    dev = state["w1"].device
    flush = torch.empty(2 ** 27, device=dev); sink = torch.empty((), device=dev)

    def tm(cfg, data):
        sr = torch.cuda.Stream(); sr.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(sr):
            for _ in range(2): _run(cfg, data, state)
        torch.cuda.current_stream().wait_stream(sr); torch.cuda.synchronize(); torch.zeros(1, device=dev)  # surfaces a failed launch before capture
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr): _run(cfg, data, state)
        ts = []
        for _ in range(10):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            sink.copy_(flush.sum()); e0.record(); gr.replay(); e1.record(); torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1e3)
        del gr
        return sorted(ts)[len(ts) // 2]
    tb, alt = state.get("tune") or {}, state.get("tune_alt") or {}
    for name, M in (("decode16", 16), ("decode32", 32), ("verify1", 8), ("verify16", 128), ("verify32", 256)):
        b = next((m for m in TUNE_M if m >= M), TUNE_M[-1])
        base = alt.get(b, tb.get(b, _default(M)))
        vs = [("wx", 0, 0, " ".join(c)) if c else ("wx", 0, 0) for c in WX_CFG]
        if M in (32, 128, 256): vs += [("wx", 0, 0, " ".join(("-DNCW=16",) + tuple(x.split()))) for x in WX_ABL if x != "-DWSTAT"]
        acc = {c: [] for c in [base] + vs}
        for sd in range(3):
            data = _synth(M, 5000 + 31 * sd + M, dev)
            for c in acc:
                try: acc[c].append(tm(c, data))
                except Exception as ex: print("WD", name, _nm(c), "failed", str(ex)[:100], flush=True); acc[c].append(float("inf"))
        line = f"WD {name} M{M}: " + "  ".join(f"{_nm(c)} {sum(v) / len(v):.1f}" for c, v in acc.items() if "-DWSTAT" not in _nm(c))
        if M in (32, 128, 256):  # wait fractions of the NCW 16 path (clock64 around the slot waits)
            try:
                st2 = dict(state, profW=torch.zeros(8, dtype=torch.int64, device=dev))
                dw = _synth(M, 5000 + M, dev)
                sink.copy_(flush.sum()); _wx(dw, st2, ("-DNCW=16", "-DWSTAT")); torch.cuda.synchronize()
                st2["profW"].zero_(); sink.copy_(flush.sum()); _wx(dw, st2, ("-DNCW=16", "-DWSTAT")); torch.cuda.synchronize()
                v = st2["profW"].double().cpu().tolist()
                line += f"  | NCW16 consumer waitFull {v[0] / max(v[1], 1):.2f} producer waitEmpty {v[2] / max(v[3], 1):.2f}"
            except Exception as ex:
                line += f"  | wstat failed {str(ex)[:60]}"
        print(line, flush=True)


def prepare(weights):
    global _T0
    _T0 = time.time()
    E, R1 = min(32, weights["w13"].shape[0]), weights["w13"].shape[1]
    I = R1 // 2
    dev = weights["w13"].device
    perm1 = torch.stack((torch.arange(I, device=dev).view(-1, 8), I + torch.arange(I, device=dev).view(-1, 8)), 1).reshape(-1)
    perm2 = torch.arange(weights["w2"].shape[1], device=dev)
    w1 = torch.empty((E, R1 // 16, 16, 2304), dtype=torch.uint8, device=dev)
    w2 = torch.empty((E, weights["w2"].shape[1] // 16, 16, 2304), dtype=torch.uint8, device=dev)
    for e in range(E):
        w1[e] = _repack(weights["w13"][e], weights["w13_sf"][e], perm1)
        w2[e] = _repack(weights["w2"][e], weights["w2_sf"][e], perm2)
    w2f = torch.empty(0,device=dev,dtype=torch.uint8)
    with ThreadPoolExecutor(8) as ex: list(ex.map(lambda f: f(), [lambda: _extL(0)] + [lambda k=k: _extS(k) for k in V17_KSF] + [lambda c=c: _extF(*c) for c in FS_CFG] + [lambda c=c: _extX(c) for c in WX_CFG] + [lambda x=x: _extG(x) for x in G_CFG]  + [lambda x=x: _extK(x) for x in K_CFG] + ([lambda x=x: _extK(x + " -DPROF") for x in K_CFG if x != K_CFG[1]] if DIAGK else []) + ([lambda x=x: _extG(("-DPROF " + x).strip()) for x in G_CFG] if DIAGG else [])
                                                + ([lambda x=x: _extX(("-DNCW=16",) + tuple(x.split())) for x in WX_ABL] if DIAGW else [])
                                                + ([lambda c=c: _extF(c[0], c[1], (c[2] + " -DPROF").strip()) for c in FS_CFG] if DIAGF else [])))
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    st = {"w1": w1, "w2": w2, "g1": weights["w13_gs"].float() * 256.0, "g2": weights["w2_gs"].float() * 256.0,
            "nw": (1.0 + weights["post_norm_w"].float()).to(torch.bfloat16).float(), "ctr": torch.zeros(64, dtype=torch.int32, device=dev),
            "tcnt": torch.zeros(32 * 768, dtype=torch.int32, device=dev),
            "prof": torch.zeros((65536 * 16 if "-DPROF" in _DBG else 0,), dtype=torch.int64, device=dev),
            "ws": torch.empty((32 * 768 * 8 * 64 * 4,), dtype=torch.float32, device=dev),
            "ctrL": torch.zeros(64 + 2 * 32 * 64 * 8, dtype=torch.int32, device=dev), "w2f": w2f,
            "w1x": _repackX(w1), "w2x": _repackX(w2), "ctrX": torch.zeros(64, dtype=torch.int32, device=dev),
            "yacc": torch.zeros((FS_MAX * 8, 4096), dtype=torch.float32, device=dev), "ctrF": torch.zeros(9280 + 64, dtype=torch.int32, device=dev),
            "profF": torch.zeros((4096 if "-DPROF" in _DBG else 0,), dtype=torch.int64, device=dev), "wsL": torch.zeros((KSL_MAX * 8 * 12288,), dtype=torch.float32, device=dev),
          "interG": torch.empty((16, 4096), dtype=torch.bfloat16, device=dev),
          "sspG": torch.zeros((1024 * 16,), dtype=torch.float32, device=dev), "ctrG": torch.zeros(64, dtype=torch.int32, device=dev),
          "interK": torch.empty((256, 4096), dtype=torch.bfloat16, device=dev), "ypK": torch.empty((4 * 256 * 4096,), dtype=torch.float32, device=dev),
          "gpK": torch.empty((2 * 256 * 4096 * 2,), dtype=torch.float32, device=dev), "sspK": torch.empty((256 * 256,), dtype=torch.float32, device=dev), "ctrK": torch.zeros(64, dtype=torch.int32, device=dev),
          "scrK": torch.empty((torch.cuda.get_device_properties(dev).multi_processor_count * 160 * 1024 if torch.cuda.get_device_capability(dev)[0] < 9 else 16,), dtype=torch.uint8, device=dev)}
    if DIAG:
        _diag(st, weights)
    el = time.time() - _T0
    print("v48 prepare %.0fs before tuning" % el, flush=True)
    if TUNE and FORCE is None and el < 200:
        st["tune"], st["tune_alt"] = _tune(st)
    if DIAGW and time.time() - _T0 < 250:
        _diagW(st)
    if DIAGF and time.time() - _T0 < 250:
        _diagF(st)
    if DIAGG and time.time() - _T0 < 280:
        _diagG(st)
    if DIAGK and time.time() - _T0 < 280:
        _diagK(st)
    return st


def _diagF(state):
    dev = state["w1"].device
    fl = torch.ones(2 ** 27, device=dev); sink = torch.empty((), device=dev)
    st = dict(state, profF=torch.zeros(4096, dtype=torch.int64, device=dev))
    wpc = 16 if _big() else 8
    for M in (1, 4, 8, 16, 32):
        data = _synth(M, 900 + M, dev)
        ids = data["topk_ids"]; l = ids[ids < 32]; c = torch.bincount(l.long(), minlength=32)
        for cfg in FS_CFG:
            ext = _extF(cfg[0], cfg[1], (cfg[2] + " -DPROF").strip())
            x = data["hidden_states"]; out = torch.empty_like(x)
            for _ in range(3):
                st["profF"].zero_(); sink.copy_(fl.sum())
                ext.moe(x, ids, data["topk_weights"], st["w1"], st["w2f"], st["g1"], st["g2"], st["nw"], st["yacc"], st["ctrF"], out, 0, st["profF"])
            torch.cuda.synchronize()
            P = st["profF"].view(-1, 16).cpu()
            n = int((P[:, 0] > 0).sum()); P = P[:n].double(); t0 = P[:, 0].min()
            q = lambda k, f: float((P[:, k] - t0).quantile(f)) / 1e3
            mx = lambda k: float((P[:, k] - t0).max()) / 1e3
            dA = (P[:, 2] - P[:, 1]).clamp(min=1); dB = (P[:, 3] - P[:, 2]).clamp(min=1)
            wA = float((P[:, 8] / (wpc * dA)).median()); wB = float((P[:, 9] / (wpc * dB)).median())
            cmp = P[:, 11].long(); n1 = cmp % 1000; n2 = (cmp // 1000) % 1000; n4 = cmp // 1000000
            cls = []
            for nm, sel in (("nt1", (n1 > 0) & (n2 == 0) & (n4 == 0)), ("nt2", (n2 > 0) & (n1 == 0) & (n4 == 0)), ("nt4", (n4 > 0) & (n1 == 0) & (n2 == 0))):
                if sel.any():
                    g = (n1 + n2 + n4)[sel].double()
                    cls.append(f"{nm} n{int(sel.sum())} A/gr {float((dA[sel] / g).median()) / 1e3:.2f} B/gr {float((dB[sel] / g).median()) / 1e3:.2f}")
            print(f"FD M{M} na {int((c > 0).sum())} maxT {int(c.max())} {_nm(('fs',) + cfg)}: pro {q(1, .5):.1f}/{q(1, 1):.1f} A {q(2, .1):.1f}/{q(2, .5):.1f}/{q(2, 1):.1f} "
                  f"B {q(3, .1):.1f}/{q(3, .5):.1f}/{q(3, 1):.1f} bar {mx(4):.1f} x6 {mx(6):.1f} x7 {mx(7):.1f} end {mx(5):.1f} waitA {wA:.2f} waitB {wB:.2f} ks/S {int(P[0, 12])} | "
                  + " | ".join(cls), flush=True)


DIAG = False  # diagnostics only: print a per-warp phase timeline from prepare() (untimed)


def _diag(state, weights):
    global _mod, _DBG
    keep, _mod, _DBG = (_mod, _DBG), None, _DBG + ["-DPROF"]
    ext = _ext()
    state = dict(state, prof=torch.zeros((65536 * 16,), dtype=torch.int64, device="cuda"))
    nw = torch.cuda.get_device_properties(0).multi_processor_count * (16 if _big() else 8)
    g = torch.Generator(device="cuda").manual_seed(7)
    for bs in (1, 8, 32, 256):
        sc = torch.rand((bs, 256), generator=g, device="cuda")
        ids = sc.topk(8, -1).indices
        pr = torch.bincount((ids // 32).flatten(), minlength=8)
        ids = ((ids - 32 * pr.argmax()) % 256).int().contiguous()
        x = torch.randn((bs, 4096), generator=g, device="cuda").bfloat16()
        tw = torch.full((bs, 8), 0.125, device="cuda")
        fl = torch.empty(2**28, device="cuda")
        for _ in range(3):
            fl.add_(1)
            inter = torch.empty((bs * 8, 4096), dtype=torch.bfloat16, device="cuda"); y = torch.empty_like(inter)
            ss = torch.empty((bs * 8,), device="cuda"); out = torch.empty_like(x)
            ext.moe(x, ids, tw, state["w1"], state["w2"], state["g1"], state["g2"], state["nw"], inter, y, ss, state["ctr"],
                    state["tcnt"], state["ws"], out, 0, state["prof"])
        torch.cuda.synchronize()
        P = state["prof"].view(-1, 16)[:nw].cpu().tolist()
        t0 = min(r[0] for r in P)
        q = lambda v, f: sorted(v)[min(len(v) - 1, int(f * len(v)))]
        col = lambda k: [(r[k] - t0) / 1e3 for r in P]
        mhz = [(r[14] - r[13]) / max(1, r[7] - r[0]) * 1e3 for r in P]
        cy = lambda k: [r[k] / q(mhz, .5) for r in P]
        print(f"DIAG bs {bs} na {P[0][11]} mode {P[0][12]} span {max(col(7)):.1f} routed {max(col(1)):.1f} "
              f"done p10/50/max {q(col(5), .1):.1f}/{q(col(5), .5):.1f}/{max(col(5)):.1f} comb {max(col(6)):.1f} mhz {q(mhz, .5):.0f} "
              f"g1 {q(cy(2), .5):.1f}/{max(cy(2)):.1f} g2 {q(cy(3), .5):.1f}/{max(cy(3)):.1f} dep {q(cy(4), .5):.1f}/{max(cy(4)):.1f} "
              f"tma {q(cy(9), .5):.1f}/{max(cy(9)):.1f} segs {max(r[8] for r in P)}", flush=True)
    _mod, _DBG = keep


def _kernel_small(data, state):
    x = data["hidden_states"].contiguous()
    M = x.shape[0]
    ids = data["topk_ids"].to(torch.int32).contiguous()
    tw = data["topk_weights"].float().contiguous()
    inter = torch.empty((M * 8, 4096), dtype=torch.bfloat16, device=x.device)
    y = torch.empty((M * 8, 4096), dtype=torch.bfloat16, device=x.device)
    ss = torch.empty((M * 8,), dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    _ext(LARGE and M > 64).moe(x, ids, tw, state["w1"], state["w2"], state["g1"], state["g2"], state["nw"], inter, y, ss, state["ctr"],
                               state["tcnt"], state["ws"], out, data["config"]["local_start"], state["prof"])
    return out


import time
from concurrent.futures import ThreadPoolExecutor
_modL = {}
TUNE_M = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)
JUDGE_M = (1, 8, 16, 32, 128, 256)  # the timed loads' M
KSL_MAX = 64  # ring K-split workspace covers M <= 64
TUNE = True
V17_KSF = (0,)  # v17 path: 0 = its own K-split rule (4/2/1 by tiles per warp group), else forced
FORCE = None  # e.g. ("ring", 2, 1) or ("v17", 0, 0): skip the tuned table


def _extL(nx1=0):
    if nx1 not in _modL:
        fl = (["-DRPW=2", "-DNSLOT=4"] if _big() else ["-DRPW=1", "-DNSLOT=2"]) + (["-DNX1"] if nx1 else []) + list(_DBG)
        _modL[nx1] = load_inline("moe_gridconstant_v30L" + "".join(fl).replace("-", "_").replace("=", ""),
                                 cpp_sources="void moe(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,torch::Tensor);",
                                 cuda_sources=CUDA_L, functions=["moe"], extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"] + fl, verbose=False)
    return _modL[nx1]


def _ring(data, state, ks, nx1):
    x = data["hidden_states"].contiguous()
    M = x.shape[0]
    ids = data["topk_ids"].to(torch.int32).contiguous()
    tw = data["topk_weights"].float().contiguous()
    inter = torch.empty((M * 8, 4096), dtype=torch.bfloat16, device=x.device)
    y = torch.empty((M * 8, 4096), dtype=torch.bfloat16, device=x.device)
    ss = torch.empty((M * 8,), dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    _extL(nx1).moe(x, ids, tw, state["w1"], state["w2"], state["g1"], state["g2"], state["nw"], inter, y, ss, state["ctrL"],
                   state["wsL"], out, data["config"]["local_start"], ks if M <= KSL_MAX else 1, state["prof"])
    return out


_modS = {}


def _extS(ksf):
    if not ksf: return _ext()
    if ksf not in _modS:
        wpc, stg = (16, 2) if _big() else (8, 2)
        _modS[ksf] = load_inline("moe_gridconstant_v30S_%d_%d_k%d" % (wpc, stg, ksf),
                                 cpp_sources="void moe(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,torch::Tensor);",
                                 cuda_sources=CUDA, functions=["moe"],
                                 extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo", "-DWPC=%d" % wpc, "-DNSTG=%d" % stg, "-DKSF=%d" % ksf] + list(_DBG), verbose=False)
    return _modS[ksf]


def _small(data, state, ksf):
    x = data["hidden_states"].contiguous()
    M = x.shape[0]
    ids = data["topk_ids"].to(torch.int32).contiguous()
    tw = data["topk_weights"].float().contiguous()
    inter = torch.empty((M * 8, 4096), dtype=torch.bfloat16, device=x.device)
    y = torch.empty((M * 8, 4096), dtype=torch.bfloat16, device=x.device)
    ss = torch.empty((M * 8,), dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    _extS(ksf).moe(x, ids, tw, state["w1"], state["w2"], state["g1"], state["g2"], state["nw"], inter, y, ss, state["ctr"],
                   state["tcnt"], state["ws"], out, data["config"]["local_start"], state["prof"])
    return out


def _run(cfg, data, state):
    if cfg[0] == "wx":
        return _wx(data, state, tuple(cfg[3].replace("+", " ").split()) if len(cfg) > 3 else ())
    if cfg[0] == "fs":
        return _fs(data, state, cfg)
    if cfg[0] == "g1":
        return _g1(data, state, cfg)
    if cfg[0] == "gk":
        return _gk(data, state, cfg)
    return _small(data, state, cfg[1]) if cfg[0] == "v17" else _ring(data, state, cfg[1], cfg[2])


def _default(M):
    return ("v17", 0, 0) if M <= 64 else ("ring", 1, 0)


def _cands(M):
    c = [("v17", k, 0) for k in V17_KSF] if M <= 64 else []
    if M <= FS_MAX: c += [("fs",) + f for f in FS_CFG]
    if M <= GS_MAX: c += [("g1", 0, 0, x) for x in G_CFG]
    c += [("gk", 0, 0, x) for x in K_CFG if GK_MIN <= M <= _gkmax(x) and (_gkmax(x) <= 16 or M >= K32_MIN)]
    for nx in (0, 1):
        for ks in ((1, 2, 4, 8) if M <= KSL_MAX else (1,)):
            c.append(("ring", ks, nx))
    if M >= WX_MIN and torch.cuda.get_device_capability()[0] >= 9:  # wgmma path: Hopper only (its smem exceeds pre-Hopper limits)
        c += [("wx", 0, 0, " ".join(x)) if x else ("wx", 0, 0) for x in WX_CFG]
    return c


def _synth(M, seed, dev):
    """busiest-GPU top-8 routing with heavy-tailed expert popularity (the judge's soft routing shape), random activations"""
    g = torch.Generator(device=dev).manual_seed(seed)
    sc = torch.rand((M, 256), generator=g, device=dev)
    z = torch.randn((256,), generator=g, device=dev)
    u = torch.rand((256,), generator=g, device=dev).clamp(min=1e-12)
    r = (sc + 0.03 * z / (-u.log()).sqrt()).topk(8, dim=-1)
    ids = r.indices
    ids = (ids - 32 * torch.bincount((ids // 32).flatten(), minlength=8).argmax()) % 256
    return {"hidden_states": torch.randn((M, 4096), generator=g, device=dev).to(torch.bfloat16), "topk_ids": ids.to(torch.int32),
            "topk_weights": (r.values / r.values.sum(-1, keepdim=True)).float(), "config": {"local_start": 0}}


def _nm(c):
    if c[0] == "g1": return "g1" + ("".join(c[3].split()) if len(c) > 3 else "")
    if c[0] == "gk": return "gk" + "".join(c[3].replace("-D", "").split()) if len(c) > 3 else "gk"
    if c[0] == "wx": return "wx" + ("".join(c[3].split()) if len(c) > 3 else "")
    if c[0] == "fs":
        x = c[3] if len(c) > 3 else ""
        return f"f{c[1]}s{c[2]}" + ("w8" if "-DWPC=8" in x else "") + ("L" if "-DLASTA" in x else "") + ("P" if "-DMT2" in x else "") + ("d" if "-DPD1=1" in x or "-DFD1=3" in x else "") + ("m" + x.split("-DSMREQ=")[1].split()[0] if "-DSMREQ=" in x else "")
    return f"{c[0][0]}{c[1]}{'n' if c[2] else ''}"


def _tune(state):
    dev = state["w1"].device
    flush = torch.empty(2 ** 27, device=dev); sink = torch.empty((), device=dev)
    table, alt, lines = {}, {}, []
    for M in TUNE_M:
        res = {}
        if time.time() - _T0 > 270:  # stay inside the 300 s prepare budget: untuned buckets keep the default
            lines.append(f"M{M}:skipped"); continue
        seeds = (11, 12, 13) if M in JUDGE_M and time.time() - _T0 < 200 else (11, 12)  # judge-generator routings per bucket
        for seed in seeds:
            data = _synth(M, seed + M, dev)
            for cfg in _cands(M):
                try:
                    sr = torch.cuda.Stream(); sr.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(sr):
                        for _ in range(2): _run(cfg, data, state)
                    torch.cuda.current_stream().wait_stream(sr); torch.cuda.synchronize(); torch.zeros(1, device=dev)  # surfaces a failed launch before capture
                    gr = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(gr): _run(cfg, data, state)
                    ts = []
                    for _ in range(12):
                        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                        sink.copy_(flush.sum()); e0.record(); gr.replay(); e1.record(); torch.cuda.synchronize()
                        ts.append(e0.elapsed_time(e1) * 1e3)
                    res[cfg] = res.get(cfg, 0.0) + sorted(ts)[len(ts) // 2]
                    del gr
                except Exception as ex:
                    res[cfg] = float("inf"); print("tune", M, cfg, "failed", ex, flush=True)
        d = _default(M)
        best = min(res, key=res.get)
        if res[best] > 0.98 * res.get(d, float("inf")): best = d
        table[M] = best
        ra = {c: v for c, v in res.items() if c[0] != "wx"}
        alt[M] = min(ra, key=ra.get) if ra else d
        lines.append(f"M{M}:" + ",".join(f"{_nm(c)}={res[c] / len(seeds):.0f}" for c in res) + f"->{_nm(best)}")
    print("v48 tune " + " | ".join(lines), flush=True)
    torch.cuda.empty_cache()
    return table, alt


def kernel_fn(data, state):
    M = data["hidden_states"].shape[0]
    if FORCE is not None and (FORCE[0] not in ("fs", "g1", "gk") or (M <= {"fs": FS_MAX, "g1": GS_MAX, "gk": _gkmax(FORCE[3] if len(FORCE) > 3 else "")}[FORCE[0]] and data["config"]["local_start"] == 0)): return _run(FORCE, data, state)
    tb = state.get("tune") or {}
    b = next((m for m in TUNE_M if m >= M), TUNE_M[-1])
    cfg = tb.get(b, _default(M))
    if cfg[0] == "fs" and (M > FS_MAX or data["config"]["local_start"] != 0): cfg = _default(M)
    if cfg[0] == "wx" and data["config"]["local_start"] != 0: cfg = _default(M)
    if cfg[0] == "g1" and (M > GS_MAX or data["config"]["local_start"] != 0): cfg = _default(M)
    if cfg[0] == "gk" and (M > _gkmax(cfg[3] if len(cfg) > 3 else "") or data["config"]["local_start"] != 0): cfg = _default(M)
    return _run(cfg, data, state)

# Lean, exact-boundary research dispatch; EP8 local_start=0 contract.
TUNE=False
GS_MAX=1
G_CFG=("-DFENCE1 -DRT=16",)
K_CFG=("-DSW=4 -DKS2=2",)
WX_CFG=()
FS_CFG=()
V17_KSF=(0,)

def kernel_fn(data,state):
    if data["config"]["local_start"] != 0 or data["config"].get("local_count",32) != 32:
        raise ValueError("This research backend is scoped to EP8 local bank zero")
    m=data["hidden_states"].shape[0]
    if m==1:return _g1(data,state,("g1",0,0,"-DFENCE1 -DRT=16"))
    if m<=16:return _gk(data,state,("gk",0,0,"-DSW=4 -DKS2=2"))
    return _run(_default(m),data,state)

# Research dispatch only; final interval requires paired evidence on held-nearby shapes.
# Source: lee101/openkernels47103fd, submission_v52.py CUDA_X, strict boundary edits above.
WX_CFG=(("-DRPW=2", "-DWXG2"),)
WX_RESEARCH_CONFIG=("-DRPW=2", "-DWXG2")
_strict_grid_fallback=kernel_fn

def kernel_fn(data,state):
    m=data["hidden_states"].shape[0]
    if 32<=m<=512:
        return _wx(data,state,WX_RESEARCH_CONFIG)
    return _strict_grid_fallback(data,state)
