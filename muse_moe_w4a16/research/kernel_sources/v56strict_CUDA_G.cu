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

// v45 "g1": decode GEMV for M <= 2 (<= 2 tokens per expert). Every warp owns one 256-K slice of every weight row the
// CTA touches, so the activation slice lives in registers and nothing is split per round across warps except a final
// 16-way smem reduction. W13: the CTA's contiguous range of 16-row tiles over all local experts -> SwiGLU -> inter
// (global). Grid barrier (the ring keeps streaming W2 meanwhile). W2: the CTA's contiguous range of 16-row tiles of the
// output, for every local expert, full K (W2 in the same 16-row tile / 256-K half-stage layout as W13) -> y rows in smem, per-route sum of squares -> grid barrier -> norm + combine
// of the CTA's rows. No atomics on y, no accumulator to clean.
#define HD 4096
#define NLOC 32
#define STG 4608
#define PC 2304
#define NW 16
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
#define SMALLSM 1  // sm_80/86 emulation: 99 KB of smem per block
#endif
#ifdef P1
#define PPS 1  // one 2304 B piece (one TMA box, one barrier) per ring slot
#else
#define PPS 2  // two pieces (two tiles' slices) per ring slot
#endif
#define SLOT (PPS * PC)
#ifndef SG
#ifdef SMALLSM
#define SG (2 / PPS)
#else
#define SG (4 / PPS)   // ring slots per warp (16 x 4 x 2304 B = 147 KB in flight per SM)
#endif
#endif
#ifdef SMALLSM
#undef RT
#define RT 2   // 99 KB of smem: small rounds whatever the candidate asks for
#endif
#ifndef RT
#define RT 8   // pieces per reduction round
#endif
#define MAXR 16
#define MAXRT 4
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define TMA 1
#else
#define TMA 0
#endif

struct P {
  const uint8_t* w1; const uint8_t* w2; const float* g1; const float* g2; const __nv_bfloat16* x; const int* ids;
  const float* tw; const float* nw; __nv_bfloat16* inter; float* ssp; __nv_bfloat16* out; int* ctr; int M, lo, smem;
  unsigned long long* prof;
};
struct Sm {
  int act[NLOC], aoff[NLOC + 1];
  int na, nr, t0, n13, r0, nrt, n2, ns13, ns2, target, stage;
  int rexp[MAXR], rid[MAXR]; float coef[MAXR], cf[MAXR], g2r[MAXR];
};

__device__ __forceinline__ uint4 lds(uint32_t a) { uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v; }
__device__ __forceinline__ uint2 lds64(uint32_t a) { uint2 v; asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\n" : "=r"(v.x), "=r"(v.y) : "r"(a)); return v; }
__device__ __forceinline__ uint32_t lds16(uint32_t a) { unsigned short v; asm volatile("ld.shared.u16 %0, [%1];\n" : "=h"(v) : "r"(a)); return v; }
__device__ __forceinline__ void cp16(uint32_t s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(g) : "memory"); }
__device__ __forceinline__ uint32_t bmul(uint32_t a, uint32_t b) { uint32_t d; asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u)); return d; }
__device__ __forceinline__ uint32_t sc2(uint32_t w, uint32_t sel) { return __byte_perm(w, 0, sel) * 16u + 0x77007700u; }
__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
#ifdef NOMMA
  c[0] += __uint_as_float(a0 ^ a1 ^ a2 ^ a3 ^ b0 ^ b1);
#else
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}
__device__ __forceinline__ int ldacq(const int* p) { int v; asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void mb_init(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(b), "r"(n)); }
__device__ __forceinline__ void mb_wait(uint32_t bar, uint32_t ph) {
#if TMA
  asm volatile("{\n.reg .pred p;\nW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" :: "r"(bar), "r"(ph) : "memory");
#else
  asm volatile("{\n.reg .pred p;\nW: mbarrier.test_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" :: "r"(bar), "r"(ph) : "memory");
#endif
}
#ifdef PROF
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
#define PST(k) do { if (threadIdx.x == 0 && p.prof) p.prof[blockIdx.x * 16 + (k)] = gtime(); } while (0)
#else
#define PST(k)
#endif
#ifdef DQHI
#define SHR5(q) __umulhi((q), 1u << 27)
#else
#define SHR5(q) ((q) >> 5)
#endif
// dequant one word of 8 e2m1 codes into an m16n8k16 A fragment (scale pair sA rows g, sB rows g + 8)
#define DQ(q, sA, sB, a0, a1, a2, a3) \
  const uint32_t a0 = bmul((q) & 0x81C081C0u, sA); \
  const uint32_t a1 = bmul(__funnelshift_l((q), (q), 10) & 0x81C081C0u, sB); \
  const uint32_t a2 = bmul(__funnelshift_l((q), (q), 13) & 0x81C081C0u, sA); \
  const uint32_t a3 = bmul((((q) & 0x001B001Bu) * 0x1084u) & 0x81C081C0u, sB);

// one 2304 B piece: 16 rows x this warp's 256-K slice (4 x 64-K chunks; lane t holds 16 contiguous K per chunk = one
// NVFP4 block, so one scale byte pair per chunk). B: the activation slice in mma B-fragment order.
__device__ __forceinline__ void piece(uint32_t pb, int lane, const uint4 (&B)[4][2], float (&acc)[2][4]) {
  const uint2 sh = lds64(pb + 2048 + lane * 8);
#pragma unroll
  for (int c = 0; c < 4; c++) {
    const uint4 w = lds(pb + c * 512 + lane * 16);
    const uint32_t sw = c < 2 ? sh.x : sh.y;
    const uint32_t sA = sc2(sw, (c & 1) ? 0x4242 : 0x4040), sB = sc2(sw, (c & 1) ? 0x4343 : 0x4141);
    const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int j = 0; j < 4; j++) {
      DQ(ww[j], sA, sB, a0, a1, a2, a3)
      const uint32_t* bb = (const uint32_t*)&B[c][j >> 1];
      mma(acc[j & 1], a0, a1, a2, a3, bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);
    }
  }
}

struct Ring {
  const P* p; Sm* sm; uint32_t ring, bars; int lane, wid; unsigned long long wns;
  // piece sequence of this warp: W13 slots (2 tiles' 256-K slices each), then W2 slots (2 (expert, row tile) slices)
  __device__ __forceinline__ void copy(uint32_t dst, uint32_t bar, const uint8_t* src) {
#if TMA
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(dst), "l"(src), "n"(PC), "r"(bar) : "memory");
#else
    for (int k = lane; k < PC / 16; k += 32) cp16(dst + k * 16, src + k * 16);
#endif
  }
  __device__ __forceinline__ void issue(int q) {
    const int ns13 = sm->ns13;
    if (q >= ns13 + sm->ns2) return;
    const uint32_t slot = q % SG, dst = ring + slot * SLOT, bar = bars + slot * 8;
    const uint8_t* s0; const uint8_t* s1 = nullptr;
    if (q < ns13) {
      const int tt = sm->t0 + PPS * q, k = wid * PC;
      s0 = p->w1 + ((size_t)(p->lo + sm->act[tt >> 9]) * 512 + (tt & 511)) * 8 * STG + k;
      if (PPS == 2 && 2 * q + 1 < sm->n13) s1 = p->w1 + ((size_t)(p->lo + sm->act[(tt + 1) >> 9]) * 512 + ((tt + 1) & 511)) * 8 * STG + k;
    } else {
      const int j = PPS * (q - ns13), nrt = sm->nrt;
      int a = j / nrt, rt = sm->r0 + j - a * nrt;
      s0 = p->w2 + (((size_t)(p->lo + sm->act[a]) * 256 + rt) * 16 + wid) * PC;
      if (PPS == 2 && j + 1 < sm->n2) { if (++rt == sm->r0 + nrt) { rt = sm->r0; a++; } s1 = p->w2 + (((size_t)(p->lo + sm->act[a]) * 256 + rt) * 16 + wid) * PC; }
    }
#ifdef NOLOAD  // diag: consumers run on stale ring contents (compute-only rate)
#if TMA
    if (lane == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(bar) : "memory");
#else
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(bar) : "memory");
#endif
    return;
#endif
#if TMA
    if (lane == 0) {
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "r"(s1 ? 2 * PC : PC) : "memory");
      copy(dst, bar, s0);
      if (s1) copy(dst + PC, bar, s1);
    }
#else
    copy(dst, bar, s0);
    if (s1) copy(dst + PC, bar, s1);
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" :: "r"(bar) : "memory");
#endif
  }
  __device__ __forceinline__ uint32_t acquire(int q) {
    const int slot = q % SG;
#ifdef PROF
    const unsigned long long w0 = gtime();
#endif
    mb_wait(bars + slot * 8, (q / SG) & 1);
    __syncwarp();
#ifdef PROF
    wns += gtime() - w0;
#endif
    return ring + slot * SLOT;
  }
  __device__ __forceinline__ void release(int q) { __syncwarp(); issue(q + SG); }
};

__global__ void __launch_bounds__(NW * 32, 1) moe_g(P p) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, g = lane >> 2, t = lane & 3, R = p.M * 8, G = gridDim.x;
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem);
  Sm* sm = (Sm*)(smem + NW * SG * 8);
  const int xo = (NW * SG * 8 + (int)sizeof(Sm) + 127) & ~127;
  __nv_bfloat16* xs = (__nv_bfloat16*)(smem + xo);
  float4* red = (float4*)(smem + xo + p.M * HD * 2);
  float* yb = (float*)(smem + xo + p.M * HD * 2 + RT * NW * 8 * 16);
  const int ro = (xo + p.M * HD * 2 + RT * NW * 8 * 16 + MAXR * MAXRT * 16 * 4 + 127) & ~127;
  const int io = ro + NW * SG * SLOT;
  PST(0);
  if (lane < SG) mb_init(sb + (wid * SG + lane) * 8, TMA ? 1 : 32);
#if TMA
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
  if (wid == 0) {  // routing in one warp (R <= 16 routes): experts ascending, routes of an expert in route order
    int e = -1;
    if (lane < R) { const unsigned u = p.ids[lane] - p.lo; e = u < NLOC ? (int)u : -1; }
    const unsigned am = __reduce_or_sync(0xffffffffu, e >= 0 ? 1u << e : 0u);
    const int na = __popc(am), nr = __popc(__ballot_sync(0xffffffffu, e >= 0));
    int k = 0;
    for (int j = 0; j < R; j++) { const int ej = __shfl_sync(0xffffffffu, e, j); k += ej >= 0 && ej < e; }
    const unsigned peers = __match_any_sync(0xffffffffu, e);
    const int rank = __popc(peers & ((1u << lane) - 1));
    if (e >= 0) {
      const int a = __popc(am & ((1u << e) - 1)), r = k + rank;
      sm->rid[r] = lane; sm->rexp[r] = a;
      if (rank == 0) { sm->act[a] = e; sm->aoff[a] = k; }
    }
    if (lane == 0) {
    sm->aoff[na] = nr; sm->na = na; sm->nr = nr;
    const int T13 = na * 512, b = blockIdx.x;
    sm->t0 = (int)((long long)b * T13 / G); sm->n13 = (int)((long long)(b + 1) * T13 / G) - sm->t0;
    sm->r0 = b * 256 / G; sm->nrt = (b + 1) * 256 / G - sm->r0;
    sm->n2 = na * sm->nrt; sm->ns13 = (sm->n13 + PPS - 1) / PPS; sm->ns2 = (sm->n2 + PPS - 1) / PPS;
    sm->stage = io + nr * HD * 2 <= p.smem;
    if (sm->nrt > MAXRT || io > p.smem || G > 256) asm volatile("trap;");
    }
  }
  __syncthreads();
  Ring C; C.p = &p; C.sm = sm; C.lane = lane; C.wid = wid; C.wns = 0;
  C.ring = sb + ro + wid * SG * SLOT; C.bars = sb + wid * SG * 8;
#pragma unroll
  for (int i = 0; i < SG; i++) C.issue(i);
  PST(1);
  const int na = sm->na, nr = sm->nr, n13 = sm->n13, n2 = sm->n2, ns13 = sm->ns13;
  for (int i = tid; i < p.M * HD / 8; i += NW * 32) ((uint4*)xs)[i] = ((const uint4*)p.x)[i];
  if (tid < nr) { const float g2 = p.g2[p.lo + sm->act[sm->rexp[tid]]]; sm->g2r[tid] = g2; sm->cf[tid] = p.tw[sm->rid[tid]]; }  // norm factors, off the tail
  __syncthreads();
  int q = 0;
  unsigned long long wsy = 0;
#ifdef PROF
#define SYNCW() do { const unsigned long long t_ = gtime(); __syncthreads(); wsy += gtime() - t_; } while (0)
  if (tid == 0 && p.prof) p.prof[blockIdx.x * 16 + 11] = clock64();
#else
#define SYNCW() __syncthreads()
#endif
  if (na) {
    // ---- W13: warp slice K [256 wid, +256); B (activation) fragments in registers, reloaded per expert
    uint4 B[4][2];
    int curA = -1;
    for (int rd = 0; rd * RT < n13; rd++) {
      const int pend = min(n13, (rd + 1) * RT);
      for (; PPS * q < pend; q++) {
        const uint32_t sbase = C.acquire(q);
#pragma unroll
        for (int pp = 0; pp < PPS; pp++) {
          const int pc = PPS * q + pp;
          if (pc < n13) {
            const int a = (sm->t0 + pc) >> 9;
            if (a != curA) {
              curA = a;
              const int T = sm->aoff[a + 1] - sm->aoff[a];
              const int row = g < T ? (sm->rid[sm->aoff[a] + g] >> 3) : -1;
#pragma unroll
              for (int c = 0; c < 4; c++)
#pragma unroll
                for (int h = 0; h < 2; h++)
                  B[c][h] = row >= 0 ? ((const uint4*)(xs + row * HD + wid * 256 + c * 64 + t * 16))[h] : make_uint4(0, 0, 0, 0);
            }
            float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
            piece(sbase + pp * PC, lane, B, acc);
            if (t == 0) red[((pc - rd * RT) * NW + wid) * 8 + g] = make_float4(acc[0][0] + acc[1][0], acc[0][1] + acc[1][1], acc[0][2] + acc[1][2], acc[0][3] + acc[1][3]);
          }
        }
        C.release(q);
      }
      SYNCW();
      if (tid < (pend - rd * RT) * 8) {
        const int rp = tid >> 3, gg = tid & 7;
        float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int w = 0; w < NW; w++) { const float4 v = red[(rp * NW + w) * 8 + gg]; s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w; }
        const int tt = sm->t0 + rd * RT + rp, a = tt >> 9, col = (tt & 511) * 8 + gg, T = sm->aoff[a + 1] - sm->aoff[a];
        const float gs = p.g1[p.lo + sm->act[a]];
#pragma unroll
        for (int n = 0; n < 2; n++)
          if (n < T) {
            const float gt = (n ? s.y : s.x) * gs, up = (n ? s.w : s.z) * gs;
            p.inter[(size_t)(sm->aoff[a] + n) * HD + col] = __float2bfloat16(muse_swiglu_bf16(gt, up));
          }
#ifndef FENCE1
        __threadfence();
#endif
      }
      SYNCW();
    }
    PST(2);
#ifdef PROF
    if (tid == 0 && p.prof) p.prof[blockIdx.x * 16 + 12] = clock64();
    if (lane == 0 && p.prof) atomicAdd(p.prof + blockIdx.x * 16 + 10, wsy);
    if (lane == 0 && p.prof) atomicAdd(p.prof + blockIdx.x * 16 + 8, C.wns);
    C.wns = 0;
#endif
    // ---- grid barrier 1: every tile of inter is written (the ring is already streaming W2)
    if (tid == 0) {
      int old;
#ifdef FENCE1
      __threadfence();  // after the __syncthreads: orders the CTA's inter stores before the release
#endif
      asm volatile("atom.add.release.gpu.global.s32 %0, [%1], 1;\n" : "=r"(old) : "l"(p.ctr) : "memory");
      const int target = (old / G + 1) * G;
      while (ldacq(p.ctr) < target) __nanosleep(20);
    }
    __syncthreads();
    PST(3);
    const __nv_bfloat16* ib = p.inter;
    if (sm->stage) {
      __nv_bfloat16* is = (__nv_bfloat16*)(smem + io);
      for (int i = tid; i < nr * HD / 8; i += NW * 32) ((uint4*)is)[i] = __ldcg((const uint4*)p.inter + i);
      __syncthreads();
      ib = is;
    }
    // ---- W2: warp slice K [256 wid, +256) of (expert, 16-row tile) units
    uint4 B2[4][2];
    int curB = -1;
    for (int rd = 0; rd * RT < n2; rd++) {
      const int pend = min(n2, (rd + 1) * RT);
      for (; PPS * (q - ns13) < pend; q++) {
        const uint32_t sbase = C.acquire(q);
#pragma unroll
        for (int pp = 0; pp < PPS; pp++) {
          const int j = PPS * (q - ns13) + pp;
          if (j < n2) {
            const int a = j / sm->nrt;
            if (a != curB) {
              curB = a;
              const int T = sm->aoff[a + 1] - sm->aoff[a];
              const uint4* src = (const uint4*)(ib + (size_t)(sm->aoff[a] + (g < T ? g : 0)) * HD + wid * 256 + t * 16);
#pragma unroll
              for (int c = 0; c < 4; c++)
#pragma unroll
                for (int h = 0; h < 2; h++) B2[c][h] = g < T ? (sm->stage ? src[c * 8 + h] : __ldcg(src + c * 8 + h)) : make_uint4(0, 0, 0, 0);
            }
            float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
            piece(sbase + pp * PC, lane, B2, acc);
            if (t == 0) red[((j - rd * RT) * NW + wid) * 8 + g] = make_float4(acc[0][0] + acc[1][0], acc[0][1] + acc[1][1], acc[0][2] + acc[1][2], acc[0][3] + acc[1][3]);
          }
        }
        C.release(q);
      }
      __syncthreads();
      if (tid < (pend - rd * RT) * 8) {
        const int rp = tid >> 3, gg = tid & 7, j = rd * RT + rp, a = j / sm->nrt, rtl = j - a * sm->nrt, T = sm->aoff[a + 1] - sm->aoff[a];
        float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int w = 0; w < NW; w++) { const float4 v = red[(rp * NW + w) * 8 + gg]; s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w; }
        float* y0 = yb + (sm->aoff[a]) * (MAXRT * 16) + rtl * 16 + gg;
        const float down_scale = sm->g2r[sm->aoff[a]];
        y0[0] = __bfloat162float(__float2bfloat16(s.x * down_scale)); y0[8] = __bfloat162float(__float2bfloat16(s.z * down_scale));
        if (T > 1) { y0[MAXRT * 16] = __bfloat162float(__float2bfloat16(s.y * down_scale)); y0[MAXRT * 16 + 8] = __bfloat162float(__float2bfloat16(s.w * down_scale)); }
      }
      __syncthreads();
    }
    PST(4);
#ifdef PROF
    if (lane == 0 && p.prof) atomicAdd(p.prof + blockIdx.x * 16 + 9, C.wns);
#endif
    // ---- per-route partial sum of squares over this CTA's rows
    const int nrows = sm->nrt * 16, row0 = sm->r0 * 16;
    if (wid < nr) {
      float ss = 0.f;
      for (int r = lane; r < nrows; r += 32) { const float v = yb[wid * MAXRT * 16 + r] * p.nw[row0 + r]; ss += v * v; }
#pragma unroll
      for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
      if (lane == 0) p.ssp[blockIdx.x * MAXR + wid] = ss;
      __threadfence();
    }
    __syncthreads();
    if (tid == 0) {
      int old;
      asm volatile("atom.add.release.gpu.global.s32 %0, [%1], 1;\n" : "=r"(old) : "l"(p.ctr) : "memory");
      const int target = (old / G + 1) * G;
      while (ldacq(p.ctr) < target) __nanosleep(20);
    }
    __syncthreads();
    PST(5);
    if (wid < nr) {
      float s = 0.f;
      float v[8];
#pragma unroll
      for (int k = 0; k < 8; k++) { const int c = lane + 32 * k; v[k] = c < G ? __ldcg(p.ssp + c * MAXR + wid) : 0.f; }  // all loads in flight at once
      s = ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7]));
#pragma unroll
      for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
      const float gs = sm->g2r[wid];
      if (lane == 0) sm->coef[wid] = sm->cf[wid] * rsqrtf(s * (1.f / HD) + 1e-8f);
    }
    __syncthreads();
  }
  // ---- combine this CTA's output rows
  {
    const int nrows = sm->nrt * 16, row0 = sm->r0 * 16;
    for (int i = tid; i < p.M * nrows; i += NW * 32) {
      const int tk = i / nrows, r = i - tk * nrows;
      float o = 0.f;
      for (int k = 0; k < nr; k++) if ((sm->rid[k] >> 3) == tk) o += sm->coef[k] * yb[k * MAXRT * 16 + r];
      p.out[(size_t)tk * HD + row0 + r] = __float2bfloat16(o * p.nw[row0 + r]);
    }
  }
  PST(6);
}

static int g_gridG = 0, g_smemG = 0;
void moe(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor w1, torch::Tensor w2, torch::Tensor g1, torch::Tensor g2,
         torch::Tensor nw, torch::Tensor inter, torch::Tensor ssp, torch::Tensor ctr, torch::Tensor out, int64_t lo, torch::Tensor prof) {
  if (!g_gridG) {
    int dev; cudaGetDevice(&dev); int mx; cudaDeviceGetAttribute(&mx, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaFuncSetAttribute(moe_g, cudaFuncAttributeMaxDynamicSharedMemorySize, mx);
    g_smemG = mx; g_gridG = sms;
  }
  P p{w1.data_ptr<uint8_t>(), w2.data_ptr<uint8_t>(), g1.data_ptr<float>(), g2.data_ptr<float>(), (const __nv_bfloat16*)x.data_ptr(),
      ids.data_ptr<int>(), tw.data_ptr<float>(), nw.data_ptr<float>(), (__nv_bfloat16*)inter.data_ptr(), ssp.data_ptr<float>(),
      (__nv_bfloat16*)out.data_ptr(), ctr.data_ptr<int>(), (int)x.size(0), (int)lo, g_smemG,
      prof.numel() ? (unsigned long long*)prof.data_ptr() : nullptr};
  moe_g<<<g_gridG, NW * 32, g_smemG, at::cuda::getCurrentCUDAStream()>>>(p);
}
