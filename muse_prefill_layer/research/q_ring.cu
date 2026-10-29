
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
// Match baseline BF16 gate/up storage before fused FP32 SiLU/product.
__device__ __forceinline__ float prefill_swiglu_bf16(float gate, float up) {
  gate = __bfloat162float(__float2bfloat16_rn(gate));
  up = __bfloat162float(__float2bfloat16_rn(up));
  return gate / (1.f + __expf(-gate)) * up;
}

#include <stdint.h>

#define HD 4096
#define NLOC 32
#define HSB 2304
#define NHS 16
#ifndef TP
#define TP 64
#endif
#define NTM (TP / 8)
#define XROW 528
#ifndef NCW
#define NCW 8
#endif
#define NTH ((NCW + 1) * 32)
#define TPR (NCW * RPW)
#define WB (TPR * HSB)
#define SLOTB (WB + TP * XROW)
#define NRB1 (8192 / (16 * TPR))
#define NRB2 (4096 / (16 * TPR))
// ctr: [0,32) W13 done per expert, 32 W2 done, 33 queue, 34 CTA done, 35 grid barrier, [64,96) expert counts
#define C_W2 32
#define C_Q 33
#define C_DONE 34
#define C_BAR 35
#define C_CNT 64
#ifndef LAG
#define LAG 64  // all W13 before W2: LAG=1 left producers spinning on W2 dependencies (H100 PROF)
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define TMA 1
#define FULLCNT 33
#else
#define TMA 0
#define FULLCNT 32
#endif

struct P {
  const uint8_t* w1; const uint8_t* w2; const float* g1; const float* g2;
  const __nv_bfloat16* x; const int* ids; const float* tw; const float* nw;
  __nv_bfloat16* inter; __nv_bfloat16* y; float* ss; __nv_bfloat16* out; int* ctr; int* list; int* pos;
  int M, lo; unsigned long long* prof;
};
struct Hdr { int g, e, rb, ps, s, ntok, flags, pad; };
struct Sm {
  Hdr hdr[NSLOT];
  int cnt[NLOC], off[NLOC], act[NLOC], np[NLOC], lc[NLOC], lb[NLOC];
  int segst[2 * NLOC + 1], segk[2 * NLOC];
  int nseg, na, ntot, nw2;
};

#ifdef PROF
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
#define PSET(k, v) do { if (lane == 0 && p.prof) p.prof[(blockIdx.x * (NTH / 32) + wid) * 16 + (k)] = (v); } while (0)
#else
#define PSET(k, v)
#endif

__device__ __forceinline__ void cp16(uint32_t s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(g) : "memory"); }
__device__ __forceinline__ uint32_t bmul(uint32_t a, uint32_t b) { uint32_t d; asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u)); return d; }
__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ int ldacq(const int* p) { int v; asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void red_rel(int* p, int v) { asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ uint32_t sc2(uint32_t w, uint32_t sel) { return __byte_perm(w, 0, sel) * 16u + 0x77007700u; }
__device__ __forceinline__ void mbinit(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(b), "r"(n)); }
__device__ __forceinline__ void mbarrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(b) : "memory"); }
__device__ __forceinline__ void cpa_arrive(uint32_t b) { asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(b) : "memory"); }
__device__ __forceinline__ void mbwait(uint32_t b, uint32_t ph) {
#if TMA
  asm volatile("{\n.reg .pred p;\nW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" :: "r"(b), "r"(ph) : "memory");
#else
  asm volatile("{\n.reg .pred p;\nW: mbarrier.test_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" :: "r"(b), "r"(ph) : "memory");
#endif
}
#if TMA
__device__ __forceinline__ void expect_tx(uint32_t bar, uint32_t n) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "r"(n) : "memory"); }
__device__ __forceinline__ void bulk(uint32_t dst, const void* src, uint32_t n, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(dst), "l"(src), "r"(n), "r"(bar) : "memory");
}
#endif
__device__ __forceinline__ void gsync(int* bar, int target) {  // grid barrier (all CTAs resident)
  __syncthreads();
  if (threadIdx.x == 0) { red_rel(bar, 1); while (ldacq(bar) < target) __nanosleep(32); }
  __syncthreads();
}

template <int NT>
__device__ __forceinline__ void stage(float (&acc)[RPW][NTM][4], const uint8_t* wsm, const uint8_t* xs, int lane) {
  const int g = lane >> 2, t = lane & 3;
  constexpr int CH = NT == 1 ? 4 : (NT == 2 ? 2 : 1);
  uint2 sc[RPW];
#pragma unroll
  for (int r = 0; r < RPW; r++) sc[r] = *(const uint2*)(wsm + r * HSB + 2048 + lane * 8);
#pragma unroll
  for (int c = 0; c < 4; c++) {
    uint4 w[RPW];
    uint32_t sA[RPW], sB[RPW];
#pragma unroll
    for (int r = 0; r < RPW; r++) {
      w[r] = *(const uint4*)(wsm + r * HSB + c * 512 + lane * 16);
      const uint32_t sw = c < 2 ? sc[r].x : sc[r].y;
      sA[r] = sc2(sw, (c & 1) ? 0x4242 : 0x4040); sB[r] = sc2(sw, (c & 1) ? 0x4343 : 0x4141);
    }
#pragma unroll
    for (int hh = 0; hh < 2; hh++) {
      uint4 b[NT];
#pragma unroll
      for (int n = 0; n < NT; n++) b[n] = *(const uint4*)(xs + (n * 8 + g) * XROW + c * 128 + t * 32 + hh * 16);
#pragma unroll
      for (int jj = 0; jj < 2; jj++) {
        const int j = hh * 2 + jj;
#pragma unroll
        for (int r = 0; r < RPW; r++) {
          const uint32_t q = j == 0 ? w[r].x : j == 1 ? w[r].y : j == 2 ? w[r].z : w[r].w;
#ifdef FAKEDQ
          const uint32_t a0 = q ^ sA[r], a1 = q ^ sB[r], a2 = q + sA[r], a3 = q + sB[r];
#else
          const uint32_t a0 = bmul(q & 0x81C081C0u, sA[r]);
          const uint32_t a1 = bmul(((q << 3) & 0x01C001C0u) | ((q << 1) & 0x80008000u), sB[r]);
          const uint32_t a2 = bmul((q << 6) & 0x81C081C0u, sA[r]);
          const uint32_t a3 = bmul(((q >> 5) & 0x01C001C0u) | ((q << 5) & 0x80008000u), sB[r]);
#endif
#pragma unroll
          for (int n = 0; n < NT; n++) {
            const uint32_t* bb = (const uint32_t*)&b[n];
            mma(acc[r][(n * CH + j % CH) % NTM], a0, a1, a2, a3, bb[jj * 2], bb[jj * 2 + 1]);
          }
        }
      }
    }
  }
}

template <int NT>
__device__ __forceinline__ void epilogue(const P& p, const Sm* sm, float (&acc)[RPW][NTM][4], const Hdr& h, int wid, int lane) {
  const int g = lane >> 2, t = lane & 3, e = h.e, ntok = h.ntok;
  const int row0 = sm->off[e] + h.ps * TP;  // compact row of this pass's first token
  constexpr int CH = NT == 1 ? 4 : (NT == 2 ? 2 : 1);
  if (CH > 1) {
#pragma unroll
    for (int r = 0; r < RPW; r++)
#pragma unroll
      for (int n = 0; n < NT; n++)
#pragma unroll
        for (int i = 0; i < 4; i++) {
          float v = acc[r][(n * CH) % NTM][i];
#pragma unroll
          for (int k = 1; k < CH; k++) v += acc[r][(n * CH + k) % NTM][i];
          acc[r][n][i] = v;
        }
  }
  if (!h.g) {
    const float gs = p.g1[p.lo + e];
#pragma unroll
    for (int r = 0; r < RPW; r++) {
      const int mt = h.rb * TPR + wid * RPW + r;
#pragma unroll
      for (int n = 0; n < NT; n++)
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          const int tok = n * 8 + 2 * t + hh;
          const float gt = acc[r][n][hh] * gs, up = acc[r][n][2 + hh] * gs;
          if (tok < ntok) p.inter[(size_t)(row0 + tok) * HD + mt * 8 + g] = __float2bfloat16(prefill_swiglu_bf16(gt, up));
        }
    }
    __syncwarp();
    if (lane == 0) red_rel(p.ctr + e, 1);
  } else {
    const float gs = p.g2[p.lo + e];
    float sa[NT], sb[NT];
#pragma unroll
    for (int n = 0; n < NT; n++) sa[n] = sb[n] = 0.f;
#pragma unroll
    for (int r = 0; r < RPW; r++) {
      const int mt = h.rb * TPR + wid * RPW + r;
      const float n0 = p.nw[mt * 16 + g] * gs, n1 = p.nw[mt * 16 + g + 8] * gs;
#pragma unroll
      for (int n = 0; n < NT; n++) {
        const float v0 = acc[r][n][0] * n0, v1 = acc[r][n][1] * n0, v2 = acc[r][n][2] * n1, v3 = acc[r][n][3] * n1;
        const int t0 = n * 8 + 2 * t;
        if (t0 < ntok) { __nv_bfloat16* yr = p.y + (size_t)(row0 + t0) * HD + mt * 16 + g; yr[0] = __float2bfloat16(v0); yr[8] = __float2bfloat16(v2); }
        if (t0 + 1 < ntok) { __nv_bfloat16* yr = p.y + (size_t)(row0 + t0 + 1) * HD + mt * 16 + g; yr[0] = __float2bfloat16(v1); yr[8] = __float2bfloat16(v3); }
        sa[n] += v0 * v0 + v2 * v2; sb[n] += v1 * v1 + v3 * v3;
      }
    }
#pragma unroll
    for (int n = 0; n < NT; n++) {
#pragma unroll
      for (int o = 4; o < 32; o <<= 1) { sa[n] += __shfl_xor_sync(0xffffffffu, sa[n], o); sb[n] += __shfl_xor_sync(0xffffffffu, sb[n], o); }
      const int t0 = n * 8 + 2 * t;
      if (g == 0) {
        if (t0 < ntok) atomicAdd(p.ss + row0 + t0, sa[n]);
        if (t0 + 1 < ntok) atomicAdd(p.ss + row0 + t0 + 1, sb[n]);
      }
    }
    __syncwarp();
    if (lane == 0) red_rel(p.ctr + C_W2, 1);
  }
}

__device__ __forceinline__ void decode(const Sm* sm, int u, int& g, int& e, int& rb, int& ps) {
  int k = 0;
  while (sm->segst[k + 1] <= u) k++;
  const int code = sm->segk[k];
  g = code >> 6; e = sm->act[code & 63];
  const int np = sm->np[e];
  const int v = u - sm->segst[k];
  ps = v % np; rb = v / np;
}

__global__ void __launch_bounds__(NTH, 1) moe_kernel(P p) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, R = p.M * 8, G = gridDim.x;
  Sm* sm = (Sm*)(smem + NSLOT * SLOTB + NSLOT * 16);
  int* rk = (int*)smem;  // prologue scratch in the ring: this CTA's routes' rank within their expert
  const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem);
  const uint32_t fullb = sbase + NSLOT * SLOTB, emptyb = fullb + NSLOT * 8;
#ifdef PROF
  PSET(0, gtime());
#endif
  if (tid < NSLOT) { mbinit(fullb + tid * 8, FULLCNT); mbinit(emptyb + tid * 8, NCW); }
  if (tid < NLOC) sm->lc[tid] = 0;
#if TMA
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
  __syncthreads();
  const int per = (R + G - 1) / G, r0 = blockIdx.x * per, r1 = min(R, r0 + per);
  for (int i = r0 + tid; i < r1; i += NTH) { const unsigned e = p.ids[i] - p.lo; rk[i - r0] = e < NLOC ? atomicAdd(&sm->lc[e], 1) : -1; }
  __syncthreads();
  if (tid < NLOC) sm->lb[tid] = sm->lc[tid] ? atomicAdd(p.ctr + C_CNT + tid, sm->lc[tid]) : 0;
  gsync(p.ctr + C_BAR, G);
  if (wid == 0) {
    const int c = __ldcg(p.ctr + C_CNT + lane);
    int v = c;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { int n = __shfl_up_sync(0xffffffffu, v, o); if (lane >= o) v += n; }
    sm->cnt[lane] = c; sm->off[lane] = v - c; sm->np[lane] = (c + TP - 1) / TP;
    int rr = 0;
#pragma unroll
    for (int j = 0; j < 32; j++) { const int cj = __shfl_sync(0xffffffffu, c, j); rr += (cj > c) || (cj == c && j < lane); }
    if (c > 0) sm->act[rr] = lane;
    const unsigned am = __ballot_sync(0xffffffffu, c > 0);
    int w2 = c > 0 ? NRB2 * ((c + TP - 1) / TP) * NCW : 0;
#pragma unroll
    for (int o = 16; o; o >>= 1) w2 += __shfl_xor_sync(0xffffffffu, w2, o);
    __syncwarp();
    if (lane == 0) {
      const int na = __popc(am);
      int ns = 0, u = 0;
      auto push = [&](int g, int a) { sm->segst[ns] = u; sm->segk[ns] = g * 64 + a; u += (g ? NRB2 : NRB1) * sm->np[sm->act[a]]; ns++; };
      for (int a = 0; a < na; a++) { push(0, a); if (a >= LAG) push(1, a - LAG); }
      for (int a = na - LAG < 0 ? 0 : na - LAG; a < na; a++) push(1, a);
      sm->segst[ns] = u; sm->nseg = ns; sm->ntot = u; sm->na = na; sm->nw2 = w2;
    }
  }
  __syncthreads();
  for (int i = r0 + tid; i < r1; i += NTH) {
    const int k = rk[i - r0];
    if (k >= 0) { const int e = p.ids[i] - p.lo, ps = sm->off[e] + sm->lb[e] + k; p.list[ps] = i; p.pos[i] = ps; } else p.pos[i] = -1;
  }
  gsync(p.ctr + C_BAR, 2 * G);
#ifdef PROF
  PSET(1, gtime());
#endif
  const int ntot = sm->ntot;
  if (wid == NCW) {  // producer
    int u = 0, nx = 0;
    if (lane == 0) { u = atomicAdd(p.ctr + C_Q, 1); nx = atomicAdd(p.ctr + C_Q, 1); }
    u = __shfl_sync(0xffffffffu, u, 0); nx = __shfl_sync(0xffffffffu, nx, 0);
    unsigned depok = 0;
    int seq = 0;
#ifdef PROF
    long long wemp = 0, wdep = 0;
#endif
    for (; u < ntot;) {
      int g, e, rb, ps; decode(sm, u, g, e, rb, ps);
      const int ntok = min(TP, sm->cnt[e] - ps * TP), row0 = sm->off[e] + ps * TP;
      const uint8_t* wsrc = (g ? p.w2 + (size_t)((p.lo + e) * 256 + rb * TPR) * NHS * HSB
                               : p.w1 + (size_t)((p.lo + e) * 512 + rb * TPR) * NHS * HSB);
      int my[TP / 32];
#pragma unroll
      for (int k = 0; k < TP / 32; k++) { const int i = k * 32 + lane; my[k] = i < ntok ? (g ? row0 + i : __ldcg(p.list + row0 + i) >> 3) : 0; }
      const uint8_t* xsrc = g ? (const uint8_t*)p.inter : (const uint8_t*)p.x;
      for (int i = 0; i < NHS; i++, seq++) {
        const int slot = seq % NSLOT;
#ifdef PROF
        long long c0 = clock64();
#endif
        if (seq >= NSLOT) mbwait(emptyb + slot * 8, ((seq / NSLOT) - 1) & 1);
#ifdef PROF
        wemp += clock64() - c0;
#endif
        if (lane == 0) { sm->hdr[slot] = Hdr{g, e, rb, ps, i, ntok, (i == 0 ? 1 : 0) | (i == NHS - 1 ? 2 : 0) | ((!g && rb == 0) ? 4 : 0), 0}; __threadfence_block(); }
        const uint32_t dst = sbase + slot * SLOTB;
        const uint32_t fb = fullb + slot * 8;
#if TMA
        if (lane == 0) expect_tx(fb, WB);
        __syncwarp();
        if (lane < TPR) bulk(dst + lane * HSB, wsrc + ((size_t)lane * NHS + i) * HSB, HSB, fb);
#else
        for (int q = lane; q < WB / 16; q += 32) {
          const int tile = q / (HSB / 16), o = q % (HSB / 16);
          cp16(dst + tile * HSB + o * 16, wsrc + ((size_t)tile * NHS + i) * HSB + o * 16);
        }
#endif
        if (g && i == 0 && !(depok >> e & 1)) {
#ifdef PROF
          long long c1 = clock64();
#endif
          if (lane == 0) { const int tgt = NRB1 * sm->np[e] * NCW; while (ldacq(p.ctr + e) < tgt) __nanosleep(64); }
          __syncwarp();
          depok |= 1u << e;
#ifdef PROF
          wdep += clock64() - c1;
#endif
        }
#pragma unroll
        for (int k = 0; k < TP / 32; k++)
          for (int rw = k * 32; rw < min(ntok, k * 32 + 32); rw++) {
            const int row = __shfl_sync(0xffffffffu, my[k], rw & 31);
            cp16(dst + WB + rw * XROW + lane * 16, xsrc + (size_t)row * (HD * 2) + i * 512 + lane * 16);
          }
        cpa_arrive(fb);
      }
      u = nx;
      if (lane == 0 && nx < ntot) nx = atomicAdd(p.ctr + C_Q, 1);
      nx = __shfl_sync(0xffffffffu, nx, 0);
    }
    const int slot = seq % NSLOT;
    if (seq >= NSLOT) mbwait(emptyb + slot * 8, ((seq / NSLOT) - 1) & 1);
    if (lane == 0) sm->hdr[slot].g = -1;
#if TMA
    if (lane == 0) mbarrive(fullb + slot * 8);
#endif
    cpa_arrive(fullb + slot * 8);
#ifdef PROF
    PSET(2, gtime()); PSET(3, wemp); PSET(4, wdep);
#endif
  } else {  // consumers
    float acc[RPW][NTM][4];
#ifdef PROF
    long long wful = 0; int nun = 0;
#endif
    for (int q = 0;; q++) {
      const int slot = q % NSLOT;
#ifdef PROF
      long long c0 = clock64();
#endif
      mbwait(fullb + slot * 8, (q / NSLOT) & 1);
#ifdef PROF
      wful += clock64() - c0;
#endif
      const Hdr h = sm->hdr[slot];
      if (h.g < 0) break;
      if (h.flags & 1) {
#pragma unroll
        for (int r = 0; r < RPW; r++)
#pragma unroll
          for (int n = 0; n < NTM; n++) acc[r][n][0] = acc[r][n][1] = acc[r][n][2] = acc[r][n][3] = 0.f;
        if ((h.flags & 4) && wid == 0) for (int i = lane; i < h.ntok; i += 32) p.ss[sm->off[h.e] + h.ps * TP + i] = 0.f;
      }
      const uint8_t* sl = smem + slot * SLOTB;
      const uint8_t* wsm = sl + wid * RPW * HSB;
      const uint8_t* xs = sl + WB;
      const int nt = (h.ntok + 7) >> 3;
#ifndef NOCOMP
      switch (nt) {
        case 1: stage<1>(acc, wsm, xs, lane); break;
        case 2: stage<2>(acc, wsm, xs, lane); break;
        case 3: stage<3>(acc, wsm, xs, lane); break;
        case 4: stage<4>(acc, wsm, xs, lane); break;
#if NTM > 4
        case 5: stage<5>(acc, wsm, xs, lane); break;
        case 6: stage<6>(acc, wsm, xs, lane); break;
        case 7: stage<7>(acc, wsm, xs, lane); break;
        default: stage<NTM>(acc, wsm, xs, lane); break;
#endif
      }
#endif
      __syncwarp();
      if (lane == 0) mbarrive(emptyb + slot * 8);
      if (h.flags & 2) {
#ifdef PROF
        nun++;
#endif
        switch (nt) {
          case 1: epilogue<1>(p, sm, acc, h, wid, lane); break;
          case 2: epilogue<2>(p, sm, acc, h, wid, lane); break;
          case 3: epilogue<3>(p, sm, acc, h, wid, lane); break;
          case 4: epilogue<4>(p, sm, acc, h, wid, lane); break;
#if NTM > 4
          case 5: epilogue<5>(p, sm, acc, h, wid, lane); break;
          case 6: epilogue<6>(p, sm, acc, h, wid, lane); break;
          case 7: epilogue<7>(p, sm, acc, h, wid, lane); break;
          default: epilogue<NTM>(p, sm, acc, h, wid, lane); break;
#endif
        }
      }
    }
#ifdef PROF
    PSET(2, gtime()); PSET(3, wful); PSET(5, nun);
#endif
  }
  const int nw2 = sm->nw2;
  if (lane == 0) while (ldacq(p.ctr + C_W2) < nw2) __nanosleep(32);
  __syncwarp();
#ifdef PROF
  PSET(6, gtime());
#endif
  for (int it = blockIdx.x * (NTH / 32) + wid; it < p.M * 16; it += G * (NTH / 32)) {
    const int t = it >> 4, col = (it & 15) * 256 + lane * 8;
    float a[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int k = 0; k < 8; k++) {
      const int r = t * 8 + k, row = __ldcg(p.pos + r);
      if (row >= 0) {
        const float cf = p.tw[r] * rsqrtf(__ldcg(p.ss + row) * (1.f / HD) + 1e-8f);
        uint4 v = __ldcg((const uint4*)(p.y + (size_t)row * HD + col));
        const __nv_bfloat162* hh = (const __nv_bfloat162*)&v;
#pragma unroll
        for (int j = 0; j < 4; j++) { float2 f = __bfloat1622float2(hh[j]); a[2 * j] += cf * f.x; a[2 * j + 1] += cf * f.y; }
      }
    }
    uint4 o; __nv_bfloat162* ob = (__nv_bfloat162*)&o;
#pragma unroll
    for (int j = 0; j < 4; j++) ob[j] = __floats2bfloat162_rn(a[2 * j], a[2 * j + 1]);
    *(uint4*)(p.out + (size_t)t * HD + col) = o;
  }
#ifdef PROF
  PSET(7, gtime());
#endif
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    if (atomicAdd(p.ctr + C_DONE, 1) == G - 1) {
      for (int i = 0; i <= C_BAR; i++) p.ctr[i] = 0;
      for (int i = 0; i < NLOC; i++) p.ctr[C_CNT + i] = 0;
      __threadfence();
    }
  }
}

static int g_grid = 0;
static size_t smem_bytes() { return (size_t)NSLOT * SLOTB + NSLOT * 16 + sizeof(Sm); }

void moe(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor w1, torch::Tensor w2, torch::Tensor g1,
         torch::Tensor g2, torch::Tensor nw, torch::Tensor inter, torch::Tensor y, torch::Tensor ss, torch::Tensor ctr,
         torch::Tensor list, torch::Tensor pos, torch::Tensor out, int64_t lo, torch::Tensor prof) {
  int M = x.size(0);
  if (!g_grid) {
    int dev; cudaGetDevice(&dev); int mx; cudaDeviceGetAttribute(&mx, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaDeviceGetAttribute(&g_grid, cudaDevAttrMultiProcessorCount, dev);
    cudaFuncSetAttribute(moe_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mx);
  }
  TORCH_CHECK((size_t)((M * 8 + g_grid - 1) / g_grid) * 4 <= (size_t)NSLOT * SLOTB, "M too large");
  P p{w1.data_ptr<uint8_t>(), w2.data_ptr<uint8_t>(), g1.data_ptr<float>(), g2.data_ptr<float>(),
      (const __nv_bfloat16*)x.data_ptr(), ids.data_ptr<int>(), tw.data_ptr<float>(), nw.data_ptr<float>(),
      (__nv_bfloat16*)inter.data_ptr(), (__nv_bfloat16*)y.data_ptr(), ss.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(),
      ctr.data_ptr<int>(), list.data_ptr<int>(), pos.data_ptr<int>(), M, (int)lo,
      prof.numel() ? (unsigned long long*)prof.data_ptr() : nullptr};
  moe_kernel<<<g_grid, NTH, smem_bytes(), at::cuda::getCurrentCUDAStream()>>>(p);
}
