#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define PDL_WAIT() asm volatile("griddepcontrol.wait;\n" ::: "memory")
#define PDL_GO() asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory")
#else
#define PDL_WAIT()
#define PDL_GO()
#endif
#ifndef PDL
#define PDL 1
#endif
#define PDL_LATE 1
#define PDL_GO_EARLY() do { if (!PDL_LATE) PDL_GO(); } while (0)
#define PDL_GO_LATE() do { if (PDL_LATE) PDL_GO(); } while (0)
static int g_pdl = 1;
void pdl_set(int64_t v) { g_pdl = (int)v; }
static inline int pdl_on() { static int a = -1; if (a < 0) { int d, mj; cudaGetDevice(&d); cudaDeviceGetAttribute(&mj, cudaDevAttrComputeCapabilityMajor, d); a = mj >= 9; } return PDL && a && g_pdl; }
template <typename... K, typename... T>
static inline void pdlk(void (*k)(K...), dim3 g, dim3 b, size_t sm, cudaStream_t st, T... a) {
  cudaLaunchConfig_t c = {}; c.gridDim = g; c.blockDim = b; c.dynamicSmemBytes = sm; c.stream = st;
  cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1;
  c.attrs = at; c.numAttrs = pdl_on();
  cudaLaunchKernelEx(&c, k, a...);
}
#include <cuda_bf16.h>
// Match baseline silu_and_mul: BF16 gate/up inputs, FP32 activation/product,
// BF16 product store at each caller. Do not add a nonexistent SiLU tensor boundary.
__device__ __forceinline__ float muse_swiglu_bf16(float gate, float up) {
  gate = __bfloat162float(__float2bfloat16_rn(gate));
  up = __bfloat162float(__float2bfloat16_rn(up));
  return gate / (1.f + __expf(-gate)) * up;
}

#include <stdint.h>

// v47 "gk": g1's structure for M <= MAXM. Activation rows of the CTA's W13 K range sit in smem (rows padded by 16 B), so
// B fragments come from LDS for any token count (n-tiles of 8 tokens, one dequant feeds every n-tile). 16 warps = TG tile
// groups of SW warps; a group's warps split the K range of each of its tiles (2304 B pieces through a per-warp TMA ring)
// and reduce SW-way behind a group-local named barrier. W13: KS1 CTA groups each own one K range (KS1 = 1: SwiGLU in the
// reduction -> inter; KS1 > 1: fp32 gate/up partials -> grid barrier -> every CTA finishes a slice of inter). Grid
// barrier. W2: KS2 CTA groups each own one K range of inter (staged in smem, chunked by whole experts when it does not
// fit) and a contiguous range of 16-row output tiles of every expert -> fp32 y partials. Grid barrier. Each CTA owns ~32
// output rows: sums the KS2 partials, per-route sum of squares -> grid barrier -> norm + combine of its rows. No atomics.
#define HD 4096
#define NLOC 32
#define STG 4608
#define PC 2304
#define NW 16
#ifndef SW
#define SW 4                 // warps per tile
#endif
#define TG (NW / SW)         // tile groups
#ifndef KS1
#define KS1 1                // W13 K split over CTA groups
#endif
#ifndef KS2
#define KS2 2                // W2 K split over CTA groups
#endif
#define PPT (16 / KS1 / SW)  // 2304 B pieces per W13 tile slice
#define W2S (16 / KS2 / SW)  // pieces per W2 unit slice
// KS2 = 1: a CTA's W2 row tiles are its output rows (y stays in smem, no y partials, one grid barrier less)
#ifndef PPS
#define PPS 1                // contiguous 2304 B pieces per ring slot (one TMA box, one barrier)
#endif
#ifndef SG
#define SG 2                 // ring slots per warp
#endif
#define SLOTB (PPS * PC)
#define ZB (2048 + 64)       // zero row for absent token slots (B of a missing token is 0: no tensor-core toggling)
static_assert(PPT % PPS == 0 && W2S % PPS == 0, "slice pieces must fill whole slots");
#ifndef MAXM
#define MAXM 16
#endif
#define MAXR (MAXM * 8)
#define MAXNT ((MAXM + 7) / 8)
#define RW ((MAXR + 31) / 32)  // routing warps
#define XS (HD / KS1 + 8)    // padded smem row (bf16 elements) of x
#define IS (HD / KS2 + 8)    // padded smem row of staged inter
#if KS2 == 1
#define YBR (MAXR * 64 * 4)  // y rows kept at the end of the union region (live while inter is staged)
#else
#define YBR 0
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define TMA 1
#else
#define TMA 0
#if defined(__CUDA_ARCH__)
#define EMUX 1  // sm_80/86 emulation (99 KB smem): the x / inter / y region lives in a global scratch, B via generic loads
#endif
#endif
#define SCR (160 * 1024)  // per-CTA scratch bytes for EMUX
#ifdef EMUX
typedef const uint8_t* BA;
#define LDB(a) (*(const uint4*)(a))
#else
typedef uint32_t BA;
#define LDB(a) lds(a)
#endif

struct P {
  const uint8_t* w1; const uint8_t* w2; const float* g1; const float* g2; const __nv_bfloat16* x; const int* ids;
  const float* tw; const float* nw; __nv_bfloat16* inter; float* gp; float* yp; float* ssp; __nv_bfloat16* out; int* ctr; uint8_t* scr;
  int M, lo, smem; unsigned long long* prof;
};
struct Sm {
  int act[NLOC], aoff[NLOC + 1], hist[RW][NLOC];
  int na, nr, k1, t0, n13, ksp, r0, nrt, o0, onr, nch, cha[NLOC + 1];
  float gs1[NLOC], gs2[NLOC], cf[MAXR];          // W13 global scale per expert; tw x W2 global scale per route
  short tr[MAXM][8];                             // route slot of (token, top-k position), -1 if not local
  float nws[64];                                 // post-norm weight of this CTA's output rows
  float coef[MAXR]; short rid[MAXR]; uint8_t rexp[MAXR];  // 16 / 8-bit: the MAXM 32 layout needs every byte
};

__device__ __forceinline__ uint4 lds(uint32_t a) { uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v; }
__device__ __forceinline__ uint2 lds64(uint32_t a) { uint2 v; asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\n" : "=r"(v.x), "=r"(v.y) : "r"(a)); return v; }
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
__device__ __forceinline__ void gbar(int id, int n) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
#ifdef PROF
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ void rd_stamp(unsigned long long* dst) {
  asm volatile("" ::: "memory");
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
  *dst = t;
  asm volatile("" ::: "memory");
}
#define RD_STAMP(k) do { if (threadIdx.x == 0 && p.prof) rd_stamp(p.prof + blockIdx.x * 32 + (k)); } while (0)
#define PST(k) do { if (threadIdx.x == 0 && p.prof) p.prof[blockIdx.x * 32 + (k)] = gtime(); } while (0)
#else
#define PST(k)
#endif
#define DQ(q, sA, sB, a0, a1, a2, a3) \
  const uint32_t a0 = bmul((q) & 0x81C081C0u, sA); \
  const uint32_t a1 = bmul(__funnelshift_l((q), (q), 10) & 0x81C081C0u, sB); \
  const uint32_t a2 = bmul(__funnelshift_l((q), (q), 13) & 0x81C081C0u, sA); \
  const uint32_t a3 = bmul((((q) & 0x001B001Bu) * 0x1084u) & 0x81C081C0u, sB);

#ifndef CW2
#define CW2 11
#endif
#ifndef CW3
#define CW3 15
#endif
#ifndef CW4
#define CW4 19
#endif
__device__ __forceinline__ int cwt(int nt) { return nt <= 1 ? 8 : (nt == 2 ? CW2 : (nt == 3 ? CW3 : CW4)); }

// grid barrier on a monotone counter (every CTA adds 1; the target is the next multiple of the grid)
__device__ __forceinline__ void grid_sync(int* ctr, int G) {
  __syncthreads();
  if (threadIdx.x == 0) {
    int old;
    __threadfence();
    asm volatile("atom.add.release.gpu.global.s32 %0, [%1], 1;\n" : "=r"(old) : "l"(ctr) : "memory");
    const int target = (old / G + 1) * G;
    while (ldacq(ctr) < target) __nanosleep(20);
  }
  __syncthreads();
}

// one 2304 B piece (16 rows x 256 K: 4 x 64-K chunks, lane t owns 16 contiguous K = one NVFP4 block per chunk) against NT
// n-tiles of activations read from smem: bx[n] = this lane's B row address (token slot 8n + g; an absent slot reads slot
// 0's row: its products only reach that slot's own, discarded, output column) at the piece's K offset
template <int NT>
__device__ __forceinline__ void piece(uint32_t pb, int lane, const BA (&bx)[NT], float (&acc)[NT][2][4]) {
  const uint2 sh = lds64(pb + 2048 + lane * 8);
#pragma unroll
  for (int c = 0; c < 4; c++) {
    const uint4 w = lds(pb + c * 512 + lane * 16);
    uint4 B[NT][2];
#pragma unroll
    for (int n = 0; n < NT; n++) { B[n][0] = LDB(bx[n] + c * 128); B[n][1] = LDB(bx[n] + c * 128 + 16); }
    const uint32_t sw = c < 2 ? sh.x : sh.y;
    const uint32_t sA = sc2(sw, (c & 1) ? 0x4242 : 0x4040), sB = sc2(sw, (c & 1) ? 0x4343 : 0x4141);
    const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int j = 0; j < 4; j++) {
      DQ(ww[j], sA, sB, a0, a1, a2, a3)
#pragma unroll
      for (int n = 0; n < NT; n++) {
        const uint32_t* bb = (const uint32_t*)&B[n][j >> 1];
        mma(acc[n][j & 1], a0, a1, a2, a3, bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);
      }
    }
  }
}

struct Ring {
  const P* p; Sm* sm; uint32_t ring, bars; int lane, grp, s; unsigned long long wns;
  int n13w, n2w;  // this warp's W13 pieces (tiles x PPT) and W2 pieces (units x W2S)
  // issue cursor (pieces are issued strictly in sequence): W13 tile tt / piece j, then W2 expert ca / row tile crt / piece j
  int nq, tt, j, ca, crt;
  __device__ __forceinline__ void init_cursor() {
    nq = 0; tt = sm->t0 + grp; j = 0;
    ca = grp / sm->nrt; crt = grp - ca * sm->nrt;
  }
  __device__ __forceinline__ void issue_next() {  // one slot: PPS consecutive pieces of one slice (contiguous half-stages)
    if (nq >= n13w + n2w) return;
    const uint32_t slot = (nq / PPS) % SG, dst = ring + slot * SLOTB, bar = bars + slot * 8;
    const uint8_t* src;
    if (nq < n13w) {
      src = p->w1 + ((size_t)(p->lo + sm->act[tt >> 9]) * 512 + (tt & 511)) * 8 * STG + (size_t)(sm->k1 * (16 / KS1) + s * PPT + j) * PC;
      j += PPS; if (j == PPT) { j = 0; tt += TG; }
    } else {
      src = p->w2 + ((size_t)(p->lo + sm->act[ca]) * 256 + sm->r0 + crt) * 16 * PC + (size_t)(sm->ksp * (16 / KS2) + s * W2S + j) * PC;
      j += PPS; if (j == W2S) { j = 0; crt += TG; while (crt >= sm->nrt) { crt -= sm->nrt; ca++; } }
    }
    nq += PPS;
#if TMA
    if (lane == 0) {
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "n"(SLOTB) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(dst), "l"(src), "n"(SLOTB), "r"(bar) : "memory");
    }
#else
    for (int k = lane; k < SLOTB / 16; k += 32) cp16(dst + k * 16, src + k * 16);
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" :: "r"(bar) : "memory");
#endif
  }
  __device__ __forceinline__ uint32_t acquire(int q) {
#ifdef PROF
    const unsigned long long w0 = gtime();
#endif
    const int qs = q / PPS;
    mb_wait(bars + (qs % SG) * 8, (qs / SG) & 1);
    __syncwarp();
#ifdef PROF
    wns += gtime() - w0;
#endif
    return ring + (qs % SG) * SLOTB;
  }
  __device__ __forceinline__ void release() { __syncwarp(); issue_next(); }
};

// one tile slice / unit slice: npc pieces accumulated over the warp's K slice, then the group's SW-way reduction in rounds
// of 2 n-tiles; out(n, f, v) consumes fragment f (row f >> 2 / + 8, token slots 8n + 2 (f & 3) + {0, 1}) of n-tile n
template <int NT, class F>
__device__ __forceinline__ void slice(Ring& C, int& q, int npc, const BA (&bx)[NT], float4* red, int wid, int grp, int s, int lane, F out) {
  float acc[NT][2][4];
#pragma unroll
  for (int n = 0; n < NT; n++)
#pragma unroll
    for (int h = 0; h < 2; h++) acc[n][h][0] = acc[n][h][1] = acc[n][h][2] = acc[n][h][3] = 0.f;
  for (int j = 0; j < npc; j += PPS, q += PPS) {
    const uint32_t sbase = C.acquire(q);
#pragma unroll
    for (int jj = 0; jj < PPS; jj++) {
      BA b2[NT];
#pragma unroll
      for (int n = 0; n < NT; n++) b2[n] = bx[n] + (j + jj) * 512;
      piece<NT>(sbase + jj * PC, lane, b2, acc);
    }
    C.release();
  }
#pragma unroll
  for (int nb = 0; nb < NT; nb += 2) {
#pragma unroll
    for (int n = nb; n < nb + 2 && n < NT; n++)
      red[(wid * 2 + n - nb) * 32 + lane] = make_float4(acc[n][0][0] + acc[n][1][0], acc[n][0][1] + acc[n][1][1], acc[n][0][2] + acc[n][1][2], acc[n][0][3] + acc[n][1][3]);
    gbar(1 + grp, SW * 32);
    if (lane < 2 * (32 / SW)) {
      const int n = lane / (32 / SW), f = s * (32 / SW) + lane % (32 / SW);
      if (nb + n < NT) {
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int w = 0; w < SW; w++) { const float4 u = red[((grp * SW + w) * 2 + n) * 32 + f]; v.x += u.x; v.y += u.y; v.z += u.z; v.w += u.w; }
        out(nb + n, f, v);
      }
    }
    gbar(1 + grp, SW * 32);  // partials consumed before they are overwritten
  }
}

// sum over the grid's per-CTA partials of route r (all loads issued before the first add)
__device__ __forceinline__ float sum_ssp(const float* ssp, int r, int G, int lane) {
  float v[8];
#pragma unroll
  for (int k = 0; k < 8; k++) { const int c = lane + 32 * k; v[k] = c < G ? __ldcg(ssp + (size_t)c * MAXR + r) : 0.f; }
  float s = ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7]));
#pragma unroll
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  return s;
}

__global__ void __launch_bounds__(NW * 32, 1) moe_k(P p) {
  PDL_WAIT();
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, R = p.M * 8, G = gridDim.x;
  const int grp = wid / SW, s = wid % SW;
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem);
  Sm* sm = (Sm*)(smem + NW * SG * 8);
  const int ro = (NW * SG * 8 + (int)sizeof(Sm) + 127) & ~127;              // ring
  const int rdo = ro + NW * SG * SLOTB;                                       // partials: [NW][2 n-tiles][32] float4
  const int zo = rdo + NW * 2 * 32 * 16;                                      // zero row
  const int xo = zo + ZB;                                                     // x rows / staged inter / y rows (union)
  float4* red = (float4*)(smem + rdo);
#ifdef EMUX
  uint8_t* rg = p.scr + (size_t)blockIdx.x * SCR;
  const int rsz = SCR;
  const BA rb = rg;
#else
  uint8_t* rg = smem + xo;
  const int rsz = p.smem - xo;
  const BA rb = sb + xo;
#endif
  __nv_bfloat16* xs = (__nv_bfloat16*)rg;
#ifdef EMUX
  const BA zb = smem + zo + (lane & 3) * 32;
#else
  const BA zb = sb + zo + (lane & 3) * 32;
#endif
  PST(0);
  if (lane < SG) mb_init(sb + (wid * SG + lane) * 8, TMA ? 1 : 32);
#if TMA
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
  RD_STAMP(20); RD_STAMP(21); RD_STAMP(13);
  // routing: deterministic slots (experts ascending, routes of an expert in route order)
  int e = -1, rank = 0;
  if (wid < RW) {
    if (tid < R) { const unsigned u = p.ids[tid] - p.lo; e = u < NLOC ? (int)u : -1; }
    const unsigned peers = __match_any_sync(0xffffffffu, e);
    rank = __popc(peers & ((1u << lane) - 1));
    sm->hist[wid][lane] = 0;
    __syncwarp();
    if (e >= 0 && rank == 0) sm->hist[wid][e] = __popc(peers);
  }
  __syncthreads();
  if (wid == 0) {
    int c = 0;
#pragma unroll
    for (int w = 0; w < RW; w++) { const int h = sm->hist[w][lane]; sm->hist[w][lane] = c; c += h; }
    const unsigned am = __ballot_sync(0xffffffffu, c > 0);
    int v = c;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { const int n = __shfl_up_sync(0xffffffffu, v, o); if (lane >= o) v += n; }
    const int na = __popc(am), nr = __shfl_sync(0xffffffffu, v, 31);
    if (c > 0) { const int a = __popc(am & ((1u << lane) - 1)); sm->act[a] = lane; sm->aoff[a] = v - c; }
    RD_STAMP(14);
    // cost-weighted contiguous W13 tile ranges (lane = expert): a tile of an expert with NT n-tiles costs CW[NT - 1] / 8
    const int b = blockIdx.x;
    int k = 0;
    while ((k + 1) * G / KS1 <= b) k++;
    int g0 = k * G / KS1, gn = (k + 1) * G / KS1 - g0;
    const int w = c > 0 ? cwt((c + 7) >> 3) : 0;
    int cs = 512 * w;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { const int n = __shfl_up_sync(0xffffffffu, cs, o); if (lane >= o) cs += n; }
    const int Ct = __shfl_sync(0xffffffffu, cs, 31), base = cs - 512 * w;
    const int lo = (b - g0) * Ct / gn, hi = (b - g0 + 1) * Ct / gn;
    const int tl = __reduce_add_sync(0xffffffffu, w && lo > base ? min(512, (lo - base + w - 1) / w) : 0);
    const int th = __reduce_add_sync(0xffffffffu, w && hi > base ? min(512, (hi - base + w - 1) / w) : 0);
    __syncwarp();  // act / aoff of every lane visible to lane 0
    if (lane == 0) {
      sm->aoff[na] = nr; sm->na = na; sm->nr = nr;
      sm->k1 = k; sm->t0 = tl; sm->n13 = th - tl;
      k = 0;
      while ((k + 1) * G / KS2 <= b) k++;
      g0 = k * G / KS2; gn = (k + 1) * G / KS2 - g0;
      sm->ksp = k; sm->r0 = (b - g0) * 256 / gn; sm->nrt = (b - g0 + 1) * 256 / gn - sm->r0;
#if KS2 == 1
      sm->o0 = sm->r0 * 16; sm->onr = sm->nrt * 16;
#else
      sm->o0 = b * HD / G; sm->onr = (b + 1) * HD / G - sm->o0;
#endif
      // W2 staging chunks: runs of whole experts whose routes fit the union region (normally one chunk)
#ifdef FITCAP
      const int fit = min((rsz - YBR) / (IS * 2), FITCAP);  // test: force chunked staging
#else
      const int fit = (rsz - YBR) / (IS * 2);
#endif
      int nch = 0, c0 = 0;
      sm->cha[0] = 0;
      for (int a = 0; a < na; a++)
        if (sm->aoff[a + 1] - sm->aoff[c0] > fit) { sm->cha[++nch] = a; c0 = a; }
      sm->cha[++nch] = na; sm->nch = nch;
      if (p.M > MAXM || p.M * XS * 2 > rsz || MAXR * 64 * 4 > rsz || sm->onr > 64 || fit < MAXM || xo > p.smem || G > 256 || (KS2 == 1 && p.M * XS * 2 > rsz - YBR)) asm volatile("trap;");
    }
  }
  __syncthreads();
  RD_STAMP(15);
  const int na = sm->na, nr = sm->nr, n13 = sm->n13, nrt = sm->nrt;
  Ring C; C.p = &p; C.sm = sm; C.lane = lane; C.grp = grp; C.s = s; C.wns = 0;
  C.ring = sb + ro + wid * SG * SLOTB; C.bars = sb + wid * SG * 8;
  const int nt13 = n13 > grp ? (n13 - grp + TG - 1) / TG : 0, nu = na * nrt, nu2 = nu > grp ? (nu - grp + TG - 1) / TG : 0;
  C.n13w = nt13 * PPT; C.n2w = nu2 * W2S;
  C.init_cursor();
#pragma unroll
  for (int i = 0; i < SG; i++) C.issue_next();
  PST(1);
  RD_STAMP(16);
  if (tid < p.M * 8) sm->tr[tid >> 3][tid & 7] = -1;
  if (tid < ZB / 16) *(uint4*)(smem + zo + tid * 16) = make_uint4(0, 0, 0, 0);
  if (tid < 64) sm->nws[tid] = tid < sm->onr ? p.nw[sm->o0 + tid] : 0.f;
  __syncthreads();
  if (lane == 0 && p.prof) rd_stamp(p.prof+gridDim.x*32+(blockIdx.x*NW+wid)*2+0);
  if (tid < R && e >= 0) {
    int a = 0;
    while (sm->act[a] != e) a++;
    const int r = sm->aoff[a] + sm->hist[wid][e] + rank;
    sm->rid[r] = tid; sm->rexp[r] = a; sm->tr[tid >> 3][tid & 7] = r;
    sm->cf[r] = p.tw[tid];
    if (rank == 0 && sm->hist[wid][e] == 0) { sm->gs1[a] = p.g1[p.lo + e]; sm->gs2[a] = p.g2[p.lo + e]; }
  }
  if (lane == 0 && p.prof) rd_stamp(p.prof+gridDim.x*32+(blockIdx.x*NW+wid)*2+1);
  RD_STAMP(18);
  {
    const int kx = sm->k1 * (HD / KS1);
    for (int i = tid; i < p.M * (HD / KS1 / 8); i += NW * 32) {
      const int row = i / (HD / KS1 / 8), c8 = i - row * (HD / KS1 / 8);
      *(uint4*)(xs + row * XS + c8 * 8) = *((const uint4*)(p.x + (size_t)row * HD + kx) + c8);
    }
  }
  __syncthreads();
  RD_STAMP(19);
  int q = 0;
#ifdef PROF
  if (tid == 0 && p.prof) p.prof[blockIdx.x * 32 + 11] = clock64();
#endif
  if (na) {
    // ---- W13: group grp takes tiles grp, grp + TG, ... of the CTA's range; warp s the K slice s of the CTA's K range
    for (int i = 0; i < nt13; i++) {
      const int tt = sm->t0 + grp + i * TG, a = tt >> 9, T = sm->aoff[a + 1] - sm->aoff[a], ao = sm->aoff[a];
      const BA xb = rb + (s * (HD / KS1 / SW) + (lane & 3) * 16) * 2;
      const int g = lane >> 2;
      const int col0 = (tt & 511) * 8;
      const float gs = sm->gs1[a];
      auto out = [&](int n, int f, float4 v) {
        const int gg = f >> 2, tt2 = f & 3, col = col0 + gg;
#pragma unroll
        for (int h = 0; h < 2; h++) {
          const int slot = n * 8 + 2 * tt2 + h;
          if (slot < T) {
#if KS1 == 1
            const float gt = (h ? v.y : v.x) * gs, up = (h ? v.w : v.z) * gs;
            p.inter[(size_t)(ao + slot) * HD + col] = __float2bfloat16(muse_swiglu_bf16(gt, up));
#else
            *(float2*)(p.gp + (((size_t)sm->k1 * MAXR + ao + slot) * HD + col) * 2) = make_float2(h ? v.y : v.x, h ? v.w : v.z);
#endif
          }
        }
      };
#define BXR(n) ((n) * 8 + g < T ? xb + (sm->rid[ao + (n) * 8 + g] >> 3) * XS * 2 : zb)
      if (T <= 8) { BA bx[1] = {BXR(0)}; slice<1>(C, q, PPT, bx, red, wid, grp, s, lane, out); }
#if MAXNT >= 2
      else if (T <= 16) { BA bx[2] = {BXR(0), BXR(1)}; slice<2>(C, q, PPT, bx, red, wid, grp, s, lane, out); }
#endif
#if MAXNT >= 3
      else if (T <= 24) { BA bx[3] = {BXR(0), BXR(1), BXR(2)}; slice<3>(C, q, PPT, bx, red, wid, grp, s, lane, out); }
      else { BA bx[4] = {BXR(0), BXR(1), BXR(2), BXR(3)}; slice<4>(C, q, PPT, bx, red, wid, grp, s, lane, out); }
#endif
#undef BXR
    }
    PST(2);
#ifdef PROF
    if (tid == 0 && p.prof) p.prof[blockIdx.x * 32 + 12] = clock64();
    if (lane == 0 && p.prof) atomicAdd(p.prof + blockIdx.x * 32 + 8, C.wns);
    C.wns = 0;
#endif
    grid_sync(p.ctr, G);
#if KS1 > 1
    // ---- inter from the KS1 gate/up partials: this CTA's slice of all (route, column) pairs
    {
      const int N = nr * HD, i0 = (int)((long long)blockIdx.x * N / G), i1 = (int)((long long)(blockIdx.x + 1) * N / G);
      for (int ib = i0; ib < i1; ib += 4 * NW * 32) {  // 4 elements per thread with all partial loads in flight together
        float2 u[4][KS1];
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const int i = ib + j * NW * 32 + tid, ic = i < i1 ? i : i0, r = ic / HD, col = ic - r * HD;
#pragma unroll
          for (int k = 0; k < KS1; k++) u[j][k] = __ldcg((const float2*)(p.gp + (((size_t)k * MAXR + r) * HD + col) * 2));
        }
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const int i = ib + j * NW * 32 + tid, r = i / HD, col = i - r * HD;
          if (i < i1) {
            float gx = 0.f, ux = 0.f;
#pragma unroll
            for (int k = 0; k < KS1; k++) { gx += u[j][k].x; ux += u[j][k].y; }
            const float gs = sm->gs1[sm->rexp[r]], gt = gx * gs, up = ux * gs;
            p.inter[(size_t)r * HD + col] = __float2bfloat16(muse_swiglu_bf16(gt, up));
          }
        }
      }
    }
    grid_sync(p.ctr, G);
#endif
    PST(3);
    // ---- W2: unit u = (expert u / nrt, row tile r0 + u % nrt), group grp takes units grp, grp + TG, ...; warp s the K slice.
    // inter[route][K range of this CTA group] is staged in smem per chunk of whole experts (normally one chunk)
    const int kb = sm->ksp * (HD / KS2);
    int i = 0;
    float* yp = p.yp + (size_t)sm->ksp * MAXR * HD;
    float* ybl = (float*)(rg + rsz - YBR);  // KS2 = 1 only
    for (int ch = 0; ch < sm->nch; ch++) {
      const int ca0 = sm->cha[ch], ca1 = sm->cha[ch + 1], rb0 = sm->aoff[ca0], nrc = sm->aoff[ca1] - rb0;
      for (int k = tid; k < nrc * (HD / KS2 / 8); k += NW * 32) {
        const int r = k / (HD / KS2 / 8), c8 = k - r * (HD / KS2 / 8);
        *(uint4*)(xs + r * IS + c8 * 8) = __ldcg((const uint4*)(p.inter + (size_t)(rb0 + r) * HD + kb) + c8);
      }
      __syncthreads();
      for (; i < nu2 && grp + i * TG < ca1 * nrt; i++) {
        const int u = grp + i * TG, a = u / nrt, rt = sm->r0 + u - a * nrt, T = sm->aoff[a + 1] - sm->aoff[a], ao = sm->aoff[a];
        const int g = lane >> 2, ab = ao - rb0;
        const BA xb = rb + (s * (HD / KS2 / SW) + (lane & 3) * 16) * 2;
        auto out = [&](int n, int f, float4 v) {
          const int gg = f >> 2, tt2 = f & 3, row = rt * 16 + gg;
#pragma unroll
          for (int h = 0; h < 2; h++) {
            const int slot = n * 8 + 2 * tt2 + h;
#if KS2 == 1
            if (slot < T) { float* y = ybl + (ao + slot) * 64 + row - sm->o0; y[0] = h ? v.y : v.x; y[8] = h ? v.w : v.z; }
#else
            if (slot < T) { float* y = yp + (size_t)(ao + slot) * HD + row; y[0] = h ? v.y : v.x; y[8] = h ? v.w : v.z; }
#endif
          }
        };
#define BXR(n) ((n) * 8 + g < T ? xb + (ab + (n) * 8 + g) * IS * 2 : zb)
        if (T <= 8) { BA bx[1] = {BXR(0)}; slice<1>(C, q, W2S, bx, red, wid, grp, s, lane, out); }
#if MAXNT >= 2
        else if (T <= 16) { BA bx[2] = {BXR(0), BXR(1)}; slice<2>(C, q, W2S, bx, red, wid, grp, s, lane, out); }
#endif
#if MAXNT >= 3
        else if (T <= 24) { BA bx[3] = {BXR(0), BXR(1), BXR(2)}; slice<3>(C, q, W2S, bx, red, wid, grp, s, lane, out); }
        else { BA bx[4] = {BXR(0), BXR(1), BXR(2), BXR(3)}; slice<4>(C, q, W2S, bx, red, wid, grp, s, lane, out); }
#endif
#undef BXR
      }
      __syncthreads();
    }
    PST(4);
#ifdef PROF
    if (lane == 0 && p.prof) atomicAdd(p.prof + blockIdx.x * 32 + 9, C.wns);
#endif
#if KS2 == 1
    float* yb = ybl;
    PST(5);
#else
    grid_sync(p.ctr, G);
    PST(5);
    // ---- this CTA's output rows: y = sum of the KS2 partials (8 elements per thread in flight together)
    float* yb = (float*)rg;
    for (int ib = 0; ib < nr * 64; ib += 8 * NW * 32) {
      float v[8][KS2];
#pragma unroll
      for (int j = 0; j < 8; j++) {
        const int i2 = ib + j * NW * 32 + tid, r = i2 >> 6, rr = i2 & 63;
        const bool ok = i2 < nr * 64 && rr < sm->onr;
#pragma unroll
        for (int k = 0; k < KS2; k++) v[j][k] = ok ? __ldcg(p.yp + ((size_t)k * MAXR + r) * HD + sm->o0 + rr) : 0.f;
      }
#pragma unroll
      for (int j = 0; j < 8; j++) {
        const int i2 = ib + j * NW * 32 + tid;
        float t = 0.f;
#pragma unroll
        for (int k = 0; k < KS2; k++) t += v[j][k];
        if (i2 < nr * 64) yb[i2] = t;
      }
    }
#endif
    __syncthreads();
    // Complete the W2 reduction and BF16 materialization before norm weighting.
    for (int i = tid; i < nr * 64; i += NW * 32) {
      const int r = i >> 6, rr = i & 63;
      if (rr < sm->onr) yb[i] = __bfloat162float(__float2bfloat16(yb[i] * sm->gs2[sm->rexp[r]]));
    }
    __syncthreads();
    const int onr = sm->onr;
    for (int r = wid; r < nr; r += NW) {
      float ss = 0.f;
      for (int rr = lane; rr < onr; rr += 32) { const float v = yb[r * 64 + rr] * sm->nws[rr]; ss += v * v; }
#pragma unroll
      for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
      if (lane == 0) p.ssp[(size_t)blockIdx.x * MAXR + r] = ss;
    }
    grid_sync(p.ctr, G);
    PST(6);
    // per-route norm: 16 threads per route, all of a thread's (<= 16) partial loads in flight together
    for (int rb2 = 0; rb2 < nr; rb2 += NW * 2) {
      const int r = rb2 + (tid >> 4), l = tid & 15;
      float v[16];
#pragma unroll
      for (int j = 0; j < 16; j++) { const int c = l + 16 * j; v[j] = r < nr && c < G ? __ldcg(p.ssp + (size_t)c * MAXR + r) : 0.f; }
      float ssum = (((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7]))) + (((v[8] + v[9]) + (v[10] + v[11])) + ((v[12] + v[13]) + (v[14] + v[15])));
#pragma unroll
      for (int o = 8; o; o >>= 1) ssum += __shfl_xor_sync(0xffffffffu, ssum, o);
      if (l == 0 && r < nr) { const float g2 = sm->gs2[sm->rexp[r]]; sm->coef[r] = sm->cf[r] * rsqrtf(ssum * (1.f / HD) + 1e-8f); }
    }
    __syncthreads();
  }
  {
    const int o0 = sm->o0, onr = sm->onr;
#if KS2 == 1
    const float* yb = (const float*)(rg + rsz - YBR);
#else
    const float* yb = (const float*)rg;
#endif
    for (int i2 = tid; i2 < p.M * onr; i2 += NW * 32) {
      const int tk = i2 / onr, rr = i2 - tk * onr;
      float o = 0.f;
#pragma unroll
      for (int k = 0; k < 8; k++) { const int r = sm->tr[tk][k]; if (r >= 0) o += sm->coef[r] * yb[r * 64 + rr]; }
      p.out[(size_t)tk * HD + o0 + rr] = __float2bfloat16(o * sm->nws[rr]);
    }
  }
  PST(7);
}

static int g_gridK = 0, g_smemK = 0;
void moe(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor w1, torch::Tensor w2, torch::Tensor g1, torch::Tensor g2,
         torch::Tensor nw, torch::Tensor inter, torch::Tensor gp, torch::Tensor yp, torch::Tensor ssp, torch::Tensor ctr, torch::Tensor scr, torch::Tensor out, int64_t lo, torch::Tensor prof) {
  if (!g_gridK) {
    int dev; cudaGetDevice(&dev); int mx; cudaDeviceGetAttribute(&mx, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaFuncSetAttribute(moe_k, cudaFuncAttributeMaxDynamicSharedMemorySize, mx);
    g_smemK = mx; g_gridK = sms;
  }
  {
    int dev; cudaGetDevice(&dev); int ma; cudaDeviceGetAttribute(&ma, cudaDevAttrComputeCapabilityMajor, dev);
    const int M = (int)x.size(0), ro = (NW * SG * 8 + (int)sizeof(Sm) + 127) & ~127, xo = ro + NW * SG * SLOTB + NW * 2 * 32 * 16 + ZB;
    const int rsz = ma >= 9 ? g_smemK - xo : SCR;
    TORCH_CHECK(M <= MAXM && xo <= g_smemK && g_gridK <= 256 && g_gridK >= 64, "gk: unsupported shape");
    TORCH_CHECK((long long)M * XS * 2 <= rsz - YBR && MAXR * 64 * 4 <= rsz && (rsz - YBR) / (IS * 2) >= MAXM, "gk: smem layout does not fit");
  }
  P p{w1.data_ptr<uint8_t>(), w2.data_ptr<uint8_t>(), g1.data_ptr<float>(), g2.data_ptr<float>(), (const __nv_bfloat16*)x.data_ptr(),
      ids.data_ptr<int>(), tw.data_ptr<float>(), nw.data_ptr<float>(), (__nv_bfloat16*)inter.data_ptr(), gp.data_ptr<float>(), yp.data_ptr<float>(),
      ssp.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(), ctr.data_ptr<int>(), scr.data_ptr<uint8_t>(), (int)x.size(0), (int)lo, g_smemK,
      prof.numel() ? (unsigned long long*)prof.data_ptr() : nullptr};
  pdlk(moe_k, dim3(g_gridK), dim3(NW * 32), (size_t)g_smemK, at::cuda::getCurrentCUDAStream(), p);
}

int64_t rd_resources() { cudaFuncAttributes a{}; auto err=cudaFuncGetAttributes(&a, moe_k); TORCH_CHECK(err==cudaSuccess,cudaGetErrorString(err)); return ((int64_t)a.numRegs<<32) | (uint32_t)a.localSizeBytes; }
