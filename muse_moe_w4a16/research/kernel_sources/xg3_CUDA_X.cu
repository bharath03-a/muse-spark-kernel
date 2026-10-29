#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
// Strict SGLang boundaries: BF16 gate/up -> FP32 SiLU/product -> BF16 inter.
__device__ __forceinline__ float strict_swiglu(float gate, float up) {
  gate=__bfloat162float(__float2bfloat16_rn(gate));
  up=__bfloat162float(__float2bfloat16_rn(up));
  return gate/(1.f+__expf(-gate))*up;
}

#include <stdint.h>

// v43 "wx": one ring of 4-chunk group slots; NCW consumer warps (8 or 16), RPW m64 blocks per warpgroup: one pass per (expert, 128-row block) over ALL the expert's tokens (N <= 256): every weight is dequantized
// once per expert. Slots are 64-K chunks: [activation tile N x 128 B, SW128-swizzled, from a per-call gathered buffer]
// [8 tiles x 576 B of weights (512 B mma A-fragment words + 64 B block scales)], both by TMA. Two consumer warpgroups
// (m64 each) run wgmma RS: A = dequantized fragment in registers, B = the activation tile; per k16 step
// wait<3> / dequant / fence / wgmma / commit (CUTLASS mixed-input RS order: four A register sets in rotation).
#define HD 4096
#define NLOC 32
#define TPU 8
#define REC 576
#define WREC (TPU * REC)
#define NCH 64
#ifndef NCW
#define NCW 8
#endif
#ifndef RPW
#define RPW 1  // m64 blocks per consumer warpgroup: unit = 128 * RPW rows sharing each activation tile
#endif
#ifndef TPX
#define TPX ((RPW == 1 && NCW == 8) ? 128 : 64)  // tokens per pass (accumulators RPW * TPX / 2 per thread)
#endif
#define XB (TPX * 128)
#define WCH 4                      // 64-K chunks per slot
#define UBL ((NCW / 8) * RPW)      // 128-row blocks per unit
#define WSB (WCH * WREC * UBL)
#define ASB (WCH * XB)
#define GSB (((ASB + WSB) + 1023) & ~1023)
#ifndef NSL
#define NSL (GSB <= 73728 ? 3 : 2)
#endif
#define RING (NSL * GSB)
#define NRB1 (64 / UBL)
#define NRB2 (32 / UBL)
#define NTH ((NCW + 1) * 32)
#define C_W2 32
#define C_Q 33
#define C_DONE 34
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define TMA 1
#else
#define TMA 0
#endif

struct P {
  const uint8_t* w1; const uint8_t* w2; const float* g1; const float* g2;
  const uint8_t* xg; uint8_t* ig; const int* ids; const float* tw; const float* nw;
  __nv_bfloat16* y; float* ss; __nv_bfloat16* out; int* ctr;
  int M, lo; unsigned long long* prof;
};
struct Hdr { int g, e, rb, ps, c, ntok, flags, pad; };
struct Sm {
  Hdr hdr[NSL];
  int cnt[NLOC], off[NLOC], act[NLOC], np[NLOC], base[NLOC], npad[NLOC];
  long long xoff[NLOC];
  int segst[2 * NLOC + 1], segk[2 * NLOC];
  int nseg, na, ntot, nw2;
  int wc[NTH / 32][NLOC];
};

#ifdef PROF
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
#define PSET(k, v) do { if (lane == 0 && p.prof) p.prof[(blockIdx.x * (NTH / 32) + wid) * 16 + (k)] = (v); } while (0)
#else
#define PSET(k, v)
#endif

__device__ __forceinline__ int kperm(int col) {  // wgmma K order inside each 64-K chunk
  const int a = col & 63;
  return (col & ~63) | (((a >> 2) & 3) << 4) | (((a & 3) >> 1) << 3) | ((a >> 4) << 1) | (a & 1);
}
__device__ __forceinline__ uint32_t xaddr_of(int pcol, int pos, int npad) {  // byte offset in an expert's gathered block
  const int b = (pcol & 63) * 2;
  return ((uint32_t)(pcol >> 6) * npad + pos) * 128u + ((((b >> 4) ^ (pos & 7))) << 4) + (b & 15);
}
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
__device__ __forceinline__ uint32_t lds32(uint32_t a) { uint32_t v; asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(v) : "r"(a)); return v; }
__device__ __forceinline__ uint32_t lds16(uint32_t a) { unsigned short v; asm volatile("ld.shared.u16 %0, [%1];\n" : "=h"(v) : "r"(a)); return v; }
__device__ __forceinline__ uint4 lds128(uint32_t a) { uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v; }

#if TMA
__device__ __forceinline__ void expect_tx(uint32_t bar, uint32_t n) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "r"(n) : "memory"); }
__device__ __forceinline__ void bulk(uint32_t dst, const void* src, uint32_t n, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(dst), "l"(src), "r"(n), "r"(bar) : "memory");
}
__device__ __forceinline__ uint64_t gdesc(uint32_t saddr) {  // K-major SW128: LBO 16, SBO 1024
  return (uint64_t)((saddr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)64 << 32) | (1ull << 62);
}
__device__ __forceinline__ uint64_t gdesc_add(uint64_t d, uint32_t off) {
  const uint32_t lo = (uint32_t)d + (off >> 4), hi = (uint32_t)(d >> 32);
  return ((uint64_t)hi << 32) | lo;
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int K> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(K) : "memory"); }
template <int N> __device__ __forceinline__ void wg_rs(float* d, const uint32_t* a, uint64_t db, int sd);
template <int N> __device__ __forceinline__ void fence_acc(float* d);
template <> __device__ __forceinline__ void wg_rs<8>(float* d, const uint32_t* a, uint64_t db, int sd) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %9, 0;\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(sd));
}
template <> __device__ __forceinline__ void fence_acc<8>(float* d) {
#pragma unroll
  for (int i = 0; i < 4; i++) asm volatile("" : "+f"(d[i]) :: "memory");
}
template <> __device__ __forceinline__ void wg_rs<16>(float* d, const uint32_t* a, uint64_t db, int sd) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %13, 0;\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, %12, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(sd));
}
template <> __device__ __forceinline__ void fence_acc<16>(float* d) {
#pragma unroll
  for (int i = 0; i < 8; i++) asm volatile("" : "+f"(d[i]) :: "memory");
}
template <> __device__ __forceinline__ void wg_rs<32>(float* d, const uint32_t* a, uint64_t db, int sd) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %21, 0;\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(sd));
}
template <> __device__ __forceinline__ void fence_acc<32>(float* d) {
#pragma unroll
  for (int i = 0; i < 16; i++) asm volatile("" : "+f"(d[i]) :: "memory");
}
template <> __device__ __forceinline__ void wg_rs<64>(float* d, const uint32_t* a, uint64_t db, int sd) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(sd));
}
template <> __device__ __forceinline__ void fence_acc<64>(float* d) {
#pragma unroll
  for (int i = 0; i < 32; i++) asm volatile("" : "+f"(d[i]) :: "memory");
}
template <> __device__ __forceinline__ void wg_rs<128>(float* d, const uint32_t* a, uint64_t db, int sd) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %69, 0;\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, {%64,%65,%66,%67}, %68, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(sd));
}
template <> __device__ __forceinline__ void fence_acc<128>(float* d) {
#pragma unroll
  for (int i = 0; i < 64; i++) asm volatile("" : "+f"(d[i]) :: "memory");
}
template <> __device__ __forceinline__ void wg_rs<256>(float* d, const uint32_t* a, uint64_t db, int sd) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %133, 0;\nwgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127}, {%128,%129,%130,%131}, %132, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(sd));
}
template <> __device__ __forceinline__ void fence_acc<256>(float* d) {
#pragma unroll
  for (int i = 0; i < 128; i++) asm volatile("" : "+f"(d[i]) :: "memory");
}

template <> __device__ __forceinline__ void wg_rs<24>(float* d, const uint32_t* a, uint64_t db, int sd) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %17, 0;\nwgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(sd));
}
template <> __device__ __forceinline__ void fence_acc<24>(float* d) {
#pragma unroll
  for (int i = 0; i < 12; i++) asm volatile("" : "+f"(d[i]) :: "memory");
}
template <> __device__ __forceinline__ void wg_rs<48>(float* d, const uint32_t* a, uint64_t db, int sd) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %29, 0;\nwgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23}, {%24,%25,%26,%27}, %28, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(sd));
}
template <> __device__ __forceinline__ void fence_acc<48>(float* d) {
#pragma unroll
  for (int i = 0; i < 24; i++) asm volatile("" : "+f"(d[i]) :: "memory");
}
#else
// sm_80/86 emulation: each warp runs mma.sync for its 16 rows on the same swizzled B tile
__device__ __forceinline__ uint64_t gdesc(uint32_t saddr) { return saddr; }
__device__ __forceinline__ uint64_t gdesc_add(uint64_t d, uint32_t off) { return d + off; }
__device__ __forceinline__ void wg_fence() {}
__device__ __forceinline__ void wg_commit() {}
template <int K> __device__ __forceinline__ void wg_wait() {}
template <int N> __device__ __forceinline__ void fence_acc(float* d) {}
template <int N> __device__ __forceinline__ void wg_rs(float* d, const uint32_t* a, uint64_t db, int sd) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const uint32_t baddr = (uint32_t)db, atom = baddr & ~127u, kc = (baddr & 127u) >> 4;
  if (!sd) {
#pragma unroll
    for (int i = 0; i < N / 2; i++) d[i] = 0.f;
  }
#pragma unroll
  for (int i = 0; i < N / 8; i++) {
    const int row = i * 8 + g;
    const uint32_t rb = atom + row * 128;
    mma(d + i * 4, a[0], a[1], a[2], a[3], lds32(rb + ((kc ^ (row & 7)) << 4) + t * 4), lds32(rb + (((kc + 1) ^ (row & 7)) << 4) + t * 4));
  }
}
#endif

#ifdef XGOLD
// gather: route i of a local expert e -> row pos(i) of e's block, K in the wgmma order, 16-B chunks swizzled
__global__ void __launch_bounds__(256) xgather(const __nv_bfloat16* x, const int* ids, int M, int lo, uint8_t* xg) {
  __shared__ int cnt[NLOC];
  __shared__ long long xo[NLOC];
  __shared__ int npad[NLOC];
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, R = M * 8;
  if (tid < NLOC) cnt[tid] = 0;
  __syncthreads();
  for (int i = tid; i < R; i += 256) { const unsigned e = ids[i] - lo; if (e < NLOC) atomicAdd(&cnt[e], 1); }
  __syncthreads();
  if (tid == 0) { long long o = 0; for (int e = 0; e < NLOC; e++) { npad[e] = (cnt[e] + 7) & ~7; xo[e] = o; o += (long long)npad[e] * NCH * 128; } }
  __syncthreads();
  const int i = blockIdx.x * 8 + wid;
  if (i >= R) return;
  const unsigned e = ids[i] - lo;
  if (e >= NLOC) return;
  int pos = 0;
  for (int j0 = 0; j0 < i; j0 += 32) {
    const int j = j0 + lane;
    const bool m = j < i && (unsigned)(ids[j] - lo) == e;
    pos += __popc(__ballot_sync(0xffffffffu, m));
  }
  const uint32_t* src = (const uint32_t*)(x + (size_t)(i >> 3) * HD);
  uint8_t* dst = xg + xo[e];
  for (int q = lane; q < HD / 2; q += 32) {
    const uint32_t v = __ldg(src + q);
    *(uint32_t*)(dst + xaddr_of(kperm(q * 2), pos, npad[e])) = v;
  }
}

#else
// gather: route i of a local expert e -> row pos(i) of e's block, K in the wgmma order, 16-B chunks swizzled.
// Every block ranks all routes once in smem (same stable order as moe_x's lists), then copies its share of rows.
#define XGT 1024
#define XGMAXR 4096
__global__ void __launch_bounds__(XGT) xgather(const __nv_bfloat16* x, const int* ids, int M, int lo, uint8_t* xg) {
#if defined(PDLX) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
  __shared__ int wc[XGT / 32][NLOC];
  __shared__ int base[NLOC], npad[NLOC];
  __shared__ long long xo[NLOC];
  __shared__ uint16_t pos_s[XGMAXR];
  __shared__ uint8_t e_s[XGMAXR];
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, R = M * 8;
  if (tid < NLOC) base[tid] = 0;
  for (int i = tid; i < R; i += XGT) { const unsigned e = ids[i] - lo; e_s[i] = e < NLOC ? e : 0xff; }
  __syncthreads();
  const unsigned ltm = (1u << lane) - 1;
  for (int c0 = 0; c0 < R; c0 += XGT) {
    const int i = c0 + tid;
    const int e = i < R ? e_s[i] : 0xff;
    const unsigned m = __match_any_sync(0xffffffffu, e);
    wc[wid][lane] = 0;
    __syncwarp();
    if (e < NLOC && (m & ltm) == 0) wc[wid][e] = __popc(m);
    __syncthreads();
    if (tid < NLOC) {
      int s = base[tid];
      for (int w = 0; w < XGT / 32; w++) { const int v = wc[w][tid]; wc[w][tid] = s; s += v; }
      base[tid] = s;
    }
    __syncthreads();
    if (e < NLOC) pos_s[i] = wc[wid][e] + __popc(m & ltm);
    __syncthreads();
  }
  if (tid == 0) { long long o = 0; for (int e = 0; e < NLOC; e++) { npad[e] = (base[e] + 7) & ~7; xo[e] = o; o += (long long)npad[e] * NCH * 128; } }
  __syncthreads();
#ifdef XGROW
  constexpr int SEG = 1;
#else
  constexpr int SEG = 4;  // quarter rows: R * 4 items spread over every warp of the grid
#endif
  for (int w = blockIdx.x * (XGT / 32) + wid; w < R * SEG; w += gridDim.x * (XGT / 32)) {
    const int i = w / SEG, sg = w - i * SEG;
    const int e = e_s[i];
    if (e >= NLOC) continue;
    const int pos = pos_s[i], np = npad[e];
    const uint4* src = (const uint4*)(x + (size_t)(i >> 3) * HD);
    uint8_t* dst = xg + xo[e];
#pragma unroll 4
    for (int q4 = sg * (HD / 8 / SEG) + lane; q4 < (sg + 1) * (HD / 8 / SEG); q4 += 32) {
      const uint4 v = __ldg(src + q4);
      const uint32_t w4[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int k = 0; k < 4; k++) *(uint32_t*)(dst + xaddr_of(kperm((q4 * 4 + k) * 2), pos, np)) = w4[k];
    }
  }
}
#endif

template <int N>
__device__ __forceinline__ void epilogue(const P& p, const Sm* sm, const uint16_t* list, float* acc, const Hdr& h, int mt, int lane) {
  const int g = lane >> 2, t = lane & 3, e = h.e, ntok = h.ntok;
  const uint16_t* lst = list + sm->off[e] + h.ps * TPX;
  if (!h.g) {
    const float gs = p.g1[p.lo + e];
    uint8_t* ib = p.ig + sm->xoff[e];
    const int pc = kperm(mt * 8 + g), npad = sm->npad[e];
#pragma unroll
    for (int i = 0; i < N / 8; i++)
#pragma unroll
      for (int hh = 0; hh < 2; hh++) {
        const int tok = i * 8 + 2 * t + hh;
        const float gt = acc[i * 4 + hh] * gs, up = acc[i * 4 + 2 + hh] * gs;
        if (tok < ntok) *(__nv_bfloat16*)(ib + xaddr_of(pc, h.ps * TPX + tok, npad)) = __float2bfloat16(strict_swiglu(gt, up));
      }
  } else {
    const float gs = p.g2[p.lo + e];
    const float n0 = p.nw[mt * 16 + g], n1 = p.nw[mt * 16 + g + 8];
#pragma unroll
    for (int i = 0; i < N / 8; i++) {
      // Complete W2, apply its own expert's global scale, then round BEFORE norm.
      const float q0=__bfloat162float(__float2bfloat16(acc[i*4]*gs)), q1=__bfloat162float(__float2bfloat16(acc[i*4+1]*gs)), q2=__bfloat162float(__float2bfloat16(acc[i*4+2]*gs)), q3=__bfloat162float(__float2bfloat16(acc[i*4+3]*gs));
      const float v0=q0*n0, v1=q1*n0, v2=q2*n1, v3=q3*n1;
      float sa = v0 * v0 + v2 * v2, sb = v1 * v1 + v3 * v3;
      const int t0 = i * 8 + 2 * t;
      if (t0 < ntok) { __nv_bfloat16* yr = p.y + (size_t)lst[t0] * HD + mt * 16 + g; yr[0] = __float2bfloat16(q0); yr[8] = __float2bfloat16(q2); }
      if (t0 + 1 < ntok) { __nv_bfloat16* yr = p.y + (size_t)lst[t0 + 1] * HD + mt * 16 + g; yr[0] = __float2bfloat16(q1); yr[8] = __float2bfloat16(q3); }
#pragma unroll
      for (int o = 4; o < 32; o <<= 1) { sa += __shfl_xor_sync(0xffffffffu, sa, o); sb += __shfl_xor_sync(0xffffffffu, sb, o); }
      if (g == 0) {
        if (t0 < ntok) atomicAdd(p.ss + lst[t0], sa);
        if (t0 + 1 < ntok) atomicAdd(p.ss + lst[t0 + 1], sb);
      }
    }
  }
}

// one group slot (4 x 64-K chunks) of a unit on the consumer side. Per k16 step (CUTLASS mixed-input RS order):
// wait<3> retires the group that last used this step's A / descriptor registers, dequant the RPW blocks, fence, RPW
// wgmmas, commit. The previous slot is released at step 3 of this slot's first chunk, once all its groups retired.
// FG: the unit's first slot (scale-d 0 on its first step, nothing to release).
template <int N, bool FG>
__device__ __forceinline__ void group(float (&acc)[RPW][N / 2], uint32_t (&A)[RPW][4][4], uint64_t (&db)[4], int& qs, int& ps_,
                                      uint32_t sbase, uint32_t full, uint32_t empty, int wid, int lane, long long& wcy) {
  const int s = qs % NSL;
#ifdef WSTAT
  const long long w0 = clock64();
#endif
  mbwait(full + s * 8, (qs / NSL) & 1);
#ifdef WSTAT
  wcy += clock64() - w0;
#endif
  qs++;
  const uint32_t sb = sbase + s * GSB;
#pragma unroll
  for (int cc = 0; cc < WCH; cc++) {
    const uint32_t xa = sb + cc * XB;
    uint32_t ww[RPW][4], sA[RPW], sB[RPW];
#pragma unroll
    for (int r = 0; r < RPW; r++) {
      const uint32_t wb = sb + ASB + (((wid >> 3) * RPW + r) * WCH + cc) * WREC + (wid & 7) * REC;
      const uint4 w = lds128(wb + lane * 16);
      const uint32_t s16 = lds16(wb + 512 + lane * 2);
      sA[r] = sc2(s16, 0x4040); sB[r] = sc2(s16, 0x4141);
      ww[r][0] = w.x; ww[r][1] = w.y; ww[r][2] = w.z; ww[r][3] = w.w;
    }
#pragma unroll
    for (int j = 0; j < 4; j++) {
      wg_wait<3>();
      if (!FG && cc == 0 && j == 3) mbarrive(empty + ps_ * 8);  // every consumer lane arrives (count NCW * 32)
#pragma unroll
      for (int r = 0; r < RPW; r++) {
        const uint32_t qq = ww[r][j];
#if defined(NOCOMP)  // NOCOMP: no dequant, no mma (stream only); NOMMA: real dequant, no mma
        A[r][j][0] = qq ^ sA[r]; A[r][j][1] = qq ^ sB[r]; A[r][j][2] = qq + sA[r]; A[r][j][3] = qq + sB[r];
#else
        A[r][j][0] = bmul(qq & 0x81C081C0u, sA[r]);
        A[r][j][1] = bmul(__funnelshift_l(qq, qq, 10) & 0x81C081C0u, sB[r]);
        A[r][j][2] = bmul(__funnelshift_l(qq, qq, 13) & 0x81C081C0u, sA[r]);
        A[r][j][3] = bmul(((qq & 0x001B001Bu) * 0x1084u) & 0x81C081C0u, sB[r]);
#endif
      }
      db[j] = gdesc(xa + j * 32);
#if defined(NOMMA) || defined(NOCOMP)
#pragma unroll
      for (int r = 0; r < RPW; r++) acc[r][j] += __uint_as_float((A[r][j][0] ^ A[r][j][1] ^ A[r][j][2] ^ A[r][j][3] ^ (uint32_t)db[j]) & 0x3fffffffu);
#else
#pragma unroll
      for (int r = 0; r < RPW; r++) fence_acc<N>(acc[r]);
      wg_fence();
#pragma unroll
      for (int r = 0; r < RPW; r++) wg_rs<N>(acc[r], A[r][j], db[j], (FG && cc == 0 && j == 0) ? 0 : 1);
      wg_commit();
#endif
    }
  }
  ps_ = s;
}

#ifdef WXG2
// WXG2: one wgmma group per 64-K chunk instead of per k16 step. The chunk's 4 words are dequantized into one of two A register
// sets (chunk parity), then fence, 4 x RPW wgmmas, commit, wait_group 1 (retires the previous chunk: its A set is free for the
// next chunk, which is dequantized while this chunk's wgmmas run). One descriptor per slot (+ constant chunk / step offsets),
// one warpgroup arrive / depbar per chunk: ~6 fewer instructions per word than the per-k16-step order.
template <int N, bool FG>
__device__ __forceinline__ void group2(float (&acc)[RPW][N / 2], uint32_t (&A)[2][RPW][4][4], int& qs, int& ps_,
                                       uint32_t sbase, uint32_t full, uint32_t empty, int wid, int lane, long long& wcy) {
  const int s = qs % NSL;
#ifdef WSTAT
  const long long w0 = clock64();
#endif
  mbwait(full + s * 8, (qs / NSL) & 1);
#ifdef WSTAT
  wcy += clock64() - w0;
#endif
  qs++;
#ifdef UDESC
  const uint32_t sb = sbase + __reduce_max_sync(0xffffffffu, (unsigned)s) * GSB;  // warp-uniform slot base: descriptors on the uniform datapath
#else
  const uint32_t sb = sbase + s * GSB;
#endif
  const uint64_t d0 = gdesc(sb);
#pragma unroll
  for (int cc = 0; cc < WCH; cc++) {
    uint32_t (&As)[RPW][4][4] = A[cc & 1];
#pragma unroll
    for (int r = 0; r < RPW; r++) {
      const uint32_t wb = sb + ASB + (((wid >> 3) * RPW + r) * WCH + cc) * WREC + (wid & 7) * REC;
      const uint4 w = lds128(wb + lane * 16);
      const uint32_t s16 = lds16(wb + 512 + lane * 2);
      const uint32_t sA = sc2(s16, 0x4040), sB = sc2(s16, 0x4141);
      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const uint32_t qq = ww[j];
#if defined(NOCOMP)
        As[r][j][0] = qq ^ sA; As[r][j][1] = qq ^ sB; As[r][j][2] = qq + sA; As[r][j][3] = qq + sB;
#else
        As[r][j][0] = bmul(qq & 0x81C081C0u, sA);
        As[r][j][1] = bmul(__funnelshift_l(qq, qq, 10) & 0x81C081C0u, sB);
        As[r][j][2] = bmul(__funnelshift_l(qq, qq, 13) & 0x81C081C0u, sA);
        As[r][j][3] = bmul(((qq & 0x001B001Bu) * 0x1084u) & 0x81C081C0u, sB);
#endif
      }
    }
#if defined(NOMMA) || defined(NOCOMP)
#pragma unroll
    for (int r = 0; r < RPW; r++)
#pragma unroll
      for (int j = 0; j < 4; j++) acc[r][j] += __uint_as_float((As[r][j][0] ^ As[r][j][1] ^ As[r][j][2] ^ As[r][j][3] ^ (uint32_t)gdesc_add(d0, cc * XB + j * 32)) & 0x3fffffffu);
#else
#pragma unroll
    for (int r = 0; r < RPW; r++) fence_acc<N>(acc[r]);
    wg_fence();
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
      for (int r = 0; r < RPW; r++) wg_rs<N>(acc[r], As[r][j], gdesc_add(d0, cc * XB + j * 32), (FG && cc == 0 && j == 0) ? 0 : 1);
    wg_commit();
    wg_wait<1>();
#endif
    if (!FG && cc == 0) mbarrive(empty + ps_ * 8);  // the previous slot's last chunk has retired (every consumer lane arrives)
  }
  ps_ = s;
}
#endif

template <int N>
__device__ __forceinline__ void unit(const P& p, const Sm* sm, const uint16_t* list, uint32_t sbase, uint32_t full, uint32_t empty,
                                     int& qs, int wid, int lane, const Hdr& h0, long long& wcy) {
  float acc[RPW][N / 2];
  uint32_t A[RPW][4][4];
  uint64_t db[4] = {0, 0, 0, 0};
  int ps_ = 0;
#if defined(NOMMA) || defined(NOCOMP)
#pragma unroll
  for (int r = 0; r < RPW; r++)
#pragma unroll
    for (int i = 0; i < N / 2; i++) acc[r][i] = 0.f;
#endif
#ifdef WXG2
  uint32_t A2[2][RPW][4][4];
  group2<N, true>(acc, A2, qs, ps_, sbase, full, empty, wid, lane, wcy);
#pragma unroll 1
  for (int c4 = 1; c4 < NCH / WCH; c4++) group2<N, false>(acc, A2, qs, ps_, sbase, full, empty, wid, lane, wcy);
#else
  group<N, true>(acc, A, db, qs, ps_, sbase, full, empty, wid, lane, wcy);
#pragma unroll 1
  for (int c4 = 1; c4 < NCH / WCH; c4++) group<N, false>(acc, A, db, qs, ps_, sbase, full, empty, wid, lane, wcy);
#endif
  wg_wait<0>();
#pragma unroll
  for (int r = 0; r < RPW; r++) fence_acc<N>(acc[r]);
  mbarrive(empty + ps_ * 8);
#ifndef NOCOMP
#pragma unroll
  for (int r = 0; r < RPW; r++) epilogue<N>(p, sm, list, acc[r], h0, ((h0.rb * UBL) + (wid >> 3) * RPW + r) * TPU + (wid & 7), lane);
#else
  if (acc[0][0] == 1234.5f) p.y[0] = __float2bfloat16(acc[0][1]);
#endif
  __syncwarp();
#if TMA
  if (!h0.g) asm volatile("fence.proxy.async.global;\n" ::: "memory");  // the W2 producer reads inter with TMA
#endif
  __syncwarp();
  if (lane == 0) red_rel(p.ctr + (h0.g ? C_W2 : h0.e), 1);
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

__global__ void __launch_bounds__(NTH, 1) moe_x(P p) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, R = p.M * 8;
  Sm* sm = (Sm*)(smem + RING + 2 * NSL * 8);
  uint16_t* list = (uint16_t*)(sm + 1);
  uint8_t* lid = smem;
  uint16_t* posi = (uint16_t*)(smem + 4096);
  const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem);
  const uint32_t full = sbase + RING, empty = full + NSL * 8;
#ifdef PROF
  PSET(0, gtime()); PSET(8, clock64());
#endif
  if (tid < NSL) { mbinit(full + tid * 8, TMA ? 1 : 32); mbinit(empty + tid * 8, NCW * 32); }
  if (tid < NLOC) sm->base[tid] = 0;
#if TMA
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
  int u = 0, nx = 0;
  if (tid == NCW * 32) u = atomicAdd(p.ctr + C_Q, 1);
  for (int i = tid; i < R; i += NTH) { unsigned e = p.ids[i] - p.lo; lid[i] = e < NLOC ? e : 0xff; }
  __syncthreads();
  const unsigned ltm = (1u << lane) - 1;
  for (int c0 = 0; c0 < R; c0 += NTH) {  // stable per-expert lists (route order), identical in every CTA and in xgather
    const int i = c0 + tid;
    const int e = i < R ? lid[i] : 0xff;
    const unsigned m = __match_any_sync(0xffffffffu, e);
    sm->wc[wid][lane] = 0;
    __syncwarp();
    if (e < NLOC && (m & ltm) == 0) sm->wc[wid][e] = __popc(m);
    __syncthreads();
    if (tid < NLOC) { int s = sm->base[tid];
      for (int w = 0; w < NTH / 32; w++) { const int v = sm->wc[w][tid]; sm->wc[w][tid] = s; s += v; }
      sm->base[tid] = s; }
    __syncthreads();
    if (e < NLOC) posi[i] = sm->wc[wid][e] + __popc(m & ltm);
    __syncthreads();
  }
  if (wid == 0) {
    const int c = sm->base[lane];
    int v = c;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { int n = __shfl_up_sync(0xffffffffu, v, o); if (lane >= o) v += n; }
    sm->cnt[lane] = c; sm->off[lane] = v - c; sm->np[lane] = (c + TPX - 1) / TPX;
    const int pd = (c + 7) & ~7;
    int vp = pd;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { int n = __shfl_up_sync(0xffffffffu, vp, o); if (lane >= o) vp += n; }
    sm->npad[lane] = pd; sm->xoff[lane] = (long long)(vp - pd) * NCH * 128;
    int rk = 0;
#pragma unroll
    for (int j = 0; j < 32; j++) { const int cj = __shfl_sync(0xffffffffu, c, j); rk += (cj > c) || (cj == c && j < lane); }
    if (c > 0) sm->act[rk] = lane;
    const unsigned am = __ballot_sync(0xffffffffu, c > 0);
    int w2 = c > 0 ? NRB2 * ((c + TPX - 1) / TPX) * NCW : 0;
#pragma unroll
    for (int o = 16; o; o >>= 1) w2 += __shfl_xor_sync(0xffffffffu, w2, o);
    __syncwarp();
    if (lane == 0) {
      const int na = __popc(am);
      int ns = 0, uu = 0;
      for (int g = 0; g < 2; g++)  // all W13 units (heaviest expert first), then all W2 units
        for (int a = 0; a < na; a++) { sm->segst[ns] = uu; sm->segk[ns] = g * 64 + a; uu += (g ? NRB2 : NRB1) * sm->np[sm->act[a]]; ns++; }
      sm->segst[ns] = uu; sm->nseg = ns; sm->ntot = uu; sm->na = na; sm->nw2 = w2;
    }
  }
  __syncthreads();
  for (int i = tid; i < R; i += NTH) { const int e = lid[i]; if (e < NLOC) list[sm->off[e] + posi[i]] = i; }
  __syncthreads();
#ifdef PROF
  PSET(1, gtime());
#endif
  const int ntot = sm->ntot;
  long long wcy = 0;  // consumer slot-wait cycles (WSTAT)
#ifdef WSTAT
  const long long ct0 = clock64();
  unsigned long long gt0; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt0));
#endif
  if (wid == NCW) {  // producer warp
    u = __shfl_sync(0xffffffffu, u, 0);
    unsigned depok = 0;
    int sq = 0, prew = 0;
#if defined(PDLX) && TMA
    if (u < ntot) {
      int g, e, rb, ps; decode(sm, u, g, e, rb, ps);
      const int ntok = min(TPX, sm->cnt[e] - ps * TPX), rows = (ntok + 7) & ~7;
      const uint8_t* wsrc = (g ? p.w2 + ((size_t)e * 32 + rb * UBL) * NCH * WREC : p.w1 + ((size_t)e * 64 + rb * UBL) * NCH * WREC);
      for (int c = 0; c < NCH && prew < NSL; c += WCH, prew++) {
        const uint32_t dst = sbase + prew * GSB, fb = full + prew * 8;
        if (lane == 0) {
          expect_tx(fb, WCH * rows * 128 + WSB);
#pragma unroll
          for (int b2 = 0; b2 < UBL; b2++) bulk(dst + ASB + b2 * WCH * WREC, wsrc + ((size_t)b2 * NCH + c) * WREC, WCH * WREC, fb);
        }
      }
    }
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
#ifdef WSTAT
    long long pw = 0;
    const long long pt0 = clock64();
#endif
    for (; u < ntot;) {
      int g, e, rb, ps; decode(sm, u, g, e, rb, ps);
      const int ntok = min(TPX, sm->cnt[e] - ps * TPX), rows = (ntok + 7) & ~7;
      const uint8_t* wsrc = (g ? p.w2 + ((size_t)e * 32 + rb * UBL) * NCH * WREC : p.w1 + ((size_t)e * 64 + rb * UBL) * NCH * WREC);
      const uint8_t* xsrc = (g ? p.ig : p.xg) + sm->xoff[e] + (size_t)ps * TPX * 128;
      const int npad = sm->npad[e];
      if (g && !(depok >> e & 1)) {  // W2 needs every W13 unit of e (all its inter columns)
        if (lane == 0) { const int tgt = NRB1 * sm->np[e] * NCW; while (ldacq(p.ctr + e) < tgt) __nanosleep(64); }
        __syncwarp();
#if TMA
        asm volatile("fence.proxy.async.global;\n" ::: "memory");
#endif
        depok |= 1u << e;
      }
      if (lane == 0) nx = atomicAdd(p.ctr + C_Q, 1);
      for (int c = 0; c < NCH; c += WCH, sq++) {
        const int s = sq % NSL;
#ifdef WSTAT
        const long long w0 = clock64();
#endif
        if (sq >= NSL) mbwait(empty + s * 8, ((sq / NSL) - 1) & 1);
#ifdef WSTAT
        pw += clock64() - w0;
#endif
        if (lane == 0) { sm->hdr[s] = Hdr{g, e, rb, ps, c, ntok, (c == 0 ? 1 : 0) | ((!g && rb == 0) ? 4 : 0), 0}; __threadfence_block(); }
        const uint32_t dst = sbase + s * GSB, fb = full + s * 8;
#if TMA
        if (lane == 0) {
          uint32_t bytes = 0;
#ifndef NOACT
          bytes += WCH * rows * 128;
#endif
#ifndef NOWGT
          bytes += WSB;
#endif
          if (sq >= prew) expect_tx(fb, bytes);
#ifndef NOACT
#pragma unroll
          for (int cc = 0; cc < WCH; cc++) bulk(dst + cc * XB, xsrc + (size_t)(c + cc) * npad * 128, rows * 128, fb);
#endif
#ifndef NOWGT
          if (sq >= prew)
#pragma unroll
          for (int b2 = 0; b2 < UBL; b2++) bulk(dst + ASB + b2 * WCH * WREC, wsrc + ((size_t)b2 * NCH + c) * WREC, WCH * WREC, fb);
#endif
        }
#else
        for (int cc = 0; cc < WCH; cc++)
          for (int k = lane; k < rows * 8; k += 32) cp16(dst + cc * XB + k * 16, xsrc + (size_t)(c + cc) * npad * 128 + k * 16);
        for (int b2 = 0; b2 < UBL; b2++)
          for (int k = lane; k < WCH * WREC / 16; k += 32) cp16(dst + ASB + b2 * WCH * WREC + k * 16, wsrc + ((size_t)b2 * NCH + c) * WREC + k * 16);
        cpa_arrive(fb);
#endif
      }
      u = __shfl_sync(0xffffffffu, nx, 0);
    }
#ifdef WSTAT
    if (lane == 0 && p.prof) { atomicAdd(p.prof + 2, (unsigned long long)pw); atomicAdd(p.prof + 3, (unsigned long long)(clock64() - pt0)); }
#endif
    const int s = sq % NSL;
    if (sq >= NSL) mbwait(empty + s * 8, ((sq / NSL) - 1) & 1);
    if (lane == 0) sm->hdr[s].g = -1;
#if TMA
    if (lane == 0) mbarrive(full + s * 8);
#else
    cpa_arrive(full + s * 8);
#endif
#ifdef PROF
    PSET(2, gtime());
#endif
  } else {  // consumers: two warpgroups, warp wid owns tile wid of every unit
    int qs = 0;
    for (;;) {
      const int s = qs % NSL;
      mbwait(full + s * 8, (qs / NSL) & 1);
      const Hdr h = sm->hdr[s];
      if (h.g < 0) break;
      if ((h.flags & 4) && wid == 0) for (int i = lane; i < h.ntok; i += 32) p.ss[list[sm->off[h.e] + h.ps * TPX + i]] = 0.f;
      const int n8 = (h.ntok + 7) >> 3;
#define U_(n) unit<n>(p, sm, list, sbase, full, empty, qs, wid, lane, h, wcy)
      if (n8 <= 1) U_(8);
      else if (n8 <= 2) U_(16);
#ifdef N8STEP
      else if (n8 <= 3) U_(24);
      else if (n8 <= 4) U_(32);
      else if (n8 <= 6) { if constexpr (TPX >= 64) U_(48); }
#else
      else if (n8 <= 4) U_(32);
#endif
      else if (n8 <= 8) { if constexpr (TPX >= 64) U_(64); }
      else if (n8 <= 16) { if constexpr (TPX >= 128) U_(128); }
      else { if constexpr (TPX >= 256) U_(256); }
    }
#ifdef PROF
    PSET(2, gtime());
#endif
  }
#ifdef WSTAT
  if (wid < NCW && lane == 0 && p.prof) {
    unsigned long long gt1; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt1));
    atomicAdd(p.prof + 0, (unsigned long long)wcy); atomicAdd(p.prof + 1, (unsigned long long)(clock64() - ct0)); atomicAdd(p.prof + 4, gt1 - gt0);
  }
#endif
  const int nw2 = sm->nw2;
  if (lane == 0) while (ldacq(p.ctr + C_W2) < nw2) __nanosleep(32);
  __syncwarp();
#ifdef PROF
  PSET(6, gtime());
#endif
  for (int it = blockIdx.x * (NTH / 32) + wid; it < p.M * 16; it += gridDim.x * (NTH / 32)) {
    const int t = it >> 4, col = (it & 15) * 256 + lane * 8;
    float a[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int k = 0; k < 8; k++) {
      const int r = t * 8 + k;
      unsigned e = p.ids[r] - p.lo;
      if (e < NLOC) {
        float cf = p.tw[r] * rsqrtf(__ldcg(p.ss + r) * (1.f / HD) + 1e-8f);
        uint4 v = __ldcg((const uint4*)(p.y + (size_t)r * HD + col));
        const __nv_bfloat162* hh = (const __nv_bfloat162*)&v;
#pragma unroll
        for (int j = 0; j < 4; j++) { float2 f = __bfloat1622float2(hh[j]); a[2 * j] += cf * f.x; a[2 * j + 1] += cf * f.y; }
      }
    }
    uint4 o; __nv_bfloat162* ob = (__nv_bfloat162*)&o;
#pragma unroll
    for (int j = 0; j < 4; j++) ob[j] = __floats2bfloat162_rn(a[2*j]*p.nw[col+2*j], a[2*j+1]*p.nw[col+2*j+1]);
    *(uint4*)(p.out + (size_t)t * HD + col) = o;
  }
#ifdef PROF
  PSET(7, gtime()); PSET(9, clock64());
#endif
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    if (atomicAdd(p.ctr + C_DONE, 1) == (int)gridDim.x - 1) {
      for (int i = 0; i <= C_DONE; i++) p.ctr[i] = 0;
      __threadfence();
    }
  }
}

static int g_gridX = 0;
static size_t smem_x(int M) { return (size_t)RING + 2 * NSL * 8 + sizeof(Sm) + (size_t)M * 8 * 2 + 1024; }

void moe(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor w1, torch::Tensor w2, torch::Tensor g1,
         torch::Tensor g2, torch::Tensor nw, torch::Tensor xg, torch::Tensor ig, torch::Tensor y, torch::Tensor ss,
         torch::Tensor ctr, torch::Tensor out, int64_t lo, torch::Tensor prof) {
  const int M = x.size(0);
  if (!g_gridX) {
    int dev; cudaGetDevice(&dev); int mx; cudaDeviceGetAttribute(&mx, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaDeviceGetAttribute(&g_gridX, cudaDevAttrMultiProcessorCount, dev);
    cudaFuncSetAttribute(moe_x, cudaFuncAttributeMaxDynamicSharedMemorySize, mx);
  }
  TORCH_CHECK(M * 8 <= 4096 && M * 8 * 3 + 4096 <= RING, "M too large");
  auto st = at::cuda::getCurrentCUDAStream();
  #ifdef XGOLD
  xgather<<<(M * 8 + 7) / 8, 256, 0, st>>>(
#else
  xgather<<<g_gridX, XGT, 0, st>>>(
#endif(const __nv_bfloat16*)x.data_ptr(), ids.data_ptr<int>(), M, (int)lo, xg.data_ptr<uint8_t>());
  P p{w1.data_ptr<uint8_t>(), w2.data_ptr<uint8_t>(), g1.data_ptr<float>(), g2.data_ptr<float>(),
      xg.data_ptr<uint8_t>(), ig.data_ptr<uint8_t>(), ids.data_ptr<int>(), tw.data_ptr<float>(), nw.data_ptr<float>(),
      (__nv_bfloat16*)y.data_ptr(), ss.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(), ctr.data_ptr<int>(), M, (int)lo,
      prof.numel() ? (unsigned long long*)prof.data_ptr() : nullptr};
#ifdef PDLX
  cudaLaunchConfig_t lc = {};
  lc.gridDim = dim3(g_gridX); lc.blockDim = dim3(NTH); lc.dynamicSmemBytes = smem_x(M); lc.stream = st;
  cudaLaunchAttribute la[1];
  la[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; la[0].val.programmaticStreamSerializationAllowed = 1;
  lc.attrs = la; lc.numAttrs = 1;
  TORCH_CHECK(cudaLaunchKernelEx(&lc, moe_x, p) == cudaSuccess, "moe_x PDL launch failed");
#else
  moe_x<<<g_gridX, NTH, smem_x(M), st>>>(p);
#endif
}
