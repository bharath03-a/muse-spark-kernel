
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <cstring>
#include <cuda.h>
#include <cudaTypedefs.h>
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define HAS_WG 1
#else
#define HAS_WG 0
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define TMA 1
#else
#define TMA 0
#endif
#ifndef BNMAX
#define BNMAX 128
#endif
#ifndef EMU_NS
#define EMU_NS 0
#endif
#define WCH 12288
#define KST 64
struct Hdr { int tile, e, rb, row0, ntok, kh, nks, pad; };
struct P { const __nv_bfloat16* A; const int* offs; const uint8_t* W; const float* g; __nv_bfloat16* out; int* ctr; unsigned long long* prof;
           int nrb256, nout, rows, ksplit, tsz, lomod, fullonly, fuse; };

__device__ __forceinline__ void mbinit(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(b), "r"(n)); }
__device__ __forceinline__ void mbarrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(b) : "memory"); }
#ifndef WDOG
#define WDOG (TMA ? (1u << 22) : (1u << 28))
#endif
__device__ int g_w4err;
__device__ __forceinline__ void mbwait(uint32_t b, uint32_t ph) {
  for (uint32_t i = 0;; i++) {
    uint32_t ok;
#if TMA
    asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(b), "r"(ph) : "memory");
#else
    asm volatile("{\n.reg .pred p;\nmbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(b), "r"(ph) : "memory");
#endif
    if (ok) return;
    if ((i & 255) == 255) {
      if (*(volatile int*)&g_w4err) return;
      if (i > WDOG) { atomicExch(&g_w4err, 1); return; }
    }
  }
}
__device__ __forceinline__ unsigned long long clk() { unsigned long long t; asm volatile("mov.u64 %0, %clock64;" : "=l"(t)); return t; }
__device__ __forceinline__ unsigned long long gns() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t lds32(uint32_t a) { uint32_t v; asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(a)); return v; }
// W4PROF phase accumulators (clock64 cycles), 16 per kernel class: consumer warps (lane 0) 0 wait-full, 1 dequant (smem loads + ALU),
// 2 wgmma fence -> issue -> wait_group 0, 3 epilogue, 4 stages, 5 span; producer (thread 0) 8 wait-empty, 9 issue (hdr + TMA + cp.async +
// commit), 10 TMA wait at the full arrival (exposed TMA latency), 11 cp.async wait + fence + arrive, 12 stages, 13 span
struct Pc { unsigned long long v[6]; };
#ifdef W4PROF
#define PWAIT(acc, b, ph) do { const unsigned long long c0_ = clk(); mbwait(b, ph); acc += clk() - c0_; } while (0)
#define PT(x) const unsigned long long x = clk()
#define PADD(pc, i, d) (pc).v[i] += (d)
#else
#define PWAIT(acc, b, ph) mbwait(b, ph)
#define PT(x)
#define PADD(pc, i, d)
#endif
__device__ __forceinline__ void cp16(uint32_t s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(g) : "memory"); }
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }
__device__ __forceinline__ void fence_async() {
#if TMA
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#endif
}
#if TMA
__device__ __forceinline__ void expect_tx(uint32_t bar, uint32_t n) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "r"(n) : "memory"); }
__device__ __forceinline__ void bulk(uint32_t dst, const void* src, uint32_t n, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(dst), "l"(src), "r"(n), "r"(bar) : "memory");
}
#endif
template <int NPT> __device__ __forceinline__ void pbar() { asm volatile("bar.sync 1, %0;\n" :: "n"(NPT) : "memory"); }
__device__ __forceinline__ uint32_t swz(uint32_t a) { return a ^ (((a >> 7) & 7) << 4); }  // SW128 on the absolute smem address
__device__ __forceinline__ uint32_t bmul(uint32_t a, uint32_t b) { uint32_t d; asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u)); return d; }
__device__ __forceinline__ uint32_t sc2(uint32_t w, uint32_t sel) { return __byte_perm(w, 0, sel) * 16u + 0x77007700u; }
__device__ __forceinline__ uint64_t gdesc(uint32_t saddr) {  // K-major, SW128, LBO 16 (unused), SBO 1024
  return (uint64_t)((saddr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)64 << 32) | (1ull << 62);
}
#if !HAS_WG
__device__ __forceinline__ uint32_t emu_kmaj(uint64_t desc, int r, int kb) {
  const uint32_t st = (uint32_t)(desc & 0x3FFF) << 4, sbo = (uint32_t)((desc >> 32) & 0x3FFF) << 4;
  return swz(st + (r >> 3) * sbo + (r & 7) * 128 + kb);
}
__device__ __forceinline__ float emu_ldb(uint32_t a) { unsigned short v; asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(a)); return __uint_as_float((uint32_t)v << 16); }
__device__ __forceinline__ float bf_lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf_hi(uint32_t v) { return __uint_as_float(v & 0xFFFF0000u); }
// RS emulation: the warpgroup's A fragments (m64 x k16) go through a per-warpgroup smem scratch; warps of a warpgroup run in lock step
// at the named barrier (id 8 + warpgroup)
template <int N>
__device__ __noinline__ void wg_emu_rs(float* d, const uint32_t* a, uint64_t db, int sd, float* scr) {
  const int wt = threadIdx.x & 127, w = wt >> 5, g = (wt & 31) >> 2, t = wt & 3;
  asm volatile("bar.sync %0, 128;" :: "r"(8 + (int)(threadIdx.x >> 7)) : "memory");
  const int r0 = 16 * w + g, r1 = r0 + 8;
  scr[r0 * 16 + 2 * t] = bf_lo(a[0]); scr[r0 * 16 + 2 * t + 1] = bf_hi(a[0]);
  scr[r1 * 16 + 2 * t] = bf_lo(a[1]); scr[r1 * 16 + 2 * t + 1] = bf_hi(a[1]);
  scr[r0 * 16 + 2 * t + 8] = bf_lo(a[2]); scr[r0 * 16 + 2 * t + 9] = bf_hi(a[2]);
  scr[r1 * 16 + 2 * t + 8] = bf_lo(a[3]); scr[r1 * 16 + 2 * t + 9] = bf_hi(a[3]);
  asm volatile("bar.sync %0, 128;" :: "r"(8 + (int)(threadIdx.x >> 7)) : "memory");
  for (int i = 0; i < N / 2; i++) {
    const int hh = (i >> 1) & 1, e = i & 1, j = i >> 2, m = 16 * w + g + 8 * hh, n = 8 * j + 2 * t + e;
    float acc = 0.f;
    for (int k = 0; k < 16; k++) acc += scr[m * 16 + k] * emu_ldb(emu_kmaj(db, n, 2 * k));
    d[i] = sd ? d[i] + acc : acc;
  }
}
#endif
__device__ __forceinline__ void wg_fence() {
#if HAS_WG
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void wg_commit() {
#if HAS_WG
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#endif
}
template <int N> __device__ __forceinline__ void wg_wait() {
#if HAS_WG
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
#endif
}
template <int N> __device__ __forceinline__ void wgm(float* d, const uint32_t* a, uint64_t db, int sd, float* scr);
template <> __device__ __forceinline__ void wgm<32>(float* d, const uint32_t* a, uint64_t db, int sd, float* scr) {
#if HAS_WG
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %21, 0;\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, p, 1, 1, 0;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(sd));
#else
  wg_emu_rs<32>(d, a, db, sd, scr);
#endif
}
template <> __device__ __forceinline__ void wgm<64>(float* d, const uint32_t* a, uint64_t db, int sd, float* scr) {
#if HAS_WG
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1, 0;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(sd));
#else
  wg_emu_rs<64>(d, a, db, sd, scr);
#endif
}
template <> __device__ __forceinline__ void wgm<128>(float* d, const uint32_t* a, uint64_t db, int sd, float* scr) {
#if HAS_WG
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %69, 0;\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, {%64,%65,%66,%67}, %68, p, 1, 1, 0;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(sd));
#else
  wg_emu_rs<128>(d, a, db, sd, scr);
#endif
}
template <> __device__ __forceinline__ void wgm<256>(float* d, const uint32_t* a, uint64_t db, int sd, float* scr) {
#if HAS_WG
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %133, 0;\nwgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127}, {%128,%129,%130,%131}, %132, p, 1, 1, 0;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(sd));
#else
  wg_emu_rs<256>(d, a, db, sd, scr);
#endif
}

template <int NT, int NCW> struct Cfg {
  static constexpr int NPT = NCW >= 2 ? 128 : 32;                 // producer threads (a whole warpgroup when registers are rebalanced)
  static constexpr int NTHR = 128 * NCW + NPT;
  static constexpr int NS = EMU_NS ? EMU_NS : (NCW == 4 ? 7 : NCW == 2 ? 5 : 6);  // ring stages
  static constexpr int LA = EMU_NS ? 0 : (NCW == 4 ? 4 : 3);       // full arrival this many stages behind issue
  static constexpr int WB = NCW * 3072;                            // weight bytes per stage (16 rows x 64 K x (codes + scales) per warp)
  static constexpr int BT = NT * 128;                              // token tile bytes per stage
  static constexpr int BM = 64 * NCW;
  static constexpr int MINB = (NT == 64 && NCW == 2) ? 2 : (NCW == 1 ? 4 : 1);  // remainder kernel: 4 CTAs / SM (6 in pf19 did not help)
  static_assert(NS >= LA + 2, "ring depth");
};
template <int NS> struct Sm { unsigned long long full[NS], empty[NS], wbar[NS]; Hdr hdr[NS]; int order[32], cum[33]; int tnext; };
template <int NT, int NCW> constexpr size_t smem_need() {
  using C = Cfg<NT, NCW>;
  return (size_t)C::NS * C::BT + (size_t)C::NS * C::WB + sizeof(Sm<C::NS>) + (HAS_WG_HOST ? 1024 : NCW * 64 * 16 * 4);
}

// this warp's 16 weight rows x 64 K of one stage: uint4 of codes (4 k16 fragments) + uint2 of scales
__device__ __forceinline__ void dq_load(uint32_t wa, int lane, uint32_t& c0, uint32_t& c1, uint32_t& c2, uint32_t& c3, uint32_t& sx, uint32_t& sy) {
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3) : "r"(wa + lane * 16));
  asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];" : "=r"(sx), "=r"(sy) : "r"(wa + 512 + lane * 8));
}
// two k16 fragments (codes q0, q1; scale word sw holding their rows-g / rows-g+8 scales) -> 8 A-fragment registers
__device__ __forceinline__ void dq_half(uint32_t q0, uint32_t q1, uint32_t sw, uint32_t (&a)[2][4]) {
#pragma unroll
  for (int i = 0; i < 2; i++) {
    const uint32_t qv = i ? q1 : q0;
    const uint32_t sA = sc2(sw, i ? 0x4242u : 0x4040u), sB = sc2(sw, i ? 0x4343u : 0x4141u);
    a[i][0] = bmul(qv & 0x81C081C0u, sA);
    a[i][1] = bmul(((qv << 3) & 0x01C001C0u) | ((qv << 1) & 0x80008000u), sB);
    a[i][2] = bmul((qv << 6) & 0x81C081C0u, sA);
    a[i][3] = bmul(((qv >> 5) & 0x01C001C0u) | ((qv << 5) & 0x80008000u), sB);
  }
}
#if TMA
__device__ __forceinline__ void tma2d(uint32_t dst, const CUtensorMap* tm, int c0, int c1, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
               :: "r"(dst), "l"((uint64_t)tm), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
#endif
// stage-ready barrier: on Hopper the TMA transaction barrier (weights bulk + token-tile tensor copy), emulated: the cp.async full barrier
template <int NS> __device__ __forceinline__ uint32_t rdy(Sm<NS>* sm, int slot) {
#if TMA
  return (uint32_t)__cvta_generic_to_shared(&sm->wbar[slot]);
#else
  return (uint32_t)__cvta_generic_to_shared(&sm->full[slot]);
#endif
}

// one 64-K stage of a warpgroup in two wgmma groups of 2 k16 each (A fragments in two 8-register sets): group 0 (set 0, dequantized
// during the previous stage) is issued, the previous stage's group 1 retires (its slot is released), set 1 is dequantized under group 0
// and issued, group 0 retires, and the next stage's first half is dequantized into set 0 under group 1. Double-buffering whole stages
// (32 A registers) made ptxas serialize the wgmmas at the 112-register budget.
#define W4STEP(K)                                                                                                                    \
  {                                                                                                                                  \
    const int qq = q + (K), slot = qq % C::NS;                                                                                       \
    const uint64_t db = gdesc(bbase0 + slot * C::BT);                                                                                \
    PT(c_b);                                                                                                                         \
    wg_fence();                                                                                                                      \
    wgm<NT>(acc, s0[0], db, (K), scr);                                                                                               \
    wgm<NT>(acc, s0[1], db + 2, 1, scr);                                                                                             \
    wg_commit();                                                                                                                     \
    if (K) {                                                                                                                         \
      wg_wait<1>();                                                                                                                  \
      if (lane == 0) mbarrive((uint32_t)__cvta_generic_to_shared(&sm->empty[(qq - 1) % C::NS]));                                     \
    }                                                                                                                                \
    PT(c_c);                                                                                                                         \
    dq_half(k2, k3, ky, s1);                                                                                                         \
    PT(c_d);                                                                                                                         \
    wg_fence();                                                                                                                      \
    wgm<NT>(acc, s1[0], db + 4, 1, scr);                                                                                             \
    wgm<NT>(acc, s1[1], db + 6, 1, scr);                                                                                             \
    wg_commit();                                                                                                                     \
    wg_wait<1>();                                                                                                                    \
    PT(c_e);                                                                                                                         \
    PADD(pc, 1, c_d - c_c); PADD(pc, 2, (c_c - c_b) + (c_e - c_d));                                                                  \
    if ((K) + 1 < h.nks) {                                                                                                           \
      PWAIT(pc.v[0], rdy<C::NS>(sm, (qq + 1) % C::NS), ((qq + 1) / C::NS) & 1);                                                      \
      PT(c_f);                                                                                                                       \
      uint32_t k0, k1, kx;                                                                                                           \
      dq_load(sbase0 + ((qq + 1) % C::NS) * C::WB + woff, lane, k0, k1, k2, k3, kx, ky);                                             \
      dq_half(k0, k1, kx, s0);                                                                                                       \
      PT(c_g);                                                                                                                       \
      PADD(pc, 1, c_g - c_f);                                                                                                        \
    }                                                                                                                                \
    PADD(pc, 4, 1);                                                                                                                  \
  }

template <int NT, int NCW>
__device__ __forceinline__ void ctile(const P& p, Sm<Cfg<NT, NCW>::NS>* sm, uint32_t bbase0, uint32_t sbase0, int q, const Hdr h, int wi,
                                      int tid, Pc& pc, float* scr) {
  using C = Cfg<NT, NCW>;
  float acc[NT / 2];
  uint32_t s0[2][4], s1[2][4];
  const int lt = tid & 127, w = lt >> 5, lane = tid & 31;
  const int woff = (wi * 4 + w) * 768;
  uint32_t k2, k3, ky;
  {
    uint32_t k0, k1, kx;
    dq_load(sbase0 + (q % C::NS) * C::WB + woff, lane, k0, k1, k2, k3, kx, ky);  // the caller waited for stage q
    dq_half(k0, k1, kx, s0);
  }
  for (int ks = 0; ks < h.nks; ks++) W4STEP(ks);
  wg_wait<0>();
  if (lane == 0) mbarrive((uint32_t)__cvta_generic_to_shared(&sm->empty[(q + h.nks - 1) % C::NS]));
  PT(c_e0);
  const int g = lane >> 2, t4 = lane & 3;
  const float gs = p.g[h.e];
  __nv_bfloat16* ob = p.out + ((size_t)h.kh * p.rows + h.row0 + 2 * t4) * p.nout + (p.fuse ? (h.rb * C::BM + wi * 64 + 16 * w) / 2 : h.rb * C::BM + wi * 64 + 16 * w) + g;
  const int nl = h.ntok - 2 * t4;
#pragma unroll
  for (int j = 0; j < NT / 8; j++)
#pragma unroll
    for (int e2 = 0; e2 < 2; e2++)
      if (8 * j + e2 < nl) {
        __nv_bfloat16* o = ob + (size_t)(8 * j + e2) * p.nout;
        if (p.fuse) {
          const float gate=__bfloat162float(__float2bfloat16(acc[4*j+e2]*gs));
          const float up=__bfloat162float(__float2bfloat16(acc[4*j+2+e2]*gs));
          o[0]=__float2bfloat16(gate/(1.f+__expf(-gate))*up);
        } else { o[0] = __float2bfloat16(acc[4 * j + e2] * gs); o[8] = __float2bfloat16(acc[4 * j + 2 + e2] * gs); }
      }
  PT(c_f0);
  PADD(pc, 3, c_f0 - c_e0);
}

// NT = BNMAX, NCW = 4: full token tiles + remainders > 32, 256-row blocks; NT = 32, NCW = 1: remainders <= 32, 64-row blocks.
// tm: the token rows (A [rows, 4096] bf16) as a TMA tensor with a (64 K x NT rows) SW128 box
template <int NT, int NCW>
__global__ void __launch_bounds__(Cfg<NT, NCW>::NTHR, Cfg<NT, NCW>::MINB) w4g_k(const __grid_constant__ P p, const __grid_constant__ CUtensorMap tm) {
  using C = Cfg<NT, NCW>;
  constexpr int NS = C::NS, LA = C::LA, NPT = C::NPT;
  extern __shared__ uint8_t smraw[];
#if HAS_WG
  uint8_t* sm0 = (uint8_t*)(((uintptr_t)smraw + 1023) & ~(uintptr_t)1023);
  float* scr = nullptr;
#else
  uint8_t* sm0 = smraw;
  float* scr = (float*)(smraw + (size_t)NS * C::BT + (size_t)NS * C::WB + sizeof(Sm<NS>)) + (threadIdx.x >> 7) * 1024;
#endif
  const uint32_t bbase0 = (uint32_t)__cvta_generic_to_shared(sm0), sbase0 = bbase0 + NS * C::BT;
  Sm<NS>* sm = (Sm<NS>*)(sm0 + NS * C::BT + NS * C::WB);
  const int tid = threadIdx.x;
  const int nrb = p.nrb256 * (4 / NCW);
#ifdef W4PROF
  const unsigned long long t_start = clk(), n_start = gns();
#endif
  Pc pc = {{0, 0, 0, 0, 0, 0}};
  if (tid == 0) {
    for (int i = 0; i < NS; i++) {
      mbinit((uint32_t)__cvta_generic_to_shared(&sm->full[i]), NPT);
      mbinit((uint32_t)__cvta_generic_to_shared(&sm->empty[i]), 4 * NCW);  // one arrival per consumer warp after its wgmmas completed
      mbinit((uint32_t)__cvta_generic_to_shared(&sm->wbar[i]), 1);
    }
#if TMA
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
  }
  if (tid < 32) {
    const int lane = tid, c = p.offs[lane] - (lane ? p.offs[lane - 1] : 0);
    int rk = 0;
    for (int j = 0; j < 32; j++) { const int cj = __shfl_sync(0xffffffffu, c, j); rk += (cj > c) || (cj == c && j < lane); }
    sm->order[rk] = lane;
    // this launch's share of the expert's tokens: c % lomod (the tokens not covered by a wider launch), in tsz-token tiles; a
    // full-only launch takes no remainder tile; the remainder kernel (NT 32) takes a <= 32-token remainder
    const int cc = p.lomod ? c % p.lomod : c, full = cc / p.tsz, rem = cc % p.tsz;
    const int nt = (NT == 64 ? (rem >= 33 && rem <= 64 ? 1 : 0) : NT == 32 ? (rem > 0 && rem <= 32 ? 1 : 0) : full + ((p.fullonly == 0 && rem > 32) || (p.fullonly == 2 && rem > 64) ? 1 : 0)) * nrb * p.ksplit;
    int ntr = 0;
    for (int j = 0; j < 32; j++) { const int rj = __shfl_sync(0xffffffffu, rk, j), nj = __shfl_sync(0xffffffffu, nt, j); if (rj == lane) ntr = nj; }
    int v = ntr;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { const int n2 = __shfl_up_sync(0xffffffffu, v, o); if (lane >= o) v += n2; }
    sm->cum[lane + 1] = v;
    if (lane == 0) sm->cum[0] = 0;
  }
  __syncthreads();
  const int total = sm->cum[32];
  const int nks = KST / p.ksplit;
  if (tid >= 128 * NCW) {  // producer
#if HAS_WG
    if constexpr (NCW == 4) asm volatile("setmaxnreg.dec.sync.aligned.u32 32;\n" ::: "memory");
    if constexpr (NCW == 2 && NT == 64) asm volatile("setmaxnreg.dec.sync.aligned.u32 32;\n" ::: "memory");
    else if constexpr (NCW == 2) asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" ::: "memory");
#endif
    const int pt = tid - 128 * NCW;
#if TMA
    // one thread: per stage one bulk copy of the NVFP4 chunk + one tensor copy of the token tile onto the stage's transaction barrier
    if (pt == 0) {
      int s = 0;
      for (;;) {
        const int u = atomicAdd(p.ctr, 1);
        if (u >= total || *(volatile int*)&g_w4err) {
          if (atomicAdd(p.ctr + 1, 1) == (int)gridDim.x - 1) { p.ctr[0] = 0; p.ctr[1] = 0; __threadfence(); }
          break;
        }
        int r = 0;
        while (sm->cum[r + 1] <= u) r++;
        const int e = sm->order[r], v = u - sm->cum[r], kh = v % p.ksplit, v2 = v / p.ksplit, tt = v2 / nrb, rb = v2 % nrb;
        const int st0 = e ? p.offs[e - 1] : 0, cnt0 = p.offs[e] - st0, cnt = p.lomod ? cnt0 % p.lomod : cnt0, sh = cnt0 - cnt;
        const int t0 = (NT == 32 || NT == 64) ? cnt / p.tsz : tt;
        const int row0 = st0 + sh + t0 * p.tsz, ntok = min(NT, cnt - t0 * p.tsz);
        const uint8_t* wsrc = p.W + ((size_t)(e * p.nrb256 + rb / (4 / NCW)) * KST + kh * nks) * WCH + (rb % (4 / NCW)) * (NCW * 3072);
        for (int ks = 0; ks < nks; ks++, s++) {
          const int slot = s % NS;
          if (s >= NS) PWAIT(pc.v[0], (uint32_t)__cvta_generic_to_shared(&sm->empty[slot]), ((s / NS) - 1) & 1);
          PT(p_a);
          sm->hdr[slot] = Hdr{u, e, rb, row0, ntok, kh, nks, 0};
          const uint32_t wb = (uint32_t)__cvta_generic_to_shared(&sm->wbar[slot]);
          expect_tx(wb, C::WB + C::BT);
          bulk(sbase0 + slot * C::WB, wsrc + (size_t)ks * WCH, C::WB, wb);
          tma2d(bbase0 + slot * C::BT, &tm, (kh * nks + ks) * 64, row0, wb);
          PT(p_b);
          PADD(pc, 1, p_b - p_a); PADD(pc, 4, 1);
        }
      }
      const int slot = s % NS;
      if (s >= NS) mbwait((uint32_t)__cvta_generic_to_shared(&sm->empty[slot]), ((s / NS) - 1) & 1);
      sm->hdr[slot].tile = -1;
      mbarrive((uint32_t)__cvta_generic_to_shared(&sm->wbar[slot]));
#ifdef W4PROF
      if (p.prof) {
        for (int i = 0; i < 5; i++) atomicAdd(p.prof + 8 + i, pc.v[i]);
        atomicAdd(p.prof + 13, clk() - t_start);
      }
#endif
    }
#else
    // emulation (no TMA): all producer threads cp.async the chunk and the token rows, full barrier after cp.async wait + fence
    int s = 0;
    for (;;) {
      if (pt == 0) sm->tnext = atomicAdd(p.ctr, 1);
      pbar<NPT>();
      const int u = sm->tnext;
      pbar<NPT>();
      if (u >= total || *(volatile int*)&g_w4err) {
        if (pt == 0 && atomicAdd(p.ctr + 1, 1) == (int)gridDim.x - 1) { p.ctr[0] = 0; p.ctr[1] = 0; __threadfence(); }
        break;
      }
      int r = 0;
      while (sm->cum[r + 1] <= u) r++;
      const int e = sm->order[r], v = u - sm->cum[r], kh = v % p.ksplit, v2 = v / p.ksplit, tt = v2 / nrb, rb = v2 % nrb;
      const int st0 = e ? p.offs[e - 1] : 0, cnt0 = p.offs[e] - st0, cnt = p.lomod ? cnt0 % p.lomod : cnt0, sh = cnt0 - cnt;
      const int t0 = (NT == 32 || NT == 64) ? cnt / p.tsz : tt;
      const int row0 = st0 + sh + t0 * p.tsz, ntok = min(NT, cnt - t0 * p.tsz);
      const uint8_t* wsrc = p.W + ((size_t)(e * p.nrb256 + rb / (4 / NCW)) * KST + kh * nks) * WCH + (rb % (4 / NCW)) * (NCW * 3072);
      const __nv_bfloat16* xsrc = p.A + (size_t)row0 * 4096 + kh * nks * 64;
      for (int ks = 0; ks < nks; ks++, s++) {
        const int slot = s % NS;
        if (s >= NS) mbwait((uint32_t)__cvta_generic_to_shared(&sm->empty[slot]), ((s / NS) - 1) & 1);
        if (pt == 0) sm->hdr[slot] = Hdr{u, e, rb, row0, ntok, kh, nks, 0};
        for (int i = pt; i < C::WB / 16; i += NPT) cp16(sbase0 + slot * C::WB + i * 16, wsrc + (size_t)ks * WCH + i * 16);
        const uint32_t bb = bbase0 + slot * C::BT;
#pragma unroll
        for (int i = 0; i < NT * 8 / NPT; i++) {
          const int qd = pt + NPT * i, row = qd >> 3, c = qd & 7;
          if (row < ntok) cp16(swz(bb + row * 128 + c * 16), xsrc + (size_t)row * 4096 + ks * 64 + c * 8);
        }
        cp_commit();
        cp_wait<0>();
        mbarrive((uint32_t)__cvta_generic_to_shared(&sm->full[slot]));
      }
    }
    const int slot = s % NS;
    if (s >= NS) mbwait((uint32_t)__cvta_generic_to_shared(&sm->empty[slot]), ((s / NS) - 1) & 1);
    if (pt == 0) sm->hdr[slot].tile = -1;
    mbarrive((uint32_t)__cvta_generic_to_shared(&sm->full[slot]));
#endif
  } else {
    // 640 threads x 96 registers at launch: producer 128 x 32 + consumers 512 x 112 = 61440 (acc 64 + 2 x 16 A-fragment registers)
#if HAS_WG
    if constexpr (NCW == 4) asm volatile("setmaxnreg.inc.sync.aligned.u32 112;\n" ::: "memory");
    // wide: 384 threads x 168 registers at launch: producer 128 x 40 + consumers 256 x 232 = 64512 (acc 128 + 16 A-fragment registers)
    if constexpr (NCW == 2 && NT == 64) asm volatile("setmaxnreg.inc.sync.aligned.u32 104;\n" ::: "memory");
    else if constexpr (NCW == 2) asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n" ::: "memory");
#endif
    const int wi = tid >> 7;
    for (int q = 0;;) {
      const int slot = q % NS;
      PWAIT(pc.v[0], rdy<NS>(sm, slot), (q / NS) & 1);
      const Hdr h = sm->hdr[slot];
      if (h.tile < 0 || *(volatile int*)&g_w4err) break;
      ctile<NT, NCW>(p, sm, bbase0, sbase0, q, h, wi, tid, pc, scr);
      q += h.nks;
    }
#ifdef W4PROF
    if ((tid & 31) == 0 && p.prof) {
      for (int i = 0; i < 5; i++) atomicAdd(p.prof + i, pc.v[i]);
      atomicAdd(p.prof + 5, clk() - t_start);
      atomicAdd(p.prof + 6, gns() - n_start);
    }
#endif
  }
}

static int g_nsm = 0;
#define KBIG w4g_k<BNMAX, 4>
#define KWIDE w4g_k<2 * BNMAX, 2>
#define KSMALL w4g_k<32, 1>
static void w4g_init() {
  if (g_nsm) return;
  int dev; cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&g_nsm, cudaDevAttrMultiProcessorCount, dev);
  cudaFuncSetAttribute(KBIG, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_need<BNMAX, 4>());
  cudaFuncSetAttribute(KWIDE, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_need<2 * BNMAX, 2>());
  cudaFuncSetAttribute(KSMALL, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_need<32, 1>());
}
static PFN_cuTensorMapEncodeTiled_v12000 g_enc = nullptr;
// A [rows, 4096] bf16 as a 2D TMA tensor, box 64 K x box rows, 128-B swizzle (the wgmma K-major SW128 layout); rows past the end read 0
static CUtensorMap tmap(const void* base, int64_t rows, int box) {
  CUtensorMap m;
  memset(&m, 0, sizeof(m));
#if HAS_WG_HOST
  if (!g_enc) {
    cudaDriverEntryPointQueryResult qr;
    TORCH_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", (void**)&g_enc, cudaEnableDefault, &qr) == cudaSuccess && g_enc, "cuTensorMapEncodeTiled");
  }
  cuuint64_t dims[2] = {4096, (cuuint64_t)rows};
  cuuint64_t strides[1] = {4096 * 2};
  cuuint32_t bx[2] = {64, (cuuint32_t)box}, es[2] = {1, 1};
  const CUresult r = g_enc(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(base), dims, strides, bx, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                           CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "w4g tensor map ", (int)r);
#endif
  return m;
}
int64_t w4g_err(int64_t reset) {
  int v = 0;
  cudaMemcpyFromSymbol(&v, g_w4err, sizeof(int));
  if (reset) { const int z = 0; cudaMemcpyToSymbol(g_w4err, &z, sizeof(int)); }
  return v;
}
// mode 0: n128 kernel (4 WG x m64n128) + remainder kernel; 1: wide kernel (2 WG x m64n256, 256-token tiles) + remainder kernel;
// 2: wide kernel on the full 256-token tiles only, n128 kernel on each expert's c % 256 leftover, remainder kernel on its <= 32 rest.
// cls bit 0: the wide / n128 kernels, bit 1: the remainder kernel. ctr: 2 ints per launch slot (6), prof: 16 per slot (48);
// slots: 0 the first big kernel, 1 the mixed-mode n128 kernel, 2 the remainder kernel. out [ksplit * rows, nrb256 * 256]
void w4g(torch::Tensor A, torch::Tensor offs, torch::Tensor W, torch::Tensor g, torch::Tensor out, torch::Tensor ctr, torch::Tensor prof,
         int64_t ksplit, int64_t cls, int64_t mode, int64_t fuse) {
  TORCH_CHECK(!fuse || ksplit == 1, "SwiGLU requires the complete W13 reduction");
  w4g_init();
  P p{(const __nv_bfloat16*)A.data_ptr(), offs.data_ptr<int>(), W.data_ptr<uint8_t>(), g.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(),
      ctr.data_ptr<int>(), prof.numel() ? (unsigned long long*)prof.data_ptr() : nullptr, (int)W.size(1), (int)out.size(1),
      (int)(out.size(0) / ksplit), (int)ksplit, BNMAX, 0, 0, (int)fuse};
  auto st = at::cuda::getCurrentCUDAStream();
  P q = p;
  if (cls & 1) {
    if (mode >= 1) {
      q.tsz = 2 * BNMAX; q.fullonly = mode == 2;
      KWIDE<<<g_nsm, Cfg<2 * BNMAX, 2>::NTHR, smem_need<2 * BNMAX, 2>(), st>>>(q, tmap(A.data_ptr(), A.size(0), 2 * BNMAX));
    } else {
      KBIG<<<g_nsm, Cfg<BNMAX, 4>::NTHR, smem_need<BNMAX, 4>(), st>>>(q, tmap(A.data_ptr(), A.size(0), BNMAX));
    }
    if (mode == 2) {
      q = p; q.ctr += 2; q.prof = p.prof ? p.prof + 16 : nullptr; q.lomod = 2 * BNMAX;
      KBIG<<<g_nsm, Cfg<BNMAX, 4>::NTHR, smem_need<BNMAX, 4>(), st>>>(q, tmap(A.data_ptr(), A.size(0), BNMAX));
    }
  }
  if (cls & 2) {
    q = p; q.ctr += 4; q.prof = p.prof ? p.prof + 32 : nullptr;
    q.tsz = mode == 1 ? 2 * BNMAX : BNMAX; q.lomod = mode == 2 ? 2 * BNMAX : 0;
    KSMALL<<<g_nsm * Cfg<32, 1>::MINB, Cfg<32, 1>::NTHR, smem_need<32, 1>(), st>>>(q, tmap(A.data_ptr(), A.size(0), 32));
  }
}
int64_t w4g_smem() { return (int64_t)(smem_need<BNMAX, 4>() * 1000000 + smem_need<32, 1>()); }

// Research only: emits just final 33..64-token tails relative to 128-token tiles.
// All other rows are intentionally untouched. Full K only; no partial BF16 sum.
void w4tail64(torch::Tensor A, torch::Tensor offs, torch::Tensor W, torch::Tensor g,
              torch::Tensor out, torch::Tensor ctr, torch::Tensor prof, int64_t fuse) {
  TORCH_CHECK(A.size(1) == 4096 && offs.numel() == 32, "tail64 input shape");
  w4g_init();
  cudaFuncSetAttribute(w4g_k<64,2>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       (int)smem_need<64,2>());
  P p{(const __nv_bfloat16*)A.data_ptr(), offs.data_ptr<int>(), W.data_ptr<uint8_t>(),
      g.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(), ctr.data_ptr<int>(),
      prof.numel() ? (unsigned long long*)prof.data_ptr() : nullptr,
      (int)W.size(1), (int)out.size(1), (int)out.size(0), 1, 128, 0, 0, (int)fuse};
  w4g_k<64,2><<<2*g_nsm, Cfg<64,2>::NTHR, smem_need<64,2>(),
                 at::cuda::getCurrentCUDAStream()>>>(p, tmap(A.data_ptr(), A.size(0), 64));
}
int64_t tail64_smem() { return smem_need<64,2>(); }
int64_t tail64_occupancy() {
  int n = 0;
  cudaFuncSetAttribute(w4g_k<64,2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_need<64,2>());
  TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,w4g_k<64,2>,384,smem_need<64,2>()) == cudaSuccess, "occupancy query");
  return n;
}
// Mode0 sequence: big kernel excludes only 33..64 tails, then old n32, then n64.
void w4filtered(torch::Tensor A, torch::Tensor offs, torch::Tensor W, torch::Tensor g,
                torch::Tensor out, torch::Tensor ctr, torch::Tensor prof, int64_t fuse) {
  w4g_init();
  P p{(const __nv_bfloat16*)A.data_ptr(), offs.data_ptr<int>(), W.data_ptr<uint8_t>(),
      g.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(), ctr.data_ptr<int>(), nullptr,
      (int)W.size(1), (int)out.size(1), (int)out.size(0), 1, 128, 0, 2, (int)fuse};
  auto st = at::cuda::getCurrentCUDAStream();
  KBIG<<<g_nsm, Cfg<128,4>::NTHR, smem_need<128,4>(),st>>>(p,tmap(A.data_ptr(),A.size(0),128));
  p.ctr += 4; p.fullonly = 0;
  KSMALL<<<4*g_nsm,Cfg<32,1>::NTHR,smem_need<32,1>(),st>>>(p,tmap(A.data_ptr(),A.size(0),32));
  w4tail64(A,offs,W,g,out,ctr.slice(0,8,10),prof,fuse);
}

