"""gk KS2=1 (M <= 8): prefetch each CTA's own W2 row-tile slab into L2 during W13 (research A/B on the judge GPU).

Mechanism: at M1/M8 the W13 phase is consumer-latency-bound, so HBM may have idle capacity. Each CTA knows its active
experts and its W2 output-row tiles right after routing; their W2 slabs (nrt x 36,864 B per expert, contiguous for
KS2=1) are prefetched with cp.async.bulk.prefetch.L2 (normal priority: nothing persists across calls).
Macros: W2PF=n experts per CTA; PFQ=q: a warp issues its share once it has issued q/4 of its W13 pieces;
NOPOL: no evict_first hint on the W13 stream. Numerics are unchanged by every variant.
First judge check (9224772201374f0a, W2PF=3 PFQ=0 with policy, thread-spread issue): d1 30.2 -> 33.4, v1 73.2 -> 82.7.

Outputs research_moe_kdiag.py: frozen strict-v56 dispatch, plus TUNE over gk variants at M1 / M8 only (prints
"v56 tune" lines: per-variant cold-L2 graph-replay medians on judge-generator routings).
"""
import ast
from pathlib import Path

root = Path(__file__).resolve().parents[1]
text = (root / 'research_moe_v56_strict.py').read_text()


def change(s, a, b, n=1):
    assert s.count(a) == n, (a[:100], s.count(a), n)
    return s.replace(a, b)


node = next(n for n in ast.parse(text).body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'CUDA_K')
c = node.value.value
c = change(c, 'struct Ring {\n  const P* p; Sm* sm; uint32_t ring, bars; int lane, grp, s; unsigned long long wns;',
           '''#ifndef PFQ
#define PFQ 0
#endif
struct Ring {
  const P* p; Sm* sm; uint32_t ring, bars; int lane, grp, s; unsigned long long wns, pol; int wid, pfat;''')
c = change(c, '''  __device__ __forceinline__ void issue_next() {  // one slot: PPS consecutive pieces of one slice (contiguous half-stages)
    if (nq >= n13w + n2w) return;''', '''  __device__ __forceinline__ void prefetch_w2() {  // this warp's share of the CTA's W2 slabs (16 KB L2 prefetches)
#if defined(W2PF) && KS2 == 1 && TMA
    const int slab = sm->nrt * 16 * PC, nchk = (slab + 16383) / 16384, npf = min(sm->na, W2PF);
    if (lane == 0)
      for (int i = wid; i < npf * nchk; i += NW) {
        const int a = i / nchk, k = i - a * nchk;
        const uint8_t* g = p->w2 + ((size_t)(p->lo + sm->act[a]) * 256 + sm->r0) * 16 * PC + (size_t)k * 16384;
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\\n" :: "l"(g), "r"((uint32_t)min(16384, slab - k * 16384)) : "memory");
      }
#endif
  }
  __device__ __forceinline__ void issue_next() {  // one slot: PPS consecutive pieces of one slice (contiguous half-stages)
#if defined(W2PF) && KS2 == 1
    if (pfat >= 0 && nq >= pfat) { pfat = -1; prefetch_w2(); }
#endif
    if (nq >= n13w + n2w) return;''')
c = change(c, '''      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\\n" :: "r"(dst), "l"(src), "n"(SLOTB), "r"(bar) : "memory");''',
           '''#if defined(W2PF) && KS2 == 1 && !defined(NOPOL)
      if (nq <= n13w)  // W13 slot (nq already advanced): evict_first, so the stream does not push the prefetched W2 out
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\\n" :: "r"(dst), "l"(src), "n"(SLOTB), "r"(bar), "l"(pol) : "memory");
      else
#endif
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\\n" :: "r"(dst), "l"(src), "n"(SLOTB), "r"(bar) : "memory");''')
c = change(c, '''  C.n13w = nt13 * PPT; C.n2w = nu2 * W2S;''', '''  C.n13w = nt13 * PPT; C.n2w = nu2 * W2S;
  C.wid = wid; C.pol = 0; C.pfat = (C.n13w * PFQ / 4) / PPS * PPS;
#if defined(W2PF) && KS2 == 1 && TMA && !defined(NOPOL)
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\\n" : "=l"(C.pol));
#endif''')
seg = ast.get_source_segment(text, node.value)
text = text.replace(seg, repr(c), 1)
text = text.replace('"moe_v56strict_v5', '"moe_kdiag_v5')
base = "-DSW=4 -DKS2=1 -DMAXM=8"
variants = [base, base + " -DW2PF=3 -DPFQ=0 -DNOPOL", base + " -DW2PF=3 -DPFQ=2 -DNOPOL",
            base + " -DW2PF=3 -DPFQ=2", base + " -DW2PF=3 -DPFQ=3 -DNOPOL", base + " -DSG=3", base + " -DSG=3 -DW2PF=3 -DPFQ=2 -DNOPOL"]
text += f'''

# ---- Research A/B (build_w2pf.py): gk variants at M1 / M8, timed in prepare on judge-generator routings.
K_CFG = {tuple(variants)!r} + ("-DSW=4 -DKS2=2",)
TUNE = True
TUNE_M = (1, 8)
JUDGE_M = (1, 8)


def _cands(M):
    return [("gk", 0, 0, x) for x in K_CFG if _gkmax(x) >= M and "KS2=1" in x]


_tune_once = _tune


def _tune(state):
    out = None
    for rep in range(3):  # three passes, different variant order effects averaged by the reader
        out = _tune_once(state)
    return out
'''
out = root / 'research_moe_kdiag.py'
out.write_text(text)
compile(text, str(out), 'exec')
(root / 'research' / 'kernel_sources' / 'kdiag_CUDA_K.cu').write_text(c)
print(out)
