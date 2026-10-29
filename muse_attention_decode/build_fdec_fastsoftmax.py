"""Port guarded FP16 lazy softmax to full batched fdec, preserving exact fallback."""
from pathlib import Path
import ast
ROOT=Path(__file__).parent
s=(ROOT/'candidate_openkernels_attention.py').read_text()
a=s.index('template <bool CL, bool MU>')
b=s.index('void prep_only(',a)
part=s[a:b]
old='''      float bm = fmaxf(fmaxf(v[0], v[1]), fmaxf(v[2], v[3]));
      bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 1));
      bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 2));
      const float nm = fmaxf(mx[h], bm), scl = ex2(mx[h] - nm);
      mx[h] = nm;
      const float p0 = ex2(v[0] - nm), p1 = ex2(v[1] - nm), p2 = ex2(v[2] - nm), p3 = ex2(v[3] - nm);
      ls[h] = ls[h] * scl + (p0 + p1 + p2 + p3);
      const uint32_t pa0 = pk(p0, p1), pa2 = pk(p2, p3);'''
new='''      float scl = 1.f, p0, p1, p2, p3;
      uint32_t pa0, pa2;
      // Full, batched decode only: other shapes keep their exact max path.
      bool exact = true;
#if FD_FASTSOFT
      if (!p.window && p.B > 1 && bi > 0) {
        p0 = ex2(v[0] - mx[h]); p1 = ex2(v[1] - mx[h]);
        p2 = ex2(v[2] - mx[h]); p3 = ex2(v[3] - mx[h]);
        pa0 = pk(p0, p1); pa2 = pk(p2, p3);
        const __half2 hm = __hmax2(*reinterpret_cast<__half2*>(&pa0), *reinterpret_cast<__half2*>(&pa2));
        const uint32_t hb = *reinterpret_cast<const uint32_t*>(&hm);
        exact = false;
        if (__any_sync(0xffffffffu, (hb & 0xffffu) > LZB || (hb >> 16) > LZB)) {
          float pm = fmaxf(fmaxf(p0, p1), fmaxf(p2, p3));
          pm = fmaxf(pm, __shfl_xor_sync(0xffffffffu, pm, 1));
          pm = fmaxf(pm, __shfl_xor_sync(0xffffffffu, pm, 2));
          if (__any_sync(0xffffffffu, pm == INFINITY)) {
            // Keep unmodified scores v[] for the stable original fallback.
            exact = true;
          } else if (pm > LZP) {
            scl = 1.f / pm;
            mx[h] += __log2f(pm);
            p0 *= scl; p1 *= scl; p2 *= scl; p3 *= scl;
            pa0 = pk(p0, p1); pa2 = pk(p2, p3);
          }
        }
      }
#endif
      if (exact) {
        float bm = fmaxf(fmaxf(v[0], v[1]), fmaxf(v[2], v[3]));
        bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 1));
        bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 2));
        const float nm = fmaxf(mx[h], bm);
        scl = ex2(mx[h] - nm); mx[h] = nm;
        p0 = ex2(v[0] - nm); p1 = ex2(v[1] - nm);
        p2 = ex2(v[2] - nm); p3 = ex2(v[3] - nm);
        pa0 = pk(p0, p1); pa2 = pk(p2, p3);
      }
      ls[h] = ls[h] * scl + (p0 + p1 + p2 + p3);
      // Do not lengthen every PV accumulator dependency when the max stayed fixed.
      if (scl != 1.f) {
#pragma unroll
        for (int n = 0; n < 8; n++) { o[h][n][0] *= scl; o[h][n][1] *= scl; }
      }'''
assert part.count(old)==1
part=part.replace(old,new)
part=part.replace('o[h][n][0] * scl, o[h][n][1] * scl','o[h][n][0], o[h][n][1]')
part=part.replace('o[h][n + 1][0] * scl, o[h][n + 1][1] * scl','o[h][n + 1][0], o[h][n + 1][1]')
s=s[:a]+part+s[b:]
s=s.replace('load_inline("attn_v60_%d"','load_inline("attn_v60_fdecfast_%d"')
s=s.replace('_DBG = []','_DBG = ["-DFD_FASTSOFT=1"]',1)
(ROOT/'candidate_openkernels_fdecfast.py').write_text(s);ast.parse(s)
# Matched control preserves accumulator-rescale placement and exact max always.
control=s.replace('attn_v60_fdecfast_%d','attn_v60_fdeccontrol_%d').replace('_DBG = ["-DFD_FASTSOFT=1"]','_DBG = ["-DFD_FASTSOFT=0"]')
(ROOT/'candidate_openkernels_fdeccontrol.py').write_text(control);ast.parse(control)
print('Generated isolated fdec fast-softmax + exact matched control; runtime restricted full batched decode')
