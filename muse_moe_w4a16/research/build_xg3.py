"""moe_x front end: all-SM O(R) xgather + PDL overlap of moe_x setup / first weight slots with the gather.

xg2 (check 061e24b8ef934b9d) cut xgather 9.8 -> 6.6 us (M128) and 11.3 -> 7.5 us (M256), but only R/32 blocks copy.
Here: every SM ranks the routes once (shared memory) and the copy is split into (row, quarter-row) items over all warps.
PDLX: xgather triggers its dependents at entry; moe_x is launched with programmatic stream serialization, its producer
pre-issues the weight TMA of the first NSL slots of its first unit, then griddepcontrol.wait, then the activation TMA.
Macros: XGOLD = original O(R^2) xgather, XGROW = xg2 block-row copy. Numerics identical in every variant.
Outputs research_moe_xg3.py: scored dispatch uses xg3 + PDLX for wx; prepare prints paired A/B "v56 tune" lines for the
wx front-end variants at M32 / M128 / M256 (cold-L2 graph medians, judge-generator routings).
"""
import ast
from pathlib import Path

root = Path(__file__).resolve().parents[1]
text = (root / 'research_moe_v56_strict.py').read_text()
node = next(n for n in ast.parse(text).body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'CUDA_X')
c = node.value.value


def change(s, a, b, n=1):
    assert s.count(a) == n, (a[:100], s.count(a), n)
    return s.replace(a, b)


a = c.index('// gather: route i of a local expert e -> row pos(i) of e\'s block')
b = c.index('template <int N>\n__device__ __forceinline__ void epilogue(')
old_kernel = c[a:b]
new_kernel = r'''#ifdef XGOLD
''' + old_kernel.replace('__global__ void __launch_bounds__(256) xgather(', '__global__ void __launch_bounds__(256) xgather(') + r'''#else
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

'''
c = c[:a] + new_kernel + c[b:]

# producer: pre-issue the first unit's first NSL weight slots, then wait for xgather, then activations
c = change(c, '''    unsigned depok = 0;
    int sq = 0;''', '''    unsigned depok = 0;
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
    asm volatile("griddepcontrol.wait;\\n" ::: "memory");
#endif''')
c = change(c, '''          uint32_t bytes = 0;
#ifndef NOACT
          bytes += WCH * rows * 128;
#endif
#ifndef NOWGT
          bytes += WSB;
#endif
          expect_tx(fb, bytes);''', '''          uint32_t bytes = 0;
#ifndef NOACT
          bytes += WCH * rows * 128;
#endif
#ifndef NOWGT
          bytes += WSB;
#endif
          if (sq >= prew) expect_tx(fb, bytes);''')
c = change(c, '''#ifndef NOWGT
#pragma unroll
          for (int b2 = 0; b2 < UBL; b2++) bulk(dst + ASB + b2 * WCH * WREC, wsrc + ((size_t)b2 * NCH + c) * WREC, WCH * WREC, fb);
#endif''', '''#ifndef NOWGT
          if (sq >= prew)
#pragma unroll
          for (int b2 = 0; b2 < UBL; b2++) bulk(dst + ASB + b2 * WCH * WREC, wsrc + ((size_t)b2 * NCH + c) * WREC, WCH * WREC, fb);
#endif''')
c = change(c, 'xgather<<<(M * 8 + 7) / 8, 256, 0, st>>>(', '''#ifdef XGOLD
  xgather<<<(M * 8 + 7) / 8, 256, 0, st>>>(
#else
  xgather<<<g_gridX, XGT, 0, st>>>(
#endif''')
c = change(c, '  moe_x<<<g_gridX, NTH, smem_x(M), st>>>(p);', '''#ifdef PDLX
  cudaLaunchConfig_t lc = {};
  lc.gridDim = dim3(g_gridX); lc.blockDim = dim3(NTH); lc.dynamicSmemBytes = smem_x(M); lc.stream = st;
  cudaLaunchAttribute la[1];
  la[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; la[0].val.programmaticStreamSerializationAllowed = 1;
  lc.attrs = la; lc.numAttrs = 1;
  TORCH_CHECK(cudaLaunchKernelEx(&lc, moe_x, p) == cudaSuccess, "moe_x PDL launch failed");
#else
  moe_x<<<g_gridX, NTH, smem_x(M), st>>>(p);
#endif''')
text = text.replace(ast.get_source_segment(text, node.value), repr(c), 1)
text = change(text, '"moe_v56strict_v56X"', '"moe_xg3_v56X"')
W1, W2 = '-DNCW=16 -DWXG2 -DUDESC', '-DNCW=16 -DN8STEP -DWXG2 -DUDESC'
text = change(text, f'(192, ("wx", 0, 0, "{W1}")),', f'(192, ("wx", 0, 0, "{W1} -DPDLX")),')
text = change(text, f'(1 << 30, ("wx", 0, 0, "{W2}")),', f'(1 << 30, ("wx", 0, 0, "{W2} -DPDLX")),')
text += f'''

# ---- Research A/B (build_xg3.py): wx front-end variants, paired in prepare on the judge GPU.
_FRONT = ("-DXGOLD", "-DXGROW", "", "-DPDLX")
WX_CFG = tuple(tuple((b + " " + f).split()) for b in ("{W1}", "{W2}") for f in _FRONT)
TUNE = True
TUNE_M = (32, 128, 256)
JUDGE_M = (32, 128, 256)


def _cands(M):
    b = "{W1}" if M <= 192 else "{W2}"
    return [("wx", 0, 0, (b + " " + f).strip()) for f in _FRONT]


_tune_once = _tune


def _tune(state):
    out = None
    for rep in range(3):
        out = _tune_once(state)
    return out
'''
out = root / 'research_moe_xg3.py'
out.write_text(text)
compile(text, str(out), 'exec')
(root / 'research' / 'kernel_sources' / 'xg3_CUDA_X.cu').write_text(c)
print(out)
