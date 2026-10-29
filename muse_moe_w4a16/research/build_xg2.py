"""O(R) xgather for moe_x (wx): rank routes once per block in shared memory instead of per-warp global rescans.

Judge profile (job 4a60fa38515e4581): xgather 9.8 us at M128 and 11.3 us at M256 (whole-decode M32: 5.9 us), fully
serial before moe_x. The old kernel has each warp count all earlier routes of its expert from global memory (O(R^2)).
New kernel: every 1024-thread block loads the ids once, builds the same stable per-expert ranks (route order) and
padded per-expert offsets as moe_x's setup, then its warps copy their rows with 16 B loads. Output bytes identical.
"""
import ast
from pathlib import Path

root = Path(__file__).resolve().parents[1]
src = root / 'research_moe_v56_strict.py'
text = src.read_text()
node = next(n for n in ast.parse(text).body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'CUDA_X')
c = node.value.value
a = c.index('// gather: route i of a local expert e -> row pos(i) of e\'s block')
b = c.index('template <int N>\n__device__ __forceinline__ void epilogue(')
new_kernel = r'''// gather: route i of a local expert e -> row pos(i) of e's block, K in the wgmma order, 16-B chunks swizzled.
// One pass of stable per-expert ranking per block (same order as moe_x's lists), then rows copied by warps.
#define XGT 1024
#define XGMAXR 4096
__global__ void __launch_bounds__(XGT) xgather(const __nv_bfloat16* x, const int* ids, int M, int lo, uint8_t* xg) {
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
  for (int i = blockIdx.x * (XGT / 32) + wid; i < R; i += gridDim.x * (XGT / 32)) {
    const int e = e_s[i];
    if (e >= NLOC) continue;
    const int pos = pos_s[i], np = npad[e];
    const uint4* src = (const uint4*)(x + (size_t)(i >> 3) * HD);
    uint8_t* dst = xg + xo[e];
#pragma unroll 4
    for (int q4 = lane; q4 < HD / 8; q4 += 32) {
      const uint4 v = __ldg(src + q4);
      const uint32_t w4[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int k = 0; k < 4; k++) *(uint32_t*)(dst + xaddr_of(kperm((q4 * 4 + k) * 2), pos, np)) = w4[k];
    }
  }
}

'''
c = c[:a] + new_kernel + c[b:]
old_launch = 'xgather<<<(M * 8 + 7) / 8, 256, 0, st>>>('
assert c.count(old_launch) == 1
c = c.replace(old_launch, 'xgather<<<min(g_gridX, (M * 8 + 31) / 32), XGT, 0, st>>>(')
c = c.replace('TORCH_CHECK(M * 8 <= 4096 && ', 'TORCH_CHECK(M * 8 <= XGMAXR && ', 1)
assert 'M * 8 <= XGMAXR' in c
text = text.replace(ast.get_source_segment(text, node.value), repr(c), 1)
text = text.replace('"moe_v56strict_v56X"', '"moe_xg2_v56X"')
assert '"moe_xg2_v56X"' in text
out = root / 'research_moe_xg2.py'
out.write_text(text)
compile(text, str(out), 'exec')
(root / 'research' / 'kernel_sources' / 'xg2_CUDA_X.cu').write_text(c)
print(out)
