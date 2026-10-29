"""No-PDL all-SM gather with optional register transpose for vector stores.

Derived from correctness-tested strict xg2, not the raw v35 embedded MoE.
Default XGSEG=4 spreads route quarters across all SMs. XGVECTOR adds an
8-lane x 4-word register transpose: one source uint4 load and one destination
uint4 store. No dequantization, numerical arithmetic, or precision changes.
Use -DXGSEG=1 as a row-copy control, -DXGVECTOR for vector-store variant.
"""
import ast
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = (root / 'research_moe_xg2.py').read_text()
node = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
            and getattr(n.targets[0], 'id', '') == 'CUDA_X')
c = node.value.value
assert 'Strict SGLang boundaries' in c
assert 'griddepcontrol' not in c and 'cudaLaunchKernelEx' not in c
old = '''  for (int i = blockIdx.x * (XGT / 32) + wid; i < R; i += gridDim.x * (XGT / 32)) {
    const int e = e_s[i];'''
new = '''#ifndef XGSEG
#define XGSEG 4
#endif
  static_assert(XGSEG == 1 || XGSEG == 2 || XGSEG == 4 || XGSEG == 8, "invalid row segmentation");
  for (int item = blockIdx.x * (XGT / 32) + wid; item < R * XGSEG; item += gridDim.x * (XGT / 32)) {
    const int i = item / XGSEG, seg = item % XGSEG;
    const int e = e_s[i];'''
assert c.count(old) == 1
c = c.replace(old, new)
old = '''    for (int q4 = lane; q4 < HD / 8; q4 += 32) {
      const uint4 v = __ldg(src + q4);
      const uint32_t w4[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int k = 0; k < 4; k++) *(uint32_t*)(dst + xaddr_of(kperm((q4 * 4 + k) * 2), pos, np)) = w4[k];
    }'''
new = '''    for (int q4 = seg * (HD / 8 / XGSEG) + lane; q4 < (seg + 1) * (HD / 8 / XGSEG); q4 += 32) {
      const uint4 v = __ldg(src + q4);
#ifdef XGVECTOR
      // Input lane bits abc and register bits de become output lane cde,
      // register ab. The 64-element kperm domain is eight uint4 lanes.
      // All lanes execute every shuffle; source lanes stay in the same octet.
      uint4 out;
      uint32_t* words = reinterpret_cast<uint32_t*>(&out);
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const int sl = (lane & ~7) | ((lane >> 2) & 1) | (j << 1);
        const uint32_t a = __shfl_sync(0xffffffffu, v.x, sl);
        const uint32_t b = __shfl_sync(0xffffffffu, v.y, sl);
        const uint32_t d = __shfl_sync(0xffffffffu, v.z, sl);
        const uint32_t eword = __shfl_sync(0xffffffffu, v.w, sl);
        const int k = lane & 3;
        words[j] = k == 0 ? a : k == 1 ? b : k == 2 ? d : eword;
      }
      *reinterpret_cast<uint4*>(dst + xaddr_of(q4 * 8, pos, np)) = out;
#else
      const uint32_t w4[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int k = 0; k < 4; k++) *(uint32_t*)(dst + xaddr_of(kperm((q4 * 4 + k) * 2), pos, np)) = w4[k];
#endif
    }'''
assert c.count(old) == 1
c = c.replace(old, new)
old = 'xgather<<<min(g_gridX, (M * 8 + 31) / 32), XGT, 0, st>>>('
assert c.count(old) == 1
c = c.replace(old, 'xgather<<<g_gridX, XGT, 0, st>>>(')
assert 'griddepcontrol' not in c and 'cudaLaunchKernelEx' not in c
source = source.replace(ast.get_source_segment(source, node.value), repr(c), 1)
source = source.replace('"moe_xg2_v56X"', '"moe_xg_vector_v56X"')
out = root / 'research_moe_xg_vector.py'
compile(source, str(out), 'exec')
out.write_text(source)
(root / 'research/kernel_sources/xg_vector_CUDA_X.cu').write_text(c)
print(out)
