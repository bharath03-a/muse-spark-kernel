"""External-only OKC preparation diagnostic; ordinary strict-v35 path unchanged.
Never use measured preparation data in kernel_fn. No reference/harness imports.
"""
import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
base=ROOT/'muse_decode_layer/candidate_v35_wxpin32_strict.py';text=base.read_text();tree=ast.parse(text)
strings={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)}
def replace(s,a,b):
 assert s.count(a)==1,(a[:100],s.count(a));return s.replace(a,b,1)
k=strings['M_CUDA_K'].replace('blockIdx.x * 16','blockIdx.x * 32')
k=replace(k,'  // routing: deterministic slots','  RD_STAMP(20); RD_STAMP(21); RD_STAMP(13);\n  // routing: deterministic slots')
k=replace(k,'    // cost-weighted contiguous W13 tile ranges','    RD_STAMP(14);\n    // cost-weighted contiguous W13 tile ranges')
k=replace(k,'  const int na = sm->na, nr = sm->nr, n13 = sm->n13, nrt = sm->nrt;', '  RD_STAMP(15);\n  const int na = sm->na, nr = sm->nr, n13 = sm->n13, nrt = sm->nrt;')
# Per-warp ownership timestamps avoid adding any synchronization to the kernel.
wm='if (lane == 0 && p.prof) rd_stamp(p.prof+gridDim.x*32+(blockIdx.x*NW+wid)*2+%d);'
k=replace(k,'  PST(1);','  PST(1);\n  RD_STAMP(16);')
k=replace(k,'  if (tid < R && e >= 0) {','  '+wm%0+'\n  if (tid < R && e >= 0) {')
k=replace(k,'  {\n    const int kx = sm->k1 * (HD / KS1);','  '+wm%1+'\n  RD_STAMP(18);\n  {\n    const int kx = sm->k1 * (HD / KS1);')
k=replace(k,'  int q = 0;\n#ifdef PROF','  RD_STAMP(19);\n  int q = 0;\n#ifdef PROF')
x=strings['M_CUDA_X'].replace('(NTH / 32) + wid) * 16','(NTH / 32) + wid) * 32')
x=replace(x,'  for (int i = tid; i < R; i += NTH) { unsigned e = p.ids[i] - p.lo;', '  if (wid == 0) { RD_STAMP(20); RD_STAMP(21); RD_STAMP(13); }\n  for (int i = tid; i < R; i += NTH) { unsigned e = p.ids[i] - p.lo;')
x=replace(x,'  if (wid == 0) {\n    const int c = sm->base[lane];','  if (wid == 0) {\n    RD_STAMP(14);\n    const int c = sm->base[lane];')
x=replace(x,'    int w2 = c > 0 ? NRB2','    RD_STAMP(15);\n    int w2 = c > 0 ? NRB2')
x=replace(x,'  for (int i = tid; i < R; i += NTH) { const int e = lid[i];','  if (wid == 0) RD_STAMP(16);\n  for (int i = tid; i < R; i += NTH) { const int e = lid[i];')
x=replace(x,'  PSET(1, gtime());','  PSET(1, gtime());\n  if (wid == 0) RD_STAMP(17);')
# Compiler-only fences around NEW stamps: no GPU fence, barrier or math changes.
stamp_helper = """__device__ __forceinline__ void rd_stamp(unsigned long long* dst) {
  asm volatile("" ::: "memory");
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
  *dst = t;
  asm volatile("" ::: "memory");
}
"""
k=replace(k,'#define PST(k) do',stamp_helper+'#define RD_STAMP(k) do { if (threadIdx.x == 0 && p.prof) rd_stamp(p.prof + blockIdx.x * 32 + (k)); } while (0)\n#define PST(k) do')
x=replace(x,'#define PSET(k, v) do',stamp_helper+'#define RD_STAMP(k) do { if (lane == 0 && p.prof) rd_stamp(p.prof + (blockIdx.x * (NTH / 32) + wid) * 32 + (k)); } while (0)\n#define PSET(k, v) do')
# Own-kernel resources only, queried during preparation without files/processes.
for_kind = [('K','moe_k'),('X','moe_x')]
for kind, kernel in for_kind:
 body = '\nint64_t rd_resources() { cudaFuncAttributes a{}; auto err=cudaFuncGetAttributes(&a, '+kernel+'); TORCH_CHECK(err==cudaSuccess,cudaGetErrorString(err)); return ((int64_t)a.numRegs<<32) | (uint32_t)a.localSizeBytes; }\n'
 if kind=='K':k+=body
 else:x+=body
# Original functions/sources remain byte-for-byte intact; cloned helpers use separate modules.
append='\n\n# EXTERNAL PREPARATION DIAGNOSTICS ONLY; no timed-path replacement.\n_RD_K='+repr(k)+'\n_RD_X='+repr(x)+'\n'
for name,new,extold,extnew in [('M__gk','_RD_gk','M__extK','_RD_extK'),('M__wx','_RD_wx','M__extX','_RD_extX')]:
 n=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
 code=ast.get_source_segment(text,n).replace('def '+name+'(','def '+new+'(',1).replace(extold+'(',extnew+'(')
 append+='\n'+code+'\n'
# Export declarations copied exactly from original builders.
for name,sym in [('M__extK','K'),('M__extX','X')]:
 n=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
 call=next(c for c in ast.walk(n) if isinstance(c,ast.Call) and isinstance(c.func,ast.Name) and c.func.id=='load_inline')
 cpp=next(ast.literal_eval(v.value) for v in call.keywords if v.arg=='cpp_sources')
 append+='\n_RD_CPP'+sym+'='+repr(cpp+' int64_t rd_resources();')+'\n'
append+='\n'+(ROOT/'muse_moe_w4a16/research/route_diagnostic_runtime.py').read_text()
out=ROOT/'muse_decode_layer/probe_strict_v35_route_phases.py';out.write_text(text+append);compile(text+append,str(out),'exec')
# Every original AST node (including all ordinary kernel_fn definitions) is unchanged.
a=ast.parse(text).body;b=ast.parse(text+append).body
assert all(ast.dump(u,include_attributes=False)==ast.dump(v,include_attributes=False) for u,v in zip(a,b[:len(a)]))
assert len(b)>len(a)
(ROOT/'muse_moe_w4a16/research/kernel_sources/diag_route_K.cu').write_text(k)
(ROOT/'muse_moe_w4a16/research/kernel_sources/diag_route_X.cu').write_text(x)
print(out,'ORIGINAL_AST_PREFIX_IDENTICAL')
