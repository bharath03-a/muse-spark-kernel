"""Prototype exact paired-row W13+SwiGLU epilogue; not enabled in full-layer dispatch."""
import ast
from pathlib import Path
r=Path(__file__).resolve().parents[1];s=(r/'research_prefill_full_bf16.py').read_text();t=ast.parse(s)
n=next(n for n in t.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='W4G_CUDA' for x in n.targets));c=ast.literal_eval(n.value)
def change(old,new):
 global c
 assert c.count(old)==1,(old,c.count(old));c=c.replace(old,new)
change('int nrb256, nout, rows, ksplit, tsz, lomod, fullonly;','int nrb256, nout, rows, ksplit, tsz, lomod, fullonly, fuse;')
change('h.rb * C::BM + wi * 64 + 16 * w + g;', '(p.fuse ? (h.rb * C::BM + wi * 64 + 16 * w) / 2 : h.rb * C::BM + wi * 64 + 16 * w) + g;')
change('o[0] = __float2bfloat16(acc[4 * j + e2] * gs); o[8] = __float2bfloat16(acc[4 * j + 2 + e2] * gs);','''if (p.fuse) {
          const float gate=__bfloat162float(__float2bfloat16(acc[4*j+e2]*gs));
          const float up=__bfloat162float(__float2bfloat16(acc[4*j+2+e2]*gs));
          o[0]=__float2bfloat16(gate/(1.f+__expf(-gate))*up);
        } else { o[0] = __float2bfloat16(acc[4 * j + e2] * gs); o[8] = __float2bfloat16(acc[4 * j + 2 + e2] * gs); }''')
change('int64_t ksplit, int64_t cls, int64_t mode) {','int64_t ksplit, int64_t cls, int64_t mode, int64_t fuse) {\n  TORCH_CHECK(!fuse || ksplit == 1, "SwiGLU requires the complete W13 reduction");')
change('(int)(out.size(0) / ksplit), (int)ksplit, BNMAX, 0, 0};','(int)(out.size(0) / ksplit), (int)ksplit, BNMAX, 0, 0, (int)fuse};')
lines=s.splitlines(True);s=''.join(lines[:n.lineno-1])+'W4G_CUDA = '+repr(c)+'\n'+''.join(lines[n.end_lineno:])
s=s.replace('"pf2_w4g_v12"','"pf2_w4g_swiglu_v1"')
old='void w4g(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);';assert s.count(old)==1;s=s.replace(old,old[:-2]+', int64_t);')
s=s.replace('def _w4gemm(A, offs, Wp, g, ctr, ksplit=1, cls=3, prof=None, ext=None, wide=0):','def _w4gemm(A, offs, Wp, g, ctr, ksplit=1, cls=3, prof=None, ext=None, wide=0, fuse=False):')
s=s.replace('out = torch.empty((ksplit * A.shape[0], Wp.shape[1] * 256), dtype=torch.bfloat16, device=A.device)','out = torch.empty((ksplit * A.shape[0], Wp.shape[1] * (128 if fuse else 256)), dtype=torch.bfloat16, device=A.device)')
s=s.replace('.w4g(A, offs, Wp, g, out, ctr, prof, ksplit, cls, wide)','.w4g(A, offs, Wp, g, out, ctr, prof, ksplit, cls, wide, int(fuse))')
p=r/'research_prefill_w13_fused.py';p.write_text(s);compile(s,str(p),'exec');print(p)
