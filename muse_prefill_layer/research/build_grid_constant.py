"""Change only W4G's kernel parameter storage; arithmetic/layout unchanged."""
import ast
from pathlib import Path
r=Path(__file__).resolve().parents[1]
s=(r/'candidate_prefill_w13_mixed.py').read_text();t=ast.parse(s)
n=next(n for n in t.body if isinstance(n,ast.Assign) and any(isinstance(a,ast.Name) and a.id=='W4G_CUDA' for a in n.targets));c=ast.literal_eval(n.value)
a='w4g_k(P p, const __grid_constant__ CUtensorMap tm)'
assert c.count(a)==1
c=c.replace(a,'w4g_k(const __grid_constant__ P p, const __grid_constant__ CUtensorMap tm)')
lines=s.splitlines(True);s=''.join(lines[:n.lineno-1])+'W4G_CUDA = '+repr(c)+'\n'+''.join(lines[n.end_lineno:])
assert s.count("'pf2_w4g_swiglu_v1'")==1 or s.count('"pf2_w4g_swiglu_v1"')==1
s=s.replace('pf2_w4g_swiglu_v1','pf2_w4g_gridp_v1')
p=r/'candidate_prefill_grid_constant.py';p.write_text(s);compile(s,str(p),'exec');print(p)
