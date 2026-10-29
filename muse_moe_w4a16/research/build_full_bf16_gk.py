"""Add exact down-output BF16 rounding to the selected shared-activation kernel."""
import ast
from pathlib import Path
root=Path(__file__).resolve().parents[1]
s=(root/'research_moe_full_bf16.py').read_text();t=ast.parse(s)
n=next(n for n in t.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='CUDA_K' for x in n.targets));c=ast.literal_eval(n.value)
def change(old,new):
 global c
 assert c.count(old)==1,(old,c.count(old));c=c.replace(old,new)
change('const float* yb = ybl;','float* yb = ybl;')
change('    __syncthreads();\n    const int onr = sm->onr;','''    __syncthreads();
    // Complete the W2 reduction and BF16 materialization before norm weighting.
    for (int i = tid; i < nr * 64; i += NW * 32) {
      const int r = i >> 6, rr = i & 63;
      if (rr < sm->onr) yb[i] = __bfloat162float(__float2bfloat16(yb[i] * sm->gs2[sm->rexp[r]]));
    }
    __syncthreads();
    const int onr = sm->onr;''')
change('sm->cf[r] = p.tw[tid] * p.g2[p.lo + e];','sm->cf[r] = p.tw[tid];')
change('sm->coef[r] = sm->cf[r] * rsqrtf(ssum * g2 * g2 * (1.f / HD) + 1e-8f);','sm->coef[r] = sm->cf[r] * rsqrtf(ssum * (1.f / HD) + 1e-8f);')
lines=s.splitlines(True);s=''.join(lines[:n.lineno-1])+'CUDA_K = '+repr(c)+'\n'+''.join(lines[n.end_lineno:])
s=s.replace('"moe_w4a16_v48K"','"moe_fullbf16_v48K"')
s=s.replace('K_CFG=()','K_CFG=("-DSW=4 -DKS2=2",)')
s=s.replace('    if m==1:return _g1(data,state,("g1",0,0,"-DFENCE1 -DRT=16"))','    if m==1:return _g1(data,state,("g1",0,0,"-DFENCE1 -DRT=16"))\n    if m<=16:return _gk(data,state,("gk",0,0,"-DSW=4 -DKS2=2"))')
p=root/'research_moe_full_bf16_gk.py';p.write_text(s);compile(s,str(p),'exec');print(p)
