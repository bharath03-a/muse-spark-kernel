"""Port only pinned v52 WXG2 into the strict/grid baseline, with exact BF16 boundaries."""
import ast
from pathlib import Path
root=Path(__file__).resolve().parents[1]
base=(root/'research_moe_grid_constant.py').read_text()
up=Path('/tmp/muse-openkernels-47103fd/muse_moe_w4a16/submission_v52.py').read_text()
def value(text,name):
 return next(ast.literal_eval(n.value) for n in ast.parse(text).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')==name)
def change(s,a,b,n=1):
 assert s.count(a)==n,(a[:100],s.count(a),n)
 return s.replace(a,b)
c=value(up,'CUDA_X')
helper='''#include <cuda_bf16.h>
// Strict SGLang boundaries: BF16 gate/up -> FP32 SiLU/product -> BF16 inter.
__device__ __forceinline__ float strict_swiglu(float gate, float up) {
  gate=__bfloat162float(__float2bfloat16_rn(gate));
  up=__bfloat162float(__float2bfloat16_rn(up));
  return gate/(1.f+__expf(-gate))*up;
}
'''
c=change(c,'#include <cuda_bf16.h>',helper)
c=change(c,'gt / (1.f + __expf(-gt)) * up','strict_swiglu(gt, up)')
c=change(c,'const float n0 = p.nw[mt * 16 + g] * gs, n1 = p.nw[mt * 16 + g + 8] * gs;','const float n0 = p.nw[mt * 16 + g], n1 = p.nw[mt * 16 + g + 8];')
c=change(c,'const float v0 = acc[i * 4] * n0, v1 = acc[i * 4 + 1] * n0, v2 = acc[i * 4 + 2] * n1, v3 = acc[i * 4 + 3] * n1;', '''// Complete W2, apply its own expert's global scale, then round BEFORE norm.
      const float q0=__bfloat162float(__float2bfloat16(acc[i*4]*gs)), q1=__bfloat162float(__float2bfloat16(acc[i*4+1]*gs)), q2=__bfloat162float(__float2bfloat16(acc[i*4+2]*gs)), q3=__bfloat162float(__float2bfloat16(acc[i*4+3]*gs));
      const float v0=q0*n0, v1=q1*n0, v2=q2*n1, v3=q3*n1;''')
for i in range(4):c=change(c,f'__float2bfloat16(v{i})',f'__float2bfloat16(q{i})')
c=change(c,'__floats2bfloat162_rn(a[2 * j], a[2 * j + 1])','__floats2bfloat162_rn(a[2*j]*p.nw[col+2*j], a[2*j+1]*p.nw[col+2*j+1])')
node=next(n for n in ast.parse(base).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='CUDA_X')
text=base.replace(ast.get_source_segment(base,node.value),repr(c),1)
text=change(text,'"moe_w4a16_v48X"','"moe_strict_wxg2_v52X"')
text=change(text,'"w1x": torch.empty(0,device=dev,dtype=torch.uint8), "w2x": torch.empty(0,device=dev,dtype=torch.uint8)','"w1x": _repackX(w1), "w2x": _repackX(w2)')
text+='''
# Research dispatch only; final interval requires paired evidence on held-nearby shapes.
# Source: lee101/openkernels47103fd, submission_v52.py CUDA_X, strict boundary edits above.
WX_CFG=(("-DRPW=2", "-DWXG2"),)
WX_RESEARCH_CONFIG=("-DRPW=2", "-DWXG2")
_strict_grid_fallback=kernel_fn

def kernel_fn(data,state):
    m=data["hidden_states"].shape[0]
    if 32<=m<=512:
        return _wx(data,state,WX_RESEARCH_CONFIG)
    return _strict_grid_fallback(data,state)
'''
out=root/'research_moe_latest_wx_strict.py';out.write_text(text);ast.parse(text)
(root/'research'/'kernel_sources'/'wx52_strict.cu').write_text(c)
# Every non-WX CUDA module must remain bit-for-bit identical to strict/grid fallback.
for n in ast.parse(base).body:
 if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str):
  name=getattr(n.targets[0],'id','')
  if name.startswith('CUDA') and name!='CUDA_X':assert value(text,name)==n.value.value,name
print(out)
