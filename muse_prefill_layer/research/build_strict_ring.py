"""Complete the BF16 down-output and norm-weight boundaries for prefill ring-Q."""
import ast
from pathlib import Path
r=Path(__file__).resolve().parents[1];s=(r/'candidate_openkernels_prefill_bf16_gate.py').read_text();t=ast.parse(s)
n=next(n for n in t.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='Q_CUDA' for x in n.targets));c=ast.literal_eval(n.value)
def change(old,new):
 global c
 assert c.count(old)==1,(old,c.count(old));c=c.replace(old,new)
change('const float n0 = p.nw[mt * 16 + g] * gs, n1 = p.nw[mt * 16 + g + 8] * gs;','const float n0 = p.nw[mt * 16 + g], n1 = p.nw[mt * 16 + g + 8];')
change('const float v0 = acc[r][n][0] * n0, v1 = acc[r][n][1] * n0, v2 = acc[r][n][2] * n1, v3 = acc[r][n][3] * n1;', '''const float q0=__bfloat162float(__float2bfloat16(acc[r][n][0]*gs)), q1=__bfloat162float(__float2bfloat16(acc[r][n][1]*gs)), q2=__bfloat162float(__float2bfloat16(acc[r][n][2]*gs)), q3=__bfloat162float(__float2bfloat16(acc[r][n][3]*gs));
        const float v0=q0*n0, v1=q1*n0, v2=q2*n1, v3=q3*n1;''')
for i in range(4):change(f'__float2bfloat16(v{i})',f'__float2bfloat16(q{i})')
change('__floats2bfloat162_rn(a[2 * j], a[2 * j + 1])','__floats2bfloat162_rn(a[2*j]*p.nw[col+2*j],a[2*j+1]*p.nw[col+2*j+1])')
lines=s.splitlines(True);s=''.join(lines[:n.lineno-1])+'Q_CUDA = '+repr(c)+'\n'+''.join(lines[n.end_lineno:])
old='"nw": 1.0 + weights["post_norm_w"].float(), "ctr": torch.zeros(128'
assert s.count(old)==1;s=s.replace(old,'"nw": (1.0 + weights["post_norm_w"].float()).to(torch.bfloat16).float(), "ctr": torch.zeros(128')
s=s.replace('"moe_w4a16_v23_"','"moe_prefill_fullbf16_v23_"')
s=s.replace('st["g_post_expert_norm"] = (1.0 + weights["post_expert_norm"].float()).contiguous()','st["g_post_expert_norm"] = (1.0 + weights["post_expert_norm"].float()).to(torch.bfloat16).float().contiguous()')
old='st["g_" + k] = (1.0 + weights[k].float()).contiguous()';new='st["g_" + k] = ((1.0 + weights[k].float()).to(torch.bfloat16).float() if k == "post_expert_norm" else 1.0 + weights[k].float()).contiguous()';s=s.replace(old,new)
p=r/'research_prefill_full_bf16.py';p.write_text(s);compile(s,str(p),'exec');print(p)
