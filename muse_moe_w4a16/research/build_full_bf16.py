import ast
from pathlib import Path
r=Path('/Users/bharathvelamala/Documents/hackathon/MuseSparkKernel/muse_moe_w4a16')
s=(r/'candidate_openkernels_moe_bf16.py').read_text()
def value(text,name):
 t=ast.parse(text)
 return next(ast.literal_eval(n.value) for n in t.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id==name for x in n.targets))
def replace_block(text,name,new):
 t=ast.parse(text);n=next(n for n in t.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id==name for x in n.targets));lines=text.splitlines(True)
 return ''.join(lines[:n.lineno-1])+name+' = '+repr(new)+'\n'+''.join(lines[n.end_lineno:])
def change(text,old,new,count=1):
 assert text.count(old)==count,(old,text.count(old),count)
 return text.replace(old,new)
for name in ('CUDA','CUDA_L'):
 c=value(s,name)
 c=change(c,'const float n0 = p.nw[mt * 16 + g] * gs, n1 = p.nw[mt * 16 + g + 8] * gs;', 'const float n0 = p.nw[mt * 16 + g], n1 = p.nw[mt * 16 + g + 8];')
 if name=='CUDA':
  c=change(c,'float v0 = f[n][0] * n0, v1 = f[n][1] * n0, v2 = f[n][2] * n1, v3 = f[n][3] * n1;', '''float q0 = __bfloat162float(__float2bfloat16(f[n][0] * gs)), q1 = __bfloat162float(__float2bfloat16(f[n][1] * gs)), q2 = __bfloat162float(__float2bfloat16(f[n][2] * gs)), q3 = __bfloat162float(__float2bfloat16(f[n][3] * gs));
      float v0=q0*n0, v1=q1*n0, v2=q2*n1, v3=q3*n1;''')
  for i in range(4):c=change(c,f'__float2bfloat16(v{i})',f'__float2bfloat16(q{i})')
 else:
  c=change(c,'const float v0 = acc[r][n][0] * n0, v1 = acc[r][n][1] * n0, v2 = acc[r][n][2] * n1, v3 = acc[r][n][3] * n1;', '''const float q0=__bfloat162float(__float2bfloat16(acc[r][n][0]*gs)), q1=__bfloat162float(__float2bfloat16(acc[r][n][1]*gs)), q2=__bfloat162float(__float2bfloat16(acc[r][n][2]*gs)), q3=__bfloat162float(__float2bfloat16(acc[r][n][3]*gs));
        const float v0=q0*n0, v1=q1*n0, v2=q2*n1, v3=q3*n1;''')
  for i in range(4):c=change(c,f'__float2bfloat16(v{i})',f'__float2bfloat16(q{i})')
  c=change(c,'const float v = __ldcg(a) * gs * p.nw[row];','const float q = __bfloat162float(__float2bfloat16(__ldcg(a) * gs));\n        const float v = q * p.nw[row];')
  c=change(c,'p.y[(size_t)r * HD + row] = __float2bfloat16(v);','p.y[(size_t)r * HD + row] = __float2bfloat16(q);')
 c=change(c,'__floats2bfloat162_rn(a[2 * j], a[2 * j + 1])','__floats2bfloat162_rn(a[2*j]*p.nw[col+2*j], a[2*j+1]*p.nw[col+2*j+1])')
 s=replace_block(s,name,c)
s=replace_block(s,'CUDA_G',value((r/'research_g1_strict_backend.py').read_text(),'CUDA_G'))
s=s.replace('"moe_w4a16_v30','"moe_fullbf16_v30').replace('"moe_w4a16_v48G"','"moe_w4a16_v48G_strict_down"')
s=change(s,'E, R1 = weights["w13"].shape[0], weights["w13"].shape[1]','E, R1 = min(32, weights["w13"].shape[0]), weights["w13"].shape[1]')
s=change(s,'w2f = torch.stack([_repack2(weights["w2"][e], weights["w2_sf"][e]) for e in range(min(E, 32))])','w2f = torch.empty(0,device=dev,dtype=torch.uint8)')
s=change(s,'[lambda: _extL(0), lambda: _extL(1)]','[lambda: _extL(0)]')
s=change(s,'"nw": 1.0 + weights["post_norm_w"].float()','"nw": (1.0 + weights["post_norm_w"].float()).to(torch.bfloat16).float()')
s=change(s,'"w1x": _repackX(w1[:32]), "w2x": _repackX(w2[:32])','"w1x": torch.empty(0,device=dev,dtype=torch.uint8), "w2x": torch.empty(0,device=dev,dtype=torch.uint8)')
s+='''
# Lean, exact-boundary research dispatch; EP8 local_start=0 contract.
TUNE=False
GS_MAX=1
G_CFG=("-DFENCE1 -DRT=16",)
K_CFG=()
WX_CFG=()
FS_CFG=()
V17_KSF=(0,)

def kernel_fn(data,state):
    if data["config"]["local_start"] != 0 or data["config"].get("local_count",32) != 32:
        raise ValueError("This research backend is scoped to EP8 local bank zero")
    m=data["hidden_states"].shape[0]
    if m==1:return _g1(data,state,("g1",0,0,"-DFENCE1 -DRT=16"))
    return _run(_default(m),data,state)
'''
p=r/'research_moe_full_bf16.py';p.write_text(s);compile(s,str(p),'exec');print(p)
