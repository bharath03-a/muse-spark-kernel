"""Strict adaptation of Apache-2.0 openkernels revision47103fd, pf32.

Preserve original candidates and proven native fallback. No GPU or submissions.
"""
import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
UP=Path('/tmp/muse-openkernels-47103fd/muse_prefill_layer/submission.py')
s=UP.read_text()
header='# okc: problem=muse_prefill_layer\n'
assert s.startswith(header)
s=header+'# Adapted from lee101/openkernels revision47103fd9c1212868018eb615a4c6467e366e9ce0.\n# Copyright2026 Lee Penkman. Apache-2.0; see OPENKERNELS_LICENSE.txt.\n'+s[len(header):]

def patch_cuda(name,fn):
 global s
 tree=ast.parse(s)
 n=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(a,ast.Name) and a.id==name for a in n.targets))
 c=fn(ast.literal_eval(n.value));lines=s.splitlines(True)
 s=''.join(lines[:n.lineno-1])+name+' = '+repr(c)+'\n'+''.join(lines[n.end_lineno:])

def q_strict(c):
 def change(old,new):
  nonlocal c
  assert c.count(old)==1,(old[:90],c.count(old));c=c.replace(old,new)
 change('#include <cuda_bf16.h>','''#include <cuda_bf16.h>
__device__ __forceinline__ float pf32_strict_swiglu(float gate,float up) {
 gate=__bfloat162float(__float2bfloat16_rn(gate));
 up=__bfloat162float(__float2bfloat16_rn(up));
 return gate/(1.f+__expf(-gate))*up;
}
''')
 change('gt / (1.f + __expf(-gt)) * up','pf32_strict_swiglu(gt, up)')
 change('const float n0 = p.nw[mt * 16 + g] * gs, n1 = p.nw[mt * 16 + g + 8] * gs;','const float n0 = p.nw[mt * 16 + g], n1 = p.nw[mt * 16 + g + 8];')
 change('const float v0 = acc[r][n][0] * n0, v1 = acc[r][n][1] * n0, v2 = acc[r][n][2] * n1, v3 = acc[r][n][3] * n1;', '''const float q0=__bfloat162float(__float2bfloat16(acc[r][n][0]*gs)),q1=__bfloat162float(__float2bfloat16(acc[r][n][1]*gs)),q2=__bfloat162float(__float2bfloat16(acc[r][n][2]*gs)),q3=__bfloat162float(__float2bfloat16(acc[r][n][3]*gs));
        const float v0=q0*n0,v1=q1*n0,v2=q2*n1,v3=q3*n1;''')
 for i in range(4):change(f'__float2bfloat16(v{i})',f'__float2bfloat16(q{i})')
 change('__floats2bfloat162_rn(a[2 * j], a[2 * j + 1])','__floats2bfloat162_rn(a[2*j]*p.nw[col+2*j],a[2*j+1]*p.nw[col+2*j+1])')
 return c
patch_cuda('Q_CUDA',q_strict)

def w4_strict(c):
 old='const float gt = acc[4 * j + e2] * gs, up = acc[4 * j + 2 + e2] * gs;'
 assert c.count(old)==1
 c=c.replace(old,'const float gt = __bfloat162float(__float2bfloat16_rn(acc[4*j+e2]*gs)), up = __bfloat162float(__float2bfloat16_rn(acc[4*j+2+e2]*gs));')
 old='int64_t ksplit, int64_t cls, int64_t mode, int64_t swiglu, torch::Tensor ss, torch::Tensor gw, int64_t pdl) {'
 assert c.count(old)==1
 c=c.replace(old,old+'\n  TORCH_CHECK(ksplit == 1, "Strict W4 requires complete K reduction before BF16 output");\n  TORCH_CHECK(!ss.numel(), "FP32 W2 norm epilogue disabled in strict variant");')
 return c
patch_cuda('W4G_CUDA',w4_strict)
old='"nw": 1.0 + weights["post_norm_w"].float(), "ctr": torch.zeros(128'
assert s.count(old)==1;s=s.replace(old,'"nw": (1.0 + weights["post_norm_w"].float()).to(torch.bfloat16).float(), "ctr": torch.zeros(128')
s=s.replace('st["g_post_expert_norm"] = (1.0 + weights["post_expert_norm"].float()).contiguous()','st["g_post_expert_norm"] = (1.0 + weights["post_expert_norm"].float()).to(torch.bfloat16).float().contiguous()')
old='st["g_" + k] = (1.0 + weights[k].float()).contiguous()'
assert s.count(old)==2;s=s.replace(old,'st["g_" + k] = ((1.0 + weights[k].float()).to(torch.bfloat16).float() if k == "post_expert_norm" else 1.0 + weights[k].float()).contiguous()')
assert s.count('for k2 in (1, 2)')==1;s=s.replace('for k2 in (1, 2)','for k2 in (1,)')
marker='''    """gathered rms-normed rows -> w4g W13 (H, gate|up) -> SwiGLU -> w4g W2 (ks2 K halves) -> post-norm weighted combine (+ add)"""'''
assert s.count(marker)==1;s=s.replace(marker,marker+'\n    if ks2 != 1 or W2SS:\n        raise ValueError("strict W2 full reduction and BF16 output required")')
lazy='''    if T <= 256 or cfg["num_q_heads"] != 16 or cfg["num_kv_heads"] != 2:
        if st.get("lean"):
            if "dl" not in st:
                st["dl"] = _prepare_dl(st["weights"])
            return _kernel_fn_dl(data, st["dl"])
        return _kernel_fn_dl(data, st)
'''
assert s.count(lazy)==1;s=s.replace(lazy,'    # Untimed native fallback handled by final strict wrapper.\n')
s=s.replace('pf2_w4g_v20','pf2_w4g_pf32_strict_v1').replace('moe_w4a16_v23_','moe_pf32_strict_v23_')
strict=(ROOT/'candidate_prefill_strict.py').read_text();marker='# Native fallback embedded from the local challenge baseline, no sibling imports.'
assert strict.count(marker)==1;s+='\n'+strict[strict.index(marker):]
out=ROOT/'candidate_prefill_pf32_strict.py';out.write_text(s);compile(s,str(out),'exec')
assert 'for k2 in (1, 2)' not in s
assert 'st["dl"] = _prepare_dl(st["weights"])' not in s
print(out)
