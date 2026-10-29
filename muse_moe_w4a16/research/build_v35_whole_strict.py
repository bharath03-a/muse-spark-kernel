"""Repair live MoE BF16 boundaries while preserving v35 whole frontend/PDL glue.
No xg2 or new performance features. Raw retired FS is excluded from all dispatch.
"""
import ast
import difflib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
WHOLE=ROOT/'muse_decode_layer/candidate_v35_wxpin32.py'
RAW=Path('/tmp/muse-openkernels-e6b48c8/muse_moe_w4a16/submission_v56.py')
STRICT=ROOT/'muse_moe_w4a16/research_moe_v56_strict.py'

def constants(source):
 return {n.targets[0].id:(n,ast.literal_eval(n.value)) for n in ast.parse(source).body
         if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)}

def replay(old,new,target):
 a,b=old.splitlines(True),new.splitlines(True);edits=[]
 for tag,i1,i2,j1,j2 in difflib.SequenceMatcher(None,a,b,autojunk=False).get_opcodes():
  if tag=='equal':continue
  lo,hi=max(i1-1,0),min(i2+1,len(a));before=''.join(a[lo:hi]);after=''.join(a[lo:i1])+''.join(b[j1:j2])+''.join(a[i2:hi])
  assert target.count(before)==1,(before[:120],target.count(before));edits.append((before,after))
 for before,after in edits:target=target.replace(before,after,1)
 return target,len(edits)

source=WHOLE.read_text();raw=constants(RAW.read_text());strict=constants(STRICT.read_text());whole=constants(source)
text=source
for name in ['CUDA','CUDA_L','CUDA_G','CUDA_K','CUDA_X']:
 node,body=whole['M_'+name];patched,count=replay(raw[name][1],strict[name][1],body)
 text=text.replace(ast.get_source_segment(source,node.value),repr(patched),1)
 print(name,'precision/grid-P hunks',count)
# Effective norm weight is materialized BF16 by baseline before the norm uses it.
a='"nw": 1.0 + weights["post_norm_w"].float()';b='"nw": (1.0 + weights["post_norm_w"].float()).to(torch.bfloat16).float()'
assert text.count(a)==1;text=text.replace(a,b,1)
# No live unpatched FS path: remove tuning candidates and make forced/stale FS use strict default.
a='    M_FS_CFG = ((1, 3, ""),)';assert text.count(a)==1;text=text.replace(a,'    M_FS_CFG = ()  # Strict repair: retired unpatched FS is never a tune candidate.',1)
a='    if cfg[0] == "fs":\n        return M__fs(data, state, cfg)';assert text.count(a)==1
text=text.replace(a,'    if cfg[0] == "fs":\n        return M__run(M__default(data["hidden_states"].shape[0]), data, state)  # strict fallback; FS is retired',1)
# New CUDA source identity; leave all projection/attention/glue extension names untouched.
text=text.replace('"moe_w4a16_v','"moe_v35strict_v')
out=ROOT/'muse_decode_layer/candidate_v35_wxpin32_strict.py';out.write_text(text);compile(text,str(out),'exec')
# Prove non-MoE top-level AST is identical, including every whole kernel/prepare wrapper.
allowed_constants={'M_CUDA','M_CUDA_L','M_CUDA_G','M_CUDA_K','M_CUDA_X'}
def scrub(t):
 t=ast.parse(t)
 for n in t.body:
  if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in allowed_constants:n.value=ast.Constant('<strict CUDA>')
  if isinstance(n,ast.FunctionDef) and n.name in {'M_prepare','M__run','M__ext','M__extF','M__extG','M__extSB','M__extK','M__extX','M__extL','M__extS'}:n.body=[ast.Pass()]
  if isinstance(n,ast.If) and any(isinstance(v,ast.Name) and v.id=='DL_PRUNE' for v in ast.walk(n.test)):
   for a in n.body:
    if isinstance(a,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='M_FS_CFG' for v in a.targets):a.value=ast.Constant('<disabled retired FS>')
 return ast.dump(t,include_attributes=False)
assert scrub(source)==scrub(text),'non-MoE glue AST changed'
print(out,'NON_MOE_GLUE_AST_IDENTICAL')
