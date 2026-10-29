"""Prevent per-thread materialization of immutable kernel parameters; no numerical edits."""
import ast
from pathlib import Path
root=Path(__file__).resolve().parents[1]
text=(root/'candidate_moe_full_strict_gk.py').read_text()
for node in ast.parse(text).body:
 if not isinstance(node,ast.Assign) or getattr(node.targets[0],'id','') not in ('CUDA','CUDA_L'):continue
 name=node.targets[0].id;source=ast.literal_eval(node.value)
 old='moe_kernel(P p)';new='moe_kernel(const __grid_constant__ P p)'
 assert source.count(old)==1
 revised=source.replace(old,new)
 assert revised.replace(new,old)==source
 text=text.replace(ast.get_source_segment(text,node.value),repr(revised),1)
 (root/'research'/'kernel_sources'/f'grid_constant_{name}.cu').write_text(revised)
text=text.replace('moe_fullbf16_v30','moe_gridconstant_v30')
out=root/'research_moe_grid_constant.py';out.write_text(text);ast.parse(text)
print(out)
