"""Isolate the existing v17 heavy-first expert ordering switch; numerical math unchanged."""
import ast
from pathlib import Path
root=Path(__file__).resolve().parents[1]
text=(root/'candidate_moe_full_strict_gk.py').read_text()
node=next(n for n in ast.parse(text).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='CUDA')
source=ast.literal_eval(node.value)
assert source.count('#if HEAVY1')==1
new='#define HEAVY1 1\n'+source
text=text.replace(ast.get_source_segment(text,node.value),repr(new),1)
text=text.replace('moe_fullbf16_v30_%d_%d','moe_heavyfirst_v30_%d_%d').replace('moe_fullbf16_v30S_','moe_heavyfirst_v30S_')
out=root/'research_moe_heavy_first.py';out.write_text(text);ast.parse(text)
(root/'research'/'kernel_sources'/'heavy_CUDA.cu').write_text(new)
print(out)
