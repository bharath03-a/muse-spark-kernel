"""Isolate immutable MoE parameter storage in the verified whole decoder."""
import ast
from pathlib import Path

root = Path(__file__).resolve().parent
text = (root / 'candidate_cp14_attention_strict_moe.py').read_text()
nodes = [node for node in ast.parse(text).body if isinstance(node, ast.Assign)
         and getattr(node.targets[0], 'id', '') in ('SW_CUDA', 'SW_CUDA_L')]
assert len(nodes) == 2
for node in reversed(nodes):
    original = ast.literal_eval(node.value)
    before = 'moe_kernel(P p)'
    after = 'moe_kernel(const __grid_constant__ P p)'
    assert original.count(before) == 1
    revised = original.replace(before, after)
    assert revised.replace(after, before) == original
    literal = ast.get_source_segment(text, node.value)
    assert text.count(literal) == 1
    text = text.replace(literal, repr(revised), 1)
text = text.replace('moe_fullbf16_v30', 'moe_gridconstant_v30')
output = root / 'candidate_cp14_attention_grid_moe.py'
output.write_text(text)
compile(text, str(output), 'exec')
print(output)
