"""Fuse exact indexing and sliced-output writes; keep all numerical math intact."""
import ast
from pathlib import Path

root = Path(__file__).resolve().parent
base = root / 'candidate_cp14_v81_minimal_v56moe.py'
probe = ast.parse((root.parent / 'research_harness/modal_launch_fusions.py').read_text())
compare = next(n for n in probe.body if isinstance(n, ast.FunctionDef) and n.name == 'compare')
functions = []
for node in compare.body:
    if isinstance(node, ast.FunctionDef) and node.name in ('localize', 'sliced_output'):
        node.name = '_launch_fused_' + node.name
        functions.append(ast.unparse(node))
assert len(functions) == 2
s = base.read_text()

def replace_once(old, new):
    global s
    assert s.count(old) == 1, (old, s.count(old))
    s = s.replace(old, new)

replace_once('        "router_buf": router_buf,', '''        "router_buf": router_buf,
        "local_ids_buf": torch.empty((MAX_TOKENS, TOP_K), dtype=torch.int32, device=device),''')
replace_once('''        local_ids = (topk_ids - data["local_start"]) % N_EXPERTS
        local_ids = torch.where(local_ids < LOCAL_EXPERTS, local_ids, -1)''', '''        local_ids = state["local_ids_buf"][:tokens]
        _launch_fused_localize[(triton.cdiv(tokens * TOP_K, 256),)](
            topk_ids, local_ids, data["local_start"], tokens * TOP_K, 256)''')
replace_once('''        ffn_output[:, :UNPOOL_SLICE].copy_(slice_out)
        ffn_output[:, UNPOOL_SLICE:].zero_()''', '''        # Both preallocated buffers have contiguous token rows.
        _launch_fused_sliced_output[(triton.cdiv(tokens * HIDDEN, 256),)](
            slice_out, ffn_output, tokens * HIDDEN, 256)''')
s += '\n\n# Exact integer localization and lossless sliced-output padding.\n' + '\n\n'.join(functions) + '\n'
out = root / 'candidate_cp14_v81_v56_launch_fusions.py'
compile(s, str(out), 'exec')
out.write_text(s)
print(out)
