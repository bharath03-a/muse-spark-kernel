"""Swap the embedded strict MoE for strict v56 (+grid fallback) with shape-only dispatch.

Evidence: actual whole-frontend routes (app ap-uVa9aO0F17EtOr1fO6Ai04): M32 1.47x, M256 1.06-1.13x,
M128 0.96-0.99x versus strict grid. Dispatch uses only M.
"""
import ast
import io
import tokenize
from pathlib import Path

root = Path(__file__).resolve().parent.parent
base = root / 'muse_decode_layer/candidate_cp14_v75_f16_grid_moe.py'
moe = root / 'muse_moe_w4a16/research_moe_v56_strict.py'

source = moe.read_text()
names = set()
for node in ast.parse(source).body:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        names.add(node.name)
    elif isinstance(node, (ast.Assign, ast.AnnAssign)):
        for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
            names.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
tokens = []
for token in tokenize.generate_tokens(io.StringIO(source).readline):
    if token.type == tokenize.NAME and token.string in names:
        token = token._replace(string='SW_' + token.string)
    tokens.append(token)
embedded = tokenize.untokenize(tokens)

text = base.read_text()
start = text.index('# Apache-2.0 persistent backend, copyright 2026 Lee Penkman;')
end = text.index('_strict_combo_prepare = prepare')
dispatch = '''
# Whole-layer shape-only dispatch (in-layer routes favour wx from M17; ap-uVa9aO0F17EtOr1fO6Ai04).
def SW_kernel_fn(data, state):
    m = data["hidden_states"].shape[0]
    if m <= 8:
        cfg = ("gk", 0, 0, "-DSW=4 -DKS2=1 -DMAXM=8")
    elif m <= 16:
        cfg = ("gk", 0, 0, "-DSW=4 -DKS2=2")
    elif m <= 192:
        cfg = ("wx", 0, 0, "-DNCW=16 -DWXG2 -DUDESC")
    else:
        cfg = ("wx", 0, 0, "-DNCW=16 -DN8STEP -DWXG2 -DUDESC")
    return SW__run(cfg, data, state)

'''
text = (text[:start]
        + '# Apache-2.0 persistent backend, copyright 2026 Lee Penkman (v56: openkernels e6b48c833d8c4c1ebbca16b80d784b9e3d5bb86b); '
          'strict BF16 modifications by this research team.\n'
        + embedded + dispatch + text[end:])
out = root / 'muse_decode_layer/candidate_cp14_v75_f16_v56moe.py'
out.write_text(text)
compile(text, str(out), 'exec')
assert text.count('\ndef SW_kernel_fn(') == 3
assert 'def OKA_kernel_fn(' in text
print(out)
