"""Embed a pinned attention snapshot into the verified dense/strict-MoE decoder.

Only source composition happens here; numerical/graph validation remains remote.
"""
import ast
import io
import tokenize
import argparse
from pathlib import Path

root = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser()
parser.add_argument('--fp8-pv', action='store_true')
parser.add_argument('--base', type=Path)
parser.add_argument('--attention', type=Path)
parser.add_argument('--output', type=Path)
args = parser.parse_args()
base = args.base or root / 'muse_decode_layer/candidate_cp14_attention_grid_moe.py'
attention = args.attention or root / ('muse_attention_decode/candidate_openkernels_v75.py' if args.fp8_pv else 'muse_attention_decode/candidate_openkernels_v75_f16pv.py')
source = attention.read_text()
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
        token = token._replace(string='OKA_' + token.string)
    tokens.append(token)
embedded = tokenize.untokenize(tokens)
text = base.read_text()
start = text.index('# Apache-2.0 attention backend,')
end = text.index('# Apache-2.0 persistent backend,', start)
text = text[:start] + '# Apache-2.0 attention backend, pinned 47103fd9c1212868018eb615a4c6467e366e9ce0.\n' + embedded + '\n' + text[end:]
output = args.output or root / ('muse_decode_layer/candidate_cp14_v75_fp8_grid_moe.py' if args.fp8_pv else 'muse_decode_layer/candidate_cp14_v75_f16_grid_moe.py')
output.write_text(text)
compile(text, str(output), 'exec')
assert text.count('def kernel_fn(data: Inputs, state: dict):') == 1
assert 'def OKA_kernel_fn(' in text
print(output)
