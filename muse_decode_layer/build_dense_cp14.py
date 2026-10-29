"""Build a self-contained decode candidate with statically namespaced cp14 projections."""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / 'muse_decode_layer/submission.py'
DENSE = ROOT / 'muse_bf16_proj/checkpoints/cp14_D_slice_check1.426.py'
LEAN = '--lean' in sys.argv
OUT = ROOT / ('muse_decode_layer/candidate_dense_cp14_v2.py' if LEAN else 'muse_decode_layer/candidate_dense_cp14.py')
tree = ast.parse(DENSE.read_text())
bindings = set()
def collect(body):
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            bindings.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                bindings.add(a.asname or a.name.split('.')[0])
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                bindings.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
        elif isinstance(node, ast.Try):
            collect(node.body); collect(node.orelse); collect(node.finalbody)
            for h in node.handlers: collect(h.body)
        elif isinstance(node, ast.If):
            collect(node.body); collect(node.orelse)
collect(tree.body)
rename = {name: 'cp14_' + name for name in bindings}
class Prefix(ast.NodeTransformer):
    def visit_Name(self, node):
        node.id = rename.get(node.id, node.id)
        return node
    def visit_FunctionDef(self, node):
        node.name = rename.get(node.name, node.name)
        return self.generic_visit(node)
    def visit_ClassDef(self, node):
        node.name = rename.get(node.name, node.name)
        return self.generic_visit(node)
    def visit_Import(self, node):
        for a in node.names:
            old = a.asname or a.name.split('.')[0]
            if old in rename: a.asname = rename[old]
        return node
    def visit_ImportFrom(self, node):
        for a in node.names:
            old = a.asname or a.name
            if old in rename: a.asname = rename[old]
        return node
embedded = ast.unparse(ast.fix_missing_locations(Prefix().visit(tree)))
wrapper = '''
# Dispatch only after both states are prepared; original warmups use original kernels.
_cp14_original_prepare = prepare
_cp14_original_gemm = _fast_gemm
_cp14_original_router = _fast_router

def prepare(weights):
    import time
    started = time.monotonic()
    state = _cp14_original_prepare(weights)
    # Reserve time for slice validation and whole-layer state construction.
    global cp14_R15_PREPARE_BUDGET_S
    cp14_R15_PREPARE_BUDGET_S = max(0.0, 210.0 - (time.monotonic() - started))
    state['cp14_projection'] = cp14_prepare({k: weights[k] for k in ('qkv_gate', 'o', 'pooler', 'unpooler', 'router')})
    print(f'cp14 whole-layer prepare seconds: {time.monotonic() - started:.1f}')
    return state

def _fast_gemm(x, key, state, out_buf=None):
    if 'cp14_projection' not in state:
        return _cp14_original_gemm(x, key, state, out_buf=out_buf)
    layer = 'unpooler' if key == 'unpooler_slice' else key
    n, k = state['dense'][key].shape
    return cp14_kernel_fn({'x': x, 'config': {'m': x.shape[0], 'n': n, 'k': k, 'layer': layer}}, state['cp14_projection'])

def _fast_router(x, state, out_buf=None):
    if 'cp14_projection' not in state:
        return _cp14_original_router(x, state, out_buf=out_buf)
    return cp14_kernel_fn({'x': x, 'config': {'m': x.shape[0], 'n': 256, 'k': x.shape[1], 'layer': 'router'}}, state['cp14_projection'])
'''
base_source = BASE.read_text()
if LEAN:
    # Dense cp4 execution is unreachable once cp14 state is installed. Preserve
    # all layer, MoE, norm, routing and buffer setup, including descriptor state.
    old_compile = """    # Compile custom CUDA extension (TMA + WGMMA + Cluster DSMEM)
    ext = None
    try:
        ext = _build_cuda()
    except Exception as exc:
        print(f"CUDA extension failed to load, falling back to Triton: {exc}")
"""
    assert base_source.count(old_compile) == 1
    base_source = base_source.replace(old_compile, "    # cp14 owns dense compilation; skip unused cp4 extension.\n    ext = None\n")
    start = base_source.index('    # Warmup all shapes to compile kernels ahead of CUDA graph capture')
    end = base_source.index('    del freqs', start)
    base_source = base_source[:start] + '    # cp14_prepare performs all dense warmups once.\n' + base_source[end:]
OUT.write_text(base_source + '\n# Embedded cp14 projections: no sibling imports.\n' + embedded + '\n' + wrapper)
compile(OUT.read_text(), str(OUT), 'exec')
print(OUT)
