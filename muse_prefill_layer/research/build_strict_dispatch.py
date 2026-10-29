"""Build strict prefill with complete-K W2 and untimed native small-shape fallback."""
import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'research_prefill_full_bf16.py').read_text()
# Do not use BF16-rounded half-K outputs as though they were a full BF16 W2 output.
assert s.count('for k2 in (1, 2)')==2
s=s.replace('for k2 in (1, 2)','for k2 in (1,)')
marker='''    """gathered rms-normed rows -> w4g W13 (H, gate|up) -> SwiGLU -> w4g W2 (ks2 K halves) -> post-norm weighted combine (+ add)"""'''
assert s.count(marker)==1
s=s.replace(marker,marker+'''\n    if ks2 != 1:
        raise ValueError("strict W2 requires one complete FP32 reduction before BF16 output")''')
old='''    if T <= 256 or cfg["num_q_heads"] != 16 or cfg["num_kv_heads"] != 2:
        if st.get("lean"):
            if "dl" not in st:
                st["dl"] = _prepare_dl(st["weights"])
            return _kernel_fn_dl(data, st["dl"])
        return _kernel_fn_dl(data, st)
'''
assert s.count(old)==1
s=s.replace(old,'    # Small-shape fallback is dispatched by the strict wrapper below.\n')

# Embed baseline with static namespacing; this preserves its FP32 small-router
# input, sliced unpool, BF16 Marlin outputs and fused residual contracts.
tree=ast.parse((ROOT/'baseline.py').read_text())
bindings=set()
def collect(body):
    for n in body:
        if isinstance(n,(ast.FunctionDef,ast.ClassDef)):bindings.add(n.name)
        elif isinstance(n,(ast.Import,ast.ImportFrom)):
            for a in n.names:bindings.add(a.asname or a.name.split('.')[0])
        elif isinstance(n,(ast.Assign,ast.AnnAssign)):
            for target in n.targets if isinstance(n,ast.Assign) else [n.target]:
                bindings.update(x.id for x in ast.walk(target) if isinstance(x,ast.Name))
        elif isinstance(n,ast.Try):
            collect(n.body);collect(n.orelse);collect(n.finalbody)
            for h in n.handlers:collect(h.body)
        elif isinstance(n,ast.If):collect(n.body);collect(n.orelse)
collect(tree.body)
rename={name:'strict_native_'+name for name in bindings}
class Prefix(ast.NodeTransformer):
    def visit_Name(self,n):n.id=rename.get(n.id,n.id);return n
    def visit_FunctionDef(self,n):n.name=rename.get(n.name,n.name);return self.generic_visit(n)
    def visit_ClassDef(self,n):n.name=rename.get(n.name,n.name);return self.generic_visit(n)
    def visit_Import(self,n):
        for a in n.names:
            key=a.asname or a.name.split('.')[0]
            if key in rename:a.asname=rename[key]
        return n
    def visit_ImportFrom(self,n):
        for a in n.names:
            key=a.asname or a.name
            if key in rename:a.asname=rename[key]
        return n
native=ast.unparse(ast.fix_missing_locations(Prefix().visit(tree)))
s+='\n# Native fallback embedded from the local challenge baseline, no sibling imports.\n'+native+'''

_strict_prefill_prepare = prepare
_strict_prefill_kernel = kernel_fn

def prepare(weights):
    # All fallback work is weight-derived and performed before timed calls.
    native_state = strict_native_prepare(weights)
    state = _strict_prefill_prepare(weights)
    state['strict_native_fallback'] = native_state
    return state

def kernel_fn(data, state):
    cfg = data['config']
    if cfg['tokens'] <= 256 or cfg['num_q_heads'] != 16 or cfg['num_kv_heads'] != 2:
        return strict_native_kernel_fn(data, state['strict_native_fallback'])
    return _strict_prefill_kernel(data, state)
'''
out=ROOT/'candidate_prefill_strict.py';out.write_text(s);compile(s,str(out),'exec')
assert 'st["dl"] = _prepare_dl(st["weights"])' not in s
print(out)
