"""CPU-only candidate identity/dispatch regression; imports no CUDA libraries."""
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'bench/top50_candidate_router_cluster_bt64.py'
source=PATH.read_bytes()
base=(ROOT/'checkpoints/cp18_cw8_meta_pooler_pass.py').read_bytes()
assert source.startswith(base)
tree=ast.parse(source)
probe=ast.parse((ROOT/'bench/router_cluster_bt64.py').read_text())
renames={'layouts':'_rcb_layouts','_copy_tile':'_rcb_copy_tile','_issue':'_rcb_issue','router_cluster_bt64':'_rcb_kernel'}
def function(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
class Rename(ast.NodeTransformer):
    def visit_Name(self,node):node.id=renames.get(node.id,node.id);return node
for original,inlined in renames.items():
    node=Rename().visit(function(probe,original));node.name=inlined
    assert ast.dump(node,include_attributes=False)==ast.dump(function(tree,inlined),include_attributes=False)

# Execute the actual final dispatch function against shape-only stand-ins.
# In particular STORE is False, but remains an enabled dictionary member.
bf16=object()
calls=[]
def run(x,state,c,mode):calls.append((x,c,mode));return ('candidate',x)
def fallback(data,state):return ('baseline',data['x'])
ns=dict(torch=SimpleNamespace(bfloat16=bf16),_rcb_run=run,_rcb_base_kernel=fallback)
node=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='kernel_fn'][-1]
exec(compile(ast.Module(body=[node],type_ignores=[]),str(PATH),'exec'),ns)
class X:
    def __init__(self,m=128,k=8192,stride=None,alignment=16,dtype=bf16):
        self.shape=(m,k);self.dtype=dtype;self._stride=stride or (k,1);self._alignment=alignment
    def stride(self):return self._stride
    def data_ptr(self):return self._alignment

def data(x,**overrides):
    cfg=dict(layer='router',m=x.shape[0],n=256,k=8192);cfg.update(overrides)
    return dict(x=x,config=cfg)
state={'rcb_selected':{128:False,256:True}}
for m,c,mode in ((128,8,False),(256,4,True)):
    for _ in range(2):
        x=X(m);result=ns['kernel_fn'](data(x),state)
        assert result==('candidate',x) and calls[-1]==(x,c,mode)
assert calls[0][0] is not calls[1][0]  # Fresh x is passed through every timed call.
for x,overrides in ((X(32),{}),(X(128,stride=(8193,1)),{}),(X(alignment=2),{}),
                    (X(dtype=object()),{}),(X(),dict(layer='pooler')),
                    (X(),dict(n=128)),(X(),dict(k=4096)),(X(),dict(m=256))):
    assert ns['kernel_fn'](data(x,**overrides),state)==('baseline',x)
x=X()
assert ns['kernel_fn'](data(x),{'rcb_selected':{}})==('baseline',x)
print('PASS: exact cp18 prefix and inlined AST; fresh-x dispatch; STORE=False membership; '
      'independent M128/256; shape/stride/alignment/dtype guards and fallback')
print('candidate SHA256',hashlib.sha256(source).hexdigest())
