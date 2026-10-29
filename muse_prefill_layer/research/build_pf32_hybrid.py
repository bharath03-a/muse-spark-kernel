"""Keep scored grid-P backend below65536 allocated rows; pf32 vector stores above.

PF32 C++ source is unchanged from the CPU-validated strict candidate. Reuse both
existing packed banks and disable PDL to isolate the measured large-path gain.
"""
import ast,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'candidate_prefill_grid_constant.py').read_text()
tree=ast.parse((ROOT/'candidate_prefill_pf32_strict.py').read_text())
c=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(a,ast.Name) and a.id=='W4G_CUDA' for a in n.targets))
gen=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_w4g_wgm')
scope={};exec(compile(ast.fix_missing_locations(ast.Module(body=[gen],type_ignores=[])),'<generator>','exec'),scope)
c=c.replace('@@WGM@@',scope['_w4g_wgm']());assert '@@WGM@@' not in c
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_w4g')
call=next(n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='load_inline')
args={k.arg:ast.literal_eval(k.value) for k in call.keywords if k.arg in ('cpp_sources','functions')}
# Keep complete original attribution and append latest backend attribution.
s+='\n# Large-path backend adapted from lee101/openkernels47103fd9c1212868018eb615a4c6467e366e9ce0.\n# Copyright2026LeePenkman, Apache-2.0; see OPENKERNELS_LICENSE.txt.\n'
s+='_PF32_HYBRID_CUDA = '+repr(c)+'\n'
s+='_PF32_HYBRID_CPP = '+repr(args['cpp_sources'])+'\n'
s+='_PF32_HYBRID_EXPORTS = '+repr(args['functions'])+'\n'
s+='''
_PF32_HYBRID_EXT = None
_PF32_HYBRID_MIN_CAPACITY = 65536
_PF32_HYBRID_ORIGINAL_PREPARE = prepare
_PF32_HYBRID_ORIGINAL_GEMM = _w4gemm

def prepare(weights):
    global _PF32_HYBRID_EXT
    import time
    started = time.monotonic()
    try:
        # Identical to the already CPU-validated pf32 strict extension; no new
        # weight banks or source compilation families are introduced.
        _PF32_HYBRID_EXT = load_inline(
            'pf2_w4g_pf32_strict_v1h', cpp_sources=_PF32_HYBRID_CPP,
            cuda_sources=_PF32_HYBRID_CUDA, functions=_PF32_HYBRID_EXPORTS,
            extra_cuda_cflags=['-O3','--use_fast_math','-lineinfo',
                '-gencode=arch=compute_90a,code=sm_90a','-DHAS_WG_HOST=1'],
            verbose=False)
    except Exception as exc:
        _PF32_HYBRID_EXT = None
        print(f'pf32 hybrid disabled; scored grid-P fallback retained: {exc}',flush=True)
    state = _PF32_HYBRID_ORIGINAL_PREPARE(weights)
    if _PF32_HYBRID_EXT is not None and _PF32_HYBRID_EXT.w4g_err(1):
        _PF32_HYBRID_EXT = None
        print('pf32 hybrid disabled after prepare watchdog; grid-P retained',flush=True)
    print(f'pf32 hybrid full prepare seconds {time.monotonic()-started:.1f}',flush=True)
    return state

def _w4gemm(A, offs, Wp, g, ctr, ksplit=1, cls=3, prof=None, ext=None, wide=0, fuse=False):
    # Preserve exact old diagnostics and the scored small-shape dispatch. The
    # selected backend depends only on capacity/config, never routing values.
    if (A.shape[0] < _PF32_HYBRID_MIN_CAPACITY or _PF32_HYBRID_EXT is None
            or prof is not None or ext is not None):
        return _PF32_HYBRID_ORIGINAL_GEMM(A,offs,Wp,g,ctr,ksplit,cls,prof,ext,wide,fuse)
    if ksplit != 1:
        raise ValueError('strict W2 complete reduction required')
    out = torch.empty((A.shape[0], Wp.shape[1]*(128 if fuse else 256)),
                      dtype=torch.bfloat16,device=A.device)
    profile = torch.empty(0,dtype=torch.int64,device=A.device)
    empty = torch.empty(0,dtype=torch.float32,device=A.device)
    # PDL explicitly disabled. This graph still captures W13 -> W2 together.
    _PF32_HYBRID_EXT.w4g(A,offs,Wp,g,out,ctr,profile,1,cls,wide,int(fuse),empty,empty,0)
    return out
'''
out=ROOT/'candidate_prefill_pf32_hybrid.py';out.write_text(s);compile(s,str(out),'exec')
# Source-level proof that CPU-compiled PF32 kernel is reused without mutation.
hashcode=hashlib.sha256(c.encode()).hexdigest()
print(out);print('UNCHANGED_PF32_CUDA_SHA256',hashcode)
