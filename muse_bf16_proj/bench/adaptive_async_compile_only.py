"""Offline SM90 compiler audit. CPU only: no tensors or CUDA allocations."""
import json
import re
import subprocess
import tempfile
from pathlib import Path
import time
import traceback
import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import get_ptxas
from triton.experimental.gluon._runtime import GluonASTSource
from adaptive_record_async_kernel import async_mm, DECODE_ASM, DECODE_CONSTRAINTS

print(json.dumps({'torch':torch.__version__,'triton':triton.__version__,
                  'audit':'CPU-only offline compiler; no GPU timing'}),flush=True)
for label,stage_x in [('weight_ring',False),('weight_x_ring',True)]:
    fn=async_mm
    names=list(fn.arg_names)
    consts=dict(N=4096,K=8192,M=32,SCALE=2.0**-4,DEPTH=3,ASM=DECODE_ASM,CONSTRAINTS=DECODE_CONSTRAINTS,STAGE_X=stage_x)
    sig={nm:('*u32' if nm in ('payload','metadata') else '*bf16') if nm in ('payload','metadata','x','out') else 'constexpr' for nm in names}
    attrs={(names.index(nm),):[['tt.divisibility',16]] for nm in ('payload','metadata','x','out')}
    src=GluonASTSource(fn=fn,signature=sig,constexprs={(names.index(nm),):v for nm,v in consts.items()},attrs=attrs)
    started=time.monotonic()
    try:
        kernel=triton.compile(src,target=GPUTarget('cuda',90,32),options={'num_warps':4})
        ptx=kernel.asm['ptx']
        print(json.dumps({'variant':label,'compile_pass':True,'seconds':time.monotonic()-started,
                          'shared':kernel.metadata.shared,'mma_sync':ptx.count('mma.sync'),
                          'cp_async_static':ptx.count('cp.async.'),'cta_barriers_static':ptx.count('bar.sync'),
                          'shared_store_instructions_static':ptx.count('st.shared')}),flush=True)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'kernel.ptx';path.write_text(ptx)
            result=subprocess.run([get_ptxas(90).path,'-arch=sm_90a','-v',str(path),'-o',str(Path(td)/'kernel.cubin')],capture_output=True,text=True)
            print(label,'PTXAS',result.returncode,result.stdout,result.stderr,flush=True)
        try:
            from triton.tools.disasm import get_sass
            print(label,'SASS_BEGIN\n'+get_sass(kernel.asm['cubin'])+'\nSASS_END',flush=True)
        except Exception as exc:
            print(label,'SASS_UNAVAILABLE',str(exc),flush=True)
        print(label,'PTX_BEGIN\n'+ptx+'\nPTX_END',flush=True)
    except Exception:
        print(label,'COMPILE_FAILED\n'+traceback.format_exc(),flush=True)
