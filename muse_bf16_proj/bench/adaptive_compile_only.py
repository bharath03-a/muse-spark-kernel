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
from adaptive_record_kernel import adaptive_mm, VECTOR_ASM, VECTOR_CONSTRAINTS
from adaptive_record_split_kernel import split_mm

print(json.dumps({'torch':torch.__version__,'triton':triton.__version__,
                  'audit':'CPU-only offline compiler; no GPU timing'}),flush=True)
for label,fn in [('direct',adaptive_mm),('intra_cta_split',split_mm)]:
    names=list(fn.arg_names)
    consts=dict(N=4096,K=8192,M=32,SCALE=2.0**-4,ASM=VECTOR_ASM,CONSTRAINTS=VECTOR_CONSTRAINTS)
    sig={nm:('*u32' if nm in ('payload','metadata') else '*bf16') if nm in ('payload','metadata','x','out') else 'constexpr' for nm in names}
    attrs={(names.index(nm),):[['tt.divisibility',16]] for nm in ('payload','metadata','x','out')}
    src=GluonASTSource(fn=fn,signature=sig,constexprs={(names.index(nm),):v for nm,v in consts.items()},attrs=attrs)
    started=time.monotonic()
    try:
        kernel=triton.compile(src,target=GPUTarget('cuda',90,32),options={'num_warps':4})
        ptx=kernel.asm['ptx']
        print(json.dumps({'variant':label,'compile_pass':True,'seconds':time.monotonic()-started,
                          'shared':kernel.metadata.shared,'mma_sync':ptx.count('mma.sync'),
                          'vector_loads_static':ptx.count('ld.global.v4.u32'),
                          'shared_store_instructions_static':ptx.count('st.shared')}),flush=True)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'kernel.ptx';path.write_text(ptx)
            result=subprocess.run([get_ptxas(90).path,'-arch=sm_90a','-v',str(path),'-o',str(Path(td)/'kernel.cubin')],capture_output=True,text=True)
            print(label,'PTXAS',result.returncode,result.stdout,result.stderr,flush=True)
        print(label,'PTX_BEGIN\n'+ptx+'\nPTX_END',flush=True)
    except Exception:
        print(label,'COMPILE_FAILED\n'+traceback.format_exc(),flush=True)
