"""Pinned SM90 offline compiler audit; no CUDA tensors/allocations/launches."""
import json,subprocess,tempfile,time,traceback
from pathlib import Path
import triton
from triton.experimental.gluon._runtime import GluonASTSource
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import get_ptxas
from router_cluster_bt64 import router_cluster_bt64,layouts

print(json.dumps(dict(triton=triton.__version__,audit='CPU-only offline SM90 compiler; GPU validation deferred')),flush=True)
for c,m in ((4,256),(8,128)):
    for mode in (False,True):
        label=f'C{c}_M{m}_'+('ASYNC' if mode else 'STORE')
        try:
            fn=router_cluster_bt64
            names=list(fn.arg_names)
            constants=dict(M=m,N=256,K=8192,C=c,DEPTH=3,ASYNC=mode,**layouts(c))
            signature={name:('*fp32' if name=='out' else '*bf16') if name in ('hi','lo','x','out') else 'constexpr' for name in names}
            attrs={(names.index(name),):[['tt.divisibility',16]] for name in ('hi','lo','x','out')}
            source=GluonASTSource(fn=fn,signature=signature,constexprs={(names.index(name),):value for name,value in constants.items()},attrs=attrs)
            started=time.monotonic()
            compiled=triton.compile(source,target=GPUTarget('cuda',90,32),options={'num_warps':4,'num_ctas':c})
            ptx=compiled.asm['ptx']
            print(json.dumps(dict(label=label,compile_pass=True,seconds=time.monotonic()-started,
                shared=compiled.metadata.shared,mma_sync=ptx.count('mma.sync'),cp_async=ptx.count('cp.async.'),
                st_async=ptx.count('st.async'),cta_barriers=ptx.count('bar.sync'),
                cluster_barriers=ptx.count('barrier.cluster'),shuffle=ptx.count('shfl'))),flush=True)
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'kernel.ptx';path.write_text(ptx)
                result=subprocess.run([get_ptxas(90).path,'-arch=sm_90a','-v',str(path),'-o',str(Path(folder)/'kernel.cubin')],capture_output=True,text=True)
                print(label,'PTXAS',result.returncode,result.stdout,result.stderr,flush=True)
            print(label,'PTX_BEGIN\n'+ptx+'\nPTX_END',flush=True)
        except Exception:
            print(label,'COMPILE_FAILED\n'+traceback.format_exc(),flush=True)
