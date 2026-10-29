"""CPU-only compile proof for the isolated pf32 strict W4 backend."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
image=(modal.Image.from_registry('lmsysorg/sglang:dev-muse-spark-open')
 .run_commands('echo /usr/local/lib/python3.12/dist-packages > /opt/sglang/lib/python3.12/site-packages/zz_system.pth','echo /usr/lib/python3/dist-packages >> /opt/sglang/lib/python3.12/site-packages/zz_system.pth')
 .entrypoint([]).add_local_file(ROOT/'candidate_prefill_pf32_strict.py','/root/pf32.py'))
app=modal.App('muse-prefill-pf32-strict-cpu',image=image)
cache=modal.Volume.from_name('muse-ext-cache',create_if_missing=True)
@app.function(timeout=900,cpu=8,memory=16384,volumes={'/cache':cache})
def compile_backend():
 import os,ast,time,json,subprocess
 from pathlib import Path
 os.environ.update(TORCH_EXTENSIONS_DIR='/cache/torch_ext',TORCH_CUDA_ARCH_LIST='9.0a',MAX_JOBS='2')
 from torch.utils.cpp_extension import load_inline
 tree=ast.parse(Path('/root/pf32.py').read_text())
 c=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(a,ast.Name) and a.id=='W4G_CUDA' for a in n.targets))
 gen=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_w4g_wgm')
 scope={};exec(compile(ast.fix_missing_locations(ast.Module(body=[gen],type_ignores=[])),'<generator>','exec'),scope)
 c=c.replace('@@WGM@@',scope['_w4g_wgm']());assert '@@WGM@@' not in c
 fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_w4g')
 call=next(n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='load_inline')
 args={k.arg:ast.literal_eval(k.value) for k in call.keywords if k.arg in ('cpp_sources','functions')}
 start=time.monotonic()
 try:
  mod=load_inline('pf2_w4g_pf32_strict_v1h',cuda_sources=c,extra_cuda_cflags=['-O3','--use_fast_math','-lineinfo','-gencode=arch=compute_90a,code=sm_90a','-DHAS_WG_HOST=1'],verbose=False,**args)
  print('PF32_CPU_OK',time.monotonic()-start,mod.__file__,flush=True)
  print(subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',mod.__file__],text=True),flush=True)
 finally:cache.commit()
