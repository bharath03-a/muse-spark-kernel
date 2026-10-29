"""Cached-only conservative v75 validation, paired timing and fresh-input S1K stress."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
image=(modal.Image.from_registry('lmsysorg/sglang:dev-muse-spark-open')
 .run_commands('echo /usr/local/lib/python3.12/dist-packages > /opt/sglang/lib/python3.12/site-packages/zz_system.pth','echo /usr/lib/python3/dist-packages >> /opt/sglang/lib/python3.12/site-packages/zz_system.pth').entrypoint([]))
for local,remote in [('candidate_openkernels_attention.py','current.py'),('candidate_openkernels_v75_f16pv.py','conservative.py'),('candidate_openkernels_v75.py','research.py'),('reference.py','reference.py'),('checks.py','checks.py')]:
 image=image.add_local_file(ROOT/'muse_attention_decode'/local,'/root/'+remote)
image=image.add_local_dir(ROOT/'research_harness','/root/research_harness',ignore=['results','__pycache__'])
cache=modal.Volume.from_name('muse-ext-cache',create_if_missing=True)
app=modal.App('muse-attention-v75-precision',image=image)
@app.function(gpu='H200',timeout=180,volumes={'/cache':cache})
def validate():
 import os,sys,importlib.util,json
 os.environ.update(TORCH_EXTENSIONS_DIR='/cache/torch_ext',TRITON_CACHE_DIR='/cache/triton',TVM_FFI_CACHE_DIR='/cache/tvm-ffi',SGLANG_JIT_CACHE_DIR='/cache/sglang_jit',TORCH_CUDA_ARCH_LIST='9.0a')
 sys.path.insert(0,'/root')
 import torch,reference
 from research_harness.gpu import compare_graphs
 modules={}
 for name,extname in [('current','attn_v60_8'),('conservative','attn_v75_8')]+[('research','attn_v75_8')]:
  libs=list(Path('/cache/torch_ext').rglob(extname+'.so'))
  if not libs:raise RuntimeError('ABORT_NO_PREBUILT '+extname)
  spec=importlib.util.spec_from_file_location(name,'/root/'+name+'.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  lib=max(libs,key=lambda p:p.stat().st_mtime)
  if m.CUDA not in (lib.parent/'cuda.cu').read_text():raise RuntimeError('ABORT_SOURCE_MISMATCH '+extname)
  spec=importlib.util.spec_from_file_location(extname,str(lib));ext=importlib.util.module_from_spec(spec);spec.loader.exec_module(ext);m._mod=ext;modules[name]=m
 class RunAudit:
  def __init__(self,ext,name):self.ext=ext;self.name=name;self.calls=0
  def __getattr__(self,name):return getattr(self.ext,name)
  def run(self,*args):
   flags=args[-3];var=args[-6]
   print('V75_NATIVE_DISPATCH',json.dumps(dict(variant=self.name,flags=flags,var=var,fp8_pv=bool(flags & 536870912))),flush=True)
   if self.name=='research':assert flags & 536870912 and var==-2, ('wrong_native_dispatch',flags,var)
   self.calls+=1
   return self.ext.run(*args)
 for name,m in modules.items():m._mod=RunAudit(m._mod,name)
 print('V75_PREBUILT',list(modules),flush=True)
 for seed in [1471,8293,19391]:
  for bs in [1,16,32]:
   for dist,scale,spread in [('soft',1.,'fixed'),('coherent',1.,'agentic'),('edges',1/64,'mixed')]:
    data=reference.generate_input(seed=seed,batch=bs,ctx=80000,heads='tp8',layer='full',q_len=8,page=1,dist=dist,kv_scale=scale,spread=spread)
    outs={}
    for name,m in modules.items():
     calls_before=m._mod.calls
     out=m.kernel_fn(data);torch.cuda.synchronize();assert torch.isfinite(out).all().item()
     assert m._mod.calls==calls_before+1, 'native_run_not_called'
     result=reference.check_implementation(data,reference.collect(data,out))
     print('V75_TARGET_CHECK',json.dumps(dict(seed=seed,batch=bs,dist=dist,kv_scale=scale,spread=spread,variant=name,passed=result[0],message=result[1])),flush=True)
     assert result[0],result
     outs[name]=out.clone()
    a=outs['research'].float();b=outs['conservative'].float();err=(a-b).norm()/b.norm().clamp_min(1e-20)
    rowerr=(a-b).reshape(-1,64).norm(dim=-1)/b.reshape(-1,64).norm(dim=-1).clamp_min(1e-20)
    print('V75_TARGET_DIFF',json.dumps(dict(seed=seed,batch=bs,dist=dist,kv_scale=scale,fp8_vs_f16_l2=err.item(),worst_head_l2=rowerr.max().item(),dtype=str(outs['research'].dtype))),flush=True)
    del data,outs,out,a,b,rowerr
 print('V75_TARGET_DONE',flush=True)
