"""One bounded cached-only H200 ablation, no compilation or scored submissions."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
image=(modal.Image.from_registry('lmsysorg/sglang:dev-muse-spark-open')
 .run_commands('echo /usr/local/lib/python3.12/dist-packages > /opt/sglang/lib/python3.12/site-packages/zz_system.pth','echo /usr/lib/python3/dist-packages >> /opt/sglang/lib/python3.12/site-packages/zz_system.pth').entrypoint([])
 .add_local_file(ROOT/'muse_attention_decode/candidate_openkernels_attention.py','/root/attention.py')
 .add_local_file(ROOT/'muse_attention_decode/reference.py','/root/reference.py')
 .add_local_file(ROOT/'muse_attention_decode/checks.py','/root/checks.py'))
cache=modal.Volume.from_name('muse-ext-cache',create_if_missing=True)
app=modal.App('muse-attention-cached-ablation',image=image)
@app.function(gpu='H200',timeout=180,volumes={'/cache':cache})
def run():
 import os,sys,importlib.util,time,statistics
 os.environ.update(TORCH_EXTENSIONS_DIR='/cache/torch_ext',TRITON_CACHE_DIR='/cache/triton',TVM_FFI_CACHE_DIR='/cache/tvm-ffi')
 sys.path.insert(0,'/root')
 import torch,reference,attention
 from pathlib import Path
 libs=list(Path('/cache/torch_ext').rglob('attn_v60_8.so'))
 if not libs:
  print('ABORT_NO_PREBUILT_EXTENSION',flush=True);return
 lib=max(libs,key=lambda p:p.stat().st_mtime)
 print('CACHED_LIB',str(lib),flush=True)
 spec=importlib.util.spec_from_file_location('attn_v60_8',str(lib));ext=importlib.util.module_from_spec(spec);spec.loader.exec_module(ext);attention._mod=ext
 variants={'CFM':131396,'BAL':324,'PCV':452,'ACV':1348}
 tests=[dict(batch=2,ctx=9000,heads='tp8',layer='full',kv_scale=1/64,q_len=8,page=1),dict(batch=4,ctx=6000,heads='tp8',layer='full',q_len=8,page=1,dist='coherent'),dict(batch=5,ctx=5000,heads='tp8',layer='sliding',q_len=8,page=1,spread='mixed')]
 for i,cfg in enumerate(tests):
  data=reference.generate_input(seed=1442+i,**cfg)
  for name,x in variants.items():
   attention.VOVR=(-2,256,8,1<<30,x,1)
   out=attention.kernel_fn(data);torch.cuda.synchronize()
   result=reference.check_implementation(data,reference.collect(data,out))
   print('CHECK',name,cfg,result,flush=True)
   if not result[0]:return
  del data,out
 flush=torch.empty(256*1024*1024,device='cuda',dtype=torch.uint8)
 for bs in [16,32]:
  data=reference.generate_input(batch=bs,ctx=80000,heads='tp8',layer='full',seed=8843+bs,q_len=8,page=1,dist='coherent',spread='agentic')
  graphs={}
  for name,x in variants.items():
   attention.VOVR=(-2,256,8,1<<30,x,1)
   attention.kernel_fn(data);torch.cuda.synchronize()
   g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):attention.kernel_fn(data)
   graphs[name]=g
  times={name:[] for name in graphs}
  for rep in range(20):
   names=list(graphs)
   if rep%2:names.reverse()
   for name in names:
    flush.zero_();a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True)
    a.record();graphs[name].replay();z.record();z.synchronize();times[name].append(a.elapsed_time(z)*1000)
  print('ABLATION_US',bs,{name:{'median':statistics.median(vals),'min':min(vals),'max':max(vals)} for name,vals in times.items()},flush=True)
 attention.VOVR=None
