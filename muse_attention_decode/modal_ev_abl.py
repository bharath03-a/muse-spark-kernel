from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
image=(modal.Image.from_registry('lmsysorg/sglang:dev-muse-spark-open')
 .run_commands('echo /usr/local/lib/python3.12/dist-packages > /opt/sglang/lib/python3.12/site-packages/zz_system.pth','echo /usr/lib/python3/dist-packages >> /opt/sglang/lib/python3.12/site-packages/zz_system.pth').entrypoint([]))
for local,remote in [('candidate_keel.py','base.py'),('reference.py','reference.py'),('checks.py','checks.py')]:
 image=image.add_local_file(ROOT/'muse_attention_decode'/local,'/root/'+remote)
cache=modal.Volume.from_name('muse-ext-cache',create_if_missing=True)
app=modal.App('muse-attn-abl',image=image)
@app.function(gpu='H200',timeout=1500,volumes={'/cache':cache})
def probe():
 import os,sys,importlib.util,time,threading
 os.environ.update(TORCH_EXTENSIONS_DIR='/cache/torch_ext',TRITON_CACHE_DIR='/cache/triton',TORCH_CUDA_ARCH_LIST='9.0a')
 sys.path.insert(0,'/root')
 import torch,reference
 spec=importlib.util.spec_from_file_location('base','/root/base.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m._zh_join()
 flags=["-O3","--use_fast_math","-gencode=arch=compute_90a,code=sm_90a","-DOKC_WST_ACC32=1","-DZH_SCH=1"]
 abls=[0,2,4,64,1,71,70]
 mods={}
 def build(a):
  src=m.ZH_WST_CUDA.replace("ABL = 0,","ABL = %d,"%a,1)
  assert src!=m.ZH_WST_CUDA or a==0
  mods[a]=m._zh_load_inline(name="okc_zh_wst_abl%d_"%a+m._zh_tag(src,*flags),cpp_sources=[m.ZH_WST_CPP],cuda_sources=[src],functions=["wst_attn_bal"],extra_cuda_cflags=flags,verbose=False)
 ths=[threading.Thread(target=build,args=(a,)) for a in abls]
 [t.start() for t in ths];[t.join() for t in ths]
 print('BUILT',sorted(mods),flush=True)
 def timeit(f):
  f();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g): f()
  fl=torch.empty(64<<20,dtype=torch.int32,device='cuda');ts=[]
  for _ in range(40):
   fl.add_(1);a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
   a.record();g.replay();b.record();b.synchronize();ts.append(a.elapsed_time(b)*1000)
  return sorted(ts)[20]
 for bs in (16,24):
  data=reference.generate_input(batch=bs,ctx=80000,heads='tp8',layer='full',q_len=8,page=1,seed=77+bs,dist='coherent',spread='agentic')
  res={}
  for rnd in range(2):
   for a in abls:
    m._ZH['wst']=mods[a]
    res.setdefault(a,[]).append(timeit(lambda:m.kernel_fn(data)))
  for a in abls: print('ABL',bs,a,['%.1f'%x for x in res[a]],flush=True)
  del data
@app.local_entrypoint()
def main(): probe.remote()
