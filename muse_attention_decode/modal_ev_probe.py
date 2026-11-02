from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
image=(modal.Image.from_registry('lmsysorg/sglang:dev-muse-spark-open')
 .run_commands('echo /usr/local/lib/python3.12/dist-packages > /opt/sglang/lib/python3.12/site-packages/zz_system.pth','echo /usr/lib/python3/dist-packages >> /opt/sglang/lib/python3.12/site-packages/zz_system.pth').entrypoint([]))
for local,remote in [('candidate_keel.py','base.py'),('candidate_ev_v1.py','cand.py'),('reference.py','reference.py'),('checks.py','checks.py')]:
 image=image.add_local_file(ROOT/'muse_attention_decode'/local,'/root/'+remote)
image=image.add_local_dir(ROOT/'research_harness','/root/research_harness',ignore=['results','__pycache__'])
cache=modal.Volume.from_name('muse-ext-cache',create_if_missing=True)
app=modal.App('muse-attn-ev',image=image)
@app.function(gpu='H200',timeout=1000,volumes={'/cache':cache})
def probe():
 import os,sys,importlib.util,json,time
 t0=time.time()
 os.environ.update(TORCH_EXTENSIONS_DIR='/cache/torch_ext',TRITON_CACHE_DIR='/cache/triton',TORCH_CUDA_ARCH_LIST='9.0a')
 sys.path.insert(0,'/root')
 import torch,reference
 from research_harness.gpu import compare_graphs
 mods={}
 for n in ['base','cand']:
  spec=importlib.util.spec_from_file_location(n,'/root/%s.py'%n);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);mods[n]=m
 print('LOADED',time.time()-t0,flush=True)
 for bs,nss in [(32,[2,4])]:
  data=reference.generate_input(batch=bs,ctx=80000,heads='tp8',layer='sliding',q_len=8,page=1,seed=77+bs,dist='soft',spread='agentic')
  for ns in nss:
   mods['cand'].EV_NS={bs:ns}
   for n,m in mods.items():
    out=m.kernel_fn(data);torch.cuda.synchronize()
    r=reference.check_implementation(data,reference.collect(data,out));print('CHECK',n,bs,ns,r[0],flush=True)
   for draw in range(2):
    rep=compare_graphs(lambda:mods['base'].kernel_fn(data),lambda:mods['cand'].kernel_fn(data),repetitions=40)
    print('PAIRED',bs,ns,draw,json.dumps(rep)[:400],flush=True)
  del data
 print('SECONDS',time.time()-t0)
@app.local_entrypoint()
def main(): probe.remote()
