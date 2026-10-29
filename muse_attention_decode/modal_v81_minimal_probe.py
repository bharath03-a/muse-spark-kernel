"""Cached-only conservative v75 validation, paired timing and fresh-input S1K stress."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
image=(modal.Image.from_registry('lmsysorg/sglang:dev-muse-spark-open')
 .run_commands('echo /usr/local/lib/python3.12/dist-packages > /opt/sglang/lib/python3.12/site-packages/zz_system.pth','echo /usr/lib/python3/dist-packages >> /opt/sglang/lib/python3.12/site-packages/zz_system.pth').entrypoint([]))
for local,remote in [('candidate_openkernels_v75_f16pv.py','current.py'),('candidate_openkernels_v75.py','conservative.py'),('candidate_openkernels_v81_minimal.py','research.py'),('reference.py','reference.py'),('checks.py','checks.py')]:
 image=image.add_local_file(ROOT/'muse_attention_decode'/local,'/root/'+remote)
image=image.add_local_dir(ROOT/'research_harness','/root/research_harness',ignore=['results','__pycache__'])
cache=modal.Volume.from_name('muse-ext-cache',create_if_missing=True)
app=modal.App('muse-attention-v81-minimal',image=image)
@app.function(gpu='H200',timeout=240,volumes={'/cache':cache})
def probe():
 import os,sys,importlib.util,json
 os.environ.update(TORCH_EXTENSIONS_DIR='/cache/torch_ext',TRITON_CACHE_DIR='/cache/triton',TVM_FFI_CACHE_DIR='/cache/tvm-ffi',SGLANG_JIT_CACHE_DIR='/cache/sglang_jit',TORCH_CUDA_ARCH_LIST='9.0a')
 sys.path.insert(0,'/root')
 import torch,reference
 from research_harness.gpu import compare_graphs
 modules={}
 for name,extname in [('current','attn_v75_8'),('conservative','attn_v75_8')]+[('research','attn_v75_8')]:
  libs=list(Path('/cache/torch_ext').rglob(extname+'.so'))
  if not libs:raise RuntimeError('ABORT_NO_PREBUILT '+extname)
  spec=importlib.util.spec_from_file_location(name,'/root/'+name+'.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  lib=max(libs,key=lambda p:p.stat().st_mtime)
  if m.CUDA not in (lib.parent/'cuda.cu').read_text():raise RuntimeError('ABORT_SOURCE_MISMATCH '+extname)
  spec=importlib.util.spec_from_file_location(extname,str(lib));ext=importlib.util.module_from_spec(spec);spec.loader.exec_module(ext);m._mod=ext;modules[name]=m
 print('V75_PREBUILT',list(modules),flush=True)
 class Audit:
  def __init__(self,ext,name):self.ext=ext;self.name=name;self.calls=0;self.flags=None
  def __getattr__(self,name):return getattr(self.ext,name)
  def run(self,*args):self.calls+=1;self.flags=args[-3];return self.ext.run(*args)
 for name,m in modules.items():m._mod=Audit(m._mod,name)
 tests=[]
 for bs in [2,16,17,32]:
  for ctx,dist,scale,spread in [(80000,'soft',1.,'fixed'),(80000,'coherent',1.,'agentic'),(5000,'edges',1/64,'mixed'),(10,'soft',1/64,'mixed')]:tests.append(dict(batch=bs,ctx=ctx,heads='tp8',layer='sliding',q_len=8,page=1,dist=dist,kv_scale=scale,spread=spread))
 for seed in [1471,8293,19391]:
  for cfg in tests:
   data=reference.generate_input(seed=seed,**cfg);outs={}
   for name,m in modules.items():
    calls=m._mod.calls;out=m.kernel_fn(data);torch.cuda.synchronize();assert torch.isfinite(out).all().item()
    assert m._mod.calls==calls+1
    flags=m._mod.flags;expect=name=='research' and cfg['batch'] in (16,32)
    assert bool(flags&536870912)==expect,(name,cfg,flags)
    result=reference.check_implementation(data,reference.collect(data,out));assert result[0],(name,cfg,result)
    print('V81_CHECK',json.dumps(dict(seed=seed,variant=name,flags=flags,fp8_pv=expect,config=cfg,passed=result[0],message=result[1])),flush=True);outs[name]=out.clone()
   a=outs['research'].float();b=outs['current'].float();print('V81_DIFF',json.dumps(dict(seed=seed,config=cfg,rel_l2=((a-b).norm()/b.norm().clamp_min(1e-20)).item())),flush=True)
   del data,outs,out,a,b
 for bs,length,layer in [(16,8,'sliding'),(32,8,'sliding'),(16,1,'sliding'),(32,1,'sliding'),(16,8,'full'),(32,8,'full')]:
  data=reference.generate_input(batch=bs,ctx=80000,heads='tp8',layer=layer,q_len=length,page=1,seed=2291+bs+length,dist='coherent' if layer=='full' else 'soft',spread='agentic')
  for name,m in modules.items():
   if name=='current':continue
   report=compare_graphs(lambda:modules['current'].kernel_fn(data),lambda:m.kernel_fn(data),repetitions=40)
   print('V81_PAIRED',json.dumps(dict(batch=bs,q_len=length,layer=layer,variant=name,**report)),flush=True)
  del data
 gen=torch.Generator(device='cuda').manual_seed(93991)
 for bs,scale in [(16,1.),(32,1/64)]:
  data=reference.generate_input(batch=bs,ctx=5000,heads='tp8',layer='sliding',q_len=8,page=1,seed=5501+bs,kv_scale=scale,spread='mixed',dist='coherent')
  kf=torch.empty_like(data['k_cache'],dtype=torch.float32);vf=torch.empty_like(kf);graphs={};outs={}
  for name,m in modules.items():
   m.kernel_fn(data);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):outs[name]=m.kernel_fn(data)
   graphs[name]=g
  for rep in range(200):
   data['qkv'].normal_(generator=gen);kf.normal_(generator=gen);kf.mul_(torch.rsqrt(kf.square().mean(-1,keepdim=True)+1e-5));vf.normal_(generator=gen)
   data['k_cache'].copy_((kf/scale).to(torch.float8_e4m3fn));data['v_cache'].copy_((vf/scale).to(torch.float8_e4m3fn))
   for name,g in graphs.items():
    g.replay();assert torch.isfinite(outs[name]).all().item(),(name,bs,rep)
    if rep%25==0 or rep==199:
     result=reference.check_implementation(data,reference.collect(data,outs[name]));assert result[0],(name,bs,rep,result)
  assert torch.isfinite(data['k_cache'].float()).all().item() and torch.isfinite(data['v_cache'].float()).all().item()
  print('V81_STRESS',json.dumps(dict(batch=bs,replays=200,kv_scale=scale,fresh_qkv_and_cache=True,passed=True)),flush=True)
  del data,kf,vf,graphs,outs
 print('V81_DONE',flush=True)
