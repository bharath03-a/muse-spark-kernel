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
app=modal.App('muse-attention-v75-probe',image=image)
@app.function(gpu='H200',timeout=300,volumes={'/cache':cache})
def probe(include_fp8:bool=False):
 import os,sys,importlib.util,json
 os.environ.update(TORCH_EXTENSIONS_DIR='/cache/torch_ext',TRITON_CACHE_DIR='/cache/triton',TVM_FFI_CACHE_DIR='/cache/tvm-ffi',SGLANG_JIT_CACHE_DIR='/cache/sglang_jit',TORCH_CUDA_ARCH_LIST='9.0a')
 sys.path.insert(0,'/root')
 import torch,reference
 from research_harness.gpu import compare_graphs
 modules={}
 for name,extname in [('current','attn_v60_8'),('conservative','attn_v75_8')]+([('research','attn_v75_8')] if include_fp8 else []):
  libs=list(Path('/cache/torch_ext').rglob(extname+'.so'))
  if not libs:raise RuntimeError('ABORT_NO_PREBUILT '+extname)
  spec=importlib.util.spec_from_file_location(name,'/root/'+name+'.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  lib=max(libs,key=lambda p:p.stat().st_mtime)
  if m.CUDA not in (lib.parent/'cuda.cu').read_text():raise RuntimeError('ABORT_SOURCE_MISMATCH '+extname)
  spec=importlib.util.spec_from_file_location(extname,str(lib));ext=importlib.util.module_from_spec(spec);spec.loader.exec_module(ext);m._mod=ext;modules[name]=m
 print('V75_PREBUILT',list(modules),flush=True)
 tests=[dict(batch=1,ctx=10,heads='tp8',layer='sliding'),dict(batch=3,ctx=2047,heads='tp8',layer='sliding'),dict(batch=5,ctx=5000,heads='tp8',layer='sliding',q_len=8,spread='mixed',page=1),dict(batch=4,ctx=3000,heads='tp8',layer='full'),dict(batch=8,ctx=8192,heads='dp',layer='full',spread='mixed'),dict(batch=2,ctx=100000,heads='tp8',layer='full',q_len=8,page=1),dict(batch=4,ctx=6000,heads='tp8',layer='sliding',dist='edges',page=1),dict(batch=4,ctx=6000,heads='tp8',layer='full',q_len=8,dist='edges'),dict(batch=1,ctx=1048576,heads='tp8',layer='full',q_len=8),dict(batch=3,ctx=4000,heads='tp8',layer='sliding',q_len=8,page=1,kv_scale=1/64),dict(batch=2,ctx=9000,heads='tp8',layer='full',kv_scale=1/64),dict(batch=4,ctx=80000,heads='tp8',layer='full',spread='mixed',dist='soft'),dict(batch=4,ctx=6000,heads='tp8',layer='sliding',q_len=8,dist='soft',page=1),dict(batch=2,ctx=20000,heads='dp',layer='full',q_len=8,dist='soft'),dict(batch=4,ctx=80000,heads='tp8',layer='full',spread='mixed',dist='coherent',page=1),dict(batch=2,ctx=20000,heads='dp',layer='full',q_len=8,dist='coherent'),dict(batch=1,ctx=2047,heads='tp8',layer='sliding',q_len=8,page=1,kv_scale=1/64),dict(batch=1,ctx=5000,heads='tp8',layer='sliding',q_len=8,page=1,spread='mixed'),dict(batch=1,ctx=80000,heads='tp8',layer='sliding',page=1,spread='mixed')]
 for seed in [1471,8293,19391]:
  for i,cfg in enumerate(tests):
   data=reference.generate_input(seed=seed,**cfg)
   compared_outputs={}
   for name,m in modules.items():
    out=m.kernel_fn(data);torch.cuda.synchronize();assert torch.isfinite(out).all().item()
    compared_outputs[name]=out.clone()
    result=reference.check_implementation(data,reference.collect(data,out))
    print('V75_CHECK',json.dumps(dict(seed=seed,index=i,variant=name,passed=result[0],message=result[1])),flush=True)
    assert result[0],result
   if include_fp8:
    a=compared_outputs['research'].float();b=compared_outputs['conservative'].float();err=(a-b).norm()/b.norm().clamp_min(1e-20)
    print('V75_DTYPE_COMPARE',json.dumps(dict(seed=seed,index=i,fp8_vs_f16_l2=err.item(),fp8_output_dtype=str(compared_outputs['research'].dtype),f16_output_dtype=str(compared_outputs['conservative'].dtype))),flush=True)
   del data,out,compared_outputs
 for bs in [1,16,32]:
  for length in [1,8]:
   for layer in ['sliding','full']:
    data=reference.generate_input(batch=bs,ctx=80000,heads='tp8',layer=layer,seed=2197+bs+length,q_len=length,page=1,dist='coherent' if layer=='full' else 'soft',spread='agentic' if bs>1 else 'fixed')
    for name,m in modules.items():
     if name=='current':continue
     report=compare_graphs(lambda:modules['current'].kernel_fn(data),lambda:m.kernel_fn(data),repetitions=40)
     print('V75_PAIRED',json.dumps(dict(batch=bs,q_len=length,layer=layer,variant=name,**report)),flush=True)
    del data
 # Stress both S1K lengths, changing all Q/K/V/gate inputs every replay.
 gen=torch.Generator(device='cuda').manual_seed(93991)
 stress_cases=[(1,1,80000,1.,'fixed','sliding'),(1,8,5000,1/64,'mixed','sliding')]
 if include_fp8:stress_cases += [(1,8,80000,1.,'fixed','full'),(16,8,80000,1/64,'agentic','full')]
 for bs,length,ctx,scale,spread,layer in stress_cases:
  data=reference.generate_input(batch=bs,ctx=ctx,heads='tp8',layer=layer,seed=8881+length,q_len=length,page=1,kv_scale=scale,spread=spread,dist='coherent' if layer=='full' else 'soft')
  kf=torch.empty_like(data['k_cache'],dtype=torch.float32);vf=torch.empty_like(kf)
  graphs={};outs={}
  for name,m in modules.items():
   m.kernel_fn(data);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):outs[name]=m.kernel_fn(data)
   graphs[name]=g
  for rep in range(200):
   data['qkv'].normal_(generator=gen);kf.normal_(generator=gen);kf.mul_(torch.rsqrt(kf.square().mean(-1,keepdim=True)+1e-5));vf.normal_(generator=gen)
   data['k_cache'].copy_((kf/scale).to(torch.float8_e4m3fn));data['v_cache'].copy_((vf/scale).to(torch.float8_e4m3fn))
   for name,g in graphs.items():
    g.replay();assert torch.isfinite(outs[name]).all().item(),(name,length,rep)
    if rep%25==0 or rep==199:
     result=reference.check_implementation(data,reference.collect(data,outs[name]));assert result[0],(name,length,rep,result)
  print('V75_STRESS',json.dumps(dict(batch=bs,q_len=length,layer=layer,replays=200,variants=list(modules),fresh_qkv_and_cache=True,passed=True)),flush=True)
  del data,kf,vf,graphs,outs
 print('V75_DONE',flush=True)
