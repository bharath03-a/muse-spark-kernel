"""Cached-only safe BF16-PV validation and paired timing; launch only with GPU allocation."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
image=(modal.Image.from_registry('lmsysorg/sglang:dev-muse-spark-open')
 .run_commands('echo /usr/local/lib/python3.12/dist-packages > /opt/sglang/lib/python3.12/site-packages/zz_system.pth','echo /usr/lib/python3/dist-packages >> /opt/sglang/lib/python3.12/site-packages/zz_system.pth').entrypoint([]))
for local,remote in [('candidate_openkernels_attention.py','current.py'),('candidate_openkernels_pvcontrol.py','control.py'),('candidate_openkernels_bf16pv.py','bf16pv.py'),('reference.py','reference.py'),('checks.py','checks.py')]:
 image=image.add_local_file(ROOT/'muse_attention_decode'/local,'/root/'+remote)
image=image.add_local_dir(ROOT/'research_harness','/root/research_harness',ignore=['results','__pycache__'])
cache=modal.Volume.from_name('muse-ext-cache',create_if_missing=True)
app=modal.App('muse-attention-bf16pv-probe',image=image)
@app.function(gpu='H200',timeout=300,volumes={'/cache':cache})
def probe():
 import os,sys,importlib.util,json,time
 os.environ.update(TORCH_EXTENSIONS_DIR='/cache/torch_ext',TRITON_CACHE_DIR='/cache/triton',TVM_FFI_CACHE_DIR='/cache/tvm-ffi',TORCH_CUDA_ARCH_LIST='9.0a',SGLANG_JIT_CACHE_DIR='/cache/sglang_jit')
 sys.path.insert(0,'/root')
 import torch,reference
 from research_harness.gpu import compare_graphs
 modules={}
 for name,extname in [('current','attn_v60_8'),('control','attn_v60_pvcontrol_8'),('bf16pv','attn_v60_bf16pv_8')]:
  libs=list(Path('/cache/torch_ext').rglob(extname+'.so'))
  if not libs:raise RuntimeError('ABORT_NO_PREBUILT '+extname)
  spec=importlib.util.spec_from_file_location(name,'/root/'+name+'.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  lib=max(libs,key=lambda p:p.stat().st_mtime)
  cuda_path=lib.parent/'cuda.cu'
  if not cuda_path.exists() or m.CUDA not in cuda_path.read_text():raise RuntimeError('ABORT_CACHE_SOURCE_MISMATCH '+extname)
  spec=importlib.util.spec_from_file_location(extname,str(lib));ext=importlib.util.module_from_spec(spec);spec.loader.exec_module(ext);m._mod=ext
  modules[name]=m
 print('PREBUILT_LOADED',list(modules),flush=True)
 cases=[('flat',dict(batch=2,ctx=6000,heads='tp8',layer='full',page=1,q_len=8)),('coherent',dict(batch=2,ctx=20000,heads='tp8',layer='full',page=1,q_len=8,dist='coherent')),('scaled',dict(batch=2,ctx=9000,heads='tp8',layer='full',page=1,q_len=8,kv_scale=1/64)),('edges',dict(batch=4,ctx=6000,heads='tp8',layer='full',page=1,q_len=8,dist='edges')),('mixed',dict(batch=5,ctx=5000,heads='tp8',layer='sliding',page=1,q_len=8,spread='mixed'))]
 for seed in [1471,8293,19391]:
  for label,cfg in cases:
   data=reference.generate_input(seed=seed,**cfg)
   if label=='flat':data['qkv'][...,:1024].zero_()
   for name,m in modules.items():
    # Force the new attnp backend even on small correctness batches.
    m.VOVR=(-2,256,8,1<<30,32836 if cfg['layer']=='sliding' else 131396,1)
    out=m.kernel_fn(data);torch.cuda.synchronize()
    assert torch.isfinite(out).all().item(), ('NONFINITE',name,seed,label)
    result=reference.check_implementation(data,reference.collect(data,out))
    print('BF16PV_CHECK',json.dumps(dict(seed=seed,case=label,variant=name,passed=result[0],message=result[1])),flush=True)
    if not result[0]:raise RuntimeError(result)
   del data,out
 for bs in [16,32]:
  for layer in ['full','sliding']:
   data=reference.generate_input(batch=bs,ctx=80000,heads='tp8',layer=layer,seed=2197+bs,q_len=8,page=1,dist='coherent' if layer=='full' else 'soft',spread='agentic')
   # Actual default dispatch: full CFM, sliding early query loads.
   for m in modules.values():m.VOVR=None
   for baseline in ['current','control']:
    report=compare_graphs(lambda:modules[baseline].kernel_fn(data),lambda:modules['bf16pv'].kernel_fn(data),repetitions=40)
    print('BF16PV_PAIRED',json.dumps(dict(batch=bs,layer=layer,baseline=baseline,candidate='bf16pv',**report)),flush=True)
   del data
 print('BF16PV_DONE',flush=True)
