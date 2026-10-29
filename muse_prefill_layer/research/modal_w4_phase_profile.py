"""Bounded CPU build and H200 phase diagnostic for current W4G; not scoring."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
image=(modal.Image.from_registry('lmsysorg/sglang:dev-muse-spark-open')
 .run_commands('echo /usr/local/lib/python3.12/dist-packages > /opt/sglang/lib/python3.12/site-packages/zz_system.pth','echo /usr/lib/python3/dist-packages >> /opt/sglang/lib/python3.12/site-packages/zz_system.pth')
 .entrypoint([]).add_local_file(ROOT/'candidate_prefill_w13_mixed.py','/root/base.py')
 .add_local_file(ROOT/'nvfp4.py','/root/nvfp4.py'))
app=modal.App('muse-prefill-w4-phase-profile',image=image)
cache=modal.Volume.from_name('muse-ext-cache',create_if_missing=True)

def _build(profile):
 import ast,os,time
 from pathlib import Path
 os.environ.update(TORCH_EXTENSIONS_DIR='/cache/torch_ext',TORCH_CUDA_ARCH_LIST='9.0a',MAX_JOBS='2')
 from torch.utils.cpp_extension import load_inline
 tree=ast.parse(Path('/root/base.py').read_text())
 c=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(a,ast.Name) and a.id=='W4G_CUDA' for a in n.targets))
 generator=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_w4g_wgm')
 scope={};exec(compile(ast.fix_missing_locations(ast.Module(body=[generator],type_ignores=[])),'<generator>','exec'),scope)
 c=c.replace('@@WGM@@',scope['_w4g_wgm']());assert '@@WGM@@' not in c
 cpp='void w4g(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t); int64_t w4g_smem(); int64_t w4g_err(int64_t);'
 flags=['-O3','--use_fast_math','-lineinfo','-gencode=arch=compute_90a,code=sm_90a','-DHAS_WG_HOST=1']+(['-DW4PROF'] if profile else [])
 name='prefill_w4g_phaseprof_h' if profile else 'prefill_w4g_paramaudit_base'
 return load_inline(name,cpp_sources=cpp,cuda_sources=c,functions=['w4g','w4g_smem','w4g_err'],extra_cuda_cflags=flags,verbose=False)

@app.function(timeout=900,cpu=8,memory=16384,volumes={'/cache':cache})
def compile_profile():
 import time
 start=time.monotonic()
 try:
  _build(True);print('PROFILE_COMPILE_SECONDS',time.monotonic()-start,flush=True)
 finally:cache.commit()

@app.function(timeout=240,gpu='H200',volumes={'/cache':cache})
def phase_profile():
 import os,sys,importlib.util,time,json
 os.environ.update(TORCH_EXTENSIONS_DIR='/cache/torch_ext',TRITON_CACHE_DIR='/cache/triton',TVM_FFI_CACHE_DIR='/cache/tvm-ffi',TORCH_CUDA_ARCH_LIST='9.0a',MAX_JOBS='2')
 sys.path.insert(0,'/root')
 import torch,nvfp4
 spec=importlib.util.spec_from_file_location('base','/root/base.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
 prof,plain=_build(True),_build(False)
 start=time.monotonic();gen=torch.Generator(device='cuda').manual_seed(8181)
 raw,sf,gs=nvfp4.random_weights(8192,4096,generator=gen,device='cuda',leading=(32,))
 packed,global_scale=base._w4pack(raw,sf,gs);del raw,sf,gs
 flush=torch.empty(256*1024*1024,dtype=torch.uint8,device='cuda');sink=torch.empty((),dtype=torch.int64,device='cuda')
 for rows in (753,3012,16384):
  x=torch.randn((rows,4096),device='cuda',dtype=torch.bfloat16,generator=gen)
  for skew in (False,True):
   ratios=torch.ones(32,dtype=torch.float64)
   if skew:ratios[:4]=16
   counts=(ratios/ratios.sum()*rows).floor().to(torch.int32);counts[0]+=rows-int(counts.sum())
   offs=counts.cumsum(0).to(device='cuda',dtype=torch.int32)
   for mode in (0,1,2):
    ctr=torch.zeros(8,dtype=torch.int32,device='cuda');pb=torch.zeros(48,dtype=torch.int64,device='cuda');empty=torch.empty(0,dtype=torch.int64,device='cuda')
    want=torch.empty((rows,8192),device='cuda',dtype=torch.bfloat16);got=torch.empty_like(want)
    plain.w4g(x,offs,packed,global_scale,want,ctr,empty,1,3,mode,0)
    sink.copy_(flush.sum());prof.w4g(x,offs,packed,global_scale,got,ctr,pb,1,3,mode,0)
    torch.cuda.synchronize();err=prof.w4g_err(1)
    good=bool(torch.isfinite(got).all()) and torch.equal(got,want)
    print('CHECK',rows,skew,mode,good,'watchdog',err,flush=True)
    assert good and err==0
    v=pb.tolist();records=[]
    for label,o in [('big',0),('n128-leftover',16),('small',32)]:
     if not v[o+5]:continue
     cs=max(1,v[o+4]);ps=max(1,v[o+12]);span=v[o+5]
     records.append(dict(kind=label,consumer_cycles_per_stage={'wait':v[o]/cs,'dq':v[o+1]/cs,'mma_issue_retire':v[o+2]/cs},consumer_span_fractions={'wait':v[o]/span,'dq':v[o+1]/span,'mma_issue_retire':v[o+2]/span,'epilogue':v[o+3]/span},producer_cycles_per_stage={'wait_empty':v[o+8]/ps,'issue':v[o+9]/ps},warp_stages=v[o+4],producer_stages=v[o+12]))
    print('PHASE',json.dumps({'rows':rows,'skew':skew,'mode':mode,'records':records}),flush=True)
    del want,got,ctr,pb,offs
    offs=counts.cumsum(0).to(device='cuda',dtype=torch.int32)
  del x
 print('PROFILE_GPU_SECONDS',time.monotonic()-start,flush=True)
