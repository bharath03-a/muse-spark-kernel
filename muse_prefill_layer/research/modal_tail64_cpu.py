"""CPU-only n64 tail resource proof. Does not launch or allocate a GPU."""
from pathlib import Path
import modal
ROOT = Path(__file__).resolve().parents[1]
image = (modal.Image.from_registry('lmsysorg/sglang:dev-muse-spark-open')
 .run_commands('echo /usr/local/lib/python3.12/dist-packages > /opt/sglang/lib/python3.12/site-packages/zz_system.pth',
               'echo /usr/lib/python3/dist-packages >> /opt/sglang/lib/python3.12/site-packages/zz_system.pth')
 .entrypoint([]).add_local_file(ROOT/'research/kernel_sources/w4tail64.cu','/root/w4tail64.cu')
 .add_local_file(ROOT/'candidate_prefill_grid_constant.py','/root/old.py')
 .add_local_file(ROOT/'nvfp4.py','/root/nvfp4.py')
 .add_local_dir(ROOT.parent/'research_harness','/root/research_harness',ignore=['results/**','__pycache__/**']))
app = modal.App('muse-prefill-tail64-resource-proof', image=image)
cache = modal.Volume.from_name('muse-ext-cache', create_if_missing=True)
@app.function(timeout=900, cpu=8, memory=16384, volumes={'/cache':cache})
def compile_resources():
 import os, re, subprocess, time, hashlib
 from pathlib import Path
 os.environ.update(TORCH_EXTENSIONS_DIR='/cache/torch_ext', TORCH_CUDA_ARCH_LIST='9.0a', MAX_JOBS='2')
 from torch.utils.cpp_extension import load_inline
 source = Path('/root/w4tail64.cu').read_text()
 print('SOURCE_SHA256', hashlib.sha256(source.encode()).hexdigest(), flush=True)
 start = time.monotonic()
 module = load_inline('prefill_tail64_resource_v2', cpp_sources='void w4tail64(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t); int64_t tail64_smem(); int64_t tail64_occupancy(); void w4filtered(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t); void w4g(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t); int64_t w4g_err(int64_t);',
  cuda_sources=source, functions=['w4tail64','tail64_smem','tail64_occupancy','w4filtered','w4g','w4g_err'],
  extra_cuda_cflags=['-O3','--use_fast_math','-lineinfo','-gencode=arch=compute_90a,code=sm_90a','-DHAS_WG_HOST=1','--ptxas-options=-v'], verbose=True)
 print('DYNAMIC_SMEM',module.tail64_smem(),flush=True)
 resource = subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',module.__file__],text=True)
 sass = subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',module.__file__],text=True)
 print('RESOURCE_USAGE',resource,flush=True)
 for chunk in sass.split('Function : '):
  name = chunk.split('\n')[0]
  if 'w4g_k' not in name: continue
  print('SASS_COUNTS', name, 'LDL',len(re.findall(r'\bLDL\b',chunk)), 'STL',len(re.findall(r'\bSTL\b',chunk)),flush=True)
 Path('/cache/prefill_tail64_v2.sass').write_text(sass)
 Path('/cache/prefill_tail64_v2.resources').write_text(resource)
 cache.commit()
 print('CPU_SECONDS',time.monotonic()-start,flush=True)

@app.function(timeout=240, gpu='H200', volumes={'/cache':cache})
def probe():
 import os,sys,time,json,importlib.util
 from pathlib import Path
 os.environ.update(TORCH_EXTENSIONS_DIR='/cache/torch_ext',TORCH_CUDA_ARCH_LIST='9.0a',TRITON_CACHE_DIR='/cache/triton',TVM_FFI_CACHE_DIR='/cache/tvm-ffi',MAX_JOBS='2')
 sys.path.insert(0,'/root')
 import torch,nvfp4
 from torch.utils.cpp_extension import load_inline
 from research_harness.gpu import compare_graphs
 cpp='void w4tail64(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t); int64_t tail64_smem(); int64_t tail64_occupancy(); void w4filtered(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t); void w4g(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t); int64_t w4g_err(int64_t);'
 e=load_inline('prefill_tail64_resource_v2',cpp_sources=cpp,cuda_sources=Path('/root/w4tail64.cu').read_text(),functions=['w4tail64','tail64_smem','tail64_occupancy','w4filtered','w4g','w4g_err'],extra_cuda_cflags=['-O3','--use_fast_math','-lineinfo','-gencode=arch=compute_90a,code=sm_90a','-DHAS_WG_HOST=1','--ptxas-options=-v'],verbose=False)
 occupancy=e.tail64_occupancy();assert occupancy>=2,occupancy
 print('TAIL64_OCCUPANCY',occupancy,flush=True)
 spec=importlib.util.spec_from_file_location('old','/root/old.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 started=time.monotonic();empty=torch.empty(0,device='cuda',dtype=torch.int64)
 perm=torch.stack((torch.arange(4096,device='cuda').view(-1,8),4096+torch.arange(4096,device='cuda').view(-1,8)),1).reshape(-1)
 for seed in (9391,9392,9393):
  gen=torch.Generator(device='cuda').manual_seed(seed)
  w,s,g=nvfp4.random_weights(8192,4096,generator=gen,device='cuda',leading=(32,))
  g=g.expand(32).contiguous()*torch.exp(torch.randn(32,generator=gen,device='cuda'))
  # Paired W13 and ordinary W2 use independent physical output extents.
  wp,gp=old._w4pack(w.index_select(1,perm),s.index_select(1,perm),g)
  w2,g2=old._w4pack(w[:,:4096],s[:,:4096],g)
  ref13=nvfp4.dequantize(w[0],s[0],g[0]);ref2=ref13[:4096]
  del w,s,g
  # Exact boundaries plus realistic profiled small-T routing. Zero and negative
  # activation controls are included without negative expert counts.
  cases=[([32,33,47,64,65]+[0]*27,'mixed'),
         ([160,161,175,192,193]+[0]*27,'negative'),
         ([33,47,64]+[0]*29,'zero'),
         ([400,314,241,174,105,57,19,15,9,9,5,2,1,1,1]+[0]*17,'profile')]
  for counts,kind in cases:
   actual=sum(counts);capacity=actual+17
   offs=torch.tensor(counts,device='cuda',dtype=torch.int32).cumsum(0).to(torch.int32)
   a=torch.randn(capacity,4096,generator=gen,device='cuda').to(torch.bfloat16)
   if kind=='negative':a=-a.abs()
   if kind=='zero':a.zero_()
   selected=torch.zeros(actual,device='cuda',dtype=torch.bool);start=0
   for count in counts:
    rem=count%128
    if 33<=rem<=64:selected[start+count-rem:start+count]=True
    start+=count
   for fuse,weights,scales,ref in [(False,w2,g2,ref2),(True,wp,gp,ref13)]:
    width=4096
    ctr0=torch.zeros(12,device='cuda',dtype=torch.int32);ctr1=torch.zeros_like(ctr0)
    baseline_out=torch.full((capacity,width),-7,device='cuda',dtype=torch.bfloat16)
    candidate_out=torch.full_like(baseline_out,-7)
    tail_out=torch.full_like(baseline_out,-7)
    def base():
     e.w4g(a,offs,weights,scales,baseline_out,ctr0,empty,1,3,0,int(fuse));return baseline_out[:actual]
    def new():
     e.w4filtered(a,offs,weights,scales,candidate_out,ctr1,empty,int(fuse));return candidate_out[:actual]
    e.w4tail64(a,offs,weights,scales,tail_out,torch.zeros_like(ctr0),empty,int(fuse))
    b,c=base(),new();torch.cuda.synchronize()
    assert torch.isfinite(b).all().item() and torch.isfinite(c).all().item()
    assert torch.equal(b,c),'filtered complete result differs'
    assert torch.equal(tail_out[:actual][selected],b[selected]),'tail output differs'
    assert (tail_out[:actual][~selected]==-7).all().item() and (tail_out[actual:]==-7).all().item()
    assert (candidate_out[actual:]==-7).all().item()
    h=(a[:min(8,counts[0])].float()@ref.T).to(torch.bfloat16)
    if fuse:
     h=(torch.nn.functional.silu(h[:,:4096].float())*h[:,4096:].float()).to(torch.bfloat16)
    err=float((c[:len(h)].float()-h.float()).norm()/h.float().norm().clamp_min(1e-20));assert err<.015,err
    assert e.w4g_err(1)==0
    print('TAIL64_CHECK',json.dumps(dict(seed=seed,kind=kind,fuse=fuse,reference_l2=err,finite=True,bitidentical=True,sentinel=True)),flush=True)
    result=compare_graphs(base,new,repetitions=20)
    print('TAIL64_PAIRED',json.dumps(dict(seed=seed,kind=kind,fuse=fuse,**result)),flush=True)
   if kind=='profile':
    i0=torch.empty(capacity,4096,device='cuda',dtype=torch.bfloat16);i1=torch.empty_like(i0)
    o0=torch.empty_like(i0);o1=torch.empty_like(i0)
    c0=torch.zeros(24,device='cuda',dtype=torch.int32);c1=torch.zeros_like(c0)
    def seq_base():
     e.w4g(a,offs,wp,gp,i0,c0[:12],empty,1,3,0,1)
     e.w4g(i0,offs,w2,g2,o0,c0[12:],empty,1,3,0,0)
     return o0[:actual]
    def seq_new():
     e.w4filtered(a,offs,wp,gp,i1,c1[:12],empty,1)
     e.w4filtered(i1,offs,w2,g2,o1,c1[12:],empty,0)
     return o1[:actual]
    bb,cc=seq_base(),seq_new();torch.cuda.synchronize()
    assert torch.isfinite(bb).all().item() and torch.equal(bb,cc)
    result=compare_graphs(seq_base,seq_new,repetitions=24)
    print('TAIL64_SEQUENCE_PAIRED',json.dumps(dict(seed=seed,finite=True,bitidentical=True,**result)),flush=True)
   del a,offs
  del wp,gp,w2,g2,ref13,ref2
 print('TAIL64_GPU_SECONDS',time.monotonic()-started,flush=True)
