"""CPU-only same-source SASS comparison of W4G parameter storage."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
image=(modal.Image.from_registry('lmsysorg/sglang:dev-muse-spark-open')
 .run_commands('echo /usr/local/lib/python3.12/dist-packages > /opt/sglang/lib/python3.12/site-packages/zz_system.pth','echo /usr/lib/python3/dist-packages >> /opt/sglang/lib/python3.12/site-packages/zz_system.pth')
 .entrypoint([])
 .add_local_file(ROOT/'candidate_prefill_w13_mixed.py','/root/base.py')
 .add_local_file(ROOT/'candidate_prefill_grid_constant.py','/root/grid.py')
 .add_local_file(ROOT/'nvfp4.py','/root/nvfp4.py')
 .add_local_dir(ROOT.parent/'research_harness','/root/research_harness',ignore=['results/**','__pycache__/**']))
app=modal.App('muse-prefill-grid-constant-cpu',image=image)
cache=modal.Volume.from_name('muse-ext-cache',create_if_missing=True)
@app.function(timeout=900,cpu=8,memory=16384,volumes={'/cache':cache})
def compare():
 import os,ast,re,json,subprocess,time
 from pathlib import Path
 os.environ.update(TORCH_EXTENSIONS_DIR='/cache/torch_ext',TORCH_CUDA_ARCH_LIST='9.0a',MAX_JOBS='2')
 from torch.utils.cpp_extension import load_inline
 cpp='void w4g(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t); int64_t w4g_smem(); int64_t w4g_err(int64_t);'
 for label,name in [('base','prefill_w4g_paramaudit_base'),('grid','prefill_w4g_paramaudit_grid')]:
  start=time.monotonic();tree=ast.parse(Path('/root/'+label+'.py').read_text())
  c=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(a,ast.Name) and a.id=='W4G_CUDA' for a in n.targets))
  # Reproduce the module's later pure-Python instruction generator; the raw
  # literal deliberately contains @@WGM@@ until module initialization completes.
  generator=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_w4g_wgm')
  scope={}
  exec(compile(ast.fix_missing_locations(ast.Module(body=[generator],type_ignores=[])),'<wgmma-generator>','exec'),scope)
  c=c.replace('@@WGM@@',scope['_w4g_wgm']())
  assert '@@WGM@@' not in c
  m=load_inline(name,cpp_sources=cpp,cuda_sources=c,functions=['w4g','w4g_smem','w4g_err'],extra_cuda_cflags=['-O3','--use_fast_math','-lineinfo','-gencode=arch=compute_90a,code=sm_90a','-DHAS_WG_HOST=1'],verbose=False)
  sass=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',m.__file__],text=True)
  resources=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',m.__file__],text=True)
  Path('/cache/'+label+'_w4g.sass').write_text(sass)
  Path('/cache/'+label+'_w4g.resources').write_text(resources)
  print('RESOURCES',label,resources,flush=True)
  for chunk in sass.split('Function : '):
   if 'w4g_k' not in chunk.split('\n')[0]:continue
   header=chunk.split('\n')[0]
   print('SASS',label,json.dumps({'kernel':header,'LDL':len(re.findall(r'\bLDL\b',chunk)),'STL':len(re.findall(r'\bSTL\b',chunk)),'LDC':len(re.findall(r'\bLDC\b',chunk)),'lines':len(chunk.splitlines())}),flush=True)
  print('CPU_SECONDS',label,time.monotonic()-start,flush=True);cache.commit()


@app.function(timeout=240,gpu='H200',volumes={'/cache':cache})
def probe():
 import os,sys,ast,importlib.util,time,json
 os.environ.update(TORCH_EXTENSIONS_DIR='/cache/torch_ext',TORCH_CUDA_ARCH_LIST='9.0a',MAX_JOBS='2')
 sys.path.insert(0,'/root')
 import torch,nvfp4
 from research_harness.gpu import compare_graphs
 from torch.utils.cpp_extension import load_inline
 from pathlib import Path
 cpp='void w4g(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t); int64_t w4g_smem(); int64_t w4g_err(int64_t);'
 ext={}
 for label,name in [('base','prefill_w4g_paramaudit_base'),('grid','prefill_w4g_paramaudit_grid')]:
  tree=ast.parse(Path('/root/'+label+'.py').read_text())
  c=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(a,ast.Name) and a.id=='W4G_CUDA' for a in n.targets))
  generator=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_w4g_wgm')
  scope={};exec(compile(ast.fix_missing_locations(ast.Module(body=[generator],type_ignores=[])),'<generator>','exec'),scope)
  c=c.replace('@@WGM@@',scope['_w4g_wgm']())
  ext[label]=load_inline(name,cpp_sources=cpp,cuda_sources=c,functions=['w4g','w4g_smem','w4g_err'],extra_cuda_cflags=['-O3','--use_fast_math','-lineinfo','-gencode=arch=compute_90a,code=sm_90a','-DHAS_WG_HOST=1'],verbose=False)
 spec=importlib.util.spec_from_file_location('base_backend','/root/base.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 started=time.monotonic()
 # Eighteen bounded cases: all modes, fresh seeds, nearby and scored row counts.
 cases=[(1001,False,0),(1001,True,2),(3008,False,1),(3008,True,0),(16385,False,2),(16385,True,1)]
 for seed in (8181,8182,8183):
  gen=torch.Generator(device='cuda').manual_seed(seed)
  raw,sf,gs=nvfp4.random_weights(8192,4096,generator=gen,device='cuda',leading=(32,))
  gs=gs.expand(32).contiguous()*torch.exp(torch.randn(32,generator=gen,device='cuda'))
  wp,gg=mod._w4pack(raw,sf,gs);del raw,sf,gs
  for rows,skew,mode in cases:
   popularity=[64,32,16,8,4,2,1,1]+[0]*24 if skew else [1]*32
   counts=[rows*p//sum(popularity) for p in popularity];counts[0]+=rows-sum(counts)
   offs=torch.tensor(counts,device='cuda',dtype=torch.int32).cumsum(0).to(torch.int32)
   x=torch.randn(rows,4096,generator=gen,device='cuda').to(torch.bfloat16)
   ctr0=torch.zeros(24,device='cuda',dtype=torch.int32);ctr1=torch.zeros_like(ctr0);empty=torch.empty(0,device='cuda',dtype=torch.int64)
   def run(label,ctr):
    out=torch.empty(rows,8192,device='cuda',dtype=torch.bfloat16)
    ext[label].w4g(x,offs,wp,gg,out,ctr,empty,1,3,mode,0)
    return out
   a,b=run('base',ctr0),run('grid',ctr1);torch.cuda.synchronize()
   assert torch.isfinite(a).all().item() and torch.isfinite(b).all().item()
   assert torch.equal(a,b),'Kernel arithmetic changed'
   assert ext['base'].w4g_err(1)==0 and ext['grid'].w4g_err(1)==0
   print('GRID_CHECK',seed,rows,skew,mode,'finite_bitidentical',flush=True)
   report=compare_graphs(lambda:run('base',ctr0),lambda:run('grid',ctr1),repetitions=24)
   print('GRID_PAIRED',json.dumps(dict(seed=seed,rows=rows,skew=skew,mode=mode,**report)),flush=True)
   del a,b,x,offs,ctr0,ctr1
  del wp,gg
 print('GRID_GPU_SECONDS',time.monotonic()-started,flush=True)
