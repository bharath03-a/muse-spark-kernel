"""CPU-only SASS comparison of exact grid-P vs lossless eight-op W4G."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
image=(modal.Image.from_registry('lmsysorg/sglang:dev-muse-spark-open')
 .run_commands('echo /usr/local/lib/python3.12/dist-packages > /opt/sglang/lib/python3.12/site-packages/zz_system.pth','echo /usr/lib/python3/dist-packages >> /opt/sglang/lib/python3.12/site-packages/zz_system.pth')
 .entrypoint([])
 .add_local_file(ROOT/'candidate_prefill_grid_constant.py','/root/base.py')
 .add_local_file(ROOT/'candidate_prefill_unpack8.py','/root/grid.py')
 .add_local_file(ROOT/'nvfp4.py','/root/nvfp4.py')
 .add_local_dir(ROOT.parent/'research_harness','/root/research_harness',ignore=['results/**','__pycache__/**']))
app=modal.App('muse-prefill-unpack8-cpu',image=image)
cache=modal.Volume.from_name('muse-ext-cache',create_if_missing=True)
@app.function(timeout=900,cpu=8,memory=16384,volumes={'/cache':cache})
def compare():
 import os,ast,re,json,subprocess,time
 from pathlib import Path
 os.environ.update(TORCH_EXTENSIONS_DIR='/cache/torch_ext',TORCH_CUDA_ARCH_LIST='9.0a',MAX_JOBS='2')
 from torch.utils.cpp_extension import load_inline
 cpp='void w4g(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t); int64_t w4g_smem(); int64_t w4g_err(int64_t);'
 for label,name in [('base','prefill_w4g_paramaudit_grid'),('grid','pf2_w4g_unpack8_v1h')]:
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
  Path('/cache/unpack8_'+label+'_w4g.sass').write_text(sass)
  Path('/cache/unpack8_'+label+'_w4g.resources').write_text(resources)
  print('RESOURCES',label,resources,flush=True)
  for chunk in sass.split('Function : '):
   if 'w4g_k' not in chunk.split('\n')[0]:continue
   header=chunk.split('\n')[0]
   print('SASS',label,json.dumps({'kernel':header,'LDL':len(re.findall(r'\bLDL\b',chunk)),'STL':len(re.findall(r'\bSTL\b',chunk)),'LDC':len(re.findall(r'\bLDC\b',chunk)),'lines':len(chunk.splitlines())}),flush=True)
  print('CPU_SECONDS',label,time.monotonic()-start,flush=True);cache.commit()


