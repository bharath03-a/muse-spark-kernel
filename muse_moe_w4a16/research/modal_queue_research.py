"""Bounded CPU compilation/SASS audit. GPU scheduling remains with coordinator."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[2] if modal.is_local() else Path("/root")
image=(modal.Image.from_registry('lmsysorg/sglang:dev-muse-spark-open')
 .run_commands('echo /usr/local/lib/python3.12/dist-packages > /opt/sglang/lib/python3.12/site-packages/zz_system.pth','echo /usr/lib/python3/dist-packages >> /opt/sglang/lib/python3.12/site-packages/zz_system.pth')
 .entrypoint([]))
if modal.is_local():
 image=(image.add_local_dir(ROOT/'muse_moe_w4a16','/root/moe',ignore=['profiles/**','__pycache__/**'])
  .add_local_dir(ROOT/'research_harness','/root/research_harness',ignore=['results/**','__pycache__/**']))
app=modal.App('muse-queue-atomic-research',image=image)
cache=modal.Volume.from_name('muse-ext-cache',create_if_missing=True)

@app.function(timeout=600,cpu=8,memory=16384,volumes={'/cache':cache})
def compile_and_inspect():
 import os,sys,importlib,time,subprocess,re
 os.environ.update(TORCH_EXTENSIONS_DIR='/cache/torch_ext',TRITON_CACHE_DIR='/cache/triton',TVM_FFI_CACHE_DIR='/cache/tvm-ffi',TORCH_CUDA_ARCH_LIST='9.0a',MAX_JOBS='2')
 sys.path.insert(0,'/root/moe')
 start=time.monotonic()
 for module_name in ['candidate_moe_full_strict_gk','research_moe_queue_atomic']:
  mod=importlib.import_module(module_name)
  mod._big=lambda:True
  for label,builder in [('v17',lambda:mod._extS(0)),('ring',lambda:mod._extL(0))]:
   ext=builder();cache.commit()
   print('COMPILE_OK',module_name,label,round(time.monotonic()-start,2),ext.__file__,flush=True)
   result=subprocess.run(['cuobjdump','--dump-sass',ext.__file__],capture_output=True,text=True,check=True)
   sass=result.stdout
   # Full files persist for instruction-level follow-up; aggregate counts are diagnostic only.
   dest=Path('/cache/moe_queue_audit');dest.mkdir(exist_ok=True)
   (dest/(module_name+'_'+label+'.sass')).write_text(sass)
   for pattern in ['ATOM','RED','SHFL','MATCH','VOTE','LDL','STL']:
    print('SASS_COUNT',module_name,label,pattern,len(re.findall(r'\b'+pattern+r'(?:\.|\b)',sass)),flush=True)
 cache.commit()

@app.function(timeout=420,gpu='H200',volumes={'/cache':cache})
def compare_queue(candidate_names: str = 'research_moe_queue_atomic,research_moe_heavy_first', cases_override: str = '', baseline_name: str = 'candidate_moe_full_strict_gk', prepare_name: str = '', vary_global_scales: bool = False):
 import os,sys,importlib,json,time,hashlib
 sys.path[:0]=['/root/moe','/root']
 from research_harness.environment import configure_h200_cache
 configure_h200_cache('/cache')
 import torch,reference
 from research_harness.gpu import compare_graphs
 baseline=importlib.import_module(baseline_name)
 candidates={name:importlib.import_module(name) for name in candidate_names.split(',')}
 start=time.monotonic()
 for name,mod in [('baseline',baseline)]+list(candidates.items()):
  print('SOURCE_SHA256',name,hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest(),flush=True)
  mod._extS(0);mod._extL(0)
  print('CACHE_LOADED',name,time.monotonic()-start,flush=True)
 # Prepare immutable weights once and clone all mutable state for each implementation.
 weights=reference.generate_weights(5252)
 if vary_global_scales:
  weights['w13_gs']=weights['w13_gs']*torch.linspace(.7,1.3,weights['w13_gs'].numel(),device='cuda')
  weights['w2_gs']=weights['w2_gs']*torch.linspace(1.4,.6,weights['w2_gs'].numel(),device='cuda')
  print('VARIED_GLOBAL_SCALES','w13:.7..1.3,w2:1.4...6 per expert',flush=True)
 prepare_mod=importlib.import_module(prepare_name) if prepare_name else baseline
 state_a=prepare_mod.prepare(weights)
 immutable={'w1','w2','g1','g2','nw','w1x','w2x','w2f'}
 states={name:{k:(v.clone() if torch.is_tensor(v) and k not in immutable else v) for k,v in state_a.items()} for name in candidates}
 print('PREPARE_SECONDS',time.monotonic()-start,flush=True)
 results=[]
 cases=[(m,seed,dist) for m in [32,128,256] for seed,dist in [(7,'normal'),(19,'normal'),(41,'skewed')]]
 cases += [(m,71,'skewed') for m in [17,33,63,65,127,257]]
 cases += [(64,7,'normal'),(64,41,'skewed')]
 if 'wx' in candidate_names:
  cases += [(96,113,'normal'),(192,113,'skewed'),(384,113,'skewed')]
 if cases_override:
  cases=[(int(m),int(seed),dist) for m,seed,dist in (row.split(':') for row in cases_override.split(','))]
 for m,seed,dist in cases:
  data=reference.generate_input(m,'ep8',seed,dist=dist)
  gold=reference.ref_kernel(data,weights)
  for name,mod,st in [('baseline',baseline,state_a)]+[(n,mod,states[n]) for n,mod in candidates.items()]:
   output=mod.kernel_fn(data,st)
   torch.cuda.synchronize()
   assert torch.isfinite(output).all().item(),(m,seed,dist,name,'nonfinite')
   ok,message=reference.compare(output,gold,reference.REL_L2_TOL)
   assert ok,(m,seed,dist,name,message)
   print('CORRECT',m,seed,dist,name,message,flush=True)
  for name,mod in candidates.items():
   report=compare_graphs(lambda:baseline.kernel_fn(data,state_a),lambda mod=mod,name=name:mod.kernel_fn(data,states[name]),repetitions=24)
   report.update(m=m,seed=seed,dist=dist,variant=name)
   results.append(report)
   print('PAIRED',json.dumps(report),flush=True)
 report_all={'cases':len(cases),'comparisons':len(results),'results':results}
 dest=Path('/cache/moe_probe_reports');dest.mkdir(exist_ok=True)
 report_path=dest/(str(time.time_ns())+'.json');report_path.write_text(json.dumps(report_all));cache.commit()
 print('PERSISTED_REPORT',str(report_path),flush=True)
 print('EXPERIMENT_DONE',json.dumps(report_all),flush=True)

@app.function(timeout=600,cpu=8,memory=16384,volumes={'/cache':cache})
def compile_heavy():
 import os,sys,importlib,time,subprocess
 os.environ.update(TORCH_EXTENSIONS_DIR='/cache/torch_ext',TRITON_CACHE_DIR='/cache/triton',TVM_FFI_CACHE_DIR='/cache/tvm-ffi',TORCH_CUDA_ARCH_LIST='9.0a',MAX_JOBS='2')
 sys.path.insert(0,'/root/moe')
 mod=importlib.import_module('research_moe_heavy_first');mod._big=lambda:True
 start=time.monotonic();ext=mod._extS(0);cache.commit()
 print('COMPILE_HEAVY_OK',time.monotonic()-start,ext.__file__,flush=True)
 sass=subprocess.run(['cuobjdump','--dump-sass',ext.__file__],capture_output=True,text=True,check=True).stdout
 dest=Path('/cache/moe_queue_audit');dest.mkdir(exist_ok=True)
 (dest/'heavy_v17.sass').write_text(sass);cache.commit()

@app.function(timeout=600,cpu=8,memory=16384,volumes={'/cache':cache})
def compile_grid_constant():
 import os,sys,importlib,time,subprocess,re
 sys.path[:0]=['/root','/root/moe']
 from research_harness.environment import configure_h200_cache
 configure_h200_cache('/cache')
 mod=importlib.import_module('research_moe_grid_constant');mod._big=lambda:True
 start=time.monotonic()
 for name,builder in [('v17',lambda:mod._extS(0)),('ring',lambda:mod._extL(0))]:
  ext=builder();cache.commit()
  print('COMPILE_GRID_OK',name,time.monotonic()-start,ext.__file__,flush=True)
  sass=subprocess.run(['cuobjdump','--dump-sass',ext.__file__],capture_output=True,text=True,check=True).stdout
  resources=subprocess.run(['cuobjdump','--dump-resource-usage',ext.__file__],capture_output=True,text=True,check=True).stdout
  dest=Path('/cache/moe_queue_audit');dest.mkdir(exist_ok=True)
  (dest/('grid_constant_'+name+'.sass')).write_text(sass)
  print('GRID_RESOURCES',name,resources,flush=True)
  for op in ['LDL','STL','LDC','SHFL','MATCH']:
   print('GRID_SASS_COUNT',name,op,len(re.findall(r'\b'+op+r'(?:\.|\b)',sass)),flush=True)
  cache.commit()

@app.function(timeout=120,cpu=2,memory=2048,volumes={'/cache':cache})
def inspect_cached_small():
 import subprocess,re
 patterns=['moe_w4a16_v48G_strict_down_DFENCE1_DRT16/*.so','moe_fullbf16_v48K_DSW4_DKS22/*.so']
 for pattern in patterns:
  files=list(Path('/cache/torch_ext').glob(pattern))
  if not files:print('CACHED_SO_MISSING',pattern,flush=True)
  for path in files:
   print('CACHED_SO',str(path),flush=True)
   resources=subprocess.run(['cuobjdump','--dump-resource-usage',str(path)],capture_output=True,text=True,check=True).stdout
   print(resources,flush=True)
   sass=subprocess.run(['cuobjdump','--dump-sass',str(path)],capture_output=True,text=True,check=True).stdout
   for op in ['LDL','STL','LDC','SHFL']:
    print('SASS_COUNT',path.name,op,len(re.findall(r'\b'+op+r'(?:\.|\b)',sass)),flush=True)
   st=[line for line in sass.splitlines() if re.search(r'\bSTL(?:\.|\b)',line)]
   print('FIRST_LOCAL_STORES','\n'.join(st[:20]),flush=True)

@app.function(timeout=600,cpu=8,memory=16384,volumes={'/cache':cache})
def compile_latest_wx():
 import sys,importlib,time,subprocess,re
 sys.path[:0]=['/root','/root/moe']
 from research_harness.environment import configure_h200_cache
 configure_h200_cache('/cache')
 mod=importlib.import_module('research_moe_latest_wx_strict');mod._big=lambda:True
 mod._sm90a=lambda:['-gencode=arch=compute_90a,code=sm_90a']
 start=time.monotonic();ext=mod._extX(('-DRPW=2','-DWXG2'));cache.commit()
 print('COMPILE_WX52_OK',time.monotonic()-start,ext.__file__,flush=True)
 sass=subprocess.run(['cuobjdump','--dump-sass',ext.__file__],capture_output=True,text=True,check=True).stdout
 resources=subprocess.run(['cuobjdump','--dump-resource-usage',ext.__file__],capture_output=True,text=True,check=True).stdout
 dest=Path('/cache/moe_queue_audit');dest.mkdir(exist_ok=True)
 (dest/'wx52_strict_rpw2.sass').write_text(sass);cache.commit()
 print('WX52_RESOURCES',resources,flush=True)
 for op in ['LDL','STL','LDC','SHFL','HGMMA','WARPGROUP']:
  print('WX52_SASS_COUNT',op,len(re.findall(r'\b'+op+r'(?:\.|\b)',sass)),flush=True)

@app.function(timeout=600,cpu=8,memory=16384,volumes={'/cache':cache})
def compile_gk_round_store():
 import sys,importlib,time,subprocess,re
 sys.path[:0]=['/root','/root/moe']
 from research_harness.environment import configure_h200_cache
 configure_h200_cache('/cache')
 start=time.monotonic()
 for name in ['research_moe_v56_strict','research_moe_v56_round_store']:
  mod=importlib.import_module(name)
  for flags in ['-DSW=4 -DKS2=1 -DMAXM=8','-DSW=4 -DKS2=2']:
   ext=mod._extK(flags);cache.commit()
   res=subprocess.run(['cuobjdump','--dump-resource-usage',ext.__file__],check=True,capture_output=True,text=True).stdout
   sass=subprocess.run(['cuobjdump','--dump-sass',ext.__file__],check=True,capture_output=True,text=True).stdout
   dest=Path('/cache/moe_round_store_audit');dest.mkdir(exist_ok=True)
   tag=name+flags.replace(' ','_').replace('=','')
   (dest/(tag+'.sass')).write_text(sass);(dest/(tag+'.resources')).write_text(res);cache.commit()
   print('GK_ROUND_STORE_RESOURCE',name,flags,ext.__file__,res,flush=True)
   for op in ['LDL','STL','BAR','LDS','STS']:
    print('GK_SASS_COUNT',name,flags,op,len(re.findall(r'\b'+op+r'(?:\.|\b)',sass)),flush=True)
 print('GK_ROUND_STORE_CPU_DONE',time.monotonic()-start,flush=True)

@app.function(timeout=240,gpu='H200',volumes={'/cache':cache})
def compare_gk_round_store():
 import sys,ast,importlib,time,json,hashlib
 sys.path[:0]=['/root','/root/moe']
 from research_harness.environment import configure_h200_cache
 configure_h200_cache('/cache')
 import torch,reference
 from research_harness.gpu import compare_graphs
 baseline=importlib.import_module('research_moe_v56_strict')
 candidate=importlib.import_module('research_moe_v56_round_store')
 # External preparation copy warms only the two GK specializations used here.
 tree=ast.parse(Path(baseline.__file__).read_text())
 prep=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prepare')
 prep.name='prepare_gk_research'
 for i,node in enumerate(prep.body):
  if isinstance(node,ast.With) and any(isinstance(n,ast.Name) and n.id=='ThreadPoolExecutor' for n in ast.walk(node)):
   prep.body[i:i+1]=ast.parse('for flags in K_CFG: _extK(flags)').body;break
 else:raise AssertionError('compile section missing')
 exec(compile(ast.fix_missing_locations(ast.Module(body=[prep],type_ignores=[])),'<external_gk_prepare>','exec'),baseline.__dict__)
 for mod in [baseline,candidate]:
  for flags in baseline.K_CFG:mod._extK(flags)
 weights=reference.generate_weights(5252)
 # Untimed immutable bank is the EP8 local range only; no selected input data cached.
 for key in ['w13','w13_sf','w13_gs','w2','w2_sf','w2_gs']:weights[key]=weights[key][:32].contiguous()
 weights['w13_gs']*=torch.linspace(.65,1.45,32,device='cuda')
 weights['w2_gs']*=torch.linspace(1.6,.5,32,device='cuda')
 state_a=baseline.prepare_gk_research(weights)
 immutable={'w1','w2','g1','g2','nw','w1x','w2x','w2f'}
 state_b={k:(v.clone() if torch.is_tensor(v) and k not in immutable else v) for k,v in state_a.items()}
 reports=[]
 cases=[(m,seed,dist) for m in [1,8,16] for seed,dist in [(103,'normal'),(217,'normal'),(331,'skewed')]]
 cases += [(m,443,'skewed') for m in [2,3,7,9,12,15]]
 cases += [(m,557,'empty_tokens') for m in [8,16]]
 for m,seed,dist in cases:
  data=reference.generate_input(m,'ep8',seed,dist='normal' if dist=='empty_tokens' else dist)
  if dist=='empty_tokens':
   data['topk_ids'].copy_(torch.arange(32,40,device='cuda',dtype=torch.int32).expand(m,-1))
   data['topk_ids'][0,0]=0  # one local expert; other experts and most tokens empty
  flags=baseline.K_CFG[0 if m<=8 else 1];cfg=('gk',0,0,flags)
  gold=reference.ref_kernel(data,weights)
  for label,mod,st in [('baseline',baseline,state_a),('roundstore',candidate,state_b)]:
   out=mod._gk(data,st,cfg);torch.cuda.synchronize();assert torch.isfinite(out).all().item()
   ok,message=reference.compare(out,gold,reference.REL_L2_TOL);assert ok,(m,seed,label,message)
   print('GK_CORRECT',m,seed,dist,label,message,flush=True)
  report=compare_graphs(lambda:baseline._gk(data,state_a,cfg),lambda:candidate._gk(data,state_b,cfg),repetitions=24)
  report.update(m=m,seed=seed,dist=dist,flags=flags,baseline_sha256=hashlib.sha256(Path(baseline.__file__).read_bytes()).hexdigest(),candidate_sha256=hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest())
  reports.append(report);print('GK_PAIRED',json.dumps(report),flush=True)
 path=Path('/cache/moe_probe_reports')/(str(time.time_ns())+'_gk_roundstore.json');path.write_text(json.dumps(reports));cache.commit()
 print('GK_REPORT',str(path),flush=True)
