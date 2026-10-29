"""External-only whole frontend route extraction; no official submission entrypoint."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[2] if modal.is_local() else Path('/root')
image=(modal.Image.from_registry('lmsysorg/sglang:dev-muse-spark-open')
 .run_commands('echo /usr/local/lib/python3.12/dist-packages > /opt/sglang/lib/python3.12/site-packages/zz_system.pth','echo /usr/lib/python3/dist-packages >> /opt/sglang/lib/python3.12/site-packages/zz_system.pth').entrypoint([]))
if modal.is_local():
 image=(image.add_local_dir(ROOT/'muse_moe_w4a16','/root/moe',ignore=['profiles/**','__pycache__/**'])
  .add_local_dir(ROOT/'muse_decode_layer','/root/decode',ignore=['profiles/**','__pycache__/**'])
  .add_local_dir(ROOT/'research_harness','/root/research_harness',ignore=['results/**','__pycache__/**']))
app=modal.App('muse-whole-route-research',image=image)
cache=modal.Volume.from_name('muse-ext-cache',create_if_missing=True)


def setup(compile_only=False):
 import sys
 sys.path[:0]=['/root/decode','/root/moe/research','/root']
 from research_harness.environment import configure_h200_cache
 configure_h200_cache('/cache')
 from whole_frontend_extract import load_frontend,load_compile_only
 return (load_compile_only if compile_only else load_frontend)('/root/decode/candidate_cp14_attention_strict_moe.py')

@app.function(timeout=1200,cpu=8,memory=16384,volumes={'/cache':cache})
def compile_frontend():
 import time,hashlib
 mod=setup(compile_only=True)
 # CPU architecture specialization changes only this isolated copied research module.
 mod.OKA__big=lambda:True
 mod.OKA__gencode=lambda:['-gencode=arch=compute_90a,code=sm_90a']
 start=time.monotonic()
 for name in ['_build_cuda','cp14__build_cuda','cp14__build_zip','cp14__r9_build','cp14__r15_build','OKA__ext']:
  try:
   ext=getattr(mod,name)();artifact=ext.__file__
  except ImportError as exc:
   if 'libcuda.so.1' not in str(exc):raise
   # load_inline has finished building before its CPU dynamic-loader failure.
   tb=exc.__traceback__;artifact=None
   while tb:
    frame=tb.tb_frame
    if frame.f_code.co_name=='_jit_compile':
     artifact=str(Path(frame.f_locals['build_directory'])/(frame.f_locals['name']+'.so'))
    tb=tb.tb_next
   assert artifact and Path(artifact).is_file(),(artifact,list(Path('/cache/torch_ext').glob('muse_proj*')))
   print('CPU_DYNAMIC_LOAD_UNAVAILABLE',name,str(exc),flush=True)
  cache.commit()
  print('FRONTEND_CACHE_READY',name,artifact,time.monotonic()-start,flush=True)
 from torch.utils.cpp_extension import load_inline
 tag=hashlib.sha1((mod.cp14_SLICE_SRC+' '.join(mod.cp14_CUDA_FLAGS)).encode()).hexdigest()[:10]
 ext=load_inline(name=f'muse_slice_{tag}',cpp_sources=mod.cp14_SLICE_CPP,cuda_sources=mod.cp14_SLICE_SRC,functions=['launch_slice'],extra_cuda_cflags=mod.cp14_CUDA_FLAGS,extra_cflags=['-O3'])
 cache.commit();print('FRONTEND_CACHE_DONE',mod.source_sha256,time.monotonic()-start,flush=True)

@app.function(timeout=600,gpu='H200',volumes={'/cache':cache})
def compare_actual_routes(seed_offset: int=917, router_seed: int=-1, layers: str='sliding,full'):
 import sys,importlib.util,json,time,hashlib,gc
 mod=setup()
 import torch
 import reference as whole_ref
 from research_harness.gpu import compare_graphs
 def load(name,path):
  spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
 baseline=load('route_grid','/root/moe/research_moe_grid_constant.py')
 candidate=load('route_wx','/root/moe/research_moe_latest_wx_strict.py')
 moe_ref=load('route_reference','/root/moe/reference.py')
 weights=whole_ref.generate_weights(5353,**({'router_seed':router_seed} if router_seed>=0 else {}))
 started=time.monotonic();frontend=mod.prepare(weights)
 print('FRONTEND_PREPARED_SECONDS',time.monotonic()-started,mod.source_sha256,flush=True)
 raw={k:weights[k] for k in ('w13','w13_sf','w13_gs','w2','w2_sf','w2_gs')};raw['post_norm_w']=weights['post_expert_norm']
 state_a=candidate.prepare(raw)
 immutable={'w1','w2','g1','g2','nw','w1x','w2x','w2f'}
 state_b={k:(v.clone() if torch.is_tensor(v) and k not in immutable else v) for k,v in state_a.items()}
 manifest=json.loads(Path('/root/moe/research/whole_route_cases.json').read_text())
 cases=[c for c in manifest['job_benchmarks'] if c['batch']*c.get('q_len',1) in (32,128,256) and c['layer'] in layers.split(',')]
 results=[]
 for case in cases:
  c=dict(case);c['seed']=(c['seed']+seed_offset)%(2**63-1);c['local_start']=0
  # Same baseline busiest-rank selection as judge; outside MoE timing.
  selected=whole_ref.case_fields(c,weights)
  c.update(selected)
  data=whole_ref.generate_input(**c)
  extracted=mod.extract_frontend(data,frontend);torch.cuda.synchronize()
  probe={k:v.clone() if torch.is_tensor(v) else dict(v) for k,v in extracted.items()}
  ids=probe['topk_ids'];hist=torch.bincount(ids[ids>=0].long(),minlength=32).cpu().tolist()
  gold=moe_ref.ref_kernel(probe,raw)
  for label,impl,st in [('grid',baseline,state_a),('wx',candidate,state_b)]:
   out=impl.kernel_fn(probe,st);torch.cuda.synchronize();assert torch.isfinite(out).all().item()
   ok,message=moe_ref.compare(out,gold,moe_ref.REL_L2_TOL);assert ok,(label,c,message)
   print('ACTUAL_REFERENCE_PASS',label,c,message,flush=True)
  report=compare_graphs(lambda:baseline.kernel_fn(probe,state_a),lambda:candidate.kernel_fn(probe,state_b),repetitions=24)
  report.update(candidate_sha256=hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(),baseline_sha256=hashlib.sha256(Path(baseline.__file__).read_bytes()).hexdigest(),weights_seed=5353,case=c,local_histogram=hist,local_routes=sum(hist),distinct=sum(x>0 for x in hist),frontend_sha256=mod.source_sha256,router_seed=router_seed,router_scope='explicit' if router_seed>=0 else 'default5353_surrogate')
  results.append(report);print('ACTUAL_PAIRED',json.dumps(report),flush=True)
  del data,extracted,probe,gold;gc.collect();torch.cuda.empty_cache()
 path=Path('/cache/moe_probe_reports')/(str(time.time_ns())+'_whole_routes.json');path.write_text(json.dumps(results));cache.commit()
 print('ACTUAL_REPORT',str(path),flush=True)
