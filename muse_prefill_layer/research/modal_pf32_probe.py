"""Bounded old-vs-pf32 strict full W13/SwiGLU/W2 graph comparison."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
image=(modal.Image.from_registry('lmsysorg/sglang:dev-muse-spark-open')
 .run_commands('echo /usr/local/lib/python3.12/dist-packages > /opt/sglang/lib/python3.12/site-packages/zz_system.pth','echo /usr/lib/python3/dist-packages >> /opt/sglang/lib/python3.12/site-packages/zz_system.pth')
 .entrypoint([]).add_local_file(ROOT/'candidate_prefill_w13_mixed.py','/root/old.py')
 .add_local_file(ROOT/'candidate_prefill_pf32_strict.py','/root/new.py')
 .add_local_file(ROOT/'nvfp4.py','/root/nvfp4.py')
 .add_local_dir(ROOT.parent/'research_harness','/root/research_harness',ignore=['results/**','__pycache__/**']))
app=modal.App('muse-prefill-pf32-strict-probe',image=image)
cache=modal.Volume.from_name('muse-ext-cache',create_if_missing=True)
@app.function(timeout=300,gpu='H200',volumes={'/cache':cache})
def probe():
 import os,sys,time,json,importlib.util
 os.environ.update(TORCH_EXTENSIONS_DIR='/cache/torch_ext',TORCH_CUDA_ARCH_LIST='9.0a',TRITON_CACHE_DIR='/cache/triton',TVM_FFI_CACHE_DIR='/cache/tvm-ffi',MAX_JOBS='2')
 sys.path.insert(0,'/root')
 import torch,nvfp4
 from research_harness.gpu import compare_graphs
 def load(name):
  spec=importlib.util.spec_from_file_location(name,'/root/'+name+'.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
 old,new=load('old'),load('new');e0,e1=old._w4g(),new._w4g();assert e0 and e1
 start=time.monotonic()
 # Actual local-route counts and allocated capacity are distinct, as in the layer.
 cases=[(1363,6024,False,0,False),(1363,6024,True,2,False),
        (5401,24096,False,1,False),(5401,24096,True,2,False),
        (29317,131072,True,2,False),(29317,131072,True,2,True)]
 for seed in (8281,8282,8283):
  gen=torch.Generator(device='cuda').manual_seed(seed)
  w13,s13,g13=nvfp4.random_weights(8192,4096,generator=gen,device='cuda',leading=(32,));g13=g13.expand(32).contiguous()*torch.exp(torch.randn(32,generator=gen,device='cuda'))
  w2,s2,g2=nvfp4.random_weights(4096,4096,generator=gen,device='cuda',leading=(32,));g2=g2.expand(32).contiguous()*torch.exp(torch.randn(32,generator=gen,device='cuda'))
  p=torch.stack((torch.arange(4096,device='cuda').view(-1,8),4096+torch.arange(4096,device='cuda').view(-1,8)),1).reshape(-1)
  op13,og13=old._w4pack(w13.index_select(1,p),s13.index_select(1,p),g13)
  np13,ng13=new._w4pack(w13,s13,g13,swiglu=True)
  op2,og2=old._w4pack(w2,s2,g2);np2,ng2=new._w4pack(w2,s2,g2)
  assert torch.equal(op13,np13) and torch.equal(og13,ng13) and torch.equal(op2,np2) and torch.equal(og2,ng2)
  print('PF32_PACK_BITIDENTICAL',seed,flush=True)
  del np13,ng13,np2,ng2
  ref13=nvfp4.dequantize(w13[0],s13[0],g13[0]);ref2=nvfp4.dequantize(w2[0],s2[0],g2[0])
  del w13,s13,g13,w2,s2,g2
  for actual,capacity,skew,mode,pdl in cases:
   popularity=[64,32,16,8,4,2,1,1]+[0]*24 if skew else [1]*32
   counts=[actual*q//sum(popularity) for q in popularity];counts[0]+=actual-sum(counts)
   offs=torch.tensor(counts,device='cuda',dtype=torch.int32).cumsum(0).to(torch.int32)
   # Identical input for large PDL on/off isolates its scheduling effect.
   xgen=torch.Generator(device='cuda').manual_seed(seed+actual+int(skew))
   a=torch.randn(capacity,4096,generator=xgen,device='cuda').to(torch.bfloat16)
   c0=torch.zeros(24,device='cuda',dtype=torch.int32);c1=torch.zeros_like(c0)
   def baseline():
    inter=old._w4gemm(a,offs,op13,og13,c0[0:8],ext=e0,wide=mode,fuse=True)
    out=old._w4gemm(inter,offs,op2,og2,c0[8:16],ext=e0,wide=mode)
    return out[:actual]
   def candidate():
    inter=new._w4gemm(a,offs,op13,og13,c1[0:8],ext=e1,wide=mode,swiglu=1,pdl=pdl)
    out=new._w4gemm(inter,offs,op2,og2,c1[8:16],ext=e1,wide=mode,pdl=pdl)
    return out[:actual]
   before,after=baseline(),candidate();torch.cuda.synchronize()
   assert torch.isfinite(before).all().item() and torch.isfinite(after).all().item()
   relative=float((after.float()-before.float()).norm()/before.float().norm())
   exact=torch.equal(before,after);assert relative<1e-5,(relative,exact)
   h=(a[:8].float()@ref13.T).to(torch.bfloat16);gate,up=h[:,:4096].float(),h[:,4096:].float()
   inter=(torch.nn.functional.silu(gate)*up).to(torch.bfloat16)
   reference=(inter.float()@ref2.T).to(torch.bfloat16)
   referr=float((after[:8].float()-reference.float()).norm()/reference.float().norm())
   assert referr<0.015,referr
   assert e0.w4g_err(1)==0 and e1.w4g_err(1)==0
   print('PF32_CHECK',json.dumps(dict(seed=seed,actual=actual,capacity=capacity,skew=skew,mode=mode,pdl=pdl,finite=True,bitidentical=exact,relative_l2=relative,reference_l2=referr)),flush=True)
   report=compare_graphs(baseline,candidate,repetitions=24)
   print('PF32_PAIRED',json.dumps(dict(seed=seed,actual=actual,capacity=capacity,skew=skew,mode=mode,pdl=pdl,**report)),flush=True)
   del a,before,after,offs,c0,c1
  del op13,op2,og13,og2,ref13,ref2
 print('PF32_GPU_SECONDS',time.monotonic()-start,flush=True)
