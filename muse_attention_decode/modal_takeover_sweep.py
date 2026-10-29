"""Bounded H200 synthetic FA3 split sweep; relative timings, not judge scores."""
from pathlib import Path
import modal
image=(modal.Image.from_registry('lmsysorg/sglang:dev-muse-spark-open')
 .run_commands('echo /usr/local/lib/python3.12/dist-packages > /opt/sglang/lib/python3.12/site-packages/zz_system.pth', 'echo /usr/lib/python3/dist-packages >> /opt/sglang/lib/python3.12/site-packages/zz_system.pth')
 .entrypoint([]).add_local_dir(Path(__file__).parent,remote_path="/root/attention",ignore=["profiles", "__pycache__"]))
app=modal.App('muse-attention-takeover', image=image)
@app.function(gpu='H200',timeout=300)
def sweep():
 import torch
 from sgl_kernel.flash_attn import flash_attn_with_kvcache, get_scheduler_metadata
 print('GPU',torch.cuda.get_device_name(),flush=True)
 torch.manual_seed(42)
 for b in [1,16,32]:
  n=80000
  k=torch.randn(b*n,1,2,64,device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn)
  v=torch.randn_like(k,dtype=torch.bfloat16).to(torch.float8_e4m3fn)
  pages=torch.randperm(b*n,device='cuda',dtype=torch.int32).view(b,n)
  lens=torch.full((b,),n,device='cuda',dtype=torch.int32)
  flush=torch.empty(64*1024*1024,device='cuda',dtype=torch.uint8)
  for qlen in [1,8]:
   q=torch.randn(b*qlen,16,64,device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn)
   cu=torch.arange(b+1,device='cuda',dtype=torch.int32)*qlen
   out=torch.empty(b*qlen,16,64,device='cuda',dtype=torch.bfloat16)
   for splits in [0,4,8,16,32,64,128]:
    def call():
     meta=get_scheduler_metadata(batch_size=b,max_seqlen_q=qlen,max_seqlen_k=n,num_heads=16,num_heads_k=2,headdim=64,cache_seqlens=lens,qkv_dtype=q.dtype,cu_seqlens_q=cu,page_size=1,causal=True,num_splits=splits,pack_gqa=True)
     return flash_attn_with_kvcache(q=q,k_cache=k,v_cache=v,page_table=pages,cache_seqlens=lens,cu_seqlens_q=cu,max_seqlen_q=qlen,softmax_scale=.6841258108,causal=True,scheduler_metadata=meta,pack_gqa=True,num_splits=splits,out=out)
    try:
     call();torch.cuda.synchronize()
     g=torch.cuda.CUDAGraph()
     with torch.cuda.graph(g): call()
     samples=[]
     for _ in range(12):
      flush.zero_();a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True)
      a.record();g.replay();z.record();z.synchronize();samples.append(a.elapsed_time(z)*1000)
     print('RESULT',b,qlen,splits,round(sorted(samples)[len(samples)//2],2),flush=True)
    except Exception as e: print('ERROR',b,qlen,splits,str(e)[:300],flush=True)
  del k,v,pages

@app.function(gpu='H200',timeout=300)
def tc_sweep():
 import sys
 sys.path.insert(0,'/root/attention')
 import torch
 import reference
 import candidate_takeover_tc64 as tc
 import submission as base
 print('GPU',torch.cuda.get_device_name(),flush=True)
 for b in [1,16,32]:
  data=reference.generate_input(batch=b,ctx=80000,heads='tp8',layer='full',seed=1234,page=1)
  expected=base.kernel_fn(data).clone()
  flush=torch.empty(64*1024*1024,device='cuda',dtype=torch.uint8)
  for split in [16,32,64,128]:
   tc.TC_SPLITS=split
   actual=tc.kernel_fn(data)
   torch.cuda.synchronize()
   err=((actual.float()-expected.float()).norm()/expected.float().norm()).item()
   g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):tc.kernel_fn(data)
   times=[]
   for _ in range(12):
    flush.zero_();a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True)
    a.record();g.replay();z.record();z.synchronize();times.append(a.elapsed_time(z)*1000)
   print('TC_RESULT',b,split,round(sorted(times)[6],2),'err',err,flush=True)

@app.function(gpu='H200',timeout=300)
def repack_sweep():
 import torch
 from sgl_kernel.flash_attn import flash_attn_with_kvcache,get_scheduler_metadata
 print('GPU',torch.cuda.get_device_name(),flush=True)
 torch.manual_seed(123)
 for b in [1,16,32]:
  n=80000
  k=torch.randn(b*n,1,2,64,device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn)
  v=torch.randn_like(k,dtype=torch.bfloat16).to(torch.float8_e4m3fn)
  pages=torch.randperm(b*n,device='cuda',dtype=torch.int32).view(b,n)
  lens=torch.full((b,),n,device='cuda',dtype=torch.int32)
  compact_k=torch.empty((b*n,2,64),device='cuda',dtype=k.dtype)
  compact_v=torch.empty_like(compact_k)
  compact_pages=torch.arange(b*n//64,device='cuda',dtype=torch.int32).reshape(b,n//64)
  flush=torch.empty(64*1024*1024,device='cuda',dtype=torch.uint8)
  for qlen in [1,8]:
   q=torch.randn(b*qlen,16,64,device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn)
   cu=torch.arange(b+1,device='cuda',dtype=torch.int32)*qlen
   out=torch.empty((b*qlen,16,64),device='cuda',dtype=torch.bfloat16)
   reference=None
   for variant in ['page1','repack_page64','repack_contiguous']:
    def call():
     if variant!='page1':
      torch.index_select(k.view(b*n,2,64),0,pages.flatten(),out=compact_k)
      torch.index_select(v.view(b*n,2,64),0,pages.flatten(),out=compact_v)
     if variant=='repack_contiguous':
      return flash_attn_with_kvcache(q=q.view(b,qlen,16,64),k_cache=compact_k.view(b,n,2,64),v_cache=compact_v.view(b,n,2,64),cache_seqlens=lens,softmax_scale=.6841258108,causal=True,pack_gqa=True,out=out.view(b,qlen,16,64))
     ps=1 if variant=='page1' else 64
     kc=k if ps==1 else compact_k.view(b*n//64,64,2,64)
     vc=v if ps==1 else compact_v.view(b*n//64,64,2,64)
     pt=pages if ps==1 else compact_pages
     meta=get_scheduler_metadata(batch_size=b,max_seqlen_q=qlen,max_seqlen_k=n,num_heads=16,num_heads_k=2,headdim=64,cache_seqlens=lens,qkv_dtype=q.dtype,cu_seqlens_q=cu,page_size=ps,causal=True,num_splits=0,pack_gqa=True)
     return flash_attn_with_kvcache(q=q,k_cache=kc,v_cache=vc,page_table=pt,cache_seqlens=lens,cu_seqlens_q=cu,max_seqlen_q=qlen,softmax_scale=.6841258108,causal=True,scheduler_metadata=meta,pack_gqa=True,out=out)
    try:
     call();torch.cuda.synchronize()
     if reference is None:reference=out.clone()
     err=((out.float()-reference.float()).norm()/reference.float().norm()).item()
     g=torch.cuda.CUDAGraph()
     with torch.cuda.graph(g):call()
     times=[]
     for _ in range(12):
      flush.zero_();a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True)
      a.record();g.replay();z.record();z.synchronize();times.append(a.elapsed_time(z)*1000)
     print('REPACK_RESULT',b,qlen,variant,round(sorted(times)[6],2),'err',err,flush=True)
    except Exception as e:print('REPACK_ERROR',b,qlen,variant,str(e)[:300],flush=True)

@app.function(gpu='H200',timeout=300)
def repack_triton_sweep():
 import sys
 sys.path.insert(0,"/root/attention")
 from gather_kv import gather_kv
 import torch
 from sgl_kernel.flash_attn import flash_attn_with_kvcache,get_scheduler_metadata
 print('GPU',torch.cuda.get_device_name(),flush=True)
 torch.manual_seed(123)
 for b in [1,16,32]:
  n=80000
  k=torch.randn(b*n,1,2,64,device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn)
  v=torch.randn_like(k,dtype=torch.bfloat16).to(torch.float8_e4m3fn)
  pages=torch.randperm(b*n,device='cuda',dtype=torch.int32).view(b,n)
  lens=torch.full((b,),n,device='cuda',dtype=torch.int32)
  compact_k=torch.empty((b*n,2,64),device='cuda',dtype=k.dtype)
  compact_v=torch.empty_like(compact_k)
  compact_pages=torch.arange(b*n//64,device='cuda',dtype=torch.int32).reshape(b,n//64)
  flush=torch.empty(64*1024*1024,device='cuda',dtype=torch.uint8)
  for qlen in [1,8]:
   q=torch.randn(b*qlen,16,64,device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn)
   cu=torch.arange(b+1,device='cuda',dtype=torch.int32)*qlen
   out=torch.empty((b*qlen,16,64),device='cuda',dtype=torch.bfloat16)
   reference=None
   for variant in ['page1','repack_page64','repack_contiguous']:
    def call():
     if variant!='page1':
      gather_kv[((b*n+63)//64,)](k.view(torch.uint8),v.view(torch.uint8),pages,compact_k.view(torch.uint8),compact_v.view(torch.uint8),b*n,ROW=64,num_warps=4)
     if variant=='repack_contiguous':
      return flash_attn_with_kvcache(q=q.view(b,qlen,16,64),k_cache=compact_k.view(b,n,2,64),v_cache=compact_v.view(b,n,2,64),cache_seqlens=lens,softmax_scale=.6841258108,causal=True,pack_gqa=True,out=out.view(b,qlen,16,64))
     ps=1 if variant=='page1' else 64
     kc=k if ps==1 else compact_k.view(b*n//64,64,2,64)
     vc=v if ps==1 else compact_v.view(b*n//64,64,2,64)
     pt=pages if ps==1 else compact_pages
     meta=get_scheduler_metadata(batch_size=b,max_seqlen_q=qlen,max_seqlen_k=n,num_heads=16,num_heads_k=2,headdim=64,cache_seqlens=lens,qkv_dtype=q.dtype,cu_seqlens_q=cu,page_size=ps,causal=True,num_splits=0,pack_gqa=True)
     return flash_attn_with_kvcache(q=q,k_cache=kc,v_cache=vc,page_table=pt,cache_seqlens=lens,cu_seqlens_q=cu,max_seqlen_q=qlen,softmax_scale=.6841258108,causal=True,scheduler_metadata=meta,pack_gqa=True,out=out)
    try:
     call();torch.cuda.synchronize()
     if reference is None:reference=out.clone()
     err=((out.float()-reference.float()).norm()/reference.float().norm()).item()
     g=torch.cuda.CUDAGraph()
     with torch.cuda.graph(g):call()
     times=[]
     for _ in range(12):
      flush.zero_();a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True)
      a.record();g.replay();z.record();z.synchronize();times.append(a.elapsed_time(z)*1000)
     print('REPACK_RESULT',b,qlen,variant,round(sorted(times)[6],2),'err',err,flush=True)
    except Exception as e:print('REPACK_ERROR',b,qlen,variant,str(e)[:300],flush=True)

@app.function(timeout=300)
def inspect_backends():
 import inspect
 import flashinfer.decode as d
 print('DECODE_FUNCTION_BEGIN',flush=True)
 print(inspect.getsource(d.trtllm_batch_decode_with_kv_cache),flush=True)
 print('DECODE_FUNCTION_END',flush=True)
 print('XQA_WRAPPER',inspect.getsource(d.xqa_batch_decode_with_kv_cache),flush=True)
 import importlib
 x=importlib.import_module('flashinfer.xqa')
 print('XQA_NATIVE',inspect.getsource(x.xqa),flush=True)
 import flashinfer.jit.xqa as jx
 print('XQA_JIT',inspect.getsource(jx.gen_xqa_module),flush=True)
 for name in dir(d):
  if any(x in name for x in ['trtllm','backend','supported']):
   obj=getattr(d,name)
   if inspect.isfunction(obj) and name!='trtllm_batch_decode_with_kv_cache':
    try:print('HELPER',name,inspect.getsource(obj)[:30000],flush=True)
    except Exception:pass

@app.function(timeout=120,volumes={'/cache':modal.Volume.from_name('muse-ext-cache',create_if_missing=True)})
def inspect_xqa_cache():
 import os
 os.environ['TVM_FFI_CACHE_DIR']='/cache/tvm-ffi'
 import flashinfer.jit.env as e
 from pathlib import Path
 for key in dir(e):
  if any(x in key for x in ['DIR','PATH']):
   value=getattr(e,key)
   if isinstance(value,(str,Path)):print('ENV_PATH',key,value,flush=True)
 for root in ['/cache','/root/.cache']:
  for folder,dirs,files in os.walk(root):
   for name in files:
    if ('xqa' in folder.lower() or 'xqa' in name.lower()) and name.endswith(('.so','.cubin','.ptx')):
     print('CACHED_XQA',os.path.join(folder,name),flush=True)
