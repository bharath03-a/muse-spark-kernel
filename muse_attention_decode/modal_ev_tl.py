from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
image=(modal.Image.from_registry('lmsysorg/sglang:dev-muse-spark-open')
 .run_commands('echo /usr/local/lib/python3.12/dist-packages > /opt/sglang/lib/python3.12/site-packages/zz_system.pth','echo /usr/lib/python3/dist-packages >> /opt/sglang/lib/python3.12/site-packages/zz_system.pth').entrypoint([]))
for local,remote in [('candidate_keel.py','base.py'),('reference.py','reference.py'),('checks.py','checks.py')]:
 image=image.add_local_file(ROOT/'muse_attention_decode'/local,'/root/'+remote)
cache=modal.Volume.from_name('muse-ext-cache',create_if_missing=True)
app=modal.App('muse-attn-tl',image=image)
def patch(src):
 def rep(a,b):
  nonlocal src
  assert src.count(a)==1,(src.count(a),a)
  src=src.replace(a,b)
 rep("#include <torch/extension.h>","#include <torch/extension.h>\n__device__ long long g_ev[256*8];\n__device__ __forceinline__ long long evn(){long long t;asm volatile(\"mov.u64 %0, %%globaltimer;\":\"=l\"(t));return t;}",)
 rep("    const int c = blockIdx.x, G = gridDim.x;\n","    const int c = blockIdx.x, G = gridDim.x;\n    if (tid==0) g_ev[c*8+0]=evn();\n")
 rep("        int pvs[PF][VP];\n","        int pvs[PF][VP];\n        if (ptid==0) g_ev[c*8+1]=evn();\n")
 rep("        if (RR && nst_total > 1 && wg == 1) asm volatile(\"bar.arrive 6, 256;\" ::: \"memory\");","        if (tid==0 && !g_ev[c*8+2]) g_ev[c*8+2]=evn();\n        if (RR && nst_total > 1 && wg == 1) asm volatile(\"bar.arrive 6, 256;\" ::: \"memory\");")
 rep("    asm volatile(\"bar.sync %0, 128;\" :: \"r\"(4 + wg) : \"memory\");   // every partial + m / l store of this WG issued","    if (tid==0) g_ev[c*8+3]=evn();\n    asm volatile(\"bar.sync %0, 128;\" :: \"r\"(4 + wg) : \"memory\");   // every partial + m / l store of this WG issued")
 rep("                     ((long)(cf + fb) * 2 + wg) * LG, L, t128);\n    }\n}","                     ((long)(cf + fb) * 2 + wg) * LG, L, t128);\n    }\n    if (tid==0) g_ev[c*8+4]=evn();\n}")
 rep("// attn-engine bal: the fulls balanced","void wst_ev(torch::Tensor o){ cudaMemcpyFromSymbol(o.data_ptr(), g_ev, sizeof(g_ev)); }\nvoid wst_ev_zero(){ cudaMemset(nullptr,0,0); long long z[256*8]={0}; cudaMemcpyToSymbol(g_ev, z, sizeof(z)); }\n// attn-engine bal: the fulls balanced")
 return src
@app.function(gpu='H200',timeout=1500,volumes={'/cache':cache})
def probe():
 import os,sys,importlib.util
 os.environ.update(TORCH_EXTENSIONS_DIR='/cache/torch_ext',TRITON_CACHE_DIR='/cache/triton',TORCH_CUDA_ARCH_LIST='9.0a')
 sys.path.insert(0,'/root')
 import torch,reference
 spec=importlib.util.spec_from_file_location('base','/root/base.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m._zh_join()
 flags=["-O3","--use_fast_math","-gencode=arch=compute_90a,code=sm_90a","-DOKC_WST_ACC32=1","-DZH_SCH=1"]
 src=patch(m.ZH_WST_CUDA)
 cpp=m.ZH_WST_CPP+"\nvoid wst_ev(torch::Tensor o);\nvoid wst_ev_zero();\n"
 m._ZH['wst']=m._zh_load_inline(name="okc_zh_wst_tl_"+m._zh_tag(src),cpp_sources=[cpp],cuda_sources=[src],functions=["wst_attn_bal","wst_ev","wst_ev_zero"],extra_cuda_cflags=flags,verbose=False)
 ext=m._ZH['wst']
 for bs in (16,24):
  data=reference.generate_input(batch=bs,ctx=80000,heads='tp8',layer='full',q_len=8,page=1,seed=77+bs,dist='coherent',spread='agentic')
  m.kernel_fn(data);torch.cuda.synchronize()
  for rep in range(3):
   ext.wst_ev_zero();torch.cuda.synchronize()
   m.kernel_fn(data);torch.cuda.synchronize()
   o=torch.zeros(256*8,dtype=torch.int64);ext.wst_ev(o)
   t=o.view(256,8)[:132].double()
   t0=t[:,0].min()
   print('TL',bs,rep,' '.join('s%d min/med/max %.1f/%.1f/%.1f'%(k,((t[:,k]-t0)/1e3).min(),((t[:,k]-t0)/1e3).median(),((t[:,k]-t0)/1e3).max()) for k in range(5)),flush=True)
@app.local_entrypoint()
def main(): probe.remote()
