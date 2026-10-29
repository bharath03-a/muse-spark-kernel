"""CPU-only compilation for safe BF16-PV experiment and its matched control."""
from pathlib import Path
import modal
ROOT=Path(__file__).parent
image=(modal.Image.from_registry('lmsysorg/sglang:dev-muse-spark-open')
 .run_commands('echo /usr/local/lib/python3.12/dist-packages > /opt/sglang/lib/python3.12/site-packages/zz_system.pth','echo /usr/lib/python3/dist-packages >> /opt/sglang/lib/python3.12/site-packages/zz_system.pth').entrypoint([])
 .add_local_file(ROOT/'candidate_openkernels_bf16pv.py','/root/bf16pv.py')
 .add_local_file(ROOT/'candidate_openkernels_pvcontrol.py','/root/pvcontrol.py')
 .add_local_file(ROOT/'candidate_openkernels_fdecfast.py','/root/fdecfast.py')
 .add_local_file(ROOT/'candidate_openkernels_fdeccontrol.py','/root/fdeccontrol.py')
 .add_local_file(ROOT/'candidate_openkernels_v75.py','/root/v75.py'))
cache=modal.Volume.from_name('muse-ext-cache',create_if_missing=True)
app=modal.App('muse-bf16pv-cpu-compile',image=image)
@app.function(timeout=900,cpu=8,memory=16384,volumes={'/cache':cache})
def compile(target:str='bf16pv'):
 import os,importlib.util,time
 os.environ.update(TORCH_EXTENSIONS_DIR='/cache/torch_ext',TRITON_CACHE_DIR='/cache/triton',TVM_FFI_CACHE_DIR='/cache/tvm-ffi',TORCH_CUDA_ARCH_LIST='9.0a',MAX_JOBS='2')
 if target not in ['bf16pv','pvcontrol','fdecfast','fdeccontrol','v75']:raise ValueError(target)
 spec=importlib.util.spec_from_file_location('candidate','/root/'+target+'.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m._big=lambda:True
 m._gencode=lambda:['-gencode=arch=compute_90a,code=sm_90a']
 start=time.monotonic()
 try:
  m._ext();print('CPU_COMPILE_OK',target,time.monotonic()-start,flush=True)
 finally:cache.commit()

@app.function(timeout=120,cpu=2,volumes={'/cache':cache})
def resources():
 import subprocess
 from pathlib import Path
 tool='/usr/local/cuda/bin/cuobjdump'
 for pattern in ['attn_v60_8.so','attn_v60_fdecfast_*.so','attn_v60_fdeccontrol_*.so']:
  libs=list(Path('/cache/torch_ext').rglob(pattern))
  if not libs:print('NO_RESOURCE_LIB',pattern,flush=True);continue
  lib=max(libs,key=lambda p:p.stat().st_mtime)
  result=subprocess.run([tool,'--dump-resource-usage',str(lib)],capture_output=True,text=True,timeout=60)
  print('RESOURCE_LIB',str(lib),'rc',result.returncode,flush=True)
  lines=result.stdout.splitlines()
  for i,line in enumerate(lines):
   if 'fdec' in line.lower():print('\n'.join(lines[i:i+5]),flush=True)

@app.function(timeout=180,cpu=2)
def inspect_fa3_baseline():
 import os,inspect,importlib,importlib.metadata
 from pathlib import Path
 spec=importlib.util.find_spec('sgl_kernel')
 wrapper=Path(spec.origin).parent/'flash_attn.py'
 print('INSTALLED_FA3_WRAPPER',str(wrapper),flush=True)
 print(wrapper.read_text()[:16000],flush=True)
 for pkg in ['sgl-kernel','sglang']:
  try:print('PACKAGE',pkg,importlib.metadata.version(pkg),flush=True)
  except Exception:pass
 names={'mainloop_fwd_sm90_tma_gmma_ws.hpp','softmax.h','flash_api.cpp','flash_fwd_launch_template.h','flash_fwd_kernel_sm90.h'}
 found=[]
 for root in ['/sgl-workspace','/workspace','/opt/sglang','/usr/local/lib/python3.12/dist-packages']:
  if not Path(root).exists():continue
  for folder,dirs,files in os.walk(root):
   dirs[:]=[d for d in dirs if d not in ['.git','__pycache__','node_modules']]
   for name in files:
    if name in names and any(x in folder.lower() for x in ['flash','sgl']):
     p=Path(folder)/name;found.append(str(p));lines=p.read_text(errors='replace').splitlines()
     print('INSTALLED_SOURCE',str(p),flush=True)
     for i,line in enumerate(lines):
      if any(x in line for x in ['ElementP','convert_type','Is_FP8','Is_FP8','max_offset','256.0','float_e4m3','ElementAccum','Element =']):
       print('SOURCE_LINES',i+1,'\n'.join(lines[max(0,i-2):i+4]),flush=True)
 print('SOURCE_COUNT',len(found),flush=True)

@app.function(timeout=180,cpu=4)
def inspect_fa3_binary(sliding:bool=False):
 import importlib.util,importlib.metadata,subprocess,re
 from pathlib import Path
 pkg=Path(importlib.util.find_spec('sgl_kernel').origin).parent
 for d in importlib.metadata.distributions():
  if 'sgl' in (d.metadata['Name'] or '').lower():print('DIST',d.metadata['Name'],d.version,flush=True)
 print('PKG_INIT', (pkg/'__init__.py').read_text()[:10000],flush=True)
 libs=list(pkg.glob('**/flash_ops*.so'))
 for p in libs:
  print('FLASH_BINARY',str(p),p.stat().st_size,flush=True)
  result=subprocess.run(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(p)],capture_output=True,text=True,timeout=90)
  funcs=[line.split('Function ',1)[1].rstrip(':') for line in result.stdout.splitlines() if 'Function ' in line and 'e4m3' in line and 'Li64E' in line and 'fwd' in line.lower()]
  print('FP8_64_FUNCTION_COUNT',len(funcs),flush=True)
  if not funcs:continue
  demangled=subprocess.run(['c++filt'],input='\n'.join(funcs)+'\n',capture_output=True,text=True,timeout=30).stdout.splitlines()
  for text in demangled[:12]:print('DEMANGLED_SAMPLE',text,flush=True)
  selected=[(symbol,text) for symbol,text in zip(funcs,demangled) if re.search(r'CollectiveMainloopFwdSm90<.*?cute::tuple<.*?cute::C<64[lL]?> >?, 64, cutlass::float_e4m3_t',text)]
  if not selected:print('NO_HEAD64_MATCH',flush=True);continue
  print('HEAD64_SPECIALIZATIONS',len(selected),flush=True)
  for _,candidate_text in selected[:4]:print('HEAD64_SAMPLE',candidate_text,flush=True)
  def match_paged(pair):
   match=re.search(r'cutlass::arch::Sm90, ((?:true|false)(?:, (?:true|false))*)',pair[1])
   flags=match.group(1).split(', ') if match else []
   return len(flags)>4 and flags[1 if sliding else 0]=='true' and flags[3]=='true' and flags[4]=='true'
  selected.sort(key=lambda pair: (not match_paged(pair), 'VarlenDynamicPersistentTileScheduler' not in pair[1]))
  print('PAGED_CAUSAL_VARLEN_MATCHES',sum(match_paged(pair) for pair in selected),flush=True)
  symbol,text=selected[0]
  print('SELECTED_SYMBOL',symbol,'DEMANGLER',text,flush=True)
  result=subprocess.run(['/usr/local/cuda/bin/cuobjdump','--dump-sass','--function',symbol,str(p)],capture_output=True,text=True,timeout=75)
  print('SASS_RC',result.returncode,'SASS_BYTES',len(result.stdout),flush=True)
  lines=result.stdout.splitlines()
  mma=[(i,l) for i,l in enumerate(lines) if 'HGMMA' in l or 'QGMMA' in l or 'HMMA' in l or 'F2FP' in l]
  for i,l in mma[:80]:print('SASS_MATH',i,l,flush=True)
  print('MMA_COUNT',len(mma),flush=True)
  for key in ['QGMMA','HGMMA','E4M3.F32','F16.F32','BF16.F32']:
   selected_lines=[l for l in lines if key in l]
   print('SASS_TYPE_COUNT',key,len(selected_lines),flush=True)
   for l in selected_lines[:24]:print('SASS_TYPE',key,l,flush=True)

@app.function(timeout=120,cpu=2)
def inspect_fp8_mma():
 import subprocess,tempfile
 from pathlib import Path
 source=r'''
#include <stdint.h>
extern "C" __global__ void pv16(const uint32_t* a,const uint32_t* b,float* out) {
 float c0=0,c1=0,c2=0,c3=0;uint32_t a0=a[2*threadIdx.x],a1=a[2*threadIdx.x+1],b0=b[threadIdx.x];
 asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};" : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3) : "r"(a0),"r"(a1),"r"(b0));
 out[4*threadIdx.x]=c0;out[4*threadIdx.x+1]=c1;out[4*threadIdx.x+2]=c2;out[4*threadIdx.x+3]=c3;
}
extern "C" __global__ void pv32(const uint32_t* a,const uint32_t* b,float* out) {
 float c0=0,c1=0,c2=0,c3=0;uint32_t a0=a[4*threadIdx.x],a1=a[4*threadIdx.x+1],a2=a[4*threadIdx.x+2],a3=a[4*threadIdx.x+3],b0=b[2*threadIdx.x],b1=b[2*threadIdx.x+1];
 asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
 out[4*threadIdx.x]=c0;out[4*threadIdx.x+1]=c1;out[4*threadIdx.x+2]=c2;out[4*threadIdx.x+3]=c3;
}
extern "C" __global__ void pv16_f16(const uint32_t* a,const uint32_t* b,float* out) {
 float c0=0,c1=0,c2=0,c3=0;uint32_t a0=a[4*threadIdx.x],a1=a[4*threadIdx.x+1],a2=a[4*threadIdx.x+2],a3=a[4*threadIdx.x+3],b0=b[2*threadIdx.x],b1=b[2*threadIdx.x+1];
 asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
 out[4*threadIdx.x]=c0;out[4*threadIdx.x+1]=c1;out[4*threadIdx.x+2]=c2;out[4*threadIdx.x+3]=c3;
}
'''
 with tempfile.TemporaryDirectory() as d:
  p=Path(d);(p/'probe.cu').write_text(source)
  cmd=['/usr/local/cuda/bin/nvcc','-cubin','-O3','-arch=sm_90a','-Xptxas=-v',str(p/'probe.cu'),'-o',str(p/'probe.cubin')]
  r=subprocess.run(cmd,capture_output=True,text=True,timeout=60);print('FP8_MMA_CONTROLS_COMPILE',r.returncode,r.stdout,r.stderr,flush=True)
  if r.returncode==0:
   r=subprocess.run(['/usr/local/cuda/bin/cuobjdump','-sass',str(p/'probe.cubin')],capture_output=True,text=True,timeout=30)
   print('FP8_MMA_CONTROLS_SASS',r.stdout,flush=True)

@app.function(timeout=120,cpu=2,volumes={'/cache':cache})
def inspect_fdec_sass():
 import subprocess,ast,hashlib
 from pathlib import Path
 source=Path('/root/v75.py').read_text()
 tree=ast.parse(source)
 cuda=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CUDA' for t in n.targets)))
 libs=list(Path('/cache/torch_ext').rglob('attn_v75_8.so'))
 if not libs:raise RuntimeError('ABORT_NO_CACHED_V75')
 lib=max(libs,key=lambda p:p.stat().st_mtime)
 assert cuda in (lib.parent/'cuda.cu').read_text(),'ABORT_SOURCE_MISMATCH'
 print('FDEC_CACHED_SOURCE_SHA256',hashlib.sha256(cuda.encode()).hexdigest(),flush=True)
 symbol='_Z4fdecILb0ELb1EEv1F'
 r=subprocess.run(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage','--function',symbol,str(lib)],capture_output=True,text=True,timeout=30)
 print('FDEC_RESOURCES',r.returncode,r.stdout,r.stderr,flush=True)
 r=subprocess.run(['/usr/local/cuda/bin/cuobjdump','--dump-sass','--function',symbol,str(lib)],capture_output=True,text=True,timeout=60)
 print('FDEC_SASS_SIZE',r.returncode,len(r.stdout),r.stderr,flush=True)
 import re
 saved=Path('/cache/research/fdec_v75_sass.txt');saved.parent.mkdir(parents=True,exist_ok=True);saved.write_text(r.stdout);cache.commit()
 lines=r.stdout.splitlines()
 for key in ['HMMA.','MUFU.EX2','LDGSTS','LDS.']:
  matches=[line for line in lines if key in line]
  print('FDEC_SASS_COUNT',key,len(matches),flush=True)
 for line in lines:
  if 'HMMA.' in line or 'MUFU.EX2' in line:print('FDEC_CHAIN',line,flush=True)

