"""CPU-only compilation of pinned public research backends into persistent cache."""
from pathlib import Path
import modal
ROOT = Path(__file__).resolve().parent
image = (modal.Image.from_registry('lmsysorg/sglang:dev-muse-spark-open')
    .run_commands('echo /usr/local/lib/python3.12/dist-packages > /opt/sglang/lib/python3.12/site-packages/zz_system.pth', 'echo /usr/lib/python3/dist-packages >> /opt/sglang/lib/python3.12/site-packages/zz_system.pth')
    .entrypoint([])
    .add_local_file(ROOT/'muse_attention_decode/candidate_openkernels_attention.py', '/root/attention.py')
    .add_local_file(ROOT/'muse_moe_w4a16/candidate_openkernels_moe.py', '/root/moe.py'))
image = (image.add_local_file(ROOT/'muse_attention_decode/reference.py', '/root/reference.py')
    .add_local_file(ROOT/'muse_attention_decode/checks.py', '/root/checks.py')
    .add_local_file(ROOT/'muse_attention_decode/submission.py', '/root/current_attention.py'))
image = (image.add_local_file(ROOT/'muse_moe_w4a16/reference.py', '/root/moe_reference.py')
    .add_local_file(ROOT/'muse_moe_w4a16/nvfp4.py', '/root/nvfp4.py'))
image = image.add_local_file(ROOT/'muse_moe_w4a16/research_g1_strict_backend.py','/root/g1.py')
image = image.add_local_file(ROOT/'muse_moe_w4a16/research_moe_full_bf16.py','/root/full.py')
image = image.add_local_file(ROOT/'muse_attention_decode/candidate_attention_interior.py','/root/interior.py')
image = image.add_local_file(ROOT/'muse_moe_w4a16/research_moe_full_bf16_gk.py','/root/fullgk.py')
image = image.add_local_file(ROOT/'muse_prefill_layer/candidate_openkernels_prefill_bf16_gate.py','/root/prefill.py')
image = (image.add_local_file(ROOT/'muse_prefill_layer/research_prefill_full_bf16.py','/root/prefillstrict.py')
    .add_local_file(ROOT/'muse_prefill_layer/research_prefill_w13_fused.py','/root/prefillfused.py'))
image = image.add_local_dir(ROOT/'research_harness', '/root/research_harness', ignore=['results/**', '__pycache__/**'])
app = modal.App('muse-reference-cpu-compile', image=image)
cache = modal.Volume.from_name('muse-ext-cache', create_if_missing=True)
@app.function(timeout=1200, cpu=8, memory=16384, volumes={'/cache': cache})
def compile_sources(target: str = 'attention'):
    import os, importlib.util, time
    os.environ.update(TORCH_EXTENSIONS_DIR='/cache/torch_ext', TRITON_CACHE_DIR='/cache/triton', SGLANG_JIT_CACHE_DIR='/cache/sglang_jit',TVM_FFI_CACHE_DIR='/cache/tvm-ffi', TORCH_CUDA_ARCH_LIST='9.0a', MAX_JOBS='2')
    spec=importlib.util.spec_from_file_location('research_backend', '/root/'+target+'.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    # These harness-only helpers select the same H200 compilation variants without a GPU.
    mod._big=lambda: True
    if hasattr(mod,'_gencode'): mod._gencode=lambda: ['-gencode=arch=compute_90a,code=sm_90a']
    if hasattr(mod,'_sm90a'): mod._sm90a=lambda: ['-gencode=arch=compute_90a,code=sm_90a']
    start=time.monotonic()
    try:
        if target in ('attention','interior'):
            mod._ext()
            print('COMPILE_OK attention', time.monotonic()-start, flush=True)
        elif target=='fullgk':
            mod._extK('-DSW=4 -DKS2=2')
            print('COMPILE_OK full_gk',time.monotonic()-start,flush=True)
        elif target=='full':
            mod._extG('-DFENCE1 -DRT=16')
            print('COMPILE_OK full_g1',time.monotonic()-start,flush=True)
            mod._extS(0)
            print('COMPILE_OK full_small',time.monotonic()-start,flush=True)
            mod._extL(0)
            print('COMPILE_OK full_ring',time.monotonic()-start,flush=True)
        elif target=='g1':
            mod._extG('-DFENCE1 -DRT=16')
            print('COMPILE_OK strict_g1',time.monotonic()-start,flush=True)
        elif target=='moe':
            tasks=[('ring0', lambda:mod._extL(0)), ('ring1',lambda:mod._extL(1))]
            tasks += [('small'+str(k),lambda k=k:mod._extS(k)) for k in mod.V17_KSF]
            tasks += [('g1'+str(k),lambda k=k:mod._extG(k)) for k in mod.G_CFG]
            tasks += [('gk'+str(k),lambda k=k:mod._extK(k)) for k in mod.K_CFG]
            tasks += [('wx'+str(k),lambda k=k:mod._extX(k)) for k in mod.WX_CFG]
            for name, fn in tasks:
                fn(); cache.commit()
                print('COMPILE_OK',name,time.monotonic()-start,flush=True)
        else: raise ValueError(target)
    finally: cache.commit()

@app.function(timeout=300, gpu='H200', volumes={'/cache': cache})
def probe_attention(target: str = "attention", comparison: str = "current_attention"):
    import os, sys, importlib.util, time, statistics
    os.environ.update(TORCH_EXTENSIONS_DIR='/cache/torch_ext', TRITON_CACHE_DIR='/cache/triton', SGLANG_JIT_CACHE_DIR='/cache/sglang_jit',TVM_FFI_CACHE_DIR='/cache/tvm-ffi', MAX_JOBS='2')
    sys.path.insert(0, '/root')
    import torch, reference
    def load(name, path):
        spec=importlib.util.spec_from_file_location(name,path)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
    new=load('new_attention','/root/'+target+'.py');old=load('comparison_attention','/root/'+comparison+'.py')
    start=time.monotonic();new._ext();print('CACHED_LOAD_SECONDS',time.monotonic()-start,flush=True)
    tests=[dict(batch=3,ctx=2047,heads='tp8',layer='sliding',page=64),
           dict(batch=2,ctx=9000,heads='tp8',layer='full',kv_scale=1/64),
           dict(batch=5,ctx=5000,heads='tp8',layer='sliding',q_len=8,page=1,spread='mixed'),
           dict(batch=2,ctx=20000,heads='dp',layer='full',q_len=8,dist='coherent')]
    for i,cfg in enumerate(tests):
        data=reference.generate_input(seed=8121+i,**cfg)
        out=new.kernel_fn(data);torch.cuda.synchronize()
        result=reference.check_implementation(data,reference.collect(data,out))
        print('CHECK',cfg,result,flush=True)
        assert result[0], result
        del data,out
    flush=torch.empty(256*1024*1024,device='cuda',dtype=torch.uint8)
    for b in (1,32):
      for length in (1,8):
       for layer in ('sliding','full'):
        data=reference.generate_input(batch=b,ctx=80000,heads='tp8',layer=layer,seed=9901+b+length,q_len=length,page=1,dist='coherent' if layer=='full' else 'soft',spread='agentic' if b>1 else 'fixed')
        graphs={}
        for name,mod in [('new',new),('old',old)]:
            for _ in range(2):mod.kernel_fn(data)
            torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):mod.kernel_fn(data)
            graphs[name]=graph
        times={name:[] for name in graphs}
        for rep in range(20):
            for name in (['new','old'] if rep%2 else ['old','new']):
                flush.zero_();a,b_ev=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record();graphs[name].replay();b_ev.record();b_ev.synchronize()
                times[name].append(a.elapsed_time(b_ev)*1000)
        print('PAIRED_US',b,length,layer,{k:statistics.median(v) for k,v in times.items()},flush=True)
        del graphs,data
    cache.commit()

@app.function(timeout=300, gpu='H200', volumes={'/cache': cache})
def probe_moe(target: str = "moe", audit: bool = False, varied_scales: bool = False):
    import os,sys,importlib.util,time,statistics
    os.environ.update(TORCH_EXTENSIONS_DIR='/cache/torch_ext',TRITON_CACHE_DIR='/cache/triton',SGLANG_JIT_CACHE_DIR='/cache/sglang_jit',TVM_FFI_CACHE_DIR='/cache/tvm-ffi',TORCH_CUDA_ARCH_LIST='9.0a',MAX_JOBS='2')
    sys.path.insert(0,'/root')
    import torch,moe_reference as ref
    from nvfp4 import random_weights
    spec=importlib.util.spec_from_file_location('probe_backend','/root/'+target+'.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mod.TUNE=False
    gen=torch.Generator(device='cuda').manual_seed(5252)
    weights={}
    for name,n in [('w13',8192),('w2',4096)]:
        w,sf,gs=random_weights(n,4096,generator=gen,device='cuda',leading=(32,))
        weights[name],weights[name+'_sf'],weights[name+'_gs']=w,sf,gs.expand(32).contiguous()
        if varied_scales:weights[name+'_gs'] *= torch.exp(torch.randn(32,generator=gen,device='cuda'))
    weights['post_norm_w']=torch.randn(4096,generator=gen,device='cuda').to(torch.bfloat16)*0.1
    t=time.monotonic();state=mod.prepare(weights);torch.cuda.synchronize()
    print('PREPARE_SECONDS',time.monotonic()-t,flush=True)
    flush=torch.empty(256*1024*1024,device='cuda',dtype=torch.uint8)
    cases=[(4,"normal"),(33,"normal"),(257,"normal"),(128,"skewed"),(1,"normal")] if audit else [(m,"normal") for m in (1,8,16,32,128,256)]
    if varied_scales:cases=[(m,"normal") for m in (1,8,16,32,128,256)]+[(4,"normal"),(33,"normal"),(257,"normal"),(128,"skewed")]
    for m,dist in cases:
        data=ref.generate_input(m,'ep8',1917+m,dist=dist)
        expected=ref.ref_kernel(data,weights)
        configs=[mod._default(m)]
        if m==1:
            configs = [('g1',0,0,'-DFENCE1 -DRT=16')] if target.startswith('full') else configs+[('g1',0,0,'-DFENCE1 -DRT=16')]
        if not target.startswith("full") and m in (8,16):configs += [('gk',0,0,'-DSW=4 -DKS2=2'),('gk',0,0,'-DSW=4 -DKS2=4')]
        if not target.startswith("full") and m==32:configs += [('ring',1,0),('gk',0,0,'-DSW=4 -DKS1=2 -DKS2=4 -DMAXM=32')]
        if target=="fullgk" and 1 < m <=16:configs=[mod._default(m),("gk",0,0,"-DSW=4 -DKS2=2")]
        for cfg in configs:
            out=mod._run(cfg,data,state);torch.cuda.synchronize()
            result=ref.compare(out,expected,ref.REL_L2_TOL)
            print('CHECK',m,cfg,result,flush=True)
            if not result[0]:continue
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):mod._run(cfg,data,state)
            times=[]
            for _ in range(20):
                flush.zero_();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record();graph.replay();b.record();b.synchronize();times.append(a.elapsed_time(b)*1000)
            print('MOE_RAW_US',m,cfg,statistics.median(times),min(times),max(times),flush=True)
            del graph
        del data,expected
    cache.commit()

@app.function(timeout=180,gpu='H200',volumes={'/cache':cache})
def probe_g1_strict():
    import os,sys,importlib.util,statistics
    os.environ.update(TORCH_EXTENSIONS_DIR='/cache/torch_ext',TRITON_CACHE_DIR='/cache/triton',SGLANG_JIT_CACHE_DIR='/cache/sglang_jit',TVM_FFI_CACHE_DIR='/cache/tvm-ffi',TORCH_CUDA_ARCH_LIST='9.0a',MAX_JOBS='2')
    sys.path.insert(0,'/root')
    import torch,moe_reference as ref
    from nvfp4 import random_weights
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
    old=load('original_g1','/root/moe.py');new=load('strict_g1','/root/g1.py');old.TUNE=False
    gen=torch.Generator(device='cuda').manual_seed(6227);w={}
    for name,n in [('w13',8192),('w2',4096)]:
        packed,sf,gs=random_weights(n,4096,generator=gen,device='cuda',leading=(32,))
        w[name],w[name+'_sf'],w[name+'_gs']=packed,sf,gs.expand(32).contiguous()
    w['post_norm_w']=(torch.randn(4096,generator=gen,device='cuda')*.1).to(torch.bfloat16)
    state=old.prepare(w);strict=dict(state)
    strict['nw']=(1+w['post_norm_w'].float()).to(torch.bfloat16).float()
    strict['ctrG']=torch.zeros_like(state['ctrG']);strict['sspG']=torch.zeros_like(state['sspG']);strict['interG']=torch.empty_like(state['interG'])
    cfg=('g1',0,0,'-DFENCE1 -DRT=16');flush=torch.empty(256*1024*1024,device='cuda',dtype=torch.uint8)
    for seed,dist in [(712,'normal'),(883,'normal'),(921,'skewed'),(1009,'skewed')]:
        d=ref.generate_input(1,'ep8',seed,dist=dist);expected=ref.ref_kernel(d,w)
        graphs={};times={'original':[],'strict':[]}
        for name,mod,st in [('original',old,state),('strict',new,strict)]:
            out=mod._g1(d,st,cfg);torch.cuda.synchronize();result=ref.compare(out,expected,ref.REL_L2_TOL)
            print('STRICT_CHECK',seed,dist,name,result,flush=True)
            assert result[0],result
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):mod._g1(d,st,cfg)
            graphs[name]=graph
        for rep in range(30):
            for name in (['original','strict'] if rep%2 else ['strict','original']):
                flush.zero_();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record();graphs[name].replay();b.record();b.synchronize();times[name].append(a.elapsed_time(b)*1000)
        print('STRICT_G1_RAW_US',seed,dist,{k:statistics.median(v) for k,v in times.items()},flush=True)
    cache.commit()

@app.function(timeout=60)
def inspect_jit_arch():
    import inspect
    import sglang.kernels.jit.utils.arch as arch
    print(inspect.getsource(arch),flush=True)

@app.function(timeout=1200,cpu=8,memory=16384,volumes={'/cache':cache})
def compile_prefill():
    import os,sys,ast,inspect,importlib.util,time
    os.environ.update(TORCH_EXTENSIONS_DIR='/cache/torch_ext',TRITON_CACHE_DIR='/cache/triton',SGLANG_JIT_CACHE_DIR='/cache/sglang_jit',TVM_FFI_CACHE_DIR='/cache/tvm-ffi',TORCH_CUDA_ARCH_LIST='9.0a',MAX_JOBS='2')
    import torch
    from sglang.kernels.jit.utils.arch import override_jit_cuda_arch
    from sglang.kernels.ops.moe.moe_wna16_marlin import _jit_moe_wna16_marlin_module
    start=time.monotonic()
    def load(name):
        import types
        path='/root/'+name+'.py'
        with open(path) as source_file:code=source_file.read()
        # CPU compilation selects the H200-only import constant; candidate source is unchanged.
        before='MOE_SMALL_MAX = 2048 if torch.cuda.get_device_capability() == (9, 0) else 256'
        assert code.count(before)==1
        code=code.replace(before,'MOE_SMALL_MAX = 2048')
        m=types.ModuleType(name);m.__file__=path;sys.modules[name]=m
        exec(compile(code,path,'exec'),m.__dict__)
        return m
    try:
        # Use the supported architecture override; no runtime GPU query is needed to compile.
        with override_jit_cuda_arch(9,0,'a'):_jit_moe_wna16_marlin_module(torch.bfloat16,True,False)
        cache.commit();print('PREFILL_CPU_OK native_marlin',time.monotonic()-start,flush=True)
        base=load('prefillstrict');base.Q__big=lambda:True;base._qfits=lambda flags:True
        base._pf2();cache.commit();print('PREFILL_CPU_OK glue',time.monotonic()-start,flush=True)
        base.Q__ext();cache.commit();print('PREFILL_CPU_OK strict_ringQ',time.monotonic()-start,flush=True)
        for name in ('prefillstrict','prefillfused'):
            mod=base if name=='prefillstrict' else load(name)
            tree=ast.parse(inspect.getsource(mod._w4g));call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='load_inline')
            decl=ast.literal_eval(next(k.value for k in call.keywords if k.arg=='cpp_sources'))
            module_name='pf2_w4g_v12h' if name=='prefillstrict' else 'pf2_w4g_swiglu_v1h'
            mod.load_inline(module_name,cpp_sources=decl,cuda_sources=mod.W4G_CUDA,functions=['w4g','w4g_smem','w4g_err'],extra_cuda_cflags=['-O3','--use_fast_math','-lineinfo','-gencode=arch=compute_90a,code=sm_90a','-DHAS_WG_HOST=1'],verbose=False)
            cache.commit();print('PREFILL_CPU_OK',name,time.monotonic()-start,flush=True)
    finally:cache.commit()

@app.function(timeout=180,gpu='H200',volumes={'/cache':cache})
def probe_prefill_ring():
    import os,sys,importlib.util,statistics
    os.environ.update(TORCH_EXTENSIONS_DIR='/cache/torch_ext',TRITON_CACHE_DIR='/cache/triton',SGLANG_JIT_CACHE_DIR='/cache/sglang_jit',TVM_FFI_CACHE_DIR='/cache/tvm-ffi',TORCH_CUDA_ARCH_LIST='9.0a',MAX_JOBS='2')
    sys.path.insert(0,'/root')
    import torch,moe_reference as ref
    from nvfp4 import random_weights
    spec=importlib.util.spec_from_file_location('strict_prefill','/root/prefillstrict.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    gen=torch.Generator(device='cuda').manual_seed(6502);weights={}
    for name,n in [('w13',8192),('w2',4096)]:
        w,sf,gs=random_weights(n,4096,generator=gen,device='cuda',leading=(32,))
        weights[name],weights[name+'_sf'],weights[name+'_gs']=w,sf,gs.expand(32).contiguous()*torch.exp(torch.randn(32,generator=gen,device='cuda'))
    weights['post_norm_w']=(torch.randn(4096,generator=gen,device='cuda')*.1).to(torch.bfloat16)
    state=mod.Q_prepare(weights);flush=torch.empty(256*1024*1024,device='cuda',dtype=torch.uint8)
    for m,dist in [(200,'normal'),(753,'normal'),(753,'skewed'),(3012,'normal')]:
        data=ref.generate_input(m,'ep8',7131+m,dist=dist);expected=ref.ref_kernel(data,weights)
        out=mod.Q_kernel_fn(data,state);torch.cuda.synchronize();result=ref.compare(out,expected,ref.REL_L2_TOL)
        print('PREFILL_Q_CHECK',m,dist,result,flush=True);assert result[0],result
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):mod.Q_kernel_fn(data,state)
        times=[]
        for _ in range(20):
            flush.zero_();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();graph.replay();b.record();b.synchronize();times.append(a.elapsed_time(b)*1000)
        print('PREFILL_Q_US',m,dist,statistics.median(times),flush=True)
        del graph,data,expected
    cache.commit()

@app.function(timeout=240,gpu='H200',volumes={'/cache':cache})
def probe_w13_fused(seed: int = 7171, modes: str = '0,1', row_counts: str = '1024,3008,16384', skewed: bool = False):
    import os,sys,importlib.util,statistics
    os.environ.update(TORCH_EXTENSIONS_DIR='/cache/torch_ext',TRITON_CACHE_DIR='/cache/triton',SGLANG_JIT_CACHE_DIR='/cache/sglang_jit',TVM_FFI_CACHE_DIR='/cache/tvm-ffi',TORCH_CUDA_ARCH_LIST='9.0a',MAX_JOBS='2')
    sys.path.insert(0,'/root')
    import torch
    from nvfp4 import random_weights
    from checks import compare
    from research_harness.gpu import compare_graphs
    def load(name):
        spec=importlib.util.spec_from_file_location(name,'/root/'+name+'.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
    old,new=load('prefillstrict'),load('prefillfused');ext0,ext1=old._w4g(),new._w4g();glue=old._pf2()
    assert ext0 is not None and ext1 is not None
    gen=torch.Generator(device='cuda').manual_seed(seed)
    raw,sf,gs=random_weights(8192,4096,generator=gen,device='cuda',leading=(32,));gs=gs.expand(32).contiguous()*torch.exp(torch.randn(32,generator=gen,device='cuda'))
    perm=torch.stack((torch.arange(4096,device='cuda').view(-1,8),4096+torch.arange(4096,device='cuda').view(-1,8)),1).reshape(-1)
    wp0,g0=old._w4pack(raw,sf,gs);wp1,g1=new._w4pack(raw.index_select(1,perm),sf.index_select(1,perm),gs)
    flush=torch.empty(256*1024*1024,device='cuda',dtype=torch.uint8)
    for rows in map(int,row_counts.split(',')):
        if skewed:
            # Deterministic long tail, including idle experts; no host input-derived work in kernel.
            popularity=[64,32,16,8,4,2,1,1]+[0]*24
            counts=[rows*p//sum(popularity) for p in popularity]
            counts[0]+=rows-sum(counts)
        else:
            counts=[rows//32+(1 if e<rows%32 else 0) for e in range(32)]
        offsets=[0]
        for count in counts:offsets.append(offsets[-1]+count)
        # w4g and act2 consume 32 inclusive cumulative ends, not 33 CSR offsets.
        offs=torch.tensor(offsets[1:],device='cuda',dtype=torch.int32);a=torch.randn(rows,4096,generator=gen,device='cuda').to(torch.bfloat16)
        ctr0=torch.zeros(24,device='cuda',dtype=torch.int32);ctr1=torch.zeros_like(ctr0)
        for wide in map(int,modes.split(',')):
            def baseline():
                h=old._w4gemm(a,offs,wp0,g0,ctr0,ext=ext0,wide=wide)
                out=torch.empty(rows,4096,device='cuda',dtype=torch.bfloat16);glue.act2(h,offs,out);return out
            def fused():return new._w4gemm(a,offs,wp1,g1,ctr1,ext=ext1,wide=wide,fuse=True)
            expected,actual=baseline(),fused();torch.cuda.synchronize()
            assert torch.isfinite(expected).all().item(), 'Baseline contains NaN/Inf'
            assert torch.isfinite(actual).all().item(), 'Candidate contains NaN/Inf'
            result=compare(actual,expected,0.001)
            print('W13_FUSED_CHECK',rows,wide,result,flush=True);assert result[0],result
            report=compare_graphs(baseline,fused,repetitions=40)
            import json
            print('W13_PAIRED_JSON',json.dumps(dict(seed=seed,rows=rows,wide=wide,skewed=skewed,**report)),flush=True)
            del actual,expected
    cache.commit()
