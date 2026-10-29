# Appended only to the external diagnostic. Ordinary kernel_fn is never replaced.
_RD_mods={}
_RD_K_FLAGS=('-DSW=4 -DKS2=1 -DMAXM=8','-DSW=4 -DKS2=2')
_RD_X_FLAGS=('-DNCW=16','-DWXG2','-DUDESC')
def _RD_build(kind,flags):
    from torch.utils.cpp_extension import load_inline
    fl=flags.split() if isinstance(flags,str) else list(flags)
    key=(kind,tuple(fl))
    module=load_inline('whole_route_phase_v2_'+kind+''.join(fl).replace('-','_').replace('=',''),
        cpp_sources=_RD_CPPK if kind=='K' else _RD_CPPX,
        cuda_sources=_RD_K if kind=='K' else _RD_X,
        functions=['moe','pdl_set','rd_resources'] if kind=='K' else ['moe','rd_resources'],
        extra_cuda_cflags=['-O3','--use_fast_math','-lineinfo','-DPROF']+(M__sm90a() if kind=='X' else [])+fl,
        verbose=False)
    if kind=='K':module.pdl_set(0)
    _RD_mods[key]=module
    return module

def _RD_extK(flags=''):
    return _RD_mods[('K',tuple(flags.replace('+',' ').split()))]
def _RD_extX(flags=()):
    return _RD_mods[('X',tuple(flags))]

def _RD_q(v):
    return [round(float(v.double().quantile(q))/1e3,3) for q in (.5,.95)]

def _RD_phase(st,deadline):
    import time
    ms=st['moe'];dev=ms['w1'].device
    G=torch.cuda.get_device_properties(dev).multi_processor_count
    rows=[]
    flush=torch.ones(64<<20,device=dev);sink=torch.empty((),device=dev)
    def cold(fn):
        sink.copy_(flush.sum());return fn()
    for T in (1,8,16,32,128,256):
        for seed in (1837,2953):
            if time.monotonic()>deadline:return rows
            data=M__synth(T,seed+T,dev)
            lid=data['topk_ids'];hist=torch.bincount(lid[(lid>=0)&(lid<32)].long(),minlength=32)
            counts=[int((hist>0).sum()),int(hist.sum()),int(hist.max())]
            isK=T<=16
            flags=_RD_K_FLAGS[0 if T<=8 else 1] if isK else _RD_X_FLAGS
            prof=torch.zeros(G*64 if isK else G*17*32,device=dev,dtype=torch.int64)
            ds=dict(ms)
            if isK:
                cfg=('gk',0,0,flags)
                a=lambda:M__gk(data,ms,cfg)
                b=lambda:_RD_gk(data,ds,cfg,prof)
            else:
                ds['profW']=prof
                a=lambda:M__wx(data,ms,flags)
                b=lambda:_RD_wx(data,ds,flags)
            # Fixed AB/BA across two fresh seeds. No retry-until-match.
            if seed==1837:oa=cold(a);ob=cold(b)
            else:ob=cold(b);oa=cold(a)
            torch.cuda.synchronize()
            finite=bool(torch.isfinite(oa).all()) and bool(torch.isfinite(ob).all())
            same=finite and torch.equal(oa,ob)
            row=[T,seed,'K' if isK else 'X' if T!=256 else 'X256nonN8',int(same)]+counts
            if not same:
                # An atomic reduction may naturally reorder. Do not claim identical phases.
                again=a();torch.cuda.synchronize()
                row+=['skip',int(torch.equal(oa,again))];rows.append(row);continue
            prof.zero_()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):out=b()
            sink.copy_(flush.sum());graph.replay();torch.cuda.synchronize()
            if not bool(torch.isfinite(out).all()) or not torch.equal(oa,out):
                rows.append(row+['graph_skip']);continue
            cpu=prof.cpu()
            if isK:
                p=cpu[:G*32].view(G,32).double()
                w=cpu[G*32:].view(G,16,2).double()[:,:((T*8+31)//32)]
                ownership=(w[:,:,1].amax(1)-w[:,:,0].amin(1)).clamp(min=0)
                row += [_RD_q((p[:,14]-p[:,13]).clamp(min=0)),_RD_q(ownership),
                        _RD_q((p[:,15]-p[:,14]).clamp(min=0)),
                        _RD_q((p[:,16]-p[:,15]).clamp(min=0)),
                        _RD_q((p[:,19]-p[:,18]).clamp(min=0))]
            else:
                p=cpu.view(G,17,32)[:,0,:].double()
                row += [_RD_q((p[:,14]-p[:,13]).clamp(min=0)),
                        _RD_q((p[:,15]-p[:,14]).clamp(min=0)),
                        _RD_q((p[:,16]-p[:,15]).clamp(min=0)),
                        _RD_q((p[:,17]-p[:,16]).clamp(min=0))]
            row.append(round(float((p[:,21]-p[:,20]).double().median()),1))
            rows.append(row)
    return rows

def _RD_overlap(st,deadline):
    import time
    rows=[];dev=st['pooler'].device
    gen=torch.Generator(device=dev).manual_seed(81173)
    flush=torch.ones(64<<20,device=dev);sink=torch.empty((),device=dev)
    # Separate events share one fork timestamp; actual operations match the whole fork/join.
    for T in (1,8,16,32,128,256):
        if time.monotonic()>deadline:break
        h=torch.randn((T,8192),generator=gen,device=dev).bfloat16()
        lg=torch.empty((T,256),device=dev,dtype=torch.float32)
        ids=torch.empty((T,8),device=dev,dtype=torch.int32);loc=torch.empty_like(ids)
        tw=torch.empty((T,8),device=dev,dtype=torch.float32)
        p=torch.empty((T,4096),device=dev,dtype=torch.bfloat16)
        ls=torch.zeros(1,device=dev,dtype=torch.int32)
        stream=torch.cuda.Stream();side=st['side'];g=_gext()
        stream.wait_stream(torch.cuda.current_stream())
        # Preserve exact small-T policy rather than inheriting the last prepare bucket.
        pol=_pdlpol(T)
        g.pdl_set(int(pol['glue']));P__gvext().pdl_set(int(pol['gv']));P__ext().pdl_set(int(pol['rtr']))
        if P__wbext():P__wbext().pdl_set(int(pol['gv']))
        ev=[torch.cuda.Event(enable_timing=True,external=True) for _ in range(3)]
        def run():
            cur=torch.cuda.current_stream();ev[0].record(cur);side.wait_stream(cur)
            with torch.cuda.stream(side):
                _router(h,lg,st);g.routing(lg,st['router_bias'],ls,ids,loc,tw);ev[2].record(side)
            pr=_proj(h,st,'pooler',4096);g.rmsgain(pr,st['g_pre_expert_norm'],p);ev[1].record(cur)
            cur.wait_stream(side)
        with torch.cuda.stream(stream):
            run();stream.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):run()
            values=[]
            for _ in range(3):
                sink.copy_(flush.sum())
                h.copy_(torch.randn(h.shape,generator=gen,device=dev).bfloat16())
                graph.replay();stream.synchronize()
                assert bool(torch.isfinite(p).all()) and bool(torch.isfinite(tw).all())
                P=ev[0].elapsed_time(ev[1])*1e3;R=ev[0].elapsed_time(ev[2])*1e3
                values.append((P,R,max(0.,P-R)))
        torch.cuda.current_stream().wait_stream(stream)
        rows.append([T]+[round(float(torch.tensor([v[i] for v in values]).median()),3) for i in range(3)])
    return rows

_RD_original_prepare=prepare
def prepare(weights):
    import concurrent.futures,time,json
    start=time.monotonic()
    pool=concurrent.futures.ThreadPoolExecutor(3)
    futures=[pool.submit(_RD_build,'K',f) for f in _RD_K_FLAGS]+[pool.submit(_RD_build,'X',_RD_X_FLAGS)]
    result={'phase':[],'overlap':[],'units':'us','identity_gate':'fixed AB/BA and graph bitidentity; skip failures',
            'Kcols':['histPrefix','ownership+GS','partition','ringPrefetch','input+barrier'],
            'Xcols':['stableRank','prefix+sort','partition','listScatter'],
            'quantiles':[50,95],'rowSuffix':'stampFloorNs','rowPrefix':['T','seed','kind','eq','experts','routes','hot'],'overlapCols':['T','mainReady','sideReady','hiddenSlack']}
    try:
        state=_RD_original_prepare(weights)
        # Do not leave compiler work running during normal timed calls.
        for future in futures:future.result(timeout=max(.01,start+270-time.monotonic()))
        pool.shutdown(wait=True)
        result['resources']=[[key[0]+('1' if '-DKS2=1' in key[1] else '2' if '-DKS2=2' in key[1] else ''),int(mod.rd_resources())>>32,int(mod.rd_resources())&0xffffffff] for key,mod in _RD_mods.items()]
        if time.monotonic()<start+255:result['phase']=_RD_phase(state,start+270)
        if time.monotonic()<start+275:result['overlap']=_RD_overlap(state,start+285)
        # Restore the ordinary whole policy and fence all diagnostic work before return.
        pol=_pdlpol(1);A__ext().pdl_set(int(pol['attn']));_gext().pdl_set(int(pol['glue']))
        P__gvext().pdl_set(int(pol['gv']));P__ext().pdl_set(int(pol['rtr']))
        if P__wbext():P__wbext().pdl_set(int(pol['gv']))
        torch.cuda.synchronize()
        result['prepareSeconds']=round(time.monotonic()-start,2)
        encoded=json.dumps(result,separators=(',',':'))
        assert len(encoded.encode())<2780,len(encoded)
        print('ROUTE_PHASE_END '+encoded,flush=True)
        return state
    except Exception:
        pool.shutdown(wait=True)
        raise
