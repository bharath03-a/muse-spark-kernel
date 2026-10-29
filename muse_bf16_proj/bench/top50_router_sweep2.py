"""Distributed split-K router; inspiration lee101/openkernels wgr.
Cooperative launch required. Separate scratch per kernel, serialized graphs only.
"""
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
from top50_qkv_mma import graph, paired
from common import load_module

@triton.jit
def distributed(X,H,L,O,W,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,S:tl.constexpr):
    pid=tl.program_id(0)
    nm:tl.constexpr=triton.cdiv(M,BM)
    mi=pid%nm
    si=(pid//nm)%S
    ni=pid//(nm*S)
    tid=ni*nm+mi
    # Read epoch before this CTA publishes; previous replays have completed.
    epoch=tl.atomic_add(C+tid,0,sem='acquire',scope='gpu')//S+1
    acc=tl.zeros((BM,BN),tl.float32)
    for ki in range(si*(K//BK)//S,(si+1)*(K//BK)//S):
        x=X.load([mi*BM,ki*BK])
        h=H.load([ki*BK,ni*BN])
        l=L.load([ki*BK,ni*BN])
        acc=tl.dot(x,h,acc)
        acc=tl.dot(x,l,acc)
    E:tl.constexpr=BM*BN
    off=tl.arange(0,BM)[:,None]*BN+tl.arange(0,BN)[None,:]
    base=W+tid*S*E
    tl.store(base+si*E+off,acc)
    tl.debug_barrier()
    tl.atomic_add(C+tid,1,sem='acq_rel',scope='gpu')
    while tl.atomic_add(C+tid,0,sem='acquire',scope='gpu')<epoch*S:
        pass
    ids=si*(E//S)+tl.arange(0,E//S)
    total=tl.full((E//S,),0,tl.float32)
    for z in tl.static_range(S):
        total+=tl.load(base+z*E+ids,cache_modifier='.cg')
    rows=mi*BM+ids//BN
    cols=ni*BN+ids%BN
    tl.store(O+rows*N+cols,total,rows<M)


def main():
    torch.manual_seed(2026)
    w=torch.randn(256,8192,device='cuda')/8192**.5
    hi=w.bfloat16();lo=(w-hi.float()).bfloat16()
    ht=hi.T.contiguous();lt=lo.T.contiguous()
    flush=torch.empty(256*2**20//4,device='cuda')
    C=load_module('checkpoints/cp14_D_slice_check1.426.py','cp14_dist_ref')
    ext=C._build_cuda()
    state={'ext':ext};hl=torch.stack([hi,lo]).contiguous()
    for m,bm,bn,s,bk in [(128,128,32,32,64),(256,128,64,32,64),(256,64,64,16,64)]:
        x=torch.randn(m,8192,device='cuda',dtype=torch.bfloat16)
        x[::8]=0;x[::8,31]=2
        out=torch.empty(m,256,device='cuda')
        ws=torch.empty((256//bn)*(m//bm)*s*bm*bn,device='cuda')
        cnt=torch.zeros((256//bn)*(m//bm),device='cuda',dtype=torch.int64)
        xd=TensorDescriptor.from_tensor(x,[bm,bk]);hd=TensorDescriptor.from_tensor(ht,[bk,bn]);ld=TensorDescriptor.from_tensor(lt,[bk,bn])
        def trial():
            distributed[((256//bn)*triton.cdiv(m,bm)*s,)](xd,hd,ld,out,ws,cnt,m,256,8192,bm,bn,bk,s,num_warps=4,num_stages=3,launch_cooperative_grid=True)
            return out
        kind,params=C._cuda_pick(state,'router',m)
        def baseline(): return C._cuda_run(state,x,hl,256,torch.float32,kind,params)
        try:
            ref=x.double()@w.double().T
            for i in range(5):
                y=trial();err=((y.double()-ref).norm()/ref.norm()).item()
                assert err<1e-4,err
            gs={key:graph(fn) for key,fn in [('distributed',trial),('cp14',baseline)]}
            print('DISTRIBUTED',m,'config',(bm,bn,s,bk),'rel',err,'raw_us',paired({k:v[0] for k,v in gs.items()},flush,30),flush=True)
        except Exception as e:
            print('DISTRIBUTED_FAIL',m,type(e).__name__,str(e)[-1500:],flush=True)
            raise
if __name__=='__main__': main()
