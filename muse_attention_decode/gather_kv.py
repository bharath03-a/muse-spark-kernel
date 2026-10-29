import triton
import triton.language as tl
@triton.jit
def gather_kv(K,V,P,KO,VO,N:tl.constexpr,ROW:tl.constexpr=16,WIDTH:tl.constexpr=128):
 r=tl.program_id(0)*ROW+tl.arange(0,ROW)
 d=tl.arange(0,WIDTH)
 idx=tl.load(P+r,r<N,other=0)
 k=tl.load(K+idx[:,None]*WIDTH+d[None,:],r[:,None]<N,other=0)
 v=tl.load(V+idx[:,None]*WIDTH+d[None,:],r[:,None]<N,other=0)
 tl.store(KO+r[:,None]*WIDTH+d[None,:],k,r[:,None]<N)
 tl.store(VO+r[:,None]*WIDTH+d[None,:],v,r[:,None]<N)
