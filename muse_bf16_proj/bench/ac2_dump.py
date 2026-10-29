import triton
from triton.backends.compiler import GPUTarget
from triton.experimental.gluon._runtime import GluonASTSource
from adaptive_record_async_kernel import DECODE_ASM, DECODE_CONSTRAINTS
from ac2_kernel import _ac2_kernel
fn=_ac2_kernel;names=list(fn.arg_names)
ptrs={'payload':'*u32','metadata':'*u32','x':'*bf16','out':'*bf16','ws':'*fp32','cnt':'*i32'}
for lab,RW in (('rw1',1),('rw2',2)):
    consts=dict(N=4096,K=8192,M=32,SCALE=2.0**-4,DEPTH=3,ASM=DECODE_ASM,CONSTRAINTS=DECODE_CONSTRAINTS,RW=RW,SPLIT=2)
    src=GluonASTSource(fn=fn,signature={nm:ptrs.get(nm,'constexpr') for nm in names},constexprs={(names.index(nm),):v for nm,v in consts.items()},attrs={(names.index(nm),):[['tt.divisibility',16]] for nm in ptrs})
    k=triton.compile(src,target=GPUTarget('cuda',90,32),options={'num_warps':4*RW})
    ir=k.asm['ttgir']
    print('=====',lab,'TTGIR');print(ir)
