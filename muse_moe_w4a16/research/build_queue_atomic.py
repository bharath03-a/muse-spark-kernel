"""Build a queue-allocation-only ablation; preserve baseline numerical source."""
import ast
from pathlib import Path
root=Path(__file__).resolve().parents[1]
base=root/'candidate_moe_full_strict_gk.py'
text=base.read_text()
helper=r'''
// Only work-ID allocation: no producer data is published by this counter.
// Match atomicAdd's relaxed device atomic semantics, without compiler aggregation.
__device__ __forceinline__ int muse_queue_next(int* ptr) {
  unsigned old;
  asm volatile("atom.add.relaxed.gpu.global.u32 %0, [%1], 1;\n"
               : "=r"(old) : "l"(ptr) : "memory");
  return (int)old;
}
'''
counts={}
for node in ast.parse(text).body:
 if not isinstance(node,ast.Assign) or not isinstance(node.value,ast.Constant):continue
 name=getattr(node.targets[0],'id','')
 if name not in ('CUDA','CUDA_L'):continue
 source=node.value.value
 counter='QI' if name=='CUDA' else 'C_Q'
 target=f'atomicAdd(p.ctr + {counter}, 1)'
 count=source.count(target)
 assert count==(3 if name=='CUDA' else 5),(name,count)
 revised=source.replace(target,f'muse_queue_next(p.ctr + {counter})')
 assert revised.count('#include <stdint.h>')==1
 revised=revised.replace('#include <stdint.h>','#include <stdint.h>\n'+helper)
 # Verify no arithmetic, precision boundary or other counter was changed.
 undone=revised.replace('#include <stdint.h>\n'+helper,'#include <stdint.h>')
 undone=undone.replace(f'muse_queue_next(p.ctr + {counter})',target)
 assert undone==source
 text=text.replace(ast.get_source_segment(text,node.value),repr(revised),1)
 counts[name]=count
 (root/'research'/'kernel_sources'/f'queue_{name}.cu').write_text(revised)
text=text.replace('moe_fullbf16_v30', 'moe_queueatomic_v30')
out=root/'research_moe_queue_atomic.py'
out.write_text(text)
ast.parse(text)
print(out,counts)
