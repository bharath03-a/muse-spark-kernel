"""Move strict BF16 materialization to completed W2 writers, saving one CTA pass.
No arithmetic-order change: KS1 full-K callback and KS2 completed sum only.
"""
from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'research_moe_v56_strict.py';source=p.read_text()
node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CUDA_K' for t in n.targets))
c=ast.literal_eval(node.value)
def edit(old,new):
 global c
 assert c.count(old)==1,(old[:100],c.count(old));c=c.replace(old,new)
edit('if (slot < T) { float* y = ybl + (ao + slot) * 64 + row - sm->o0; y[0] = h ? v.y : v.x; y[8] = h ? v.w : v.z; }',
'''if (slot < T) { float* y = ybl + (ao + slot) * 64 + row - sm->o0;
              // KS2=1: slice has completed all K and the SW reduction.
              const float gs=sm->gs2[a];
              y[0]=__bfloat162float(__float2bfloat16((h ? v.y : v.x)*gs));
              y[8]=__bfloat162float(__float2bfloat16((h ? v.w : v.z)*gs)); }''')
edit('if (i2 < nr * 64) yb[i2] = t;',
'''if (i2 < nr * 64) {
          // KS2>1: t contains the same ordered sum of every FP32 K partial.
          const int r=i2>>6, rr=i2&63;
          yb[i2]=rr<sm->onr ? __bfloat162float(__float2bfloat16(t*sm->gs2[sm->rexp[r]])) : t;
        }''')
edit('''    // Complete the W2 reduction and BF16 materialization before norm weighting.
    for (int i = tid; i < nr * 64; i += NW * 32) {
      const int r = i >> 6, rr = i & 63;
      if (rr < sm->onr) yb[i] = __bfloat162float(__float2bfloat16(yb[i] * sm->gs2[sm->rexp[r]]));
    }
    __syncthreads();
''','''    // BF16 down outputs were materialized by their complete-reduction writers.
    // The preceding CTA barrier remains: norm warps read other writers' rows.
''')
text=source.replace(ast.get_source_segment(source,node.value),repr(c),1)
assert text.count('"moe_v56strict_v56K"')==1
text=text.replace('"moe_v56strict_v56K"','"moe_v56strict_roundstoreK"')
out=ROOT/'research_moe_v56_round_store.py';out.write_text(text);compile(text,str(out),'exec')
(ROOT/'research/kernel_sources/gk_round_store.cu').write_text(c)
print(out)
