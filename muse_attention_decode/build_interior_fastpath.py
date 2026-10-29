"""Create an exact-copy interior-block fast path; sentinel/tail path remains unchanged."""
import ast
from pathlib import Path
root=Path(__file__).resolve().parent
s=(root/'candidate_openkernels_attention.py').read_text()
t=ast.parse(s);node=next(n for n in t.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='CUDA' for x in n.targets));c=ast.literal_eval(node.value)
old='auto issue = [&](int slot, int rl, bool tabok) {  // the new token\'s row comes from the table once it is written'
assert c.count(old)==1
c=c.replace(old,'auto issue = [&](int slot, int rl, bool tabok, bool interior) {  // exact full-valid blocks bypass sentinel handling')
needle='#ifdef FD_NOLOAD\n    asm volatile("cp.async.commit_group;\\n"); return;\n#endif\n'
assert c.count(needle)==1
fast='''    if (interior) {
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int r = i * 4 + (lane >> 3), rw = __shfl_sync(0xffffffffu, rl, r);
        const uint32_t dd = d0 + r * 128 + ((c ^ (r & 7)) << 4);
        const size_t off = (size_t)rw * 128 + c * 16;
        asm volatile("cp.async.cg.shared.global" FDH " [%0], [%1], 16;\\n" :: "r"(dd), "l"(p.kc + off));
        asm volatile("cp.async.cg.shared.global" FDH " [%0], [%1], 16;\\n" :: "r"(dd + 2048), "l"(p.vc + off));
      }
      asm volatile("cp.async.commit_group;\\n");
      return;
    }
'''
c=c.replace(needle,needle+fast)
a='issue(i, rpro[i], false);';b='issue((bi + RBF - 1) % RBF, rnext, true);'
assert c.count(a)==1 and c.count(b)==1
c=c.replace(a,'issue(i, rpro[i], false, ws + (i + 1) * 16 <= min(we, seq));').replace(b,'issue((bi + RBF - 1) % RBF, rnext, true, kb + RBF * 16 <= min(we, seq));')
lines=s.splitlines(True);s=''.join(lines[:node.lineno-1])+'CUDA = '+repr(c)+'\n'+''.join(lines[node.end_lineno:])
s=s.replace('"attn_v60_%d"','"attn_v60_interior_%d"')
out=root/'candidate_attention_interior.py';out.write_text(s);compile(s,str(out),'exec');print(out)
