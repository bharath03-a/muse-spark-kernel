"""Lossless W4G-only eight-op code placement. Keep ring/native banks untouched.

Derived from Apache-2.0 upstream 47103fd/e6b48c8, copyright2026 Lee Penkman.
Full license: ../OPENKERNELS_LICENSE.txt. Never edits the promoted submission.
"""
import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'candidate_prefill_grid_constant.py'
s=base.read_text();tree=ast.parse(s)
node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(a,ast.Name) and a.id=='W4G_CUDA' for a in n.targets))
c=ast.literal_eval(node.value)
replacements={
 'bmul(((qv << 3) & 0x01C001C0u) | ((qv << 1) & 0x80008000u), sB)':
 'bmul(__funnelshift_l(qv, qv, 10) & 0x81C081C0u, sB)',
 'bmul((qv << 6) & 0x81C081C0u, sA)':
 'bmul(__funnelshift_l(qv, qv, 13) & 0x81C081C0u, sA)',
 'bmul(((qv >> 5) & 0x01C001C0u) | ((qv << 5) & 0x80008000u), sB)':
 'bmul(((qv & 0x001B001Bu) * 0x1084u) & 0x81C081C0u, sB)',
}
for old,new in replacements.items():
 assert c.count(old)==1,old
 c=c.replace(old,new)
lines=s.splitlines(keepends=True);s=''.join(lines[:node.lineno-1])+'W4G_CUDA = '+repr(c)+'\n'+''.join(lines[node.end_lineno:])
# Only W4G packing changes; Q ring preparation retains its native code layout.
tree=ast.parse(s);pack=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_w4pack')
lines=s.splitlines(keepends=True);part=''.join(lines[pack.lineno-1:pack.end_lineno]);assert part.count('Q__place(')==1
part=part.replace('Q__place(', '_w4_unpack8_place(')
helper='''
# W4G private lossless bit placement; A/B/A/B scale ownership is unchanged.
_W4_UNPACK8_ROT = ((0,0,0,0),(10,10,10,10),(13,13,13,13),(2,7,7,12))
def _w4_unpack8_place(word, code, pair, hi):
    for bit, target, delta in zip(range(4),(6,7,8,15),_W4_UNPACK8_ROT[pair]):
        word |= ((code >> bit) & 1) << ((target + 16*hi - delta) % 32)
    return word

'''
s=''.join(lines[:pack.lineno-1])+helper+part+''.join(lines[pack.end_lineno:])
assert s.count('pf2_w4g_gridp_v1')==1
s=s.replace('pf2_w4g_gridp_v1','pf2_w4g_unpack8_v1')
out=ROOT/'candidate_prefill_unpack8.py';compile(s,str(out),'exec');out.write_text(s)
print(out)
