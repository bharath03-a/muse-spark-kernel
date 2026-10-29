"""CPU exact bit proof for v56's lossless eight-code 32bit packing/decode.
Bit-permutation paths checked on every basis vector. Multiply path exhaustive over
its eight input bits (all256possibilities), so covers all2**32packed words.
Scale arithmetic follows the independently exhaustive prove_unpack.py check.
"""
import ast
from pathlib import Path
s=(Path(__file__).resolve().parents[1]/'research_moe_v56_strict.py').read_text()
t=ast.parse(s);ns={}
for n in t.body:
    if isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='_ROT' for v in n.targets):ns['_ROT']=ast.literal_eval(n.value)
    if isinstance(n,ast.FunctionDef) and n.name=='_place':exec(compile(ast.Module(body=[n],type_ignores=[]),'<place>','exec'),ns)
place=ns['_place'];mask=0x81C081C0
rot=lambda q,n:((q<<n)|(q>>(32-n)))&0xffffffff

def output(q,pair):
    if pair==0:return q&mask
    if pair in (1,2):return rot(q,ns['_ROT'][pair][0])&mask
    return (((q&0x001B001B)*0x1084)&0xffffffff)&mask

def wanted(lo,hi):return ((lo&7)<<6)|((lo&8)<<12)|((hi&7)<<22)|((hi&8)<<28)
positions=[]
for pair in range(4):
    for hi in range(2):
        for bit in range(4):
            q=place(0,1<<bit,pair,hi)
            assert q and q&(q-1)==0
            positions.append(q.bit_length()-1)
            for outpair in range(4):
                expected=wanted((1<<bit) if hi==0 else 0,(1<<bit) if hi==1 else 0) if outpair==pair else 0
                assert output(q,outpair)==expected,(pair,hi,bit,outpair)
assert sorted(positions)==list(range(32))
for lo in range(16):
    for hi in range(16):
        q=place(place(0,lo,3,0),hi,3,1)
        assert q&~0x001B001B==0
        assert output(q,3)==wanted(lo,hi),(lo,hi)
print('PASS packing is bijective over32bits; linear decoders exact by32basisvectors; multiply decoder exact over all256masked inputs. Covers all2**32words.')
