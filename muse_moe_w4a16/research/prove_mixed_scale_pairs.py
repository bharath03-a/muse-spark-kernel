"""Isolated exact CPU layout proof, not a CUDA/MMA fragment implementation.
Interleave two original 16-value NVFP4 blocks into BF16x2 register pairs.
Each of the 16 output pairs shares the same (scaleA, scaleB) pair.
Weights and activations MUST receive the same permutation. Stored FP4 codes
and FP8 scale bytes are unchanged; only address/lane organization changes.
"""
from fractions import Fraction
from random import Random
GRID=[Fraction(x,2) for x in [0,1,2,3,4,6,8,12]]
def scale(byte):
    e,m=byte>>3,byte&7
    assert 1<=e<=15 and byte<=0x7e
    return Fraction(2)**(e-7)*(1+Fraction(m,8))
def val(code):return (-1 if code&8 else 1)*GRID[code&7]
perm=[j for i in range(16) for j in [i,i+16]]
assert sorted(perm)==list(range(32))
rng=Random(19027)
for trial in range(256):
    codes=[rng.randrange(16) for _ in range(32)]
    sf=[rng.randrange(0x08,0x7f) for _ in range(2)]
    x=[Fraction(rng.randrange(-128,129),64) for _ in range(32)]
    w=[val(codes[k])*scale(sf[k//16]) for k in range(32)]
    # One reusable scale pair, as opposed to separate replicated A/A and B/B pairs.
    scale_pair=(scale(sf[0]),scale(sf[1]))
    wp=[val(codes[perm[k]])*scale_pair[k%2] for k in range(32)]
    xp=[x[k] for k in perm]
    assert wp==[w[k] for k in perm]
    assert sum(a*b for a,b in zip(w,x))==sum(a*b for a,b in zip(wp,xp))
# Adversarial counterexample establishes that weights-only permutation is wrong.
x=list(map(Fraction,range(32)));w=list(map(Fraction,range(32)))
assert sum(a*b for a,b in zip(w,x))!=sum(w[k]*x[i] for i,k in enumerate(perm))
print('PASS 256 exact rational dot products; paired-scale reuse and shared K permutation')
print('NOT PROVEN: mma.sync fragment register/lane mapping, repack implementation, instruction count or GPU speed')
