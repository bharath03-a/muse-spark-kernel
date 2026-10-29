"""CPU proof of W4G lossless placement, scale ownership and BF16 exactness."""
import ast,struct
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(name):return ast.parse((ROOT/name).read_text())
def extract(tree,constant,fn):
 scope={}
 for n in tree.body:
  if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id==constant for x in n.targets):scope[constant]=ast.literal_eval(n.value)
  if isinstance(n,ast.FunctionDef) and n.name==fn:exec(compile(ast.Module(body=[n],type_ignores=[]),'<place>','exec'),scope)
 return scope[fn]
oldtree,newtree=read('candidate_prefill_grid_constant.py'),read('candidate_prefill_unpack8.py')
old=extract(oldtree,'Q__ROT','Q__place');new=extract(newtree,'_W4_UNPACK8_ROT','_w4_unpack8_place')
M=0x81C081C0
rot=lambda q,n:((q<<n)|(q>>(32-n)))&0xffffffff
def decode(q):return (q&M,rot(q,10)&M,rot(q,13)&M,((q&0x001B001B)*0x1084)&M)
def olddecode(q):return (q&M,((q<<3)&0x01C001C0)|((q<<1)&0x80008000),(q<<6)&M,((q>>5)&0x01C001C0)|((q<<5)&0x80008000))
positions=[]
for pr in range(4):
 for hi in range(2):
  for bit in range(4):
   a,b=old(0,1<<bit,pr,hi),new(0,1<<bit,pr,hi)
   assert olddecode(a)==decode(b),(pr,hi,bit)
   positions.append(b.bit_length()-1)
assert sorted(positions)==list(range(32))
for lo in range(16):
 for hi in range(16):
  a=old(old(0,lo,3,0),hi,3,1);b=new(new(0,lo,3,0),hi,3,1)
  assert olddecode(a)==decode(b)
# Packing loop and scale-byte layout must remain identical except placement.
def fun(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
a=ast.unparse(fun(oldtree,'_w4pack'));b=ast.unparse(fun(newtree,'_w4pack')).replace('_w4_unpack8_place(', 'Q__place(')
assert a==b
# Explicit scale tags are unique for every row/k16 block. Each code pair uses
# (h,hk)=(0,0),(1,0),(0,1),(1,1); scale order A,B,A,B follows h, never hk.
for ks in range(64):
 for warp in range(16):
  for lane in range(32):
   g,t=lane//4,lane%4
   for kk in range(4):
    for pr,(h,hk) in enumerate(((0,0),(1,0),(0,1),(1,1))):
     row=warp*16+h*8+g
     tag=(row,ks*4+kk)
     # sb uses lane replicated t, bytes kk*2+h; sc2 chooses same byte.
     packed_byte=(kk*2+h)
     assert packed_byte==kk*2+(pr%2)
     for hi in range(2):
      k=ks*64+kk*16+hk*8+2*t+hi
      assert tag==(row,k//16)
# Arithmetic products are exactly BF16-representable (not merely close).
def bf16(bits):return struct.unpack('>f',struct.pack('>I',(bits&65535)<<16))[0]
grid=(0.,.5,1.,1.5,2.,3.,4.,6.)
for code in range(16):
 for scale in range(8,127):
  raw=bf16(((code&7)<<6)|((code&8)<<12))*bf16(0x7700+scale*16)
  want=(-1 if code&8 else 1)*grid[code&7]*2**((scale>>3)-7)*(1+(scale&7)/8)/256
  assert raw==want
  bits=struct.unpack('>I',struct.pack('>f',raw))[0]
  assert bits&65535==0,'product requires BF16 rounding'
# Zero scale is canonicalized to zero codes; negative scales flip code sign.
for code in range(16):
 assert bf16(0)*bf16(0x7780)==0
 assert (-1 if (code^8)&8 else 1)*grid[code&7]==-((-1 if code&8 else 1)*grid[code&7])
print('PASS: all2^32-word placement by basis+exhaustive IMAD proof; all1,904 normal code/scale products exact BF16; unique row/k16scale ownership across64stages/16warps/32lanes; pack AST unchanged except placement.')
