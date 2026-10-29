"""CPU proof: paired WGMMA output rows map bijectively to original gate/up rows."""
perm=[]
for block in range(4096//8):perm.extend(range(block*8,block*8+8));perm.extend(range(4096+block*8,4096+block*8+8))
assert sorted(perm)==list(range(8192))
for bm in (64,128,256):
 seen=set()
 for rb in range(8192//bm):
  for wi in range(bm//64):
   for warp in range(4):
    for g in range(8):
     base=rb*bm+wi*64+warp*16
     out=base//2+g
     assert perm[base+g]==out
     assert perm[base+g+8]==4096+out
     assert out not in seen
     seen.add(out)
 assert seen==set(range(4096))
print('PASS all4096gate/up pairs for64/128/256-row WGMMAtiles; no duplicated or missing outputs')
