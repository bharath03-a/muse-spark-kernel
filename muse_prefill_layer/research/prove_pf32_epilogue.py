"""CPU address/shuffle proof for pf32 paired BF16 epilogues, including partial tokens."""
for ntok in (1,7,8,15,31,32,33,65,127,128,129,255,256):
  gate_seen=set(); down_seen=set()
  for warp in range(4):
    for lane in range(32):
      g=lane>>2;t=lane&3;odd=g&1
      for j in range(32):
        for e2 in range(2):
          token=2*t+8*j+e2
          if token>=ntok:continue
          # XOR lane4 swaps adjacent g rows; paired store executed at even g.
          if not odd:
            for feature in (warp*8+(g&6),warp*8+(g&6)+1):
              key=(token,feature);assert key not in gate_seen;gate_seen.add(key)
          # Down branch even lanes store low8 and odd lanes high8 pairs.
          start=warp*16+(g&6)+(8 if odd else 0)
          for feature in (start,start+1):
            key=(token,feature);assert key not in down_seen;down_seen.add(key)
  assert gate_seen=={(t,f) for t in range(ntok) for f in range(32)}
  assert down_seen=={(t,f) for t in range(ntok) for f in range(64)}
print('PASS pf32 pairedstore/XOR4 mapping, all64down/32fusedfeatures,13partial-token boundaries')
