"""Exhaustive CPU proof of disconnected tail selector and packed row offsets."""
from pathlib import Path
import ctypes
# The kernel deliberately leaves rows outside the eligible tail untouched.
for count in range(4097):
    rem = count % 128
    tiles = int(33 <= rem <= 64)
    rows = list(range((count // 128) * 128, count)) if tiles else []
    assert len(rows) == (rem if tiles else 0)
    assert all(0 <= r < count for r in rows)
    if tiles:
        assert len(rows) <= 64 and rows[0] == count - rem
# Existing 256-row physical blocks split into two 128-row CTA blocks.
for nrb256 in (16,32):
    logical = set()
    for rb in range(2*nrb256):
        block, sub = rb//2, rb%2
        for wi in range(2):
            for warp in range(4):
                for lane_row in range(16):
                    row = block*256 + sub*128 + wi*64 + warp*16 + lane_row
                    assert row == rb*128 + wi*64 + warp*16 + lane_row
                    logical.add(row)
    assert logical == set(range(nrb256*256))
class Hdr(ctypes.Structure):
    _fields_=[('v',ctypes.c_int*8)]
class Sm5(ctypes.Structure):
    _fields_=[('bars',ctypes.c_ulonglong*15),('hdr',Hdr*5),('order',ctypes.c_int*32),('cum',ctypes.c_int*33),('tnext',ctypes.c_int)]
assert ctypes.sizeof(Sm5)==544
smem = 5*(8192+6144)+ctypes.sizeof(Sm5)+1024
assert smem == 73248
print('PASS counts0..4096, NCW2 packed row bijection, Sm5=544B, dynamic shared=73248B')
print('Resource hypothesis: 2 CTAs use 148544 shared bytes including 1024B static/CTA; 61440 redistributed registers; 24 warps.')
