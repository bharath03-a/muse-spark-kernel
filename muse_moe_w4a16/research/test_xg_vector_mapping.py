"""CPU-only byte-placement proof; does not establish GPU correctness or speed."""
import ast
from collections import Counter
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]

def perm(col):
    a = col & 63
    return (col & ~63) | (((a >> 2) & 3) << 4) | (((a & 3) >> 1) << 3) | ((a >> 4) << 1) | (a & 1)

def addr(col, pos, pad):
    b = (col & 63) * 2
    return ((col >> 6) * pad + pos) * 128 + (((b >> 4) ^ (pos & 7)) << 4) + (b & 15)

def check_row(pos, pad, segs):
    old, new = {}, {}
    for seg in range(segs):
        for base in range(seg * (512 // segs), (seg + 1) * (512 // segs), 32):
            for lane in range(32):
                q = base + lane
                for j in range(4):
                    # Unique word identifier encodes source q4 and word index.
                    key = addr(perm(q * 8 + j * 2), pos, pad)
                    assert key not in old
                    old[key] = q * 4 + j
                dst = addr(q * 8, pos, pad)
                assert dst % 16 == 0
                for j in range(4):
                    sl = (lane & ~7) | ((lane >> 2) & 1) | (j << 1)
                    value = (base + sl) * 4 + (lane & 3)
                    assert dst + j * 4 not in new
                    new[dst + j * 4] = value
    assert len(new) == 2048 and old == new

for pad in (8, 16, 64, 128, 256, 512):
    for pos in (0, 1, min(7, pad - 1), pad - 1):
        for segs in (1, 2, 4, 8):
            check_row(pos, pad, segs)

# Launch mapping and stable list semantics for scored and nearby cases.
rng = random.Random(343)
for m in (1, 4, 8, 16, 32, 33, 128, 256, 257, 512):
    ids = [rng.randrange(-5, 40) for _ in range(m * 8)]
    counts = Counter()
    ranks = []
    for e in ids:
        ranks.append(counts[e])
        if 0 <= e < 32:
            counts[e] += 1
    offsets, off = {}, 0
    for e in range(32):
        offsets[e] = off
        off += ((counts[e] + 7) & ~7) * 8192
    assert all(0 <= ranks[i] < counts[e] for i, e in enumerate(ids) if 0 <= e < 32)
    for segs in (1, 2, 4, 8):
        visits = Counter()
        for block in range(132):
            for warp in range(32):
                for item in range(block * 32 + warp, m * 8 * segs, 132 * 32):
                    visits[item] += 1
        assert len(visits) == m * 8 * segs and all(v == 1 for v in visits.values())

# Lossless front-end-only mutation: math and strict BF16 boundaries unchanged.
def cuda(path):
    tree = ast.parse(path.read_text())
    return next(n.value.value for n in tree.body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], 'id', '') == 'CUDA_X')
old = cuda(ROOT / 'research_moe_xg2.py')
new = cuda(ROOT / 'research_moe_xg_vector.py')
marker = 'template <int N>\n__device__ __forceinline__ void epilogue('
assert old[old.index(marker):].replace(
    'xgather<<<min(g_gridX, (M * 8 + 31) / 32), XGT, 0, st>>>(',
    'xgather<<<g_gridX, XGT, 0, st>>>(') == new[new.index(marker):]
assert 'griddepcontrol' not in new and 'cudaLaunchKernelEx' not in new
print('PASS: 96 row-layout proofs, ten scored/nearby route maps, 4 segmentation controls, strict downstream CUDA identity, no PDL.')
