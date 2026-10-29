"""[B] z9 config sweep per shape, cold L2. exp_B_sweep.py [--compile] [--reps 60]"""
from __future__ import annotations
import hashlib, os, sys, threading
import torch
from common import load_module, weights

threading.Timer(240, lambda: (print("WATCHDOG", flush=True), os._exit(0))).start()
C = load_module("checkpoints/cp11_lt_check1.401.py", "cp11")
# id: (RB, NT, W, CS, D, MINB)
NEW = {
    137: (2, 2, 7, 1, 2, 2), 200: (4, 2, 8, 1, 2, 1), 201: (4, 2, 8, 2, 2, 1), 202: (4, 2, 4, 2, 2, 2),
    203: (8, 2, 8, 2, 1, 1), 204: (8, 2, 4, 2, 2, 1), 205: (8, 2, 8, 4, 1, 1), 206: (8, 2, 4, 4, 2, 1),
    207: (4, 2, 6, 2, 2, 1), 210: (4, 4, 8, 2, 2, 1), 211: (8, 4, 4, 4, 2, 1), 212: (4, 4, 8, 1, 2, 1),
    213: (8, 4, 4, 2, 2, 1), 214: (4, 4, 6, 2, 2, 1), 215: (2, 4, 8, 1, 2, 2), 216: (2, 4, 8, 2, 2, 2),
}
OLD = {25: (2, 1, 8, 1, 2, 2), 27: (2, 2, 8, 1, 2, 2), 28: (2, 2, 8, 1, 1, 2), 31: (1, 1, 8, 1, 2, 2),
       39: (2, 1, 8, 2, 2, 2), 40: (1, 1, 8, 2, 2, 2), 44: (1, 1, 8, 1, 2, 3), 37: (1, 1, 8, 1, 3, 2)}
CFG = {**OLD, **NEW}
SRC = C.ZIP_SRC.replace('  TORCH_CHECK(false, "zip: config id not compiled ", id);',
    " ".join(f"Z9_CASE({i}, {', '.join(map(str, c))})" for i, c in NEW.items()) +
    '\n  TORCH_CHECK(false, "zip: config id not compiled ", id);')
assert SRC != C.ZIP_SRC
tag = hashlib.sha1((SRC + " ".join(C.CUDA_FLAGS)).encode()).hexdigest()[:10]
from torch.utils.cpp_extension import load_inline
ext = load_inline(name=f"muse_zipB_{tag}", cpp_sources=C.ZIP_CPP, cuda_sources=SRC, functions=["launch_z7"],
                  extra_cuda_cflags=C.CUDA_FLAGS, extra_cflags=["-O3"], extra_ldflags=C.CUDA_LDFLAGS)
print("built", tag, flush=True)
if "--compile" in sys.argv:
    os._exit(0)
reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 60
W = weights()
gen = torch.Generator(device="cuda").manual_seed(0)
flush = torch.zeros(64 * 2**20, device="cuda", dtype=torch.float32)
packs = {}
def pack(layer, rb):
    if (layer, rb) not in packs:
        w = W[layer].cuda().contiguous()
        p = C._zip_pack(w, rb); p["ext"] = ext
        packs[layer, rb] = (w, p)
    return packs[layer, rb]
for layer in ("pooler", "unpooler", "qkv_gate"):
    for m in (16, 32):
        res = []
        for cid, (rb, nt, *_r) in sorted(CFG.items()):
            if m > nt * 8:
                continue
            try:
                w, p = pack(layer, rb)
                x = torch.randn(m, w.shape[1], device="cuda", generator=gen).to(torch.bfloat16)
                t = C._graph_cold_us(lambda: C._zip_launch(x, w.shape[0], cid, p), flush, reps)
                res.append((t, cid))
            except Exception as e:
                res.append((999.0, cid)); print(f"  {layer} m{m} id{cid} ERR {str(e)[:80]}", flush=True)
        res.sort()
        print(f"{layer} m={m}: " + " ".join(f"{c}:{t:.2f}" for t, c in res[:8]) + "  | base " +
              " ".join(f"{c}:{t:.2f}" for t, c in res if c in OLD), flush=True)
        for t, cid in res[:2]:
            if t < 900:
                w, p = pack(layer, CFG[cid][0])
                try:
                    C._zip_check((m, m, cid, p), w, gen)
                except Exception as e:
                    print(f"  CHECK FAIL {layer} m{m} id{cid}: {e}", flush=True)
print("DONE", flush=True)
os._exit(0)
