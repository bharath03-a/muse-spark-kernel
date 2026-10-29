"""[D] hand CUDA mma.sync slice kernel m=1..8 vs current path."""
from __future__ import annotations
import os, statistics, threading
import torch
from torch.utils.cpp_extension import load_inline
from common import load_module, weights
from slice_mma_src import SLICE_CPP, SLICE_SRC
threading.Timer(110, lambda: (print("WATCHDOG", flush=True), os._exit(0))).start()
C = load_module("checkpoints/cp13_B_check1.415.py", "cp13")
ext2 = load_inline(name="muse_slice_d3", cpp_sources=SLICE_CPP, cuda_sources=SLICE_SRC, functions=["launch_slice"],
                   extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
FL = torch.empty(256 * 2**20, dtype=torch.uint8, device="cuda"); D = torch.zeros(1, device="cuda")
def T(fn, reps=60):
    for _ in range(2): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    ts = []
    for _ in range(reps):
        FL.sum(); a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b) * 1e3)
    return statistics.median(ts)
EMPTY = T(lambda: D.add_(1)); print(f"empty {EMPTY:.2f}", flush=True)
fmt = lambda t: f"{t:.2f}/{t - EMPTY:.2f}"
w = weights()["unpooler"][:1024].cuda().contiguous()
st = {"ws": torch.empty(C.WS_FLOATS, device="cuda"), "cnt": torch.zeros(C.COUNTERS, device="cuda", dtype=torch.int32), "ext": C._build_cuda(), "desc": {}}
gen = torch.Generator(device="cuda").manual_seed(0)
for m in (1, 2, 3, 4, 5, 8):
    x = torch.randn(m, 4096, device="cuda", generator=gen).to(torch.bfloat16)
    ref = x.double() @ w.double().t()
    err = lambda o: f"{((o.double() - ref).norm() / ref.norm()).item():.0e}"
    def cur():
        if m == 1:
            o = torch.empty(1, 1024, device="cuda", dtype=torch.bfloat16)
            C._bf16_launch(x, w, o, st, "unpooler_slice", C.Tile(1, 2, 2048, 1, 1, 8, 2, "fma")); return o
        pk = ("gemv", (2, 2, 2)) if m == 2 else ("gemv", (4, 4, 2)) if m <= 4 else ("wgmma", (64, 8, 128, 8, 4))
        if m == 5:
            o = torch.empty(m, 1024, device="cuda", dtype=torch.bfloat16)
            C._bf16_launch(x, w, o, st, "unpooler_slice", C.BF16_TILES["unpooler_slice"][1]); return o
        return C._cuda_run(st, x, w, 1024, torch.bfloat16, *pk)
    o = cur(); print(f"m{m} current {fmt(T(cur))} err {err(o)}", flush=True)
    for nb, nw in ((1, 8), (1, 16), (1, 4), (2, 8), (2, 16)):
        def f():
            o = torch.empty(m, 1024, device="cuda", dtype=torch.bfloat16); ext2.launch_slice(x, w, o, nb, nw); return o
        o = f(); torch.cuda.synchronize()
        eq = all(torch.equal(f(), o) for _ in range(5))
        print(f"  m{m} mma NB{nb} NW{nw} ctas{1024 // (nb * 8)}: {fmt(T(f))} err {err(o)} det {eq}", flush=True)
os._exit(0)
