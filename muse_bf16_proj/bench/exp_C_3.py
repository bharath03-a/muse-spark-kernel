"""[C] pooler/unpooler m32 gluon tile variants (cp13), cold-L2 graph replay."""
import os, statistics, threading
threading.Timer(200, lambda: (print('WATCHDOG', flush=True), os._exit(0))).start()
import torch
from common import load_module, reference
mod = load_module("checkpoints/cp13_B_check1.415.py", "cp13")
w = reference.generate_weights(reference.WEIGHTS_SEED)
state = {k: v.cuda().contiguous() for k, v in w.items()}
state["ws"] = torch.empty(mod.WS_FLOATS, device="cuda", dtype=torch.float32)
state["cnt"] = torch.zeros(mod.COUNTERS, device="cuda", dtype=torch.int32)
state["gl_desc"] = {}; state["gl_xl"] = {}
flush = torch.empty(256 * 2**20, dtype=torch.uint8, device="cuda")
def tm(fn, reps=100):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        flush.sum()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b) * 1e3)
    return statistics.median(ts)
dummy = torch.zeros(1, device="cuda"); print("dummy", tm(lambda: dummy.add_(1)), flush=True)
G = mod.GTile
for key in ("pooler", "unpooler"):
    n, k = state[key].shape
    m = 32
    x = (torch.randn(m, k, device="cuda") / k**.5).to(torch.bfloat16)
    ref = x.double() @ state[key].double().t()
    cfgs = [(64,128,8,1),(64,128,6,1),(64,128,4,1),(64,256,4,1),(64,64,8,1),(32,128,8,1),(32,128,4,1),(32,256,4,1),(16,128,8,1),
            (64,128,8,2),(64,128,6,2),(64,128,4,2),(64,64,8,2),(64,256,4,2),(128,128,4,1),(128,64,8,1)]
    for bn, bk, st, sp in cfgs:
        t = G(m, m, bn, bk, m, st, sp)
        name = f"{key} bn{bn} bk{bk} st{st} sp{sp} ctas={n//bn*sp}"
        try:
            if not mod._gluon_fits(t, n, k): print(name, "nofit"); continue
            def f():
                out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
                mod._gluon_launch(x, out, state, key, t); return out
            o = f(); torch.cuda.synchronize()
            err = ((o.double() - ref).norm() / ref.norm()).item()
            print(f"{name:42s} {tm(f):7.2f} err={err:.1e}", flush=True)
        except Exception as e:
            print(name, "FAIL", type(e).__name__, str(e)[:80], flush=True)
