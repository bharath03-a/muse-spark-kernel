"""[C] router m8/16/32 + floors: empty/touch kernels, wgmma csize variants, fp32 gemv_mt."""
import os, statistics, threading
threading.Timer(240, lambda: (print('WATCHDOG', flush=True), os._exit(0))).start()
import torch
from common import load_module, reference

mod = load_module("bench/cand_C.py", "cand_C")
ext = mod._build_cuda()
w = reference.generate_weights(reference.WEIGHTS_SEED)
r = w["router"].float().cuda()
hi = r.to(torch.bfloat16); lo = (r - hi.float()).to(torch.bfloat16)
hl = torch.stack([hi, lo]).contiguous()
flush = torch.empty(256 * 2**20, dtype=torch.uint8, device="cuda")
REPS = 80


def graph_of(fn):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    torch.cuda.synchronize()
    return g


def t_us(g):
    ts = []
    for _ in range(REPS):
        flush.sum()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)
    return statistics.median(ts)


def run(name, fn, ref=None):
    try:
        out = fn()
        torch.cuda.synchronize()
        err = ""
        if ref is not None:
            err = f" err={((out.double()-ref).norm()/ref.norm()).item():.1e}"
        print(f"{name:34s} {t_us(graph_of(fn)):6.2f}{err}", flush=True)
    except Exception as e:
        print(f"{name:34s} FAIL {type(e).__name__}: {str(e)[:100]}", flush=True)


dummy = torch.zeros(1, device="cuda")
run("graph dummy add", lambda: dummy.add_(1))
x1 = torch.randn(32, 8192, device="cuda", dtype=torch.bfloat16)
o = torch.empty(32, 256, device="cuda", dtype=torch.float32)
run("empty kernel", lambda: ext.launch_misc(x1, r, o, 0))
run("touch 1B of W", lambda: ext.launch_misc(x1, r, o, 1))
for m in (8, 16, 32):
    x = x1[:m].contiguous()
    ref = x.double() @ r.double().t()
    print(f"--- router m={m}")
    run("base wgmma c8 bk128 st8", lambda: _w(x, m, 128, 8, 8), ref) if False else None
    def W(bt, bk, st, cs):
        def f():
            out = torch.empty(m, 256, device="cuda", dtype=torch.float32)
            ext.launch_wgmma(x, hl, out, 32, bt, bk, st, cs)
            return out
        return f
    run("wgmma c8 bk128 st8 (base)", W(m, 128, 8, 8), ref)
    for kind, mm, nm in ((4, 8, "gemv_mt R1 MT8"), (30, 8, "cs R4 MT8 S2"), (31, 8, "cs R4 MT8 S4"), (32, 8, "cs R2 MT8 S2"),
                         (33, 8, "cs R2 MT8 S4"), (34, 8, "cs R8 MT8 S2"), (35, 16, "cs R4 MT16 S2"), (36, 16, "cs R2 MT16 S2"),
                         (37, 16, "cs R2 MT16 S4"), (38, 16, "cs R4 MT16 S4")):
        if mm != m: continue
        def f(kind=kind):
            out = torch.empty(m, 256, device="cuda", dtype=torch.float32)
            ext.launch_misc(x, r, out, kind)
            return out
        run(nm, f, ref)
