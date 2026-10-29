"""qkv_gate m=128/256 split-K warp-specialized kernel: correctness + timing vs cuBLAS. --compile = build only."""
import hashlib, os, statistics, sys, threading
COMPILE = "--compile" in sys.argv
threading.Timer(480 if COMPILE else 240, lambda: os._exit(3)).start()  # Timer(240 run)
import torch
from common import load_module
from exp_r6_mc_src import MC_SRC, MC_CPP
from torch.utils.cpp_extension import load_inline
import reference

sub = load_module("submission.py", "sub")
DBG = '--dbg' in sys.argv
src = ("#define MC_DBG 1\n" if DBG else "") + sub.CUDA_SRC + MC_SRC
tag = hashlib.sha1(src.encode()).hexdigest()[:8]
ext = load_inline(name=f"r6mcqkv_{tag}", cpp_sources=sub.CUDA_CPP + "\n" + MC_CPP + ("\nvoid get_dbg(torch::Tensor o);" if DBG else ""), cuda_sources=src,
                  functions=["launch_qkv_mc"] + (["get_dbg"] if DBG else []), extra_cuda_cflags=sub.CUDA_FLAGS, extra_cflags=["-O3"],
                  extra_ldflags=sub.CUDA_LDFLAGS)
print("built", tag, flush=True)
if COMPILE:
    os._exit(0)

def dbg_report(m, x, wr, out):
    flushb = flush
    for _ in range(3):
        flushb.sum(); ext.launch_qkv_mc(x, wr, out, ws, cnt)
    torch.cuda.synchronize()
    flushb.sum(); ext.launch_qkv_mc(x, wr, out, ws, cnt); torch.cuda.synchronize()
    o = torch.zeros(256 * 6, dtype=torch.int64); ext.get_dbg(o)
    d = o.view(256, 6)[:126].double() / 1000.0
    t0 = d[:, 0].min()
    f = lambda v: f"med {v.median():.2f} max {v.max():.2f}"
    print("start skew", f(d[:, 0] - t0), "| main", f(d[:, 1] - d[:, 0]), "| wswrite", f(d[:, 2] - d[:, 1]),
          "| spin", f(d[:, 3] - d[:, 2]), "| reduce", f(d[:, 4] - d[:, 3]), "| total", f(d[:, 4] - t0), flush=True)

w = reference.generate_weights(reference.WEIGHTS_SEED)["qkv_gate"]
n, k = w.shape
wr = w.view(n // 128, 128, k // 64, 64).permute(0, 2, 1, 3).contiguous().view(-1, 64)
ws = torch.empty(18 * 7 * 256 * 128, device="cuda", dtype=torch.float32)
cnt = torch.zeros(128, device="cuda", dtype=torch.int32)
flush = torch.empty(64 * 2**20, device="cuda", dtype=torch.float32).normal_()

def graph_of(fn):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    torch.cuda.synchronize(); return g
def t(g):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    flush.sum(); s.record(); g.replay(); e.record(); e.synchronize(); return s.elapsed_time(e) * 1e3

for m in (128, 256):
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    ext.launch_qkv_mc(x, wr, out, ws, cnt); torch.cuda.synchronize()
    ref = (x.float() @ w.float().t())
    err = ((out.float() - ref).norm() / ref.norm()).item()
    outs = [out.clone()]
    for _ in range(200):
        ext.launch_qkv_mc(x, wr, out, ws, cnt); outs.append(out.clone())
    det = all(torch.equal(o, outs[0]) for o in outs)
    print(f"m={m} rel_l2={err:.2e} deterministic={det} cnt_zero={int(cnt.sum())==0}", flush=True)
    if DBG: dbg_report(m, x, wr, out)
    fns = {"cublas": lambda: torch.mm(x, w.t()), "mc": lambda: ext.launch_qkv_mc(x, wr, out, ws, cnt)}
    gs = {nm: graph_of(f) for nm, f in fns.items()}
    res = {nm: [] for nm in gs}
    for r in range(150):
        for nm in (list(gs) if r % 2 == 0 else list(gs)[::-1]):
            res[nm].append(t(gs[nm]))
    print(m, {nm: round(statistics.median(v) - 4.0, 2) for nm, v in res.items()}, flush=True)
