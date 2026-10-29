"""Hypothesis: x re-reads from L2 bound m=128/256. Time existing CUDA wgmma kernel with and without x TMA."""
import hashlib, statistics, torch
from common import load_module
import reference
from torch.utils.cpp_extension import load_inline

sub = load_module("submission.py", "sub")
SRC = sub.CUDA_SRC
nox = SRC.replace("mbar_expect_tx(&ready[slot], STAGE_BYTES);", "mbar_expect_tx(&ready[slot], STAGE_BYTES - BT * BK * 2);")
a = "tma_load_2d(x_box(slot, kb), &x_map, kc, t0, &ready[slot]);"
assert a in nox
nox = nox.replace(a, "")

def build(src, nm):
    tag = hashlib.sha1(src.encode()).hexdigest()[:8]
    return load_inline(name=f"r6hyp_{nm}_{tag}", cpp_sources=sub.CUDA_CPP, cuda_sources=src,
                       functions=["launch_wgmma", "launch_gemv"], extra_cuda_cflags=sub.CUDA_FLAGS,
                       extra_cflags=["-O3"], extra_ldflags=sub.CUDA_LDFLAGS)

G = "wgmma_m64n<BT>(acc, da_base + (uint64_t)(ks * 2), db_base + (uint64_t)(ks * 2));"
nocomp = SRC.replace(G, "if (ks == 99) wgmma_m64n<BT>(acc, da_base, db_base);")
noxnc = nox.replace(G, "if (ks == 99) wgmma_m64n<BT>(acc, da_base, db_base);")
assert nocomp != SRC and noxnc != nox
ext_a, ext_b, ext_c, ext_d = build(SRC, "a"), build(nox, "b"), build(nocomp, "c"), build(noxnc, "d")
flush = torch.empty(64 * 2**20, device="cuda", dtype=torch.float32).normal_()
W = reference.generate_weights(reference.WEIGHTS_SEED)

def graph_of(fn):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    torch.cuda.synchronize()
    return g

def t(g):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    flush.sum(); s.record(); g.replay(); e.record(); e.synchronize()
    return s.elapsed_time(e) * 1e3

dummy = torch.zeros(1, device="cuda"); gd = graph_of(lambda: dummy.add_(1))
for layer in ("qkv_gate", "unpooler"):
    w = W[layer]; n, k = w.shape
    for m in (128,):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        fns = {"cublas": lambda: torch.mm(x, w.t()),
               "cuda_c2": lambda: ext_a.launch_wgmma(x, w, out, 64, 128, 64, 6, 2),
               "nox_c2": lambda: ext_b.launch_wgmma(x, w, out, 64, 128, 64, 6, 2),
               "nocomp": lambda: ext_c.launch_wgmma(x, w, out, 64, 128, 64, 6, 2),
               "nox_nocomp": lambda: ext_d.launch_wgmma(x, w, out, 64, 128, 64, 6, 2)}
        gs = {nm: graph_of(f) for nm, f in fns.items()}
        res = {nm: [] for nm in gs}
        for r in range(150):
            for nm in (list(gs) if r % 2 == 0 else list(gs)[::-1]):
                res[nm].append(t(gs[nm]))
        base = statistics.median(t(gd) for _ in range(50))
        print(layer, m, {nm: round(statistics.median(v) - 4.0, 2) for nm, v in res.items()}, flush=True)
