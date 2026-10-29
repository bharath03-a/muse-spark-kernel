"""[D] multi-token FMA gemv for unpooler slice / o at m=2..8."""
from __future__ import annotations
import os, statistics, threading, sys
import torch, triton, triton.language as tl
from common import load_module, weights
threading.Timer(140, lambda: (print("WATCHDOG", flush=True), os._exit(0))).start()
C = load_module("checkpoints/cp13_B_check1.415.py", "cp13")
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

@triton.jit(do_not_specialize=["M"])
def gemv_mt(x_ptr, w_ptr, out_ptr, ws_ptr, cnt_ptr, M, stride_xm, N: tl.constexpr, K: tl.constexpr,
            SPLIT: tl.constexpr, BLOCK_N: tl.constexpr, BT: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    KL: tl.constexpr = K // SPLIT
    offs_k = tl.max_contiguous(tl.multiple_of(pid_s * KL + tl.arange(0, KL), KL), KL)
    w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :], eviction_policy="evict_first").to(tl.float32)
    for t in tl.static_range(BT):
        if t < M:
            xv = tl.load(x_ptr + t * stride_xm + offs_k, eviction_policy="evict_last").to(tl.float32)
            r = tl.sum(w * xv[None, :], axis=1)
            if SPLIT == 1:
                tl.store(out_ptr + t * N + offs_n, r.to(out_ptr.dtype.element_ty))
            else:
                tl.store(ws_ptr + (pid_s * BT + t) * N + offs_n, r)
    if SPLIT > 1:
        tl.debug_barrier()
        done = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
        if done == SPLIT - 1:
            for t in tl.static_range(BT):
                if t < M:
                    tot = tl.zeros((BLOCK_N,), dtype=tl.float32)
                    for s in tl.static_range(SPLIT):
                        tot += tl.load(ws_ptr + (s * BT + t) * N + offs_n, cache_modifier=".cg")
                    tl.store(out_ptr + t * N + offs_n, tot.to(out_ptr.dtype.element_ty))
            tl.atomic_xchg(cnt_ptr + pid_n, 0)

def launch(x, w, out, ws, cnt, R, split, warps):
    n, k = w.shape
    gemv_mt[(n // R, split)](x, w, out, ws, cnt, x.shape[0], x.stride(0), N=n, K=k, SPLIT=split, BLOCK_N=R, BT=8, num_warps=warps)

Wt = weights()
ws = torch.empty(C.WS_FLOATS, device="cuda", dtype=torch.float32); cnt = torch.zeros(C.COUNTERS, device="cuda", dtype=torch.int32)
ext = C._build_cuda(); st = {"ws": ws, "cnt": cnt, "ext": ext, "desc": {}}
gen = torch.Generator(device="cuda").manual_seed(0)
cases = [("slice", Wt["unpooler"][:1024].cuda().contiguous(), "unpooler_slice", C.Tile(1, 2, 2048, 1, 1, 8, 2, "fma"))]
cases.append(("o", Wt["o"].cuda().contiguous(), "o", C.TILES.get("o") and C.BF16_TILES["o"][0]))
for name, w, key, t1 in cases:
    n, k = w.shape; print(name, w.shape, flush=True)
    st[key] = w
    for m in (1, 2, 8):
        x = torch.randn(m, k, device="cuda", generator=gen).to(torch.bfloat16)
        ref = x.float() @ w.float().t()
        err = lambda o: f"{((o.float() - ref).norm() / ref.norm()).item():.0e}"
        def cur():
            o = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
            if name == "slice" and m == 8:
                return C._cuda_run(st, x, w, n, torch.bfloat16, "wgmma", (64, 8, 128, 8, 4))
            if name == "slice" and m == 2:
                return C._cuda_run(st, x, w, n, torch.bfloat16, "gemv", (2, 2, 2))
            if m == 1:
                C._bf16_launch(x, w, o, st, key, t1); return o
            C._bf16_launch(x, w, o, st, key, C._bf16_pick(key, m)); return o
        o = cur(); print(f"  m{m} current {fmt(T(cur))} err {err(o)}", flush=True)
        for R in (4, 8, 16):
            for sp in (1, 2):
                for wp in (4, 8):
                    def f():
                        o = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
                        launch(x, w, o, ws, cnt, R, sp, wp); return o
                    try:
                        o = f(); torch.cuda.synchronize()
                        print(f"  m{m} mt R{R} S{sp} w{wp} ctas{n // R * sp}: {fmt(T(f))} err {err(o)}", flush=True)
                    except Exception as e:
                        print(f"  m{m} mt R{R} S{sp} w{wp} ERR {str(e)[:80]}", flush=True)
os._exit(0)
