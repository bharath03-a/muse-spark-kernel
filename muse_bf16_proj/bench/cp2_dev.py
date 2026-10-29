"""cp2 sweeps, paired against cp1 in one process: Gluon GEMV (pooler/unpooler), router m=128/256 two-dot
kernel, router m=9-16 CUDA cluster tiles. Prints median paired deltas (candidate - cp1) in us.

    python bench/cp2_dev.py [--reps 50] [--only gluon,router_big,router16] [--cold]
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import tempfile
import time

if "--cold" in sys.argv:
    os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp()

import torch

from common import load_module, reference, weights
from harness import FLUSH_BYTES, time_replay

CP1 = "checkpoints/cp1_router_gemv_cuda_bigpooler.py"
OVERHEAD = 4.0


def graph_of(fn):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    g.replay()
    torch.cuda.synchronize()
    return g, out


def paired(g_base, g_cand, flush, reps):
    d, tb, tc = [], [], []
    for r in range(reps):
        order = (g_base, g_cand) if r % 2 == 0 else (g_cand, g_base)
        t = {}
        for g in order:
            t[id(g)] = time_replay(g, flush) - OVERHEAD
        tb.append(t[id(g_base)])
        tc.append(t[id(g_cand)])
        d.append(t[id(g_cand)] - t[id(g_base)])
    return statistics.median(tb), statistics.median(tc), statistics.median(d)


def check(data, out, w):
    ok, msg = reference.check_implementation(data, out, w)
    return ok, msg


def run_cands(label, data, w, base_fn, cands, flush, reps):
    g_base, out_b = graph_of(base_fn)
    ok, msg = check(data, out_b, w)
    print(f"== {label} cp1 {'ok' if ok else 'FAIL ' + msg}", flush=True)
    best = None
    for name, fn in cands:
        try:
            g, out = graph_of(fn)
            ok, msg = check(data, out, w)
            if not ok:
                print(f"  {name}: WRONG {msg}", flush=True)
                continue
            tb, tc, d = paired(g_base, g, flush, reps)
            print(f"  {name}: cp1 {tb:6.2f} cand {tc:6.2f} delta {d:+6.2f} us", flush=True)
            if best is None or d < best[1]:
                best = (name, d)
        except Exception as exc:
            print(f"  {name}: FAILED {type(exc).__name__}: {str(exc)[:400]}", flush=True)
    print(f"  BEST {label}: {best}", flush=True)


def gluon_sweep(sub, st, cp1, cst, w, flush, reps):
    for layer, m in (("pooler", 16), ("pooler", 32), ("unpooler", 16), ("pooler", 8), ("unpooler", 32),
                     ("pooler", 12), ("pooler", 24)):
        data = reference.generate_input(m, layer, 1)
        x = data["x"]
        n, k = st[layer].shape
        cands = []
        bts = (8, 16) if m <= 8 else ((16,) if m <= 16 else (32,))
        for bt in bts:
            for bk, stages in ((128, 8), (128, 6), (128, 10), (256, 4), (256, 3)):
                t = sub.GTile(1, m, 64, bk, bt, stages)
                if not sub._gluon_fits(t, n, k):
                    continue

                def fn(t=t):
                    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
                    sub._gluon_launch(x, out, st, layer, t)
                    return out
                cands.append((f"BT{bt} BK{bk} S{stages}", fn))
        run_cands(f"gluon {layer} m={m}", data, w, lambda: cp1.kernel_fn(data, cst), cands, flush, reps)


def router_big_sweep(sub, st, cp1, cst, w, flush, reps):
    cfgs = []
    for swap in (0, 1):
        for bm, bn in ((64, 64), (64, 32), (128, 64), (128, 32)):
            for split in (4, 8, 16):
                cfgs.append((bm, bn, 128, split, swap, 4, 3))
    cfgs += [(64, 64, 64, 16, 0, 4, 4), (64, 64, 64, 32, 0, 4, 4), (128, 64, 64, 16, 0, 4, 4),
             (64, 64, 128, 8, 0, 8, 3), (128, 64, 128, 8, 0, 8, 3)]
    n, k = st["router"].shape
    for m in (128, 256):
        data = reference.generate_input(m, "router", 1)
        x = data["x"]
        cands = []
        for cfg in cfgs:
            bm, bn, bk, split = cfg[:4]
            ctas = -(-m // bm) * (n // bn) * split
            if not sub._router_big_fits(cfg, m, n, k) or not 32 <= ctas <= 528:
                continue
            sub._router_descs(st, bn, bk)
            cands.append((f"{cfg} ctas={ctas}", lambda cfg=cfg: sub._router_big_launch(x, st, cfg)))
        run_cands(f"router_big m={m}", data, w, lambda: cp1.kernel_fn(data, cst), cands, flush, reps)


def router16_sweep(sub, st, cp1, cst, w, flush, reps):
    for m in (16, 12, 9):
        data = reference.generate_input(m, "router", 1)
        x = data["x"]
        cands = []
        for nt in (1, 4):
            for c in (2, 4, 8, 16):
                for wk in (8, 4):
                    if 8192 % (c * wk * 64):
                        continue
                    p = (1, nt, c, wk)
                    cands.append((f"cluster {p}", lambda p=p: sub._cuda_run(
                        st, x, st["router_hl"], 256, torch.float32, "cluster", p)))
        run_cands(f"router m={m}", data, w, lambda: cp1.kernel_fn(data, cst), cands, flush, reps)


def gluon2_sweep(sub, st, cp1, cst, w, flush, reps):
    for layer, m in (("unpooler", 16), ("unpooler", 32), ("pooler", 16), ("unpooler", 12)):
        data = reference.generate_input(m, layer, 1)
        x = data["x"]
        n, k = st[layer].shape
        bt = 16 if m <= 16 else 32
        cands = []
        for bk, stages in ((128, 8), (256, 4), (64, 16), (64, 8)):
            t = sub.GTile(1, m, 64, bk, bt, stages)
            if not sub._gluon_fits(t, n, k):
                continue

            def fn(t=t):
                out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
                sub._gluon_launch(x, out, st, layer, t)
                return out
            cands.append((f"BT{bt} BK{bk} S{stages}", fn))
        run_cands(f"gluon2 {layer} m={m}", data, w, lambda: cp1.kernel_fn(data, cst), cands, flush, reps)


def router_big2_sweep(sub, st, cp1, cst, w, flush, reps):
    cfgs = [(64, 32, 128, 8, 0, 4, 3), (64, 32, 128, 8, 0, 4, 4), (64, 32, 128, 4, 0, 4, 4),
            (64, 32, 64, 8, 0, 4, 4), (64, 32, 64, 4, 0, 4, 6), (64, 16, 128, 4, 0, 4, 3),
            (64, 16, 128, 8, 0, 4, 3), (64, 16, 64, 8, 0, 4, 4), (64, 32, 128, 2, 0, 4, 4),
            (64, 16, 128, 2, 0, 4, 4), (128, 32, 64, 4, 0, 4, 4), (64, 32, 256, 4, 0, 4, 2),
            (64, 32, 128, 8, 0, 2, 3), (64, 16, 128, 4, 0, 2, 4)]
    n, k = st["router"].shape
    for m in (128, 256):
        data = reference.generate_input(m, "router", 1)
        x = data["x"]
        cands = []
        for cfg in cfgs:
            bm, bn, bk, split = cfg[:4]
            ctas = -(-m // bm) * (n // bn) * split
            if not sub._router_big_fits(cfg, m, n, k):
                continue
            sub._router_descs(st, bn, bk)
            cands.append((f"{cfg} ctas={ctas}", lambda cfg=cfg: sub._router_big_launch(x, st, cfg)))
        run_cands(f"router_big2 m={m}", data, w, lambda: cp1.kernel_fn(data, cst), cands, flush, reps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--only", default="gluon,router_big,router16")
    ap.add_argument("--cold", action="store_true")
    args = ap.parse_args()
    w = weights()
    t0 = time.time()
    sub = load_module("submission.py", "sub")
    st = sub.prepare({k: v.clone() for k, v in w.items()})
    print(f"submission prepare {time.time() - t0:.1f}s (cold triton cache: {args.cold})", flush=True)
    cp1 = load_module(CP1, "cp1")
    cst = cp1.prepare({k: v.clone() for k, v in w.items()})
    flush = torch.empty(FLUSH_BYTES // 4, device="cuda", dtype=torch.float32).normal_()
    only = args.only.split(",")
    for name, fn in (("router16", router16_sweep), ("gluon", gluon_sweep), ("router_big", router_big_sweep),
                     ("gluon2", gluon2_sweep), ("router_big2", router_big2_sweep)):
        if name in only:
            t1 = time.time()
            fn(sub, st, cp1, cst, w, flush, args.reps)
            print(f"[{name}] {time.time() - t1:.0f}s", flush=True)


if __name__ == "__main__":
    main()
