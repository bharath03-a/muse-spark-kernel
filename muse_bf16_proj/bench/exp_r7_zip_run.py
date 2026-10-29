"""r7 zip runner: build (+ptxas), exactness, S1 streaming microbench, paired timing vs submission.

    exp_r7_zip_run.py --compile [--src exp_r7_zip_k1]     # CPU: nvcc + ptxas -v
    exp_r7_zip_run.py [--reps 60] [--idx 0,3] [--nosub]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import os
import statistics
import sys
import threading
import time

import torch

from common import load_module, reference

FLAGS = ["-O3", "-gencode=arch=compute_90a,code=sm_90a"]

# (layer, m, config id, ilv)
PLAN = [
    ("pooler", 32, 40, 0), ("pooler", 32, 41, 0), ("pooler", 32, 47, 0), ("pooler", 16, 42, 0),
    ("unpooler", 32, 43, 0), ("unpooler", 16, 44, 0),
    ("qkv_gate", 32, 45, 0), ("qkv_gate", 16, 46, 0),
]
# (layer, m, config id, ilv, k fraction): loads-only kernels on a K prefix give slope (BW) and intercept
S1 = []


def esc_slots(p: dict, w: torch.Tensor, ze: int = 8) -> torch.Tensor:
    """[N, ze, 2] int32 {k or -1, true weight fp32 bits} from the packer's per-row CSR."""
    rp = p["rowptr"].long()
    cnt = rp[1:] - rp[:-1]
    if int(cnt.max()) > ze:
        raise ValueError(f"row with {int(cnt.max())} escapes > {ze} slots")
    n = cnt.numel()
    e = int(rp[-1])
    rows = torch.repeat_interleave(torch.arange(n, device=w.device), cnt)
    pos = torch.arange(e, device=w.device) - rp[rows]
    k = p["esck"][:e].long()
    slots = torch.full((n, ze, 2), -1, dtype=torch.int32, device=w.device)
    slots[:, :, 1] = 0
    slots[rows, pos, 0] = k.to(torch.int32)
    slots[rows, pos, 1] = w[rows, k].float().view(torch.int32)
    return slots.contiguous()


def build(Z, ptxas: bool = False):
    from torch.utils.cpp_extension import load_inline
    src = Z.source()
    flags = FLAGS + (["-Xptxas=-v"] if ptxas else [])
    tag = hashlib.sha1((src + " ".join(FLAGS)).encode()).hexdigest()[:10]
    return load_inline(name=f"muse_z7_{tag}", cpp_sources=Z.ZIP_CPP, cuda_sources=src, functions=["launch_z7"],
                       extra_cuda_cflags=flags, extra_cflags=["-O3"], extra_ldflags=["-lcuda"], verbose=ptxas)


def graph_time(fn, flush, reps, overhead=4.0):
    from harness import time_replay
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return statistics.median(max(time_replay(g, flush) - overhead, 1e-3) for _ in range(reps))


def bench_pair(fa, fb, flush, reps, overhead=4.0):
    from harness import time_replay
    fa(); fb(); torch.cuda.synchronize()
    ga, gb = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(ga):
        fa()
    with torch.cuda.graph(gb):
        fb()
    torch.cuda.synchronize()
    ta, tb = [], []
    for rep in range(reps):
        order = (ga, gb) if rep % 2 == 0 else (gb, ga)
        t = {}
        for g in order:
            t[id(g)] = max(time_replay(g, flush) - overhead, 1e-3)
        ta.append(t[id(ga)]); tb.append(t[id(gb)])
    return statistics.median(ta), statistics.median(tb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="exp_r7_zip_k13")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--idx", default="")
    ap.add_argument("--nosub", action="store_true")
    ap.add_argument("--s1only", action="store_true")
    ap.add_argument("--wd", type=float, default=240)
    args = ap.parse_args()
    wd = threading.Timer(args.wd, lambda: (print("WATCHDOG: exiting", flush=True), os._exit(0)))
    wd.daemon = True
    wd.start()
    Z = importlib.import_module(args.src)
    prb = getattr(Z, "pack_rb", lambda c: Z.CONFIGS[c][0])
    t0 = time.time()
    ext = build(Z, ptxas=args.compile)
    print(f"build {time.time() - t0:.1f}s", flush=True)
    if args.compile or args.build:
        return
    import exp_r6_zip_v2 as P
    from harness import FLUSH_BYTES

    w = reference.generate_weights(reference.WEIGHTS_SEED)
    flush = torch.empty(FLUSH_BYTES // 4, device="cuda", dtype=torch.float32).normal_()
    packs = {}
    for layer, rb in sorted({(l, prb(c)) for l, _, c, _ in PLAN}):
        torch.cuda.synchronize(); t0 = time.time()
        packs[layer, rb] = p = P.pack(w[layer], rb)
        torch.cuda.synchronize()
        p["esc"] = esc_slots(p, w[layer])
        print(f"pack {layer} rb={rb}: {time.time() - t0:.2f}s eb={p['eb']} esc={p['n_esc']} "
              f"maxrow={int((p['rowptr'][1:] - p['rowptr'][:-1]).max())} B/w={p['buf'].numel() / w[layer].numel():.4f}",
              flush=True)

    def run(layer, x, cid, ilv):
        p = packs[layer, prb(cid)]
        out = torch.empty(x.shape[0], w[layer].shape[0], device="cuda", dtype=torch.bfloat16)
        ext.launch_z7(x, p["buf"], out, p["esc"], p["scale"], cid, ilv)
        return out

    plan = [PLAN[int(i)] for i in args.idx.split(",")] if args.idx else PLAN
    if args.s1only:
        plan = []
    print("--- exactness: rel L2 vs fp64, one-hot column readback mismatches", flush=True)
    for layer, m, cid, ilv in plan:
        x = reference.generate_input(m, layer, 3)["x"]
        out = run(layer, x, cid, ilv)
        ref = x.double() @ w[layer].double().t()
        rel = float((out.double() - ref).norm() / ref.norm())
        k = w[layer].shape[1]
        bad = 0
        for b in range(4):
            xo = torch.zeros(m, k, device="cuda", dtype=torch.bfloat16)
            cols = torch.randint(0, k // 64, (m,), device="cuda") * 64 + (b * 16 + torch.arange(m, device="cuda")) % 64
            xo[torch.arange(m), cols] = 1.0
            bad += int((run(layer, xo, cid, ilv) != w[layer][:, cols].t()).sum())
        # escape columns: one-hot on an escaped weight's column must read back exactly too
        pk = packs[layer, prb(cid)]
        rp = pk["rowptr"]
        e = int(rp[100])
        if e < int(rp[-1]):
            xo = torch.zeros(m, k, device="cuda", dtype=torch.bfloat16)
            xo[:, int(pk["esck"][e])] = 1.0
            bad += int((run(layer, xo, cid, ilv) != w[layer][:, int(pk["esck"][e])][None].expand(m, -1)).sum())
        print(f"{layer:9} m={m:3d} id={cid:2d} ilv={ilv} {Z.CONFIGS[cid]}: rel={rel:.2e} mismatch={bad}", flush=True)

    print("--- S1 streaming (cold L2, graph): compressed bytes / time", flush=True)
    for layer, m, cid, ilv, kf in S1:
        xf = reference.generate_input(m, layer, 5)["x"]
        x = xf[:, : xf.shape[1] // kf]
        tt = graph_time(lambda: run(layer, x, cid, ilv), flush, args.reps)
        nb = packs[layer]["buf"].numel() // kf
        print(f"S1 {layer} m={m} id={cid} ilv={ilv} 1/{kf} mode={Z.CONFIGS[cid][-1]}: {tt:6.2f} us  {nb / tt / 1e6:.2f} TB/s "
              f"(raw+4: {tt + 4:.2f} us)", flush=True)

    if args.nosub:
        return
    sub = load_module("submission.py", "sub")
    t0 = time.time()
    state = sub.prepare({k: v.clone() for k, v in w.items()})
    print(f"submission prepare {time.time() - t0:.1f}s", flush=True)
    print("--- paired: submission (A) vs z7 (B), cold L2, overhead 4.0", flush=True)
    for layer, m, cid, ilv in plan:
        data = reference.generate_input(m, layer, 5)
        x = data["x"].clone()
        xs = {"x": x, "config": data["config"]}
        ta, tb = bench_pair(lambda: sub.kernel_fn(xs, state), lambda: run(layer, x, cid, ilv), flush, args.reps)
        print(f"{layer:9} m={m:3d} id={cid:2d} ilv={ilv}: base {ta:6.2f} z7 {tb:6.2f} delta {tb - ta:+.2f} us", flush=True)


if __name__ == "__main__":
    main()
