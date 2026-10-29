"""R4: sweep m=128/256 candidates for qkv_gate / o / unpooler against cp5 under cold-L2 graph replay.

    python bench/exp_r4_sweep.py [--reps 40] [--only qkv_gate:128,o:256]
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
import triton

from common import load_module, reference, weights
from harness import FLUSH_BYTES, time_replay

BASE = "submission.py"
OVERHEAD = 4.0

# big: (BM, BN, BK, SPLIT, warps, stages, SWAP, TMA_STORE)
# gl:  (BN, BK, BT, stages, SPLIT, warps)
# cu:  (bn, bt, bk, stages, csize)
CANDS = {
    ("qkv_gate", 128): [
        ("big", (128, 64, 64, 3, 4, 8, 0, 0)), ("big", (128, 64, 64, 4, 4, 6, 0, 0)),
        ("big", (128, 128, 64, 7, 8, 4, 0, 0)), ("big", (128, 128, 64, 4, 8, 4, 0, 0)),
        ("big", (64, 128, 64, 4, 4, 6, 1, 0)), ("big", (128, 64, 64, 11, 4, 3, 0, 0)),
        ("big", (128, 64, 128, 3, 4, 4, 0, 0)),
        ("gl", (64, 128, 128, 4, 3, 4)), ("gl", (64, 64, 128, 8, 3, 4)), ("gl", (64, 64, 128, 8, 4, 4)),
        ("gl", (128, 64, 128, 4, 7, 8)), ("gl", (64, 128, 128, 4, 2, 4)), ("gl", (64, 64, 128, 8, 11, 4)),
        ("cu", (64, 128, 64, 6, 2)),
    ],
    ("qkv_gate", 256): [
        ("big", (128, 64, 64, 2, 4, 8, 0, 0)), ("big", (128, 128, 64, 4, 8, 4, 0, 0)),
        ("big", (128, 128, 64, 3, 8, 4, 0, 0)), ("big", (256, 128, 64, 7, 8, 3, 0, 0)),
        ("big", (256, 64, 64, 3, 8, 4, 0, 0)), ("big", (128, 64, 64, 3, 4, 8, 0, 0)),
        ("gl", (64, 64, 256, 4, 3, 4)), ("gl", (64, 64, 256, 4, 4, 4)), ("gl", (128, 64, 256, 4, 7, 8)),
        ("cu", (64, 128, 64, 6, 2)),
    ],
    ("o", 128): [
        ("big", (128, 64, 64, 1, 4, 8, 0, 0)), ("big", (64, 128, 64, 1, 4, 8, 1, 0)),
        ("big", (128, 128, 64, 1, 8, 8, 0, 0)), ("big", (128, 128, 64, 2, 8, 4, 0, 0)),
        ("big", (64, 64, 64, 1, 4, 8, 1, 0)), ("big", (128, 64, 64, 1, 4, 8, 0, 1)),
        ("big", (128, 64, 128, 1, 4, 4, 0, 0)),
        ("gl", (64, 64, 128, 8, 1, 4)), ("gl", (64, 128, 128, 4, 1, 4)), ("gl", (128, 64, 128, 4, 2, 8)),
        ("gl", (64, 64, 128, 4, 2, 4)),
        ("cu", (64, 128, 64, 6, 2)),
    ],
    ("o", 256): [
        ("big", (128, 128, 64, 1, 8, 8, 0, 0)), ("big", (128, 128, 64, 1, 4, 6, 0, 0)),
        ("big", (128, 64, 64, 1, 4, 8, 0, 0)), ("big", (256, 64, 64, 1, 8, 6, 0, 0)),
        ("big", (64, 128, 64, 1, 4, 8, 1, 0)), ("big", (128, 128, 64, 1, 8, 8, 0, 1)),
        ("big", (128, 256, 64, 1, 8, 4, 0, 0)),
        ("gl", (64, 64, 256, 4, 1, 4)), ("gl", (128, 64, 256, 4, 1, 8)), ("gl", (64, 64, 256, 4, 2, 4)),
    ],
    ("unpooler", 128): [
        ("big", (128, 64, 64, 1, 4, 8, 0, 0)), ("big", (64, 128, 64, 1, 4, 8, 1, 0)),
        ("big", (128, 128, 64, 2, 8, 4, 0, 0)), ("big", (128, 128, 64, 1, 8, 6, 0, 0)),
        ("big", (64, 128, 128, 1, 4, 4, 0, 0)), ("big", (128, 256, 64, 4, 8, 3, 0, 0)),
        ("big", (128, 64, 64, 1, 4, 8, 0, 1)),
        ("gl", (64, 64, 128, 8, 1, 4)), ("gl", (64, 128, 128, 4, 1, 4)), ("gl", (128, 64, 128, 4, 2, 8)),
        ("cu", (64, 128, 64, 6, 2)),
    ],
    ("unpooler", 256): [
        ("big", (128, 128, 64, 1, 8, 6, 0, 0)), ("big", (256, 64, 64, 1, 8, 6, 0, 0)),
        ("big", (128, 128, 64, 1, 4, 6, 0, 0)), ("big", (128, 256, 64, 2, 8, 3, 0, 0)),
        ("big", (128, 128, 64, 1, 8, 6, 0, 1)),
        ("gl", (128, 64, 256, 4, 2, 8)), ("gl", (64, 64, 256, 4, 1, 4)),
    ],
}


def make_fn(sub, state, kind, p, x, layer):
    w = state[layer]
    m, (n, k) = x.shape[0], w.shape
    dev = x.device
    if kind == "cp5":
        data = {"x": x, "config": {"m": m, "n": n, "k": k, "layer": layer}}
        return lambda: sub.kernel_fn(data, state)
    if kind == "cublas":
        return lambda: torch.nn.functional.linear(x, w)
    if kind == "big":
        bm, bn, bk, split, warps, stages, swap, ts = p
        assert n % bn == 0 and k % bk == 0
        assert split == 1 or triton.cdiv(m, bm) * (n // bn) * split * bm * bn <= sub.WS_FLOATS
        wd = sub.TensorDescriptor.from_tensor(w, [bn, bk])

        def f():
            out = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
            tgt = sub.TensorDescriptor.from_tensor(out, [bm, bn]) if ts else out
            sub._big_gemm[(triton.cdiv(m, bm) * (n // bn) * split,)](
                sub.TensorDescriptor.from_tensor(x, [bm, bk]), wd, tgt, state["ws"], state["cnt"], m,
                N=n, K=k, SPLIT=split, BM=bm, BN=bn, BK=bk, SWAP=swap, TMA_STORE=ts,
                num_warps=warps, num_stages=stages)
            return out
        return f
    if kind == "gl":
        bn, bk, bt, stages, split, warps = p
        gl = sub.gl
        assert n % bn == 0 and (k // bk) // split >= stages and bt >= m
        assert split == 1 or (n // bn) * split * bn * bt <= sub.WS_FLOATS
        wl = gl.NVMMASharedLayout.get_default_for([bn, bk], gl.bfloat16)
        wd = sub.GluonDesc.from_tensor(w, [bn, bk], wl)
        xl = gl.NVMMASharedLayout.get_default_for([bt, bk], gl.bfloat16)

        def f():
            out = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
            xd = sub.GluonDesc.from_tensor(x, [bt, bk], xl)
            sub._gluon_gemv[(n // bn, split)](wd, xd, out, state["ws"], state["cnt"], m, N=n, K=k,
                                              STAGES=stages, SPLIT=split, num_warps=warps)
            return out
        return f
    if kind == "cu":
        def f():
            out = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
            state["ext"].launch_wgmma(x, w, out, *p)
            return out
        return f
    raise ValueError(kind)


def graph_of(fn):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    torch.cuda.synchronize()
    return g, out


def run_shape(sub, state, w, layer, m, cands, reps, flush):
    data = reference.generate_input(m, layer, 7)
    x = data["x"].clone()
    ref = torch.nn.functional.linear(x, state[layer]).float()
    entries = []
    for kind, p in [("cp5", None), ("cublas", None)] + cands:
        name = kind if p is None else f"{kind}{p}"
        t0 = time.time()
        try:
            g, out = graph_of(make_fn(sub, state, kind, p, x, layer))
            g.replay()
            torch.cuda.synchronize()
            err = ((out.float() - ref).norm() / ref.norm()).item()
            g.replay()
            torch.cuda.synchronize()
            first = out.clone()
            g.replay()
            torch.cuda.synchronize()
            det = torch.equal(first, out)
        except Exception as exc:
            print(f"  {name}: FAILED {type(exc).__name__}: {str(exc)[:200]}")
            continue
        entries.append([name, g, [], err, det, time.time() - t0])
    for r in range(reps):
        order = entries[r % len(entries):] + entries[: r % len(entries)]
        if r % 2:
            order = order[::-1]
        for e in order:
            e[2].append(time_replay(e[1], flush) - OVERHEAD)
    base = statistics.median(entries[0][2])
    print(f"== {layer} m={m}")
    for name, _, ts, err, det, ct in sorted(entries, key=lambda e: statistics.median(e[2])):
        med = statistics.median(ts)
        paired = statistics.median([a - b for a, b in zip(ts, entries[0][2])])
        print(f"  {med:7.2f} us  d_vs_cp5 {paired:+6.2f}  err {err:.1e} det {int(det)}  build {ct:4.1f}s  {name}")
    del entries
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    t0 = time.time()
    w = weights()
    sub = load_module(BASE, "sub")
    state = sub.prepare({k: v.clone() for k, v in w.items()})
    print(f"prepare {time.time() - t0:.1f}s")
    flush = torch.empty(FLUSH_BYTES // 4, device="cuda", dtype=torch.float32).normal_()
    keys = list(CANDS) if not args.only else [(s.split(":")[0], int(s.split(":")[1]))
                                               for s in args.only.split(",")]
    for layer, m in keys:
        ts = time.time()
        run_shape(sub, state, w, layer, m, CANDS[(layer, m)], args.reps, flush)
        print(f"  [{layer} m={m} took {time.time() - ts:.0f}s]", flush=True)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
