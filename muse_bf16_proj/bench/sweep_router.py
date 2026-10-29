"""Router tile sweep: CUDA-graph replay, cold L2, per m; prints the best tiles and baseline times.

    python bench/sweep_router.py [--ms 1,8,16,32,128,256] [--reps 100]
"""

from __future__ import annotations

import argparse
import itertools
import statistics

import torch

from common import load_module, reference, weights
from harness import FLUSH_BYTES, graph_overhead_us, time_replay


def graph_of(fn):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    torch.cuda.synchronize()
    return g, out


def timed(fn, flush, overhead, reps) -> tuple[float, torch.Tensor]:
    g, out = graph_of(fn)
    times = [time_replay(g, flush) - overhead for _ in range(reps)]
    return statistics.median(times), out


def kernel_us(fn, flush, iters: int = 20) -> str:
    """Per-kernel mean device time per call, excluding the L2 flush."""
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            flush.sum()
            fn()
        torch.cuda.synchronize()
    rows = [(e.key[:40], e.self_device_time_total / iters) for e in prof.key_averages()
            if e.self_device_time_total > 0 and e.count >= iters and "at::native::reduce_kernel" not in e.key]
    return ", ".join(f"{k} {t:.2f}" for k, t in rows)


SMEM = 227 * 1024


def candidates(m: int, sub):
    if m <= 32:
        bt = 16 if m <= 16 else 32
        grid = itertools.product((16, 32, 64), (128, 256), (8, 16, 32), (bt,), (2, 4), (3, 4, 5))
    else:
        grid = itertools.product((32, 64, 128), (64, 128), (4, 8, 16), (64, 128), (4, 8), (2, 3, 4))
    for bn, bk, split, bt, warps, stages in grid:
        if 16384 % (split * bk) or stages * (bn * bk + bk * bt) * 2 > SMEM:
            continue
        if warps == 8 and bn * bt < 64 * 64 or warps == 2 and bn * bt > 32 * 32:
            continue
        ctas = (256 // bn) * split * -(-m // bt)
        if not 96 <= ctas <= 600:
            continue
        yield sub.Tile(m, bn, bk, bt, split, warps, stages)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="submission.py")
    ap.add_argument("--ms", default="1,8,16,32,128,256")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    w = weights()
    sub, base = load_module(args.sub, "sub"), load_module("baseline.py", "base")
    state = sub.prepare({k: v.clone() for k, v in w.items()})
    base_state = base.prepare({k: v.clone() for k, v in w.items()})
    flush = torch.empty(FLUSH_BYTES // 4, device="cuda", dtype=torch.float32).normal_()
    overhead = graph_overhead_us(flush)
    hilo = state["router_hilo"]
    try:
        from sglang.kernels.ops.gemm import router_gemv as sgl_router
    except Exception as exc:
        sgl_router = None
        print("sglang router_gemv unavailable:", exc)

    for m in (int(v) for v in args.ms.split(",")):
        data = reference.generate_input(m, "router", 7)
        x = data["x"]
        t_base, _ = timed(lambda: base.kernel_fn(data, base_state), flush, overhead, args.reps)
        t_cur, _ = timed(lambda: sub.kernel_fn(data, state), flush, overhead, args.reps)
        print(f"m={m}: baseline {t_base:.2f} us [{kernel_us(lambda: base.kernel_fn(data, base_state), flush)}]")
        print(f"m={m}: current TILES {t_cur:.2f} us [{kernel_us(lambda: sub.kernel_fn(data, state), flush)}]")
        if sgl_router is not None:
            x2 = torch.cat([x, x], dim=1)
            try:
                t_sgl, out = timed(lambda: sgl_router.router_gemv(x2, hilo), flush, overhead, args.reps)
                ok, _ = reference.check_implementation(data, out, w)
                print(f"m={m}: sglang router_gemv on hi|lo {t_sgl:.2f} us correct={ok}")
            except Exception as exc:
                print(f"m={m}: sglang router_gemv failed: {type(exc).__name__}: {str(exc)[:120]}")
        results = []
        for tile in candidates(m, sub):
            fresh = {**state, "cnt": torch.zeros_like(state["cnt"])}

            def run(tile=tile, fresh=fresh):
                out = torch.empty(m, 256, device=x.device, dtype=torch.float32)
                sub._launch(x, hilo, out, fresh, tile, kx=x.shape[1])
                return out

            try:
                t, out = timed(run, flush, overhead, args.reps)
            except Exception as exc:
                print(f"  skip {tile}: {type(exc).__name__}: {str(exc)[:120]}")
                continue
            ok, msg = reference.check_implementation(data, out, w)
            if not ok:
                print(f"  WRONG {tile}: {msg}")
                continue
            results.append((t, tile))
        results.sort(key=lambda r: r[0])
        for t, tile in results[: args.top]:
            print(f"  {t:6.2f} us  {tile}")
        if results:
            t, tile = results[0]
            fresh = {**state, "cnt": torch.zeros_like(state["cnt"])}
            out = torch.empty(m, 256, device=x.device, dtype=torch.float32)
            k = kernel_us(lambda: sub._launch(x, hilo, out, fresh, tile, kx=x.shape[1]), flush)
            print(f"BEST m={m} {t:.2f} us [{k}] {tuple(tile)}")


if __name__ == "__main__":
    main()
