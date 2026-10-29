"""Local replica of the judge's timing: CUDA-graph replays, cold L2, paired ratios vs baseline.

    python bench/harness.py [--sub submission.py] [--base baseline.py] [--reps 200] [--only layer]
"""

from __future__ import annotations

import argparse
import math
import statistics
import time

import torch

from common import LOADS, load_module, reference, scored_shapes, weights

FLUSH_BYTES = 256 * 2**20  # several times H200's L2


class Timed:
    """One side's graph of kernel_fn on a static input buffer."""

    def __init__(self, mod, state, data):
        self.x = data["x"].clone()
        self.data = {"x": self.x, "config": data["config"]}
        for _ in range(3):
            mod.kernel_fn(self.data, state)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = mod.kernel_fn(self.data, state)
        torch.cuda.synchronize()


def graph_overhead_us(flush, reps: int = 200) -> float:
    dummy = torch.zeros(1, device="cuda")
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        dummy.add_(1)
    return statistics.median(time_replay(g, flush) for _ in range(reps))


def time_replay(graph, flush) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    flush.sum()  # reads the buffer, evicting L2 like the judge
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1e3


def bench_shape(sub, sub_state, base, base_state, m, layer, reps, flush, overhead):
    data = reference.generate_input(m, layer, 1)
    mine, theirs = Timed(sub, sub_state, data), Timed(base, base_state, data)
    ratios, t_base = [], []
    for rep in range(reps):
        fresh = reference.generate_input(m, layer, 1000 + rep)["x"]
        mine.x.copy_(fresh)
        theirs.x.copy_(fresh)
        order = (mine, theirs) if rep % 2 == 0 else (theirs, mine)
        times = {}
        for side in order:
            times[id(side)] = max(time_replay(side.graph, flush) - overhead, 1e-3)
        a, b = times[id(mine)], times[id(theirs)]
        t_base.append(b)
        ratios.append(b / a)
    speedup = statistics.median(ratios)
    base_med = statistics.median(t_base)
    return base_med, base_med / speedup, speedup


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="submission.py")
    ap.add_argument("--base", default="baseline.py")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--only", default="")
    # The measured cost includes the probe graph's own tiny kernel (~1.2 us more than the judge subtracts).
    ap.add_argument("--overhead", type=float, default=4.0, help="fixed us to subtract; negative = measure")
    args = ap.parse_args()

    w = weights()
    sub, base = load_module(args.sub, "sub"), load_module(args.base, "base")
    t0 = time.time()
    sub_state = sub.prepare({k: v.clone() for k, v in w.items()})
    print(f"{args.sub} prepare {time.time() - t0:.1f}s")
    base_state = base.prepare({k: v.clone() for k, v in w.items()})
    flush = torch.empty(FLUSH_BYTES // 4, device="cuda", dtype=torch.float32).normal_()
    overhead = graph_overhead_us(flush) if args.overhead < 0 else args.overhead
    print(f"graph replay fixed cost ~{overhead:.2f} us (subtracted)")

    load_base = {load: 0.0 for load in LOADS}
    load_mine = {load: 0.0 for load in LOADS}
    print(f"{'load':10} {'layer':9} {'m':>4} {'base us':>8} {'mine us':>8} {'speedup':>8} {'sol':>6}")
    for load, m, layer in scored_shapes():
        if args.only and layer != args.only:
            continue
        b, a, s = bench_shape(sub, sub_state, base, base_state, m, layer, args.reps, flush, overhead)
        sol = reference.speed_of_light_ns(m, layer, {"hbm_bytes_per_s": 4.8e12,
                                                     "bf16_dense_flops_per_s": 989e12}) / 1e3
        load_base[load] += b
        load_mine[load] += a
        print(f"{load:10} {layer:9} {m:4d} {b:8.2f} {a:8.2f} {s:7.3f}x {sol:6.2f}")
    if not args.only:
        ratios = [load_base[l] / load_mine[l] for l in LOADS]
        for l, r in zip(LOADS, ratios):
            print(f"load {l:10} base {load_base[l]:7.2f} mine {load_mine[l]:7.2f}  {r:.3f}x")
        geo = math.exp(sum(map(math.log, ratios)) / len(ratios))
        print(f"GEOMEAN SPEEDUP {geo:.3f}x")


if __name__ == "__main__":
    main()
