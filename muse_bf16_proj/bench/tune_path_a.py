import time
import math
import statistics
import torch
import reference
from common import weights, scored_shapes, LOADS, LAYER_ORDER
from submission import prepare, kernel_fn
import baseline

def time_replay(graph, flush):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    flush.sum()
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1e3

class Timed:
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

def bench_full(sub_mod, sub_state, base_mod, base_state, reps=100, overhead=4.0):
    flush = torch.empty(256 * 1024 * 1024 // 4, device="cuda", dtype=torch.float32).normal_()
    load_base = {load: 0.0 for load in LOADS}
    load_mine = {load: 0.0 for load in LOADS}
    
    print(f"{'load':10} {'layer':9} {'m':>4} {'base us':>8} {'mine us':>8} {'speedup':>8}")
    for load, m, layer in scored_shapes():
        data = reference.generate_input(m, layer, 1)
        mine, theirs = Timed(sub_mod, sub_state, data), Timed(base_mod, base_state, data)
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
        s = statistics.median(ratios)
        b = statistics.median(t_base)
        a = b / s
        load_base[load] += b
        load_mine[load] += a
        print(f"{load:10} {layer:9} {m:4d} {b:8.2f} {a:8.2f} {s:7.3f}x")
        
    ratios = [load_base[l] / load_mine[l] for l in LOADS]
    for l, r in zip(LOADS, ratios):
        print(f"load {l:10} base {load_base[l]:7.2f} mine {load_mine[l]:7.2f}  {r:.3f}x")
    geo = math.exp(sum(map(math.log, ratios)) / len(ratios))
    print(f"\n>>> GEOMEAN SPEEDUP: {geo:.4f}x <<<")
    return geo

def main():
    w = weights()
    import submission as sub
    sub_state = sub.prepare({k: v.clone() for k, v in w.items()})
    base_state = baseline.prepare({k: v.clone() for k, v in w.items()})
    bench_full(sub, sub_state, baseline, base_state, reps=100)

if __name__ == "__main__":
    main()
