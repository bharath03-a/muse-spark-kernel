"""One-shot MoE check: correctness of the new Triton NVFP4 path + CUDA-graph timing vs Marlin (cold L2)."""
import torch

import moe_reference as ref
import moe_sub_new as new
import moe_sub_old as old

SHAPES = [(1, "normal"), (16, "normal"), (32, "normal"), (8, "normal"), (128, "normal"), (256, "normal"),
          (4, "normal"), (33, "normal"), (257, "normal"), (128, "skewed")]


def graph_time(fn, data, state, flush, iters=30):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn(data, state)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn(data, state)
    ts = []
    for _ in range(iters):
        flush.zero_()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize()
        ts.append(a.elapsed_time(b) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    w = ref.generate_weights(ref.WEIGHTS_SEED)
    st_new, st_old = new.prepare(w), old.prepare(w)
    flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    for bs, dist in SHAPES:
        data = ref.generate_input(bs, "ep8", 1234 + bs, dist=dist)
        out = new.kernel_fn(data, st_new)
        torch.cuda.synchronize()
        ok, msg = ref.check_implementation(data, out, w)
        t_new = graph_time(new.kernel_fn, data, st_new, flush)
        t_old = graph_time(old.kernel_fn, data, st_old, flush)
        print(f"bs={bs:4d} {dist:7s} ok={ok} {str(msg)[:60]:60s} new={t_new:8.1f}us marlin={t_old:8.1f}us "
              f"speedup={t_old / t_new:5.2f}x", flush=True)


main()
