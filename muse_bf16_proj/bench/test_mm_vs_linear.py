import time
import torch
from common import weights

def bench_fn(fn, *args, reps=100):
    for _ in range(5):
        fn(*args)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn(*args)
    torch.cuda.synchronize()
    flush = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    times = []
    for _ in range(reps):
        flush.fill_(1)
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        g.replay()
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0 - 4.0)
    times.sort()
    return times[len(times) // 2]

def main():
    w_dict = weights()
    for layer in ("qkv_gate", "unpooler", "o", "pooler"):
        w = w_dict[layer].cuda().contiguous()
        w_t = w.t().contiguous()
        for m in (128, 256):
            x = torch.randn(m, w.shape[1], dtype=torch.bfloat16, device="cuda")
            t_linear = bench_fn(torch.nn.functional.linear, x, w)
            t_mm = bench_fn(torch.mm, x, w_t)
            print(f"{layer} m={m}: F.linear={t_linear:.2f} us | torch.mm(x, w_t)={t_mm:.2f} us | diff={t_linear - t_mm:+.2f} us")

if __name__ == "__main__":
    main()
