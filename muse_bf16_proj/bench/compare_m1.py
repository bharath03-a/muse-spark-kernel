import time
import torch
import reference
import submission
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
    return times[len(times) // 2], out

def main():
    print("=== Comparing m=1 implementations on H200 ===")
    w_dict = weights()
    state_sub = submission.prepare(w_dict)

    for layer in ("pooler", "qkv_gate", "o", "unpooler"):
        w = w_dict[layer].cuda().contiguous()
        n, k = w.shape
        x = (torch.randn(1, k, dtype=torch.bfloat16, device="cuda") / (k ** 0.5)).contiguous()
        
        # 1. Reference baseline (cuBLAS)
        t_base, out_base = bench_fn(torch.nn.functional.linear, x, w)
        
        # 2. Submission current (Triton / CUDA)
        t_sub, out_sub = bench_fn(submission.kernel_fn, {"x": x, "config": {"m": 1, "n": n, "k": k, "layer": layer}}, state_sub)
        
        print(f"{layer:15s} (1x{n}x{k}): cuBLAS = {t_base:6.2f} us | submission = {t_sub:6.2f} us | Speedup = {t_base/t_sub:.3f}x")

if __name__ == "__main__":
    main()
