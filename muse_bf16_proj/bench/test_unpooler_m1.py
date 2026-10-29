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
    print("=== Testing row_gemv on unpooler_slice m=1 ===")
    w_dict = weights()
    state = submission.prepare(w_dict)
    ext = state["ext"]
    w = state["unpooler_slice"] # 1024 x 4096
    n, k = w.shape
    x = (torch.randn(1, k, dtype=torch.bfloat16, device="cuda") / (k ** 0.5)).contiguous()
    
    # 1. cuBLAS baseline
    t_base, out_base = bench_fn(torch.nn.functional.linear, x, w)
    
    # 2. Triton _bf16_gemv currently in submission
    t_curr, out_curr = bench_fn(submission.kernel_fn, {"x": x, "config": {"m": 1, "n": n, "k": k, "layer": "unpooler"}}, state)
    
    print(f"unpooler_slice m=1: cuBLAS = {t_base:.2f} us | Triton = {t_curr:.2f} us")

    # 3. CUDA row_gemv with (rows=2, mt=1, per=2)
    out_cuda = torch.empty(1, n, dtype=torch.bfloat16, device="cuda")
    def run_cuda():
        ext.launch_gemv(x, w, out_cuda, 2, 1, 2)
        return out_cuda
    t_cuda, out_cuda = bench_fn(run_cuda)
    err = torch.norm((out_base - out_cuda).float()) / torch.norm(out_base.float())
    print(f"CUDA row_gemv (rows=2, mt=1, per=2): {t_cuda:.2f} us | Speedup vs Triton: {t_curr/t_cuda:.3f}x | Diff: {err:.2e}")

if __name__ == "__main__":
    main()
