import time
import torch
import reference
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
    print("=== Testing cuBLAS layout and options for m=128, 256 ===")
    w_dict = weights()
    
    for layer in ("o", "unpooler", "qkv_gate", "pooler"):
        w = w_dict[layer].cuda().contiguous()
        n, k = w.shape
        w_t = w.t().contiguous() # Column-major or pre-transposed
        
        for m in (128, 256):
            x = (torch.randn(m, k, dtype=torch.bfloat16, device="cuda") / (k ** 0.5)).contiguous()
            
            # Option 1: torch.nn.functional.linear(x, w) -> x @ w.T
            t_lin, out_lin = bench_fn(torch.nn.functional.linear, x, w)
            
            # Option 2: torch.mm(x, w_t) -> x @ w_t
            t_mm, out_mm = bench_fn(torch.mm, x, w_t)
            
            # Option 3: torch.matmul(x, w.T)
            t_matmul, out_matmul = bench_fn(torch.matmul, x, w.T)
            
            # Option 4: torch.addmm
            out_buf = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
            def run_addmm():
                torch.addmm(out_buf, x, w_t, beta=0.0, alpha=1.0)
                return out_buf
            t_addmm, _ = bench_fn(run_addmm)
            
            print(f"{layer:10s} m={m:3d}: linear={t_lin:5.2f} us | mm(w_t)={t_mm:5.2f} us | matmul={t_matmul:5.2f} us | addmm={t_addmm:5.2f} us")

if __name__ == "__main__":
    main()
