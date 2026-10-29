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
    print("=== Comparing linear(x, w) vs mm(x, w_t) across all layers and m ===")
    w_dict = weights()
    
    layers = ["qkv_gate", "o", "pooler", "unpooler"]
    m_values = [1, 8, 16, 32, 128, 256]
    
    for layer in layers:
        w = w_dict[layer].cuda().contiguous()
        n, k = w.shape
        w_t = w.t().contiguous()
        
        print(f"\n--- Layer: {layer} (N={n}, K={k}) ---")
        for m in m_values:
            curr_n = 1024 if layer == "unpooler" and m <= 8 else n
            curr_w = w[:curr_n]
            curr_wt = curr_w.t().contiguous()
            x = (torch.randn(m, k, dtype=torch.bfloat16, device="cuda") / (k ** 0.5)).contiguous()
            
            t_lin, out_lin = bench_fn(torch.nn.functional.linear, x, curr_w)
            t_mm, out_mm = bench_fn(torch.mm, x, curr_wt)
            
            err = torch.norm((out_lin - out_mm).float()) / torch.norm(out_lin.float())
            speedup = t_lin / t_mm
            print(f"  m={m:3d}: linear={t_lin:5.2f} us | mm(w_t)={t_mm:5.2f} us | speedup={speedup:.3f}x | Diff={err:.2e}")

if __name__ == "__main__":
    main()
