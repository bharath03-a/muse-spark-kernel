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
    print("=== Testing WGMMA <128, 64, 6, 1, 2> on unpooler, qkv_gate, o at m=128 ===")
    w_dict = weights()
    state = submission.prepare(w_dict)
    ext = state["ext"]
    assert ext is not None, "CUDA extension not available"

    for layer in ("unpooler", "qkv_gate", "o"):
        w = w_dict[layer].cuda().contiguous()
        n, k = w.shape
        x = (torch.randn(128, k, dtype=torch.bfloat16, device="cuda") / (k ** 0.5)).contiguous()
        
        # cuBLAS baseline
        t_base, out_base = bench_fn(torch.nn.functional.linear, x, w)
        
        # WGMMA kernel
        out_wgmma = torch.empty(128, n, dtype=torch.bfloat16, device="cuda")
        def run_wgmma():
            ext.launch_wgmma(x, w, out_wgmma, 64, 128, 64, 6, 2)
            return out_wgmma
        
        try:
            t_wgmma, _ = bench_fn(run_wgmma)
            err = torch.norm((out_base - out_wgmma).float()) / torch.norm(out_base.float())
            speedup = t_base / t_wgmma
            print(f"{layer:10s} m=128: cuBLAS = {t_base:5.2f} us | WGMMA = {t_wgmma:5.2f} us | Speedup = {speedup:.3f}x | Diff = {err:.2e}")
        except Exception as e:
            print(f"{layer:10s} m=128: FAILED: {e}")

if __name__ == "__main__":
    main()
