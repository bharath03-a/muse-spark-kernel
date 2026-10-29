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
    print("=== Testing WGMMA for unpooler and pooler at m=8, 16, 32 ===")
    w_dict = weights()
    state = submission.prepare(w_dict)
    ext = state["ext"]
    assert ext is not None, "CUDA ext not compiled"

    for layer in ("unpooler", "pooler"):
        w = w_dict[layer].cuda().contiguous()
        n, k = w.shape
        for m in (8, 16, 32):
            x = (torch.randn(m, k, dtype=torch.bfloat16, device="cuda") / (k ** 0.5)).contiguous()
            
            # 1. cuBLAS baseline
            t_base, out_base = bench_fn(torch.nn.functional.linear, x, w)
            
            # 2. Current submission (e.g. Gluon)
            t_curr, out_curr = bench_fn(submission.kernel_fn, {"x": x, "config": {"m": m, "n": n, "k": k, "layer": layer}}, state)
            
            # 3. WGMMA candidate
            out_wgmma = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
            cfg = None
            if m == 8:
                cfg = (64, 8, 128, 8, 4)
            elif m == 16:
                cfg = (64, 16, 128, 4, 4)
            elif m == 32:
                cfg = (64, 32, 128, 4, 4)
            
            try:
                def run_wgmma(x, w, out):
                    ext.launch_wgmma(x, w, out, *cfg)
                    return out
                t_wgmma, out_wgmma = bench_fn(run_wgmma, x, w, out_wgmma)
                err = torch.norm((out_base - out_wgmma).float()) / torch.norm(out_base.float())
                print(f"{layer:10s} m={m:2d}: cuBLAS={t_base:5.2f} us | curr={t_curr:5.2f} us | WGMMA={t_wgmma:5.2f} us | vs curr: {t_curr/t_wgmma:.2f}x | Diff={err:.2e}")
            except Exception as e:
                print(f"{layer:10s} m={m:2d}: WGMMA FAILED: {e}")

if __name__ == "__main__":
    main()
