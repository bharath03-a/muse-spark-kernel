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
    print("=== Testing o layer Gluon configurations on Modal H200 ===")
    w_dict = weights()
    state = submission.prepare(w_dict)
    w = w_dict["o"].cuda().contiguous()
    n, k = w.shape

    for m in (8, 16, 32):
        x = (torch.randn(m, k, dtype=torch.bfloat16, device="cuda") / (k ** 0.5)).contiguous()
        t_base, out_base = bench_fn(torch.nn.functional.linear, x, w)
        t_curr, out_curr = bench_fn(submission.kernel_fn, {"x": x, "config": {"m": m, "n": n, "k": k, "layer": "o"}}, state)
        print(f"\n--- o m={m} | cuBLAS: {t_base:.2f} us | Current: {t_curr:.2f} us ---")

        for bk in (128, 256):
            for stages in (4, 6, 8):
                try:
                    gt = submission.GTile(m, m, 64, bk, m, stages, split=1)
                    if not submission._gluon_fits(gt, n, k):
                        continue
                    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
                    def run_g():
                        submission._gluon_launch(x, out, state, "o", gt)
                        return out
                    t, _ = bench_fn(run_g)
                    err = torch.norm((out_base - out).float()) / torch.norm(out_base.float())
                    print(f"  BK={bk}, stages={stages}: {t:.2f} us | vs curr: {t_curr/t:.3f}x | Diff: {err:.2e}")
                except Exception as e:
                    pass

if __name__ == "__main__":
    main()
