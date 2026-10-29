import time
import torch
import reference
import submission
from common import weights
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma, warpgroup_mma, warpgroup_mma_wait
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor as GluonDesc

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
    print("=== Testing pooler Gluon split=2 vs split=1 on Modal H200 ===")
    w_dict = weights()
    state = submission.prepare(w_dict)
    w = w_dict["pooler"].cuda().contiguous()
    n, k = w.shape

    for m in (8, 16, 32):
        x = (torch.randn(m, k, dtype=torch.bfloat16, device="cuda") / (k ** 0.5)).contiguous()
        
        # Baseline cuBLAS
        t_base, out_base = bench_fn(torch.nn.functional.linear, x, w)
        
        # Current submission (split=1)
        t_curr, out_curr = bench_fn(submission.kernel_fn, {"x": x, "config": {"m": m, "n": n, "k": k, "layer": "pooler"}}, state)
        
        print(f"\n--- pooler m={m} | cuBLAS: {t_base:.2f} us | Current (split=1): {t_curr:.2f} us ---")

        # Test Gluon with split=2
        # GTile(min_m, max_m, block_n, block_k, block_t, stages, split=2)
        for stages in (4, 6, 8):
            for bk in (64, 128):
                if k % (bk * 2) != 0:
                    continue
                try:
                    gt = submission.GTile(m, m, 64, bk, m, stages, split=2)
                    if not submission._gluon_fits(gt, n, k):
                        continue
                    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
                    def run_g():
                        submission._gluon_launch(x, out, state, "pooler", gt)
                        return out
                    t, _ = bench_fn(run_g)
                    err = torch.norm((out_base - out).float()) / torch.norm(out_base.float())
                    print(f"  split=2 (BK={bk}, stages={stages}): {t:.2f} us | vs curr: {t_curr/t:.3f}x | Diff: {err:.2e}")
                except Exception as e:
                    print(f"  split=2 (BK={bk}, stages={stages}) FAILED: {e}")

if __name__ == "__main__":
    main()
