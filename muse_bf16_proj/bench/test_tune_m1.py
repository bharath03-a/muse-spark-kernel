import time
import torch
import reference
from common import weights
import triton
import triton.language as tl

@triton.jit
def _bf16_gemv_test(x_ptr, w_ptr, out_ptr, N: tl.constexpr, K: tl.constexpr,
                    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for kk in range(0, K, BLOCK_K):
        offs_k = tl.max_contiguous(tl.multiple_of(kk + tl.arange(0, BLOCK_K), BLOCK_K), BLOCK_K)
        w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :], eviction_policy="evict_first")
        xv = tl.load(x_ptr + offs_k, eviction_policy="evict_last")
        acc += w.to(tl.float32) * xv.to(tl.float32)[None, :]
    r = tl.sum(acc, axis=1)
    tl.store(out_ptr + offs_n, r.to(out_ptr.dtype.element_ty))

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
    w_dict = weights()
    for layer in ("pooler", "qkv_gate", "o"):
        w = w_dict[layer].cuda().contiguous()
        n, k = w.shape
        x = (torch.randn(1, k, dtype=torch.bfloat16, device="cuda") / (k ** 0.5)).contiguous()
        t_base, out_base = bench_fn(torch.nn.functional.linear, x, w)
        print(f"\n--- {layer} (N={n}, K={k}) | Baseline: {t_base:.2f} us ---")
        
        candidates = [
            (2, 1024, 4, 2),
            (2, 1024, 4, 3),
            (4, 1024, 4, 2),
            (4, 1024, 4, 3),
            (4, 1024, 4, 4),
            (8, 1024, 4, 2),
            (8, 1024, 4, 3),
            (8, 512, 4, 3),
            (16, 256, 8, 2),
            (16, 256, 8, 3),
            (16, 512, 8, 2),
        ]
        
        best_t = 1e9
        best_cfg = None
        for bn, bk, warps, stages in candidates:
            if n % bn != 0 or k % bk != 0:
                continue
            try:
                out = torch.empty(1, n, dtype=torch.bfloat16, device="cuda")
                def launch():
                    _bf16_gemv_test[(n // bn,)](x, w, out, N=n, K=k, BLOCK_N=bn, BLOCK_K=bk,
                                                num_warps=warps, num_stages=stages)
                    return out
                t, out_t = bench_fn(launch)
                err = torch.norm((out_base - out_t).float()) / torch.norm(out_base.float())
                speedup = t_base / t
                print(f"  bn={bn:2d}, bk={bk:4d}, w={warps}, st={stages}: {t:.2f} us | {speedup:.3f}x | Diff: {err:.2e}")
                if t < best_t:
                    best_t = t
                    best_cfg = (bn, bk, warps, stages)
            except Exception as e:
                pass
        if best_cfg:
            print(f"  => BEST: {best_cfg} @ {best_t:.2f} us ({t_base / best_t:.3f}x)")

if __name__ == "__main__":
    main()
