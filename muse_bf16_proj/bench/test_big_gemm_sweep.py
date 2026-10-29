import time
import torch
import reference
from common import weights
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
from submission import _big_gemm

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
    ws = torch.empty(1024, device="cuda", dtype=torch.float32)
    cnt = torch.zeros(1024, device="cuda", dtype=torch.int32)
    
    shapes = [
        ("unpooler", 256),
        ("qkv_gate", 256),
        ("o", 256),
        ("unpooler", 128),
        ("o", 128),
    ]
    
    print("=== Testing 2D TMA _big_gemm on m=128, 256 ===")
    for layer, m in shapes:
        w = w_dict[layer].cuda().contiguous()
        n, k = w.shape
        x = (torch.randn(m, k, dtype=torch.bfloat16, device="cuda") / (k ** 0.5)).contiguous()
        t_base, out_base = bench_fn(torch.nn.functional.linear, x, w)
        print(f"\n--- {layer} (M={m}, N={n}, K={k}) | Baseline: {t_base:.2f} us ---")
        
        candidates = [
            (64, 128, 64, 1, 4, 8, 0),
            (64, 64, 64, 1, 4, 8, 0),
            (64, 128, 64, 1, 4, 6, 0),
            (64, 128, 128, 1, 4, 4, 0),
            (32, 128, 64, 1, 4, 6, 0),
            (64, 128, 64, 1, 4, 8, 1),
            (128, 64, 64, 1, 4, 8, 0),
        ]
        
        best_t = 1e9
        best_cfg = None
        for bm, bn, bk, split, warps, stages, swap in candidates:
            if n % bn != 0 or k % bk != 0 or m % bm != 0:
                continue
            try:
                x_desc = TensorDescriptor.from_tensor(x, [bm, bk])
                w_desc = TensorDescriptor.from_tensor(w, [bn, bk])
                out = torch.empty(m, n, device=x.device, dtype=torch.bfloat16)
                grid = (triton.cdiv(m, bm) * (n // bn) * split,)
                
                def launch():
                    _big_gemm[grid](
                        x_desc, w_desc, out, ws, cnt, m, N=n, K=k,
                        SPLIT=split, BM=bm, BN=bn, BK=bk, SWAP=swap, TMA_STORE=0,
                        num_warps=warps, num_stages=stages)
                    return out
                
                t_mine, out_m = bench_fn(launch)
                err = torch.norm((out_base - out_m).float()) / torch.norm(out_base.float())
                speedup = t_base / t_mine
                print(f"  bm={bm:2d}, bn={bn:3d}, bk={bk:3d}, w={warps}, st={stages}, sw={swap}: {t_mine:.2f} us | Speedup: {speedup:.3f}x | Diff: {err:.2e}")
                if t_mine < best_t:
                    best_t = t_mine
                    best_cfg = (bm, bn, bk, split, warps, stages, swap)
            except Exception as e:
                pass
        if best_cfg:
            print(f"  => BEST: {best_cfg} @ {best_t:.2f} us ({t_base / best_t:.3f}x)")

if __name__ == "__main__":
    main()
