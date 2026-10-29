import time
import torch
import reference
from common import weights
from torch.utils.cpp_extension import load_inline
from submission import CUDA_SRC, CUDA_CPP, CUDA_FLAGS, CUDA_LDFLAGS, prepare, kernel_fn

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
    print("=== Testing WGMMA vs Gluon for m in (8, 16, 32) on Hopper ===")
    ext = load_inline(
        name="test_wgmma_decode_ext",
        cpp_sources=CUDA_CPP,
        cuda_sources=CUDA_SRC,
        functions=["launch_wgmma"],
        extra_cuda_cflags=CUDA_FLAGS,
        extra_cflags=["-O3"],
        extra_ldflags=CUDA_LDFLAGS,
    )
    
    w_dict = weights()
    state = prepare({k: v.clone() for k, v in w_dict.items()})
    
    # Layers to test
    test_cases = [
        ("pooler", 8),
        ("pooler", 16),
        ("pooler", 32),
        ("unpooler", 16),
        ("unpooler", 32),
        ("o", 8),
        ("o", 16),
        ("o", 32),
    ]
    
    for layer, m in test_cases:
        w = state[layer]
        n, k = w.shape
        x = (torch.randn(m, k, dtype=torch.bfloat16, device="cuda") / (k ** 0.5)).contiguous()
        
        # 1. Baseline
        t_base, out_base = bench_fn(torch.nn.functional.linear, x, w)
        
        # 2. Checkpoint 4 (currently Gluon)
        data = {"x": x, "config": {"m": m, "n": n, "k": k, "layer": layer}}
        t_cp4, out_cp4 = bench_fn(kernel_fn, data, state)
        
        # 3. Custom WGMMA
        # Try candidate configs
        configs = []
        if m == 8:
            configs = [(64, 8, 128, 8, 4)]
        elif m == 16:
            configs = [(64, 16, 128, 4, 4)]
        elif m == 32:
            configs = [(64, 32, 128, 4, 4)]
            
        print(f"\n--- {layer} (m={m}, N={n}, K={k}) ---")
        print(f"  Baseline: {t_base:.2f} us")
        print(f"  Checkpoint 4 (Gluon): {t_cp4:.2f} us (Speedup: {t_base / t_cp4:.3f}x)")
        
        for bn, bt, bk, stages, csize in configs:
            try:
                out_wgmma = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
                t_wgmma, _ = bench_fn(ext.launch_wgmma, x, w, out_wgmma, bn, bt, bk, stages, csize)
                err = torch.norm((out_base - out_wgmma).float()) / torch.norm(out_base.float())
                speedup = t_base / t_wgmma
                delta = t_cp4 - t_wgmma
                print(f"  Custom WGMMA (bt={bt}, bk={bk}, c={csize}): {t_wgmma:.2f} us (Speedup: {speedup:.3f}x, Delta vs CP4: {delta:+.2f} us) | Diff: {err:.2e}")
            except Exception as e:
                print(f"  Custom WGMMA failed: {e}")

if __name__ == "__main__":
    main()
