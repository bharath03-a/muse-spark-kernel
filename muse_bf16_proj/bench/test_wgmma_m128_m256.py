import time
import torch
import reference
from common import weights
from torch.utils.cpp_extension import load_inline
from submission import CUDA_SRC, CUDA_CPP, CUDA_FLAGS, CUDA_LDFLAGS

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
    print("=== Testing WGMMA for m=128 and m=256 on Hopper sm_90a ===")
    ext = load_inline(
        name="test_wgmma_m128_ext",
        cpp_sources=CUDA_CPP,
        cuda_sources=CUDA_SRC,
        functions=["launch_wgmma"],
        extra_cuda_cflags=CUDA_FLAGS,
        extra_cflags=["-O3"],
        extra_ldflags=CUDA_LDFLAGS,
    )
    
    w_dict = weights()
    shapes = [
        ("qkv_gate", 128),
        ("o", 128),
        ("unpooler", 128),
        ("pooler", 128),
        ("qkv_gate", 256),
        ("o", 256),
        ("unpooler", 256),
        ("pooler", 256),
    ]
    
    for layer, m in shapes:
        w = w_dict[layer].cuda().contiguous()
        n, k = w.shape
        x = (torch.randn(m, k, dtype=torch.bfloat16, device="cuda") / (k ** 0.5)).contiguous()
        
        # 1. Baseline: torch.nn.functional.linear(x, w)
        t_base, out_base = bench_fn(torch.nn.functional.linear, x, w)
        
        # 2. Test WGMMA configs: (bn, bt, bk, stages, csize)
        # Note: bt must divide or equal m
        configs = []
        if m == 128:
            configs.append((64, 128, 64, 6, 2))
        elif m == 256:
            configs.append((64, 128, 64, 6, 2))
            
        print(f"\n--- {layer} (M={m}, N={n}, K={k}) | Baseline: {t_base:.2f} us ---")
        for bn, bt, bk, stages, csize in configs:
            try:
                out_wgmma = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
                t_wgmma, _ = bench_fn(ext.launch_wgmma, x, w, out_wgmma, bn, bt, bk, stages, csize)
                err = torch.norm((out_base - out_wgmma).float()) / torch.norm(out_base.float())
                speedup = t_base / t_wgmma
                print(f"  WGMMA (bt={bt}, bk={bk}, st={stages}, c={csize}): {t_wgmma:.2f} us | Speedup: {speedup:.3f}x | Diff: {err:.2e}")
            except Exception as e:
                print(f"  WGMMA (bt={bt}, bk={bk}, st={stages}, c={csize}) failed: {e}")

if __name__ == "__main__":
    main()
