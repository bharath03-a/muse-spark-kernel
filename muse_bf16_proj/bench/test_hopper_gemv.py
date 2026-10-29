import time
import torch
import reference
from common import weights
from sglang.kernels.ops.gemm.hopper_bf16_gemv import hopper_bf16_gemv, use_hopper_bf16_gemv
from submission import prepare, kernel_fn

def bench_op(fn, *args, reps=100):
    # Warmup
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
    w = weights()
    state = prepare({k: v.clone() for k, v in w.items()})
    
    print("=== Testing SGLang hopper_bf16_gemv vs Checkpoint 4 on m=1 ===")
    for layer in ("pooler", "qkv_gate", "o", "unpooler"):
        w_layer = state[layer]
        if layer == "unpooler":
            w_layer = w_layer[:1024]
        n, k = w_layer.shape
        x = torch.randn(1, k, dtype=torch.bfloat16, device="cuda")
        
        can_use = use_hopper_bf16_gemv(1, n, k)
        print(f"\nLayer {layer} (N={n}, K={k}) | use_hopper_bf16_gemv: {can_use}")
        
        # Checkpoint 4
        data = {"x": x, "config": {"m": 1, "n": n, "k": k, "layer": layer}}
        t_cp4, out_cp4 = bench_op(kernel_fn, data, state)
        print(f"Checkpoint 4: {t_cp4:.2f} us")
        
        # Hopper BF16 GEMV
        try:
            t_sgl, out_sgl = bench_op(hopper_bf16_gemv, x, w_layer)
            err = torch.norm((out_cp4 - out_sgl).float()) / torch.norm(out_cp4.float())
            print(f"hopper_bf16_gemv: {t_sgl:.2f} us | Rel Diff: {err:.2e}")
        except Exception as e:
            print(f"hopper_bf16_gemv failed: {e}")

if __name__ == "__main__":
    main()
