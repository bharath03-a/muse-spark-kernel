import time
import torch
import triton
import triton.language as tl
import submission
from common import weights

@triton.jit
def _split_k_finish(acc, ws_ptrs, ws_stride, out_ptrs, mask, cnt_ptr, pid_s, SPLIT: tl.constexpr):
    tl.store(ws_ptrs + pid_s * ws_stride, acc, mask=mask)
    tl.debug_barrier()
    done = tl.atomic_add(cnt_ptr, 1, sem="acq_rel", scope="gpu")
    if done == SPLIT - 1:
        total = tl.zeros_like(acc)
        for s in tl.static_range(SPLIT):
            total += tl.load(ws_ptrs + s * ws_stride, mask=mask, other=0.0, cache_modifier=".cg")
        tl.store(out_ptrs, total.to(out_ptrs.dtype.element_ty), mask=mask)
        tl.atomic_xchg(cnt_ptr, 0)

@triton.jit
def _bf16_gemv(x_ptr, w_ptr, out_ptr, ws_ptr, cnt_ptr, N: tl.constexpr, K: tl.constexpr,
               SPLIT: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_len: tl.constexpr = K // SPLIT
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for kk in range(0, k_len, BLOCK_K):
        offs_k = tl.max_contiguous(tl.multiple_of(pid_s * k_len + kk + tl.arange(0, BLOCK_K), BLOCK_K), BLOCK_K)
        w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :], eviction_policy="evict_first")
        xv = tl.load(x_ptr + offs_k, eviction_policy="evict_last")
        acc += w.to(tl.float32) * xv.to(tl.float32)[None, :]
    r = tl.sum(acc, axis=1)
    if SPLIT == 1:
        tl.store(out_ptr + offs_n, r.to(out_ptr.dtype.element_ty))
    else:
        _split_k_finish(r, ws_ptr + offs_n, N, out_ptr + offs_n, offs_n < N, cnt_ptr + pid_n, pid_s,
                        SPLIT)

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

def tune_gemv_for(layer, w, ws, cnt):
    n, k = w.shape
    x = (torch.randn(1, k, dtype=torch.bfloat16, device="cuda") / (k ** 0.5)).contiguous()
    out_ref = torch.nn.functional.linear(x, w)
    t_base, _ = bench_fn(torch.nn.functional.linear, x, w)

    print(f"\n=======================================================")
    print(f"Layer {layer}: N={n}, K={k} | Baseline: {t_base:.2f} us")
    print(f"=======================================================")

    best_cfg = None
    best_t = 99999.0

    # Configurations to test: (BLOCK_N, BLOCK_K, SPLIT, num_warps, num_stages)
    candidates = []
    for bn in (1, 2, 4, 8, 16):
        if n % bn != 0:
            continue
        for split in (1, 2, 3, 4, 6, 8):
            if k % split != 0:
                continue
            k_len = k // split
            for bk in (256, 512, 1024, 2048):
                if k_len % bk != 0:
                    continue
                for warps in (4, 8):
                    for stages in (1, 2, 3, 4):
                        candidates.append((bn, bk, split, warps, stages))

    print(f"Total candidates to evaluate: {len(candidates)}")

    for bn, bk, split, warps, stages in candidates:
        if split * bn > 4 * 1024 * 1024 or n // bn > 8192:
            continue
        out = torch.empty(1, n, dtype=torch.bfloat16, device="cuda")
        grid = (n // bn, split)
        
        try:
            def run_k():
                _bf16_gemv[grid](
                    x, w, out, ws, cnt, N=n, K=k, SPLIT=split,
                    BLOCK_N=bn, BLOCK_K=bk, num_warps=warps, num_stages=stages
                )
            t, _ = bench_fn(run_k)
            err = torch.norm((out_ref - out).float()) / torch.norm(out_ref.float())
            if err > 1e-2:
                continue
            if t < best_t:
                best_t = t
                best_cfg = (bn, bk, split, warps, stages)
                print(f"  --> NEW BEST: BN={bn}, BK={bk}, SPLIT={split}, warps={warps}, stages={stages}: {t:.2f} us ({t_base/t:.3f}x)")
        except Exception:
            pass

    print(f"WINNER for {layer}: {best_cfg} -> {best_t:.2f} us ({t_base/best_t:.3f}x)")
    return best_cfg, best_t

def main():
    w_dict = weights()
    ws = torch.empty(4 * 1024 * 1024, device="cuda", dtype=torch.float32)
    cnt = torch.zeros(8192, device="cuda", dtype=torch.int32)

    # 1. pooler (N=4096, K=8192)
    tune_gemv_for("pooler", w_dict["pooler"].cuda().contiguous(), ws, cnt)

    # 2. qkv_gate (N=2304, K=8192)
    tune_gemv_for("qkv_gate", w_dict["qkv_gate"].cuda().contiguous(), ws, cnt)

    # 3. o (N=8192, K=1024)
    tune_gemv_for("o", w_dict["o"].cuda().contiguous(), ws, cnt)

    # 4. unpooler_slice (N=1024, K=4096)
    tune_gemv_for("unpooler_slice", w_dict["unpooler"][:1024].cuda().contiguous(), ws, cnt)

if __name__ == "__main__":
    main()
