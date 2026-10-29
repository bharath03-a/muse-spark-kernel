import time
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
from common import weights

# Import or define _big_gemm
@triton.jit(do_not_specialize=["M"])
def _big_gemm(x_desc, w_desc, out_ref, ws_ptr, cnt_ptr, M,
              N: tl.constexpr, K: tl.constexpr, SPLIT: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
              SWAP: tl.constexpr, TMA_STORE: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    pid_m = pid % num_m
    pid_s = (pid // num_m) % SPLIT
    pid_n = pid // (num_m * SPLIT)
    KI: tl.constexpr = K // BK
    k0 = pid_s * KI // SPLIT
    k1 = (pid_s + 1) * KI // SPLIT

    if SWAP:
        acc = tl.zeros((BN, BM), dtype=tl.float32)
    else:
        acc = tl.zeros((BM, BN), dtype=tl.float32)

    for ki in range(k0, k1):
        a = x_desc.load([pid_m * BM, ki * BK])
        b = w_desc.load([pid_n * BN, ki * BK])
        if SWAP:
            acc = tl.dot(b, a.T, acc)
        else:
            acc = tl.dot(a, b.T, acc)

    if SPLIT == 1:
        if TMA_STORE:
            val = acc.to(tl.bfloat16)
            if SWAP:
                val = val.T
            out_ref.store([pid_m * BM, pid_n * BN], val)
        else:
            local_m = tl.arange(0, BM)
            local_n = tl.arange(0, BN)
            offs_m = pid_m * BM + local_m
            offs_n = pid_n * BN + local_n
            if SWAP:
                out_idx = offs_m[None, :] * N + offs_n[:, None]
                mask = (offs_m < M)[None, :]
            else:
                out_idx = offs_m[:, None] * N + offs_n[None, :]
                mask = (offs_m < M)[:, None]
            tl.store(out_ref + out_idx, acc.to(tl.bfloat16), mask=mask)
    else:
        TILE_ELEMS: tl.constexpr = BM * BN
        local_m = tl.arange(0, BM)
        local_n = tl.arange(0, BN)
        if SWAP:
            acc_store = acc.T
        else:
            acc_store = acc
        tile_id = pid_n * num_m + pid_m
        ws_tile = ws_ptr + tile_id * (SPLIT * TILE_ELEMS) + local_m[:, None] * BN + local_n[None, :]
        tl.store(ws_tile + pid_s * TILE_ELEMS, acc_store)
        tl.debug_barrier()
        cnt = cnt_ptr + tile_id
        done = tl.atomic_add(cnt, 1, sem="acq_rel", scope="gpu")
        if done == SPLIT - 1:
            if SPLIT == 2:
                total = acc_store + tl.load(ws_tile + (1 - pid_s) * TILE_ELEMS, cache_modifier=".cg")
            else:
                total = tl.zeros((BM, BN), dtype=tl.float32)
                for s in tl.static_range(SPLIT):
                    total += tl.load(ws_tile + s * TILE_ELEMS, cache_modifier=".cg")
            offs_m = pid_m * BM + local_m
            offs_n = pid_n * BN + local_n
            out_idx = offs_m[:, None] * N + offs_n[None, :]
            mask = (offs_m < M)[:, None]
            tl.store(out_ref + out_idx, total.to(tl.bfloat16), mask=mask)
            tl.atomic_xchg(cnt, 0)

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

def tune_layer_m(layer, m, w_dict):
    w = w_dict[layer].cuda().contiguous()
    n, k = w.shape
    x = (torch.randn(m, k, dtype=torch.bfloat16, device="cuda") / (k ** 0.5)).contiguous()
    
    t_base, out_base = bench_fn(torch.nn.functional.linear, x, w)
    print(f"\n=== {layer} m={m} (N={n}, K={k}) | cuBLAS baseline: {t_base:.2f} us ===")

    ws = torch.empty(4 * 1024 * 1024, device="cuda", dtype=torch.float32)
    cnt = torch.zeros(8192, device="cuda", dtype=torch.int32)

    # Search candidates: (BM, BN, BK, SPLIT, warps, stages, SWAP, TMA_STORE)
    candidates = [
        # (BM, BN, BK, SPLIT, warps, stages, SWAP, TMA_STORE)
        (128, 64, 64, 1, 4, 4, 0, 0),
        (128, 64, 64, 1, 4, 8, 0, 0),
        (64, 128, 64, 1, 4, 4, 0, 0),
        (64, 128, 64, 1, 4, 8, 0, 0),
        (64, 128, 64, 1, 4, 8, 1, 0),
        (128, 128, 64, 1, 4, 4, 0, 0),
        (128, 128, 64, 1, 4, 6, 0, 0),
        (128, 128, 64, 1, 8, 4, 0, 0),
        (128, 64, 64, 2, 4, 6, 0, 0),
        (128, 64, 64, 2, 4, 8, 0, 0),
        (64, 128, 64, 2, 4, 6, 1, 0),
        (64, 128, 64, 2, 4, 8, 1, 0),
        (64, 64, 128, 1, 4, 4, 0, 0),
        (64, 64, 128, 1, 4, 6, 0, 0),
        (128, 64, 128, 1, 4, 4, 0, 0),
    ]

    best_cfg = None
    best_t = t_base

    for cfg in candidates:
        bm, bn, bk, split, warps, stages, swap, tma_store = cfg
        if n % bn != 0 or k % bk != 0:
            continue
        if k % (bk * split) != 0:
            continue
        try:
            x_desc = TensorDescriptor.from_tensor(x, [bm, bk])
            w_desc = TensorDescriptor.from_tensor(w, [bn, bk])
            out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)

            grid = (triton.cdiv(m, bm) * (n // bn) * split,)
            def run_kernel():
                _big_gemm[grid](
                    x_desc, w_desc, out, ws, cnt, m,
                    N=n, K=k, SPLIT=split, BM=bm, BN=bn, BK=bk,
                    SWAP=swap, TMA_STORE=tma_store,
                    num_warps=warps, num_stages=stages
                )
            
            t, _ = bench_fn(run_kernel)
            err = torch.norm((out_base - out).float()) / torch.norm(out_base.float())
            if err > 1e-2:
                print(f"  CFG {cfg}: ERR {err:.2e} TOO HIGH")
                continue
            speedup = t_base / t
            print(f"  CFG {cfg}: {t:5.2f} us | Speedup: {speedup:.3f}x | Diff: {err:.2e}")
            if t < best_t:
                best_t = t
                best_cfg = cfg
        except Exception as e:
            pass

    if best_cfg is not None:
        print(f"  ==> WINNER for {layer} m={m}: {best_cfg} ({best_t:.2f} us, {t_base/best_t:.3f}x)")
    else:
        print(f"  ==> cuBLAS remains fastest ({t_base:.2f} us)")

def main():
    w_dict = weights()
    # Scored shapes at m=128 and m=256
    tune_layer_m("o", 128, w_dict)
    tune_layer_m("o", 256, w_dict)
    tune_layer_m("unpooler", 128, w_dict)
    tune_layer_m("unpooler", 256, w_dict)
    tune_layer_m("qkv_gate", 128, w_dict)
    tune_layer_m("qkv_gate", 256, w_dict)
    tune_layer_m("pooler", 128, w_dict)
    tune_layer_m("pooler", 256, w_dict)

if __name__ == "__main__":
    main()
