import time
import torch
import reference
from common import weights
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma, warpgroup_mma, warpgroup_mma_wait
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor as GluonDesc

@gluon.jit
def _kfold_issue(w_desc, x_desc, w_bufs, x_bufs, bar, s, i, n0, pred,
                 R: gl.constexpr, BT: gl.constexpr, BK: gl.constexpr, Q: gl.constexpr):
    NBYTES: gl.constexpr = Q * (w_desc.block_type.nbytes + x_desc.block_type.nbytes)
    mbarrier.expect(bar, NBYTES, pred=pred)
    for q in gl.static_range(Q):
        k_off = (Q * i + q) * BK
        tma.async_load(w_desc, [n0, k_off], bar, w_bufs.index(s).slice(q * R, R, dim=0), pred=pred)
        tma.async_load(x_desc, [0, k_off], bar, x_bufs.index(s).slice(q * BT, BT, dim=0), pred=pred)

@gluon.jit
def _gluon_gemv_kfold(w_desc, x_desc, out_ptr, M, N: gl.constexpr, K: gl.constexpr,
                      STAGES: gl.constexpr, Q: gl.constexpr, num_warps: gl.constexpr):
    R: gl.constexpr = w_desc.block_type.shape[0]
    BK: gl.constexpr = w_desc.block_type.shape[1]
    BT: gl.constexpr = x_desc.block_type.shape[0]
    BN: gl.constexpr = R * Q
    KI: gl.constexpr = K // (BK * Q)
    mma: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[num_warps, 1],
                                                  instr_shape=[16, Q * BT, 16])
    n0 = gl.program_id(0) * R
    w_bufs = gl.allocate_shared_memory(w_desc.dtype, [STAGES, BN, BK], w_desc.layout)
    x_bufs = gl.allocate_shared_memory(x_desc.dtype, [STAGES, Q * BT, BK], x_desc.layout)
    ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(STAGES):
        mbarrier.init(ready.index(i), count=1)
    for i in gl.static_range(STAGES - 1):
        _kfold_issue(w_desc, x_desc, w_bufs, x_bufs, ready.index(i), i, i, n0, i < KI, R, BT, BK, Q)
    acc = gl.zeros([BN, Q * BT], gl.float32, layout=mma)
    for i in range(KI):
        s = i % STAGES
        mbarrier.wait(ready.index(s), (i // STAGES) & 1)
        acc = warpgroup_mma(w_bufs.index(s), x_bufs.index(s).permute((1, 0)), acc, is_async=True)
        acc = warpgroup_mma_wait(1, deps=(acc,))
        j = i + STAGES - 1
        sj = j % STAGES
        _kfold_issue(w_desc, x_desc, w_bufs, x_bufs, ready.index(sj), sj, j, n0, j < KI, R, BT, BK, Q)
    acc = warpgroup_mma_wait(0, deps=(acc,))
    TW: gl.constexpr = 32 if BT >= 32 else BT
    epi: gl.constexpr = gl.BlockedLayout([1, BT // TW], [32 // TW, TW], [num_warps, 1], [1, 0])
    red = gl.allocate_shared_memory(gl.float32, [BN, Q * BT],
                                    gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0]))
    red.store(acc)
    gl.barrier()
    tot = gl.zeros([R, BT], gl.float32, layout=epi)
    for q in gl.static_range(Q):
        tot += red.slice(q * R, R, dim=0).slice(q * BT, BT, dim=1).load(epi)
    offs_n = n0 + gl.arange(0, R, layout=gl.SliceLayout(1, epi))
    offs_t = gl.arange(0, BT, layout=gl.SliceLayout(0, epi))
    gl.store(out_ptr + offs_t[None, :] * N + offs_n[:, None], tot.to(gl.bfloat16),
             mask=(offs_t < M)[None, :])

def bench_op(fn, *args, reps=50):
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
    print("=== Testing K-Fold GEMV on pooler m=8 and qkv_gate m=8 ===")
    for layer in ("pooler", "qkv_gate"):
        w = w_dict[layer].cuda().contiguous()
        n, k = w.shape
        m = 8
        x = (torch.randn(m, k, dtype=torch.bfloat16, device="cuda") / (k ** 0.5)).contiguous()
        t_base, out_base = bench_op(torch.nn.functional.linear, x, w)
        print(f"\n--- {layer} (m=8, N={n}, K={k}) | Baseline: {t_base:.2f} us ---")
        
        # Test Q=2, R in (32, 64), BK in (128, 256), stages in (4, 6)
        candidates = [
            (32, 128, 4, 2),
            (32, 256, 4, 2),
            (64, 128, 4, 2),
            (32, 128, 6, 2),
        ]
        for r, bk, stages, q in candidates:
            bn, qbt = r * q, m * q
            smem = stages * (bn + qbt) * bk * 2 + bn * qbt * 4
            if smem > 220 * 1024 or n % r != 0 or k % (bk * q) != 0:
                continue
            try:
                wl = gl.NVMMASharedLayout.get_default_for([r, bk], gl.bfloat16)
                xl = gl.NVMMASharedLayout.get_default_for([m, bk], gl.bfloat16)
                wd = GluonDesc.from_tensor(w, [r, bk], wl)
                xd = GluonDesc.from_tensor(x, [m, bk], xl)
                out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
                
                def launch():
                    _gluon_gemv_kfold[(n // r,)](wd, xd, out, m, N=n, K=k, STAGES=stages, Q=q, num_warps=4)
                    return out
                
                t_kfold, out_k = bench_op(launch)
                err = torch.norm((out_base - out_k).float()) / torch.norm(out_base.float())
                speedup = t_base / t_kfold
                print(f"  R={r:2d}, BK={bk:3d}, st={stages}, Q={q}: {t_kfold:.2f} us | Speedup: {speedup:.3f}x | Diff: {err:.2e}")
            except Exception as e:
                pass

if __name__ == "__main__":
    main()
