import time
import torch
import reference
from common import weights
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma, warpgroup_mma, warpgroup_mma_wait
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor as GluonDesc

# Replicate Gluon GEMM for arbitrary M
@gluon.jit
def _gluon_gemm(w_desc, x_desc, out_ptr, M, N: gl.constexpr, K: gl.constexpr,
                STAGES: gl.constexpr, num_warps: gl.constexpr):
    BN: gl.constexpr = w_desc.block_type.shape[0]
    BK: gl.constexpr = w_desc.block_type.shape[1]
    BT: gl.constexpr = x_desc.block_type.shape[0]
    KI: gl.constexpr = K // BK
    mma: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[num_warps, 1],
                                                  instr_shape=[16, BT, 16])
    pid_n = gl.program_id(0)
    pid_t = gl.program_id(1)
    n0 = pid_n * BN
    t0 = pid_t * BT
    
    w_bufs = gl.allocate_shared_memory(w_desc.dtype, [STAGES] + w_desc.block_type.shape, w_desc.layout)
    x_bufs = gl.allocate_shared_memory(x_desc.dtype, [STAGES] + x_desc.block_type.shape, x_desc.layout)
    ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(STAGES):
        mbarrier.init(ready.index(i), count=1)
    nbytes: gl.constexpr = w_desc.block_type.nbytes + x_desc.block_type.nbytes
    for i in gl.static_range(STAGES - 1):
        mbarrier.expect(ready.index(i), nbytes)
        tma.async_load(w_desc, [n0, i * BK], ready.index(i), w_bufs.index(i))
        tma.async_load(x_desc, [t0, i * BK], ready.index(i), x_bufs.index(i))
    acc = gl.zeros([BN, BT], gl.float32, layout=mma)
    for i in range(KI):
        s = i % STAGES
        mbarrier.wait(ready.index(s), (i // STAGES) & 1)
        acc = warpgroup_mma(w_bufs.index(s), x_bufs.index(s).permute((1, 0)), acc, is_async=True)
        acc = warpgroup_mma_wait(1, deps=(acc,))
        j = i + STAGES - 1
        sj = j % STAGES
        pred = j < KI
        mbarrier.expect(ready.index(sj), nbytes, pred=pred)
        tma.async_load(w_desc, [n0, j * BK], ready.index(sj), w_bufs.index(sj), pred=pred)
        tma.async_load(x_desc, [t0, j * BK], ready.index(sj), x_bufs.index(sj), pred=pred)
    acc = warpgroup_mma_wait(0, deps=(acc,))
    offs_nl = gl.arange(0, BN, layout=gl.SliceLayout(1, mma))
    offs_t = gl.arange(0, BT, layout=gl.SliceLayout(0, mma))
    out_ptrs = out_ptr + (t0 + offs_t)[None, :] * N + (n0 + offs_nl)[:, None]
    mask = (t0 + offs_t < M)[None, :]
    gl.store(out_ptrs, acc.to(gl.bfloat16), mask=mask)

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
    shapes = [
        ("qkv_gate", 128),
        ("pooler", 128),
        ("unpooler", 128),
        ("o", 128),
        ("qkv_gate", 256),
        ("pooler", 256),
        ("unpooler", 256),
        ("o", 256),
    ]
    
    print("=== Testing Gluon 2D GEMM on m=128 and m=256 ===")
    for layer, m in shapes:
        w = w_dict[layer].cuda().contiguous()
        n, k = w.shape
        x = (torch.randn(m, k, dtype=torch.bfloat16, device="cuda") / (k ** 0.5)).contiguous()
        
        t_base, out_base = bench_op(torch.nn.functional.linear, x, w)
        print(f"\n--- {layer} (M={m}, N={n}, K={k}) | Baseline: {t_base:.2f} us ---")
        
        candidates = [
            (64, 64, 64, 4),
            (64, 64, 64, 6),
            (64, 128, 64, 4),
            (128, 64, 64, 4),
            (128, 128, 64, 3),
        ]
        
        best_t = 1e9
        best_cfg = None
        for bn, bk, bt, stages in candidates:
            if n % bn != 0 or k % bk != 0 or k // bk < stages:
                continue
            smem = stages * (bn + bt) * bk * 2
            if smem > 220 * 1024:
                continue
            try:
                wl = gl.NVMMASharedLayout.get_default_for([bn, bk], gl.bfloat16)
                xl = gl.NVMMASharedLayout.get_default_for([bt, bk], gl.bfloat16)
                wd = GluonDesc.from_tensor(w, [bn, bk], wl)
                xd = GluonDesc.from_tensor(x, [bt, bk], xl)
                out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
                grid = (n // bn, (m + bt - 1) // bt)
                
                def launch():
                    _gluon_gemm[grid](wd, xd, out, m, N=n, K=k, STAGES=stages, num_warps=4)
                    return out
                
                t_gluon, out_g = bench_op(launch)
                err = torch.norm((out_base - out_g).float()) / torch.norm(out_base.float())
                speedup = t_base / t_gluon
                print(f"  bn={bn:3d}, bk={bk:3d}, bt={bt:3d}, st={stages}: {t_gluon:.2f} us | Speedup: {speedup:.3f}x | Diff: {err:.2e}")
                if t_gluon < best_t:
                    best_t = t_gluon
                    best_cfg = (bn, bk, bt, stages)
            except Exception as e:
                pass
        if best_cfg:
            print(f"  => BEST: {best_cfg} @ {best_t:.2f} us ({t_base / best_t:.3f}x)")

if __name__ == "__main__":
    main()
