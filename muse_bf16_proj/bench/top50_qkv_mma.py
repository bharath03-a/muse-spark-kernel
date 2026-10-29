"""Dense BN32 no-split mma_v2: 72 independent qkv CTAs, no reduction.
Modal: modal run modal_app.py::micro --name top50_qkv_mma
Reference APIs: existing zip_gluon_dev.py mma_v2 + gluon_probe.py TMA ring.
Reports raw CUDA-event graph time; no unsupported subtraction/comparison to judge.
"""
from __future__ import annotations
import statistics
import time
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

@gluon.jit
def dense_qkv(wdesc, xdesc, out, M: gl.constexpr, N: gl.constexpr,
              K: gl.constexpr, D: gl.constexpr):
    BN: gl.constexpr = wdesc.block_type.shape[0]
    BK: gl.constexpr = wdesc.block_type.shape[1]
    BT: gl.constexpr = xdesc.block_type.shape[0]
    mma: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[2, 2], instr_shape=[16, 8])
    ad: gl.constexpr = gl.DotOperandLayout(0, mma, 2)
    bd: gl.constexpr = gl.DotOperandLayout(1, mma, 2)
    wb = gl.allocate_shared_memory(wdesc.dtype, [D, BN, BK], wdesc.layout)
    xb = gl.allocate_shared_memory(xdesc.dtype, [D, BT, BK], xdesc.layout)
    bars = gl.allocate_shared_memory(gl.int64, [D, 1], mbarrier.MBarrierLayout())
    for init_s in gl.static_range(D):
        mbarrier.init(bars.index(init_s), count=1)
    n0 = gl.program_id(0) * BN
    for pre_s in gl.static_range(D):
        mbarrier.expect(bars.index(pre_s), wdesc.block_type.nbytes + xdesc.block_type.nbytes)
        tma.async_load(wdesc, [n0, pre_s * BK], bars.index(pre_s), wb.index(pre_s))
        tma.async_load(xdesc, [0, pre_s * BK], bars.index(pre_s), xb.index(pre_s))
    acc = gl.zeros([BN, BT], gl.float32, mma)
    for i in range(K // BK):
        s = i % D
        mbarrier.wait(bars.index(s), (i // D) & 1)
        a = wb.index(s).load(ad)
        b = xb.index(s).permute((1, 0)).load(bd)
        # All warps finish shared loads before the same slot is refilled.
        gl.barrier()
        j = i + D
        if j < K // BK:
            mbarrier.expect(bars.index(s), wdesc.block_type.nbytes + xdesc.block_type.nbytes)
            tma.async_load(wdesc, [n0, j * BK], bars.index(s), wb.index(s))
            tma.async_load(xdesc, [0, j * BK], bars.index(s), xb.index(s))
        acc = mma_v2(a, b, acc)
    ns = n0 + gl.arange(0, BN, gl.SliceLayout(1, mma))
    ms = gl.arange(0, BT, gl.SliceLayout(0, mma))
    gl.store(out + ms[None, :] * N + ns[:, None], acc.to(gl.bfloat16), ms[None, :] < M)
    for final_s in gl.static_range(D):
        mbarrier.invalidate(bars.index(final_s))


def make(w, x, bk, depth):
    n, k = w.shape
    m = x.shape[0]
    bt = max(16, triton.next_power_of_2(m))
    layout = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    wd = TensorDescriptor(w, [n, k], [k, 1], [32, bk], layout)
    xd = TensorDescriptor(x, [m, k], [k, 1], [bt, bk], layout)
    out = torch.empty(m, n, device='cuda', dtype=torch.bfloat16)
    def run():
        dense_qkv[(n // 32,)](wd, xd, out, m, n, k, depth, num_warps=4)
        return out
    return run


def graph(fn):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    return g, out


def paired(graphs, flush, reps=30):
    times = {key: [] for key in graphs}
    keys = list(graphs)
    for rep in range(reps):
        for key in (keys if rep % 2 == 0 else keys[::-1]):
            flush.fill_(rep)
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            graphs[key].replay()
            b.record()
            b.synchronize()
            times[key].append(a.elapsed_time(b) * 1000)
    return {key: round(statistics.median(v), 3) for key, v in times.items()}


def main():
    t0 = time.monotonic()
    torch.manual_seed(20261027)
    n, k = 2304, 8192
    w = (torch.randn(n, k, device='cuda') / k**0.5).to(torch.bfloat16)
    flush = torch.empty(256 * 2**20 // 4, device='cuda')
    configs = [(128, 4), (256, 4)]
    for m in (16, 32):
        x = torch.randn(m, k, device='cuda').to(torch.bfloat16)
        funcs = {'torch_cublas': lambda: torch.nn.functional.linear(x, w)}
        for bk, d in configs:
            funcs[f'BN32_BK{bk}_D{d}'] = make(w, x, bk, d)
        # FP64 reference, outliers, exact one-hot columns; all use same captured pointers.
        for dist in ('normal', 'outliers', 'onehot'):
            x.normal_()
            if dist == 'outliers':
                x[:, ::97] *= 30
            if dist == 'onehot':
                x.zero_()
                x[torch.arange(m), torch.arange(m) * 97] = 2
            ref = (x.double() @ w.double().T)
            denom = ref.norm(dim=1).clamp_min(ref.norm(dim=1).mean() / 10)
            for key, fn in funcs.items():
                out = fn()
                diff = (out.double() - ref).norm(dim=1) / denom
                rel = (out.double() - ref).norm() / ref.norm()
                print(f'CHECK m={m} {key} {dist} rel={rel.item():.3g} rowmax={diff.max().item():.3g}', flush=True)
                assert torch.isfinite(out).all()
                assert rel <= (0.001 if dist == 'onehot' else 0.01)
                assert diff.max() <= (0.001 if dist == 'onehot' else 0.05)
        x.normal_()
        graphs = {key: graph(fn)[0] for key, fn in funcs.items()}
        print(f'RAW_COLD_EVENT_US m={m} {paired(graphs, flush)}', flush=True)
    print(f'QKV probe total seconds {time.monotonic()-t0:.1f}', flush=True)

if __name__ == '__main__':
    main()
