"""GPU-UNVALIDATED: cp14 Gluon ring, outstanding-Q1 versus Q2.
No canonical kernels modified. --syntax needs no torch/Triton/GPU.
Run manually only with authorized GPU budget:
 python bench/expert_ring_probe.py --layer pooler --m 32
Raw cold CUDA-event graph times, no graph-overhead subtraction.
"""
from __future__ import annotations
import argparse
import ast
import pathlib
import statistics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--syntax', action='store_true')
parser.add_argument('--layer', choices=['pooler', 'unpooler'], default='pooler')
parser.add_argument('--m', type=int, choices=[16, 32], default=32)
parser.add_argument('--bk', type=int, choices=[128, 256], default=128)
parser.add_argument('--stages', type=int, default=8)
parser.add_argument('--reps', type=int, default=30)
args = parser.parse_args()
if args.syntax:
    ast.parse(pathlib.Path(__file__).read_text())
    print('Python syntax PASS; GPU compile, correctness and performance UNVALIDATED')
    raise SystemExit(0)

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma, warpgroup_mma, warpgroup_mma_wait
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor as GluonDesc

@gluon.jit
def ring(w_desc, x_desc, out_ptr, M: gl.constexpr, N: gl.constexpr,
         K: gl.constexpr, STAGES: gl.constexpr, Q: gl.constexpr,
         num_warps: gl.constexpr):
    BN: gl.constexpr = w_desc.block_type.shape[0]
    BK: gl.constexpr = w_desc.block_type.shape[1]
    BT: gl.constexpr = x_desc.block_type.shape[0]
    KI: gl.constexpr = K // BK
    mma: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[num_warps, 1], instr_shape=[16, BT, 16])
    n0 = gl.program_id(0) * BN
    wb = gl.allocate_shared_memory(w_desc.dtype, [STAGES] + w_desc.block_type.shape, w_desc.layout)
    xb = gl.allocate_shared_memory(x_desc.dtype, [STAGES] + x_desc.block_type.shape, x_desc.layout)
    ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    nbytes: gl.constexpr = w_desc.block_type.nbytes + x_desc.block_type.nbytes
    for init_s in gl.static_range(STAGES):
        mbarrier.init(ready.index(init_s), count=1)
    # Q outstanding MMAs need Q protected slots. Remaining slots are TMA lead.
    for pre_s in gl.static_range(STAGES - Q):
        mbarrier.expect(ready.index(pre_s), nbytes)
        tma.async_load(w_desc, [n0, pre_s * BK], ready.index(pre_s), wb.index(pre_s))
        tma.async_load(x_desc, [0, pre_s * BK], ready.index(pre_s), xb.index(pre_s))
    acc = gl.zeros([BN, BT], gl.float32, layout=mma)
    for i in range(KI):
        s = i % STAGES
        mbarrier.wait(ready.index(s), (i // STAGES) & 1)
        acc = warpgroup_mma(wb.index(s), xb.index(s).permute((1, 0)), acc, is_async=True)
        acc = warpgroup_mma_wait(Q, deps=(acc,))
        # This slot formerly held i-Q, whose MMA is now complete.
        j = i + STAGES - Q
        sj = j % STAGES
        pred = j < KI
        mbarrier.expect(ready.index(sj), nbytes, pred=pred)
        tma.async_load(w_desc, [n0, j * BK], ready.index(sj), wb.index(sj), pred=pred)
        tma.async_load(x_desc, [0, j * BK], ready.index(sj), xb.index(sj), pred=pred)
    acc = warpgroup_mma_wait(0, deps=(acc,))
    ns = n0 + gl.arange(0, BN, layout=gl.SliceLayout(1, mma))
    ts = gl.arange(0, BT, layout=gl.SliceLayout(0, mma))
    gl.store(out_ptr + ts[None, :] * N + ns[:, None], acc.to(gl.bfloat16), mask=(ts < M)[None, :])


def capture(fn):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    return graph, out


def main():
    torch.manual_seed(20261026)
    n, k = (4096, 8192) if args.layer == 'pooler' else (8192, 4096)
    m, bn, bk, stages = args.m, 64, args.bk, args.stages
    assert stages > 2 and k // bk >= stages
    smem = stages * (bn + m) * bk * 2 + stages * 8
    if smem > 227 * 1024:
        raise ValueError(f'Ring needs {smem} bytes; choose fewer stages/BK')
    w = (torch.randn(n, k, device='cuda') / k**0.5).to(torch.bfloat16)
    x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
    # Exactly cp14 _gluon_wdesc / _gluon_xlayout conventions.
    wl = gl.NVMMASharedLayout.get_default_for([bn, bk], gl.bfloat16)
    xl = gl.NVMMASharedLayout.get_default_for([m, bk], gl.bfloat16)
    wd = GluonDesc.from_tensor(w, [bn, bk], wl)
    xd = GluonDesc.from_tensor(x, [m, bk], xl)
    outs = {q: torch.empty(m, n, device='cuda', dtype=torch.bfloat16) for q in [1, 2]}
    def run(q):
        ring[(n // bn,)](wd, xd, outs[q], m, n, k, stages, q, num_warps=4)
        return outs[q]
    # Compilation happens before capture/timing. Print resources for diagnosis.
    for q in [1, 2]:
        compiled = ring[(n // bn,)](wd, xd, outs[q], m, n, k, stages, q, num_warps=4)
        print('COMPILE', q, 'registers', compiled.n_regs, 'spills', compiled.n_spills,
              'shared', compiled.metadata.shared, flush=True)
    graphs = {q: capture(lambda q=q: run(q))[0] for q in [1, 2]}
    previous = {}
    for dist in ['normal', 'outliers', 'onehot', 'changed_input']:
        x.normal_()
        if dist == 'outliers':
            x[:, ::97] *= 30
        if dist == 'onehot':
            x.zero_()
            x[torch.arange(m), torch.arange(m) * 97] = 2
        ref = x.double() @ w.double().T
        denom = ref.norm(dim=1).clamp_min(ref.norm(dim=1).mean() / 10)
        for q in [1, 2]:
            for repeat in range(3):
                graphs[q].replay()
                out = outs[q].double()
                rel = (out - ref).norm() / ref.norm()
                row = (out - ref).norm(dim=1) / denom
                assert torch.isfinite(out).all()
                assert rel <= (0.001 if dist == 'onehot' else 0.01)
                assert row.max() <= (0.001 if dist == 'onehot' else 0.05)
            if dist == 'changed_input':
                assert not torch.equal(previous[q], outs[q]), 'stale output'
            previous[q] = outs[q].clone()
            print('CHECK', dist, q, float(rel), float(row.max()), flush=True)
    flush = torch.empty(256 * 2**20 // 4, device='cuda', dtype=torch.float32)
    times = {q: [] for q in [1, 2]}
    events = {q: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for q in [1, 2]}
    for rep in range(args.reps):
        for q in ([1, 2] if rep % 2 == 0 else [2, 1]):
            flush.fill_(rep)
            a, b = events[q]
            a.record()
            graphs[q].replay()
            b.record()
            b.synchronize()
            times[q].append(a.elapsed_time(b) * 1000)
    print('RAW_COLD_GRAPH_US', {q: statistics.median(v) for q, v in times.items()})
    print('MEDIAN_PAIRED_Q1_OVER_Q2', statistics.median(a / b for a, b in zip(times[1], times[2])))

if __name__ == '__main__':
    main()
