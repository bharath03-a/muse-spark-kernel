"""Unmeasured Hopper W-multicast feasibility probe, Triton 3.8.

Paired variants must use the SAME BN/BK/STAGES/CTAS and input tensors.
The conservative per-iteration cluster barrier protects remote ring reuse.
This is deliberately not a canonical candidate or a fast peer-credit ring.
No compilation, CUDA work, or timing takes place at import.
"""
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    cluster, mbarrier, tma, warpgroup_mma, warpgroup_mma_wait,
)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


@gluon.jit
def weight_multicast(wd, xd, out, M: gl.constexpr, N: gl.constexpr,
                     K: gl.constexpr, BN: gl.constexpr, BT: gl.constexpr,
                     BK: gl.constexpr, STAGES: gl.constexpr,
                     CTAS: gl.constexpr, MULTICAST: gl.constexpr):
    gl.static_assert(CTAS == 2 or CTAS == 4)
    gl.static_assert(STAGES >= 2)
    gl.static_assert(K % BK == 0)
    gl.static_assert(K // BK >= STAGES - 1)
    CGA: gl.constexpr = [[0, 1]] if CTAS == 2 else [[0, 1], [0, 2]]
    mma: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=[4, 1],
        instr_shape=[16, BT // CTAS, 16], cga_layout=CGA)
    wbuf = gl.allocate_shared_memory(gl.bfloat16, [STAGES, BN, BK], wd.layout)
    xbuf = gl.allocate_shared_memory(gl.bfloat16, [STAGES, BT, BK], xd.layout)
    ready = mbarrier.allocate_mbarrier(batch=STAGES)
    for init_s in gl.static_range(STAGES):
        mbarrier.init(ready.index(init_s), count=1)
    mbarrier.fence_init_release_cluster()
    cluster.barrier(relaxed=True)
    n0 = gl.program_id(0) * BN
    t0 = gl.program_id(1) * BT
    NB: gl.constexpr = wd.nbytes_per_cta + xd.nbytes_per_cta
    KI: gl.constexpr = K // BK
    for pre_s in gl.static_range(STAGES - 1):
        mbarrier.expect(ready.index(pre_s), NB)
        tma.async_load(wd, [n0, pre_s * BK], ready.index(pre_s), wbuf.index(pre_s), multicast=MULTICAST)
        tma.async_load(xd, [t0, pre_s * BK], ready.index(pre_s), xbuf.index(pre_s))
    acc = gl.zeros([BN, BT], gl.float32, layout=mma)
    for i in range(KI):
        s = i % STAGES
        mbarrier.wait(ready.index(s), (i // STAGES) & 1)
        acc = warpgroup_mma(wbuf.index(s), xbuf.index(s).permute((1, 0)), acc, is_async=True)
        acc = warpgroup_mma_wait(1, deps=(acc,))
        # Every peer has retired i-1 before any peer can overwrite its slot.
        # The current i MMA can remain outstanding across this barrier.
        cluster.barrier()
        j = i + STAGES - 1
        sj = j % STAGES
        pred = j < KI
        mbarrier.expect(ready.index(sj), NB, pred=pred)
        tma.async_load(wd, [n0, j * BK], ready.index(sj), wbuf.index(sj),
                       pred=pred, multicast=MULTICAST)
        tma.async_load(xd, [t0, j * BK], ready.index(sj), xbuf.index(sj), pred=pred)
    acc = warpgroup_mma_wait(0, deps=(acc,))
    epi: gl.constexpr = gl.BlockedLayout([2, 1], [32, 1], [1, 4], [0, 1], cga_layout=CGA)
    out_val = gl.convert_layout(acc.to(gl.bfloat16), epi)
    nr = n0 + gl.arange(0, BN, layout=gl.SliceLayout(1, epi))
    mt = t0 + gl.arange(0, BT, layout=gl.SliceLayout(0, epi))
    out_ptrs = out + mt[None, :] * N + nr[:, None]
    out_mask = (mt < M)[None, :] & (nr < N)[:, None]
    gl.store(out_ptrs, out_val, mask=out_mask)
    # No CTA exits while peers can still target its shared memory/barriers.
    cluster.barrier()
    for final_s in gl.static_range(STAGES):
        mbarrier.invalidate(ready.index(final_s))


def run(x, w, *, multicast=True, bn=64, bk=128, stages=4, ctas=2, _return_kernel=False):
    m, k = x.shape
    n = w.shape[0]
    if ctas not in (2, 4) or stages < 2 or k // bk < stages - 1:
        raise ValueError("invalid cluster/ring geometry")
    if x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
        raise ValueError("probe requires BF16 inputs and weights")
    if stages * (bn + m // ctas) * bk * 2 + stages * 8 > 220 * 1024:
        raise ValueError("probe exceeds conservative shared-memory budget")
    if m not in (128, 256) or n % bn or k % bk:
        raise ValueError("probe supports exact m128/256 and divisible N/K only")
    bt = m
    wcga = [[0, 0]] if ctas == 2 else [[0, 0], [0, 0]]
    xcga = [[1, 0]] if ctas == 2 else [[1, 0], [2, 0]]
    wl = gl.NVMMASharedLayout(128, 16, cga_layout=wcga)
    xl = gl.NVMMASharedLayout(128, 16, cga_layout=xcga)
    wd = TensorDescriptor.from_tensor(w, [bn, bk], wl)
    xd = TensorDescriptor.from_tensor(x, [bt, bk], xl)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    compiled = weight_multicast[(triton.cdiv(n, bn), 1)](
        wd, xd, out, M=m, N=n, K=k, BN=bn, BT=bt, BK=bk,
        STAGES=stages, CTAS=ctas, MULTICAST=multicast,
        num_warps=4, num_ctas=ctas)
    return (out, compiled) if _return_kernel else out


def prepare(weights):
    return dict(weights)


def kernel_fn(data, state):
    """Default multicast probe; use run(...multicast=False) as paired control."""
    x, cfg = data["x"], data["config"]
    w = state[cfg["layer"]][:cfg["n"]]
    if cfg["layer"] == "router":
        return x.float() @ w.T
    if x.shape[0] in (128, 256):
        return run(x, w)
    return x @ w.T


def resource_estimate(m, *, bn=64, bk=128, stages=4, ctas=2):
    """Lower bounds; inspect compiled n_regs/n_spills/shared and PTX as well."""
    return dict(shared_bytes_per_cta=stages * (bn + m // ctas) * bk * 2 + stages * 8,
                accumulator_fp32_per_thread=bn * (m // ctas) // 128,
                required_ptx="cp.async.bulk.tensor.*.multicast::cluster",
                multicast_weight_requests_per_cluster=1,
                unicast_weight_requests_per_cluster=ctas,
                note="unicast control performs CTAS reads; HBM may serve repeats from L2")
