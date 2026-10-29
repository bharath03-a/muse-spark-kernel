"""Untested dual-token-warpgroup W-reuse probe for unpooler m256.

No compilation or CUDA work on import. run(token_tile=256) uses8 warps;
run(token_tile=128) is the4-warp/two-token-tile comparison. CPU syntax only.
"""
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    mbarrier, tma, warpgroup_mma, warpgroup_mma_wait,
)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


@gluon.jit
def token_warpgroups(wd, xd, out, N: gl.constexpr, K: gl.constexpr,
                     BT: gl.constexpr, STAGES: gl.constexpr):
    BN: gl.constexpr = 64
    BK: gl.constexpr = 64
    NW: gl.constexpr = BT // 128
    gl.static_assert(BT == 128 or BT == 256)
    gl.static_assert(STAGES >= 2 and K // BK >= STAGES - 1)
    mma: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=[4, NW], instr_shape=[16, 128, 16])
    wb = gl.allocate_shared_memory(gl.bfloat16, [STAGES, BN, BK], wd.layout)
    xb = gl.allocate_shared_memory(gl.bfloat16, [STAGES, BT, BK], xd.layout)
    ready = mbarrier.allocate_mbarrier(batch=STAGES)
    for init_s in gl.static_range(STAGES):
        mbarrier.init(ready.index(init_s), count=1)
    n0 = gl.program_id(0) * BN
    t0 = gl.program_id(1) * BT
    NB: gl.constexpr = (BN + BT) * BK * 2
    KI: gl.constexpr = K // BK
    for pre_s in gl.static_range(STAGES - 1):
        mbarrier.expect(ready.index(pre_s), NB)
        tma.async_load(wd, [n0, pre_s * BK], ready.index(pre_s), wb.index(pre_s))
        tma.async_load(xd, [t0, pre_s * BK], ready.index(pre_s), xb.index(pre_s))
    acc = gl.zeros([BN, BT], gl.float32, layout=mma)
    for i in range(KI):
        s = i % STAGES
        mbarrier.wait(ready.index(s), (i // STAGES) & 1)
        acc = warpgroup_mma(wb.index(s), xb.index(s).permute((1, 0)), acc, is_async=True)
        acc = warpgroup_mma_wait(1, deps=(acc,))
        # Both token WGs must retire the old shared stage before CTA-leader refill.
        gl.barrier()
        j = i + STAGES - 1
        sj = j % STAGES
        pred = j < KI
        mbarrier.expect(ready.index(sj), NB, pred=pred)
        tma.async_load(wd, [n0, j * BK], ready.index(sj), wb.index(sj), pred=pred)
        tma.async_load(xd, [t0, j * BK], ready.index(sj), xb.index(sj), pred=pred)
    acc = warpgroup_mma_wait(0, deps=(acc,))
    epi: gl.constexpr = gl.BlockedLayout([2, 1], [32, 1], [1, 4 * NW], [0, 1])
    out_val = gl.convert_layout(acc.to(gl.bfloat16), epi)
    nr = n0 + gl.arange(0, BN, layout=gl.SliceLayout(1, epi))
    mt = t0 + gl.arange(0, BT, layout=gl.SliceLayout(0, epi))
    out_ptrs = out + mt[None, :] * N + nr[:, None]
    gl.store(out_ptrs, out_val)
    gl.barrier()
    for final_s in gl.static_range(STAGES):
        mbarrier.invalidate(ready.index(final_s))


def run(x, w, *, token_tile=256, stages=4, _return_kernel=False):
    if x.shape[0] != 256 or token_tile not in (128, 256):
        raise ValueError('probe requires m256 and token_tile128/256')
    if x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
        raise ValueError('BF16 required')
    n, k = w.shape
    if n % 64 or k % 64 or x.shape[1] != k or stages < 2 or k // 64 < stages - 1:
        raise ValueError('invalid probe geometry')
    if stages * (64 + token_tile) * 64 * 2 + stages * 8 > 220 * 1024:
        raise ValueError('probe exceeds shared-memory budget')
    layout = gl.NVMMASharedLayout(128, 16)
    wd = TensorDescriptor.from_tensor(w, [64, 64], layout)
    xd = TensorDescriptor.from_tensor(x, [token_tile, 64], layout)
    out = torch.empty((256, n), device=x.device, dtype=torch.bfloat16)
    compiled = token_warpgroups[(n // 64, 256 // token_tile)](
        wd, xd, out, N=n, K=k, BT=token_tile, STAGES=stages,
        num_warps=4 * (token_tile // 128))
    return (out, compiled) if _return_kernel else out
