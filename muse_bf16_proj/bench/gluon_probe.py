"""Probe: does Gluon (Triton 3.8) work on sm_90 for our swap-AB GEMV, and does it beat _bf16_gemm?

    modal run modal_app.py::micro --name gluon_probe

Variants (out^T[BN, BT] = W[BN, K] @ x^T, W rows on the wgmma M side, tokens on N):
  ws     1 TMA producer warp + 1 consumer warpgroup (warp_specialize), STAGES-deep mbarrier ring
  pipe   same ring, single partition, prefetch distance STAGES-1
  rega   pipe, but W goes smem -> registers (DotOperandLayout) -> wgmma A from registers
Times are cold-L2 CUDA-graph replays of one kernel, against submission._bf16_gemm on the same shape.
"""

from __future__ import annotations

import statistics
import sys
import traceback

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared, mbarrier, tma, warpgroup_mma, warpgroup_mma_wait)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor as GDesc

sys.path.insert(0, "/root/proj")
import submission as sub  # noqa: E402

FLUSH = None


@gluon.jit
def _producer(w_desc, x_desc, w_bufs, x_bufs, ready, empty, n0, KI: gl.constexpr, BK: gl.constexpr):
    S: gl.constexpr = w_bufs.type.shape[0]
    for i in range(KI):
        s = i % S
        ph = (i // S) & 1
        mbarrier.wait(empty.index(s), ph ^ 1)
        mbarrier.expect(ready.index(s), w_desc.block_type.nbytes + x_desc.block_type.nbytes)
        tma.async_load(w_desc, [n0, i * BK], ready.index(s), w_bufs.index(s))
        tma.async_load(x_desc, [0, i * BK], ready.index(s), x_bufs.index(s))


@gluon.jit
def _consumer(w_bufs, x_bufs, ready, empty, KI: gl.constexpr, BN: gl.constexpr, BT: gl.constexpr,
              mma: gl.constexpr):
    S: gl.constexpr = w_bufs.type.shape[0]
    acc = gl.zeros([BN, BT], gl.float32, layout=mma)
    for i in range(KI):
        s = i % S
        mbarrier.wait(ready.index(s), (i // S) & 1)
        acc = warpgroup_mma(w_bufs.index(s), x_bufs.index(s).permute((1, 0)), acc, is_async=True)
        acc = warpgroup_mma_wait(1, deps=(acc,))
        mbarrier.arrive(empty.index((i + S - 1) % S), pred=i > 0)
    acc = warpgroup_mma_wait(0, deps=(acc,))
    return acc


@gluon.jit
def _epilogue(acc, out_ptr, M, n0, N: gl.constexpr, BN: gl.constexpr, BT: gl.constexpr, mma: gl.constexpr):
    offs_n = n0 + gl.arange(0, BN, layout=gl.SliceLayout(1, mma))
    offs_t = gl.arange(0, BT, layout=gl.SliceLayout(0, mma))
    ptrs = out_ptr + offs_t[None, :] * N + offs_n[:, None]
    gl.store(ptrs, acc.to(gl.bfloat16), mask=(offs_t < M)[None, :])


@gluon.jit
def gemv_ws(w_desc, x_desc, out_ptr, M, N: gl.constexpr, K: gl.constexpr, STAGES: gl.constexpr,
            num_warps: gl.constexpr):
    BN: gl.constexpr = w_desc.block_type.shape[0]
    BK: gl.constexpr = w_desc.block_type.shape[1]
    BT: gl.constexpr = x_desc.block_type.shape[0]
    KI: gl.constexpr = K // BK
    mma: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[num_warps, 1],
                                                  instr_shape=[16, BT, 16])
    n0 = gl.program_id(0) * BN
    w_bufs = gl.allocate_shared_memory(w_desc.dtype, [STAGES] + w_desc.block_type.shape, w_desc.layout)
    x_bufs = gl.allocate_shared_memory(x_desc.dtype, [STAGES] + x_desc.block_type.shape, x_desc.layout)
    ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    empty = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(STAGES):
        mbarrier.init(ready.index(i), count=1)
        mbarrier.init(empty.index(i), count=1)
    acc = gl.warp_specialize([
        (_consumer, (w_bufs, x_bufs, ready, empty, KI, BN, BT, mma)),
        (_producer, (w_desc, x_desc, w_bufs, x_bufs, ready, empty, n0, KI, BK)),
    ], [1], [24])
    _epilogue(acc, out_ptr, M, n0, N, BN, BT, mma)


@gluon.jit
def gemv_pipe(w_desc, x_desc, out_ptr, M, N: gl.constexpr, K: gl.constexpr, STAGES: gl.constexpr,
              REG_A: gl.constexpr, num_warps: gl.constexpr):
    BN: gl.constexpr = w_desc.block_type.shape[0]
    BK: gl.constexpr = w_desc.block_type.shape[1]
    BT: gl.constexpr = x_desc.block_type.shape[0]
    KI: gl.constexpr = K // BK
    mma: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[num_warps, 1],
                                                  instr_shape=[16, BT, 16])
    a_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=2)
    n0 = gl.program_id(0) * BN
    w_bufs = gl.allocate_shared_memory(w_desc.dtype, [STAGES] + w_desc.block_type.shape, w_desc.layout)
    x_bufs = gl.allocate_shared_memory(x_desc.dtype, [STAGES] + x_desc.block_type.shape, x_desc.layout)
    ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(STAGES):
        mbarrier.init(ready.index(i), count=1)
    nbytes: gl.constexpr = w_desc.block_type.nbytes + x_desc.block_type.nbytes
    for i in gl.static_range(STAGES - 1):
        mbarrier.expect(ready.index(i), nbytes)
        tma.async_load(w_desc, [n0, i * BK], ready.index(i), w_bufs.index(i))
        tma.async_load(x_desc, [0, i * BK], ready.index(i), x_bufs.index(i))
    acc = gl.zeros([BN, BT], gl.float32, layout=mma)
    for i in range(KI):
        s = i % STAGES
        mbarrier.wait(ready.index(s), (i // STAGES) & 1)
        if REG_A:
            a = w_bufs.index(s).load(a_layout)
            acc = warpgroup_mma(a, x_bufs.index(s).permute((1, 0)), acc, is_async=True)
        else:
            acc = warpgroup_mma(w_bufs.index(s), x_bufs.index(s).permute((1, 0)), acc, is_async=True)
        acc = warpgroup_mma_wait(1, deps=(acc,))
        # wgmma i-1 has retired, so its slot ((i - 1) % STAGES) can be refilled with tile i + STAGES - 1.
        j = i + STAGES - 1
        sj = j % STAGES
        pred = j < KI
        mbarrier.expect(ready.index(sj), nbytes, pred=pred)
        tma.async_load(w_desc, [n0, j * BK], ready.index(sj), w_bufs.index(sj), pred=pred)
        tma.async_load(x_desc, [0, j * BK], ready.index(sj), x_bufs.index(sj), pred=pred)
    acc = warpgroup_mma_wait(0, deps=(acc,))
    _epilogue(acc, out_ptr, M, n0, N, BN, BT, mma)


def _descs(w, x, bn, bk, bt):
    wl = gl.NVMMASharedLayout.get_default_for([bn, bk], gl.bfloat16)
    xl = gl.NVMMASharedLayout.get_default_for([bt, bk], gl.bfloat16)
    return GDesc.from_tensor(w, [bn, bk], wl), GDesc.from_tensor(x, [bt, bk], xl)


def launch_gluon(kind, w, x, out, bn, bk, bt, stages, warps):
    n, k = w.shape
    wd, xd = _descs(w, x, bn, bk, bt)
    grid = (n // bn,)
    if kind == "ws":
        gemv_ws[grid](wd, xd, out, x.shape[0], N=n, K=k, STAGES=stages, num_warps=warps, maxnreg=168)
    else:
        gemv_pipe[grid](wd, xd, out, x.shape[0], N=n, K=k, STAGES=stages, REG_A=kind == "rega",
                        num_warps=warps)


def launch_triton(key, w, x, out, ws, cnt):
    tile = sub._bf16_pick(key, x.shape[0])
    state = {"ws": ws, "cnt": cnt, "desc": {}}
    if tile.kind == "tma":
        state["desc"][(key, tile)] = sub.TensorDescriptor.from_tensor(w, [tile.block_n, tile.block_k])
    sub._bf16_launch(x, w, out, state, key, tile)
    return tile


def time_graph(fn, reps=60):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        FLUSH.sum()
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)
    return statistics.median(ts)


def main():
    global FLUSH
    FLUSH = torch.empty(256 * 2**20 // 4, device="cuda", dtype=torch.float32)
    torch.manual_seed(0)
    ws = torch.empty(4 * 2**20, device="cuda", dtype=torch.float32)
    cnt = torch.zeros(8192, device="cuda", dtype=torch.int32)
    shapes = [("pooler", 4096, 8192, 16), ("unpooler", 8192, 4096, 16), ("pooler", 4096, 8192, 32),
              ("qkv_gate", 2304, 8192, 16)]
    variants = [("ws", 64, 128, 8, 4), ("ws", 64, 64, 16, 4), ("pipe", 64, 128, 8, 4),
                ("rega", 64, 128, 8, 4), ("ws", 128, 64, 10, 8)]
    for layer, n, k, m in shapes:
        w = (torch.randn(n, k, device="cuda") / k**0.5).to(torch.bfloat16)
        x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
        ref = x.float() @ w.float().T
        gb = n * k * 2 / 1e3
        out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        tile = launch_triton(layer, w, x, out, ws, cnt)
        t = time_graph(lambda: launch_triton(layer, w, x, out, ws, cnt))
        print(f"{layer} m={m} triton {tile} {t:.2f} us {gb / t:.2f} GB/s", flush=True)
        for kind, bn, bk, st, warps in variants:
            bt = max(16, m)
            tag = f"{layer} m={m} gluon {kind} BN{bn} BK{bk} S{st} w{warps}"
            try:
                o = torch.zeros(m, n, device="cuda", dtype=torch.bfloat16)
                launch_gluon(kind, w, x, o, bn, bk, bt, st, warps)
                torch.cuda.synchronize()
                err = ((o.float() - ref).norm() / ref.norm()).item()
                t = time_graph(lambda: launch_gluon(kind, w, x, o, bn, bk, bt, st, warps))
                print(f"{tag}: {t:.2f} us {gb / t:.2f} GB/s relerr {err:.1e}", flush=True)
            except Exception as exc:
                print(f"{tag}: FAILED {type(exc).__name__}: {str(exc)[:600]}", flush=True)
                traceback.print_exc(limit=2)


if __name__ == "__main__":
    main()
