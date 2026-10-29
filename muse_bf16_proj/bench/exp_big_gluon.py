"""Track B: Large-m (m=128, 256) Gluon & Split-K GEMM experiments vs checkpoints/cp2_gluon_router.py.

Architectures tested:
  B1 (Router m=128, 256, N=256, K=8192, FP32 out from BF16 hi|lo):
    - _router_big_v2 (Triton TMA): wider tiles (BM=128/256, BN=32/64/128), SWAP=0/1, packed stride-1
      tile-contiguous workspace, and (SPLIT-1)-load in-register reduction (total = acc + others).
    - _gluon_router_pipe (Gluon single-partition TMA ring + 2x warpgroup_mma per step + wait(2)):
      explicit mbarrier ring loading x once and multiplying by both hi and lo without Triton pipeliner
      overhead.
    - _gluon_router_ws (Gluon warp_specialize: 1 TMA producer warp [24 regs] + consumer warpgroup(s)).

  B2 (BF16 layers qkv_gate, pooler, unpooler, o at m=128, 256):
    - _gluon_big_pipe (Gluon single-partition TMA ring + async warpgroup_mma + wait(1)):
      supports SWAP=0 (acc [BM, BN]) and SWAP=1 (acc [BN, BM]), SPLIT >= 1 with packed stride-1
      in-kernel atomic last-CTA reduction (eliminating cuBLAS's separate 2.7 us splitKreduce_kernel).
    - _gluon_big_ws (Gluon warp_specialize: 1 TMA producer warp [24 regs] + 4/8 consumer warps):
      overlaps TMA HBM streaming with wgmma at the Hopper ridge point (m=128, 256).

Usage:
    modal run modal_app.py::compile_only --cmd "exp_big_gluon --compile"
    modal run modal_app.py::micro --name exp_big_gluon --args "--reps 35"
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from typing import NamedTuple

if "--cold" in sys.argv:
    os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp()

import torch
import triton
import triton.language as tl

try:
    from triton.tools.tensor_descriptor import TensorDescriptor
except ImportError:
    TensorDescriptor = None

try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language.nvidia.hopper import (
        mbarrier,
        tma,
        warpgroup_mma,
        warpgroup_mma_wait,
    )
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor as GDesc

    HAVE_GLUON = True
except ImportError:
    gluon = None
    HAVE_GLUON = False

BASE_CP = "checkpoints/cp2_gluon_router.py"
SHAPES = {
    "qkv_gate": (2304, 8192),
    "o": (8192, 1024),
    "pooler": (4096, 8192),
    "unpooler": (8192, 4096),
    "router": (256, 8192),
}
HW = {"hbm_bytes_per_s": 4.8e12, "bf16_dense_flops_per_s": 989e12}
FLUSH_BYTES = 256 * 2**20
OVERHEAD_US = 4.0
WS_FLOATS = 4 * 2**20  # 16 MB workspace for sweep (fits all split configs up to m=512)
SUB_WS_FLOATS = 2 * 2**20  # 8 MB cap for submission viability
COUNTERS = 8192
GLUON_SMEM = 220 * 1024


class RCfg(NamedTuple):
    kind: str       # "triton_v2", "gluon_pipe", "gluon_ws"
    bm: int
    bn: int
    bk: int
    split: int
    swap: int       # 0: acc [BM, BN], 1: acc [BN, BM]
    warps: int
    stages: int
    maxnreg: int = 0


class BCfg(NamedTuple):
    kind: str       # "gluon_pipe", "gluon_ws"
    bm: int
    bn: int
    bk: int
    split: int
    swap: int       # 0: acc [BM, BN], 1: acc [BN, BM]
    warps: int
    stages: int
    maxnreg: int = 0


def r_smem(c: RCfg) -> int:
    return c.stages * (c.bm + 2 * c.bn) * c.bk * 2


def b_smem(c: BCfg) -> int:
    return c.stages * (c.bm + c.bn) * c.bk * 2


def wgmma_valid(bm: int, bn: int, swap: int, warps: int) -> bool:
    d0 = bn if swap else bm
    d1 = bm if swap else bn
    if d0 < 64 or d0 % 64 != 0 or d1 < 8 or d1 % 8 != 0 or d1 > 256:
        return False
    if warps == 4:
        return True
    if warps == 8:
        return d0 >= 128
    return False


def r_valid(c: RCfg, m: int, n: int = 256, k: int = 8192) -> bool:
    num_m = triton.cdiv(m, c.bm)
    if n % c.bn != 0 or k % c.bk != 0 or (k // c.bk) < c.split:
        return False
    if (k // c.bk) // c.split < c.stages or c.stages < 2:
        return False
    if r_smem(c) > GLUON_SMEM:
        return False
    if c.split * num_m * c.bm * n > SUB_WS_FLOATS or num_m * (n // c.bn) > COUNTERS:
        return False
    if c.kind == "gluon_ws" and (c.split != 1 or c.warps != 4):
        return False
    if c.kind != "triton_v2":
        if not wgmma_valid(c.bm, c.bn, c.swap, c.warps):
            return False
        # Filter out 255-reg spilling configs identified by ptxas -v
        if c.bm * c.bn >= 8192 and c.swap == 1 and c.warps == 4:
            return False
        if c.bm >= 256 and c.split > 1 and c.warps == 4:
            return False
    return True


def b_valid(c: BCfg, layer: str, m: int) -> bool:
    n, k = SHAPES[layer]
    num_m = triton.cdiv(m, c.bm)
    if n % c.bn != 0 or k % c.bk != 0 or (k // c.bk) < c.split:
        return False
    if (k // c.bk) // c.split < c.stages or c.stages < 2:
        return False
    if b_smem(c) > GLUON_SMEM:
        return False
    if c.split > 1 and c.split * num_m * c.bm * n > SUB_WS_FLOATS:
        return False
    if num_m * (n // c.bn) > COUNTERS:
        return False
    if c.kind == "gluon_ws" and (c.split != 1 or c.warps != 4):
        return False
    if c.bm >= 256 and c.split > 1 and c.warps == 4:
        return False
    return wgmma_valid(c.bm, c.bn, c.swap, c.warps)


# --------------------------------------------------------------------------- Architecture B1: Triton _router_big_v2

@triton.jit(do_not_specialize=["M"])
def _router_big_v2(
    x_desc, hi_desc, lo_desc, out_ptr, ws_ptr, cnt_ptr, M,
    N: tl.constexpr, K: tl.constexpr, SPLIT: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SWAP: tl.constexpr,
):
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
        hi = hi_desc.load([pid_n * BN, ki * BK])
        lo = lo_desc.load([pid_n * BN, ki * BK])
        if SWAP:
            acc = tl.dot(hi, a.T, acc)
            acc = tl.dot(lo, a.T, acc)
        else:
            acc = tl.dot(a, hi.T, acc)
            acc = tl.dot(a, lo.T, acc)
    local_m = tl.arange(0, BM)
    local_n = tl.arange(0, BN)
    offs_m = pid_m * BM + local_m
    offs_n = pid_n * BN + local_n
    if SWAP:
        out_idx = offs_m[None, :] * N + offs_n[:, None]
        mask = (offs_m < M)[None, :]
        ws_offs = local_n[:, None] * BM + local_m[None, :]
    else:
        out_idx = offs_m[:, None] * N + offs_n[None, :]
        mask = (offs_m < M)[:, None]
        ws_offs = local_m[:, None] * BN + local_n[None, :]
    if SPLIT == 1:
        tl.store(out_ptr + out_idx, acc, mask=mask)
    else:
        TILE_ELEMS: tl.constexpr = BM * BN
        tile_id = pid_n * num_m + pid_m
        ws_tile = ws_ptr + tile_id * (SPLIT * TILE_ELEMS) + ws_offs
        tl.store(ws_tile + pid_s * TILE_ELEMS, acc)
        tl.debug_barrier()
        cnt = cnt_ptr + tile_id
        done = tl.atomic_add(cnt, 1, sem="acq_rel", scope="gpu")
        if done == SPLIT - 1:
            total = acc
            for ds in tl.static_range(1, SPLIT):
                s = (pid_s + ds) % SPLIT
                total += tl.load(ws_tile + s * TILE_ELEMS, cache_modifier=".cg")
            tl.store(out_ptr + out_idx, total, mask=mask)
            tl.atomic_xchg(cnt, 0)


# --------------------------------------------------------------------------- Gluon helpers & B1/B2 kernels

if HAVE_GLUON:

    @gluon.jit
    def _gluon_epilogue(
        acc, out_ptr, ws_ptr, cnt_ptr, M, pid_m, pid_n, pid_s, num_m,
        N: gl.constexpr, SPLIT: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr,
        SWAP: gl.constexpr, OUT_FP32: gl.constexpr, mma: gl.constexpr,
    ):
        if SWAP:
            local_n = gl.arange(0, BN, layout=gl.SliceLayout(1, mma))
            local_m = gl.arange(0, BM, layout=gl.SliceLayout(0, mma))
            offs_n = pid_n * BN + local_n
            offs_m = pid_m * BM + local_m
            out_ptrs = out_ptr + offs_m[None, :] * N + offs_n[:, None]
            mask = (offs_m < M)[None, :]
            ws_offs = local_n[:, None] * BM + local_m[None, :]
        else:
            local_m = gl.arange(0, BM, layout=gl.SliceLayout(1, mma))
            local_n = gl.arange(0, BN, layout=gl.SliceLayout(0, mma))
            offs_m = pid_m * BM + local_m
            offs_n = pid_n * BN + local_n
            out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
            mask = (offs_m < M)[:, None]
            ws_offs = local_m[:, None] * BN + local_n[None, :]

        if SPLIT == 1:
            val = acc if OUT_FP32 else acc.to(gl.bfloat16)
            gl.store(out_ptrs, val, mask=mask)
        else:
            TILE_ELEMS: gl.constexpr = BM * BN
            tile_id = pid_n * num_m + pid_m
            ws_tile = ws_ptr + tile_id * (SPLIT * TILE_ELEMS) + ws_offs
            gl.store(ws_tile + pid_s * TILE_ELEMS, acc)
            gl.barrier()
            cnt = cnt_ptr + tile_id
            done = gl.atomic_add(cnt, 1, sem="acq_rel", scope="gpu")
            if done == SPLIT - 1:
                total = acc
                for ds in gl.static_range(1, SPLIT):
                    s = (pid_s + ds) % SPLIT
                    total += gl.load(ws_tile + s * TILE_ELEMS, cache_modifier=".cg")
                val = total if OUT_FP32 else total.to(gl.bfloat16)
                gl.store(out_ptrs, val, mask=mask)
                gl.atomic_xchg(cnt, 0)

    # ---- Architecture B1: Gluon Router Pipe ----

    @gluon.jit
    def _gluon_router_pipe(
        x_desc, hi_desc, lo_desc, out_ptr, ws_ptr, cnt_ptr, M,
        N: gl.constexpr, K: gl.constexpr, SPLIT: gl.constexpr,
        STAGES: gl.constexpr, SWAP: gl.constexpr, num_warps: gl.constexpr,
    ):
        BM: gl.constexpr = x_desc.block_type.shape[0]
        BK: gl.constexpr = x_desc.block_type.shape[1]
        BN: gl.constexpr = hi_desc.block_type.shape[0]
        KI: gl.constexpr = K // BK
        D0: gl.constexpr = BN if SWAP else BM
        D1: gl.constexpr = BM if SWAP else BN
        mma: gl.constexpr = gl.NVMMADistributedLayout(
            version=[3, 0], warps_per_cta=[num_warps, 1], instr_shape=[16, D1, 16]
        )

        pid = gl.program_id(0)
        num_m = (M + BM - 1) // BM
        pid_m = pid % num_m
        pid_s = (pid // num_m) % SPLIT
        pid_n = pid // (num_m * SPLIT)
        m0 = pid_m * BM
        n0 = pid_n * BN
        k0 = pid_s * KI // SPLIT
        k_steps: gl.constexpr = KI // SPLIT

        x_bufs = gl.allocate_shared_memory(x_desc.dtype, [STAGES] + x_desc.block_type.shape, x_desc.layout)
        hi_bufs = gl.allocate_shared_memory(hi_desc.dtype, [STAGES] + hi_desc.block_type.shape, hi_desc.layout)
        lo_bufs = gl.allocate_shared_memory(lo_desc.dtype, [STAGES] + lo_desc.block_type.shape, lo_desc.layout)
        ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
        for i in gl.static_range(STAGES):
            mbarrier.init(ready.index(i), count=1)

        nbytes: gl.constexpr = (
            x_desc.block_type.nbytes + hi_desc.block_type.nbytes + lo_desc.block_type.nbytes
        )
        for i in gl.static_range(STAGES - 1):
            mbarrier.expect(ready.index(i), nbytes)
            tma.async_load(x_desc, [m0, (k0 + i) * BK], ready.index(i), x_bufs.index(i))
            tma.async_load(hi_desc, [n0, (k0 + i) * BK], ready.index(i), hi_bufs.index(i))
            tma.async_load(lo_desc, [n0, (k0 + i) * BK], ready.index(i), lo_bufs.index(i))

        acc = gl.zeros([D0, D1], gl.float32, layout=mma)
        for i in range(k_steps):
            s = i % STAGES
            mbarrier.wait(ready.index(s), (i // STAGES) & 1)
            if SWAP:
                xt = x_bufs.index(s).permute((1, 0))
                acc = warpgroup_mma(hi_bufs.index(s), xt, acc, is_async=True)
                acc = warpgroup_mma(lo_bufs.index(s), xt, acc, is_async=True)
            else:
                xb = x_bufs.index(s)
                acc = warpgroup_mma(xb, hi_bufs.index(s).permute((1, 0)), acc, is_async=True)
                acc = warpgroup_mma(xb, lo_bufs.index(s).permute((1, 0)), acc, is_async=True)
            acc = warpgroup_mma_wait(2, deps=(acc,))
            if num_warps > 4:
                gl.barrier()
            j = i + STAGES - 1
            sj = j % STAGES
            pred = j < k_steps
            mbarrier.expect(ready.index(sj), nbytes, pred=pred)
            tma.async_load(x_desc, [m0, (k0 + j) * BK], ready.index(sj), x_bufs.index(sj), pred=pred)
            tma.async_load(hi_desc, [n0, (k0 + j) * BK], ready.index(sj), hi_bufs.index(sj), pred=pred)
            tma.async_load(lo_desc, [n0, (k0 + j) * BK], ready.index(sj), lo_bufs.index(sj), pred=pred)

        acc = warpgroup_mma_wait(0, deps=(acc,))
        _gluon_epilogue(
            acc, out_ptr, ws_ptr, cnt_ptr, M, pid_m, pid_n, pid_s, num_m,
            N, SPLIT, BM, BN, SWAP, True, mma,
        )

    # ---- Architecture B2: Gluon BF16 GEMM Pipe & Warp-Specialized (SPLIT=1, W=4) ----

    @gluon.jit
    def _gluon_big_pipe(
        x_desc, w_desc, out_ptr, ws_ptr, cnt_ptr, M,
        N: gl.constexpr, K: gl.constexpr, SPLIT: gl.constexpr,
        STAGES: gl.constexpr, SWAP: gl.constexpr, num_warps: gl.constexpr,
    ):
        BM: gl.constexpr = x_desc.block_type.shape[0]
        BK: gl.constexpr = x_desc.block_type.shape[1]
        BN: gl.constexpr = w_desc.block_type.shape[0]
        KI: gl.constexpr = K // BK
        D0: gl.constexpr = BN if SWAP else BM
        D1: gl.constexpr = BM if SWAP else BN
        mma: gl.constexpr = gl.NVMMADistributedLayout(
            version=[3, 0], warps_per_cta=[num_warps, 1], instr_shape=[16, D1, 16]
        )

        pid = gl.program_id(0)
        num_m = (M + BM - 1) // BM
        pid_m = pid % num_m
        pid_s = (pid // num_m) % SPLIT
        pid_n = pid // (num_m * SPLIT)
        m0 = pid_m * BM
        n0 = pid_n * BN
        if SPLIT == 1:
            k0 = 0
            k_steps = KI
        else:
            k0 = pid_s * KI // SPLIT
            k_steps = (pid_s + 1) * KI // SPLIT - k0

        x_bufs = gl.allocate_shared_memory(x_desc.dtype, [STAGES] + x_desc.block_type.shape, x_desc.layout)
        w_bufs = gl.allocate_shared_memory(w_desc.dtype, [STAGES] + w_desc.block_type.shape, w_desc.layout)
        ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
        for i in gl.static_range(STAGES):
            mbarrier.init(ready.index(i), count=1)

        nbytes: gl.constexpr = x_desc.block_type.nbytes + w_desc.block_type.nbytes
        for i in gl.static_range(STAGES - 1):
            mbarrier.expect(ready.index(i), nbytes)
            tma.async_load(x_desc, [m0, (k0 + i) * BK], ready.index(i), x_bufs.index(i))
            tma.async_load(w_desc, [n0, (k0 + i) * BK], ready.index(i), w_bufs.index(i))

        acc = gl.zeros([D0, D1], gl.float32, layout=mma)
        for i in range(k_steps):
            s = i % STAGES
            mbarrier.wait(ready.index(s), (i // STAGES) & 1)
            if SWAP:
                acc = warpgroup_mma(w_bufs.index(s), x_bufs.index(s).permute((1, 0)), acc, is_async=True)
            else:
                acc = warpgroup_mma(x_bufs.index(s), w_bufs.index(s).permute((1, 0)), acc, is_async=True)
            acc = warpgroup_mma_wait(1, deps=(acc,))
            if num_warps > 4:
                gl.barrier()
            j = i + STAGES - 1
            sj = j % STAGES
            pred = j < k_steps
            mbarrier.expect(ready.index(sj), nbytes, pred=pred)
            tma.async_load(x_desc, [m0, (k0 + j) * BK], ready.index(sj), x_bufs.index(sj), pred=pred)
            tma.async_load(w_desc, [n0, (k0 + j) * BK], ready.index(sj), w_bufs.index(sj), pred=pred)

        acc = warpgroup_mma_wait(0, deps=(acc,))
        _gluon_epilogue(
            acc, out_ptr, ws_ptr, cnt_ptr, M, pid_m, pid_n, pid_s, num_m,
            N, SPLIT, BM, BN, SWAP, False, mma,
        )

    @gluon.jit
    def _big_ws_producer(
        x_desc, w_desc, x_bufs, w_bufs, ready, empty,
        m0, n0, KI: gl.constexpr, BK: gl.constexpr,
    ):
        S: gl.constexpr = x_bufs.type.shape[0]
        nbytes: gl.constexpr = x_desc.block_type.nbytes + w_desc.block_type.nbytes
        for i in range(KI):
            s = i % S
            ph = (i // S) & 1
            mbarrier.wait(empty.index(s), ph ^ 1)
            mbarrier.expect(ready.index(s), nbytes)
            tma.async_load(x_desc, [m0, i * BK], ready.index(s), x_bufs.index(s))
            tma.async_load(w_desc, [n0, i * BK], ready.index(s), w_bufs.index(s))

    @gluon.jit
    def _big_ws_consumer(
        x_bufs, w_bufs, ready, empty,
        KI: gl.constexpr, D0: gl.constexpr, D1: gl.constexpr,
        SWAP: gl.constexpr, mma: gl.constexpr,
    ):
        S: gl.constexpr = x_bufs.type.shape[0]
        acc = gl.zeros([D0, D1], gl.float32, layout=mma)
        for i in range(KI):
            s = i % S
            mbarrier.wait(ready.index(s), (i // S) & 1)
            if SWAP:
                acc = warpgroup_mma(w_bufs.index(s), x_bufs.index(s).permute((1, 0)), acc, is_async=True)
            else:
                acc = warpgroup_mma(x_bufs.index(s), w_bufs.index(s).permute((1, 0)), acc, is_async=True)
            acc = warpgroup_mma_wait(1, deps=(acc,))
            mbarrier.arrive(empty.index((i + S - 1) % S), pred=i > 0)
        acc = warpgroup_mma_wait(0, deps=(acc,))
        return acc

    @gluon.jit
    def _gluon_big_ws(
        x_desc, w_desc, out_ptr, ws_ptr, cnt_ptr, M,
        N: gl.constexpr, K: gl.constexpr, SPLIT: gl.constexpr,
        STAGES: gl.constexpr, SWAP: gl.constexpr, num_warps: gl.constexpr,
    ):
        BM: gl.constexpr = x_desc.block_type.shape[0]
        BK: gl.constexpr = x_desc.block_type.shape[1]
        BN: gl.constexpr = w_desc.block_type.shape[0]
        KI: gl.constexpr = K // BK
        D0: gl.constexpr = BN if SWAP else BM
        D1: gl.constexpr = BM if SWAP else BN
        mma: gl.constexpr = gl.NVMMADistributedLayout(
            version=[3, 0], warps_per_cta=[num_warps, 1], instr_shape=[16, D1, 16]
        )

        pid = gl.program_id(0)
        num_m = (M + BM - 1) // BM
        pid_m = pid % num_m
        pid_n = pid // num_m
        m0 = pid_m * BM
        n0 = pid_n * BN

        x_bufs = gl.allocate_shared_memory(x_desc.dtype, [STAGES] + x_desc.block_type.shape, x_desc.layout)
        w_bufs = gl.allocate_shared_memory(w_desc.dtype, [STAGES] + w_desc.block_type.shape, w_desc.layout)
        ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
        empty = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
        for i in gl.static_range(STAGES):
            mbarrier.init(ready.index(i), count=1)
            mbarrier.init(empty.index(i), count=1)

        acc = gl.warp_specialize(
            [
                (_big_ws_consumer, (x_bufs, w_bufs, ready, empty, KI, D0, D1, SWAP, mma)),
                (_big_ws_producer, (x_desc, w_desc, x_bufs, w_bufs, ready, empty, m0, n0, KI, BK)),
            ],
            [1],
            [24],
        )
        _gluon_epilogue(
            acc, out_ptr, ws_ptr, cnt_ptr, M, pid_m, pid_n, 0, num_m,
            N, 1, BM, BN, SWAP, False, mma,
        )


# --------------------------------------------------------------------------- Candidate Sets (B1 & B2, 0-spill only)

def b1_router_candidates(m: int) -> list[RCfg]:
    raw: list[RCfg] = []
    # 1. Triton _router_big_v2 (packed stride-1 ws + (SPLIT-1)-load reduce + wider tiles)
    triton_specs = [
        (64, 32, 128, 8, 0, 4, 4),
        (64, 32, 128, 4, 0, 4, 4),
        (64, 32, 128, 8, 1, 4, 4),
        (64, 32, 128, 4, 1, 4, 4),
        (64, 64, 128, 8, 0, 4, 3),
        (64, 64, 128, 4, 0, 4, 3),
        (64, 64, 64, 8, 0, 4, 4),
        (128, 32, 128, 8, 0, 4, 3),
        (128, 32, 64, 8, 0, 4, 4),
        (128, 64, 64, 8, 0, 4, 4),
        (128, 64, 64, 4, 0, 4, 4),
        (128, 64, 128, 8, 0, 8, 3),
        (128, 128, 64, 8, 0, 8, 3),
    ]
    for bm, bn, bk, sp, sw, w, st in triton_specs:
        raw.append(RCfg("triton_v2", bm, bn, bk, sp, sw, w, st))

    # 2. Gluon _gluon_router_pipe (0-spill configs verified by ptxas -v)
    gluon_specs = [
        # (bm, bn, bk, split, swap, warps, stages)
        (64, 32, 128, 8, 0, 4, 4),
        (64, 32, 128, 4, 0, 4, 4),
        (64, 32, 64, 8, 0, 4, 8),
        (64, 64, 128, 8, 0, 4, 4),
        (64, 64, 128, 4, 0, 4, 4),
        (64, 64, 128, 8, 1, 4, 4),
        (64, 64, 64, 8, 0, 4, 4),
        (128, 32, 128, 8, 0, 4, 4),
        (128, 32, 128, 16, 0, 4, 2),
        (128, 32, 64, 8, 0, 4, 4),
        (128, 64, 128, 8, 0, 4, 2),
        (128, 64, 64, 8, 0, 4, 4),
        (128, 64, 64, 16, 0, 4, 4),
        (128, 64, 64, 8, 0, 8, 4),
        (128, 64, 64, 16, 0, 8, 4),
        (128, 128, 64, 8, 0, 8, 4),
        (128, 128, 64, 16, 0, 8, 4),
    ]
    if m >= 256:
        gluon_specs += [
            (256, 64, 64, 8, 0, 8, 4),
            (256, 64, 64, 16, 0, 8, 4),
        ]
    for bm, bn, bk, sp, sw, w, st in gluon_specs:
        raw.append(RCfg("gluon_pipe", bm, bn, bk, sp, sw, w, st, 0))

    out: list[RCfg] = []
    for c in raw:
        if r_valid(c, m) and c not in out:
            out.append(c)
    return out


def b2_bf16_candidates(layer: str, m: int) -> list[BCfg]:
    specs: list[tuple[int, int, int, int, int, int, int]] = []
    if layer == "qkv_gate":
        if m <= 192:
            specs = [
                # (bm, bn, bk, split, swap, warps, stages)
                (128, 64, 128, 2, 0, 4, 4),
                (128, 64, 128, 3, 0, 4, 4),
                (128, 64, 128, 4, 0, 4, 4),
                (128, 64, 128, 2, 1, 4, 4),
                (128, 64, 128, 3, 1, 4, 4),
                (128, 64, 64, 2, 0, 4, 8),
                (128, 64, 64, 3, 0, 4, 8),
                (128, 64, 64, 4, 0, 4, 8),
                (128, 64, 64, 3, 1, 4, 8),
                (128, 64, 64, 3, 0, 8, 8),
                (128, 64, 128, 3, 0, 8, 4),
                (64, 64, 128, 1, 0, 4, 4),
                (64, 64, 128, 2, 0, 4, 4),
                (64, 64, 64, 1, 0, 4, 8),
                (64, 64, 64, 2, 0, 4, 8),
                (64, 128, 128, 2, 0, 4, 4),
                (64, 128, 128, 3, 0, 4, 4),
                (64, 128, 64, 3, 0, 4, 8),
                (128, 128, 64, 3, 0, 8, 4),
                (128, 128, 64, 4, 0, 8, 4),
            ]
        else:
            specs = [
                (128, 64, 128, 1, 0, 4, 4),
                (128, 64, 128, 2, 0, 4, 4),
                (128, 64, 128, 1, 1, 4, 4),
                (128, 64, 128, 2, 1, 4, 4),
                (128, 64, 64, 1, 0, 4, 8),
                (128, 64, 64, 2, 0, 4, 8),
                (128, 64, 64, 2, 0, 8, 8),
                (256, 64, 64, 2, 0, 8, 4),
                (256, 64, 64, 3, 0, 8, 4),
                (256, 64, 128, 2, 0, 8, 2),
                (256, 64, 128, 3, 0, 8, 2),
                (128, 128, 64, 2, 0, 8, 4),
                (128, 128, 64, 3, 0, 8, 4),
                (128, 128, 128, 2, 0, 8, 2),
                (128, 128, 128, 3, 0, 8, 2),
                (64, 64, 128, 1, 0, 4, 4),
            ]
    elif layer == "pooler":
        if m <= 192:
            specs = [
                (128, 64, 128, 1, 0, 4, 4),
                (128, 64, 128, 2, 0, 4, 4),
                (128, 64, 128, 1, 1, 4, 4),
                (128, 64, 128, 2, 1, 4, 4),
                (128, 64, 64, 1, 0, 4, 8),
                (128, 64, 64, 2, 0, 4, 8),
                (128, 64, 64, 2, 1, 4, 8),
                (128, 64, 64, 2, 0, 8, 8),
                (128, 64, 128, 2, 0, 8, 4),
                (64, 128, 128, 1, 0, 4, 4),
                (64, 128, 128, 2, 0, 4, 4),
                (64, 128, 64, 2, 0, 4, 8),
                (64, 128, 64, 2, 1, 4, 8),
                (64, 64, 128, 1, 0, 4, 4),
                (64, 64, 64, 1, 0, 4, 8),
                (128, 128, 64, 2, 0, 8, 4),
                (128, 128, 64, 4, 0, 8, 4),
            ]
        else:
            specs = [
                (128, 64, 128, 1, 0, 4, 4),
                (128, 64, 128, 1, 1, 4, 4),
                (128, 64, 64, 1, 0, 4, 8),
                (128, 64, 64, 1, 1, 4, 8),
                (128, 64, 64, 1, 0, 8, 8),
                (256, 64, 64, 1, 1, 4, 4),
                (256, 64, 64, 1, 0, 8, 4),
                (256, 64, 64, 2, 0, 8, 4),
                (256, 64, 128, 1, 1, 4, 2),
                (256, 64, 128, 1, 0, 8, 2),
                (128, 128, 64, 1, 0, 8, 4),
                (128, 128, 64, 2, 0, 8, 4),
                (128, 128, 128, 1, 0, 8, 2),
                (128, 128, 128, 2, 0, 8, 2),
                (64, 128, 128, 1, 0, 4, 4),
            ]
    elif layer == "unpooler":
        if m <= 192:
            specs = [
                (128, 64, 128, 1, 0, 4, 4),
                (128, 64, 128, 1, 1, 4, 4),
                (128, 64, 128, 1, 0, 8, 4),
                (128, 64, 64, 1, 0, 4, 8),
                (128, 64, 64, 1, 1, 4, 8),
                (128, 64, 64, 1, 0, 8, 8),
                (64, 128, 128, 1, 0, 4, 4),
                (64, 128, 128, 1, 1, 4, 4),
                (64, 128, 64, 1, 0, 4, 8),
                (64, 64, 128, 1, 0, 4, 4),
                (128, 128, 64, 1, 0, 8, 4),
                (128, 128, 64, 2, 0, 8, 4),
                (128, 128, 128, 1, 0, 8, 2),
            ]
        else:
            specs = [
                (256, 64, 64, 1, 1, 4, 4),
                (256, 64, 64, 1, 0, 8, 4),
                (256, 64, 128, 1, 1, 4, 2),
                (256, 64, 128, 1, 0, 8, 2),
                (128, 128, 64, 1, 0, 8, 4),
                (128, 128, 128, 1, 0, 8, 2),
                (128, 64, 128, 1, 0, 4, 4),
                (128, 64, 128, 1, 1, 4, 4),
                (128, 64, 64, 1, 0, 4, 8),
                (128, 64, 64, 1, 0, 8, 8),
            ]
    elif layer == "o":
        if m <= 192:
            specs = [
                (128, 64, 128, 1, 0, 4, 4),
                (128, 64, 128, 1, 1, 4, 4),
                (128, 64, 128, 1, 0, 8, 4),
                (128, 64, 64, 1, 0, 4, 8),
                (128, 64, 64, 1, 0, 4, 4),
                (128, 64, 64, 1, 1, 4, 8),
                (128, 64, 64, 1, 0, 8, 4),
                (64, 128, 128, 1, 0, 4, 4),
                (64, 128, 64, 1, 0, 4, 8),
                (64, 64, 128, 1, 0, 4, 4),
                (128, 128, 128, 1, 0, 8, 2),
                (128, 128, 64, 1, 0, 8, 4),
            ]
        else:
            specs = [
                (256, 64, 128, 1, 1, 4, 2),
                (256, 64, 128, 1, 0, 8, 2),
                (256, 64, 64, 1, 1, 4, 4),
                (256, 64, 64, 1, 0, 8, 4),
                (128, 128, 128, 1, 0, 8, 2),
                (128, 128, 64, 1, 0, 8, 4),
                (128, 64, 128, 1, 0, 4, 4),
                (128, 64, 128, 1, 1, 4, 4),
                (128, 64, 64, 1, 0, 4, 8),
                (128, 64, 64, 1, 0, 4, 4),
            ]

    out: list[BCfg] = []
    for bm, bn, bk, sp, sw, w, st in specs:
        for kind, nreg in (("gluon_pipe", 0), ("gluon_ws", 168)):
            c = BCfg(kind, bm, bn, bk, sp, sw, w, st, nreg)
            if b_valid(c, layer, m) and c not in out:
                out.append(c)
    return out


# --------------------------------------------------------------------------- CPU Compile & ptxas Diagnostics

def _run_ptxas(ptx: str, ptxas_bin: str) -> str:
    with tempfile.TemporaryDirectory() as td:
        path = f"{td}/k.ptx"
        with open(path, "w") as f:
            f.write(ptx)
        res = subprocess.run([ptxas_bin, "-arch=sm_90a", "-v", path, "-o", f"{td}/k.o"],
                             capture_output=True, text=True)
        lines = []
        for line in (res.stdout + res.stderr).splitlines():
            if any(k in line for k in ("registers", "spill", "smem", "cmem", "Warning", "error")):
                lines.append(line.strip())
        return " | ".join(lines) if lines else "ok (no verbose lines)"


def compile_report(only: str = "") -> None:
    if not HAVE_GLUON:
        raise RuntimeError("Gluon is not available in this Python environment")
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import get_ptxas
    from triton.experimental.gluon._runtime import GluonASTSource

    ptxas_bin = get_ptxas(90).path
    target = GPUTarget("cuda", 90, 32)

    print("===== B1 Router Gluon Compile Report (sm_90a, 0-spill) =====", flush=True)
    r_test = [
        RCfg("gluon_pipe", 64, 32, 128, 8, 0, 4, 4, 0),
        RCfg("gluon_pipe", 64, 64, 128, 8, 0, 4, 4, 0),
        RCfg("gluon_pipe", 128, 64, 64, 8, 0, 4, 4, 0),
        RCfg("gluon_pipe", 128, 64, 64, 8, 0, 8, 4, 0),
        RCfg("gluon_pipe", 128, 128, 64, 8, 0, 8, 4, 0),
        RCfg("gluon_pipe", 256, 64, 64, 8, 0, 8, 4, 0),
    ]
    n_r, k_r = SHAPES["router"]
    for c in r_test:
        tag = f"router {c.kind} BM{c.bm} BN{c.bn} BK{c.bk} SP{c.split} SW{c.swap} W{c.warps} S{c.stages}"
        if only and only not in tag:
            continue
        x = torch.zeros(max(128, c.bm), k_r, dtype=torch.bfloat16)
        hi = torch.zeros(n_r, k_r, dtype=torch.bfloat16)
        lo = torch.zeros(n_r, k_r, dtype=torch.bfloat16)
        xl = gl.NVMMASharedLayout.get_default_for([c.bm, c.bk], gl.bfloat16)
        wl = gl.NVMMASharedLayout.get_default_for([c.bn, c.bk], gl.bfloat16)
        xd = GDesc.from_tensor(x, [c.bm, c.bk], xl)
        hid = GDesc.from_tensor(hi, [c.bn, c.bk], wl)
        lod = GDesc.from_tensor(lo, [c.bn, c.bk], wl)
        fn = _gluon_router_pipe
        sig = {
            "x_desc": xd.__mangle__(),
            "hi_desc": hid.__mangle__(),
            "lo_desc": lod.__mangle__(),
            "out_ptr": "*fp32",
            "ws_ptr": "*fp32",
            "cnt_ptr": "*i32",
            "M": "i32",
        }
        consts = dict(N=n_r, K=k_r, SPLIT=c.split, STAGES=c.stages, SWAP=bool(c.swap), num_warps=c.warps)
        names = list(fn.arg_names)
        full = {nm: sig.get(nm, "constexpr") for nm in names}
        attrs = {(names.index(nm),): [["tt.divisibility", 16]] for nm in ("out_ptr", "ws_ptr", "cnt_ptr")}
        src = GluonASTSource(
            fn=fn, signature=full, constexprs={(names.index(nm),): v for nm, v in consts.items()}, attrs=attrs
        )
        opts = {"num_warps": c.warps}
        if c.maxnreg:
            opts["maxnreg"] = c.maxnreg
        t0 = time.time()
        try:
            ker = triton.compile(src, target=target, options=opts)
            ptx = ker.asm["ptx"]
            wgmma_cnt = len(re.findall(r"wgmma\.mma_async", ptx))
            info = _run_ptxas(ptx, ptxas_bin)
            print(f"  [OK] {tag} ({time.time() - t0:.1f}s) smem={ker.metadata.shared}B wgmma={wgmma_cnt} | {info}",
                  flush=True)
        except Exception as exc:
            print(f"  [FAIL] {tag}: {type(exc).__name__}: {str(exc)[:600]}", flush=True)

    print("\n===== B2 BF16 GEMM Gluon Compile Report (sm_90a, 0-spill) =====", flush=True)
    b_test = [
        ("qkv_gate", BCfg("gluon_pipe", 128, 64, 128, 3, 0, 4, 4, 0)),
        ("qkv_gate", BCfg("gluon_pipe", 128, 64, 128, 3, 1, 4, 4, 0)),
        ("qkv_gate", BCfg("gluon_pipe", 256, 64, 64, 3, 0, 8, 4, 0)),
        ("pooler", BCfg("gluon_pipe", 128, 64, 128, 1, 0, 4, 4, 0)),
        ("pooler", BCfg("gluon_ws", 128, 64, 128, 1, 0, 4, 4, 168)),
        ("pooler", BCfg("gluon_pipe", 128, 64, 128, 2, 0, 4, 4, 0)),
        ("pooler", BCfg("gluon_pipe", 256, 64, 64, 1, 1, 4, 4, 0)),
        ("pooler", BCfg("gluon_ws", 256, 64, 64, 1, 1, 4, 4, 168)),
        ("unpooler", BCfg("gluon_pipe", 128, 64, 128, 1, 0, 4, 4, 0)),
        ("unpooler", BCfg("gluon_ws", 128, 64, 128, 1, 0, 4, 4, 168)),
        ("unpooler", BCfg("gluon_pipe", 256, 64, 64, 1, 1, 4, 4, 0)),
        ("unpooler", BCfg("gluon_ws", 256, 64, 64, 1, 1, 4, 4, 168)),
        ("o", BCfg("gluon_pipe", 128, 64, 128, 1, 0, 4, 4, 0)),
        ("o", BCfg("gluon_ws", 128, 64, 128, 1, 0, 4, 4, 168)),
        ("o", BCfg("gluon_pipe", 256, 64, 128, 1, 1, 4, 2, 0)),
        ("o", BCfg("gluon_ws", 256, 64, 128, 1, 1, 4, 2, 168)),
    ]
    for layer, c in b_test:
        n_b, k_b = SHAPES[layer]
        tag = f"{layer} {c.kind} BM{c.bm} BN{c.bn} BK{c.bk} SP{c.split} SW{c.swap} W{c.warps} S{c.stages}"
        if only and only not in tag:
            continue
        x = torch.zeros(max(128, c.bm), k_b, dtype=torch.bfloat16)
        w = torch.zeros(n_b, k_b, dtype=torch.bfloat16)
        xl = gl.NVMMASharedLayout.get_default_for([c.bm, c.bk], gl.bfloat16)
        wl = gl.NVMMASharedLayout.get_default_for([c.bn, c.bk], gl.bfloat16)
        xd = GDesc.from_tensor(x, [c.bm, c.bk], xl)
        wd = GDesc.from_tensor(w, [c.bn, c.bk], wl)
        fn = _gluon_big_ws if c.kind == "gluon_ws" else _gluon_big_pipe
        sig = {
            "x_desc": xd.__mangle__(),
            "w_desc": wd.__mangle__(),
            "out_ptr": "*bf16",
            "ws_ptr": "*fp32",
            "cnt_ptr": "*i32",
            "M": "i32",
        }
        consts = dict(N=n_b, K=k_b, SPLIT=c.split, STAGES=c.stages, SWAP=bool(c.swap), num_warps=c.warps)
        names = list(fn.arg_names)
        full = {nm: sig.get(nm, "constexpr") for nm in names}
        attrs = {(names.index(nm),): [["tt.divisibility", 16]] for nm in ("out_ptr", "ws_ptr", "cnt_ptr")}
        src = GluonASTSource(
            fn=fn, signature=full, constexprs={(names.index(nm),): v for nm, v in consts.items()}, attrs=attrs
        )
        opts = {"num_warps": c.warps}
        if c.maxnreg:
            opts["maxnreg"] = c.maxnreg
        t0 = time.time()
        try:
            ker = triton.compile(src, target=target, options=opts)
            ptx = ker.asm["ptx"]
            wgmma_cnt = len(re.findall(r"wgmma\.mma_async", ptx))
            info = _run_ptxas(ptx, ptxas_bin)
            print(f"  [OK] {tag} ({time.time() - t0:.1f}s) smem={ker.metadata.shared}B wgmma={wgmma_cnt} | {info}",
                  flush=True)
        except Exception as exc:
            print(f"  [FAIL] {tag}: {type(exc).__name__}: {str(exc)[:600]}", flush=True)


# --------------------------------------------------------------------------- GPU Execution & Timing Context

class BenchCtx:
    def __init__(self, w: dict[str, torch.Tensor]) -> None:
        self.w = {k: v.contiguous() for k, v in w.items()}
        router_f32 = self.w["router"].float()
        hi = router_f32.to(torch.bfloat16).contiguous()
        lo = (router_f32 - hi.float()).to(torch.bfloat16).contiguous()
        self.router_hl = (hi, lo)
        self.ws = torch.zeros(WS_FLOATS, device="cuda", dtype=torch.float32)
        self.cnt = torch.zeros(COUNTERS, device="cuda", dtype=torch.int32)
        self.triton_r_desc: dict = {}
        self.gluon_r_desc: dict = {}
        self.gluon_w_desc: dict = {}
        self.gluon_x_layout: dict = {}

    def get_x_layout(self, bm: int, bk: int):
        key = (bm, bk)
        if key not in self.gluon_x_layout:
            self.gluon_x_layout[key] = gl.NVMMASharedLayout.get_default_for([bm, bk], gl.bfloat16)
        return self.gluon_x_layout[key]

    def get_triton_router_desc(self, bn: int, bk: int):
        key = (bn, bk)
        if key not in self.triton_r_desc:
            hi, lo = self.router_hl
            self.triton_r_desc[key] = (
                TensorDescriptor.from_tensor(hi, [bn, bk]),
                TensorDescriptor.from_tensor(lo, [bn, bk]),
            )
        return self.triton_r_desc[key]

    def get_gluon_router_desc(self, bn: int, bk: int):
        key = (bn, bk)
        if key not in self.gluon_r_desc:
            hi, lo = self.router_hl
            wl = gl.NVMMASharedLayout.get_default_for([bn, bk], gl.bfloat16)
            self.gluon_r_desc[key] = (
                GDesc.from_tensor(hi, [bn, bk], wl),
                GDesc.from_tensor(lo, [bn, bk], wl),
            )
        return self.gluon_r_desc[key]

    def get_gluon_w_desc(self, layer: str, bn: int, bk: int):
        key = (layer, bn, bk)
        if key not in self.gluon_w_desc:
            wl = gl.NVMMASharedLayout.get_default_for([bn, bk], gl.bfloat16)
            self.gluon_w_desc[key] = GDesc.from_tensor(self.w[layer], [bn, bk], wl)
        return self.gluon_w_desc[key]

    def launch_router(self, x: torch.Tensor, c: RCfg) -> torch.Tensor:
        m = x.shape[0]
        n, k = SHAPES["router"]
        out = torch.empty(m, n, device=x.device, dtype=torch.float32)
        grid = (triton.cdiv(m, c.bm) * (n // c.bn) * c.split,)
        if c.kind == "triton_v2":
            hi_d, lo_d = self.get_triton_router_desc(c.bn, c.bk)
            xd = TensorDescriptor.from_tensor(x, [c.bm, c.bk])
            _router_big_v2[grid](
                xd, hi_d, lo_d, out, self.ws, self.cnt, m,
                N=n, K=k, SPLIT=c.split, BM=c.bm, BN=c.bn, BK=c.bk, SWAP=bool(c.swap),
                num_warps=c.warps, num_stages=c.stages,
            )
            return out
        hi_d, lo_d = self.get_gluon_router_desc(c.bn, c.bk)
        xl = self.get_x_layout(c.bm, c.bk)
        xd = GDesc.from_tensor(x, [c.bm, c.bk], xl)
        _gluon_router_pipe[grid](
            xd, hi_d, lo_d, out, self.ws, self.cnt, m,
            N=n, K=k, SPLIT=c.split, STAGES=c.stages, SWAP=bool(c.swap),
            num_warps=c.warps,
        )
        return out

    def launch_bf16(self, layer: str, x: torch.Tensor, c: BCfg) -> torch.Tensor:
        m = x.shape[0]
        n, k = SHAPES[layer]
        out = torch.empty(m, n, device=x.device, dtype=torch.bfloat16)
        grid = (triton.cdiv(m, c.bm) * (n // c.bn) * c.split,)
        wd = self.get_gluon_w_desc(layer, c.bn, c.bk)
        xl = self.get_x_layout(c.bm, c.bk)
        xd = GDesc.from_tensor(x, [c.bm, c.bk], xl)
        if c.kind == "gluon_ws":
            _gluon_big_ws[grid](
                xd, wd, out, self.ws, self.cnt, m,
                N=n, K=k, SPLIT=c.split, STAGES=c.stages, SWAP=bool(c.swap),
                num_warps=c.warps, maxnreg=c.maxnreg if c.maxnreg else 168,
            )
        else:
            _gluon_big_pipe[grid](
                xd, wd, out, self.ws, self.cnt, m,
                N=n, K=k, SPLIT=c.split, STAGES=c.stages, SWAP=bool(c.swap),
                num_warps=c.warps,
            )
        return out


def graph_of(fn):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    g.replay()
    torch.cuda.synchronize()
    return g, out


def paired_time(g_base, g_cand, flush, reps: int, time_replay_fn) -> tuple[float, float, float]:
    d, tb, tc = [], [], []
    for r in range(reps):
        order = (g_base, g_cand) if r % 2 == 0 else (g_cand, g_base)
        t = {}
        for g in order:
            t[id(g)] = max(time_replay_fn(g, flush) - OVERHEAD_US, 1e-3)
        tb.append(t[id(g_base)])
        tc.append(t[id(g_cand)])
        d.append(t[id(g_cand)] - t[id(g_base)])
    return statistics.median(tb), statistics.median(tc), statistics.median(d)


def verify_nearby(ctx: BenchCtx, w: dict, reference, layer: str, c_128, c_256) -> bool:
    ok_all = True
    for m_test in (128, 130, 256, 512):
        cfg = c_128 if m_test <= 192 else c_256
        if cfg is None:
            continue
        # Check workspace fit for m_test=512 inside WS_FLOATS
        n, _ = SHAPES[layer]
        num_m = triton.cdiv(m_test, cfg.bm)
        if cfg.split * num_m * cfg.bm * n > WS_FLOATS or num_m * (n // cfg.bn) > COUNTERS:
            continue
        for dist, seed in (("normal", 11), ("outliers", 12)):
            data = reference.generate_input(m_test, layer, seed, dist=dist)
            out = (
                ctx.launch_router(data["x"], cfg)
                if layer == "router"
                else ctx.launch_bf16(layer, data["x"], cfg)
            )
            torch.cuda.synchronize()
            ok, msg = reference.check_implementation(data, out, w)
            if not ok:
                ok_all = False
                print(f"  [FAIL] nearby check {layer} m={m_test} {dist}: {msg}", flush=True)
    print(f"  [CHECK] {layer} correctness (m=128, 130, 256, 512 normal+outliers): {'PASS' if ok_all else 'FAIL'}",
          flush=True)
    return ok_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile", action="store_true", help="CPU-only Gluon/Triton compile + ptxas report")
    ap.add_argument("--reps", type=int, default=35, help="Paired cold-L2 graph timing reps per candidate")
    ap.add_argument("--final-reps", type=int, default=90, help="Paired reps for top-2 re-verification")
    ap.add_argument("--only", default="", help="Comma-separated layers or layer:m filters")
    ap.add_argument("--arch", default="all", choices=("all", "b1", "b2", "pipe", "ws", "triton"))
    ap.add_argument("--cold", action="store_true")
    args = ap.parse_args()

    if args.compile:
        compile_report(args.only)
        return

    from common import LAYER_ORDER, load_module, reference, weights
    from harness import time_replay

    t0 = time.time()
    w = weights()
    cp2 = load_module(BASE_CP, "cp2")
    cp2_state = cp2.prepare({k: v.clone() for k, v in w.items()})
    ctx = BenchCtx(w)
    flush = torch.empty(FLUSH_BYTES // 4, device="cuda", dtype=torch.float32).normal_()
    print(f"Loaded {BASE_CP} in {time.time() - t0:.1f}s | reps={args.reps} final_reps={args.final_reps}", flush=True)

    filters = [s.strip() for s in args.only.split(",") if s.strip()]

    def want(layer: str, m: int) -> bool:
        if args.arch == "b1" and layer != "router":
            return False
        if args.arch == "b2" and layer == "router":
            return False
        if not filters:
            return True
        return any(f == layer or f == f"{layer}:{m}" or f == str(m) for f in filters)

    base_load = {128: 0.0, 256: 0.0}
    best_load = {128: 0.0, 256: 0.0}
    winners: dict[tuple[str, int], tuple] = {}

    for layer in LAYER_ORDER:
        n, k = SHAPES[layer]
        for m in (128, 256):
            if not want(layer, m):
                continue
            sol = reference.speed_of_light_ns(m, layer, HW) / 1e3
            data = reference.generate_input(m, layer, 1)
            x = data["x"]
            g_base, out_base = graph_of(lambda: cp2.kernel_fn(data, cp2_state))
            ok_b, msg_b = reference.check_implementation(data, out_base, w)
            print(f"\n================ {layer} m={m} (N={n}, K={k}, SOL={sol:.2f} us) base={'ok' if ok_b else msg_b} ================",
                  flush=True)

            cands = b1_router_candidates(m) if layer == "router" else b2_bf16_candidates(layer, m)
            if args.arch == "pipe":
                cands = [c for c in cands if c.kind == "gluon_pipe"]
            elif args.arch == "ws":
                cands = [c for c in cands if c.kind == "gluon_ws"]
            elif args.arch == "triton":
                cands = [c for c in cands if c.kind == "triton_v2"]

            scored = []
            last_tb = None
            for c in cands:
                ctas = triton.cdiv(m, c.bm) * (n // c.bn) * c.split
                smem_kb = (r_smem(c) if layer == "router" else b_smem(c)) // 1024
                name = (f"{c.kind:10s} BM{c.bm:3d} BN{c.bn:3d} BK{c.bk:3d} SP{c.split:2d} "
                        f"SW{c.swap} W{c.warps} S{c.stages} (ctas={ctas:3d}, {smem_kb:3d}KB)")
                fn = (lambda c=c: ctx.launch_router(x, c)) if layer == "router" else (lambda c=c: ctx.launch_bf16(layer, x, c))
                try:
                    g_cand, out_cand = graph_of(fn)
                    ok, msg = reference.check_implementation(data, out_cand, w)
                    if not ok:
                        print(f"  {name}: WRONG {msg}", flush=True)
                        continue
                    tb, tc, d = paired_time(g_base, g_cand, flush, args.reps, time_replay)
                    last_tb = tb
                    scored.append((d, tc, tb, c, name, fn))
                    print(f"  {name}: cp2 {tb:6.2f} | cand {tc:6.2f} | delta {d:+6.2f} us", flush=True)
                except Exception as exc:
                    print(f"  {name}: FAILED {type(exc).__name__}: {str(exc)[:300]}", flush=True)

            if not scored:
                continue
            scored.sort(key=lambda item: item[0])
            finals = []
            for _, _, _, c, name, fn in scored[:2]:
                g_cand, _ = graph_of(fn)
                tb, tc, d = paired_time(g_base, g_cand, flush, args.final_reps, time_replay)
                finals.append((d, tc, tb, c, name))
            finals.sort(key=lambda item: item[0])
            best_d, best_tc, best_tb, best_c, best_name = finals[0]
            winners[(layer, m)] = (best_c, best_tb, best_tc, best_d, sol)
            base_load[m] += best_tb
            best_load[m] += min(best_tb, best_tc)
            print(f"  >>> BEST {layer} m={m}: {best_name} -> cp2 {best_tb:.2f} us | cand {best_tc:.2f} us | "
                  f"delta {best_d:+.2f} us (SOL {sol:.2f} us)", flush=True)

        c128 = winners.get((layer, 128), (None,))[0]
        c256 = winners.get((layer, 256), (None,))[0]
        if c128 is not None or c256 is not None:
            verify_nearby(ctx, w, reference, layer, c128, c256)

    print("\n================ TRACK B SUMMARY (vs cp2_gluon_router.py) ================", flush=True)
    for (layer, m), (c, tb, tc, d, sol) in winners.items():
        status = "WIN" if d < -0.15 else ("TIE" if d <= 0.15 else "SLOWER")
        print(f"  {layer:9s} m={m:3d}: cp2 {tb:6.2f} -> cand {tc:6.2f} us ({d:+6.2f} us, SOL {sol:5.2f} us) "
              f"[{status}] {c}", flush=True)
    for m, load_name in ((128, "verify 16"), (256, "verify 32")):
        if base_load[m] > 0:
            saved = base_load[m] - best_load[m]
            ratio = base_load[m] / best_load[m]
            print(f"  LOAD {load_name} (m={m}): cp2 sum {base_load[m]:.2f} us -> best sum {best_load[m]:.2f} us "
                  f"(saved {saved:+.2f} us, {ratio:.3f}x)", flush=True)
    print(f"Total elapsed: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
