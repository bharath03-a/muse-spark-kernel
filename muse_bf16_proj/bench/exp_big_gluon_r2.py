"""Track B Round 2: _big_gemm_v2 & _router_big_v2 (@triton.jit TMA + SWAP + WARP_SPEC + deterministic
packed split-K) + Gluon SWAP=1 STAGES=8 pipe vs checkpoints/cp3_gluon_v2.py.

Key findings from Round 1 incorporated here:
  1. When SPLIT > 1, @triton.jit TMA software-pipelined loops beat Gluon manual rings by ~0.7-1.0 us,
     while our packed tile-contiguous stride-1 workspace cut router m=128/256 by -0.83/-0.90 us.
  2. Determinism requirement (5000 CUDA graph replays bit-exact):
     - SPLIT == 2: `total = acc + load(1 - pid_s)` is 100% bitwise deterministic (IEEE-754 addition is
       commutative) and cuts workspace L2 reads in half (1 load instead of 2).
     - SPLIT > 2: sums in canonical order `s = 0, 1, ..., SPLIT - 1` from the packed stride-1 tile
       workspace (`ws_tile + s * TILE_ELEMS`), preserving 100% bitwise determinism across 5000 replays.
  3. On pooler m=256 (SPLIT=1), Gluon `SWAP=1, STAGES=8` (`BM=128, BN=64, BK=64`) won by -0.61 us;
     we test both `_big_gemm_v2` (`SWAP=0/1`, `WARP_SPEC=False/True`, `BN=32/64/128`, `BK=64/128`) and
     `gluon_pipe` (`SWAP=1, STAGES=8`) across qkv_gate, pooler, unpooler, and o.

Usage:
    modal run modal_app.py::compile_only --cmd "exp_big_gluon_r2 --compile"
    modal run modal_app.py::micro --name exp_big_gluon_r2 --args "--reps 35"
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

BASE_CP = "checkpoints/cp3_gluon_v2.py"
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
WS_FLOATS = 4 * 2**20
SUB_WS_FLOATS = 2 * 2**20
COUNTERS = 8192
SMEM_LIMIT = 224 * 1024
SMEM_BUDGET = 200 * 1024


class Cfg(NamedTuple):
    kind: str       # "triton_v2", "gluon_pipe"
    bm: int
    bn: int
    bk: int
    split: int
    swap: int       # 0: acc [BM, BN], 1: acc [BN, BM]
    warps: int
    stages: int
    wspec: int = 0  # tl.range(..., warp_specialize=bool(wspec))


def smem_bytes(c: Cfg, is_router: bool = False) -> int:
    mult = 2 if is_router else 1
    return c.stages * (c.bm + mult * c.bn) * c.bk * 2


def fit_stages(c: Cfg, is_router: bool = False) -> Cfg:
    s = c.stages
    limit = SMEM_BUDGET if c.kind == "triton_v2" else SMEM_LIMIT
    while s > 2 and smem_bytes(c._replace(stages=s), is_router) > limit:
        s -= 1
    return c._replace(stages=s)


def wgmma_tile_ok(bm: int, bn: int, swap: int, warps: int, kind: str) -> bool:
    d0 = bn if swap else bm
    d1 = bm if swap else bn
    if d0 < 64 or d0 % 64 != 0 or d1 < 16 or d1 % 8 != 0 or d1 > 256:
        return False
    if kind == "gluon_pipe":
        if warps == 4:
            return True
        if warps == 8:
            return d0 >= 128
        return False
    return True


def valid_cfg(c: Cfg, layer: str, m: int) -> bool:
    n, k = SHAPES[layer]
    is_router = layer == "router"
    num_m = triton.cdiv(m, c.bm)
    if n % c.bn != 0 or k % c.bk != 0 or (k // c.bk) < c.split:
        return False
    if (k // c.bk) // c.split < c.stages or c.stages < 2:
        return False
    if smem_bytes(c, is_router) > SMEM_LIMIT:
        return False
    if c.split > 1 and c.split * num_m * c.bm * n > SUB_WS_FLOATS:
        return False
    if num_m * (n // c.bn) > COUNTERS:
        return False
    if not wgmma_tile_ok(c.bm, c.bn, c.swap, c.warps, c.kind):
        return False
    if c.kind == "gluon_pipe" and c.bm >= 256 and c.split > 1 and c.warps == 4:
        return False
    return True


# --------------------------------------------------------------------------- @triton.jit Kernels (_big_gemm_v2, _router_big_v2)

@triton.jit(do_not_specialize=["M"])
def _big_gemm_v2(
    x_desc, w_desc, out_ptr, ws_ptr, cnt_ptr, M,
    N: tl.constexpr, K: tl.constexpr, SPLIT: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    SWAP: tl.constexpr, WSPEC: tl.constexpr,
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
    for ki in tl.range(k0, k1, warp_specialize=WSPEC):
        a = x_desc.load([pid_m * BM, ki * BK])
        b = w_desc.load([pid_n * BN, ki * BK])
        if SWAP:
            acc = tl.dot(b, a.T, acc)
        else:
            acc = tl.dot(a, b.T, acc)
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
        tl.store(out_ptr + out_idx, acc.to(out_ptr.dtype.element_ty), mask=mask)
    else:
        TILE_ELEMS: tl.constexpr = BM * BN
        tile_id = pid_n * num_m + pid_m
        ws_tile = ws_ptr + tile_id * (SPLIT * TILE_ELEMS) + ws_offs
        tl.store(ws_tile + pid_s * TILE_ELEMS, acc)
        tl.debug_barrier()
        cnt = cnt_ptr + tile_id
        done = tl.atomic_add(cnt, 1, sem="acq_rel", scope="gpu")
        if done == SPLIT - 1:
            if SPLIT == 2:
                # IEEE-754 addition is commutative (a + b == b + a): 100% bitwise deterministic & 1 load!
                total = acc + tl.load(ws_tile + (1 - pid_s) * TILE_ELEMS, cache_modifier=".cg")
            else:
                # Canonical order s = 0..SPLIT-1 guarantees bitwise determinism across 5000 replays.
                total = tl.zeros_like(acc)
                for s in tl.static_range(SPLIT):
                    total += tl.load(ws_tile + s * TILE_ELEMS, cache_modifier=".cg")
            tl.store(out_ptr + out_idx, total.to(out_ptr.dtype.element_ty), mask=mask)
            tl.atomic_xchg(cnt, 0)


@triton.jit(do_not_specialize=["M"])
def _router_big_v2(
    x_desc, hi_desc, lo_desc, out_ptr, ws_ptr, cnt_ptr, M,
    N: tl.constexpr, K: tl.constexpr, SPLIT: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    SWAP: tl.constexpr, WSPEC: tl.constexpr,
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
    for ki in tl.range(k0, k1, warp_specialize=WSPEC):
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
            if SPLIT == 2:
                total = acc + tl.load(ws_tile + (1 - pid_s) * TILE_ELEMS, cache_modifier=".cg")
            else:
                total = tl.zeros_like(acc)
                for s in tl.static_range(SPLIT):
                    total += tl.load(ws_tile + s * TILE_ELEMS, cache_modifier=".cg")
            tl.store(out_ptr + out_idx, total, mask=mask)
            tl.atomic_xchg(cnt, 0)


# --------------------------------------------------------------------------- Gluon Kernels (_gluon_big_pipe)

if HAVE_GLUON:

    @gluon.jit
    def _gluon_epilogue(
        acc, out_ptr, ws_ptr, cnt_ptr, M, pid_m, pid_n, pid_s, num_m,
        N: gl.constexpr, SPLIT: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr,
        SWAP: gl.constexpr, D0: gl.constexpr, D1: gl.constexpr, mma: gl.constexpr,
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
            gl.store(out_ptrs, acc.to(gl.bfloat16), mask=mask)
        else:
            TILE_ELEMS: gl.constexpr = BM * BN
            tile_id = pid_n * num_m + pid_m
            ws_tile = ws_ptr + tile_id * (SPLIT * TILE_ELEMS) + ws_offs
            gl.store(ws_tile + pid_s * TILE_ELEMS, acc)
            gl.barrier()
            cnt = cnt_ptr + tile_id
            done = gl.atomic_add(cnt, 1, sem="acq_rel", scope="gpu")
            if done == SPLIT - 1:
                if SPLIT == 2:
                    total = acc + gl.load(ws_tile + (1 - pid_s) * TILE_ELEMS, cache_modifier=".cg")
                else:
                    total = gl.zeros([D0, D1], gl.float32, layout=mma)
                    for s in gl.static_range(SPLIT):
                        total += gl.load(ws_tile + s * TILE_ELEMS, cache_modifier=".cg")
                gl.store(out_ptrs, total.to(gl.bfloat16), mask=mask)
                gl.atomic_xchg(cnt, 0)

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
            N, SPLIT, BM, BN, SWAP, D0, D1, mma,
        )


# --------------------------------------------------------------------------- Candidate Sets

def candidates_for(layer: str, m: int) -> list[Cfg]:
    raw: list[Cfg] = []
    T = lambda bm, bn, bk, sp, sw, w, st, ws=0: fit_stages(
        Cfg("triton_v2", bm, bn, bk, sp, sw, w, st, ws), is_router=(layer == "router")
    )
    G = lambda bm, bn, bk, sp, sw, w, st: Cfg("gluon_pipe", bm, bn, bk, sp, sw, w, st, 0)

    if layer == "qkv_gate":
        if m <= 192:
            raw = [
                # BM=128, BN=32 (72 N-tiles -> SPLIT=2 gives 144 CTAs, SPLIT=1 gives 72 CTAs)
                T(128, 32, 128, 2, 0, 4, 4),
                T(128, 32, 128, 2, 0, 4, 5),
                T(128, 32, 64, 2, 0, 4, 7),
                T(128, 32, 64, 2, 0, 4, 8),
                # BM=64, BN=64 (72 M*N tiles -> SPLIT=2 gives 144 CTAs with 1-load deterministic reduce!)
                T(64, 64, 128, 2, 0, 4, 4),
                T(64, 64, 128, 2, 0, 4, 5),
                T(64, 64, 128, 2, 1, 4, 4),
                T(64, 64, 128, 2, 0, 8, 4),
                T(64, 64, 64, 2, 0, 4, 7),
                T(64, 64, 64, 2, 0, 4, 8),
                T(64, 64, 64, 2, 1, 4, 7),
                # BM=128, BN=64 (36 N-tiles -> SPLIT=2 is 72 CTAs, SPLIT=3 is 108 CTAs, SPLIT=4 is 144 CTAs)
                T(128, 64, 128, 2, 0, 4, 4),
                T(128, 64, 128, 2, 1, 4, 4),
                T(128, 64, 128, 3, 0, 4, 4),
                T(128, 64, 128, 3, 1, 4, 4),
                T(128, 64, 128, 4, 0, 4, 4),
                T(128, 64, 64, 2, 0, 4, 7),
                T(128, 64, 64, 2, 1, 4, 7),
                T(128, 64, 64, 3, 0, 4, 7),
                T(128, 64, 64, 3, 0, 4, 8),
                T(128, 64, 64, 3, 1, 4, 7),
                T(128, 64, 64, 3, 0, 8, 7),
                T(128, 64, 64, 3, 0, 4, 6, 1),
                T(128, 64, 64, 4, 0, 4, 7),
                # BM=64, BN=128 (36 M*N tiles -> SPLIT=2 is 72 CTAs, SPLIT=3 is 108 CTAs)
                T(64, 128, 64, 2, 0, 4, 7),
                T(64, 128, 64, 3, 0, 4, 7),
                T(64, 128, 64, 3, 1, 4, 7),
                T(64, 128, 128, 3, 0, 4, 4),
            ]
        else:
            raw = [
                # BM=128, BN=32, SPLIT=1 -> 2 * 72 = 144 CTAs with ZERO split-K!
                T(128, 32, 128, 1, 0, 4, 4),
                T(128, 32, 128, 1, 0, 4, 5),
                T(128, 32, 64, 1, 0, 4, 7),
                T(128, 32, 64, 1, 0, 4, 8),
                G(128, 32, 128, 1, 0, 4, 4),
                G(128, 32, 64, 1, 0, 4, 8),
                # BM=256, BN=32, SPLIT=1/2 -> reads W strictly ONCE!
                T(256, 32, 64, 1, 0, 4, 5),
                T(256, 32, 64, 2, 0, 4, 5),
                T(256, 32, 64, 2, 0, 8, 5),
                # BM=128, BN=64, SPLIT=1 (72 CTAs) & SPLIT=2 (144 CTAs, 1-load deterministic reduce)
                T(128, 64, 128, 1, 0, 4, 4),
                T(128, 64, 128, 1, 1, 4, 4),
                T(128, 64, 64, 1, 0, 4, 7),
                T(128, 64, 64, 1, 1, 4, 7),
                G(128, 64, 64, 1, 1, 4, 8),
                T(128, 64, 128, 2, 0, 4, 4),
                T(128, 64, 128, 2, 1, 4, 4),
                T(128, 64, 64, 2, 0, 4, 7),
                T(128, 64, 64, 2, 1, 4, 7),
                T(128, 64, 64, 2, 0, 8, 7),
                T(128, 64, 64, 2, 0, 4, 6, 1),
                # BM=64, BN=128, SPLIT=1, 2
                T(64, 128, 64, 1, 0, 4, 7),
                T(64, 128, 64, 2, 0, 4, 7),
                T(64, 128, 128, 2, 0, 4, 4),
                # BM=128, BN=128, SPLIT=2, 3 (72 / 108 CTAs)
                T(128, 128, 64, 2, 0, 4, 6),
                T(128, 128, 64, 2, 0, 8, 6),
                T(128, 128, 64, 3, 0, 4, 6),
                T(128, 128, 64, 3, 0, 8, 6),
                T(128, 128, 64, 3, 0, 4, 6, 1),
            ]
    elif layer == "pooler":
        if m <= 192:
            raw = [
                # BM=64, BN=128, SPLIT=2 (128 CTAs; tests 1-load deterministic reduce vs cp3)
                T(64, 128, 64, 2, 0, 4, 7),
                T(64, 128, 64, 2, 1, 4, 7),
                T(64, 128, 64, 2, 0, 8, 7),
                T(64, 128, 128, 2, 0, 4, 4),
                T(64, 128, 64, 2, 0, 4, 6, 1),
                # BM=128, BN=64, SPLIT=2 (128 CTAs; reads W strictly ONCE + 1-load deterministic reduce)
                T(128, 64, 64, 2, 0, 4, 7),
                T(128, 64, 64, 2, 0, 4, 8),
                T(128, 64, 64, 2, 1, 4, 7),
                T(128, 64, 64, 2, 0, 8, 7),
                T(128, 64, 128, 2, 0, 4, 4),
                T(128, 64, 128, 2, 1, 4, 4),
                T(128, 64, 64, 2, 0, 4, 6, 1),
                # BM=128, BN=32, SPLIT=1 (128 CTAs; reads W strictly ONCE + ZERO split-K!)
                T(128, 32, 128, 1, 0, 4, 4),
                T(128, 32, 64, 1, 0, 4, 8),
                G(128, 32, 128, 1, 0, 4, 4),
                # BM=64, BN=64, SPLIT=1 (128 CTAs; ZERO split-K)
                T(64, 64, 128, 1, 0, 4, 4),
                T(64, 64, 64, 1, 0, 4, 8),
                G(64, 64, 64, 1, 1, 4, 8),
            ]
        else:
            raw = [
                # cp3 winner is gluon_pipe(128, 64, 64, 1, 1, 4, 8); test nearby triton_v2 & gluon_pipe
                G(128, 64, 64, 1, 1, 4, 8),
                G(128, 64, 128, 1, 1, 4, 4),
                T(128, 64, 64, 1, 0, 4, 8),
                T(128, 64, 64, 1, 1, 4, 8),
                T(128, 64, 128, 1, 0, 4, 4),
                T(128, 64, 128, 1, 1, 4, 4),
                # BM=256, BN=32, SPLIT=1 (128 CTAs; reads W strictly ONCE + ZERO split-K!)
                T(256, 32, 64, 1, 0, 4, 5),
                T(256, 32, 64, 1, 0, 8, 5),
                # BM=128, BN=128, SPLIT=2 (128 CTAs; 1-load deterministic reduce)
                T(128, 128, 64, 2, 0, 4, 6),
                T(128, 128, 64, 2, 0, 8, 6),
                T(128, 128, 64, 2, 0, 4, 6, 1),
                T(128, 128, 128, 2, 0, 4, 3),
                # BM=256, BN=64, SPLIT=2 (128 CTAs; reads W strictly ONCE)
                T(256, 64, 64, 2, 0, 8, 5),
                T(256, 64, 64, 2, 1, 8, 5),
            ]
    elif layer == "unpooler":
        if m <= 192:
            raw = [
                # BM=128, BN=64, SPLIT=1 -> 128 CTAs, reads W strictly ONCE, ZERO split-K
                T(128, 64, 64, 1, 0, 4, 7),
                T(128, 64, 64, 1, 0, 4, 8),
                T(128, 64, 64, 1, 1, 4, 7),
                T(128, 64, 64, 1, 0, 8, 7),
                T(128, 64, 128, 1, 0, 4, 4),
                T(128, 64, 128, 1, 1, 4, 4),
                T(128, 64, 64, 1, 0, 4, 6, 1),
                G(128, 64, 64, 1, 1, 4, 8),
                # BM=64, BN=128, SPLIT=1 -> 128 CTAs
                T(64, 128, 64, 1, 0, 4, 7),
                T(64, 128, 64, 1, 1, 4, 7),
                T(64, 128, 64, 1, 0, 8, 7),
                T(64, 128, 128, 1, 0, 4, 4),
                T(64, 128, 64, 1, 0, 4, 6, 1),
                # BM=128, BN=128, SPLIT=2 -> 128 CTAs, 1-load deterministic reduce
                T(128, 128, 64, 2, 0, 4, 6),
                T(128, 128, 64, 2, 0, 8, 6),
            ]
        else:
            raw = [
                # BM=128, BN=128, SPLIT=1 -> 128 CTAs, ZERO split-K
                T(128, 128, 64, 1, 0, 4, 6),
                T(128, 128, 64, 1, 0, 8, 6),
                T(128, 128, 64, 1, 1, 4, 6),
                T(128, 128, 64, 1, 0, 4, 6, 1),
                T(128, 128, 64, 1, 0, 8, 6, 1),
                T(128, 128, 128, 1, 0, 4, 3),
                T(128, 128, 128, 1, 0, 8, 3),
                # BM=256, BN=64, SPLIT=1 -> 128 CTAs, reads W strictly ONCE, ZERO split-K
                T(256, 64, 64, 1, 0, 8, 5),
                T(256, 64, 64, 1, 1, 8, 5),
                T(256, 64, 64, 1, 1, 4, 5),
                T(256, 64, 128, 1, 0, 8, 2),
                # BM=128, BN=64, SPLIT=1 -> 256 CTAs (including gluon_pipe STAGES=8 that won on pooler m=256)
                G(128, 64, 64, 1, 1, 4, 8),
                T(128, 64, 64, 1, 0, 4, 8),
                T(128, 64, 64, 1, 1, 4, 8),
                T(128, 64, 128, 1, 0, 4, 4),
            ]
    elif layer == "o":
        if m <= 192:
            raw = [
                # BM=128, BN=64, SPLIT=1 -> 128 CTAs, K=1024 is 8 steps of BK=128 or 16 steps of BK=64
                T(128, 64, 128, 1, 0, 4, 4),
                T(128, 64, 128, 1, 1, 4, 4),
                T(128, 64, 128, 1, 0, 8, 4),
                T(128, 64, 64, 1, 0, 4, 7),
                T(128, 64, 64, 1, 0, 4, 8),
                T(128, 64, 64, 1, 1, 4, 7),
                T(128, 64, 64, 1, 0, 4, 6, 1),
                G(128, 64, 64, 1, 1, 4, 8),
                G(128, 64, 128, 1, 1, 4, 4),
                # BM=64, BN=128, SPLIT=1 -> 128 CTAs
                T(64, 128, 128, 1, 0, 4, 4),
                T(64, 128, 64, 1, 0, 4, 7),
                T(64, 128, 64, 1, 1, 4, 7),
                # BM=128, BN=128, SPLIT=1 -> 64 CTAs
                T(128, 128, 128, 1, 0, 4, 3),
                T(128, 128, 64, 1, 0, 4, 6),
                T(128, 128, 64, 1, 0, 8, 6),
            ]
        else:
            raw = [
                # BM=128, BN=128, SPLIT=1 -> 128 CTAs
                T(128, 128, 128, 1, 0, 4, 3),
                T(128, 128, 128, 1, 0, 8, 3),
                T(128, 128, 64, 1, 0, 4, 5),
                T(128, 128, 64, 1, 0, 4, 6),
                T(128, 128, 64, 1, 0, 8, 6),
                T(128, 128, 64, 1, 1, 4, 6),
                T(128, 128, 64, 1, 0, 4, 6, 1),
                # BM=256, BN=64, SPLIT=1 -> 128 CTAs, reads W strictly ONCE
                T(256, 64, 128, 1, 0, 8, 2),
                T(256, 64, 128, 1, 1, 4, 2),
                T(256, 64, 64, 1, 0, 8, 5),
                T(256, 64, 64, 1, 1, 4, 5),
                # BM=128, BN=64, SPLIT=1 -> 256 CTAs
                G(128, 64, 64, 1, 1, 4, 8),
                G(128, 64, 128, 1, 1, 4, 4),
                T(128, 64, 128, 1, 0, 4, 4),
                T(128, 64, 64, 1, 0, 4, 6),
            ]
    elif layer == "router":
        if m <= 192:
            raw = [
                # Baseline in cp3: (64, 32, 128, 8, 0, 4, 4); test BN=16, SPLIT=2/4/6/8/16, stages, wspec
                T(64, 32, 128, 8, 0, 4, 4),
                T(64, 32, 128, 8, 0, 4, 3),
                T(64, 32, 128, 8, 0, 4, 5),
                T(64, 32, 128, 6, 0, 4, 4),
                T(64, 32, 128, 4, 0, 4, 4),
                T(64, 32, 128, 2, 0, 4, 4),
                T(64, 32, 64, 8, 0, 4, 6),
                T(64, 32, 64, 16, 0, 4, 4),
                T(64, 32, 128, 8, 0, 4, 4, 1),
                # BN=16 (16 N-tiles -> SPLIT=4 gives 128 CTAs, SPLIT=2 gives 64 CTAs with 1-load reduce!)
                T(64, 16, 128, 4, 0, 4, 4),
                T(64, 16, 128, 4, 0, 4, 3),
                T(64, 16, 128, 8, 0, 4, 3),
                T(64, 16, 128, 2, 0, 4, 4),
                T(64, 16, 64, 8, 0, 4, 4),
                # BM=128, BN=16 (num_m=1 -> reads hi/lo strictly ONCE! 16 N-tiles * SPLIT=8 = 128 CTAs)
                T(128, 16, 128, 8, 0, 4, 3),
                T(128, 16, 128, 4, 0, 4, 4),
                T(128, 16, 128, 2, 0, 4, 4),
                T(128, 16, 64, 8, 0, 4, 4),
                # BM=128, BN=32 (num_m=1 -> reads hi/lo strictly ONCE! 8 N-tiles)
                T(128, 32, 128, 8, 0, 4, 3),
                T(128, 32, 128, 4, 0, 4, 4),
                T(128, 32, 128, 2, 0, 4, 4),
                T(128, 32, 64, 8, 0, 4, 4),
                T(128, 32, 64, 16, 0, 4, 4),
            ]
        else:
            raw = [
                # Baseline in cp3: (64, 32, 128, 4, 0, 4, 4)
                T(64, 32, 128, 4, 0, 4, 4),
                T(64, 32, 128, 4, 0, 4, 3),
                T(64, 32, 128, 4, 0, 4, 5),
                T(64, 32, 128, 6, 0, 4, 4),
                T(64, 32, 128, 8, 0, 4, 4),
                T(64, 32, 128, 2, 0, 4, 4),
                T(64, 32, 64, 4, 0, 4, 6),
                T(64, 32, 64, 8, 0, 4, 4),
                T(64, 32, 128, 4, 0, 4, 4, 1),
                # BN=16: at m=256, 4 * 16 * SPLIT=2 = 128 CTAs with 1-load deterministic reduce!
                T(64, 16, 128, 2, 0, 4, 4),
                T(64, 16, 128, 2, 0, 4, 3),
                T(64, 16, 128, 4, 0, 4, 4),
                T(64, 16, 64, 4, 0, 4, 4),
                T(64, 16, 64, 2, 0, 4, 6),
                # BM=128, BN=16: 2 * 16 * SPLIT=4 = 128 CTAs, or SPLIT=2 = 64 CTAs (1-load reduce)
                T(128, 16, 128, 4, 0, 4, 3),
                T(128, 16, 128, 4, 0, 4, 4),
                T(128, 16, 128, 2, 0, 4, 4),
                T(128, 16, 64, 4, 0, 4, 4),
                # BM=128, BN=32: 2 * 8 * SPLIT=8 = 128 CTAs, or SPLIT=4 = 64 CTAs, or SPLIT=2 = 32 CTAs
                T(128, 32, 128, 4, 0, 4, 3),
                T(128, 32, 128, 8, 0, 4, 3),
                T(128, 32, 128, 2, 0, 4, 4),
                T(128, 32, 64, 8, 0, 4, 4),
                T(128, 32, 64, 4, 0, 4, 4),
            ]

    out: list[Cfg] = []
    for c in raw:
        if c.wspec != 0 or (c.bm == 128 and c.bn == 128 and c.split > 1 and c.warps == 4):
            continue
        if valid_cfg(c, layer, m) and c not in out:
            out.append(c)
    return out


# --------------------------------------------------------------------------- CPU Compile & ptxas Diagnostics

def _run_ptxas(ptx: str, ptxas_bin: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as td:
        path = f"{td}/k.ptx"
        with open(path, "w") as f:
            f.write(ptx)
        res = subprocess.run([ptxas_bin, "-arch=sm_90a", "-v", path, "-o", f"{td}/k.o"],
                             capture_output=True, text=True)
        lines = []
        spilled = False
        for line in (res.stdout + res.stderr).splitlines():
            if "spill" in line and not ("0 bytes spill stores, 0 bytes spill loads" in line):
                spilled = True
            if any(k in line for k in ("registers", "spill", "smem", "cmem", "Warning", "error")):
                lines.append(line.strip())
        return spilled, (" | ".join(lines) if lines else "ok")


def compile_report(only: str = "") -> None:
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import get_ptxas
    from triton.compiler import ASTSource

    ptxas_bin = get_ptxas(90).path
    target = GPUTarget("cuda", 90, 32)

    print("===== Track B Round 2 Compile Report (sm_90a) =====", flush=True)
    seen: set[tuple[str, Cfg]] = set()
    total = ok_cnt = spill_cnt = fail_cnt = 0
    for layer in SHAPES:
        n, k = SHAPES[layer]
        for m in (128, 256):
            for c in candidates_for(layer, m):
                if (layer, c) in seen:
                    continue
                seen.add((layer, c))
                tag = (f"{layer:9s} {c.kind:10s} BM{c.bm:3d} BN{c.bn:3d} BK{c.bk:3d} "
                       f"SP{c.split:2d} SW{c.swap} W{c.warps} S{c.stages} WS{c.wspec}")
                if only and only not in tag:
                    continue
                total += 1
                t0 = time.time()
                try:
                    if c.kind == "triton_v2":
                        fn = _router_big_v2 if layer == "router" else _big_gemm_v2
                        if layer == "router":
                            sig = {
                                "x_desc": f"tensordesc<bf16[{c.bm}, {c.bk}]>",
                                "hi_desc": f"tensordesc<bf16[{c.bn}, {c.bk}]>",
                                "lo_desc": f"tensordesc<bf16[{c.bn}, {c.bk}]>",
                                "out_ptr": "*fp32",
                                "ws_ptr": "*fp32",
                                "cnt_ptr": "*i32",
                                "M": "i32",
                            }
                        else:
                            sig = {
                                "x_desc": f"tensordesc<bf16[{c.bm}, {c.bk}]>",
                                "w_desc": f"tensordesc<bf16[{c.bn}, {c.bk}]>",
                                "out_ptr": "*bf16",
                                "ws_ptr": "*fp32",
                                "cnt_ptr": "*i32",
                                "M": "i32",
                            }
                        consts = dict(
                            N=n, K=k, SPLIT=c.split, BM=c.bm, BN=c.bn, BK=c.bk,
                            SWAP=bool(c.swap), WSPEC=bool(c.wspec),
                        )
                        names = list(fn.arg_names)
                        attrs = {(names.index(nm),): [["tt.divisibility", 16]] for nm in ("out_ptr", "ws_ptr", "cnt_ptr")}
                        src = ASTSource(fn=fn, signature=sig, constexprs=consts, attrs=attrs)
                        opts = {"num_warps": c.warps, "num_stages": c.stages}
                        ker = triton.compile(src, target=target, options=opts)
                    else:
                        from triton.experimental.gluon._runtime import GluonASTSource

                        x = torch.zeros(max(128, c.bm), k, dtype=torch.bfloat16)
                        w = torch.zeros(n, k, dtype=torch.bfloat16)
                        xl = gl.NVMMASharedLayout.get_default_for([c.bm, c.bk], gl.bfloat16)
                        wl = gl.NVMMASharedLayout.get_default_for([c.bn, c.bk], gl.bfloat16)
                        xd = GDesc.from_tensor(x, [c.bm, c.bk], xl)
                        wd = GDesc.from_tensor(w, [c.bn, c.bk], wl)
                        fn = _gluon_big_pipe
                        sig = {
                            "x_desc": xd.__mangle__(),
                            "w_desc": wd.__mangle__(),
                            "out_ptr": "*bf16",
                            "ws_ptr": "*fp32",
                            "cnt_ptr": "*i32",
                            "M": "i32",
                        }
                        consts = dict(N=n, K=k, SPLIT=c.split, STAGES=c.stages, SWAP=bool(c.swap), num_warps=c.warps)
                        names = list(fn.arg_names)
                        full = {nm: sig.get(nm, "constexpr") for nm in names}
                        attrs = {(names.index(nm),): [["tt.divisibility", 16]] for nm in ("out_ptr", "ws_ptr", "cnt_ptr")}
                        src = GluonASTSource(
                            fn=fn,
                            signature=full,
                            constexprs={(names.index(nm),): v for nm, v in consts.items()},
                            attrs=attrs,
                        )
                        ker = triton.compile(src, target=target, options={"num_warps": c.warps})

                    ptx = ker.asm["ptx"]
                    spilled, info = _run_ptxas(ptx, ptxas_bin)
                    if spilled:
                        spill_cnt += 1
                        print(f"  [SPILL] {tag} ({time.time() - t0:.1f}s) smem={ker.metadata.shared}B | {info}",
                              flush=True)
                    else:
                        ok_cnt += 1
                        print(f"  [OK]    {tag} ({time.time() - t0:.1f}s) smem={ker.metadata.shared}B | {info}",
                              flush=True)
                except Exception as exc:
                    fail_cnt += 1
                    print(f"  [FAIL]  {tag}: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
    print(f"\nCompile summary: {ok_cnt}/{total} clean (0 spills), {spill_cnt} spilled, {fail_cnt} failed",
          flush=True)


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
        self.triton_w_desc: dict = {}
        self.gluon_w_desc: dict = {}
        self.gluon_x_layout: dict = {}

    def get_triton_router_desc(self, bn: int, bk: int):
        key = (bn, bk)
        if key not in self.triton_r_desc:
            hi, lo = self.router_hl
            self.triton_r_desc[key] = (
                TensorDescriptor.from_tensor(hi, [bn, bk]),
                TensorDescriptor.from_tensor(lo, [bn, bk]),
            )
        return self.triton_r_desc[key]

    def get_triton_w_desc(self, layer: str, bn: int, bk: int):
        key = (layer, bn, bk)
        if key not in self.triton_w_desc:
            self.triton_w_desc[key] = TensorDescriptor.from_tensor(self.w[layer], [bn, bk])
        return self.triton_w_desc[key]

    def get_gluon_w_desc(self, layer: str, bn: int, bk: int):
        key = (layer, bn, bk)
        if key not in self.gluon_w_desc:
            wl = gl.NVMMASharedLayout.get_default_for([bn, bk], gl.bfloat16)
            self.gluon_w_desc[key] = GDesc.from_tensor(self.w[layer], [bn, bk], wl)
        return self.gluon_w_desc[key]

    def get_x_layout(self, bm: int, bk: int):
        key = (bm, bk)
        if key not in self.gluon_x_layout:
            self.gluon_x_layout[key] = gl.NVMMASharedLayout.get_default_for([bm, bk], gl.bfloat16)
        return self.gluon_x_layout[key]

    def launch(self, layer: str, x: torch.Tensor, c: Cfg) -> torch.Tensor:
        m = x.shape[0]
        n, k = SHAPES[layer]
        grid = (triton.cdiv(m, c.bm) * (n // c.bn) * c.split,)
        if layer == "router":
            out = torch.empty(m, n, device=x.device, dtype=torch.float32)
            hi_d, lo_d = self.get_triton_router_desc(c.bn, c.bk)
            xd = TensorDescriptor.from_tensor(x, [c.bm, c.bk])
            _router_big_v2[grid](
                xd, hi_d, lo_d, out, self.ws, self.cnt, m,
                N=n, K=k, SPLIT=c.split, BM=c.bm, BN=c.bn, BK=c.bk,
                SWAP=bool(c.swap), WSPEC=bool(c.wspec),
                num_warps=c.warps, num_stages=c.stages,
            )
            return out
        out = torch.empty(m, n, device=x.device, dtype=torch.bfloat16)
        if c.kind == "triton_v2":
            wd = self.get_triton_w_desc(layer, c.bn, c.bk)
            xd = TensorDescriptor.from_tensor(x, [c.bm, c.bk])
            _big_gemm_v2[grid](
                xd, wd, out, self.ws, self.cnt, m,
                N=n, K=k, SPLIT=c.split, BM=c.bm, BN=c.bn, BK=c.bk,
                SWAP=bool(c.swap), WSPEC=bool(c.wspec),
                num_warps=c.warps, num_stages=c.stages,
            )
            return out
        wd = self.get_gluon_w_desc(layer, c.bn, c.bk)
        xl = self.get_x_layout(c.bm, c.bk)
        xd = GDesc.from_tensor(x, [c.bm, c.bk], xl)
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


def check_determinism_and_nearby(ctx: BenchCtx, w: dict, reference, layer: str, c_128, c_256) -> bool:
    ok_all = True
    # 1. Nearby shapes & distributions correctness
    for m_test in (128, 130, 256, 512):
        cfg = c_128 if m_test <= 192 else c_256
        if cfg is None:
            continue
        n, _ = SHAPES[layer]
        num_m = triton.cdiv(m_test, cfg.bm)
        if cfg.split * num_m * cfg.bm * n > WS_FLOATS or num_m * (n // cfg.bn) > COUNTERS:
            continue
        for dist, seed in (("normal", 21), ("outliers", 22)):
            data = reference.generate_input(m_test, layer, seed, dist=dist)
            out = ctx.launch(layer, data["x"], cfg)
            torch.cuda.synchronize()
            ok, msg = reference.check_implementation(data, out, w)
            if not ok:
                ok_all = False
                print(f"  [FAIL] nearby {layer} m={m_test} {dist}: {msg}", flush=True)

    # 2. Bitwise determinism across 100 CUDA graph replays
    for m_det, cfg in ((128, c_128), (256, c_256)):
        if cfg is None or cfg.split == 1:
            continue
        data = reference.generate_input(m_det, layer, 99)
        x = data["x"]
        g, out_g = graph_of(lambda: ctx.launch(layer, x, cfg))
        ref_copy = out_g.clone()
        diffs = 0
        for _ in range(100):
            g.replay()
            if not torch.equal(out_g, ref_copy):
                diffs += 1
        if diffs > 0:
            ok_all = False
            print(f"  [FAIL] determinism {layer} m={m_det} {cfg}: {diffs}/100 replays differed!", flush=True)

    print(f"  [CHECK] {layer} correctness + determinism (m=128, 130, 256, 512): {'PASS' if ok_all else 'FAIL'}",
          flush=True)
    return ok_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile", action="store_true", help="CPU compile + ptxas -v spill/reg check")
    ap.add_argument("--reps", type=int, default=35, help="Paired cold-L2 graph timing reps per candidate")
    ap.add_argument("--final-reps", type=int, default=90, help="Paired reps for top-2 re-verification")
    ap.add_argument("--only", default="", help="Comma-separated layers or layer:m filters")
    ap.add_argument("--cold", action="store_true")
    args = ap.parse_args()

    if args.compile:
        compile_report(args.only)
        return

    from common import LAYER_ORDER, load_module, reference, weights
    from harness import time_replay

    t0 = time.time()
    w = weights()
    cp3 = load_module(BASE_CP, "cp3")
    cp3_state = cp3.prepare({k: v.clone() for k, v in w.items()})
    ctx = BenchCtx(w)
    flush = torch.empty(FLUSH_BYTES // 4, device="cuda", dtype=torch.float32).normal_()
    print(f"Loaded {BASE_CP} in {time.time() - t0:.1f}s | reps={args.reps} final_reps={args.final_reps}", flush=True)

    filters = [s.strip() for s in args.only.split(",") if s.strip()]

    def want(layer: str, m: int) -> bool:
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
            g_base, out_base = graph_of(lambda: cp3.kernel_fn(data, cp3_state))
            ok_b, msg_b = reference.check_implementation(data, out_base, w)
            print(f"\n================ {layer} m={m} (N={n}, K={k}, SOL={sol:.2f} us) cp3={'ok' if ok_b else msg_b} ================",
                  flush=True)

            cands = candidates_for(layer, m)
            scored = []
            for c in cands:
                ctas = triton.cdiv(m, c.bm) * (n // c.bn) * c.split
                smem_kb = smem_bytes(c, is_router=(layer == "router")) // 1024
                name = (f"{c.kind:10s} BM{c.bm:3d} BN{c.bn:3d} BK{c.bk:3d} SP{c.split:2d} "
                        f"SW{c.swap} W{c.warps} S{c.stages} WS{c.wspec} (ctas={ctas:3d}, {smem_kb:3d}KB)")
                fn = lambda c=c: ctx.launch(layer, x, c)
                try:
                    g_cand, out_cand = graph_of(fn)
                    ok, msg = reference.check_implementation(data, out_cand, w)
                    if not ok:
                        print(f"  {name}: WRONG {msg}", flush=True)
                        continue
                    tb, tc, d = paired_time(g_base, g_cand, flush, args.reps, time_replay)
                    scored.append((d, tc, tb, c, name, fn))
                    print(f"  {name}: cp3 {tb:6.2f} | cand {tc:6.2f} | delta {d:+6.2f} us", flush=True)
                except Exception as exc:
                    print(f"  {name}: FAILED {type(exc).__name__}: {str(exc)[:240]}", flush=True)

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
            print(f"  >>> BEST {layer} m={m}: {best_name} -> cp3 {best_tb:.2f} us | cand {best_tc:.2f} us | "
                  f"delta {best_d:+.2f} us (SOL {sol:.2f} us)", flush=True)

        c128 = winners.get((layer, 128), (None,))[0]
        c256 = winners.get((layer, 256), (None,))[0]
        if c128 is not None or c256 is not None:
            check_determinism_and_nearby(ctx, w, reference, layer, c128, c256)

    print("\n================ TRACK B ROUND 2 SUMMARY (vs cp3_gluon_v2.py) ================", flush=True)
    for (layer, m), (c, tb, tc, d, sol) in winners.items():
        status = "WIN" if d < -0.15 else ("TIE" if d <= 0.15 else "SLOWER")
        print(f"  {layer:9s} m={m:3d}: cp3 {tb:6.2f} -> cand {tc:6.2f} us ({d:+6.2f} us, SOL {sol:5.2f} us) "
              f"[{status}] {c}", flush=True)
    for m, load_name in ((128, "verify 16"), (256, "verify 32")):
        if base_load[m] > 0:
            saved = base_load[m] - best_load[m]
            ratio = base_load[m] / best_load[m]
            print(f"  LOAD {load_name} (m={m}): cp3 sum {base_load[m]:.2f} us -> best sum {best_load[m]:.2f} us "
                  f"(saved {saved:+.2f} us, {ratio:.3f}x)", flush=True)
    print(f"Total elapsed: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
