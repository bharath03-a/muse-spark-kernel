"""Track A Round 3: First-Principles Attack on `decode 1` (m=1) and Small-m (`pooler` m=8,16,32 & `unpooler` m=16,32).

Compares three architectures against `checkpoints/cp3_gluon_v2.py` using paired cold-L2 CUDA-graph timing:
  1. Architecture M1 (`_bf16_gemv_v2`, `@triton.jit`):
     - Cache-modifier sweeps (`w_mod` in {"", ".cg", ".cs"}, `x_mod` in {"", ".ca"}) to keep `x` (2-16 KB) pinned in
       L1/L2 while streaming `W` (16-64 MB) without cache pollution.
     - Tile & occupancy sweeps: `BLOCK_N in {4, 8, 16, 32}`, `BLOCK_K in {256, 512, 1024, 2048}`, `warps in {4, 8}`,
       `stages in {1, 2, 3, 4}`, including single-step `BLOCK_K=1024` (zero-loop) for `o` (K=1024).
     - `SPLIT=2` with 1-load commutative IEEE-754 atomic last-CTA reduction (`total = r + other`), 100% bitwise
       deterministic across 5000 replays.
  2. Architecture M2 (TMA & Gluon at m=1):
     - M2a (`_triton_tma_gemv`): `@triton.jit` using Hopper `TensorDescriptor` for `W` `[BLOCK_N, BLOCK_K]`.
     - M2b (`_gluon_m1_fma`): `@gluon.jit` TMA + mbarrier ring streaming `W` `[BLOCK_N, BLOCK_K]` into shared memory
       and accumulating `w * x` in FP32 across warps.
     - M2c (`_gluon_gemv_kfold_r2` / `_gluon_gemv_split_r2` at `m=1, BT=8`): WGMMA TMA-ring at `m=1`.
  3. Architecture M3 (Small-m `pooler` m=8,16,32 and `unpooler` m=16,32 refinements):
     - `BN=64, BK=256, STAGES=4` (`UNROLL in {0, 1, 2}`) on `unpooler` m=16,32 and `pooler` m=8,16,32.
     - `BN=128, BK=128, STAGES=4` (`warps=8` -> 2 warpgroups/CTA sharing `x` in smem across 128 rows of `W`, and
       `warps=4`) with `SPLIT in {1, 2}` and `UNROLL in {0, 1, 2}`.
     - `kfold` `R=32, Q=2, BK=256, STAGES=3/4` (`UNROLL in {0, 1, 2}`, 128 CTAs on `pooler`!).

Usage:
    modal run modal_app.py::compile_only --cmd "exp_m1_gemv --compile"
    modal run modal_app.py::micro --name exp_m1_gemv --args "--reps 40"
"""

from __future__ import annotations

import argparse
import math
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
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor as GluonDesc

    HAVE_GLUON = True
except ImportError:
    gluon = None
    HAVE_GLUON = False

BASE_CP = "checkpoints/cp3_gluon_v2.py"
OVERHEAD_US = 4.0
GLUON_SMEM = 220 * 1024
WS_FLOATS = 2 * 2**20
COUNTERS = 8192

# Judge load totals (us) from okc check #1 (job 2fcb56e653a549ad) for geomean impact calculation.
JUDGE_LOAD_TOTALS_US = {
    "decode 1": 40.31,
    "verify 1": 47.40,
    "decode 16": 59.75,
    "decode 32": 63.25,
    "verify 16": 77.57,
    "verify 32": 92.44,
}
SGLANG_LOAD_TOTALS_US = {
    "decode 1": 59.29,
    "verify 1": 75.93,
    "decode 16": 76.29,
    "decode 32": 76.93,
    "verify 16": 86.73,
    "verify 32": 99.62,
}
M_TO_LOAD = {1: "decode 1", 8: "verify 1", 16: "decode 16", 32: "decode 32", 128: "verify 16", 256: "verify 32"}
NEARBY_M = {1: (1,), 8: (2, 7, 8), 16: (9, 12, 16), 32: (17, 24, 32)}


# ==============================================================================
# Architecture M1 & M2a: @triton.jit m=1 GEMV kernels
# ==============================================================================

@triton.jit
def _m1_split_finish(r, out_ptr, ws_ptr, cnt_ptr, offs_n, pid_n, pid_s,
                     SPLIT: tl.constexpr, BLOCK_N: tl.constexpr):
    if SPLIT == 1:
        tl.store(out_ptr + offs_n, r.to(out_ptr.dtype.element_ty))
    else:
        ws_base = ws_ptr + pid_n * (SPLIT * BLOCK_N) + tl.arange(0, BLOCK_N)
        tl.store(ws_base + pid_s * BLOCK_N, r)
        tl.debug_barrier()
        done = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
        if done == SPLIT - 1:
            if SPLIT == 2:
                # Commutative IEEE-754 addition: 100% bitwise deterministic & saves 1 global load
                total = r + tl.load(ws_base + (1 - pid_s) * BLOCK_N, cache_modifier=".cg")
            else:
                total = tl.zeros((BLOCK_N,), dtype=tl.float32)
                for s in tl.static_range(SPLIT):
                    total += tl.load(ws_base + s * BLOCK_N, cache_modifier=".cg")
            tl.store(out_ptr + offs_n, total.to(out_ptr.dtype.element_ty))
            tl.atomic_xchg(cnt_ptr + pid_n, 0)


@triton.jit
def _bf16_gemv_v2(x_ptr, w_ptr, out_ptr, ws_ptr, cnt_ptr,
                  N: tl.constexpr, K: tl.constexpr, SPLIT: tl.constexpr,
                  BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  W_MOD: tl.constexpr, X_MOD: tl.constexpr):
    """Architecture M1: Cache-hinted FMA GEMV with optional 1-step unroll and Split-K."""
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_len: tl.constexpr = K // SPLIT
    STEPS: tl.constexpr = k_len // BLOCK_K

    if STEPS == 1:
        offs_k = tl.max_contiguous(tl.multiple_of(pid_s * k_len + tl.arange(0, BLOCK_K), BLOCK_K), BLOCK_K)
        if W_MOD == 1:
            w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :], cache_modifier=".cg")
        elif W_MOD == 2:
            w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :], cache_modifier=".cs")
        else:
            w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :], eviction_policy="evict_first")
        if X_MOD == 1:
            xv = tl.load(x_ptr + offs_k, cache_modifier=".ca")
        else:
            xv = tl.load(x_ptr + offs_k, eviction_policy="evict_last")
        acc = w.to(tl.float32) * xv.to(tl.float32)[None, :]
    else:
        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for kk in range(0, k_len, BLOCK_K):
            offs_k = tl.max_contiguous(tl.multiple_of(pid_s * k_len + kk + tl.arange(0, BLOCK_K), BLOCK_K), BLOCK_K)
            if W_MOD == 1:
                w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :], cache_modifier=".cg")
            elif W_MOD == 2:
                w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :], cache_modifier=".cs")
            else:
                w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :], eviction_policy="evict_first")
            if X_MOD == 1:
                xv = tl.load(x_ptr + offs_k, cache_modifier=".ca")
            else:
                xv = tl.load(x_ptr + offs_k, eviction_policy="evict_last")
            acc += w.to(tl.float32) * xv.to(tl.float32)[None, :]
    r = tl.sum(acc, axis=1)
    _m1_split_finish(r, out_ptr, ws_ptr, cnt_ptr, offs_n, pid_n, pid_s, SPLIT, BLOCK_N)


@triton.jit
def _triton_tma_gemv(x_ptr, w_desc, out_ptr, ws_ptr, cnt_ptr,
                     N: tl.constexpr, K: tl.constexpr, SPLIT: tl.constexpr,
                     BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """Architecture M2a: TMA TensorDescriptor load of W [BLOCK_N, BLOCK_K] + cached x load."""
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_len: tl.constexpr = K // SPLIT
    k_start = pid_s * k_len
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for kk in range(0, k_len, BLOCK_K):
        offs_k = tl.max_contiguous(tl.multiple_of(k_start + kk + tl.arange(0, BLOCK_K), BLOCK_K), BLOCK_K)
        w = w_desc.load([pid_n * BLOCK_N, k_start + kk])
        xv = tl.load(x_ptr + offs_k, cache_modifier=".ca")
        acc += w.to(tl.float32) * xv.to(tl.float32)[None, :]
    r = tl.sum(acc, axis=1)
    _m1_split_finish(r, out_ptr, ws_ptr, cnt_ptr, offs_n, pid_n, pid_s, SPLIT, BLOCK_N)


# ==============================================================================
# Architecture M2b, M2c & M3: @gluon.jit kernels
# ==============================================================================

if HAVE_GLUON:
    @gluon.jit
    def _gluon_m1_fma(w_desc, x_ptr, out_ptr, ws_ptr, cnt_ptr,
                      N: gl.constexpr, K: gl.constexpr, SPLIT: gl.constexpr,
                      STAGES: gl.constexpr, num_warps: gl.constexpr):
        """Architecture M2b: Gluon TMA + mbarrier ring for W [BN, BK] + register FMA for m=1."""
        BN: gl.constexpr = w_desc.block_type.shape[0]
        BK: gl.constexpr = w_desc.block_type.shape[1]
        KI: gl.constexpr = K // BK
        K_PER: gl.constexpr = KI // SPLIT
        # 32 threads along K (8 bf16/thread = 256 per warp-row), num_warps along BN
        lay: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [num_warps, 1], [1, 0])
        k_lay: gl.constexpr = gl.SliceLayout(0, lay)
        n_lay: gl.constexpr = gl.SliceLayout(1, lay)

        pid_n = gl.program_id(0)
        pid_s = gl.program_id(1)
        n0 = pid_n * BN
        k_start = pid_s * K_PER

        w_bufs = gl.allocate_shared_memory(w_desc.dtype, [STAGES] + w_desc.block_type.shape, w_desc.layout)
        ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
        for i in gl.static_range(STAGES):
            mbarrier.init(ready.index(i), count=1)
        nbytes: gl.constexpr = w_desc.block_type.nbytes
        for i in gl.static_range(STAGES):
            pred = i < K_PER
            mbarrier.expect(ready.index(i), nbytes, pred=pred)
            tma.async_load(w_desc, [n0, (k_start + i) * BK], ready.index(i), w_bufs.index(i), pred=pred)

        acc = gl.zeros([BN, BK], gl.float32, layout=lay)
        offs_k_base = gl.arange(0, BK, layout=k_lay)
        for i in range(K_PER):
            s = i % STAGES
            mbarrier.wait(ready.index(s), (i // STAGES) & 1)
            w_val = w_bufs.index(s).load(lay).to(gl.float32)
            x_val = gl.load(x_ptr + (k_start + i) * BK + offs_k_base, cache_modifier=".ca").to(gl.float32)
            gl.barrier()
            j = i + STAGES
            pred = j < K_PER
            mbarrier.expect(ready.index(s), nbytes, pred=pred)
            tma.async_load(w_desc, [n0, (k_start + j) * BK], ready.index(s), w_bufs.index(s), pred=pred)
            acc += w_val * x_val[None, :]

        r = gl.sum(acc, axis=1)
        offs_nl = gl.arange(0, BN, layout=n_lay)
        if SPLIT == 1:
            gl.store(out_ptr + n0 + offs_nl, r.to(gl.bfloat16))
        else:
            ws_base = ws_ptr + pid_n * (SPLIT * BN) + offs_nl
            gl.store(ws_base + pid_s * BN, r)
            gl.barrier()
            done = gl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
            if done == SPLIT - 1:
                if SPLIT == 2:
                    total = r + gl.load(ws_base + (1 - pid_s) * BN, cache_modifier=".cg")
                else:
                    total = gl.zeros([BN], gl.float32, layout=n_lay)
                    for s in gl.static_range(SPLIT):
                        total += gl.load(ws_base + s * BN, cache_modifier=".cg")
                gl.store(out_ptr + n0 + offs_nl, total.to(gl.bfloat16))
                gl.atomic_xchg(cnt_ptr + pid_n, 0)

    @gluon.jit
    def _kfold_issue(w_desc, x_desc, w_bufs, x_bufs, bar, s, k_idx, n0, pred,
                     R: gl.constexpr, BT: gl.constexpr, BK: gl.constexpr, Q: gl.constexpr):
        NBYTES: gl.constexpr = Q * (w_desc.block_type.nbytes + x_desc.block_type.nbytes)
        mbarrier.expect(bar, NBYTES, pred=pred)
        for q in gl.static_range(Q):
            k_off = (Q * k_idx + q) * BK
            tma.async_load(w_desc, [n0, k_off], bar, w_bufs.index(s).slice(q * R, R, dim=0), pred=pred)
            tma.async_load(x_desc, [0, k_off], bar, x_bufs.index(s).slice(q * BT, BT, dim=0), pred=pred)

    @gluon.jit
    def _kfold_issue_unpred(w_desc, x_desc, w_bufs, x_bufs, bar, s, k_idx, n0,
                            R: gl.constexpr, BT: gl.constexpr, BK: gl.constexpr, Q: gl.constexpr):
        NBYTES: gl.constexpr = Q * (w_desc.block_type.nbytes + x_desc.block_type.nbytes)
        mbarrier.expect(bar, NBYTES)
        for q in gl.static_range(Q):
            k_off = (Q * k_idx + q) * BK
            tma.async_load(w_desc, [n0, k_off], bar, w_bufs.index(s).slice(q * R, R, dim=0))
            tma.async_load(x_desc, [0, k_off], bar, x_bufs.index(s).slice(q * BT, BT, dim=0))

    @gluon.jit
    def _gluon_gemv_kfold_r3(w_desc, x_desc, out_ptr, ws_ptr, cnt_ptr, M,
                             N: gl.constexpr, K: gl.constexpr, STAGES: gl.constexpr,
                             SPLIT: gl.constexpr, Q: gl.constexpr, UNROLL: gl.constexpr,
                             num_warps: gl.constexpr):
        R: gl.constexpr = w_desc.block_type.shape[0]
        BK: gl.constexpr = w_desc.block_type.shape[1]
        BT: gl.constexpr = x_desc.block_type.shape[0]
        BN: gl.constexpr = R * Q
        KI: gl.constexpr = K // (BK * Q)
        EVEN: gl.constexpr = (KI % SPLIT) == 0
        K_PER: gl.constexpr = KI // SPLIT
        FULL: gl.constexpr = (K_PER // STAGES) * STAGES
        REM: gl.constexpr = K_PER - FULL
        mma: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[num_warps, 1],
                                                      instr_shape=[16, Q * BT, 16])
        pid_n = gl.program_id(0)
        pid_s = gl.program_id(1)
        n0 = pid_n * R
        k_start = 0 if SPLIT == 1 else pid_s * K_PER
        k_steps = KI if SPLIT == 1 else K_PER

        w_bufs = gl.allocate_shared_memory(w_desc.dtype, [STAGES, BN, BK], w_desc.layout)
        x_bufs = gl.allocate_shared_memory(x_desc.dtype, [STAGES, Q * BT, BK], x_desc.layout)
        ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
        for i in gl.static_range(STAGES):
            mbarrier.init(ready.index(i), count=1)
        for i in gl.static_range(STAGES - 1):
            _kfold_issue_unpred(w_desc, x_desc, w_bufs, x_bufs, ready.index(i), i, k_start + i, n0, R, BT, BK, Q)

        acc = gl.zeros([BN, Q * BT], gl.float32, layout=mma)
        if UNROLL == 2 and EVEN:
            for i in gl.static_range(K_PER):
                s: gl.constexpr = i % STAGES
                ph: gl.constexpr = (i // STAGES) & 1
                mbarrier.wait(ready.index(s), ph)
                acc = warpgroup_mma(w_bufs.index(s), x_bufs.index(s).permute((1, 0)), acc, is_async=True)
                acc = warpgroup_mma_wait(1, deps=(acc,))
                j: gl.constexpr = i + STAGES - 1
                sj: gl.constexpr = j % STAGES
                if j < K_PER:
                    _kfold_issue_unpred(w_desc, x_desc, w_bufs, x_bufs, ready.index(sj), sj, k_start + j, n0,
                                        R, BT, BK, Q)
        elif UNROLL == 1 and EVEN:
            ph = 0
            for base in range(0, FULL, STAGES):
                for s in gl.static_range(STAGES):
                    i = base + s
                    mbarrier.wait(ready.index(s), ph)
                    acc = warpgroup_mma(w_bufs.index(s), x_bufs.index(s).permute((1, 0)), acc, is_async=True)
                    acc = warpgroup_mma_wait(1, deps=(acc,))
                    j = i + STAGES - 1
                    sj: gl.constexpr = (s + STAGES - 1) % STAGES
                    _kfold_issue(w_desc, x_desc, w_bufs, x_bufs, ready.index(sj), sj, k_start + j, n0,
                                 j < K_PER, R, BT, BK, Q)
                ph = ph ^ 1
            for s in gl.static_range(REM):
                mbarrier.wait(ready.index(s), ph)
                acc = warpgroup_mma(w_bufs.index(s), x_bufs.index(s).permute((1, 0)), acc, is_async=True)
                acc = warpgroup_mma_wait(1, deps=(acc,))
                j: gl.constexpr = FULL + s + STAGES - 1
                sj: gl.constexpr = (s + STAGES - 1) % STAGES
                if j < K_PER:
                    _kfold_issue_unpred(w_desc, x_desc, w_bufs, x_bufs, ready.index(sj), sj, k_start + j, n0,
                                        R, BT, BK, Q)
        else:
            for i in range(k_steps):
                s = i % STAGES
                mbarrier.wait(ready.index(s), (i // STAGES) & 1)
                acc = warpgroup_mma(w_bufs.index(s), x_bufs.index(s).permute((1, 0)), acc, is_async=True)
                acc = warpgroup_mma_wait(1, deps=(acc,))
                j = i + STAGES - 1
                sj = j % STAGES
                _kfold_issue(w_desc, x_desc, w_bufs, x_bufs, ready.index(sj), sj, k_start + j, n0,
                             j < k_steps, R, BT, BK, Q)
        acc = warpgroup_mma_wait(0, deps=(acc,))

        TW: gl.constexpr = 32 if BT >= 32 else BT
        epi: gl.constexpr = gl.BlockedLayout([1, BT // TW], [32 // TW, TW], [num_warps, 1], [1, 0])
        gl.barrier()
        red = w_bufs._reinterpret(gl.float32, [BN, Q * BT], gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0]))
        red.store(acc)
        gl.barrier()
        tot = gl.zeros([R, BT], gl.float32, layout=epi)
        for q in gl.static_range(Q):
            tot += red.slice(q * R, R, dim=0).slice(q * BT, BT, dim=1).load(epi)

        offs_rl = gl.arange(0, R, layout=gl.SliceLayout(1, epi))
        offs_t = gl.arange(0, BT, layout=gl.SliceLayout(0, epi))
        out_ptrs = out_ptr + offs_t[None, :] * N + (n0 + offs_rl)[:, None]
        mask = (offs_t < M)[None, :]
        if SPLIT == 1:
            gl.store(out_ptrs, tot.to(gl.bfloat16), mask=mask)
        else:
            ws_base = ws_ptr + pid_n * (SPLIT * R * BT) + offs_rl[:, None] * BT + offs_t[None, :]
            gl.store(ws_base + pid_s * (R * BT), tot)
            gl.barrier()
            done = gl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
            if done == SPLIT - 1:
                if SPLIT == 2:
                    total = tot + gl.load(ws_base + (1 - pid_s) * (R * BT), cache_modifier=".cg")
                else:
                    total = gl.zeros([R, BT], gl.float32, layout=epi)
                    for s in gl.static_range(SPLIT):
                        total += gl.load(ws_base + s * (R * BT), cache_modifier=".cg")
                gl.store(out_ptrs, total.to(gl.bfloat16), mask=mask)
                gl.atomic_xchg(cnt_ptr + pid_n, 0)

    @gluon.jit
    def _gluon_gemv_split_r3(w_desc, x_desc, out_ptr, ws_ptr, cnt_ptr, M,
                             N: gl.constexpr, K: gl.constexpr, SPLIT: gl.constexpr,
                             STAGES: gl.constexpr, UNROLL: gl.constexpr, num_warps: gl.constexpr):
        BN: gl.constexpr = w_desc.block_type.shape[0]
        BK: gl.constexpr = w_desc.block_type.shape[1]
        BT: gl.constexpr = x_desc.block_type.shape[0]
        KI: gl.constexpr = K // BK
        EVEN: gl.constexpr = (KI % SPLIT) == 0
        K_PER: gl.constexpr = KI // SPLIT
        FULL: gl.constexpr = (K_PER // STAGES) * STAGES
        REM: gl.constexpr = K_PER - FULL
        mma: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[num_warps, 1],
                                                      instr_shape=[16, BT, 16])
        pid_n = gl.program_id(0)
        pid_s = gl.program_id(1)
        n0 = pid_n * BN
        k_start = 0 if SPLIT == 1 else pid_s * K_PER
        k_steps = KI if SPLIT == 1 else K_PER

        w_bufs = gl.allocate_shared_memory(w_desc.dtype, [STAGES] + w_desc.block_type.shape, w_desc.layout)
        x_bufs = gl.allocate_shared_memory(x_desc.dtype, [STAGES] + x_desc.block_type.shape, x_desc.layout)
        ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
        for i in gl.static_range(STAGES):
            mbarrier.init(ready.index(i), count=1)
        nbytes: gl.constexpr = w_desc.block_type.nbytes + x_desc.block_type.nbytes
        for i in gl.static_range(STAGES - 1):
            mbarrier.expect(ready.index(i), nbytes)
            tma.async_load(w_desc, [n0, (k_start + i) * BK], ready.index(i), w_bufs.index(i))
            tma.async_load(x_desc, [0, (k_start + i) * BK], ready.index(i), x_bufs.index(i))

        acc = gl.zeros([BN, BT], gl.float32, layout=mma)
        if UNROLL == 2 and EVEN:
            for i in gl.static_range(K_PER):
                s: gl.constexpr = i % STAGES
                ph: gl.constexpr = (i // STAGES) & 1
                mbarrier.wait(ready.index(s), ph)
                acc = warpgroup_mma(w_bufs.index(s), x_bufs.index(s).permute((1, 0)), acc, is_async=True)
                acc = warpgroup_mma_wait(1, deps=(acc,))
                j: gl.constexpr = i + STAGES - 1
                sj: gl.constexpr = j % STAGES
                if j < K_PER:
                    mbarrier.expect(ready.index(sj), nbytes)
                    tma.async_load(w_desc, [n0, (k_start + j) * BK], ready.index(sj), w_bufs.index(sj))
                    tma.async_load(x_desc, [0, (k_start + j) * BK], ready.index(sj), x_bufs.index(sj))
        elif UNROLL == 1 and EVEN:
            ph = 0
            for base in range(0, FULL, STAGES):
                for s in gl.static_range(STAGES):
                    i = base + s
                    mbarrier.wait(ready.index(s), ph)
                    acc = warpgroup_mma(w_bufs.index(s), x_bufs.index(s).permute((1, 0)), acc, is_async=True)
                    acc = warpgroup_mma_wait(1, deps=(acc,))
                    j = i + STAGES - 1
                    sj: gl.constexpr = (s + STAGES - 1) % STAGES
                    pred = j < K_PER
                    mbarrier.expect(ready.index(sj), nbytes, pred=pred)
                    tma.async_load(w_desc, [n0, (k_start + j) * BK], ready.index(sj), w_bufs.index(sj), pred=pred)
                    tma.async_load(x_desc, [0, (k_start + j) * BK], ready.index(sj), x_bufs.index(sj), pred=pred)
                ph = ph ^ 1
            for s in gl.static_range(REM):
                mbarrier.wait(ready.index(s), ph)
                acc = warpgroup_mma(w_bufs.index(s), x_bufs.index(s).permute((1, 0)), acc, is_async=True)
                acc = warpgroup_mma_wait(1, deps=(acc,))
                j: gl.constexpr = FULL + s + STAGES - 1
                sj: gl.constexpr = (s + STAGES - 1) % STAGES
                if j < K_PER:
                    mbarrier.expect(ready.index(sj), nbytes)
                    tma.async_load(w_desc, [n0, (k_start + j) * BK], ready.index(sj), w_bufs.index(sj))
                    tma.async_load(x_desc, [0, (k_start + j) * BK], ready.index(sj), x_bufs.index(sj))
        else:
            for i in range(k_steps):
                s = i % STAGES
                mbarrier.wait(ready.index(s), (i // STAGES) & 1)
                acc = warpgroup_mma(w_bufs.index(s), x_bufs.index(s).permute((1, 0)), acc, is_async=True)
                acc = warpgroup_mma_wait(1, deps=(acc,))
                j = i + STAGES - 1
                sj = j % STAGES
                pred = j < k_steps
                mbarrier.expect(ready.index(sj), nbytes, pred=pred)
                tma.async_load(w_desc, [n0, (k_start + j) * BK], ready.index(sj), w_bufs.index(sj), pred=pred)
                tma.async_load(x_desc, [0, (k_start + j) * BK], ready.index(sj), x_bufs.index(sj), pred=pred)
        acc = warpgroup_mma_wait(0, deps=(acc,))

        offs_nl = gl.arange(0, BN, layout=gl.SliceLayout(1, mma))
        offs_t = gl.arange(0, BT, layout=gl.SliceLayout(0, mma))
        out_ptrs = out_ptr + offs_t[None, :] * N + (n0 + offs_nl)[:, None]
        mask = (offs_t < M)[None, :]
        if SPLIT == 1:
            gl.store(out_ptrs, acc.to(gl.bfloat16), mask=mask)
        else:
            ws_base = ws_ptr + pid_n * (SPLIT * BN * BT) + offs_nl[:, None] * BT + offs_t[None, :]
            gl.store(ws_base + pid_s * (BN * BT), acc)
            gl.barrier()
            done = gl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
            if done == SPLIT - 1:
                if SPLIT == 2:
                    total = acc + gl.load(ws_base + (1 - pid_s) * (BN * BT), cache_modifier=".cg")
                else:
                    total = gl.zeros([BN, BT], gl.float32, layout=mma)
                    for s in gl.static_range(SPLIT):
                        total += gl.load(ws_base + s * (BN * BT), cache_modifier=".cg")
                gl.store(out_ptrs, total.to(gl.bfloat16), mask=mask)
                gl.atomic_xchg(cnt_ptr + pid_n, 0)


# ==============================================================================
# Candidate specification & generator
# ==============================================================================

class CandR3(NamedTuple):
    kind: str       # "m1_fma", "m1_tma", "m1_gl_fma", "kfold", "split"
    block_n: int
    block_k: int
    block_t: int
    stages: int
    split: int = 1
    warps: int = 4
    w_mod: int = 0   # 0: evict_first, 1: .cg, 2: .cs
    x_mod: int = 0   # 0: evict_last, 1: .ca
    q: int = 1
    unroll: int = 0

    @property
    def name(self) -> str:
        if self.kind == "m1_fma":
            wm = ("ev1", "cg", "cs")[self.w_mod]
            xm = ("evL", "ca")[self.x_mod]
            return f"m1_fma_BN{self.block_n}_BK{self.block_k}_S{self.split}_w{self.warps}_st{self.stages}_{wm}_{xm}"
        if self.kind == "m1_tma":
            return f"m1_tma_BN{self.block_n}_BK{self.block_k}_S{self.split}_w{self.warps}_st{self.stages}"
        if self.kind == "m1_gl_fma":
            return f"m1_gl_fma_BN{self.block_n}_BK{self.block_k}_S{self.split}_w{self.warps}_st{self.stages}"
        if self.kind == "kfold":
            return (f"kfold_R{self.block_n}Q{self.q}_BK{self.block_k}_BT{self.block_t}"
                    f"_ST{self.stages}_S{self.split}_U{self.unroll}_w{self.warps}")
        return (f"split_BN{self.block_n}_BK{self.block_k}_BT{self.block_t}"
                f"_ST{self.stages}_S{self.split}_U{self.unroll}_w{self.warps}")

    def smem_bytes(self) -> int:
        if self.kind in ("m1_fma", "m1_tma"):
            return self.stages * (self.block_n * self.block_k + self.block_k) * 2
        if self.kind == "m1_gl_fma":
            return self.stages * self.block_n * self.block_k * 2
        bn = self.block_n * self.q
        qbt = self.block_t * self.q
        w_bytes = self.stages * bn * self.block_k * 2
        x_bytes = self.stages * qbt * self.block_k * 2
        red_bytes = bn * qbt * 4 if self.kind == "kfold" else 0
        return max(w_bytes, red_bytes) + x_bytes

    def fits(self, n: int, k: int, m: int) -> bool:
        if n % self.block_n != 0 or k % (self.block_k * self.q * self.split) != 0:
            return False
        if self.smem_bytes() > GLUON_SMEM:
            return False
        if self.kind in ("m1_fma", "m1_tma", "m1_gl_fma"):
            if m != 1:
                return False
            if self.kind == "m1_tma" and (TensorDescriptor is None or self.block_k > 256):
                return False
            if self.kind == "m1_gl_fma" and (not HAVE_GLUON or self.block_k < 256 or (k // self.block_k // self.split) < self.stages):
                return False
            return True
        if not HAVE_GLUON or self.block_t < m:
            return False
        ki = k // (self.block_k * self.q)
        if ki // self.split < self.stages:
            return False
        if self.kind == "kfold":
            if self.block_n * self.q != 16 * self.warps:
                return False
            if (self.block_n * self.q) * (self.q * self.block_t) * 4 > self.stages * (self.block_n * self.q) * self.block_k * 2:
                return False
        if self.unroll == 2 and (ki % self.split != 0 or (ki // self.split) > 32):
            return False
        return True


def candidates_for(layer: str, m: int, n: int, k: int) -> list[CandR3]:
    raw: list[CandR3] = []

    if m == 1:
        if layer == "o":
            # o (N=8192, K=1024): test single-step BLOCK_K=1024 (no loop!) + BLOCK_K=256, 512
            for bn in (4, 8, 16, 32):
                for warps in (4, 8):
                    for w_mod, x_mod in ((0, 0), (1, 1), (2, 1)):
                        raw.append(CandR3("m1_fma", bn, 1024, 1, 1, split=1, warps=warps, w_mod=w_mod, x_mod=x_mod))
            for bn in (8, 16, 32):
                for bk in (256, 512):
                    for warps in (4, 8):
                        for st in (1, 2, 3):
                            for w_mod, x_mod in ((0, 0), (1, 1), (2, 1)):
                                raw.append(CandR3("m1_fma", bn, bk, 1, st, split=1, warps=warps, w_mod=w_mod, x_mod=x_mod))
            # M2: TMA and Gluon WGMMA at m=1
            for bn in (16, 32):
                raw.append(CandR3("m1_tma", bn, 256, 1, 3, split=1, warps=4))
                raw.append(CandR3("m1_gl_fma", bn, 256, 1, 2, split=1, warps=4))
            raw.append(CandR3("split", 64, 256, 8, 4, split=1, unroll=2, warps=4))
        else:
            # qkv_gate (N=2304, K=8192) & pooler (N=4096, K=8192)
            for bn in (2, 4, 8, 16):
                for bk in (512, 1024, 2048):
                    for warps in (4, 8):
                        if bn * bk // (warps * 32) > 64:
                            continue  # avoid register spills from oversized [BN, BK] accumulator
                        for st in (1, 2, 3, 4):
                            for w_mod, x_mod in ((0, 0), (1, 1), (2, 1)):
                                raw.append(CandR3("m1_fma", bn, bk, 1, st, split=1, warps=warps, w_mod=w_mod, x_mod=x_mod))
            # SPLIT=2 with 1-load commutative reduction
            for bn in (8, 16, 32):
                for bk in (512, 1024):
                    for warps in (4, 8):
                        if bn * bk // (warps * 32) > 64:
                            continue
                        for st in (1, 2, 3):
                            for w_mod, x_mod in ((0, 0), (1, 1)):
                                raw.append(CandR3("m1_fma", bn, bk, 1, st, split=2, warps=warps, w_mod=w_mod, x_mod=x_mod))
            # M2a/b/c: TMA descriptor, Gluon TMA+FMA, and Gluon K-fold/WGMMA at m=1
            for bn in (16, 32):
                raw.append(CandR3("m1_tma", bn, 256, 1, 3, split=1, warps=4))
                raw.append(CandR3("m1_tma", bn, 256, 1, 3, split=2, warps=4))
                raw.append(CandR3("m1_gl_fma", bn, 256, 1, 4, split=1, warps=4))
                raw.append(CandR3("m1_gl_fma", bn, 256, 1, 4, split=2, warps=4))
            raw.append(CandR3("kfold", 32, 256, 8, 4, split=1, q=2, unroll=2, warps=4))
            raw.append(CandR3("kfold", 16, 256, 8, 4, split=1, q=4, unroll=2, warps=4))
            raw.append(CandR3("split", 64, 256, 8, 4, split=2, unroll=2, warps=4))
    else:
        # Architecture M3: Small-m (m in {8, 16, 32}) on pooler and unpooler
        bt = 8 if m <= 8 else (16 if m <= 16 else 32)
        for u in (0, 1, 2):
            # BN=64, BK=256, ST=4 vs BN=64, BK=128, ST=8
            raw.append(CandR3("split", 64, 256, bt, 4, split=1, unroll=u, warps=4))
            raw.append(CandR3("split", 64, 128, bt, 8, split=1, unroll=u, warps=4))
            # BN=128 (2 warpgroups per CTA sharing x in smem across 128 rows of W, or 1 warpgroup)
            raw.append(CandR3("split", 128, 128, bt, 4, split=1, unroll=u, warps=8))
            raw.append(CandR3("split", 128, 128, bt, 4, split=1, unroll=u, warps=4))
            raw.append(CandR3("split", 128, 256, bt, 3, split=1, unroll=u, warps=8))
            if layer == "pooler":
                # Pooler has N=4096 (only 32 CTAs at BN=128, 64 CTAs at BN=64): test SPLIT=2 and kfold R=32, Q=2 (128 CTAs)
                raw.append(CandR3("split", 128, 128, bt, 4, split=2, unroll=u, warps=8))
                raw.append(CandR3("split", 64, 256, bt, 4, split=2, unroll=u, warps=4))
                raw.append(CandR3("kfold", 32, 256, bt, 4, split=1, q=2, unroll=u, warps=4))
                raw.append(CandR3("kfold", 32, 256, bt, 3, split=1, q=2, unroll=u, warps=4))
                raw.append(CandR3("kfold", 32, 128, bt, 8, split=1, q=2, unroll=u, warps=4))
                raw.append(CandR3("kfold", 32, 128, bt, 6, split=1, q=2, unroll=u, warps=4))

    seen = set()
    out = []
    for c in raw:
        if c not in seen and c.fits(n, k, m):
            seen.add(c)
            out.append(c)
    return out


# ==============================================================================
# CPU compile report (--compile)
# ==============================================================================

def compile_report(only: str) -> None:
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import get_ptxas
    from triton.compiler import ASTSource
    from triton.experimental.gluon._runtime import GluonASTSource

    compile_shapes = [
        ("qkv_gate", 1, 2304, 8192),
        ("o", 1, 8192, 1024),
        ("pooler", 16, 4096, 8192),
        ("pooler", 32, 4096, 8192),
        ("unpooler", 16, 8192, 4096),
    ]
    seen_configs = set()
    print(f"{'CANDIDATE':64s} | {'SMEM':>7s} | {'REGS':>4s} | {'SPILL':>5s} | {'LDG128':>6s} | {'TIME':>5s}", flush=True)
    print("-" * 104, flush=True)
    for label, m, n, k in compile_shapes:
        for c in candidates_for(label, m, n, k):
            cfg_key = (c, n, k)
            if cfg_key in seen_configs:
                continue
            seen_configs.add(cfg_key)
            tag = f"{label}_m{m}_{c.name}"
            if only and only not in tag:
                continue
            t0 = time.time()
            try:
                if c.kind == "m1_fma":
                    fn = _bf16_gemv_v2
                    sig = {"x_ptr": "*bf16", "w_ptr": "*bf16", "out_ptr": "*bf16", "ws_ptr": "*fp32", "cnt_ptr": "*i32"}
                    consts = dict(N=n, K=k, SPLIT=c.split, BLOCK_N=c.block_n, BLOCK_K=c.block_k, W_MOD=c.w_mod, X_MOD=c.x_mod)
                    names = list(fn.arg_names)
                    full = {nm: sig.get(nm, "constexpr") for nm in names}
                    attrs = {(names.index(nm),): [["tt.divisibility", 16]] for nm in sig}
                    src = ASTSource(fn=fn, signature=full, constexprs=consts, attrs=attrs)
                    ker = triton.compile(src, target=GPUTarget("cuda", 90, 32),
                                         options={"num_warps": c.warps, "num_stages": c.stages})
                elif c.kind == "m1_tma":
                    continue  # TensorDescriptor in triton.tools requires CUDA tensor on some builds
                else:
                    w = torch.zeros(n, k, dtype=torch.bfloat16)
                    x = torch.zeros(max(m, c.block_t), k, dtype=torch.bfloat16)
                    wl = gl.NVMMASharedLayout.get_default_for([c.block_n, c.block_k], gl.bfloat16)
                    w_desc = GluonDesc.from_tensor(w, [c.block_n, c.block_k], wl)
                    if c.kind == "m1_gl_fma":
                        fn = _gluon_m1_fma
                        sig = {"w_desc": w_desc.__mangle__(), "x_ptr": "*bf16", "out_ptr": "*bf16",
                               "ws_ptr": "*fp32", "cnt_ptr": "*i32"}
                        consts = dict(N=n, K=k, SPLIT=c.split, STAGES=c.stages, num_warps=c.warps)
                    elif c.kind == "kfold":
                        xl = gl.NVMMASharedLayout.get_default_for([c.block_t, c.block_k], gl.bfloat16)
                        x_desc = GluonDesc.from_tensor(x, [c.block_t, c.block_k], xl)
                        fn = _gluon_gemv_kfold_r3
                        sig = {"w_desc": w_desc.__mangle__(), "x_desc": x_desc.__mangle__(), "out_ptr": "*bf16",
                               "ws_ptr": "*fp32", "cnt_ptr": "*i32", "M": "i32"}
                        consts = dict(N=n, K=k, STAGES=c.stages, SPLIT=c.split, Q=c.q, UNROLL=c.unroll, num_warps=c.warps)
                    else:
                        xl = gl.NVMMASharedLayout.get_default_for([c.block_t, c.block_k], gl.bfloat16)
                        x_desc = GluonDesc.from_tensor(x, [c.block_t, c.block_k], xl)
                        fn = _gluon_gemv_split_r3
                        sig = {"w_desc": w_desc.__mangle__(), "x_desc": x_desc.__mangle__(), "out_ptr": "*bf16",
                               "ws_ptr": "*fp32", "cnt_ptr": "*i32", "M": "i32"}
                        consts = dict(N=n, K=k, SPLIT=c.split, STAGES=c.stages, UNROLL=c.unroll, num_warps=c.warps)
                    names = list(fn.arg_names)
                    full = {nm: sig.get(nm, "constexpr") for nm in names}
                    attrs = {(names.index(nm),): [["tt.divisibility", 16]] for nm in sig if sig[nm].startswith("*")}
                    src = GluonASTSource(
                        fn=fn,
                        signature=full,
                        constexprs={(names.index(nm),): v for nm, v in consts.items()},
                        attrs=attrs,
                    )
                    ker = triton.compile(src, target=GPUTarget("cuda", 90, 32), options={"num_warps": c.warps})
            except Exception as exc:
                print(f"{tag:64s} | COMPILE FAILED: {type(exc).__name__}: {str(exc)[:180]}", flush=True)
                continue
            ptx = ker.asm["ptx"]
            ldg128_cnt = len(re.findall(r"ld\.global(?:\.[a-z0-9]+)*\.v4\.b32", ptx))
            regs, spills = "?", "0"
            with tempfile.TemporaryDirectory() as td:
                path = f"{td}/k.ptx"
                open(path, "w").write(ptx)
                res = subprocess.run(
                    [get_ptxas(90).path, "-arch=sm_90a", "-v", path, "-o", f"{td}/k.o"],
                    capture_output=True,
                    text=True,
                )
                out_txt = res.stdout + res.stderr
                m_reg = re.search(r"Used\s+(\d+)\s+registers", out_txt)
                m_spl = re.search(r"(\d+)\s+bytes\s+spill\s+stores", out_txt)
                if m_reg:
                    regs = m_reg.group(1)
                if m_spl:
                    spills = m_spl.group(1)
            print(f"{tag:64s} | {ker.metadata.shared // 1024:5d}KB | {regs:>4s} | {spills:>5s} | {ldg128_cnt:6d} | {time.time() - t0:4.1f}s",
                  flush=True)


# ==============================================================================
# GPU Benchmark Runner
# ==============================================================================

class RunnerStateR3:
    def __init__(self, weights: dict[str, torch.Tensor]) -> None:
        self.w = {k: v.contiguous() for k, v in weights.items()}
        self.ws = torch.empty(WS_FLOATS, device="cuda", dtype=torch.float32)
        self.cnt = torch.zeros(COUNTERS, device="cuda", dtype=torch.int32)
        self.tma_descs: dict[tuple[str, int, int], object] = {}
        self.gl_w_descs: dict[tuple[str, int, int], object] = {}
        self.gl_x_layouts: dict[tuple[int, int], object] = {}

    def tma_desc(self, key: str, bn: int, bk: int):
        dk = (key, bn, bk)
        if dk not in self.tma_descs:
            self.tma_descs[dk] = TensorDescriptor.from_tensor(self.w[key], [bn, bk])
        return self.tma_descs[dk]

    def gl_w_desc(self, key: str, bn: int, bk: int):
        dk = (key, bn, bk)
        if dk not in self.gl_w_descs:
            wl = gl.NVMMASharedLayout.get_default_for([bn, bk], gl.bfloat16)
            self.gl_w_descs[dk] = GluonDesc.from_tensor(self.w[key], [bn, bk], wl)
        return self.gl_w_descs[dk]

    def gl_x_desc(self, x: torch.Tensor, bt: int, bk: int):
        lk = (bt, bk)
        if lk not in self.gl_x_layouts:
            self.gl_x_layouts[lk] = gl.NVMMASharedLayout.get_default_for([bt, bk], gl.bfloat16)
        return GluonDesc.from_tensor(x, [bt, bk], self.gl_x_layouts[lk])

    def launch(self, c: CandR3, key: str, x: torch.Tensor) -> torch.Tensor:
        if x.stride(1) != 1:
            x = x.contiguous()
        m = x.shape[0]
        w = self.w[key]
        n, k = w.shape
        out = torch.empty(m, n, device=x.device, dtype=torch.bfloat16)
        grid = (n // c.block_n, c.split)
        if c.kind == "m1_fma":
            _bf16_gemv_v2[grid](
                x, w, out, self.ws, self.cnt,
                N=n, K=k, SPLIT=c.split, BLOCK_N=c.block_n, BLOCK_K=c.block_k,
                W_MOD=c.w_mod, X_MOD=c.x_mod, num_warps=c.warps, num_stages=c.stages,
            )
        elif c.kind == "m1_tma":
            _triton_tma_gemv[grid](
                x, self.tma_desc(key, c.block_n, c.block_k), out, self.ws, self.cnt,
                N=n, K=k, SPLIT=c.split, BLOCK_N=c.block_n, BLOCK_K=c.block_k,
                num_warps=c.warps, num_stages=c.stages,
            )
        elif c.kind == "m1_gl_fma":
            _gluon_m1_fma[grid](
                self.gl_w_desc(key, c.block_n, c.block_k), x, out, self.ws, self.cnt,
                N=n, K=k, SPLIT=c.split, STAGES=c.stages, num_warps=c.warps,
            )
        elif c.kind == "kfold":
            _gluon_gemv_kfold_r3[grid](
                self.gl_w_desc(key, c.block_n, c.block_k), self.gl_x_desc(x, c.block_t, c.block_k),
                out, self.ws, self.cnt, m,
                N=n, K=k, STAGES=c.stages, SPLIT=c.split, Q=c.q, UNROLL=c.unroll, num_warps=c.warps,
            )
        else:
            _gluon_gemv_split_r3[grid](
                self.gl_w_desc(key, c.block_n, c.block_k), self.gl_x_desc(x, c.block_t, c.block_k),
                out, self.ws, self.cnt, m,
                N=n, K=k, SPLIT=c.split, STAGES=c.stages, UNROLL=c.unroll, num_warps=c.warps,
            )
        return out


def graph_of(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    g.replay()
    torch.cuda.synchronize()
    return g, out


def paired(g_base, g_cand, flush, reps: int, time_replay):
    d, tb, tc = [], [], []
    for r in range(reps):
        order = (g_base, g_cand) if r % 2 == 0 else (g_cand, g_base)
        t = {}
        for g in order:
            t[id(g)] = time_replay(g, flush) - OVERHEAD_US
        tb.append(t[id(g_base)])
        tc.append(t[id(g_cand)])
        d.append(t[id(g_cand)] - t[id(g_base)])
    return statistics.median(tb), statistics.median(tc), statistics.median(d)


def check_cand_correctness(reference, rs: RunnerStateR3, c: CandR3, layer: str, key: str, m: int, w: dict,
                           include_nearby: bool = False) -> tuple[bool, str]:
    ms = NEARBY_M.get(m, (m,)) if include_nearby else (m,)
    for m_test in ms:
        if not c.fits(rs.w[key].shape[0], rs.w[key].shape[1], m_test):
            return False, f"does not fit m={m_test}"
        for dist, seed in (("normal", 1), ("outliers", 2)):
            data = reference.generate_input(m_test, layer, seed, dist=dist)
            out = rs.launch(c, key, data["x"])
            torch.cuda.synchronize()
            ok, msg = reference.check_implementation(data, out, w)
            if not ok:
                return False, f"m={m_test} {dist}: {msg}"
    return True, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile", action="store_true", help="CPU-only Triton/Gluon compile + ptxas report")
    ap.add_argument("--only", default="", help="Substring filter on candidate/shape tags")
    ap.add_argument("--shapes", default="qkv_gate:1,pooler:1,o:1,unpooler:16,unpooler:32,pooler:8,pooler:16,pooler:32")
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--cold", action="store_true")
    args = ap.parse_args()

    if args.compile:
        compile_report(args.only)
        return

    from common import load_module, reference, weights
    from harness import FLUSH_BYTES, time_replay

    t0 = time.time()
    w = weights()
    cp3 = load_module(BASE_CP, "cp3")
    cst = cp3.prepare({k: v.clone() for k, v in w.items()})
    rs = RunnerStateR3(w)
    flush = torch.empty(FLUSH_BYTES // 4, device="cuda", dtype=torch.float32).normal_()
    print(f"cp3 prepare done in {time.time() - t0:.1f}s; running paired cold-L2 CUDA-graph sweeps (reps={args.reps})",
          flush=True)

    shapes = [(s.split(":")[0], int(s.split(":")[1])) for s in args.shapes.split(",") if s.strip()]
    results = {}

    for layer, m in shapes:
        key = layer
        if args.only and args.only not in f"{key}:{m}" and args.only not in layer:
            continue
        n, k = rs.w[key].shape
        sol_us = reference.speed_of_light_ns(m, layer, {"hbm_bytes_per_s": 4.8e12, "bf16_dense_flops_per_s": 989e12}) / 1e3
        data = reference.generate_input(m, layer, 1)
        x_buf = data["x"].clone()
        data_static = {"x": x_buf, "config": data["config"]}

        g_base, out_b = graph_of(lambda: cp3.kernel_fn(data_static, cst))
        ok_b, msg_b = reference.check_implementation(data_static, out_b, w)
        print(f"\n==================== {key} m={m} (N={n}, K={k}, SOL={sol_us:.2f} us) "
              f"cp3={'ok' if ok_b else 'FAIL ' + msg_b} ====================", flush=True)

        cands = candidates_for(layer, m, n, k)
        best = None
        for c in cands:
            ctas = (n // c.block_n) * c.split
            tag = f"{c.name} (ctas={ctas})"
            try:
                ok_c, msg_c = check_cand_correctness(reference, rs, c, layer, key, m, w, include_nearby=False)
                if not ok_c:
                    print(f"  {tag:64s}: WRONG ({msg_c})", flush=True)
                    continue
                g_cand, out_c = graph_of(lambda c=c: rs.launch(c, key, x_buf))
                ok_g, msg_g = reference.check_implementation(data_static, out_c, w)
                if not ok_g:
                    print(f"  {tag:64s}: GRAPH WRONG ({msg_g})", flush=True)
                    continue
                ref_bits = out_c.clone().view(torch.int16)
                det_ok = True
                for _ in range(50):
                    g_cand.replay()
                    if not torch.equal(out_c.view(torch.int16), ref_bits):
                        det_ok = False
                        break
                torch.cuda.synchronize()
                if not det_ok:
                    print(f"  {tag:64s}: NON-DETERMINISTIC across graph replays!", flush=True)
                    continue
                tb, tc, d = paired(g_base, g_cand, flush, args.reps, time_replay)
                bytes_moved = n * k * 2 + m * k * 2 + m * n * 2
                tb_s = bytes_moved / tc / 1e6
                print(f"  {tag:64s}: cp3 {tb:6.2f} | cand {tc:6.2f} us ({tb_s:4.2f} TB/s) | delta {d:+6.2f} us",
                      flush=True)
                if best is None or d < best["delta"]:
                    best = {"cand": c, "tb": tb, "tc": tc, "delta": d, "tag": tag}
            except Exception as exc:
                print(f"  {tag:64s}: FAILED {type(exc).__name__}: {str(exc)[:260]}", flush=True)

        if best is not None:
            ok_near, msg_near = check_cand_correctness(
                reference, rs, best["cand"], layer, key, m, w, include_nearby=True
            )
            best["nearby_ok"] = ok_near
            best["nearby_msg"] = msg_near
            results[(layer, m)] = best
            print(f"  >> BEST {key} m={m}: {best['cand'].name} -> cp3 {best['tb']:.2f} vs cand {best['tc']:.2f} us "
                  f"(delta {best['delta']:+.2f} us) | nearby m {NEARBY_M.get(m, (m,))}: {msg_near}", flush=True)

    print("\n==================== SUMMARY & WHOLE-LOAD SCORE IMPACT vs CP3 ====================", flush=True)
    load_deltas = {ld: 0.0 for ld in JUDGE_LOAD_TOTALS_US}
    for (layer, m), b in results.items():
        ld = M_TO_LOAD.get(m, "")
        gain = min(0.0, b["delta"])
        if ld in load_deltas:
            load_deltas[ld] += gain
        status = "WIN" if b["delta"] < -0.08 and b["nearby_ok"] else "TIE/LOSE"
        print(f"  {layer:10s} m={m:2d} ({ld:9s}): {b['cand'].name:50s} cp3={b['tb']:6.2f} cand={b['tc']:6.2f} "
              f"delta={b['delta']:+6.2f} us  nearby={b['nearby_ok']} [{status}]", flush=True)

    old_ratios = [SGLANG_LOAD_TOTALS_US[ld] / JUDGE_LOAD_TOTALS_US[ld] for ld in JUDGE_LOAD_TOTALS_US]
    new_ratios = [
        SGLANG_LOAD_TOTALS_US[ld] / max(1.0, JUDGE_LOAD_TOTALS_US[ld] + load_deltas[ld])
        for ld in JUDGE_LOAD_TOTALS_US
    ]
    old_geo = math.exp(sum(map(math.log, old_ratios)) / len(old_ratios))
    new_geo = math.exp(sum(map(math.log, new_ratios)) / len(new_ratios))
    for ld in ("decode 1", "verify 1", "decode 16", "decode 32"):
        old_t = JUDGE_LOAD_TOTALS_US[ld]
        new_t = old_t + load_deltas[ld]
        print(f"  Load {ld:10s}: cp3 {old_t:6.2f} us -> projected {new_t:6.2f} us (delta {load_deltas[ld]:+6.2f} us, "
              f"{SGLANG_LOAD_TOTALS_US[ld] / old_t:.3f}x -> {SGLANG_LOAD_TOTALS_US[ld] / new_t:.3f}x)", flush=True)
    print(f"  Projected Geomean Speedup: {old_geo:.4f}x -> {new_geo:.4f}x ({new_geo - old_geo:+.4f}x)", flush=True)


if __name__ == "__main__":
    main()
