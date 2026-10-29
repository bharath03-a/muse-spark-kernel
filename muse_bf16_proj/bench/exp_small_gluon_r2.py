"""Track A Round 2: Unrolled Stage Rings, Zero-Overhead SMEM Reuse (`_reinterpret`), and R=16/32 K-Fold.

Compares Round 2 candidates against `checkpoints/cp3_gluon_v2.py`:
  1. `_gluon_gemv_kfold` with:
     - Zero extra shared memory for `red` via `w_bufs._reinterpret(gl.float32, [BN, Q * BT], ...)` after `gl.barrier()`,
       unlocking `R=32, Q=2, BK=256, STAGES=3` (192 KB) and `BK=128, STAGES=6` (192 KB) at `m=32`, and `R=16, Q=4`
       (144 CTAs on `qkv_gate`!).
     - Stage-unrolled (`unroll=1`) and fully-unrolled (`unroll=2`) TMA rings that replace runtime `i % STAGES` and
       `(i // STAGES) & 1` with compile-time `gl.static_range` indices (eliminating modulo/div PTX overhead for
       non-power-of-2 `STAGES=3, 5, 6` and folding constant smem offsets for `STAGES=4, 8`).
     - Optional `SPLIT >= 1` support on `kfold` (e.g., `R=32, Q=2, SPLIT=2` -> 144 CTAs on `qkv_gate` m=32) with
       in-register self-partial (`total = tot`).
  2. `_gluon_gemv_split` with stage-unrolled / fully-unrolled TMA rings and in-register self-partial (`total = acc`).
  3. `pooler` (`N=4096, K=8192`) at m=8, 16, 32 with `kfold R=32, Q=2, BK=256` (128 CTAs, no global split-K) and
     unrolled `BN=64` rings.

Usage:
    modal run modal_app.py::compile_only --cmd "exp_small_gluon_r2 --compile"
    modal run modal_app.py::micro --name exp_small_gluon_r2 --args "--reps 35"
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

# Estimated cp3 load totals (us) for geomean delta calculation.
CP3_LOAD_TOTALS_US = {
    "decode 1": 41.70,
    "verify 1": 45.37,
    "decode 16": 59.13,
    "decode 32": 62.63,
    "verify 16": 76.76,
    "verify 32": 91.50,
}
SGLANG_LOAD_TOTALS_US = {
    "decode 1": 59.48,
    "verify 1": 74.48,
    "decode 16": 74.66,
    "decode 32": 76.61,
    "verify 16": 86.20,
    "verify 32": 100.31,
}
M_TO_LOAD = {1: "decode 1", 8: "verify 1", 16: "decode 16", 32: "decode 32", 128: "verify 16", 256: "verify 32"}
NEARBY_M = {8: (2, 7, 8), 16: (9, 12, 16), 32: (17, 24, 32)}


class CandR2(NamedTuple):
    kind: str       # "kfold" or "split"
    block_n: int    # real rows per CTA (16, 32, or 64)
    block_k: int    # K per TMA box (128 or 256)
    block_t: int    # token tile size (8, 16, or 32)
    stages: int     # ring buffer depth (3, 4, 5, 6, 8)
    split: int = 1  # global split-K factor (1, 2, 3, 4)
    q: int = 1      # K-fold factor (1 for split, 2 or 4 for kfold)
    unroll: int = 1  # 0: dynamic i%STAGES, 1: stage-unrolled static_range(STAGES), 2: full static_range(K_STEPS)
    warps: int = 4

    @property
    def name(self) -> str:
        u_tag = f"U{self.unroll}"
        if self.kind == "kfold":
            return (f"kfold_R{self.block_n}Q{self.q}_BK{self.block_k}_BT{self.block_t}"
                    f"_ST{self.stages}_S{self.split}_{u_tag}")
        return (f"split_BN{self.block_n}_BK{self.block_k}_BT{self.block_t}"
                f"_ST{self.stages}_S{self.split}_{u_tag}")

    def smem_bytes(self) -> int:
        # With w_bufs._reinterpret for red in kfold, red uses 0 additional shared memory bytes!
        bn = self.block_n * self.q
        qbt = self.block_t * self.q
        w_bytes = self.stages * bn * self.block_k * 2
        x_bytes = self.stages * qbt * self.block_k * 2
        red_bytes = bn * qbt * 4 if self.kind == "kfold" else 0
        return max(w_bytes, red_bytes) + x_bytes

    def fits(self, n: int, k: int, m: int) -> bool:
        if self.block_t < m or n % self.block_n != 0 or k % (self.block_k * self.q) != 0:
            return False
        if self.smem_bytes() > GLUON_SMEM:
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
        if self.split > 1:
            if self.split * n * self.block_t > WS_FLOATS or (n // self.block_n) > COUNTERS:
                return False
        return True


if HAVE_GLUON:
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
    def _gluon_gemv_kfold_r2(w_desc, x_desc, out_ptr, ws_ptr, cnt_ptr, M,
                             N: gl.constexpr, K: gl.constexpr, STAGES: gl.constexpr,
                             SPLIT: gl.constexpr, Q: gl.constexpr, UNROLL: gl.constexpr,
                             num_warps: gl.constexpr):
        """Round 2 K-fold kernel: zero-extra-smem `_reinterpret` epilogue, unrolled stage ring, optional SPLIT."""
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
        if SPLIT == 1:
            k_start = 0
            k_steps = KI
        elif EVEN:
            k_start = pid_s * K_PER
            k_steps = K_PER
        else:
            k_base: gl.constexpr = KI // SPLIT
            k_start = pid_s * k_base
            k_steps = KI - k_start if pid_s == SPLIT - 1 else k_base

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
                s = i % STAGES
                ph = (i // STAGES) & 1
                mbarrier.wait(ready.index(s), ph)
                acc = warpgroup_mma(w_bufs.index(s), x_bufs.index(s).permute((1, 0)), acc, is_async=True)
                acc = warpgroup_mma_wait(1, deps=(acc,))
                j = i + STAGES - 1
                sj = j % STAGES
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
                    sj = (s + STAGES - 1) % STAGES
                    _kfold_issue(w_desc, x_desc, w_bufs, x_bufs, ready.index(sj), sj, k_start + j, n0,
                                 j < K_PER, R, BT, BK, Q)
                ph = ph ^ 1
            for s in gl.static_range(REM):
                mbarrier.wait(ready.index(s), ph)
                acc = warpgroup_mma(w_bufs.index(s), x_bufs.index(s).permute((1, 0)), acc, is_async=True)
                acc = warpgroup_mma_wait(1, deps=(acc,))
                j = FULL + s + STAGES - 1
                sj = (s + STAGES - 1) % STAGES
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
        red = gl.allocate_shared_memory(gl.float32, [BN, Q * BT], gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0]))
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
                    # IEEE-754 addition is commutative (a + b == b + a): bitwise deterministic & saves 1 load
                    total = tot + gl.load(ws_base + (1 - pid_s) * (R * BT), cache_modifier=".cg")
                else:
                    # Canonical s = 0..SPLIT-1 order guarantees bitwise determinism regardless of arrival order
                    total = gl.zeros([R, BT], gl.float32, layout=epi)
                    for s in gl.static_range(SPLIT):
                        total += gl.load(ws_base + s * (R * BT), cache_modifier=".cg")
                gl.store(out_ptrs, total.to(gl.bfloat16), mask=mask)
                gl.atomic_xchg(cnt_ptr + pid_n, 0)

    @gluon.jit
    def _gluon_gemv_split_r2(w_desc, x_desc, out_ptr, ws_ptr, cnt_ptr, M,
                             N: gl.constexpr, K: gl.constexpr, SPLIT: gl.constexpr,
                             STAGES: gl.constexpr, UNROLL: gl.constexpr, num_warps: gl.constexpr):
        """Round 2 Split-K wgmma kernel: unrolled stage ring + deterministic Split-K reduction."""
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
        if SPLIT == 1:
            k_start = 0
            k_steps = KI
        elif EVEN:
            k_start = pid_s * K_PER
            k_steps = K_PER
        else:
            k_base: gl.constexpr = KI // SPLIT
            k_start = pid_s * k_base
            k_steps = KI - k_start if pid_s == SPLIT - 1 else k_base

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
                s = i % STAGES
                ph = (i // STAGES) & 1
                mbarrier.wait(ready.index(s), ph)
                acc = warpgroup_mma(w_bufs.index(s), x_bufs.index(s).permute((1, 0)), acc, is_async=True)
                acc = warpgroup_mma_wait(1, deps=(acc,))
                j = i + STAGES - 1
                sj = j % STAGES
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
                    sj = (s + STAGES - 1) % STAGES
                    pred = j < K_PER
                    mbarrier.expect(ready.index(sj), nbytes, pred=pred)
                    tma.async_load(w_desc, [n0, (k_start + j) * BK], ready.index(sj), w_bufs.index(sj), pred=pred)
                    tma.async_load(x_desc, [0, (k_start + j) * BK], ready.index(sj), x_bufs.index(sj), pred=pred)
                ph = ph ^ 1
            for s in gl.static_range(REM):
                mbarrier.wait(ready.index(s), ph)
                acc = warpgroup_mma(w_bufs.index(s), x_bufs.index(s).permute((1, 0)), acc, is_async=True)
                acc = warpgroup_mma_wait(1, deps=(acc,))
                j = FULL + s + STAGES - 1
                sj = (s + STAGES - 1) % STAGES
                if j < K_PER:
                    mbarrier.expect(ready.index(sj), nbytes)
                    tma.async_load(w_desc, [n0, (k_start + j) * BK], ready.index(sj), w_bufs.index(sj))
                    tma.async_load(x_desc, [0, (k_start + j) * BK], ready.index(sj), x_bufs.index(sj))
        elif UNROLL == 1 and not EVEN:
            K_BASE: gl.constexpr = KI // SPLIT
            FULL_B: gl.constexpr = (K_BASE // STAGES) * STAGES
            ph = 0
            for base in range(0, FULL_B, STAGES):
                for s in gl.static_range(STAGES):
                    i = base + s
                    mbarrier.wait(ready.index(s), ph)
                    acc = warpgroup_mma(w_bufs.index(s), x_bufs.index(s).permute((1, 0)), acc, is_async=True)
                    acc = warpgroup_mma_wait(1, deps=(acc,))
                    j = i + STAGES - 1
                    sj = (s + STAGES - 1) % STAGES
                    pred = j < k_steps
                    mbarrier.expect(ready.index(sj), nbytes, pred=pred)
                    tma.async_load(w_desc, [n0, (k_start + j) * BK], ready.index(sj), w_bufs.index(sj), pred=pred)
                    tma.async_load(x_desc, [0, (k_start + j) * BK], ready.index(sj), x_bufs.index(sj), pred=pred)
                ph = ph ^ 1
            for i in range(FULL_B, k_steps):
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


def candidates_for(layer: str, m: int, n: int, k: int) -> list[CandR2]:
    bt = 8 if m <= 8 else (16 if m <= 16 else 32)
    raw: list[CandR2] = []

    if layer == "qkv_gate":
        # 1. R=32, Q=2 K-fold with 0 extra smem + unroll=0, 2
        for u in (0, 2):
            raw.append(CandR2("kfold", 32, 256, bt, 4, split=1, q=2, unroll=u))
            raw.append(CandR2("kfold", 32, 256, bt, 3, split=1, q=2, unroll=u))
            raw.append(CandR2("kfold", 32, 128, bt, 6, split=1, q=2, unroll=u))
            raw.append(CandR2("kfold", 32, 128, bt, 8, split=1, q=2, unroll=u))
            raw.append(CandR2("kfold", 32, 128, bt, 5, split=1, q=2, unroll=u))
        # 2. R=16, Q=4 K-fold (144 CTAs = 2304 // 16, zero global split-K!)
        for u in (0, 2):
            raw.append(CandR2("kfold", 16, 256, bt, 4, split=1, q=4, unroll=u))
            raw.append(CandR2("kfold", 16, 256, bt, 3, split=1, q=4, unroll=u))
            raw.append(CandR2("kfold", 16, 128, bt, 4, split=1, q=4, unroll=u))
            raw.append(CandR2("kfold", 16, 128, bt, 6, split=1, q=4, unroll=u))
            raw.append(CandR2("kfold", 16, 128, bt, 3, split=1, q=4, unroll=u))
        # 3. Hybrid K-fold R=32, Q=2 + SPLIT=2 (144 CTAs = 72 * 2)
        for u in (0, 2):
            raw.append(CandR2("kfold", 32, 256, bt, 3, split=2, q=2, unroll=u))
            raw.append(CandR2("kfold", 32, 256, bt, 4, split=2, q=2, unroll=u))
            raw.append(CandR2("kfold", 32, 128, bt, 4, split=2, q=2, unroll=u))
        # 4. Split-K BN=64 with self-partial in register + unrolled stage ring
        for split in (2, 3, 4):
            for u in (0, 2):
                raw.append(CandR2("split", 64, 256, bt, 4, split=split, unroll=u))
                raw.append(CandR2("split", 64, 128, bt, 6, split=split, unroll=u))
                raw.append(CandR2("split", 64, 128, bt, 8, split=split, unroll=u))

    elif layer == "pooler":
        # Pooler (N=4096, K=8192): kfold R=32, Q=2 gives 128 CTAs (vs 64 CTAs for BN=64)!
        for u in (0, 2):
            raw.append(CandR2("kfold", 32, 256, bt, 4, split=1, q=2, unroll=u))
            raw.append(CandR2("kfold", 32, 256, bt, 3, split=1, q=2, unroll=u))
            raw.append(CandR2("kfold", 32, 128, bt, 6, split=1, q=2, unroll=u))
            raw.append(CandR2("kfold", 32, 128, bt, 8, split=1, q=2, unroll=u))
            raw.append(CandR2("split", 64, 128, bt, 8, split=1, unroll=u))
            raw.append(CandR2("split", 64, 256, bt, 4, split=1, unroll=u))
            raw.append(CandR2("split", 64, 256, bt, 4, split=2, unroll=u))

    elif layer == "unpooler" and n == 8192:
        for u in (0, 2):
            raw.append(CandR2("split", 64, 256, bt, 4, split=1, unroll=u))
            raw.append(CandR2("split", 64, 128, bt, 8, split=1, unroll=u))
            raw.append(CandR2("split", 64, 128, bt, 6, split=1, unroll=u))

    elif layer == "o":
        for u in (0, 2):
            raw.append(CandR2("split", 64, 256, bt, 4, split=1, unroll=u))
            raw.append(CandR2("split", 64, 128, bt, 8, split=1, unroll=u))
            raw.append(CandR2("split", 64, 128, bt, 4, split=1, unroll=u))

    seen = set()
    out = []
    for c in raw:
        if c not in seen and c.fits(n, k, m):
            seen.add(c)
            out.append(c)
    return out


# ---------------------------------------------------------------------------------- CPU compile mode

def compile_report(only: str) -> None:
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import get_ptxas
    from triton.experimental.gluon._runtime import GluonASTSource

    compile_shapes = [
        ("qkv_gate", 8, 2304, 8192),
        ("qkv_gate", 16, 2304, 8192),
        ("qkv_gate", 32, 2304, 8192),
        ("pooler", 16, 4096, 8192),
        ("pooler", 32, 4096, 8192),
        ("o", 16, 8192, 1024),
    ]
    seen_configs = set()
    print(f"{'CANDIDATE':52s} | {'SMEM':>7s} | {'REGS':>4s} | {'SPILL':>5s} | {'WGMMA':>5s} | {'TIME':>5s}", flush=True)
    print("-" * 92, flush=True)
    for label, m, n, k in compile_shapes:
        for c in candidates_for(label, m, n, k):
            cfg_key = (c, n, k)
            if cfg_key in seen_configs:
                continue
            seen_configs.add(cfg_key)
            tag = f"{label}_m{m}_{c.name}"
            if only and only not in tag:
                continue
            w = torch.zeros(n, k, dtype=torch.bfloat16)
            x = torch.zeros(m, k, dtype=torch.bfloat16)
            wl = gl.NVMMASharedLayout.get_default_for([c.block_n, c.block_k], gl.bfloat16)
            xl = gl.NVMMASharedLayout.get_default_for([c.block_t, c.block_k], gl.bfloat16)
            w_desc = GluonDesc.from_tensor(w, [c.block_n, c.block_k], wl)
            x_desc = GluonDesc.from_tensor(x, [c.block_t, c.block_k], xl)
            sig = {
                "w_desc": w_desc.__mangle__(),
                "x_desc": x_desc.__mangle__(),
                "out_ptr": "*bf16",
                "ws_ptr": "*fp32",
                "cnt_ptr": "*i32",
                "M": "i32",
            }
            if c.kind == "kfold":
                fn = _gluon_gemv_kfold_r2
                consts = dict(N=n, K=k, STAGES=c.stages, SPLIT=c.split, Q=c.q, UNROLL=c.unroll, num_warps=c.warps)
            else:
                fn = _gluon_gemv_split_r2
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
            t0 = time.time()
            try:
                ker = triton.compile(src, target=GPUTarget("cuda", 90, 32), options={"num_warps": c.warps})
            except Exception as exc:
                print(f"{tag:52s} | COMPILE FAILED: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
                continue
            ptx = ker.asm["ptx"]
            wgmma_cnt = len(re.findall(r"wgmma\.mma_async", ptx))
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
            print(f"{tag:52s} | {ker.metadata.shared // 1024:5d}KB | {regs:>4s} | {spills:>5s} | {wgmma_cnt:5d} | {time.time() - t0:4.1f}s",
                  flush=True)


# ---------------------------------------------------------------------------------- GPU benchmark mode

class RunnerStateR2:
    def __init__(self, weights: dict[str, torch.Tensor]) -> None:
        self.w = {k: v.contiguous() for k, v in weights.items()}
        self.ws = torch.empty(WS_FLOATS, device="cuda", dtype=torch.float32)
        self.cnt = torch.zeros(COUNTERS, device="cuda", dtype=torch.int32)
        self.w_descs: dict[tuple[str, int, int], object] = {}
        self.x_layouts: dict[tuple[int, int], object] = {}

    def w_desc(self, key: str, bn: int, bk: int):
        dk = (key, bn, bk)
        if dk not in self.w_descs:
            wl = gl.NVMMASharedLayout.get_default_for([bn, bk], gl.bfloat16)
            self.w_descs[dk] = GluonDesc.from_tensor(self.w[key], [bn, bk], wl)
        return self.w_descs[dk]

    def x_desc(self, x: torch.Tensor, bt: int, bk: int):
        lk = (bt, bk)
        if lk not in self.x_layouts:
            self.x_layouts[lk] = gl.NVMMASharedLayout.get_default_for([bt, bk], gl.bfloat16)
        return GluonDesc.from_tensor(x, [bt, bk], self.x_layouts[lk])

    def launch(self, c: CandR2, key: str, x: torch.Tensor) -> torch.Tensor:
        if x.stride(1) != 1:
            x = x.contiguous()
        m = x.shape[0]
        n, k = self.w[key].shape
        out = torch.empty(m, n, device=x.device, dtype=torch.bfloat16)
        wd = self.w_desc(key, c.block_n, c.block_k)
        xd = self.x_desc(x, c.block_t, c.block_k)
        grid = (n // c.block_n, c.split)
        if c.kind == "kfold":
            _gluon_gemv_kfold_r2[grid](
                wd, xd, out, self.ws, self.cnt, m,
                N=n, K=k, STAGES=c.stages, SPLIT=c.split, Q=c.q, UNROLL=c.unroll, num_warps=c.warps,
            )
        else:
            _gluon_gemv_split_r2[grid](
                wd, xd, out, self.ws, self.cnt, m,
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


def check_cand_correctness(reference, rs: RunnerStateR2, c: CandR2, layer: str, key: str, m: int, w: dict,
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
    ap.add_argument("--shapes", default="qkv_gate:8,qkv_gate:16,qkv_gate:32,pooler:8,pooler:16,pooler:32,unpooler:16,unpooler:32,o:8,o:16,o:32")
    ap.add_argument("--reps", type=int, default=35)
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
    rs = RunnerStateR2(w)
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
            tag = f"{c.name} (ctas={ctas}, smem={c.smem_bytes() // 1024}KB)"
            try:
                ok_c, msg_c = check_cand_correctness(reference, rs, c, layer, key, m, w, include_nearby=False)
                if not ok_c:
                    print(f"  {tag:58s}: WRONG ({msg_c})", flush=True)
                    continue
                g_cand, out_c = graph_of(lambda c=c: rs.launch(c, key, x_buf))
                ok_g, msg_g = reference.check_implementation(data_static, out_c, w)
                if not ok_g:
                    print(f"  {tag:58s}: GRAPH WRONG ({msg_g})", flush=True)
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
                    print(f"  {tag:58s}: NON-DETERMINISTIC across graph replays!", flush=True)
                    continue
                tb, tc, d = paired(g_base, g_cand, flush, args.reps, time_replay)
                bytes_moved = n * k * 2 + m * k * 2 + m * n * 2
                tb_s = bytes_moved / tc / 1e6
                print(f"  {tag:58s}: cp3 {tb:6.2f} | cand {tc:6.2f} us ({tb_s:4.2f} TB/s) | delta {d:+6.2f} us",
                      flush=True)
                if best is None or d < best["delta"]:
                    best = {"cand": c, "tb": tb, "tc": tc, "delta": d, "tag": tag}
            except Exception as exc:
                print(f"  {tag:58s}: FAILED {type(exc).__name__}: {str(exc)[:300]}", flush=True)

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
    load_deltas = {ld: 0.0 for ld in CP3_LOAD_TOTALS_US}
    for (layer, m), b in results.items():
        ld = M_TO_LOAD.get(m, "")
        gain = min(0.0, b["delta"])
        if ld in load_deltas:
            load_deltas[ld] += gain
        status = "WIN" if b["delta"] < -0.08 and b["nearby_ok"] else "TIE/LOSE"
        print(f"  {layer:10s} m={m:2d} ({ld:9s}): {b['cand'].name:42s} cp3={b['tb']:6.2f} cand={b['tc']:6.2f} "
              f"delta={b['delta']:+6.2f} us  nearby={b['nearby_ok']} [{status}]", flush=True)

    old_ratios = [SGLANG_LOAD_TOTALS_US[ld] / CP3_LOAD_TOTALS_US[ld] for ld in CP3_LOAD_TOTALS_US]
    new_ratios = [
        SGLANG_LOAD_TOTALS_US[ld] / max(1.0, CP3_LOAD_TOTALS_US[ld] + load_deltas[ld])
        for ld in CP3_LOAD_TOTALS_US
    ]
    old_geo = math.exp(sum(map(math.log, old_ratios)) / len(old_ratios))
    new_geo = math.exp(sum(map(math.log, new_ratios)) / len(new_ratios))
    for ld in ("verify 1", "decode 16", "decode 32"):
        old_t = CP3_LOAD_TOTALS_US[ld]
        new_t = old_t + load_deltas[ld]
        print(f"  Load {ld:10s}: cp3 {old_t:6.2f} us -> projected {new_t:6.2f} us (delta {load_deltas[ld]:+6.2f} us, "
              f"{SGLANG_LOAD_TOTALS_US[ld] / old_t:.3f}x -> {SGLANG_LOAD_TOTALS_US[ld] / new_t:.3f}x)", flush=True)
    print(f"  Projected Geomean Speedup: {old_geo:.4f}x -> {new_geo:.4f}x ({new_geo - old_geo:+.4f}x)", flush=True)


if __name__ == "__main__":
    main()
