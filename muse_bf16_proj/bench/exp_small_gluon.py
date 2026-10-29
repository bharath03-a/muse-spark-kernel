"""Track A: Small-m (m in {8, 16, 32}) Gluon & Split-K Architecture Sweep.

Compares three small-m Gluon architectures against `checkpoints/cp2_gluon_router.py`:
  A1a (`v2`):     No-split BN=32 TMA-ring with Ampere/Hopper `mma_v2` (72 CTAs on qkv_gate)
  A1b (`kfold`):  No-split R=32, Q=2 interleaved K-fold TMA-ring with `warpgroup_mma` (72 CTAs, no W repack)
  A2  (`split`):  BN=64 `warpgroup_mma` TMA-ring with deterministic last-CTA Split-K in {1, 2, 3, 4, 8}
  A3  (`o` / `unpooler_slice` / `pooler` / `unpooler` stage, BK, BN=128, and Split-K tuning)

Usage:
    modal run modal_app.py::compile_only --cmd "exp_small_gluon --compile"
    modal run modal_app.py::micro --name exp_small_gluon --args "--reps 40"
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
    from triton.experimental.gluon.language.nvidia.ampere import mma_v2
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

CP2_PATH = "checkpoints/cp2_gluon_router.py"
OVERHEAD_US = 4.0
GLUON_SMEM = 220 * 1024
WS_FLOATS = 2 * 2**20
COUNTERS = 8192

# Judge load totals (us) from cp2_gluon_router profile (job c2e2b791b06c4cb9) for geomean impact estimation.
CP2_LOAD_TOTALS_US = {
    "decode 1": 41.70,
    "verify 1": 47.37,
    "decode 16": 60.61,
    "decode 32": 63.61,
    "verify 16": 76.76,
    "verify 32": 93.00,
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


class Cand(NamedTuple):
    kind: str       # "split" (wgmma SPLIT>=1), "v2" (mma_v2 BN=32/64), "kfold" (wgmma R=32, Q=2)
    block_n: int    # real rows per CTA (32, 64, or 128)
    block_k: int    # K per TMA box (64, 128, or 256)
    block_t: int    # token tile size (8, 16, or 32)
    stages: int     # ring buffer depth (4, 8)
    split: int = 1  # global split-K factor (for kind="split")
    warps: int = 4  # num_warps per CTA
    warps_m: int = 4  # for kind="v2": warps along M (BN)
    warps_n: int = 1  # for kind="v2": warps along N (BT)
    q: int = 1      # for kind="kfold": K-fold factor (2)

    @property
    def name(self) -> str:
        if self.kind == "v2":
            return (f"v2_BN{self.block_n}_BK{self.block_k}_BT{self.block_t}_S{self.stages}"
                    f"_w{self.warps_m}x{self.warps_n}")
        if self.kind == "kfold":
            return (f"kfold_R{self.block_n}Q{self.q}_BK{self.block_k}_BT{self.block_t}"
                    f"_S{self.stages}_w{self.warps}")
        return (f"wgmma_BN{self.block_n}_BK{self.block_k}_BT{self.block_t}_S{self.stages}"
                f"_SP{self.split}_w{self.warps}")

    def smem_bytes(self) -> int:
        if self.kind == "kfold":
            ring = self.stages * self.q * (self.block_n + self.block_t) * self.block_k * 2
            red = (self.block_n * self.q) * (self.q * self.block_t) * 4
            return ring + red
        return self.stages * (self.block_n + self.block_t) * self.block_k * 2

    def fits(self, n: int, k: int, m: int) -> bool:
        if self.block_t < m or n % self.block_n != 0 or k % self.block_k != 0:
            return False
        if self.smem_bytes() > GLUON_SMEM:
            return False
        ki = k // self.block_k
        if self.kind == "split":
            if ki // self.split < self.stages:
                return False
            if self.split > 1 and (self.split * n * self.block_t > WS_FLOATS or n // self.block_n > COUNTERS):
                return False
        elif self.kind == "v2":
            if ki < self.stages or self.block_n % (16 * self.warps_m) != 0 or self.block_t % (8 * self.warps_n) != 0:
                return False
        elif self.kind == "kfold":
            if ki % self.q != 0 or (ki // self.q) < self.stages:
                return False
            if self.block_n * self.q != 16 * self.warps:
                return False
        return True


if HAVE_GLUON:
    @gluon.jit
    def _gluon_gemv_split(w_desc, x_desc, out_ptr, ws_ptr, cnt_ptr, M,
                          N: gl.constexpr, K: gl.constexpr, SPLIT: gl.constexpr,
                          STAGES: gl.constexpr, num_warps: gl.constexpr):
        """Architecture A2 & A3: wgmma TMA-ring GEMV with optional deterministic last-CTA Split-K."""
        BN: gl.constexpr = w_desc.block_type.shape[0]
        BK: gl.constexpr = w_desc.block_type.shape[1]
        BT: gl.constexpr = x_desc.block_type.shape[0]
        KI: gl.constexpr = K // BK
        EVEN: gl.constexpr = (KI % SPLIT) == 0
        K_PER: gl.constexpr = KI // SPLIT
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
            k_start = (pid_s * KI) // SPLIT
            k_end = ((pid_s + 1) * KI) // SPLIT
            k_steps = k_end - k_start

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
            # Workspace layout [N // BN, SPLIT, BN, BT] has BT as stride-1, matching wgmma register pairs.
            ws_base = ws_ptr + pid_n * (SPLIT * BN * BT) + offs_nl[:, None] * BT + offs_t[None, :]
            gl.store(ws_base + pid_s * (BN * BT), acc)
            gl.barrier()
            done = gl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
            if done == SPLIT - 1:
                total = gl.zeros([BN, BT], gl.float32, layout=mma)
                for s in gl.static_range(SPLIT):
                    total += gl.load(ws_base + s * (BN * BT), cache_modifier=".cg")
                gl.store(out_ptrs, total.to(gl.bfloat16), mask=mask)
                gl.atomic_xchg(cnt_ptr + pid_n, 0)

    @gluon.jit
    def _gluon_gemv_v2(w_desc, x_desc, out_ptr, M, N: gl.constexpr, K: gl.constexpr, STAGES: gl.constexpr,
                       WARPS_M: gl.constexpr, WARPS_N: gl.constexpr, num_warps: gl.constexpr):
        """Architecture A1a: BN=32 (or 64) TMA-ring GEMV using mma_v2 (m16n8k16) from registers."""
        BN: gl.constexpr = w_desc.block_type.shape[0]
        BK: gl.constexpr = w_desc.block_type.shape[1]
        BT: gl.constexpr = x_desc.block_type.shape[0]
        KI: gl.constexpr = K // BK
        mma: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[WARPS_M, WARPS_N],
                                                      instr_shape=[16, 8])
        adot: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=2)
        bdot: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma, k_width=2)
        n0 = gl.program_id(0) * BN
        w_bufs = gl.allocate_shared_memory(w_desc.dtype, [STAGES] + w_desc.block_type.shape, w_desc.layout)
        x_bufs = gl.allocate_shared_memory(x_desc.dtype, [STAGES] + x_desc.block_type.shape, x_desc.layout)
        ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
        for i in gl.static_range(STAGES):
            mbarrier.init(ready.index(i), count=1)
        nbytes: gl.constexpr = w_desc.block_type.nbytes + x_desc.block_type.nbytes
        for i in gl.static_range(STAGES):
            pred = i < KI
            mbarrier.expect(ready.index(i), nbytes, pred=pred)
            tma.async_load(w_desc, [n0, i * BK], ready.index(i), w_bufs.index(i), pred=pred)
            tma.async_load(x_desc, [0, i * BK], ready.index(i), x_bufs.index(i), pred=pred)
        acc = gl.zeros([BN, BT], gl.float32, layout=mma)
        for i in range(KI):
            s = i % STAGES
            mbarrier.wait(ready.index(s), (i // STAGES) & 1)
            w_reg = w_bufs.index(s).load(adot)
            x_reg = x_bufs.index(s).permute((1, 0)).load(bdot)
            gl.barrier()
            j = i + STAGES
            pred = j < KI
            mbarrier.expect(ready.index(s), nbytes, pred=pred)
            tma.async_load(w_desc, [n0, j * BK], ready.index(s), w_bufs.index(s), pred=pred)
            tma.async_load(x_desc, [0, j * BK], ready.index(s), x_bufs.index(s), pred=pred)
            acc = mma_v2(w_reg, x_reg, acc)
        offs_n = n0 + gl.arange(0, BN, layout=gl.SliceLayout(1, mma))
        offs_t = gl.arange(0, BT, layout=gl.SliceLayout(0, mma))
        gl.store(out_ptr + offs_t[None, :] * N + offs_n[:, None], acc.to(gl.bfloat16),
                 mask=(offs_t < M)[None, :])

    @gluon.jit
    def _kfold_issue(w_desc, x_desc, w_bufs, x_bufs, bar, s, i, n0, pred,
                     R: gl.constexpr, BT: gl.constexpr, BK: gl.constexpr, Q: gl.constexpr):
        NBYTES: gl.constexpr = Q * (w_desc.block_type.nbytes + x_desc.block_type.nbytes)
        mbarrier.expect(bar, NBYTES, pred=pred)
        for q in gl.static_range(Q):
            k_off = (Q * i + q) * BK
            tma.async_load(w_desc, [n0, k_off], bar, w_bufs.index(s).slice(q * R, R, dim=0), pred=pred)
            tma.async_load(x_desc, [0, k_off], bar, x_bufs.index(s).slice(q * BT, BT, dim=0), pred=pred)

    @gluon.jit
    def _gluon_gemv_kfold(w_desc, x_desc, out_ptr, M, N: gl.constexpr, K: gl.constexpr,
                          STAGES: gl.constexpr, Q: gl.constexpr, num_warps: gl.constexpr):
        """Architecture A1b: R=32 real rows per CTA, Q=2 interleaved K-blocks into BN=64 wgmma (no W repack)."""
        R: gl.constexpr = w_desc.block_type.shape[0]
        BK: gl.constexpr = w_desc.block_type.shape[1]
        BT: gl.constexpr = x_desc.block_type.shape[0]
        BN: gl.constexpr = R * Q
        KI: gl.constexpr = K // (BK * Q)
        mma: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[num_warps, 1],
                                                      instr_shape=[16, Q * BT, 16])
        n0 = gl.program_id(0) * R
        w_bufs = gl.allocate_shared_memory(w_desc.dtype, [STAGES, BN, BK], w_desc.layout)
        x_bufs = gl.allocate_shared_memory(x_desc.dtype, [STAGES, Q * BT, BK], x_desc.layout)
        ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
        for i in gl.static_range(STAGES):
            mbarrier.init(ready.index(i), count=1)
        for i in gl.static_range(STAGES - 1):
            _kfold_issue(w_desc, x_desc, w_bufs, x_bufs, ready.index(i), i, i, n0, i < KI, R, BT, BK, Q)
        acc = gl.zeros([BN, Q * BT], gl.float32, layout=mma)
        for i in range(KI):
            s = i % STAGES
            mbarrier.wait(ready.index(s), (i // STAGES) & 1)
            acc = warpgroup_mma(w_bufs.index(s), x_bufs.index(s).permute((1, 0)), acc, is_async=True)
            acc = warpgroup_mma_wait(1, deps=(acc,))
            j = i + STAGES - 1
            sj = j % STAGES
            _kfold_issue(w_desc, x_desc, w_bufs, x_bufs, ready.index(sj), sj, j, n0, j < KI, R, BT, BK, Q)
        acc = warpgroup_mma_wait(0, deps=(acc,))
        TW: gl.constexpr = 32 if BT >= 32 else BT
        epi: gl.constexpr = gl.BlockedLayout([1, BT // TW], [32 // TW, TW], [num_warps, 1], [1, 0])
        red = gl.allocate_shared_memory(gl.float32, [BN, Q * BT],
                                        gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0]))
        red.store(acc)
        gl.barrier()
        tot = gl.zeros([R, BT], gl.float32, layout=epi)
        for q in gl.static_range(Q):
            tot += red.slice(q * R, R, dim=0).slice(q * BT, BT, dim=1).load(epi)
        offs_n = n0 + gl.arange(0, R, layout=gl.SliceLayout(1, epi))
        offs_t = gl.arange(0, BT, layout=gl.SliceLayout(0, epi))
        gl.store(out_ptr + offs_t[None, :] * N + offs_n[:, None], tot.to(gl.bfloat16),
                 mask=(offs_t < M)[None, :])


def candidates_for(layer: str, m: int, n: int, k: int) -> list[Cand]:
    """Generate architecture candidates for (layer, m, n, k)."""
    bt_min = 8 if m <= 8 else (16 if m <= 16 else 32)
    bts = (8, 16) if m <= 8 else ((16,) if m <= 16 else (32,))
    raw: list[Cand] = []

    if layer == "qkv_gate":
        # Architecture A1: no-split (SPLIT=1) BN=64 baseline + A1a (mma_v2 BN=32) + A1b (kfold R=32, Q=2)
        for bt in bts:
            raw.append(Cand("split", 64, 128, bt, 8, split=1))
            raw.append(Cand("split", 64, 256, bt, 4, split=1))
            # A1a: mma_v2 BN=32 (72 CTAs)
            raw.append(Cand("v2", 32, 128, bt, 4, warps=2, warps_m=2, warps_n=1))
            raw.append(Cand("v2", 32, 128, bt, 8, warps=2, warps_m=2, warps_n=1))
            raw.append(Cand("v2", 32, 64, bt, 8, warps=2, warps_m=2, warps_n=1))
            if bt >= 16:
                raw.append(Cand("v2", 32, 128, bt, 4, warps=4, warps_m=2, warps_n=2))
                raw.append(Cand("v2", 32, 128, bt, 8, warps=4, warps_m=2, warps_n=2))
            # A1b: kfold R=32, Q=2 -> BN=64 wgmma (72 CTAs, no W repack)
            raw.append(Cand("kfold", 32, 128, bt, 4, warps=4, q=2))
            raw.append(Cand("kfold", 32, 128, bt, 8, warps=4, q=2))
            raw.append(Cand("kfold", 32, 256, bt, 4, warps=4, q=2))
        # Architecture A2: BN=64 wgmma TMA-ring with SPLIT in {2, 3, 4}
        for split in (2, 3, 4):
            for bt in bts:
                for bk, st in ((128, 8), (128, 4), (256, 4)):
                    raw.append(Cand("split", 64, bk, bt, st, split=split))

    elif layer == "o":
        # Architecture A3: o (N=8192, K=1024) at m=8, 16, 32
        for bt in bts:
            raw.append(Cand("split", 64, 128, bt, 4, split=1, warps=4))
            raw.append(Cand("split", 64, 128, bt, 8, split=1, warps=4))
            raw.append(Cand("split", 64, 256, bt, 4, split=1, warps=4))
            raw.append(Cand("split", 128, 128, bt, 4, split=1, warps=4))
            raw.append(Cand("split", 128, 128, bt, 4, split=1, warps=8))
            raw.append(Cand("split", 128, 128, bt, 8, split=1, warps=4))

    elif layer == "unpooler" and n == 1024:
        # Architecture A3: unpooler_slice (N=1024, K=4096, m=8)
        for bt in bts:
            for split in (2, 4, 8):
                for bk, st in ((128, 4), (128, 8), (256, 4), (64, 8), (64, 4)):
                    raw.append(Cand("split", 64, bk, bt, st, split=split))
            raw.append(Cand("kfold", 32, 128, bt, 4, warps=4, q=2))
            raw.append(Cand("kfold", 32, 128, bt, 8, warps=4, q=2))
            raw.append(Cand("v2", 32, 128, bt, 4, warps=2, warps_m=2, warps_n=1))

    elif layer == "pooler":
        # Architecture A3: pooler (N=4096, K=8192) - 64 CTAs no-split vs 128 CTAs (SPLIT=2 or kfold R=32)
        for bt in bts:
            raw.append(Cand("split", 64, 128, bt, 8, split=1))
            raw.append(Cand("split", 64, 128, bt, 4, split=1))
            raw.append(Cand("split", 64, 256, bt, 4, split=1))
            for split in (2,):
                for bk, st in ((128, 8), (128, 4), (256, 4)):
                    raw.append(Cand("split", 64, bk, bt, st, split=split))
            raw.append(Cand("kfold", 32, 128, bt, 4, warps=4, q=2))
            raw.append(Cand("kfold", 32, 128, bt, 8, warps=4, q=2))

    elif layer == "unpooler":
        # Architecture A3: unpooler (N=8192, K=4096, m=16, 32)
        raw.append(Cand("split", 64, 128, bt_min, 8, split=1))
        raw.append(Cand("split", 64, 128, bt_min, 4, split=1))
        raw.append(Cand("split", 64, 256, bt_min, 4, split=1))
        raw.append(Cand("split", 128, 128, bt_min, 4, split=1, warps=4))
        raw.append(Cand("split", 128, 128, bt_min, 4, split=1, warps=8))

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
        ("qkv_gate", 16, 2304, 8192),
        ("qkv_gate", 32, 2304, 8192),
        ("o", 16, 8192, 1024),
        ("unpooler_slice", 8, 1024, 4096),
        ("pooler", 16, 4096, 8192),
    ]
    seen_configs = set()
    for label, m, n, k in compile_shapes:
        layer = "unpooler" if label == "unpooler_slice" else label
        for c in candidates_for(layer, m, n, k):
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
            if c.kind == "split":
                fn = _gluon_gemv_split
                sig = {
                    "w_desc": w_desc.__mangle__(),
                    "x_desc": x_desc.__mangle__(),
                    "out_ptr": "*bf16",
                    "ws_ptr": "*fp32",
                    "cnt_ptr": "*i32",
                    "M": "i32",
                }
                consts = dict(N=n, K=k, SPLIT=c.split, STAGES=c.stages, num_warps=c.warps)
            elif c.kind == "v2":
                fn = _gluon_gemv_v2
                sig = {
                    "w_desc": w_desc.__mangle__(),
                    "x_desc": x_desc.__mangle__(),
                    "out_ptr": "*bf16",
                    "M": "i32",
                }
                consts = dict(N=n, K=k, STAGES=c.stages, WARPS_M=c.warps_m, WARPS_N=c.warps_n, num_warps=c.warps)
            else:
                fn = _gluon_gemv_kfold
                sig = {
                    "w_desc": w_desc.__mangle__(),
                    "x_desc": x_desc.__mangle__(),
                    "out_ptr": "*bf16",
                    "M": "i32",
                }
                consts = dict(N=n, K=k, STAGES=c.stages, Q=c.q, num_warps=c.warps)

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
                print(f"== {tag}: compile FAILED {type(exc).__name__}: {str(exc)[:1000]}", flush=True)
                continue
            ptx = ker.asm["ptx"]
            ops = re.findall(r"^\s*(?:@%p\d+\s+)?([a-z][a-z0-9_]*(?:\.[a-z0-9_]+)*)", ptx, flags=re.M)
            hist: dict[str, int] = {}
            for o in ops:
                key = o if o.startswith(("ld.shared", "wgmma", "mma", "ldmatrix", "cp.async.bulk", "atom", "bar")) else o.split(".")[0]
                hist[key] = hist.get(key, 0) + 1
            top = sorted(hist.items(), key=lambda kv: -kv[1])[:16]
            ptxas_info = []
            with tempfile.TemporaryDirectory() as td:
                path = f"{td}/k.ptx"
                open(path, "w").write(ptx)
                res = subprocess.run(
                    [get_ptxas(90).path, "-arch=sm_90a", "-v", path, "-o", f"{td}/k.o"],
                    capture_output=True,
                    text=True,
                )
                for line in (res.stdout + res.stderr).splitlines():
                    if any(s in line for s in ("registers", "spill", "Compile time", "Warning", "error")):
                        ptxas_info.append(line.strip())
            print(
                f"== {tag}: ok ({time.time() - t0:.1f}s) smem={ker.metadata.shared}B | "
                f"{' | '.join(ptxas_info)}",
                flush=True,
            )
            print("   ptx:", " ".join(f"{o}:{cnt}" for o, cnt in top), flush=True)


# ---------------------------------------------------------------------------------- GPU benchmark mode

class RunnerState:
    def __init__(self, weights: dict[str, torch.Tensor]) -> None:
        self.w = {k: v.contiguous() for k, v in weights.items()}
        self.w["unpooler_slice"] = self.w["unpooler"][:1024]
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

    def launch(self, c: Cand, key: str, x: torch.Tensor) -> torch.Tensor:
        if x.stride(1) != 1:
            x = x.contiguous()
        m = x.shape[0]
        n, k = self.w[key].shape
        out = torch.empty(m, n, device=x.device, dtype=torch.bfloat16)
        wd = self.w_desc(key, c.block_n, c.block_k)
        xd = self.x_desc(x, c.block_t, c.block_k)
        if c.kind == "split":
            _gluon_gemv_split[(n // c.block_n, c.split)](
                wd, xd, out, self.ws, self.cnt, m,
                N=n, K=k, SPLIT=c.split, STAGES=c.stages, num_warps=c.warps,
            )
        elif c.kind == "v2":
            _gluon_gemv_v2[(n // c.block_n,)](
                wd, xd, out, m,
                N=n, K=k, STAGES=c.stages, WARPS_M=c.warps_m, WARPS_N=c.warps_n, num_warps=c.warps,
            )
        elif c.kind == "kfold":
            _gluon_gemv_kfold[(n // c.block_n,)](
                wd, xd, out, m,
                N=n, K=k, STAGES=c.stages, Q=c.q, num_warps=c.warps,
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


def check_cand_correctness(reference, rs: RunnerState, c: Cand, layer: str, key: str, m: int, w: dict,
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
    ap.add_argument("--shapes", default="qkv_gate:8,qkv_gate:16,qkv_gate:32,o:8,o:16,o:32,unpooler:8,pooler:8,pooler:16,pooler:32,unpooler:16,unpooler:32")
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
    cp2 = load_module(CP2_PATH, "cp2")
    cst = cp2.prepare({k: v.clone() for k, v in w.items()})
    rs = RunnerState(w)
    flush = torch.empty(FLUSH_BYTES // 4, device="cuda", dtype=torch.float32).normal_()
    print(f"cp2 prepare done in {time.time() - t0:.1f}s; running paired cold-L2 CUDA-graph sweeps (reps={args.reps})",
          flush=True)

    shapes = [(s.split(":")[0], int(s.split(":")[1])) for s in args.shapes.split(",") if s.strip()]
    results = {}

    for layer, m in shapes:
        key = "unpooler_slice" if layer == "unpooler" and m <= 8 else layer
        if args.only and args.only not in f"{key}:{m}" and args.only not in layer:
            continue
        n, k = rs.w[key].shape
        ctas_64 = n // 64
        sol_us = reference.speed_of_light_ns(m, layer, {"hbm_bytes_per_s": 4.8e12, "bf16_dense_flops_per_s": 989e12}) / 1e3
        data = reference.generate_input(m, layer, 1)
        x_buf = data["x"].clone()
        data_static = {"x": x_buf, "config": data["config"]}

        g_base, out_b = graph_of(lambda: cp2.kernel_fn(data_static, cst))
        ok_b, msg_b = reference.check_implementation(data_static, out_b, w)
        print(f"\n==================== {key} m={m} (N={n}, K={k}, BN64_CTAs={ctas_64}, SOL={sol_us:.2f} us) "
              f"cp2={'ok' if ok_b else 'FAIL ' + msg_b} ====================", flush=True)

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
                tb, tc, d = paired(g_base, g_cand, flush, args.reps, time_replay)
                bytes_moved = n * k * 2 + m * k * 2 + m * n * 2
                tb_s = bytes_moved / tc / 1e6
                print(f"  {tag:58s}: cp2 {tb:6.2f} | cand {tc:6.2f} us ({tb_s:4.2f} TB/s) | delta {d:+6.2f} us",
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
            print(f"  >> BEST {key} m={m}: {best['cand'].name} -> cp2 {best['tb']:.2f} vs cand {best['tc']:.2f} us "
                  f"(delta {best['delta']:+.2f} us) | nearby m {NEARBY_M.get(m, (m,))}: {msg_near}", flush=True)

    # Whole-load score impact summary
    print("\n==================== SUMMARY & WHOLE-LOAD SCORE IMPACT vs CP2 ====================", flush=True)
    load_deltas = {ld: 0.0 for ld in CP2_LOAD_TOTALS_US}
    for (layer, m), b in results.items():
        ld = M_TO_LOAD.get(m, "")
        gain = min(0.0, b["delta"])  # only adopt improvements
        if ld in load_deltas:
            load_deltas[ld] += gain
        status = "WIN" if b["delta"] < -0.10 and b["nearby_ok"] else "TIE/LOSE"
        print(f"  {layer:10s} m={m:2d} ({ld:9s}): {b['cand'].name:42s} cp2={b['tb']:6.2f} cand={b['tc']:6.2f} "
              f"delta={b['delta']:+6.2f} us  nearby={b['nearby_ok']} [{status}]", flush=True)

    old_ratios = [SGLANG_LOAD_TOTALS_US[ld] / CP2_LOAD_TOTALS_US[ld] for ld in CP2_LOAD_TOTALS_US]
    new_ratios = [
        SGLANG_LOAD_TOTALS_US[ld] / max(1.0, CP2_LOAD_TOTALS_US[ld] + load_deltas[ld])
        for ld in CP2_LOAD_TOTALS_US
    ]
    old_geo = math.exp(sum(map(math.log, old_ratios)) / len(old_ratios))
    new_geo = math.exp(sum(map(math.log, new_ratios)) / len(new_ratios))
    for ld in ("verify 1", "decode 16", "decode 32"):
        old_t = CP2_LOAD_TOTALS_US[ld]
        new_t = old_t + load_deltas[ld]
        print(f"  Load {ld:10s}: cp2 {old_t:6.2f} us -> projected {new_t:6.2f} us (delta {load_deltas[ld]:+6.2f} us, "
              f"{SGLANG_LOAD_TOTALS_US[ld] / old_t:.3f}x -> {SGLANG_LOAD_TOTALS_US[ld] / new_t:.3f}x)", flush=True)
    print(f"  Projected Geomean Speedup: {old_geo:.4f}x -> {new_geo:.4f}x ({new_geo - old_geo:+.4f}x)", flush=True)


if __name__ == "__main__":
    main()
