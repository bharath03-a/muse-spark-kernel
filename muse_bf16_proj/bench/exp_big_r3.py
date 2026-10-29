"""Track B Round 3: Closing Remaining Large-m Gaps (qkv_gate m=256, pooler m=256, o m=128/256, unpooler m=128/256).

Key Round 3 architectural upgrades in `_big_gemm_v3`:
  1. Hardware TMA Epilogue Store (`TMA_STORE=1`, `out_desc.store([pid_m * BM, pid_n * BN], val)`):
     - Uses Hopper's TMA engine (`cp.async.bulk.tensor.2d.global.shared::cta`) instead of per-thread
       register `tl.store` + address arithmetic + mask predication.
     - Works for both `SWAP=0` (`val = acc.to(tl.bfloat16)`) and `SWAP=1` (`val = acc.to(tl.bfloat16).T`,
       which lowers to `cvt.rn.bf16x2.f32` + `stmatrix.sync.aligned.m8n8.x4.trans.shared.b16`).
     - Eliminates ~25-30 epilogue address/predicate registers and cuts epilogue latency on short-K
       layers like `o` (K=1024, only 8-16 K-loop iterations).
  2. Single-M-Tile (`BM=256, BN=64/128, num_warps=8` at m=256; `BM=128, BN=64/128` at m=128):
     - At m=256, `BM=256` (`num_m = 1`) loads the weight matrix `W` (36 MB for qkv_gate, 64 MB for
       pooler/unpooler, 16 MB for o) STRICTLY ONCE across all 256 tokens instead of twice (`BM=128`),
       cutting `W` HBM/L2 traffic in half!
     - With `num_warps=8` (256 threads), `BM=256, BN=64` uses only 64 FP32 accumulator registers per
       thread (identical per-thread register pressure to `BM=128, BN=64, num_warps=4`).
     - Gives ideal 1-wave CTA counts on H200 (132 SMs):
       * `qkv_gate` m=256: `BM=256, BN=64, SPLIT=3` -> `1 * 36 * 3 = 108` CTAs (1 wave, W read 1x!)
       * `pooler`   m=256: `BM=256, BN=64, SPLIT=2` -> `1 * 64 * 2 = 128` CTAs (1 wave, W read 1x, 1-load reduce!)
       * `unpooler` m=256: `BM=256, BN=64, SPLIT=1` -> `1 * 128 * 1 = 128` CTAs (1 wave, W read 1x, 0 reduce!)
       * `o`        m=256: `BM=256, BN=64, SPLIT=1` -> `1 * 128 * 1 = 128` CTAs (1 wave, W read 1x, TMA store!)
  3. Unlocked `SPLIT=2..4` Workspace (`WS_FLOATS = 4 * 2**20`) + Tile Ordering (`ORDER in {0, 1, 2}`):
     - Tests `pooler` m=256 with `SPLIT=2` (`BM=128, BN=128, SPLIT=2` -> 128 CTAs; `BM=256, BN=64, SPLIT=2` -> 128 CTAs).
     - Tests `ORDER=0` (M innermost, W reuse), `ORDER=1` (N innermost, full 16 MB `o` W cached in 50 MB L2),
       and `ORDER=2` (SPLIT innermost, then N, then M so split-K siblings co-schedule on simultaneous SMs).

Usage:
    modal run modal_app.py::compile_only --cmd "exp_big_r3 --compile"
    modal run modal_app.py::micro --name exp_big_r3 --args "--reps 35"
"""

from __future__ import annotations

import argparse
import os
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

BASE_CP = "checkpoints/cp3_gluon_v2.py"
SHAPES = {
    "qkv_gate": (2304, 8192),
    "o": (8192, 1024),
    "pooler": (4096, 8192),
    "unpooler": (8192, 4096),
}
HW = {"hbm_bytes_per_s": 4.8e12, "bf16_dense_flops_per_s": 989e12}
FLUSH_BYTES = 256 * 2**20
OVERHEAD_US = 4.0
WS_FLOATS = 8 * 2**20
SUB_WS_FLOATS = 4 * 2**20  # 16 MB workspace budget (supports SPLIT=2..4 at m=256)
COUNTERS = 8192
SMEM_LIMIT = 224 * 1024
SMEM_BUDGET = 208 * 1024


class Cfg(NamedTuple):
    bm: int
    bn: int
    bk: int
    split: int
    swap: int          # 0: acc [BM, BN], 1: acc [BN, BM]
    warps: int
    stages: int
    tma_store: int = 0 # 0: tl.store, 1: out_desc.store (Hopper TMA bulk tensor store)
    order: int = 0     # 0: M->S->N, 1: N->M->S, 2: S->N->M


def loop_smem_bytes(c: Cfg) -> int:
    return c.stages * (c.bm + c.bn) * c.bk * 2


def total_smem_est(c: Cfg) -> int:
    epi = (c.bm * c.bn * 2) if c.tma_store else 0
    return loop_smem_bytes(c) + epi


def fit_stages(c: Cfg) -> Cfg:
    s = c.stages
    while s > 2 and total_smem_est(c._replace(stages=s)) > SMEM_BUDGET:
        s -= 1
    return c._replace(stages=s)


def wgmma_tile_ok(bm: int, bn: int, swap: int, warps: int) -> bool:
    d0 = bn if swap else bm
    d1 = bm if swap else bn
    if d0 < 64 or d0 % 64 != 0 or d1 < 16 or d1 % 8 != 0 or d1 > 256:
        return False
    # Avoid >128 accumulator registers per thread
    if (bm * bn) // (warps * 32) > 128:
        return False
    return True


def valid_cfg(c: Cfg, layer: str, m: int) -> bool:
    n, k = SHAPES[layer]
    num_m = triton.cdiv(m, c.bm)
    if n % c.bn != 0 or k % c.bk != 0 or (k // c.bk) < c.split:
        return False
    if (k // c.bk) // c.split < c.stages or c.stages < 2:
        return False
    if total_smem_est(c) > SMEM_LIMIT:
        return False
    if c.split > 1 and c.split * num_m * c.bm * n > SUB_WS_FLOATS:
        return False
    if num_m * (n // c.bn) > COUNTERS:
        return False
    if not wgmma_tile_ok(c.bm, c.bn, c.swap, c.warps):
        return False
    # 128 acc regs/thread with split-K reduction spills if not enough headroom
    if (c.bm * c.bn) // (c.warps * 32) >= 128 and c.split > 1:
        return False
    return True


# --------------------------------------------------------------------------- @triton.jit Kernel (_big_gemm_v3)

@triton.jit(do_not_specialize=["M"])
def _big_gemm_v3(
    x_desc, w_desc, out_ref, ws_ptr, cnt_ptr, M,
    N: tl.constexpr, K: tl.constexpr, SPLIT: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    SWAP: tl.constexpr, TMA_STORE: tl.constexpr, ORDER: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n: tl.constexpr = N // BN
    if ORDER == 0:
        pid_m = pid % num_m
        pid_s = (pid // num_m) % SPLIT
        pid_n = pid // (num_m * SPLIT)
    elif ORDER == 1:
        pid_n = pid % num_n
        pid_m = (pid // num_n) % num_m
        pid_s = pid // (num_n * num_m)
    else:
        pid_s = pid % SPLIT
        pid_n = (pid // SPLIT) % num_n
        pid_m = pid // (SPLIT * num_n)

    KI: tl.constexpr = K // BK
    k0 = pid_s * KI // SPLIT
    k1 = (pid_s + 1) * KI // SPLIT
    if SWAP:
        acc = tl.zeros((BN, BM), dtype=tl.float32)
    else:
        acc = tl.zeros((BM, BN), dtype=tl.float32)
    for ki in range(k0, k1):
        a = x_desc.load([pid_m * BM, ki * BK])
        b = w_desc.load([pid_n * BN, ki * BK])
        if SWAP:
            acc = tl.dot(b, a.T, acc)
        else:
            acc = tl.dot(a, b.T, acc)

    if SPLIT == 1:
        if TMA_STORE:
            val = acc.to(tl.bfloat16)
            if SWAP:
                val = val.T
            out_ref.store([pid_m * BM, pid_n * BN], val)
        else:
            local_m = tl.arange(0, BM)
            local_n = tl.arange(0, BN)
            offs_m = pid_m * BM + local_m
            offs_n = pid_n * BN + local_n
            if SWAP:
                out_idx = offs_m[None, :] * N + offs_n[:, None]
                mask = (offs_m < M)[None, :]
            else:
                out_idx = offs_m[:, None] * N + offs_n[None, :]
                mask = (offs_m < M)[:, None]
            tl.store(out_ref + out_idx, acc.to(tl.bfloat16), mask=mask)
    else:
        TILE_ELEMS: tl.constexpr = BM * BN
        local_m = tl.arange(0, BM)
        local_n = tl.arange(0, BN)
        if SWAP:
            ws_offs = local_n[:, None] * BM + local_m[None, :]
        else:
            ws_offs = local_m[:, None] * BN + local_n[None, :]
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
            if TMA_STORE:
                val = total.to(tl.bfloat16)
                if SWAP:
                    val = val.T
                out_ref.store([pid_m * BM, pid_n * BN], val)
            else:
                offs_m = pid_m * BM + local_m
                offs_n = pid_n * BN + local_n
                if SWAP:
                    out_idx = offs_m[None, :] * N + offs_n[:, None]
                    mask = (offs_m < M)[None, :]
                else:
                    out_idx = offs_m[:, None] * N + offs_n[None, :]
                    mask = (offs_m < M)[:, None]
                tl.store(out_ref + out_idx, total.to(tl.bfloat16), mask=mask)
            tl.atomic_xchg(cnt, 0)


# --------------------------------------------------------------------------- Candidate Sweeps per (layer, m)

def candidates_for(layer: str, m: int) -> list[Cfg]:
    def T(bm, bn, bk, split, swap, warps, stages, tma_store=0, order=0):
        return fit_stages(Cfg(bm, bn, bk, split, swap, warps, stages, tma_store, order))

    raw: list[Cfg] = []

    if layer == "qkv_gate":
        # N=2304, K=8192 (n_tiles: 36 for BN=64, 18 for BN=128)
        if m == 128:
            raw = [
                # Round 2 winner: (128, 64, 64, 3, 0, 4, 8) -> 16.38 us (-0.59 us vs cuBLAS)
                T(128, 64, 64, 3, 0, 4, 8, 0, 0),
                T(128, 64, 64, 3, 0, 4, 7, 1, 0),
                T(128, 64, 64, 3, 1, 4, 8, 0, 0),
                T(128, 64, 64, 3, 1, 4, 7, 1, 0),
                T(128, 64, 64, 3, 0, 4, 8, 0, 1),
                # SPLIT=2 (72 CTAs, 1-load commutative reduction)
                T(128, 64, 64, 2, 0, 4, 8, 0, 0),
                T(128, 64, 64, 2, 0, 4, 7, 1, 0),
                T(128, 64, 64, 2, 1, 4, 8, 0, 0),
                T(128, 64, 64, 2, 1, 4, 7, 1, 0),
                # SPLIT=4 (144 CTAs)
                T(128, 64, 64, 4, 0, 4, 8, 0, 0),
                T(128, 64, 64, 4, 1, 4, 8, 0, 0),
                # BM=128, BN=128 (18 N-tiles: SPLIT=6 -> 108 CTAs, SPLIT=4 -> 72 CTAs, W read 1x!)
                T(128, 128, 64, 6, 0, 8, 5, 0, 0),
                T(128, 128, 64, 6, 1, 8, 5, 0, 0),
                T(128, 128, 64, 4, 0, 8, 5, 0, 0),
                T(128, 128, 64, 4, 0, 8, 5, 1, 0),
                T(128, 128, 64, 3, 0, 8, 5, 0, 0),
                # BM=64, BN=64 (2 * 36 = 72 tiles, SPLIT=2 -> 144 CTAs with 1-load reduce)
                T(64, 64, 64, 2, 0, 4, 8, 0, 0),
                T(64, 64, 64, 2, 1, 4, 8, 0, 0),
                T(64, 64, 64, 2, 0, 4, 8, 0, 2),
            ]
        else:
            # m=256:
            # 1) BM=256, BN=64 (num_m=1 -> W (36 MB) loaded STRICTLY ONCE across all 256 tokens!)
            #    n_tiles = 36: SPLIT=3 -> 108 CTAs (1 single wave on 132 SMs!), SPLIT=2 -> 72 CTAs (1-load reduce!)
            raw = [
                T(256, 64, 64, 3, 0, 8, 5, 0, 0),
                T(256, 64, 64, 3, 0, 8, 4, 1, 0),
                T(256, 64, 64, 3, 1, 8, 5, 0, 0),
                T(256, 64, 64, 3, 1, 8, 4, 1, 0),
                T(256, 64, 64, 3, 0, 8, 4, 0, 0),
                T(256, 64, 64, 3, 0, 8, 5, 0, 1),
                T(256, 64, 64, 2, 0, 8, 5, 0, 0),
                T(256, 64, 64, 2, 0, 8, 4, 1, 0),
                T(256, 64, 64, 2, 1, 8, 5, 0, 0),
                T(256, 64, 64, 2, 1, 8, 4, 1, 0),
                T(256, 64, 64, 4, 0, 8, 5, 0, 0),
                T(256, 64, 64, 4, 1, 8, 5, 0, 0),
                T(256, 64, 64, 1, 0, 8, 4, 1, 0),
                T(256, 64, 64, 1, 1, 8, 4, 1, 0),
                # 2) BM=128, BN=128 (num_m=2, n_tiles=18: SPLIT=3 -> 108 CTAs in 1 wave! SPLIT=2 -> 72 CTAs)
                T(128, 128, 64, 3, 0, 8, 5, 0, 0),
                T(128, 128, 64, 3, 0, 8, 5, 1, 0),
                T(128, 128, 64, 3, 1, 8, 5, 0, 0),
                T(128, 128, 64, 3, 1, 8, 5, 1, 0),
                T(128, 128, 64, 3, 0, 8, 5, 0, 2),
                T(128, 128, 64, 2, 0, 8, 5, 0, 0),
                T(128, 128, 64, 2, 0, 8, 5, 1, 0),
                T(128, 128, 64, 2, 1, 8, 5, 0, 0),
                T(128, 128, 64, 2, 1, 8, 5, 1, 0),
                # 3) BM=128, BN=64 (num_m=2, n_tiles=36: SPLIT=2 -> 144 CTAs, SPLIT=1 -> 72 CTAs)
                T(128, 64, 64, 2, 0, 4, 8, 0, 0),
                T(128, 64, 64, 2, 0, 4, 7, 1, 0),
                T(128, 64, 64, 2, 0, 4, 8, 0, 2),
                T(128, 64, 64, 2, 1, 4, 8, 0, 0),
                T(128, 64, 64, 2, 1, 4, 8, 0, 2),
                T(128, 64, 64, 1, 0, 4, 7, 1, 0),
                T(128, 64, 64, 1, 1, 4, 7, 1, 0),
            ]

    elif layer == "pooler":
        # N=4096, K=8192 (n_tiles: 64 for BN=64, 32 for BN=128)
        if m == 128:
            raw = [
                # Round 2 winner: (64, 128, 64, 2, 1, 4, 7) -> 22.78 us (-0.67 us)
                T(64, 128, 64, 2, 1, 4, 7, 0, 0),
                T(64, 128, 64, 2, 1, 4, 7, 1, 0),
                T(64, 128, 64, 2, 0, 4, 7, 0, 0),
                T(64, 128, 64, 2, 0, 4, 7, 1, 0),
                T(64, 128, 64, 2, 1, 4, 8, 0, 0),
                T(64, 128, 64, 2, 1, 4, 7, 0, 2),
                # Single-M-tile BM=128, BN=64 (reads W (64 MB) STRICTLY ONCE! 1 * 64 * 2 = 128 CTAs)
                T(128, 64, 64, 2, 0, 4, 8, 0, 0),
                T(128, 64, 64, 2, 0, 4, 7, 1, 0),
                T(128, 64, 64, 2, 1, 4, 8, 0, 0),
                T(128, 64, 64, 2, 1, 4, 7, 1, 0),
                T(128, 64, 64, 2, 0, 8, 7, 0, 0),
                T(128, 64, 64, 2, 1, 8, 7, 0, 0),
                # Single-M-tile BM=128, BN=128 (1 * 32 * 4 = 128 CTAs or SPLIT=2 -> 64 CTAs)
                T(128, 128, 64, 4, 0, 8, 5, 0, 0),
                T(128, 128, 64, 4, 1, 8, 5, 0, 0),
                T(128, 128, 64, 2, 0, 8, 5, 0, 0),
                T(128, 128, 64, 2, 0, 8, 5, 1, 0),
            ]
        else:
            # m=256:
            raw = [
                # 1) Single-M-tile BM=256, BN=64 (reads W (64 MB) STRICTLY ONCE across all 256 tokens!)
                #    SPLIT=2 -> 1 * 64 * 2 = 128 CTAs (1 single wave on 132 SMs + 1-load reduction!)
                T(256, 64, 64, 2, 0, 8, 5, 0, 0),
                T(256, 64, 64, 2, 0, 8, 4, 1, 0),
                T(256, 64, 64, 2, 1, 8, 5, 0, 0),
                T(256, 64, 64, 2, 1, 8, 4, 1, 0),
                T(256, 64, 64, 2, 0, 8, 4, 0, 0),
                T(256, 64, 64, 2, 1, 8, 4, 0, 0),
                #    SPLIT=1 -> 1 * 64 * 1 = 64 CTAs (reads W 1x, zero workspace reduction, TMA store!)
                T(256, 64, 64, 1, 0, 8, 5, 0, 0),
                T(256, 64, 64, 1, 0, 8, 4, 1, 0),
                T(256, 64, 64, 1, 1, 8, 5, 0, 0),
                T(256, 64, 64, 1, 1, 8, 4, 1, 0),
                # 2) Two-M-tile BM=128, BN=128, SPLIT=2 (2 * 32 * 2 = 128 CTAs in 1 wave + 1-load reduce!)
                T(128, 128, 64, 2, 0, 8, 5, 0, 0),
                T(128, 128, 64, 2, 0, 8, 5, 1, 0),
                T(128, 128, 64, 2, 1, 8, 5, 0, 0),
                T(128, 128, 64, 2, 1, 8, 5, 1, 0),
                T(128, 128, 64, 2, 0, 8, 4, 0, 0),
                T(128, 128, 64, 2, 0, 8, 5, 0, 2),
                # 3) SPLIT=1 with 128 CTAs (BM=128, BN=64 or BM=64, BN=128) + SWAP=0/1 + TMA_STORE=0/1
                T(128, 64, 64, 1, 0, 4, 8, 0, 0),
                T(128, 64, 64, 1, 0, 4, 7, 1, 0),
                T(128, 64, 64, 1, 1, 4, 8, 0, 0),
                T(128, 64, 64, 1, 1, 4, 7, 1, 0),
                T(128, 64, 64, 1, 1, 8, 7, 1, 0),
                T(64, 128, 64, 1, 0, 4, 8, 0, 0),
                T(64, 128, 64, 1, 0, 4, 7, 1, 0),
                T(64, 128, 64, 1, 1, 4, 8, 0, 0),
                T(64, 128, 64, 1, 1, 4, 7, 1, 0),
                T(128, 128, 64, 1, 0, 8, 5, 1, 0),
                T(128, 128, 64, 1, 1, 8, 5, 1, 0),
            ]

    elif layer == "unpooler":
        # N=8192, K=4096 (n_tiles: 128 for BN=64, 64 for BN=128)
        if m == 128:
            raw = [
                # Round 2 winner: (64, 128, 128, 1, 0, 4, 4) -> 19.33 us (-0.51 us vs cuBLAS)
                T(64, 128, 128, 1, 0, 4, 4, 0, 0),
                T(64, 128, 128, 1, 0, 4, 4, 1, 0),
                T(64, 128, 128, 1, 1, 4, 4, 0, 0),
                T(64, 128, 128, 1, 1, 4, 4, 1, 0),
                T(64, 128, 128, 1, 0, 8, 4, 1, 0),
                T(64, 128, 64, 1, 0, 4, 7, 1, 0),
                T(64, 128, 64, 1, 1, 4, 7, 1, 0),
                # Single-M-tile BM=128, BN=64 (1 * 128 * 1 = 128 CTAs in 1 wave, reads W (64 MB) STRICTLY ONCE!)
                T(128, 64, 128, 1, 0, 4, 4, 0, 0),
                T(128, 64, 128, 1, 0, 4, 4, 1, 0),
                T(128, 64, 128, 1, 1, 4, 4, 0, 0),
                T(128, 64, 128, 1, 1, 4, 4, 1, 0),
                T(128, 64, 128, 1, 0, 8, 4, 1, 0),
                T(128, 64, 64, 1, 0, 4, 8, 0, 0),
                T(128, 64, 64, 1, 0, 4, 7, 1, 0),
                T(128, 64, 64, 1, 1, 4, 8, 0, 0),
                T(128, 64, 64, 1, 1, 4, 7, 1, 0),
                # Single-M-tile BM=128, BN=128 (SPLIT=2 -> 128 CTAs in 1 wave, reads W 1x, 1-load reduce!)
                T(128, 128, 64, 2, 0, 8, 5, 0, 0),
                T(128, 128, 64, 2, 0, 8, 5, 1, 0),
                T(128, 128, 64, 2, 1, 8, 5, 0, 0),
                T(128, 128, 64, 1, 0, 8, 5, 1, 0),
            ]
        else:
            # m=256:
            raw = [
                # Round 2 winner: (128, 128, 64, 1, 0, 4, 6) -> 24.32 us (-0.56 us vs cuBLAS)
                T(128, 128, 64, 1, 0, 4, 6, 0, 0),
                T(128, 128, 64, 1, 0, 4, 5, 1, 0),
                T(128, 128, 64, 1, 1, 4, 6, 0, 0),
                T(128, 128, 64, 1, 1, 4, 5, 1, 0),
                T(128, 128, 64, 1, 0, 8, 5, 0, 0),
                T(128, 128, 64, 1, 0, 8, 5, 1, 0),
                T(128, 128, 64, 1, 1, 8, 5, 0, 0),
                T(128, 128, 64, 1, 1, 8, 5, 1, 0),
                T(128, 128, 128, 1, 0, 8, 3, 0, 0),
                T(128, 128, 128, 1, 0, 8, 2, 1, 0),
                # Single-M-tile BM=256, BN=64, SPLIT=1 (1 * 128 = 128 CTAs in 1 wave, reads W (64 MB) 1x, 0 reduce!)
                T(256, 64, 64, 1, 0, 8, 5, 0, 0),
                T(256, 64, 64, 1, 0, 8, 4, 1, 0),
                T(256, 64, 64, 1, 1, 8, 5, 0, 0),
                T(256, 64, 64, 1, 1, 8, 4, 1, 0),
                T(256, 64, 64, 1, 0, 8, 4, 0, 0),
                T(256, 64, 128, 1, 0, 8, 2, 0, 0),
                T(256, 64, 128, 1, 0, 8, 2, 1, 0),
                T(256, 64, 128, 1, 1, 8, 2, 0, 0),
                T(256, 64, 128, 1, 1, 8, 2, 1, 0),
                # Single-M-tile BM=256, BN=128, SPLIT=1 (64 CTAs)
                T(256, 128, 64, 1, 0, 8, 4, 0, 0),
                T(256, 128, 64, 1, 0, 8, 3, 1, 0),
                T(256, 128, 64, 1, 1, 8, 4, 0, 0),
                T(256, 128, 64, 1, 1, 8, 3, 1, 0),
            ]

    elif layer == "o":
        # N=8192, K=1024 (K is only 8x128 or 16x64 -> epilogue TMA store & 1-wave 128 CTAs are critical!)
        if m == 128:
            raw = [
                # 1) Single-M-tile BM=128, BN=64, SPLIT=1 (1 * 128 = 128 CTAs in 1 wave, W (16 MB) read 1x!)
                T(128, 64, 128, 1, 0, 4, 4, 1, 0),
                T(128, 64, 128, 1, 1, 4, 4, 1, 0),
                T(128, 64, 128, 1, 0, 8, 4, 1, 0),
                T(128, 64, 128, 1, 1, 8, 4, 1, 0),
                T(128, 64, 128, 1, 0, 4, 4, 0, 0),
                T(128, 64, 128, 1, 1, 4, 4, 0, 0),
                T(128, 64, 64, 1, 0, 4, 6, 1, 0),
                T(128, 64, 64, 1, 1, 4, 6, 1, 0),
                T(128, 64, 64, 1, 0, 8, 6, 1, 0),
                T(128, 64, 64, 1, 0, 4, 6, 0, 0),
                # 2) Two-M-tile BM=64, BN=128, SPLIT=1 (2 * 64 = 128 CTAs in 1 wave!)
                T(64, 128, 128, 1, 0, 4, 4, 1, 0),
                T(64, 128, 128, 1, 1, 4, 4, 1, 0),
                T(64, 128, 128, 1, 0, 8, 4, 1, 0),
                T(64, 128, 128, 1, 1, 8, 4, 1, 0),
                T(64, 128, 128, 1, 0, 4, 4, 0, 0),
                T(64, 128, 64, 1, 0, 4, 6, 1, 0),
                T(64, 128, 64, 1, 1, 4, 6, 1, 0),
                # 3) Single-M-tile BM=128, BN=128 (64 CTAs at SPLIT=1, or 128 CTAs at SPLIT=2)
                T(128, 128, 128, 1, 0, 8, 2, 1, 0),
                T(128, 128, 128, 1, 1, 8, 2, 1, 0),
                T(128, 128, 64, 1, 0, 8, 5, 1, 0),
                T(128, 128, 64, 1, 1, 8, 5, 1, 0),
                T(128, 128, 64, 2, 0, 8, 4, 1, 0),
                T(128, 128, 64, 2, 0, 8, 4, 0, 0),
            ]
        else:
            # m=256:
            raw = [
                # 1) Single-M-tile BM=256, BN=64, SPLIT=1 (1 * 128 = 128 CTAs in 1 wave, W (16 MB) read 1x!)
                T(256, 64, 128, 1, 0, 8, 2, 1, 0),
                T(256, 64, 128, 1, 1, 8, 2, 1, 0),
                T(256, 64, 128, 1, 0, 8, 2, 0, 0),
                T(256, 64, 128, 1, 1, 8, 2, 0, 0),
                T(256, 64, 64, 1, 0, 8, 4, 1, 0),
                T(256, 64, 64, 1, 1, 8, 4, 1, 0),
                T(256, 64, 64, 1, 0, 8, 5, 0, 0),
                T(256, 64, 64, 1, 1, 8, 5, 0, 0),
                # 2) Two-M-tile BM=128, BN=128, SPLIT=1 (2 * 64 = 128 CTAs in 1 wave!)
                T(128, 128, 128, 1, 0, 8, 2, 1, 0),
                T(128, 128, 128, 1, 1, 8, 2, 1, 0),
                T(128, 128, 128, 1, 0, 8, 3, 0, 0),
                T(128, 128, 128, 1, 0, 4, 3, 0, 0),
                T(128, 128, 64, 1, 0, 8, 5, 1, 0),
                T(128, 128, 64, 1, 1, 8, 5, 1, 0),
                T(128, 128, 64, 1, 0, 8, 5, 0, 0),
                T(128, 128, 64, 1, 0, 4, 5, 1, 0),
                # 3) Multi-wave 256 CTAs with TMA_STORE=1 and ORDER=0 vs ORDER=1 (16 MB W fits in 50 MB L2!)
                T(128, 64, 128, 1, 0, 4, 4, 1, 0),
                T(128, 64, 128, 1, 0, 4, 4, 1, 1),
                T(128, 64, 128, 1, 1, 4, 4, 1, 0),
                T(128, 64, 128, 1, 1, 4, 4, 1, 1),
                T(64, 128, 128, 1, 0, 4, 4, 1, 0),
                T(64, 128, 128, 1, 0, 4, 4, 1, 1),
                T(64, 128, 128, 1, 1, 4, 4, 1, 0),
                T(64, 128, 128, 1, 1, 4, 4, 1, 1),
            ]

    out: list[Cfg] = []
    for c in raw:
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

    print("===== Track B Round 3 Compile Report (sm_90a) =====", flush=True)
    seen: set[tuple[str, Cfg]] = set()
    total = ok_cnt = spill_cnt = fail_cnt = 0
    for layer in SHAPES:
        n, k = SHAPES[layer]
        for m in (128, 256):
            for c in candidates_for(layer, m):
                if (layer, c) in seen:
                    continue
                seen.add((layer, c))
                tag = (f"{layer:9s} BM{c.bm:3d} BN{c.bn:3d} BK{c.bk:3d} "
                       f"SP{c.split:2d} SW{c.swap} W{c.warps} S{c.stages} TMA{c.tma_store} ORD{c.order}")
                if only and only not in tag:
                    continue
                total += 1
                t0 = time.time()
                try:
                    out_sig = f"tensordesc<bf16[{c.bm}, {c.bn}]>" if c.tma_store else "*bf16"
                    sig = {
                        "x_desc": f"tensordesc<bf16[{c.bm}, {c.bk}]>",
                        "w_desc": f"tensordesc<bf16[{c.bn}, {c.bk}]>",
                        "out_ref": out_sig,
                        "ws_ptr": "*fp32",
                        "cnt_ptr": "*i32",
                        "M": "i32",
                    }
                    consts = dict(
                        N=n, K=k, SPLIT=c.split, BM=c.bm, BN=c.bn, BK=c.bk,
                        SWAP=bool(c.swap), TMA_STORE=bool(c.tma_store), ORDER=c.order,
                    )
                    names = list(_big_gemm_v3.arg_names)
                    ptr_names = ("ws_ptr", "cnt_ptr") if c.tma_store else ("out_ref", "ws_ptr", "cnt_ptr")
                    attrs = {(names.index(nm),): [["tt.divisibility", 16]] for nm in ptr_names}
                    src = ASTSource(fn=_big_gemm_v3, signature=sig, constexprs=consts, attrs=attrs)
                    opts = {"num_warps": c.warps, "num_stages": c.stages}
                    ker = triton.compile(src, target=target, options=opts)
                    ptx = ker.asm["ptx"]
                    spilled, info = _run_ptxas(ptx, ptxas_bin)
                    smem = ker.metadata.shared
                    if smem > 232448:
                        fail_cnt += 1
                        print(f"  [SMEM_OVER] {tag} ({time.time() - t0:.1f}s) smem={smem}B > 232448B | {info}",
                              flush=True)
                    elif spilled:
                        spill_cnt += 1
                        print(f"  [SPILL] {tag} ({time.time() - t0:.1f}s) smem={smem}B | {info}", flush=True)
                    else:
                        ok_cnt += 1
                        print(f"  [OK]    {tag} ({time.time() - t0:.1f}s) smem={smem}B | {info}", flush=True)
                except Exception as exc:
                    fail_cnt += 1
                    print(f"  [FAIL]  {tag}: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
    print(f"\nCompile summary: {ok_cnt}/{total} clean (0 spills), {spill_cnt} spilled, {fail_cnt} failed",
          flush=True)


# --------------------------------------------------------------------------- GPU Execution & Timing Context

class BenchCtx:
    def __init__(self, w: dict[str, torch.Tensor]) -> None:
        self.w = {k: v.contiguous() for k, v in w.items() if k in SHAPES}
        self.ws = torch.zeros(WS_FLOATS, device="cuda", dtype=torch.float32)
        self.cnt = torch.zeros(COUNTERS, device="cuda", dtype=torch.int32)
        self.triton_w_desc: dict = {}

    def get_triton_w_desc(self, layer: str, bn: int, bk: int):
        key = (layer, bn, bk)
        if key not in self.triton_w_desc:
            self.triton_w_desc[key] = TensorDescriptor.from_tensor(self.w[layer], [bn, bk])
        return self.triton_w_desc[key]

    def launch(self, layer: str, x: torch.Tensor, c: Cfg) -> torch.Tensor:
        m = x.shape[0]
        n, k = SHAPES[layer]
        grid = (triton.cdiv(m, c.bm) * (n // c.bn) * c.split,)
        out = torch.empty(m, n, device=x.device, dtype=torch.bfloat16)
        wd = self.get_triton_w_desc(layer, c.bn, c.bk)
        xd = TensorDescriptor.from_tensor(x, [c.bm, c.bk])
        out_ref = TensorDescriptor.from_tensor(out, [c.bm, c.bn]) if c.tma_store else out
        _big_gemm_v3[grid](
            xd, wd, out_ref, self.ws, self.cnt, m,
            N=n, K=k, SPLIT=c.split, BM=c.bm, BN=c.bn, BK=c.bk,
            SWAP=bool(c.swap), TMA_STORE=bool(c.tma_store), ORDER=c.order,
            num_warps=c.warps, num_stages=c.stages,
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

    from common import load_module, reference, weights
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

    for layer in ("qkv_gate", "o", "pooler", "unpooler"):
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
                smem_kb = total_smem_est(c) // 1024
                name = (f"BM{c.bm:3d} BN{c.bn:3d} BK{c.bk:3d} SP{c.split:2d} "
                        f"SW{c.swap} W{c.warps} S{c.stages} TMA{c.tma_store} ORD{c.order} "
                        f"(ctas={ctas:3d}, {smem_kb:3d}KB)")
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
            for _, _, _, c, name, fn in scored[:3]:
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

    print("\n================ TRACK B ROUND 3 SUMMARY (vs cp3_gluon_v2.py) ================", flush=True)
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
