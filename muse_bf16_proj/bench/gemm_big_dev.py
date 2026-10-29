"""Triton TMA/wgmma GEMM for the BF16 layers at m = 97..320 (verify 16/32), optional in-kernel split-K."""

from __future__ import annotations

from typing import NamedTuple

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

SHAPES = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096)}
WS_FLOATS = 16 * 2**20
COUNTERS = 8192
SMEM_LIMIT = 227 * 1024
SMEM_BUDGET = 200 * 1024


class Cfg(NamedTuple):
    bm: int      # token tile
    bn: int      # weight-row tile
    bk: int
    split: int
    warps: int
    stages: int
    swap: int = 0  # 1: acc = W x^T ([bn, bm]), token tile is the wgmma N
    wspec: int = 0  # tl.range warp_specialize


@triton.jit(do_not_specialize=["M"])
def _gemm_big(x_desc, w_desc, out_ptr, ws_ptr, cnt_ptr, M,
              N: tl.constexpr, K: tl.constexpr, SPLIT: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SWAP: tl.constexpr, WSPEC: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    pid_m = pid % num_m
    rest = pid // num_m
    pid_s = rest % SPLIT
    pid_n = rest // SPLIT
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
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    if SWAP:
        idx = offs_m[None, :] * N + offs_n[:, None]
        mask = (offs_m < M)[None, :]
    else:
        idx = offs_m[:, None] * N + offs_n[None, :]
        mask = (offs_m < M)[:, None]
    if SPLIT == 1:
        tl.store(out_ptr + idx, acc.to(out_ptr.dtype.element_ty), mask=mask)
    else:
        rows = num_m * BM
        tl.store(ws_ptr + pid_s * rows * N + idx, acc)
        tl.debug_barrier()
        cnt = cnt_ptr + pid_n * num_m + pid_m
        done = tl.atomic_add(cnt, 1, sem="acq_rel", scope="gpu")
        if done == SPLIT - 1:
            total = tl.zeros_like(acc)
            for s in tl.static_range(SPLIT):
                total += tl.load(ws_ptr + s * rows * N + idx, cache_modifier=".cg")
            tl.store(out_ptr + idx, total.to(out_ptr.dtype.element_ty), mask=mask)
            tl.atomic_xchg(cnt, 0)


def smem_bytes(c: Cfg) -> int:
    return (c.bm + c.bn) * c.bk * 2 * c.stages


def valid(c: Cfg, layer: str, m: int) -> bool:
    n, k = SHAPES[layer]
    num_m = triton.cdiv(m, c.bm)
    return (n % c.bn == 0 and k % c.bk == 0 and k // c.bk >= c.split and smem_bytes(c) <= SMEM_LIMIT
            and c.split * num_m * c.bm * n <= WS_FLOATS and num_m * (n // c.bn) <= COUNTERS)


def ctas(c: Cfg, layer: str, m: int) -> int:
    return triton.cdiv(m, c.bm) * (SHAPES[layer][0] // c.bn) * c.split


def w_desc(w: torch.Tensor, c: Cfg) -> TensorDescriptor:
    return TensorDescriptor.from_tensor(w, [c.bn, c.bk])


def launch(x: torch.Tensor, w: torch.Tensor, wd: TensorDescriptor, out: torch.Tensor,
           ws: torch.Tensor, cnt: torch.Tensor, c: Cfg) -> None:
    m = x.shape[0]
    n, k = w.shape
    xd = TensorDescriptor.from_tensor(x, [c.bm, c.bk])
    grid = (triton.cdiv(m, c.bm) * (n // c.bn) * c.split,)
    _gemm_big[grid](xd, wd, out, ws, cnt, m, N=n, K=k, SPLIT=c.split, BM=c.bm, BN=c.bn, BK=c.bk,
                    SWAP=c.swap, WSPEC=c.wspec, num_warps=c.warps, num_stages=c.stages)


def fit(c: Cfg) -> Cfg:
    s = c.stages
    while s > 2 and smem_bytes(c._replace(stages=s)) > SMEM_BUDGET:
        s -= 1
    return c._replace(stages=s)


def candidates(layer: str, m: int) -> list[Cfg]:
    """One-wave grids (<= 2 CTAs/SM) around the plan's starting points."""
    base: list[tuple] = []
    if layer == "qkv_gate":
        base = [(128, 128, 64, 7, 4), (128, 256, 64, 14, 8), (128, 64, 64, 3, 4), (64, 128, 64, 3, 4),
                (128, 128, 64, 4, 4), (128, 64, 64, 4, 4)] if m <= 192 else \
               [(128, 128, 64, 3, 4), (128, 256, 64, 7, 8), (256, 128, 64, 7, 8), (128, 128, 64, 4, 8),
                (128, 64, 64, 2, 4)]
    elif layer == "o":
        base = [(128, 64, 64, 1, 4), (128, 128, 64, 2, 4), (64, 128, 64, 1, 4), (128, 128, 64, 1, 4)] \
            if m <= 192 else [(128, 128, 64, 1, 4), (128, 128, 64, 1, 8), (256, 64, 64, 1, 8),
                              (128, 256, 64, 1, 8), (128, 64, 64, 1, 4)]
    elif layer == "pooler":
        base = [(128, 128, 64, 4, 4), (128, 256, 64, 8, 8), (128, 64, 64, 2, 4), (64, 128, 64, 2, 4),
                (128, 128, 64, 2, 4), (128, 32, 64, 1, 4)] if m <= 192 else \
               [(128, 128, 64, 2, 4), (128, 128, 64, 2, 8), (128, 256, 64, 4, 8), (256, 128, 64, 4, 8),
                (128, 64, 64, 1, 4), (128, 256, 64, 2, 8)]
    elif layer == "unpooler":
        base = [(128, 64, 64, 1, 4), (128, 128, 64, 2, 4), (128, 256, 64, 4, 8), (64, 128, 64, 1, 4)] \
            if m <= 192 else [(128, 128, 64, 1, 4), (128, 128, 64, 1, 8), (256, 128, 64, 2, 8),
                              (128, 256, 64, 2, 8), (256, 64, 64, 1, 8)]
    out = []
    for bm, bn, bk, split, warps in base:
        for stages in (4, 6):
            c = fit(Cfg(bm, bn, bk, split, warps, stages))
            if valid(c, layer, m) and c not in out:
                out.append(c)
        if bm == 64 or bn <= 64:
            c = fit(Cfg(bm, bn, bk, split, warps, 5, swap=1))
            if valid(c, layer, m) and c not in out:
                out.append(c)
    return out


def _r3() -> dict:
    C = Cfg
    s = {
        ("pooler", 128): [C(128, 64, 64, 2, 4, s) for s in (6, 7, 8)] + [C(64, 128, 64, 2, 4, s) for s in (7, 8)]
        + [C(128, 64, 128, 2, 4, s) for s in (3, 4)] + [C(64, 64, 64, 1, 4, s) for s in (6, 8, 10)]
        + [C(128, 64, 64, 2, 4, 6, 0, 1), C(128, 128, 64, 4, 8, 6)],
        ("unpooler", 128): [C(64, 128, 64, 1, 4, s) for s in (7, 8)] + [C(128, 64, 64, 1, 4, s) for s in (7, 8)]
        + [C(64, 64, 64, 1, 4, s) for s in (6, 8, 10)] + [C(128, 64, 128, 1, 4, 4), C(64, 128, 64, 1, 4, 6, 0, 1)],
        ("qkv_gate", 128): [C(128, 64, 64, 3, 4, s) for s in (7, 8)] + [C(64, 128, 64, 3, 4, s) for s in (7, 8)]
        + [C(64, 64, 64, 2, 4, s) for s in (6, 8)] + [C(64, 64, 64, 1, 4, 10), C(128, 64, 128, 3, 4, 4),
                                                       C(128, 64, 64, 3, 4, 6, 0, 1)],
        ("o", 128): [C(128, 64, 64, 1, 4, s) for s in (7, 8)] + [C(64, 64, 64, 1, 4, s) for s in (4, 6, 8)]
        + [C(64, 128, 64, 1, 4, 8), C(128, 64, 64, 1, 4, 6, 0, 1)],
        ("qkv_gate", 256): [C(128, 64, 64, 1, 4, s) for s in (6, 8)] + [C(64, 128, 64, 1, 4, s) for s in (6, 8)]
        + [C(64, 64, 64, 1, 4, s) for s in (5, 8)] + [C(128, 128, 64, 3, 4, 6, 0, 1), C(128, 128, 64, 3, 8, 6)],
        ("o", 256): [C(128, 64, 64, 1, 4, s) for s in (3, 5)] + [C(128, 128, 64, 1, 4, s) for s in (5, 6)]
        + [C(64, 128, 64, 1, 4, 6), C(128, 128, 64, 1, 4, 6, 0, 1), C(128, 64, 64, 1, 4, 4, 0, 1)],
        ("pooler", 256): [C(128, 128, 64, 2, 4, 6, 0, 1), C(128, 128, 64, 2, 8, 6, 0, 1), C(128, 128, 128, 2, 4, 3),
                          C(128, 128, 32, 2, 4, 12), C(128, 64, 64, 1, 4, 8), C(128, 256, 64, 4, 8, 4, 0, 1),
                          C(128, 128, 64, 2, 4, 6)],
        ("unpooler", 256): [C(128, 128, 64, 1, 4, 6, 0, 1), C(128, 128, 64, 1, 8, 6, 0, 1), C(128, 128, 32, 1, 4, 12),
                            C(128, 128, 128, 1, 4, 3), C(128, 128, 64, 1, 4, 6), C(256, 64, 64, 1, 8, 5, 1)],
    }
    return s


SETS = {"r3": _r3}


def _r4() -> dict:
    C = Cfg
    return {
        ("pooler", 128): [C(64, 128, 64, 2, 4, 7), C(64, 128, 64, 2, 8, 7), C(64, 256, 64, 4, 8, 4),
                          C(64, 64, 64, 2, 4, 7), C(64, 128, 128, 2, 4, 4), C(128, 64, 64, 2, 4, 6)],
        ("pooler", 256): [C(128, 64, 64, 1, 4, s) for s in (7, 8, 9)]
        + [C(128, 64, 64, 1, 8, 8), C(128, 64, 128, 1, 4, 4), C(256, 64, 64, 2, 8, 5)],
        ("qkv_gate", 128): [C(128, 64, 64, 3, 4, 7), C(128, 64, 64, 3, 8, 7), C(128, 64, 128, 3, 4, 4),
                            C(64, 64, 64, 3, 4, 7), C(128, 64, 64, 3, 4, 8)],
        ("unpooler", 128): [C(64, 128, 64, 1, 4, 7), C(64, 128, 64, 1, 8, 7), C(64, 256, 64, 2, 8, 4)],
    }


SETS["r4"] = _r4
