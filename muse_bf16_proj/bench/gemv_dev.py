"""Dev copy of the m<=40 BF16 swap-AB GEMV/GEMM for qkv_gate, o, pooler, unpooler (+1024-row slice).

Three kernel kinds share one split-K epilogue (fp32 partials in ws, last CTA reduces in fixed
order, self-resetting counter):
  dot  pointer loads, acc[BN, BT] = W[BN, BK] @ x^T[BK, BT]; x loaded [BT, BK] then tl.trans
  tma  same, W through a host TMA descriptor
  dotk / tmak  same, x loaded directly as [BK, BT]
  fma  m == 1 only, acc[BN, BK] += W * x, one tl.sum at the end
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import triton
import triton.language as tl

try:
    from triton.tools.tensor_descriptor import TensorDescriptor
except ImportError:  # older Triton: tma kind unavailable
    TensorDescriptor = None

SHAPES = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192),
          "unpooler": (8192, 4096), "unpooler_slice": (1024, 4096)}
SLICE_ROWS, SLICE_MAX_M = 1024, 8
WS_FLOATS = 4 * 2**20
COUNTERS = 8192
SMEM_LIMIT = 227 * 1024


class Tile(NamedTuple):
    max_m: int
    block_n: int
    block_k: int
    block_t: int
    split: int
    warps: int
    stages: int
    kind: str = "dot"


# Tuned on H200 (sweep_gemv.py full run). m > 32 -> cuBLAS.
TILES: dict[str, tuple[Tile, ...]] = {
    "pooler": (Tile(1, 4, 1024, 1, 1, 4, 3, "fma"), Tile(8, 32, 256, 16, 1, 4, 5),
               Tile(16, 32, 256, 16, 1, 4, 5), Tile(32, 64, 128, 32, 2, 4, 5)),
    "qkv_gate": (Tile(1, 4, 1024, 1, 1, 4, 2, "fma"), Tile(8, 64, 128, 16, 4, 4, 4),
                 Tile(16, 64, 128, 16, 4, 4, 4, "tma"), Tile(32, 64, 128, 32, 4, 4, 4, "tma")),
    "o": (Tile(1, 16, 256, 1, 1, 8, 3, "fma"), Tile(8, 64, 128, 16, 1, 4, 6),
          Tile(16, 64, 128, 16, 1, 4, 5), Tile(32, 64, 128, 32, 1, 4, 5)),
    "unpooler": (Tile(16, 64, 128, 16, 1, 4, 6), Tile(32, 64, 256, 32, 1, 4, 4, "tma")),
    "unpooler_slice": (Tile(1, 4, 1024, 1, 1, 8, 2, "fma"), Tile(8, 64, 128, 16, 8, 4, 4)),
}


@triton.jit
def _finish(acc, out_ptr, ws_ptr, cnt_ptr, offs_n, offs_t, t_mask, pid_n, pid_s, pid_t,
            N: tl.constexpr, SPLIT: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_T: tl.constexpr):
    out_ptrs = out_ptr + offs_t[None, :] * N + offs_n[:, None]
    if SPLIT == 1:
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=t_mask[None, :])
    else:
        rows = tl.num_programs(2) * BLOCK_T
        ws_tile = ws_ptr + offs_t[None, :] * N + offs_n[:, None]
        tl.store(ws_tile + pid_s * rows * N, acc)
        tl.debug_barrier()
        tile = pid_t * tl.num_programs(0) + pid_n
        done = tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel", scope="gpu")
        if done == SPLIT - 1:
            total = tl.zeros((BLOCK_N, BLOCK_T), dtype=tl.float32)
            for s in range(SPLIT):
                total += tl.load(ws_tile + s * rows * N, cache_modifier=".cg")
            tl.store(out_ptrs, total.to(out_ptr.dtype.element_ty), mask=t_mask[None, :])
            tl.atomic_xchg(cnt_ptr + tile, 0)


@triton.jit(do_not_specialize=["M"])
def _gemm_dot(x_ptr, w_ptr, out_ptr, ws_ptr, cnt_ptr, M, stride_xm,
              N: tl.constexpr, K: tl.constexpr, SPLIT: tl.constexpr,
              BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_T: tl.constexpr,
              X_KMAJOR: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_t = tl.program_id(2)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = offs_t < M
    k_len: tl.constexpr = K // SPLIT
    k_start = pid_s * k_len
    acc = tl.zeros((BLOCK_N, BLOCK_T), dtype=tl.float32)
    for kk in range(0, k_len, BLOCK_K):
        offs_k = tl.max_contiguous(tl.multiple_of(k_start + kk + tl.arange(0, BLOCK_K), BLOCK_K), BLOCK_K)
        w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :], eviction_policy="evict_first")
        if X_KMAJOR:
            xt = tl.load(x_ptr + offs_t[None, :] * stride_xm + offs_k[:, None],
                         mask=t_mask[None, :], other=0.0, eviction_policy="evict_last")
        else:
            xt = tl.trans(tl.load(x_ptr + offs_t[:, None] * stride_xm + offs_k[None, :],
                                  mask=t_mask[:, None], other=0.0, eviction_policy="evict_last"))
        acc = tl.dot(w, xt, acc)
    _finish(acc, out_ptr, ws_ptr, cnt_ptr, offs_n, offs_t, t_mask, pid_n, pid_s, pid_t,
            N, SPLIT, BLOCK_N, BLOCK_T)


@triton.jit(do_not_specialize=["M"])
def _gemm_tma(x_ptr, w_desc, out_ptr, ws_ptr, cnt_ptr, M, stride_xm,
              N: tl.constexpr, K: tl.constexpr, SPLIT: tl.constexpr,
              BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_T: tl.constexpr,
              X_KMAJOR: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_t = tl.program_id(2)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = offs_t < M
    k_len: tl.constexpr = K // SPLIT
    k_start = pid_s * k_len
    acc = tl.zeros((BLOCK_N, BLOCK_T), dtype=tl.float32)
    for kk in range(0, k_len, BLOCK_K):
        offs_k = tl.max_contiguous(tl.multiple_of(k_start + kk + tl.arange(0, BLOCK_K), BLOCK_K), BLOCK_K)
        w = w_desc.load([pid_n * BLOCK_N, k_start + kk])
        if X_KMAJOR:
            xt = tl.load(x_ptr + offs_t[None, :] * stride_xm + offs_k[:, None],
                         mask=t_mask[None, :], other=0.0, eviction_policy="evict_last")
        else:
            xt = tl.trans(tl.load(x_ptr + offs_t[:, None] * stride_xm + offs_k[None, :],
                                  mask=t_mask[:, None], other=0.0, eviction_policy="evict_last"))
        acc = tl.dot(w, xt, acc)
    _finish(acc, out_ptr, ws_ptr, cnt_ptr, offs_n, offs_t, t_mask, pid_n, pid_s, pid_t,
            N, SPLIT, BLOCK_N, BLOCK_T)


@triton.jit
def _gemv_fma(x_ptr, w_ptr, out_ptr, ws_ptr, cnt_ptr,
              N: tl.constexpr, K: tl.constexpr, SPLIT: tl.constexpr,
              BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_len: tl.constexpr = K // SPLIT
    k_start = pid_s * k_len
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for kk in range(0, k_len, BLOCK_K):
        offs_k = tl.max_contiguous(tl.multiple_of(k_start + kk + tl.arange(0, BLOCK_K), BLOCK_K), BLOCK_K)
        w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :], eviction_policy="evict_first")
        xv = tl.load(x_ptr + offs_k, eviction_policy="evict_last")
        acc += w.to(tl.float32) * xv.to(tl.float32)[None, :]
    r = tl.sum(acc, axis=1)
    if SPLIT == 1:
        tl.store(out_ptr + offs_n, r.to(out_ptr.dtype.element_ty))
    else:
        tl.store(ws_ptr + pid_s * N + offs_n, r)
        tl.debug_barrier()
        done = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
        if done == SPLIT - 1:
            total = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for s in range(SPLIT):
                total += tl.load(ws_ptr + s * N + offs_n, cache_modifier=".cg")
            tl.store(out_ptr + offs_n, total.to(out_ptr.dtype.element_ty))
            tl.atomic_xchg(cnt_ptr + pid_n, 0)


def smem_bytes(t: Tile) -> int:
    if t.kind == "fma":
        return t.stages * (t.block_n * t.block_k + t.block_k) * 2
    return t.stages * (t.block_n * t.block_k + t.block_k * t.block_t) * 2


def valid(t: Tile, n: int, k: int) -> bool:
    if n % t.block_n or k % (t.split * t.block_k):
        return False
    if t.kind == "fma" and t.max_m != 1:
        return False
    if t.kind.startswith("tma") and (TensorDescriptor is None or t.block_k > 256):
        return False
    rows = triton.cdiv(t.max_m, t.block_t) * t.block_t
    if t.split > 1 and (t.split * rows * n > WS_FLOATS or (n // t.block_n) * triton.cdiv(t.max_m, t.block_t) > COUNTERS):
        return False
    return smem_bytes(t) <= SMEM_LIMIT


def _descriptor(w: torch.Tensor, t: Tile, cache: dict):
    key = (w.data_ptr(), w.shape[0], t.block_n, t.block_k)
    desc = cache.get(key)
    if desc is None:
        desc = TensorDescriptor.from_tensor(w, [t.block_n, t.block_k])
        cache[key] = desc
    return desc


def launch(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor, state: dict, t: Tile):
    m, n, k = x.shape[0], w.shape[0], w.shape[1]
    ws, cnt = state["ws"], state["cnt"]
    if t.kind == "fma":
        return _gemv_fma[(n // t.block_n, t.split)](x, w, out, ws, cnt, N=n, K=k, SPLIT=t.split,
                                                    BLOCK_N=t.block_n, BLOCK_K=t.block_k,
                                                    num_warps=t.warps, num_stages=t.stages)
    if m > t.block_t:
        raise ValueError(f"m={m} > block_t={t.block_t}: W would be re-read per token tile")
    tma = t.kind.startswith("tma")
    w_arg = _descriptor(w, t, state["desc"]) if tma else w
    fn = _gemm_tma if tma else _gemm_dot
    return fn[(n // t.block_n, t.split, 1)](x, w_arg, out, ws, cnt, m, x.stride(0), N=n, K=k,
                                            SPLIT=t.split, BLOCK_N=t.block_n, BLOCK_K=t.block_k,
                                            BLOCK_T=t.block_t, X_KMAJOR=t.kind.endswith("k"),
                                            num_warps=t.warps, num_stages=t.stages)


def layer_key(layer: str, m: int) -> str:
    return "unpooler_slice" if layer == "unpooler" and m <= SLICE_MAX_M else layer


def pick(tiles: dict, key: str, m: int) -> Tile | None:
    for t in tiles.get(key, ()):
        if m <= t.max_m:
            return t
    return None


def prepare(weights: dict, tiles: dict | None = None) -> dict:
    device = weights["pooler"].device
    state = {name: weights[name].contiguous() for name in ("qkv_gate", "o", "pooler", "unpooler")}
    state["unpooler_slice"] = state["unpooler"][:SLICE_ROWS]
    state["ws"] = torch.empty(WS_FLOATS, device=device, dtype=torch.float32)
    state["cnt"] = torch.zeros(COUNTERS, device=device, dtype=torch.int32)
    state["desc"] = {}
    state["tiles"] = tiles or TILES
    for key, ts in state["tiles"].items():
        n, k = SHAPES[key]
        for t in ts:
            if not valid(t, n, k):
                raise ValueError(f"{key} tile {t} invalid for n={n}, k={k}")
    return state


def kernel_fn(data: dict, state: dict) -> torch.Tensor:
    x, cfg = data["x"], data["config"]
    if x.stride(1) != 1:
        x = x.contiguous()
    m, key = x.shape[0], layer_key(cfg["layer"], x.shape[0])
    w = state[key]
    t = pick(state["tiles"], key, m)
    if t is None or (t.kind == "fma" and m != 1):
        return x @ w.T
    out = torch.empty(m, w.shape[0], device=x.device, dtype=torch.bfloat16)
    launch(x, w, out, state, t)
    return out
