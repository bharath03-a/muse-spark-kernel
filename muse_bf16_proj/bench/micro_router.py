"""Router kernel variants A/B: x-load orientation, k wrap, eviction hints; then a tile sweep of the winner.

    python bench/micro_router.py [--reps 100]
"""

from __future__ import annotations

import argparse
import itertools

import torch
import triton
import triton.language as tl

from common import load_module, reference, weights
from harness import FLUSH_BYTES, graph_overhead_us
from sweep_router import SMEM, kernel_us, timed


@triton.jit(do_not_specialize=["M"])
def _variant(x_ptr, w_ptr, out_ptr, ws_ptr, cnt_ptr, M, stride_xm,
             N: tl.constexpr, KX: tl.constexpr, KW: tl.constexpr, SPLIT: tl.constexpr,
             BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_T: tl.constexpr,
             X_ROWS: tl.constexpr, LANE_MOD: tl.constexpr, EVICT: tl.constexpr,
             REDUCE: tl.constexpr = 1):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_t = tl.program_id(2)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = offs_t < M
    k_len: tl.constexpr = KW // SPLIT
    k_start = pid_s * k_len
    acc = tl.zeros((BLOCK_N, BLOCK_T), dtype=tl.float32)
    for kk in range(0, k_len, BLOCK_K):
        k0 = k_start + kk
        offs_k = k0 + tl.arange(0, BLOCK_K)
        if LANE_MOD:
            offs_kx = offs_k % KX
        else:
            offs_kx = k0 % KX + tl.arange(0, BLOCK_K)
        w_ptrs = w_ptr + offs_n[:, None] * KW + offs_k[None, :]
        if EVICT >= 1:
            w = tl.load(w_ptrs, eviction_policy="evict_first")
        else:
            w = tl.load(w_ptrs)
        if X_ROWS:
            x_ptrs = x_ptr + offs_t[:, None] * stride_xm + offs_kx[None, :]
            if EVICT >= 2:
                xr = tl.load(x_ptrs, mask=t_mask[:, None], other=0.0, eviction_policy="evict_last")
            else:
                xr = tl.load(x_ptrs, mask=t_mask[:, None], other=0.0)
            acc = tl.dot(w, tl.trans(xr), acc)
        else:
            x_ptrs = x_ptr + offs_t[None, :] * stride_xm + offs_kx[:, None]
            if EVICT >= 2:
                xt = tl.load(x_ptrs, mask=t_mask[None, :], other=0.0, eviction_policy="evict_last")
            else:
                xt = tl.load(x_ptrs, mask=t_mask[None, :], other=0.0)
            acc = tl.dot(w, xt, acc)
    out_ptrs = out_ptr + offs_t[None, :] * N + offs_n[:, None]
    rows = tl.num_programs(2) * BLOCK_T
    ws_tile = ws_ptr + offs_t[None, :] * N + offs_n[:, None]
    tl.store(ws_tile + pid_s * rows * N, acc, mask=t_mask[None, :])
    if REDUCE:
        _reduce(ws_tile, out_ptrs, cnt_ptr, t_mask, rows, N, SPLIT, BLOCK_N, BLOCK_T)


@triton.jit
def _reduce(ws_tile, out_ptrs, cnt_ptr, t_mask, rows, N: tl.constexpr, SPLIT: tl.constexpr,
            BLOCK_N: tl.constexpr, BLOCK_T: tl.constexpr):
    tl.debug_barrier()
    tile = tl.program_id(2) * tl.num_programs(0) + tl.program_id(0)
    done = tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel", scope="gpu")
    if done == SPLIT - 1:
        total = tl.zeros((BLOCK_N, BLOCK_T), dtype=tl.float32)
        for s in tl.static_range(SPLIT):
            total += tl.load(ws_tile + s * rows * N, mask=t_mask[None, :], other=0.0, cache_modifier=".cg")
        tl.store(out_ptrs, total, mask=t_mask[None, :])
        tl.atomic_xchg(cnt_ptr + tile, 0)


# Best tiles from the first sweep (per-lane modulo kernel): (block_n, block_k, block_t, split, warps, stages).
SEED = {1: (32, 256, 16, 16, 4, 5), 8: (32, 256, 16, 16, 4, 5), 16: (32, 128, 16, 16, 4, 5),
        32: (32, 256, 32, 16, 4, 5), 128: (64, 256, 64, 16, 8, 3), 256: (64, 256, 64, 8, 4, 3)}
VARIANTS = {  # name: (X_ROWS, LANE_MOD, EVICT)
    "A lane-mod xT evict2": (0, 1, 2),
    "B rows+trans evict2": (1, 0, 2),
    "C rows+trans evict0": (1, 0, 0),
    "D rows+trans evict1": (1, 0, 1),
    "E lane-mod xT evict0": (0, 1, 0),
}


def launch(x, w, out, ws, cnt, tile, flags, reduce=1):
    bn, bk, bt, split, warps, stages = tile
    m = x.shape[0]
    grid = (w.shape[0] // bn, split, triton.cdiv(m, bt))
    _variant[grid](x, w, out, ws, cnt, m, x.stride(0), N=w.shape[0], KX=x.shape[1], KW=w.shape[1],
                   SPLIT=split, BLOCK_N=bn, BLOCK_K=bk, BLOCK_T=bt, X_ROWS=flags[0],
                   LANE_MOD=flags[1], EVICT=flags[2], REDUCE=reduce, num_warps=warps, num_stages=stages)


def neighbours(m, seed):
    bt0, warps0 = seed[2], seed[4]
    if m <= 32:
        grid = itertools.product((16, 32, 64), (128, 256), (16, 32), (bt0,), (4,), (3, 4, 5))
    else:
        grid = itertools.product((32, 64, 128), (64, 128, 256), (8, 16), (64, 128), {4, warps0}, (2, 3))
    for bn, bk, split, bt, warps, stages in grid:
        if 16384 % (split * bk) or stages * (bn * bk + bk * bt) * 2 > SMEM or bn < 16:
            continue
        ctas = (256 // bn) * split * -(-m // bt)
        if 96 <= ctas <= 600:
            yield bn, bk, bt, split, warps, stages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--ms", default="1,8,16,32,128,256")
    args = ap.parse_args()
    w = weights()
    sub = load_module("submission.py", "sub")
    state = sub.prepare({k: v.clone() for k, v in w.items()})
    hilo, ws = state["router_hilo"], state["ws"]
    flush = torch.empty(FLUSH_BYTES // 4, device="cuda", dtype=torch.float32).normal_()
    overhead = graph_overhead_us(flush)

    def measure(m, data, tile, flags):
        x = data["x"]
        cnt = torch.zeros_like(state["cnt"])

        def run():
            out = torch.empty(m, 256, device=x.device, dtype=torch.float32)
            launch(x, hilo, out, ws, cnt, tile, flags)
            return out

        t, out = timed(run, flush, overhead, args.reps)
        ok, msg = reference.check_implementation(data, out, w)
        return t, ok, msg, run

    for m in (int(v) for v in args.ms.split(",")):
        data = reference.generate_input(m, "router", 7)
        best = None
        for name, flags in VARIANTS.items():
            try:
                t, ok, msg, run = measure(m, data, SEED[m], flags)
            except Exception as exc:
                print(f"m={m} {name}: failed {type(exc).__name__}: {str(exc)[:100]}")
                continue
            print(f"m={m} {name}: {t:.2f} us ok={ok} [{kernel_us(run, flush)}]")
            if ok and (best is None or t < best[0]):
                best = (t, name, flags)
        if best is None:
            continue
        print(f"WINNER m={m}: {best[1]} {best[0]:.2f} us; sweeping tiles")
        results = []
        for tile in neighbours(m, SEED[m]):
            try:
                t, ok, msg, _ = measure(m, data, tile, best[2])
            except Exception:
                continue
            if ok:
                results.append((t, tile))
            else:
                print(f"  WRONG {tile}: {msg}")
        results.sort()
        for t, tile in results[:5]:
            print(f"  {t:6.2f} us  {tile}")
        if results:
            print(f"BEST m={m} {best[1]} {results[0][0]:.2f} us {results[0][1]}")


if __name__ == "__main__":
    main()
