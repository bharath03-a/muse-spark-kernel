"""Router diagnostics: streaming vs reduce tail, wide token tiles, BK 512, FP32 CUDA-core path.

    python bench/diag_router.py [--reps 100]
"""

from __future__ import annotations

import argparse

import torch
import triton
import triton.language as tl

from common import load_module, reference, weights
from harness import FLUSH_BYTES, graph_overhead_us
from micro_router import SMEM, VARIANTS, launch
from sweep_router import kernel_us, timed

LANE = VARIANTS["A lane-mod xT evict2"]
# (block_n, block_k, block_t, split, warps, stages)
BEST = {1: (32, 256, 16, 16, 4, 4), 8: (32, 256, 16, 16, 4, 4), 16: (32, 256, 16, 16, 4, 4),
        32: (32, 256, 32, 16, 4, 4), 128: (64, 256, 64, 16, 8, 3), 256: (64, 256, 64, 8, 4, 3)}
EXTRA = {
    "small": [(16, 512, 16, 8, 4, 3), (32, 512, 16, 8, 4, 3), (16, 512, 16, 16, 4, 3),
              (32, 512, 16, 16, 4, 3), (16, 256, 16, 8, 4, 4), (32, 256, 16, 8, 4, 4),
              (64, 512, 16, 8, 4, 3), (16, 256, 16, 32, 4, 4)],
    "wide": [(32, 64, 128, 16, 8, 3), (64, 64, 128, 16, 8, 3), (32, 128, 128, 16, 8, 3),
             (64, 128, 128, 16, 8, 2), (32, 64, 128, 32, 8, 3), (64, 64, 128, 32, 8, 3),
             (32, 64, 256, 16, 8, 3), (64, 64, 256, 16, 8, 3), (32, 64, 256, 32, 8, 3),
             (64, 64, 256, 32, 8, 3), (32, 128, 256, 16, 8, 2), (64, 128, 256, 32, 8, 2)],
}
FMA_TILES = [(8, 512, 16, 4, 3), (16, 256, 16, 4, 3), (16, 512, 8, 4, 3), (32, 256, 8, 4, 3),
             (8, 256, 32, 4, 3), (16, 256, 32, 4, 2), (32, 512, 8, 8, 2), (16, 1024, 8, 8, 2)]


@triton.jit(do_not_specialize=["M"])
def _fma32(x_ptr, w_ptr, out_ptr, ws_ptr, cnt_ptr, M, stride_xm,
           N: tl.constexpr, K: tl.constexpr, SPLIT: tl.constexpr, BLOCK_N: tl.constexpr,
           BLOCK_K: tl.constexpr, M_PAD: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, M_PAD)
    k_len: tl.constexpr = K // SPLIT
    acc = tl.zeros((M_PAD, BLOCK_N), dtype=tl.float32)
    for kk in range(0, k_len, BLOCK_K):
        offs_k = pid_s * k_len + kk + tl.arange(0, BLOCK_K)
        w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :], eviction_policy="evict_first")
        for t in tl.static_range(M_PAD):
            xv = tl.load(x_ptr + t * stride_xm + offs_k, mask=(offs_k < K) & (t < M), other=0.0).to(tl.float32)
            col = tl.sum(w * xv[None, :], axis=1)
            acc = tl.where(offs_m[:, None] == t, acc + col[None, :], acc)
    m_mask = offs_m < M
    ws_tile = ws_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(ws_tile + pid_s * M_PAD * N, acc, mask=m_mask[:, None])
    tl.debug_barrier()
    done = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
    if done == SPLIT - 1:
        total = tl.zeros((M_PAD, BLOCK_N), dtype=tl.float32)
        for s in tl.static_range(SPLIT):
            total += tl.load(ws_tile + s * M_PAD * N, mask=m_mask[:, None], other=0.0, cache_modifier=".cg")
        tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], total, mask=m_mask[:, None])
        tl.atomic_xchg(cnt_ptr + pid_n, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    args = ap.parse_args()
    w = weights()
    sub = load_module("submission.py", "sub")
    state = sub.prepare({k: v.clone() for k, v in w.items()})
    hilo, ws, w32 = state["router_hilo"], state["ws"], state["router"]
    flush = torch.empty(FLUSH_BYTES // 4, device="cuda", dtype=torch.float32).normal_()
    overhead = graph_overhead_us(flush)
    print(f"graph overhead {overhead:.2f} us; flush alone: [{kernel_us(lambda: None, flush)}]")
    copy_src = torch.empty(4 * 2**20, device="cuda", dtype=torch.bfloat16)
    copy_dst = torch.empty_like(copy_src)
    t, _ = timed(lambda: copy_dst.copy_(copy_src), flush, overhead, args.reps)
    print(f"8MB copy (16MB traffic) {t:.2f} us = {16 * 2**20 / t / 1e6:.2f} TB/s")
    t, _ = timed(lambda: hilo.sum(dtype=torch.float32), flush, overhead, args.reps)
    print(f"8MB read (hilo.sum) {t:.2f} us = {8 * 2**20 / t / 1e6:.2f} TB/s")

    def run_tc(m, data, tile, reduce=1):
        x = data["x"]
        cnt = torch.zeros_like(state["cnt"])

        def run():
            out = torch.empty(m, 256, device=x.device, dtype=torch.float32)
            launch(x, hilo, out, ws, cnt, tile, LANE, reduce)
            return out
        return run

    def report(label, m, data, run, check=True):
        try:
            t, out = timed(run, flush, overhead, args.reps)
        except Exception as exc:
            print(f"m={m} {label}: failed {type(exc).__name__}: {str(exc)[:100]}")
            return
        ok = reference.check_implementation(data, out, w)[0] if check else "-"
        print(f"m={m} {label}: {t:.2f} us ok={ok}")

    for m in (1, 8, 16, 32, 128, 256):
        data = reference.generate_input(m, "router", 7)
        tile = BEST[m]
        report(f"tc best {tile}", m, data, run_tc(m, data, tile))
        report(f"tc best NO-REDUCE {tile}", m, data, run_tc(m, data, tile, reduce=0), check=False)
        for extra in EXTRA["small" if m <= 32 else "wide"]:
            bn, bk, bt, split, warps, stages = extra
            if m <= 32:
                extra = (bn, bk, BEST[m][2], split, warps, stages)
                bt = extra[2]
            if stages * (bn * bk + bk * bt) * 2 > SMEM or (m > 32 and bt > 128 and m <= 128):
                continue
            report(f"tc {extra}", m, data, run_tc(m, data, extra))
        if m <= 8:
            x = data["x"]
            m_pad = triton.next_power_of_2(m)
            for bn, bk, split, warps, stages in FMA_TILES:
                cnt = torch.zeros_like(state["cnt"])

                def run(bn=bn, bk=bk, split=split, warps=warps, stages=stages, cnt=cnt):
                    out = torch.empty(m, 256, device=x.device, dtype=torch.float32)
                    _fma32[(256 // bn, split)](x, w32, out, ws, cnt, m, x.stride(0), N=256, K=8192,
                                               SPLIT=split, BLOCK_N=bn, BLOCK_K=bk, M_PAD=m_pad,
                                               num_warps=warps, num_stages=stages)
                    return out
                report(f"fma32 bn{bn} bk{bk} split{split} w{warps} s{stages}", m, data, run)


if __name__ == "__main__":
    main()
