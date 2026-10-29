"""BF16 tiles in submission.py vs flashinfer tinygemm_bf16 vs cuBLAS at m=1/8, plus PTX/SASS load widths.

    modal run modal_app.py::session --cmds "tinygemm_ab"
"""

from __future__ import annotations

import inspect
import re
import statistics

import torch

from common import load_module, reference, weights

FLUSH_BYTES = 256 * 2**20
OVERHEAD_US = 4.0
REPS = 100
CASES = [("qkv_gate", 1), ("qkv_gate", 8), ("pooler", 1), ("pooler", 8), ("unpooler", 1), ("unpooler", 8),
         ("o", 1), ("o", 8)]


def capture(fn):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    torch.cuda.synchronize()
    return g, out


def timed(g, flush, sink) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(REPS):
        torch.sum(flush, dim=0, out=sink)
        start.record()
        g.replay()
        end.record()
        end.synchronize()
        ts.append(start.elapsed_time(end) * 1e3 - OVERHEAD_US)
    return statistics.median(ts)


def tinygemm_fn():
    try:
        import flashinfer.gemm as fg
        fn = fg.tinygemm_bf16
    except Exception as exc:
        print("tinygemm_bf16 unavailable:", exc)
        return None
    try:
        print("tinygemm_bf16 signature:", inspect.signature(fn))
    except (TypeError, ValueError):
        pass
    return fn


def call_tiny(fn, x, w):
    n = w.shape[0]
    out = torch.empty(x.shape[0], n, device=x.device, dtype=torch.bfloat16)
    for args in ((x, w), (x, w, out), (x, w, None), (x, w, None, out)):
        try:
            r = fn(*args)
            return r if isinstance(r, torch.Tensor) else out
        except TypeError:
            continue
    raise TypeError("no tinygemm_bf16 call form matched")


def load_widths(sub, state) -> None:
    pats = {"ld.global.v4": r"ld\.global[^;]*\.v4", "ld.global(other)": r"ld\.global(?![^;]*\.v4)[^;]*;",
            "cp.async 16B": r"cp\.async\.c[ag]\.shared\.global[^;]*, 16", "cp.async.bulk.tensor": r"cp\.async\.bulk\.tensor"}
    sass_pats = {"LDG.128": r"LDG\.E[\w.]*\.128", "LDGSTS.128": r"LDGSTS[\w.]*\.128", "UTMALDG": r"UTMALDG",
                 "LDG(other)": r"LDG\.E(?![\w.]*\.128)"}
    print("load instructions per BF16 tile:")
    for key, tiles in sub.BF16_TILES.items():
        w = state[key]
        n, k = w.shape
        for t in tiles:
            x = torch.zeros(t.max_m, k, device="cuda", dtype=torch.bfloat16)
            out = torch.empty(t.max_m, n, device="cuda", dtype=torch.bfloat16)
            if t.kind == "fma":
                ck = sub._bf16_gemv[(n // t.block_n, t.split)](
                    x, w, out, state["ws"], state["cnt"], N=n, K=k, SPLIT=t.split, BLOCK_N=t.block_n,
                    BLOCK_K=t.block_k, num_warps=t.warps, num_stages=t.stages)
            else:
                desc = state["desc"][(key, t)] if t.kind == "tma" else w
                ck = sub._bf16_gemm[(n // t.block_n, t.split)](
                    x, w, desc, out, state["ws"], state["cnt"], t.max_m, x.stride(0), N=n, K=k,
                    SPLIT=t.split, BLOCK_N=t.block_n, BLOCK_K=t.block_k, BLOCK_T=t.block_t,
                    USE_TMA=t.kind == "tma", num_warps=t.warps, num_stages=t.stages)
            ptx = ck.asm.get("ptx", "")
            try:
                sass = ck.asm.get("sass", "") or ""
            except Exception as exc:
                sass = ""
                print("  sass unavailable:", exc)
            p = {name: len(re.findall(pat, ptx)) for name, pat in pats.items()}
            s = {name: len(re.findall(pat, sass)) for name, pat in sass_pats.items()} if sass else "n/a"
            print(f"  {key:15} m<={t.max_m:2d} {t.kind:3} BN{t.block_n} BK{t.block_k} grid ({n // t.block_n},{t.split},1)"
                  f" ptx {p} sass {s}", flush=True)


def main() -> None:
    w = weights()
    sub, base = load_module("submission.py", "sub"), load_module("baseline.py", "base")
    state = sub.prepare({k: v.clone() for k, v in w.items()})
    bstate = base.prepare({k: v.clone() for k, v in w.items()})
    flush = torch.empty(FLUSH_BYTES // 4, device="cuda").normal_()
    sink = torch.empty((), device="cuda")
    tiny = tinygemm_fn()
    print(f"{'shape':16} {'mine':>7} {'tiny':>7} {'cublas':>7}  (us, {OVERHEAD_US} subtracted)")
    for layer, m in CASES:
        data = reference.generate_input(m, layer, 1)
        x, n = data["x"], data["config"]["n"]
        wl = w[layer][:n]
        ref = x.float() @ wl.float().T
        g_mine, _ = capture(lambda: sub.kernel_fn(data, state))
        g_base, _ = capture(lambda: base.kernel_fn(data, bstate))
        t_tiny, err = float("nan"), float("nan")
        if tiny is not None:
            try:
                wc = wl.contiguous()
                g_tiny, o = capture(lambda: call_tiny(tiny, x, wc))
                err = float((o.float() - ref).norm() / ref.norm())
                t_tiny = timed(g_tiny, flush, sink)
            except Exception as exc:
                print(f"  tinygemm failed on {layer} m={m}: {type(exc).__name__}: {str(exc)[:200]}")
        t_mine, t_base = timed(g_mine, flush, sink), timed(g_base, flush, sink)
        print(f"{layer + ' m' + str(m):16} {t_mine:7.2f} {t_tiny:7.2f} {t_base:7.2f}  tiny err {err:.1e}", flush=True)
    load_widths(sub, state)


if __name__ == "__main__":
    main()
