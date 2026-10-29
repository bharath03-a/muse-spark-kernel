"""Sweep bench/cluster_dev.py tiles: correctness (normal + outliers, probe rows, router 1e-4), then
graph-replay timing with a cold L2, interleaved with the current submission kernel and cuBLAS.

    python bench/sweep_cluster.py [--reps 40] [--only router]
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

import cluster_dev as D
from common import load_module, reference, weights
from harness import FLUSH_BYTES, time_replay

OVERHEAD_US = 4.0
GEMV_CASES = [("router", 1), ("router", 8), ("router", 5), ("unpooler", 1), ("unpooler", 2), ("unpooler", 4), ("unpooler", 8),
              ("qkv_gate", 1), ("pooler", 1)]
CASES = [("router", m) for m in (1, 8, 16, 32, 128, 256)] + [("qkv_gate", m) for m in (8, 16, 32)] + [
    ("unpooler", m) for m in (1, 2, 4, 8)]
NEARBY = {1: (1,), 2: (2,), 4: (3, 4), 8: (5, 7, 8), 16: (9, 12, 16), 32: (17, 24, 31, 32),
          128: (100, 127, 128, 129), 256: (200, 255, 256, 257)}
CSIZES = (2, 4, 8, 16)
WKS = (8, 4, 2, 1)


def graph_of(fn):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    torch.cuda.synchronize()
    return g, out


def candidates(layer: str, m: int, n: int, k: int):
    p = 2 if layer == "router" else 1
    for mt, nt, pp in D.CONFIGS_ALL:
        if pp != p:
            continue
        bt = 8 * nt
        if m <= 32 and not (bt >= m and (bt < 2 * m or nt == 1)):
            continue
        if m > 32 and nt < 8:
            continue
        for c in CSIZES:
            for wk in WKS:
                tiles = [D.CTile(mt, nt, c, wk)]
                for ks in (256, 512):
                    nst = k // c // ks
                    for st in sorted({min(sv, nst) for sv in (2, 4, 6, 8)}):
                        tiles.append(D.CTile(mt, nt, c, wk, 1, ks, st))
                for t in tiles:
                    if not D.valid(t, p, n, k):
                        continue
                    ctas = n // D.block_n(t) * c * -(-m // bt)
                    if 32 <= ctas <= 1200:
                        yield t


def gemv_candidates(layer: str, m: int, n: int, k: int):
    f32 = layer == "router"
    vec = 4 if f32 else 8
    for f, r, mt, per in D.GEMV_ALL:
        if f == f32 and mt >= m and (mt < 2 * m or mt == 1) and n % r == 0 and k == per * 256 * vec:
            yield D.GTile(r, mt, per)


def gemv_sweep(ext, sub, state, w, flush, reps) -> list:
    rows = []
    for layer, m in GEMV_CASES:
        data = reference.generate_input(m, layer, 1)
        n, k = data["config"]["n"], data["config"]["k"]
        wl = w[layer]
        x = data["x"].clone()
        sides = {}
        for t in gemv_candidates(layer, m, n, k):
            bad = []
            for seed, dist in ((3, "normal"), (4, "outliers")):
                d = reference.generate_input(m, layer, seed, dist=dist)
                out = D.run_gemv(ext, d["x"], wl, n, t)
                torch.cuda.synchronize()
                ok, msg = reference.check_implementation(d, out, w)
                if not ok:
                    bad.append(f"{dist}: {msg}")
            if bad:
                print(f"FAIL gemv {layer} m={m} {t}: {bad}", flush=True)
                continue
            sides[t] = graph_of(lambda t=t: D.run_gemv(ext, x, wl, n, t))[0]
        sd = {"x": x, "config": data["config"]}
        sides["triton"] = graph_of(lambda: sub.kernel_fn(sd, state))[0]
        names = list(sides)
        times = {nm: [] for nm in names}
        for rep in range(reps):
            for nm in (names if rep % 2 else names[::-1]):
                times[nm].append(time_replay(sides[nm], flush) - OVERHEAD_US)
        med = {nm: statistics.median(v) for nm, v in times.items()}
        nbytes = n * k * wl.element_size() + m * k * 2 + m * n * wl.element_size()
        ranked = sorted((nm for nm in names if nm != "triton"), key=med.get)
        print(f"== gemv {layer} m={m} triton {med['triton']:.2f}", flush=True)
        for nm in ranked:
            print(f"   {nm}  {med[nm]:.2f}us  {nbytes / med[nm] / 1e6:.2f} TB/s", flush=True)
        if ranked:
            best = ranked[0]
            g, out = graph_of(lambda: D.run_gemv(ext, x, wl, n, best))
            g.replay()
            torch.cuda.synchronize()
            first = out.clone()
            diff = sum(int((g.replay(), not torch.equal(out, first))[1]) for _ in range(200))
            rows.append((layer, m, best, med[best], med["triton"], nbytes, diff))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--only", default="")
    ap.add_argument("--gemv-only", action="store_true")
    args = ap.parse_args()
    t_start = time.time()

    ext = D.build()
    print(f"ext ready in {time.time() - t_start:.1f}s", flush=True)
    w = weights()
    sub = load_module("submission.py", "sub")
    state = sub.prepare({kk: v.clone() for kk, v in w.items()})
    router = w["router"].float()
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    planes = {"router": torch.stack([hi, lo]).contiguous(), "qkv_gate": w["qkv_gate"].contiguous(),
              "unpooler": w["unpooler"].contiguous()}
    flush = torch.empty(FLUSH_BYTES // 4, device="cuda", dtype=torch.float32).normal_()
    print(f"setup done at {time.time() - t_start:.1f}s", flush=True)

    gemv_rows = gemv_sweep(ext, sub, state, w, flush, args.reps) if not args.only else []
    summary = []
    for layer, m in CASES if not args.gemv_only else ():
        if args.only and layer != args.only:
            continue
        data = reference.generate_input(m, layer, 1)
        n, k = data["config"]["n"], data["config"]["k"]
        wl = planes[layer]
        x = data["x"].clone()
        sides = {}
        for t in candidates(layer, m, n, k):
            try:
                bad = []
                for seed, dist in ((3, "normal"), (4, "outliers")):
                    d = reference.generate_input(m, layer, seed, dist=dist)
                    out = D.run(ext, d["x"], wl, n, t)
                    torch.cuda.synchronize()
                    ok, msg = reference.check_implementation(d, out, w)
                    if not ok:
                        bad.append(f"{dist}: {msg}")
                if bad:
                    print(f"FAIL {layer} m={m} {t}: {bad}", flush=True)
                    continue
                sides[t] = graph_of(lambda t=t: D.run(ext, x, wl, n, t))[0]
            except Exception as exc:
                print(f"ERR {layer} m={m} {t}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
                torch.cuda.synchronize()
        sd = {"x": x, "config": data["config"]}
        sides["triton"] = graph_of(lambda: sub.kernel_fn(sd, state))[0]
        if layer == "router":
            def cublas():
                acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
                acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
                return acc
        else:
            wv = w[layer][:n]

            def cublas():
                return x @ wv.T
        sides["cublas"] = graph_of(cublas)[0]

        names = list(sides)
        times = {nm: [] for nm in names}
        for rep in range(args.reps):
            order = names[rep % len(names):] + names[: rep % len(names)]
            if rep % 2:
                order = order[::-1]
            for nm in order:
                times[nm].append(time_replay(sides[nm], flush) - OVERHEAD_US)
        med = {nm: statistics.median(v) for nm, v in times.items()}
        wbytes = 4 if layer == "router" else 2
        nbytes = n * k * wbytes + m * k * 2 + m * n * wbytes
        sol = reference.speed_of_light_ns(m, layer, {"hbm_bytes_per_s": 4.8e12,
                                                     "bf16_dense_flops_per_s": 989e12}) / 1e3
        ranked = sorted((nm for nm in names if nm not in ("triton", "cublas")), key=med.get)
        print(f"\n== {layer} m={m} n={n} k={k} sol {sol:.2f}us | triton {med['triton']:.2f} | "
              f"cublas {med['cublas']:.2f}", flush=True)
        for nm in ranked[:10]:
            print(f"   {nm}  {med[nm]:.2f}us  {nbytes / med[nm] / 1e6:.2f} TB/s", flush=True)
        if ranked:
            best = ranked[0]
            fails = []
            for mm in NEARBY[m]:
                if mm > 32 and best.nt < 8 or mm <= 32 and 8 * best.nt < mm:
                    continue
                for seed, dist in ((5, "normal"), (6, "outliers")):
                    d = reference.generate_input(mm, layer, seed, dist=dist)
                    nn = d["config"]["n"]
                    out = D.run(ext, d["x"], wl, nn, best)
                    torch.cuda.synchronize()
                    ok, msg = reference.check_implementation(d, out, w)
                    if not ok:
                        fails.append((mm, dist, msg))
            g, out = graph_of(lambda: D.run(ext, x, wl, n, best))
            g.replay()
            torch.cuda.synchronize()
            first = out.clone()
            diff = 0
            for _ in range(200):
                g.replay()
                diff += int(not torch.equal(out, first))
            torch.cuda.synchronize()
            print(f"   best {best}: nearby fails {fails or 'none'}; determinism {diff}/200 differ", flush=True)
            diag = {md: graph_of(lambda md=md: D.run(ext, x, wl, n, best, md))[0] for md in (0, 1, 2)}
            dt = {md: [] for md in diag}
            for rep in range(args.reps):
                for md in (diag if rep % 2 else list(diag)[::-1]):
                    dt[md].append(time_replay(diag[md], flush) - OVERHEAD_US)
            print("   diag full {:.2f} | no-reduce {:.2f} | no-stream {:.2f}".format(
                *(statistics.median(dt[md]) for md in (0, 1, 2))), flush=True)
            summary.append((layer, m, best, med[best], med["triton"], med["cublas"], nbytes, sol))
        print(f"   [t={time.time() - t_start:.0f}s]", flush=True)

    print(f"\n{'layer':9} {'m':>4} {'best':28} {'cluster':>8} {'triton':>8} {'cublas':>8} {'TB/s':>6} {'sol':>6}")
    for layer, m, best, a, b, c, nbytes, sol in summary:
        print(f"{layer:9} {m:4d} {str(tuple(best)):28} {a:8.2f} {b:8.2f} {c:8.2f} {nbytes / a / 1e6:6.2f} {sol:6.2f}")
    for layer, m, best, a, b, nbytes, diff in gemv_rows:
        print(f"gemv {layer:9} {m:4d} {str(tuple(best)):16} {a:8.2f} triton {b:8.2f} {nbytes / a / 1e6:6.2f} TB/s"
              f" determinism {diff}/200")
    print(f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
