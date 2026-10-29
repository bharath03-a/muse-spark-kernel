"""m = 128/256 BF16 layers: cuBLAS variants, DeepGEMM/FlashInfer probes, Triton TMA GEMM sweep.

    modal run modal_app.py::session --cmds "sweep_gemm_big --mode full"

Every candidate is timed paired against baseline.kernel_fn (CUDA-graph replay, 256 MB read-flush,
4.0 us subtracted); top 3 per shape re-timed; winners checked with reference.check_implementation.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

import gemm_big_dev as G
from common import load_module, reference

HERE = Path(__file__).resolve().parent
FLUSH_BYTES = 256 * 2**20
OVERHEAD_US = 4.0
HW = {"hbm_bytes_per_s": 4.8e12, "bf16_dense_flops_per_s": 989e12}
CASES = [(layer, m) for m in (128, 256) for layer in ("qkv_gate", "o", "pooler", "unpooler")]
NEARBY = (100, 127, 129, 160, 200, 255, 257, 300)
BUCKET_SPLIT = 192  # m <= 192 -> the m=128 winner, else the m=256 winner


class Timer:
    def __init__(self) -> None:
        self.flush = torch.empty(FLUSH_BYTES // 4, device="cuda").normal_()
        self.sink = torch.empty((), device="cuda")
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def once(self, graph) -> float:
        torch.sum(self.flush, dim=0, out=self.sink)
        self.start.record()
        graph.replay()
        self.end.record()
        self.end.synchronize()
        return max(self.start.elapsed_time(self.end) * 1e3 - OVERHEAD_US, 1e-3)


def capture(fn):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    g.replay()
    torch.cuda.synchronize()
    return g, out


def paired(timer: Timer, g, base_g, reps: int) -> tuple[float, float, float]:
    for _ in range(3):
        g.replay()
        base_g.replay()
    ratios, base = [], []
    for r in range(reps):
        order = (g, base_g) if r % 2 == 0 else (base_g, g)
        ts = {id(x): timer.once(x) for x in order}
        ratios.append(ts[id(base_g)] / ts[id(g)])
        base.append(ts[id(base_g)])
    sp = statistics.median(ratios)
    b = statistics.median(base)
    return b / sp, b, sp


class Ctx:
    def __init__(self, w: dict) -> None:
        self.w = {k: v.contiguous() for k, v in w.items()}
        self.ws = torch.zeros(G.WS_FLOATS, device="cuda")
        self.cnt = torch.zeros(G.COUNTERS, device="cuda", dtype=torch.int32)
        self.desc: dict = {}

    def triton_fn(self, layer: str, c: G.Cfg, x: torch.Tensor):
        w = self.w[layer]
        key = (layer, c.bn, c.bk)
        if key not in self.desc:
            self.desc[key] = G.w_desc(w, c)
        wd = self.desc[key]

        def fn():
            out = torch.empty(x.shape[0], w.shape[0], device="cuda", dtype=torch.bfloat16)
            G.launch(x, w, wd, out, self.ws, self.cnt, c)
            return out
        return fn


def alt_fns(layer: str, x: torch.Tensor, w: torch.Tensor, extras: dict) -> dict:
    n = w.shape[0]
    zero_bias = torch.zeros(n, device="cuda", dtype=torch.bfloat16)
    fns = {
        "mm": lambda: x @ w.T,
        "swapT": lambda: torch.mm(w, x.t()).t().contiguous(),
        "addmm_bias0": lambda: torch.addmm(zero_bias, x, w.t()),
    }

    def swap_out():
        out = torch.empty(x.shape[0], n, device="cuda", dtype=torch.bfloat16)
        torch.mm(w, x.t(), out=out.t())
        return out
    fns["swap_out"] = swap_out

    def lt():
        prev = torch.backends.cuda.preferred_blas_library()
        torch.backends.cuda.preferred_blas_library("cublaslt")
        try:
            return x @ w.T
        finally:
            torch.backends.cuda.preferred_blas_library(prev)
    fns["cublaslt"] = lt

    def nored():
        prev = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = not prev
        try:
            return x @ w.T
        finally:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = prev
    fns["flip_bf16_red"] = nored

    dg = extras.get("deep_gemm")
    if dg is not None:
        def dgf():
            out = torch.empty(x.shape[0], n, device="cuda", dtype=torch.bfloat16)
            dg(x, w, out)
            return out
        fns["deep_gemm"] = dgf
    fi = extras.get("fi_mm")
    if fi is not None:
        fns["flashinfer"] = lambda: fi(x, w.t())
    return fns


def probe_libs() -> dict:
    extras = {}
    try:
        import deep_gemm
        names = [a for a in dir(deep_gemm) if "gemm" in a.lower() or "bf16" in a.lower()]
        print("deep_gemm", getattr(deep_gemm, "__version__", "?"), names, flush=True)
        for name in ("bf16_gemm_nt", "gemm_bf16_nt"):
            if hasattr(deep_gemm, name):
                extras["deep_gemm"] = getattr(deep_gemm, name)
                print("  using deep_gemm." + name, inspect.signature(getattr(deep_gemm, name)), flush=True)
                break
    except Exception as exc:
        print("deep_gemm unavailable:", type(exc).__name__, exc, flush=True)
    try:
        import flashinfer.gemm as fg
        names = [a for a in dir(fg) if "bf16" in a.lower()]
        print("flashinfer.gemm bf16 names", names, flush=True)
        if hasattr(fg, "mm_bf16"):
            extras["fi_mm"] = fg.mm_bf16
            print("  mm_bf16", inspect.signature(fg.mm_bf16), flush=True)
    except Exception as exc:
        print("flashinfer unavailable:", type(exc).__name__, exc, flush=True)
    return extras


def worker(path: str) -> None:
    items = json.loads(Path(path).read_text())
    ctx = Ctx({layer: torch.zeros(n, k, device="cuda", dtype=torch.bfloat16)
               for layer, (n, k) in G.SHAPES.items()})
    failed = []
    for layer, m, cfg in items:
        c = G.Cfg(*cfg)
        x = torch.zeros(m, G.SHAPES[layer][1], device="cuda", dtype=torch.bfloat16)
        try:
            ctx.triton_fn(layer, c, x)()
            torch.cuda.synchronize()
        except Exception as exc:
            failed.append([layer, m, cfg])
            print(f"compile fail {layer} {c}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
    Path(path + ".fail").write_text(json.dumps(failed))


def precompile(items: list, workers: int) -> set:
    t0 = time.time()
    tmp = Path("/tmp/sweep_gemm_big")
    tmp.mkdir(exist_ok=True)
    env = {**os.environ, "PYTHONPATH": f"{HERE.parent}:{HERE}"}
    procs = []
    for i in range(workers):
        chunk = items[i::workers]
        if chunk:
            p = tmp / f"chunk{i}.json"
            p.write_text(json.dumps(chunk))
            procs.append(subprocess.Popen([sys.executable, str(HERE / "sweep_gemm_big.py"), "--worker", str(p)],
                                          env=env))
    for p in procs:
        p.wait()
    bad = set()
    for f in tmp.glob("chunk*.json.fail"):
        bad |= {(layer, tuple(cfg)) for layer, _, cfg in json.loads(f.read_text())}
        f.unlink()
    print(f"precompiled {len(items)} ({len(bad)} failed) on {len(procs)} workers in {time.time() - t0:.0f}s",
          flush=True)
    return bad


def rel_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    return float((out.float() - ref).norm() / ref.norm())


def metric(layer: str, m: int, us: float) -> str:
    n, k = G.SHAPES[layer]
    tb = (n * k + m * k + m * n) * 2 / us / 1e6
    tf = 2 * m * n * k / us / 1e6
    return f"{tb:.2f}TB/s {tf:.0f}TF/s"


def check_winners(ctx: Ctx, w: dict, winners: dict) -> None:
    total = fails = 0
    for layer in G.SHAPES:
        for m in (128, 256) + NEARBY:
            c = winners.get((layer, 128 if m <= BUCKET_SPLIT else 256))
            if c is None:
                continue
            for dist, seed in (("normal", 3), ("outliers", 4)):
                data = reference.generate_input(m, layer, seed, dist=dist)
                out = ctx.triton_fn(layer, c, data["x"])()
                ok, msg = reference.check_implementation(data, out, w)
                total += 1
                if not ok:
                    fails += 1
                    print(f"FAIL {layer} m={m} {dist} {c}: {msg}", flush=True)
    print(f"correctness: {total - fails}/{total} passed", flush=True)


def nearby_timing(ctx: Ctx, timer: Timer, base, base_state, winners: dict, reps: int) -> None:
    keep = []
    for (layer, mb), c in winners.items():
        for m in (x for x in NEARBY if (x <= BUCKET_SPLIT) == (mb == 128)):
            data = reference.generate_input(m, layer, 1)
            print(f"  nearby {layer} m={m} base capture", flush=True)
            bg, _ = capture(lambda: base.kernel_fn(data, base_state))
            print(f"  nearby {layer} m={m} triton capture", flush=True)
            g, _ = capture(ctx.triton_fn(layer, c, data["x"]))
            keep.append((data, bg, g))
            us, b, sp = paired(timer, g, bg, reps)
            print(f"  nearby {layer:9} m={m:3d} triton {us:6.2f} cublas {b:6.2f} x{sp:.3f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default="")
    ap.add_argument("--mode", default="full", choices=("full", "final"))
    ap.add_argument("--cfgs", default="", help="final: json {layer:m: [cfg,...]}")
    ap.add_argument("--set", default="")
    ap.add_argument("--no-nearby", action="store_true")
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--final-reps", type=int, default=300)
    ap.add_argument("--no-alts", action="store_true")
    args = ap.parse_args()
    if args.worker:
        worker(args.worker)
        return
    t0 = time.time()
    import triton
    print(f"triton {triton.__version__} torch {torch.__version__} blas {torch.backends.cuda.preferred_blas_library()}"
          f" bf16_red {torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}", flush=True)
    w = reference.generate_weights(reference.WEIGHTS_SEED)
    base = load_module("baseline.py", "base")
    base_state = base.prepare({k: v.clone() for k, v in w.items()})
    try:
        src = inspect.getsource(base._bf16_gemm_dispatch_impl)
        print("dispatch impl:\n" + "\n".join(src.splitlines()[:40]), flush=True)
    except Exception as exc:
        print("no dispatch source:", exc)
    ctx = Ctx(w)
    timer = Timer()

    if args.set:
        cands = G.SETS[args.set]()
    elif args.mode == "full":
        cands = {(layer, m): G.candidates(layer, m) for layer, m in CASES}
    else:
        spec = json.loads(args.cfgs)
        cands = {(k.split(":")[0], int(k.split(":")[1])): [G.Cfg(*c) for c in v] for k, v in spec.items()}
    items, seen = [], set()
    for (layer, m), cs in cands.items():
        for c in cs:
            if (layer, c) not in seen:
                seen.add((layer, c))
                items.append([layer, m, list(c)])
    ncpu = len(os.sched_getaffinity(0))
    bad = precompile(items, max(1, min(16, ncpu)))
    extras = {} if args.no_alts else probe_libs()

    winners, report = {}, {}
    for layer, m in CASES:
        if (layer, m) not in cands:
            continue
        data = reference.generate_input(m, layer, 1)
        x, wl = data["x"], ctx.w[layer]
        ref = x.float() @ wl.float().T
        floor = reference.speed_of_light_ns(m, layer, HW) / 1e3
        base_g, _ = capture(lambda: base.kernel_fn(data, base_state))
        res = []
        if not args.no_alts:
            for name, fn in alt_fns(layer, x, wl, extras).items():
                try:
                    g, out = capture(fn)
                    e = rel_err(out, ref)
                    us, b, sp = paired(timer, g, base_g, args.reps)
                    print(f"{layer}:m{m} alt {name:14} {us:6.2f} cublas {b:6.2f} x{sp:.3f} err {e:.1e}", flush=True)
                except Exception as exc:
                    print(f"{layer}:m{m} alt {name} failed: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
        for c in cands[(layer, m)]:
            if (layer, tuple(c)) in bad:
                continue
            try:
                g, out = capture(ctx.triton_fn(layer, c, x))
                e = rel_err(out, ref)
                if e > 5e-3:
                    print(f"  WRONG {layer}:m{m} {c} err {e:.2e}", flush=True)
                    continue
                us, b, sp = paired(timer, g, base_g, args.reps)
                res.append((us, c))
                print(f"{layer}:m{m} {str(tuple(c)):34} ctas {G.ctas(c, layer, m):4d} {us:6.2f} cublas {b:6.2f}"
                      f" x{sp:.3f}", flush=True)
            except Exception as exc:
                print(f"  fail {layer}:m{m} {c}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
        res.sort(key=lambda r: r[0])
        finals = []
        for _, c in res[:3]:
            g, _ = capture(ctx.triton_fn(layer, c, x))
            finals.append((*paired(timer, g, base_g, args.final_reps), c))
        if not finals:
            continue
        us, b, sp, c = min(finals, key=lambda r: r[0])
        winners[(layer, m)] = c
        report[f"{layer}:{m}"] = {"cfg": list(c), "us": us, "cublas": b, "speedup": sp, "floor": floor}
        print(f"BEST {layer}:m{m} {tuple(c)} {us:.2f} vs cublas {b:.2f} x{sp:.3f} floor {floor:.2f} "
              f"[{metric(layer, m, us)}] cublas [{metric(layer, m, b)}]  t={time.time() - t0:.0f}s", flush=True)
    print("BEST_JSON " + json.dumps(report), flush=True)
    check_winners(ctx, w, winners)
    if not args.no_nearby:
        nearby_timing(ctx, timer, base, base_state, winners, 50)
    print(f"total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
