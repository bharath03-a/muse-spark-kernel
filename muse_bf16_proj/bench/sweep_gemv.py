"""Tile sweep for bench/gemv_dev.py on H200, timed like the judge, paired against cuBLAS.

    modal run modal_app.py::micro --name sweep_gemv --args "--mode full"

Phases: A (BN, BK, split at warps 4) -> B (top bases x warps x stages x kind) -> final
(top 3 per shape re-timed paired with cuBLAS) -> table + correctness of the chosen table.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

import gemv_dev as G
from common import load_module, reference

FLUSH_BYTES = 256 * 2**20
HW = {"hbm_bytes_per_s": 4.8e12, "bf16_dense_flops_per_s": 989e12}
CASES = [("pooler", 1), ("pooler", 8), ("pooler", 16), ("pooler", 32),
         ("unpooler", 1), ("unpooler", 8), ("unpooler", 16), ("unpooler", 32),
         ("qkv_gate", 1), ("qkv_gate", 8), ("qkv_gate", 16), ("qkv_gate", 32),
         ("o", 1), ("o", 8), ("o", 16), ("o", 32)]
SPLITS = (1, 2, 4, 8, 16)
MIN_CTAS, MAX_CTAS = 96, 1200
SPLIT_MAX_TILES = 144  # split-K only when there are too few n-tiles to fill 132 SMs


def ctas_ok(n_tiles: int, split: int) -> bool:
    return MIN_CTAS <= n_tiles * split <= MAX_CTAS and (split == 1 or n_tiles <= SPLIT_MAX_TILES)
HERE = Path(__file__).resolve().parent


def bt_for(m: int) -> int:
    return 16 if m <= 16 else 32 if m <= 32 else 64


def group_of(key: str, m: int, kind: str) -> tuple:
    return (key, "fma") if kind == "fma" else (key, bt_for(m))


def fit_stages(t: G.Tile, want: int) -> G.Tile:
    for s in range(want, 0, -1):
        c = t._replace(stages=s)
        if G.smem_bytes(c) <= G.SMEM_LIMIT:
            return c
    return t


def phase_a(key: str, bt: int) -> list[G.Tile]:
    n, k = G.SHAPES[key]
    out = []
    bks = (64, 128, 256) if key == "o" else (128, 256, 512)
    for bn in (16, 32, 64, 128):
        for bk in bks:
            for split in SPLITS:
                t = fit_stages(G.Tile(bt, bn, bk, bt, split, 4, 4), 4)
                if t.stages >= 2 and ctas_ok(n // bn, split) and G.valid(t, n, k):
                    out.append(t)
    return out


def phase_a_fma(key: str) -> list[G.Tile]:
    n, k = G.SHAPES[key]
    out = []
    for bn in (4, 8, 16):
        for bk in (256, 512, 1024):
            for split in SPLITS:
                t = G.Tile(1, bn, bk, 1, split, 4, 3, "fma")
                if t.stages >= 2 and ctas_ok(n // bn, split) and G.valid(t, n, k):
                    out.append(t)
    return out


def phase_b(base: G.Tile, key: str) -> list[G.Tile]:
    n, k = G.SHAPES[key]
    if base.kind == "fma":
        combos = [("fma", w, s) for w in (2, 4, 8) for s in (1, 2, 3, 4, 6)]
    else:
        combos = [("dot", w, s) for w in (4, 8) for s in (2, 3, 4, 5, 6, 8)]
        combos += [("tma", w, s) for w in (4, 8) for s in (3, 4, 5)]
    out = []
    for kind, w, s in combos:
        t = base._replace(kind=kind, warps=w, stages=s)
        if G.valid(t, n, k):
            out.append(t)
    return out


# ---------------------------------------------------------------- compile workers

def precompile(cands: list[tuple[str, G.Tile]], workers: int) -> set:
    if not cands:
        return set()
    t0 = time.time()
    chunks = [cands[i::workers] for i in range(workers)]
    tmp = Path("/tmp/sweep_gemv")
    tmp.mkdir(exist_ok=True)
    procs = []
    env = {**os.environ, "PYTHONPATH": f"{HERE.parent}:{HERE}"}
    for i, chunk in enumerate(chunks):
        if not chunk:
            continue
        path = tmp / f"chunk{i}.json"
        path.write_text(json.dumps([[key, list(t)] for key, t in chunk]))
        procs.append(subprocess.Popen([sys.executable, str(HERE / "sweep_gemv.py"), "--worker", str(path)],
                                      env=env))
    for p in procs:
        p.wait()
    failed = set()
    for f in tmp.glob("chunk*.json.fail"):
        failed |= {(key, G.Tile(*t)) for key, t in json.loads(f.read_text())}
        f.unlink()
    print(f"precompiled {len(cands)} candidates ({len(failed)} failed) on {len(procs)} workers in "
          f"{time.time() - t0:.1f}s", flush=True)
    return failed


def worker(path: str) -> None:
    items = [(key, G.Tile(*t)) for key, t in json.loads(Path(path).read_text())]
    w_cache: dict[str, torch.Tensor] = {}
    state = {"ws": torch.zeros(G.WS_FLOATS, device="cuda"),
             "cnt": torch.zeros(G.COUNTERS, device="cuda", dtype=torch.int32), "desc": {}}
    failed = []
    for key, t in items:
        n, k = G.SHAPES[key]
        if key not in w_cache:
            w_cache[key] = torch.zeros(n, k, device="cuda", dtype=torch.bfloat16)
        m = 1 if t.kind == "fma" else t.block_t
        x = torch.zeros(m, k, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        try:
            G.launch(x, w_cache[key], out, state, t)
            torch.cuda.synchronize()
        except Exception as exc:
            failed.append([key, list(t)])
            print(f"compile fail {key} {t}: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
    Path(path + ".fail").write_text(json.dumps(failed))


# ---------------------------------------------------------------- timing

class Timer:
    def __init__(self) -> None:
        self.flush = torch.empty(FLUSH_BYTES // 4, device="cuda", dtype=torch.float32).normal_()
        self.sink = torch.empty((), device="cuda", dtype=torch.float32)
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        g = torch.cuda.CUDAGraph()
        dummy = torch.zeros(1, device="cuda")
        with torch.cuda.graph(g):
            dummy.add_(1)
        self.overhead = 0.0
        self.overhead = statistics.median(self.once(g) for _ in range(300))

    def once(self, graph) -> float:
        torch.sum(self.flush, dim=0, out=self.sink)
        self.start.record()
        graph.replay()
        self.end.record()
        self.end.synchronize()
        return self.start.elapsed_time(self.end) * 1e3 - self.overhead

    def median(self, graph, reps: int) -> float:
        for _ in range(3):
            graph.replay()
        return statistics.median(self.once(graph) for _ in range(reps))


def capture(fn) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    torch.cuda.synchronize()
    return g, out


def rel_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    return float((out.float() - ref).norm() / ref.norm().clamp(min=1e-12))


class Shape:
    def __init__(self, layer: str, m: int, state: dict, base, base_state) -> None:
        self.layer, self.m = layer, m
        self.key = G.layer_key(layer, m)
        self.data = reference.generate_input(m, layer, 1)
        self.x = self.data["x"]
        self.w = state[self.key]
        self.n, self.k = self.w.shape
        self.ref = self.x.float() @ self.w.float().T
        self.floor_us = reference.speed_of_light_ns(m, layer, HW) / 1e3
        self.bytes = self.n * self.k * 2 + m * self.k * 2 + m * self.n * 2
        self.base_graph, _ = capture(lambda: base.kernel_fn(self.data, base_state))

    def name(self) -> str:
        return f"{self.key}:m{self.m}"


def mine_fn(shape: Shape, state: dict, t: G.Tile):
    def fn():
        out = torch.empty(shape.m, shape.n, device="cuda", dtype=torch.bfloat16)
        G.launch(shape.x, shape.w, out, state, t)
        return out
    return fn


def bench_tile(timer: Timer, shape: Shape, state: dict, t: G.Tile, reps: int) -> float | None:
    try:
        g, out = capture(mine_fn(shape, state, t))
        g.replay()
        torch.cuda.synchronize()
        e1 = rel_err(out, shape.ref)
        us = timer.median(g, reps)
        e2 = rel_err(out, shape.ref)
    except Exception as exc:
        print(f"  fail {shape.name()} {t}: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
        return None
    if not (e1 < 5e-3 and e2 < 5e-3):
        print(f"  WRONG {shape.name()} {t}: err {e1:.2e}/{e2:.2e}", flush=True)
        return None
    return us


def paired(timer: Timer, shape: Shape, state: dict, t: G.Tile, reps: int) -> dict:
    g, _ = capture(mine_fn(shape, state, t))
    for _ in range(3):
        g.replay()
        shape.base_graph.replay()
    mine, base, ratios = [], [], []
    for r in range(reps):
        order = (g, shape.base_graph) if r % 2 == 0 else (shape.base_graph, g)
        ts = {id(x): max(timer.once(x), 1e-3) for x in order}
        a, b = ts[id(g)], ts[id(shape.base_graph)]
        mine.append(a)
        base.append(b)
        ratios.append(b / a)
    sp = statistics.median(ratios)
    b_med = statistics.median(base)
    us = b_med / sp
    return {"tile": list(t), "us": us, "cublas_us": b_med, "speedup": sp,
            "tbps": shape.bytes / us / 1e6, "pct_floor": 100 * shape.floor_us / us,
            "floor_us": shape.floor_us, "mine_med": statistics.median(mine)}


# ---------------------------------------------------------------- driver

def sweep(shapes: list[Shape], state: dict, timer: Timer, cands: dict, reps: int,
          results: dict) -> None:
    for s in shapes:
        for t in cands.get(group_of(s.key, s.m, "dot"), []) + (
                cands.get((s.key, "fma"), []) if s.m == 1 else []):
            k = (s.name(), t)
            if k in results:
                continue
            us = bench_tile(timer, s, state, t, reps)
            if us is not None:
                results[k] = us
        print(f"{s.name()}: {sum(1 for (n, _) in results if n == s.name())} timed", flush=True)


def top_bases(shapes: list[Shape], results: dict, group: tuple, n_top: int) -> list[G.Tile]:
    fma = group[1] == "fma"
    names = [s.name() for s in shapes
             if (fma and s.key == group[0] and s.m == 1) or (not fma and group_of(s.key, s.m, "dot") == group)]
    best_per: dict[str, float] = {}
    for (name, t), us in results.items():
        if name in names and (t.kind == "fma") == fma:
            best_per[name] = min(us, best_per.get(name, us))
    score: dict[tuple, list[float]] = {}
    for (name, t), us in results.items():
        if name in best_per and (t.kind == "fma") == fma:
            score.setdefault((t.max_m, t.block_n, t.block_k, t.block_t, t.split), []).append(us / best_per[name])
    ranked = sorted((statistics.mean(v), b) for b, v in score.items() if len(v) == len(best_per))
    return [G.Tile(*b, 4, 3 if fma else 4, "fma" if fma else "dot") for _, b in ranked[:n_top]]


def build_table(best: dict) -> dict:
    tiles: dict[str, tuple] = {}
    for key, ms in (("pooler", (1, 8, 16, 32)), ("qkv_gate", (1, 8, 16, 32)), ("o", (1, 8, 16, 32)),
                    ("unpooler", (16, 32)), ("unpooler_slice", (1, 8))):
        tiles[key] = tuple(G.Tile(*best[f"{key}:m{m}"]["tile"])._replace(max_m=m) for m in ms)
    return tiles


def ptx_report(tiles: dict, state: dict) -> None:
    import re
    pats = {"ld.v4": r"ld\.global[\w.]*\.v4", "cp.async16": r"cp\.async\.cg\.shared\.global[^;]*16",
            "cp.async(any)": r"cp\.async\.c[ag]", "tma": r"cp\.async\.bulk\.tensor", "ld.scalar": r"ld\.global[\w.]*\.b16 "}
    print("PTX load instructions per chosen tile (grid t-dim is always 1 for dot/tma; fma is m=1):")
    for key, ts in tiles.items():
        n, k = G.SHAPES[key]
        for t in ts:
            m = t.max_m
            x = torch.zeros(m, k, device="cuda", dtype=torch.bfloat16)
            out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
            ck = G.launch(x, state[key], out, state, t)
            ptx = ck.asm.get("ptx", "") if ck is not None else ""
            counts = {name: len(re.findall(pat, ptx)) for name, pat in pats.items()}
            grid = (n // t.block_n, t.split, 1)
            print(f"  {key:15} m<={m:2d} {t.kind} BN{t.block_n} BK{t.block_k} BT{t.block_t} split{t.split} "
                  f"w{t.warps} s{t.stages} grid {grid} ctas {grid[0] * grid[1]} "
                  f"W/CTA {n // grid[0] * k // t.split * 2 // 1024}KB {counts}", flush=True)


def correctness(tiles: dict, w: dict) -> int:
    state = G.prepare(w, tiles)
    fails = total = 0
    for layer in ("qkv_gate", "o", "pooler", "unpooler"):
        for m in range(1, 41):
            for dist in ("normal", "outliers"):
                for seed in ((1, 2) if m in (1, 8, 16, 32) else (3,)):
                    data = reference.generate_input(m, layer, seed, dist=dist)
                    out = G.kernel_fn(data, state)
                    torch.cuda.synchronize()
                    ok, msg = reference.check_implementation(data, out, w)
                    total += 1
                    if not ok:
                        fails += 1
                        print(f"FAIL {layer} m={m} {dist} seed={seed}: {msg}", flush=True)
    print(f"correctness: {total - fails}/{total} passed", flush=True)
    return fails


def nearby(tiles: dict, w: dict, timer: Timer, base, base_state, reps: int) -> None:
    state = G.prepare(w, tiles)
    print("nearby m (unpaired medians):")
    for layer in ("qkv_gate", "o", "pooler", "unpooler"):
        for m in (2, 5, 7, 9, 12, 24, 33, 40):
            data = reference.generate_input(m, layer, 1)
            g, _ = capture(lambda: G.kernel_fn(data, state))
            bg, _ = capture(lambda: base.kernel_fn(data, base_state))
            a, b = timer.median(g, reps), timer.median(bg, reps)
            print(f"  {layer:9} m={m:3d} mine {a:7.2f} cublas {b:7.2f} ratio {b / a:5.3f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default="")
    ap.add_argument("--mode", default="full", choices=("smoke", "full", "ab"))
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--final-reps", type=int, default=300)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--top", type=int, default=2)
    args = ap.parse_args()
    if args.worker:
        worker(args.worker)
        return

    t_start = time.time()
    ncpu = len(os.sched_getaffinity(0))
    workers = args.workers or max(1, min(16, ncpu))
    import triton
    print(f"triton {triton.__version__} torch {torch.__version__} cpus {ncpu} workers {workers} "
          f"tma {'yes' if G.TensorDescriptor is not None else 'no'}", flush=True)

    w = reference.generate_weights(reference.WEIGHTS_SEED)
    state = G.prepare(w)
    base = load_module("baseline.py", "base")
    base_state = base.prepare({k: v.clone() for k, v in w.items()})
    timer = Timer()
    print(f"empty graph replay {timer.overhead:.2f} us (subtracted)", flush=True)
    shapes = [Shape(layer, m, state, base, base_state) for layer, m in CASES]
    cublas = {s.name(): timer.median(s.base_graph, args.reps) for s in shapes}
    for s in shapes:
        print(f"cublas {s.name():22} {cublas[s.name()]:7.2f} us  floor {s.floor_us:6.2f}", flush=True)

    groups = sorted({group_of(s.key, s.m, "dot") for s in shapes} |
                    {(s.key, "fma") for s in shapes if s.m == 1}, key=str)
    cands: dict[tuple, list[G.Tile]] = {}
    for g in groups:
        cands[g] = phase_a_fma(g[0]) if g[1] == "fma" else phase_a(g[0], g[1])
    if args.mode == "ab":
        cands = {}
        for s in shapes:
            t = G.pick(G.TILES, s.key, s.m)
            if t.kind == "fma":
                vs = [t]
            else:
                kk = t.kind.rstrip("k") + "k"
                flip = ("tma" if t.kind.startswith("dot") else "dot") + "k"
                vs = [t, t._replace(kind=kk), t._replace(kind=kk, stages=t.stages + 1),
                      t._replace(kind=kk, stages=max(2, t.stages - 1)), t._replace(kind=flip)]
            n, k = G.SHAPES[s.key]
            g = group_of(s.key, s.m, t.kind)
            cands.setdefault(g, [])
            cands[g] += [v for v in vs if G.valid(v, n, k) and v not in cands[g]]
    elif args.mode == "smoke":
        cands = {g: c[:2] for g, c in cands.items()}
        base_t = G.Tile(16, 32, 128, 16, 1, 4, 4)
        cands[("pooler", 16)].append(base_t._replace(kind="tma"))
        cands[("unpooler_slice", 16)].append(G.Tile(16, 32, 128, 16, 4, 4, 4, "tma"))
    print(f"phase A: {sum(map(len, cands.values()))} candidates", flush=True)
    bad = precompile([(g[0], t) for g, ts in cands.items() for t in ts], workers)
    cands = {g: [t for t in ts if (g[0], t) not in bad] for g, ts in cands.items()}
    results: dict = {}
    sweep(shapes, state, timer, cands, args.reps, results)
    print(f"phase A done at {time.time() - t_start:.0f}s", flush=True)

    if args.mode == "full":
        cands_b: dict[tuple, list[G.Tile]] = {}
        for g in groups:
            seen = set(cands[g])
            cands_b[g] = [t for b in top_bases(shapes, results, g, args.top) for t in phase_b(b, g[0])
                          if t not in seen]
        print(f"phase B: {sum(map(len, cands_b.values()))} candidates", flush=True)
        bad = precompile([(g[0], t) for g, ts in cands_b.items() for t in ts], workers)
        cands_b = {g: [t for t in ts if (g[0], t) not in bad] for g, ts in cands_b.items()}
        sweep(shapes, state, timer, cands_b, args.reps, results)
        print(f"phase B done at {time.time() - t_start:.0f}s", flush=True)

    best = {}
    print("\nfinal (paired vs cuBLAS):")
    for s in shapes:
        mine = sorted((us, t) for (name, t), us in results.items() if name == s.name())
        if not mine:
            print(f"{s.name()}: no candidate", flush=True)
            continue
        finals = [paired(timer, s, state, t, args.final_reps) for _, t in mine[:3]]
        win = min(finals, key=lambda r: r["us"])
        best[s.name()] = win
        print(f"{s.name():22} {str(G.Tile(*win['tile'])):80} mine {win['us']:6.2f} cublas {win['cublas_us']:6.2f} "
              f"x{win['speedup']:.3f} {win['tbps']:.2f}TB/s floor {win['floor_us']:.2f} "
              f"({win['pct_floor']:.0f}%)", flush=True)
        for _, t in mine[:6]:
            print(f"    cand {t} {results[(s.name(), t)]:.2f}", flush=True)
    print("BEST_JSON " + json.dumps(best), flush=True)

    if len(best) == len(shapes):
        tiles = build_table(best)
        print("TILES = {")
        for key, ts in tiles.items():
            print(f"    {key!r}: (" + ", ".join(repr(t) for t in ts) + "),")
        print("}", flush=True)
        ptx_report(tiles, state)
        correctness(tiles, w)
        nearby(tiles, w, timer, base, base_state, args.reps)
    print(f"total {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
