"""Phase 2.5 go/no-go: fused-decode 12-bit weight GEMV/GEMM vs our BF16 Triton kernels vs cuBLAS.

    modal run modal_app.py::session --cmds "sweep_compress --reps 50"

Per case: every compressed candidate is checked (fp32 ref) and timed (CUDA graph, 256 MB read
flush per replay, 4.0 us subtracted); the best is then timed paired, 3-way, against our BF16
kernel (submission's tile) and cuBLAS, and run through reference.check_implementation
(normal + outliers inputs, seed-5151 and held-out seed-777 weights).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

import compress_dev as C
from common import reference

HERE = Path(__file__).resolve().parent
FLUSH_BYTES = 256 * 2**20
OVERHEAD_US = 4.0
HELD_OUT_SEED = 777
SLICE = 1024
T = C.CTile

FMA8K = [T("fma", 8, 1024, 1, 1, 4, 2), T("fma", 8, 1024, 1, 1, 4, 3), T("fma", 8, 1024, 1, 1, 4, 4),
         T("fma", 8, 1024, 1, 1, 2, 2), T("fma", 4, 1024, 1, 1, 2, 3), T("fma", 4, 2048, 1, 1, 4, 3),
         T("fma", 4, 1024, 1, 2, 4, 2), T("fma", 8, 512, 1, 2, 4, 3), T("fma", 8, 1024, 1, 1, 8, 2)]
FMA4K = [T("fma", 4, 1024, 1, 1, 8, 2)]
POOL16 = [T("dot", 64, 128, 16, 4, 4, 3), T("dot", 64, 128, 16, 4, 4, 3, True), T("dot", 32, 128, 16, 4, 4, 4, True),
          T("dot", 64, 128, 16, 2, 4, 4, True), T("dot", 64, 256, 16, 2, 4, 3, True), T("dot", 32, 256, 16, 2, 4, 3, True),
          T("dot", 64, 128, 16, 1, 4, 5, True), T("dot", 128, 128, 16, 1, 4, 3, True), T("dot", 128, 64, 16, 1, 4, 6, True),
          T("dot", 64, 64, 16, 2, 4, 6, True)]
POOL32 = [T("dot", 64, 128, 32, 4, 4, 3, True), T("dot", 64, 128, 32, 2, 4, 4, True), T("dot", 64, 128, 32, 1, 4, 5, True),
          T("dot", 128, 128, 32, 1, 4, 3, True), T("dot", 128, 64, 32, 1, 4, 6, True), T("dot", 64, 64, 32, 2, 4, 6, True),
          T("dot", 64, 256, 32, 2, 4, 3, True)]
QKV16 = [T("dot", 32, 128, 16, 8, 4, 4), T("dot", 32, 128, 16, 8, 4, 4, True), T("dot", 64, 128, 16, 8, 4, 3, True),
         T("dot", 64, 128, 16, 4, 4, 4, True), T("dot", 64, 64, 16, 8, 4, 6, True)]
QKV32 = [T("dot", 64, 128, 32, 8, 4, 3, True), T("dot", 64, 128, 32, 4, 4, 4, True), T("dot", 64, 64, 32, 8, 4, 6, True)]
SLICE16 = [T("dot", 32, 256, 16, 16, 4, 2, True), T("dot", 64, 128, 16, 8, 4, 4, True)]
UNP32 = [T("dot", 64, 128, 32, 2, 4, 4, True), T("dot", 64, 128, 32, 1, 4, 5, True), T("dot", 128, 128, 32, 1, 4, 3, True)]
CASES = [("pooler", 1, FMA8K), ("pooler", 8, POOL16), ("pooler", 16, POOL16), ("pooler", 32, POOL32),
         ("qkv_gate", 1, FMA8K), ("qkv_gate", 8, QKV16), ("qkv_gate", 32, QKV32),
         ("unpooler", 1, FMA4K), ("unpooler", 8, SLICE16), ("unpooler", 32, UNP32)]
LAYERS = ("pooler", "qkv_gate", "unpooler")


def key_of(layer: str, m: int) -> str:
    return "unpooler_slice" if layer == "unpooler" and m <= reference.SLICED_MAX_TOKENS else layer


class Timer:
    def __init__(self) -> None:
        self.flush = torch.empty(FLUSH_BYTES // 4, device="cuda", dtype=torch.float32).normal_()
        self.sink = torch.empty((), device="cuda", dtype=torch.float32)
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def once(self, graph) -> float:
        torch.sum(self.flush, dim=0, out=self.sink)
        self.start.record()
        graph.replay()
        self.end.record()
        self.end.synchronize()
        return max(self.start.elapsed_time(self.end) * 1e3 - OVERHEAD_US, 1e-3)

    def median(self, graph, reps: int) -> float:
        for _ in range(3):
            graph.replay()
        return statistics.median(self.once(graph) for _ in range(reps))


def capture(fn):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    torch.cuda.synchronize()
    return g, out


def rel_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    return float((out.float() - ref).norm() / ref.norm().clamp(min=1e-12))


class Ctx:
    def __init__(self, e_max: dict[int, int]) -> None:
        self.ws = torch.zeros(4 * 2**20, device="cuda", dtype=torch.float32)
        self.cnt = torch.zeros(8192, device="cuda", dtype=torch.int32)
        self.e_max = e_max

    def comp_fn(self, x: torch.Tensor, p: C.Packed, n: int, t: C.CTile):
        def fn():
            out = torch.empty(x.shape[0], n, device="cuda", dtype=torch.bfloat16)
            C.launch(x, p, n, out, self.ws, self.cnt, t, self.e_max[t.block_n])
            return out
        return fn


# ------------------------------------------------------------------------- parallel precompile

def _dummy_packed(n: int, k: int, plain: bool) -> C.Packed:
    z = torch.zeros(0, device="cuda", dtype=torch.int64)
    return C.Packed(torch.zeros(n, k // 2, device="cuda", dtype=torch.int16),
                    torch.zeros(n, k // 2, device="cuda", dtype=torch.uint8), 1.0, 54, z, z,
                    torch.zeros(0, device="cuda", dtype=torch.bfloat16), plain, C.GROUP)


def worker(path: str) -> None:
    specs = json.loads(Path(path).read_text())
    e_max = {int(k): v for k, v in specs["e_max"].items()}
    ctx = Ctx(e_max)
    bad = []
    for n, k, m, t in specs["items"]:
        t = C.CTile(*t)
        try:
            x = torch.zeros(m, k, device="cuda", dtype=torch.bfloat16)
            ctx.comp_fn(x, _dummy_packed(n, k, t.plain), n, t)()
            torch.cuda.synchronize()
        except Exception as exc:
            bad.append(list(t))
            print(f"compile fail {t}: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
    Path(path + ".fail").write_text(json.dumps(bad))


def precompile(items: list, e_max: dict[int, int], workers: int) -> None:
    t0 = time.time()
    tmp = Path("/tmp/sweep_compress")
    tmp.mkdir(exist_ok=True)
    env = {**os.environ, "PYTHONPATH": f"{HERE.parent}:{HERE}"}
    procs = []
    for i in range(workers):
        chunk = items[i::workers]
        if chunk:
            f = tmp / f"chunk{i}.json"
            f.write_text(json.dumps({"e_max": e_max, "items": chunk}))
            procs.append(subprocess.Popen([sys.executable, str(HERE / "sweep_compress.py"), "--worker", str(f)],
                                          env=env))
    for p in procs:
        p.wait()
    print(f"precompiled {len(items)} kernels on {len(procs)} workers in {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------------------------ driver

def encode_all(weights: dict, label: str) -> dict:
    packs = {}
    for layer in LAYERS:
        w = weights[layer]
        for plain in (False, True):
            t0 = time.time()
            p = C.encode(w, plain=plain)
            C.round_trip(w, p)
            packs[(layer, plain)] = p
            if not plain:
                n, k = w.shape
                esc4, esc32, esc64 = (C.max_block_escapes(p, b) for b in (4, 32, 64))
                print(f"[{label}] {layer:9} eb={p.eb} E window [{2 * p.eb},{2 * p.eb + 15}] escapes {p.n_esc} "
                      f"({p.n_esc / w.numel():.2e}) max/block BN4 {esc4} BN32 {esc32} BN64 {esc64}; "
                      f"bytes/w (BN32 lists) {C.nbytes(p, 32) / w.numel():.4f} -> ratio "
                      f"{2 * w.numel() / C.nbytes(p, 32):.3f}x; round trip bit-exact ({time.time() - t0:.1f}s)",
                      flush=True)
    return packs


def decode_kernel_test(w: torch.Tensor, packs: dict, layer: str) -> None:
    n, k = w.shape
    for plain in (False, True):
        p = packs[(layer, plain)]
        out = torch.empty(n, k, device="cuda", dtype=torch.bfloat16)
        C._decode_store[(n // 16, k // 256)](p.a, p.b, out, K=k, G=p.group, BLOCK_N=16, BLOCK_K=256, PLAIN=plain)
        got = out.view(torch.int16).to(torch.int32) & 0xFFFF
        want = C.expected_rebased(w, p)
        bad = int((got != want).sum())
        print(f"in-kernel decode {layer} plain={plain}: {bad} mismatching weights of {n * k}"
              f"{' (bit-exact)' if bad == 0 else ' FAIL'}", flush=True)


def bf16_state(sub, weights: dict) -> dict:
    state = {name: weights[name].contiguous() for name in ("qkv_gate", "o", "pooler", "unpooler")}
    state["unpooler_slice"] = state["unpooler"][:SLICE]
    state["ws"] = torch.empty(sub.WS_FLOATS, device="cuda", dtype=torch.float32)
    state["cnt"] = torch.zeros(sub.COUNTERS, device="cuda", dtype=torch.int32)
    state["desc"] = {}
    sub._check_bf16_tiles(state)
    return state


def ours_fn(sub, state: dict, key: str, x: torch.Tensor):
    w = state[key]
    tile = sub._bf16_pick(key, x.shape[0])

    def fn():
        if tile is None:
            return x @ w.T
        out = torch.empty(x.shape[0], w.shape[0], device="cuda", dtype=torch.bfloat16)
        sub._bf16_launch(x, w, out, state, key, tile)
        return out
    return fn, tile


def paired3(timer: Timer, graphs: list, reps: int) -> list[list[float]]:
    perms = list(itertools.permutations(range(len(graphs))))
    for g in graphs:
        for _ in range(3):
            g.replay()
    times = [[] for _ in graphs]
    for r in range(reps):
        for i in perms[r % len(perms)]:
            times[i].append(timer.once(graphs[i]))
    return times


def check_ref(ctx: Ctx, weights: dict, p: C.Packed, layer: str, m: int, t: C.CTile, seeds) -> list[str]:
    msgs = []
    n = reference.out_features(layer, m)
    for seed, dist in seeds:
        data = reference.generate_input(m, layer, seed, dist=dist)
        out = ctx.comp_fn(data["x"], p, n, t)()
        ok, msg = reference.check_implementation(data, out, weights)
        msgs.append(f"{'PASS' if ok else 'FAIL'} seed{seed}/{dist}: {msg}")
    return msgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--worker", default="")
    args = ap.parse_args()
    if args.worker:
        worker(args.worker)
        return
    t_start = time.time()
    import submission as sub

    weights = reference.generate_weights(reference.WEIGHTS_SEED)
    held = reference.generate_weights(HELD_OUT_SEED)
    packs = encode_all(weights, "seed5151")
    held_packs = encode_all(held, "seed777 ")
    decode_kernel_test(weights["pooler"], packs, "pooler")

    cases = [c for c in CASES if not args.only or c[0] in args.only.split(",")]
    e_max = {}
    for _, _, cands in cases:
        for t in cands:
            need = max(C.esc_lists(pk, t.block_n)[2] for pk in (*packs.values(), *held_packs.values()))
            e_max[t.block_n] = max(e_max.get(t.block_n, 16), need)
    print("E_MAX per BN:", e_max, flush=True)
    items, seen = [], set()
    for layer, m, cands in cases:
        n, k = reference.out_features(layer, m), reference.LAYERS[layer][1]
        for t in cands:
            if C.tile_fits(t, n, k) and (k, t) not in seen:
                seen.add((k, t))
                items.append([n, k, m, list(t)])
    precompile(items, e_max, min(args.workers, os.cpu_count() or 1))

    ctx = Ctx(e_max)
    timer = Timer()
    state = bf16_state(sub, weights)
    hw = {"hbm_bytes_per_s": 4.8e12, "bf16_dense_flops_per_s": 989e12}
    rows = []
    for layer, m, cands in cases:
        key = key_of(layer, m)
        n, k = reference.out_features(layer, m), reference.LAYERS[layer][1]
        data = reference.generate_input(m, layer, 1)
        x = data["x"]
        ref = x.float() @ weights[layer][:n].float().T
        best = None
        for t in cands:
            if not C.tile_fits(t, n, k):
                continue
            p = packs[(layer, t.plain)]
            try:
                g, out = capture(ctx.comp_fn(x, p, n, t))
                g.replay()
                torch.cuda.synchronize()
                err = rel_err(out, ref)
                us = timer.median(g, args.reps)
            except Exception as exc:
                print(f"  fail {key} m{m} {t}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
                continue
            flag = "" if err < 5e-3 else "  WRONG"
            print(f"  {key:14} m{m:<3} {t}  {us:7.2f} us  err {err:.1e}{flag}", flush=True)
            if not flag and (best is None or us < best[0]):
                best = (us, t)
            if t.plain:
                rows.append({"key": key, "m": m, "plain_us": us, "plain_tile": list(t)})
        if best is None:
            print(f"!! {key} m{m}: no valid compressed candidate", flush=True)
            continue
        t = best[1]
        p = packs[(layer, t.plain)]
        gc, _ = capture(ctx.comp_fn(x, p, n, t))
        fo, tile_ours = ours_fn(sub, state, key, x)
        go, _ = capture(fo)
        w = state[key]
        gb, _ = capture(lambda: x @ w.T)
        tc, to, tb = paired3(timer, [gc, go, gb], args.reps)
        r_ours = statistics.median(b / a for a, b in zip(tc, to))
        r_cub = statistics.median(b / a for a, b in zip(tc, tb))
        mc, mo, mb = (statistics.median(v) for v in (tc, to, tb))
        nk, ew, _ = C.esc_lists(p, t.block_n, e_max[t.block_n])
        blocks = n // t.block_n
        comp_bytes = n * k * 1.5 + blocks * nk.shape[1] * 6 + m * k * 2 + m * n * 2
        bf_bytes = n * k * 2 + m * k * 2 + m * n * 2
        checks = check_ref(ctx, weights, p, layer, m, t, [(1, "normal"), (2, "normal"), (3, "outliers")])
        checks += check_ref(ctx, held, held_packs[(layer, t.plain)], layer, m, t, [(4, "normal"), (5, "outliers")])
        row = {"key": key, "m": m, "tile": list(t), "comp_us": mc, "ours_us": mo, "cublas_us": mb,
               "speedup_vs_ours": r_ours, "speedup_vs_cublas": r_cub, "ours_tile": list(tile_ours) if tile_ours else "cublas",
               "comp_tbps": comp_bytes / mc / 1e6, "comp_equiv_bf16_tbps": bf_bytes / mc / 1e6,
               "ours_tbps": bf_bytes / mo / 1e6, "cublas_tbps": bf_bytes / mb / 1e6,
               "floor_us": reference.speed_of_light_ns(m, layer, hw) / 1e3, "checks": checks}
        rows.append(row)
        print(f"== {key} m{m}: comp {mc:.2f} us ({row['comp_tbps']:.2f} TB/s compressed, "
              f"{row['comp_equiv_bf16_tbps']:.2f} bf16-equiv) | ours {mo:.2f} us ({row['ours_tbps']:.2f}) | "
              f"cuBLAS {mb:.2f} us ({row['cublas_tbps']:.2f}) | vs ours {r_ours:.3f}x vs cuBLAS {r_cub:.3f}x "
              f"tile {t}", flush=True)
        for c in checks:
            print("     ", c, flush=True)
    print("\nSUMMARY")
    print(f"{'shape':16} {'m':>3} {'comp us':>8} {'ours us':>8} {'cuBLAS':>8} {'vs ours':>8} {'vs cuBL':>8} "
          f"{'compTB/s':>8} {'eqTB/s':>7} {'oursTB/s':>8}")
    for r in rows:
        if "comp_us" in r:
            print(f"{r['key']:16} {r['m']:3d} {r['comp_us']:8.2f} {r['ours_us']:8.2f} {r['cublas_us']:8.2f} "
                  f"{r['speedup_vs_ours']:7.3f}x {r['speedup_vs_cublas']:7.3f}x {r['comp_tbps']:8.2f} "
                  f"{r['comp_equiv_bf16_tbps']:7.2f} {r['ours_tbps']:8.2f}")
    print("JSON", json.dumps(rows))
    print(f"sweep_compress wall {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
