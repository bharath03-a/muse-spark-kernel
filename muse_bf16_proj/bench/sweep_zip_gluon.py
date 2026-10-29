"""Step-3 go/no-go: 12-bit weights decoded into MMA A-operand registers (bench/zip_gluon_dev.py).

    modal run modal_app.py::compile_only --cmd "sweep_zip_gluon --compile"        # CPU: IR/SASS/regs
    modal run modal_app.py::micro --name sweep_zip_gluon --args "--shapes pooler:16"

Per shape, in one process: every candidate (MMA v2/v3, warps, groups/stage, stages) is checked against
an fp32 ref and timed alone; then the best compressed config, its dense twin (same layout/MMA, bf16)
and the current best kernel (gluon_probe gemv_pipe or submission's tile, whichever is faster) are timed
3-way paired: CUDA-graph replay, 256 MB read flush before every replay, 4.0 us subtracted (judge-like).
Correctness: in-kernel decode bit-exact vs torch, reference.check_implementation (normal + outliers,
probe rows, seeds 5151 and held-out 777).
"""

from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
import tempfile
import time
import traceback

import torch

import zip_gluon_dev as Z

FLUSH_BYTES = 256 * 2**20
OVERHEAD_US = 4.0
HELD_OUT = 777
# (mma, warps, groups per stage, stages)
# (mma, warps, groups per stage, stages (0 = fill SMEM_BUDGET), K fold q)
CANDS = [(3, 4, 1, 8, 1), (3, 4, 2, 5, 1), (3, 8, 1, 0, 2), (3, 8, 1, 0, 4), (3, 4, 1, 0, 2), (3, 4, 1, 0, 4),
         (3, 8, 1, 0, 8)]
SMEM_BUDGET = 200 * 1024


def stages_for(cand, comp: bool, bt: int) -> int:
    mma, warps, kg, stages, q = cand
    if stages:
        return stages
    bn = 16 * warps
    per = kg * (bn * 128 * (3 if comp else 4) // 2) + 2 * kg * q * bt * 128
    return max(2, min(8, SMEM_BUDGET // per))


class Packs:
    def __init__(self) -> None:
        self.d: dict = {}

    def get(self, layer: str, w: torch.Tensor, warps: int, q: int, comp: bool) -> "Z.ZPack":
        key = (layer, warps, q, comp, w.data_ptr())
        if key not in self.d:
            rows = 16 * warps // q
            wv = Z.fold(w, rows, q) if q > 1 else w
            p = Z.encode(wv, warps, dense=not comp)
            if comp:
                Z.round_trip(wv, p)
                print(f"  [{layer}] pack warps {warps} q {q}: eb {p.eb} escapes {p.n_esc} E_MAX {p.e_max} "
                      f"bytes/w {p.nbytes() / w.numel():.4f}; torch round trip bit-exact", flush=True)
            self.d[key] = p
        return self.d[key]


def bt_for(m: int, mma: int) -> int:
    bt = 8
    while bt < m:
        bt *= 2
    return max(bt, 16) if mma == 3 and m > 8 else bt


# ---------------------------------------------------------------------------------- CPU compile

def compile_report(only: str) -> None:
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import get_ptxas
    import triton
    from triton.experimental.gluon._runtime import GluonASTSource
    from triton.tools.disasm import get_sass

    n, k = 4096, 8192
    for cand in CANDS:
        mma, warps, kg, _, q = cand
        for comp in (True, False):
            for m in (16,):
                bt = bt_for(m, mma)
                stages = stages_for(cand, comp, bt)
                tag = f"mma{mma} w{warps} kg{kg} s{stages} q{q} {'comp' if comp else 'dense'} m{m}"
                if only and only not in tag:
                    continue
                bn = 16 * warps
                rows = bn // 4
                a = torch.zeros(rows * 64, 256, dtype=torch.int16 if comp else torch.bfloat16)
                b = torch.zeros(rows * 64, 256, dtype=torch.uint8)
                x = torch.zeros(m, k, dtype=torch.bfloat16)
                a_desc = Z.GDesc.from_tensor(a, [rows, 256], Z.PLANE16)
                b_desc = Z.GDesc.from_tensor(b, [rows, 256], Z.PLANE8) if comp else a_desc
                x_desc = Z.x_desc_for(x, bt)
                dll, sha, shb = Z.layouts(warps)
                fn = Z.zgemv
                sig = {"a_desc": a_desc.__mangle__(), "b_desc": b_desc.__mangle__(), "x_desc": x_desc.__mangle__(),
                       "enk_ptr": "*i32", "ew_ptr": "*bf16", "x_ptr": "*bf16", "out_ptr": "*bf16", "M": "i32",
                       "stride_xm": "i32", "scale": "fp32"}
                consts = dict(N=n, K=k // q, KG=kg, STAGES=stages, COMP=comp, MMA=mma, E_MAX=128 if comp else 0,
                              DLL=dll, SHA=sha, SHB=shb, Q=q, num_warps=warps)
                names = list(fn.arg_names)
                full = {nm: sig.get(nm, "constexpr") for nm in names}
                attrs = {(names.index(nm),): [["tt.divisibility", 16]] for nm in sig
                         if sig[nm].startswith("*") or nm == "stride_xm"}
                src = GluonASTSource(fn=fn, signature=full, constexprs={(names.index(nm),): v for nm, v in consts.items()},
                                     attrs=attrs)
                t0 = time.time()
                try:
                    ker = triton.compile(src, target=GPUTarget("cuda", 90, 32), options={"num_warps": warps})
                except Exception as exc:
                    print(f"== {tag}: compile FAILED {type(exc).__name__}: {str(exc)[:3000]}", flush=True)
                    continue
                ttgir, ptx = ker.asm["ttgir"], ker.asm["ptx"]
                print(f"== {tag}: compiled {time.time() - t0:.1f}s smem {ker.metadata.shared}", flush=True)
                print("   ttgir: convert_layout", ttgir.count("ttg.convert_layout"), "local_load",
                      ttgir.count("ttg.local_load"), "inline_asm", ttgir.count("elementwise_inline_asm"),
                      "warp_group_dot", ttgir.count("warp_group_dot "), "tt.dot", ttgir.count("tt.dot "))
                ops = re.findall(r"^\s*(?:@%p\d+\s+)?([a-z][a-z0-9_]*(?:\.[a-z0-9_]+)*)", ptx, flags=re.M)
                hist: dict[str, int] = {}
                for o in ops:
                    key = o if o.startswith(("ld.shared", "wgmma", "mma", "ldmatrix", "prmt", "lop3", "shf",
                                             "cp.async.bulk")) else o.split(".")[0]
                    hist[key] = hist.get(key, 0) + 1
                top = sorted(hist.items(), key=lambda kv: -kv[1])[:26]
                print("   ptx:", " ".join(f"{o}:{c}" for o, c in top), flush=True)
                with tempfile.TemporaryDirectory() as td:
                    path = f"{td}/k.ptx"
                    open(path, "w").write(ptx)
                    res = subprocess.run([get_ptxas(90).path, "-arch=sm_90a", "-v", path, "-o", f"{td}/k.o"],
                                         capture_output=True, text=True)
                    for line in (res.stdout + res.stderr).splitlines():
                        if any(s in line for s in ("registers", "spill", "wgmma", "Warning", "error")):
                            print("   ptxas:", line.strip()[:300], flush=True)
                try:
                    sass = get_sass(ker.asm["cubin"])
                    sops = [o.split(".")[0] for o in re.findall(r"^\s+/\*[0-9a-f]+\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)",
                                                                  sass, flags=re.M)]
                    sh: dict[str, int] = {}
                    for o in sops:
                        sh[o] = sh.get(o, 0) + 1
                    print("   sass:", " ".join(f"{o}:{c}" for o, c in sorted(sh.items(), key=lambda kv: -kv[1])[:30]),
                          flush=True)
                    if "--sass" in sys.argv:
                        print(sass[:20000])
                except Exception as exc:
                    print("   sass unavailable:", type(exc).__name__, str(exc)[:200], flush=True)


# ------------------------------------------------------------------------------------- GPU run

class Timer:
    def __init__(self) -> None:
        self.flush = torch.empty(FLUSH_BYTES // 4, device="cuda", dtype=torch.float32).normal_()
        self.sink = torch.empty((), device="cuda", dtype=torch.float32)
        self.ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

    def once(self, graph) -> float:
        torch.sum(self.flush, dim=0, out=self.sink)
        self.ev[0].record()
        graph.replay()
        self.ev[1].record()
        self.ev[1].synchronize()
        return self.ev[0].elapsed_time(self.ev[1]) * 1e3 - OVERHEAD_US

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


def rel(out: torch.Tensor, ref: torch.Tensor) -> float:
    return float((out.float() - ref).norm() / ref.norm().clamp(min=1e-12))


def zfn(runner, x: torch.Tensor, n: int):
    def fn():
        out = torch.empty(x.shape[0], n, device="cuda", dtype=torch.bfloat16)
        runner(x, out)
        return out
    return fn


def best_fns(layer: str, w: torch.Tensor, x: torch.Tensor):
    """Current best candidates: gluon_probe gemv_pipe and submission's BF16 tile (if it has one)."""
    fns = {}
    m, n = x.shape[0], w.shape[0]
    try:
        import gluon_probe as gp

        bt = max(16, m)

        def pipe():
            out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
            gp.launch_gluon("pipe", w, x, out, 64, 128, bt, 8, 4)
            return out
        fns["probe_pipe"] = pipe
    except Exception as exc:
        print(f"  gluon_probe unavailable: {type(exc).__name__}: {exc}", flush=True)
    try:
        import submission as sub

        key = layer
        tile = sub._bf16_pick(key, m)
        if tile is not None:
            ws = torch.empty(sub.WS_FLOATS, device="cuda", dtype=torch.float32)
            cnt = torch.zeros(sub.COUNTERS, device="cuda", dtype=torch.int32)
            state = {"ws": ws, "cnt": cnt, "desc": {}}
            if tile.kind == "tma":
                state["desc"][(key, tile)] = sub.TensorDescriptor.from_tensor(w, [tile.block_n, tile.block_k])

            def subm():
                out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
                sub._bf16_launch(x, w, out, state, key, tile)
                return out
            fns[f"sub_{tile.kind}{tile.block_n}x{tile.block_k}s{tile.split}"] = subm
    except Exception as exc:
        print(f"  submission tile unavailable: {type(exc).__name__}: {exc}", flush=True)
    fns["cublas"] = lambda: x @ w.T
    return fns


def decode_test(w: torch.Tensor, packs: dict, label: str) -> bool:
    ok = True
    for warps, p in packs.items():
        got = Z.decode_full(p).view(torch.int16).to(torch.int32) & 0xFFFF
        if p.comp:
            want, _ = Z.expected_rebased(w, p)
        else:
            want = w.view(torch.int16).to(torch.int32) & 0xFFFF
        bad = int((got != want).sum())
        ok &= bad == 0
        print(f"  [{label}] in-kernel decode {'comp' if p.comp else 'dense'} warps {warps}: {bad} mismatches of "
              f"{w.numel()}{' (bit-exact)' if bad == 0 else ' FAIL'}", flush=True)
    return ok


def check_ref(reference, layer: str, m: int, fn_of_x, weights: dict, label: str) -> bool:
    ok_all = True
    for dist, seed in (("normal", 1), ("outliers", 2), ("normal", 3)):
        data = reference.generate_input(m, layer, seed, dist=dist)
        out = fn_of_x(data["x"])
        torch.cuda.synchronize()
        ok, msg = reference.check_implementation(data, out, weights)
        ok_all &= ok
        print(f"    check {label} {dist} seed {seed}: {'PASS' if ok else 'FAIL'} {msg}", flush=True)
    return ok_all


def run_shape(reference, layer: str, m: int, w: torch.Tensor, w_held: torch.Tensor, reps: int, timer: Timer,
              cache: dict) -> dict:
    n, k = w.shape
    print(f"\n##### {layer} m={m} (N={n}, K={k})", flush=True)
    x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
    ref = x.float() @ w.float().T
    dense_bytes = n * k * 2 + m * k * 2 + m * n * 2
    res = {}
    for cand in CANDS:
        mma, warps, kg, _, q = cand
        if n % (16 * warps // q) or (k // q // 128) % kg:
            continue
        bt = bt_for(m, mma)
        for comp in (True, False):
            stages = stages_for(cand, comp, bt)
            tag = f"{'comp ' if comp else 'dense'} mma{mma} w{warps} kg{kg} s{stages} q{q} bt{bt}"
            try:
                p = cache.get(layer, w, warps, q, comp)
                fn = zfn(Z.Runner(p, mma, kg, stages, bt, q), x, n)
                out = fn()
                torch.cuda.synchronize()
                err = rel(out, ref)
                g, _ = capture(fn)
                t = timer.median(g, reps)
                res[(comp, cand)] = (t, fn, err)
                print(f"  {tag}: {t:6.2f} us  {dense_bytes / t / 1e6:5.2f} TB/s eff  "
                      f"{(p.nbytes() + m * k * 2 + m * n * 2) / t / 1e6:5.2f} TB/s real  err {err:.1e}"
                      f"{'' if err < 1e-2 else '  BAD'}", flush=True)
            except Exception as exc:
                print(f"  {tag}: FAILED {type(exc).__name__}: {str(exc)[:1500]}", flush=True)
    best = {}
    for name, fn in best_fns(layer, w, x).items():
        try:
            out = fn()
            torch.cuda.synchronize()
            err = rel(out, ref)
            g, _ = capture(fn)
            t = timer.median(g, reps)
            best[name] = (t, fn)
            print(f"  best-cand {name}: {t:6.2f} us  {dense_bytes / t / 1e6:5.2f} TB/s  err {err:.1e}", flush=True)
        except Exception as exc:
            print(f"  best-cand {name}: FAILED {type(exc).__name__}: {str(exc)[:600]}", flush=True)
    good = {kc: v for kc, v in res.items() if v[2] < 1e-2}
    comp_c = [(v[0], kc[1]) for kc, v in good.items() if kc[0]]
    if not comp_c or not best:
        print("  no valid compressed kernel or no baseline; skip gate", flush=True)
        return {}
    _, cand = min(comp_c)
    bname = min(best, key=lambda nm: best[nm][0])
    trio = {"comp": res[(True, cand)][1], "best": best[bname][1]}
    if (False, cand) in good:
        trio["dense"] = res[(False, cand)][1]
    graphs = {nm: capture(fn)[0] for nm, fn in trio.items()}
    for g in graphs.values():
        for _ in range(3):
            g.replay()
    names = list(graphs)
    ts = {nm: [] for nm in names}
    for r in range(3 * reps):
        order = names[r % len(names):] + names[:r % len(names)]
        for nm in order:
            ts[nm].append(timer.once(graphs[nm]))
    med = {nm: statistics.median(v) for nm, v in ts.items()}
    d_cb = statistics.median(a - b for a, b in zip(ts["comp"], ts["best"]))
    line = (f"  PAIRED {layer} m={m} cand mma{cand[0]} w{cand[1]} kg{cand[2]} q{cand[4]} vs {bname}: "
            f"comp {med['comp']:.2f} | best {med['best']:.2f}")
    if "dense" in med:
        d_db = statistics.median(a - b for a, b in zip(ts["dense"], ts["best"]))
        d_cd = statistics.median(a - b for a, b in zip(ts["comp"], ts["dense"]))
        line += f" | dense {med['dense']:.2f} us; dense-best {d_db:+.2f} comp-dense {d_cd:+.2f}"
    line += f"; comp-best {d_cb:+.2f} us"
    print(line, flush=True)
    for nm in names:
        print(f"    {nm}: {med[nm]:.2f} us, {dense_bytes / med[nm] / 1e6:.2f} TB/s dense-equivalent", flush=True)
    # correctness of the chosen compressed kernel (and dense twin) on the judge's checks
    mma, warps, kg, _, q = cand
    weights = {layer: w}

    def mk(p):
        r = Z.Runner(p, mma, kg, stages_for(cand, p.comp, bt_for(m, mma)), bt_for(m, mma), q)

        def f(xx):
            out = torch.empty(xx.shape[0], n, device="cuda", dtype=torch.bfloat16)
            r(xx, out)
            return out
        return f
    ok = check_ref(reference, layer, m, mk(cache.get(layer, w, warps, q, True)), weights, "comp w5151")
    ok &= check_ref(reference, layer, m, mk(cache.get(layer, w, warps, q, False)), weights, "dense w5151")
    ok &= check_ref(reference, layer, m, mk(cache.get(layer, w_held, warps, q, True)), {layer: w_held}, "comp w777")
    return {"comp_best": d_cb, "ok": ok, "comp": med["comp"], "best": med["best"], "dense": med.get("dense")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--sass", action="store_true")
    ap.add_argument("--shapes", default="pooler:16")
    ap.add_argument("--reps", type=int, default=40)
    args = ap.parse_args()
    if args.compile:
        compile_report(args.only)
        return
    from common import reference

    t0 = time.time()
    timer = Timer()
    shapes = [(s.split(":")[0], int(s.split(":")[1])) for s in args.shapes.split(",")]
    layers = list(dict.fromkeys(ly for ly, _ in shapes))
    wts = reference.generate_weights(reference.WEIGHTS_SEED)
    held = reference.generate_weights(HELD_OUT)
    cache = Packs()
    if "--decode" in sys.argv:
        layer = layers[0]
        w = wts[layer]
        try:
            decode_test(w, {(wp, c): cache.get(layer, w, wp, 1, c) for wp, c in ((4, True), (4, False))}, layer)
        except Exception:
            traceback.print_exc()
    summary = {}
    for layer, m in shapes:
        try:
            summary[(layer, m)] = run_shape(reference, layer, m, wts[layer], held[layer], args.reps, timer, cache)
        except Exception:
            traceback.print_exc()
    print("\n##### SUMMARY (local us, 4.0 subtracted)", flush=True)
    for (layer, m), r in summary.items():
        if r:
            gate = "GO" if r["comp_best"] <= -2.0 and r["ok"] else "NO-GO"
            print(f"  {layer} m={m}: comp {r['comp']:.2f} best {r['best']:.2f} dense {r['dense']} "
                  f"comp-best {r['comp_best']:+.2f} us correct={r['ok']} -> {gate}", flush=True)
    print(f"total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
