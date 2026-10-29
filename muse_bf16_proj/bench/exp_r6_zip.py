"""ZipGEMM 12-bit experiment: build, CPU self-test, GPU correctness/exactness, decode microbench, paired timing.

    exp_r6_zip.py --cpu        # CPU: packer + decode emulation + escape fix vs fp64 reference
    exp_r6_zip.py --compile    # build the extension only
    exp_r6_zip.py [--reps N] [--only pooler,unpooler,qkv_gate]
"""
from __future__ import annotations

import argparse
import hashlib
import statistics
import sys
import time

import torch

from common import load_module, reference
import importlib
Z = None


def preamble(sub_src: str) -> str:
    a = sub_src.index("constexpr int BN = 64;")
    b = sub_src.index("__device__ __forceinline__ uint32_t smem_u32")
    c = sub_src.index("__device__ __forceinline__ uint64_t make_smem_desc")
    d = sub_src.index("inline cudaLaunchConfig_t cluster_cfg")
    e = sub_src.index("template <int BT, int BK, int STAGES, int P, int CSIZE>\nvoid launch_wgmma_64")
    return sub_src[:a] + sub_src[b:c] + sub_src[d:e]


def build(sub, ptxas=False):
    from torch.utils.cpp_extension import load_inline
    src = preamble(sub.CUDA_SRC) + Z.ZIP_SRC
    flags = list(sub.CUDA_FLAGS) + (["-Xptxas=-v"] if ptxas else [])
    tag = hashlib.sha1((src + " ".join(flags)).encode()).hexdigest()[:10]
    return load_inline(name=f"muse_zip_exp_{tag}", cpp_sources=Z.ZIP_CPP, cuda_sources=src,
                       functions=["launch_zip"], extra_cuda_cflags=flags, extra_cflags=["-O3"],
                       extra_ldflags=sub.CUDA_LDFLAGS, verbose=ptxas)


def cpu_test():
    dev = "cpu"
    torch.manual_seed(0)
    for n, k in ((256, 512), (128, 1024)):
        w = (torch.randn(n, k) / k ** 0.5).to(torch.bfloat16)
        w[3, 5] = 0.0
        w[7, 9] = 1e-9
        w[11, 13] = 3.0
        t0 = time.time()
        p = Z.pack(w, 4)
        dec = Z.unpack(p, n, k)
        exact = (dec * p["scale"])
        # escapes replaced by 2^-15*scale; everything else must equal w exactly
        eq = exact == w.float()
        n_bad = int((~eq).sum())
        print(f"cpu {n}x{k}: eb={p['eb']} esc={p['n_esc']} mismatch(non-escape expected)={n_bad} pack {time.time()-t0:.2f}s")
        assert n_bad == p["n_esc"], (n_bad, p["n_esc"])
        x = torch.randn(16, k).to(torch.bfloat16)
        got = Z.reference_out(p, dec, x)
        ref = x.double() @ w.double().t()
        err = (got.double() - ref).abs().max() / ref.abs().max()
        print(f"  max rel err vs fp64: {float(err):.2e}")
        assert err < 1e-5
    print("CPU OK")


def bench_pair(fa, fb, x_a, x_b, flush, reps, overhead=4.0):
    from harness import time_replay
    ga = torch.cuda.CUDAGraph()
    gb = torch.cuda.CUDAGraph()
    fa(); fb(); torch.cuda.synchronize()
    with torch.cuda.graph(ga):
        oa = fa()
    with torch.cuda.graph(gb):
        ob = fb()
    torch.cuda.synchronize()
    ratios, ta, tb = [], [], []
    for rep in range(reps):
        order = (ga, gb) if rep % 2 == 0 else (gb, ga)
        t = {}
        for g in order:
            t[id(g)] = max(time_replay(g, flush) - overhead, 1e-3)
        ta.append(t[id(ga)]); tb.append(t[id(gb)])
        ratios.append(t[id(ga)] / t[id(gb)])
    return statistics.median(ta), statistics.median(tb), statistics.median(ratios)


def main():
    import os, threading
    wd = threading.Timer(240 if "--wd" not in sys.argv else float(sys.argv[sys.argv.index("--wd")+1]), lambda: (print("WATCHDOG: exiting", flush=True), os._exit(0)))
    wd.daemon = True
    wd.start()
    global Z
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="exp_r6_zip_v2")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--ptxas", action="store_true")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--only", default="pooler,unpooler,qkv_gate")
    ap.add_argument("--wd", type=float, default=240)
    ap.add_argument("--idx", default="")
    args = ap.parse_args()
    Z = importlib.import_module(args.src)
    if args.cpu:
        cpu_test()
        return
    sub = load_module("submission.py", "sub")
    if args.ptxas:
        build(sub, ptxas=True)
        return
    ext = build(sub)
    if args.compile:
        cpu_test()
        return

    from harness import FLUSH_BYTES
    w = reference.generate_weights(reference.WEIGHTS_SEED)
    flush = torch.empty(FLUSH_BYTES // 4, device="cuda", dtype=torch.float32).normal_()
    packs = {}
    rbs = sorted({c[1] for _, _, c, _ in getattr(Z, "PLAN", [(0, 0, (0, 4, 0, 0, 0), 0)])})
    for layer in ("pooler", "unpooler", "qkv_gate"):
      for rbv in rbs:
        torch.cuda.synchronize(); t0 = time.time()
        packs[(layer, rbv)] = Z.pack(w[layer], rbv)
        torch.cuda.synchronize()
        p = packs[(layer, rbv)]
        print(f"pack {layer} rb={rbv}: {time.time()-t0:.2f}s eb={p['eb']} esc={p['n_esc']} ({p['n_esc']/w[layer].numel():.2e}) "
              f"bytes={p['buf'].numel()/w[layer].numel():.4f}/w", flush=True)

    def run(layer, x, cfg, n=None, wrap=0):
        p = packs[(layer, cfg[1])]
        n = n or w[layer].shape[0]
        out = torch.empty(x.shape[0], n, device="cuda", dtype=torch.bfloat16)
        ext.launch_zip(x, p["buf"], out, p["rowptr"], p["esck"], p["escd"], p["scale"], wrap, *cfg)
        return out

    # (layer, m, cfg=(nt, rb, cs, warps, stages), n)
    plan = getattr(Z, "PLAN", None) or [
        ("pooler", 1, (1, 4, 2, 8, 16), None), ("pooler", 8, (1, 4, 2, 8, 16), None),
        ("pooler", 16, (2, 4, 2, 8, 12), None),
        ("pooler", 16, (2, 4, 2, 8, 8), None),
        ("unpooler", 16, (2, 4, 1, 8, 12), None),
        ("qkv_gate", 1, (1, 4, 4, 8, 16), None), ("qkv_gate", 8, (1, 4, 4, 8, 16), None),
        ("qkv_gate", 16, (2, 4, 4, 8, 16), None),
    ]
    if args.idx:
        plan = [plan[int(i)] for i in args.idx.split(",")]
    only = set(args.only.split(","))
    print("--- correctness (vs fp64, rel L2) + bit-exact probes")
    for layer, m, cfg, n in plan:
        if layer not in only:
            continue
        data = reference.generate_input(m, layer, 3)
        x = data["x"]
        out = run(layer, x, cfg)
        ref = (x.double() @ w[layer].double().t())
        rel = float((out.double() - ref).norm() / ref.norm())
        # exactness: one-hot x rows read back weight columns
        k = w[layer].shape[1]
        bad = 0
        for b in range(4):
            xo = torch.zeros(m, k, device="cuda", dtype=torch.bfloat16)
            cols = (torch.randint(0, k // 64, (m,), device="cuda") * 64 + (b * 16 + torch.arange(m, device="cuda")) % 64)
            xo[torch.arange(m), cols] = 1.0
            o = run(layer, xo, cfg)
            want = w[layer][:, cols].t()
            bad += int((o != want).sum())
        print(f"{layer} m={m} cfg={cfg}: rel={rel:.2e} onehot_mismatch={bad}", flush=True)

    print("--- paired timing: submission.kernel_fn (A) vs zip (B), cold L2, overhead 4.0")
    state = sub.prepare({k: v.clone() for k, v in w.items()})
    for layer, m, cfg, n in plan:
        if layer not in only:
            continue
        data = reference.generate_input(m, layer, 5)
        x = data["x"].clone()
        xs = {"x": x, "config": data["config"]}
        ta, tb, r = bench_pair(lambda: sub.kernel_fn(xs, state), lambda: run(layer, x, cfg), x, x, flush, args.reps)
        print(f"{layer:9} m={m:3d} cfg={cfg}: base {ta:6.2f} zip {tb:6.2f}  delta {tb-ta:+.2f} us  ratio {r:.3f}", flush=True)

    print("--- decode microbench: graph of 20 launches, L2-resident weights (wrap 2 MB) vs streamed")
    for layer, m, cfg, n in plan[:1] + plan[3:4]:
        data = reference.generate_input(m, layer, 5)
        x = data["x"].clone()
        for wrap in (0, (2 << 20) // 16 // 384 * 384):
            run(layer, x, cfg, wrap=wrap)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(20):
                    run(layer, x, cfg, wrap=wrap)
            g.replay(); torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); g.replay(); e1.record(); e1.synchronize()
            print(f"{layer} m={m} cfg={cfg} wrap={wrap}: {e0.elapsed_time(e1)*1e3/20:.2f} us/launch (back-to-back graph)", flush=True)


if __name__ == "__main__":
    main()
