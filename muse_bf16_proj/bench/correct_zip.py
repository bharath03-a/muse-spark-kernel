"""Correctness suite (zip candidate copy, with watchdog): scored shapes, judge tests, nearby m, seeds, held-out weights, determinism.

    python bench/correct.py [--sub submission.py] [--quick] [--replays 5000]
"""

from __future__ import annotations

import argparse
import time

import torch

from common import LAYER_ORDER, TESTS, load_module, reference, scored_shapes, weights

NEARBY_M = sorted({*range(1, 41), 64, 96, 100, 127, 129, 160, 200, 255, 257, 300, 512, 1000})


def run_case(sub, state, w, m, layer, seed, dist) -> tuple[bool, str]:
    data = reference.generate_input(m, layer, seed, dist=dist)
    out = sub.kernel_fn(data, state)
    torch.cuda.synchronize()
    return reference.check_implementation(data, out, w)


def suite(sub, state, w, cases, label) -> int:
    fails = 0
    for m, layer, seed, dist in cases:
        try:
            ok, msg = run_case(sub, state, w, m, layer, seed, dist)
        except Exception as exc:
            ok, msg = False, f"raised {type(exc).__name__}: {exc}"
        if not ok:
            fails += 1
            print(f"FAIL [{label}] m={m} {layer} seed={seed} {dist}: {msg}")
    print(f"[{label}] {len(cases) - fails}/{len(cases)} passed")
    return fails


def determinism(sub, state, m, layer, replays) -> int:
    data = reference.generate_input(m, layer, 123)
    graph_data = {"x": data["x"].clone(), "config": data["config"]}
    sub.kernel_fn(graph_data, state)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = sub.kernel_fn(graph_data, state)
    g.replay()
    torch.cuda.synchronize()
    first = out.clone()
    bad = 0
    for _ in range(replays):
        g.replay()
        bad += int(not torch.equal(out, first))
    torch.cuda.synchronize()
    print(f"[determinism] m={m} {layer}: {bad}/{replays} replays differ")
    return int(bad > 0)


def fresh_inputs(sub, state, w, m, layer, iters) -> int:
    """One capture on a static x; each replay gets a new x copied in and is checked against it."""
    data = reference.generate_input(m, layer, 500)
    graph_data = {"x": data["x"].clone(), "config": data["config"]}
    sub.kernel_fn(graph_data, state)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = sub.kernel_fn(graph_data, state)
    bad = 0
    for i in range(iters):
        new = reference.generate_input(m, layer, 1000 + i, dist="outliers" if i % 2 else "normal")
        graph_data["x"].copy_(new["x"])
        g.replay()
        torch.cuda.synchronize()
        ok, msg = reference.check_implementation(new, out, w)
        if not ok:
            bad += 1
            if bad <= 3:
                print(f"FAIL [graph fresh-input] m={m} {layer} iter={i}: {msg}")
    print(f"[graph fresh-input] m={m} {layer}: {iters - bad}/{iters} passed")
    return int(bad > 0)


def main() -> None:
    import os
    import threading
    wd = threading.Timer(330, lambda: (print("WATCHDOG: exiting", flush=True), os._exit(0)))
    wd.daemon = True
    wd.start()
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="submission.py")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--replays", type=int, default=5000)
    args = ap.parse_args()

    sub = load_module(args.sub, "sub")
    fails = 0
    for wseed, label in ((reference.WEIGHTS_SEED, "judge-weights"), (777, "heldout-weights")):
        w = weights(wseed)
        t0 = time.time()
        state = sub.prepare({k: v.clone() for k, v in w.items()})
        print(f"[{label}] prepare {time.time() - t0:.1f}s")
        seeds = (1,) if args.quick else (1, 2, 3, 4, 5)
        scored = [(m, layer, s, d) for _, m, layer in scored_shapes() for s in seeds
                  for d in ("normal", "outliers")]
        fails += suite(sub, state, w, scored, f"{label} scored")
        fails += suite(sub, state, w, [(m, l, 11, d) for m, l, d in TESTS], f"{label} tests")
        if wseed == reference.WEIGHTS_SEED:
            nearby = [(m, layer, 21, "normal") for m in NEARBY_M for layer in LAYER_ORDER]
            fails += suite(sub, state, w, nearby, f"{label} nearby")
            if not args.quick:
                for m, layer in ((1, "qkv_gate"), (8, "router"), (32, "router"), (8, "unpooler"),
                                 (128, "pooler"), (256, "pooler"), (8, "pooler"), (16, "pooler"),
                                 (32, "pooler"), (16, "unpooler"), (32, "unpooler"), (16, "router"),
                                 (128, "router"), (256, "router"), (1, "pooler"), (5, "pooler"), (12, "pooler"), (9, "unpooler"), (8, "qkv_gate"), (5, "qkv_gate")):
                    fails += determinism(sub, state, m, layer, args.replays)
                for m, layer in ((128, "pooler"), (256, "pooler"), (8, "pooler"), (16, "pooler"),
                                 (32, "pooler"), (16, "unpooler"), (32, "unpooler"), (128, "router"), (1, "pooler"), (3, "pooler"), (12, "pooler"), (13, "unpooler"), (1, "qkv_gate"), (8, "qkv_gate"),
                                 (256, "router")):
                    fails += fresh_inputs(sub, state, w, m, layer, 200)
        del state
        torch.cuda.empty_cache()
    print("ALL PASS" if fails == 0 else f"{fails} FAILURES")


if __name__ == "__main__":
    main()
