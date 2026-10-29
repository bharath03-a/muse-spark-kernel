"""Test checkpoints/cp4_gluon_wgmma_v3.py against checkpoints/cp3_gluon_v2.py."""
import math
import statistics
import sys
import time
import torch

from common import load_module, reference, weights
from harness import FLUSH_BYTES, time_replay

BASE_CP = "checkpoints/cp3_gluon_v2.py"
CAND_CP = "checkpoints/cp4_gluon_wgmma_v3.py"
OVERHEAD_US = 4.0

SCORED_CASES = [
    # decode 1
    ("qkv_gate", 1), ("o", 1), ("pooler", 1), ("unpooler", 1), ("router", 1),
    # verify 1
    ("qkv_gate", 8), ("o", 8), ("pooler", 8), ("unpooler", 8), ("router", 8),
    # decode 16
    ("qkv_gate", 16), ("o", 16), ("pooler", 16), ("unpooler", 16), ("router", 16),
    # decode 32
    ("qkv_gate", 32), ("o", 32), ("pooler", 32), ("unpooler", 32), ("router", 32),
    # verify 16
    ("qkv_gate", 128), ("o", 128), ("pooler", 128), ("unpooler", 128), ("router", 128),
    # verify 32
    ("qkv_gate", 256), ("o", 256), ("pooler", 256), ("unpooler", 256), ("router", 256),
]

NEARBY_M = [2, 5, 7, 9, 12, 17, 24, 31, 33, 100, 130, 200, 512]


def graph_of(fn):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    torch.cuda.synchronize()
    return g, out


def main():
    reps = 30
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        reps = int(sys.argv[1])

    print(f"Loading base: {BASE_CP} and cand: {CAND_CP}...")
    t0 = time.time()
    w = weights()
    base_mod = load_module(BASE_CP, "base")
    cand_mod = load_module(CAND_CP, "cand")

    print(f"Preparing base state...")
    base_state = base_mod.prepare({k: v.clone() for k, v in w.items()})
    print(f"Preparing cand state...")
    cand_state = cand_mod.prepare({k: v.clone() for k, v in w.items()})
    print(f"Prepare done in {time.time() - t0:.1f}s")

    # 1. Correctness Checks
    print("\n--- Correctness Checks (Scored Cases) ---")
    all_ok = True
    for layer, m in SCORED_CASES:
        for seed, dist in [(1, "normal"), (2, "outliers")]:
            data = reference.generate_input(m, layer, seed, dist=dist)
            out_c = cand_mod.kernel_fn(data, cand_state)
            torch.cuda.synchronize()
            ok, msg = reference.check_implementation(data, out_c, w)
            if not ok:
                print(f"FAIL: {layer} m={m} dist={dist}: {msg}")
                all_ok = False
                break
        if not all_ok:
            break

    print("\n--- Correctness Checks (Nearby m values) ---")
    for m in NEARBY_M:
        for layer in ("router", "unpooler", "qkv_gate", "pooler", "o"):
            data = reference.generate_input(m, layer, 42)
            out_c = cand_mod.kernel_fn(data, cand_state)
            torch.cuda.synchronize()
            ok, msg = reference.check_implementation(data, out_c, w)
            if not ok:
                print(f"FAIL NEARBY: {layer} m={m}: {msg}")
                all_ok = False
                break
        if not all_ok:
            break

    if not all_ok:
        print("\nABORTING TIMING: Correctness check failed!")
        sys.exit(1)
    print("ALL 30 SCORED + 65 NEARBY CORRECTNESS CHECKS PASSED!\n")

    # 2. Paired CUDA Graph Timing
    print(f"--- Paired CUDA Graph Timing ({reps} reps, cold-L2) ---")
    flush = torch.empty(FLUSH_BYTES // 4, device="cuda", dtype=torch.float32).normal_()

    results = []
    base_load_times = {"decode 1": 0.0, "verify 1": 0.0, "decode 16": 0.0, "decode 32": 0.0, "verify 16": 0.0, "verify 32": 0.0}
    cand_load_times = {"decode 1": 0.0, "verify 1": 0.0, "decode 16": 0.0, "decode 32": 0.0, "verify 16": 0.0, "verify 32": 0.0}

    def get_load(m):
        if m == 1: return "decode 1"
        if m == 8: return "verify 1"
        if m == 16: return "decode 16"
        if m == 32: return "decode 32"
        if m == 128: return "verify 16"
        if m == 256: return "verify 32"
        return "other"

    for layer, m in SCORED_CASES:
        data = reference.generate_input(m, layer, 100)
        # Capture graphs
        g_base, _ = graph_of(lambda: base_mod.kernel_fn(data, base_state))
        g_cand, _ = graph_of(lambda: cand_mod.kernel_fn(data, cand_state))

        times_b, times_c = [], []
        for r in range(reps):
            order = [("base", g_base), ("cand", g_cand)] if r % 2 == 0 else [("cand", g_cand), ("base", g_base)]
            for name, g in order:
                t = time_replay(g, flush) - OVERHEAD_US
                if name == "base":
                    times_b.append(t)
                else:
                    times_c.append(t)

        tb = statistics.median(times_b)
        tc = statistics.median(times_c)
        delta = tc - tb
        pct = (delta / tb) * 100
        load = get_load(m)
        base_load_times[load] += tb * 62
        cand_load_times[load] += tc * 62
        tag = "[WIN]" if delta < -0.1 else ("[LOSE]" if delta > 0.1 else "[TIE]")
        print(f"{layer:<12} m={m:<3} ({load:<9}): base={tb:6.2f} us | cand={tc:6.2f} us | delta={delta:+6.2f} us ({pct:+5.1f}%) {tag}")
        results.append((layer, m, tb, tc, delta))

    print("\n--- Workload Aggregate Times (62 calls per shape) ---")
    geomean_base = 1.0
    geomean_cand = 1.0
    num_loads = len(base_load_times)

    for load in sorted(base_load_times.keys()):
        tb_load = base_load_times[load]
        tc_load = cand_load_times[load]
        delta_load = tc_load - tb_load
        speedup_cand_vs_base = tb_load / tc_load
        print(f"Load {load:<9}: base={tb_load:8.2f} us | cand={tc_load:8.2f} us | delta={delta_load:+8.2f} us ({speedup_cand_vs_base:.4f}x)")
        geomean_base *= tb_load
        geomean_cand *= tc_load

    geomean_base = geomean_base ** (1.0 / num_loads)
    geomean_cand = geomean_cand ** (1.0 / num_loads)
    geomean_speedup = geomean_base / geomean_cand
    cp3_known_score = 1.2807
    projected_score = cp3_known_score * geomean_speedup

    print(f"\n========================================================")
    print(f"Geomean Step Time : base={geomean_base:.2f} us | cand={geomean_cand:.2f} us")
    print(f"Speedup vs cp3    : {geomean_speedup:.4f}x")
    print(f"Projected Score   : {projected_score:.4f}x (cp3 was 1.2807x)")
    print(f"========================================================")

if __name__ == "__main__":
    main()
