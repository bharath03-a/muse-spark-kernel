"""R5: qkv_gate m=128/256 sweep vs submission.py (cold-L2 graph replay)."""
import argparse
import time

import torch

import exp_r4_sweep as r4
from common import load_module, weights
from harness import FLUSH_BYTES

from exp_r5_qkv_cands import CANDS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--only", default="")
    ap.add_argument("--sub", default="submission.py")
    a = ap.parse_args()
    t0 = time.time()
    sub = load_module(a.sub, "sub")
    state = sub.prepare({k: v.clone() for k, v in weights().items()})
    flush = torch.empty(FLUSH_BYTES // 4, device="cuda", dtype=torch.float32).normal_()
    for (layer, m), c in CANDS.items():
        if a.only and str(m) != a.only:
            continue
        r4.run_shape(sub, state, None, layer, m, c, a.reps, flush)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
