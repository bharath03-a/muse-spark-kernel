"""Sweep stream-K qkv_gate configs (cand_qkv.py) vs submission.py kernel_fn, paired cold-L2."""
import argparse
import statistics

import torch
import triton

from common import load_module, reference, weights
from harness import FLUSH_BYTES, time_replay
from exp_r4_sweep import graph_of

OVERHEAD = 4.0
# (P, STAGES, GRID)
CFGS = {  # (P, STAGES, GRID, DBG)
    128: [(2, 4, 132, 0), (2, 4, 132, 1), (2, 4, 132, 2), (1, 6, 132, 0), (1, 6, 132, 2), (1, 6, 108, 2)],
    256: [(2, 4, 132, 0)],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=50)
    a = ap.parse_args()
    w = weights()
    base = load_module("submission.py", "sub")
    cand = load_module("bench/cand_qkv.py", "cand")
    bst = base.prepare({k: v.clone() for k, v in w.items()})
    cst = cand.prepare({k: v.clone() for k, v in w.items()})
    wq = cst["qkv_gate"]
    flush = torch.empty(FLUSH_BYTES // 4, device="cuda", dtype=torch.float32).normal_()
    for m, cfgs in CFGS.items():
        data = reference.generate_input(m, "qkv_gate", 7)
        x = data["x"].clone()
        d = {"x": x, "config": data["config"]}
        ref = torch.nn.functional.linear(x, wq).float()
        keep = []
        ents = [("base", graph_of(lambda: base.kernel_fn(d, bst)))]
        ents.append(("cand_default", graph_of(lambda: cand.kernel_fn(d, cst))))
        for p, st, g, dbg in cfgs:
            tiles = (wq.shape[0] // (64 * p)) * triton.cdiv(m, 128)
            ws = torch.empty(tiles * 10 * 64 * p * 128, device="cuda", dtype=torch.float32)
            cnt = torch.zeros(2 * tiles, device="cuda", dtype=torch.int32)

            keep.append((ws, cnt))

            def f(p=p, st=st, g=g, dbg=dbg, ws=ws, cnt=cnt):
                out = torch.empty(m, wq.shape[0], device="cuda", dtype=torch.bfloat16)
                cst["ext"].launch_qkv_sk(x, wq, out, ws, cnt, p, 128, st, g, dbg)
                return out
            try:
                print("try", p, st, g, flush=True)
                gg = graph_of(f)
                gg[0].replay(); torch.cuda.synchronize()
                ents.append((f"sk P{p} S{st} G{g} dbg{dbg}", gg))
            except Exception as e:
                print("FAIL", p, st, g, str(e)[:200])
        res = []
        for name, (gr, out) in ents:
            gr.replay(); torch.cuda.synchronize()
            err = ((out.float() - ref).norm() / ref.norm()).item()
            f1 = out.clone(); gr.replay(); torch.cuda.synchronize()
            res.append([name, gr, out, err, torch.equal(f1, out), []])
        for r in range(a.reps):
            order = res[r % len(res):] + res[:r % len(res)]
            if r % 2:
                order = order[::-1]
            for e in order:
                e[5].append(time_replay(e[1], flush) - OVERHEAD)
        print(f"== qkv_gate m={m}")
        for e in sorted(res, key=lambda e: statistics.median(e[5])):
            d_ = statistics.median([x_ - y for x_, y in zip(e[5], res[0][5])])
            print(f"  {statistics.median(e[5]):7.2f} us d_vs_base {d_:+6.2f} err {e[3]:.1e} det {int(e[4])}  {e[0]}")


if __name__ == "__main__":
    main()
