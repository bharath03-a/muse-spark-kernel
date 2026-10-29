"""Floors: cp5 at small m (bandwidth-bound) and a plain full-W read, vs m=128/256."""
import statistics
import torch
from common import load_module, reference, weights
from harness import FLUSH_BYTES, time_replay

OV = 4.0


def graph_of(fn):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def main():
    w = weights()
    sub = load_module("submission.py", "sub")
    st = sub.prepare({k: v.clone() for k, v in w.items()})
    flush = torch.empty(FLUSH_BYTES // 4, device="cuda", dtype=torch.float32).normal_()
    ents = []
    for layer in ("o", "unpooler"):
        W = st[layer]
        for m in (1, 16, 32, 128, 256):
            n, k = W.shape
            x = reference.generate_input(m, layer, 3)["x"].clone()
            d = {"x": x, "config": {"m": m, "n": n, "k": k, "layer": layer}}
            ents.append((f"{layer} cp5 m={m}", graph_of(lambda d=d: sub.kernel_fn(d, st))))
    ts = {n: [] for n, _ in ents}
    for r in range(60):
        for n, g in (ents if r % 2 else ents[::-1]):
            ts[n].append(time_replay(g, flush) - OV)
    for n, _ in ents:
        print(f"{n:22s} {statistics.median(ts[n]):7.2f} us")


main()
