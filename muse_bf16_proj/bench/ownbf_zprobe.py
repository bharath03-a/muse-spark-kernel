"""OWN: zip config probe (accuracy + cold graph us) without full prepare."""
import sys, torch
from common import load_module, weights
mod = load_module(sys.argv[1], "c")
w = weights()
ext = mod._build_zip()
gen = torch.Generator(device="cuda").manual_seed(0)
flush = torch.zeros(64 * 2**20, device="cuda")
for layer, cases in (("qkv_gate", ((1, (25, 39)), (8, (25, 39)), (16, (204, 221, 222, 224)), (32, (213, 223)))),
                     ("pooler", ((16, (137, 221, 222, 224)), (32, (223,))))):
    W = w[layer].cuda().contiguous()
    p = mod._zip_pack(W, 2); mod._zip_verify(p, W); p["ext"] = ext
    n, k = W.shape
    for m, ids in cases:
        x = torch.randn(m, k, device="cuda", generator=gen).bfloat16()
        ref = x.double() @ W.double().t()
        for cid in ids:
            try:
                got = mod._zip_launch(x, n, cid, p).double()
                err = float((got - ref).norm() / ref.norm())
                t = mod._graph_cold_us(lambda: mod._zip_launch(x, n, cid, p), flush, 41)
                print(f"{layer} m{m} id{cid} err {err:.2e} us {t:.2f}", flush=True)
            except Exception as e:
                print(f"{layer} m{m} id{cid} FAIL {e}", flush=True)
