"""[B] z9 ablation: abl modes 0 full,1 loads-only,2 +decode,3 +mma (no fix/epilogue),4 full w/o fix warp. --compile"""
from __future__ import annotations
import hashlib, os, sys, threading
import torch
from common import load_module, weights

threading.Timer(240, lambda: (print("WATCHDOG", flush=True), os._exit(0))).start()
C = load_module("checkpoints/cp13_B_check1.415.py", "cp13")
S = C.ZIP_SRC
def rep(a, b, cnt=1):
    global S
    assert S.count(a) == cnt, (a, S.count(a))
    S = S.replace(a, b)
rep("int xstride, float scale) {", "int xstride, float scale, int abl) {")
rep("double scale) {\n  constexpr int BT", "double scale, int abl) {\n  constexpr int BT")
rep("(float)scale));", "(float)scale, abl));")
rep("(x, wp, out, esc, scale); return; }", "(x, wp, out, esc, scale, (int)ilv); return; }")
rep("      uint4 w0[RB], w1[RB], wb[RB], xc[NT][2];", "      if (abl != 1) {\n      uint4 w0[RB], w1[RB], wb[RB], xc[NT][2];")
rep("      __syncwarp();\n      if (lane == 0 && u + D < NU) {", "      }\n      __syncwarp();\n      if (lane == 0 && u + D < NU) {")
rep("            z7_mma(acc[rb][nt], r0, r1, r2, r3, xw[(j & 1) * 2], xw[(j & 1) * 2 + 1]);",
    "            if (abl == 2) acc[rb][nt][0] += __uint_as_float(r0 ^ r1 ^ r2 ^ r3 ^ xw[(j & 1) * 2]);\n            else z7_mma(acc[rb][nt], r0, r1, r2, r3, xw[(j & 1) * 2], xw[(j & 1) * 2 + 1]);")
rep("} else if (rank == 0) {", "} else if (rank == 0 && abl == 0) {")
rep("  __syncthreads();\n  if (warp < W) {\n    float* dst = red", "  __syncthreads();\n  if (abl >= 1 && abl <= 3) { if (tid == 0) out[0] = __float2bfloat16(acc[0][0][0]); return; }\n  if (warp < W) {\n    float* dst = red")
tag = hashlib.sha1((S + " ".join(C.CUDA_FLAGS)).encode()).hexdigest()[:10]
from torch.utils.cpp_extension import load_inline
ext = load_inline(name=f"muse_zipA_{tag}", cpp_sources=C.ZIP_CPP, cuda_sources=S, functions=["launch_z7"],
                  extra_cuda_cflags=C.CUDA_FLAGS, extra_cflags=["-O3"], extra_ldflags=C.CUDA_LDFLAGS)
print("built", tag, flush=True)
if "--compile" in sys.argv:
    os._exit(0)
W = weights()
gen = torch.Generator(device="cuda").manual_seed(0)
flush = torch.zeros(64 * 2**20, device="cuda", dtype=torch.float32)
ov = C._graph_cold_us(lambda: flush.add_(0), flush, 60)  # near-empty graph
print(f"empty-ish graph {ov:.2f}", flush=True)
for layer, rb, cid, m in (("pooler", 2, 25, 1), ("pooler", 2, 25, 8), ("qkv_gate", 2, 121, 1), ("qkv_gate", 2, 121, 8),
                          ("pooler", 2, 137, 16)):
    w = W[layer].cuda().contiguous()
    p = C._zip_pack(w, rb)
    x = torch.randn(m, w.shape[1], device="cuda", generator=gen).to(torch.bfloat16)
    out = torch.empty(m, w.shape[0], device="cuda", dtype=torch.bfloat16)
    nb = p["buf"].numel() / 1e6
    ref = p["buf"].view(torch.int64)
    tr = C._graph_cold_us(lambda: ref.sum(), flush, 60)
    row = []
    for abl in range(5):
        t = C._graph_cold_us(lambda: ext.launch_z7(x, p["buf"], out, p["esc"], p["scale"], cid, abl), flush, 60)
        row.append(f"abl{abl}:{t:.2f}")
    print(f"{layer} m{m} id{cid} {nb:.1f}MB torch.sum-ref {tr:.2f} | " + " ".join(row), flush=True)
print("DONE", flush=True)
os._exit(0)
