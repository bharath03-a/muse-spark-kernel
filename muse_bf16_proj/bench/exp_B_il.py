"""[B] loads-only variants: abl bit1 loads-only, bit2 interleaved addr, bit8 no x. --compile"""
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
rep("      uint4 w0[RB], w1[RB], wb[RB], xc[NT][2];", "      if (!(abl & 1)) {\n      uint4 w0[RB], w1[RB], wb[RB], xc[NT][2];")
rep("      __syncwarp();\n      if (lane == 0 && u + D < NU) {", "      }\n      __syncwarp();\n      if (lane == 0 && u + D < NU) {")
rep("            z7_mma(acc[rb][nt], r0, r1, r2, r3, xw[(j & 1) * 2], xw[(j & 1) * 2 + 1]);",
    "            if (abl == 2) acc[rb][nt][0] += __uint_as_float(r0 ^ r1 ^ r2 ^ r3 ^ xw[(j & 1) * 2]);\n            else z7_mma(acc[rb][nt], r0, r1, r2, r3, xw[(j & 1) * 2], xw[(j & 1) * 2 + 1]);")
rep("} else if (rank == 0) {", "} else if (rank == 0 && abl == 0) {")
rep("  __syncthreads();\n  if (warp < W) {\n    float* dst = red", "  __syncthreads();\n  if (abl & 1) { if (tid == 0) out[0] = __float2bfloat16(acc[0][0][0]); return; }\n  if (warp < W) {\n    float* dst = red")

rep("    const unsigned char* gsrc = wp", "    const size_t GW = (size_t)gridDim.x * W, gw = (size_t)blockIdx.x * W + warp;\n    const unsigned char* gsrc = wp")
rep("const uint32_t tx = SLOT + XB;", "const uint32_t tx = SLOT + ((abl & 8) ? 0 : XB);")
rep("gsrc + (size_t)d * SLOT, SLOT", "(abl & 2 ? wp + ((size_t)d * GW + gw) * SLOT : gsrc + (size_t)d * SLOT), SLOT")
rep("gsrc + (size_t)(u + D) * SLOT, SLOT", "(abl & 2 ? wp + ((size_t)(u + D) * GW + gw) * SLOT : gsrc + (size_t)(u + D) * SLOT), SLOT")
rep("          z10_tma_x(xring + d * XB, &xmap, (u0 + d) * 64, &bar[d]);", "          if (!(abl & 8)) z10_tma_x(xring + d * XB, &xmap, (u0 + d) * 64, &bar[d]);")
rep("        z10_tma_x(xring + d * XB, &xmap, (u0 + u + D) * 64, &bar[d]);", "        if (!(abl & 8)) z10_tma_x(xring + d * XB, &xmap, (u0 + u + D) * 64, &bar[d]);")
NEW = {400: (4, 1, 8, 1, 2, 1), 401: (4, 1, 8, 1, 3, 1), 403: (8, 1, 8, 2, 1, 1), 404: (2, 1, 8, 1, 3, 2), 405: (2, 1, 16, 1, 1, 2)}
rep("  TORCH_CHECK(false, \"zip: config id not compiled \", id);", " ".join(f"Z9_CASE({i}, {', '.join(map(str, c))})" for i, c in NEW.items()) + "\n  TORCH_CHECK(false, \"zip: config id not compiled \", id);")
tag = hashlib.sha1((S + " ".join(C.CUDA_FLAGS)).encode()).hexdigest()[:10]
from torch.utils.cpp_extension import load_inline
ext = load_inline(name=f"muse_zipA_{tag}", cpp_sources=C.ZIP_CPP, cuda_sources=S, functions=["launch_z7"],
                  extra_cuda_cflags=C.CUDA_FLAGS, extra_cflags=["-O3"], extra_ldflags=C.CUDA_LDFLAGS)
print("built", tag, flush=True)
if "--compile" in sys.argv:
    os._exit(0)
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bw_sweep.py")).read()
ns = {}
exec(src[:src.index("ap = argparse")], ns)
ref = load_inline(name="bw_sweep_v1", cpp_sources=ns["CPP"], cuda_sources=ns["SRC"], functions=["ldg", "bulk", "tma"],
                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], extra_ldflags=["-lcuda"])
SM = torch.cuda.get_device_properties(0).multi_processor_count
sink = torch.zeros(4, device="cuda", dtype=torch.int32)
Wt = weights()
gen = torch.Generator(device="cuda").manual_seed(0)
flush = torch.zeros(64 * 2**20, device="cuda", dtype=torch.float32)
T = lambda f: C._graph_cold_us(f, flush, 60)
for layer, ids, m in (("pooler", (25, 400, 401, 403, 404, 405), 1), ("qkv_gate", (121, 110), 1)):
    w = Wt[layer].cuda().contiguous()
    p = C._zip_pack(w, {400: 4, 401: 4, 403: 8}.get(0, 2)); p["ext"] = ext
    x = torch.randn(m, w.shape[1], device="cuda", generator=gen).to(torch.bfloat16)
    out = torch.empty(m, w.shape[0], device="cuda", dtype=torch.bfloat16)
    buf = torch.randint(0, 255, (p["buf"].numel(),), device="cuda", dtype=torch.uint8)
    print(f"{layer} {p['buf'].numel()/1e6:.1f}MB  ref bulk32K d2 inter: {T(lambda: ref.bulk(buf, SM*2, 32768, 2, 1, 1)):.2f}  "
          f"ldg cps8 U16: {T(lambda: ref.ldg(buf, SM*8, 16, 0, sink)):.2f}", flush=True)
    for cid in ids:
        rb = {400: 4, 401: 4, 403: 8}.get(cid, 2)
        pk = C._zip_pack(w, rb); pk["ext"] = ext
        row = []
        for abl in (0, 1, 3, 9, 11):
            try:
                row.append(f"abl{abl}:{T(lambda: ext.launch_z7(x, pk['buf'], out, pk['esc'], pk['scale'], cid, abl)):.2f}")
            except Exception as e:
                row.append(f"abl{abl}:ERR"); print(str(e)[:80], flush=True)
        print(f"  id{cid} " + " ".join(row), flush=True)
print("DONE", flush=True)
os._exit(0)
