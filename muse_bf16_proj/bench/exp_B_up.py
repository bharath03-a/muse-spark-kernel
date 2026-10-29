"""[B] z9 with UP units per stage (bigger bulk ops). exp_B_up.py [--compile]"""
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
rep("template <int RB, int NT, int W, int CS, int D, int MINB>", "template <int RB, int NT, int W, int CS, int D, int MINB, int UP = 1>", 2)
rep("constexpr uint32_t SLOT = RB * 1536, XB = BT * 128, XOFF", "constexpr uint32_t SLOT0 = RB * 1536, SLOT = SLOT0 * UP, XB0 = BT * 128, XB = XB0 * UP, XOFF")
rep("constexpr size_t XOFF = ((W * D * RB * 1536 + 1023) / 1024) * 1024, RING = XOFF + W * D * BT * 128;",
    "constexpr size_t XOFF = ((W * D * RB * 1536 * UP + 1023) / 1024) * 1024, RING = XOFF + W * D * BT * 128 * UP;")
rep("(K / 64) >= CS * W,", "(K / 64) % (CS * W * UP) == 0,")
rep("auto kern = z9_kernel<RB, NT, W, CS, D, MINB>;", "auto kern = z9_kernel<RB, NT, W, CS, D, MINB, UP>;")
rep("    const unsigned char* gsrc = wp + ((size_t)rg * U + u0) * SLOT;", "    const int NS = NU / UP;\n    const unsigned char* gsrc = wp + ((size_t)rg * U + u0) * SLOT0;")
rep("        if (d < NU) {\n          z7_expect_tx(&bar[d], tx);\n          z7_bulk(ring + d * SLOT, gsrc + (size_t)d * SLOT, SLOT, &bar[d], pol);\n          z10_tma_x(xring + d * XB, &xmap, (u0 + d) * 64, &bar[d]);",
    "        if (d < NS) {\n          z7_expect_tx(&bar[d], tx);\n          z7_bulk(ring + d * SLOT, gsrc + (size_t)d * SLOT, SLOT, &bar[d], pol);\n          for (int q = 0; q < UP; ++q) z10_tma_x(xring + d * XB + q * XB0, &xmap, (u0 + d * UP + q) * 64, &bar[d]);")
rep("    for (int u = 0; u < NU; ++u) {", "    for (int u = 0; u < NS; ++u) {")
rep("      const uint4* sp = reinterpret_cast<const uint4*>(ring + d * SLOT) + lane;\n      const unsigned char* xst = xring + d * XB;",
    "#pragma unroll 1\n      for (int q = 0; q < UP; ++q) {\n      const uint4* sp = reinterpret_cast<const uint4*>(ring + d * SLOT + q * SLOT0) + lane;\n      const unsigned char* xst = xring + d * XB + q * XB0;")
rep("      __syncwarp();\n      if (lane == 0 && u + D < NU) {", "      }\n      __syncwarp();\n      if (lane == 0 && u + D < NS) {")
rep("        z7_bulk(ring + d * SLOT, gsrc + (size_t)(u + D) * SLOT, SLOT, &bar[d], pol);\n        z10_tma_x(xring + d * XB, &xmap, (u0 + u + D) * 64, &bar[d]);",
    "        z7_bulk(ring + d * SLOT, gsrc + (size_t)(u + D) * SLOT, SLOT, &bar[d], pol);\n        for (int q = 0; q < UP; ++q) z10_tma_x(xring + d * XB + q * XB0, &xmap, (u0 + (u + D) * UP + q) * 64, &bar[d]);")
NEW = {  # id: RB,NT,W,CS,D,MINB,UP
    301: (2, 1, 8, 1, 2, 1, 2), 302: (2, 1, 8, 1, 1, 2, 2), 303: (2, 1, 8, 1, 1, 1, 4), 305: (2, 1, 4, 1, 2, 1, 4),
    306: (2, 1, 4, 1, 1, 2, 4), 307: (2, 1, 4, 1, 2, 2, 2), 308: (2, 1, 4, 1, 1, 1, 8), 309: (2, 1, 8, 1, 2, 2, 1),
    311: (2, 2, 8, 1, 1, 2, 2), 312: (2, 2, 8, 1, 2, 1, 2), 313: (2, 2, 4, 1, 2, 2, 2), 314: (2, 2, 4, 1, 2, 1, 4),
    315: (2, 2, 4, 1, 1, 2, 4), 316: (2, 2, 8, 1, 1, 1, 4),
}
rep("void launch_z7(torch::Tensor x,",
    "#define Z9_CASEU(ID, RB_, NT_, W_, CS_, D_, MINB_, UP_) \\\n  if (id == ID) { z9_launch<RB_, NT_, W_, CS_, D_, MINB_, UP_>(x, wp, out, esc, scale); return; }\nvoid launch_z7(torch::Tensor x,")
rep('  TORCH_CHECK(false, "zip: config id not compiled ", id);',
    " ".join(f"Z9_CASEU({i}, {', '.join(map(str, c))})" for i, c in NEW.items()) + '\n  TORCH_CHECK(false, "zip: config id not compiled ", id);')
tag = hashlib.sha1((S + " ".join(C.CUDA_FLAGS)).encode()).hexdigest()[:10]
from torch.utils.cpp_extension import load_inline
ext = load_inline(name=f"muse_zipU_{tag}", cpp_sources=C.ZIP_CPP, cuda_sources=S, functions=["launch_z7"],
                  extra_cuda_cflags=C.CUDA_FLAGS, extra_cflags=["-O3"], extra_ldflags=C.CUDA_LDFLAGS)
print("built", tag, flush=True)
if "--compile" in sys.argv:
    os._exit(0)
W = weights()
gen = torch.Generator(device="cuda").manual_seed(0)
flush = torch.zeros(64 * 2**20, device="cuda", dtype=torch.float32)
BASE = {25: 2, 121: 2, 137: 2, 28: 2, 27: 2}
packs = {}
def run_set(layer, m, base_ids):
    w = W[layer].cuda().contiguous()
    p = packs.setdefault(layer, C._zip_pack(w, 2)); p["ext"] = ext
    x = torch.randn(m, w.shape[1], device="cuda", generator=gen).to(torch.bfloat16)
    res = []
    for cid in base_ids + [i for i, c in NEW.items() if m <= c[1] * 8 and (m > 8 or c[1] == 1) or (m <= 8 and c[1] == 1)]:
        try:
            res.append((C._graph_cold_us(lambda: C._zip_launch(x, w.shape[0], cid, p), flush, 60), cid))
        except Exception as e:
            print(f"  {layer} m{m} id{cid} ERR {str(e)[:70]}", flush=True)
    res.sort()
    print(f"{layer} m{m}: " + " ".join(f"{c}:{t:.2f}" for t, c in res[:8]) + " | base " +
          " ".join(f"{c}:{t:.2f}" for t, c in res if c in base_ids), flush=True)
    for t, cid in res[:2]:
        try:
            C._zip_check((m, m, cid, p), w, gen)
        except Exception as e:
            print(f"  CHECK FAIL {layer} m{m} id{cid}: {e}", flush=True)
run_set("pooler", 1, [25]); run_set("qkv_gate", 1, [121]); run_set("qkv_gate", 8, [121])
run_set("pooler", 16, [137]); run_set("unpooler", 16, [28])
print("DONE", flush=True)
os._exit(0)
