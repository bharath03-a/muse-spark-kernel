"""[D] E1 ingress-cap F, E2 z9 no-x-TMA kill test, E3 unpooler-slice m8 alternatives. exp_D.py [--compile] [e1|e2|e3 ...]
One harness T(): flush by reading 256MB, CUDA graph replay, median 60; raw and raw-empty-graph."""
from __future__ import annotations
import hashlib, os, statistics, sys, threading
import torch
from common import load_module, weights
from torch.utils.cpp_extension import load_inline

threading.Timer(240, lambda: (print("WATCHDOG", flush=True), os._exit(0))).start()
which = [a for a in sys.argv[1:] if a in ("e1", "e2", "e3")] or ["e1", "e2", "e3"]
COMPILE = "--compile" in sys.argv
FL = torch.empty(256 * 2**20, dtype=torch.uint8, device="cuda")
D = torch.zeros(1, device="cuda")

def T(fn, reps=60):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    ts = []
    for _ in range(reps):
        FL.sum()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)
    return statistics.median(ts)

EMPTY = T(lambda: D.add_(1)) if not COMPILE else 0.0
print(f"empty graph {EMPTY:.2f}", flush=True)
def fmt(t): return f"{t:.2f}/{t - EMPTY:.2f}"

E1_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
template <int U>
__global__ void __launch_bounds__(256) k(const uint4* s, long ns, const uint4* r, long nr, int q, int* sink) {
  long tid = (long)blockIdx.x * 256 + threadIdx.x, st = (long)gridDim.x * 256;
  unsigned acc = 0; long e = tid;
  for (long i = tid; i < ns; i += st * U) {
    uint4 v[U];
    #pragma unroll
    for (int u = 0; u < U; ++u) { long j = i + (long)u * st; if (j < ns) asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(v[u].x),"=r"(v[u].y),"=r"(v[u].z),"=r"(v[u].w):"l"(s + j)); else v[u] = make_uint4(0,0,0,0); }
    for (int b = 0; b < q; b += 8) {
      uint4 w[8];
      #pragma unroll
      for (int u = 0; u < 8; ++u) { long j = (e + (long)u * st) % nr; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(w[u].x),"=r"(w[u].y),"=r"(w[u].z),"=r"(w[u].w):"l"(r + j)); }
      e += 8 * st;
      #pragma unroll
      for (int u = 0; u < 8; ++u) acc ^= w[u].x ^ w[u].y ^ w[u].z ^ w[u].w;
    }
    #pragma unroll
    for (int u = 0; u < U; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  if (acc == 0x9e3779b9u) sink[0] = acc;
}
void run(torch::Tensor s, torch::Tensor r, int64_t blocks, int64_t q, torch::Tensor sink) {
  k<8><<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>((const uint4*)s.data_ptr(), s.numel() / 16, (const uint4*)r.data_ptr(), r.numel() / 16, (int)q, sink.data_ptr<int>());
}
"""
if "e1" in which:
    ext1 = load_inline(name="muse_d_e1", cpp_sources="void run(torch::Tensor s, torch::Tensor r, int64_t blocks, int64_t q, torch::Tensor sink);",
                       cuda_sources=E1_SRC, functions=["run"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
    print("e1 built", flush=True)
if COMPILE and "e1" in which and len(which) == 1:
    os._exit(0)

C = load_module("checkpoints/cp13_B_check1.415.py", "cp13")
S = C.ZIP_SRC
def rep(a, b, cnt=1):
    global S
    assert S.count(a) == cnt, (a, S.count(a))
    S = S.replace(a, b)
NEW = {210: (4, 4, 8, 2, 2, 1), 212: (4, 4, 8, 1, 2, 1), 215: (2, 4, 8, 1, 2, 2), 216: (2, 4, 8, 2, 2, 2)}
if "e2" in which:
    rep("int xstride, float scale) {", "int xstride, float scale, int abl) {")
    rep("double scale) {\n  constexpr int BT", "double scale, int abl) {\n  constexpr int BT")
    rep("(float)scale));", "(float)scale, abl));")
    rep("(x, wp, out, esc, scale); return; }", "(x, wp, out, esc, scale, (int)ilv); return; }")
    rep("const uint32_t tx = SLOT + XB;", "const uint32_t tx = SLOT + ((abl & 8) ? 0 : XB);")
    rep("          z10_tma_x(xring + d * XB, &xmap, (u0 + d) * 64, &bar[d]);", "          if (!(abl & 8)) z10_tma_x(xring + d * XB, &xmap, (u0 + d) * 64, &bar[d]);")
    rep("        z10_tma_x(xring + d * XB, &xmap, (u0 + u + D) * 64, &bar[d]);", "        if (!(abl & 8)) z10_tma_x(xring + d * XB, &xmap, (u0 + u + D) * 64, &bar[d]);")
    rep("  TORCH_CHECK(false, \"zip: config id not compiled \", id);", " ".join(f"Z9_CASE({i}, {', '.join(map(str, c))})" for i, c in NEW.items()) + "\n  TORCH_CHECK(false, \"zip: config id not compiled \", id);")
    tag = hashlib.sha1((S + " ".join(C.CUDA_FLAGS)).encode()).hexdigest()[:10]
    ext2 = load_inline(name=f"muse_zipD_{tag}", cpp_sources=C.ZIP_CPP, cuda_sources=S, functions=["launch_z7"],
                       extra_cuda_cflags=C.CUDA_FLAGS, extra_cflags=["-O3"], extra_ldflags=C.CUDA_LDFLAGS)
    print("e2 built", flush=True)
if COMPILE:
    os._exit(0)

gen = torch.Generator(device="cuda").manual_seed(0)
if "e1" in which:
    sink = torch.zeros(4, device="cuda", dtype=torch.int32)
    s = torch.randint(0, 255, (50_331_648,), device="cuda", dtype=torch.uint8)
    r = torch.randint(0, 255, (2 * 2**20,), device="cuda", dtype=torch.uint8)
    nb = 132 * 2
    st = nb * 256
    for R in (0, 4, 8, 16, 32, 64):
        q = (R * (2 * 2**20 // 16)) // (st * max(1, (s.numel() // 16 + st * 8 - 1) // (st * 8)))
        q = (q + 7) // 8 * 8 if R else 0
        t = T(lambda: ext1.run(s, r, nb, q, sink))
        tot = (s.numel() + (q * st * ((s.numel() // 16 + st * 8 - 1) // (st * 8)) * 16 if R else 0)) / 1e6
        print(f"E1 R={R} q={q} total_MB={tot:.1f} raw/net {fmt(t)} ingress {tot / 1e6 / ((t - EMPTY) * 1e-6):.2f} TB/s", flush=True)

if "e2" in which:
    Wt = weights()
    for layer, m, ids in (("pooler", 32, (210, 212, 215, 216)), ("pooler", 16, (137, 27)), ("pooler", 1, (25,)), ("unpooler", 16, (28,))):
        w = Wt[layer].cuda().contiguous()
        x = torch.randn(m, w.shape[1], device="cuda", generator=gen).to(torch.bfloat16)
        out = torch.empty(m, w.shape[0], device="cuda", dtype=torch.bfloat16)
        for cid in ids:
            rb = {**NEW, 137: (2,), 27: (2,), 25: (2,), 28: (2,)}[cid][0]
            pk = C._zip_pack(w, rb)
            row = []
            for abl in (0, 8, 0, 8):
                try:
                    row.append(fmt(T(lambda: ext2.launch_z7(x, pk["buf"], out, pk["esc"], pk["scale"], cid, abl))))
                except Exception as e:
                    row.append("ERR"); print(str(e)[:100], flush=True)
            print(f"E2 {layer} m{m} id{cid} raw/net full {row[0]} | noX {row[1]} | full {row[2]} | noX {row[3]}", flush=True)

_W, _ST = [None], [None]
def _b(x, tile):
    o = torch.empty(x.shape[0], 1024, device="cuda", dtype=torch.bfloat16)
    C._bf16_launch(x, _W[0], o, _ST[0], "unpooler_slice", tile)
    return o
if "e3" in which:
    ext = C._build_cuda()
    Wt = weights()
    w = Wt["unpooler"][:C.UNPOOL_SLICE].cuda().contiguous()
    st = {"ws": torch.empty(C.WS_FLOATS, device="cuda", dtype=torch.float32), "cnt": torch.zeros(C.COUNTERS, device="cuda", dtype=torch.int32), "ext": ext, "desc": {}}
    _W[0], _ST[0] = w, st
    print("E3 dispatch m8: CUDA_TILES", C.CUDA_TILES["unpooler_slice"], "-> wgmma bn64 bt8 bk128 stages8 csize4 => grid (1024/64)*4=64 CTAs; m1 = triton _bf16_gemv 512 CTAs (n/2)", flush=True)
    for m in (1, 8):
        x = torch.randn(m, w.shape[1], device="cuda", generator=gen).to(torch.bfloat16)
        ref = (x.float() @ w.float().t())
        def chk(o): return f"{((o.float() - ref).norm() / ref.norm()).item():.0e}"
        cands = []
        if m == 1:
            cands.append(("m1 triton fma bn2 bk2048", lambda x=x: _b(x, C.Tile(1, 2, 2048, 1, 1, 8, 2, "fma"))))
        cands.append(("cur wgmma 64/8/128/8/c4", lambda x=x: C._cuda_run(st, x, w, 1024, torch.bfloat16, "wgmma", (64, 8, 128, 8, 4))))
        cands.append(("gemv R2 MT2 P2 (512 CTA)", lambda x=x: C._cuda_run(st, x, w, 1024, torch.bfloat16, "gemv", (2, 2, 2))))
        cands.append(("gemv R4 MT4 P2 (256 CTA)", lambda x=x: C._cuda_run(st, x, w, 1024, torch.bfloat16, "gemv", (4, 4, 2))))
        for bn, bk, sp, wp, stg in ((64, 128, 8, 4, 4), (32, 128, 8, 4, 4), (32, 256, 4, 4, 4), (32, 128, 16, 4, 4), (16, 128, 8, 4, 4), (32, 256, 8, 4, 3)):
            cands.append((f"triton gemm bn{bn} bk{bk} S{sp} w{wp} st{stg}", lambda x=x, t=C.Tile(8, bn, bk, 16, sp, wp, stg): _b(x, t)))
        for name, fn in cands:
            try:
                o = fn(); torch.cuda.synchronize()
                print(f"E3 m{m} {name:32s} {fmt(T(fn))} err {chk(o)}", flush=True)
            except Exception as e:
                print(f"E3 m{m} {name} ERR {str(e)[:90]}", flush=True)
