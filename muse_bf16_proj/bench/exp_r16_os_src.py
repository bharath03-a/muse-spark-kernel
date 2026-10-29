# r16 one-shot small-matrix kernels: every CTA issues its whole W slice at kernel start (registers via ldg, or smem via
import hashlib

import torch

CUDA_FLAGS = ["-O3", "-gencode=arch=compute_90a,code=sm_90a"]
# cp.async.bulk chunks), x via per-row bulk copies, mma.sync m16n8k16 with W packed as 16x32 A-fragment records.
OS_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>

namespace cg = cooperative_groups;
using bf16 = __nv_bfloat16;

__device__ __forceinline__ uint32_t os_smem(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void os_cl_arrive_relaxed() { asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void os_cl_arrive_release() { asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void os_cl_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory"); }

__device__ __forceinline__ void os_bar_init(uint64_t* b) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(os_smem(b)) : "memory");
}
__device__ __forceinline__ void os_expect(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(os_smem(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void os_wait(uint64_t* b) {
  uint32_t ok = 0;
  do {
    asm volatile("{\n .reg .pred P1;\n mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], 0;\n selp.u32 %0, 1, 0, P1;\n}\n"
                 : "=r"(ok) : "r"(os_smem(b)) : "memory");
  } while (!ok);
}
__device__ __forceinline__ void os_bulk(void* dst, const void* src, uint32_t bytes, uint64_t* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               ::"r"(os_smem(dst)), "l"(src), "r"(bytes), "r"(os_smem(b)) : "memory");
}
__device__ __forceinline__ uint4 os_ldw(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ void os_mma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                       uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <int NT, int NB, int WK, int R, int S, int HILO, int SMEMW>
struct OsCfg {
  static constexpr int WARPS = NB * WK;
  static constexpr int THREADS = WARPS * 32;
  static constexpr int KC = WK * R * 32;          // K per CTA
  static constexpr int XS = KC + 8;               // x smem row stride (elements), +16B kills bank conflicts
  static constexpr int RPB = HILO ? 8 : 16;       // W rows per 16-row A block
  static constexpr int RC = NB * RPB;             // output rows per CTA
  static constexpr int C = NT * 8;                // padded tokens
  static constexpr int E = RC * C;
  static constexpr int ES = E / S;
  static constexpr int CTA_REC = R * WARPS;       // 1 KB records per CTA
  static constexpr int CHUNK_BYTES = WARPS * 1024;
  static constexpr int BAR_BYTES = 128;
  static constexpr int W_BYTES = SMEMW ? CTA_REC * 1024 : 0;
  static constexpr int X_OFF = BAR_BYTES + W_BYTES;
  static constexpr int X_BYTES = C * XS * 2;
  static constexpr int RED_BYTES = WARPS * 16 * C * 4;
  static constexpr int RED_OFF = (SMEMW && RED_BYTES <= W_BYTES) ? BAR_BYTES : X_OFF + X_BYTES;
  static constexpr int RECV_OFF = (RED_OFF == BAR_BYTES ? X_OFF + X_BYTES : RED_OFF + RED_BYTES);
  static constexpr int SMEM = RECV_OFF + (S > 1 ? E * 4 : 0);
  static_assert(E % S == 0, "E % S");
  static_assert(R <= 15, "barriers");
};

template <int NT, int NB, int WK, int R, int S, int HILO, int SMEMW, int F32OUT>
__global__ void __launch_bounds__(NB * WK * 32, 1)
os_kernel(const uint4* __restrict__ wp, const bf16* __restrict__ x, void* __restrict__ out, int m, int n, int stride_x) {
  using Cf = OsCfg<NT, NB, WK, R, S, HILO, SMEMW>;
  constexpr int WARPS = Cf::WARPS, THREADS = Cf::THREADS, KC = Cf::KC, XS = Cf::XS, RPB = Cf::RPB, RC = Cf::RC;
  constexpr int C = Cf::C, E = Cf::E, ES = Cf::ES;
  extern __shared__ __align__(128) unsigned char smem[];
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem);
  bf16* xs = reinterpret_cast<bf16*>(smem + Cf::X_OFF);
  const uint4* ws = reinterpret_cast<const uint4*>(smem + Cf::BAR_BYTES);
  float* red = reinterpret_cast<float*>(smem + Cf::RED_OFF);
  float* recv = reinterpret_cast<float*>(smem + Cf::RECV_OFF);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, q = lane & 3;
  const int cta = blockIdx.x, grp = cta / S, s = cta % S;
  const uint4* wcta = wp + (size_t)cta * Cf::CTA_REC * 64;

  uint4 wa[SMEMW ? 1 : R][2];
  if constexpr (!SMEMW) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const uint4* rec = wcta + (r * WARPS + warp) * 64;
      wa[r][0] = os_ldw(rec + lane);
      wa[r][1] = os_ldw(rec + 32 + lane);
    }
  }
  if constexpr (S > 1) os_cl_arrive_relaxed();
  if (tid == 0) {
    os_bar_init(&bars[0]);
    if constexpr (SMEMW) {
#pragma unroll
      for (int r = 0; r < R; ++r) os_bar_init(&bars[1 + r]);
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  if (warp == 0) {
    if (lane == 0) {
      os_expect(&bars[0], (uint32_t)(m * KC * 2));
      if constexpr (SMEMW) {
#pragma unroll
        for (int r = 0; r < R; ++r) os_expect(&bars[1 + r], Cf::CHUNK_BYTES);
      }
    }
    __syncwarp();
    if constexpr (SMEMW) {
      if (lane < R) os_bulk(smem + Cf::BAR_BYTES + lane * Cf::CHUNK_BYTES, wcta + lane * (Cf::CHUNK_BYTES / 16),
                            Cf::CHUNK_BYTES, &bars[1 + lane]);
    }
    for (int t = lane; t < m; t += 32)
      os_bulk(xs + t * XS, x + (size_t)t * stride_x + (size_t)s * KC, KC * 2, &bars[0]);
  }

  const int wk = warp % WK;
  float acc[NT][4];
#pragma unroll
  for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  os_wait(&bars[0]);
#pragma unroll
  for (int r = 0; r < R; ++r) {
    uint4 a0, a1;
    if constexpr (SMEMW) {
      os_wait(&bars[1 + r]);
      a0 = ws[(r * WARPS + warp) * 64 + lane];
      a1 = ws[(r * WARPS + warp) * 64 + 32 + lane];
    } else {
      a0 = wa[r][0];
      a1 = wa[r][1];
    }
    const int kl = (wk * R + r) * 32 + 8 * q;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const uint4 xb = *reinterpret_cast<const uint4*>(xs + (j * 8 + g) * XS + kl);
      os_mma(acc[j], a0.x, a1.x, a0.y, a1.y, xb.x, xb.y);
      os_mma(acc[j], a0.z, a1.z, a0.w, a1.w, xb.z, xb.w);
    }
  }
  if constexpr (Cf::RED_OFF == Cf::BAR_BYTES) __syncthreads();  // red aliases the W smem
#pragma unroll
  for (int j = 0; j < NT; ++j) {
    const int col = j * 8 + 2 * q;
    red[(warp * 16 + g) * C + col] = acc[j][0];
    red[(warp * 16 + g) * C + col + 1] = acc[j][1];
    red[(warp * 16 + g + 8) * C + col] = acc[j][2];
    red[(warp * 16 + g + 8) * C + col + 1] = acc[j][3];
  }
  __syncthreads();
  if constexpr (S > 1) os_cl_wait();
  for (int e = tid; e < E; e += THREADS) {
    const int c = e / RC, rowc = e % RC, bb = rowc / RPB, rr = rowc % RPB;
    float v = 0.f;
#pragma unroll
    for (int kk = 0; kk < WK; ++kk) {
      const int w = bb * WK + kk;
      v += red[(w * 16 + rr) * C + c];
      if constexpr (HILO) v += red[(w * 16 + rr + 8) * C + c];
    }
    if (c >= m) continue;
    if constexpr (S == 1) {
      const size_t o = (size_t)c * n + (size_t)grp * RC + rowc;
      if constexpr (F32OUT) static_cast<float*>(out)[o] = v;
      else static_cast<bf16*>(out)[o] = __float2bfloat16_rn(v);
    } else {
      cg::this_cluster().map_shared_rank(recv, e / ES)[s * ES + e % ES] = v;
    }
  }
  if constexpr (S > 1) {
    os_cl_arrive_release();
    os_cl_wait();
    for (int j = tid; j < ES; j += THREADS) {
      const int e = s * ES + j, c = e / RC, rowc = e % RC;
      if (c >= m) continue;
      float v = 0.f;
#pragma unroll
      for (int ss = 0; ss < S; ++ss) v += recv[ss * ES + j];
      const size_t o = (size_t)c * n + (size_t)grp * RC + rowc;
      if constexpr (F32OUT) static_cast<float*>(out)[o] = v;
      else static_cast<bf16*>(out)[o] = __float2bfloat16_rn(v);
    }
  }
}

template <int NT, int NB, int WK, int R, int S, int HILO, int SMEMW, int F32OUT>
int64_t os_launch(const torch::Tensor& x, const torch::Tensor& wp, torch::Tensor& out, int64_t smem_min, int64_t query) {
  using Cf = OsCfg<NT, NB, WK, R, S, HILO, SMEMW>;
  auto kern = os_kernel<NT, NB, WK, R, S, HILO, SMEMW, F32OUT>;
  const int smem = (int)std::max<int64_t>(Cf::SMEM, smem_min);
  C10_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(kern), cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  const int n = out.size(1);
  const int grid = (n / Cf::RC) * S;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid, 1, 1);
  cfg.blockDim = dim3(Cf::THREADS, 1, 1);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream().stream();
  cudaLaunchAttribute attr;
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = S;
  attr.val.clusterDim.y = 1;
  attr.val.clusterDim.z = 1;
  cfg.attrs = &attr;
  cfg.numAttrs = S > 1 ? 1 : 0;
  if (query) {
    if (S > 1) {
      int nc = 0;
      C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&nc, kern, &cfg));
      return nc;
    }
    int nb = 0, dev = 0, sms = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kern, Cf::THREADS, smem));
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    return (int64_t)nb * sms;
  }
  const int m = x.size(0), k = x.size(1);
  TORCH_CHECK(k == S * Cf::KC && n % Cf::RC == 0 && m >= 1 && m <= Cf::C, "os: shape mismatch");
  TORCH_CHECK(wp.numel() * wp.element_size() == (int64_t)grid * Cf::CTA_REC * 1024, "os: packed W size");
  TORCH_CHECK(x.stride(1) == 1 && x.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "os: x layout");
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, reinterpret_cast<const uint4*>(wp.data_ptr()),
                                    reinterpret_cast<const bf16*>(x.data_ptr()), out.data_ptr(), m, n, (int)x.stride(0)));
  return 0;
}

int64_t launch_os(torch::Tensor x, torch::Tensor wp, torch::Tensor out, int64_t id, int64_t smem_min, int64_t query) {
#define OS_CASE(ID, NT, NB, WK, R, S, HILO, SMEMW, F32) \
  if (id == ID) return os_launch<NT, NB, WK, R, S, HILO, SMEMW, F32>(x, wp, out, smem_min, query);
OS_CASES
#undef OS_CASE
  TORCH_CHECK(false, "os: id not compiled ", id);
  return 0;
}
"""
OS_CPP = "int64_t launch_os(torch::Tensor x, torch::Tensor wp, torch::Tensor out, int64_t id, int64_t smem_min, int64_t query);"

# id -> (NT, NB, WK, R, S, HILO, SMEMW)
OS_CFGS = {}
_FAMS = {
    # router (hi+lo, fp32 out)
    "rA_l": ((1, 2, 4), 1, 8, 8, 4, True, False),
    "rB_l": ((1, 2, 4), 2, 4, 8, 8, True, False),
    "rA_s": ((1, 2, 4), 1, 8, 8, 4, True, True),
    "rB_s": ((1, 2, 4), 2, 4, 8, 8, True, True),
    "rC_l": ((1, 2, 4), 1, 4, 8, 8, True, False),
    # unpooler slice
    "uA_l": ((1,), 1, 8, 8, 2, False, False),
    "uB_l": ((1,), 2, 4, 8, 4, False, False),
    "uA_s": ((1,), 1, 8, 8, 2, False, True),
    "uB_s": ((1,), 2, 4, 8, 4, False, True),
    "uC_l": ((1,), 1, 4, 8, 4, False, False),
    # o
    "oA_l": ((1, 2, 4), 4, 4, 8, 1, False, False),
    "oA_s": ((1, 2, 4), 4, 4, 8, 1, False, True),
    "oC_l": ((1, 2, 4), 2, 4, 8, 1, False, False),
}
OS_IDS = {}
for _fam, (_nts, *_rest) in _FAMS.items():
    for _nt in _nts:
        _id = len(OS_CFGS)
        OS_CFGS[_id] = (_nt, *_rest)
        OS_IDS[(_fam, _nt)] = _id
OS_SRC = OS_SRC.replace("OS_CASES", "\n".join(
    f"  OS_CASE({i}, {nt}, {nb}, {wk}, {r}, {s}, {int(hl)}, {int(sm)}, {int(hl)})" for i, (nt, nb, wk, r, s, hl, sm) in OS_CFGS.items()))

# (key, m) -> families to try; key "router" / "unpooler_slice" / "o"
OS_PLAN = {
    ("router", 1): ("rA_l", "rB_l", "rA_s", "rB_s", "rC_l"),
    ("router", 8): ("rA_l", "rB_l", "rA_s", "rB_s", "rC_l"),
    ("router", 16): ("rA_l", "rB_l", "rA_s", "rB_s", "rC_l"),
    ("router", 32): ("rA_l", "rB_l", "rA_s", "rB_s", "rC_l"),
    ("unpooler_slice", 1): ("uA_l", "uB_l", "uA_s", "uB_s", "uC_l"),
    ("unpooler_slice", 8): ("uA_l", "uB_l", "uA_s", "uB_s", "uC_l"),
    ("o", 1): ("oA_l", "oA_s", "oC_l"),
    ("o", 8): ("oA_l", "oA_s", "oC_l"),
    ("o", 16): ("oA_l", "oA_s", "oC_l"),
    ("o", 32): ("oA_l", "oA_s", "oC_l"),
}
OS_SMEM_FORCE = 120 * 1024  # >half an SM's smem: forces one CTA per SM
OS_REPS = 40
OS_MARGIN_US = 0.15
OS_DET_REPS = 10


def _os_build():
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((OS_SRC + " ".join(CUDA_FLAGS)).encode()).hexdigest()[:10]
    return load_inline(name=f"muse_os_{tag}", cpp_sources=OS_CPP, cuda_sources=OS_SRC, functions=["launch_os"],
                       extra_cuda_cflags=CUDA_FLAGS, extra_cflags=["-O3"])


def _os_pack(w_a: torch.Tensor, nb: int, wk: int, r: int, s: int) -> torch.Tensor:
    """w_a: [NA*16, K] A rows. Returns [cta][rec r][warp][16 x 32] bf16 records, flat."""
    rows, k = w_a.shape
    na, kb = rows // 16, k // 32
    g = na // nb
    t = w_a.view(na, 16, kb, 32).permute(0, 2, 1, 3)  # [na, kb, 16, 32]
    t = t.reshape(g, nb, s, wk, r, 512).permute(0, 2, 4, 1, 3, 5)
    return t.contiguous().view(-1)


def _os_arows(state: dict, key: str) -> torch.Tensor:
    if key == "router":
        hi, lo = state["router_hl"][0], state["router_hl"][1]
        n, k = hi.shape
        return torch.stack([hi.view(n // 8, 8, k), lo.view(n // 8, 8, k)], dim=1).reshape(n * 2, k)
    return state[key]


def _os_run(ext, x: torch.Tensor, wp: torch.Tensor, n: int, cid: int, smem: int, f32: bool) -> torch.Tensor:
    out = torch.empty(x.shape[0], n, device=x.device, dtype=torch.float32 if f32 else torch.bfloat16)
    ext.launch_os(x, wp, out, cid, smem, 0)
    return out


def _os_selftest(fn, w_ref: torch.Tensor, m: int, f32: bool, gen: torch.Generator) -> None:
    k = w_ref.shape[1]
    x = torch.randn(m, k, device=w_ref.device, generator=gen)
    x[:, ::97] *= 30.0
    if m >= 2:
        x[m // 2] = 0
        x[m // 2, k // 3 + 5] = -64.0
    x = x.to(torch.bfloat16)
    ref = x.double() @ w_ref.double().t()
    got = fn(x)
    gd = got.double()
    rel = float((gd - ref).norm() / ref.norm())
    tol = 2e-5 if f32 else 4e-3
    if not bool(torch.isfinite(gd).all()) or rel > tol:
        raise ValueError(f"smoke rel={rel:.2e} > {tol:g}")
    rows = ((x != 0).sum(-1) == 1).nonzero().flatten()
    if len(rows):
        pr = float(((gd[rows] - ref[rows]).norm(dim=-1) / ref[rows].norm(dim=-1)).max())
        if pr > 2e-4:
            raise ValueError(f"probe rel={pr:.2e}")
    for _ in range(OS_DET_REPS):
        if not torch.equal(fn(x), got):
            raise ValueError("non-deterministic")


def _os_prepare(state: dict) -> None:
    import time
    t_start = time.time()
    state["os"] = {}
    if state.get("ext") is None:
        return
    try:
        ext = _os_build()
        print(f"os build {time.time() - t_start:.1f}s")
    except Exception as exc:
        print(f"os path disabled: {type(exc).__name__}: {str(exc)[-3000:]}")
        return
    dev = state["o"].device
    gen = torch.Generator(device=dev).manual_seed(16)
    flush = torch.empty(64 * 2**20, device=dev, dtype=torch.float32)
    flush.normal_()
    packs, arows = {}, {}
    picks = {}
    for (key, m), fams in OS_PLAN.items():
        f32 = key == "router"
        w_ref = state["router"] if f32 else state[key]
        n, k = w_ref.shape
        good = {}
        for fam in fams:
            nt = (m + 7) // 8
            nt = 4 if nt > 2 else nt
            cid = OS_IDS.get((fam, nt))
            if cid is None:
                continue
            _, nb, wk, r, s, hl, sm = OS_CFGS[cid]
            for smem in ((0, OS_SMEM_FORCE) if not sm else (0,)):
                nm = f"{fam}{'+f' if smem else ''}"
                try:
                    if key not in arows:
                        arows[key] = _os_arows(state, key)
                    pk = (key, nb, wk, r, s)
                    if pk not in packs:
                        packs[pk] = _os_pack(arows[key], nb, wk, r, s)
                    wp = packs[pk]
                    rpb = 8 if hl else 16
                    grid = (n // (nb * rpb)) * s
                    cap = int(ext.launch_os(torch.empty(0), wp, torch.empty(0, n), cid, smem, 1))
                    need = grid // s
                    if cap < need:
                        print(f"os skip {key} m={m} {nm}: co-resident {cap} < {need}")
                        continue
                    fn = (lambda xx, wp=wp, cid=cid, smem=smem: _os_run(ext, xx, wp, n, cid, smem, f32))
                    _os_selftest(fn, w_ref, m, f32, gen)
                    good[nm] = fn
                except Exception as exc:
                    print(f"os selftest FAILED {key} m={m} {nm}: {type(exc).__name__}: {str(exc)[-1500:]}")
        if not good:
            continue
        try:
            x = torch.randn(m, k, device=dev, generator=gen).to(torch.bfloat16)
            layer = "unpooler" if key == "unpooler_slice" else key
            data = {"x": x, "config": {"m": m, "n": n, "k": k, "layer": layer}}
            graphs = {"cur": _r9_graph(lambda: kernel_fn(data, state))}
            for nm, fn in good.items():
                graphs[nm] = _r9_graph(lambda fn=fn: fn(x))
            med = _os_time(graphs, flush)
            best = min(good, key=lambda nm: med[nm])
            win = med[best] < med["cur"] - OS_MARGIN_US
            print(f"os bench {key} m={m} (raw us): " + " ".join(f"{nm}={v:.2f}" for nm, v in med.items())
                  + (f" -> {best}" if win else " -> cur"))
            if win:
                picks[(key, m)] = good[best]
        except Exception as exc:
            print(f"os bench error {key} m={m}: {type(exc).__name__}: {exc}")
    del flush
    state["os"] = picks
    torch.cuda.synchronize()
    print(f"os prepare {time.time() - t_start:.1f}s picks={sorted(picks)}")


def _os_time(graphs: dict, flush: torch.Tensor) -> dict:
    import statistics
    res = {nm: [] for nm in graphs}
    names = list(graphs)
    for r in range(OS_REPS):
        for nm in (names if r % 2 == 0 else names[::-1]):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            flush.sum()
            a.record()
            graphs[nm].replay()
            b.record()
            b.synchronize()
            res[nm].append(a.elapsed_time(b) * 1e3)
    return {nm: statistics.median(v) for nm, v in res.items()}


def _os_try(x: torch.Tensor, key: str, n: int, state: dict):
    fn = state.get("os", {}).get((key, x.shape[0]))
    if fn is None:
        return None
    w = state["router"] if key == "router" else state[key]
    if n != w.shape[0] or x.shape[1] != w.shape[1]:
        return None
    return fn(x)
