"""r9 (k4: NKW k-interleaved warpgroups, KB-deep stages, fence-once) cluster split-K bf16 GEMM for m=128/256: W rows on the wgmma M side (NWG consumer warpgroups x 64 rows),
tokens on N (BT = 64/128/256), one producer warp issuing 2D TMA for W and x into a D-stage mbarrier ring.
The S CTAs of a cluster split K; fp32 partials are pushed into the owner rank's smem (token-stripe ownership)
through DSMEM and summed in fixed rank order (deterministic). Spliced into the candidate by exp_r9_cl_build.py.
"""


def _r9_gen_wgmma() -> str:
    parts = ["template <int BT>\n__device__ __forceinline__ void r9_mma(float (&d)[BT / 8][4], uint64_t da, uint64_t db);\n"]
    for n in (64, 128, 256):
        nt = n // 8
        regs = ", ".join(f"%{i}" for i in range(n // 2))
        outs = ", ".join(f'"+f"(d[{j}][{r}])' for j in range(nt) for r in range(4))
        parts.append(
            "template <>\n"
            f"__device__ __forceinline__ void r9_mma<{n}>(float (&d)[{nt}][4], uint64_t da, uint64_t db) {{\n"
            f'  asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 "\n'
            f'      "{{{regs}}}, %{n // 2}, %{n // 2 + 1}, 1, 1, 1, 0, 0;\\n"\n'
            f"      : {outs}\n"
            '      : "l"(da), "l"(db));\n'
            "}\n")
    return "".join(parts)


R9_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda.h>

namespace cg = cooperative_groups;
using bf16 = __nv_bfloat16;

__device__ __forceinline__ uint32_t r9_su(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void r9_cl_arrive_release() { asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void r9_cl_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void r9_mbar_init(uint64_t* b, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(r9_su(b)), "r"(count) : "memory");
}
__device__ __forceinline__ void r9_expect_tx(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(r9_su(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void r9_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(r9_su(b)) : "memory");
}
__device__ __forceinline__ void r9_wait(uint64_t* b, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred P1;\n R9_WAIT:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      " @P1 bra R9_DONE;\n bra R9_WAIT;\n R9_DONE:\n}\n"
      ::"r"(r9_su(b)), "r"(parity) : "memory");
}
__device__ __forceinline__ void r9_tma(void* dst, const CUtensorMap* map, int c0, int c1, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
      ::"r"(r9_su(dst)), "l"(map), "r"(c0), "r"(c1), "r"(r9_su(bar)) : "memory");
}
__device__ __forceinline__ void r9_tma_h(void* dst, const CUtensorMap* map, int c0, int c1, uint64_t* bar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;\n"
      ::"r"(r9_su(dst)), "l"(map), "r"(c0), "r"(c1), "r"(r9_su(bar)), "l"(pol) : "memory");
}
__device__ __forceinline__ uint64_t r9_gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
// K-major operand, 128B swizzle, 8-row groups 1024 B apart
__device__ __forceinline__ uint64_t r9_desc(const void* p) {
  return (uint64_t)((r9_su(p) >> 4) & 0x3FFF) | (1ull << 16) | (64ull << 32) | (1ull << 62);
}
__device__ __forceinline__ void r9_wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void r9_wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void r9_wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"); }
""" + _r9_gen_wgmma() + r"""
template <int NWG, int BT, int S, int KB, int NKW>
struct R9Geo {
  static constexpr int BN = 64 * NWG;
  static constexpr int NT = BT / 8;
  static constexpr int NCT = NWG * 128;
  static constexpr int NCONS = NCT * NKW;           // NKW warpgroups per row block take alternate stages
  static constexpr int THREADS = NCONS + 32;
  static constexpr int JPR = (NT + S - 1) / S;      // max 8-token column blocks owned per rank
  static constexpr uint32_t WBOX = BN * 128;        // one 64-k W box (BN rows)
  static constexpr uint32_t XBOX = BT * 128;        // one 64-k x box (BT tokens)
  static constexpr uint32_t WST = KB * WBOX;
  static constexpr uint32_t STB = KB * (WBOX + XBOX);
  static constexpr uint32_t SLOT = JPR * NCT * 16;  // one peer's partials, fragment order (float4 per thread per j)
  static constexpr uint32_t RECV = (S - 1) * SLOT;
  static constexpr int OPITCH = BN + 8;             // bf16 staging row pitch: conflict-free, 16B rows
  static constexpr uint32_t OST = 8 * JPR * OPITCH * 2;
  __device__ static constexpr int owner(int j) { return (j * S) / NT; }
  __device__ static constexpr int j0(int q) { return (q * NT + S - 1) / S; }
};

template <int NWG, int BT, int S, int D, int KB, int NKW, int F1>
__global__ void __launch_bounds__(NWG * NKW * 128 + 32, 1)
r9_kernel(const __grid_constant__ CUtensorMap wmap, const __grid_constant__ CUtensorMap xmap,
          bf16* __restrict__ out, int m, int N, int K, int TT, unsigned long long* __restrict__ prof) {
  using G = R9Geo<NWG, BT, S, KB, NKW>;
  static_assert(G::RECV + G::OST <= D * G::STB, "r9: reduce buffers exceed the stage ring");
  static_assert(G::NT >= S, "r9: every rank must own a token stripe");
  static_assert(D > NKW, "r9: stage ring must outlast one round of k-interleaved warpgroups");
  static_assert((NKW - 1) * G::NT * G::NCT * 16 <= D * G::STB, "r9: k-interleave scratch exceeds the ring");
  constexpr uint32_t BOFF = D * G::STB;
  extern __shared__ __align__(1024) unsigned char r9_raw[];
  unsigned char* sm = r9_raw + ((1024 - (r9_su(r9_raw) & 1023)) & 1023);
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + BOFF);
  uint64_t* empty = full + D;

  int rank = 0;
  if constexpr (S > 1) rank = (int)cg::this_cluster().block_rank();
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int c = blockIdx.x / S;
  const int tt = c % TT, tile = c / TT;
  const int n0 = tile * G::BN, t0 = tt * BT;
  const int U = K / (64 * KB), u0 = rank * U / S, nk = (rank + 1) * U / S - u0;

  if (tid == 0) {
    for (int d = 0; d < D; ++d) { r9_mbar_init(&full[d], 1); r9_mbar_init(&empty[d], NWG * 4); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  unsigned long long* pf = prof ? prof + (size_t)blockIdx.x * 6 : nullptr;
  if (pf && tid == 0) pf[0] = pf[1] = r9_gtime();

  float acc[G::NT][4];
#pragma unroll
  for (int j = 0; j < G::NT; ++j)
#pragma unroll
    for (int r = 0; r < 4; ++r) acc[j][r] = 0.f;

  if (warp == NWG * NKW * 4) {
    if (lane == 0) {
      for (int i = 0; i < nk; ++i) {
        const int s = i % D;
        if (i >= D) {
          r9_wait(&empty[s], ((i / D) - 1) & 1);
          asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        }
        unsigned char* st = sm + s * G::STB;
        r9_expect_tx(&full[s], G::STB);  // x rows >= m are zero-filled and still counted
#pragma unroll
        for (int kb = 0; kb < KB; ++kb) {
          const int kc = ((u0 + i) * KB + kb) * 64;
          r9_tma(st + kb * G::WBOX, &wmap, kc, n0, &full[s]);
          r9_tma(st + G::WST + kb * G::XBOX, &xmap, kc, t0, &full[s]);
        }
      }
    }
    __syncwarp();
  } else {
    const int cw = warp >> 2, wg = cw % NWG, wk = cw / NWG;
    int it = 0;
    for (int i = wk; i < nk; i += NKW, ++it) {
      const int s = i % D;
      r9_wait(&full[s], (i / D) & 1);
      const unsigned char* st = sm + s * G::STB;
      if (!F1 || it == 0) r9_wg_fence();
#pragma unroll
      for (int kb = 0; kb < KB; ++kb) {
        const uint64_t da = r9_desc(st + kb * G::WBOX + wg * 8192);
        const uint64_t db = r9_desc(st + G::WST + kb * G::XBOX);
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) r9_mma<BT>(acc, da + (uint64_t)(2 * ks), db + (uint64_t)(2 * ks));
      }
      r9_wg_commit();
      r9_wg_wait<1>();
      if (it > 0 && lane == 0) r9_arrive(&empty[(i - NKW) % D]);
    }
    r9_wg_wait<0>();
    if (pf && tid == 0) pf[2] = r9_gtime();
    if constexpr (NKW > 1) {
      // fixed-order in-CTA reduce of the k-interleaved partials into the wk == 0 warpgroups
      float4* scr = reinterpret_cast<float4*>(sm);
      const int lt = tid - wk * G::NCT;
      asm volatile("bar.sync 2, %0;\n" ::"n"(G::NCONS) : "memory");
      if (wk > 0) {
#pragma unroll
        for (int j = 0; j < G::NT; ++j)
          scr[((wk - 1) * G::NT + j) * G::NCT + lt] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);
      }
      asm volatile("bar.sync 2, %0;\n" ::"n"(G::NCONS) : "memory");
      if (wk == 0) {
#pragma unroll
        for (int q = 1; q < NKW; ++q)
#pragma unroll
          for (int j = 0; j < G::NT; ++j) {
            const float4 p = scr[((q - 1) * G::NT + j) * G::NCT + lt];
            acc[j][0] += p.x;
            acc[j][1] += p.y;
            acc[j][2] += p.z;
            acc[j][3] += p.w;
          }
      }
    }
  }
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  // every CTA of the cluster is past its mainloop: the stage ring is free for reduce slots everywhere
  if constexpr (S > 1) {
    r9_cl_arrive_release();
    r9_cl_wait();
  } else {
    __syncthreads();
  }
  if (pf && tid == 0) pf[3] = r9_gtime();

  float4* recv = reinterpret_cast<float4*>(sm);
  bf16* ost = reinterpret_cast<bf16*>(sm + G::RECV);
  if constexpr (S > 1) {
    if (warp < NWG * 4) {
      cg::cluster_group cluster = cg::this_cluster();
#pragma unroll
      for (int j = 0; j < G::NT; ++j) {
        const int q = G::owner(j);
        if (q != rank) {
          const int slot = rank < q ? rank : rank - 1;
          float4* dst = cluster.map_shared_rank(recv + (slot * G::JPR + (j - G::j0(q))) * G::NCT + tid, q);
          *dst = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);
        }
      }
    }
    r9_cl_arrive_release();
    r9_cl_wait();
  }
  if (pf && tid == 0) pf[4] = r9_gtime();

  if (warp < NWG * 4) {
    const int wg = warp >> 2, wq = warp & 3, g = lane >> 2, t4 = lane & 3;
    const int nl = wg * 64 + wq * 16 + g;
    const int jb = G::j0(rank);
#pragma unroll
    for (int j = 0; j < G::NT; ++j) {
      if (G::owner(j) == rank) {
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int src = 0; src < S; ++src) {
          float4 p;
          if (src == rank) {
            p = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);
          } else {
            const int slot = src < rank ? src : src - 1;
            p = recv[(slot * G::JPR + (j - jb)) * G::NCT + tid];
          }
          v.x += p.x;
          v.y += p.y;
          v.z += p.z;
          v.w += p.w;
        }
        const int tl = 8 * (j - jb) + 2 * t4;
        ost[tl * G::OPITCH + nl] = __float2bfloat16(v.x);
        ost[(tl + 1) * G::OPITCH + nl] = __float2bfloat16(v.y);
        ost[tl * G::OPITCH + nl + 8] = __float2bfloat16(v.z);
        ost[(tl + 1) * G::OPITCH + nl + 8] = __float2bfloat16(v.w);
      }
    }
    asm volatile("bar.sync 1, %0;\n" ::"n"(G::NCT) : "memory");
    constexpr int V = G::BN / 8;
    const int ntok = 8 * (G::j0(rank + 1) - jb);
    for (int e = tid; e < ntok * V; e += G::NCT) {
      const int tl = e / V, cv = e % V;
      const int t = t0 + 8 * jb + tl;
      if (t < m) {
        *reinterpret_cast<uint4*>(out + (size_t)t * N + n0 + cv * 8) =
            *reinterpret_cast<const uint4*>(ost + tl * G::OPITCH + cv * 8);
      }
    }
    if (pf && tid == 0) pf[5] = r9_gtime();
  }
}

inline CUtensorMap r9_map(const void* ptr, int rows, int cols, int row_stride, int box_rows) {
  CUtensorMap map{};
  cuuint64_t gdim[2] = {(cuuint64_t)cols, (cuuint64_t)rows};
  cuuint64_t gstride[1] = {(cuuint64_t)row_stride * sizeof(bf16)};
  cuuint32_t bdim[2] = {64u, (cuuint32_t)box_rows};
  cuuint32_t estride[2] = {1u, 1u};
  CUresult res = cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), gdim, gstride,
                                        bdim, estride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(res == CUDA_SUCCESS, "r9: cuTensorMapEncodeTiled failed ", (int)res);
  return map;
}

template <int NWG, int BT, int S, int D, int KB, int NKW, int F1>
constexpr size_t r9_smem() { return 1024 + (size_t)D * R9Geo<NWG, BT, S, KB, NKW>::STB + (size_t)2 * D * 8; }

template <int NWG, int BT, int S, int D, int KB, int NKW, int F1>
cudaLaunchConfig_t r9_cfg(int blocks, cudaLaunchAttribute* attr) {
  using G = R9Geo<NWG, BT, S, KB, NKW>;
  constexpr size_t smem = r9_smem<NWG, BT, S, D, KB, NKW, F1>();
  static_assert(smem <= 232448, "r9: smem");
  auto kern = r9_kernel<NWG, BT, S, D, KB, NKW, F1>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(kern),
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  if (S > 8) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(kern),
                                        cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(blocks, 1, 1);
  cfg.blockDim = dim3(G::THREADS, 1, 1);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream().stream();
  attr->id = cudaLaunchAttributeClusterDimension;
  attr->val.clusterDim.x = S;
  attr->val.clusterDim.y = 1;
  attr->val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  return cfg;
}

template <int NWG, int BT, int S, int D, int KB, int NKW, int F1>
void r9_launch(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& out, const torch::Tensor& prof) {
  using G = R9Geo<NWG, BT, S, KB, NKW>;
  const int m = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(m >= 1 && N % G::BN == 0 && K % (64 * KB) == 0 && K / (64 * KB) >= S && w.size(1) == K, "r9: shape");
  TORCH_CHECK(out.size(0) == m && out.size(1) == N && out.is_contiguous() && w.is_contiguous(), "r9: out/w layout");
  TORCH_CHECK(x.stride(1) == 1 && x.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "r9: x layout");
  const int TT = (m + BT - 1) / BT;
  const CUtensorMap wmap = r9_map(w.data_ptr(), N, K, K, G::BN);
  const CUtensorMap xmap = r9_map(x.data_ptr(), m, K, (int)x.stride(0), BT);
  cudaLaunchAttribute attr;
  const int blocks = (N / G::BN) * TT * S;
  unsigned long long* pp = nullptr;
  if (prof.numel() > 0) {
    TORCH_CHECK(prof.is_cuda() && prof.scalar_type() == at::kLong && prof.numel() >= (int64_t)blocks * 6, "r9: prof");
    pp = reinterpret_cast<unsigned long long*>(prof.data_ptr());
  }
  cudaLaunchConfig_t cfg = r9_cfg<NWG, BT, S, D, KB, NKW, F1>(blocks, &attr);
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, r9_kernel<NWG, BT, S, D, KB, NKW, F1>, wmap, xmap,
                                    static_cast<bf16*>(out.data_ptr()), m, N, K, TT, pp));
}

template <int NWG, int BT, int S, int D, int KB, int NKW, int F1>
int64_t r9_clusters() {
  cudaLaunchAttribute attr;
  cudaLaunchConfig_t cfg = r9_cfg<NWG, BT, S, D, KB, NKW, F1>(132 * S, &attr);
  int n = -1;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&n, reinterpret_cast<const void*>(r9_kernel<NWG, BT, S, D, KB, NKW, F1>), &cfg));
  return n;
}

#define R9_CASE(ID, NWG_, BT_, S_, D_, KB_, NKW_, F1_)                    \
  if (id == ID) {                                                         \
    if (query) return r9_clusters<NWG_, BT_, S_, D_, KB_, NKW_, F1_>();   \
    r9_launch<NWG_, BT_, S_, D_, KB_, NKW_, F1_>(x, w, out, prof);        \
    return 0;                                             \
  }

int64_t launch_r9(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t id, int64_t query,
                  torch::Tensor prof) {
R9_CASES
  TORCH_CHECK(false, "r9: config id not compiled ", id);
  return 0;
}
"""
# id -> (NWG, BT, S, D, KB, NKW, F1): BN = 64 NWG W rows per CTA, BT tokens, S-way cluster split-K, D-stage ring of
# KB 64-k boxes, NKW warpgroups per row block taking alternate stages (hides the per-stage wgmma drain), F1 = fence once
R9_CFGS = {
    3: (1, 128, 3, 8, 1, 1, 0),
    16: (1, 128, 1, 8, 1, 1, 0),
    20: (1, 128, 3, 4, 2, 1, 0),
    41: (1, 128, 3, 8, 1, 2, 0),
    42: (1, 128, 3, 4, 2, 2, 0),
    43: (1, 256, 3, 5, 1, 2, 0),
    45: (1, 128, 1, 8, 1, 2, 0),
    46: (1, 256, 1, 5, 1, 2, 0),
    47: (1, 128, 2, 8, 1, 2, 0),
    48: (1, 256, 2, 5, 1, 2, 0),
    49: (2, 128, 3, 6, 1, 2, 0),
}
R9_SRC = R9_SRC.replace("R9_CASES", "\n".join(f"  R9_CASE({i}, {', '.join(map(str, v))})" for i, v in R9_CFGS.items()))
R9_CPP = ("int64_t launch_r9(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t id, int64_t query,"
          " torch::Tensor prof);")
# (layer, m) -> candidate config ids, timed in prepare against the cp8 path; fastest wins if it beats cp8
R9_PLAN = {
    ("qkv_gate", 128): (3, 20, 41, 42),
    ("qkv_gate", 256): (43, 49),
    ("pooler", 128): (47,),
    ("pooler", 256): (48,),
    ("unpooler", 128): (45,),
    ("unpooler", 256): (46,),
    ("o", 128): (16, 45),
    ("o", 256): (46,),
}
# log-only per-CTA phase stamps: (layer, m, id)
R9_DIAG = (("qkv_gate", 128, 41), ("qkv_gate", 256, 43), ("o", 128, 45), ("o", 256, 46))
R9_DET_REPS = 10
R9_BENCH_REPS = 30
R9_MARGIN_US = 0.2
_R9_NOPROF = {}


def _r9_build():
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((R9_SRC + " ".join(CUDA_FLAGS)).encode()).hexdigest()[:10]
    return load_inline(name=f"muse_r9_{tag}", cpp_sources=R9_CPP, cuda_sources=R9_SRC, functions=["launch_r9"],
                       extra_cuda_cflags=CUDA_FLAGS, extra_cflags=["-O3"], extra_ldflags=CUDA_LDFLAGS)


def _r9_noprof(device) -> torch.Tensor:
    t = _R9_NOPROF.get(device)
    if t is None:
        t = _R9_NOPROF[device] = torch.empty(0, dtype=torch.int64, device=device)
    return t


def _r9_run(ext, x: torch.Tensor, w: torch.Tensor, cid: int, prof=None) -> torch.Tensor:
    out = torch.empty(x.shape[0], w.shape[0], device=x.device, dtype=torch.bfloat16)
    ext.launch_r9(x, w, out, cid, 0, _r9_noprof(x.device) if prof is None else prof)
    return out


def _r9_blocks(cid: int, n: int, m: int) -> int:
    nwg, bt, s = R9_CFGS[cid][:3]
    return (n // (64 * nwg)) * ((m + bt - 1) // bt) * s


def _r9_fits(cid: int, n: int, k: int, m: int, max_clusters: int) -> str | None:
    nwg, bt, s, _, kb = R9_CFGS[cid][:5]
    if n % (64 * nwg) or k % (64 * kb) or k // (64 * kb) < s:
        return "shape"
    need = (n // (64 * nwg)) * ((m + bt - 1) // bt)
    if need > max_clusters:
        return f"needs {need} clusters > {max_clusters} co-resident"
    return None


def _r9_selftest(ext, w: torch.Tensor, m: int, cid: int, gen: torch.Generator) -> None:
    k = w.shape[1]
    x = torch.randn(m, k, device=w.device, generator=gen).to(torch.bfloat16)
    x[:, ::97] *= 30.0
    ref = x.double() @ w.double().t()
    got = _r9_run(ext, x, w, cid)
    gd = got.double()
    rel = float((gd - ref).norm() / ref.norm())
    mx = float((gd - ref).abs().max() / ref.abs().max())
    if not bool(torch.isfinite(gd).all()) or rel > 4e-3 or mx > 2e-2:
        raise ValueError(f"smoke rel={rel:.2e} max={mx:.2e}")
    for _ in range(R9_DET_REPS):
        if not torch.equal(_r9_run(ext, x, w, cid), got):
            raise ValueError("non-deterministic")


def _r9_time(graphs: dict, flush: torch.Tensor) -> dict:
    import statistics
    res = {nm: [] for nm in graphs}
    for r in range(R9_BENCH_REPS):
        for nm in (list(graphs) if r % 2 == 0 else list(graphs)[::-1]):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            flush.sum()
            a.record()
            graphs[nm].replay()
            b.record()
            b.synchronize()
            res[nm].append(a.elapsed_time(b) * 1e3)
    return {nm: statistics.median(v) for nm, v in res.items()}


def _r9_graph(fn):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def _r9_diag(ext, state: dict, flush: torch.Tensor, gen: torch.Generator) -> None:
    for layer, m, cid in R9_DIAG:
        try:
            w = state[layer]
            n, k = w.shape
            x = torch.randn(m, k, device=w.device, generator=gen).to(torch.bfloat16)
            nb = _r9_blocks(cid, n, m)
            prof = torch.zeros(nb * 6, dtype=torch.int64, device=w.device)
            st = []
            for _ in range(5):
                flush.sum()
                _r9_run(ext, x, w, cid, prof)
                torch.cuda.synchronize()
                p = prof.view(nb, 6).double() / 1e3
                t0 = p[:, 0].min()
                ph = torch.stack([p[:, 0] - t0, p[:, 2] - p[:, 0], p[:, 3] - p[:, 2], p[:, 4] - p[:, 3],
                                  p[:, 5] - p[:, 4]], 1)
                st.append(torch.cat([ph.median(0).values, ph.max(0).values, (p[:, 5].max() - t0).view(1)]))
            sm = torch.stack(st).median(0).values.tolist()
            names = ("skew", "main", "bar1", "red", "store")
            print(f"r9 diag {layer} m={m} id{cid} blocks={nb}: med " + " ".join(f"{a}={b:.2f}" for a, b in zip(names, sm[:5]))
                  + " | max " + " ".join(f"{a}={b:.2f}" for a, b in zip(names, sm[5:10])) + f" | span={sm[10]:.2f}")
        except Exception as exc:
            print(f"r9 diag error {layer} m={m} id={cid}: {type(exc).__name__}: {exc}")


def _r9_prepare(state: dict) -> None:
    import time
    t_start = time.time()
    state["r9"] = {}
    if state.get("ext") is None:
        return
    try:
        ext = _r9_build()
        print(f"r9 build {time.time() - t_start:.1f}s")
    except Exception as exc:
        print(f"r9 path disabled: {type(exc).__name__}: {str(exc)[-3000:]}")
        return
    dev = state["pooler"].device
    clusters = {}
    for cid in R9_CFGS:
        try:
            e = torch.empty(0, device=dev)
            clusters[cid] = int(ext.launch_r9(e, e, e, cid, 1, _r9_noprof(dev)))
        except Exception as exc:
            print(f"r9 id={cid} occupancy query failed: {type(exc).__name__}: {exc}")
            clusters[cid] = 0
    print("r9 max clusters: " + " ".join(f"id{c}{R9_CFGS[c]}={v}" for c, v in clusters.items()))
    gen = torch.Generator(device=dev).manual_seed(9)
    flush = torch.empty(64 * 2**20, device=dev, dtype=torch.float32)
    flush.normal_()
    picks = {}
    for (layer, m), cands in R9_PLAN.items():
        w = state[layer]
        n, k = w.shape
        good = []
        for cid in cands:
            why = _r9_fits(cid, n, k, m, clusters.get(cid, 0))
            if why:
                print(f"r9 skip {layer} m={m} id={cid}: {why}")
                continue
            try:
                _r9_selftest(ext, w, m, cid, gen)
                good.append(cid)
            except Exception as exc:
                print(f"r9 selftest FAILED {layer} m={m} id={cid}: {type(exc).__name__}: {str(exc)[-1500:]}")
        if not good:
            continue
        try:
            x = torch.randn(m, k, device=w.device, generator=gen).to(torch.bfloat16)
            data = {"x": x, "config": {"m": m, "n": n, "k": k, "layer": layer}}
            graphs = {"cp8": _r9_graph(lambda: kernel_fn(data, state))}
            for cid in good:
                graphs[f"id{cid}"] = _r9_graph(lambda c=cid: _r9_run(ext, x, w, c))
            med = _r9_time(graphs, flush)
            best = min(good, key=lambda c: med[f"id{c}"])
            win = med[f"id{best}"] < med["cp8"] - R9_MARGIN_US
            print(f"r9 bench {layer} m={m} (raw us, ~4 graph overhead): "
                  + " ".join(f"{nm}={v:.2f}" for nm, v in med.items())
                  + (f" -> id{best}" if win else " -> cp8"))
            if win:
                picks[(layer, m)] = best
        except Exception as exc:
            print(f"r9 bench error {layer} m={m}: {type(exc).__name__}: {exc}")
    _r9_diag(ext, state, flush, gen)
    state["r9_ext"] = ext
    state["r9"] = picks
    torch.cuda.synchronize()
    print(f"r9 prepare {time.time() - t_start:.1f}s picks={picks}")


def _r9_try(x: torch.Tensor, layer: str, n: int, state: dict):
    pick = state.get("r9", {}).get((layer, x.shape[0]))
    w = state[layer]
    if pick is None or n != w.shape[0] or x.shape[1] != w.shape[1]:
        return None
    return _r9_run(state["r9_ext"], x, w, pick)
