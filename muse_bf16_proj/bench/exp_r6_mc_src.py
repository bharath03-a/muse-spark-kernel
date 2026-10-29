"""CUDA source for the split-K qkv kernel: 1 TMA producer warp + 2 consumer warpgroups (alternating k-blocks)."""


def _n256():
    regs = ", ".join(f"%{i}" for i in range(128))
    outs = ",\n        ".join(", ".join(f'"+f"(d[{j}][{r}])' for r in range(4)) for j in range(32))
    return f'''
template <>
__device__ __forceinline__ void wgmma_m64n<256>(float (&d)[32][4], uint64_t da, uint64_t db) {{
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
      "{{{regs}}}, "
      "%128, %129, 1, 1, 1, 0, 0;\\n"
      : {outs}
      : "l"(da), "l"(db));
}}
'''


MC_SRC = _n256() + r"""
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void bar_consumers() { asm volatile("bar.sync 1, 256;\n" ::: "memory"); }

#ifdef MC_DBG
__device__ unsigned long long g_dbg[256 * 6];
__device__ __forceinline__ unsigned long long gt() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
#define DBG(i) if (threadIdx.x == 0) g_dbg[blockIdx.x * 6 + (i)] = gt()
#else
#define DBG(i)
#endif
template <int BT, int STAGES, int S>
__global__ void __launch_bounds__(288, 1)
qkv_mc_kernel(const __grid_constant__ CUtensorMap w_map, const __grid_constant__ CUtensorMap x_map,
              bf16* __restrict__ out, float* __restrict__ ws, int* __restrict__ cnt, int m, int n, int k) {
  constexpr int W_BYTES = 128 * 64 * 2;
  constexpr int X_BYTES = BT * 64 * 2;
  constexpr int STAGE_BYTES = W_BYTES + X_BYTES;
  constexpr int NT = BT / 8;
  constexpr int F = BT * 32;
  extern __shared__ __align__(1024) unsigned char smem[];
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + (size_t)STAGES * STAGE_BYTES);
  uint64_t* empty = full + STAGES;

  DBG(0);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int tile = blockIdx.x / S, split = blockIdx.x % S;
  const int n0 = tile * 128;
  const int KB = k / 64;
  const int kb0 = split * KB / S, kb1 = (split + 1) * KB / S;
  const int nk = kb1 - kb0;

  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) { mbar_init(&full[s], 1); mbar_init(&empty[s], 8); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();

  if (warp == 8) {
    if (lane == 0) {
      for (int i = 0; i < nk; ++i) {
        const int s = i % STAGES;
        if (i >= STAGES) mbar_wait(&empty[s], ((i / STAGES) - 1) & 1);
        mbar_expect_tx(&full[s], STAGE_BYTES);
        const int kc = (kb0 + i) * 64;
        unsigned char* sb = smem + (size_t)s * STAGE_BYTES;
        const int wrow = (tile * KB + kb0 + i) * 128;
        tma_load_2d(sb, &w_map, 0, wrow, &full[s]);
        tma_load_2d(sb + 8192, &w_map, 0, wrow + 64, &full[s]);
        tma_load_2d(sb + W_BYTES, &x_map, kc, 0, &full[s]);
      }
    }
    return;
  }

  const int wg = warp >> 2;
  float acc[NT][4];
#pragma unroll
  for (int j = 0; j < NT; ++j)
#pragma unroll
    for (int r = 0; r < 4; ++r) acc[j][r] = 0.f;

  int prev = -1;
  for (int i = 0; i < nk; ++i) {
    const int s = i % STAGES;
    mbar_wait(&full[s], (i / STAGES) & 1);
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    wgmma_fence();
    unsigned char* sb = smem + (size_t)s * STAGE_BYTES;
    const uint64_t da = make_smem_desc(sb + wg * 8192), db = make_smem_desc(sb + W_BYTES);
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) wgmma_m64n<BT>(acc, da + (uint64_t)(ks * 2), db + (uint64_t)(ks * 2));
    wgmma_commit();
    wgmma_wait<1>();
    if (prev >= 0 && lane == 0) mbar_arrive(&empty[prev]);
    prev = s;
  }
  wgmma_wait<0>();
  bar_consumers();
  DBG(1);

  constexpr int NE = NT * 4;
  const int ctid = tid;
  float* my_ws = ws + (size_t)(tile * S + split) * NE * 256;
#pragma unroll
  for (int j = 0; j < NT; ++j)
#pragma unroll
    for (int r = 0; r < 4; ++r) __stcg(&my_ws[(j * 4 + r) * 256 + ctid], acc[j][r]);
  bar_consumers();
  DBG(2);
  if (tid == 0) {
    asm volatile("red.release.gpu.global.add.s32 [%0], 1;\n" ::"l"(&cnt[tile]) : "memory");
    int v;
    do {
      asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(&cnt[tile]) : "memory");
    } while (v < S);
  }
  bar_consumers();
  DBG(3);
  const float* tws = ws + (size_t)tile * S * NE * 256;
  const int e0 = split * NE / S, e1 = (split + 1) * NE / S;
  const int lg = (lane >> 2), lq = lane & 3;
  const int row_base = wg * 64 + (warp & 3) * 16 + lg;
  for (int e = e0; e < e1; ++e) {
    float v = __ldcg(&tws[e * 256 + ctid]);
#pragma unroll
    for (int sp = 1; sp < S; ++sp) v += __ldcg(&tws[(size_t)sp * NE * 256 + e * 256 + ctid]);
    const int j = e >> 2, r = e & 3;
    const int tok = 8 * j + 2 * lq + (r & 1);
    const int row = row_base + (r >> 1) * 8;
    if (tok < m) out[(size_t)tok * n + n0 + row] = __float2bfloat16(v);
  }
  bar_consumers();
  DBG(4);
  if (tid == 0) {
    if (atomicAdd(&cnt[64 + tile], 1) == S - 1) {
      cnt[tile] = 0;
      cnt[64 + tile] = 0;
    }
  }
}

template <int BT, int STAGES>
void run_qkv_mc(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& out, torch::Tensor& ws,
                torch::Tensor& cnt) {
  constexpr int S = 7;
  const int m = x.size(0), k = x.size(1), n = out.size(1);
  TORCH_CHECK(n % 128 == 0 && k % 64 == 0 && k / 64 >= S * STAGES / 2, "qkv_mc: shape");
  const int tiles = n / 128;
  TORCH_CHECK((int64_t)tiles * S * BT * 128 <= ws.numel() && tiles <= 64 && cnt.numel() >= 128, "qkv_mc: workspace");
  const int smem = STAGES * (128 * 64 * 2 + BT * 128) + 2 * STAGES * 8 + 16;
  CUtensorMap w_map = make_tma_2d(w.data_ptr(), tiles * (k / 64) * 128, 64, 64, 64);
  CUtensorMap x_map = make_tma_2d(x.data_ptr(), m, k, (int)x.stride(0), BT);
  auto kern = qkv_mc_kernel<BT, STAGES, S>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(kern),
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  kern<<<tiles * S, 288, smem, at::cuda::getCurrentCUDAStream()>>>(
      w_map, x_map, static_cast<bf16*>(out.data_ptr()), ws.data_ptr<float>(), cnt.data_ptr<int>(), m, n, k);
  C10_CUDA_CHECK(cudaGetLastError());
}

#ifdef MC_DBG
void get_dbg(torch::Tensor o) {
  C10_CUDA_CHECK(cudaMemcpyFromSymbol(o.data_ptr(), g_dbg, sizeof(unsigned long long) * 256 * 6));
}
#endif
void launch_qkv_mc(torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && out.is_cuda(), "qkv_mc: CUDA tensors expected");
  const int m = x.size(0);
  if (m == 128) { run_qkv_mc<128, 7>(x, w, out, ws, cnt); return; }
  if (m == 256) { run_qkv_mc<256, 4>(x, w, out, ws, cnt); return; }
  TORCH_CHECK(false, "qkv_mc: m not compiled");
}
"""
MC_CPP_DBG = "void get_dbg(torch::Tensor o);"
MC_CPP = ("void launch_qkv_mc(torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor ws,"
          " torch::Tensor cnt);")
