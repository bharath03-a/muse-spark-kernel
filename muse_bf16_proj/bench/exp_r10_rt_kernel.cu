template <>
__device__ __forceinline__ void wgmma_m64n<64>(float (&d)[8][4], uint64_t da, uint64_t db) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
        "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
        "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
        "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
        "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
        "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
        "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3])
      : "l"(da), "l"(db));
}

// Router m in {128, 256}: 4 expert tiles (64 rows) x RSK_S k-splits = 128 CTAs, one per SM. Each CTA loads its whole
// k-slice up front (hi, lo, x), runs hi and lo wgmmas into one fp32 accumulator (2 warpgroups split the tokens),
// writes its partial, then after all splits of the tile arrive reduces 1/RSK_S of the tile in fixed split order.
constexpr int RSK_S = 32, RSK_STEPS = 4, RSK_SPIN = 1 << 22, RSK_PAD = 68;

__device__ __forceinline__ int rsk_ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}

template <int BT>
__global__ void __launch_bounds__(256, 1)
router_sk(const __grid_constant__ CUtensorMap hi_map, const __grid_constant__ CUtensorMap lo_map,
          const __grid_constant__ CUtensorMap x_map, float* __restrict__ out, float* __restrict__ ws,
          int* __restrict__ cnt, int m) {
  constexpr int NH = BT / 2, NT = NH / 8, WB = 64 * 64, XE = BT * 64, STAGE_E = 2 * WB + XE;
  constexpr uint32_t STAGE_BYTES = STAGE_E * sizeof(bf16);
  constexpr int TPS = BT / RSK_S, SV4 = TPS * 16, NG = 256 / SV4, PG = RSK_S / NG;
  extern __shared__ __align__(1024) unsigned char smem[];
  bf16* st = reinterpret_cast<bf16*>(smem);
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + RSK_STEPS * STAGE_BYTES);
  float4* gsum = reinterpret_cast<float4*>(smem + RSK_STEPS * STAGE_BYTES + 64);

  const int tid = threadIdx.x, wg = tid >> 7;
  const int tile = blockIdx.x / RSK_S, s = blockIdx.x % RSK_S;
  const int e0 = tile * 64, k0 = s * RSK_STEPS * 64;

  if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];\n" ::"l"(reinterpret_cast<uint64_t>(&hi_map)) : "memory");
    asm volatile("prefetch.tensormap [%0];\n" ::"l"(reinterpret_cast<uint64_t>(&lo_map)) : "memory");
    asm volatile("prefetch.tensormap [%0];\n" ::"l"(reinterpret_cast<uint64_t>(&x_map)) : "memory");
    for (int i = 0; i < RSK_STEPS; ++i) mbar_init(&bars[i], 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    for (int i = 0; i < RSK_STEPS; ++i) {
      bf16* b = st + i * STAGE_E;
      mbar_expect_tx(&bars[i], STAGE_BYTES);
      tma_load_2d(b, &hi_map, k0 + i * 64, e0, &bars[i]);
      tma_load_2d(b + WB, &lo_map, k0 + i * 64, e0, &bars[i]);
      tma_load_2d(b + 2 * WB, &x_map, k0 + i * 64, 0, &bars[i]);
    }
  }
  __syncthreads();

  float acc[NT][4];
#pragma unroll
  for (int j = 0; j < NT; ++j)
#pragma unroll
    for (int r = 0; r < 4; ++r) acc[j][r] = 0.f;

#pragma unroll
  for (int i = 0; i < RSK_STEPS; ++i) {
    mbar_wait(&bars[i], 0);
    wgmma_fence();
    const bf16* b = st + i * STAGE_E;
    const uint64_t dh = make_smem_desc(b), dl = make_smem_desc(b + WB);
    const uint64_t dx = make_smem_desc(b + 2 * WB + wg * NH * 64);
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      wgmma_m64n<NH>(acc, dh + (uint64_t)(ks * 2), dx + (uint64_t)(ks * 2));
      wgmma_m64n<NH>(acc, dl + (uint64_t)(ks * 2), dx + (uint64_t)(ks * 2));
    }
    wgmma_commit();
  }
  wgmma_wait<0>();
  __syncthreads();

  float* P = reinterpret_cast<float*>(smem);  // [BT][RSK_PAD]: token-major partial, reuses the stages
  {
    const int w = (tid >> 5) & 3, lane = tid & 31, g = lane >> 2, q = lane & 3;
    const int r0 = w * 16 + g;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const int c = wg * NH + 8 * j + 2 * q;
      P[c * RSK_PAD + r0] = acc[j][0];
      P[(c + 1) * RSK_PAD + r0] = acc[j][1];
      P[c * RSK_PAD + r0 + 8] = acc[j][2];
      P[(c + 1) * RSK_PAD + r0 + 8] = acc[j][3];
    }
  }
  __syncthreads();
  float* part = ws + (size_t)(tile * RSK_S + s) * BT * 64;
#pragma unroll 4
  for (int v = tid; v < BT * 16; v += 256) {
    const int tk = v >> 4, e4 = (v & 15) << 2;
    __stcg(reinterpret_cast<float4*>(part + tk * 64 + e4), *reinterpret_cast<const float4*>(&P[tk * RSK_PAD + e4]));
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    atomicAdd(&cnt[tile], 1);
    for (int it = 0; it < RSK_SPIN && rsk_ld_acquire(&cnt[tile]) < RSK_S; ++it) {
    }
  }
  __syncthreads();

  {
    const int v = tid % SV4, grp = tid / SV4;
    const int tk = s * TPS + (v >> 4), e4 = (v & 15) << 2;
    const float* src = ws + (size_t)tile * RSK_S * BT * 64 + (size_t)tk * 64 + e4;
    float4 p[PG];
#pragma unroll
    for (int i = 0; i < PG; ++i) p[i] = __ldcg(reinterpret_cast<const float4*>(src + (size_t)(grp * PG + i) * BT * 64));
    float4 a = p[0];
#pragma unroll
    for (int i = 1; i < PG; ++i) { a.x += p[i].x; a.y += p[i].y; a.z += p[i].z; a.w += p[i].w; }
    gsum[grp * SV4 + v] = a;
  }
  __syncthreads();
  if (tid < SV4) {
    const int tk = s * TPS + (tid >> 4), e4 = (tid & 15) << 2;
    float4 a = gsum[tid];
#pragma unroll
    for (int gg = 1; gg < NG; ++gg) {
      const float4 b = gsum[gg * SV4 + tid];
      a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
    }
    if (tk < m) *reinterpret_cast<float4*>(out + (size_t)tk * 256 + e0 + e4) = a;
  }
  if (tid == 0) {
    // the last CTA past the wait resets both counters for the next launch
    if (atomicAdd(&cnt[4 + tile], 1) == RSK_S - 1) {
      atomicExch(&cnt[tile], 0);
      atomicExch(&cnt[4 + tile], 0);
    }
  }
}

template <int BT>
void launch_rsk_t(const torch::Tensor& x, const torch::Tensor& hl, torch::Tensor& out, torch::Tensor& ws,
                  torch::Tensor& cnt) {
  const int m = x.size(0), k = x.size(1);
  TORCH_CHECK(m >= 1 && m <= BT && k == RSK_S * RSK_STEPS * 64 && hl.size(0) == 2 && hl.size(1) == 256 &&
              hl.size(2) == k && out.size(1) == 256, "router_sk: shape");
  TORCH_CHECK(ws.numel() >= (int64_t)4 * RSK_S * BT * 64 && cnt.numel() >= 8, "router_sk: workspace");
  constexpr int SV4 = BT / RSK_S * 16;
  const int smem = RSK_STEPS * (2 * 64 * 64 + BT * 64) * (int)sizeof(bf16) + 64 + 256 / SV4 * SV4 * 16;
  const bf16* hp = static_cast<const bf16*>(hl.data_ptr());
  CUtensorMap hi_map = make_tma_2d(hp, 256, k, k, 64);
  CUtensorMap lo_map = make_tma_2d(hp + (size_t)256 * k, 256, k, k, 64);
  CUtensorMap x_map = make_tma_2d(x.data_ptr(), m, k, (int)x.stride(0), BT);
  auto kern = router_sk<BT>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(kern),
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  kern<<<4 * RSK_S, 256, smem, at::cuda::getCurrentCUDAStream().stream()>>>(
      hi_map, lo_map, x_map, out.data_ptr<float>(), ws.data_ptr<float>(), cnt.data_ptr<int>(), m);
  C10_CUDA_CHECK(cudaGetLastError());
}

void launch_rsk(torch::Tensor x, torch::Tensor hl, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt,
                int64_t bt) {
  TORCH_CHECK(x.is_cuda() && hl.is_cuda() && out.is_cuda() && out.scalar_type() == at::kFloat, "router_sk: args");
  if (bt == 128) { launch_rsk_t<128>(x, hl, out, ws, cnt); return; }
  if (bt == 256) { launch_rsk_t<256>(x, hl, out, ws, cnt); return; }
  TORCH_CHECK(false, "router_sk: bt not compiled");
}
