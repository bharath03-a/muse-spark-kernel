# Triton / Gluon ideas before moving to CUDA C++ (Triton 3.8.0, H200)

Baseline: judge 1.235× (cp1, job 2f5964e97a3648a9). **Leverage:** 1 µs saved on one shape = 0.40% geomean on
decode1, 0.33% on verify1, 0.27% on decode16, 0.25% on decode32, 0.21% on verify16, 0.18% on verify32
(1/6 of 1/load-total).

## 0. What I measured (one H200 run, ~2 min, `bench/gluon_probe.py`)

**Gluon works on sm_90 in Triton 3.8.0.** `gl.warp_specialize` with a 1-warp TMA producer and a 4-warp wgmma
consumer compiles, runs and is correct. So does a single-partition TMA + mbarrier ring, and so does wgmma
with **A from registers** (`DotOperandLayout(0, mma, k_width=2)`). The sm_90 warp-specialize failure we hit
before is in the *automatic* pass (`tl.range(warp_specialize=True)` → `add_hopper_warpspec`). Gluon skips the
whole TTGIR optimisation pipeline (`gluon_to_ttgir` runs no pipeliner and no autoWS), so that bug doesn't
apply.

Swap-AB GEMV, BN=64 W rows (wgmma M), BT=16/32 tokens (wgmma N), BK=128, 8-stage ring, no split. Times are
local cold-L2 graph replays, which include ~4 µs graph cost (judge ≈ local − 4):

| shape | current Triton (`_bf16_gemm`) | Gluon pipe | Gluon WS | Gluon reg-A |
|---|---|---|---|---|
| pooler m=16 | 23.09 | **21.86** (−1.2) | 22.02 | 22.66 |
| unpooler m=16 | 22.37 | **21.86** (−0.5) | 22.08 | 22.30 |
| pooler m=32 | 24.58 (split 2) | **22.50** (−2.1) | 22.64 | 22.91 |
| qkv m=16 (36 CTAs!) | 17.63 (split 4) | 18.40 (+0.8) | 18.72 | 21.79 |

What the numbers say:
- An *untuned* Gluon TMA ring already beats the tuned Triton pipeliner on the big streaming shapes by
  3–10% of kernel time. Pooler gets there with only **64 CTAs**.
- Warp specialization adds nothing for pure streaming: the pipe and WS variants are within ±1%. WS matters only
  where MMA and memory both matter (m=128/256).
- qkv with no split runs **36 CTAs at ~2.6 TB/s, i.e. ~73 GB/s per SM**. So a TMA-fed SM can pull roughly
  70–75 GB/s, and ≥60 streaming CTAs saturate HBM. Without a split, qkv is capped by its CTA count.
- BN=128 with 8 warps was bad (32 µs; probably layout or registers; I didn't investigate). BK=64 with 16 stages is
  worse than BK=128 with 8 stages: bigger TMA boxes help.
- Reg-A costs 1–4% in smem→register traffic, but it's what makes in-register decompression possible (idea 6).

## Ranked list (expected geomean gain per H200-minute)

| # | Idea | Shapes | Expected | Conf. | Effort | Test cost |
|---|---|---|---|---|---|---|
| 1 | Gluon TMA-ring GEMV for pooler/unpooler m 2–32 (drop-in) | pooler m8/16/32, unpooler m16/32 | −0.5…−2 µs each → **+1.5–2%** | high (measured) | 0.5 day | 1 sweep run |
| 2 | Gluon ring + split-K 2/3 for qkv m≤32 (≥72 CTAs) | qkv m8/16/32 | −2.5…−3.5 µs each → **+2–2.8%** | medium | 0.5 day on top of #1 | 1 run |
| 3 | Router m=128/256: read x once, two dots per x tile, no swap-AB | router m128/256 | −4…−6 µs each → **+1.7–2.2%** | medium | 0.5 day (plain Triton) | 1 run |
| 4 | Distributed split-K reduce (every split CTA reduces 1/S of the tile; cooperative grid) | qkv/pooler m=128/256, router m≥16 | −1…−3 µs on big-tile splits → **+0.5–1.5%** | medium | 0.5 day | 1 run |
| 5 | Gluon WS GEMM for m=128/256 (TMA producer + 2 consumer WGs, swap-AB for qkv) | qkv/pooler/unpooler/o m128/256 | −2…−5 µs each → **+2–5%** (wide range) | low-medium | 2–3 days | several runs |
| 6 | Compression in Gluon: decode in registers → wgmma A from registers | all BF16 m≤128 | bytes −25% → up to **+6–10%** | low-medium | 3–5 days | several runs |
| 7 | Tile-sweep knobs on #1/#2: BK 128/256, stages, 1 vs 2 CTAs/SM, maxnreg, `num_ctas`/multicast x | m 2–32 | +0.3–0.8% | medium | hours | 1 sweep |
| 8 | Router m 9–16 → the CUDA cluster path (today it falls into Triton `_skinny_gemm`) | router m16 | −1…−2 µs → +0.3–0.5% | medium | 1 hour (table edit) | 1 run |
| 9 | Inline-asm loads (`ld.global.nc.L1::no_allocate.L2::256B`), `.cg/.cs`, prefetch | m=1 FMA | ≤0.3% | low | hours | – |
| 10 | Disable autotuning/SLP issues, `enable_fp_fusion`, `launch_pdl` | – | 0 / forbidden | – | – | – |

### 1. Gluon TMA-ring GEMV (m 2–32, pooler / unpooler) — **do first**
- Kernel: `gemv_pipe` in `bench/gluon_probe.py`. W [BN=64, BK=128] and x [BT, BK] go through one TMA
  descriptor each into an S-stage mbarrier ring, with `warpgroup_mma(w_s, x_s.permute((1,0)), acc, is_async=True)`
  and `warpgroup_mma_wait(1)`, and each slot is refilled once the wgmma that read it retires. TMA zero-fills
  x rows ≥ m, so masks and padding are free. The x descriptor is built on the host per call. It is a
  by-value kernel argument, so graph capture is fine; check the build cost with the judge's per-call overhead.
- Measured locally: −1.2 (pooler m16), −0.5 (unpooler m16), −2.1 µs (pooler m32). Expected m8 to behave like m16.
- To test: add `stages∈{6,8,10}`, `BK∈{128,256}`, BN 64 and grid N/64. Run pooler/unpooler at m 8/16/32 in
  `sweep_gemv.py` style, then put the winner behind `Tile.kind="gluon"`. One run.
- Risk: Gluon is "experimental", and its API changes between versions. That's fine here, since the image is
  pinned at 3.8.0.

### 2. qkv m≤32: same kernel + split-K so ≥72 CTAs stream
- Evidence: with no split, 36 CTAs × ~73 GB/s gives ~2.6 TB/s. The current split-4 kernel (144 CTAs, cuBLAS at m32) is at
  13.4–14.4 µs on the judge against a 7.9 µs floor. qkv's 37.7 MB fits in L2, so the profiler's "GPU time"
  (10.5 µs) is warm-L2. Judge cold-L2 time is ~13.7 µs, i.e. **~2.75 TB/s**, so qkv is our worst-bandwidth big
  shape.
- Design: split 2 (72 CTAs, 32 k-blocks each) or 3 (108 CTAs, 21/21/22 k-blocks; uneven K ranges are fine in
  Gluon). Partials are 64×BT fp32 (4–8 KB per CTA), reduced by the last CTA as today, or as in #4. Target ~10 µs.
  → −3 µs × (0.33+0.27+0.25)% ≈ **+2.5%**.
- To test: add `SPLIT` plus the existing `_split_k_finish` logic (write it in Gluon with `gl.atomic_add(sem="acq_rel")`).
  Benchmark split 1/2/3/4 at m 8/16/32. One run.

### 3. Router m=128/256 (10.7 and 13.0 µs on the judge against a 2.2/2.7 µs floor) — plain Triton
- Today: swap-AB `_skinny_gemm` over [hi|lo] with K=16384, BN=64 → 4 n-tiles × 2–4 t-tiles × split. Every n-tile
  re-reads all of x, *twice* (hi and lo halves via `k % KX`). At m=256 that's ~8× 4 MB = 32 MB of x L2 traffic
  for 8 MB of weights.
- Fix: put tokens on M. x tile via TMA [BM=128, BK=64], loaded once per k-step. W_hi and W_lo tiles
  [BN=256 (all experts), BK] give **two `tl.dot`s per x tile into one fp32 accumulator**; this is the
  untested idea from the plan. Grid: m/128 × split 16–32 (64–128 CTAs). x is read once and W m/128 times
  (L2). The split partial is 128×256 fp32 = 128 KB/CTA, so use the distributed reduce (#4) or split ≤8 with
  BN=128 halves.
- Expected 5–7 µs each → **+1.7–2.2%**. To test: one new Triton kernel in `bench/sweep_router.py`. One run.

### 4. Split-K reduce that doesn't serialise on one SM
- With big tiles the last-CTA reduce is a single-SM read of S × tile bytes. Example: qkv m=128 BN=128 split 7
  is 7 × 64 KB = 448 KB through one SM at ~70–100 GB/s ≈ 4–6 µs. (For m≤32 partials are tiny, which is why
  the CUDA cluster/DSMEM experiment was flat there.)
- Fix: each of the S CTAs writes its partial and arrives on the tile counter, then spins until count == S
  and reduces rows `[r·BM/S, (r+1)·BM/S)` of the tile. Every CTA must be resident: use
  `launch_cooperative_grid=True` (a 3.8 `CUDAOptions` field). Confirm a cooperative launch captures into a
  CUDA graph; I haven't tested that. If it doesn't, keep grid ≤ #SMs × occupancy, which is guaranteed in practice
  but not formally. Alternative inside a cluster: Gluon `num_ctas>1` + `hopper.async_store`/DSMEM, as in
  `reg_gemm`.
- Applies to: #3, #5, and today's `_big_gemm` split 2. Expect −1…−3 µs on large-tile splits.

### 5. m=128/256 GEMMs in Gluon (qkv, pooler, unpooler, o): where cuBLAS is beatable
- Gaps on the judge: qkv m128 17.3 (cuBLAS splitK + a separate 3.1 µs reduce kernel), qkv m256 19.7, pooler m128
  23.3 (our `_big_gemm`, ~2.75 TB/s), unpooler m128/256 20.2/24.6, o m128/256 7.4/9.6. Intensity m=128..256 sits near
  the H200 ridge (206), so these need MMA *and* HBM overlapped. That's exactly what Gluon WS gives (1 TMA warp,
  2 consumer warpgroups with `worker_num_regs=[24]` and `maxnreg`), and what Triton's autoWS can't do on sm_90.
- Plan: start from Triton's `python/tutorials/gluon/07-persistence.py` (Hopper wgmma persistent matmul) and
  add the 08 WS partitioning. qkv: swap-AB (W rows on M=128 via 2 WGs, 128/256 tokens on N), split 3–4 with
  the #4 reduce. pooler/unpooler m128: one HBM pass of W (n-tile owns all tokens; BN_w=128 × BT=128).
- Expected −2…−5 µs per shape, **+2–5%** in total. Low confidence: cuBLAS's nvjet kernels are already WS+TMA.
  Our edge is only shape-specific (no reduce kernel, a single HBM pass, exact-m tiles). Effort 2–3 days.

### 6. Compression without the smem round trip — now expressible in Gluon
- Why Triton lost (0.6–0.75×): `tl.dot` wants operands in smem, so decoded bf16 got written back to smem and
  re-read. Gluon's Hopper `warpgroup_mma(a, b, ...)` accepts **`a` as a register tensor in
  `DotOperandLayout(0, mma, k_width=2)`**. Measured: it works and is correct, and costs 1–4% against smem-A on
  uncompressed W.
- Path: TMA the compressed planes (u8 sign+mantissa, 4-bit exponent codes) into the ring → `smem.load(L)` in a
  layout L whose per-thread elements are exactly the thread's wgmma A fragment → decode elementwise
  (`gl.inline_asm_elementwise` with `prmt/lop3`, 2–4 weights per op) → `wgmma(a_regs, x_smem)`.
  To keep loads vectorised, permute K inside each 32-wide block at `prepare()` time, so a thread's fragment
  (k {2q,2q+1,2q+8,2q+9} per k16, the same trick `reg_gemm` uses) is contiguous. Build L as a
  `DistributedLinearLayout` and `gl.convert_layout` it to the DotOperandLayout; that's free when the linear
  layouts match, and the TTGIR will show whether a real convert was emitted. x needs no permutation: permuting
  K identically for W and x leaves the dot unchanged, and x goes into the smem B operand with the same
  permuted K order via an index gather, which is cheap because x is tiny.
- Upside: −25% bytes on every BF16 streaming shape. Only if the decode stays under the memory shadow, which is
  doubtful at m≥16 with 4 consumer warps. Use 2 consumer warpgroups. Effort 3–5 days. **This is the one idea that
  could move us into top-3 range and still stay in Python.**
- Cheap go/no-go first (one run): take `gemv_pipe` with `REG_A=True`, insert a fake decode (e.g. 6 `lop3/prmt`
  per pair via inline asm, on bf16 bits) and see whether pooler m=16 stays within 5% of 17.9 µs (local − 4).

### 7. Knobs to sweep on #1/#2 (cheap, small)
- `BK=256` (32 KB boxes), stages 5–10, `maxnreg` 128–232 (with WS, the default partition gets what's left after
  `worker_num_regs`).
- `num_ctas=2` clusters + `tma.async_load(..., multicast=True)` for x, so cluster peers share one x fetch. That
  matters only where x L2 re-reads are big (m=32 with many n-tiles). Probably ≤0.3%.
- 2 CTAs/SM (smem ≤113 KB each) for pooler at 64 tiles vs. split-K 2 at 128 tiles; per-SM ~73 GB/s says ≥60
  CTAs are needed, so both are at the edge.

### 8. Router m=16 uses the slow fallback
- `CUDA_TILES["router"]` covers m 1, 2–8 and 17–32. m 9–16 falls into Triton `_skinny_gemm` (judge 6.74 µs; m=8 is
  5.93 on CUDA). Add `(9, 16, "cluster", ...)` or a Gluon router GEMV and test with the existing sweep. Expect
  −1…−2 µs → +0.3–0.5%.

### 9–10. Low value or not allowed
- Load hints: `.ca/.cg/.cs/.cv` and `eviction_policy` exist. `L1::no_allocate`/`L2::256B` needs
  `inline_asm_elementwise` pointer loads. Eviction hints were neutral before, so expect ≤0.3%.
- `tl.extra.cuda.gdc_wait / gdc_launch_dependents` and `launch_pdl=True` are **PDL, forbidden by the rules**.
- Persistent single-wave scheduling for m 2–32: Gluon makes it easy (`program_id` loop + `tl.extra.cuda.smid`
  for diagnostics). It only helps where CTA count isn't a multiple of 132, and the per-SM cap suggests
  more, smaller units instead. Fold into #7.

## Triton 3.8-specific notes
- `CUDAOptions` in 3.8: `num_warps, num_ctas, num_stages, maxnreg, enable_fp_fusion, enable_reflect_ftz,
  launch_cooperative_grid, launch_pdl, ptx_options, ir_override`. There's **no `cluster_dims` option**: clusters come
  from `num_ctas` (CGA layouts in Gluon). `ir_override` lets you hand-edit TTGIR/PTX for a single experiment.
- `tl.range(num_stages, loop_unroll_factor, disallow_acc_multi_buffer, flatten, warp_specialize, disable_licm)`
  all exist. On sm_90, `warp_specialize=True` goes through `add_hopper_warpspec`, which is what failed for us.
  Gluon avoids it.
- **3.8 sm_90 workaround that touches us:** `make_llir` disables LLVM's SLP vectorizer when the TTGIR has both
  `bf16` and `tt.reduce` (llvm#203741). Our m=1 FMA kernel `_bf16_gemv` matches (bf16 loads + `tl.sum`), so it
  compiles without SLP. Probably harmless since it's memory-bound, but worth a check: compare its SASS
  load widths (`LDG.E.128`) with an fp32-reduce variant.
- Gluon Hopper API present in 3.8: `tma.async_load(multicast=)`, `tma.async_store`, `tma.async_atomic_add`
  (bulk reduce-add to global; fp32 split-K without counters, but nondeterministic and it needs a zeroed
  output), `mbarrier.*`, `cluster.arrive/wait/barrier`, `async_store` to a peer CTA's smem (DSMEM),
  `warpgroup_mma` (A from smem or registers), `ampere.mma_v2` (m16n8k16 with N=8, for m≤8), and `warp_specialize`
  with `worker_num_regs`.
- Caveat: I read the Python sources from the PyPI 3.8.0 wheel, not from the image; the probe's run in the image
  agrees with them.

## Gluon swap-AB GEMV sketch (the measured kernel, condensed)
```python
@gluon.jit
def gemv_pipe(w_desc, x_desc, out_ptr, M, N: gl.constexpr, K: gl.constexpr, STAGES: gl.constexpr,
              num_warps: gl.constexpr):
    BN: gl.constexpr = w_desc.block_type.shape[0]; BK: gl.constexpr = w_desc.block_type.shape[1]
    BT: gl.constexpr = x_desc.block_type.shape[0]; KI: gl.constexpr = K // BK
    mma: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[num_warps, 1],
                                                  instr_shape=[16, BT, 16])
    n0 = gl.program_id(0) * BN
    w_bufs = gl.allocate_shared_memory(w_desc.dtype, [STAGES] + w_desc.block_type.shape, w_desc.layout)
    x_bufs = gl.allocate_shared_memory(x_desc.dtype, [STAGES] + x_desc.block_type.shape, x_desc.layout)
    ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(STAGES):
        mbarrier.init(ready.index(i), count=1)
    nbytes: gl.constexpr = w_desc.block_type.nbytes + x_desc.block_type.nbytes
    for i in gl.static_range(STAGES - 1):                      # prologue
        mbarrier.expect(ready.index(i), nbytes)
        tma.async_load(w_desc, [n0, i * BK], ready.index(i), w_bufs.index(i))
        tma.async_load(x_desc, [0, i * BK], ready.index(i), x_bufs.index(i))
    acc = gl.zeros([BN, BT], gl.float32, layout=mma)
    for i in range(KI):
        s = i % STAGES
        mbarrier.wait(ready.index(s), (i // STAGES) & 1)
        acc = warpgroup_mma(w_bufs.index(s), x_bufs.index(s).permute((1, 0)), acc, is_async=True)
        acc = warpgroup_mma_wait(1, deps=(acc,))               # wgmma i-1 retired -> its slot is free
        j = i + STAGES - 1; sj = j % STAGES; p = j < KI
        mbarrier.expect(ready.index(sj), nbytes, pred=p)
        tma.async_load(w_desc, [n0, j * BK], ready.index(sj), w_bufs.index(sj), pred=p)
        tma.async_load(x_desc, [0, j * BK], ready.index(sj), x_bufs.index(sj), pred=p)
    acc = warpgroup_mma_wait(0, deps=(acc,))
    offs_n = n0 + gl.arange(0, BN, layout=gl.SliceLayout(1, mma))
    offs_t = gl.arange(0, BT, layout=gl.SliceLayout(0, mma))
    gl.store(out_ptr + offs_t[None, :] * N + offs_n[:, None], acc.to(gl.bfloat16), mask=(offs_t < M)[None, :])

# host: from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
# wl = gl.NVMMASharedLayout.get_default_for([64, 128], gl.bfloat16); same for x with [BT, 128]
# gemv_pipe[(N // 64,)](TensorDescriptor.from_tensor(w, [64, 128], wl), TensorDescriptor.from_tensor(x, [BT, 128], xl),
#                       out, m, N=N, K=K, STAGES=8, num_warps=4)
```
For split-K (#2): grid `(N//64, SPLIT)`, k range `[pid_s*KI//SPLIT, (pid_s+1)*KI//SPLIT)`, then the existing
counter epilogue. For the WS variant: `gl.warp_specialize([(consumer, args), (producer, args)], [1], [24])` with
`maxnreg=168`, as in `gemv_ws` in the probe.

## Recommendation
**Worth doing in Triton/Gluon now (≈1.5 days, ~4–6 checks of H200 time):** #1 + #2 (Gluon ring GEMV with split
for qkv), #3 (router m=128/256 restructure, plain Triton), #8 (router m=16 table fix), then #4. Together
**~+5–7% geomean → ~1.30–1.32×**, all in Python with no new toolchain. #1 is measured; #2 and #3 are cheap to
check.

**Where to stop and use C++ (or commit to deep Gluon):**
- **m=128/256 vs cuBLAS (#5):** Gluon *can* now express WS+TMA+wgmma on sm_90. So this is no longer a Triton
  wall; it's a "write CUTLASS-grade kernels by hand" job in either language. If the team is already fluent in
  CUDA/CUTLASS (the cluster `reg_gemm` exists), CUTLASS 3.x sm90 WS kernels with custom tile shapes are the
  lower-risk route to the ~5%.
- **Compression (#6):** Gluon removes the hard blocker (register-A wgmma works), but decode speed decides the
  outcome, and PTX-level control (`prmt/lop3` scheduling, register pressure) is easier in CUDA. Run the fake-decode
  go/no-go in Gluon first (1 run). If decode overhead is >10% at m=16, do it in CUDA ZipGEMM-style.
- **Small-shape fixed costs** (o, unpooler slice, router m≤8 at 4–6 µs against 1.8–3.5 floors) are launch/
  prologue/latency-bound. Neither language helps much more; the CUDA versions already exist.

Sources: Gluon tutorials [05-wgmma](https://triton-lang.org/main/getting-started/tutorials/gluon/wgmma.html),
[08-warp-specialization](https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/08-warp-specialization.py),
[07-persistence](https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/07-persistence.py),
[gl.warp_specialize API](https://triton-lang.org/main/gluon/api/generated/triton.experimental.gluon.language.warp_specialize.html),
[PyTorch: warp specialization in Triton](https://pytorch.org/blog/warp-specialization-in-triton-design-and-roadmap/),
[Tilus Hopper WS matmul](https://nvidia.github.io/tilus/latest/tutorials/matmul-hopper/v3.html); Triton 3.8.0 wheel
sources (`backends/nvidia/compiler.py`, `experimental/gluon/language/nvidia/hopper/*`).
