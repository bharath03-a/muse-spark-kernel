# muse_bf16_proj — complete state analysis (2026-10-27)

Purpose: hand-off brief for a fresh expert review. Everything below is measured unless marked *(inferred)*.

## 1. Problem and scoring
- Open Kernel Challenge, H200 (132 SMs, 60 MB L2, HBM3e). `out = x @ W.T`, x bf16 [m, k]; W bf16 [n, k] except router (fp32 W, fp32 out; allowed as bf16 hi+lo pair with fp32 accumulation).
- Layers (n×k): qkv_gate 2304×8192 (37.7 MB), o 8192×1024 (16.8 MB), pooler 4096×8192 (67 MB), unpooler 8192×4096 (67 MB; for m≤8 only the first 1024 rows = 8.4 MB "slice"), router 256×8192 fp32 (8.4 MB).
- m ∈ {1, 8, 16, 32, 128, 256}. A "load" = sum of the 5 layers at one m. **Score = geomean over the 6 loads of (SGLang load time / ours)** — each load counts equally, so a µs saved on a small load is worth more.
- Timing: each shape is a CUDA graph replayed with cold L2 (the judge reads a >L2 buffer first), paired against SGLang, median of paired ratios, ~4 µs graph cost subtracted. Judge noise ≈ ±0.4 µs per shape, ±1.5% overall.
- Rules: lossless repack/compression only (precision and values unchanged; dequant in-kernel OK); prepare() untimed ≤300 s; fresh output tensor every call; no PDL, no L2 persistence, no caching x/outputs, no host sync or value-dependent branching in kernel_fn; `load_inline` CUDA allowed; whole run ≤10 min incl. compile. Final ranking = medians of reruns, incl. held-out weights and nearby shapes.
- Weights in the judge: `randn(n,k)/sqrt(k)` → bf16 (random each run). x: randn, 1% "outlier" channels ×30, plus probe rows (one-hot power-of-two) checked to 1e-3.
- Limits: 20 checks + 10 submits/day per problem. Deadline Nov 4 2026 23:59 PT.

## 2. Standing
- Board best 1.409× (rank ~181 of 1112). Latest candidate checkpoints/cp13_B_check1.415.py: 1.415× (score 3412.9 µs); being submitted.
- Leader 1.637×, top-20 ≈ 1.52×, 1.5× ≈ rank 121. Gap to 1.5× = −6% load time on average; to 1.6× = −12%.

## 3. Per-shape judge µs (cp13, job bc3cf943) — SGLang / **ours** / "sol" (judge's 4.8 TB/s, 989 TFLOP/s bound)
| load | qkv_gate | o | pooler | unpooler | router | load sum ours / SGLang | load speedup |
|---|---|---|---|---|---|---|---|
| m=1 | 13.9 / **9.2** / 7.9 | 6.7 / **5.4** / 3.5 | 21.3 / **13.8** / 14.0 | 6.9 / **3.7** / 1.75 (slice) | 10.2 / **3.7** / 1.75 | 35.7 / 59.0 | 1.65 |
| m=8 | 13.7 / **9.2** / 7.9 | 6.9 / **5.7** / 3.5 | 21.2 / **13.7** / 14.0 | 7.3 / **5.1** / 1.76 (slice) | 26.4 / **4.7** / 1.78 | 38.3 / 75.6 | 1.97 |
| m=16 | 13.9 / **11.9** / 7.9 | 7.0 / **6.2** / 3.6 | 21.1 / **16.0** / 14.1 | 18.5 / **15.2** / 14.1 | 14.9 / **5.1** / 1.8 | 54.5 / 75.5 | 1.39 |
| m=32 | 14.0 / **12.1** / 8.0 | 7.0 / **6.1** / 3.6 | 21.7 / **18.9** / 14.1 | 18.4 / **17.5** / 14.1 | 15.2 / **4.8** / 1.9 | 59.5 / 76.3 | 1.28 |
| m=128 | 17.2 / **14.6** / 8.4 | 7.6 / **7.5** / 4.0 | 25.3 / **21.1** / 14.6 | 20.1 / **20.1** / 14.6 | 16.4 / **7.0** / 2.2 | 70.4 / 86.6 | 1.23 |
| m=256 | 19.7 / **18.4** / 9.8 | 9.3 / **9.3** / 4.5 | 28.6 / **27.0** / 17.4 | 24.3 / **24.2** / 17.4 | 17.5 / **9.1** / 2.7 | 87.9 / 99.4 | 1.13 |

Weakest loads: m=256 (1.13), m=128 (1.23), m=32 (1.28), m=16 (1.39). At m=128/256 we tie SGLang (cuBLAS) on o and unpooler, and nearly on qkv/pooler.

## 4. What the current code does (cp13, single file ~2600 lines: Triton + Gluon + load_inline CUDA)
- **12-bit lossless "zip" format** (1.5 B/weight, bit-exact): sign+mantissa byte plane + 3/4-bit exponent code over a per-layer window; out-of-window "escapes" (~1.4e-4 of weights) stored exactly in an unbounded per-row list and fixed by a 9th warp. Planes pre-permuted to mma.sync m16n8k16 A-fragment order; decode = lop3/prmt, ~7 ops per 8 weights.
- **z9 kernel** (zip): per-warp private 1D cp.async.bulk rings (depth 1–3), x via swizzled 2D TMA on the same mbarrier, 2 CTAs/SM, cluster DSMEM reduce where split. Used for qkv m1/m8, pooler m1/m8/m16, unpooler m16.
- bf16 paths: Triton/Gluon GEMV and TMA+wgmma kernels for m≤32; qkv m128 custom cluster split-K (3-CTA, DSMEM reduce, 108 CTAs); cuBLASLt heuristic-selected algos for qkv m256 / pooler m128; cuBLAS (torch) for o/unpooler m128/256; Triton _big_gemm for pooler m256.
- Router: hi|lo bf16 split; m=1 fp32 FMA row-GEMV; m≥8 TMA+wgmma with 8-CTA cluster reduce.
- prepare() (~120–140 s): builds extensions, packs zip, self-tests every config (fp64 smoke, one-hot bit-exact incl. escaped column, 10–20× determinism), times candidates vs fallback with cold-L2 graph replays and picks per shape.

## 5. Measured hardware facts
- Cold-L2 pure read floors (graph replay minus empty graph; bench/bw_sweep.py): 8.4 MB 2.4 µs, 16.8 MB 4.24, 28.3 MB 6.8, 37.7 MB 9.0, 50.3 MB 11.4, 67.1 MB 15.1 → ~0.65 µs fixed + ~4.5 TB/s asymptote. Best pattern: LDG.128 grid-stride, 132×4–8 CTAs×256 threads, interleaved chunk assignment. cp.async.bulk 32 KB depth-2 is +0.3–0.5 µs; small chunks with deep rings much worse.
- Same-run comparison: z9 (pooler m1, 50 MB zip) is only ~0.8–1.0 µs above a pure LDG stream of the same bytes; decode+MMA+escape-fix+epilogue together ≈ 0.6 µs; interleaved addressing would recover ~0.3–0.5 µs.
- Judge µs for small layers ≈ floor + ~1–1.5 µs (launch/ramp/tail, and the judge subtracts only ~4 µs vs a ~5.25 µs empty graph).
- Thread-block cluster co-residency on the judge H200: S=2 → 66 clusters, S=3 → 39, S=4 → 30, S=8 → 15.
- At m=256 these GEMMs are near compute-bound (pooler 17.2 GFLOP); our hand wgmma kernels reach a lower MMA rate than cuBLAS nvjet.
- Judge image has CUTLASS 4.8 headers (/opt/cutlass/include), cuBLASLt 13.1, nvcc 13.0, deep_gemm, flashinfer, tilelang.

## 6. Tried and lost (do not repeat without a new reason)
- Zip at m=32 (all designs), zip qkv m16, zip for o and unpooler slice (fixed cost dominates 8–17 MB), zip at m=128 (register-A wgmma; transform-warpgroup smem decode → compute/pipeline-bound, x ingress dominates).
- x-resident split-K / bigger row blocks for zip m16/32 (x traffic ≤1 µs, not the bottleneck).
- Bigger bulk copies (6–24 KB units), 16 warps, 4 warps, deeper rings, 1 CTA/SM, stream-K balanced zip.
- One-shot whole-slice loads for router/o/slice — slower.
- Router m8–32 alternatives (cluster 2/4/16, BK64, fp32 FMA rows, K-split clusters); router m128/256 split-K kernel.
- m128/256: hand cluster split-K (only qkv m128 won), multicast-x hand kernels, CUTLASS 4.8 stream-K / pingpong / cooperative (1–8 µs slower), enumerated cuBLASLt algos, more SMs for pooler/unpooler m32 (split-2 +1.8 µs).
- Triton-only compressed paths; eviction / L2 hints.

## 7. Tried and won (history 0.907 → 1.415)
Router hi|lo split-K (+21%), Triton/Gluon GEMVs, CUDA wgmma kernels, 12-bit zip + z9 (+5%), robust escapes, qkv m128 cluster split-K (−2.4 µs), cuBLASLt heuristic picks (−1.4/−0.7 µs), z9 config tuning (qkv m1/m8 −0.3/−1.3, pooler m16 −0.9).

## 8. Open questions for a fresh expert
1. Leaders are ~13% faster per load on average. Where could that come from, given small-m kernels are ~1 µs from a cold-read floor and m128/256 tie cuBLAS? Which loads do they likely win?
2. Better lossless format? Gaussian bf16 entropy ≈ 10.6 bits (sign 1 + mantissa 7 + exponent ~2.6). 11-bit codes → ~1% escapes; entropy-coded exponents ~10.7 bits (−11% bytes vs 12). Can decode keep up at 4.5 TB/s with in-band outliers?
3. m=128/256: any known way to beat cuBLAS by >10% on these skinny bf16 GEMMs on H200 (compressed W with tensor-core-friendly decode, 2-SM cooperative tiles, other split strategies)?
4. m=16/32: zip loses as NT grows (decode in the MMA path). Is there a decode-to-smem + wgmma design that wins at m=16/32, where W streaming still dominates?
5. Is the ~1–1.5 µs per-shape fixed cost reducible (launch latency, ramp, tail, prologue)? 30 shapes × 1 µs ≈ 2–3% of score.
6. Any measurement trap we may be falling into?

## 9. Key files
- Best code: checkpoints/cp13_B_check1.415.py. History: checkpoints/README.md.
- Judge profiles: bench/profiles/*.txt (per-shape µs) and *.log (prepare prints).
- Team log: bench/FINDINGS.md. Bandwidth sweep: bench/bw_sweep.py / .out. Ablations: bench/exp_B_abl.py, exp_B_il.py.
- GPU runner with 30-min budget: bench/gpu_q.py (1181 s used).

## 10. Fresh expert review (2026-10-27) — summary
- Diagnosis: at m≥16 the binding limit is L2→SM ingress F ≈ 6.6–8.5 TB/s on (W + re-read x + split-K partials), not HBM/decode/compute. Model: t ≈ max(HBM/4.5 TB/s, ingress/F). Judge floors = bw_sweep + ~1.25 µs → m=1 already at floor (≤0.7 µs left). Log-gap by load: m128 0.24, m32 0.23, m16 0.15, m256 0.15, m8 0.05, m1 0.02. Ingress-aware floors give geomean ≈1.62 ≈ leader → leader is simply near these floors everywhere.
- E1 first (≤3 min): measure F directly (stream 50 MB from HBM while re-reading a 2 MB L2-resident buffer R=0..64 times); verify model predicts m16–m128 judge µs within ~1 µs.
- Ranked ideas: (1) TMA multicast of x across a 2-CTA cluster in z9 (m16/m32), then retry zip m32 / qkv m16 — kill test: z9 no-x variant at m32 (expect 28.5→21–22 raw) — +3–5%. (2) m128 ingress-aware kernel: swap-AB wgmma, 128–256 W rows/CTA, all 128 tokens as N, x multicast cluster-2, split-K 4–8, last-arriver reduce; avoid cluster-4 (only 30 co-resident) — +2–4%. (3) m256 tile shaping (pooler 128×256 split-K4 + pairwise DSMEM; qkv 18×S7; o 64-row tiles) — +1.5–2%; quick test cuBLASLt explicit SPLITK 2/3/4 and transposed problem. (4) Router m128/256: hi|lo interleaved along K, one wgmma stream, split-K 32–64, last-arriver — +1%. (5) slice m8 (5.06 vs 3.9 floor — kernel-choice bug) and o m8–32 via m1 LDG GEMV with 8 tokens/pass — +0.5–0.8%. (6) router lo compression; (7) denser zip (base-9 exponents, ~11.2 b/w) — last.
- Stop: z9 config/ring/warp sweeps, m=1 work, CUTLASS/cuBLASLt enumeration at m128, router m8–32 micro-variants, cross-harness raw comparisons (build one judge-replica harness: cold flush, graph, paired median, −4.0 µs), "x < W" no-go criterion.
- Risk flagged: exact-m dispatch (e.g. cuBLASLt at exactly m=128/256, qkv m128 cluster kernel) — final review reruns nearby shapes; dispatch by m-range so m=127/129/255/33 also hit fast paths.
- Realistic target: 1.48–1.52 (ideas 1+5+4 ≈ 1.48–1.50; + working idea 2 ≈ 1.52–1.55).
- cp14 = current best code base (2026-10-27).
