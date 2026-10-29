# Independent architecture audit, 2026-10-26

Scope: TASK.md, ANALYSIS.md, FINDINGS.md, cp14 inspected; primary references checked. No GPU measurements performed by this reviewer. All proposed gains below are hypotheses.

## What the measurements actually rule out

The fresh expert's ingress diagnosis in ANALYSIS section 10 did not survive the later [D] no-x experiment. Removing x loads changed pooler m32 from 27.8 to 28.7 us, and pooler m16 did not move. Thus multicast-x and the predicted 3–5% score gain are not a reasonable next investment. The hardware ingress microbenchmark measured a lower bound, not a binding 7–8 TB/s ceiling. Do not derive leader architecture from that ceiling.

Likewise, dense WGMMA BN32 being impossible does not mean dense BN32 MMA is impossible: mma.sync supports a smaller M tile than WGMMA. Compression failures tested decode overhead plus their scheduling, not this dense geometry in isolation.

## Ranked actionable tests

1. Dense no-split BN32 Gluon mma_v2, pooler m32 first, qkv m16/32 next. Use BN32, BT32 or16, BK128/256, four warps, two-to-four stages, no workspace. Pooler gets 128 independent CTAs instead of64; qkv72 instead of36 output tiles and avoids cluster split4 reduction. Do not expect more bandwidth merely from more CTAs: useful outcome requires saved reduction/underutilization to exceed shared-to-register and synchronous-MMA cost. A 1 us saving on each m16/m32 load gives score multiplier ((54.5/53.5)*(59.5/58.5))^(1/6)=1.0059, approximately +0.0084 from1.426. This experiment alone cannot reach top50.

2. Router large-m compare representation axes, only after consulting existing router agent. Existing CUDA stacks hi and lo along weight-row M and sums two accumulator halves. Existing _router_big already loads x once for two tl.dot operations. Therefore a 'single pass hi+lo' claim is not a novel experiment by itself. K-interleaving changes accumulator footprint and output CTA count; it is an architectural alternative only if those resources improve measured occupancy. Benchmark equal total flops and include reduction. Optimistically saving4us at m128 and6us at m256 gives multiplier ((70.4/66.4)*(87.9/81.9))^(1/6)=1.022, about +0.031 score. Real target must come from actual measurements.

3. Persistent cross-output-tile large GEMM overlap. Inspect against current scheduling before implementation; adjacent-token reuse already exists. Novel part must be retained asynchronous loads across epilogue/output boundaries or a proven smaller register footprint. Ordinary warp-specialization and CUTLASS persistence have already lost. Initial test o K1024 where startup/drain is a larger fraction; target >=0.5us net before broadening. No paper benchmark proves a gain on these shapes.

## References supporting mechanisms, not speed claims

- Triton3.8 mma_v2 export: https://github.com/triton-lang/triton/blob/v3.8.0/python/triton/experimental/gluon/language/nvidia/ampere/__init__.py
- Triton3.8 WGMMA layouts and register operands: https://github.com/triton-lang/triton/blob/v3.8.0/python/tutorials/gluon/05-wgmma.py
- Triton3.8 persistent Hopper pipeline examples: https://github.com/triton-lang/triton/blob/v3.8.0/python/tutorials/gluon/07-persistence.py
- NVIDIA PTX ISA, mma and wgmma instruction contracts: https://docs.nvidia.com/cuda/parallel-thread-execution/

These establish expressibility and instruction constraints only. They do not establish a H200 benchmark number for this challenge. Current local negative results outrank broad paper speedups.

## Correctness and nearby-shape audit

- cp14 line1087 dormant TMA_STORE path returns a fresh out allocation while descriptor points at prepare's dummy_out (line1289). All active BIG_TILES have TMA_STORE false, so latent, not current corruption. Fix by constructing output descriptor from the timed call's out before enabling this branch.
- CUDA_TILES router128/256 and r9/r15 exact-m choices are performance cliffs. Do not mechanically widen ranges: descriptor dimensions, padded tokens, divisibility, scratch sizes, and algorithm validity must first be checked. Test127/129/255/257 plus scored shapes.
- Current test passing says nothing about unexecuted TMA_STORE or alternate descriptors. Keep those branches out of canonical candidate until exercised.
- The scored-shape floor in TASK is for uncompressed weights. Existing lossless-compressed pooler can beat it. Summing those floors into a purported ultimate ceiling is invalid for compressed implementations.

## Score target discipline

Use live rank50 score, not 1.45 as a synonym for top50. At1.426, reaching1.50 requires4.93% less geometric-mean load time;1.55 requires8.00%. A 1us change restricted to one50us load changes score by only about0.34%. Router-only and dense-BN32 improvements may close part of the gap but do not establish top50. Prefer candidates with several independently measured wins, retain per-shape fallbacks, and use paired cold graph timing plus judge checks before making any score claim.
