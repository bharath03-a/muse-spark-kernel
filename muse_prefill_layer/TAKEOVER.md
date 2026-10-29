# Prefill takeover, October 26, 2026

## Verified state

Live board query: team `bharath03-a`, best job `4fdebc07c2984fd7`, speedup 1.0012873x, score 386,905.84 us. Rank was 483, then 484 as another team joined. Top 50 at query time: 1.6838x / 230,076.16 us. The supplied 297,235.47 us / 1.303x narrative is not this account’s verified result.

`submission.py` implements full prefill with BF16 dense projections, fused QK-norm/RoPE/KV write, FA3 paged FP8 attention, norm/gate, two residual boundaries, BF16-hi+lo router, Marlin NVFP4 local experts, SwiGLU, expert normalization/weighted reduction, and unpooling. It preallocates intermediates and overlaps router/pooler streams. It remains close to baseline speed.

## Experiments already attempted

- `test_prefill_opt_v2.py`: 1.00032–1.00066x, no significant gain.
- `test_prefill_packgqa.py`: 0.99945x, no gain.
- `test_prefill_tile.py`: 0.98464x, regression.
- `test_tuned_prefill.py`: 0.99749x and 0.88641x, regression/variance.
- `test_prefill_fp32reduce.py`: 0.99222x, no gain; do not infer its behavior from its filename alone.

All of the above local profile summaries passed 5/5. These historical checks do not validate new edits to the same filenames.

## New controlled experiment

`test_prefill_m32.py` changes only the expert alignment M tile to 32 for token counts below 1024, retains the existing heuristic otherwise, and preserves FP32 Marlin reduction. Hypothesis: reduced padding under sparse local routes outweighs smaller tiles. Check job `5a552622d52f4677`. Original submission is preserved. Do not promote without actual correctness, timing and profile evidence.

## Bottleneck and priorities

Profiles attribute roughly 65–70% of sliding prefill time to two Marlin expert GEMMs; full-context attention contributes up to 30%. Fusing SwiGLU alone can save only its own roughly 1–1.5% plus intermediate memory traffic, so large rank jumps cannot be inferred from that change. The main target is expert GEMM throughput and padding, followed by first-turn dense GEMMs and full-context attention.

Hopper has no native FP4 tensor cores. NVIDIA documents Hopper NVFP4 execution as weight-only Marlin dequantization to BF16: https://docs.nvidia.com/nim/large-language-models/2.1.4/reference/advanced-configuration.html . The Marlin primary paper describes asynchronous weight loads, cache eviction policy and dequantization pipelining: https://spcl.inf.ethz.ch/Publications/.pdf/2025_marlin.pdf . Blackwell native NVFP4 compute is not an H200 optimization option.

## Compute workflow

Authenticated Modal CLI: `../muse_bf16_proj/.venv/bin/modal`. Existing `../muse_bf16_proj/modal_app.py` uses judge image `lmsysorg/sglang:dev-muse-spark-open`, H200 by default, shared build cache. It currently uploads only the BF16 challenge, so prefill needs a dedicated upload/runner before GPU benchmarking. CPU-only inspection works using `MUSE_GPU=''` with its `inspect` function. The image reports torch 2.14.1, Triton 3.8.0, CUDA 13.0, flashinfer-python 0.7.0.post1.

Use CUDA-graph replay with cold L2 and full correctness before leaderboard promotion. Avoid precision changes disallowed by TASK.md and output caching. Prefill has a separate board and does not count toward whole decode ranking; whole decode top 50 is the main priority.
