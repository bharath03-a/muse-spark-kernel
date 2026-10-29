# Immutable kernel-parameter experiment

The strict ring kernel receives `P p` by value and passes it by reference to
its epilogue and non-inlined finish routine. Its H200 SASS allocates exactly
136 bytes of stack, the size of P: seventeen prologue STL.64 instructions
copy offsets0x00 through0x80. All106 static LDL reload fields in this frame.
There are no later STL instructions. These are parameter copies, not evidence
that dequantized accumulator registers spilled under pressure.

NVIDIA documents that taking a kernel parameter's address normally creates
a per-thread local-memory copy. A `const __grid_constant__ P p` parameter
instead has one read-only object/address per kernel grid and avoids that
copy. P members are not mutated by these kernels; writes through its tensor
pointers remain valid. CUDA supports this annotation on the target H200.
Primary reference:
https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html#__grid_constant__

`research/build_grid_constant.py` changes exactly the v17 and ring kernel
signatures and assigns unique extension names. It verifies the inverse
signature transform reconstructs the original CUDA source. No arithmetic,
BF16 boundary, layout, counter, synchronization or dispatch change is made.
G1/GK source is identical to strict fullGK baseline.

CPU compilation and cuobjdump on sm90a confirm:

| Path | Original | Grid constant |
|---|---|---|
| ring stack bytes |136|0|
| ring LDL / STL |106 /17|0 /0|
| ring registers |not measured in this comparison|130|
| v17 LDL / STL |26 /6|26 /6|
| v17 stack bytes |not measured in this comparison|24|

The v17 result suggests most parameter copying was already optimized away
there. Larger ring shapes are the principal target. Eliminating local
instructions does not by itself prove a latency win: loads may move to a
different cache or register live ranges may change.

CPU log `/tmp/moe_grid_constant_compile_v2.log` (121.5 seconds total);
SASS artifacts `/cache/moe_queue_audit/grid_constant_{v17,ring}.sass`.
Coordinator authorized one bounded420-second H200 probe, launched with
shared configure_h200_cache and the validated paired research harness:
local session85350, log `/tmp/moe_grid_constant_gpu.log`. It covers17 fresh
normal/skewed/scored/nearby cases with explicit finiteness and FP32-reference
checks before AB/BA graph timing. No performance outcome recorded yet.


## Measured H200 result

The one bounded probe completed. All 17 cases (34 baseline/candidate finite
and FP32-reference checks) and replay-stability guards passed. Both modules
loaded from CPU-populated cache in 3.60 seconds; shared preparation took
5.25 seconds total.

| M / distribution / seed | Baseline us | Candidate us | Paired speedup | 95% interval |
|---|---:|---:|---:|---|
| 128 / normal / 7 | 268.13 | 259.17 | 1.0351x | 1.0196–1.0506 |
| 128 / normal / 19 | 238.13 | 230.59 | 1.0352x | 1.0197–1.0507 |
| 128 / skewed / 41 | 167.34 | 163.10 | 1.0246x | 1.0093–1.0400 |
| 256 / normal / 7 | 316.90 | 306.96 | 1.0325x | 1.0170–1.0480 |
| 256 / normal / 19 | 325.46 | 316.78 | 1.0274x | 1.0120–1.0429 |
| 256 / skewed / 41 | 295.76 | 291.22 | 1.0148x | 0.9996–1.0300 |

Median speedup across ring shapes M>64 was 1.0246x; unchanged small-v17
controls had median1.0023x. Consistent normal-route gains around3% with lower
intervals above1 support one official check, not an assumed leaderboard
improvement. No new OKC check/submission was made by this expert.

Raw reports (all AB/BA samples, source hashes, GPU/CUDA/Torch metadata) are
saved in profiles/research_grid_constant_gpu.json and .log. Shared persistent
report: /cache/moe_probe_reports/1793132755240884692.json. GPU allocation was
returned to the coordinator immediately after completion.

## Weighted extrapolation (not an official score)

Across three fresh routes per scored target shape, geometric mean speedups
are1.03163x (M128) and1.02489x (M256). Keeping the other four equally weighted
loads at1x gives component geometric speedup1.00933x, or roughly0.93% lower
score:7,760.36→7,688.6us. Normal-only route evidence predicts7,677.9us.
These extrapolations fall inside the judge's minimum1.5% uncertainty guard;
only an actual official result may determine promotion.

For whole decode, existing profile56c0b49b63f542d9 attributes MoE69.0%/52.3%
of verify16 sliding/full and69.7%/47.0% of verify32 sliding/full. Amdahl scaling
with the observed shape factors and actual46:16 layer weights predicts load
gains1.01991x/1.01528x; unchanged four loads yield whole gain1.00583x (~0.58%),
14,177.52→14,095.4us. This is an approximate inference: whole-layer routing,
clock behavior and cache pressure can differ from the isolated component.

Coordinator launched official checks041ed31d40cf4296 (standalone) and
a56d540dabca47ed (combined whole) after reviewing this evidence. Expert did
not duplicate these requests and has returned GPU allocation to other work.
