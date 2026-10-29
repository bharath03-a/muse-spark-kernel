# MoE microarchitecture audit — October 26, 2026

Pinned source: lee101/openkernels d2057db2618075e960b7b8465180729bd87ccc38,
local `/tmp/muse-openkernels-research`. Upstream diary measurements below
are research evidence, not our authenticated leaderboard results.

## Evidence and decisions

- NVFP4 unpack already uses a carefully packed integer representation: four
  BF16x2 multiplies plus masks/shifts per eight values, paired scale bytes
  adjacent to MMA fragments. It does not perform generic nibble lookup.
  `research/prove_unpack.py` exhaustively verifies all 1,904 combinations of
  16 FP4 codes and 119 normal finite positive E4M3 scales. The exact product
  is FP4*scale/256, restored by state global scales multiplied by 256.
  Negative scales fold into code signs; zero scales fold codes to zero.
  Subnormal scales are rejected by upstream prepare, a documented limitation.
- Upstream v48 SASS diary: per 32 words,200 LOP3,114 IMAD.SHL,55 SHF,
  128 HMUL2,32 HMMA; total core 16.5 instructions/word. Scale PRMT/LEA,
  LDS, addressing and stage issue lift static total to ~22/word. A generic
  unpack rewrite is unlikely to beat this without changing representation.
- Scale-pair reuse is a concrete longer-term hypothesis: mixed-block A
  registers can halve scale-pair operations, but require a matching K
  permutation in packed weights AND gathered activations. Changing only
  weights silently changes dot products. Requires independent CPU mapping
  proof, compiled SASS verification and reference checks before GPU scoring.
- Upstream g1 diagnostics: full 2.63 TB/s; removing MMA 2.73; removing weight TMA
  loads 3.81; removing both 4.10. NOLOAD bypasses weight TMA, so this does not isolate activation-B dependency.
  It implicates weight streaming/shared-memory traffic or consumer scheduling
  more strongly than raw tensor-core throughput. g1
  keeps activation slices in 32 registers, reloaded once per expert from smem;
  gk stages all token rows in smem and adds LDS perfragment. Choose based on
  routed token count and register lifetime, not a blanket shared-memory rule.
- Existing g1 rings have 147 KB/SM inflight and only~2% TMA wait. More ring
  stages or an extra producer warp do not address that measured bottleneck.
  Historical producer/consumer redesign regressed 1.2–2.2x. Per-slot overhead
  reduction (PPS2) is already a v48candidate.
- Register pressure matters: older fs spilled 144/628 bytes at 128 register cap;
  g1 ~121 registers had no spills; gk 93–128 likewise no spills; wx N8STEP 96
  registers spilled 24 bytes. Source arrays alone cannot prove spilling:
  compare ptxas metadata and SASS LDS/STL for the exact compiled configuration.
- Whole-grid synchronization assumes resident persistent CTAs: gk launches
  one CTA/SM and uses launch_bounds(...,1); ring CPB>1 explicitly checks CUDA
  occupancy. Do not increase grid to multiplewaves while leaving spin
  barriers: queued CTAs could be required to release resident waiting CTAs.
  Monotone grid counters are fresh-work synchronization, not output caching.
- Dynamic routes create tail imbalance. Small-M contiguous assignments and
  gk splitK can reduce per expert imbalance, but W13→SwiGLU→W2 dependencies
  and extra partial-sum/barrier traffic can cost more. Upstream gk-M32 lost
  to the ring default despite better small-M behavior; retain shape dispatch.

## Precision variant and peer review

`candidate_openkernels_moe_bf16.py` is a NEW unmeasured candidate. It rounds
scaled gate/up accumulators to BF16 before SiLU, retaining FP32 SiLU/product
and existing BF16 product stores. All9 sites across 6 CUDA strings are covered:
CUDA 1, CUDA_L 2, CUDA_F 2, CUDA_G 1, CUDA_K 2, CUDA_X 1.
Prefill expert independently confirmed coverage and preservation of stores.

Default SGLang silu_and_mul loads BF16 gate/up, promotes to FP32, then rounds
the final product. BF16 rounding SiLU itself belongs to its optional
kRoundActivation=true variant, not the default used by baseline.py.
Primary source: https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/kernels/jit/csrc/elementwise/activation.cuh
(lines93–110). A pinned judge-image binding check remains useful before
claiming bit parity. This variant addresses only gate/up boundaries; expert
output/norm rounding in persistent source still deserves a separate audit.

## Preparation and outstanding runs

Original persistent prepare repacks 256 experts, then compiles many configurations
before its tuning 270 s guard. Thus that guard does not bound cold compilation.
Packed W1/W2 outputs alone are~7.25GB vs~0.906GB for 32 local experts. CPU
precompile exact source hashes and bound first-rollout configs; avoid cold
GPU compilation. Root owns current sequential CPU compile; do not duplicate.
The earlier independent Marlin CPU compile was never launched after the
previous usage interruption. Earlier GPU baseline comparison timed out at 300 s;
tile-v2 had include-path issues and produced no successful benchmark.

At this audit: original persistent MoE check e591e5372bbb4a1a and combined
whole decode fba499920caa4ef9 queued. No new GPU runs or checks launched by this expert
research pass. Original submission.py remains unchanged.

## Evidence-backed next proposal

1. Finish original persistent checks and root's CPU cache build.
2. CPU compile only matching default variants for BF16-boundary candidate;
   source hashes differ from original and cannot reuse its compiled modules.
3. If root authorizes, compare original and rounded variant on fresh normal
   and skewed routes plus graph replay; inspect phase waits and ptxas metadata.
4. Only after identifying a remaining consumer instruction bottleneck, prototype
   mixed-block scale-pair reuse with an exact CPU permutation proof. Do not
   spend scoring checks on generic unpack or deeper-ring changes first.
