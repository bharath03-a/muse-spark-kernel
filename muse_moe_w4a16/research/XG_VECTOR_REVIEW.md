# No-PDL gather review, 2026-10-28

Source base: `research_moe_xg2.py`, already passing check `061e24b8ef934b9d`.
The new isolated builder is `research/build_xg_vector.py`; it emits
`research_moe_xg_vector.py` and readable `research/kernel_sources/xg_vector_CUDA_X.cu`.
Scored `submission.py` is unchanged. No OKC job or Modal GPU job was launched.

## Concrete changes

1. Default `XGSEG=4`: divide each route's 8192-byte activation row into four
   2048-byte segments; distribute items across all 132 CTAs instead of only
   32 CTAs at M128 and 64 at M256. Same 1024-thread CTA and stable route ranks.
   `-DXGSEG=1` provides a control with all-SM launch but no segmentation.
2. Optional `-DXGVECTOR`: retain each lane's 16-byte source load, shuffle words
   inside octets, write one contiguous aligned 16-byte destination vector.
   The old path issues four separate 32-bit stores per lane.
   The vector path uses 16 warp shuffles and four word selections, so latency
   may regress: this is a mechanism hypothesis, not a performance claim.

The source uint4 lane q contains words numbered q*4+k. For destination lane l
and output word j, source lane is `(l & ~7) | ((l >> 2) & 1) | (j << 1)`, source
word is `l & 3`. This inverts the existing kperm exactly. SW128 row swizzle
changes only the aligned output-vector address, not word order inside it.
All lanes of a warp share the route and segment, so every shuffle uses the
full active mask. Segment boundaries are multiples of 32 vectors.

CPU sector enumeration for a warp's 512 output bytes, at row positions 0/1/7:
old four store instructions each request eight 32-byte sectors = 32 sector
requests; new vector store requests 16 sectors. This is **2x fewer sector
requests and 4x fewer store instructions**, not 4x less HBM traffic. Store
merging downstream may reduce the practical benefit.

## Proof performed

`/usr/bin/python3 research/test_xg_vector_mapping.py` passes:
96 full-row address/value comparisons over padded sizes8..512, row-boundary
positions, and XGSEG1/2/4/8; ten scored/nearby token shapes1..512; exactly-once
item coverage across132CTAs; unchanged downstream CUDA source (apart from
xgather launch); no griddepcontrol or programmatic-launch API.
Python syntax compilation passes for builder and emitted candidate.

Strict base audit: gate/up use strict_swiglu, which rounds each BF16 before
FP32 SiLU/product and then rounds the intermediate. W2 multiplies its global
scale and rounds q0..q3 to BF16 before the norm. The new gather changes no
arithmetic or weight storage. This is independent of the noncompliant old
whole-decode v35 embedded kernels. Existing `build_xg3.py` contains PDL and
must not be used under this task's no-PDL constraint.

No CUDA compiler, GPU numerical validation, or timing proof has been run for
this candidate. Do not submit/promote it on the CPU proof alone. Future
paired comparisons should use xg2, all-SM XGSEG1, all-SM XGSEG4, and vector
XGSEG4, on identical weights/routes, cold captured graphs and fresh inputs.

## Ceiling and next mechanism

Current xg2 gather is6.6us(M128),7.5us(M256), total shape times214.5/252.1us.
Even eliminating those gathers completely gives only
`((214.5/(214.5-6.6))*(252.1/(252.1-7.5)))**(1/6) = 1.0103x` overall if the
other four loads are unchanged. Thus this work alone cannot close the
6822->5759us top-50 gap. A plausible2us/shape gather saving is only0.29% overall.
The all-SM path may lose because every CTA redundantly ranks the full routes:
132/32=4.125x aggregate rank work atM128 and132/64=2.0625x atM256. This should
be evaluated before adding any more gather complexity.

The larger independent opportunity is the measured10–14us last-consumer tail.
Current scheduler already dynamically claims units and orders experts by
decreasing token count; generic dynamic scheduling is therefore not new.
K-splitting only last-wave units requires separate FP32 partial sums and a
completion/reduction protocol **before** strict gate/up or W2 BF16 rounding.
Existing epilogues immediately publish BF16 gate/up, inter, and W2/norm sums;
adding atomic partials there would change rounding and race downstream work.
Do not implement a superficial split-K tweak. A safe dedicated last-wave
partial buffer/reduction needs source review and GPU budget and could itself
cost the1–3us reduction tail observed elsewhere. This gather candidate is the
bounded low-risk source work; no claim of top50 or1.45x guarantee is justified.

Emitted candidate SHA256:
`c98d6791625b6a3587e169e6a2fc33090616fa2db06b665d0952aff974546eaf`.
Ablation suffixes for the existing `_extX(flags)` interface:

- xg2 original source: existing WX flags, original bounded grid.
- new source `-DXGSEG=1`: all-SM grid only.
- new source `-DXGSEG=4`: all-SM grid plus quarter-row work items (default).
- new source `-DXGSEG=4 -DXGVECTOR`: add register transpose/vector stores.

Preserve `-DNCW=16 -DWXG2 -DUDESC`, plus `-DN8STEP` for the current M256
variant. The emitted candidate freezes the prior dispatch and does not
self-enable prepare-time tuning; an authorized coordinator must explicitly
wire those ablations into its existing paired harness before a judge check.

Independent peer review (`attention_pipeline_review`) also ran the CPU test
and approved shuffle participation, segment/octect boundaries, stable ranks,
and16B alignment. Scope caveat: SEG4 has only `min(132, M)` active-copy CTAs,
although all132rank. It reaches128copyCTAs atM128 and132 atM256; it is a poor
blanket choice for smallM. Frozen standalone dispatch uses wx only aboveM64,
so scored smallM paths are unchanged. Do not port this launch indiscriminately
to whole-decode's wx M32 pin. Inspect SASS before treating source-level store
counts as measured hardware transactions: compiler transformations can alter
the practical difference.
