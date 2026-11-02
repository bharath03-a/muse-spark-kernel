# RESEARCH_V2 (scout, 2026-11-01)

## 0. Headline
The shared recipe is public: **github.com/vjeux/okc-2026** (Apache-2.0, 5 single-file native-CUDA submissions, commit f35a41e, 2026-10-27).
Clone: `/tmp/scout/vjeux_okc-2026/`. Leaderboard: vjeux attention 2.61 (#211), moe 2.327 (#322), decode 2.102 (#356), bf16 1.501 (#408).
Fork-with-tweaks by alygg77 (`/tmp/scout/alygg77_okc-muse-spark-kernels/`, 2026-10-29): attention 2.653 (#120, vjeux file UNCHANGED),
moe 2.4015 (#115), decode 2.2058 (#236), bf16 1.5502 (#233). Plus rohanarun (#109 attn 2.661, #101 moe 2.405, #101 decode 2.255, #30 bf16 1.595).
Leaderboard plateaus (rank 100-300): attn 2.57-2.67 (modes 2.58-2.65, dozens each), moe 2.34-2.41, decode 2.15-2.26, bf16 1.53-1.57 = vjeux-derived files.
`/tmp/up` (lee101/openkernels, HEAD c565e37) is our own lineage: its attention submission.py differs from ours only in DFL/OVR tuning tables;
no hidden recipe there. Its docs (attention_gap_analysis.md, moe_gap_analysis.md, upstream_ideas.md) are good but already known.
Only vjeux/alygg public files reach the plateau; ours (3-launch attention, 1.98) never adopted them (grep of our tree for vjeux/okc-2026: no hits).

Policy note: copying public Apache-2.0 code is legal under challenge terms (alygg77 did it, with NOTICE). Lead decides; keep attribution header.
Rules check still applies per file: alygg/vjeux files claim OKC-rule compliance, but we have not run okc check on them.

## 1. muse_attention_decode (ours 1.983, #300 = 2.571)  -- biggest, most certain gain
Per-shape judge us (alygg77 RESULTS.md, vjeux file): d1s 5.19, d1f 10.95, d16s 7.91, d16f 83.65, d32s 9.70, d32f 159.93,
v1s 6.77, v1f 14.25, v16s 11.77, v16f 99.98, v32s 14.82, v32f 182.13. Ours (v56-era, older doc, may be ~5% better now):
d1s 8.62, d1f 17.45, d16s 11.21, d16f 110.5, d32s 13.57, d32f 201, v1s 12.4, v1f 21.6, v16s 16.7, v16f 130.9, v32s 19.8, v32f 243.
Gap is largest on sliding + B=1 (3-6 us) and verify full (v32f -60 us), exactly where our gap doc says we are latency bound.

Ranked mechanisms (all in `/tmp/scout/vjeux_okc-2026/muse_attention_decode.py`, CUDA_SRC, plan in `_plan()` line ~3012):
1. **Single fused launch** (prep + new-K/V build/FP8 cache write + split-KV attention + merge + rmsnorm*sigmoid(gate) epilogue). We run 3 launches
   (prep -> split-KV -> combine). Each extra graph node costs ~1.5-3 us; accounts for most of the 3.4 us on d1s and 5.6 us on v1s. Expected +15-25% on sliding loads.
   Apply: port as whole file (candidate file + okc check); do not graft piecemeal.
2. **FP32-atomic merge at a fixed reference max** (QK RMSNorm bounds logits, so no running-max rescale across splits): every CTA atomically adds
   rescaled (o, l) partials; the last-arriving CTA (counter) normalizes, applies epilogue, writes output and **zeroes the workspace for the next call**
   (no memset node). Replaces our partial store + separate combine kernel (our 2.5-5 us merge). Check precision margin vs checks.py before trusting.
3. **Launch plan per shape** (`_plan`): full+B>1 -> "flat" kernel (equal key shares of all requests, one pass per SM pair, 2 CTAs/SM, target=2*SMs);
   B=1 full -> plain split-KV, 2 CTAs/SM (clusters slower); sliding few requests -> 1 tile/CTA, nc=1 workspace copy; sliding L=8 -> cooperative finalize
   (every CTA normalizes a slice) ; sliding mid -> DSMEM clusters of 4-16 (cs halves until reqs*cs <= limit; residency limit 2*SMs for L=1, SMs for L=8,
   15/16 SMs when 1 CTA/SM). Compare `OVR/DFL` in our submission.py:6276.
4. **PV on FP8 wgmma for q_len 8 on sm_90** (P in E4M3 as FA3-FP8 does); QK on FP8 MMA with Q rounded to E4M3. Our candidate_v81_fp8pv explored this;
   vjeux ships it. Rules: SGLang's FA3 FP8 path rounds P to e4m3, so claimed in-spec; our precision sentinel in AGENTS.md decides.
5. PDL bit (bit 16 of mode word) + cp.async multi-stage 64-key tiles (no TMA gather needed; 128 B rows).
Not worth it (alygg measured, `docs/NEGATIVE_RESULTS.md`): merges of 128-318 splits, L2 prefetch hints, last-arriver merge via global + threadfence (2-4 us slower than combine kernel for prefill).
Facts: same-address red.v4.f32 serializes ~17 ns/op in L2; per-SM L2 write ~40-50 GB/s; DSMEM read ~30 GB/s/SM; 2 CTAs/SM vs 1 only 16-18% on v16f/v32f.

Remaining headroom after vjeux (leader 3.34 at 17 attempts; top-10 ~2.9-3.3): d1s/v1s/v16s still 5-12 us vs floors <0.2 us; byte floors
at 3.87 TB/s gather ceiling (our gap doc). Novel: one-key-block-per-CTA B=1 sliding with one-round-trip merge; FA3-style QK(i+1) during softmax(i)
for verify full (needs >112 regs, so split across 2 warpgroups).

## 2. muse_moe_w4a16 (ours 2.032, #300 = 2.337)
vjeux file `/tmp/scout/vjeux_okc-2026/muse_moe_w4a16.py` (3497 lines; ours 1143). alygg per-shape us (with KPERM3):
d1 24.6 (3.16x), d16 92.3, d32 124.0, v1 63.6, v16 175.2, v32 207.8. Ours (old doc v34-era): d1 50.4, d16 126, d32 175, v1 101, v16 259, v32 340 -- recheck with current profile,
but d1 and v16/v32 are where a 2x gap (not 15%) shows.
1. **One persistent grid, routing done redundantly inside every CTA** (counts, offsets, sorted entry list, job list from topk_ids; no route launch), phase A
   (gate+up+SwiGLU) and phase B (down) in the same kernel, grid barrier, then final norm + weighted combine pass. Saves 2+ launches.
2. **Job classes**: experts > SKINNY_MAX_TOKENS become fat jobs (256-row units, RS-wgmma "WGS" build `OKC_WGS` for the large-batch band, K-stage smem ring);
   rest skinny (16 rows x full K, warps split K, mma.sync). Fat first, largest first; skinny split by cost so every CTA has equal cost; small atomic
   stream-K tail pool. Cross-phase prefill: last ring refills of phase A prefetch this CTA's first phase-B units.
3. **Four compile-time variants by M** (MAIN/TINY/TINY2/MID); d1 uses TINY (3 distinct experts, per-CTA streaming). This is the d1 2x.
4. **KPERM3** (alygg77, +1.25% board): permute K inside each wgmma stage so one 8-byte scale load serves 4 k16 steps -> decode loop 20.1 -> 18.4 instr/word,
   no spill. Lossless repack in prepare(); gain only on the WGS path (d32, v16, v32: +2.6/+3.9/+3.1%). Needs prepare() < 300 s (they measured 208-311 s; keep build lean).
   Source: `/tmp/scout/alygg77_okc-muse-spark-kernels/kernels/muse_moe_w4a16.py`.
5. alygg measured: WGS stream is unit-boundary-bound not rate-bound (1-7 us stall per 16-stage unit end, ends synchronized across 40-90 CTAs); 3 weight slots saturate;
   a GPU-scope fence in the producer costs 0.13 us each; unit release 0.9 us. Next-level idea (unbuilt): desynchronize unit ends / finer phase-B tail (end spread 3-8 us).
6. Ours-known (docs/upstream_ideas.md): Humming-style RS-wgmma W4A16 and m8 mma.sync stream-K tiles for bs<=32 (Humming drops TMA+producer warp on short expert grids).
   vjeux's WGS is already the RS-wgmma variant.
Floors: weight bytes at 3.9 TB/s -> 2.74x geomean; 2.40 = ~85% of that, so only ~10% more is physically left to #1 (2.76).

## 3. muse_decode_layer (ours 1.861, #300 = 2.147)
vjeux file (11177 lines) = composition; alygg 2.2058 per-shape us: d1s 69.3, d1f 74.4, d16s 129, d16f 204, d32s 149, d32f 304,
v1s 110, v1f 117, v16s 244, v16f 328, v32s 304, v32f 464.
1. Compose from the component boards' kernels (copied verbatim): fused attention (kernel above), MoE (3 builds by T), bf16_proj routes. Wins automatically as 1-2 land.
2. **prepare() autotunes on the judge GPU**: times each backend (cuBLAS / bf16_proj / Triton split-K) per (T, matrix) and the expert kernels per T with CUDA-graph
   replays after L2 flush, pins the winner; untuned T uses the next bucket. Sentinel words after scratch buffers checked after the tuner.
3. **FP32 router at T<=8** (found by "Matsvei"/kernarc matvey): reference uses FP32 normalized activation x FP32 [256,8192] weight at T<=8, BF16 hi+lo above 8.
   AUDIT OURS: a BF16-activation router at T<=8 deviates from the reference (act differs ~2e-3 rel). alygg fused it into the pooler stream (v1 cost ~1.5-2.5 us, d1 gain 1 us).
4. alygg facts: at T<=8 the layer is one serial PDL path with no gaps; skipping all 3 glue launches saves only 4-4.8 us (d1s/v1s); 2 pre-wait weight stages optimal;
   side streams lose (graph fork 21-33 us late; verify attention slows 50-80% beside any GEMM); cuBLASLt algo sweep headroom ~2 us/layer (m256); split-K never critical.
5. KPERM3 for the expert stream at T>=17: d32s +3.1%, v16s +5.2% on the judge.
Do not repeat (negative): L2 prefetch of next-kernel weights, starting pooler stream before attention, more pre-wait stages, stream-K qkv (96 CTAs saturate).

## 4. muse_bf16_proj (ours 1.481, #300 = 1.534)
We already have c12 (12-bit lossless + escapes), same as vjeux. Remaining recipe differences:
1. **Nathen Fernandes small-m routes** (MIT, github.com/NathenFernandes/muse-bf16-proj, clone `/tmp/scout/n/`, v18 = 1.5186, was #1 on Oct 24): wgmma/TMA/cluster
   shape-specific kernels, pinned cuBLASLt algos for larger shapes, prepare 54 s. Helpful routes (8/8 local runs): o m8 (-0.38 us), pooler m8 (-0.60), pooler m16 (-0.47);
   lost: unpooler m8, router m1, o m16. Source is zlib/base64-compressed (1 MB limit): `python3 tools/expand_source.py submission.py out.py`.
2. alygg 1.5502 stack = vjeux base + custom WGMMA O-projection (FP32 acc) + pooler tuning + Nathen routes + __deepcopy__ fix (state owns weights) + cuBLASLt plan pins
   + per-thread CUDA context bind (fresh judge thread has no context; raw cuBLASLt handles failed without it).
3. Weak loads everywhere: verify16/32 (m128/256) at 1.24-1.28x vs decode1/verify1 1.7-2.0x. Facts: qkv m256 transfer-bound (66 B/clk/SM L2->SM), o m256 tensor-bound 87%,
   per-SM L2 write ~40-50 GB/s kills split-K at m256; 12-bit decode at m>=128 LOSES (int ALU does not overlap HGMMA: 286+256 -> 489 clk);
   cuBLASLt route search at m256 found nothing. So keep cuBLAS/pinned algo for m>=128 and spend effort on m<=32 only.
4. Judge noise on one unchanged bf16 file: 1.4866-1.5201 over hours, cluster offset (sfo3 vs sfo3c) 0.4-1%: need paired A/B with a control in the same check; only trust >2%.
5. Router m1 already 2.5 us; unpooler m1 3.6 us; qkv m1 8.8 us, pooler m1 13.4 us are the weights-bound floors (bytes). Top-3 bf16 are 1.65 (arashattar, naklecha, olety) - unknown recipe.

## 5. Leaderboard plateaus (lb_*.json, speedup by rank)
- attention: r10 2.93, r50 2.725, r100 2.668, r150 2.633, r200 2.614, r300 2.571, r400 2.261, r500 2.01. Mode cluster 2.58-2.65 (~35 rows each) = vjeux file/forks. 25 rows at 1.13 (SGLang-baseline-ish).
  Big cliff 2.57 -> 2.26 between r300 and r400: everyone who adopted vjeux sits above it. #1 pjmullin3 3.34 (17 attempts) - no public source found.
- moe: r10 2.62, r50 2.459, r100 2.406, r150 2.389, r200 2.379, r300 2.337, r400 2.216, r500 2.05. Mode 2.37-2.40 (vjeux + KPERM3). 31 rows at 1.27 (Marlin tweaks, e.g. ScrashOff).
- decode_layer: r10 2.41, r50 2.293, r100 2.255, r200 2.218, r300 2.147, r400 2.047, r500 1.83. Mode 2.20-2.26.
- bf16: r10 1.617, r30 1.595 (rohanarun), r50 1.587, r100 1.573, r200 1.555, r300 1.534, r400 1.506. Mode 1.55-1.57; 1.16-1.22 cluster = baseline-tweak teams.
Plateau = a public file with small per-team tweaks, confirming a vjeux-derived common base. Paths to top-50 need ~+5% beyond it (attn +6%, moe +2.5%, decode +4%, bf16 +2.5% over alygg).

## 6. Other repos (low value)
- lilaclilac09/open-kernel-challenge (bf16 1.07), ozgureyilmaz (1.17), ScrashOff (Triton, 1.2-1.4), msaroufim/openkernelchallenge-audit (judge audit: binary/evaluator/probes - read for timing semantics, not
  a source of speed; do not mine for judge gaming, forbidden by AGENTS.md), rohanarun/Auto-research-agent (LLM-generated starters/candidates, no strong results found).
- Generic techniques web-search was not needed for: FlashDecoding split-KV, lean attention, Marlin/Machete, DeepGEMM, PDL are all already in vjeux/alygg or ours; the binding constraints
  are launch count, merge latency and 128 B random-gather ceiling (3.87 TB/s), not algorithm class.

## 7. Recommended order
1. Attention: drop in vjeux file as candidate (probe file with ours vs vjeux, one check), submit if correct and >=1% better. Expect ~2.6 (+30%): crosses #300 alone.
2. MoE: alygg `kernels/muse_moe_w4a16.py` (vjeux + KPERM3) as candidate; check prepare() time (<300 s limit). Expect ~2.38-2.40.
3. Decode_layer: after 1-2, alygg's decode file (2.206); audit our T<=8 router precision regardless.
4. bf16: alygg file / Nathen routes into ours (m<=32 routes only); expect 1.53-1.55.
5. Then innovate on top: see per-board "remaining headroom" lines above.
Caveat: all per-shape numbers for vjeux/alygg come from their judge profiles in README/RESULTS; our per-shape numbers are from older docs and unverified against current v-latest.
