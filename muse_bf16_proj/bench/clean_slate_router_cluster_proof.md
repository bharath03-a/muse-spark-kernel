# Router cluster ownership proof and prototype handoff

CPU/source work only, 2026-10-28. No GPU/Modal/OKC jobs launched by this agent. No canonical/checkpoint edits.

## Executed proof

`/usr/bin/python3 bench/router_cluster_mapping_test.py` passes C2/C4/C8 plus the complete M128/C8 and M256/C4 router grid/address checks. It imports only standard-library code.

Logical partials are P[source,row,token] with shape[C,32,64]. Source CGA bases [[1,0,0],[2,0,0],[4,0,0]] as applicable give source CTA s exactly plane s. Destination/shared CGA bases move these bits onto the token tile axis. Each destination d owns the contiguous interval `[d*(64/C),(d+1)*(64/C))`, not tokens modulo C.

For every source s, row r and token t:

- owner d = t//(64/C)
- local token u = t%(64/C)
- local mailbox byte offset =4*((s*32+r)*(64/C)+u)

The mapping is bijective onto every destination's offsets0..8191 in4B steps. Source→destination traffic is8192/C bytes; each destination receives C equal contributions, including self, totaling8192 bytes. Its local mailbox has shape[C,32,64/C], exactly2048 FP32 elements. Thus `mbarrier.expect(bar,smem.nbytes_per_cta)` must resolve8192 bytes on every CTA for C2/4/8; it must not use total-cluster bytes.

Source BlockedLayout([1,1,2],[1,8,4],[1,4,1],[2,1,0]) enumerates row=warp*8+lane//4 and token=(lane%4)*2+(reg%2)+(reg//2)*8. Destination retains the same row mapping, repeats registers first over its64/C tokens and then over all C source planes. Every output[row,token] sees exactly sources0..C-1, without duplicate/omitted values. The test exhaustively enumerates source ownership, byte destinations, destination register reads and integer sums.

Primary layout contract: [pinned3.8 layouts and shape-per-CTA calculation](https://github.com/triton-lang/triton/blob/v3.8.0/python/triton/experimental/gluon/language/_layouts.py). Host CGA bases describe CTA tiles; scaling by the local token extent yields contiguous token intervals.

## Full source prototype

- `router_cluster_bt64.py`, SHA256 `82882366c662a229fdbf9fab158aac26a32b95d180123b9252d70a491febb0d0`.
- `router_cluster_bt64_compile_only.py`, offline SM90 driver, no tensor allocations or launches.
- The initial offline compile reported a Python-bool async-copy mask in the statically unrolled prologue. Fixed only that issue with a full COPY-layout predicate `(step<KI)&(offsets>=0)`. Revised source syntax passes; subsequent compile result belongs to the parent.

Mainloop: rank3 MMA2 `[C,32,64]`, warps[1,2,2], BF16 hi/lo weights and BF16 x, separate FP32 accumulators added only in FP32. K split4/8, BK128, depth3 cp.async stage buffers total98304 bytes/CTA. M128/C8 has16 clusters×8=128 CTAs, KI8. M256/C4 has32 clusters×4=128 CTAs, KI16. Output tiles and K shards cover the full operation exactly once. No global partial workspace or separate reduction launch.

Host-created constexpr layouts avoid the pinned nested-CGA equality bug. The leading stage allocation/index pattern follows existing shared descriptors; indexed descriptors carry rank3 batch-axis0 CGA. A rank4 CGA rewrite should not assume `.index()` shifts layout bits: pinned `_semantic.memdesc_index` preserves the layout object while dropping the first shape dimension. Actual compiler validation, rather than guessing, decides whether this allocation pattern lowers here.

Two CTA barriers per iteration protect cross-lane shared operand loads and slot refill. Final wait0+CTA barrier drains the ring before shared-memory reuse. Both generic and async epilogues rendezvous across the cluster BEFORE any remote write, because a fast sender's mailbox may alias a slow peer's live ring. Both rendezvous after all mailbox loads before any CTA exits or invalidates its barrier. Async barrier initialization has release fencing and a cluster rendezvous; self-target st.async is legal, and every byte has the same destination CTA for its mailbox and completion barrier.

Independent compiler peer approved source protocol and precision, subject to actual compile/lowering. Numerical correctness and performance remain UNPROVEN. MMA2 mainloop performance cannot be inferred from the faster CUDA WGMMA baseline or from isolated epilogue compilation.

## Full-kernel offline compilation now passes

Parent compiled revised frozen source in pinned Triton3.8 / SM90. Read `router_cluster_bt64_compile_fixed.out`: all four kernels PASS, shared98304B, zero stack/spills,64 static MMA2 instructions,70 cp.async instructions, zero shuffles. Registers are147/148 for C4 STORE/ASYNC and149/148 for C8 STORE/ASYNC. Generic tail has6 static cluster-barrier instructions, async4; local barriers8 versus11. The leading-stage shared layout/index pattern is accepted by the actual compiler; do not replace it based only on rank speculation. CPU compilation still does not establish GPU correctness or performance.

## Self-contained gated candidate

`top50_candidate_router_cluster_bt64.py`, SHA256 `e98d58899ed2f04cf0dce36aff7fff610e3c9b9a301477f68e3fcbb933192494`, begins with exact cp18 bytes. Layout/copy/issue/kernel normalized ASTs equal the compiled standalone prototype. Only new router M128/C8 and M256/C4 are eligible, independently. Timed `_rcb_run` always consumes the current `data['x']`; no standalone `make` closure or prepared activation is retained.

Prepare splits the original FP32 router once into allowed BF16 hi/lo; rejects nonfinite values/overflow. Both STORE and ASYNC must pass compile/resources, `_r15_valid(...,tol=1e-4)` and a new m-parameterized captured-input check. The latter changes normal/outlier inputs before every replay, includes one-hot power-of-two probes, uses FP64 targets with conservative global5e-5 / row2.5e-4 / probe5e-5 bounds, and checks exact zero output twice. The faster valid control must beat the exact cp18 path in two31-repetition cold alternating-order rounds by more than max(0.5us,2.5%). Outer260s timer starts before cp18 prepare; no selections survive a deadline failure. Existing cp18 pooler/unpooler selections remain untouched.

`router_cluster_candidate_cpu_test.py` executed PASS: exact cp18 prefix, inlined kernel AST identity, fresh runtime x forwarding, STORE=False membership, independent per-M mode choices, and shape/config/dtype/stride/alignment/fallback dispatch checks. No GPU jobs or judge checks by this agent. Parent owns any decision to launch.

Final wrapper peer review: expert_compiler approved for one bounded parent-owned judge check, with no blocking issue found. Confirmed exact cp18 prefix, fresh-input dispatch, independent per-shape selection, strict correctness gates, two timing rounds, deadline handling and fallback. Candidate source/hash unchanged after this review. GPU correctness/performance remain unverified.
