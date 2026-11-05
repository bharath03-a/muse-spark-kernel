# Top-100 execution plan — November 2, 2026

## 1. Objective, evidence, and unresolved decisions

Target: reach top 100 before November 4, 11:59 pm Pacific (November 5, 12:59 am Phoenix), through a real repeatable improvement. Focus on the complete decoder; standalone kernel improvements matter only after integration improves the ranked workload.

Last live decoder reading in this chat: ours #141 at 2.2556x; #100 at 2.2680x. Required latency reduction at that snapshot is 1 - 2.2556/2.2680 = 0.5467%. This is below single-run judge uncertainty and will change as others submit. Aim experimentally for 1–1.5% repeatable full-decoder improvement, not a lucky leaderboard draw.

Ranking verification is Gate 0. The user and public keell0renz README identify decoder rank with overall rank, while local TASK.md says the overall board combines best runs. Query both current overall and decoder leaderboards and reconcile the current official definition before claiming an overall result. Until resolved, describe the 0.5467% figure as the decoder top-100 gap only.

Ownership is also unresolved for shipping: the user requires our own implementation and has not explicitly approved a partly derived final submission. Research, independent prototypes and comparison against the credited baseline can proceed. A candidate that retains credited code must be labeled partly derived and needs an explicit decision before it becomes the proposed final submission. Do not assume this planning request relaxes ownership. A complete independent decoder rewrite is a separate track with no demonstrated near-term score path.

No GPU launch or submission is part of writing this plan.

## 2. Strategy

Search for one small removable decoder boundary, using open source to identify an implementation mechanism. Do not replace the fast credited MoE with the much slower independent prototype. Do not start a persistent whole-layer rewrite.

Order:
1. Establish scoring, source identity, active runtime paths and baseline measurements.
2. Compare SGLang's attention-to-expert boundary against those paths.
3. Build one independently authored narrow boundary replacement if the comparison exposes recoverable time.
4. If no useful boundary exists, inspect MonoMoE's weight-major scheduling as a second, higher-risk mechanism.
5. Confirm the complete decoder, then decide promotion under the ownership constraint.

All percentages below are planning gates, not promised gains.

## 3. Phase A — establish an auditable baseline (45–90 minutes of local work)

### A1. Freeze inputs to the investigation

Create a research directory under `muse_decode_layer/research/top100/` without changing the live submission. Record:
- SHA-256 of current submission, relevant local dependency files, TASK.md and reference/check files.
- Current git commit and a list of pre-existing modifications; never reset or overwrite them.
- Credited upstream versions and license/NOTICE files.
- Existing official job IDs, prepare duration, source identity where available and all 12 shape timings.
- Current quotas, Modal active apps and reconciled remaining dollar allowance.

The older handoff budget is an estimate, not a reliable billing ledger. Inspect actual usage before more paid work. No background or detached jobs.

### A2. Resolve ranking and score arithmetic

Run bounded reads of `okc leaderboard --json`, `okc leaderboard muse_decode_layer --json`, and `okc problems --json`. Save raw responses, timestamps and current #100 rows. Inspect CLI help if the current command syntax differs.

For decoder load l:
- L_l = 46*t_sliding,l + 16*t_full,l.
- Candidate/baseline improvement factor G = exp(mean_l(log(L_baseline,l / L_candidate,l))) across six loads.
- Equivalent time reduction = 1 - 1/G.

Use paired-derived times from the harness, not independent noisy averages. Do not average twelve percentage changes or weight all twelve cases equally. Approximate new speedup S_old*G is for planning only; official scoring can differ with hardware and baseline timing.

Illustration: a 3% reduction in only one of six loads yields about 0.51% aggregate improvement; 3% in two loads yields about 1.02%. A faster microkernel can therefore be irrelevant if its affected load share is too small.

### A3. Map the actual execution path

The submission contains stacked prepare/kernel wrappers. Reading an early definition does not identify the active path. Resolve final bindings and shape dispatch for all twelve cases, then trace one representative case per branch.

Produce `runtime_map.md` with these columns:
shape family; final wrapper; CUDA/Triton entrypoint; producer; consumer; intermediate shape/dtype; launch dependency; scratch owner; existing fusion; source provenance.

Start at `_after_attention`, `_gle_after_attention`, `_pool_router_fp32`, `_gle_route`, the MoE call, and the final residual/norm boundary. These are inspection anchors, not assumed active paths. Confirm with the final module bindings and a GPU trace only if needed.

Exit: know which boundaries actually exist at T=1,8,16,32,128,256, and which have already been fused. Otherwise do not write a replacement.

## 4. Phase B — source comparison and feasibility (60–120 minutes, primarily CPU)

### B1. SGLang boundary source

Sources:
- https://github.com/sgl-project/sglang/pull/44668
- https://raw.githubusercontent.com/sgl-project/sglang/codex/muse-spark-stack-05-helion-verify/python/sglang/kernels/ops/muse_spark/helion_megakernels/attention_boundary_verify_megakernel.py

Pin the fetched commit and record license. The implementation covers O-projection, residual/norm, pooler, router, routing and pre-expert quantization for TP4 T=8/16. Our contract is TP8/EP8 and BF16 activations with NVFP4 weights. It is an architectural reference, not a compatible replacement. Never import its activation quantization merely because its tests allow it.

Compare operation by operation:
- Projection output layout and whether a full tensor is materialized.
- Row statistics: how sums of squares are produced, combined and consumed.
- The precision of residual and router input at every rounding point.
- Where pooler/router share activation loads.
- Where top-k consumes logits and where expert inputs are normalized/gathered.
- Every inter-CTA dependency and how producer completion becomes visible.
- Scratch reset and repeated graph-replay behavior.

Write `source_delta.md`: each difference, whether ours already implements it, applicable shapes, required precision changes (must be none outside the rules), proposed independent implementation and maximum plausible saved work.

Reject full-layer megakernel transplantation. keell0renz's lessons report cross-SM synchronization becoming much more expensive during saturated memory traffic. An alternative must explain how it avoids the measured failure, not merely use different syntax.

### B2. Quantify a ceiling before coding

For each surviving difference compute bytes removed, launches removed, duplicate work added, synchronization introduced and affected load fraction. Use measured boundary latency to cap potential savings; theoretical bandwidth alone is insufficient for tiny operations.

Select at most one candidate whose optimistic full-decoder ceiling exceeds roughly 0.8–1.0%. If every candidate is below that ceiling, investigate one larger mechanism or stop; do not consume the budget with many negligible tweaks. A lower ceiling may still justify a cheap experiment if its implementation is already complete and evidence is unusually strong.

Deliver `experiment_card.md` before compilation: hypothesis, baseline hash, exact path, new code boundary, required buffers, expected bottleneck, ceiling, correctness hazards, compile budget and explicit no-go result.

## 5. Phase C — narrow boundary experiments

These are alternatives. Selection depends on Phase B, not on their order here.

### C1. Projection epilogue plus row-statistics production

Hypothesis: the projection already visits output fragments, so it may produce partial row statistics without a separate full-output pass.

Design requirements:
- Establish whether output fragments are final or split-K partials. Squared partial sums cannot replace the square of the completed sum.
- Preserve the exact rounding point used by the downstream norm.
- Emit statistics only after the relevant reduction is complete.
- Retain a legal small reduction/consumer launch if the whole row spans CTAs; avoid a new global spin barrier.
- Count the added register pressure, stores and reduction traffic.

Test the complete producer-to-normalized-consumer segment, then decoder. Stop if statistics overhead or changed occupancy cancels the removed pass, or if the incumbent already performs this fusion.

### C2. Shared boundary input for pooler and exact router

Hypothesis: one remaining boundary may redundantly read or reconstruct normalized residual activations for pooler and router.

Design requirements:
- Inspect the existing fused pooler/router path first; it already handles some shapes.
- Preserve the FP32 router-input semantics. Earlier code rounded this path incorrectly, so BF16-only reuse is not acceptable.
- Share only mathematically compatible loads/statistics. Keep separate representations where required.
- Limit the prototype to a previously unfused supported shape range.

Stop before coding if this is merely the failed T=8 GLE expansion or already covered by the incumbent. A different shape dispatch alone is not a new mechanism.

### C3. Routing-to-expert-input boundary

Hypothesis: an independently implemented consumer can load original activation rows via timed routing metadata and avoid one global gathered tensor.

Design requirements:
- Keep all routing and input-dependent preparation within the timed call.
- Preserve routing publication synchronization; CTA-local barriers cannot publish a full grid's work.
- Gather directly into the operand layout of the first expert projection.
- Initially preserve the second expert projection and output reduction unchanged.
- Compare removed materialization against scattered-load overhead and loss of activation reuse.

Stop if the incumbent already has equivalent direct gathering, or if complete route+first-projection time does not improve. Do not use a gather-only microbenchmark as a score claim.

### C4. Final residual/norm boundary

If earlier boundaries are exhausted, inspect whether the final unpooler output can feed an independently authored residual/norm operation with fewer passes. Account for the T<=8 sliced unpooler semantics: the remaining channels still participate in residual/norm computation. Keep FP32 residual output and correct BF16 next-layer input.

Existing cluster-based boundary code may already eliminate the available passes. If so, mark this closed without a GPU experiment.

## 6. Phase D — MonoMoE fallback research

Source: https://arxiv.org/abs/2609.04244
Claimed implementation: https://github.com/flashinfer-ai/flashinfer/tree/main/csrc/fused_moe/monomoe

The paper describes weight-major scheduling with the token tile on tensor-core N, avoiding expert-local token materialization. Its H200 results are not measurements against our credited NVFP4 decoder. Retrieval of the linked source directory failed during the initial survey; verify source availability and license before planning a port.

Feasibility gates:
1. Confirm actual supported quantization, activation precision, tensor-core shapes and Hopper architecture.
2. Compare the design against our current stream-K MoE. If the same scheduling and materialization savings are already present, stop.
3. Identify a mechanism that works with unchanged NVFP4 weights/scales and BF16 activation requirements.
4. Build a single expert/tile microbenchmark with real weight bytes and representative route occupancy.
5. Include scheduling, scatter/gather, activation, reduction and scratch reset in the next benchmark before judging benefit.

Do not implement a complete new MoE unless this prototype beats the current matched segment. Preserve variable routes and skew across seeds; never bake known expert assignments into dispatch.

Deadline priority: Phase D is higher risk than a narrow boundary change. If unresolved by November 3, stop it for the deadline campaign and preserve findings for later ownership work.

## 7. Implementation and provenance discipline

Keep public sources as read-only research snapshots, with commit hashes and licenses. Independently authored prototypes should be small, readable files with design notes linking conceptual sources. Since we have read upstream source, do not claim a formal clean-room process.

An adapter around a credited decoder is acceptable as a clearly labeled research comparison. It does not make the candidate independently authored end to end. Reused implementation retains attribution. Record provenance at function/kernel level and distinguish our new code from inherited helpers and generated CUDA strings.

Do not stage unrelated dirty files, delete notices, or replace the live submission during research. Final source promotion must identify exactly what remains third-party derived.

## 8. Benchmark protocol and gates

Use the existing external research harness after reviewing its semantics:
- `research_harness/gpu.py`: paired CUDA-graph timing and mutable-input reset.
- `research_harness/stats.py`: weighted-load aggregation.
- `research_harness/loop.py`: fingerprinting, serial launch, logs and bounded jobs.

### Gate 1: harness sanity

- Same baseline versus itself must be numerically identical and time near 1.0x.
- Replay a captured graph before reading its outputs; this was missed in the earlier BF16 probe.
- Calibrate fixed overhead or report raw timing consistently; do not assume an arbitrary subtraction for tiny differences.
- Keep graph outputs, workspaces and tensor maps alive for the entire test.
- Reset mutable KV-cache/input state identically outside both timing regions. Reset must not calculate algorithm intermediates.
- Compare repeated replays to detect stale counters or scratch contamination.

### Gate 2: local correctness

Use canonical checks externally; submissions must not import reference modules. Check all returned tensors, expert IDs, in-place cache updates, finite values and per-row bounds. Preserve router behavior, outliers, page sizes, non-unit KV scales, long contexts, uneven lengths and nearby batch sizes.

Start with the changed path; run all 13 canonical decoder cases before official checking. Scored cases also need correctness on fresh inputs, not only timing.

### Gate 3: cheap paired screen

Warm both variants equally. Start with approximately 30 reversed-order pairs on affected shapes, cold L2 and multiple legal seeds. Count samples explicitly as pairs, not ambiguous repetitions. Keep source and compiler flags fixed.

For code-generation-sensitive edits, build a control with the new structure disabled to distinguish algorithm changes from incidental compiler shifts. Reject candidates that only win against a differently built baseline.

No-go: correctness failure, scratch drift, repeated slowdown, or no meaningful affected-segment win. Investigate one concrete failure; do not launch blind parameter sweeps.

### Gate 4: full-decoder confirmation

For a survivor, measure all 12 scored cases with around 100 paired observations per case, subject to time budget. Run a second independent draw with different seeds. Retain raw samples, medians and paired uncertainty. Aggregate into six weighted loads and the geometric score.

The harness currently applies a conservative uncertainty floor near judge noise. Do not remove that floor just to call a 0.55% gain significant. Report the paired signal and the floor separately, and label inconclusive results honestly. Target a larger, repeated 1–1.5% effect where possible.

A candidate can proceed to an official correctness check based on a reproducible local signal, but a small apparent gain is not yet grounds for a score claim or submission.

### Gate 5: official confirmation and promotion

Read remaining quotas first. Plan for one candidate check plus a confirmation check for gains under 2%, keeping at least one scored attempt available. Respect the CLI's same-code deduplication; determine supported rerun behavior rather than changing comments to fish for timing draws.

Inspect canonical test outcomes, preparation time, per-shape timing, code/source identity and warnings. Reconcile local/official discrepancies before promotion. Submit the exact passing source only after provenance/ownership requirements are resolved and the evidence supports a real benefit.

Record final source hash, job ID, before/after score and whether top 100 was actually reached. Do not infer overall rank from an unverified board mapping.

## 9. Resource allocation and schedule

All launches run in the foreground with local process and remote function timeouts. Compile on CPU into the shared cache; use consistent sm_90a architecture and image settings to avoid GPU recompilation. After timeout/interruption, check Modal state and stop an orphan before retrying.

Reconcile the remaining dollar allowance first. Conditional GPU allocation for one research cycle:
- Up to 60 seconds: baseline/harness sanity.
- Up to 90 seconds: one affected-segment screen.
- Up to 150 seconds: full-shape confirmation of a survivor.
- Reserve up to 180 seconds: independent confirmation or one narrowly justified fix.
Total initial ceiling: eight H200 minutes, only if remaining dollars permit it. Compilation/startup overhead must be included in cost accounting. If validation needs more time, report that rather than silently skipping required cases.

November 2: scoring/provenance snapshots, runtime map, source delta, ceiling calculation, choose one candidate. CPU compile if feasible.
November 3 morning: sanity, correctness and paired screen. Reject or fix one concrete issue. November 3 afternoon: full-decoder confirmation and first official check for a survivor.
November 4: confirm, resolve ownership for the concrete candidate, and submit with queue margin. Suggested freeze for new mechanisms: noon Pacific. Aim to have a validated submission ready by 6 pm Pacific; these are planning targets, not queue guarantees. Final deadline is 11:59 pm Pacific.

If the first mechanism fails, permit one fallback with a distinct evidence-backed explanation. Do not spend the remaining day repeating stages, PDL flags, T=8 glue, early-weight-prefetch or tail-steal variants already recorded as no-go.

## 10. Required artifacts and completion criteria

Deliverables for the execution phase:
1. `baseline_manifest.json`: source hashes, environment, scores, quotas, budgets and job IDs.
2. `runtime_map.md`: actual per-shape dispatch and dependencies.
3. `source_delta.md`: upstream mechanisms, overlap and applicability.
4. `experiment_card.md`: one selected falsifiable hypothesis.
5. Independent prototype plus a clearly labeled research adapter if needed.
6. Raw paired results and `decoder_comparison.md` with per-case and per-load effects.
7. `provenance.md`: original versus derived implementation and retained notices.
8. Decision: promote, inconclusive, or no-go, with exact reasons and cost.

Success is a verified improvement and observed top-100 rank under the current official definition, with honest provenance. A failed experiment still delivers useful evidence, but is not success on score. If the score route requires retaining credited kernels and that remains unacceptable, state the conflict and continue only the independent implementation track without promising a deadline rank gain.
