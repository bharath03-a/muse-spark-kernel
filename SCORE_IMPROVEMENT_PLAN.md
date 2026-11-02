# Score improvement plan — November 2–4, 2026

## Objective and decision

Improve real, repeatable performance before November 4, 11:59 pm Pacific (November 5, 12:59 am Phoenix). Overall top 50 is the target, not a promised outcome. The user has clarified that submission implementation must be our own. Independently authored replacements are now on the critical path. We may study published techniques and improve upon them, but modifying copied code does not establish independent authorship. Retain all required attribution wherever inherited code remains.

Keep live submissions intact while researching. A candidate must have a specific mechanism, a measured benefit, and canonical correctness before promotion. Nothing in this plan permits reduced precision, cached inputs/outputs, work moved outside the timed call, or modifications to the judge.

## Revised execution order: ownership first

This section supersedes the earlier experiment ordering below. Those mechanisms remain research hypotheses, but must be implemented in our own candidate rather than patched into the copied submissions.

1. **Audit provenance before choosing a base.** Inventory every active kernel, helper, generated CUDA string, dispatch table and embedded backend. Classify as independently authored, third-party derived, or uncertain. Do not trust `submission_before_public.py` or any other filename as proof. Preserve notices and document uncertain origins before reusing code.
2. **Write a small independent BF16 submission first.** Build from the problem specification and verified original components. Use standard library operations as a correct fallback. Create our own prepare/kernel interfaces, dispatch and benchmark harness; do not transplant the nested backend factories from the current submission.
3. **Close the measured BF16 gaps one family at a time.** Start with m16/m32 qkv, pooler and unpooler, then m128 projections and m256 router. Investigate lossless weight packing and register/shared-memory reuse for small m, and independently implemented pipelined GEMM for larger m. Previously failed zip/cluster/register-direct variants remain negative evidence: require a specific changed mechanism before trying them again. Preserve FP32 router accuracy.
4. **Compare against two baselines.** Measure the candidate against our provenance-verified original baseline and the current credited implementation. Record both absolute latency and the remaining gap. Matching our old baseline is progress in reconstruction, not a score improvement.
5. **Build our own MoE next.** Start with a simple correct expert-grouped implementation, then test direct activation gather and expert-local weight reuse. Benchmark each stage before integrating it. Use keell0renz's timings as a performance reference, not his implementation as the candidate scaffold.
6. **Build our own attention and compose the decoder.** Start from verified original attention code if available. Reuse our independently implemented primitives in the decoder; test fusion only after the unfused version passes. Treat the decoder as an integration milestone rather than a fourth unrelated rewrite.
7. **Promote only with an explicit result.** Report originality/provenance, correctness, latency versus our old code, and latency versus the live submission separately. Do not silently replace a faster live submission with a slower independent rewrite; present that tradeoff if it arises.

For every studied technique, write a short design note covering the mathematical operation, data layout, synchronization, expected bytes moved, our proposed improvement, and the experiment that could disprove it. Credit intellectual sources. Because we have already inspected upstream source, do not describe this workflow as a formal clean-room process. Cosmetic rewrites and header removal do not meet the user's goal.

The first concrete deliverable is a provenance inventory and an independent BF16 candidate with a per-shape comparison report. No additional GPU spend is required for the inventory and design stage. The earlier cuBLASLt probe is optional research only and does not satisfy the ownership objective.

## Baseline and headroom

Live OKC leaderboard readings from this session on November 2, not a prediction of final rank:

| Board | Current speedup | Reported #50 cutoff | Approximate time reduction to that cutoff |
|---|---:|---:|---:|
| BF16 projections | 1.5724x | 1.5910x | 1.17% |
| Attention decode | 2.6397x | 2.7390x | 3.63% |
| Routed experts | 2.3842x | 2.4710x | 3.51% |
| Whole decoder | 2.2556x | 2.3111x | 2.40% |

Time reduction = 1 - current speedup / cutoff. Individual-board #50 is only a reference: it does not establish the overall ranking formula or guarantee overall top 50. Refresh standings and quotas before submitting; cutoffs can move.

Evidence constrains the search:

- Attention full-context streaming already reaches roughly 4.0 TB/s. Removing its arithmetic saved only about 3.6 microseconds at batch 16. More arithmetic optimization is a weak lead.
- MoE at batch 256 reads about 792 MB and achieves about 85% of the measured practical bandwidth ceiling. Its route/gather ramp is about 7.7 microseconds, with a final tail around 5 microseconds. Early weight issue did not reduce the ramp.
- Whole-decoder MoE consumes roughly 41–65% of verify execution in the recorded profile. A portable MoE improvement has two-board value, but integration must be measured independently.
- BF16 backend selection is exhausted among the previously tested backends. Re-running that selection is not a new design.

## Priority 0: investigate the new whole-decoder result

The live leaderboard now reports keell0renz at 4.1519x (job f5b964350c0b440e), much higher than the handoff-era comparison. Our current board ranks are BF16 138, attention 158, MoE 196, decoder 137. The new score is provisional and provides no evidence by itself about implementation or correctness.

Before spending more GPU budget:

1. Refresh the public repository and inspect newly published changes since the handoff. Do not assume the submitted source is public or matches repository HEAD.
2. Identify any structural changes: intermediate precision/layout allowed by the rules, eliminated materializations, cross-operation fusion, routing representation, or scheduling changes.
3. If a technique is public, derive its dataflow and traffic savings and check whether it applies to our workload. Credit any reused implementation. If pursuing an original implementation, author and test it ourselves rather than relabeling copied code.
4. Build a tiny isolated proof of the mechanism before adopting a large upstream file.
5. If no matching source or explanation is available, record that uncertainty and proceed with the independent experiments below. Do not stall waiting for publication.

This source review is a newly justified read-only step because the live state differs materially from the handoff. It does not commit us to using another author's implementation.

## Priority 1: remove the MoE activation-gather boundary

### Hypothesis

The failed early-weight-prefetch experiment tells us the gather dependency remains. Change where activation gathering happens, rather than issuing weights earlier behind the same dependency.

### Design

1. Trace route -> gather -> first expert GEMM in the existing standalone MoE implementation. Enumerate every intermediate array, synchronization point, and consumer.
2. Keep routing and expert token lists correct, but have an expert CTA load its own input rows from the original activation tensor using those lists.
3. Gather a tile directly into the shared-memory operand layout consumed by the first GEMM. Remove the global gathered-activation write/read where possible.
4. Preserve the existing second GEMM and reduction at first. This isolates the change and limits correctness risk.
5. Start with small token counts, where launch and setup costs matter more. Test verify sizes only after the direct-gather path is credible.

All routing and input-dependent preparation stays inside the timed invocation. A kernel boundary may be necessary for routing publication; do not replace it with an unsafe grid-wide spin barrier.

### First decisive experiment

Build a minimal first-GEMM variant using the real token lists and original activation rows. Compare the COMPLETE route/gather/first-GEMM segment against the original, including extra indexing and scattered reads. Then compare complete MoE calls.

Measure batch sizes 1, 8, 16, 32, 128, 256; distinct expert counts; gather bytes removed; first-consumer start; register/shared-memory use; and full cold-L2 graph latency. Use normal inputs, outliers, skewed routes, empty experts, and expert-boundary cases allowed by the spec.

### Success and stop conditions

- Advance only if total MoE time improves on a meaningful subset without offsetting regressions elsewhere.
- A 2–4% board gain is a research target, not an estimate supported by existing measurements.
- Stop if eliminating the global gather merely shifts more time into the consumer, or occupancy falls enough to erase the benefit.
- If the tiny prototype loses, do not rewrite the full 7,000-line kernel around it.

## Priority 2: reduce repeated expert-weight traffic at small batches

### Hypothesis

At small batches, separate token work units may reload a popular expert's weights. A compact expert-local token tile might reuse each weight fragment across more tokens, reducing actual memory traffic rather than trying to exceed bandwidth.

This may already be implemented. Source/traffic inspection is the first gate, not optional preparatory work.

### Design and experiment

1. Record real expert occupancy and the mapping from tokens to weight-loading CTAs on multiple legal random seeds.
2. Count how often each expert's weight tiles are loaded. Separate compulsory traffic from redundant traffic.
3. If duplication exists, build one expert-major microkernel that groups its routed tokens into a small tile and reuses a loaded weight fragment.
4. Compare padded work, scheduling overhead, imbalance, and total bytes with the current skinny path.
5. Select by public shape information, or by routing metadata computed during the timed call. Never specialize on known judge routes or seeds.

Advance if the end-to-end gain survives varying expert occupancy. Stop immediately if the current implementation already achieves this reuse, or if extra padding costs more than the eliminated loads. This is an alternative to Priority 1, not a simultaneous full rewrite.

## Priority 3: fuse sliding-attention preparation with partial attention

### Hypothesis

Sliding-window cases have a larger launch/setup component than full-context cases. Removing a real preparation boundary could help more than changing split counts or pipeline stages.

### Design

1. Trace the exact tensors produced by query/preparation kernels and consumed by sliding attention for decode and verify separately.
2. Check whether one CTA can produce the needed query representation and consume it locally without repeating expensive preparation per split.
3. Build a sliding-only kernel that fuses that preparation into partial attention. Keep the existing merge/reduction initially.
4. Favor batch 16 and 32 if their dependency mapping permits useful fusion; retain the existing path elsewhere.
5. Measure a second stage that fuses the merge only if the first stage wins and a legal synchronization strategy is clear.

The prior verify-sliding vfg variant lost. This proposal must remove an actual kernel/intermediate boundary that that variant retained; if inspection shows it does not, stop before compiling.

### Measurement and gate

Compare complete sliding attention calls, not just partial kernels. Verify output, probe rows, sequence tails, page boundaries, and nearby batch sizes. Report savings weighted by the 46 sliding versus 16 full layers used in the recorded scoring profile, then evaluate the actual load-based geometric score.

A few microseconds in sliding can matter, but convert the measured result into the board score before predicting rank. Stop if fusion duplicates enough query work to erase the launch savings.

## Priority 4: bounded BF16 library-configuration search

### Hypothesis

Choosing among fixed backends did not help; searching configurations within the large-m cuBLASLt backend could still reveal a better algorithm configuration on the installed judge library.

### Experiment

- Start with the existing large-m plans for qkv_gate, pooler, unpooler and o, plus pooler m128.
- Vary tile, cluster, stage count, swizzle and split-K/reduction configurations within valid library combinations.
- First compare each incumbent with itself; require this control to pass correctness and return a ratio near one.
- Replay every captured graph before reading its output. Validate each candidate before recording a time.
- Short-screen candidates, then remeasure finalists with interleaved cold-L2 replays on fresh inputs and a separate draw.
- Include workspace limits and prepare-time cost. A configuration must fit the actual submission constraints.

The initial probe in this session is INVALID: it checked captured outputs before replay. It produced zero accepted candidates, which says nothing about configuration quality. The harness now replays before checking; it has not been rerun after that fix. No judge quota was spent.

This is a bounded fallback, not a reason to resume endless knob searches. One valid sweep; stop if there is no repeatable material win. If there is a win, reproduce it in a minimal candidate with attribution intact before an official check.

## Priority 5: move a winning primitive into the whole decoder

Only attempt after a standalone MoE or attention win.

1. Locate the matching integrated kernel and compare its layout, precision, routing semantics and synchronization with the standalone version.
2. Port one winning mechanism, retaining existing interfaces where possible.
3. Measure the affected integrated stage and total decoder latency.
4. Check all layer outputs and intermediates required by the canonical tests. Preserve exact router behavior and supported nearby shapes.
5. Run an official whole-decoder check only after the local end-to-end result is credible.

If the MoE fraction is f and its measured latency reduction is r, the rough whole-layer reduction is f*r before integration overhead. For example, 5% off a 50% component suggests about 2.5% overall. This is a planning estimate, not a benchmark. Do not assume a standalone win transfers automatically.

## Optional follow-up: small MoE tail repair

Recorded CTA cost skew suggests a fixed scheduler correction might shorten the tail. First build a cost model from actual token counts and expert work; do not hard-code which CTA IDs happened to finish late on one seed.

Compare predicted and observed completion time across routing distributions. Only implement redistribution if it balances real work without extra launches or dangerous global synchronization. The few-microsecond ceiling makes this secondary to gather removal and weight reuse. Avoid a wholesale dynamic work queue unless a microbenchmark first proves its overhead is lower than the recoverable tail.

## Experiments we will not repeat without new evidence

- BF16 self-calibrating selection over the same backend set.
- Attention NPW/PF/stage sweeps, arithmetic-only optimization, widened fold participation, or fine-grained tail stealing with the already measured setup cost.
- MoE early weight issue, the failed larger shared-memory ring, or PDL toggles.
- Whole-decoder GLE T=8 without a new supported kernel design; its estimated score ceiling was too small anyway.
- Public-code substitution presented as original work.

## Execution sequence and resource limits

### November 2: establish one defensible lead

1. Freeze baseline hashes and preserve the dirty workspace.
2. Confirm no active Modal apps; this was done at session start and both short probes subsequently completed.
3. Review the newly changed public-source situation above, then inspect whether direct gather or expert-weight reuse is absent from the current kernel. Choose ONE prototype based on evidence.
4. Use a short isolated experiment to falsify the mechanism before building a complete candidate.
5. If it fails, test sliding-attention fusion feasibility; use the BF16 sweep as the bounded fallback.

### November 3: validate and integrate

1. Reproduce the best local result in a second paired draw with different legal seeds.
2. Run full correctness and nearby-shape coverage.
3. Spend an official check; investigate discrepancies rather than submitting through them.
4. Port a verified standalone win into the whole decoder if the layouts permit it.
5. Write down every negative result so the next session does not repeat it.

### November 4: protect and submit measured wins

Stop major rewrites early enough to allow checks, reruns, queues and submission before 11:59 pm Pacific. Refresh quotas and standings. Recheck gains under 2%; submit only a passing, justified candidate. Record job IDs, source hashes, score intervals and attribution. Never use a last-minute unverified candidate to replace a known passing submission.

### Budget

The handoff estimates $3–3.5 spent against a $5–6 allowance; remaining dollars are approximate, not verified billing. Check actual usage before further paid work. Cap the next discovery phase at roughly 8 H200 minutes total, split into short foreground runs. This is a time ceiling, not authorization to exceed the dollar allowance. Reserve capacity for confirmation/integration. Do not launch multiple independent searches at once.

Every remote job gets both a server function timeout and a local process timeout. On timeout or interruption, inspect the app and stop it if still running. Never leave a detached research job.

## Measurement and promotion protocol

For each experiment record:

- Hypothesis and what differs from failed attempts.
- Baseline/candidate source hashes, exact GPU/image, shapes and legal seeds.
- Full timed work included; no cached activations or prepared input-dependent results.
- Canonical correctness, per-row/probe checks and nearby-shape results.
- Paired cold-L2 graph times, repetition counts, variability and independent confirmation.
- Per-load speedups and the resulting geometric aggregate, not an average of per-shape percentage gains.
- GPU seconds, quota spent, job IDs and a clear go/no-go conclusion.

Use official checks to validate promising candidates, not to discover whether a blind edit works. Gain under 2% needs another official check before submission per the handoff. A passing but statistically flat result remains a no-go.

## Expected outcome

The realistic target is one original, portable improvement worth a few percent, potentially benefiting two boards. Evidence may rule all of these out. The plan is designed to reveal that quickly and preserve budget and good submissions; it cannot guarantee top 50 or an entirely independent four-board implementation by the deadline.

## Outcome — November 2, 2026

### Verified findings
- **Overall rank = decoder rank.** Overall rank equals `muse_decode_layer` rank exactly for the first 516 teams (leaderboard JSON, Nov 2). Other boards show no visible effect. We were at 2.2556x (#138); the cutoffs were 2.268x for #100 and 2.3142x for #50.
- **Decoder score:** each load is scored as L = 46·t_sliding + 16·t_full, and the speedup is the geomean over the 6 loads. Baseline check `681c0802458c4462`: 2.250x. Load-weighted time shares: MoE 47%, projections 24%, attention 15%, glue 14%.
- **Provenance:** the BF16 base (`muse_bf16_proj/submission_before_public.py`) is ours; the user confirmed this. See `muse_bf16_proj/PROVENANCE.md`. The MoE `submission_before_public.py` is derived from lee101/openkernels (same v56 history) and has no attribution header, so it is **not ours**.
- **Ownership:** the user approved a partly derived decoder submission: credited kernels plus our attributed changes.

### Own MoE (independent, `muse_moe_w4a16/own_moe_v*.py`)
| Version | Speedup | Notes |
|---|---|---|
| v1 | 0.417x | Triton grouped GEMM, PyTorch routing glue |
| v2 | 0.677x | Fused routing kernel, interleaved gate/up, lookup decode, split-K |
| v3 | 0.681x | Per-bs config sweep in prepare; decoding once per expert lost (serial tail) |
| v4 | 1–12% slower than v3 (Modal) | 16-byte repack; ncu shows integer-instruction-bound, 151–188 registers, 12–17% occupancy |

All versions are correct; rel-L2 is about 2.5e-3 against the 1.5% limit. The credited MoE reaches 2.38x. Next idea if resumed: hand-written CUDA with 32-bit weight loads, decode output written directly as wgmma A operands, and register use under 128.

### Decoder experiments (credited base), both dropped
- **Boundary fusion:** the credited code already fuses T≤8, C2 and C3. The best remaining candidate (O-proj + norm at T=16/32) is worth ≤0.34% geomean, below the 0.8% gate and the ±1.5% judge noise.
- **Projection GEMMs at T=128/256:** the ceiling is about 1.2% geomean, but every reusable kernel (our `_big_gemm`, our BF16 rewrite, rq, cuBLASLt sweeps, a pooler/router split) already ties or loses. Reaching the ceiling needs a new GEMM design.

### Decision
Score attempt stopped. Live submissions are unchanged (decoder 2.2556x). No submits made.
