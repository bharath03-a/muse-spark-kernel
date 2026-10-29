# Whole-specific routing metadata overlap (source-only)

Baseline: promoted strict whole `candidate_v35_wxpin32_strict.py`, scored5648767e139d441b,12501.65us. Preserve all defaults. No implementation, check, CPU/GPU launch, or BF16-projection edit in this investigation. Existing XG_VECTOR_REVIEW.md already covers all-SM segmented/vector gather; do not duplicate it.

## Actual dependency timeline

The original v35 kernel (unchanged by strict repair) starts side work after resid1 produces BF16 h. `side.wait_stream(cur)` precedes `_router(h,lg)` then `g.routing(lg,...,ids,loc,tw)`. Independently the current stream computes `_proj(h,pooler)` then BF16 `g.rmsgain(...,p)`. `cur.wait_stream(side)` joins the physical side-stream completion before either GK or WX consumes p/loc/tw. Kernel lines13647–13658 in the readable uncollapsed v35 source show the fork and join. DL_FUSE is false. The small-T PDL policy does not remove this full side-event dependency.

Therefore metadata depending ONLY on completed ids/loc/tw can run after routing while the pooler is unfinished. Activation gathering cannot move into this interval: it reads p, which becomes valid only after the main-stream pooler+RMS boundary. No BF16 boundary moves.

Let P be main-stream pooler+norm completion time after the fork; R be side router+routing completion. Hidden slack H=max(0,P-R). For a metadata epilogue duration E, pooler contention deltaP, consumer setup saving S, and extra metadata-load cost L, the gain is:

`S - (max(P+deltaP,R+E)-max(P,R)) - L`.

If E<=H and deltaP≈0, metadata production adds no critical-path latency. This is possible in the existing graph; it is not proved by subtracting aggregate kernel durations.

## Per-load duration-based window estimates

The table uses scored5648767 profile kernel durations and its preparation log. Small pooler/router identities are known from their selected GV template tuples. The lower endpoint ignores rmsg duration entirely; the upper adds the reported rmsg span. Both are estimates, NOT guaranteed stream-timestamp bounds: PDL kernel spans may contain predecessor waits, and concurrent kernels can slow one another. The guaranteed H lower bound from exposed records remains0 for every load.

|tokens/load|pooler profile us|router profile us|route us|duration-based H estimate us|
|---|---:|---:|---:|---:|
|1/decode1 sliding|14.4|4.0|6.0|4.4–7.4|
|1/decode1 full|14.1|3.6|6.3|4.2–9.5|
|8/verify1 sliding|14.4|4.5|5.8|4.1–11.2|
|8/verify1 full|14.8|4.9|5.8|4.1–11.0|
|16/decode16 sliding|14.2|4.3|5.8|4.1–7.1|
|16/decode16 full|14.1|4.4|5.6|4.1–6.9|
|32/decode32 sliding|17.2|6.1|5.6|5.5–8.7|
|32/decode32 full|16.9|5.9|5.6|5.4–8.6|
|128/verify16|23.52 prepare-time pooler proxy|7.8 profile WGR|5.8–6.0|about9.7–14.3, proxy only|
|256/verify32|27.62 prepare-time pooler proxy|about9.2 including splitKreduce|5.7–5.9|about12.5–20.2, proxy only|

Large GEMM profile labels do not uniquely identify the pooler, so prepare-time latency is used only as a proxy, not an actual overlap bound. The existing router selection benchmark deliberately overlaps `_router` with `_proj(pooler)`, confirming the intended concurrency, but omits routing/RMS/metadata and cannot establish H.

A compact1–3us metadata epilogue is plausibly hidden, especially at small T where few routes exist. Do not assume4us or more of hidden work without measuring fork-to-main-ready and fork-to-side-ready timestamps on matched fresh inputs.

## Producer ownership that avoids extra launches

The existing route kernel is one warp per token,4warps/CTA. Invalid final-CTA warps return early. Consequently a block-wide barrier added after that return would deadlock nearby token counts.

A safe source design is a last-VALID-token-warp epilogue:

1. Each valid warp writes its existing eight ids/loc/tw entries unchanged. Every writer device-fences its OWN stores, then all32lanes execute a warp sync. A leader-only fence does not publish peer-lane stores.
2. Warp leader makes an acq_rel GPU-scope arrival on a per-invocation counter; expected arrivals=T valid token warps, not ceil(T/4) blocks. Last-arrival status is broadcast within that warp.
3. Only the last valid warp builds stable metadata by ascending route INDEX, never arrival order. Input metadata reads should bypass stale L1 or use appropriate acquire semantics. The metadata order must match GK/WX's current deterministic expert/rank conventions.
4. It writes the final metadata and resets the arrival counter only after its work. The ordinary side-stream completion event and current-stream join publish the completed table to MoE. Existing early PDL_GO is NOT data readiness. Same-side-stream ordering/full join protects counter reuse on subsequent calls.
5. There are no CTA/global blocking barriers in this producer, no waits on p/main-stream work, and no circular dependency. No activation is read before rmsg completes.

Attention specialist independently approved this ownership/event model subject to the exact fences/acquire/order conditions above. Stable counts/positions/list may be generated using warp match masks and shared32expert counters, followed by a prefix/scatter. Nonlocal route entries need explicit sentinels each invocation. The table must be freshly overwritten inside timed work; prepare holds only storage.

## What setup can actually disappear

GK/WX consumers can copy stable counts/offsets/list/rank data, avoiding repeated all-SM match/prefix/scatter. But they still need their own barriers/TMA initialization, output-row/K partition calculation, ring cursor and weight prefetch, and activation staging. GK's source has only one route-histogram batch at scored M<=16; its initial PROF span also includes unrelated work. It is therefore unjustified to call the entire initial span removable metadata overhead.

WX has more repeated route-ranking waves and CTA barriers at R=1024/2048; metadata reuse is more promising there, but its gather must still copy p after the join. Shared-table loading and any changed register usage must be counted. This design is distinct from xg_vector's store/segmentation work and must first be evaluated alone.

## Whole aggregate threshold

Using the exact scored5648767 per-case medians and weights46/16 within each of6loads, uniform NET per-case reductions would yield:

|net savings per case|whole aggregate speedup|
|---:|---:|
|2us|1.01175x|
|3us|1.01775x|
|4us|1.02382x|
|5us|1.02999x|

Thus hiding producer work genuinely bypasses the standalone extra-launch penalty, and4–5us NET consumer savings could clear2–3% whole improvement. Existing records do not establish that much removable GK setup. The source model supports measuring this opportunity; it does NOT yet justify implementing/checking a claimed2–3% improvement.

Before prototype allocation: obtain matched main/side ready timestamps and existing GK/WX setup/tail phase durations; identify which instructions/barriers are removed, and subtract metadata table loads plus pooler contention. Preserve the canonical BF16 gate/up/inter/down/normweight boundaries and all FP32 full-K reductions. No checkpoint caching or host branch on route values.
