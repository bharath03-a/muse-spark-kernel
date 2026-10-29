# Scheduling / reduction expert review

CPU/source review only; no new GPU, Modal, or judge jobs. Existing 1800-second cap respected. Read TASK.md, cp14, ANALYSIS.md, FINDINGS.md, TOP50_CAMPAIGN.md, actual top50 kernel sources/results, and current upstream references. All projected savings below are hypotheses.

## Evidence that supersedes older recommendations

- Dense BN32 no-split mma_v2 was implemented and passed correctness but lost badly: qkv m16 BK256 24.352 us vs torch21.328; m32 26.048 vs21.248 (raw paired cold graph). This is no longer an untried first recommendation.
- Cross-tile continuous pipeline helped its own persistent control, 29.072->26.528 us, but nonpersistent19.856 and cuBLAS14.000 were substantially better. Do not claim the relative improvement establishes a useful kernel.
- Distributed cooperative router reduction passed repeated graphs but m12813.376 vs cp1412.192 and m25615.296 vs15.264; later sweeps lost. More global counters and split variants are not evidence-backed priorities.
- no-x ablation directly falsified the claimed x-ingress bottleneck. The relevant microbenchmark measured a lower bound, not a hard L2 bandwidth ceiling. Multicast-x cannot be justified by byte arithmetic alone.
- No measurements establish that a scheduling-only patch reaches top50. Parent's actual-profile calculation requires 6.62% uniform time reduction from cp14 to1.5271. A handful of0.3us savings will not close that gap.

## Best remaining narrow mechanism: register-source asynchronous cluster pushes

Actual cp14 uncompressed WGMMA epilogue (lines383–443): wait all MMA, CTA sync, accumulator scalar stores into padded MPAD68 local shared memory, CTA sync, initialization cluster rendezvous, float4 local reload, remote owner scatter, full-cluster release/acquire rendezvous, fixed-order per-owner reduction, final BF16 cast. This provides a specific alternative beyond the already-failed global distributed reducer.

Use `st.async.shared::cluster.mbarrier::complete_tx::bytes` to publish accumulator registers directly into distinct sender-rank mailboxes at owner CTAs. The owner waits for expected byte completion and sums sender slots in the same fixed FP32 order. Local owner stores use an ordinary local path. This can remove padded local accumulator staging/reload and replace the post-scatter cluster-wide barrier with per-owner completion. Scalar/b64 forms map naturally to accumulator fragments; v4 is useful only if packing does not add expensive shuffles.

Important distinction: `cp.async.bulk.shared::cluster.shared::cta` has a SHARED source and therefore cannot eliminate accumulator staging. `st.async` has a REGISTER source and can. PTX supports its cluster weak/mbarrier form on sm90 since ISA8.1. Use f32/u32/b64 or v4.f32; b128 was only introduced in ISA9.2 and is inappropriate to assume in this fixed CUDA13 environment.

Safety protocol: initialize owner mailboxes/barriers before remote operations; retain cluster initialization rendezvous; exact expected bytes including masks; destination and mbarrier must belong to the same peer CTA; never use remote-only forms for local owner; sender and destination shared memory must remain alive until operations complete; fixed sender-slot sum, cast once. Do not delete the lifetime synchronization merely because every owner has received its own data. Router hi/lo halves need a register-layout/shuffle plan before claiming staging is removable.

Primary references:
- https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-st-async
- https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk
- https://raw.githubusercontent.com/lee101/openkernels/master/muse_bf16_proj/submission_v22.py (PUSH at594, v4.f32 at718)
- https://raw.githubusercontent.com/lee101/openkernels/master/docs/ideas.md (lines565–568: H100 bundled PUSH+uncapped qkv m16 12.54->11.46, m32 14.85->12.45; not an isolated PUSH experiment, not H200 proof)

Expected opportunity: 0.2–0.7us per affected uncompressed qkv/router shape, with zero or regression possible. Zip ablation places compute+fix+epilogue together around0.6us, so a multi-microsecond zip epilogue promise is untenable. If hypothetical savings0.5us qkv m16/m32,0.3us router m8/16/32,0.5us router m128/256 occurred, first-order log-score gain is only about0.8%, around+0.011 score from1.426; not top50 by itself.

Smallest future authorized test: compile current and direct-push identical mainloops; inspect PTX/SASS for no local accumulator transpose/reload; benchmark qkv m32 and router m128 first with30 paired cold graphs, then prove repeated/changed-input correctness and nearby shapes. Do NOT couple register limits, tile sizes, split count, and push in one comparison.

Peer-reviewed layout detail: current acc[j][0:2] is adjacent tokens at one row, acc[j][2:4] adjacent tokens at row+8. Current `mine` and receive layout are [token,row]; a direct v4 push is therefore NOT contiguous. First correctness version uses scalar f32 stores into the existing mailbox. A second version may change mailbox to [sender,row,token] and use v2 for adjacent tokens, at the price of different final coalescing/conversion. Existing owner slices generally partition contiguous tokens when BT is divisible by CSIZE; boundary masks must still be explicit. Do not assume v4 packing is free.

Triton-only route: retain Gluon distributed accumulator layout, allocate receive/barrier shared memory in a cluster-compatible layout, use supported num_ctas and inline assembly for mapa/st.async only after checking physical CTA ownership and allocated smem pointers. The language exposes inline assembly, but this review has not proven the required shared pointer extraction/distributed layout lowering on installed3.8. No fake runnable probe or CUDA-only rewrite was produced; API/lowering validation is a prerequisite.

## Second narrow mechanism: legally increase outstanding WGMMA groups

Compiler peer identified that current wait1 is tied to refill i+S-1. Blind wait2 is unsafe. A valid Q-group pipeline prefills S-Q and refills i+S-Q, reusing slot i-Q only after corresponding retirement. This trades shorter TMA lookahead for more outstanding tensor work; potentially useful compute-heavy router, likely negligible or negative on streaming layers. Parent should prioritize assembly evidence of exposed WGMMA waits before implementation; no quantified speed claim yet.

## Cross-review consensus

Direct discussion with expert_memory and expert_compiler occurred. Memory agrees no-x falsification and zip epilogue upper opportunity. Compiler corrected shared-source bulk versus register-source async distinction and supplied the ring retirement constraint. Neither candidate currently supports a top50 promise. Best group strategy is isolate one synchronization or register-lifetime change at a time and retain only improvements versus cp14, not slower experimental controls. Existing CUDA paths remain measured baseline; a Triton/Gluon implementation may require layout-supported distributed shared memory plus inline PTX and explicit num_ctas, whose generated layout must be inspected before assuming physical cluster ownership.
