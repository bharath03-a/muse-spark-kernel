# Expert memory / compression audit — 2026-10-26

Read TASK.md, cp14, ANALYSIS.md, FINDINGS.md, TOP50_CAMPAIGN.md and the actual zip pack/decode/reduction source. No GPU jobs, Modal sessions or judge checks launched. Existing cap respected. No kernel edits.

## Established findings

Current zip payload is exactly 12 bits/weight, before CSR: low byte contains 7 mantissa bits plus exponent LSB; four-bit plane contains sign plus three-bit high-exponent offset. It reconstructs MMA fragments directly. Thus suggestions to do register decode are already implemented, not new research.

CSR uses two int32 per exception plus row offsets. For reported exception fraction p=0.00014, its cost is 64p=0.00896 bits/weight plus 32/K bits for row offsets. At K8192 that is about 12.0129 bits/weight total. Optimizing CSR storage alone saves only tens of KB, not microseconds.

The no-x ablation falsifies the previous x-ingress diagnosis. Dense BN32 mma_v2 and cross-tile persistence have also now lost in actual tests. None should be presented as untried.

Zip full-minus-loads-only ablation is approximately 0.6us for the whole decode/MMA/fix/epilogue contribution. This bounds what epilogue-only changes can remove in those measured cases; it is not a reusable multi-us opportunity. Different cluster geometries need their own ablation.

## Primary sources: what transfers and what does not

- ZipServ https://arxiv.org/html/2603.17435v1 : constant-length fragment-aware decode is conceptually already present. Its advertised 2.21x is not a shape-matched H200 proof.
- Unweight https://research.cloudflare.com/papers/unweight-2026.pdf : section3.3 exceptions apply to **64-element tile rows**, not full K rows. Its row-verbatim method must not be rejected using K8192 whole-row probabilities. Conversely, whole-K adaptation would mark roughly 68% of rows at p0.00014 and is unsuitable. Its staged Huffman/transcode pipeline and cross-layer overlap cannot be assumed profitable in a single isolated timed projection.
- openkernels local reviewed snapshot d2057db2618075e960b7b8465180729bd87ccc38, https://github.com/lee101/openkernels . Current compression already uses comparable fragment packing. New async partial pushes / uncapped register variants merit scheduling/compiler review, not a compression claim. PDL remains prohibited.

## Narrow codec research, no promised speed gain

An 11-bit code (sign/mantissa byte + three-bit exponent palette) saves one payload bit but increases exception rate. With current 64-bit CSR, it beats current storage only when p11 < (1+64*p12)/64, approximately 1.5765%, ignoring other metadata. This threshold must be checked on actual prepared weights BEFORE building a kernel. More expensive exception computation can still lose below the threshold.

Alternative: 64-element tile-row verbatim exponent fallback, with one metadata bit per row. Approximate bits/weight for b exponent bits is:

    8 + b + (8-b)*(1-(1-p)**64) + 1/64

before offsets/alignment. At b3 this beats twelve bits only if fewer than roughly19.7% tile rows fall back. Physical byte rounding and offset reads must be included; reserving worst-case space everywhere erases compression. To preserve vector accesses and avoid warp divergence, align format units to warp-owned fragment groups. This is a CPU format feasibility study, not a recommended immediate GPU experiment.

Even a perfect 11-bit format saves just 4.194MB on a full67.109MB matrix relative to current12bit payload: at4.4TB/s the traffic opportunity is0.953us, before added decode. Qkv opportunity is0.536us. This cannot establish top50 by itself and is lower priority than compiler/scheduling evidence.

## Peer review: async register-source cluster pushes

Scheduler identified precise uncompressed WGMMA target: acc registers -> padded mine shared buffer -> CTA barrier -> float4 reload -> remote mailbox. A register-source async push can potentially remove that staging/transpose, preserving separate source-rank mailboxes and fixed-order FP32 sum. Router hi+lo needs register lane shuffles first and may not remove staging cheaply.

Safety requirements: remote destination CTA/shared allocation stays alive until transaction completion; acquire after matching completion before consumers read; complete receiver reads before reusing mailbox; all source regs consumed before repurposing; no early CTA exit that invalidates remote shared storage. Preserve FP32 partials and sum, then cast once. When changing mailbox padding, calculate bank mapping from actual lane address functions, not a blanket claim that float4 is conflict-free.

Consensus with scheduling/compiler peers: prioritize evidence of avoidable accumulator spills/caps and uncompressed staging overhead; no measured justification for a broad compression rewrite or revived x multicast. Scheduler's plausible0.2–0.7us benefit range is explicitly hypothetical and insufficient alone for top50. Compiler notes heterogeneous compressed strides require offset/metadata loads and can break aligned vector accesses; prove net bytes before any new codec.

## Vendor physical transpose audit

`bench/test_mm_vs_linear.py` already authors the exact offline transpose experiment: `w_t=w.t().contiguous()` followed by `torch.mm(x,w_t)`, versus F.linear, all four dense layers m128/256. No accompanying saved results found. Its timer uses host perf_counter plus synchronization and dirty flush.fill_, so its eventual results would not directly establish judge gains. cp14 `lt_make` fixes column-major A[k,n] opT and B[k,m] opN, so its algorithm enumeration does not cover the alternate physical weight orientation. Do not claim the transpose experiment is new or proven failed; classify as authored but unvalidated. A proper paired cold event harness is useful when GPU budget is authorized. This is a vendor fallback optimization, not a Triton-only kernel.
