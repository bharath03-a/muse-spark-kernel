# Research team operating plan

Goal: improve all five challenge scores, reach top50 overall, retain top50 whole-decode qualification target. Scores and rank require live verification; off-board checks and Modal timings are evidence, not leaderboard entries.

## Ownership and peer review
- Attention expert: page1 FP8 traffic, tensor layouts, async producer/consumer overlap, split reduction, softmax/numerical bounds. Reviews MoE TMA/register and full-layer stream interactions.
- MoE expert: exact NVFP4 unpack/scale instruction chains, MMA scheduling, activation reuse, persistent queues, load imbalance, barriers and normalization. Reviews attention producer/consumer and memory hypotheses.
- Whole-layer/prefill expert: per-load critical path, weighted performance model, dense/norm/residual fusion, prefill grouped GEMM, preparation/compilation constraints. Reviews component gains for portability and whole-layer impact; researches BF16 projections without overwriting independently managed candidates.
- Root: integrates candidates, manages shared experiment/check budgets, verifies actual scores, promotes only validated meaningful gains, maintains source provenance.

Agents communicate directly with each other through collaboration messages. Every proposed change should get one peer review covering both the bottleneck model and integration risks. Findings are shared in each challenge TAKEOVER.md; cross-cutting decisions go into root TAKEOVER.md.

## Experiment proposal
1. Observed bottleneck: exact profile/shape/source reference.
2. Mechanism: which bytes, instructions, dependency, occupancy or launch is removed.
3. Bound: plausible maximum time saving on that shape and impact on weighted whole-layer loads.
4. Small falsifying experiment: ablation or kernel benchmark before full check.
5. Precision/fresh-input/graph constraints and held-out nearby shapes.
6. Paired cold-L2 CUDA graph timing; report raw/overhead-adjusted distinction, median and variance.
7. Source hash, candidate path, command, job/app ID, result and decision.

Avoid tiny changes justified only by noise. Do not sum standalone geometric-mean score differences as whole-layer latency savings. Root approves shared compute allocation internally; this does not require interrupting the user for routine reversible work.

## Current evidence and budget

Use root TAKEOVER.md and VERIFIED_CANDIDATES.json for source-to-job provenance; PROGRESS.json is a timestamped snapshot, so ranks may have drifted. Current valid scored whole is strict v35 at 12,501.6459 us (5648767e139d441b); the raw v35wx run was withdrawn for missing BF16 boundaries. MoE strict v56 is 6,827.2569 us and attention conservative v75 is 1,942.7804 us. Top 50 is not achieved.

As of the user's latest checkpoint, Modal GPU usage is 1,782.6 / 1,800 seconds. No new Modal GPU jobs are allocated; the remaining 17.4 seconds is insufficient for controlled validation. Source review and bounded CPU compilation can continue. Any research diagnostic OKC check needs a root allocation and must never be scored or promoted as a kernel.

Corrected attention sweep 0653d51aad4c4281 completed; retained B32 prefetch results regressed and earlier rows were truncated. No winning setting was selected. Current priority: inspect completed cached SASS for head-chain overlap and bound whether routing metadata can be hidden behind whole-layer pooler work. MoE xg2 passes but its ~1% aggregate change is inconclusive; source work targets gather distribution or tail reduction. Prefill tests exact eight-operation unpacking without changing weights or scales.

## Ownership clarification from user
BF16 projection optimization belongs exclusively to other chat `Optimize H200 GEMV kernels` (01a16fae-7d31-73d0-85e8-637414fc218b). User authorized cross-chat coordination; root sent findings and requested stable integration snapshot. This team's agents must not edit BF16 files. Root and experts focus attention/MoE/wholedecode/prefill.
Peer exchange verified: attention and MoE experts jointly reviewed ALUconversion/producerconversion hypotheses; exact cachedcompiledvariants identified to avoid silentfallback. One180sH200cachedablation authorized. MoE/prefill experts caught BF16gate/up boundary discrepancy and refined proposal against actual SGLang defaultactivation semantics (FP32SiLU/product following BF16gate/up; BF16finalproduct). Reviewbeforepromotion.
