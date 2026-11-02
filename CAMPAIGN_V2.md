# Campaign V2 — staged climb (2026-11-01)

Overall rank #484/1826 (ranks only; 4 boards: bf16, attention, moe, decode_layer; prefill not counted).
Stages: top 300 → 150 → 100 → 50. Per-board cutoffs (speedup / µs):

| board | ours | #300 | #150 | #100 | #50 | #1 |
|---|---|---|---|---|---|---|
| muse_bf16_proj | 1.481 (#446) | 1.534 / 3148 | 1.563 / 3091 | 1.573 / 3071 | 1.587 / 3043 | 1.652 |
| muse_attention_decode | 1.983 (#526) | 2.571 / 1465 | 2.633 / 1431 | 2.668 / 1412 | 2.725 / 1382 | 3.339 |
| muse_moe_w4a16 | 2.032 (#516) | 2.337 / 5754 | 2.390 / 5628 | 2.406 / 5590 | 2.459 / 5470 | 2.764 |
| muse_decode_layer | 1.861 (#484) | 2.147 / 10534 | 2.236 / 10115 | 2.255 / 10032 | 2.294 / 9863 | 2.484 |

Reading: cutoffs are tightly clustered → hundreds of teams share a common known-good recipe we lack (esp. attention: +30% to reach #300). Find it first (upstream lee101/openkernels newest commits, public repos/forks, papers), then innovate.

## Rules
- AGENTS.md in each folder: no judge gaming, no output caching/replay, no precision below rules. Never print OKC_TOKEN.
- OKC: `~/.okc/bin/okc`. Limits per problem/day: 20 checks, 10 scored (reset 00:00 UTC). Check `okc problems --json` before queueing; a separate cloud session may also spend quota. Never duplicate a pending job.
- Each board agent owns only its folder + its board's quota. Use multi-variant probe files (time several variants in one check) to stretch quota.
- Local free pre-check: Docker nvcc for sm_90a compile / ptxas regs/spills (if unavailable, skip).
- Scored submit only if correct and ≥1% better than our current scored value beyond noise. Back up old default as `submission_before_<tag>.py`.
- Log every experiment as one line in `<folder>/EXPERIMENTS_V2.md` (mechanism, job, µs, verdict). Don't commit; lead commits.
- Upstream clone: /tmp/up (lee101/openkernels, `git -C /tmp/up pull`). Diff its newest versions vs our submission.

## Handoff (2026-11-01 ~17:30 MST)
- Research: RESEARCH_V2.md. Plateaus = public repos vjeux/okc-2026 (Apache-2.0), alygg77/okc-muse-spark-kernels (fork), NathenFernandes/muse-bf16-proj (MIT); local copies /tmp/scout/.
- User rule (updated): copy public file as base with attribution kept, submit if better, then improve on top.
- In-flight benchmark checks: attention candidate_vjeux.py (/tmp/vj_check.json) + probe_small_v1.py; bf16 candidate_A_alygg.py (/tmp/candA.json) + probe_v2_big.py; decode candidate_alygg.py (/tmp/alygg_check.json); MoE alygg port. Delete candidate_*vjeux*/ *alygg* files after numbers recorded.
- Key mechanisms to reimplement: attention single fused launch + FP32-atomic last-CTA merge + per-shape plan (wins on sliding/b1); MoE persistent grid, in-CTA routing, stream-K tails, K-permute scales (KPERM3); bf16 small-m routes o m8/pooler m8,m16 + backend pinning; decode compose + FP32 router T≤8 + per-(T,matrix) pinning.
- Local Mac memory-starved: use /usr/bin/python3 (pyenv python killed, exit 137).

## Paused (2026-11-02 01:40 UTC)
All agents stopped (judge queue stalled ~2h). Pending checks on judge: bf16 b0e37a8e/817484ad, attention 88ddfe93/4c25b44c, decode c8015a8e/c3e00e6a, moe 478c0e9c/b0cea1e7. Results: `okc status <id>`.
Done today: attention scored 51e80b11 1,886 µs (1.997×, #514). MoE 8943fb06 6,858, decode d10a67c1 12,567 (not better). bf16 Triton m128/m256 probe 385a845d: NO-GO.
Resume: one Sonnet agent per board — read public-file baseline result; if better, promote with attribution + submit; then improve per CHECKLIST.md.

## 2026-11-02 ~07:00 UTC
Overall #295→#300 (others moved). Round of tweaks on vjeux/alygg bases all no-go (attn plan optimal; moe ns4 crash, PDL flat; bf16 router ext worse).
Found keell0renz/muse-open-kernel-challenge (Apache-2.0, judge: decode 2.25, moe 2.39, attn ~2.64, bf16 1.57; LESSONS.md has ~460 experiments). Copied as */candidate_keel.py with attribution + NOTICE_keell0renz; scored: attn a7ab30813ae34de3, moe 958bd92a752e4369, decode df0ab3f05b064876, bf16 e9283643e06e4677. Promote winners to submission.py when scored.
Next: read /tmp/scout/k/repo/LESSONS.md before any new experiment (avoid their dead ends).
