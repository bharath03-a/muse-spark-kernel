# Handoff — OKC campaign (written 2026-11-02)

## Goal (user)
1. Overall top 50 (4 boards: bf16_proj, attention_decode, moe_w4a16, decode_layer; prefill not counted). Top 50 at kernel close gets serving-track priority.
2. Submissions must be genuinely OUR OWN code.

Deadline: kernel submissions close **Nov 4, 11:59 pm PT**. Serving track opens Nov 5 noon PT (prizes are serving-only).

## Hard constraints
- Live `*/submission.py` = keell0renz public code (Apache-2.0) + attribution line 2 + `NOTICE_keell0renz`. Attribution **must stay** while any of their code is in the file (license; submitting certifies right to license). "Own" = rewrite we actually author (clean-room: learn techniques, write code ourselves), not stripping headers.
- OKC rules: CUDA-graph capturable; no caching/replay, no work outside timed calls, no precision below rules (FP8 Q allowed; router BF16 hi+lo ok). Final ranks = medians of organizer reruns + manual review.
- Never print OKC_TOKEN / Modal tokens. Repo private. Subagents: Sonnet only.
- Quota/day/problem: 5 scored, 10 checks (`~/.okc/bin/okc problems`), reset 00:00 UTC. Judge noise ±1.5% (bf16 ±2%): re-check before submitting gains <2%.
- Local Mac: use `/usr/bin/python3` (pyenv OOMs). `modal` CLI not on PATH locally. Agents must run Modal/okc in foreground with timeouts (two agents hung 10h on background waits).
- Modal budget user allowed: $5–6; ~$3–3.5 spent (estimate). Check Modal dashboard for orphaned apps.

## Current state (overall #134)
| board | live (keel) | our own best | rank | #50 cutoff (Nov 2) |
|---|---|---|---|---|
| bf16 | 3,072.3 µs, 1.572× | 1.481× (`submission_before_public.py`) | ~117 | 1.589× |
| attention | 1,426.8 µs, 2.640× | 1.997× | ~146 | 2.721× |
| moe | 5,640.7 µs, 2.384× | 2.032× | ~187 | 2.470× |
| decode | 10.03 ms, 2.256× | 1.861× | ~115 | 2.298× |

Backups: `*/submission_before_keel.py` (vjeux/alygg-based), `*/submission_before_public.py` (ours). Cutoffs drift up daily: `okc leaderboard <p> --json`.

## Evidence (Nov 2 round, all in `*/EXPERIMENTS_V2.md`)
- attention: wst verify-full streams at 4.0 TB/s ceiling; overhead = ramp 5.4 µs + CTA spread ~4 µs + fold 3.6–5 µs. Dead: wst→b32, NPW/PF knobs, 5th stage (smem), fold spread (+2 µs), tail-steal (setup 3 tiles > 4-tile spread). FP8 Q/PV already used. Compute fully hidden (ablation −3.6 µs).
- moe: 85% of achievable HBM at bs256 (792 MB, 28 distinct experts); timeline ramp 7.7, tail 5, cost skew 3 µs. Dead: early weight issue, ns4 (smem), PDL, wsi per-shape.
- bf16: v7 self-calibrating dispatch found 0 better routes; scored 3,074 (tie). Own rewrite gap vs keel ≈3 µs/case at m32/m128 qkv/pooler/unpooler + m256 router (`muse_bf16_proj/OWN_REWRITE_NOTES.md`, harness `bench/own_ab.py`).
- decode: GLE T=8 ≈0.19% (no-go); MoE = 41–65% of verify, attention 37–52%.
- Working harnesses: `muse_moe_w4a16/modal_ab.py`, `modal_tl.py`; `muse_attention_decode/modal_ev_{abl,abl2,tl,fold}.py`.
- Public repos: none stronger than keell0renz (checked Nov 2). Their LESSONS.md (~460 experiments): re-clone github.com/keell0renz/muse-open-kernel-challenge if /tmp wiped.

## Honest assessment
Tuning keel's kernels is exhausted; top 50 needs ~+1–3.6% per board via new designs. Making submissions fully our own means rebuilding from our 1.48–2.03× code to ≥ keel level — multi-day per board. Both goals by Nov 4 are unlikely together; decide per board.

## Suggested plan
1. Decide per board: (a) own clean-room rewrite (start bf16 — smallest, gap list known), or (b) keep attributed keel + own improvements.
2. bf16 own rewrite: find independently why keel's m32/m128 kernels are ~3 µs faster (tile shape, swizzle, split, epilogue), implement in our code, A/B via `bench/own_ab.py` on Modal, check, submit when ≥ keel.
3. One agent per board, evidence first (roofline + timeline), one change at a time, foreground only.
