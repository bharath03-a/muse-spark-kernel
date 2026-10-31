# Single-session brief — OKC top-50 campaign

Goal: top-50 overall on the Muse Open Kernel Challenge. Overall rank uses 4 boards: muse_bf16_proj, muse_attention_decode, muse_moe_w4a16, muse_decode_layer (prefill does NOT count). You own all of them.

## Setup
- `OKC_TOKEN` is in the environment; never print it. Install okc per any `*/AGENTS.md` if missing.
- Daily limits per problem: 20 checks, 10 scored. Check `okc problems --json` before queueing.
- Upstream reference: `git clone https://github.com/lee101/openkernels /tmp/up` (compare against e6b48c8; port anything newer and faster).
- Rules in `*/AGENTS.md`: no judge gaming, no output caching/replay, no precision below the rules.

## Step 0 — recover prior wins (do first)
Earlier sessions scored these, but their code may not be in `main`:
| board | job | speedup | rank | #50 cutoff |
|---|---|---|---|---|
| bf16 | d587f4c14d824847 | 1.481× | 352 | ~1.570× |
| attention | 8b3c761e26054d6f | 1.977× | 427 | ~2.679× |
| moe | b0013852514e4e7e | 2.032× | 417 | ~2.424× |
| decode | 48d31a264c934fef | 1.861× (12,158 µs) | 379 | ~2.240× |

`git fetch origin && git branch -r` — merge any `origin/claude/*` branches. If a board's best code is missing, continue from the folder's `submission.py` and note it in TAKEOVER.md.

## Priorities (biggest overall-rank gain per OKC run)
1. **BF16** (smallest gap, ~6%): router path dominates cost; large-M (qkv m256 split-K, register-A WGMMA m128). Closed: ac2 m16/m32 family. Read `muse_bf16_proj/bench/TOP50_CAMPAIGN.md` by line range.
2. **Whole decode** (~20% gap): `moe_x` ≈ 60% of verify-b16. Integrate every component win into `muse_decode_layer/`; then layer fusion (router on second stream, residual+RMSNorm epilogue fusion, PDL chaining, zero host syncs). Validate with full-layer check.
3. **MoE** (~19% gap): expert-grouped persistent schedule, weight-load pipelining, gate/up+activation fusion. Wins feed decode.
4. **Attention** (~35% gap): split-KV per batch for full-context b16/b32, GQA KV reuse, cheaper combine. Wins feed decode.

## Team structure — you are the lead
Spawn parallel subagents (Agent tool), one per board: `bf16`, `decode`, `moe`, `attention`. Keep at most 4 alive; credits are shared and limited.
- Each subagent gets: its board's folder, the state row above, its leads, the AGENTS.md rules, and this file's workflow rules. It owns only its folder.
- **Innovate, don't sweep.** Each subagent must first propose 3 materially different mechanisms with an expected gain and why (profile evidence, roofline math, upstream diff), then build the most promising one. No parameter sweeps of closed lines.
- **Lead owns OKC quota and git.** Subagents ask the lead before any `okc check`/`submit`; the lead dedupes, budgets ~5 checks per board per round, and is the only one who commits/pushes.
- Each round: collect results → kill dead lines → forward every component win (bf16/moe/attention) to the `decode` subagent for integration → respawn with sharper briefs. Spawn a fresh subagent when one stalls twice on the same mechanism.
- Optional reviewer subagent before each scored submit: verify correctness rules and noise margin.

## Workflow
- Submit scored only on correctness + gain ≥1% beyond noise. Back up old default as `submission_before_*.py`, record in `VERIFIED_CANDIDATES.json`, log in `TAKEOVER.md`.
- After every scored submission: commit and push to `main` (or `claude/all-boards` if main is refused).
- Token economy: read large files by line range, never re-read, one blocking `okc status <job> --wait`, terse output.
- Stall on one mechanism → switch mechanism or board. Stop only when all four boards' daily checks are used or overall rank ≤ 50; before stopping, push and append a 10-line handoff to TAKEOVER.md.
