# Cloud session tasks

Goal: top-50 overall rank on the Muse Open Kernel Challenge leaderboard.

## Setup (both sessions)

- Requires `OKC_TOKEN` in the cloud environment (output of `okc api token`). Install okc per any `*/AGENTS.md`.
- Upstream reference: `git clone https://github.com/lee101/openkernels /tmp/muse-openkernels-e6b48c8 && git -C /tmp/muse-openkernels-e6b48c8 checkout e6b48c8`.
- Read `TAKEOVER.md`, `VERIFIED_CANDIDATES.json`, `RESEARCH_TEAM.md` first. Inspect pending OKC jobs before queueing; never duplicate.
- No Modal GPU (cap 1800 s, 1782.6 used). Validate with OKC checks only.
- Submit a scored run only after correctness + gain ≥1% beyond noise. Back up the old default as `submission_before_*.py`, record provenance in `VERIFIED_CANDIDATES.json`, log in `TAKEOVER.md`.
- Commit and push results to `main` (rebase first; the other session pushes too). Only touch files you own.

## Task A — full-context attention b16/b32 (owns `muse_attention_decode/`, `muse_decode_layer/`)

State: attention 1,943 µs (#260, cutoff ~1,455). Whole decode 12,484 µs (scored 5ab92e7e91d64475, #210, cutoff ~10,638); base `muse_decode_layer/candidate_v35_wxpin32.py`.
Gather probe 82ccf4fc showed ~20–25% headroom on full-context b16/b32. Sweep check 36749dbf05214f06 (`muse_attention_decode/probe_fdec_sweep.py`) may have timed out — read its result; split into smaller probes if needed.
Build the real kernel (split-KV tuned per batch, GQA KV reuse, PDL, cheaper combine), validate standalone, then integrate into whole decode.

## Task B — MoE expert kernel (owns `muse_moe_w4a16/`)

State: MoE 6,827 µs scored (strict v56, #263, cutoff ~5,759). `muse_moe_w4a16/research_moe_xg2.py` (check 061e24b8ef934b9d) at 6,756 µs is within CI — push its mechanism further or combine levers until clearly beyond noise. No param sweeps; new mechanisms (expert-grouped persistent scheduling, weight-load pipelining, gate/up + activation fusion). `moe_x` is ~60% of whole-decode verify-b16: post verified wins under a "FOR WHOLE-DECODE" heading in `TAKEOVER.md`.

## Task C — BF16 dense projections (owns `muse_bf16_proj/`)

State: canonical `muse_bf16_proj/submission.py` (cp17) at 1.433×, #385. Live #50 cutoff 1.570× (~3,077 µs), #10 1.597×. Smallest gap of the four boards that count toward overall rank (prefill does not count).
Read first: `muse_bf16_proj/bench/TOP50_CAMPAIGN.md`, `bench/EXPERT_PANEL.md`, `bench/FINDINGS.md`, `ANALYSIS.md`, and the latest `profiles/submission.<job>.txt`.
Closed lines — do not reopen without a materially different mechanism: compressed-weight ac2 m16/m32 family (best case ~1.50×), adaptive direct / intra-CTA split, epilogue-only fix.
Leads:
1. Router: public hint that the router (different precision rule from BF16 projections) dominates observed cost. Re-read its correctness rules, measure its share of weighted load cost from the official profile, and build a faster precision-preserving router path.
2. Large M: the m256 load scores only ~1.15×. Candidates: tile-shaped split-K GEMM for qkv at m256; register-A WGMMA for pooler/unpooler at m128.
Any shape-based dispatch must stay valid on all test shapes. Use the official per-case weighted results to justify each change.

## Task D — Whole decoder layer (owns `muse_decode_layer/`; Task A is now attention-only)

Priority board. State: scored 12,484 µs, 1.809×, #401 (job 5ab92e7e91d64475), default `muse_decode_layer/submission.py` = `candidate_v35_wxpin32.py` (upstream v35 + M32 MoE route pinned to `wx -DNCW=16 -DWXG2 -DUDESC`). Live #50 ≈ 2.240×, #10 ≈ 2.345× → need ~24% less time. Check live: `okc leaderboard muse_decode_layer --json`.
Known profile: `moe_x` expert kernel ≈ 60% of verify-b16; batch-1 decode/verify close to upstream; old-path vs v35 per-shape picking worth ≤0.5% (done, don't redo).
Leads, biggest first:
1. Upstream first: `git ls-remote https://github.com/lee101/openkernels` and diff its newest whole-decoder/MoE/attention versions against e6b48c8. Port anything newer and faster.
2. Integrate sibling wins as they land: Task A attention (scored 1.977× standalone), Task B MoE, Task C projections — read TAKEOVER.md "FOR WHOLE-DECODE" and git log.
3. Layer-level fusion the standalone boards can't capture: router launched on a second stream overlapping the attention tail; residual-add + RMSNorm fused into attn-out / unpooler epilogues; PDL / programmatic launch chaining across the whole layer; one persistent residual workspace; zero host syncs.
4. b16/b32 verify: MoE dominates — specialized grouped expert schedule for the decoder's real routing distribution.
Validate with full-layer `okc check`; a component gain does not prove an end-to-end gain.
