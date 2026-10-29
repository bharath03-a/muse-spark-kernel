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
