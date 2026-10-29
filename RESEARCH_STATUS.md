# Research status

Updated after the latest source audit and check launches. Use `okc status` for live job state; this file is a checkpoint, not a claim that queued work completed.

## Verified defaults

| Challenge | Scored latency | Job |
| --- | ---: | --- |
| Whole decoder | 12,501.65 µs | 5648767e139d441b |
| MoE | 6,827.26 µs | 4a60fa38515e4581 |
| Attention | 1,942.78 µs | 7da917a2ea564131 |
| Prefill | 268,118.10 µs | 0d00b0439c5d4a54 |

BF16 projections are owned by the other chat. Do not edit that folder.

The newer whole score 12,484.08 µs (job 5ab92e7e91d64475) was withdrawn after a source audit found missing BF16 intermediate boundaries. It must not be restored as a valid best result. The repaired strict default above is promoted. See `muse_moe_w4a16/research/V35_STRICT_BOUNDARY_AUDIT.md`.

## Latest completed validation

- Attention diagnostic check **0653d51aad4c4281** completed successfully. Retained B32 results reject PFF4/PFF8 (~202→353–357 µs); balancing was roughly tied. The service kept only the final 16 KB of logs, so earlier B16 results are missing. No winning kernel was selected. Do not score this diagnostic. Future probes need compact final summaries after raw samples.
- Whole strict repair check **474298298c394dd8** passed at **12,499.28 µs**, with no flags; scored job **5648767e139d441b** passed at **12,501.65 µs** and is promoted. Candidate: `muse_decode_layer/candidate_v35_wxpin32_strict.py`. Preserves non-MoE AST and PDL/glue, restores BF16 gate/up/down and effective norm-weight boundaries, and removes unpatched FS from reachable dispatch. Source SHA256: `8dc4c65ec5facdd12dfe345fab90a85ccd2324370c0a79a8f2a6d84cb0815140`. An immutable snapshot is under `research_harness/results/sources/`.

Do not duplicate either job because observation times out. Query actual state first.

## Budget and next decisions

- User-reported Modal GPU use: **1,782.6 / 1,800 seconds**. No new GPU launches are allocated. CPU/source research can continue.
- MoE `research_moe_xg2.py` passed at 6,756 µs against 6,822 µs in separate checks, but the aggregate difference is inconclusive. Next isolated lead: distribute its gather copy over more SMs without changing bytes or adding synchronization mechanisms at the same time.
- Prefill eight-operation unpack candidate has exact bit/scale proofs and unchanged registers with no spills. It still needs GPU correctness and timing; no speedup claimed.
- Prefill n64-tail experiment was correct but about 22% slower; rejected.
- Root launch-fusion draft has a corrected runtime GPU pointer for local_start, but that correction has not been rerun on GPU. Do not integrate it.

Promote only after rules compliance, fresh-input correctness, meaningful measured improvement, source-to-job identity, and an actual scored result. Top 50 has not been reached.
