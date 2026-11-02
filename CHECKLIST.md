# Implementation checklist — ordered by rank gain per okc check

Rules for every item: public Apache-2.0/MIT files may be used as base (keep license + attribution headers verbatim), then improved with our changes.
Loop per item: build → local compile (/usr/bin/python3, Docker nvcc) → `okc check` → log line in `<board>/EXPERIMENTS_V2.md` → if correct and ≥1% better (bf16 ≥2%, re-check once) back up `submission_before_<tag>.py`, promote, `okc submit` → commit + push.
Quota per board/day: 20 checks, 10 scored (reset 00:00 UTC). Check `okc problems --json` first.

## Phase 0 — baselines (once)
- [ ] Record per-case µs of our current scored file per board (latest profile) in EXPERIMENTS_V2.md.
- [ ] Record per-case targets from the benchmark-only check of each public file; delete the copies.
- [ ] Rank cases by (our µs / target µs) × weight → the work order inside each board.

## Phase 1 — attention (biggest gap: 1.98× → 2.57× for #300)
- [ ] 1.1 Single fused launch: prep + KV-cache write + split-KV attention + merge + rmsnorm·sigmoid epilogue. Target: sliding and b1 cases.
- [ ] 1.2 Merge via FP32 atomics with fixed reference max; last-arriving CTA finalizes and zeroes workspace (no memset, no combine kernel).
- [ ] 1.3 Per-shape launch plan: flat grid for full layers B>1, cluster/cooperative finalize for sliding, 2 CTAs/SM.
- [ ] 1.4 Graft our FP8-PV for q_len 8 (candidate_v81_fp8pv lineage).
- [ ] 1.5 Verify-full b32: deeper cp.async ring / more bytes in flight (230 µs vs 163 µs ceiling).
- [ ] 1.6 Forward the winning kernel to decode (FOR WHOLE-DECODE section).

## Phase 2 — MoE (2.03× → 2.34× for #300; feeds 40–59% of decode verify)
- [ ] 2.1 One persistent grid; routing computed redundantly in each CTA (no route launch).
- [ ] 2.2 Phase A (gate/up+SiLU) → phase B (down) → combine in the same grid, grid-wide sync via counters.
- [ ] 2.3 Fat/skinny job classes with cost-balanced stream-K tails.
- [ ] 2.4 Build variants selected by token count (d1 / b16 / b32).
- [ ] 2.5 K-permuted weight layout so one scale load serves 4 k16 steps (done in prepare(); keep prepare < 300 s).
- [ ] 2.6 Forward to decode.

## Phase 3 — whole decode (1.86× → 2.15× #300 → 2.29× #50)
- [ ] 3.1 FP32 router at T≤8 (correctness audit vs reference), fused into pooler stream.
- [ ] 3.2 prepare(): time each backend per (T, matrix) with CUDA-graph replays, pin winner.
- [ ] 3.3 Integrate Phase 1 attention kernel → check.
- [ ] 3.4 Integrate Phase 2 MoE kernel → check.
- [ ] 3.5 Glue fusion: resid ×2 (21 µs → ~10), xgather/rmsg/route into neighbours.
- [ ] 3.6 One probe check: whole-layer CUDA graph and router-on-stream-2 A/B (alygg measured streams as a loss — verify on ours, don't assume).
- [ ] 3.7 PDL chaining between remaining launches.

## Phase 4 — bf16 (1.481× → 1.534× #300 → 1.587× #50; noise ~2%)
- [ ] 4.1 Small-m routes: own kernels for o m8, pooler m8/m16.
- [ ] 4.2 Custom WGMMA O-projection.
- [ ] 4.3 Per-route backend pinning measured in prepare().
- [ ] 4.4 m128/m256 multi-variant probe (register-A WGMMA, persistent tiles, split-K qkv m256, TMA multicast).
- [ ] 4.5 Forward winners to decode.

## Phase 5 — beyond plateau (top 100 → 50)
- [ ] Re-profile all boards; attack the lowest per-case speedup again.
- [ ] Share any kernel win into decode the same day.
- [ ] New mechanisms only (no sweeps of closed lines); one multi-variant probe per idea set.

## Milestones
- [ ] Overall #300  - [ ] #150  - [ ] #100  - [ ] #50
